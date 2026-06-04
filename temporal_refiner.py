"""
Path 1: Temporal Refiner — small model that takes (12, 234) per-window predictions
and refines them using temporal structure across windows in the same file.

Key insight: per-window prediction independence is information bottleneck.
Refiner learns the joint structure: a bird calling in window 5 likely calls in 4 or 6,
file-level species consistency, etc.

Training: file-based holdout (47 train / 12 val of labeled soundscape files).
Architecture: lightweight Conv1d across window axis to avoid overfitting.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import soundfile as sf
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
import librosa.filters as lf

SR = 32000; WIN_SAMPLES = SR * 5; FILE_SAMPLES = 60 * SR; N_WINDOWS = 12

def build_mel(n, fmin, fmax):
    return lf.mel(sr=SR, n_fft=2048, n_mels=n, fmin=fmin, fmax=fmax,
                   htk=True, norm='slaney').astype(np.float32)

def fast_mel(chunks, mb, hann):
    out = []
    for ch in chunks:
        cp = np.pad(ch, 1024, mode='reflect')
        nf = 1 + (len(cp) - 2048) // 512
        frames = np.lib.stride_tricks.as_strided(cp, (nf, 2048), (cp.strides[0]*512, cp.strides[0])).copy()
        spec = np.abs(np.fft.rfft(frames * hann, axis=1))**2
        mel = mb @ spec.T
        mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
        mn, mx = mel_db.min(), mel_db.max()
        out.append((mel_db - mn) / (mx - mn + 1e-7))
    return np.repeat(np.stack(out)[:, np.newaxis], 3, axis=1).astype(np.float32)

def sed_pp(p):
    from scipy.ndimage import gaussian_filter1d
    if p.shape[0] <= 1: return p
    fm = p.max(axis=0, keepdims=True)
    x = p + 0.05*fm; sh = np.power(x, 1.5)
    sm = gaussian_filter1d(sh, sigma=0.7, axis=0)
    return np.power(np.maximum(sm, 1e-10), 1.0/1.5)

def run_model(path, mels, apply_sigmoid=False):
    sopts = ort.SessionOptions(); sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), sopts)
    inp = sess.get_inputs()[0].name
    preds = []
    for mel in mels:
        p = sess.run(None, {inp: mel})[0].astype(np.float32)
        if apply_sigmoid: p = 1.0/(1.0+np.exp(-np.clip(p,-50,50)))
        preds.append(sed_pp(p))
    return np.stack(preds)  # (n_files, 12, 234)


class TemporalRefiner(nn.Module):
    """Small 1D conv across window axis. Lightweight to avoid overfitting."""
    def __init__(self, n_classes=234, n_windows=12, hidden=64, kernel=3, n_layers=2):
        super().__init__()
        # Input: (B, 12, 234) → reshape to (B*234, 1, 12) for per-species 1D conv
        # OR: (B, 234, 12) and conv over 12 axis
        # We use shared per-species temporal kernel
        layers = []
        in_ch = 1
        for _ in range(n_layers):
            layers.append(nn.Conv1d(in_ch, hidden, kernel, padding=kernel//2))
            layers.append(nn.GELU())
            in_ch = hidden
        layers.append(nn.Conv1d(hidden, 1, kernel, padding=kernel//2))
        self.conv = nn.Sequential(*layers)
        # Residual connection: refined = raw + δ * conv_output
        self.delta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: (B, 12, 234) — per-window logits/probs
        B, T, C = x.shape
        # Reshape to per-species sequences: (B*C, 1, T)
        x_flat = x.permute(0, 2, 1).reshape(B*C, 1, T)
        delta = self.conv(x_flat).reshape(B, C, T).permute(0, 2, 1)  # (B, T, C)
        return x + self.delta * delta


def auc_per_species(pred, Y, min_pos=3):
    """Returns mean macro-AUC and per-species AUC dict."""
    aucs = []
    for c in range(pred.shape[-1]):
        y = Y[..., c].flatten()
        p = pred[..., c].flatten()
        if y.sum() >= min_pos and y.sum() < len(y):
            try: aucs.append(roc_auc_score(y, p))
            except: pass
    return float(np.mean(aucs)), len(aucs)


def main():
    t0 = time.time()
    tax = pd.read_csv('data/taxonomy.csv')
    tax['primary_label'] = tax['primary_label'].astype(str)
    label_cols = sorted(tax['primary_label'].tolist())
    l2i = {c:i for i,c in enumerate(label_cols)}
    N = len(label_cols)

    sc = pd.read_csv('data/train_soundscapes_labels.csv').drop_duplicates()
    sc_c = sc.groupby(['filename','start','end'])['primary_label'].apply(
        lambda s: sorted(set(l.strip() for x in s for l in str(x).split(';') if l.strip()))
    ).reset_index(name='label_list')
    sc_c['end_sec'] = pd.to_timedelta(sc_c['end']).dt.total_seconds().astype(int)
    wpf = sc_c.groupby('filename').size()
    full_files = sorted(wpf[wpf==12].index.tolist())
    full_sc = sc_c[sc_c['filename'].isin(full_files)].sort_values(['filename','end_sec']).reset_index(drop=True)

    # Build per-file labels (n_files, 12, 234)
    Y_per_file = np.zeros((len(full_files), 12, N), dtype=np.float32)
    file_to_idx = {f: i for i, f in enumerate(full_files)}
    for _, row in full_sc.iterrows():
        fi = file_to_idx[row['filename']]
        wi = (row['end_sec'] // 5) - 1
        for lbl in row['label_list']:
            if lbl in l2i: Y_per_file[fi, wi, l2i[lbl]] = 1.0

    # Compute mels + run models
    print('Computing mels + running models...')
    mb224 = build_mel(224, 0, 16000); mb128 = build_mel(128, 50, 14000)
    hann = np.hanning(2048).astype(np.float32)
    mels224, mels128 = [], []
    for fn in full_files:
        a, _ = sf.read(str(Path('data/train_soundscapes')/fn), dtype='float32')
        if a.ndim == 2: a = a.mean(axis=1)
        if len(a) < FILE_SAMPLES: a = np.pad(a, (0, FILE_SAMPLES-len(a)))
        a = a[:FILE_SAMPLES]
        chunks = a.reshape(N_WINDOWS, WIN_SAMPLES)
        mels224.append(fast_mel(chunks, mb224, hann))
        mels128.append(fast_mel(chunks, mb128, hann))

    sed = run_model('kaggle_model/LB872_soup3.onnx', mels224)
    nfnet = run_model('kaggle_model/eca_nfnet_l0.onnx', mels128, apply_sigmoid=True)
    v2s = run_model('kaggle_model/effv2s_r2_fold1.onnx', mels224)
    print(f'Models done: {time.time()-t0:.0f}s')

    # Simple 3-model average as baseline (no Perch — too slow to compute locally)
    base_pred = (sed + nfnet + v2s) / 3.0  # (n_files, 12, 234)

    # File-based 5-fold CV — train temporal refiner on 4 folds, test on held-out
    np.random.seed(42)
    perm = np.random.permutation(len(full_files))
    n_files = len(full_files)
    fold_size = n_files // 5

    all_refined = np.zeros_like(base_pred)
    all_baseline = base_pred.copy()  # No refinement

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training temporal refiner with 5-fold CV (device={device})...')

    for fold in range(5):
        val_idx = perm[fold*fold_size:(fold+1)*fold_size] if fold < 4 else perm[fold*fold_size:]
        train_idx = np.array([i for i in perm if i not in val_idx])

        X_train = torch.from_numpy(base_pred[train_idx]).float().to(device)  # (n_train, 12, 234)
        Y_train = torch.from_numpy(Y_per_file[train_idx]).float().to(device)
        X_val = torch.from_numpy(base_pred[val_idx]).float().to(device)

        model = TemporalRefiner(n_classes=N, n_windows=12, hidden=32, kernel=3, n_layers=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.BCELoss()

        # Train: 100 epochs, full batch (small data)
        best_val_loss = 1e9
        best_state = None
        for ep in range(100):
            model.train()
            opt.zero_grad()
            pred = model(X_train).clamp(1e-7, 1-1e-7)
            loss = loss_fn(pred, Y_train)
            loss.backward()
            opt.step()

            # Val check
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).clamp(1e-7, 1-1e-7)
                val_loss = loss_fn(val_pred, torch.from_numpy(Y_per_file[val_idx]).float().to(device)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Apply best model to val
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            refined = model(X_val).cpu().numpy()
        all_refined[val_idx] = refined
        print(f'  Fold {fold}: train_loss={loss.item():.4f} best_val_loss={best_val_loss:.4f} '
              f'delta={model.delta.item():.4f}')

    # Compare AUCs
    auc_baseline, n = auc_per_species(all_baseline, Y_per_file)
    auc_refined, _ = auc_per_species(all_refined, Y_per_file)
    print(f'\n=== Path 1 results (file-based 5-fold CV) ===')
    print(f'Baseline (no refinement): AUC = {auc_baseline:.5f}  (N_species={n})')
    print(f'Temporal refiner:         AUC = {auc_refined:.5f}')
    print(f'Improvement:              {auc_refined - auc_baseline:+.5f}')

    if auc_refined > auc_baseline:
        print(f'\nSignal POSITIVE — temporal refiner adds value.')
    else:
        print(f'\nSignal NEUTRAL/NEGATIVE — temporal structure not exploitable here.')


if __name__ == '__main__':
    main()

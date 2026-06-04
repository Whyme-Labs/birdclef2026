"""
Path 2 small experiment: Transformer refiner with cross-window self-attention.

Tests the CONCEPT of Path 2 (joint cross-window reasoning) without full retrain.
Attention learns to share info between the 12 windows of a file.

If this beats the conv refiner (Path 1) → cross-window structure exists, full Path 2 worth committing.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return np.stack(preds)


class CrossWindowTransformer(nn.Module):
    """
    Transformer over the 12-window axis. Each window's prediction is refined
    by attending to all 12 windows in the same file.

    Input/output shape: (B, T=12, C=234)

    Architecture:
    1. Project (12, 234) → (12, d_model) — embed each window's species predictions
    2. Self-attention across the 12 windows (capture file-level patterns)
    3. Project back to (12, 234) as residual delta
    4. final = raw + δ * delta
    """
    def __init__(self, n_classes=234, n_windows=12, d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(n_classes, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.unembed = nn.Linear(d_model, n_classes)
        self.delta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: (B, T, C)
        h = self.embed(x) + self.pos_emb  # (B, T, d_model)
        h = self.transformer(h)             # cross-window attention
        delta = self.unembed(h)             # (B, T, C)
        return x + self.delta * delta


def auc_score(pred, Y, min_pos=3):
    aucs = []
    for c in range(pred.shape[-1]):
        y = Y[..., c].flatten(); p = pred[..., c].flatten()
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

    Y = np.zeros((len(full_files), 12, N), dtype=np.float32)
    file_to_idx = {f: i for i, f in enumerate(full_files)}
    for _, row in full_sc.iterrows():
        fi = file_to_idx[row['filename']]
        wi = (row['end_sec'] // 5) - 1
        for lbl in row['label_list']:
            if lbl in l2i: Y[fi, wi, l2i[lbl]] = 1.0

    print('Computing predictions...')
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
    base_pred = (sed + nfnet + v2s) / 3.0
    print(f'Predictions: {time.time()-t0:.0f}s')

    # Train transformer refiner on ALL labeled data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.from_numpy(base_pred).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)

    model = CrossWindowTransformer(n_classes=N, n_windows=12, d_model=128, n_heads=4, n_layers=2, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
    loss_fn = nn.BCELoss()

    print(f'Training cross-window transformer ({sum(p.numel() for p in model.parameters())/1e3:.0f}k params)...')
    for ep in range(200):
        model.train()
        opt.zero_grad()
        # SpecAugment-style on input: randomly mask some windows
        x_in = X.clone()
        if ep > 20:  # warmup with no aug
            mask = torch.rand(X.shape[0], X.shape[1], device=device) < 0.1
            x_in[mask] = 0  # mask 10% of windows during training
        pred = model(x_in).clamp(1e-7, 1-1e-7)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (ep+1) % 40 == 0:
            print(f'  Ep {ep+1}: loss={loss.item():.5f} delta={model.delta.item():.4f} lr={sched.get_last_lr()[0]:.5f}')

    print(f'\nFinal delta = {model.delta.item():.4f}')

    # Save to ONNX
    model.eval()
    dummy = torch.zeros(1, 12, N).to(device)
    onnx_path = 'kaggle_model/attn_refiner.onnx'
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['probs_in'], output_names=['probs_out'],
        dynamic_axes={'probs_in': {0: 'batch'}, 'probs_out': {0: 'batch'}},
        opset_version=17,
    )
    print(f'ONNX exported to {onnx_path}')

    sess = ort.InferenceSession(onnx_path)
    test_in = np.random.uniform(0, 1, (3, 12, N)).astype(np.float32)
    onnx_out = sess.run(None, {'probs_in': test_in})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_in).to(device)).cpu().numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f'ONNX vs PyTorch max diff: {diff:.2e}')


if __name__ == '__main__':
    main()

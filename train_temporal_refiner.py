"""
Train a single temporal refiner on labeled soundscapes (no fold), save weights.
Apply at test time in submission notebook.

Even if local val is saturated, the refiner can still help on test (unsaturated).
At worst, learns δ=0 (identity).
"""
import time, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class TemporalRefiner(nn.Module):
    """
    Lightweight temporal refiner. Applies a learned residual smoothing kernel
    over window axis. Works per-species (shared kernel) for parameter efficiency.
    """
    def __init__(self, hidden=32, kernel=3, n_layers=2):
        super().__init__()
        layers = []
        in_ch = 1
        for _ in range(n_layers):
            layers.append(nn.Conv1d(in_ch, hidden, kernel, padding=kernel//2))
            layers.append(nn.GELU())
            in_ch = hidden
        layers.append(nn.Conv1d(hidden, 1, kernel, padding=kernel//2))
        self.conv = nn.Sequential(*layers)
        # Residual scale, init small to preserve baseline at start
        self.delta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: (B, T=12, C=234)
        B, T, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B*C, 1, T)
        delta = self.conv(x_flat).reshape(B, C, T).permute(0, 2, 1)
        return x + self.delta * delta


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

    print('Computing mels + predictions...')
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

    # Train on ALL files (no holdout — accept that local AUC is saturated)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.from_numpy(base_pred).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)

    model = TemporalRefiner(hidden=32, kernel=3, n_layers=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = nn.BCELoss()

    print('Training temporal refiner (BCE on labeled soundscapes)...')
    for ep in range(150):
        model.train()
        opt.zero_grad()
        pred = model(X).clamp(1e-7, 1-1e-7)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        opt.step()
        if (ep+1) % 30 == 0:
            print(f'  Ep {ep+1}: loss={loss.item():.5f} delta={model.delta.item():.4f}')

    # Final delta value
    print(f'\nFinal delta = {model.delta.item():.4f}')
    print(f'  delta=0 means refiner is identity (no help)')
    print(f'  delta!=0 means refiner contributes a learned correction')

    # Save weights for use in submission
    out_path = 'kaggle_model/temporal_refiner.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'config': {'hidden': 32, 'kernel': 3, 'n_layers': 2},
    }, out_path)
    print(f'Saved to {out_path}')

    # Export to ONNX for use in Kaggle inference
    model.eval()
    dummy = torch.zeros(1, 12, N).to(device)
    onnx_path = 'kaggle_model/temporal_refiner.onnx'
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['probs_in'], output_names=['probs_out'],
        dynamic_axes={'probs_in': {0: 'batch'}, 'probs_out': {0: 'batch'}},
        opset_version=17,
    )
    print(f'ONNX exported to {onnx_path}')

    # Verify ONNX matches PyTorch
    sess = ort.InferenceSession(onnx_path)
    test_in = np.random.uniform(0, 1, (3, 12, N)).astype(np.float32)
    onnx_out = sess.run(None, {'probs_in': test_in})[0]
    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_in).to(device)).cpu().numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f'ONNX vs PyTorch max diff: {diff:.2e}')


if __name__ == '__main__':
    main()

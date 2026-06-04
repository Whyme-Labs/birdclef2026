"""Test if local validation signal predicts V149 (replace) vs V137 direction."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
import librosa.filters as lf

SR = 32000; WIN_SAMPLES = SR * 5; FILE_SAMPLES = 60*SR; N_WINDOWS = 12

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

def pitch_shift(mel, s):
    if s == 0: return mel
    sh = np.roll(mel, s, axis=2)
    if s > 0: sh[:, :, :s] = mel.min()
    else: sh[:, :, s:] = mel.min()
    return sh

def sed_pp(probs):
    from scipy.ndimage import gaussian_filter1d
    if probs.shape[0] <= 1: return probs
    fm = probs.max(axis=0, keepdims=True)
    p = probs + 0.05 * fm
    sh = np.power(p, 1.5)
    sm = gaussian_filter1d(sh, sigma=0.7, axis=0)
    return np.power(np.maximum(sm, 1e-10), 1.0/1.5)

def rank_probs(p):
    n, c = p.shape
    out = np.zeros_like(p)
    for j in range(c):
        order = p[:, j].argsort()
        out[order, j] = np.arange(1, n+1, dtype=np.float32) / n
    return out

def qmix(probs_list, weights, alpha=0.5):
    w = np.array(weights); w = w/w.sum()
    prob_avg = np.zeros_like(probs_list[0])
    for p, wi in zip(probs_list, w): prob_avg += wi * p
    mn, mx = prob_avg.min(0, keepdims=True), prob_avg.max(0, keepdims=True)
    prob_avg = (prob_avg - mn) / (mx - mn + 1e-8)
    rank_avg = np.zeros_like(probs_list[0])
    for p, wi in zip(probs_list, w): rank_avg += wi * rank_probs(p)
    return alpha * prob_avg + (1-alpha) * rank_avg

def run_onnx_tta(path, mels, shifts=[0, 2], apply_sigmoid=False):
    sopts = ort.SessionOptions(); sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(path), sopts)
    inp = sess.get_inputs()[0].name
    accum = None
    for s in shifts:
        preds = []
        for mel in mels:
            m = pitch_shift(mel, s) if s != 0 else mel
            p = sess.run(None, {inp: m})[0].astype(np.float32)
            if apply_sigmoid: p = 1.0/(1.0+np.exp(-np.clip(p,-50,50)))
            preds.append(sed_pp(p))
        a = np.concatenate(preds, axis=0)
        accum = a if accum is None else accum + a
    return accum / len(shifts)


taxonomy = pd.read_csv('data/taxonomy.csv')
label_cols = sorted(taxonomy['primary_label'].astype(str).tolist())
l2i = {c:i for i,c in enumerate(label_cols)}
N = len(label_cols)

sc = pd.read_csv('data/train_soundscapes_labels.csv').drop_duplicates()
sc_c = sc.groupby(['filename','start','end'])['primary_label'].apply(
    lambda s: sorted(set(l.strip() for x in s for l in str(x).split(';') if l.strip()))
).reset_index(name='label_list')
sc_c['end_sec'] = pd.to_timedelta(sc_c['end']).dt.total_seconds().astype(int)
wpf = sc_c.groupby('filename').size()
full_files = sorted(wpf[wpf == 12].index.tolist())
full_sc = sc_c[sc_c['filename'].isin(full_files)].sort_values(['filename','end_sec']).reset_index(drop=True)
Y = np.zeros((len(full_sc), N), dtype=np.float32)
for i, ll in enumerate(full_sc['label_list']):
    for lbl in ll:
        if lbl in l2i: Y[i, l2i[lbl]] = 1.0

mb224 = build_mel(224, 0, 16000); mb128 = build_mel(128, 50, 14000)
hann = np.hanning(2048).astype(np.float32)
mels224, mels128 = [], []
t0 = time.time()
print('Computing mels...')
for fn in full_files:
    a, _ = sf.read(str(Path('data/train_soundscapes')/fn), dtype='float32')
    if a.ndim == 2: a = a.mean(axis=1)
    if len(a) < FILE_SAMPLES: a = np.pad(a, (0, FILE_SAMPLES-len(a)))
    a = a[:FILE_SAMPLES]
    chunks = a.reshape(N_WINDOWS, WIN_SAMPLES)
    mels224.append(fast_mel(chunks, mb224, hann))
    mels128.append(fast_mel(chunks, mb128, hann))

print('SED...'); sed = run_onnx_tta('kaggle_model/LB872_soup3.onnx', mels224)
print('NFNet...'); nfnet = run_onnx_tta('kaggle_model/eca_nfnet_l0.onnx', mels128, shifts=[0], apply_sigmoid=True)
print('V2S R2...'); v2s_r2 = run_onnx_tta('kaggle_model/effv2s_r2_fold1.onnx', mels224)
print('V2S merged...'); v2s_merged = run_onnx_tta('kaggle_model/v2s_2025merged.onnx', mels224)

def macro_auc(ens, Y, min_pos=3):
    aucs = []
    for j in range(ens.shape[1]):
        y = Y[:, j]
        if y.sum() >= min_pos and y.sum() < len(y):
            try: aucs.append(roc_auc_score(y, ens[:, j]))
            except: pass
    return np.mean(aucs), len(aucs)

# V137: SED + NFNet + V2S_R2
v137 = qmix([sed, nfnet, v2s_r2], [0.30, 0.20, 0.20], alpha=0.5)
# V149: SED + NFNet + V2S_merged (REPLACES V2S_R2)
v149 = qmix([sed, nfnet, v2s_merged], [0.30, 0.20, 0.20], alpha=0.5)
# V150: V137 + V2S_merged at 0.05
v150 = qmix([sed, nfnet, v2s_r2, v2s_merged], [0.30, 0.20, 0.20, 0.05], alpha=0.5)

auc_137, n = macro_auc(v137, Y)
auc_149, _ = macro_auc(v149, Y)
auc_150, _ = macro_auc(v150, Y)

print(f'\n=== Local val (3-model approx, no Perch; N_species={n}) ===')
print(f'V137-like: {auc_137:.5f}')
print(f'V149-like: {auc_149:.5f}  (LB: 0.938, diff vs V137 = -0.003)')
print(f'V150-like: {auc_150:.5f}  (LB: 0.940, diff vs V137 = -0.001)')
print(f'\nLocal-LB correlation check:')
print(f'  V149 local diff: {auc_149 - auc_137:+.5f} | LB: -0.003 | sign_match={"YES" if (auc_149-auc_137)<0 else "NO"}')
print(f'  V150 local diff: {auc_150 - auc_137:+.5f} | LB: -0.001 | sign_match={"YES" if (auc_150-auc_137)<0 else "NO"}')

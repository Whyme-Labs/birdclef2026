"""
Diagnostic analysis: WHERE does our ensemble ceiling at 0.941 come from?

Deep questions:
1. Per-species AUC distribution — which species are we failing?
2. Model agreement — are errors orthogonal (ensemble can help) or redundant (need new info)?
3. Per-taxon patterns — birds/amphibians/insects fail differently?
4. Training data vs performance — does more data = better AUC?
"""
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
    return np.concatenate(preds, axis=0).astype(np.float32)

tax = pd.read_csv('data/taxonomy.csv')
tax['primary_label'] = tax['primary_label'].astype(str)
label_cols = sorted(tax['primary_label'].tolist())
l2i = {c:i for i,c in enumerate(label_cols)}
N = len(label_cols)
class_name = dict(zip(tax['primary_label'], tax['class_name']))

train_df = pd.read_csv('data/train.csv')
train_counts = train_df['primary_label'].astype(str).value_counts().to_dict()

sc = pd.read_csv('data/train_soundscapes_labels.csv').drop_duplicates()
sc_c = sc.groupby(['filename','start','end'])['primary_label'].apply(
    lambda s: sorted(set(l.strip() for x in s for l in str(x).split(';') if l.strip()))
).reset_index(name='label_list')
sc_c['end_sec'] = pd.to_timedelta(sc_c['end']).dt.total_seconds().astype(int)
wpf = sc_c.groupby('filename').size()
full_files = sorted(wpf[wpf==12].index.tolist())
full_sc = sc_c[sc_c['filename'].isin(full_files)].sort_values(['filename','end_sec']).reset_index(drop=True)
Y = np.zeros((len(full_sc), N), dtype=np.float32)
for i, ll in enumerate(full_sc['label_list']):
    for lbl in ll:
        if lbl in l2i: Y[i, l2i[lbl]] = 1.0

t0 = time.time()
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
print(f'Model inference: {time.time()-t0:.0f}s')

results = []
for c_idx, sp in enumerate(label_cols):
    y = Y[:, c_idx]
    n_pos = int(y.sum())
    if n_pos < 3 or n_pos == len(y): continue
    row = {'species': sp, 'class': class_name.get(sp, '?'), 'n_pos': n_pos,
           'train_n': train_counts.get(sp, 0)}
    try:
        row['sed_auc'] = roc_auc_score(y, sed[:, c_idx])
        row['nfnet_auc'] = roc_auc_score(y, nfnet[:, c_idx])
        row['v2s_auc'] = roc_auc_score(y, v2s[:, c_idx])
        avg = (sed[:, c_idx] + nfnet[:, c_idx] + v2s[:, c_idx]) / 3
        row['avg_auc'] = roc_auc_score(y, avg)
        results.append(row)
    except ValueError: pass

df = pd.DataFrame(results)
print(f'\n=== Per-species AUC distribution (N={len(df)} species ≥3 pos) ===')
print(f'Mean avg_auc: {df["avg_auc"].mean():.4f}')
print(f'Median avg_auc: {df["avg_auc"].median():.4f}')
print(f'AUC < 0.70: {(df["avg_auc"]<0.70).sum()}  (HARD)')
print(f'AUC < 0.80: {(df["avg_auc"]<0.80).sum()}')
print(f'AUC < 0.90: {(df["avg_auc"]<0.90).sum()}')
print(f'AUC >= 0.95: {(df["avg_auc"]>=0.95).sum()}  (SATURATED)')

model_aucs = df[['sed_auc', 'nfnet_auc', 'v2s_auc']].values
df['auc_std'] = model_aucs.std(axis=1)
df['auc_min'] = model_aucs.min(axis=1)
df['auc_max'] = model_aucs.max(axis=1)
df['auc_range'] = df['auc_max'] - df['auc_min']

print('\n=== KEY DIAGNOSTIC: error orthogonality ===')
for thr in [0.70, 0.80, 0.90]:
    sub = df[df['avg_auc'] < thr]
    if len(sub) == 0: continue
    print(f'Species with avg_auc<{thr:.2f}: N={len(sub)}')
    print(f'  mean inter-model std: {sub["auc_std"].mean():.4f}')
    print(f'  mean max-min range:   {sub["auc_range"].mean():.4f}')
    # If >0.05, models disagree meaningfully
    disagree = (sub['auc_range'] > 0.10).sum()
    print(f'  species where models disagree (range>0.10): {disagree} ({100*disagree/len(sub):.0f}%)')

print('\n=== Per-taxon breakdown ===')
for taxon in sorted(df['class'].unique()):
    sub = df[df['class']==taxon]
    if len(sub) == 0: continue
    print(f'{taxon:12s}  N={len(sub):3d}  mean_auc={sub["avg_auc"].mean():.4f}  '
          f'hard(<0.80)={(sub["avg_auc"]<0.80).sum()}/{len(sub)}')

print('\n=== Training data vs performance ===')
df['log_train_n'] = np.log1p(df['train_n'])
corr = df[['log_train_n', 'avg_auc']].corr().iloc[0,1]
print(f'Pearson r(log_train_n, avg_auc) = {corr:.3f}')
zs = df[df['train_n']==0]
if len(zs) > 0:
    print(f'Zero-shot species (train_n=0): N={len(zs)}, mean AUC={zs["avg_auc"].mean():.4f}')

print('\n=== BOTTOM 15 SPECIES (hardest) ===')
bot = df.nsmallest(15, 'avg_auc')[['species','class','n_pos','train_n','sed_auc','nfnet_auc','v2s_auc','avg_auc','auc_range']]
print(bot.to_string(index=False))

df.to_csv('diagnostic_v137.csv', index=False)
print(f'\nSaved to diagnostic_v137.csv')

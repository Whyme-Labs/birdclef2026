"""V189 MVT: Sparse motif token minimum viable test.

PCEN spectrograms → sparse event patches via spectral flux → KMeans codebook
→ bag-of-tokens histogram per chunk → LogReg per species.

Tests whether sparse acoustic motifs carry per-species signal on labeled soundscape.

PREDICTION (Predict-Then-Run):
  - Held-out macro AUC: 0.55-0.70 (cited paper got ~0.52 token-only, our setup
    smaller codebook + simpler patches expected similar to slightly better)
  - Direction: weak signal at best, won't beat single-probe AUC of ~0.75
  - Confidence: medium

DISCONFIRM (kill the direction):
  - AUC < 0.55 → tokens carry no extractable signal in our setup
  - AUC < probe-only AUC (~0.75) → won't add orthogonal signal vs V183 stack

CONFIRM (worth pipeline investment):
  - AUC > 0.70 AND combined-with-V137 AUC notably > V137 alone

Cost: ~30-60 min CPU, no GPU needed.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


SR = 32000
HOP = 320  # 10ms hop @ 32k
N_FFT = 2048
N_MELS = 128
FMIN = 50
FMAX = 14000
PATCH_T = 16  # ~160ms patches
PATCH_F = 32  # half mel range per patch
N_TOKENS_CODEBOOK = 512
TOP_K_PATCHES = 8  # sparse: keep top 8 events per chunk


def pcen(mel):
    """Per-Channel Energy Normalization."""
    eps = 1e-6
    s = 0.025  # smoothing
    alpha = 0.98
    delta = 2.0
    r = 0.5
    M = np.zeros_like(mel)
    M[:, 0] = mel[:, 0]
    for t in range(1, mel.shape[1]):
        M[:, t] = (1 - s) * M[:, t - 1] + s * mel[:, t]
    return (mel / (eps + M) ** alpha + delta) ** r - delta ** r


def load_chunk_pcen(filename, end_sec, audio_dir):
    """Load 5s chunk and compute PCEN mel."""
    fp = audio_dir / filename
    audio, sr = sf.read(str(fp))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    start = (end_sec - 5) * SR
    end = end_sec * SR
    if start < 0:
        target = audio[0:max(0, end)]
        if len(target) < 5*SR:
            target = np.pad(target, (5*SR - len(target), 0))
    elif end > len(audio):
        target = audio[start:]
        if len(target) < 5*SR:
            target = np.pad(target, (0, 5*SR - len(target)))
    else:
        target = audio[start:end]

    target = target.astype(np.float32)
    # Mel
    mel = librosa.feature.melspectrogram(
        y=target, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0)
    return pcen(mel)


def extract_patches(pcen_mel, top_k=TOP_K_PATCHES):
    """Extract top-K event patches from PCEN spectrogram via spectral flux."""
    # Spectral flux: positive temporal differences in PCEN
    if pcen_mel.shape[1] < 2:
        return np.zeros((0, PATCH_F * PATCH_T), dtype=np.float32)
    flux = np.maximum(0, np.diff(pcen_mel, axis=1))  # (n_mels, T-1)
    # Per-time energy
    flux_sum = flux.sum(axis=0)  # (T-1,)
    # Smooth flux
    if len(flux_sum) >= 5:
        flux_sum = np.convolve(flux_sum, np.ones(5)/5, mode='same')
    # Find top-K time peaks
    if len(flux_sum) <= top_k:
        peak_times = np.arange(len(flux_sum))
    else:
        peak_times = np.argpartition(flux_sum, -top_k)[-top_k:]
    # Sort by time
    peak_times = np.sort(peak_times)

    patches = []
    half_t = PATCH_T // 2
    for t in peak_times:
        # Find frequency center of max activity around this time
        win = pcen_mel[:, max(0, t - half_t):min(pcen_mel.shape[1], t + half_t)]
        if win.shape[1] == 0:
            continue
        # Frequency-domain peak
        freq_energy = win.sum(axis=1)
        f_peak = int(np.argmax(freq_energy))
        # Extract patch
        f_lo = max(0, f_peak - PATCH_F // 2)
        f_hi = min(N_MELS, f_lo + PATCH_F)
        f_lo = f_hi - PATCH_F  # ensure exact size
        if f_lo < 0:
            continue
        t_lo = max(0, t - half_t)
        t_hi = min(pcen_mel.shape[1], t_lo + PATCH_T)
        t_lo = t_hi - PATCH_T
        if t_lo < 0:
            continue
        patch = pcen_mel[f_lo:f_hi, t_lo:t_hi]
        if patch.shape != (PATCH_F, PATCH_T):
            continue
        patches.append(patch.flatten())
    if not patches:
        return np.zeros((0, PATCH_F * PATCH_T), dtype=np.float32)
    return np.stack(patches).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    args = ap.parse_args()

    tax = pd.read_csv(args.taxonomy)
    label_cols = tax["primary_label"].astype(str).tolist()
    l2i = {c: i for i, c in enumerate(label_cols)}

    sc = pd.read_csv(args.soundscape_csv)
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)

    def parse_lbl(x):
        if pd.isna(x) or x == "nan": return set()
        return set(t.strip() for t in str(x).split(";") if t.strip())
    sc["label_set"] = sc["primary_label"].apply(parse_lbl)
    grouped = sc.groupby(["filename", "end_sec"])["label_set"].apply(
        lambda s: set().union(*s)).reset_index()

    audio_dir = Path(args.soundscape_dir)
    existing = set(p.name for p in audio_dir.glob("*.ogg"))
    grouped = grouped[grouped["filename"].isin(existing)].reset_index(drop=True)
    print(f"Chunks: {len(grouped)}")

    # Phase 1: Extract all patches
    print("\nPhase 1: Extract sparse patches (PCEN + flux peaks)...")
    t0 = time.time()
    all_patches = []
    chunk_patch_counts = []
    chunk_groups = []  # filename for grouped CV
    chunk_labels = []
    skipped = 0

    by_file = grouped.groupby("filename")
    files_done = 0
    for fn, group in by_file:
        for _, r in group.iterrows():
            try:
                pcen_mel = load_chunk_pcen(fn, int(r["end_sec"]), audio_dir)
                patches = extract_patches(pcen_mel)
            except Exception as e:
                skipped += 1
                continue
            chunk_patch_counts.append(len(patches))
            if len(patches) > 0:
                all_patches.append(patches)
            # Label
            y = np.zeros(len(label_cols), dtype=np.float32)
            for lbl in r["label_set"]:
                if lbl in l2i:
                    y[l2i[lbl]] = 1.0
            chunk_labels.append(y)
            chunk_groups.append(fn)
        files_done += 1
        if files_done % 10 == 0:
            print(f"  {files_done}/{len(by_file)} files — {time.time()-t0:.0f}s")

    patches_all = np.concatenate(all_patches, axis=0) if all_patches else np.zeros((0, PATCH_F*PATCH_T), dtype=np.float32)
    print(f"  total patches: {patches_all.shape}, chunks: {len(chunk_labels)}, skipped: {skipped}")
    print(f"  extraction time: {time.time()-t0:.0f}s")

    # Phase 2: KMeans codebook
    print(f"\nPhase 2: KMeans codebook ({N_TOKENS_CODEBOOK} clusters)...")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=N_TOKENS_CODEBOOK, random_state=42, batch_size=512, n_init=3)
    km.fit(patches_all)
    print(f"  KMeans fit: {time.time()-t0:.0f}s")

    # Phase 3: Compute per-chunk token histograms
    print(f"\nPhase 3: Token histograms per chunk...")
    t0 = time.time()
    X = np.zeros((len(chunk_labels), N_TOKENS_CODEBOOK), dtype=np.float32)
    p_offset = 0
    for i, npat in enumerate(chunk_patch_counts):
        if npat == 0:
            continue
        ptokens = km.predict(patches_all[p_offset:p_offset+npat])
        for tk in ptokens:
            X[i, tk] += 1
        p_offset += npat
    # L1 normalize
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    X = X / row_sum
    Y = np.stack(chunk_labels)
    groups = np.array(chunk_groups)
    print(f"  histograms: X={X.shape}, Y={Y.shape}, time {time.time()-t0:.0f}s")

    # Phase 4: 5-fold grouped CV with LR
    print(f"\nPhase 4: 5-fold GroupKFold LR...")
    gkf = GroupKFold(n_splits=5)
    auc_folds = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]
        valid = (Y_tr.sum(axis=0) >= 3) & (Y_te.sum(axis=0) >= 1)
        if valid.sum() == 0:
            continue
        # Train one LR per valid class
        scores = np.zeros((len(Y_te), len(label_cols)), dtype=np.float32)
        for ci in np.where(valid)[0]:
            try:
                lr = LogisticRegression(C=1.0, max_iter=200, solver="liblinear", class_weight="balanced")
                lr.fit(X_tr, Y_tr[:, ci])
                scores[:, ci] = lr.predict_proba(X_te)[:, 1]
            except Exception:
                pass
        try:
            auc = roc_auc_score(Y_te[:, valid], scores[:, valid], average="macro")
        except ValueError:
            auc = 0.0
        n_classes = valid.sum()
        print(f"  fold {fold}: AUC={auc:.4f} on {n_classes} classes")
        auc_folds.append(auc)

    if auc_folds:
        mean_auc = np.mean(auc_folds)
        std_auc = np.std(auc_folds)
        print(f"\n=== TOKEN MVT RESULT ===")
        print(f"Mean macro AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"Decision:")
        if mean_auc < 0.55:
            print(f"  ABANDON: AUC {mean_auc:.4f} < 0.55 → tokens carry no signal")
        elif mean_auc < 0.65:
            print(f"  WEAK: AUC {mean_auc:.4f} ≈ paper-baseline 0.52 → not worth full pipeline")
        elif mean_auc < 0.75:
            print(f"  AMBIGUOUS: AUC {mean_auc:.4f} → marginal, test as 5th probe")
        else:
            print(f"  CONFIRMED: AUC {mean_auc:.4f} ≥ 0.75 → invest in full pipeline")


if __name__ == "__main__":
    main()

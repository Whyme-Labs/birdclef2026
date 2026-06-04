"""
Local evaluation on labeled soundscapes — simulates V137 pipeline.

Purpose: Build a local metric that ranks submissions similarly to LB.
Used to guide future experiments without blind submissions.

Note: models were trained on labeled soundscapes, so this is IN-SAMPLE.
Purpose is RELATIVE comparison, not absolute AUC estimation.
"""
import os
import re
import sys
import gc
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
import librosa.filters as lf


SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12


def build_mel_basis(n_mels, fmin, fmax):
    return lf.mel(sr=SR, n_fft=2048, n_mels=n_mels, fmin=fmin, fmax=fmax,
                   htk=True, norm='slaney').astype(np.float32)


def fast_mel(chunks, mel_basis, hann):
    out = []
    for ch in chunks:
        cp = np.pad(ch, 1024, mode='reflect')
        nf = 1 + (len(cp) - 2048) // 512
        frames = np.lib.stride_tricks.as_strided(
            cp, (nf, 2048), (cp.strides[0]*512, cp.strides[0])).copy()
        spec = np.abs(np.fft.rfft(frames * hann, axis=1))**2
        mel = mel_basis @ spec.T
        mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel_db = np.maximum(mel_db, mel_db.max() - 80.0)
        mn, mx = mel_db.min(), mel_db.max()
        out.append((mel_db - mn) / (mx - mn + 1e-7))
    return np.repeat(np.stack(out)[:, np.newaxis], 3, axis=1).astype(np.float32)


def read_60s(path):
    y, sr = sf.read(str(path), dtype='float32')
    if y.ndim == 2: y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]


def pitch_shift_mel(mel, shift):
    if shift == 0: return mel
    shifted = np.roll(mel, shift, axis=2)
    if shift > 0: shifted[:, :, :shift] = mel.min()
    else: shifted[:, :, shift:] = mel.min()
    return shifted


def sed_pp(probs):
    if probs.shape[0] <= 1: return probs
    from scipy.ndimage import gaussian_filter1d
    fm = probs.max(axis=0, keepdims=True)
    probs = probs + 0.05 * fm
    sh = np.power(probs, 1.5)
    sm = gaussian_filter1d(sh, sigma=0.7, axis=0)
    return np.power(np.maximum(sm, 1e-10), 1.0/1.5)


def infer_onnx_tta(model_path, mels_list, shifts=[0, 2]):
    """Run ONNX model on list of mel arrays with pitch-shift TTA."""
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), sopts)
    inp = sess.get_inputs()[0].name
    accum = None
    for shift in shifts:
        preds = []
        for mel in mels_list:
            mel_s = pitch_shift_mel(mel, shift) if shift != 0 else mel
            p = sess.run(None, {inp: mel_s})[0].astype(np.float32)
            preds.append(sed_pp(p))
        a = np.concatenate(preds, axis=0).astype(np.float32)
        accum = a if accum is None else accum + a
    return accum / len(shifts)


def infer_onnx(model_path, mels_list, apply_sigmoid=False):
    """Run ONNX without TTA (for NFNet — no TTA in V137)."""
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(model_path), sopts)
    inp = sess.get_inputs()[0].name
    preds = []
    for mel in mels_list:
        p = sess.run(None, {inp: mel})[0].astype(np.float32)
        if apply_sigmoid:
            p = 1.0 / (1.0 + np.exp(-np.clip(p, -50, 50)))
        preds.append(sed_pp(p))
    return np.concatenate(preds, axis=0).astype(np.float32)


def rank_probs(probs):
    """Per-species rank normalization."""
    n, c = probs.shape
    out = np.zeros_like(probs, dtype=np.float32)
    for j in range(c):
        order = probs[:, j].argsort()
        out[order, j] = np.arange(1, n+1, dtype=np.float32) / n
    return out


def qmix_ensemble(probs_list, weights, alpha=0.5):
    """QMix: blend prob avg + rank avg."""
    # Normalize weights
    w = np.array(weights, dtype=np.float32); w = w / w.sum()
    # Probability average
    prob_avg = np.zeros_like(probs_list[0], dtype=np.float32)
    for p, wi in zip(probs_list, w):
        prob_avg += wi * p
    # Min-max normalize per species
    mn = prob_avg.min(axis=0, keepdims=True)
    mx = prob_avg.max(axis=0, keepdims=True)
    prob_avg = (prob_avg - mn) / (mx - mn + 1e-8)
    # Rank average
    rank_avg = np.zeros_like(probs_list[0], dtype=np.float32)
    for p, wi in zip(probs_list, w):
        rank_avg += wi * rank_probs(p)
    return alpha * prob_avg + (1 - alpha) * rank_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sed', default='kaggle_model/LB872_soup3.onnx')
    parser.add_argument('--nfnet', default='kaggle_model/eca_nfnet_l0.onnx')
    parser.add_argument('--v2s', default='kaggle_model/effv2s_r2_fold1.onnx')
    parser.add_argument('--perch_probs', default=None,
                        help='Path to precomputed Perch probs on labeled soundscapes')
    parser.add_argument('--out', default='local_val_v137.csv')
    args = parser.parse_args()

    t0 = time.time()
    # Load label data
    taxonomy = pd.read_csv('data/taxonomy.csv')
    label_cols = sorted(taxonomy['primary_label'].astype(str).tolist())
    label_to_idx = {c:i for i,c in enumerate(label_cols)}
    N_CLASSES = len(label_cols)

    sc = pd.read_csv('data/train_soundscapes_labels.csv').drop_duplicates().reset_index(drop=True)
    sc_clean = sc.groupby(['filename','start','end'])['primary_label'].apply(
        lambda s: sorted(set(l.strip() for x in s for l in str(x).split(';') if l.strip()))
    ).reset_index(name='label_list')
    sc_clean['end_sec'] = pd.to_timedelta(sc_clean['end']).dt.total_seconds().astype(int)

    wpf = sc_clean.groupby('filename').size()
    full_files = sorted(wpf[wpf == N_WINDOWS].index.tolist())
    full_sc = sc_clean[sc_clean['filename'].isin(full_files)].sort_values(
        ['filename','end_sec']).reset_index(drop=True)
    print(f'Full-labeled files: {len(full_files)}, windows: {len(full_sc)}')

    # Build label matrix
    Y = np.zeros((len(full_sc), N_CLASSES), dtype=np.float32)
    for i, labels in enumerate(full_sc['label_list']):
        for lbl in labels:
            if lbl in label_to_idx: Y[i, label_to_idx[lbl]] = 1.0

    # Compute mels (224-mel + 128-mel)
    print('Computing mels (224 + 128 mel)...')
    mel_224 = build_mel_basis(224, 0, 16000)
    mel_128 = build_mel_basis(128, 50, 14000)
    hann = np.hanning(2048).astype(np.float32)

    mels_224 = []
    mels_128 = []
    for fname in full_files:
        audio = read_60s(Path('data/train_soundscapes') / fname)
        chunks = audio.reshape(N_WINDOWS, WIN_SAMPLES)
        mels_224.append(fast_mel(chunks, mel_224, hann))
        mels_128.append(fast_mel(chunks, mel_128, hann))
    print(f'Mels computed: 224-mel {len(mels_224)} files, 128-mel {len(mels_128)} files — {time.time()-t0:.0f}s')

    # Run models
    print(f'Running SED ({args.sed})...')
    sed_probs = infer_onnx_tta(args.sed, mels_224, shifts=[0, 2])
    print(f'  SED done: {sed_probs.shape} — {time.time()-t0:.0f}s')

    print(f'Running NFNet ({args.nfnet})...')
    # NFNet outputs logits, apply sigmoid
    nfnet_probs = infer_onnx(args.nfnet, mels_128, apply_sigmoid=True)
    print(f'  NFNet done: {nfnet_probs.shape} — {time.time()-t0:.0f}s')

    print(f'Running V2S ({args.v2s})...')
    v2s_probs = infer_onnx_tta(args.v2s, mels_224, shifts=[0, 2])
    print(f'  V2S done: {v2s_probs.shape} — {time.time()-t0:.0f}s')

    # Perch probs: use raw model logits if no precomputed file
    # For a quick baseline, use uniform (will need to precompute later)
    if args.perch_probs and Path(args.perch_probs).exists():
        print(f'Loading Perch probs from {args.perch_probs}')
        perch_probs = np.load(args.perch_probs)
    else:
        print('WARNING: No Perch probs available, using zeros (3-model ensemble)')
        perch_probs = np.zeros_like(sed_probs)

    # V137 ensemble: Perch 0.50 + SED 0.30 + NFNet 0.20 + V2S 0.20, QMix α=0.5
    if perch_probs.any():
        ensemble = qmix_ensemble(
            [perch_probs, sed_probs, nfnet_probs, v2s_probs],
            [0.50, 0.30, 0.20, 0.20],
            alpha=0.5
        )
    else:
        # 3-model without Perch
        ensemble = qmix_ensemble(
            [sed_probs, nfnet_probs, v2s_probs],
            [0.30, 0.20, 0.20],
            alpha=0.5
        )

    # Macro-AUC on species with >=1 positive in labels
    aucs_per_species = {}
    for j, sp in enumerate(label_cols):
        y = Y[:, j]
        if y.sum() >= 3 and y.sum() < len(y):
            try:
                auc = roc_auc_score(y, ensemble[:, j])
                aucs_per_species[sp] = auc
            except ValueError:
                pass

    macro_auc = np.mean(list(aucs_per_species.values()))
    print(f'\n=== Local val results ===')
    print(f'Species evaluated: {len(aucs_per_species)}')
    print(f'Macro-AUC: {macro_auc:.4f}')
    print(f'Species >= 0.95: {sum(a >= 0.95 for a in aucs_per_species.values())}')
    print(f'Species < 0.70: {sum(a < 0.70 for a in aucs_per_species.values())}')

    # Save per-species AUCs
    df_auc = pd.DataFrame([
        {'species': sp, 'auc': auc} for sp, auc in aucs_per_species.items()
    ]).sort_values('auc')
    df_auc.to_csv(args.out, index=False)
    print(f'Saved per-species AUCs to {args.out}')
    print(f'Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()

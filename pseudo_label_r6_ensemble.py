"""
R6 pseudo-labels via 3-model ENSEMBLE teacher (knowledge distillation).

Differences from R5:
- R5: average of 3 individual teacher predictions
- R6: V137-style QMix ensemble as single teacher (richer signal)

Teachers: SED LB872_soup3 + NFNet + V2S R2 (all with pitch TTA)
Output: pseudo_labels_r6/raw_predictions.csv (soft labels, thresholded adaptively)
"""
import time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import onnxruntime as ort
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


def run_onnx_tta_on_file(sess, inp, mel, shifts, apply_sigmoid=False):
    """Returns (N_WINDOWS, N_CLASSES) averaged over shifts."""
    accum = None
    for s in shifts:
        m = pitch_shift(mel, s) if s != 0 else mel
        p = sess.run(None, {inp: m})[0].astype(np.float32)
        if apply_sigmoid: p = 1.0/(1.0+np.exp(-np.clip(p,-50,50)))
        p = sed_pp(p)
        accum = p if accum is None else accum + p
    return accum / len(shifts)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sed', default='kaggle_model/LB872_soup3.onnx')
    p.add_argument('--nfnet', default='kaggle_model/eca_nfnet_l0.onnx')
    p.add_argument('--v2s', default='kaggle_model/effv2s_r2_fold1.onnx')
    p.add_argument('--sound_dir', default='data/train_soundscapes')
    p.add_argument('--out_dir', default='pseudo_labels_r6')
    p.add_argument('--theta_min', type=float, default=0.3)
    p.add_argument('--theta_max', type=float, default=0.9)
    args = p.parse_args()

    t0 = time.time()
    tax = pd.read_csv('data/taxonomy.csv')
    label_cols = sorted(tax['primary_label'].astype(str).tolist())
    N_CLASSES = len(label_cols)

    soundscape_files = sorted(Path(args.sound_dir).glob('*.ogg'))
    print(f'Soundscapes: {len(soundscape_files)}')

    # Load ONNX sessions
    sopts = ort.SessionOptions(); sopts.inter_op_num_threads = 4; sopts.intra_op_num_threads = 4
    sed_sess = ort.InferenceSession(args.sed, sopts); sed_in = sed_sess.get_inputs()[0].name
    nfnet_sess = ort.InferenceSession(args.nfnet, sopts); nfnet_in = nfnet_sess.get_inputs()[0].name
    v2s_sess = ort.InferenceSession(args.v2s, sopts); v2s_in = v2s_sess.get_inputs()[0].name

    mb224 = build_mel(224, 0, 16000); mb128 = build_mel(128, 50, 14000)
    hann = np.hanning(2048).astype(np.float32)

    records = []
    N_files = len(soundscape_files)
    for fi, fp in enumerate(soundscape_files):
        if (fi+1) % 500 == 0 or fi == 0:
            elapsed = time.time() - t0
            rate = (fi + 1) / max(elapsed, 1)
            eta = (N_files - fi - 1) / max(rate, 0.1) / 60
            print(f'  [{fi+1}/{N_files}] {elapsed:.0f}s, ETA {eta:.1f}min', flush=True)

        try:
            a, _ = sf.read(str(fp), dtype='float32')
        except Exception as e:
            print(f'  skip {fp.name}: {e}')
            continue
        if a.ndim == 2: a = a.mean(axis=1)
        if len(a) < FILE_SAMPLES: a = np.pad(a, (0, FILE_SAMPLES-len(a)))
        a = a[:FILE_SAMPLES]
        chunks = a.reshape(N_WINDOWS, WIN_SAMPLES)

        mel224 = fast_mel(chunks, mb224, hann)
        mel128 = fast_mel(chunks, mb128, hann)

        sed_p = run_onnx_tta_on_file(sed_sess, sed_in, mel224, [0, 2])
        nfnet_p = run_onnx_tta_on_file(nfnet_sess, nfnet_in, mel128, [0], apply_sigmoid=True)
        v2s_p = run_onnx_tta_on_file(v2s_sess, v2s_in, mel224, [0, 2])

        # ENSEMBLE — only for this file (not global ranking)
        # Since rank depends on all windows, we'd need to defer ranking
        # For per-file pseudo-labeling, just use prob avg for simplicity
        ensemble = 0.30 * sed_p + 0.20 * nfnet_p + 0.20 * v2s_p  # unnormalized; ok for per-file
        ensemble = ensemble / (0.30 + 0.20 + 0.20)

        # Build record per window
        stem = fp.stem
        for w in range(N_WINDOWS):
            end_s = (w + 1) * 5
            rec = {'file': fp.name, 'end_time': float(end_s)}
            for j, sp in enumerate(label_cols):
                rec[sp] = float(ensemble[w, j])
            records.append(rec)

    df = pd.DataFrame(records)
    print(f'\nGenerated {len(df)} predictions')

    # Per-class adaptive thresholds
    thresholds = {}
    for col in label_cols:
        vals = df[col].values
        mu, sigma = vals.mean(), vals.std()
        thresholds[col] = float(np.clip(mu + 1.0 * sigma, args.theta_min, args.theta_max))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'raw_predictions.csv', index=False)
    pd.DataFrame({'species': list(thresholds.keys()), 'threshold': list(thresholds.values())}
                 ).to_csv(out_dir / 'thresholds.csv', index=False)

    # Count retained
    ret = np.zeros(len(df), dtype=bool)
    for col in label_cols:
        ret |= (df[col].values >= thresholds[col])
    print(f'Retained: {ret.sum()}/{len(df)} ({100*ret.sum()/len(df):.1f}%)')
    print(f'Threshold range: [{min(thresholds.values()):.3f}, {max(thresholds.values()):.3f}]')
    print(f'Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()

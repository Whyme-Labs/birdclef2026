"""
Diagnostic: evaluate all models on labeled train soundscapes.
This gives us ground truth AUC on actual soundscapes (same domain as test).
"""
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from src.config import Config
from src.models import SEDModel
import time


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = SEDModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt["label_cols"]


def macro_auc(y_true, y_pred):
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return np.mean(aucs) if aucs else 0.0


def predict_soundscapes(model, config, soundscape_dir, filenames, end_times, device):
    """Predict on specific windows from soundscape files."""
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate, n_fft=config.n_fft,
        hop_length=config.hop_length, n_mels=config.n_mels,
        f_min=config.fmin, f_max=config.fmax,
        power=2.0, norm="slaney", mel_scale="htk",
    )
    db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)

    chunk_samples = config.target_samples
    audio_cache = {}
    all_probs = []

    for fname, end_t in zip(filenames, end_times):
        path = soundscape_dir / fname
        if str(path) not in audio_cache:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != config.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, config.sample_rate)
            audio_cache[str(path)] = wav
            if len(audio_cache) > 100:
                del audio_cache[next(iter(audio_cache))]

        wav = audio_cache[str(path)]
        start_sample = int((end_t - 5.0) * config.sample_rate)
        chunk = wav[..., start_sample:start_sample + chunk_samples]
        if chunk.shape[-1] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

        mel = mel_tf(chunk)
        mel = db_tf(mel)
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
        mel = mel.expand(3, -1, -1).unsqueeze(0).to(device)

        with torch.no_grad():
            clip_logits, _ = model(mel)
            prob = torch.sigmoid(clip_logits).cpu().numpy()[0]
        all_probs.append(prob)

    return np.array(all_probs)


def postprocess(probs, filenames, sharpen_exp=1.5, smooth_sigma=0.7, file_max_weight=0.05):
    """Apply post-processing per file."""
    result = probs.copy()
    unique_files = list(dict.fromkeys(filenames))  # preserve order
    for f in unique_files:
        mask = [i for i, fn in enumerate(filenames) if fn == f]
        if len(mask) <= 1:
            continue
        file_probs = result[mask]
        file_max = file_probs.max(axis=0, keepdims=True)
        file_probs = file_probs + file_max_weight * file_max
        sharpened = np.power(file_probs, sharpen_exp)
        smoothed = gaussian_filter1d(sharpened, sigma=smooth_sigma, axis=0)
        file_probs = np.power(np.maximum(smoothed, 1e-10), 1.0 / sharpen_exp)
        result[mask] = file_probs
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labeled soundscape ground truth
    sl = pd.read_csv("data/train_soundscapes_labels.csv")
    taxonomy = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())

    # Check: do label_cols match sample submission columns?
    ss = pd.read_csv("data/sample_submission.csv")
    ss_cols = [c for c in ss.columns if c != "row_id"]
    sorted_ss_cols = sorted(ss_cols)
    print("=== LABEL COLUMN CHECK ===")
    print(f"Our label_cols: {len(label_cols)} species")
    print(f"Sample submission cols: {len(ss_cols)} species")
    print(f"Match (sorted): {sorted_ss_cols == label_cols}")
    print(f"Match (original order): {ss_cols == label_cols}")
    mismatched = set(label_cols) ^ set(ss_cols)
    if mismatched:
        print(f"MISMATCHED: {mismatched}")
    else:
        print("All column names match")
    print()

    # Build ground truth labels
    label_to_idx = {l: i for i, l in enumerate(label_cols)}
    filenames = sl["filename"].tolist()
    end_times = []
    for _, row in sl.iterrows():
        parts = str(row["end"]).split(":")
        end_times.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))

    y_true = np.zeros((len(sl), len(label_cols)))
    for i, (_, row) in enumerate(sl.iterrows()):
        species = str(row["primary_label"]).split(";")
        for sp in species:
            sp = sp.strip()
            if sp in label_to_idx:
                y_true[i, label_to_idx[sp]] = 1.0

    print(f"Ground truth: {len(sl)} windows, {y_true.sum():.0f} positive labels")
    print(f"Avg labels per window: {y_true.sum(1).mean():.1f}")
    print()

    # Models to evaluate
    models_to_test = {
        "B0_baseline": "checkpoints/baseline_b0_fold0/best_fold0.pt",
        "B3_teacher": "checkpoints/teacher_b3_fold0/best_fold0.pt",
        "B0_pseudo": "checkpoints/student_b0_pseudo_r1/best_fold0.pt",
    }

    soundscape_dir = Path("data/train_soundscapes")
    results = {}

    for name, ckpt_path in models_to_test.items():
        print(f"--- Evaluating {name} ---")
        model, config, model_label_cols = load_model(ckpt_path, device)

        # Verify label ordering matches
        if model_label_cols != label_cols:
            print(f"  WARNING: label_cols mismatch! Model has {len(model_label_cols)}, expected {len(label_cols)}")

        t0 = time.time()
        preds = predict_soundscapes(model, config, soundscape_dir, filenames, end_times, device)
        elapsed = time.time() - t0

        # Raw AUC
        raw_auc = macro_auc(y_true, preds)

        # With post-processing
        preds_pp = postprocess(preds, filenames)
        pp_auc = macro_auc(y_true, preds_pp)

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Raw macro AUC: {raw_auc:.4f}")
        print(f"  Post-processed AUC: {pp_auc:.4f}")
        print(f"  Prediction stats: mean={preds.mean():.4f}, std={preds.std():.4f}, max={preds.max():.4f}")
        print()
        results[name] = {"raw": preds, "pp": preds_pp, "raw_auc": raw_auc, "pp_auc": pp_auc}

        del model
        torch.cuda.empty_cache()

    # Ensemble evaluations
    print("=== ENSEMBLE EVALUATIONS ===")
    blend_configs = [
        ("B0 only", {"B0_baseline": 1.0}),
        ("B3 only", {"B3_teacher": 1.0}),
        ("Pseudo only", {"B0_pseudo": 1.0}),
        ("B0+B3 (0.5/0.5)", {"B0_baseline": 0.5, "B3_teacher": 0.5}),
        ("B0+B3 (0.4/0.6)", {"B0_baseline": 0.4, "B3_teacher": 0.6}),
        ("Current sub (0.25/0.35/0.40)", {"B0_baseline": 0.25, "B3_teacher": 0.35, "B0_pseudo": 0.40}),
        ("B0+B3 heavy (0.35/0.45/0.20)", {"B0_baseline": 0.35, "B3_teacher": 0.45, "B0_pseudo": 0.20}),
        ("No pseudo (0.4/0.6/0.0)", {"B0_baseline": 0.4, "B3_teacher": 0.6, "B0_pseudo": 0.0}),
    ]

    for desc, weights in blend_configs:
        blended = np.zeros_like(results["B0_baseline"]["raw"])
        w_sum = sum(weights.values())
        for model_name, w in weights.items():
            if w > 0:
                blended += (w / w_sum) * results[model_name]["raw"]

        raw_auc = macro_auc(y_true, blended)
        blended_pp = postprocess(blended, filenames)
        pp_auc = macro_auc(y_true, blended_pp)
        marker = " <<<" if desc == "Current sub (0.25/0.35/0.40)" else ""
        print(f"  {desc:40s}  raw={raw_auc:.4f}  pp={pp_auc:.4f}{marker}")

    # Per-class analysis for best single model
    print("\n=== PER-CLASS AUC (B0 baseline, bottom 20) ===")
    preds_b0 = results["B0_baseline"]["raw"]
    per_class_auc = {}
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        try:
            per_class_auc[label_cols[c]] = roc_auc_score(col, preds_b0[:, c])
        except ValueError:
            continue

    for sp, auc in sorted(per_class_auc.items(), key=lambda x: x[1])[:20]:
        n_pos = int(y_true[:, label_to_idx[sp]].sum())
        print(f"  {sp:15s}  AUC={auc:.4f}  n_pos={n_pos}")


if __name__ == "__main__":
    main()

"""
Evaluate student model + model soups on labeled soundscapes.
Compares: LB872, student best, and various soup blends.
"""
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.models_v2 import SEDModelV2


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


def load_model(path, device):
    model = SEDModelV2()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


def make_soup(ckpt_a, ckpt_b, alpha, device):
    """Weighted average of two model state dicts: alpha*A + (1-alpha)*B."""
    model = SEDModelV2()
    sd_a = ckpt_a["model_state_dict"]
    sd_b = ckpt_b["model_state_dict"]
    sd_soup = {}
    for key in sd_a:
        sd_soup[key] = alpha * sd_a[key].float() + (1 - alpha) * sd_b[key].float()
    model.load_state_dict(sd_soup, strict=True)
    model.to(device).eval()
    return model


def predict_soundscapes(model, soundscape_dir, labels_csv, label_cols, device):
    """Predict on labeled soundscape windows."""
    df = pd.read_csv(labels_csv)
    label_to_idx = {l: i for i, l in enumerate(label_cols)}

    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=32000, n_fft=2048, hop_length=512,
        n_mels=224, f_min=0, f_max=16000,
        power=2.0, norm="slaney", mel_scale="htk",
    )
    db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)

    audio_cache = {}
    all_preds, all_labels = [], []

    for _, row in df.iterrows():
        fname = row["filename"]
        path = soundscape_dir / fname

        # Parse time
        parts = str(row["start"]).split(":")
        start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

        # Load audio
        if str(path) not in audio_cache:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 32000:
                wav = torchaudio.functional.resample(wav, sr, 32000)
            audio_cache[str(path)] = wav
            if len(audio_cache) > 200:
                oldest = next(iter(audio_cache))
                del audio_cache[oldest]

        waveform = audio_cache[str(path)]
        start_sample = int(start_sec * 32000)
        chunk = waveform[..., start_sample:start_sample + 160000]
        if chunk.shape[-1] < 160000:
            chunk = torch.nn.functional.pad(chunk, (0, 160000 - chunk.shape[-1]))

        # Mel spectrogram
        mel = db_tf(mel_tf(chunk))
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        mel = mel.expand(3, -1, -1).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = model(mel)["clipwise_prob"].cpu().numpy()[0]
        all_preds.append(probs)

        # Labels
        label = np.zeros(len(label_cols))
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in label_to_idx:
                label[label_to_idx[sp]] = 1.0
        all_labels.append(label)

    return np.array(all_preds), np.array(all_labels)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", type=str, required=True,
                        help="Directory with student checkpoints")
    parser.add_argument("--baseline", type=str, default="kaggle_model/LB872.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taxonomy = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(taxonomy["primary_label"].astype(str).tolist())

    sl_csv = "data/train_soundscapes_labels.csv"
    sl_dir = Path("data/train_soundscapes")

    # Evaluate baseline
    print("=== Baseline (LB872) ===")
    model_base, ckpt_base = load_model(args.baseline, device)
    preds_base, labels = predict_soundscapes(model_base, sl_dir, sl_csv, label_cols, device)
    auc_base = macro_auc(labels, preds_base)
    print(f"  sl_auc = {auc_base:.4f}")
    del model_base

    # Find best student checkpoint
    student_dir = Path(args.student_dir)
    best_ckpt = student_dir / "best_fold0.pt"
    if not best_ckpt.exists():
        print(f"No best checkpoint found in {student_dir}")
        return

    print(f"\n=== Student (best) ===")
    model_stu, ckpt_stu = load_model(best_ckpt, device)
    ckpt_info = {k: v for k, v in ckpt_stu.items() if k != "model_state_dict" and k != "label_cols"}
    print(f"  Checkpoint info: {ckpt_info}")
    preds_stu, _ = predict_soundscapes(model_stu, sl_dir, sl_csv, label_cols, device)
    auc_stu = macro_auc(labels, preds_stu)
    print(f"  sl_auc = {auc_stu:.4f}")
    del model_stu

    # Prediction-level ensemble (not soup)
    print(f"\n=== Prediction Ensemble (LB872 + student) ===")
    for w in [0.9, 0.8, 0.7, 0.5]:
        preds_ens = w * preds_base + (1 - w) * preds_stu
        auc_ens = macro_auc(labels, preds_ens)
        print(f"  {w:.1f}*LB872 + {1-w:.1f}*student: sl_auc = {auc_ens:.4f}")

    # Model soup (weight-space averaging)
    print(f"\n=== Model Soup (weight-space avg) ===")
    for alpha in [0.95, 0.9, 0.85, 0.8, 0.7, 0.5]:
        soup = make_soup(ckpt_base, ckpt_stu, alpha, device)
        preds_soup, _ = predict_soundscapes(soup, sl_dir, sl_csv, label_cols, device)
        auc_soup = macro_auc(labels, preds_soup)
        print(f"  {alpha:.2f}*LB872 + {1-alpha:.2f}*student: sl_auc = {auc_soup:.4f}")
        del soup

    # SWA: average multiple epoch checkpoints
    epoch_ckpts = sorted(student_dir.glob("epoch*_fold0.pt"))
    if len(epoch_ckpts) >= 3:
        print(f"\n=== SWA (avg last 3 epoch checkpoints) ===")
        last3 = epoch_ckpts[-3:]
        sd_avg = None
        for ep_path in last3:
            ckpt = torch.load(ep_path, map_location="cpu", weights_only=False)
            if sd_avg is None:
                sd_avg = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
            else:
                for k in sd_avg:
                    sd_avg[k] += ckpt["model_state_dict"][k].float()
            print(f"  Including: {ep_path.name} (sl_auc={ckpt.get('sl_auc', '?')})")
        for k in sd_avg:
            sd_avg[k] /= len(last3)
        swa_model = SEDModelV2()
        swa_model.load_state_dict(sd_avg, strict=True)
        swa_model.to(device).eval()
        preds_swa, _ = predict_soundscapes(swa_model, sl_dir, sl_csv, label_cols, device)
        auc_swa = macro_auc(labels, preds_swa)
        print(f"  SWA sl_auc = {auc_swa:.4f}")


if __name__ == "__main__":
    main()

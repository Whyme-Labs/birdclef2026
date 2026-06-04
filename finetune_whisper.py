"""V211 Whisper-Small LoRA finetune for multi-label bird classification.

Bird Whisperer (Interspeech 2024): frozen Whisper FAILS, finetuned Whisper gives
+15% F1 over baseline. Our 30+ submissions stuck at 0.942 follow exactly this
pattern — we've only used frozen probes. This is the unlock.

Architecture:
  Whisper-Medium encoder (24 layers, dim 1024, 16 heads, 307M params)
  + LoRA r=16 on q/k/v/out projections in attention
  + Mean-pool over time → Linear(1024, 234) head
  + BCE loss (multi-label)

Inputs:
  Focal training audio + V137 ensemble pseudo-labels
  Audio: 32kHz mono → 16kHz → 30s pad/crop → log-mel n80 hop160

Output: checkpoint with LoRA-merged weights, then exportable to ONNX for Kaggle.
"""
import argparse
import os
import sys
import time
import json
from pathlib import Path

# Force unbuffered stdout so progress shows immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import soundfile as sf
import librosa
from sklearn.model_selection import StratifiedKFold

from transformers import WhisperModel, WhisperFeatureExtractor


WHISPER_MODEL = "openai/whisper-small"
WHISPER_SR = 16000
WHISPER_DURATION_S = 30  # Whisper expects 30-sec inputs (zero-padded shorter ones)
N_CLASSES = 234


class WhisperFTClassifier(nn.Module):
    """Whisper encoder + multi-label head."""

    def __init__(self, model_id=WHISPER_MODEL, n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        full_model = WhisperModel.from_pretrained(model_id)
        self.encoder = full_model.encoder  # discard decoder
        self.encoder.config.return_dict = True
        d = self.encoder.config.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, n_classes),
        )

    def forward(self, mel):
        # mel: (B, n_mels=80, 3000)
        # When peft-wrapped, self.encoder is a PeftModel; call its base_model
        # directly to keep WhisperEncoder's input_features signature.
        enc = self.encoder.base_model.model if hasattr(self.encoder, "base_model") else self.encoder
        out = enc(input_features=mel)
        h = out.last_hidden_state  # (B, T=1500, d=1024)
        pooled = h.mean(dim=1)
        return self.head(pooled)  # (B, 234) logits


def apply_lora(model, r=16, alpha=32, dropout=0.05):
    """LoRA on attention projections."""
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    # Apply only to encoder
    model.encoder = get_peft_model(model.encoder, cfg)
    return model


class BirdAudioDataset(Dataset):
    """Multi-label audio with optional V137 pseudo soft labels."""

    def __init__(self, df, audio_root, label_cols, feature_extractor, pseudo=None,
                 is_train=True, hard_label_weight=1.0, pseudo_weight=1.0):
        self.df = df.reset_index(drop=True)
        self.audio_root = Path(audio_root)
        self.label_cols = label_cols
        self.l2i = {c: i for i, c in enumerate(label_cols)}
        self.fe = feature_extractor
        self.pseudo = pseudo  # dict: (file, end_time) → np.array(234)
        self.is_train = is_train
        self.hard_label_weight = hard_label_weight
        self.pseudo_weight = pseudo_weight

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = str(row.get("filename", row.get("file", "")))
        # Try multiple roots
        candidates = [self.audio_root / rel, Path("data/train_audio") / rel,
                      Path("data_external/xc_pantanal_audio") / rel,
                      Path("data/train_soundscapes") / rel]
        # Also try basename
        for r in [self.audio_root, Path("data/train_audio"),
                  Path("data_external/xc_pantanal_audio")]:
            candidates.append(r / Path(rel).name)
        wav = None
        sr = None
        for c in candidates:
            if c.exists():
                try:
                    wav, sr = sf.read(str(c), dtype="float32")
                    break
                except Exception:
                    continue
        if wav is None:
            wav = np.zeros(WHISPER_SR * 5, dtype=np.float32)
            sr = WHISPER_SR
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != WHISPER_SR:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=WHISPER_SR)

        # Take 5s window (random crop in train, center in val) then Whisper pads to 30s
        n_target = WHISPER_SR * 5
        if len(wav) > n_target:
            if self.is_train:
                start = np.random.randint(0, len(wav) - n_target + 1)
            else:
                start = (len(wav) - n_target) // 2
            wav = wav[start:start + n_target]
        elif len(wav) < n_target:
            wav = np.pad(wav, (0, n_target - len(wav)))

        # Feature extractor returns log-mel (80, 3000) padded to 30s
        feat = self.fe(wav, sampling_rate=WHISPER_SR, return_tensors="pt")["input_features"][0]

        # Build labels (multi-label vector)
        y = np.zeros(N_CLASSES, dtype=np.float32)
        # Hard labels from primary_label column
        primary = str(row.get("primary_label", "")).strip()
        if primary and primary != "nan" and primary in self.l2i:
            y[self.l2i[primary]] = self.hard_label_weight
        # secondary_labels column may have a list
        sec = row.get("secondary_labels", "")
        if isinstance(sec, str) and sec.startswith("["):
            try:
                sec_list = eval(sec)
                for s in sec_list:
                    if s in self.l2i:
                        y[self.l2i[s]] = max(y[self.l2i[s]], 0.5)
            except Exception:
                pass

        # Optional pseudo (soft labels)
        if self.pseudo is not None:
            key = (rel, float(row.get("end_time", 5)))
            if key in self.pseudo:
                p = self.pseudo[key].astype(np.float32)
                y = self.hard_label_weight * y + self.pseudo_weight * p
                y = np.clip(y, 0, 1)

        return {"mel": feat, "label": torch.from_numpy(y)}


def asl_loss(logits, target, gamma_pos=0.0, gamma_neg=4.0, eps=1e-8):
    """Asymmetric loss for multi-label."""
    p = torch.sigmoid(logits)
    pt_pos = p * target
    pt_neg = (1 - p) * (1 - target)
    pt = pt_pos + pt_neg
    one_minus_pt = (1 - p) * target + p * (1 - target)
    g = gamma_pos * target + gamma_neg * (1 - target)
    log_pt = torch.log(pt.clamp(min=eps))
    loss = -(one_minus_pt ** g) * log_pt
    return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--out_dir", default="checkpoints/v211_whisper_small_lora")
    ap.add_argument("--use_pseudo", action="store_true", default=True,
                    help="Mix V137 ensemble pseudo-labels (soft) into targets")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data: focal training (data/train.csv) — primary_label column
    tax = pd.read_csv("data/taxonomy.csv")
    label_cols = sorted(tax["primary_label"].astype(str).tolist())
    df = pd.read_csv("data/train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    print(f"Focal train: {len(df)} rows")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(df, df["primary_label"]))[0]
    df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
    print(f"  train {len(df_train)} / val {len(df_val)}")

    # Pseudo (optional): pseudo_labels_v137/raw_predictions.csv
    pseudo_dict = None
    if args.use_pseudo and Path("pseudo_labels_v137/raw_predictions.csv").exists():
        print("Loading V137 pseudo...")
        ps = pd.read_csv("pseudo_labels_v137/raw_predictions.csv")
        pseudo_dict = {}
        for _, r in ps.iterrows():
            key = (r["file"], float(r["end_time"]))
            pseudo_dict[key] = r[label_cols].values.astype(np.float32)
        print(f"  pseudo dict: {len(pseudo_dict)} keys")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)

    train_ds = BirdAudioDataset(df_train, "data/train_audio", label_cols,
                                 feature_extractor, pseudo=pseudo_dict, is_train=True)
    val_ds = BirdAudioDataset(df_val, "data/train_audio", label_cols,
                               feature_extractor, pseudo=None, is_train=False)
    print(f"Datasets: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print(f"Loading {WHISPER_MODEL} on {device}...")
    model = WhisperFTClassifier(WHISPER_MODEL, n_classes=N_CLASSES)
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    # Gradient checkpointing on encoder to fit batch on 11GB VRAM
    model.encoder.base_model.model.gradient_checkpointing_enable()
    model.to(device)
    model.encoder.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train/1e6:.1f}M / Total: {n_total/1e6:.1f}M")

    # Param groups
    head_params = list(model.head.parameters())
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "head" not in n]
    optim = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr_lora},
         {"params": head_params, "lr": args.lr_head}],
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=len(train_loader) * args.epochs)
    scaler = GradScaler("cuda", enabled=device == "cuda")

    print("\nTraining...")
    best_val_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n = 0, 0
        optim.zero_grad()
        for step, batch in enumerate(train_loader):
            mel = batch["mel"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with autocast("cuda", enabled=device == "cuda", dtype=torch.float16):
                logits = model(mel)
                loss = asl_loss(logits, y, gamma_pos=0.0, gamma_neg=4.0) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                optim.zero_grad()
                sched.step()
            loss_sum += float(loss) * args.grad_accum; n += 1
            if step % 50 == 0:
                el = time.time() - t0
                print(f"  ep {epoch} step {step}/{len(train_loader)} "
                      f"loss={loss_sum/n:.4f} {el:.0f}s", flush=True)
        train_loss = loss_sum / max(n, 1)

        # Val
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device, non_blocking=True)
                y = batch["label"]
                with autocast("cuda", enabled=device == "cuda", dtype=torch.float16):
                    logits = model(mel)
                p = torch.sigmoid(logits).float().cpu().numpy()
                all_p.append(p); all_y.append(y.numpy())
        all_p = np.concatenate(all_p)
        all_y = np.concatenate(all_y)
        # Macro AUC on classes with positives
        from sklearn.metrics import roc_auc_score
        valid = (all_y > 0.5).sum(0) > 0
        try:
            val_auc = roc_auc_score(all_y[:, valid], all_p[:, valid], average="macro")
        except Exception:
            val_auc = 0.0
        el = time.time() - t0
        is_best = val_auc > best_val_auc
        marker = " *BEST*" if is_best else ""
        print(f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} val_auc={val_auc:.4f} "
              f"{el:.0f}s{marker}")
        # Save EVERY epoch (not just best) to debug + retry from any checkpoint
        ckpt = {
            "model_state_dict": model.state_dict(),
            "val_auc": val_auc,
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")
        if is_best:
            best_val_auc = val_auc
            torch.save(ckpt, out_dir / "best.pt")

    print(f"\nBest val_auc: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()

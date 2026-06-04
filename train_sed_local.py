"""Local SED training on GTX 1080 Ti — own SED to beat public distilled SED on rare classes.

Architecture matches public SED (EfficientNet-B0, 1-ch mel 256x313 -> clip+frame logits 234)
so the trained model is a drop-in replacement / blend partner.

Anti-collapse vs V3/V4 (which used EffV2-M, collapsed to val 0.53):
  - B0 is ~10x smaller, far more stable
  - LR warmup + cosine, gradient clip
  - correct val AUC computation (no try/except swallowing)
  - class-balanced sampling so rare classes get gradient budget

Usage: python train_sed_local.py --fold 0 --epochs 34
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DATA = "/home/soh/birdclef-2026/data"
CKPT_DIR = "/home/soh/birdclef-2026/checkpoints/sed_local"
os.makedirs(CKPT_DIR, exist_ok=True)

SR = 32000
WINDOW_SEC = 5
WS = SR * WINDOW_SEC
N_MELS = 256
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 16000
N_TIME = 313
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=34)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone", type=str, default="tf_efficientnet_b0.ns_jft_in1k")
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


# ── Mel
def make_mel(wav):
    s = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    s = librosa.power_to_db(s, top_db=80)
    s = (s - s.mean()) / (s.std() + 1e-6)
    if s.shape[-1] < N_TIME:
        s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
    else:
        s = s[:, :N_TIME]
    return s.astype(np.float32)


class SEDDataset(Dataset):
    def __init__(self, df, Y, training):
        self.df = df.reset_index(drop=True)
        self.Y = Y
        self.training = training

    def __len__(self):
        return len(self.df)

    def _load_wav(self, fn):
        try:
            wav, sr = sf.read(f"{DATA}/train_audio/{fn}", dtype="float32", always_2d=False)
        except Exception:
            return None
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav

    def _augment(self, wav):
        # random gain +/-6 dB
        wav = wav * (10 ** (np.random.uniform(-0.3, 0.3)))
        # random soft EQ filter (channel distortion sim)
        if np.random.rand() < 0.3:
            spec = np.fft.rfft(wav)
            freqs = np.fft.rfftfreq(len(wav), d=1.0 / SR)
            eq = np.ones_like(freqs)
            for _ in range(np.random.randint(1, 3)):
                fc = np.random.uniform(500, 8000)
                bw = np.random.uniform(500, 2000)
                g = np.random.uniform(0.5, 1.5)
                eq *= 1 + (g - 1) * np.exp(-((freqs - fc) ** 2) / (2 * bw ** 2))
            wav = np.fft.irfft(spec * eq, n=len(wav)).astype(np.float32)
        return wav

    def __getitem__(self, i):
        fn = self.df.iloc[i]["filename"]
        y = self.Y[i].astype(np.float32)
        wav = self._load_wav(fn)
        if wav is None or len(wav) < 100:
            return torch.zeros(1, N_MELS, N_TIME), torch.from_numpy(y)
        n = len(wav)
        if n < WS:
            wav = np.tile(wav, (WS // n) + 1)[:WS]
        else:
            if self.training:
                st = np.random.randint(0, n - WS + 1)
            else:
                st = max(0, (n - WS) // 2)
            wav = wav[st:st + WS]
        if self.training:
            wav = self._augment(wav)
        mel = make_mel(wav)
        m = torch.from_numpy(mel).unsqueeze(0)  # (1, 256, 313)
        if self.training:
            # SpecAugment
            if np.random.rand() < 0.6:
                f0 = np.random.randint(0, N_MELS - 24)
                m[:, f0:f0 + np.random.randint(8, 24), :] = 0
            if np.random.rand() < 0.6:
                t0 = np.random.randint(0, N_TIME - 40)
                m[:, :, t0:t0 + np.random.randint(15, 40)] = 0
        return m, torch.from_numpy(y)


class SEDModel(nn.Module):
    def __init__(self, backbone_name, n_classes=234):
        super().__init__()
        self.bk = timm.create_model(backbone_name, pretrained=True, num_classes=0,
                                     global_pool="", in_chans=1)
        d = self.bk.num_features
        self.gem_p = nn.Parameter(torch.tensor(3.0))
        self.drop = nn.Dropout(0.3)
        self.frame_head = nn.Linear(d, n_classes)
        self.att_head = nn.Linear(d, n_classes)

    def forward(self, x):
        f = self.bk(x)                       # (B, D, H', W')
        p = self.gem_p.clamp(min=1.0)
        f_freq = (f.clamp(min=1e-6).pow(p)).mean(dim=2).pow(1.0 / p)  # (B, D, W')
        f_t = f_freq.transpose(1, 2)         # (B, W', D)
        f_t = self.drop(f_t)
        frame_logits = self.frame_head(f_t)  # (B, W', C)
        att = torch.softmax(self.att_head(f_t), dim=1)
        clip_logits = (att * frame_logits).sum(dim=1)  # (B, C)
        return clip_logits, frame_logits


def focal_bce(logits, targets, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    return (ce * (1 - pt).pow(gamma)).mean()


def main():
    args = parse_args()
    print(f"=== Local SED training | fold {args.fold} | {args.backbone} ===", flush=True)
    torch.manual_seed(42 + args.fold)
    np.random.seed(42 + args.fold)

    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {l: i for i, l in enumerate(PRIMARY)}
    NC = len(PRIMARY)

    df = pd.read_csv(f"{DATA}/train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df = df.reset_index(drop=True)

    # Build multi-hot labels (primary=1.0, secondary=0.5)
    Y = np.zeros((len(df), NC), dtype=np.float32)
    for i, r in df.iterrows():
        pl = r["primary_label"]
        if pl in l2i:
            Y[i, l2i[pl]] = 1.0
        sec = r.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec.startswith("["):
            try:
                for s in eval(sec):
                    s = str(s).strip()
                    if s in l2i:
                        Y[i, l2i[s]] = max(Y[i, l2i[s]], 0.5)
            except Exception:
                pass
    print(f"{len(df)} recordings, {NC} classes, "
          f"{(Y.sum(0) > 0).sum()} classes with positives", flush=True)

    # 5-fold split, stratified by primary, grouped by author
    groups = df["author"].fillna("unk").astype(str).values if "author" in df.columns else np.arange(len(df))
    strat = df["primary_label"].values
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(df, strat, groups))
    tr_idx, va_idx = folds[args.fold]
    print(f"fold {args.fold}: train={len(tr_idx)} val={len(va_idx)}", flush=True)

    tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]
    tr_Y, va_Y = Y[tr_idx], Y[va_idx]

    # class-balanced sampler: inverse-sqrt frequency on primary label
    prim_idx = np.array([l2i.get(p, 0) for p in tr_df["primary_label"]])
    cls_count = np.bincount(prim_idx, minlength=NC).astype(np.float32)
    cls_w = 1.0 / np.sqrt(cls_count + 1.0)
    samp_w = cls_w[prim_idx]
    sampler = WeightedRandomSampler(samp_w, num_samples=len(tr_df), replacement=True)

    tr_ds = SEDDataset(tr_df, tr_Y, training=True)
    va_ds = SEDDataset(va_df, va_Y, training=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    model = SEDModel(args.backbone, NC).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params: {n_par:.1f}M", flush=True)

    bk_params = list(model.bk.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("bk.")]
    opt = torch.optim.AdamW([
        {"params": bk_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    steps = args.epochs * len(tr_ld)
    warmup = len(tr_ld)  # 1 epoch warmup
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = GradScaler()

    best_auc = 0.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for step, (x, y) in enumerate(tr_ld):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            # mixup
            if np.random.rand() < 0.5:
                lam = np.random.beta(0.4, 0.4)
                perm = torch.randperm(x.size(0), device=DEVICE)
                x = lam * x + (1 - lam) * x[perm]
                y = torch.maximum(y, y[perm])
            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                clip_logits, frame_logits = model(x)
                frame_max = frame_logits.max(dim=1).values
                loss = focal_bce(clip_logits, y) + 0.5 * focal_bce(frame_max, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            n += 1
            if step % 100 == 0:
                print(f"  ep{ep+1} step {step}/{len(tr_ld)} loss={tot/n:.4f} "
                      f"t={time.time()-t0:.0f}s", flush=True)
        # validate
        model.eval()
        ap, ay = [], []
        with torch.no_grad():
            for x, y in va_ld:
                x = x.to(DEVICE, non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    clip_logits, _ = model(x)
                ap.append(torch.sigmoid(clip_logits).float().cpu().numpy())
                ay.append(y.numpy())
        AP, AY = np.concatenate(ap), np.concatenate(ay)
        valid = (AY > 0.5).sum(0) > 0
        try:
            va_auc = roc_auc_score((AY[:, valid] > 0.5).astype(int), AP[:, valid], average="macro")
        except Exception as e:
            va_auc = 0.0
            print(f"  auc error: {e}", flush=True)
        print(f"[ep {ep+1}/{args.epochs}] loss={tot/n:.4f} val_macroAUC={va_auc:.4f} "
              f"t={time.time()-t0:.0f}s  (classes eval={valid.sum()})", flush=True)
        ckpt = {"state": model.state_dict(), "epoch": ep + 1, "val_auc": va_auc,
                "backbone": args.backbone, "fold": args.fold}
        torch.save(ckpt, f"{CKPT_DIR}/fold{args.fold}_last.pt")
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(ckpt, f"{CKPT_DIR}/fold{args.fold}_best.pt")
            print(f"  -> new best {best_auc:.4f}", flush=True)

    print(f"=== DONE fold {args.fold}: best val_macroAUC={best_auc:.4f} ===", flush=True)


if __name__ == "__main__":
    main()

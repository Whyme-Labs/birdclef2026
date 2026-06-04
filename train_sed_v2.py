"""SED v2 — soundscape-native training. Fixes V256 (focal-only -> 0.931 domain-shift fail).

Trains on a MIX of:
  - focal recordings (hard multi-hot labels)         -> rare-species identity coverage
  - pseudo-labeled soundscape windows (soft labels)  -> soundscape domain adaptation
  - 1478 truly-labeled soundscape windows (hard)     -> real soundscape anchor

Per batch ~ 40% focal / 50% pseudo-soundscape / 10% labeled-soundscape.
Architecture identical to v1 (EfficientNet-B0, 1-ch mel 256x313, clip+frame heads)
so the ONNX is a drop-in blend partner for the public SED.

Usage: python train_sed_v2.py --fold 0 --epochs 22
"""
import os
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
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DATA = "/home/soh/birdclef-2026/data"
CKPT_DIR = "/home/soh/birdclef-2026/checkpoints/sed_v2"
PSEUDO_NPZ = "/home/soh/birdclef-2026/soundscape_pseudo.npz"
os.makedirs(CKPT_DIR, exist_ok=True)

SR = 32000
WS = SR * 5
N_MELS, N_FFT, HOP, FMIN, FMAX, N_TIME = 256, 2048, 512, 20, 16000, 313
DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=22)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--backbone", type=str, default="tf_efficientnet_b0.ns_jft_in1k")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--steps_per_epoch", type=int, default=900)
    return p.parse_args()


def make_mel(wav):
    s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    s = librosa.power_to_db(s, top_db=80)
    s = (s - s.mean()) / (s.std() + 1e-6)
    if s.shape[-1] < N_TIME:
        s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
    else:
        s = s[:, :N_TIME]
    return s.astype(np.float32)


def parse_t(t):
    p = str(t).split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])


class MixDataset(Dataset):
    """Yields a stream of mixed focal / pseudo-soundscape / labeled-soundscape samples."""
    def __init__(self, focal_df, focal_Y, focal_w, ss_files, ss_win, ss_probs,
                 lab_rows, n_steps, batch_size):
        self.focal_df = focal_df.reset_index(drop=True)
        self.focal_Y = focal_Y
        self.focal_w = focal_w / focal_w.sum()
        self.ss_files = ss_files
        self.ss_win = ss_win
        self.ss_probs = ss_probs
        self.lab_rows = lab_rows  # list of (filename, start_s, end_s, Y_vec)
        self.n = n_steps * batch_size
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return self.n

    def _focal(self):
        i = np.random.choice(len(self.focal_df), p=self.focal_w)
        fn = self.focal_df.iloc[i]["filename"]
        y = self.focal_Y[i].astype(np.float32)
        try:
            wav, sr = sf.read(f"{DATA}/train_audio/{fn}", dtype="float32", always_2d=False)
        except Exception:
            return np.zeros((N_MELS, N_TIME), np.float32), y, 1.0
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        n = len(wav)
        if n < WS:
            wav = np.tile(wav, (WS // n) + 1)[:WS]
        else:
            st = np.random.randint(0, n - WS + 1)
            wav = wav[st:st + WS]
        return make_mel(self._aug(wav)), y, 1.0

    def _soundscape(self):
        j = np.random.randint(len(self.ss_files))
        fn = self.ss_files[j]
        wi = int(self.ss_win[j])
        y = self.ss_probs[j].astype(np.float32)
        try:
            wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
        except Exception:
            return np.zeros((N_MELS, N_TIME), np.float32), y, 1.0
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        seg = wav[wi * WS:(wi + 1) * WS]
        if len(seg) < WS:
            seg = np.pad(seg, (0, WS - len(seg)))
        return make_mel(self._aug(seg)), y, 1.0

    def _labeled(self):
        fn, st_s, en_s, y = self.lab_rows[np.random.randint(len(self.lab_rows))]
        try:
            wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
        except Exception:
            return np.zeros((N_MELS, N_TIME), np.float32), y, 1.0
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        seg = wav[int(st_s * SR):int(en_s * SR)]
        if len(seg) < WS:
            seg = np.pad(seg, (0, WS - len(seg)))
        else:
            seg = seg[:WS]
        return make_mel(self._aug(seg)), y.astype(np.float32), 1.0

    def _aug(self, wav):
        wav = wav * (10 ** (np.random.uniform(-0.3, 0.3)))
        return wav

    def __getitem__(self, idx):
        r = np.random.rand()
        if r < 0.40:
            mel, y, w = self._focal()
        elif r < 0.90:
            mel, y, w = self._soundscape()
        else:
            mel, y, w = self._labeled()
        m = torch.from_numpy(mel).unsqueeze(0)
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
        f = self.bk(x)
        p = self.gem_p.clamp(min=1.0)
        f_freq = (f.clamp(min=1e-6).pow(p)).mean(dim=2).pow(1.0 / p)
        f_t = self.drop(f_freq.transpose(1, 2))
        frame_logits = self.frame_head(f_t)
        att = torch.softmax(self.att_head(f_t), dim=1)
        clip_logits = (att * frame_logits).sum(dim=1)
        return clip_logits, frame_logits


def focal_bce(logits, targets, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = p * targets + (1 - p) * (1 - targets)
    return (ce * (1 - pt).pow(gamma)).mean()


def main():
    a = parse_args()
    print(f"=== SED v2 soundscape-native | fold {a.fold} ===", flush=True)
    torch.manual_seed(42 + a.fold); np.random.seed(42 + a.fold)

    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {l: i for i, l in enumerate(PRIMARY)}
    NC = len(PRIMARY)

    # focal
    df = pd.read_csv(f"{DATA}/train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df = df.reset_index(drop=True)
    Y = np.zeros((len(df), NC), dtype=np.float32)
    for i, r in df.iterrows():
        if r["primary_label"] in l2i:
            Y[i, l2i[r["primary_label"]]] = 1.0
        sec = r.get("secondary_labels", "[]")
        if isinstance(sec, str) and sec.startswith("["):
            try:
                for s in eval(sec):
                    s = str(s).strip()
                    if s in l2i:
                        Y[i, l2i[s]] = max(Y[i, l2i[s]], 0.5)
            except Exception:
                pass

    groups = df["author"].fillna("unk").astype(str).values
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, df["primary_label"], groups))
    tr_idx, va_idx = folds[a.fold]
    tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]
    tr_Y, va_Y = Y[tr_idx], Y[va_idx]
    prim = np.array([l2i.get(p, 0) for p in tr_df["primary_label"]])
    cnt = np.bincount(prim, minlength=NC).astype(np.float32)
    focal_w = (1.0 / np.sqrt(cnt + 1.0))[prim]

    # pseudo soundscapes
    pz = np.load(PSEUDO_NPZ, allow_pickle=True)
    ss_files = pz["files"]; ss_win = pz["win_idx"]; ss_probs = pz["probs"].astype(np.float32)
    print(f"pseudo soundscapes: {ss_probs.shape}", flush=True)

    # labeled soundscapes (hard)
    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv")
    lab_rows = []
    for _, r in lab.iterrows():
        yv = np.zeros(NC, dtype=np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in l2i:
                yv[l2i[s]] = 1.0
        lab_rows.append((str(r["filename"]), parse_t(r["start"]), parse_t(r["end"]), yv))
    print(f"labeled soundscape rows: {len(lab_rows)}", flush=True)

    tr_ds = MixDataset(tr_df, tr_Y, focal_w, ss_files, ss_win, ss_probs, lab_rows,
                       a.steps_per_epoch, a.batch_size)
    tr_ld = DataLoader(tr_ds, batch_size=a.batch_size, shuffle=False,
                       num_workers=a.num_workers, pin_memory=True, drop_last=True,
                       persistent_workers=True)

    # validation: focal held-out fold (sanity), plus labeled-soundscape AUC (real signal)
    class ValDS(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            fn, st, en, y = self.rows[i]
            try:
                wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                seg = wav[int(st*SR):int(en*SR)]
                if len(seg) < WS: seg = np.pad(seg, (0, WS-len(seg)))
                else: seg = seg[:WS]
                return torch.from_numpy(make_mel(seg)).unsqueeze(0), torch.from_numpy(y)
            except Exception:
                return torch.zeros(1, N_MELS, N_TIME), torch.from_numpy(y)
    va_ld = DataLoader(ValDS(lab_rows), batch_size=96, shuffle=False, num_workers=6)

    model = SEDModel(a.backbone, NC).to(DEVICE)
    # warm-start from v1 focal checkpoint if available
    v1 = f"/home/soh/birdclef-2026/checkpoints/sed_local/fold{a.fold}_best.pt"
    if os.path.exists(v1):
        try:
            model.load_state_dict(torch.load(v1, map_location="cpu")["state"])
            print(f"warm-started from v1 {v1}", flush=True)
        except Exception as e:
            print(f"warm-start skipped: {e}", flush=True)

    bk = list(model.bk.parameters())
    hd = [p for n, p in model.named_parameters() if not n.startswith("bk.")]
    opt = torch.optim.AdamW([{"params": bk, "lr": a.lr * 0.1},
                             {"params": hd, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * len(tr_ld)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=a.lr * 0.02)
    scaler = GradScaler()

    best = 0.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = n = 0
        for step, (x, y) in enumerate(tr_ld):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if np.random.rand() < 0.5:
                lam = np.random.beta(0.4, 0.4)
                perm = torch.randperm(x.size(0), device=DEVICE)
                x = lam * x + (1 - lam) * x[perm]
                y = torch.maximum(y, y[perm])
            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                clip, frame = model(x)
                fm = frame.max(dim=1).values
                loss = focal_bce(clip, y) + 0.5 * focal_bce(fm, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
        # validate on labeled soundscapes
        model.eval(); ap, ay = [], []
        with torch.no_grad():
            for x, y in va_ld:
                x = x.to(DEVICE, non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    clip, _ = model(x)
                ap.append(torch.sigmoid(clip).float().cpu().numpy())
                ay.append(y.numpy())
        AP, AY = np.concatenate(ap), np.concatenate(ay)
        v = (AY > 0.5).sum(0) > 0
        try:
            auc = roc_auc_score((AY[:, v] > 0.5).astype(int), AP[:, v], average="macro")
        except Exception:
            auc = 0.0
        print(f"[ep {ep+1}/{a.epochs}] loss={tot/n:.4f} ss_val_macroAUC={auc:.4f} "
              f"t={time.time()-t0:.0f}s (cls={v.sum()})", flush=True)
        ck = {"state": model.state_dict(), "epoch": ep+1, "ss_val_auc": auc,
              "backbone": a.backbone, "fold": a.fold}
        torch.save(ck, f"{CKPT_DIR}/fold{a.fold}_last.pt")
        if auc > best:
            best = auc
            torch.save(ck, f"{CKPT_DIR}/fold{a.fold}_best.pt")
            print(f"  -> best {best:.4f}", flush=True)
    print(f"=== DONE fold {a.fold}: best ss_val_macroAUC={best:.4f} ===", flush=True)


if __name__ == "__main__":
    main()

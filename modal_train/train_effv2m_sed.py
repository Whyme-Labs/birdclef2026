"""Train EffV2-M SED from scratch on labeled soundscape + focal recordings.
Goal: produce a SED model that REPLACES tuckerarrants' public distilled SED in the kernel.

Decorrelation strategy: this is fundamentally different from all our ProtoSSM sidecar
experiments. The public SED is shared across every 0.947 fork — having OUR OWN SED
gives us a genuinely decorrelated component that the leaderboard top teams have.

Modal A100, target ~6-10 hours for 25 epochs on 50k corpus.
"""
import modal
app = modal.App("birdclef-effv2m-sed")
vol = modal.Volume.from_name("birdclef-effv2m-artifacts", create_if_missing=True)
audio_vol = modal.Volume.from_name("birdclef-audio-data", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.3.0", "torchaudio==2.3.0", "timm==1.0.11",
                 "numpy", "pandas", "scikit-learn", "soundfile", "librosa",
                 "kaggle"))


@app.function(image=image, volumes={"/audio": audio_vol}, timeout=2*3600, cpu=4)
def download_data(kaggle_token_b64: bytes):
    """One-time: download BirdCLEF 2026 audio to Modal volume."""
    import os, base64, subprocess
    os.makedirs("/root/.kaggle", exist_ok=True)
    open("/root/.kaggle/kaggle.json", "wb").write(kaggle_token_b64)
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    os.chdir("/audio")
    if not os.path.exists("birdclef-2026.zip") and not os.path.exists("data/train_audio"):
        print("Downloading birdclef-2026...")
        subprocess.run(["kaggle","competitions","download","-c","birdclef-2026","-p","/audio"], check=True)
    if not os.path.exists("data/train_audio"):
        print("Extracting...")
        subprocess.run(["unzip","-q","/audio/birdclef-2026.zip","-d","/audio/data"], check=True)
    print("audio root contents:", os.listdir("/audio"))
    if os.path.exists("/audio/data/train_audio"):
        n = sum(1 for _ in os.scandir("/audio/data/train_audio"))
        print(f"train_audio species dirs: {n}")
    audio_vol.commit()


@app.function(image=image, gpu="A100", volumes={"/vol": vol, "/audio": audio_vol},
              timeout=12*3600, cpu=4)
def train_effv2m(n_epochs: int = 20, batch_size: int = 32, lr: float = 5e-4):
    import os, time, numpy as np, pandas as pd
    import torch, torch.nn as nn, torchaudio, timm
    import soundfile as sf
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import autocast, GradScaler

    SR = 32_000; WINDOW_SAMPLES = SR * 5; N_WINDOWS = 12
    DATA_ROOT = "/audio/data"
    tax = pd.read_csv(f"{DATA_ROOT}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    N_CLASSES = len(PRIMARY)
    l2i = {l:i for i,l in enumerate(PRIMARY)}
    print(f"N_CLASSES={N_CLASSES}")
    df = pd.read_csv(f"{DATA_ROOT}/train.csv")
    df["primary_label"] = df["primary_label"].astype(str)
    print(f"train.csv rows={len(df)}")

    # Multi-hot labels
    Y = np.zeros((len(df), N_CLASSES), dtype=np.float32)
    for i, row in df.iterrows():
        p = row["primary_label"]
        if p in l2i: Y[i, l2i[p]] = 1.0
        # secondary
        sec = row.get("secondary_labels", "")
        if isinstance(sec, str) and sec.startswith("["):
            try:
                for s in eval(sec):
                    s = str(s).strip()
                    if s in l2i: Y[i, l2i[s]] = 0.5
            except: pass

    class FocalDS(Dataset):
        def __init__(self, df, Y, root):
            self.df = df; self.Y = Y; self.root = root
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=2048, hop_length=512, n_mels=224,
                f_min=0, f_max=16000, power=2.0, norm="slaney", mel_scale="htk")
            self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            fn = self.df.iloc[idx]["filename"]
            try:
                wav, sr = sf.read(f"{self.root}/train_audio/{fn}", dtype="float32", always_2d=False)
            except Exception:
                wav = np.zeros(WINDOW_SAMPLES, np.float32); sr = SR
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            n = len(wav)
            if n < WINDOW_SAMPLES:
                reps = (WINDOW_SAMPLES // n) + 1
                wav = np.tile(wav, reps)[:WINDOW_SAMPLES]
            else:
                # random 5s crop in training
                start = np.random.randint(0, n - WINDOW_SAMPLES + 1)
                wav = wav[start:start+WINDOW_SAMPLES]
            w = torch.from_numpy(wav).float().unsqueeze(0)
            m = self.db(self.mel(w))
            mn, mx = m.min(), m.max()
            m = (m - mn) / (mx - mn + 1e-8)
            m = m.expand(3, -1, -1)
            return m, torch.from_numpy(self.Y[idx]).float()

    ds = FocalDS(df, Y, DATA_ROOT)
    print(f"dataset size: {len(ds)}")
    # 90/10 split
    n_val = len(ds) // 10
    idx = np.arange(len(ds)); np.random.default_rng(42).shuffle(idx)
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    from torch.utils.data import Subset
    train_ds = Subset(ds, tr_idx); val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    # Build SED head over EffV2-M backbone
    class SED(nn.Module):
        def __init__(self):
            super().__init__()
            self.bk = timm.create_model("tf_efficientnetv2_m.in21k", pretrained=True, num_classes=0, global_pool="")
            d = self.bk.num_features
            # GEM pool over frequency
            self.gem_p = nn.Parameter(torch.tensor(3.0))
            self.head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, N_CLASSES))
        def forward(self, x):
            f = self.bk(x)  # (B, C, H, W)
            # GEM over freq (H)
            p = self.gem_p.clamp(min=1.0)
            f = (f.clamp(min=1e-6) ** p).mean(dim=2) ** (1.0/p)  # (B, C, W)
            # Avg over time
            f = f.mean(dim=2)  # (B, C)
            return self.head(f)
    device = "cuda"
    model = SED().to(device)
    print(f"model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    # Diff LR
    bk_params = list(model.bk.parameters())
    head_params = list(model.gem_p.unsqueeze(0)) + list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": bk_params, "lr": lr*0.1},
        {"params": head_params, "lr": lr}], weight_decay=1e-5)
    steps = n_epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr*0.1, lr], total_steps=steps, pct_start=0.05)
    scaler = GradScaler()
    # ASL-like loss
    def asl_loss(logits, target):
        # logits: (B, C). target: (B, C) in [0,1]
        p = torch.sigmoid(logits)
        eps = 1e-8
        pos = -target * torch.log(p.clamp(min=eps))  # positive part
        neg = -(1-target) * ((p ** 4) * torch.log((1-p).clamp(min=eps)))  # gamma_neg=4
        return (pos + neg).mean()

    from sklearn.metrics import roc_auc_score
    os.makedirs("/vol/eff", exist_ok=True)
    best_val = 0.0
    for ep in range(n_epochs):
        model.train()
        t0 = time.time(); loss_sum, n = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                out = model(x); loss = asl_loss(out, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item(); n += 1
            if step % 100 == 0:
                print(f"  ep{ep+1} step {step}/{len(train_loader)} loss={loss_sum/n:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        # val
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    out = model(x)
                p = torch.sigmoid(out).float().cpu().numpy()
                all_p.append(p); all_y.append(y.numpy())
        ap = np.concatenate(all_p); ay = np.concatenate(all_y)
        valid = (ay > 0.5).sum(0) > 0
        try: va = roc_auc_score(ay[:, valid], ap[:, valid], average="macro")
        except: va = 0.0
        print(f"Epoch {ep+1}/{n_epochs}: train_loss={loss_sum/n:.4f} val_macroAUC={va:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        ckpt = {"state": model.state_dict(), "epoch": ep+1, "val_auc": va}
        torch.save(ckpt, f"/vol/eff/effv2m_sed_ep{ep+1}.pt")
        if va > best_val:
            best_val = va
            torch.save(ckpt, "/vol/eff/effv2m_sed_best.pt")
        vol.commit()
    print(f"DONE best_val_macroAUC={best_val:.4f}")
    return best_val


@app.local_entrypoint()
def main():
    import sys, os
    token_path = os.path.expanduser("~/.kaggle/kaggle.json")
    token = open(token_path, "rb").read()
    # Step 1: download data to volume (skips if already there)
    print("Step 1: download data")
    download_data.remote(token)
    print("Step 2: train")
    auc = train_effv2m.remote(n_epochs=20)
    print(f"Best val_macroAUC: {auc}")

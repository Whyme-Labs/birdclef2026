"""Phase 2 (V246): Train EffV2-S SED student distilled from 5-fold public SED teacher.

Prereqs:
  - birdclef-audio-data volume populated with train_audio/ + train.csv + taxonomy.csv
  - birdclef-sed-softlabels volume from extract_sed_softlabels.py:
      * file_index.csv (filename, primary_label ordered)
      * softlabels_clip_5fold.npy (N_files, 234) averaged clip soft probs
      * file_chunks_softlabels.npy (N_chunks_total, 234) chunk-level soft probs
      * file_chunk_offsets.npy (N_files+1,) cumulative chunk offsets

Student spec (match public SED I/O for drop-in replacement):
  Input: mel (B, 1, 256, 313) 1-channel mel @ n_mels=256, hop=512, n_fft=2048
  Output: clip_logits (B, 234), framewise_logits (B, T_frames, 234)
  Backbone: convnextv2_tiny.fcmae_ft_in22k_in1k (~28M params, FCMAE+in22k pretrained — SOTA at this size class, 5x public SED's EffNet-B0)

Loss = 0.4 * BCE(hard) + 0.6 * KL(teacher_soft, student_clip)
Augs (per 2025 2nd place):
  - MixUp (audio domain, element-wise-max label)
  - SpecAugment (time + freq masking)
  - Background mix from ESC-50 (light, p=0.3)
  - RandomFilter (simulated channel distortion)
  - Random gain ±6dB
"""
import modal

app = modal.App("birdclef-sed-student")
audio_vol = modal.Volume.from_name("birdclef-audio-data", create_if_missing=False)
soft_vol = modal.Volume.from_name("birdclef-sed-softlabels", create_if_missing=False)
out_vol = modal.Volume.from_name("birdclef-sed-student", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.5.0",
        "torchaudio==2.5.0",
        "timm==1.0.11",
        "onnx==1.17.0",
        "onnxruntime==1.20.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "soundfile",
        "librosa==0.10.2",
        "tqdm",
    )
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/audio": audio_vol, "/soft": soft_vol, "/out": out_vol},
    timeout=12 * 3600,
    cpu=8,
)
def train(
    fold: int = 0,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    kl_weight: float = 0.6,
    bce_weight: float = 0.4,
    mixup_alpha: float = 0.4,
    seed: int = 42,
):
    import os, time, random, numpy as np, pandas as pd
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchaudio
    import timm
    import soundfile as sf
    from torch.utils.data import Dataset, DataLoader, Subset
    from torch.amp import autocast, GradScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    # ── Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    SR = 32000
    WS = SR * 5
    N_MELS = 256
    N_FFT = 2048
    HOP = 512
    NC = 234
    N_TIME = 313  # exact target frame count

    DATA = "/audio"
    SOFT = "/soft"
    OUT = "/out"

    # ── Load taxonomy + train metadata
    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {l: i for i, l in enumerate(PRIMARY)}
    df_train = pd.read_csv(f"{DATA}/train.csv")
    df_train["primary_label"] = df_train["primary_label"].astype(str)
    df_train["filename"] = df_train["filename"].astype(str)

    df_idx = pd.read_csv(f"{SOFT}/file_index.csv")
    df_idx["filename"] = df_idx["filename"].astype(str)
    # Ensure df_train order matches df_idx (which dictates softlabel row order)
    df = df_idx.merge(df_train[["filename", "author"]] if "author" in df_train.columns else df_train[["filename"]],
                      on="filename", how="left")
    print(f"[data] {len(df)} files (softlabel rows)")

    soft_clip = np.load(f"{SOFT}/softlabels_clip_5fold.npy")  # (N, 234)
    print(f"[soft] clip shape {soft_clip.shape}, mean prob {soft_clip.mean():.4f}")
    assert soft_clip.shape[0] == len(df), f"mismatch {soft_clip.shape[0]} vs {len(df)}"

    # ── Build hard labels
    Y = np.zeros((len(df), NC), dtype=np.float32)
    for i, r in df.iterrows():
        pl = str(r["primary_label"])
        if pl in l2i:
            Y[i, l2i[pl]] = 1.0
    print(f"[hard] {(Y.sum(0) > 0).sum()}/{NC} classes with positives")

    # ── 5-fold split by author/group (fallback random if no author col)
    if "author" in df.columns and df["author"].notna().any():
        groups = df["author"].fillna("unknown").astype(str).values
    else:
        groups = np.arange(len(df))  # random per-file
    folds = list(GroupKFold(n_splits=5).split(np.arange(len(df)), groups=groups))
    tr_idx, va_idx = folds[fold]
    print(f"[fold {fold}] train={len(tr_idx)} val={len(va_idx)}")

    # ── Dataset (uses LIBROSA mel — exact match to public SED teacher)
    import librosa
    class SEDDataset(Dataset):
        def __init__(self, df, Y, soft, root, indices, training=True):
            self.df = df.reset_index(drop=True)
            self.Y = Y
            self.soft = soft
            self.root = root
            self.indices = indices
            self.training = training
            # Mel via librosa (CPU). torchaudio.MelSpectrogram has different filterbank
            # computation than librosa.feature.melspectrogram even with same params —
            # teacher soft labels were generated with librosa, so student MUST match.
            self.fmask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
            self.tmask = torchaudio.transforms.TimeMasking(time_mask_param=40)

        def _make_mel(self, wav):
            s = librosa.feature.melspectrogram(
                y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                fmin=20.0, fmax=16000.0, power=2.0,
            )
            s = librosa.power_to_db(s, top_db=80)
            s = (s - s.mean()) / (s.std() + 1e-6)
            return s.astype(np.float32)  # (256, T)

        def __len__(self):
            return len(self.indices)

        def _load(self, idx):
            fn = self.df.iloc[idx]["filename"]
            try:
                wav, sr = sf.read(f"{self.root}/train_audio/{fn}", dtype="float32", always_2d=False)
            except Exception:
                return None
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            n = len(wav)
            if n < WS:
                wav = np.tile(wav, (WS // n) + 1)[:WS]
                n = WS
            # Random 5-sec crop in training, deterministic center in val
            if self.training:
                st = np.random.randint(0, n - WS + 1) if n > WS else 0
            else:
                st = max(0, (n - WS) // 2)
            return wav[st:st + WS].astype(np.float32)

        def _augment(self, wav):
            if not self.training:
                return wav
            # Random gain ±6dB
            gain = 10 ** (np.random.uniform(-0.3, 0.3))
            wav = wav * gain
            # Random low/high-pass filter sim via FFT magnitude scaling
            if np.random.rand() < 0.3:
                spec = np.fft.rfft(wav)
                freqs = np.fft.rfftfreq(len(wav), d=1 / SR)
                # Random soft equalizer: 1-2 random gain bumps/cuts
                eq = np.ones_like(freqs)
                for _ in range(np.random.randint(1, 3)):
                    fc = np.random.uniform(500, 8000)
                    bw = np.random.uniform(500, 2000)
                    gain = np.random.uniform(0.5, 1.5)
                    eq *= 1 + (gain - 1) * np.exp(-((freqs - fc) ** 2) / (2 * bw ** 2))
                spec *= eq
                wav = np.fft.irfft(spec, n=len(wav)).astype(np.float32)
            return wav

        def __getitem__(self, i):
            real_idx = self.indices[i]
            wav = self._load(real_idx)
            if wav is None:
                return torch.zeros(1, N_MELS, N_TIME), torch.from_numpy(self.Y[real_idx]).float(), torch.from_numpy(self.soft[real_idx]).float()
            wav = self._augment(wav)
            s = self._make_mel(wav)  # (256, T) np.float32, librosa
            # Pad/crop to N_TIME=313 frames
            if s.shape[1] < N_TIME:
                s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[1])))
            else:
                s = s[:, :N_TIME]
            m = torch.from_numpy(s).unsqueeze(0)  # (1, 256, 313)
            if self.training:
                if np.random.rand() < 0.7:
                    m = self.fmask(m)
                if np.random.rand() < 0.7:
                    m = self.tmask(m)
            return m, torch.from_numpy(self.Y[real_idx]).float(), torch.from_numpy(self.soft[real_idx]).float()

    tr_ds = SEDDataset(df, Y, soft_clip, DATA, tr_idx, training=True)
    va_ds = SEDDataset(df, Y, soft_clip, DATA, va_idx, training=False)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # ── Student model: ConvNeXtV2-Tiny (FCMAE+in22k+in1k) — standard SED head
    class SEDStudent(nn.Module):
        def __init__(self, backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k", n_classes=NC):
            super().__init__()
            # 3-channel pretrained — replicate 1-channel mel at input
            self.bk = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="", in_chans=3)
            d = self.bk.num_features
            self.gem_p = nn.Parameter(torch.tensor(3.0))
            self.frame_head = nn.Linear(d, n_classes)
            self.att_head = nn.Linear(d, n_classes)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            # x: (B, 1, 256, 313) — replicate to 3 channels for pretrained
            if x.shape[1] == 1:
                x = x.expand(-1, 3, -1, -1)
            f = self.bk(x)  # (B, D, H', W')
            # Frequency-axis GEM pool: (B, D, W')
            p = self.gem_p.clamp(min=1.0)
            f_freq = (f.clamp(min=1e-6).pow(p)).mean(dim=2).pow(1.0 / p)
            f_t = f_freq.transpose(1, 2)  # (B, W', D)
            f_t_dp = self.dropout(f_t)
            # Per-frame, per-class outputs
            frame_logits = self.frame_head(f_t_dp)  # (B, W', NC)
            # Per-class attention weights over time
            att_logits = self.att_head(f_t_dp)  # (B, W', NC)
            att_weights = torch.softmax(att_logits, dim=1)  # softmax over time axis
            # Attention-pooled clip-level logits (standard SED head)
            clip_logits = (att_weights * frame_logits).sum(dim=1)  # (B, NC)
            return clip_logits, frame_logits

    model = SEDStudent().cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] params={n_params:.1f}M")

    bk_p = list(model.bk.parameters())
    head_p = [p for n, p in model.named_parameters() if not n.startswith("bk.")]
    opt = torch.optim.AdamW([
        {"params": bk_p, "lr": lr * 0.1},
        {"params": head_p, "lr": lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * len(tr_ld), eta_min=lr * 0.01)
    scaler = GradScaler()

    bce_fn = nn.BCEWithLogitsLoss()

    def mixup_batch(x, y_hard, y_soft, alpha=0.4):
        if alpha <= 0:
            return x, y_hard, y_soft
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1 - lam) * x[idx]
        # Element-wise max for multilabel
        y_hard_mix = torch.maximum(y_hard, y_hard[idx])
        # Soft labels: convex combination
        y_soft_mix = lam * y_soft + (1 - lam) * y_soft[idx]
        return x_mix, y_hard_mix, y_soft_mix

    os.makedirs(f"{OUT}/fold{fold}", exist_ok=True)
    best_val = 0.0
    for ep in range(n_epochs):
        model.train()
        t0 = time.time()
        tot_loss = tot_bce = tot_kl = n = 0
        for step, (x, y_hard, y_soft) in enumerate(tr_ld):
            x = x.cuda(non_blocking=True)
            y_hard = y_hard.cuda(non_blocking=True)
            y_soft = y_soft.cuda(non_blocking=True).clamp(min=1e-6, max=1 - 1e-6)
            x, y_hard, y_soft = mixup_batch(x, y_hard, y_soft, alpha=mixup_alpha)
            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                clip_logits, _ = model(x)
                bce = bce_fn(clip_logits, y_hard)
                # KL: teacher_soft is a probability, student logits → sigmoid then KL via BCE soft target
                # Use binary KL: targets are soft probabilities
                kl = F.binary_cross_entropy_with_logits(clip_logits, y_soft)
                loss = bce_weight * bce + kl_weight * kl
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item()
            tot_bce += bce.item()
            tot_kl += kl.item()
            n += 1
            if step % 100 == 0:
                print(f"  ep{ep+1} step {step}/{len(tr_ld)} loss={tot_loss/n:.4f} bce={tot_bce/n:.4f} kl={tot_kl/n:.4f} t={time.time()-t0:.0f}s", flush=True)

        # ── Validate
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x, y, _ in va_ld:
                x = x.cuda(non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    clip_logits, _ = model(x)
                all_probs.append(torch.sigmoid(clip_logits).float().cpu().numpy())
                all_labels.append(y.numpy())
        AP = np.concatenate(all_probs)
        AY = np.concatenate(all_labels)
        v = (AY > 0.5).sum(0) > 0
        try:
            val_auc = roc_auc_score(AY[:, v], AP[:, v], average="macro")
        except Exception:
            val_auc = 0.0
        print(f"[ep {ep+1}] loss={tot_loss/n:.4f} val_auc={val_auc:.4f} t={time.time()-t0:.0f}s", flush=True)

        ckpt = {"state": model.state_dict(), "epoch": ep + 1, "val_auc": val_auc, "config": {"bce_w": bce_weight, "kl_w": kl_weight, "mixup": mixup_alpha}}
        torch.save(ckpt, f"{OUT}/fold{fold}/ep{ep+1}.pt")
        if val_auc > best_val:
            best_val = val_auc
            torch.save(ckpt, f"{OUT}/fold{fold}/best.pt")
            print(f"  ↑ best val_auc → {best_val:.4f}", flush=True)
        out_vol.commit()

    print(f"[DONE fold {fold}] best_val={best_val:.4f}")

    # ── ONNX export of best checkpoint for V246 kernel integration
    best_ckpt = torch.load(f"{OUT}/fold{fold}/best.pt", map_location="cuda")
    model.load_state_dict(best_ckpt["state"])
    model.eval()
    onnx_path = f"{OUT}/fold{fold}/sed_fold{fold}.onnx"
    dummy = torch.randn(1, 1, N_MELS, N_TIME, device="cuda", dtype=torch.float32)

    class ONNXWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, mel):
            clip_logits, frame_logits = self.inner(mel)
            return clip_logits, frame_logits

    wrapper = ONNXWrapper(model).cuda().eval()
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["mel"],
        output_names=["clip_logits", "framewise_logits"],
        dynamic_axes={
            "mel": {0: "batch"},
            "clip_logits": {0: "batch"},
            "framewise_logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[ONNX] exported to {onnx_path}")
    out_vol.commit()

    return {"fold": fold, "best_val_auc": best_val, "epochs": n_epochs, "n_train": len(tr_idx), "n_val": len(va_idx), "onnx_path": onnx_path}


@app.local_entrypoint()
def main(fold: int = 0, n_epochs: int = 50):
    print(train.remote(fold=fold, n_epochs=n_epochs))

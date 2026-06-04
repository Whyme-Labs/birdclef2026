"""Modal: train a strong EffV2-S SED student to beat the public distilled SED.

Pipeline:
  - extract uploaded data tar to the data volume (one-time)
  - train_fold(): EffV2-S SED on focal(hard) + soundscape-pseudo(soft) + labeled-soundscape(hard),
    heavy aug, clean FILE-LEVEL held-out soundscape validation, and a DIRECT comparison
    against the public distilled SED teacher on the same held-out files.
  - export_onnx(): export checkpoint to ONNX matching public SED I/O
    (input 'mel' (B,1,256,313) -> 'clip_logits' (B,234), 'framewise_logits').

Mel config EXACTLY matches the public SED so the ONNX is a drop-in 40%-weight blend partner.

Local driver commands:
  modal run sed_modal.py::extract_data
  modal run sed_modal.py::train_fold --fold 0 --epochs 30
  modal run sed_modal.py::export_onnx --fold 0
"""
import modal

app = modal.App("birdclef-sed-student")

BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"

# Diverse-architecture SED ensemble (decorrelated errors → ensemble gain)
ENSEMBLE_BACKBONES = [
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "tf_efficientnet_b3.ns_jft_in1k",
    "seresnext50_32x4d.racm_in1k",
]


def _cache_backbone():
    """Bake ALL ensemble backbones into the image at build time (avoids runtime HF-Hub hang; offline-safe)."""
    import timm
    for b in ["tf_efficientnetv2_s.in21k_ft_in1k", "convnext_small.fb_in22k_ft_in1k",
              "tf_efficientnet_b3.ns_jft_in1k", "seresnext50_32x4d.racm_in1k"]:
        timm.create_model(b, pretrained=True, num_classes=0, global_pool="", in_chans=1)
        print(f"cached {b}", flush=True)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "timm==1.0.11", "librosa==0.10.2", "soundfile==0.12.1",
        "scikit-learn==1.5.2", "numpy==1.26.4", "pandas==2.2.3",
        "onnx==1.17.0", "onnxruntime-gpu==1.20.1", "kaggle==2.1.0", "kagglesdk==0.1.21",
    )
    .run_function(_cache_backbone)
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
)

# Separate image for probe experiments (needs transformers + HF online for foundation models)
probe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1", "timm==1.0.11", "librosa==0.10.2",
        "soundfile==0.12.1", "scikit-learn==1.5.2", "numpy==1.26.4", "pandas==2.2.3",
        "onnx==1.17.0", "onnxruntime-gpu==1.20.1", "transformers==4.46.3",
    )
)

KAGGLE_SECRET = modal.Secret.from_name("kaggle-api-token")

data_vol = modal.Volume.from_name("birdclef-sed-data", create_if_missing=True)
out_vol = modal.Volume.from_name("birdclef-sed-out", create_if_missing=True)

DATA = "/data"
OUT = "/out"

SR = 32000
WS = SR * 5
N_MELS, N_FFT, HOP, FMIN, FMAX, N_TIME = 256, 2048, 512, 20, 16000, 313


# ───────────────────────── data prep (download on Modal + pseudo-labels) ─────────────────────────
@app.function(image=image, secrets=[KAGGLE_SECRET], volumes={"/data": data_vol},
              gpu="A10G", timeout=4 * 3600)
def prepare_data(force: bool = False):
    import os, json, glob, subprocess, time
    import numpy as np, soundfile as sf, librosa

    # kaggle auth via KAGGLE_API_TOKEN env var (set by the secret) — matches local CLI 2.1.0
    assert os.environ.get("KAGGLE_API_TOKEN"), "KAGGLE_API_TOKEN not set"
    print("KAGGLE_API_TOKEN present:", bool(os.environ.get("KAGGLE_API_TOKEN")), flush=True)

    import zipfile
    def dl_unzip(cmd, zip_path, extract_to):
        print(f"running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print(f"unzipping {zip_path} -> {extract_to}", flush=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_to)
        os.remove(zip_path)

    # 1) competition data (train_audio + train_soundscapes + csvs) — datacenter-fast
    if force or not os.path.isdir(f"{DATA}/train_audio"):
        print("downloading competition data...", flush=True)
        dl_unzip(["kaggle", "competitions", "download", "-c", "birdclef-2026", "-p", DATA],
                 f"{DATA}/birdclef-2026.zip", DATA)
    else:
        print("train_audio present, skip competition download", flush=True)
    # 2) teacher SED ONNX
    if force or not os.path.isdir(f"{DATA}/public_sed"):
        print("downloading teacher SED dataset...", flush=True)
        os.makedirs(f"{DATA}/public_sed", exist_ok=True)
        dl_unzip(["kaggle", "datasets", "download", "tuckerarrants/bc2026-distilled-sed-public",
                  "-p", f"{DATA}/public_sed"],
                 f"{DATA}/public_sed/bc2026-distilled-sed-public.zip", f"{DATA}/public_sed")
    print("=== /data contents ===", flush=True)
    for f in sorted(os.listdir(DATA)):
        p = f"{DATA}/{f}"
        n = len(os.listdir(p)) if os.path.isdir(p) else os.path.getsize(p)
        print(f"  {f}{'/' if os.path.isdir(p) else ''}  {n}", flush=True)
    data_vol.commit()

    # 3) generate pseudo-labels: teacher 5-fold on all train_soundscapes
    pseudo_path = f"{DATA}/soundscape_pseudo.npz"
    if not force and os.path.exists(pseudo_path):
        print("pseudo-labels present, skip", flush=True)
        return
    import onnxruntime as ort
    N_WIN = 12
    def make_mel(wav):
        s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=80)
        s = (s - s.mean()) / (s.std() + 1e-6)
        if s.shape[-1] < N_TIME: s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
        else: s = s[:, :N_TIME]
        return s.astype(np.float32)
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    sessions = [ort.InferenceSession(f"{DATA}/public_sed/sed_fold{f}.onnx",
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                for f in range(5)]
    inp = sessions[0].get_inputs()[0].name
    files = sorted(glob.glob(f"{DATA}/train_soundscapes/*.ogg"))
    print(f"pseudo-labeling {len(files)} soundscapes x {N_WIN} win", flush=True)
    all_probs, all_files, all_win = [], [], []
    mel_buf, meta_buf = [], []
    t0 = time.time()
    def flush():
        if not mel_buf: return
        x = np.stack(mel_buf)[:, None].astype(np.float32)
        psum = np.zeros((len(mel_buf), 234), np.float32)
        for s in sessions:
            o = s.run(None, {inp: x})
            clip = o[0]
            fm = o[1].max(axis=1) if len(o) > 1 and o[1].ndim == 3 else clip
            psum += 0.5 * sigmoid(clip) + 0.5 * sigmoid(fm)
        psum /= len(sessions)
        for k, (fn, wi) in enumerate(meta_buf):
            all_probs.append(psum[k].astype(np.float16)); all_files.append(fn); all_win.append(wi)
        mel_buf.clear(); meta_buf.clear()
    for fi, path in enumerate(files):
        fn = os.path.basename(path)
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            continue
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        need = N_WIN * WS
        if len(wav) < need: wav = np.pad(wav, (0, need - len(wav)))
        for wi in range(N_WIN):
            mel_buf.append(make_mel(wav[wi * WS:(wi + 1) * WS])); meta_buf.append((fn, wi))
            if len(mel_buf) >= 128: flush()
        if (fi + 1) % 1000 == 0:
            print(f"  {fi+1}/{len(files)} t={time.time()-t0:.0f}s", flush=True)
    flush()
    probs = np.stack(all_probs)
    np.savez_compressed(pseudo_path, probs=probs, files=np.array(all_files),
                        win_idx=np.array(all_win, dtype=np.int16))
    print(f"pseudo-labels saved: {probs.shape}, mean={probs.astype(np.float32).mean():.4f}", flush=True)
    data_vol.commit()


# ───────────────────────── model ─────────────────────────
def build_model_code():
    """Returned as a string-free module: defined inline in training fn for Modal serialization."""
    pass


# ───────────────────────── training ─────────────────────────
@app.function(image=image, gpu="A100", volumes={"/data": data_vol, "/out": out_vol},
              timeout=6 * 3600)
def train_fold(fold: int = 0, epochs: int = 30, backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k",
               batch_size: int = 64, lr: float = 1e-3, steps_per_epoch: int = 1200,
               holdout_frac: float = 0.2, tag: str = ""):
    import os, time, numpy as np, pandas as pd, soundfile as sf, librosa
    import torch, torch.nn as nn, torch.nn.functional as F, timm
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import autocast, GradScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    DEVICE = "cuda"
    torch.manual_seed(42 + fold); np.random.seed(42 + fold)
    os.makedirs(OUT, exist_ok=True)

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

    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {l: i for i, l in enumerate(PRIMARY)}
    NC = len(PRIMARY)
    print(f"NC={NC}", flush=True)

    # focal hard labels
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
    prim = np.array([l2i.get(p, 0) for p in df["primary_label"]])
    cnt = np.bincount(prim, minlength=NC).astype(np.float32)
    focal_w = (1.0 / np.sqrt(cnt + 1.0))[prim]
    print(f"focal: {len(df)} clips", flush=True)

    # labeled soundscapes — FILE-LEVEL holdout for clean validation
    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv")
    all_files = sorted(lab["filename"].unique().tolist())
    rng = np.random.default_rng(123)
    rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * holdout_frac))
    val_files = set(all_files[:n_val])
    print(f"labeled soundscape files: {len(all_files)} total, {len(val_files)} held out for val", flush=True)

    def rows_from(lab_df, keep):
        out = []
        for _, r in lab_df.iterrows():
            fn = str(r["filename"])
            if (fn in val_files) != (not keep):
                continue
            yv = np.zeros(NC, dtype=np.float32)
            for s in str(r["primary_label"]).split(";"):
                s = s.strip()
                if s in l2i:
                    yv[l2i[s]] = 1.0
            out.append((fn, parse_t(r["start"]), parse_t(r["end"]), yv))
        return out
    train_lab_rows = rows_from(lab, keep=True)    # files NOT in val
    val_lab_rows = rows_from(lab, keep=False)     # held-out files
    print(f"train labeled rows: {len(train_lab_rows)}, val labeled rows: {len(val_lab_rows)}", flush=True)

    # pseudo soundscapes (exclude held-out val files to avoid soft leak)
    pz = np.load(f"{DATA}/soundscape_pseudo.npz", allow_pickle=True)
    ss_files = pz["files"]; ss_win = pz["win_idx"]; ss_probs = pz["probs"].astype(np.float32)
    keep_mask = np.array([str(f) not in val_files for f in ss_files])
    ss_files, ss_win, ss_probs = ss_files[keep_mask], ss_win[keep_mask], ss_probs[keep_mask]
    print(f"pseudo soundscape windows (val excluded): {ss_probs.shape}", flush=True)

    class MixDataset(Dataset):
        def __init__(self, n_steps, bs):
            self.n = n_steps * bs
        def __len__(self):
            return self.n
        def _aug_wav(self, wav):
            return wav * (10 ** (np.random.uniform(-0.4, 0.4)))
        def _focal(self):
            i = np.random.choice(len(df), p=focal_w / focal_w.sum())
            fn = df.iloc[i]["filename"]; y = Y[i].astype(np.float32)
            try:
                wav, sr = sf.read(f"{DATA}/train_audio/{fn}", dtype="float32", always_2d=False)
            except Exception:
                return np.zeros((N_MELS, N_TIME), np.float32), y
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            n = len(wav)
            if n < WS: wav = np.tile(wav, (WS // max(n, 1)) + 1)[:WS]
            else:
                st = np.random.randint(0, n - WS + 1); wav = wav[st:st + WS]
            return make_mel(self._aug_wav(wav)), y
        def _pseudo(self):
            j = np.random.randint(len(ss_files))
            fn = ss_files[j]; wi = int(ss_win[j]); y = ss_probs[j].astype(np.float32)
            try:
                wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
            except Exception:
                return np.zeros((N_MELS, N_TIME), np.float32), y
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            seg = wav[wi * WS:(wi + 1) * WS]
            if len(seg) < WS: seg = np.pad(seg, (0, WS - len(seg)))
            return make_mel(self._aug_wav(seg)), y
        def _labeled(self):
            fn, st_s, en_s, y = train_lab_rows[np.random.randint(len(train_lab_rows))]
            try:
                wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
            except Exception:
                return np.zeros((N_MELS, N_TIME), np.float32), y
            if wav.ndim > 1: wav = wav.mean(axis=1)
            if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            seg = wav[int(st_s * SR):int(en_s * SR)]
            seg = np.pad(seg, (0, WS - len(seg))) if len(seg) < WS else seg[:WS]
            return make_mel(self._aug_wav(seg)), y.astype(np.float32)
        def __getitem__(self, idx):
            r = np.random.rand()
            if r < 0.40: mel, y = self._focal()
            elif r < 0.90: mel, y = self._pseudo()
            else: mel, y = self._labeled()
            m = torch.from_numpy(mel).unsqueeze(0)
            if np.random.rand() < 0.6:
                f0 = np.random.randint(0, N_MELS - 24)
                m[:, f0:f0 + np.random.randint(8, 24), :] = 0
            if np.random.rand() < 0.6:
                t0 = np.random.randint(0, N_TIME - 40)
                m[:, :, t0:t0 + np.random.randint(15, 40)] = 0
            return m, torch.from_numpy(y)

    class ValDS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            fn, st, en, y = self.rows[i]
            try:
                wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                seg = wav[int(st * SR):int(en * SR)]
                seg = np.pad(seg, (0, WS - len(seg))) if len(seg) < WS else seg[:WS]
                return torch.from_numpy(make_mel(seg)).unsqueeze(0), torch.from_numpy(y)
            except Exception:
                return torch.zeros(1, N_MELS, N_TIME), torch.from_numpy(y)

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

    tr_ld = DataLoader(MixDataset(steps_per_epoch, batch_size), batch_size=batch_size,
                       shuffle=False, num_workers=16, pin_memory=True, drop_last=True,
                       persistent_workers=True)
    va_ld = DataLoader(ValDS(val_lab_rows), batch_size=128, shuffle=False, num_workers=8)

    model = SEDModel(backbone, NC).to(DEVICE)
    bk = list(model.bk.parameters())
    hd = [p for n, p in model.named_parameters() if not n.startswith("bk.")]
    opt = torch.optim.AdamW([{"params": bk, "lr": lr * 0.1}, {"params": hd, "lr": lr}],
                            weight_decay=1e-4)
    steps = epochs * len(tr_ld)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    scaler = GradScaler()

    # validation labels matrix
    VY = np.stack([r[3] for r in val_lab_rows]) if val_lab_rows else np.zeros((0, NC))

    best = 0.0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot = nb = 0
        for x, y in tr_ld:
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
            tot += loss.item(); nb += 1
        # validate
        model.eval(); ap = []
        with torch.no_grad():
            for x, _ in va_ld:
                x = x.to(DEVICE, non_blocking=True)
                with autocast("cuda", dtype=torch.float16):
                    clip, _ = model(x)
                ap.append(torch.sigmoid(clip).float().cpu().numpy())
        AP = np.concatenate(ap) if ap else np.zeros((0, NC))
        v = (VY > 0.5).sum(0) > 0
        try:
            auc = roc_auc_score((VY[:, v] > 0.5).astype(int), AP[:, v], average="macro")
        except Exception:
            auc = 0.0
        print(f"[ep {ep+1}/{epochs}] loss={tot/max(nb,1):.4f} heldout_ss_AUC={auc:.4f} "
              f"t={time.time()-t0:.0f}s (cls={int(v.sum())})", flush=True)
        ck = {"state": model.state_dict(), "epoch": ep + 1, "heldout_auc": float(auc),
              "backbone": backbone, "fold": fold, "val_files": sorted(val_files)}
        sfx = f"_{tag}" if tag else ""
        torch.save(ck, f"{OUT}/fold{fold}_last{sfx}.pt")
        if auc > best:
            best = auc
            torch.save(ck, f"{OUT}/fold{fold}_best{sfx}.pt")
            print(f"  -> best {best:.4f}", flush=True)
        out_vol.commit()
    print(f"=== DONE fold {fold}: best heldout_ss_AUC={best:.4f} ===", flush=True)
    return best


# ───────────────────────── teacher comparison ─────────────────────────
@app.function(image=image, gpu="A10G", volumes={"/data": data_vol, "/out": out_vol}, timeout=3600)
def compare_to_teacher(fold: int = 0):
    """Run the public SED teacher on the SAME held-out files our student used; report both AUCs."""
    import numpy as np, pandas as pd, soundfile as sf, librosa, torch
    from sklearn.metrics import roc_auc_score
    import onnxruntime as ort

    def make_mel(wav):
        s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=80)
        s = (s - s.mean()) / (s.std() + 1e-6)
        if s.shape[-1] < N_TIME: s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
        else: s = s[:, :N_TIME]
        return s.astype(np.float32)

    def parse_t(t):
        p = str(t).split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {l: i for i, l in enumerate(PRIMARY)}; NC = len(PRIMARY)
    ck = torch.load(f"{OUT}/fold{fold}_best.pt", map_location="cpu")
    val_files = set(ck["val_files"])
    print(f"held-out val files: {len(val_files)}", flush=True)

    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv")
    rows = []
    for _, r in lab.iterrows():
        fn = str(r["filename"])
        if fn not in val_files: continue
        yv = np.zeros(NC, dtype=np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in l2i: yv[l2i[s]] = 1.0
        rows.append((fn, parse_t(r["start"]), parse_t(r["end"]), yv))

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    sessions = [ort.InferenceSession(f"{DATA}/public_sed/sed_fold{f}.onnx",
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                for f in range(5)]
    inp = sessions[0].get_inputs()[0].name
    preds, ys = [], []
    for fn, st, en, y in rows:
        wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        seg = wav[int(st * SR):int(en * SR)]
        seg = np.pad(seg, (0, WS - len(seg))) if len(seg) < WS else seg[:WS]
        x = make_mel(seg)[None, None].astype(np.float32)
        ps = np.zeros(NC, np.float32)
        for s in sessions:
            o = s.run(None, {inp: x})
            clip = o[0][0]
            fm = o[1][0].max(0) if len(o) > 1 and o[1].ndim == 3 else clip
            ps += 0.5 * sigmoid(clip) + 0.5 * sigmoid(fm)
        preds.append(ps / len(sessions)); ys.append(y)
    P, A = np.stack(preds), np.stack(ys)
    v = (A > 0.5).sum(0) > 0
    teacher_auc = roc_auc_score((A[:, v] > 0.5).astype(int), P[:, v], average="macro")
    print(f"=== PUBLIC SED teacher held-out AUC: {teacher_auc:.4f} (cls={int(v.sum())}) ===", flush=True)
    print(f"=== student fold{fold} held-out AUC: {ck['heldout_auc']:.4f} ===", flush=True)
    print(f"=== DELTA (student - teacher): {ck['heldout_auc'] - teacher_auc:+.4f} ===", flush=True)
    return {"teacher": float(teacher_auc), "student": float(ck["heldout_auc"])}


# ───────────────────────── probe orthogonality evaluation ─────────────────────────
@app.function(image=probe_image, gpu="A10G", volumes={"/data": data_vol, "/out": out_vol},
              timeout=2 * 3600)
def eval_birdmae_probe(holdout_frac: float = 0.2):
    """Decisive offline test: does a Bird-MAE probe add ORTHOGONAL signal over the public SED?

    Trains Bird-MAE linear probes on train-split labeled soundscapes, evaluates on the SAME
    13 held-out files the SED student used. Reports:
      - Bird-MAE probe held-out macro-AUC (standalone strength)
      - public SED teacher held-out macro-AUC (the base)
      - best blend (teacher, probe) held-out macro-AUC + weight  ← does blending HELP?
      - per-class: how many classes the probe BEATS the teacher on (orthogonality)
    """
    import os, numpy as np, pandas as pd, soundfile as sf, librosa, torch
    import torchaudio.compliance.kaldi as kaldi
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModel
    import onnxruntime as ort

    SR = 32000
    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {l: i for i, l in enumerate(PRIMARY)}; NC = len(PRIMARY)

    def parse_t(t):
        p = str(t).split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv")
    all_files = sorted(lab["filename"].unique().tolist())
    rng = np.random.default_rng(123); rng.shuffle(all_files)
    n_val = max(1, int(len(all_files) * holdout_frac))
    val_files = set(all_files[:n_val])
    print(f"{len(all_files)} files, {len(val_files)} held out (same seed=123 as SED student)", flush=True)

    # build per-(file,end) rows with union labels
    lab["primary_label"] = lab["primary_label"].astype(str)
    rows = []
    for _, r in lab.iterrows():
        yv = np.zeros(NC, np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in l2i: yv[l2i[s]] = 1.0
        rows.append((str(r["filename"]), parse_t(r["start"]), parse_t(r["end"]), yv))

    def fbank_mel(wav_t, target_length=512):
        fb = kaldi.fbank(wav_t, htk_compat=True, sample_frequency=SR, use_energy=False,
                         window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
        n = fb.shape[0]
        if n < target_length: fb = torch.cat([fb, torch.zeros(target_length - n, 128)], 0)
        else: fb = fb[:target_length]
        return (fb - (-7.2)) / (4.43 * 2.0)

    print("loading Bird-MAE-Base...", flush=True)
    model = AutoModel.from_pretrained("DBD-research-group/Bird-MAE-Base", trust_remote_code=True).cuda().eval()

    # extract embeddings per row
    X = np.zeros((len(rows), 768), np.float32); Y = np.zeros((len(rows), NC), np.float32)
    is_val = np.zeros(len(rows), bool)
    wav_cache = {}
    import time as _t; t0 = _t.time()
    for i, (fn, st, en, yv) in enumerate(rows):
        if fn not in wav_cache:
            w, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
            if w.ndim > 1: w = w.mean(1)
            if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
            wav_cache[fn] = w
        w = wav_cache[fn]
        seg = w[int(st * SR):int(en * SR)]
        seg = np.pad(seg, (0, 5 * SR - len(seg))) if len(seg) < 5 * SR else seg[:5 * SR]
        mel = fbank_mel(torch.from_numpy(seg).unsqueeze(0)).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            X[i] = model(mel).last_hidden_state.cpu().numpy().reshape(-1)[:768]
        Y[i] = yv; is_val[i] = fn in val_files
        if (i + 1) % 300 == 0: print(f"  emb {i+1}/{len(rows)} t={_t.time()-t0:.0f}s", flush=True)

    tr, va = ~is_val, is_val
    print(f"train rows {tr.sum()}, val rows {va.sum()}", flush=True)

    # train probes on train split, predict val
    probe_val = np.full((va.sum(), NC), np.nan, np.float32)
    trained = 0
    for ci in range(NC):
        if Y[tr, ci].sum() < 3 or Y[va, ci].sum() < 1: continue
        try:
            clf = LogisticRegression(C=1.0, max_iter=200, solver="liblinear", class_weight="balanced")
            clf.fit(X[tr], Y[tr, ci].astype(int))
            probe_val[:, ci] = clf.predict_proba(X[va])[:, 1]; trained += 1
        except Exception:
            pass
    print(f"probes trained: {trained}", flush=True)

    # public SED teacher on val files
    def make_mel(wav):
        s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=80); s = (s - s.mean()) / (s.std() + 1e-6)
        if s.shape[-1] < N_TIME: s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
        else: s = s[:, :N_TIME]
        return s.astype(np.float32)
    def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    sess = [ort.InferenceSession(f"{DATA}/public_sed/sed_fold{f}.onnx", providers=["CPUExecutionProvider"]) for f in range(5)]
    inp = sess[0].get_inputs()[0].name
    val_rows = [r for r, v in zip(rows, is_val) if v]
    teacher_val = np.zeros((len(val_rows), NC), np.float32)
    for i, (fn, st, en, yv) in enumerate(val_rows):
        w = wav_cache[fn]; seg = w[int(st * SR):int(en * SR)]
        seg = np.pad(seg, (0, 5 * SR - len(seg))) if len(seg) < 5 * SR else seg[:5 * SR]
        x = make_mel(seg)[None, None]
        ps = np.zeros(NC, np.float32)
        for s_ in sess:
            o = s_.run(None, {inp: x}); clip = o[0][0]
            fm = o[1][0].max(0) if len(o) > 1 and o[1].ndim == 3 else clip
            ps += 0.5 * sig(clip) + 0.5 * sig(fm)
        teacher_val[i] = ps / len(sess)

    VY = Y[va]
    valid = (VY.sum(0) > 0) & ~np.isnan(probe_val).all(0)
    pv = np.nan_to_num(probe_val[:, valid]); tv = teacher_val[:, valid]; yy = (VY[:, valid] > 0.5).astype(int)
    def macro(P): return roc_auc_score(yy, P, average="macro")
    probe_auc = macro(pv); teacher_auc = macro(tv)
    print(f"\n=== Bird-MAE probe held-out AUC: {probe_auc:.4f} ({valid.sum()} cls) ===", flush=True)
    print(f"=== public SED teacher held-out AUC: {teacher_auc:.4f} ===", flush=True)
    best = (0, teacher_auc)
    for wt in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        # rank-blend (matches kernel's rank fusion)
        from scipy.stats import rankdata
        rp = np.apply_along_axis(rankdata, 0, pv); rt = np.apply_along_axis(rankdata, 0, tv)
        blend = (1 - wt) * rt + wt * rp
        a = macro(blend)
        print(f"  blend w_probe={wt:.2f}: {a:.4f}  ({'+' if a>teacher_auc else ''}{a-teacher_auc:+.4f})", flush=True)
        if a > best[1]: best = (wt, a)
    # orthogonality: classes where probe beats teacher
    beats = sum(1 for j in range(valid.sum())
                if roc_auc_score(yy[:, j], pv[:, j]) > roc_auc_score(yy[:, j], tv[:, j]) + 0.01)
    print(f"\n=== best blend: w_probe={best[0]:.2f} -> {best[1]:.4f} (teacher alone {teacher_auc:.4f}) ===", flush=True)
    print(f"=== probe beats teacher on {beats}/{valid.sum()} held-out classes (orthogonality) ===", flush=True)
    print(f"=== VERDICT: {'PROBE HELPS — build kernel' if best[1] > teacher_auc + 0.001 else 'no held-out gain (but held-out saturated; rare-class effect unmeasurable here)'} ===", flush=True)
    return {"probe": float(probe_auc), "teacher": float(teacher_auc), "best_blend": float(best[1]), "best_w": float(best[0]), "beats": int(beats), "ncls": int(valid.sum())}


# ───────────────────────── rare-taxa specialist eval ─────────────────────────
@app.function(image=probe_image, gpu="A10G", volumes={"/data": data_vol, "/out": out_vol},
              timeout=2 * 3600)
def eval_rare_taxa_specialist(holdout_frac: float = 0.2, model_id: str = "audiomae"):
    """THE genuinely-novel test: does a texture-appropriate probe beat the bird-tuned public SED
    SPECIFICALLY on rare Amphibia/Insecta classes (where Perch+SED are weakest)?

    model_id: 'audiomae' (general audio, hypothesis: better for insect/frog textures) or 'birdmae'.
    Evaluates on held-out files, SLICED to rare Amphibia/Insecta classes with enough positives.
    """
    import os, numpy as np, pandas as pd, soundfile as sf, librosa, torch
    import torchaudio.compliance.kaldi as kaldi
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModel
    import onnxruntime as ort

    SR = 32000
    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    l2i = {l: i for i, l in enumerate(PRIMARY)}; NC = len(PRIMARY)
    clsname = tax.set_index("primary_label")["class_name"].to_dict()

    def parse_t(t):
        p = str(t).split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv"); lab["primary_label"] = lab["primary_label"].astype(str)
    all_files = sorted(lab["filename"].unique().tolist())
    rng = np.random.default_rng(123); rng.shuffle(all_files)
    val_files = set(all_files[:max(1, int(len(all_files) * holdout_frac))])
    rows = []
    for _, r in lab.iterrows():
        yv = np.zeros(NC, np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in l2i: yv[l2i[s]] = 1.0
        rows.append((str(r["filename"]), parse_t(r["start"]), parse_t(r["end"]), yv))
    is_val = np.array([r[0] in val_files for r in rows])

    # Bird-MAE-Base and AudioMAE both consume kaldi fbank (B,1,512,128); AudioMAE generic-audio norm
    HF = {"birdmae": "DBD-research-group/Bird-MAE-Base", "audiomae": "gaunernst/vit_base_patch16_1024_128.audiomae_as2m"}
    print(f"loading {model_id} ({HF[model_id]})...", flush=True)
    use_timm = model_id == "audiomae"
    if use_timm:
        import timm
        enc = timm.create_model("hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
                                pretrained=True, num_classes=0).cuda().eval()
        MEAN, STD = -4.27, 4.57  # audioset norm
    else:
        enc = AutoModel.from_pretrained(HF["birdmae"], trust_remote_code=True).cuda().eval()
        MEAN, STD = -7.2, 4.43

    def fbank(seg, L=1024 if use_timm else 512):
        wt = torch.from_numpy(seg).unsqueeze(0)
        fb = kaldi.fbank(wt, htk_compat=True, sample_frequency=SR, use_energy=False,
                         window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
        n = fb.shape[0]
        fb = torch.cat([fb, torch.zeros(L - n, 128)], 0) if n < L else fb[:L]
        return (fb - MEAN) / (STD * 2.0)

    wav_cache = {}; import time as _t; t0 = _t.time()
    D = 768
    X = np.zeros((len(rows), D), np.float32); Y = np.zeros((len(rows), NC), np.float32)
    for i, (fn, st, en, yv) in enumerate(rows):
        if fn not in wav_cache:
            w, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
            if w.ndim > 1: w = w.mean(1)
            if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
            wav_cache[fn] = w
        w = wav_cache[fn]; seg = w[int(st * SR):int(en * SR)]
        seg = np.pad(seg, (0, 5 * SR - len(seg))) if len(seg) < 5 * SR else seg[:5 * SR]
        mel = fbank(seg).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            if use_timm:
                f = enc(mel)
                emb = f.mean(1) if f.ndim == 3 else f
                X[i] = emb.cpu().numpy().reshape(-1)[:D]
            else:
                X[i] = enc(mel).last_hidden_state.cpu().numpy().reshape(-1)[:D]
        Y[i] = yv
        if (i + 1) % 400 == 0: print(f"  emb {i+1}/{len(rows)} t={_t.time()-t0:.0f}s", flush=True)

    tr, va = ~is_val, is_val
    # rare Amphibia/Insecta classes measurable on this split
    rare = [ci for ci in range(NC) if clsname.get(PRIMARY[ci]) in ("Amphibia", "Insecta")
            and Y[tr, ci].sum() >= 3 and Y[va, ci].sum() >= 1]
    print(f"measurable rare Amphibia/Insecta classes: {len(rare)}", flush=True)

    # train probes on rare classes
    probe_va = np.zeros((va.sum(), len(rare)), np.float32)
    for j, ci in enumerate(rare):
        clf = LogisticRegression(C=1.0, max_iter=300, solver="liblinear", class_weight="balanced")
        clf.fit(X[tr], Y[tr, ci].astype(int))
        probe_va[:, j] = clf.predict_proba(X[va])[:, 1]

    # public SED teacher on val rows, sliced to rare classes
    def make_mel(wav):
        s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=80); s = (s - s.mean()) / (s.std() + 1e-6)
        if s.shape[-1] < N_TIME: s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
        else: s = s[:, :N_TIME]
        return s.astype(np.float32)
    def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    sess = [ort.InferenceSession(f"{DATA}/public_sed/sed_fold{f}.onnx", providers=["CPUExecutionProvider"]) for f in range(5)]
    inp = sess[0].get_inputs()[0].name
    val_rows = [r for r, v in zip(rows, is_val) if v]
    teach = np.zeros((len(val_rows), len(rare)), np.float32)
    for i, (fn, st, en, yv) in enumerate(val_rows):
        w = wav_cache[fn]; seg = w[int(st * SR):int(en * SR)]
        seg = np.pad(seg, (0, 5 * SR - len(seg))) if len(seg) < 5 * SR else seg[:5 * SR]
        x = make_mel(seg)[None, None]; ps = np.zeros(NC, np.float32)
        for s_ in sess:
            o = s_.run(None, {inp: x}); clip = o[0][0]
            fm = o[1][0].max(0) if len(o) > 1 and o[1].ndim == 3 else clip
            ps += 0.5 * sig(clip) + 0.5 * sig(fm)
        ps /= len(sess)
        teach[i] = ps[rare]

    YY = (Y[va][:, rare] > 0.5).astype(int)
    from scipy.stats import rankdata
    def macro(P): return roc_auc_score(YY, P, average="macro")
    pa, ta = macro(probe_va), macro(teach)
    print(f"\n=== RARE-TAXA ({len(rare)} Amphibia/Insecta classes, {int(YY.sum())} val pos) ===", flush=True)
    print(f"=== {model_id} specialist AUC: {pa:.4f} ===", flush=True)
    print(f"=== public SED teacher AUC (same rare classes): {ta:.4f} ===", flush=True)
    beats = sum(1 for j in range(len(rare)) if roc_auc_score(YY[:, j], probe_va[:, j]) > roc_auc_score(YY[:, j], teach[:, j]) + 0.01)
    print(f"=== specialist BEATS teacher on {beats}/{len(rare)} rare classes ===", flush=True)
    best = (0.0, ta)
    for wt in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        rp = np.apply_along_axis(rankdata, 0, probe_va); rt = np.apply_along_axis(rankdata, 0, teach)
        a = macro((1 - wt) * rt + wt * rp)
        print(f"  rare-only blend w_spec={wt:.1f}: {a:.4f} ({a-ta:+.4f})", flush=True)
        if a > best[1]: best = (wt, a)
    print(f"\n=== VERDICT: {'SPECIALIST WINS on rare taxa — build rare-class blend kernel' if best[1] > ta + 0.005 else 'no rare-taxa gain'} (best {best[1]:.4f} @ w={best[0]:.1f} vs teacher {ta:.4f}) ===", flush=True)
    return {"model": model_id, "specialist": float(pa), "teacher": float(ta), "beats": int(beats),
            "nrare": len(rare), "best_blend": float(best[1]), "best_w": float(best[0])}


# ───────────────────────── diverse SED ensemble gate ─────────────────────────
@app.function(image=image, gpu="A10G", volumes={"/data": data_vol, "/out": out_vol}, timeout=2 * 3600)
def eval_sed_ensemble(tags: str = "effv2s,convnext,effb3,seresnext"):
    """Gate: does an ensemble of diverse-architecture SEDs GAIN over the best single (diversity works)?
    Also reports vs public SED teacher (leak-biased reference). Uses the shared 13-file held-out split.
    `tags` = comma list matching saved checkpoints fold0_best_<tag>.pt
    """
    import os, numpy as np, pandas as pd, soundfile as sf, librosa, torch, torch.nn as nn, timm
    from torch.amp import autocast
    from sklearn.metrics import roc_auc_score
    import onnxruntime as ort
    DEVICE = "cuda"

    def make_mel(wav):
        s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=80); s = (s - s.mean()) / (s.std() + 1e-6)
        if s.shape[-1] < N_TIME: s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
        else: s = s[:, :N_TIME]
        return s.astype(np.float32)
    def parse_t(t):
        p = str(t).split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

    class SEDModel(nn.Module):
        def __init__(self, bn, n=234):
            super().__init__()
            self.bk = timm.create_model(bn, pretrained=False, num_classes=0, global_pool="", in_chans=1)
            d = self.bk.num_features
            self.gem_p = nn.Parameter(torch.tensor(3.0)); self.drop = nn.Dropout(0.3)
            self.frame_head = nn.Linear(d, n); self.att_head = nn.Linear(d, n)
        def forward(self, x):
            f = self.bk(x); p = self.gem_p.clamp(min=1.0)
            ff = (f.clamp(min=1e-6).pow(p)).mean(dim=2).pow(1.0 / p); ft = ff.transpose(1, 2)
            fr = self.frame_head(ft); at = torch.softmax(self.att_head(ft), dim=1)
            return (at * fr).sum(dim=1), fr

    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist()); l2i = {l: i for i, l in enumerate(PRIMARY)}; NC = len(PRIMARY)
    taglist = [t.strip() for t in tags.split(",")]
    cks = {}
    for t in taglist:
        p = f"{OUT}/fold0_best_{t}.pt"
        if os.path.exists(p): cks[t] = torch.load(p, map_location="cpu")
        else: print(f"MISSING checkpoint {p}", flush=True)
    assert cks, "no checkpoints found"
    val_files = set(next(iter(cks.values()))["val_files"])
    print(f"held-out files: {len(val_files)} | models: {list(cks)}", flush=True)

    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv"); lab["primary_label"] = lab["primary_label"].astype(str)
    rows = []
    for _, r in lab.iterrows():
        fn = str(r["filename"])
        if fn not in val_files: continue
        yv = np.zeros(NC, np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in l2i: yv[l2i[s]] = 1.0
        rows.append((fn, parse_t(r["start"]), parse_t(r["end"]), yv))
    YY = np.stack([r[3] for r in rows])
    wav_cache = {}
    def seg_mel(fn, st, en):
        if fn not in wav_cache:
            w, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
            if w.ndim > 1: w = w.mean(1)
            if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
            wav_cache[fn] = w
        w = wav_cache[fn]; sg = w[int(st * SR):int(en * SR)]
        sg = np.pad(sg, (0, WS - len(sg))) if len(sg) < WS else sg[:WS]
        return make_mel(sg)

    preds = {}
    for t, ck in cks.items():
        m = SEDModel(ck["backbone"], NC).to(DEVICE); m.load_state_dict(ck["state"]); m.eval()
        P = np.zeros((len(rows), NC), np.float32)
        with torch.no_grad():
            for i, (fn, st, en, yv) in enumerate(rows):
                x = torch.from_numpy(seg_mel(fn, st, en))[None, None].to(DEVICE)
                with autocast("cuda", dtype=torch.float16):
                    clip, _ = m(x)
                P[i] = torch.sigmoid(clip).float().cpu().numpy()
        preds[t] = P
        del m; torch.cuda.empty_cache()

    v = (YY > 0.5).sum(0) > 0; yy = (YY[:, v] > 0.5).astype(int)
    def macro(P): return roc_auc_score(yy, P[:, v], average="macro")
    print(f"\n=== held-out ({int(v.sum())} classes w/ positives) ===", flush=True)
    singles = {}
    for t in preds:
        singles[t] = macro(preds[t]); print(f"  single {t}: {singles[t]:.4f}", flush=True)
    best_single = max(singles.values())
    # rank-average ensemble
    from scipy.stats import rankdata
    rk = sum(np.apply_along_axis(rankdata, 0, preds[t]) for t in preds) / len(preds)
    ens = macro(rk)
    print(f"=== ENSEMBLE ({len(preds)} diverse archs): {ens:.4f} ===", flush=True)
    print(f"=== best single: {best_single:.4f}  ->  ensemble gain: {ens-best_single:+.4f} ===", flush=True)
    # teacher (leak-biased reference)
    def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    sess = [ort.InferenceSession(f"{DATA}/public_sed/sed_fold{f}.onnx", providers=["CPUExecutionProvider"]) for f in range(5)]
    inp = sess[0].get_inputs()[0].name
    T = np.zeros((len(rows), NC), np.float32)
    for i, (fn, st, en, yv) in enumerate(rows):
        x = seg_mel(fn, st, en)[None, None]; ps = np.zeros(NC, np.float32)
        for s_ in sess:
            o = s_.run(None, {inp: x}); clip = o[0][0]
            fm = o[1][0].max(0) if len(o) > 1 and o[1].ndim == 3 else clip
            ps += 0.5 * sig(clip) + 0.5 * sig(fm)
        T[i] = ps / len(sess)
    print(f"=== public SED teacher (LEAK-biased, trained on these files): {macro(T):.4f} ===", flush=True)
    # ensemble + teacher blend
    rt = np.apply_along_axis(rankdata, 0, T)
    for w in [0.2, 0.3, 0.4, 0.5]:
        print(f"  teacher+ensemble blend w_ens={w}: {macro((1-w)*rt + w*rk):.4f}", flush=True)
    print(f"\n=== VERDICT: {'diverse-ensemble GAINS over single — proceed to LB test' if ens > best_single + 0.005 else 'no ensemble gain'} ===", flush=True)
    return {"singles": singles, "ensemble": float(ens), "best_single": float(best_single), "teacher": float(macro(T))}


# ───────────────────────── ONNX export ─────────────────────────
@app.function(image=image, volumes={"/out": out_vol}, timeout=1800)
def export_onnx(fold: int = 0):
    import torch, torch.nn as nn, timm
    ck = torch.load(f"{OUT}/fold{fold}_best.pt", map_location="cpu")
    backbone = ck["backbone"]

    class SEDModel(nn.Module):
        def __init__(self, backbone_name, n_classes=234):
            super().__init__()
            self.bk = timm.create_model(backbone_name, pretrained=False, num_classes=0,
                                         global_pool="", in_chans=1)
            d = self.bk.num_features
            self.gem_p = nn.Parameter(torch.tensor(3.0))
            self.drop = nn.Dropout(0.3)
            self.frame_head = nn.Linear(d, n_classes)
            self.att_head = nn.Linear(d, n_classes)
        def forward(self, mel):
            f = self.bk(mel)
            p = self.gem_p.clamp(min=1.0)
            f_freq = (f.clamp(min=1e-6).pow(p)).mean(dim=2).pow(1.0 / p)
            f_t = f_freq.transpose(1, 2)
            frame_logits = self.frame_head(f_t)
            att = torch.softmax(self.att_head(f_t), dim=1)
            clip_logits = (att * frame_logits).sum(dim=1)
            return clip_logits, frame_logits

    m = SEDModel(backbone, 234); m.load_state_dict(ck["state"]); m.eval()
    dummy = torch.randn(1, 1, N_MELS, N_TIME)
    out_path = f"{OUT}/student_sed_fold{fold}.onnx"
    torch.onnx.export(m, dummy, out_path, input_names=["mel"],
                      output_names=["clip_logits", "framewise_logits"],
                      dynamic_axes={"mel": {0: "batch"}, "clip_logits": {0: "batch"},
                                    "framewise_logits": {0: "batch"}},
                      opset_version=17, do_constant_folding=True)
    out_vol.commit()
    print(f"exported {out_path} (heldout_auc={ck['heldout_auc']:.4f}, backbone={backbone})", flush=True)

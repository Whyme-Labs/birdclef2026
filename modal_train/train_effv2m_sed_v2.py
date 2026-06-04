"""V2: Train EffV2-M SED from audio already uploaded to Modal volume."""
import modal
app = modal.App("birdclef-effv2m-v2")
vol = modal.Volume.from_name("birdclef-effv2m-artifacts", create_if_missing=True)
audio_vol = modal.Volume.from_name("birdclef-audio-data", create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.3.0", "torchaudio==2.3.0", "timm==1.0.11",
                 "numpy", "pandas", "scikit-learn", "soundfile", "librosa"))


@app.function(image=image, gpu="A100", volumes={"/vol": vol, "/audio": audio_vol},
              timeout=12*3600, cpu=4)
def train(n_epochs: int = 18, batch_size: int = 32, lr: float = 1e-4):
    import os, time, numpy as np, pandas as pd
    import torch, torch.nn as nn, torchaudio, timm
    import soundfile as sf
    from torch.utils.data import Dataset, DataLoader, Subset
    from torch.amp import autocast, GradScaler
    from sklearn.metrics import roc_auc_score

    SR = 32_000; WS = SR * 5; NC = 234
    DATA = "/audio"
    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist())
    NC = len(PRIMARY); l2i = {l:i for i,l in enumerate(PRIMARY)}
    df = pd.read_csv(f"{DATA}/train.csv")
    df["primary_label"] = df["primary_label"].astype(str)

    Y = np.zeros((len(df), NC), dtype=np.float32)
    for i, r in df.iterrows():
        if r["primary_label"] in l2i: Y[i, l2i[r["primary_label"]]] = 1.0
        sec = r.get("secondary_labels", "")
        if isinstance(sec, str) and sec.startswith("["):
            try:
                for s in eval(sec):
                    s = str(s).strip()
                    if s in l2i: Y[i, l2i[s]] = 0.5
            except: pass
    print(f"{len(df)} recordings, {NC} classes, {(Y.sum(0)>0).sum()} classes with positives")

    class DS(Dataset):
        def __init__(self, df, Y, root):
            self.df=df; self.Y=Y; self.root=root
            self.mel=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=2048,hop_length=512,
                n_mels=224,f_min=0,f_max=16000,power=2.0,norm="slaney",mel_scale="htk")
            self.db=torchaudio.transforms.AmplitudeToDB(top_db=80)
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            fn = self.df.iloc[idx]["filename"]
            try:
                wav, sr = sf.read(f"{self.root}/train_audio/{fn}", dtype="float32", always_2d=False)
            except Exception:
                return torch.zeros(3,224,313), torch.from_numpy(self.Y[idx]).float()
            if wav.ndim>1: wav=wav.mean(axis=1)
            if sr!=SR:
                import librosa; wav=librosa.resample(wav, orig_sr=sr, target_sr=SR)
            n=len(wav)
            if n<WS: wav=np.tile(wav,(WS//n)+1)[:WS]
            else:
                st=np.random.randint(0,n-WS+1); wav=wav[st:st+WS]
            w=torch.from_numpy(wav).float().unsqueeze(0)
            m=self.db(self.mel(w))
            mn,mx=m.min(),m.max(); m=(m-mn)/(mx-mn+1e-8); m=m.expand(3,-1,-1)
            return m, torch.from_numpy(self.Y[idx]).float()

    ds = DS(df, Y, DATA)
    n_val = len(ds)//10
    idx = np.arange(len(ds)); np.random.default_rng(42).shuffle(idx)
    val_i = idx[:n_val]; tr_i = idx[n_val:]
    tr_ds = Subset(ds, tr_i); va_ds = Subset(ds, val_i)
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    class SED(nn.Module):
        def __init__(self):
            super().__init__()
            self.bk = timm.create_model("tf_efficientnetv2_m.in21k", pretrained=True, num_classes=0, global_pool="")
            d = self.bk.num_features
            self.gem_p = nn.Parameter(torch.tensor(3.0))
            self.head = nn.Sequential(nn.Linear(d,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,NC))
        def forward(self,x):
            f = self.bk(x)
            p = self.gem_p.clamp(min=1.0)
            f = (f.clamp(min=1e-6)**p).mean(dim=2)**(1.0/p)
            f = f.mean(dim=2)
            return self.head(f)
    model = SED().cuda()
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    bk_p = list(model.bk.parameters())
    head_p = [model.gem_p] + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params":bk_p,"lr":lr*0.1},{"params":head_p,"lr":lr}], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr*0.05,lr], total_steps=n_epochs*len(tr_ld), pct_start=0.1)
    scaler = GradScaler()

    loss_fn = nn.BCEWithLogitsLoss()  # stable, no NaN

    os.makedirs("/vol/eff", exist_ok=True)
    best = 0.0
    for ep in range(n_epochs):
        model.train(); t0 = time.time(); ls, n = 0, 0
        for step, (x,y) in enumerate(tr_ld):
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            opt.zero_grad()
            with autocast("cuda", dtype=torch.float16):
                o = model(x); l = loss_fn(o, y)
            scaler.scale(l).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt); scaler.update(); sched.step()
            ls += l.item(); n += 1
            if step % 100 == 0:
                print(f"  ep{ep+1} step {step}/{len(tr_ld)} loss={ls/n:.4f} t={time.time()-t0:.0f}s", flush=True)
        model.eval(); ap, ay = [], []
        with torch.no_grad():
            for x,y in va_ld:
                x = x.cuda(non_blocking=True)
                with autocast("cuda", dtype=torch.float16): o = model(x)
                ap.append(torch.sigmoid(o).float().cpu().numpy()); ay.append(y.numpy())
        AP = np.concatenate(ap); AY = np.concatenate(ay)
        v = (AY > 0.5).sum(0) > 0
        try: va = roc_auc_score(AY[:,v], AP[:,v], average="macro")
        except: va = 0.0
        print(f"ep{ep+1}: loss={ls/n:.4f} val_auc={va:.4f} t={time.time()-t0:.0f}s", flush=True)
        torch.save({"state":model.state_dict(),"epoch":ep+1,"val_auc":va}, f"/vol/eff/ep{ep+1}.pt")
        if va > best:
            best = va
            torch.save({"state":model.state_dict(),"epoch":ep+1,"val_auc":va}, "/vol/eff/best.pt")
        vol.commit()
    print(f"DONE best_val={best:.4f}")
    return best


@app.local_entrypoint()
def main():
    print(train.remote(n_epochs=18))

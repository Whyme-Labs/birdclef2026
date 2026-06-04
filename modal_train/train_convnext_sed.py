"""ConvNeXt-Small SED from scratch — different architecture lineage than EffV2-M.
Goal: 2nd own-trained backbone for ensemble diversity in V239."""
import modal
app = modal.App("birdclef-convnext-sed")
vol = modal.Volume.from_name("birdclef-effv2m-artifacts", create_if_missing=True)
audio_vol = modal.Volume.from_name("birdclef-audio-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch==2.3.0", "torchaudio==2.3.0", "timm==1.0.11",
                 "numpy", "pandas", "scikit-learn", "soundfile", "librosa"))

@app.function(image=image, gpu="A100", volumes={"/vol": vol, "/audio": audio_vol},
              timeout=12*3600, cpu=4)
def train(n_epochs: int = 14, batch_size: int = 32, lr: float = 1e-4):
    import os, time, numpy as np, pandas as pd
    import torch, torch.nn as nn, torchaudio, timm
    import soundfile as sf
    from torch.utils.data import Dataset, DataLoader, Subset
    from torch.amp import autocast, GradScaler
    SR=32_000; WS=SR*5
    DATA = "/audio"
    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist()); NC=len(PRIMARY); l2i={l:i for i,l in enumerate(PRIMARY)}
    df = pd.read_csv(f"{DATA}/train.csv"); df["primary_label"] = df["primary_label"].astype(str)
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
    print(f"{len(df)} recs, {NC} classes")

    class DS(Dataset):
        def __init__(self, df, Y, root):
            self.df=df; self.Y=Y; self.root=root
            self.mel=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=2048,hop_length=512,n_mels=224,f_min=0,f_max=16000,power=2.0,norm="slaney",mel_scale="htk")
            self.db=torchaudio.transforms.AmplitudeToDB(top_db=80)
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            fn = self.df.iloc[idx]["filename"]
            try: wav, sr = sf.read(f"{self.root}/train_audio/{fn}", dtype="float32", always_2d=False)
            except: return torch.zeros(3,224,313), torch.from_numpy(self.Y[idx]).float()
            if wav.ndim>1: wav=wav.mean(axis=1)
            if sr!=SR:
                import librosa; wav=librosa.resample(wav, orig_sr=sr, target_sr=SR)
            n=len(wav)
            if n<WS: wav=np.tile(wav,(WS//n)+1)[:WS]
            else: st=np.random.randint(0,n-WS+1); wav=wav[st:st+WS]
            w=torch.from_numpy(wav).float().unsqueeze(0); m=self.db(self.mel(w))
            mn,mx=m.min(),m.max(); m=(m-mn)/(mx-mn+1e-8); m=m.expand(3,-1,-1)
            return m, torch.from_numpy(self.Y[idx]).float()
    ds = DS(df, Y, DATA); n_val=len(ds)//10
    idx=np.arange(len(ds)); np.random.default_rng(123).shuffle(idx)
    tr_ld = DataLoader(Subset(ds, idx[n_val:]), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    va_ld = DataLoader(Subset(ds, idx[:n_val]), batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    class SED(nn.Module):
        def __init__(self):
            super().__init__()
            self.bk = timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=True, num_classes=0, global_pool="")
            d = self.bk.num_features
            self.gem_p = nn.Parameter(torch.tensor(3.0))
            self.head = nn.Sequential(nn.Linear(d,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,NC))
        def forward(self,x):
            f = self.bk(x); p = self.gem_p.clamp(min=1.0)
            f = (f.clamp(min=1e-6)**p).mean(dim=2)**(1.0/p); f = f.mean(dim=2)
            return self.head(f)
    m = SED().cuda(); print(f"params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    bk_p = list(m.bk.parameters()); head_p = [m.gem_p] + list(m.head.parameters())
    opt = torch.optim.AdamW([{"params":bk_p,"lr":lr*0.05},{"params":head_p,"lr":lr}], weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr*0.05,lr], total_steps=n_epochs*len(tr_ld), pct_start=0.1)
    scaler = GradScaler(); loss_fn = nn.BCEWithLogitsLoss()
    from sklearn.metrics import roc_auc_score
    os.makedirs("/vol/cnxt", exist_ok=True); best=0.0
    for ep in range(n_epochs):
        m.train(); t0=time.time(); ls,n=0.0,0
        for step,(x,y) in enumerate(tr_ld):
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True); opt.zero_grad()
            with autocast("cuda", dtype=torch.float16): o=m(x); l=loss_fn(o,y)
            scaler.scale(l).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
            scaler.step(opt); scaler.update(); sch.step()
            ls+=l.item(); n+=1
            if step%100==0: print(f"  cnxt ep{ep+1} step {step}/{len(tr_ld)} loss={ls/n:.4f} t={time.time()-t0:.0f}s", flush=True)
        # val
        m.eval(); ap,ay=[],[]
        with torch.no_grad():
            for x,y in va_ld:
                x=x.cuda(non_blocking=True)
                with autocast("cuda", dtype=torch.float16): o=m(x)
                ap.append(torch.sigmoid(o).float().cpu().numpy()); ay.append(y.numpy())
        AP=np.concatenate(ap); AY=np.concatenate(ay)
        # Per-class AUC with skip-on-error
        per_class_aucs = []
        for c in range(NC):
            yc = (AY[:,c] > 0.5).astype(int)
            if yc.sum() == 0 or yc.sum() == len(yc): continue
            try: per_class_aucs.append(roc_auc_score(yc, AP[:,c]))
            except: pass
        va = float(np.mean(per_class_aucs)) if per_class_aucs else 0.0
        print(f"cnxt ep{ep+1}: loss={ls/n:.4f} val_auc={va:.4f} (over {len(per_class_aucs)} classes) t={time.time()-t0:.0f}s", flush=True)
        torch.save({"state":m.state_dict(),"epoch":ep+1,"val_auc":va}, f"/vol/cnxt/ep{ep+1}.pt")
        if va > best: best=va; torch.save({"state":m.state_dict(),"epoch":ep+1,"val_auc":va}, "/vol/cnxt/best.pt")
        vol.commit()
    print(f"DONE best={best:.4f}"); return best

@app.local_entrypoint()
def main(): print(train.remote())

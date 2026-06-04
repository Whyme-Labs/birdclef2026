"""Second variant: ProtoSSM(d_model=384) 4-fold, different seed.
Adds ensemble diversity to V233's d=320 sidecar.
"""
import modal
app = modal.App("birdclef-protossm-v2")
volume = modal.Volume.from_name("birdclef-protossm320-artifacts", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.3.0", "numpy", "pandas", "scikit-learn", "pyarrow")

@app.function(image=image, gpu="A100", volumes={"/vol": volume}, timeout=3600)
def train_v2(d_model: int = 384, seed: int = 137, n_epochs: int = 80, prefix: str = "moe_protossm384"):
    import os, time, numpy as np, pandas as pd, torch
    import torch.nn as nn, torch.nn.functional as F
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score

    arr = np.load("/vol/inputs/all_perch_arrays.npz")
    emb_full = arr["emb_full"].astype(np.float32)
    scores_full = arr["scores_full_raw"].astype(np.float32)
    meta = pd.read_parquet("/vol/inputs/all_perch_meta.parquet")
    tax = pd.read_csv("/vol/inputs/taxonomy.csv")
    PRIMARY_LABELS = sorted(tax["primary_label"].astype(str).tolist())
    N_CLASSES = len(PRIMARY_LABELS); l2i = {l:i for i,l in enumerate(PRIMARY_LABELS)}
    N_WINDOWS = 12

    sl = pd.read_csv("/vol/inputs/train_soundscapes_labels.csv")
    sl["end_sec"] = pd.to_timedelta(sl["end"]).dt.total_seconds().astype(int)
    sl["row_id"] = sl["filename"].str.replace(".ogg","",regex=False) + "_" + sl["end_sec"].astype(str)
    sl_by_row = {}
    for _, r in sl.iterrows():
        labs = str(r["primary_label"]).split(";")
        sl_by_row.setdefault(r["row_id"], set()).update(s.strip() for s in labs)
    Y = np.zeros((len(meta), N_CLASSES), dtype=np.float32)
    for i, rid in enumerate(meta["row_id"]):
        for s in sl_by_row.get(rid, []):
            if s in l2i: Y[i, l2i[s]] = 1.0
    labeled_mask = Y.sum(axis=1) > 0
    n_files_full = len(emb_full) // N_WINDOWS
    file_has_label = labeled_mask.reshape(n_files_full, N_WINDOWS).any(axis=1)
    keep_files = np.where(file_has_label)[0]
    keep_idx = np.concatenate([np.arange(f*N_WINDOWS,(f+1)*N_WINDOWS) for f in keep_files])
    emb_full = emb_full[keep_idx]; scores_full = scores_full[keep_idx]; Y = Y[keep_idx]
    meta_keep = meta.iloc[keep_idx].reset_index(drop=True)
    n_files = len(emb_full) // N_WINDOWS

    sites_unique = sorted(meta_keep["site"].dropna().astype(str).unique())
    site2i = {s:i+1 for i,s in enumerate(sites_unique)}
    N_SITES_CAP = max(32, len(sites_unique)+2)
    fnames = meta_keep.drop_duplicates("filename")["filename"].tolist()
    site_ids = np.array([min(site2i.get(str(meta_keep.loc[meta_keep["filename"]==fn,"site"].iloc[0]),0),N_SITES_CAP-1) for fn in fnames], dtype=np.int64)
    hour_ids = np.array([int(meta_keep.loc[meta_keep["filename"]==fn,"hour_utc"].iloc[0])%24 for fn in fnames], dtype=np.int64)

    class _SSM(nn.Module):
        def __init__(self, d_model, d_state=24, d_conv=4):
            super().__init__(); self.d_state=d_state
            self.in_proj=nn.Linear(d_model,2*d_model,bias=False)
            self.conv1d=nn.Conv1d(d_model,d_model,d_conv,padding=d_conv-1,groups=d_model)
            self.dt_proj=nn.Linear(d_model,d_model,bias=True)
            A=torch.arange(1,d_state+1,dtype=torch.float32).unsqueeze(0).expand(d_model,-1)
            self.A_log=nn.Parameter(torch.log(A)); self.D=nn.Parameter(torch.ones(d_model))
            self.B_proj=nn.Linear(d_model,d_state,bias=False); self.C_proj=nn.Linear(d_model,d_state,bias=False)
        def forward(self,x):
            B_sz,T,D=x.shape; xz=self.in_proj(x); x_ssm,_=xz.chunk(2,dim=-1)
            x_conv=F.silu(self.conv1d(x_ssm.transpose(1,2))[:,:,:T].transpose(1,2))
            dt=F.softplus(self.dt_proj(x_conv)); A=-torch.exp(self.A_log)
            B=self.B_proj(x_conv); C=self.C_proj(x_conv)
            h=torch.zeros(B_sz,D,self.d_state,device=x.device); ys=[]
            for t in range(T):
                dA=torch.exp(A[None]*dt[:,t,:,None]); dB=dt[:,t,:,None]*B[:,t,None,:]
                h=h*dA+x[:,t,:,None]*dB; ys.append((h*C[:,t,None,:]).sum(-1))
            return torch.stack(ys,dim=1)+x*self.D[None,None,:]

    class ProtoSSM(nn.Module):
        def __init__(self, d_model, n_classes=N_CLASSES, n_sites=N_SITES_CAP, n_layers=3, d_state=24):
            super().__init__()
            self.input_proj=nn.Sequential(nn.Linear(1536,d_model),nn.LayerNorm(d_model),nn.GELU(),nn.Dropout(0.12))
            self.pos_enc=nn.Parameter(torch.randn(1,12,d_model)*0.02)
            self.site_emb=nn.Embedding(n_sites,16); self.hour_emb=nn.Embedding(24,16)
            self.meta_proj=nn.Linear(32,d_model)
            self.ssm_fwd=nn.ModuleList([_SSM(d_model,d_state) for _ in range(n_layers)])
            self.ssm_bwd=nn.ModuleList([_SSM(d_model,d_state) for _ in range(n_layers)])
            self.merge=nn.ModuleList([nn.Linear(2*d_model,d_model) for _ in range(n_layers)])
            self.norms=nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.drop=nn.Dropout(0.12)
            self.prototypes=nn.Parameter(torch.randn(n_classes,d_model)*0.02)
            self.proto_temp=nn.Parameter(torch.tensor(5.0))
            self.class_bias=nn.Parameter(torch.zeros(n_classes))
            self.fusion_alpha=nn.Parameter(torch.zeros(n_classes))
        def forward(self,emb,teacher_logits,site_ids,hours):
            h=self.input_proj(emb)+self.pos_enc[:,:emb.size(1),:]
            meta=self.meta_proj(torch.cat([self.site_emb(site_ids),self.hour_emb(hours)],dim=-1))
            h=h+meta[:,None,:]
            for fwd,bwd,merge,norm in zip(self.ssm_fwd,self.ssm_bwd,self.merge,self.norms):
                res=h; h_f=fwd(h); h_b=bwd(h.flip(1)).flip(1)
                h=norm(res+self.drop(merge(torch.cat([h_f,h_b],dim=-1))))
            h_n=F.normalize(h,dim=-1); p_n=F.normalize(self.prototypes,dim=-1)
            sim=torch.matmul(h_n,p_n.T)*F.softplus(self.proto_temp)+self.class_bias[None,None,:]
            alpha=torch.sigmoid(self.fusion_alpha)[None,None,:]
            return alpha*sim+(1-alpha)*teacher_logits

    emb_f = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f = scores_full.reshape(n_files, N_WINDOWS, -1)
    Y_f   = Y.reshape(n_files, N_WINDOWS, -1)
    rng = np.random.default_rng(seed); file_idx = np.arange(n_files); rng.shuffle(file_idx)
    kf = KFold(n_splits=4, shuffle=False)
    device = "cuda"; print(f"d_model={d_model}, seed={seed}, training 4 folds, {n_epochs} epochs")

    os.makedirs("/vol/checkpoints", exist_ok=True)
    aucs = []
    for fold_id, (tr_pos, va_pos) in enumerate(kf.split(file_idx), 1):
        tr_i = file_idx[tr_pos]; va_i = file_idx[va_pos]
        model = ProtoSSM(d_model).to(device)
        with torch.no_grad():
            emb_tr_flat = torch.from_numpy(emb_f[tr_i].reshape(-1,1536)).float().to(device)
            Y_tr_flat = torch.from_numpy(Y_f[tr_i].reshape(-1,N_CLASSES)).float().to(device)
            h0 = model.input_proj(emb_tr_flat)
            for c in range(N_CLASSES):
                m = Y_tr_flat[:,c] > 0.5
                if m.sum() > 0:
                    model.prototypes.data[c] = F.normalize(h0[m].mean(0), dim=0)
        e_tr = torch.from_numpy(emb_f[tr_i]).float().to(device)
        l_tr = torch.from_numpy(log_f[tr_i]).float().to(device)
        y_tr = torch.from_numpy(Y_f[tr_i]).float().to(device)
        s_tr = torch.from_numpy(site_ids[tr_i]).long().to(device)
        h_tr = torch.from_numpy(hour_ids[tr_i]).long().to(device)
        e_va = torch.from_numpy(emb_f[va_i]).float().to(device)
        l_va = torch.from_numpy(log_f[va_i]).float().to(device)
        y_va = Y_f[va_i]
        s_va = torch.from_numpy(site_ids[va_i]).long().to(device)
        h_va = torch.from_numpy(hour_ids[va_i]).long().to(device)
        pos_cnt = y_tr.sum(dim=(0,1))
        pw = ((y_tr.shape[0]*y_tr.shape[1] - pos_cnt)/(pos_cnt+1)).clamp(max=25.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=n_epochs, steps_per_epoch=1, pct_start=0.1, anneal_strategy="cos")
        swa_model = torch.optim.swa_utils.AveragedModel(model); swa_start = int(n_epochs*0.65)
        swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)
        best = 0.0
        for ep in range(n_epochs):
            model.train()
            out = model(e_tr, l_tr, s_tr, h_tr)
            loss = F.binary_cross_entropy_with_logits(out, y_tr, pos_weight=pw[None,None,:]) + 0.15*F.mse_loss(out, l_tr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if ep >= swa_start: swa_model.update_parameters(model); swa_sched.step()
            else: sched.step()
            if (ep+1)%20==0 or ep==n_epochs-1:
                model.eval()
                with torch.no_grad():
                    o_v = model(e_va, l_va, s_va, h_va).cpu().numpy()
                p_v = 1/(1+np.exp(-np.clip(o_v,-30,30)))
                pf = p_v.reshape(-1,N_CLASSES); yf = y_va.reshape(-1,N_CLASSES)
                valid = (yf.sum(0)>0)
                try: va = roc_auc_score(yf[:,valid], pf[:,valid], average="macro")
                except: va = 0.0
                print(f"  fold{fold_id} ep{ep+1}: val_macroAUC={va:.4f}")
                best = max(best, va)
        try: torch.optim.swa_utils.update_bn(e_tr.unsqueeze(0), swa_model)
        except: pass
        save = f"/vol/checkpoints/{prefix}_seed{seed}_fold{fold_id}.pt"
        torch.save(swa_model.module.state_dict() if hasattr(swa_model,'module') else swa_model.state_dict(), save)
        print(f"  saved {save}, best={best:.4f}")
        aucs.append(best)
    print(f"DONE prefix={prefix} seed={seed}: {aucs}, mean={np.mean(aucs):.4f}")
    volume.commit()
    return aucs

@app.local_entrypoint()
def main():
    # Variant 1: d=384, seed=137
    aucs1 = train_v2.remote(d_model=384, seed=137, prefix="moe_protossm384")
    print(f"d=384 seed=137: mean={sum(aucs1)/len(aucs1):.4f}")

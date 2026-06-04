"""ProtoSSM(d=320) 4-fold, seed=314 — for ensemble diversity."""
import modal
app = modal.App("p320-seed-314")
vol = modal.Volume.from_name("birdclef-protossm320-artifacts", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.3.0","numpy","pandas","scikit-learn","pyarrow")

@app.function(image=image, gpu="A100", volumes={"/vol":vol}, timeout=1800)
def train():
    import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    arr = np.load("/vol/inputs/all_perch_arrays.npz")
    emb = arr["emb_full"].astype(np.float32); sc = arr["scores_full_raw"].astype(np.float32)
    meta = pd.read_parquet("/vol/inputs/all_perch_meta.parquet")
    tax = pd.read_csv("/vol/inputs/taxonomy.csv")
    PR = sorted(tax["primary_label"].astype(str).tolist()); NC = len(PR); l2i = {l:i for i,l in enumerate(PR)}
    sl = pd.read_csv("/vol/inputs/train_soundscapes_labels.csv")
    sl["end_sec"] = pd.to_timedelta(sl["end"]).dt.total_seconds().astype(int)
    sl["row_id"] = sl["filename"].str.replace(".ogg","",regex=False)+"_"+sl["end_sec"].astype(str)
    sbr = {}
    for _,r in sl.iterrows(): sbr.setdefault(r["row_id"],set()).update(s.strip() for s in str(r["primary_label"]).split(";"))
    Y = np.zeros((len(meta),NC),dtype=np.float32)
    for i,rid in enumerate(meta["row_id"]):
        for s in sbr.get(rid,[]):
            if s in l2i: Y[i,l2i[s]]=1.0
    fhl = (Y.sum(1)>0).reshape(len(meta)//12,12).any(1); kf = np.where(fhl)[0]
    ki = np.concatenate([np.arange(f*12,(f+1)*12) for f in kf])
    emb,sc,Y = emb[ki],sc[ki],Y[ki]; mk = meta.iloc[ki].reset_index(drop=True); nf = len(emb)//12
    su = sorted(mk["site"].dropna().astype(str).unique()); s2i = {s:i+1 for i,s in enumerate(su)}; NCAP = max(32,len(su)+2)
    fns = mk.drop_duplicates("filename")["filename"].tolist()
    si = np.array([min(s2i.get(str(mk.loc[mk["filename"]==fn,"site"].iloc[0]),0),NCAP-1) for fn in fns],dtype=np.int64)
    hi = np.array([int(mk.loc[mk["filename"]==fn,"hour_utc"].iloc[0])%24 for fn in fns],dtype=np.int64)
    class S(nn.Module):
        def __init__(self,d,ds=24):
            super().__init__(); self.ds=ds; self.ip=nn.Linear(d,2*d,bias=False); self.c=nn.Conv1d(d,d,4,padding=3,groups=d)
            self.dt=nn.Linear(d,d,bias=True); A=torch.arange(1,ds+1,dtype=torch.float32).unsqueeze(0).expand(d,-1)
            self.A=nn.Parameter(torch.log(A)); self.D=nn.Parameter(torch.ones(d))
            self.B=nn.Linear(d,ds,bias=False); self.C=nn.Linear(d,ds,bias=False)
        def forward(self,x):
            B_sz,T,D=x.shape; xz=self.ip(x); xs,_=xz.chunk(2,-1)
            xc=F.silu(self.c(xs.transpose(1,2))[:,:,:T].transpose(1,2))
            dt=F.softplus(self.dt(xc)); A=-torch.exp(self.A); B=self.B(xc); C=self.C(xc)
            h=torch.zeros(B_sz,D,self.ds,device=x.device); ys=[]
            for t in range(T):
                dA=torch.exp(A[None]*dt[:,t,:,None]); dB=dt[:,t,:,None]*B[:,t,None,:]
                h=h*dA+x[:,t,:,None]*dB; ys.append((h*C[:,t,None,:]).sum(-1))
            return torch.stack(ys,1)+x*self.D[None,None,:]
    class P(nn.Module):
        def __init__(self,d=320,nc=NC,ns=NCAP,nl=3,ds=24):
            super().__init__()
            self.proj=nn.Sequential(nn.Linear(1536,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(0.12))
            self.pe=nn.Parameter(torch.randn(1,12,d)*0.02)
            self.se=nn.Embedding(ns,16); self.he=nn.Embedding(24,16); self.mp=nn.Linear(32,d)
            self.f=nn.ModuleList([S(d,ds) for _ in range(nl)]); self.b=nn.ModuleList([S(d,ds) for _ in range(nl)])
            self.m=nn.ModuleList([nn.Linear(2*d,d) for _ in range(nl)]); self.n=nn.ModuleList([nn.LayerNorm(d) for _ in range(nl)])
            self.dr=nn.Dropout(0.12); self.pt=nn.Parameter(torch.randn(nc,d)*0.02)
            self.tp=nn.Parameter(torch.tensor(5.0)); self.cb=nn.Parameter(torch.zeros(nc)); self.fa=nn.Parameter(torch.zeros(nc))
        def forward(self,e,tl,si_,hi_):
            h=self.proj(e)+self.pe[:,:e.size(1),:]; m=self.mp(torch.cat([self.se(si_),self.he(hi_)],-1)); h=h+m[:,None,:]
            for f,b,mm,nn_ in zip(self.f,self.b,self.m,self.n):
                r=h; hf=f(h); hb=b(h.flip(1)).flip(1); h=nn_(r+self.dr(mm(torch.cat([hf,hb],-1))))
            hn=F.normalize(h,dim=-1); pn=F.normalize(self.pt,dim=-1)
            si2=torch.matmul(hn,pn.T)*F.softplus(self.tp)+self.cb[None,None,:]
            al=torch.sigmoid(self.fa)[None,None,:]; return al*si2+(1-al)*tl
    ef=emb.reshape(nf,12,-1); lf=sc.reshape(nf,12,-1); yf=Y.reshape(nf,12,-1)
    rng=np.random.default_rng(314); fi=np.arange(nf); rng.shuffle(fi); kfs=KFold(4,shuffle=False)
    os.makedirs("/vol/checkpoints",exist_ok=True); aucs=[]
    for fid,(tp,vp) in enumerate(kfs.split(fi),1):
        ti=fi[tp]; vi=fi[vp]; m=P(320).cuda()
        with torch.no_grad():
            eft=torch.from_numpy(ef[ti].reshape(-1,1536)).float().cuda()
            yft=torch.from_numpy(yf[ti].reshape(-1,NC)).float().cuda(); h0=m.proj(eft)
            for c in range(NC):
                mm=yft[:,c]>0.5
                if mm.sum()>0: m.pt.data[c]=F.normalize(h0[mm].mean(0),dim=0)
        et=torch.from_numpy(ef[ti]).float().cuda(); lt=torch.from_numpy(lf[ti]).float().cuda()
        yt=torch.from_numpy(yf[ti]).float().cuda(); st=torch.from_numpy(si[ti]).long().cuda(); ht=torch.from_numpy(hi[ti]).long().cuda()
        ev=torch.from_numpy(ef[vi]).float().cuda(); lv=torch.from_numpy(lf[vi]).float().cuda()
        yv=yf[vi]; sv=torch.from_numpy(si[vi]).long().cuda(); hv=torch.from_numpy(hi[vi]).long().cuda()
        pc=yt.sum((0,1)); pw=((yt.shape[0]*yt.shape[1]-pc)/(pc+1)).clamp(max=25.0)
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-3)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-3,epochs=80,steps_per_epoch=1,pct_start=0.1,anneal_strategy="cos")
        sw=torch.optim.swa_utils.AveragedModel(m); sws=torch.optim.swa_utils.SWALR(opt,swa_lr=4e-4); ss=52
        best=0
        for ep in range(80):
            m.train(); o=m(et,lt,st,ht)
            l=F.binary_cross_entropy_with_logits(o,yt,pos_weight=pw[None,None,:])+0.15*F.mse_loss(o,lt)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            if ep>=ss: sw.update_parameters(m); sws.step()
            else: sch.step()
            if (ep+1)%20==0 or ep==79:
                m.eval()
                with torch.no_grad(): ov=m(ev,lv,sv,hv).cpu().numpy()
                pv=1/(1+np.exp(-np.clip(ov,-30,30))); pvf=pv.reshape(-1,NC); yvf=yv.reshape(-1,NC); v=yvf.sum(0)>0
                try: va=roc_auc_score(yvf[:,v],pvf[:,v],average="macro")
                except: va=0.0
                best=max(best,va)
        try: torch.optim.swa_utils.update_bn(et.unsqueeze(0),sw)
        except: pass
        sp=f"/vol/checkpoints/moe_protossm320_seed314_fold{fid}.pt"
        torch.save(sw.module.state_dict() if hasattr(sw,'module') else sw.state_dict(), sp)
        print(f"fold{fid} best={best:.4f}",flush=True); aucs.append(best)
    vol.commit(); print(f"DONE seed=314: {aucs} mean={sum(aucs)/4:.4f}"); return aucs

@app.local_entrypoint()
def main(): print(train.remote())

import torch, torch.nn as nn, timm, os
CKPTS="sed_ens_ckpts"; OUT="own_sed_ensemble_onnx"; os.makedirs(OUT,exist_ok=True)
N_MELS,N_TIME,NC=256,313,234
TAGS=["effv2s","convnext","effb3","seresnext"]
class SEDModel(nn.Module):
    def __init__(self,bn,n=234):
        super().__init__()
        self.bk=timm.create_model(bn,pretrained=False,num_classes=0,global_pool="",in_chans=1)
        d=self.bk.num_features
        self.gem_p=nn.Parameter(torch.tensor(3.0));self.drop=nn.Dropout(0.3)
        self.frame_head=nn.Linear(d,n);self.att_head=nn.Linear(d,n)
    def forward(self,mel):
        f=self.bk(mel);p=self.gem_p.clamp(min=1.0)
        ff=(f.clamp(min=1e-6).pow(p)).mean(dim=2).pow(1.0/p);ft=ff.transpose(1,2)
        fr=self.frame_head(ft);at=torch.softmax(self.att_head(ft),dim=1)
        return (at*fr).sum(dim=1),fr
for i,t in enumerate(TAGS):
    ck=torch.load(f"{CKPTS}/fold0_best_{t}.pt",map_location="cpu")
    m=SEDModel(ck["backbone"],NC);m.load_state_dict(ck["state"]);m.eval()
    d=torch.randn(1,1,N_MELS,N_TIME)
    p=f"{OUT}/sed_fold{5+i}.onnx"  # 5-8 (public are 0-4)
    torch.onnx.export(m,d,p,input_names=["mel"],output_names=["clip_logits","framewise_logits"],
        dynamic_axes={"mel":{0:"batch"},"clip_logits":{0:"batch"},"framewise_logits":{0:"batch"}},
        opset_version=17,do_constant_folding=True)
    print(f"exported {p} ({t}, {ck['backbone']}, heldout {ck['heldout_auc']:.4f})")
print("done")

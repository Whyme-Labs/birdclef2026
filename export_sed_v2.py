"""Export trained local SED checkpoints to ONNX matching public SED I/O.

Public SED I/O: input 'mel' (B,1,256,313) -> 'clip_logits' (B,234), 'framewise_logits' (B,W',234)
Our SEDModel outputs (clip_logits, frame_logits) — export with matching names so the
kernel's SED loop ( 0.5*sigmoid(clip) + 0.5*sigmoid(frame_max) ) works unchanged.
"""
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

CKPT_DIR = "/home/soh/birdclef-2026/checkpoints/sed_v2"
OUT_DIR = "/home/soh/birdclef-2026/own_sed_v2_onnx"
os.makedirs(OUT_DIR, exist_ok=True)
N_MELS, N_TIME, NC = 256, 313, 234


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


def export_one(ckpt_path, out_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")
    model = SEDModel(backbone, NC)
    model.load_state_dict(ckpt["state"])
    model.eval()
    dummy = torch.randn(1, 1, N_MELS, N_TIME)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["mel"],
        output_names=["clip_logits", "framewise_logits"],
        dynamic_axes={"mel": {0: "batch"},
                      "clip_logits": {0: "batch"},
                      "framewise_logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"  exported {out_path}  (epoch {ckpt['epoch']}, val_auc {ckpt.get('ss_val_auc', ckpt.get('val_auc', 0)):.4f})")


if __name__ == "__main__":
    best = sorted(glob.glob(f"{CKPT_DIR}/fold*_best.pt"))
    if not best:
        print("No checkpoints found")
        raise SystemExit(1)
    for ck in best:
        fold = os.path.basename(ck).split("_")[0].replace("fold", "")
        export_one(ck, f"{OUT_DIR}/sed_fold{fold}.onnx")
    print(f"Done. {len(best)} models in {OUT_DIR}")

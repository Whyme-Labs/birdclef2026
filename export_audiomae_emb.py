"""Export AudioMAE backbone-only (returns 768-d global-pool embedding) for V160 template retrieval."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import timm
from timm.layers.pos_embed import resample_abs_pos_embed


class AudioMAEBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
            pretrained=False, num_classes=0,
            img_size=(320, 128), dynamic_img_size=False,
        )

    def forward(self, x):
        feats = self.backbone(x)  # (B, 768) global-pool (avg over patch tokens)
        # L2-normalize so cosine similarity = dot product downstream
        feats = feats / torch.clamp(feats.norm(dim=-1, keepdim=True), min=1e-9)
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/audiomae_ft_v159/best_fold0.pt")
    ap.add_argument("--out", default="kaggle_model/audiomae_emb_v160.onnx")
    args = ap.parse_args()

    print(f"Loading {args.ckpt}…")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    print(f"checkpoint val_auc={ckpt.get('val_auc', 'n/a')}")

    model = AudioMAEBackbone()

    if "backbone.pos_embed" in sd:
        old_pe = sd["backbone.pos_embed"]
        new_pe = resample_abs_pos_embed(
            old_pe, new_size=(20, 8), old_size=(64, 8), num_prefix_tokens=1,
        )
        sd["backbone.pos_embed"] = new_pe
        print(f"resampled pos_embed: {tuple(old_pe.shape)} -> {tuple(new_pe.shape)}")

    backbone_state = {
        k: v for k, v in sd.items() if k.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        print(f"load: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval()

    dummy = torch.zeros(1, 1, 320, 128)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["mel"], output_names=["embedding"],
        dynamic_axes={"mel": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {out_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    test = rng.uniform(0, 1, (3, 1, 320, 128)).astype(np.float32)
    out_onnx = sess.run(None, {inp: test})[0]
    with torch.no_grad():
        out_torch = model(torch.from_numpy(test)).numpy()
    diff = np.abs(out_onnx - out_torch).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}, output shape: {out_onnx.shape}")
    print(f"sample row norm: {np.linalg.norm(out_onnx[0]):.4f} (should be 1.0)")


if __name__ == "__main__":
    main()

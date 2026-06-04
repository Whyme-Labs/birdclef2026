"""Export AudioMAE finetuned classifier to ONNX for V159 inference."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import timm


class AudioMAEClassifier(nn.Module):
    def __init__(self, num_classes=234, dropout=0.2):
        super().__init__()
        # dynamic_img_size=True triggers bicubic_aa upsampling on pos_embed which
        # ONNX opset<=20 doesn't support. We fix input to (320, 128) at export time.
        self.backbone = timm.create_model(
            'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m',
            pretrained=False, num_classes=0,
            img_size=(320, 128), dynamic_img_size=False,
        )
        self.embed_dim = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return torch.sigmoid(self.head(feats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/audiomae_ft_v159/best_fold0.pt")
    ap.add_argument("--out", default="kaggle_model/audiomae_v159.onnx")
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--time_frames", type=int, default=320)
    args = ap.parse_args()

    print(f"Loading {args.ckpt}…")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    print(f"checkpoint val_auc={ckpt.get('val_auc', 'n/a')}, epoch={ckpt.get('epoch', 'n/a')}")

    model = AudioMAEClassifier(num_classes=234, dropout=0.2)

    # Resample the 513-token pos_embed (original 1024×128) down to the 161-token
    # version expected at our 320×128 fixed input, then load the rest verbatim.
    from timm.layers.pos_embed import resample_abs_pos_embed
    if "backbone.pos_embed" in sd:
        old_pe = sd["backbone.pos_embed"]
        new_pe = resample_abs_pos_embed(
            old_pe, new_size=(20, 8), old_size=(64, 8),
            num_prefix_tokens=1,
        )
        sd["backbone.pos_embed"] = new_pe
        print(f"resampled pos_embed: {tuple(old_pe.shape)} -> {tuple(new_pe.shape)}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"load: missing={missing}, unexpected={unexpected}")
    model.eval()

    # AudioMAE expects (B, 1, time_frames, n_mels). Target ~5s @ 32kHz, hop=500 → 320 frames, 128 mels.
    dummy = torch.zeros(1, 1, args.time_frames, args.n_mels)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["mel"], output_names=["probs"],
        dynamic_axes={"mel": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {out_path}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    test = rng.uniform(0, 1, (3, 1, args.time_frames, args.n_mels)).astype(np.float32)
    out_onnx = sess.run(None, {inp: test})[0]
    with torch.no_grad():
        out_torch = model(torch.from_numpy(test)).numpy()
    diff = np.abs(out_onnx - out_torch).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}, output shape: {out_onnx.shape}")


if __name__ == "__main__":
    main()

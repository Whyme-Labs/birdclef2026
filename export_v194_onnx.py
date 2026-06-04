"""Export V194 iter 1 (SEDModelV2 with effnetv2-s backbone) to ONNX.

Used in V196 = V137 with V2S = avg(V194_iter1, R2_fold1) — same compute as V137,
sidesteps the recent Kaggle compute-tightening that's been killing 4-probe stacks.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.models_v2 import SEDModelV2


class V194Wrapper(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        out = self.m(x)
        # SEDModelV2 returns AttentionSEDHead output: dict or tuple?
        # Match V137 V2S ONNX convention: output = sigmoid(clip_logits).
        if isinstance(out, dict):
            clip = out.get("clipwise_logit", out.get("clipwise"))
        elif isinstance(out, tuple):
            clip = out[0]
        else:
            clip = out
        clip = torch.clamp(clip, min=-30.0, max=30.0)
        return torch.sigmoid(clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v194_ns_iter1/best_fold0.pt")
    ap.add_argument("--output", default="kaggle_model/effv2s_v194_iter1.onnx")
    ap.add_argument("--n_mels", type=int, default=224)
    ap.add_argument("--n_classes", type=int, default=234)
    ap.add_argument("--backbone", default="tf_efficientnetv2_s.in21k")
    args = ap.parse_args()

    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    model = SEDModelV2(backbone=args.backbone, num_classes=args.n_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    wrap = V194Wrapper(model).eval()

    # Probe forward to see output shape/type
    dummy = torch.rand(1, 3, args.n_mels, 313)
    with torch.no_grad():
        try:
            o = wrap(dummy)
            print(f"  forward OK: {o.shape}, mean={o.mean().item():.4f}")
        except Exception as e:
            print(f"  forward shape probe failed: {e}")
            # Try to inspect what the raw model returns
            with torch.no_grad():
                raw = model(dummy)
            print(f"  raw model output type: {type(raw)}")
            if isinstance(raw, dict):
                print(f"  keys: {list(raw.keys())}")
            elif isinstance(raw, tuple):
                print(f"  tuple len {len(raw)}, shapes: {[r.shape for r in raw]}")
            else:
                print(f"  shape: {raw.shape}")
            return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap, dummy, str(out_path),
        input_names=["input"], output_names=["probs"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"Exported {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path))
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = wrap(dummy).numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f"Max diff: {diff:.2e} — {'OK' if diff < 1e-4 else 'WARNING'}")


if __name__ == "__main__":
    main()

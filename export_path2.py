"""
Export trained Path 2 multi-window model to ONNX for Kaggle inference.

Output shape: (B, W=12, 3, n_mels, T) → (B, W=12, num_classes) probabilities (sigmoid).

Sigmoid is included so the inference notebook can treat outputs uniformly with
other ensemble members (which already output probabilities).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.config import Config
from src.models_path2 import MultiWindowSED


class Path2Wrapper(nn.Module):
    """Apply sigmoid + return only clip-level (B, W, C) probabilities."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        clip_logits, _ = self.m(x)
        return torch.sigmoid(clip_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="kaggle_model/path2_v1.onnx")
    ap.add_argument("--n_mels", type=int, default=224)
    ap.add_argument("--n_windows", type=int, default=12)
    ap.add_argument("--cw_layers", type=int, default=None,
                    help="Override cw_attn layer count; otherwise read from checkpoint args.")
    ap.add_argument("--cw_heads", type=int, default=None)
    ap.add_argument("--cw_dropout", type=float, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    saved_cfg = ckpt.get("config", {})
    saved_args = ckpt.get("args", {})
    label_cols = ckpt.get("label_cols", [])

    cfg = Config()
    for k, v in saved_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_mels = args.n_mels
    cfg.num_classes = len(label_cols) if label_cols else cfg.num_classes

    cw_layers = args.cw_layers if args.cw_layers is not None else saved_args.get("cw_layers", 2)
    cw_heads = args.cw_heads if args.cw_heads is not None else saved_args.get("cw_heads", 4)
    cw_dropout = args.cw_dropout if args.cw_dropout is not None else saved_args.get("cw_dropout", 0.1)
    print(f"Building MultiWindowSED with cw_layers={cw_layers}, cw_heads={cw_heads}, cw_dropout={cw_dropout}")

    model = MultiWindowSED(
        cfg, n_windows=args.n_windows,
        cw_layers=cw_layers, cw_heads=cw_heads, cw_dropout=cw_dropout,
    )
    model.load_state_dict(sd)
    model.eval()
    wrapped = Path2Wrapper(model)

    cfg_T = (cfg.target_samples // cfg.hop_length) + 1
    dummy = torch.zeros(1, args.n_windows, cfg.in_chans, cfg.n_mels, cfg_T)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped, dummy, str(out_path),
        input_names=["mel_5d"], output_names=["probs"],
        dynamic_axes={
            "mel_5d": {0: "batch"},
            "probs": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported to {out_path}")
    print(f"Input dummy shape: {tuple(dummy.shape)}")
    print(f"cw_gate at export = {model.cw_gate.item():.4f}")
    print(f"Val AUC at save: {ckpt.get('val_auc', 'n/a')}")

    # Verify with the SAME deterministic input on both runtimes (no extra randomness
    # between PyTorch forward and ONNX forward — that caused spurious "large diff").
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    test = rng.uniform(0, 1, (2, args.n_windows, cfg.in_chans, cfg.n_mels, cfg_T)).astype(np.float32)
    out_onnx = sess.run(None, {inp: test})[0]
    with torch.no_grad():
        out_torch = wrapped(torch.from_numpy(test)).numpy()
    diff = np.abs(out_onnx - out_torch).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}, output shape: {out_onnx.shape}")
    if diff > 1e-3:
        print("WARNING: large diff", file=sys.stderr)


if __name__ == "__main__":
    main()

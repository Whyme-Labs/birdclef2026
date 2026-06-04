"""V177 prep: average V2S R2 fold0 + fold1 weights (model soup) → single ONNX.

Model soups (Wortsman et al. 2022): averaging weights of models with identical
architecture trained from the same init with different data splits often
outperforms the best single model AND prediction averaging.

V175 used PREDICTION averaging (output mean) and got 0.941 (identity with V137).
V177 uses WEIGHT averaging — single forward pass, may behave differently due to
implicit regularization in weight space.
"""
import torch, json, argparse
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.config import Config as ExperimentConfig
from src.models import SEDModel


class SEDModelForExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        clip_logits, _ = self.model(x)
        clip_logits = torch.clamp(clip_logits, min=-30.0, max=30.0)
        return torch.sigmoid(clip_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt0", default="checkpoints/effv2s_r2_fold0/best_fold0.pt")
    ap.add_argument("--ckpt1", default="checkpoints/effv2s_r2_fold1/best_fold1.pt")
    ap.add_argument("--output", default="kaggle_model/effv2s_r2_soup.onnx")
    args = ap.parse_args()

    print(f"Loading fold 0: {args.ckpt0}")
    ck0 = torch.load(args.ckpt0, map_location="cpu", weights_only=False)
    print(f"  epoch={ck0.get('epoch')}, val_auc={ck0.get('val_auc')}")
    print(f"Loading fold 1: {args.ckpt1}")
    ck1 = torch.load(args.ckpt1, map_location="cpu", weights_only=False)
    print(f"  epoch={ck1.get('epoch')}, val_auc={ck1.get('val_auc')}")

    sd0 = ck0["model_state_dict"]
    sd1 = ck1["model_state_dict"]

    # Verify keys match
    assert set(sd0.keys()) == set(sd1.keys()), f"Key mismatch: {set(sd0.keys()) ^ set(sd1.keys())}"

    soup = {}
    for k in sd0:
        v0, v1 = sd0[k], sd1[k]
        if v0.dtype == v1.dtype and v0.dtype.is_floating_point:
            soup[k] = (v0 + v1) / 2.0
        else:
            # Integer (e.g., num_batches_tracked) — take fold 0
            soup[k] = v0

    print(f"Soup state_dict: {len(soup)} keys")

    # Build config from fold 0
    cfg_dict = ck0["config"]
    config = ExperimentConfig(**{k: v for k, v in cfg_dict.items()
                                  if k in ExperimentConfig.__dataclass_fields__})
    config.num_classes = 234
    print(f"Backbone: {config.backbone}, n_mels: {config.n_mels}")

    model = SEDModel(config)
    model.load_state_dict(soup, strict=True)
    model.eval()

    export_model = SEDModelForExport(model).eval()

    dummy = torch.rand(1, 3, config.n_mels, 313)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model, dummy, str(out_path),
        input_names=["input"], output_names=["probs"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Validate
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = export_model(dummy).numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}")


if __name__ == "__main__":
    main()

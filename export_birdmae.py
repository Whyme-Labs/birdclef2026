"""Export Bird-MAE-Base backbone to ONNX for V182 Kaggle inference.

Bird-MAE input: (B, 1, 512, 128) mel from kaldi fbank, mean=-7.2, std=4.43.
Output: (B, 768) — already mean-pooled per config (global_pool='mean').
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


class BirdMAEBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (B, 1, 512, 128)
        out = self.model(x)
        return out.last_hidden_state  # (B, 768)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="kaggle_model/birdmae_base.onnx")
    args = ap.parse_args()

    print("Loading Bird-MAE-Base...")
    model = AutoModel.from_pretrained("DBD-research-group/Bird-MAE-Base", trust_remote_code=True)
    model.eval()
    backbone = BirdMAEBackbone(model).eval()

    dummy = torch.zeros(1, 1, 512, 128)
    with torch.no_grad():
        out = backbone(dummy)
    print(f"PyTorch output: {out.shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {out_path}...")
    torch.onnx.export(
        backbone, dummy, str(out_path),
        input_names=["mel"], output_names=["features"],
        dynamic_axes={"mel": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported: {out_path} ({out_path.stat().st_size/1e6:.0f}MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    test = rng.uniform(-1, 1, (3, 1, 512, 128)).astype(np.float32)
    o = sess.run(None, {"mel": test})[0]
    with torch.no_grad():
        ot = backbone(torch.from_numpy(test)).numpy()
    diff = np.abs(o - ot).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}, out shape: {o.shape}")


if __name__ == "__main__":
    main()

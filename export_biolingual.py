"""Export BioLingual audio encoder to ONNX for V174 inference.

Inputs:
  - input_features: (B, 1, 1001, 64) — log mel spec
  - is_longer: (B, 1) bool — false for our 5s chunks
Output:
  - audio_features: (B, 512) — L2-normalized
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import ClapModel


class BioLingualAudioEncoder(nn.Module):
    """Wrapper that exposes audio encoder + projection + L2 norm."""
    def __init__(self, clap_model):
        super().__init__()
        self.audio_model = clap_model.audio_model
        self.audio_projection = clap_model.audio_projection

    def forward(self, input_features, is_longer):
        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=True,
        )
        pooled = audio_outputs[1]  # pooler_output
        feats = self.audio_projection(pooled)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="kaggle_model/biolingual_audio.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print("Loading BioLingual...")
    model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
    model.eval()

    encoder = BioLingualAudioEncoder(model)
    encoder.eval()

    # Test forward
    dummy_features = torch.zeros(1, 1, 1001, 64, dtype=torch.float32)
    dummy_is_longer = torch.zeros(1, 1, dtype=torch.bool)
    with torch.no_grad():
        out = encoder(dummy_features, dummy_is_longer)
    print(f"PyTorch output: {out.shape}, dtype={out.dtype}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {out_path} (opset={args.opset})...")
    torch.onnx.export(
        encoder,
        (dummy_features, dummy_is_longer),
        str(out_path),
        input_names=["input_features", "is_longer"],
        output_names=["audio_features"],
        dynamic_axes={
            "input_features": {0: "batch"},
            "is_longer": {0: "batch"},
            "audio_features": {0: "batch"},
        },
        opset_version=args.opset,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Exported: {out_path} ({size_mb:.0f}MB)")

    # Validate
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    test_feat = rng.uniform(-1, 1, (3, 1, 1001, 64)).astype(np.float32)
    test_long = np.zeros((3, 1), dtype=bool)
    out_onnx = sess.run(None, {
        "input_features": test_feat,
        "is_longer": test_long,
    })[0]
    with torch.no_grad():
        out_torch = encoder(torch.from_numpy(test_feat), torch.from_numpy(test_long)).numpy()
    diff = np.abs(out_onnx - out_torch).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}, output shape: {out_onnx.shape}")


if __name__ == "__main__":
    main()

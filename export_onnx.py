"""Export SEDModel (v1) to ONNX with sigmoid output (matching SED ONNX convention)."""
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from src.config import Config as ExperimentConfig
from src.models import SEDModel


class SEDModelForExport(nn.Module):
    """Wraps SEDModel to output sigmoid probabilities (clip-level only)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        clip_logits, _ = self.model(x)
        # Clamp to prevent fp32 sigmoid instability at extreme logits
        clip_logits = torch.clamp(clip_logits, min=-30.0, max=30.0)
        return torch.sigmoid(clip_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_mels", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    if "config" in ckpt:
        config_dict = ckpt["config"]
        config = ExperimentConfig(**{k: v for k, v in config_dict.items()
                                     if k in ExperimentConfig.__dataclass_fields__})
    else:
        config = ExperimentConfig()

    config.n_mels = args.n_mels
    print(f"Backbone: {config.backbone}, n_mels: {config.n_mels}")

    # Build model
    model = SEDModel(config)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    export_model = SEDModelForExport(model)
    export_model.eval()

    # Export — use realistic mel input (values in [0, 1]) not random noise
    dummy = torch.rand(1, 3, args.n_mels, 313)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model, dummy, str(output_path),
        input_names=["input"],
        output_names=["probs"],
        dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(output_path))
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = export_model(dummy).numpy()
    diff = np.abs(onnx_out - torch_out).max()
    print(f"Max diff: {diff:.2e} — {'OK' if diff < 1e-4 else 'WARNING'}")


if __name__ == "__main__":
    main()

"""Export V173 DANN-trained EffV2S to ONNX (species head only — domain head dropped at inference)."""
import argparse, json
import torch, torch.nn as nn
import numpy as np
from pathlib import Path
from src.config import Config as ExperimentConfig
from src.models import SEDModel
import sys; sys.path.insert(0, '.')
from finetune_dann import DANN


class DANNForExport(nn.Module):
    """Wraps DANN to output sigmoid species probs only (drops domain head)."""
    def __init__(self, dann):
        super().__init__()
        self.dann = dann

    def forward(self, x):
        # DANN.forward returns (clip_logits, frame_logits, domain_logits)
        clip_logits, _, _ = self.dann(x, lambda_=0.0)
        clip_logits = torch.clamp(clip_logits, min=-30.0, max=30.0)
        return torch.sigmoid(clip_logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/effv2s_dann/best_fold0.pt")
    ap.add_argument("--config_json", default="checkpoints/effv2s_dann/config.json")
    ap.add_argument("--output", default="kaggle_model/effv2s_dann.onnx")
    args = ap.parse_args()

    with open(args.config_json) as f:
        cfg_dict = json.load(f)
    config = ExperimentConfig(**{k: v for k, v in cfg_dict.items()
                                  if k in ExperimentConfig.__dataclass_fields__})
    config.num_classes = 234
    print(f"Backbone: {config.backbone}, n_mels: {config.n_mels}")

    model = DANN(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded ep{ckpt.get('epoch')} sl_auc={ckpt.get('sl_auc')}")

    export_model = DANNForExport(model)
    export_model.eval()

    dummy = torch.zeros(1, 3, config.n_mels, 320)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model, dummy, str(out_path),
        input_names=["mel"], output_names=["probs"],
        dynamic_axes={"mel": {0: "batch", 3: "time"},
                      "probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported: {out_path} ({out_path.stat().st_size/1e6:.0f}MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    test = rng.uniform(0, 1, (3, 3, config.n_mels, 320)).astype(np.float32)
    out_onnx = sess.run(None, {inp: test})[0]
    with torch.no_grad():
        out_torch = export_model(torch.from_numpy(test)).numpy()
    diff = np.abs(out_onnx - out_torch).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}, output shape: {out_onnx.shape}")


if __name__ == "__main__":
    main()

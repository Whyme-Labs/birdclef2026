"""Export Bird-MAE finetuned classifier to ONNX (V186 inference path).

Loads checkpoints/birdmae_ft/best_fold0.pt and exports a sigmoid output ONNX
suitable for Kaggle inference at weight 0.10 in the V137 ensemble.

Input: (B, 1, 512, 128) mel from kaldi fbank (32kHz, 128 mels, 512 frames).
Output: (B, 234) sigmoid probabilities.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


class BirdMAEClassifier(nn.Module):
    def __init__(self, num_classes=234, dropout=0.0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "DBD-research-group/Bird-MAE-Base", trust_remote_code=True
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        feat = out.last_hidden_state
        feat = self.dropout(feat)
        return self.head(feat)


class BirdMAEForExport(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x):
        logits = self.classifier(x)
        logits = torch.clamp(logits, min=-30.0, max=30.0)
        return torch.sigmoid(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/birdmae_ft/best_fold0.pt")
    ap.add_argument("--output", default="kaggle_model/birdmae_ft.onnx")
    args = ap.parse_args()

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  epoch={ckpt.get('epoch')}, val_auc={ckpt.get('val_auc'):.4f}")

    model = BirdMAEClassifier(num_classes=234, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    export_model = BirdMAEForExport(model).eval()

    dummy = torch.zeros(1, 1, 512, 128, dtype=torch.float32)
    with torch.no_grad():
        out = export_model(dummy)
    print(f"PyTorch output: {out.shape}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {out_path}...")
    torch.onnx.export(
        export_model, dummy, str(out_path),
        input_names=["mel"], output_names=["probs"],
        dynamic_axes={"mel": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported: {out_path} ({out_path.stat().st_size/1e6:.0f}MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    test = rng.uniform(-1, 1, (3, 1, 512, 128)).astype(np.float32)
    o = sess.run(None, {"mel": test})[0]
    with torch.no_grad():
        ot = export_model(torch.from_numpy(test)).numpy()
    diff = np.abs(o - ot).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}, out shape: {o.shape}")


if __name__ == "__main__":
    main()

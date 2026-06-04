"""Export WavJEPA-Nat to ONNX for Kaggle CPU inference.

WavJEPA via transformers.PyTorch on Kaggle CPU is ~4 s/chunk on dummy test,
projecting to ~9h for the full test set — breaks budget when stacked with V137.
ONNX export should give 3-4x speedup, putting it at ~2-3h, comfortably safe.

Strategy:
  Load model via the same manual-import workaround used in V192 kernel,
  export via torch.onnx.export at fixed shape (B=1, 2ch, 80000 samples).
  Validate fp32 ONNX output matches PyTorch.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_wavjepa(snapshot_dir: Path):
    pkg = "_wj_pkg_export"
    sys.modules[pkg] = type(sys)(pkg)
    sys.modules[pkg].__path__ = [str(snapshot_dir)]

    def _load(name, fname):
        spec = importlib.util.spec_from_file_location(
            f"{pkg}.{name}", str(snapshot_dir / fname))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg}.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    for sub in ("types", "utils", "pos_embed", "audio_extractor",
                "configuration_wavjepa_nat", "model", "modeling_wavjepa_nat"):
        _load(sub, f"{sub}.py")

    cfg_class = sys.modules[f"{pkg}.configuration_wavjepa_nat"].WavJEPANatConfig
    model_class = sys.modules[f"{pkg}.modeling_wavjepa_nat"].WavJEPANatModel
    with open(snapshot_dir / "config.json") as f:
        cfg = cfg_class(**json.load(f))
    m = model_class(cfg)

    from safetensors.torch import load_file
    state = load_file(str(snapshot_dir / "model.safetensors"))
    missing, unexpected = m.load_state_dict(state, strict=False)
    print(f"  load: missing={len(missing)}, unexpected={len(unexpected)}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot",
                    default="/home/soh/.cache/huggingface/hub/models--labhamlet--wavjepa-nat-base/snapshots/15d95ff67fa98117b17e83a1653bbca97877ff6f")
    ap.add_argument("--out_fp32", default="kaggle_model/wavjepa_fp32.onnx")
    ap.add_argument("--out_int8", default="kaggle_model/wavjepa_int8.onnx")
    args = ap.parse_args()

    snapshot = Path(args.snapshot)
    print(f"Loading WavJEPA from {snapshot}")
    t0 = time.time()
    m = load_wavjepa(snapshot)
    m.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Force the SLOW path of nn.TransformerEncoderLayer.forward by monkey-patching:
    # the fused fast-path calls aten::_transformer_encoder_layer_fwd which has
    # no ONNX symbolic. We replace forward() with the manual eager-mode body.
    def _slow_layer_forward(self, src, src_mask=None, src_key_padding_mask=None,
                              is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask,
                                    is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask,
                                                is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    torch.nn.TransformerEncoderLayer.forward = _slow_layer_forward
    # Also disable nested tensor at encoder level
    for sub in m.modules():
        if isinstance(sub, torch.nn.TransformerEncoder):
            sub.enable_nested_tensor = False
            if hasattr(sub, "use_nested_tensor"):
                sub.use_nested_tensor = False

    # Wrap so output is mean-pooled (B, 768) — matches probe path
    class WJWrapper(torch.nn.Module):
        def __init__(self, mdl): super().__init__(); self.m = mdl
        def forward(self, wav):
            # wav: (B, 2, T)
            out = self.m(wav)  # tuple, [0] is (B, 2ch, T_tokens, 768)
            feats = out[0]
            return feats.mean(dim=(1, 2))  # (B, 768)

    wrapper = WJWrapper(m).eval()

    # Dummy input: (B, 2, 80000) for 5s @ 16kHz
    B = 1
    T = 80000
    dummy = torch.randn(B, 2, T)
    print(f"\nDummy input shape: {dummy.shape}")
    print("Testing PyTorch forward...")
    t0 = time.time()
    with torch.no_grad():
        feat_pt = wrapper(dummy)
    print(f"  PT forward: {time.time() - t0:.2f}s; output {feat_pt.shape}, "
          f"mean={feat_pt.mean().item():.4f}, std={feat_pt.std().item():.4f}")

    Path(args.out_fp32).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting to {args.out_fp32}...")
    t0 = time.time()
    try:
        torch.onnx.export(
            wrapper, dummy, args.out_fp32,
            input_names=["wav"], output_names=["feat"],
            dynamic_axes={"wav": {0: "batch"}, "feat": {0: "batch"}},
            opset_version=17, do_constant_folding=True,
        )
        sz = Path(args.out_fp32).stat().st_size / 1024 / 1024
        print(f"  exported in {time.time() - t0:.0f}s; size {sz:.0f} MB")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    print("\nValidating ONNX vs PyTorch...")
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out_fp32, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    feat_onnx = sess.run(None, {inp_name: dummy.numpy()})[0]
    diff = np.abs(feat_pt.numpy() - feat_onnx).max()
    rel = diff / (np.abs(feat_pt.numpy()).max() + 1e-9)
    print(f"  max abs diff: {diff:.6f}; relative: {rel:.6f}")

    # Speed
    print("\nSpeed benchmark (10 forward passes, batch=1)...")
    for _ in range(3):
        sess.run(None, {inp_name: dummy.numpy()})
    t0 = time.time()
    for _ in range(10):
        sess.run(None, {inp_name: dummy.numpy()})
    onnx_time = (time.time() - t0) / 10
    n_chunks = 8400
    print(f"  ONNX FP32: {onnx_time*1000:.0f} ms/chunk; "
          f"est test set: {onnx_time*n_chunks/60:.0f} min")


if __name__ == "__main__":
    main()

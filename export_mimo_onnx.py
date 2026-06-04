"""Export MiMo-Audio-Tokenizer encoder (without quantizer) to ONNX.

Goal: get MiMo's pre-RVQ continuous features exportable for Kaggle CPU inference.
Without ONNX + int8, MiMo's 640M-param transformer takes ~4.6h CPU on test set —
breaks the 9h Kaggle budget. ONNX int8 quantization typically gives 3-4x speedup
which would put it at ~70 min.

Strategy:
  1. Build a wrapper module that takes (B, T_mel, n_mels) → (B, d_model) mean-pooled
     encoder output. Fixed batch size, fixed T_mel = 501 (5s @ 24kHz mel).
  2. Use the flash_attn → SDPA shim so attention is ONNX-friendly.
  3. Export at batch=1, then validate output matches PyTorch on a few inputs.

If export succeeds AND outputs match: kick off int8 dynamic quantization, validate
again, save as kaggle_model/mimo_encoder_int8.onnx.
"""
import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# flash_attn → SDPA shim (same as build_mimo_probe.py)
def _flash_attn_varlen_func_sdpa(q, k, v, cu_q, cu_k, max_q, max_k,
                                  causal=False, window_size=(-1, -1), **kw):
    bsz = cu_q.shape[0] - 1
    H, D = q.shape[1], q.shape[2]
    seq_lens = (cu_q[1:] - cu_q[:-1]).tolist()
    if all(s == seq_lens[0] for s in seq_lens):
        T = seq_lens[0]
        q_b = q.view(bsz, T, H, D).transpose(1, 2).contiguous()
        k_b = k.view(bsz, T, H, D).transpose(1, 2).contiguous()
        v_b = v.view(bsz, T, H, D).transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=causal)
        return out.transpose(1, 2).reshape(bsz * T, H, D)
    out = torch.zeros_like(q)
    for i in range(bsz):
        s = int(cu_q[i].item())
        e = int(cu_q[i + 1].item())
        if e == s:
            continue
        q_i = q[s:e].transpose(0, 1).unsqueeze(0)
        k_i = k[s:e].transpose(0, 1).unsqueeze(0)
        v_i = v[s:e].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=causal)
        out[s:e] = o.squeeze(0).transpose(0, 1)
    return out


_fake = types.ModuleType("flash_attn")
_fake.flash_attn_varlen_func = _flash_attn_varlen_func_sdpa
sys.modules["flash_attn"] = _fake

sys.path.insert(0, "/tmp/MiMo-Audio/src")
from mimo_audio_tokenizer import MiMoAudioTokenizer, MiMoAudioTokenizerConfig  # noqa


class MiMoEncoderWrapper(nn.Module):
    """Wraps MiMo encoder to (mel, length) → mean-pooled encoder output (B, d_model).

    Uses the model's encode() with use_quantizer=False. Mean-pools over output_length.
    """

    def __init__(self, mimo_model: MiMoAudioTokenizer):
        super().__init__()
        self.mimo = mimo_model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, T_mel, n_mels)
        B, T_mel, n_mels = mel.shape
        # MiMo encode() expects packed mels: (sum_T, n_mels) + lens (B,)
        packed = mel.reshape(B * T_mel, n_mels)
        lens = torch.full((B,), T_mel, dtype=torch.long, device=mel.device)
        h, _, output_length, _ = self.mimo.encode(packed, lens, use_quantizer=False)
        # h: (B, T_out, d_model). output_length is per-batch (B,) but with fixed
        # T_mel and same length, all are equal: T_out_full.
        T_out = h.shape[1]
        # Mean-pool over time. With same lengths across batch, simple mean works.
        feat = h.mean(dim=1)  # (B, d_model)
        return feat


def load_mimo(model_id="XiaomiMiMo/MiMo-Audio-Tokenizer"):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    p = snapshot_download(model_id)
    cfg = MiMoAudioTokenizerConfig.from_pretrained(p)
    m = MiMoAudioTokenizer(cfg)
    sd = load_file(f"{p}/model.safetensors")
    m.load_state_dict(sd, strict=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_fp32", default="kaggle_model/mimo_encoder_fp32.onnx")
    ap.add_argument("--out_int8", default="kaggle_model/mimo_encoder_int8.onnx")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--t_mel", type=int, default=501)  # 5s @ 24kHz mel hop=240
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--device", default="cpu")  # CPU export is more robust
    args = ap.parse_args()

    Path(args.out_fp32).parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading MiMo on {args.device}...")
    t0 = time.time()
    m = load_mimo()
    m.eval().to(args.device)
    print(f"  loaded in {time.time() - t0:.1f}s; d_model={m.config.d_model}")

    wrapper = MiMoEncoderWrapper(m).to(args.device).eval()

    # Dummy input
    dummy = torch.randn(args.batch_size, args.t_mel, args.n_mels, device=args.device)
    print(f"\nDummy input shape: {dummy.shape}")
    print("Testing PyTorch forward...")
    with torch.no_grad():
        feat_pt = wrapper(dummy)
    print(f"  PT output: {feat_pt.shape}, mean={feat_pt.mean().item():.4f}, "
          f"std={feat_pt.std().item():.4f}")

    # ONNX export
    print(f"\nExporting to {args.out_fp32}...")
    t0 = time.time()
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            args.out_fp32,
            input_names=["mel"],
            output_names=["feat"],
            dynamic_axes={"mel": {0: "batch"}, "feat": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  exported in {time.time() - t0:.0f}s; file size {Path(args.out_fp32).stat().st_size / 1024 / 1024:.0f} MB")
    except Exception as e:
        print(f"ONNX export FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return

    # Validate ONNX matches PyTorch
    print("\nValidating ONNX vs PyTorch...")
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out_fp32, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    feat_onnx = sess.run(None, {inp_name: dummy.cpu().numpy()})[0]
    diff = np.abs(feat_pt.cpu().numpy() - feat_onnx).max()
    rel = diff / (np.abs(feat_pt.cpu().numpy()).max() + 1e-9)
    print(f"  max abs diff: {diff:.6f}; relative: {rel:.6f}")
    if rel > 1e-2:
        print(f"  WARNING: large divergence; ONNX export may have issues")

    # Try a different random input
    dummy2 = torch.randn(args.batch_size, args.t_mel, args.n_mels)
    with torch.no_grad():
        feat_pt2 = wrapper(dummy2)
    feat_onnx2 = sess.run(None, {inp_name: dummy2.numpy()})[0]
    diff2 = np.abs(feat_pt2.cpu().numpy() - feat_onnx2).max()
    print(f"  diff on input 2: {diff2:.6f}")

    # INT8 quantization
    print(f"\nQuantizing to INT8 → {args.out_int8}...")
    t0 = time.time()
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            args.out_fp32,
            args.out_int8,
            weight_type=QuantType.QUInt8,
        )
        sz_int8 = Path(args.out_int8).stat().st_size / 1024 / 1024
        sz_fp32 = Path(args.out_fp32).stat().st_size / 1024 / 1024
        print(f"  quantized in {time.time() - t0:.0f}s")
        print(f"  fp32 size: {sz_fp32:.0f} MB → int8 size: {sz_int8:.0f} MB "
              f"({sz_fp32/sz_int8:.1f}x)")
    except Exception as e:
        print(f"INT8 quantization FAILED: {e}")
        return

    # Validate INT8
    print("\nValidating INT8 vs FP32 ONNX...")
    sess8 = ort.InferenceSession(args.out_int8, providers=["CPUExecutionProvider"])
    inp_name8 = sess8.get_inputs()[0].name
    feat8 = sess8.run(None, {inp_name8: dummy.cpu().numpy()})[0]
    diff8 = np.abs(feat_pt.cpu().numpy() - feat8).max()
    rel8 = diff8 / (np.abs(feat_pt.cpu().numpy()).max() + 1e-9)
    print(f"  fp32 vs int8 max diff: {diff8:.6f}; relative: {rel8:.6f}")

    # Speed benchmark
    print("\nSpeed benchmark (10 forward passes, batch=1, t_mel=501)...")
    n_warm = 3
    n_test = 10
    for _ in range(n_warm):
        sess.run(None, {inp_name: dummy.cpu().numpy()})
    t0 = time.time()
    for _ in range(n_test):
        sess.run(None, {inp_name: dummy.cpu().numpy()})
    fp32_time = (time.time() - t0) / n_test
    for _ in range(n_warm):
        sess8.run(None, {inp_name8: dummy.cpu().numpy()})
    t0 = time.time()
    for _ in range(n_test):
        sess8.run(None, {inp_name8: dummy.cpu().numpy()})
    int8_time = (time.time() - t0) / n_test
    print(f"  FP32: {fp32_time*1000:.0f} ms/chunk")
    print(f"  INT8: {int8_time*1000:.0f} ms/chunk  ({fp32_time/int8_time:.2f}x speedup)")
    n_chunks = 8400  # rough: 700 test files × 12 chunks
    print(f"  Estimated test-set time at INT8: {int8_time * n_chunks / 60:.0f} min")
    print(f"  Estimated test-set time at FP32: {fp32_time * n_chunks / 60:.0f} min")


if __name__ == "__main__":
    main()

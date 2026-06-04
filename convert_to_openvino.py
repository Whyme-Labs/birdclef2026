"""Convert ONNX models to OpenVINO IR fp16 for Kaggle CPU inference speedup.

OpenVINO fp16 typically gives 2-3× speedup over ONNX fp32 on Intel CPUs (Kaggle).
2nd place 2025 uses this exact pipeline. Critical for unblocking V184/V187/V191/
V193/V196/V200 timeout pattern.
"""
import argparse
import time
from pathlib import Path
import numpy as np


def convert(onnx_path: Path, ir_dir: Path, fp16: bool = True):
    import openvino as ov
    print(f"\n→ {onnx_path.name}")
    t0 = time.time()
    model = ov.Core().read_model(str(onnx_path))
    if fp16:
        from openvino import save_model
        ir_path = ir_dir / (onnx_path.stem + "_fp16.xml")
        save_model(model, str(ir_path), compress_to_fp16=True)
    else:
        ir_path = ir_dir / (onnx_path.stem + "_fp32.xml")
        ov.save_model(model, str(ir_path), compress_to_fp16=False)
    print(f"  saved {ir_path.name} ({(ir_path.stat().st_size + ir_path.with_suffix('.bin').stat().st_size) / 1e6:.1f} MB)")
    print(f"  conversion time: {time.time() - t0:.1f}s")
    return ir_path


def benchmark(onnx_path: Path, ir_path: Path, dummy_input: np.ndarray, n_iter: int = 10):
    import onnxruntime as ort
    import openvino as ov

    print(f"\nBenchmark {onnx_path.stem}:")

    # ONNX FP32
    sopts = ort.SessionOptions()
    sopts.inter_op_num_threads = 4
    sopts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sopts, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    for _ in range(3):
        sess.run(None, {inp_name: dummy_input})
    t0 = time.time()
    for _ in range(n_iter):
        sess.run(None, {inp_name: dummy_input})
    onnx_time = (time.time() - t0) / n_iter

    # OpenVINO FP16
    core = ov.Core()
    compiled = core.compile_model(str(ir_path), "CPU", config={
        "INFERENCE_NUM_THREADS": 4,
        "PERFORMANCE_HINT": "LATENCY",
    })
    for _ in range(3):
        compiled(dummy_input)
    t0 = time.time()
    for _ in range(n_iter):
        compiled(dummy_input)
    ov_time = (time.time() - t0) / n_iter

    print(f"  ONNX FP32: {onnx_time*1000:.1f} ms/inference")
    print(f"  OV   FP16: {ov_time*1000:.1f} ms/inference  ({onnx_time/ov_time:.2f}x speedup)")
    return onnx_time, ov_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="kaggle_model/openvino_ir")
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    ir_dir = Path(args.out_dir)
    ir_dir.mkdir(parents=True, exist_ok=True)

    # Models we want to convert (heaviest = biggest LB-payoff if we can speed them up)
    targets = [
        # (onnx_path, dummy_input_shape, dtype)
        ("kaggle_model/eca_nfnet_l0.onnx", (1, 3, 224, 313), np.float32),
        ("kaggle_model/effv2s_r2_fold1.onnx", (1, 3, 224, 313), np.float32),
        ("kaggle_model/effv2s_r2_fold0.onnx", (1, 3, 224, 313), np.float32),
        ("kaggle_model/effv2s_v194_iter1.onnx", (1, 3, 224, 313), np.float32),
        ("kaggle_model/effv2s_v199_soup.onnx", (1, 3, 224, 313), np.float32),
        ("kaggle_model/audiomae_emb_v160.onnx", (1, 1024, 128), np.float32),
        ("kaggle_model/birdmae_base.onnx", (1, 1024, 128), np.float32),
        ("kaggle_model/biolingual_audio.onnx", (1, 1, 1001, 64), np.float32),
    ]

    results = []
    for onnx_p, shape, dtype in targets:
        op = Path(onnx_p)
        if not op.exists():
            print(f"SKIP missing: {onnx_p}")
            continue
        try:
            ir_p = convert(op, ir_dir, fp16=True)
            dummy = np.random.rand(*shape).astype(dtype)
            onnx_t, ov_t = benchmark(op, ir_p, dummy)
            results.append((op.name, onnx_t, ov_t, onnx_t / ov_t))
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")

    if results:
        print("\n=== SUMMARY ===")
        print(f"{'Model':<40} {'ONNX(ms)':>10} {'OV(ms)':>10} {'Speedup':>10}")
        for name, ot, vt, sp in results:
            print(f"{name:<40} {ot*1000:>10.1f} {vt*1000:>10.1f} {sp:>10.2f}x")


if __name__ == "__main__":
    main()

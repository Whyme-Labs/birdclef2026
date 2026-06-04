"""V212 distillation: train fresh EffV2-S on blended (V137 pseudo + Whisper-FT pseudo).

Path 1 of Whisper distillation: Whisper-FT is teacher, EffV2-S is Kaggle-deployable
student. Same architecture as V137's V2S R2, so calibration recovery via souping
with R2_fold1 should work (V199 pattern).

Distinct from V194 iter 2:
  V194_iter2 used (V137 pseudo + V194_iter1 pseudo) — same encoder family, val_auc
  compounded but LB regressed (-0.005).
  V212 uses (V137 pseudo + Whisper-FT pseudo) — Whisper brings genuinely orthogonal
  signal from speech/music/ambient pretraining (680k hrs). Should transfer to LB
  where V194 didn't.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper_pseudo",
                    default="pseudo_labels_whisper_ft/raw_predictions.csv")
    ap.add_argument("--v137_pseudo",
                    default="pseudo_labels_v137/raw_predictions.csv")
    ap.add_argument("--w_v137", type=float, default=0.5,
                    help="V137 pseudo weight (Whisper-FT gets 1-w)")
    ap.add_argument("--blend_dir", default="pseudo_labels_v212_blend")
    ap.add_argument("--checkpoint", default="checkpoints/effv2s_r2_fold0/best_fold0.pt",
                    help="Warm-start backbone")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--experiment_name", default="v212_whisper_distill")
    args = ap.parse_args()

    if not Path(args.whisper_pseudo).exists():
        print(f"ERROR: {args.whisper_pseudo} not found.")
        print("Run build_whisper_ft_pseudo.py first.")
        sys.exit(1)

    # Step 1: blend pseudo
    blend_csv = Path(args.blend_dir) / "raw_predictions.csv"
    if not blend_csv.exists():
        print(f"\n=== Step 1: Blend pseudo (w_v137={args.w_v137}) ===")
        subprocess.run([
            sys.executable, "blend_v137_v194_pseudo.py",
            "--v137", args.v137_pseudo,
            "--v194", args.whisper_pseudo,  # script is generic; just point at Whisper-FT
            "--out", str(blend_csv),
            "--w_v137", str(args.w_v137),
        ], check=True)
    else:
        print(f"Reusing existing blend: {blend_csv}")

    # Step 2: train V212 student
    print(f"\n=== Step 2: Train V212 student ===")
    subprocess.run([
        sys.executable, "finetune_v2.py",
        "--checkpoint", args.checkpoint,
        "--pseudo_label_dir", args.blend_dir,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--experiment_name", args.experiment_name,
        "--loss_type", "combined_asl_auc",
    ], check=True)

    # Step 3: export to ONNX
    print(f"\n=== Step 3: Export to ONNX ===")
    ckpt_path = f"checkpoints/{args.experiment_name}/best_fold0.pt"
    onnx_path = "kaggle_model/effv2s_v212_distill.onnx"
    subprocess.run([
        sys.executable, "export_v194_onnx.py",
        "--checkpoint", ckpt_path,
        "--output", onnx_path,
    ], check=True)

    # Step 4: soup with R2_fold1 for calibration recovery
    print(f"\n=== Step 4: Soup with R2_fold1 ===")
    soup_pt = "checkpoints/v212_soup/best_fold0.pt"
    subprocess.run([
        sys.executable, "soup_v194iter2_r2.py",
        "--ckpts", ckpt_path, "checkpoints/effv2s_r2_fold1/best_fold1.pt",
        "--weights", "0.5", "0.5",
        "--out", soup_pt,
    ], check=True)
    soup_onnx = "kaggle_model/effv2s_v212_soup.onnx"
    subprocess.run([
        sys.executable, "export_v194_onnx.py",
        "--checkpoint", soup_pt,
        "--output", soup_onnx,
    ], check=True)
    print(f"\nDone. Use {soup_onnx} as V137's V2S replacement in V213 kernel.")


if __name__ == "__main__":
    main()

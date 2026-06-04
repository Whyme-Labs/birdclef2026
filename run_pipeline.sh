#!/bin/bash
# Full training pipeline for BirdCLEF 2026 — Model A + B + ONNX export
# Run from /home/soh/birdclef-2026
#
# Usage: nohup bash run_pipeline.sh > /tmp/pipeline_log.txt 2>&1 &

set -e
PYTHON=/home/soh/miniconda3/envs/birdclef/bin/python
export PYTHONUNBUFFERED=1
export TORCH_DYNAMO_DISABLE=1

cd /home/soh/birdclef-2026

echo "=== Pipeline started at $(date) ==="

# ── Step 1: Check if Model A training is still running ────────────────────────
if pgrep -f "train.py.*b3_softauc_r2" > /dev/null 2>&1; then
    echo "Model A (b3_softauc_r2) is still training. Waiting..."
    while pgrep -f "train.py.*b3_softauc_r2" > /dev/null 2>&1; do
        sleep 300
        grep -E "Epoch" /tmp/train_b3_softauc_r2_log.txt | tail -3
    done
    echo "Model A training finished at $(date)"
fi

# Verify Model A checkpoint exists
if [ ! -f checkpoints/b3_softauc_r2/best_fold0.pt ]; then
    echo "ERROR: Model A checkpoint not found! Check training logs."
    exit 1
fi
echo "Model A checkpoint found: $(ls -la checkpoints/b3_softauc_r2/best_fold0.pt)"

# ── Step 2: Export Model A to ONNX ───────────────────────────────────────────
echo ""
echo "=== Exporting Model A to ONNX at $(date) ==="
$PYTHON export_onnx.py \
    --checkpoint checkpoints/b3_softauc_r2/best_fold0.pt \
    --output kaggle_model/b3_softauc_r2.onnx
echo "Model A export done at $(date)"

# ── Step 3: Train Model B (EfficientNetV2-S fold 1) ─────────────────────────
echo ""
echo "=== Training Model B (effv2s_r2_fold1) at $(date) ==="
$PYTHON train.py \
    --backbone tf_efficientnetv2_s.in21k \
    --batch_size 32 \
    --grad_accum_steps 1 \
    --lr 5e-4 \
    --backbone_lr_mult 0.1 \
    --epochs 25 \
    --loss_type combined_asl_auc \
    --pseudo_label_dir pseudo_labels_r2 \
    --pseudo_label_weight 0.5 \
    --drop_rate 0.3 \
    --mixup_alpha 0.4 \
    --fold 1 \
    --experiment_name effv2s_r2_fold1 \
    --warmup_epochs 3
echo "Model B training done at $(date)"

# Verify Model B checkpoint
if [ ! -f checkpoints/effv2s_r2_fold1/best_fold1.pt ]; then
    echo "ERROR: Model B checkpoint not found!"
    exit 1
fi

# ── Step 4: Export Model B to ONNX ───────────────────────────────────────────
echo ""
echo "=== Exporting Model B to ONNX at $(date) ==="
$PYTHON export_onnx.py \
    --checkpoint checkpoints/effv2s_r2_fold1/best_fold1.pt \
    --output kaggle_model/effv2s_r2_fold1.onnx
echo "Model B export done at $(date)"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Pipeline complete at $(date) ==="
echo "ONNX models:"
ls -la kaggle_model/b3_softauc_r2.onnx kaggle_model/effv2s_r2_fold1.onnx
echo ""
echo "All models in kaggle_model/:"
ls -la kaggle_model/*.onnx

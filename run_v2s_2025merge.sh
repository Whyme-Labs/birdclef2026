#!/bin/bash
# V2S 2025-merged pipeline:
#   1. Wait for the currently-running EffNetV2-M training (PID 449475) to finish.
#   2. Train tf_efficientnetv2_s on 2026 + 2025 merged data with R5 pseudo-labels.
#   3. Export best checkpoint to ONNX.
#   4. Upload to Kaggle dataset.
set -e

PYTHON=/home/soh/miniconda3/envs/birdclef/bin/python
KAGGLE=/home/soh/miniconda3/bin/kaggle
EXP=v2s_2025merged
PID_TO_WAIT=449475

export PYTHONUNBUFFERED=1
cd /home/soh/birdclef-2026

echo "=== V2S 2025-merged pipeline started at $(date) ==="
echo "Waiting for PID ${PID_TO_WAIT} (current EffNetV2-M training) to finish..."

# Poll every 60s until the process is gone.
while kill -0 ${PID_TO_WAIT} 2>/dev/null; do
    sleep 60
done

echo "PID ${PID_TO_WAIT} finished at $(date). Starting V2S training."

# Train V2S on merged CSV + R5 pseudo labels.
echo "=== Training ${EXP} at $(date) ==="
${PYTHON} train.py \
    --backbone tf_efficientnetv2_s.in21k \
    --batch_size 32 \
    --grad_accum_steps 1 \
    --lr 5e-4 \
    --backbone_lr_mult 0.1 \
    --epochs 20 \
    --loss_type combined_asl_auc \
    --pseudo_label_dir pseudo_labels_r5 \
    --pseudo_label_weight 0.5 \
    --drop_rate 0.3 \
    --mixup_alpha 0.4 \
    --fold 0 \
    --experiment_name ${EXP} \
    --warmup_epochs 3 \
    --train_csv train_merged.csv
echo "=== Training done at $(date) ==="

# Export to ONNX.
echo "=== Exporting ${EXP} to ONNX ==="
${PYTHON} export_onnx.py \
    --checkpoint checkpoints/${EXP}/best_fold0.pt \
    --output kaggle_model/${EXP}.onnx
echo "=== Export done at $(date) ==="

# Upload Kaggle dataset version.
echo "=== Uploading to Kaggle ==="
cd kaggle_model
${KAGGLE} datasets version -p . -m "Add ${EXP}.onnx (V2S trained on 2026 + 2025 merged + R5 pseudo)"

echo "=== V2S 2025-merged pipeline complete at $(date) ==="

#!/bin/bash
# R4 pipeline: wait for pseudo-labels, then train, then export
set -e
PYTHON=/home/soh/miniconda3/envs/birdclef/bin/python
export PYTHONUNBUFFERED=1

cd /home/soh/birdclef-2026

echo "=== R4 Pipeline started at $(date) ==="

# Wait for R4 pseudo-labels
echo "Waiting for R4 pseudo-labels..."
while [ ! -f pseudo_labels_r4/raw_predictions.csv ]; do
    sleep 60
done
echo "R4 pseudo-labels ready at $(date)"

# Train R4 V2S
echo "=== Training R4 V2S at $(date) ==="
$PYTHON train.py \
    --backbone tf_efficientnetv2_s.in21k \
    --batch_size 32 \
    --grad_accum_steps 1 \
    --lr 5e-4 \
    --backbone_lr_mult 0.1 \
    --epochs 20 \
    --loss_type combined_asl_auc \
    --pseudo_label_dir pseudo_labels_r4 \
    --pseudo_label_weight 0.5 \
    --drop_rate 0.3 \
    --mixup_alpha 0.4 \
    --fold 0 \
    --experiment_name effv2s_r4_fold0 \
    --warmup_epochs 2
echo "R4 training done at $(date)"

# Export to ONNX
echo "=== Exporting R4 to ONNX ==="
$PYTHON export_onnx.py \
    --checkpoint checkpoints/effv2s_r4_fold0/best_fold0.pt \
    --output kaggle_model/effv2s_r4_fold0.onnx
echo "R4 export done at $(date)"

# Upload
echo "=== Uploading to Kaggle ==="
cd kaggle_model
/home/soh/miniconda3/bin/kaggle datasets version -p . -m "Add effv2s_r4_fold0.onnx"
echo "=== R4 Pipeline complete at $(date) ==="

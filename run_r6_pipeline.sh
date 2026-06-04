#!/bin/bash
# R6 pipeline: wait for pseudo-labels, train V2S distillation student, export, upload
set -e
PYTHON=/home/soh/miniconda3/envs/birdclef/bin/python
export PYTHONUNBUFFERED=1
cd /home/soh/birdclef-2026

echo "=== R6 Pipeline started at $(date) ==="

echo "Waiting for R6 pseudo-labels..."
while [ ! -f pseudo_labels_r6/raw_predictions.csv ]; do
    sleep 60
done
echo "R6 pseudo-labels ready at $(date)"

echo "Wait 30s to ensure file is fully written..."
sleep 30

echo "=== Training V2S R6 student at $(date) ==="
$PYTHON train.py \
    --backbone tf_efficientnetv2_s.in21k \
    --batch_size 32 \
    --grad_accum_steps 1 \
    --lr 5e-4 \
    --backbone_lr_mult 0.1 \
    --epochs 20 \
    --loss_type combined_asl_auc \
    --pseudo_label_dir pseudo_labels_r6 \
    --pseudo_label_weight 0.5 \
    --drop_rate 0.3 \
    --mixup_alpha 0.4 \
    --fold 0 \
    --experiment_name v2s_r6_student \
    --warmup_epochs 3
echo "V2S R6 training done at $(date)"

if [ ! -f checkpoints/v2s_r6_student/best_fold0.pt ]; then
    echo "ERROR: V2S R6 checkpoint missing!"
    exit 1
fi

echo "=== Exporting V2S R6 to ONNX ==="
$PYTHON export_onnx.py \
    --checkpoint checkpoints/v2s_r6_student/best_fold0.pt \
    --output kaggle_model/v2s_r6_student.onnx

echo "=== Uploading to Kaggle ==="
cd kaggle_model
/home/soh/miniconda3/bin/kaggle datasets version -p . -m "Add v2s_r6_student.onnx (V137-ensemble distillation)"
echo "=== R6 Pipeline complete at $(date) ==="

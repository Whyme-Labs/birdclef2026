#!/bin/bash
# Wait for R5 pseudo-labeling to complete, then launch DaViT-tiny training.
set -u

cd /home/soh/birdclef-2026

PSEUDO_OUT=/home/soh/birdclef-2026/pseudo_labels_r5/raw_predictions.csv
PSEUDO_LOG=/tmp/pseudo_label_r5_log.txt
TRAIN_LOG=/tmp/train_swin_log.txt

echo "[$(date)] Waiting for R5 pseudo-labels at $PSEUDO_OUT ..."
while [ ! -f "$PSEUDO_OUT" ]; do
    sleep 60
done

# One more sleep to make sure file write completes
sleep 30

echo "[$(date)] R5 pseudo-labels ready. Launching DaViT-tiny training." | tee -a "$TRAIN_LOG"

PYTHONUNBUFFERED=1 /home/soh/miniconda3/envs/birdclef/bin/python train.py \
    --backbone davit_tiny \
    --batch_size 24 \
    --grad_accum_steps 2 \
    --lr 3e-4 \
    --backbone_lr_mult 0.1 \
    --epochs 20 \
    --loss_type combined_asl_auc \
    --pseudo_label_dir pseudo_labels_r5 \
    --pseudo_label_weight 0.5 \
    --drop_rate 0.3 \
    --mixup_alpha 0.4 \
    --fold 0 \
    --experiment_name davit_tiny_r5 \
    --warmup_epochs 3 >> "$TRAIN_LOG" 2>&1

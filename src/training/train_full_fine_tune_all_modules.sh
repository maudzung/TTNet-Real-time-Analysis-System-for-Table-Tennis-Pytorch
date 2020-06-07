#!/bin/bash

python main.py \
  --saved_fn 'ttnet_full_finetune' \
  --arch 'ttnet' \
  --no-val \
  --batch_size 32 \
  --num_workers 8 \
  --sigma 1. \
  --thresh_ball_pos_mask 0.01 \
  --start_epoch 1 \
  --num_epochs 9 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 5 \
  --lr_factor 0.2 \
  --world-size 1 \
  --rank 0 \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --weight_decay 0. \
  --global_weight 1. \
  --seg_weight 1. \
  --event_weight 1. \
  --local_weight 1. \
  --pretrained_path ../../checkpoints/ttnet_full_freeze_global_freeze_seg/ttnet_full_freeze_global_freeze_seg_epoch_21.pth

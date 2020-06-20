#!/bin/bash

python main.py \
  --saved_fn 'ttnet_1st_phase' \
  --arch 'ttnet' \
  --no-val \
  --batch_size 32 \
  --num_workers 4 \
  --sigma 1. \
  --thresh_ball_pos_mask 0.05 \
  --start_epoch 1 \
  --num_epochs 21 \
  --lr 0.001 \
  --lr_type 'step_lr' \
  --lr_step_size 5 \
  --lr_factor 0.2 \
  --world-size 1 \
  --rank 0 \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --weight_decay 0. \
  --global_weight 5. \
  --seg_weight 1. \
  --no_local \
  --no_event \
  --smooth-labelling \
  --print_freq 50
#!/bin/bash

python main.py \
  --working_dir '../' \
  --saved_fn 'ttnet_3rd_phase' \
  --arch 'ttnet' \
  --no-val \
  --batch_size 32 \
  --num_workers 4 \
  --sigma 1. \
  --thresh_ball_pos_mask 0.05 \
  --start_epoch 1 \
  --num_epochs 30 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
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
  --pretrained_path ../checkpoints/ttnet_2nd_phase/ttnet_2nd_phase_epoch_30.pth \
  --smooth-labelling \
  --print_freq 50

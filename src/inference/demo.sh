#!/bin/bash

python demo.py \
  --saved_fn 'demo' \
  --gpu_idx 0 \
  --pretrained_path ../../checkpoints/ttnet_full_finetune/ttnet_full_finetune_epoch_9.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --thresh_ball_pos_mask 0.05 \
  --video_path ../../dataset/test/videos/test_2.mp4 \
  --show_image

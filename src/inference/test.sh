#!/bin/bash

python test.py \
  --gpu_idx 0 \
  --evaluate \
  --batch_size 1 \
  --pretrained_path ../../checkpoints/ttnet_full_freeze_global_freeze_seg/ttnet_full_freeze_global_freeze_seg_epoch_21.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5
#!/bin/bash

python test.py \
  --saved_fn 'ttnet_full_finetune' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../../checkpoints/ttnet_full_finetune/ttnet_full_finetune_epoch_9.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5
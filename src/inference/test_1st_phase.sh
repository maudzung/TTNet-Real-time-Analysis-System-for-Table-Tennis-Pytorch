#!/bin/bash

python test.py \
  --saved_fn 'ttnet_no_local_no_event' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../../checkpoints/ttnet_no_local_no_event/ttnet_no_local_no_event_epoch_21.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --no_local \
  --no_event
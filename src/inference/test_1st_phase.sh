#!/bin/bash

python test.py \
  --saved_fn 'ttnet_1st_phase' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../../checkpoints/ttnet_1st_phase/ttnet_1st_phase_epoch_21.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --no_local \
  --no_event \
  --smooth-labelling
#!/bin/bash

python test.py \
  --working_dir '../' \
  --saved_fn 'ttnet_3rd_phase' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --smooth-labelling
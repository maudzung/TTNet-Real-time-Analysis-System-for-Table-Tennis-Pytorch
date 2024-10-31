#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=August

# python test.py \
#   --working-dir '../' \
#   --saved_fn 'ttnet_3rd_phase' \
#   --gpu_idx 0 \
#   --batch_size 1 \
#   --pretrained_path ../checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth \
#   --seg_thresh 0.5 \
#   --event_thresh 0.5 \
#   --smooth-labelling \
#   --thresh_ball_pos_mask 0.0001



python test.py \
  --working-dir '../' \
  --saved_fn 'ttnet_3rd_phase_128_320' \
  --gpu_idx 0 \
  --batch_size 32 \
  --pretrained_path ../checkpoints/ttnet_3rd_phase_128_320/ttnet_3rd_phase_128_320_best.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --smooth_labelling \
  --thresh_ball_pos_mask 0.00001  \
  --no_seg  \
  --no_event \

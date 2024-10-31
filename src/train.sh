#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=AugustTT

# The first phase: No local, no event

python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_1st_phase_128_320' \
  --gpu_idx 0 \
  --num_epochs 50\
  --batch_size 128 \
  --num_workers 4 \
  --lr 0.001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.1 \
  --global_weight 5. \
  --seg_weight 1. \
  --no_local \
  --no_event \
  --no_test \
  --smooth_labelling  \
# The second phase: Freeze the segmentation and the global modules

python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_2nd_phase_128_320' \
  --num_epochs 50\
  --batch_size 32 \
  --num_workers 10 \
  --lr 0.001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.1 \
  --gpu_idx 0 \
  --global_weight 0. \
  --seg_weight 0. \
  --event_weight 2. \
  --local_weight 1. \
  --pretrained_path ../checkpoints/ttnet_1st_phase_128_320/ttnet_1st_phase_128_320_best.pth \
  --overwrite_global_2_local \
  --freeze_seg \
  --freeze_global \
  --smooth_labelling  \
  --no_event \
  --no_seg  \
  --no_test \

# # The third phase: Finetune all modules

python main.py \
  --working-dir '../' \
  --saved_fn 'ttnet_3rd_phase_128_320' \
  --num_epochs 30\
  --batch_size 32 \
  --num_workers 10 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.2 \
  --gpu_idx 0 \
  --global_weight 1. \
  --no_seg  \
  --seg_weight 1. \
  --event_weight 1. \
  --local_weight 1. \
  --pretrained_path ../checkpoints/ttnet_2nd_phase_128_320/ttnet_2nd_phase_128_320_best.pth \
  --smooth_labelling \
  --no_event \
  --no_seg  \
  --no_test \
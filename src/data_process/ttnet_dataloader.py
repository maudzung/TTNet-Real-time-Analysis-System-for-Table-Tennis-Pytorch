"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import sys

import torch
from torch.utils.data import DataLoader, Subset

sys.path.append('../')

from data_process.ttnet_dataset import TTNet_Dataset, Occlusion_Dataset
from data_process.ttnet_data_utils import get_events_infor, train_val_data_separation
from data_process.transformation import Compose, Random_Crop, Resize, Normalize, Random_Rotate, Random_HFlip, Random_Ball_Mask


def create_train_val_dataloader(configs):
    """Create dataloader for training and validate"""

    train_transform = Compose([
        # Random_Crop(max_reduction_percent=0.15, p=0.5),
        # Random_HFlip(p=0.5),
        # Random_Rotate(rotation_angle_limit=10, p=0.5),
        Random_Ball_Mask(mask_size=(128//20, 320//20), p=0.25),
    ], p=1.)

    train_events_infor, val_events_infor, *_ = train_val_data_separation(configs)
    train_dataset = TTNet_Dataset(train_events_infor, configs.org_size, configs.input_size, transform=train_transform,
                                  num_samples=configs.num_samples)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)

    val_dataloader = None
    if not configs.no_val:
 
        val_transform = Compose([
            Random_Ball_Mask(mask_size=(128//20, 320//20), p=0.25),
        ],  p=1.)
        val_sampler = None
        val_dataset = TTNet_Dataset(val_events_infor, configs.org_size, configs.input_size, transform=val_transform,
                                    num_samples=configs.num_samples)
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)

    return train_dataloader, val_dataloader, train_sampler


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_transform = Compose([
            Random_Ball_Mask(mask_size=(128//20, 320//20), p=1.0),
        ],  p=1.)
    dataset_type = 'test'
    test_events_infor, test_events_labels = get_events_infor(configs.test_game_list, configs, dataset_type)
    test_dataset = TTNet_Dataset(test_events_infor, configs.org_size, configs.input_size, transform=test_transform,
                                 num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler)

    return test_dataloader


def create_occlusion_train_val_dataloader(configs, subset_size=None):
    """Create dataloader for training and validation, with an option to use a subset of the data."""

    train_transform = Compose([
        Resize(new_size=configs.img_size, p=1.0),
        Random_Ball_Mask(mask_size=(128//20, 320//20), p=0.25),
    ], p=1.)

    # Load train and validation data information
    train_events_infor, val_events_infor, train_events_label, val_events_label = train_val_data_separation(configs)

    # Create train dataset
    train_dataset = Occlusion_Dataset(train_events_infor, train_events_label, transform=train_transform,
                                   num_samples=configs.num_samples)
    
    # If subset_size is provided, create a subset for training
    if subset_size is not None:
        train_indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
        train_dataset = Subset(train_dataset, train_indices)
    
    # Create train sampler if distributed
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # Create train dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, 
                                  sampler=train_sampler, drop_last=True)

    # Create validation dataloader (without transformations)
    val_dataloader = None
    if not configs.no_val:
        val_transform = Compose([
            Resize(new_size=configs.img_size, p=1.0),
            Random_Ball_Mask(mask_size=(128//5, 320//5), p=0.5),
        ], p=1.)
        val_dataset = Occlusion_Dataset(val_events_infor, val_events_label, transform=val_transform,
                                     num_samples=configs.num_samples)

        # If subset_size is provided, create a subset for validation
        if subset_size is not None:
            val_indices = torch.randperm(len(val_dataset))[:subset_size].tolist()
            val_dataset = Subset(val_dataset, val_indices)
        
        # Create validation sampler if distributed
        val_sampler = None
        if configs.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        
        # Create validation dataloader
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler, drop_last=True)

    return train_dataloader, val_dataloader, train_sampler


if __name__ == '__main__':
    from config.config import parse_configs
    configs = parse_configs()
    configs.distributed = False  # For testing

    # Create dataloaders
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))

    test_dataloader = create_test_dataloader(configs)
    print(f"len test_loader {len(test_dataloader)}")

    # Get one batch from train_dataloader
    for batch in train_dataloader:
        # Assuming batch contains both input data and labels
        inputs, labels = batch
        print(f"Train batch data shape: {inputs.shape}")
        print(f"Train batch labels shape: {labels.shape}")
        break  # Exit after printing the first batch

    # Get one batch from val_dataloader
    for batch in val_dataloader:
        inputs, labels = batch
        print(f"Val batch data shape: {inputs.shape}")
        print(f"Val batch labels shape: {labels.shape}")
        break

    # Get one batch from test_dataloader
    for batch in test_dataloader:
        # Test dataloader might have only inputs
        print(f"Test batch data shape: {batch.shape}")
        break

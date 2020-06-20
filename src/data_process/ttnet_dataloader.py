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
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.ttnet_dataset import TTNet_Dataset
from data_process.ttnet_data_utils import get_events_infor, train_val_data_separation
from data_process.transformation import Compose, Random_Crop, Resize, Normalize, Random_Rotate, Random_HFlip


def create_train_val_dataloader(configs):
    """Create dataloader for training and validate"""

    train_transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=1.),
        Random_HFlip(p=0.5),
        Random_Rotate(rotation_angle_limit=15, p=0.5),
    ], p=1.)
    val_transform = None

    train_events_infor, val_events_infor, *_ = train_val_data_separation(configs)

    train_dataset = TTNet_Dataset(train_events_infor, configs.org_size, configs.input_size, transform=train_transform,
                                  num_samples=configs.num_samples, no_local=configs.no_local)
    if not configs.no_val:
        val_dataset = TTNet_Dataset(val_events_infor, configs.org_size, configs.input_size, transform=val_transform,
                                    num_samples=configs.num_samples, no_local=configs.no_local)
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        if not configs.no_val:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        if not configs.no_val:
            val_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)
    if not configs.no_val:
        val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                    pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler)
    else:
        val_dataloader = None
    return train_dataloader, val_dataloader, train_sampler


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_transform = None

    dataset_type = 'test'
    test_events_infor, test_events_labels = get_events_infor(configs.test_game_list, configs, dataset_type)
    test_dataset = TTNet_Dataset(test_events_infor, configs.org_size, configs.input_size, transform=test_transform,
                                 num_samples=configs.num_samples, no_local=configs.no_local)

    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers)
    return test_dataloader


if __name__ == '__main__':
    from config.config import parse_configs

    configs = parse_configs()
    configs.distributed = False  # For testing
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    print('len val_dataloader: {}'.format(len(val_dataloader)))
    # for b_idx, data in enumerate(val_dataloader):
    #     print(b_idx)

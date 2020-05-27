import torch
from torch.utils.data import DataLoader

from data_process.ttnet_dataset import TTNet_Dataset
from data_process.ttnet_data_utils import get_events_infor, train_val_data_separation
from data_process.transformation import Compose, Random_Crop, Resize, Normalize, Random_Rotate, Random_HFlip


def create_train_val_dataloader(configs):
    """
    Create dataloader for training and validate
    Args:
        configs:

    Returns:

    """
    train_transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=1.),
        Resize(new_size=(320, 128), p=1.0),
        Random_HFlip(p=0.5),
        Random_Rotate(rotation_angle_limit=15, p=0.5),
        Normalize(p=1.)
    ], p=1.)

    val_transform = Compose([
        Resize(new_size=(320, 128), p=1.0),
        Normalize(p=1.)
    ], p=1.)

    train_events_infor, val_events_infor = train_val_data_separation(configs)

    train_dataset = TTNet_Dataset(train_events_infor, configs.events_dict, transformations=train_transform)
    val_dataset = TTNet_Dataset(val_events_infor, configs.events_dict, transformations=val_transform)
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers)

    return train_dataloader, val_dataloader, train_sampler


def create_test_dataloader(configs):
    """
    Create dataloader for testing phase
    Args:
        configs:

    Returns:

    """
    test_transform = Compose([
        Resize(new_size=(320, 128), p=1.0),
        Normalize(p=1.)
    ], p=1.)
    dataset_type = 'test'
    test_events_infor = get_events_infor(configs.train_game_list, configs, dataset_type)
    test_dataset = TTNet_Dataset(test_events_infor, configs.events_dict, transformations=test_transform)

    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers)
    return test_dataloader


if __name__ == '__main__':
    from config.config import parse_configs

    configs = parse_configs()
    configs.distributed = False # For testing
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    print('len val_dataloader: {}'.format(len(val_dataloader)))
    # for b_idx, data in enumerate(val_dataloader):
    #     print(b_idx)
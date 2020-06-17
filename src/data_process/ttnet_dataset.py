"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: The script for loading TTNet dataset
"""

import sys
import os
import numpy as np

from torch.utils.data import Dataset

sys.path.append('../')

from data_process.ttnet_data_utils import load_raw_img


class TTNet_Dataset(Dataset):
    def __init__(self, events_infor, input_size, transform=None, resize=None, num_samples=None):
        self.events_infor = events_infor
        self.w = input_size[0]
        self.h = input_size[1]
        self.transform = transform
        self.resize = resize
        assert self.resize is not None, "At lease, need to resize images to input_size"
        if num_samples is not None:
            self.events_infor = self.events_infor[:num_samples]

    def __len__(self):
        return len(self.events_infor)

    def __getitem__(self, index):
        img_path_list, org_ball_pos_xy, event_class, seg_path = self.events_infor[index]
        # Load segmentation
        seg_img = load_raw_img(seg_path)

        # Load list of images (-4, 4)
        origin_imgs = []
        for img_path_idx, img_path in enumerate(img_path_list):
            origin_imgs.append(load_raw_img(img_path))
        # loading process faster 3 times with np.dstack() function
        origin_imgs = np.dstack(origin_imgs)  # (1080, 1920, 27)

        # Apply augmentation
        if self.transform:
            origin_imgs, org_ball_pos_xy, seg_img = self.transform(origin_imgs, org_ball_pos_xy, seg_img)
        # resize for the global ball stage
        resized_imgs, global_ball_pos_xy, seg_img = self.resize(origin_imgs, org_ball_pos_xy, seg_img)
        # If the ball position is outside of the resized image, set position to -1, -1 --> No ball (just for safety)
        if (global_ball_pos_xy[0] >= self.w) or (global_ball_pos_xy[1] >= self.h) or (global_ball_pos_xy[0] < 0) or (
                global_ball_pos_xy[1] < 0):
            global_ball_pos_xy[0] = -1
            global_ball_pos_xy[1] = -1

        # Transpose (H, W, C) to (C, H, W) --> fit input of TTNet model
        resized_imgs = resized_imgs.transpose(2, 0, 1)
        origin_imgs = origin_imgs.transpose(2, 0, 1)
        target_seg = seg_img.transpose(2, 0, 1).astype(np.float)
        # Segmentation mask should be 0 or 1
        target_seg[target_seg < 75] = 0.
        target_seg[target_seg >= 75] = 1.

        return origin_imgs, resized_imgs, np.array(org_ball_pos_xy), np.array(global_ball_pos_xy), np.array(
            event_class), target_seg


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    from config.config import parse_configs
    from data_process.ttnet_data_utils import get_events_infor, train_val_data_separation
    from data_process.transformation import Compose, Random_Crop, Resize, Random_HFlip, Random_Rotate

    configs = parse_configs()
    game_list = ['game_1']
    dataset_type = 'training'
    train_events_infor, val_events_infor = train_val_data_separation(configs)
    print('len(train_events_infor): {}'.format(len(train_events_infor)))
    # Test transformation
    transform = Compose([
        Random_Crop(max_reduction_percent=0.15, p=1.),
        Random_HFlip(p=1.),
        Random_Rotate(rotation_angle_limit=15, p=1.)
    ], p=1.)
    resize_transform = Resize(new_size=tuple(configs.input_size), p=1.0)

    ttnet_dataset = TTNet_Dataset(train_events_infor, configs.input_size, transform=transform,
                                  resize=resize_transform)

    print('len(ttnet_dataset): {}'.format(len(ttnet_dataset)))
    example_index = 100
    origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg = ttnet_dataset.__getitem__(
        example_index)

    print('target_seg shape: {}'.format(target_seg.shape))

    origin_imgs = origin_imgs.transpose(1, 2, 0)
    print('origin_imgs shape: {}'.format(origin_imgs.shape))

    out_images_dir = os.path.join(configs.working_dir, 'docs', 'out_images')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.ravel()

    for i in range(configs.num_frames_sequence):
        img = origin_imgs[:, :, (i * 3): (i + 1) * 3]
        axes[i].imshow(img)
        axes[i].set_title('image {}'.format(i))
    fig.suptitle(
        'Event: {}, ball_position_xy: (x= {}, y= {})'.format(event_class, org_ball_pos_xy[0], org_ball_pos_xy[1]),
        fontsize=16)
    plt.savefig(os.path.join(out_images_dir, 'org_all_imgs_{}.jpg'.format(example_index)))
    target_seg = target_seg.transpose(1, 2, 0)

    resized_imgs = resized_imgs.transpose(1, 2, 0)
    resized_imgs = np.array(resized_imgs)
    print('resized_imgs shape: {}'.format(resized_imgs.shape))

    plt.imsave(os.path.join(out_images_dir, 'augment_seg_img_{}.jpg'.format(example_index)), target_seg)
    for i in range(configs.num_frames_sequence):
        img = resized_imgs[:, :, (i * 3): (i + 1) * 3]
        if (i == (configs.num_frames_sequence - 1)):
            img = cv2.resize(img, (img.shape[1], img.shape[0]))
            ball_img = cv2.circle(img, tuple(global_ball_pos_xy), radius=5, color=(255, 0, 0), thickness=2)
            ball_img = cv2.cvtColor(ball_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_images_dir, 'augment_img_{}.jpg'.format(example_index)),
                        ball_img)

        axes[i].imshow(img)
        axes[i].set_title('image {}'.format(i))
    fig.suptitle(
        'Event: {}, ball_position_xy: (x= {}, y= {})'.format(event_class, global_ball_pos_xy[0], global_ball_pos_xy[1]),
        fontsize=16)
    plt.savefig(os.path.join(out_images_dir, 'augment_all_imgs_{}.jpg'.format(example_index)))

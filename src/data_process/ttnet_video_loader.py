"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.10
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: This script creates the video loader for testing with an input video
"""

import os
from collections import deque

import cv2
import numpy as np


class TTNet_Video_Loader:
    """The loader for demo with a video input"""

    def __init__(self, video_path, input_size=(320, 128), num_frames_sequence=9):
        assert os.path.isfile(video_path), "No video at {}".format(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = input_size[0]
        self.height = input_size[1]
        self.count = 0
        self.num_frames_sequence = num_frames_sequence
        print('Length of the video: {:d} frames'.format(self.video_num_frames))

        self.images_sequence = deque(maxlen=num_frames_sequence)
        self.get_first_images_sequence()

    def get_first_images_sequence(self):
        # Load (self.num_frames_sequence - 1) images
        while (self.count < self.num_frames_sequence):
            self.count += 1
            ret, frame = self.cap.read()  # BGR
            assert ret, 'Failed to load frame {:d}'.format(self.count)
            self.images_sequence.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height)))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image

        ret, frame = self.cap.read()  # BGR
        assert ret, 'Failed to load frame {:d}'.format(self.count)
        self.images_sequence.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (self.width, self.height)))
        resized_imgs = np.dstack(self.images_sequence)  # (128, 320, 27)
        # Transpose (H, W, C) to (C, H, W) --> fit input of TTNet model
        resized_imgs = resized_imgs.transpose(2, 0, 1)  # (27, 128, 320)

        return self.count, resized_imgs

    def __len__(self):
        return self.video_num_frames - self.num_frames_sequence + 1  # number of sequences


if __name__ == '__main__':
    import time

    import matplotlib.pyplot as plt
    from config.config import parse_configs

    configs = parse_configs()

    video_path = os.path.join(configs.dataset_dir, 'test', 'videos', 'test_1.mp4')
    video_loader = TTNet_Video_Loader(video_path, input_size=(320, 128),
                                      num_frames_sequence=configs.num_frames_sequence)
    out_images_dir = os.path.join(configs.results_dir, 'debug', 'ttnet_video_loader')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.ravel()

    for example_index in range(1, 10):
        print('process the sequence index: {}'.format(example_index))
        start_time = time.time()
        count, resized_imgs = video_loader.__next__()
        print('time to load sequence {}: {}'.format(example_index, time.time() - start_time))

        resized_imgs = resized_imgs.transpose(1, 2, 0)
        for i in range(configs.num_frames_sequence):
            img = resized_imgs[:, :, (i * 3): (i + 1) * 3]
            axes[i].imshow(img)
            axes[i].set_title('image {}'.format(i))
        plt.savefig(os.path.join(out_images_dir, 'augment_all_imgs_{}.jpg'.format(example_index)))

        origin_imgs = cv2.resize(resized_imgs, (1920, 1080))
        for i in range(configs.num_frames_sequence):
            img = origin_imgs[:, :, (i * 3): (i + 1) * 3]
            axes[i].imshow(img)
            axes[i].set_title('image {}'.format(i))
        plt.savefig(os.path.join(out_images_dir, 'org_all_imgs_{}.jpg'.format(example_index)))

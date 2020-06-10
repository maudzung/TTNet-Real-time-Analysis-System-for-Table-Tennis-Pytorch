"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.10
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: This script for demonstration
"""

import os
import sys
import torch

import cv2

sys.path.append('../')

from data_process.ttnet_video_loader import TTNet_Video_Loader
from training.train_utils import get_model, load_pretrained_model
from config.config import parse_configs


def demo(configs):
    video_loader = TTNet_Video_Loader(configs.video_path, configs.input_size, configs.num_frames_sequence)
    result_filename = os.path.join(configs.result_root, 'results.txt')
    frame_rate = video_loader.video_fps

    configs.frame_dir = None if configs.output_format == 'text' else os.path.join(configs.saved_dir, 'frame')
    if not os.path.isdir(configs.frame_dir):
        os.makedirs(configs.frame_dir)
    try:
        eval_seq(configs, video_loader, result_filename, configs.show_image, frame_rate=frame_rate)
    except Exception as e:
        print('Exception: ', e)

    if configs.output_format == 'video':
        output_video_path = os.path.join(configs.saved_dir, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(
            os.path.join(configs.frame_dir), output_video_path)
        os.system(cmd_str)


def eval_seq(configs, video_loader, result_filename, show_image, frame_rate):
    configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    # model
    model = get_model(configs)
    model.cuda()

    assert configs.pretrained_path is not None, "Need to load the pre-trained model"
    model = load_pretrained_model(model, configs.pretrained_path, configs.gpu_idx, configs.overwrite_global_2_local)
    # Load dataset
    frame_idx = 0
    for count, origin_imgs, resized_imgs in video_loader:
        img = origin_imgs[0:3, :, :]  # Just dummy, change latter
        resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
        origin_imgs = origin_imgs.to(configs.device, non_blocking=True).float()
        pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
            origin_imgs, resized_imgs, None, None, None, None)
        img = img.transpose(1, 2, 0)
        online_im = plot_detection(img)

        if show_image:
            cv2.imshow('online_im', online_im)

        if configs.save_demo_output:
            cv2.imwrite(os.path.join(configs.saved_dir, '{:06d}.jpg'.format(frame_idx)), online_im)

        frame_idx += 1


def plot_detection(img):
    return img


if __name__ == '__main__':
    configs = parse_configs()
    demo(configs=configs)

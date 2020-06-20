"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.20
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
# refer from: https://github.com/NVIDIA/apex/blob/c3fad1ad120b23055f6630da0b029c8b626db78f/tests/L1/common/main_amp.py#L519
-----------------------------------------------------------------------------------
# Description: utils for TTNet dataset
"""

import torch


class data_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_origin_imgs, self.next_resized_imgs, self.next_org_ball_pos_xy, self.next_global_ball_pos_xy, \
            self.next_target_events, self.next_target_seg = next(self.loader)
        except StopIteration:
            self.next_origin_imgs, self.next_resized_imgs, self.next_org_ball_pos_xy, self.next_global_ball_pos_xy, \
            self.next_target_events, self.next_target_seg = None, None, None, None, None, None
            return
        with torch.cuda.stream(self.stream):
            self.next_origin_imgs = self.next_origin_imgs.to(self.device, non_blocking=True)
            self.next_resized_imgs = self.next_resized_imgs.to(self.device, non_blocking=True)
            self.next_org_ball_pos_xy = self.next_org_ball_pos_xy.to(self.device, non_blocking=True)
            self.next_global_ball_pos_xy = self.next_global_ball_pos_xy.to(self.device, non_blocking=True)
            self.next_target_events = self.next_target_events.to(self.device, non_blocking=True)
            self.next_target_seg = self.next_target_seg.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        origin_imgs = self.next_origin_imgs
        resized_imgs = self.next_resized_imgs
        org_ball_pos_xy = self.next_org_ball_pos_xy
        global_ball_pos_xy = self.next_global_ball_pos_xy
        target_events = self.next_target_events
        target_seg = self.next_target_seg
        self.preload()
        return origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg

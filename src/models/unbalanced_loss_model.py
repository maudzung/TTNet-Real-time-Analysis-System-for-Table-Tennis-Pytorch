"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: The unbalanced loss model that train with unweighted/manually weighted losses
"""

import sys

import torch
import torch.nn as nn

sys.path.append('../')

from losses.losses import Ball_Detection_Loss, Events_Spotting_Loss, Segmentation_Loss
from data_process.ttnet_data_utils import create_target_ball


class Unbalance_Loss_Model(nn.Module):
    def __init__(self, model, tasks_loss_weight, weights_events, input_size, sigma, thresh_ball_pos_mask, device):
        super(Unbalance_Loss_Model, self).__init__()
        self.model = model
        self.tasks_loss_weight = torch.tensor(tasks_loss_weight)
        self.tasks_loss_weight = self.tasks_loss_weight / self.tasks_loss_weight.sum()
        self.num_events = len(tasks_loss_weight)
        self.w = input_size[0]
        self.h = input_size[1]
        self.sigma = sigma
        self.thresh_ball_pos_mask = thresh_ball_pos_mask
        self.device = device
        self.ball_loss_criterion = Ball_Detection_Loss(self.w, self.h)
        self.event_loss_criterion = Events_Spotting_Loss(weights=weights_events, num_events=self.num_events)
        self.seg_loss_criterion = Segmentation_Loss()

    def forward(self, resize_batch_input, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg):
        pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy = self.model(resize_batch_input,
                                                                                                 org_ball_pos_xy)
        # Create target for events spotting and ball position (local and global)
        batch_size = pred_ball_global.size(0)
        target_ball_global = torch.zeros_like(pred_ball_global)
        task_idx = 0
        for sample_idx in range(batch_size):
            target_ball_global[sample_idx] = create_target_ball(global_ball_pos_xy[sample_idx], sigma=self.sigma,
                                                                w=self.w, h=self.h,
                                                                thresh_mask=self.thresh_ball_pos_mask,
                                                                device=self.device)
        global_ball_loss = self.ball_loss_criterion(pred_ball_global, target_ball_global)
        total_loss = global_ball_loss * self.tasks_loss_weight[task_idx]

        if pred_ball_local is not None:
            task_idx += 1
            target_ball_local = torch.zeros_like(pred_ball_local)
            for sample_idx in range(batch_size):
                target_ball_local[sample_idx] = create_target_ball(local_ball_pos_xy[sample_idx], sigma=self.sigma,
                                                                   w=self.w, h=self.h,
                                                                   thresh_mask=self.thresh_ball_pos_mask,
                                                                   device=self.device)
            local_ball_loss = self.ball_loss_criterion(pred_ball_local, target_ball_local)
            total_loss += local_ball_loss * self.tasks_loss_weight[task_idx]

        if pred_events is not None:
            task_idx += 1
            target_events = target_events.to(device=self.device)
            event_loss = self.event_loss_criterion(pred_events, target_events)
            total_loss += event_loss * self.tasks_loss_weight[task_idx]

        if pred_seg is not None:
            task_idx += 1
            seg_loss = self.seg_loss_criterion(pred_seg, target_seg)
            total_loss += seg_loss * self.tasks_loss_weight[task_idx]

        return pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, None

    def run_demo(self, resize_batch_input):
        pred_ball_global, pred_ball_local, pred_events, pred_seg = self.model.run_demo(resize_batch_input)
        return pred_ball_global, pred_ball_local, pred_events, pred_seg

"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.05.21
# email: nguyenmaudung93.kstn@gmail.com
# project repo: https://github.com/maudzung/TTNet-Realtime-for-Table-Tennis-Pytorch
-----------------------------------------------------------------------------------
# Description: The multi-task learning model that train with learnt weights of losses
"""

import sys

import torch
import torch.nn as nn

sys.path.append('../')

from losses.losses import Ball_Detection_Loss, Events_Spotting_Loss, Segmentation_Loss
from data_process.ttnet_data_utils import create_target_ball


class Multi_Task_Learning_Model(nn.Module):
    """
    Original paper: "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics" - CVPR 2018
    url: https://arxiv.org/pdf/1705.07115.pdf
    refer code: https://github.com/Hui-Li/multi-task-learning-example-PyTorch
    """

    def __init__(self, model, tasks, num_events, weights_events, input_size, sigma, thresh_ball_pos_mask, device):
        super(Multi_Task_Learning_Model, self).__init__()
        self.model = model
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.log_vars = nn.Parameter(torch.zeros((self.num_tasks)))
        self.w = input_size[0]
        self.h = input_size[1]
        self.sigma = sigma
        self.thresh_ball_pos_mask = thresh_ball_pos_mask
        self.device = device
        self.ball_loss_criterion = Ball_Detection_Loss(self.w, self.h)
        self.event_loss_criterion = Events_Spotting_Loss(weights=weights_events, num_events=num_events)
        self.seg_loss_criterion = Segmentation_Loss()

    def forward(self, resize_batch_input, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg):
        log_vars_idx = 0
        pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy = self.model(resize_batch_input,
                                                                                                 org_ball_pos_xy)
        # Create target for events spotting and ball position (local and global)
        batch_size = pred_ball_global.size(0)
        target_ball_global = torch.zeros_like(pred_ball_global)
        for sample_idx in range(batch_size):
            target_ball_global[sample_idx] = create_target_ball(global_ball_pos_xy[sample_idx], sigma=self.sigma,
                                                                w=self.w, h=self.h,
                                                                thresh_mask=self.thresh_ball_pos_mask,
                                                                device=self.device)
        global_ball_loss = self.ball_loss_criterion(pred_ball_global, target_ball_global)
        total_loss = global_ball_loss / (torch.exp(2 * self.log_vars[log_vars_idx])) + self.log_vars[log_vars_idx]

        if pred_ball_local is not None:
            log_vars_idx += 1
            target_ball_local = torch.zeros_like(pred_ball_local)
            for sample_idx in range(batch_size):
                target_ball_local[sample_idx] = create_target_ball(local_ball_pos_xy[sample_idx], sigma=self.sigma,
                                                                   w=self.w, h=self.h,
                                                                   thresh_mask=self.thresh_ball_pos_mask,
                                                                   device=self.device)
            local_ball_loss = self.ball_loss_criterion(pred_ball_local, target_ball_local)
            total_loss += local_ball_loss / (torch.exp(2 * self.log_vars[log_vars_idx])) + self.log_vars[log_vars_idx]

        if pred_events is not None:
            log_vars_idx += 1
            target_events = target_events.to(device=self.device)
            event_loss = self.event_loss_criterion(pred_events, target_events)
            total_loss += event_loss / (2 * torch.exp(self.log_vars[log_vars_idx])) + self.log_vars[log_vars_idx]

        if pred_seg is not None:
            log_vars_idx += 1
            seg_loss = self.seg_loss_criterion(pred_seg, target_seg)
            total_loss += seg_loss / (2 * torch.exp(self.log_vars[log_vars_idx])) + self.log_vars[log_vars_idx]

        # Final weights: [math.exp(log_var) ** 0.5 for log_var in log_vars]

        return pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, self.log_vars.data.tolist()

    def run_demo(self, resize_batch_input):
        pred_ball_global, pred_ball_local, pred_events, pred_seg = self.model.run_demo(resize_batch_input)
        return pred_ball_global, pred_ball_local, pred_events, pred_seg

import sys

import torch
import torch.nn as nn

sys.path.append('../')

from losses.losses import Ball_Detection_Loss, Events_Spotting_Loss, Segmentation_Loss
from data_process.ttnet_data_utils import create_target_ball, create_target_events


class Multi_Task_Learning_Model(nn.Module):
    """
    Original paper: "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics" - CVPR 2018
    url: https://arxiv.org/pdf/1705.07115.pdf
    refer code: https://github.com/Hui-Li/multi-task-learning-example-PyTorch
    """

    def __init__(self, model, num_tasks=4, num_events=2, weights_events=(1, 3), input_size=(320, 128), device=None):
        super(Multi_Task_Learning_Model, self).__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros((num_tasks)))
        self.w = input_size[0]
        self.h = input_size[1]
        self.ball_loss_criterion = Ball_Detection_Loss(self.w, self.h)
        self.event_loss_criterion = Events_Spotting_Loss(weights=weights_events, num_events=num_events)
        self.seg_loss_criterion = Segmentation_Loss()
        self.device = device

    def forward(self, original_batch_input, resize_batch_input, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg):
        pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy = self.model(original_batch_input,
                                                                                                resize_batch_input,
                                                                                                org_ball_pos_xy)
        # Create target for events spotting and ball position (local and global)
        batch_size = pred_ball_global.size(0)
        target_ball_global = torch.zeros_like(pred_ball_global)
        target_ball_local = torch.zeros_like(pred_ball_global)
        target_events = torch.zeros((batch_size, 2), device=self.device)
        for idx in range(batch_size):
            target_ball_global[idx] = create_target_ball(global_ball_pos_xy[idx], sigma=1., w=self.w, h=self.h, thresh_mask=0.01, device=self.device)
            target_ball_local[idx] = create_target_ball(local_ball_pos_xy[idx], sigma=1., w=self.w, h=self.h, thresh_mask=0.01, device=self.device)
            target_events[idx] = create_target_events(event_class[idx], device=self.device)

        global_ball_loss = self.ball_loss_criterion(pred_ball_global, target_ball_global)
        local_ball_loss = self.ball_loss_criterion(pred_ball_local, target_ball_local)
        event_loss = self.event_loss_criterion(pred_events, target_events)
        seg_loss = self.seg_loss_criterion(pred_seg, target_seg)

        total_loss = global_ball_loss / (torch.exp(self.log_vars[0])) + self.log_vars[0]
        total_loss += local_ball_loss / (torch.exp(self.log_vars[1])) + self.log_vars[1]
        total_loss += event_loss / (torch.exp(self.log_vars[2])) + self.log_vars[2]
        total_loss += seg_loss / (torch.exp(self.log_vars[3])) + self.log_vars[3]

        # Final weights: [math.exp(log_var) ** 0.5 for log_var in log_vars]

        return pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, self.log_vars.data.tolist()

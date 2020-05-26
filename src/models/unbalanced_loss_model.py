import sys

import torch.nn as nn

sys.path.append('../')

from losses.losses import Ball_Detection_Loss, Events_Spotting_Loss, Segmentation_Loss


class Unbalance_Loss_Model(nn.Module):
    def __init__(self, model, num_events=2, weights_events=(1, 3), thresh_seg=0.5, input_size=(320, 128)):
        super(Unbalance_Loss_Model, self).__init__()
        self.model = model
        self.num_events = num_events
        self.w = input_size[0]
        self.h = input_size[1]
        self.ball_loss_criterion = Ball_Detection_Loss(self.w, self.h)
        self.event_loss_criterion = Events_Spotting_Loss(weights=weights_events, num_events=num_events)
        self.seg_loss_criterion = Segmentation_Loss(thresh_seg=thresh_seg)

    def forward(self, original_batch_input, resize_batch_input, target_ball_position, target_events, target_seg):
        pred_ball_position_global, pred_ball_position_local, pred_events, pred_seg = self.model(original_batch_input,
                                                                                                resize_batch_input)
        global_ball_loss = self.ball_loss_criterion(pred_ball_position_global, target_ball_position)
        local_ball_loss = self.ball_loss_criterion(pred_ball_position_local, target_ball_position)
        event_loss = self.event_loss_criterion(pred_events, target_events)
        seg_loss = self.seg_loss_criterion(pred_seg, target_seg)

        total_loss = global_ball_loss + local_ball_loss + event_loss + seg_loss

        return pred_ball_position_global, pred_ball_position_local, pred_events, pred_seg, total_loss, None

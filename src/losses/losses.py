import torch.nn as nn
import torch


class Ball_Detection_Loss(nn.Module):
    def __init__(self, w, h, reduction='mean'):
        super(Ball_Detection_Loss, self).__init__()
        self.w = w
        self.h = h
        self.reduction = reduction

    def forward(self, pred_ball_position, target_ball_position):
        x_pred = pred_ball_position[:self.w]
        y_pred = pred_ball_position[self.w: (self.w + self.h)]

        x_target = target_ball_position[:self.w]
        y_target = target_ball_position[self.w: (self.w + self.h)]
        loss_ball = - torch.sum(x_pred * torch.log(x_target), dim=-1) / self.w - torch.sum(y_pred * torch.log(y_target),
                                                                                           dim=-1) / self.h
        if self.reduction == 'mean':
            loss_ball = loss_ball.mean()

        return loss_ball


class Events_Spotting_Loss(nn.Module):
    def __init__(self, weights=(1, 3), num_events=2, reduction='mean'):
        super(Events_Spotting_Loss, self).__init__()
        self.weights = weights
        self.num_events = num_events
        self.reduction = reduction

    def forward(self, pred_events, target_events):
        loss_event = - torch.sum(self.weights * pred_events * torch.log(target_events), dim=-1) / self.num_events
        if self.reduction == 'mean':
            loss_event = loss_event.mean()

        return loss_event


class DICE_Smotth_Loss(nn.Module):
    def __init__(self, thresh_seg=0.5, epsilon=1e-9, reduction='mean'):
        super(DICE_Smotth_Loss, self).__init__()
        self.thresh_seg = thresh_seg
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred_seg, target_seg):
        pred_seg = pred_seg >= self.thresh_seg
        union = pred_seg * target_seg
        # if self.reduction == 'mean':
        loss_dice_smooth = (torch.sum(2 * union) + self.epsilon) / (
                torch.sum(pred_seg) + torch.sum(target_seg) + self.epsilon)

        return loss_dice_smooth


class Segmentation_Loss(nn.Module):
    def __init__(self, thresh_seg=0.5, reduction='mean'):
        super(Segmentation_Loss, self).__init__()
        self.bce_criterion = torch.nn.BCELoss(reduction=reduction)
        self.dice_criterion = DICE_Smotth_Loss(thresh_seg=thresh_seg, epsilon=1e-9, reduction=reduction)

    def forward(self, pred_seg, target_seg):
        loss_bce = self.bce_criterion(pred_seg, target_seg)
        loss_dice = self.dice_criterion(pred_seg, target_seg)

        loss_seg = loss_bce + loss_dice

        return loss_seg


class Compute_Loss(nn.Module):
    def __init__(self, num_events=2, weights_events=(1, 3), thresh_seg=0.5, input_size=(320, 128), reduction='mean'):
        super(Compute_Loss, self).__init__()
        self.num_events = num_events
        self.w = input_size[0]
        self.h = input_size[1]
        self.ball_loss_criterion = Ball_Detection_Loss(self.w, self.h, reduction=reduction)
        self.event_loss_criterion = Events_Spotting_Loss(weights=weights_events, num_events=num_events,
                                                         reduction=reduction)
        self.seg_loss_criterion = Segmentation_Loss(thresh_seg=thresh_seg, reduction='mean')

    def forward(self, pred_ball_position, target_ball_position, pred_events, target_events, pred_seg, target_seg):
        ball_loss = self.ball_loss_criterion(pred_ball_position, target_ball_position)
        event_loss = self.event_loss_criterion(pred_events, target_events)
        seg_loss = self.seg_loss_criterion(pred_seg, target_seg)
        total_loss = ball_loss + event_loss + seg_loss

        return total_loss


class Multi_Task_Learning_Model(nn.Module):
    """
    Original paper: "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics" - CVPR 2018
    url: https://arxiv.org/pdf/1705.07115.pdf
    """

    def __init__(self, model, num_tasks, num_events=2, weights_events=(1, 3), thresh_seg=0.5, input_size=(320, 128),
                 reduction='mean'):
        super(Multi_Task_Learning_Model, self).__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.vars = nn.Parameter(torch.zeros((num_tasks)))
        self.w = input_size[0]
        self.h = input_size[1]
        self.ball_loss_criterion = Ball_Detection_Loss(self.w, self.h, reduction=reduction)
        self.event_loss_criterion = Events_Spotting_Loss(weights=weights_events, num_events=num_events,
                                                         reduction=reduction)
        self.seg_loss_criterion = Segmentation_Loss(thresh_seg=thresh_seg, reduction='mean')

    def forward(self, original_batch_input, resize_batch_input, target_ball_position, target_events, target_seg):
        pred_ball_position_global, pred_ball_position_local, pred_events, pred_seg = self.model(original_batch_input,
                                                                                                resize_batch_input)
        global_ball_loss = self.ball_loss_criterion(pred_ball_position_global, target_ball_position)
        local_ball_loss = self.ball_loss_criterion(pred_ball_position_local, target_ball_position)
        event_loss = self.event_loss_criterion(pred_events, target_events)
        seg_loss = self.seg_loss_criterion(pred_seg, target_seg)
        loss = 0.
        for subloss_idx, subloss in enumerate([global_ball_loss, local_ball_loss, event_loss, seg_loss]):
            loss += subloss / (self.vars[subloss_idx] ** 2) + torch.log(self.vars[subloss_idx])

        return loss, self.vars.data.tolist()

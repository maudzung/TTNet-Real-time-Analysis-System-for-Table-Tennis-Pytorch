import torch.nn as nn
import torch


class Ball_Detection_Loss(nn.Module):
    def __init__(self, w, h):
        super(Ball_Detection_Loss, self).__init__()
        self.w = w
        self.h = h

    def forward(self, pred_ball_position, target_ball_position):
        x_pred = pred_ball_position[:, :self.w]
        y_pred = pred_ball_position[:, self.w: (self.w + self.h)]

        x_target = target_ball_position[:, :self.w]
        y_target = target_ball_position[:, self.w: (self.w + self.h)]

        loss_ball = - torch.sum(x_target * torch.log(x_pred), dim=-1) / self.w - torch.sum(y_target * torch.log(y_pred),
                                                                                           dim=-1) / self.h

        return loss_ball


class Events_Spotting_Loss(nn.Module):
    def __init__(self, weights=(1, 3), num_events=2):
        super(Events_Spotting_Loss, self).__init__()
        self.weights = torch.tensor(weights).view(1, 2)
        self.num_events = num_events

    def forward(self, pred_events, target_events):
        self.weights = self.weights.cuda()
        loss_event = - torch.sum(self.weights * target_events * torch.log(pred_events), dim=-1) / self.num_events

        return loss_event


class DICE_Smotth_Loss(nn.Module):
    def __init__(self, thresh_seg=0.5, epsilon=1e-9):
        super(DICE_Smotth_Loss, self).__init__()
        self.thresh_seg = thresh_seg
        self.epsilon = epsilon

    def forward(self, pred_seg, target_seg):
        pred_seg = pred_seg >= self.thresh_seg
        union = pred_seg * target_seg
        loss_dice_smooth = (torch.sum(2 * union, dim=(1, 2, 3)) + self.epsilon) / (
                torch.sum(pred_seg, dim=(1, 2, 3)) + torch.sum(target_seg, dim=(1, 2, 3)) + self.epsilon)

        return loss_dice_smooth


class Segmentation_Loss(nn.Module):
    def __init__(self, thresh_seg=0.5):
        super(Segmentation_Loss, self).__init__()
        self.bce_criterion = torch.nn.BCELoss(reduction='none')  # Keep the size
        self.dice_criterion = DICE_Smotth_Loss(thresh_seg=thresh_seg, epsilon=1e-9)

    def forward(self, pred_seg, target_seg):
        target_seg = target_seg.float()
        loss_bce = self.bce_criterion(pred_seg, target_seg.float())
        loss_bce = loss_bce.mean(dim=(1, 2, 3))
        loss_dice = self.dice_criterion(pred_seg, target_seg)
        loss_seg = loss_bce + loss_dice

        return loss_seg


class Imbalance_Loss_Model(nn.Module):
    def __init__(self, model, num_events=2, weights_events=(1, 3), thresh_seg=0.5, input_size=(320, 128)):
        super(Imbalance_Loss_Model, self).__init__()
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


class Multi_Task_Learning_Model(nn.Module):
    """
    Original paper: "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics" - CVPR 2018
    url: https://arxiv.org/pdf/1705.07115.pdf
    refer code: https://github.com/Hui-Li/multi-task-learning-example-PyTorch
    """

    def __init__(self, model, num_tasks=4, num_events=2, weights_events=(1, 3), thresh_seg=0.5, input_size=(320, 128)):
        super(Multi_Task_Learning_Model, self).__init__()
        self.model = model
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros((num_tasks)))
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

        total_loss = global_ball_loss / (torch.exp(self.log_vars[0])) + self.log_vars[0]
        total_loss += local_ball_loss / (torch.exp(self.log_vars[1])) + self.log_vars[1]
        total_loss += event_loss / (torch.exp(self.log_vars[2])) + self.log_vars[2]
        total_loss += seg_loss / (torch.exp(self.log_vars[3])) + self.log_vars[3]

        # Final weights: [math.exp(log_var) ** 0.5 for log_var in log_vars]

        return pred_ball_position_global, pred_ball_position_local, pred_events, pred_seg, total_loss, self.log_vars

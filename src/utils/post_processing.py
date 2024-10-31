import torch
import numpy as np


def post_processing(pred_ball_global, pred_ball_local, pred_events, pred_seg, w, thresh_ball_pos_prob, seg_thresh,
                    event_thresh):
    prediction_global = get_prediction_ball_pos(pred_ball_global, w, thresh_ball_pos_prob)
    prediction_local = get_prediction_ball_pos(pred_ball_local, w, thresh_ball_pos_prob)
    prediction_seg = get_prediction_seg(pred_seg, seg_thresh)
    prediction_events = prediction_get_events(pred_events, event_thresh)

    return prediction_global, prediction_local, prediction_seg, prediction_events


def get_prediction_ball_pos(pred_ball, w, thresh_ball_pos_prob):
    if pred_ball.is_cuda:
        pred_ball = pred_ball.cpu()
    pred_ball = torch.squeeze(pred_ball).numpy()
    pred_ball[pred_ball < thresh_ball_pos_prob] = 0.
    prediction_ball_x = np.argmax(pred_ball[:w])
    prediction_ball_y = np.argmax(pred_ball[w:])

    return (prediction_ball_x, prediction_ball_y)

def get_prediction_ball_pos_right(pred_ball, thresh_ball_pos_prob):
    # pred_ball is in shape ((b, w),(b,h))
    # convert them into [b*(w,h)]
    converted_pred_balls = [(pred_ball[0][i], pred_ball[1][i]) for i in range(pred_ball[0].shape[0])]
    results = []
    for converted_pred_ball in converted_pred_balls:
        pred_ball_coords_x = converted_pred_ball[0].cpu()
        pred_ball_coords_y = converted_pred_ball[1].cpu()
        pred_ball_coords_x = pred_ball_coords_x.detach().numpy()
        pred_ball_coords_y = pred_ball_coords_y.detach().numpy()
        pred_ball_coords_x [pred_ball_coords_x  < thresh_ball_pos_prob] = 0.
        pred_ball_coords_y [pred_ball_coords_y  < thresh_ball_pos_prob] = 0.
        
        prediction_ball_x = np.argmax(pred_ball_coords_x)
        prediction_ball_y = np.argmax(pred_ball_coords_y)
        results.append([prediction_ball_x, prediction_ball_y])

    return results

def get_prediction_ball_pos_right_test(pred_ball, thresh_ball_pos_prob):
    # pred_ball is in shape (h,w)
    pred_ball_coords_x = pred_ball[0].cpu()
    pred_ball_coords_y = pred_ball[1].cpu()
    pred_ball_coords_x = pred_ball_coords_x.detach().numpy()
    pred_ball_coords_y = pred_ball_coords_y.detach().numpy()
    pred_ball_coords_x [pred_ball_coords_x  < thresh_ball_pos_prob] = 0.
    pred_ball_coords_y [pred_ball_coords_y  < thresh_ball_pos_prob] = 0.
    
    prediction_ball_x = np.argmax(pred_ball_coords_x)
    prediction_ball_y = np.argmax(pred_ball_coords_y)


    return (prediction_ball_x, prediction_ball_y)


def prediction_get_events(pred_events, event_thresh):
    if pred_events.is_cuda:
        pred_events = pred_events.cpu()
    pred_events = torch.squeeze(pred_events).numpy()
    # prediction_events = (pred_events > event_thresh).astype(np.int)
    prediction_events = pred_events
    return prediction_events


def get_prediction_seg(pred_seg, seg_thresh):
    if pred_seg.is_cuda:
        pred_seg = pred_seg.cpu()
    pred_seg = torch.squeeze(pred_seg).numpy().transpose(1, 2, 0)
    prediction_seg = (pred_seg > seg_thresh).astype(int)

    return prediction_seg

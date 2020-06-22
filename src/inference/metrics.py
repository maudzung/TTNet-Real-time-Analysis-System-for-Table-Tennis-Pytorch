import numpy as np


def PCE(sample_prediction_events, sample_target_events):
    """Percentage of Correct Events

    :param sample_prediction_events: prediction of event spotting, size: (2,)
    :param sample_target_events: target/ground-truth of event spotting, size: (2,)
    :return:
    """
    sample_prediction_events[sample_prediction_events >= 0.5] = 1.
    sample_prediction_events[sample_prediction_events < 0.5] = 0.
    sample_target_events[sample_target_events >= 0.5] = 1.
    sample_target_events[sample_target_events < 0.5] = 0.
    diff = sample_prediction_events - sample_target_events
    # Check correct or not
    if np.sum(diff) != 0:  # Incorrect
        ret_pce = 0
    else:  # Correct
        ret_pce = 1
    return ret_pce


def SPCE(sample_prediction_events, sample_target_events, thresh=0.25):
    """Smooth Percentage of Correct Events

    :param sample_prediction_events: prediction of event spotting, size: (2,)
    :param sample_target_events: target/ground-truth of event spotting, size: (2,)
    :param thresh: the threshold for the difference between the prediction and ground-truth
    :return:
    """
    diff = np.abs(sample_prediction_events - sample_target_events)
    if np.sum(diff > thresh) > 0:
        ret_spce = 0
    else:
        ret_spce = 1
    return ret_spce

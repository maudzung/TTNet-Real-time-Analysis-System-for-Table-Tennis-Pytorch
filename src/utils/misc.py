import os

import torch


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # or os.makedirs(folder_name, exist_ok=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(checkpoints_dir, saved_fn, saved_state, is_best, logger=None):
    """
    Save checkpoint every epoch
    Args:
        checkpoints_dir:
        saved_fn:
        saved_state:
        is_best:
        logger:

    Returns:

    """
    if is_best:
        save_path = os.path.join(checkpoints_dir, '{}_best.pth'.format(saved_fn))
    else:
        save_path = os.path.join(checkpoints_dir, '{}.pth'.format(saved_fn))

    torch.save(saved_state, save_path)
    if logger:
        logger.info('save the checkpoint at {}'.format(save_path))

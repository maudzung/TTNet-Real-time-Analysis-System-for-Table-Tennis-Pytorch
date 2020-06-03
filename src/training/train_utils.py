import torch
import sys
import copy
import os

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

sys.path.append('../')

from models.TTNet import TTNet
from models.unbalanced_loss_model import Unbalance_Loss_Model
from models.multi_task_learning_model import Multi_Task_Learning_Model


def get_model(configs):
    """
    Create model based on backbone name
    Args:
        configs:

    Returns:

    """
    if configs.arch == 'ttnet':
        ttnet_model = TTNet(dropout_p=configs.dropout_p, tasks=configs.tasks)
    else:
        assert False, 'Undefined model backbone'

    if configs.multitask_learning == True:
        model = Multi_Task_Learning_Model(ttnet_model, tasks=configs.tasks, num_events=2, weights_events=(1, 3),
                                          input_size=(320, 128), device=configs.device)
    else:
        model = Unbalance_Loss_Model(ttnet_model, num_events=2, weights_events=(1, 3),
                                     input_size=(320, 128), device=configs.device)

    return model


def get_num_parameters(model):
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters


def resume_model(resume_path, arch, gpu_idx):
    assert os.path.isfile(resume_path), "=> no checkpoint found at '{}'".format(resume_path)
    if gpu_idx is None:
        checkpoint = torch.load(resume_path, map_location='cpu')
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu_idx)
        checkpoint = torch.load(resume_path, map_location=loc)
    assert arch == checkpoint['configs'].arch, "Load the different arch..."
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))

    return checkpoint


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.batch_size = int(configs.batch_size / configs.ngpus_per_node)
            configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model


def get_optimizer(configs, model, is_warm_up):
    """
    Initialize optimizer for training process
    Args:
        configs:
        model:
        is_warm_up:

    Returns:

    """
    # trainable_vars = [param for param in model.module.parameters() if param.requires_grad]
    if is_warm_up:
        lr = configs.train_warmup_lr
        momentum = configs.train_warmup_momentum
        weight_decay = configs.train_warmup_weight_decay
        optimizer_type = configs.train_warmup_optimizer_type
    else:
        lr = configs.lr
        momentum = configs.momentum
        weight_decay = configs.weight_decay
        optimizer_type = configs.optimizer_type
    if hasattr(model, 'module'):
        train_params = model.module.parameters()
    else:
        train_params = model.parameters()

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(train_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=lr, weight_decay=weight_decay)
    else:
        assert False, "Unknown optimizer type"

    return optimizer


def get_lr_scheduler(optimizer, configs):
    if configs.lr_type == 'step_lr':
        lr_scheduler = StepLR(optimizer, step_size=configs.lr_step_size, gamma=configs.lr_factor)
    elif configs.lr_type == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=configs.lr_factor, patience=configs.lr_patience)
    else:
        raise TypeError

    return lr_scheduler


def get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss, earlystop_count):
    """
    Get the information to save with checkpoints
    Args:
        model:
        optimizer:
        epoch:
        configs:

    Returns:

    """
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    saved_state = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': lr_scheduler.state_dict(),
        'state_dict': model_state_dict,
        'best_val_loss': best_val_loss,
        'earlystop_count': earlystop_count,
    }

    return saved_state


def save_checkpoint(checkpoints_dir, saved_fn, saved_state, is_best, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch
    """
    if is_best:
        save_path = os.path.join(checkpoints_dir, '{}_best.pth'.format(saved_fn))
    else:
        save_path = os.path.join(checkpoints_dir, '{}_epoch_{}.pth'.format(saved_fn, epoch))

    torch.save(saved_state, save_path)
    print('save a checkpoint at {}'.format(save_path))


if __name__ == '__main__':
    from config.config import get_default_configs

    configs = get_default_configs()
    model = get_model(configs)

import torch
import sys
import copy
import os

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
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
    if configs.model_backbone == 'ttnet':
        ttnet_model = TTNet(dropout_p=configs.model_dropout_p)
    else:
        assert False, 'Undefined model backbone'

    if configs.multitask_learning == True:
        model = Multi_Task_Learning_Model(ttnet_model, num_tasks=4, num_events=2, weights_events=(1, 3), thresh_seg=0.5,
                                          input_size=(320, 128), device=configs.device)
    else:
        model = Unbalance_Loss_Model(ttnet_model, num_events=2, weights_events=(1, 3), thresh_seg=0.5,
                                     input_size=(320, 128), device=configs.device)

    return model


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
        lr = configs.train_lr
        momentum = configs.train_momentum
        weight_decay = configs.train_weight_decay
        optimizer_type = configs.train_optimizer_type
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
    if configs.train_lr_type == 'step_lr':
        lr_scheduler = StepLR(optimizer, step_size=configs.train_lr_step_size, gamma=configs.train_lr_factor)
    elif configs.train_lr_type == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=configs.train_lr_factor, patience=configs.train_lr_patience)
    else:
        raise TypeError

    return lr_scheduler


def get_saved_state(model, optimizer, epoch, configs):
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
        'state_dict': model_state_dict,
    }

    return saved_state


def get_metrics(all_targets, all_preds):
    """
    Calculate evaluation metrics during training/validate/testing phase
    Args:
        all_targets:
        all_preds:

    Returns:

    """
    acc = accuracy_score(all_targets, all_preds) * 100.
    micro_f1 = f1_score(all_targets, all_preds, average='micro')
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    confusion_mat = confusion_matrix(y_true=all_targets, y_pred=all_preds)

    return acc, micro_f1, macro_f1, confusion_mat


def write_sumup_results(configs, test_acc, test_micro_f1, test_macro_f1):
    """
    Write the results on test set to a file when the training process is done.
    Args:
        configs:
        test_acc:
        test_micro_f1:
        test_macro_f1:

    Returns:

    """
    result_filepath = os.path.join(configs.results_dir, 'sumup_results.csv')
    if os.path.isfile(result_filepath):
        f_write_results = open(result_filepath, 'a+')
    else:
        f_write_results = open(result_filepath, 'w')
        row = 'model_name,num_points,min_num_points,'
        row += 'loss_type,dropout_p,'
        row += 'data_sampler,uniform,feature_transform,'
        row += 'test_acc,test_micro_f1,test_macro_f1\n'
        f_write_results.write(row)

    row = '{},{},{},'.format(configs.model_backbone, configs.num_points, configs.min_num_points)
    row += '{},{:.2f},'.format(configs.loss_type, configs.model_dropout_p)
    row += 'Yes,' if configs.data_sampler else 'No,'
    row += 'Yes,' if configs.uniform else 'No,'
    row += 'Yes,' if configs.feature_transform else 'No,'
    row += '{:.2f},{:.4f},{:.4f}\n'.format(test_acc, test_micro_f1, test_macro_f1)
    f_write_results.write(row)

    f_write_results.close()


if __name__ == '__main__':
    from config.config import get_default_configs

    configs = get_default_configs()
    model = get_model(configs)

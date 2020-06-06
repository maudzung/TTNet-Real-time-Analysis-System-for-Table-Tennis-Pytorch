import torch
import os
import datetime
import argparse
from easydict import EasyDict as edict
import sys

sys.path.append('../')

from utils.misc import make_folder


def parse_configs():
    parser = argparse.ArgumentParser(description='TTNet Implementation')
    parser.add_argument('--seed', type=int, default=2020,
                        help='re-produce the results with seed random')
    parser.add_argument('--saved_fn', type=str, default='ttnet', metavar='FN',
                        help='The name using for saving logs, models,...')
    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    parser.add_argument('-a', '--arch', type=str, default='ttnet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--dropout_p', type=float, default=0.5, metavar='P',
                        help='The dropout probability of the model')
    parser.add_argument('--multitask_learning', action='store_true',
                        help='If true, the weights of different losses will be learnt (train).'
                             'If false, a regular sum of different losses will be applied')
    parser.add_argument('--no_local', action='store_true',
                        help='If true, no local stage for ball detection.')
    parser.add_argument('--no_event', action='store_true',
                        help='If true, no event spotting detection.')
    parser.add_argument('--no_seg', action='store_true',
                        help='If true, no segmentation module.')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--no-val', action='store_true',
                        help='If true, use all data for training, no validation set')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='The size of validation set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='mini-batch size (default: 16), this is the total'
                             'batch size of all GPUs on the current node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--print_freq', type=int, default=10, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--checkpoint_freq', type=int, default=3, metavar='N',
                        help='frequency of saving checkpoints (default: 3)')
    parser.add_argument('--sigma', type=float, default=0.5, metavar='SIGMA',
                        help='standard deviation of the 1D Gaussian for the ball position target')
    parser.add_argument('--thresh_ball_pos_mask', type=float, default=0.01, metavar='THRESH',
                        help='the lower thresh for the 1D Gaussian of the ball position target')
    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################

    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=40, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-7, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6, metavar='WD',
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')
    parser.add_argument('--lr_type', type=str, default='plateau', metavar='SCHEDULER',
                        help='the type of the learning rate scheduler (steplr or ReduceonPlateau)')
    parser.add_argument('--lr_factor', type=float, default=0.5, metavar='FACTOR',
                        help='reduce the learning rate with this factor')
    parser.add_argument('--lr_step_size', type=int, default=5, metavar='STEP_SIZE',
                        help='step_size of the learning rate when using steplr scheduler')
    parser.add_argument('--lr_patience', type=int, default=3, metavar='N',
                        help='patience of the learning rate when using ReduceoPlateau scheduler')
    parser.add_argument('--earlystop_patience', type=int, default=12, metavar='N',
                        help='Early stopping the training process if performance is not improved within this value')

    ####################################################################
    ##############     Loss weight            ###################
    ####################################################################
    parser.add_argument('--bce_weight', type=float, default=0.5,
                        help='The weight of BCE loss in segmentation module, the dice_loss weight = 1- bce_weight')
    parser.add_argument('--global_weight', type=float, default=1.,
                        help='The weight of loss of the global stage for ball detection')
    parser.add_argument('--local_weight', type=float, default=1.,
                        help='The weight of loss of the local stage for ball detection')
    parser.add_argument('--event_weight', type=float, default=1.,
                        help='The weight of loss of the event spotting module')
    parser.add_argument('--seg_weight', type=float, default=1.,
                        help='The weight of BCE loss in segmentation module')

    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate the model, not training')
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    parser.add_argument('--use_best_checkpoint', action='store_true',
                        help='If true, choose the best model on val set, otherwise choose the last model')

    configs = edict(vars(parser.parse_args()))

    ####################################################################
    ############## Hardware configurations ############################
    ####################################################################
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda')
    configs.ngpus_per_node = torch.cuda.device_count()

    configs.pin_memory = True

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################
    configs.working_dir = '../../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset')
    configs.train_game_list = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5']
    configs.test_game_list = ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6', 'test_7']
    configs.events_dict = {
        'bounce': 0,
        'net': 1,
        'empty_event': 2
    }
    configs.events_weights_loss_dict = {
        'bounce': 1.,
        'net': 3.,
    }
    configs.events_weights_loss = (configs.events_weights_loss_dict['bounce'], configs.events_weights_loss_dict['net'])
    configs.num_events = len(configs.events_weights_loss_dict)  # Just "bounce" and "net hits"
    configs.num_frames_sequence = 9

    configs.input_size = (320, 128)

    configs.tasks = ['global', 'local', 'event', 'seg']
    if configs.no_local:
        if 'local' in configs.tasks:
            configs.tasks.remove('local')
        if 'event' in configs.tasks:
            configs.tasks.remove('event')
    if configs.no_event:
        if 'event' in configs.tasks:
            configs.tasks.remove('event')
    if configs.no_seg:
        if 'seg' in configs.tasks:
            configs.tasks.remove('seg')

    # Compose loss weight for tasks, normalize the weights later
    loss_weight_dict = {
        'global': configs.global_weight,
        'local': configs.local_weight,
        'event': configs.event_weight,
        'seg': configs.seg_weight
    }
    configs.tasks_loss_weight = []
    for task in configs.tasks:
        configs.tasks_loss_weight.append(loss_weight_dict[task])
    ####################################################################
    ############## logs, Checkpoints, and results dir ########################
    ####################################################################
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.saved_fn)
    configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.saved_fn)
    configs.use_best_checkpoint = True

    if configs.use_best_checkpoint:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}_best.pth'.format(configs.saved_fn))
    else:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}.pth'.format(configs.saved_fn))

    configs.results_dir = os.path.join(configs.working_dir, 'results')

    make_folder(configs.checkpoints_dir)
    make_folder(configs.logs_dir)
    make_folder(configs.results_dir)

    return configs


if __name__ == "__main__":
    configs = parse_configs()
    print(configs)

    print(datetime.date.today())
    print(datetime.datetime.now().year)

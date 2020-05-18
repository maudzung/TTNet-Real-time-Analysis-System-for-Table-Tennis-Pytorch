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
    parser.add_argument('--seed', type=int, default=2020, help='re-produce the results with seed random')

    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    parser.add_argument('--model_backbone', type=str, default='ttnet')
    parser.add_argument('--model_dropout_p', type=float, default=0.5)

    ####################################################################
    ##############     Losses configs            ###################
    ####################################################################
    parser.add_argument('--loss_type', type=str, default='CE', help='CE Focal')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--num_filepaths', type=int, default=None, help='Test with small dataset')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_sampler', type=bool, default=False)

    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--verbose', type=bool, default=True)

    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################

    parser.add_argument('--train_num_epochs', type=int, default=200)
    parser.add_argument('--train_lr', type=float, default=1e-4)
    parser.add_argument('--train_minimum_lr', type=float, default=1e-7)
    parser.add_argument('--train_momentum', type=float, default=0.9)
    parser.add_argument('--train_weight_decay', type=float, default=1e-6)
    parser.add_argument('--train_optimizer_type', type=str, default='adam', help='sgd or adam')
    parser.add_argument('--train_lr_step_size', type=int, default=20)
    parser.add_argument('--train_lr_type', type=str, default='step_lr')
    parser.add_argument('--train_lr_factor', type=float, default=0.5)

    parser.add_argument('--train_lr_patience', type=int, default=3)
    parser.add_argument('--train_earlystop_patience', type=int, default=12)

    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--is_test_during_training', type=bool, default=True)
    parser.add_argument('--use_best_checkpoint', type=bool, default=True)

    configs = edict(vars(parser.parse_args()))

    ####################################################################
    ############## Hardware configurations ############################
    ####################################################################
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs.num_gpus = torch.cuda.device_count()
    configs.pin_memory = False

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################
    configs.working_dir = '../../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset')
    configs.train_game_list = ['game_1', 'game_2', 'game_3', 'game_4', 'game_5']
    configs.events_dict = {
        'bounce': 0,
        'net': 1,
        'empty_event': 2
    }
    configs.events_weights_loss_dict = {
        'bounce': 1.,
        'net': 3.,
    }

    configs.events_weights_loss = [configs.events_weights_loss_dict['bounce'], configs.events_weights_loss_dict['net']]

    configs.num_events = 2  # Just "bounce" and "net hits"
    configs.num_frames_sequence = 9

    ####################################################################
    ############## logs, Checkpoints, and results dir ########################
    ####################################################################
    configs.task = 'ttnet'.format()

    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.task)
    configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.task)
    configs.saved_fn = configs.task
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

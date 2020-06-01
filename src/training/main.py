import argparse
import time
import numpy as np
import sys
import random
import os
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('../')

from data_process.ttnet_dataloader import create_train_val_dataloader, create_test_dataloader
from training.train_utils import get_model, get_optimizer, get_lr_scheduler, get_saved_state, get_metrics, \
    write_sumup_results
from utils.misc import AverageMeter, save_checkpoint, ProgressMeter
from utils.logger import Logger
from config.config import parse_configs


def main():
    configs = parse_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        configs.world_size = configs.ngpus_per_node * configs.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        # Simply call main_worker function
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):


    configs.gpu_idx = gpu_idx

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    configs.is_master_node = (not configs.distributed) or (
                configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))

        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None
    # model
    model = get_model(configs)
    # summary(model.cuda(), (27, 1024))

    # Data Parallel
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

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_loader, val_loader, train_sampler = create_train_val_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in train set: {}, val set: {}'.format(len(train_loader), len(val_loader)))

    optimizer = get_optimizer(configs, model, is_warm_up=False)
    lr_scheduler = get_lr_scheduler(optimizer, configs)
    best_val_loss = np.inf
    lr = configs.train_lr
    earlystop_count = 0
    for epoch in range(1, configs.train_num_epochs + 1):
        # train_loader, val_loader = get_dataloader(configs)
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.train_num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}] learning rate: {}'.format(epoch, configs.train_num_epochs, lr))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, configs, logger)
        # evaluate on validation set
        val_loss = validate_one_epoch(val_loader, model, epoch, configs, logger)

        is_best = val_loss <= best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        print_string = '\t--- train_loss: {:.4f}, val_loss: {:.4f}, best_val_loss: {:.4f}\t'.format(
            train_loss,
            val_loss,
            best_val_loss)
        if tb_writer is not None:
            tb_writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

        saved_state = get_saved_state(model, optimizer, epoch, configs)
        if configs.is_master_node:
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=is_best, logger=None)

        # Adjust learning rate
        if configs.train_lr_type == 'step_lr':
            lr_scheduler.step()
        elif configs.train_lr_type == 'plateau':
            lr_scheduler.step(val_loss)
        # Get next learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        if configs.train_earlystop_patience:
            earlystop_count = 0 if is_best else (earlystop_count + 1)
            print_string += ' |||\t earlystop_count: {}'.format(earlystop_count)

        if configs.train_earlystop_patience:
            if configs.train_earlystop_patience <= earlystop_count:
                print_string += '\n\t--- Early stopping!!!'
                break
            else:
                print_string += '\n\t--- Continue training..., earlystop_count: {}'.format(earlystop_count)

        if logger is not None:
            logger.info(print_string)
    if tb_writer is not None:
        tb_writer.close()
    cleanup()


def cleanup():
    dist.destroy_process_group()


def train_one_epoch(train_loader, model, optimizer, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Train - Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    start_time = time.time()
    for batch_idx, (origin_imgs, aug_imgs, target_ball_position, target_events, target_seg, _, _) in enumerate(
            tqdm(train_loader)):
        data_time.update(time.time() - start_time)
        b_size = origin_imgs.size(0)
        target_ball_position = target_ball_position.to(configs.device, non_blocking=True)
        target_events = target_events.to(configs.device, non_blocking=True)
        target_seg = target_seg.to(configs.device, non_blocking=True)

        aug_imgs = aug_imgs.to(configs.device, non_blocking=True).float()
        origin_imgs = origin_imgs.to(configs.device, non_blocking=True).float()

        # compute output
        pred_ball_position_global, pred_ball_position_local, pred_events, pred_seg, total_loss, _ = model(
            origin_imgs, aug_imgs, target_ball_position, target_events, target_seg)
        # For multiple GPU
        total_loss = torch.mean(total_loss)

        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradient and perform backpropagation
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), b_size)
        # measure elapsed time
        batch_time.update(time.time() - start_time)

        # Log message
        if ((batch_idx + 1) % configs.print_freq) == 0:
            logger.info(progress.get_message(batch_idx))

        start_time = time.time()

    return losses.avg


def validate_one_epoch(val_loader, model, epoch, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses],
        prefix="Validation - Epoch: [{}]".format(epoch))
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (origin_imgs, aug_imgs, target_ball_position, target_events, target_seg, _, _) in enumerate(
                tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            b_size = origin_imgs.size(0)
            target_ball_position = target_ball_position.to(configs.device, non_blocking=True)
            target_events = target_events.to(configs.device, non_blocking=True)
            target_seg = target_seg.to(configs.device, non_blocking=True)

            aug_imgs = aug_imgs.to(configs.device, non_blocking=True).float()
            origin_imgs = origin_imgs.to(configs.device, non_blocking=True).float()
            # compute output
            pred_ball_position_global, pred_ball_position_local, pred_events, pred_seg, total_loss, _ = model(
                origin_imgs, aug_imgs, target_ball_position, target_events, target_seg)
            total_loss = torch.mean(total_loss)

            losses.update(total_loss.item(), b_size)
            # measure elapsed time
            batch_time.update(time.time() - start_time)

            # Log message
            if ((batch_idx + 1) % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))

            start_time = time.time()

    return losses.avg


if __name__ == '__main__':
    main()

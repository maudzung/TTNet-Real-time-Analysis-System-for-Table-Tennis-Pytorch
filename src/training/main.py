import time
import numpy as np
import sys
import random
import os
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('../')

from data_process.ttnet_dataloader import create_train_val_dataloader
from models.model_utils import create_model, load_pretrained_model, make_data_parallel, resume_model, get_num_parameters
from models.model_utils import freeze_model
from training.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.misc import AverageMeter, ProgressMeter
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
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

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
    model = create_model(configs)

    # Data Parallel
    model = make_data_parallel(model, configs)

    # Freeze model
    model = freeze_model(model, configs.freeze_modules_list)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    best_val_loss = np.inf
    earlystop_count = 0

    # optionally load weight from a checkpoint
    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx, configs.overwrite_global_2_local)
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    # optionally resume from a checkpoint
    if configs.resume_path is not None:
        checkpoint = resume_model(configs.resume_path, configs.arch, configs.gpu_idx)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_val_loss = checkpoint['best_val_loss']
        earlystop_count = checkpoint['earlystop_count']
        configs.start_epoch = checkpoint['epoch'] + 1

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_loader, val_loader, train_sampler = create_train_val_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in train set: {}'.format(len(train_loader)))
        if val_loader is not None:
            logger.info('number of batches in val set: {}'.format(len(val_loader)))

    if configs.evaluate:
        assert val_loader is not None, "The validation should not be None"
        val_loss = validate_one_epoch(val_loader, model, configs.start_epoch - 1, configs, logger)
        print('Evaluate, val_loss: {}'.format(val_loss))
        return

    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        # Get the current learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}] learning rate: {:.2e}'.format(epoch, configs.num_epochs, lr))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch, configs, logger)
        # evaluate on validation set
        if not configs.no_val:
            val_loss = validate_one_epoch(val_loader, model, epoch, configs, logger)

        # Adjust learning rate
        if configs.lr_type == 'step_lr':
            lr_scheduler.step()
        elif configs.lr_type == 'plateau':
            assert (not configs.no_val), "Only use plateau when having validation set"
            lr_scheduler.step(val_loss)

        if not configs.no_val:
            is_best = val_loss <= best_val_loss
            best_val_loss = min(val_loss, best_val_loss)
            print_string = '\t--- train_loss: {:.4f}, val_loss: {:.4f}, best_val_loss: {:.4f}\t'.format(
                train_loss,
                val_loss,
                best_val_loss)
            if tb_writer is not None:
                tb_writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

            if configs.is_master_node and (is_best or ((epoch % configs.checkpoint_freq) == 0)):
                saved_state = get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss,
                                              earlystop_count)
                save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best, epoch)

            if configs.earlystop_patience:
                earlystop_count = 0 if is_best else (earlystop_count + 1)
                print_string += ' |||\t earlystop_count: {}'.format(earlystop_count)
                if configs.earlystop_patience <= earlystop_count:
                    print_string += '\n\t--- Early stopping!!!'
                    break
                else:
                    print_string += '\n\t--- Continue training..., earlystop_count: {}'.format(earlystop_count)

            if logger is not None:
                logger.info(print_string)
        else:
            if tb_writer is not None:
                tb_writer.add_scalars('Loss', {'train': train_loss}, epoch)
            if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
                saved_state = get_saved_state(model, optimizer, lr_scheduler, epoch, configs, best_val_loss,
                                              earlystop_count)
                save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, False, epoch)

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
    for batch_idx, (
            origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg) in enumerate(
        tqdm(train_loader)):
        data_time.update(time.time() - start_time)
        batch_size = resized_imgs.size(0)
        target_seg = target_seg.to(configs.device, non_blocking=True)
        resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
        if not configs.no_local:
            origin_imgs = origin_imgs.to(configs.device, non_blocking=True).float()
            pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)
        else:
            pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                None, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)
        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # zero the parameter gradients
        optimizer.zero_grad()
        # compute gradient and perform backpropagation
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - start_time)

        # Log message
        if logger is not None:
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
        for batch_idx, (
                origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg) in enumerate(
            tqdm(val_loader)):
            data_time.update(time.time() - start_time)
            batch_size = resized_imgs.size(0)
            target_seg = target_seg.to(configs.device, non_blocking=True)
            resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
            # Only move origin_imgs to cuda if the model has local stage for ball detection
            if not configs.no_local:
                origin_imgs = origin_imgs.to(configs.device, non_blocking=True).float()
                # compute output
                pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                    origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)
            else:
                pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                    None, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, target_events, target_seg)
            # For torch.nn.DataParallel case
            if (not configs.distributed) and (configs.gpu_idx is None):
                total_loss = torch.mean(total_loss)

            losses.update(total_loss.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()

    return losses.avg


if __name__ == '__main__':
    main()

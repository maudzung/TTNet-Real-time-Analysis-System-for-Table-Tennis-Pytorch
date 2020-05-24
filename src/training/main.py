import argparse
import time
import numpy as np
import sys
import random
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

sys.path.append('../')

from data_process.ttnet_dataloader import create_train_val_dataloader, create_test_dataloader
from training.train_utils import get_model, get_optimizer, get_lr_scheduler, get_saved_state, get_metrics, \
    write_sumup_results
from utils.misc import AverageMeter, save_checkpoint, ProgressMeter
from utils.logger import Logger
from config.config import parse_configs


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
            train_loader):
        data_time.update(time.time() - start_time)
        b_size = origin_imgs.size(0)
        target_ball_position = target_ball_position.to(configs.device)
        target_events = target_events.to(configs.device)
        target_seg = target_seg.to(configs.device)

        aug_imgs = aug_imgs.to(configs.device).float()
        # origin_imgs = origin_imgs.to(configs.device).float()

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
                val_loader):
            data_time.update(time.time() - start_time)
            b_size = origin_imgs.size(0)
            target_ball_position = target_ball_position.to(configs.device)
            target_events = target_events.to(configs.device)
            target_seg = target_seg.to(configs.device)

            aug_imgs = aug_imgs.to(configs.device).float()
            # origin_imgs = origin_imgs.to(configs.device).float()
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

    return losses.avg


def main():
    configs = parse_configs()

    # Re-produce results
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = Logger(configs.logs_dir, configs.saved_fn)
    logger.info('>>> Created a new logger')
    logger.info('>>> configs: {}'.format(configs))

    writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))

    logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_loader, val_loader = create_train_val_dataloader(configs)
    print('train_loader: {}, val_loader: {}'.format(len(train_loader), len(val_loader)))
    # model
    model = get_model(configs).to(configs.device)
    # summary(model.cuda(), (27, 1024))
    # Data Parallel
    if configs.num_gpus > 1:
        model = torch.nn.DataParallel(model)

    optimizer = get_optimizer(configs, model, is_warm_up=False)
    lr_scheduler = get_lr_scheduler(optimizer, configs)
    best_val_loss = np.inf
    lr = configs.train_lr
    earlystop_count = 0
    for epoch in range(1, configs.train_num_epochs + 1):
        # train_loader, val_loader = get_dataloader(configs)
        logger.info('{}'.format('*-' * 40))
        logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.train_num_epochs, '=' * 35))
        logger.info('{}'.format('*-' * 40))

        logger.info('>>> Epoch: [{}/{}] learning rate: {}'.format(epoch, configs.train_num_epochs, lr))
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

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)

        saved_state = get_saved_state(model, optimizer, epoch, configs)
        save_checkpoint(configs.checkpoints_dir, configs.saved_fn, saved_state, is_best=is_best, logger=None)

        # Adjust learning rate
        lr_scheduler.step()
        # Get next learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        if configs.train_earlystop_patience:
            earlystop_count = 0 if is_best else (earlystop_count + 1)
            print_string += ' |||\t earlystop_count: {}'.format(earlystop_count)

        logger.info(print_string)

        if configs.train_earlystop_patience:
            if configs.train_earlystop_patience <= earlystop_count:
                logger.info('\t--- Early stopping!!!')
                break
            else:
                logger.info('\t--- Continue training..., earlystop_count: {}'.format(earlystop_count))

    writer.close()


if __name__ == '__main__':
    main()

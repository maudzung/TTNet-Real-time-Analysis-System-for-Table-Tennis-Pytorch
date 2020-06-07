import time
import sys
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('../')

from data_process.ttnet_dataloader import create_test_dataloader
from training.train_utils import get_model, load_pretrained_model
from training.train_utils import make_data_parallel, get_num_parameters
from utils.misc import AverageMeter
from config.config import parse_configs


def main():
    configs = parse_configs()

    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
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
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    # model
    model = get_model(configs)
    model = make_data_parallel(model, configs)

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        print('number of trained parameters of the model: {}'.format(num_parameters))

    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx, configs.overwrite_global_2_local)
    # Load dataset
    test_loader = create_test_dataloader(configs)
    test(test_loader, model, configs)


def test(test_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    acc_event = AverageMeter('Acc_Event', ':6.4f')
    iou_seg = AverageMeter('IoU_Seg', ':6.4f')
    mse_global = AverageMeter('MSE_Global', ':6.4f')
    mse_local = AverageMeter('MSE_Local', ':6.4f')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (
                origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg) in enumerate(
            tqdm(test_loader)):
            data_time.update(time.time() - start_time)
            batch_size = resized_imgs.size(0)
            target_seg = target_seg.to(configs.device, non_blocking=True)
            resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
            # compute output
            if 'local' in configs.tasks:
                origin_imgs = origin_imgs.to(configs.device, non_blocking=True).float()
                pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                    origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg)
            else:
                pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(
                    None, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg)
            # Transfer output to cpu
            pred_ball_global = pred_ball_global.cpu().numpy()
            global_ball_pos_xy = global_ball_pos_xy.numpy()
            if pred_ball_local is not None:
                pred_ball_local = pred_ball_local.cpu().numpy()
                local_ball_pos_xy = local_ball_pos_xy.cpu().numpy()  # Ground truth of the local stage
            if pred_events is not None:
                pred_events = pred_events.cpu().numpy()
            if pred_seg is not None:
                pred_seg = pred_seg.cpu().numpy()
                target_seg = target_seg.cpu().numpy()

            org_ball_pos_xy = org_ball_pos_xy.numpy()

            for sample_idx in range(batch_size):
                w, h = configs.input_size
                # Get target
                sample_org_ball_pos_xy = org_ball_pos_xy[sample_idx]
                sample_global_ball_pos_xy = global_ball_pos_xy[sample_idx]  # Target
                # Process the global stage
                sample_pred_ball_global = pred_ball_global[sample_idx]
                sample_pred_ball_global[sample_pred_ball_global < configs.thresh_ball_pos_mask] = 0.
                sample_pred_ball_global_x = np.argmax(sample_pred_ball_global[:w])
                sample_pred_ball_global_y = np.argmax(sample_pred_ball_global[w:])

                # Calculate the MSE
                if (sample_global_ball_pos_xy[0] > 0) and (sample_global_ball_pos_xy[1] > 0):
                    mse = (sample_pred_ball_global_x - sample_global_ball_pos_xy[0]) ** 2 + (
                            sample_pred_ball_global_y - sample_global_ball_pos_xy[1]) ** 2
                    mse_global.update(mse)

                print('Global stage: (x, y) - org: ({}, {}), gt = ({}, {}), prediction = ({}, {})'.format(
                    sample_org_ball_pos_xy[0], sample_org_ball_pos_xy[1],
                    sample_global_ball_pos_xy[0], sample_global_ball_pos_xy[1], sample_pred_ball_global_x,
                    sample_pred_ball_global_y))

                # Process local ball stage
                if pred_ball_local is not None:
                    # Get target
                    sample_local_ball_pos_xy = local_ball_pos_xy[sample_idx]  # Target
                    # Process the local stage
                    sample_pred_ball_local = pred_ball_local[sample_idx]
                    sample_pred_ball_local[sample_pred_ball_local < configs.thresh_ball_pos_mask] = 0.
                    sample_pred_ball_local_x = np.argmax(sample_pred_ball_local[:w])
                    sample_pred_ball_local_y = np.argmax(sample_pred_ball_local[w:])

                    # Calculate the MSE
                    if (sample_local_ball_pos_xy[0] > 0) and (sample_local_ball_pos_xy[1] > 0):
                        mse = (sample_pred_ball_local_x - sample_local_ball_pos_xy[0]) ** 2 + (
                                sample_pred_ball_local_y - sample_local_ball_pos_xy[1]) ** 2
                        mse_local.update(mse)

                    print('Local stage: (x, y) - gt = ({}, {}), prediction = ({}, {})'.format(
                        sample_local_ball_pos_xy[0], sample_local_ball_pos_xy[1], sample_pred_ball_local_x,
                        sample_pred_ball_local_y))

                # Process event stage
                if pred_events is not None:
                    sample_target_event = event_class[sample_idx].item()
                    vec_sample_target_event = np.zeros((2,), dtype=np.int)
                    if sample_target_event < 2:
                        vec_sample_target_event[sample_target_event] = 1
                    sample_pred_event = (pred_events[sample_idx] > configs.event_thresh).astype(np.int)
                    print('Event stage: gt = {}, prediction: {}'.format(sample_target_event, pred_events[sample_idx]))
                    diff = sample_pred_event - vec_sample_target_event
                    # Check correct or not
                    if np.sum(diff) != 0:
                        # Incorrect
                        acc_event.update(0)
                    else:
                        # Correct
                        acc_event.update(1)

                # Process segmentation stage
                if pred_seg is not None:
                    sample_target_seg = target_seg[sample_idx].transpose(1, 2, 0)
                    sample_pred_seg = pred_seg[sample_idx].transpose(1, 2, 0)
                    sample_target_seg = sample_target_seg.astype(np.int)
                    sample_pred_seg = (sample_pred_seg > configs.seg_thresh).astype(np.int)

                    # Calculate the IoU
                    iou = 2 * np.sum(sample_target_seg * sample_pred_seg) / (
                            np.sum(sample_target_seg) + np.sum(sample_pred_seg) + 1e-9)
                    iou_seg.update(iou)
                    if configs.save_test_output:
                        fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(10, 5))
                        plt.tight_layout()
                        axes.ravel()
                        axes[2 * sample_idx].imshow(sample_target_seg * 255)
                        axes[2 * sample_idx + 1].imshow(sample_pred_seg * 255)
                        # title
                        target_title = 'target seg'
                        pred_title = 'pred seg'
                        if pred_events is not None:
                            target_title += ', is bounce: {}, is net: {}'.format(vec_sample_target_event[0],
                                                                                 vec_sample_target_event[1])
                            pred_title += ', is bounce: {}, is net: {}'.format(sample_pred_event[0],
                                                                               sample_pred_event[1])

                        axes[2 * sample_idx].set_title(target_title)
                        axes[2 * sample_idx + 1].set_title(pred_title)

                        plt.savefig(os.path.join(configs.saved_dir,
                                                 'batch_idx_{}_sample_idx_{}.jpg'.format(batch_idx, sample_idx)))

            if ((batch_idx + 1) % configs.print_freq) == 0:
                print('batch_idx: {} - Average acc_event: {}, iou_seg: {}, mse_global: {}, mse_local: {}'.format(
                    batch_idx, acc_event.avg, iou_seg.avg, mse_global.avg, mse_local.avg))

            batch_time.update(time.time() - start_time)

            start_time = time.time()

    print('Average acc_event: {}, iou_seg: {}, mse_global: {}, mse_local: {}'.format(acc_event.avg, iou_seg.avg,
                                                                                     mse_global.avg, mse_local.avg))
    print('Done testing')


if __name__ == '__main__':
    main()

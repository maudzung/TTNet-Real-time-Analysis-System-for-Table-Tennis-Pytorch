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
from training.train_utils import get_model
from training.train_utils import make_data_parallel, resume_model
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

    # optionally resume from a checkpoint
    if configs.resume_path is not None:
        checkpoint = resume_model(configs.resume_path, configs.arch, configs.gpu_idx)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        earlystop_count = checkpoint['earlystop_count']
        start_epoch = checkpoint['epoch'] + 1
        print('best_val_loss: {}, earlystop_count: {}, start_epoch: {}'.format(best_val_loss, earlystop_count,
                                                                               start_epoch))

    test_loader = create_test_dataloader(configs)

    if configs.evaluate:
        test(test_loader, model, configs)
        return

def test(test_loader, model, configs):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    # switch to evaluate mode
    model.train() # if use model.val(), the performance become worse
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (origin_imgs, resized_imgs, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg) in enumerate(
            tqdm(test_loader)):
            data_time.update(time.time() - start_time)
            batch_size = resized_imgs.size(0)
            target_seg = target_seg.to(configs.device, non_blocking=True)
            resized_imgs = resized_imgs.to(configs.device, non_blocking=True).float()
            # compute output
            if 'local' in configs.tasks:
                origin_imgs = origin_imgs.to(configs.device, non_blocking=True).float()
                pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(origin_imgs,
                    resized_imgs, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg)
            else:
                pred_ball_global, pred_ball_local, pred_events, pred_seg, local_ball_pos_xy, total_loss, _ = model(None,
                    resized_imgs, org_ball_pos_xy, global_ball_pos_xy, event_class, target_seg)
            print('total_loss: {}'.format(total_loss.item()))
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

            seg_thresh = 0.5
            event_thresh = 0.5
            events_idx_to_names = {
                0: 'bounce',
                1: 'net',
                2: 'empty'
            }
            fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(10, 5))
            plt.tight_layout()
            axes.ravel()
            saved_dir = '../../docs/test_output_full'
            if not os.path.isdir(saved_dir):
                os.makedirs(saved_dir)
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
                print('Global stage: (x, y) - org: ({}, {}), gt = ({}, {}), prediction = ({}, {})'.format(
                    sample_org_ball_pos_xy[0], sample_org_ball_pos_xy[1],
                    sample_global_ball_pos_xy[0], sample_global_ball_pos_xy[1], sample_pred_ball_global_x,
                    sample_pred_ball_global_y))

                # Process event stage
                if pred_events is not None:
                    sample_target_event = event_class[sample_idx].item()
                    sample_pred_event = (pred_events[sample_idx] > event_thresh).astype(np.int)
                    print('Event stage: gt = {}, prediction: {}'.format(sample_target_event, pred_events[sample_idx]))

                if pred_seg is not None:
                    sample_target_seg = target_seg[sample_idx].transpose(1, 2, 0)
                    sample_pred_seg = pred_seg[sample_idx].transpose(1, 2, 0)
                    print('Segmentation: Shape sample_target_seg: {}, sample_pred_seg: {}'.format(
                        sample_target_seg.shape, sample_pred_seg.shape))
                    print('Segmentation: Max values sample_target_seg: {}, sample_pred_seg: {}'.format(
                        sample_target_seg.max(), sample_pred_seg.max()))

                    print('Before cast Segmentation sample_target_seg R: {}, G: {}, B: {}'.format(sample_target_seg[:, :, 0].sum(),
                                                                    sample_target_seg[:, :, 1].sum(),
                                                                    sample_target_seg[:, :, 2].sum()))
                    print('Before cast Segmentation sample_pred_seg R: {}, G: {}, B: {}'.format(
                        sample_pred_seg[:, :, 0].sum(),
                        sample_pred_seg[:, :, 1].sum(),
                        sample_pred_seg[:, :, 2].sum()))
                    sample_target_seg = sample_target_seg.astype(np.int)
                    sample_pred_seg = (sample_pred_seg > seg_thresh).astype(np.int)
                    print('After Segmentation sample_target_seg R: {}, G: {}, B: {}'.format(sample_target_seg[:, :, 0].sum(),
                                                                    sample_target_seg[:, :, 1].sum(),
                                                                    sample_target_seg[:, :, 2].sum()))
                    print('After Segmentation sample_pred_seg R: {}, G: {}, B: {}'.format(
                        sample_pred_seg[:, :, 0].sum(),
                        sample_pred_seg[:, :, 1].sum(),
                        sample_pred_seg[:, :, 2].sum()))
                    axes[2 * sample_idx].imshow(sample_target_seg  * 255)
                    axes[2 * sample_idx + 1].imshow(sample_pred_seg  * 255)
                    # title
                    target_title = 'target seg'
                    pred_title = 'pred seg'
                    if pred_events is not None:
                        target_title += ', event: {}'.format(events_idx_to_names[sample_target_event])
                        pred_title += ', is bounce: {}, is net: {}'.format(sample_pred_event[0], sample_pred_event[1])

                    axes[2 * sample_idx].set_title(target_title)
                    axes[2 * sample_idx + 1].set_title(pred_title)


                    plt.savefig(
                        os.path.join(saved_dir, 'batch_idx_{}_sample_idx_{}.jpg'.format(batch_idx, sample_idx)))

            batch_time.update(time.time() - start_time)

            start_time = time.time()
    print('Done testing')


if __name__ == '__main__':
    main()

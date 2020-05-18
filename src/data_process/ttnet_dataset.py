import sys
import os
import numpy as np

from torch.utils.data import Dataset

sys.path.append('../')

from data_process.ttnet_data_utils import load_raw_img, create_target_ball_possition, create_target_events_spotting


class TTNet_Dataset(Dataset):
    def __init__(self, events_infor, events_dict, transformations=None):
        self.events_infor = events_infor
        self.events_dict = events_dict
        self.transformations = transformations

    def __len__(self):
        return len(self.events_infor)

    def __getitem__(self, index):
        img_path_list, ball_position_xy, event_name, seg_path = self.events_infor[index]
        event_class = self.events_dict[event_name]
        # Load segmentation
        seg_img = load_raw_img(seg_path)

        # Load list of images (-4, 4)
        imgs = None
        for img_path_idx, img_path in enumerate(img_path_list):
            img = load_raw_img(img_path)
            if img_path_idx == 0:
                imgs = img
            else:
                imgs = np.concatenate((imgs, img), axis=-1)

        # Apply augmentation
        if self.transformations:
            imgs, ball_position_xy, seg_img = self.transformations(imgs, ball_position_xy, seg_img)

        # Transpose (H, W, C) to (C, H, W) --> fit input of TTNet model
        imgs = imgs.transpose(2, 0, 1)

        target_ball_possition = create_target_ball_possition(ball_position_xy, sigma=1., w=320., h=128.)
        target_events_spotting = create_target_events_spotting(event_name, configs.events_dict)

        return imgs, target_ball_possition, target_events_spotting, seg_img


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    from config.config import parse_configs
    from data_process.ttnet_data_utils import get_events_infor, train_val_data_separation
    from data_process.transformation import Compose, Random_Crop, Resize, Random_HFlip, Random_Rotate

    configs = parse_configs()
    game_list = ['game_1']
    dataset_type = 'training'
    train_events_infor, val_events_infor = train_val_data_separation(configs)
    print(len(train_events_infor))
    ttnet_dataset = TTNet_Dataset(train_events_infor, configs.events_dict, transformations=None)

    print(len(ttnet_dataset))
    example_index = 150
    imgs, event_name, ball_position_xy, seg_img = ttnet_dataset.__getitem__(example_index)

    imgs = imgs.transpose(1, 2, 0)
    print(imgs.shape)

    out_images_dir = os.path.join(configs.working_dir, 'out_images')
    if not os.path.isdir(out_images_dir):
        os.makedirs(out_images_dir)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    axes = axes.ravel()

    for i in range(configs.num_frames_sequence):
        img = imgs[:, :, (i * 3): (i + 1) * 3]
        axes[i].imshow(img)
        axes[i].set_title('image {}'.format(i))
    fig.suptitle(
        'Event: {}, ball_position_xy: (x= {}, y= {})'.format(event_name, ball_position_xy[0], ball_position_xy[1]),
        fontsize=16)
    plt.savefig(os.path.join(out_images_dir, 'img_example_{}.jpg'.format(example_index)))
    plt.imsave(os.path.join(out_images_dir, 'org_seg_img_example_{}.jpg'.format(example_index)), seg_img)

    # Test transformation
    augment = Compose([
        Random_Crop(max_height_reduction_percent=0.15, max_width_reduction_percent=0.15, p=1.),
        Resize(new_size=(320, 128), p=1.0),
        Random_HFlip(p=1.),
        Random_Rotate(rotation_angle_limit=15, p=1.)
    ], p=1.)
    imgs, ball_position_xy, seg_img = augment(imgs, ball_position_xy, seg_img)
    plt.imsave(os.path.join(out_images_dir, 'af_seg_img_crop_{}.jpg'.format(example_index)), seg_img)
    for i in range(configs.num_frames_sequence):
        img = imgs[:, :, (i * 3): (i + 1) * 3]
        if (i == (configs.num_frames_sequence - 1)):
            ball_img = cv2.circle(img, tuple(ball_position_xy), radius=5, color=(255, 0, 0), thickness=2)
            ball_img = cv2.cvtColor(ball_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_images_dir, 'af_crop_resize_ball_img_{}.jpg'.format(example_index)),
                        ball_img)

        axes[i].imshow(img)
        axes[i].set_title('image {}'.format(i))
    fig.suptitle(
        'Event: {}, ball_position_xy: (x= {}, y= {})'.format(event_name, ball_position_xy[0], ball_position_xy[1]),
        fontsize=16)
    plt.savefig(os.path.join(out_images_dir, 'af_crop_resize_img_{}.jpg'.format(example_index)))

import sys
import os
import numpy as np

from torch.utils.data import Dataset

sys.path.append('../')

from data_process.ttnet_data_utils import load_raw_img


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
        # imgs = []
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
        # imgs = imgs.transpose(2, 0, 1)

        # return imgs, event_class, ball_position_xy, seg_img

        return imgs, event_name, ball_position_xy, seg_img

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from config.config import parse_configs
    from data_process.ttnet_data_utils import get_events_infor

    configs = parse_configs()
    game_list = ['game_1']
    dataset_type = 'training'
    events_infor = get_events_infor(game_list, configs, dataset_type, num_frames_sequence=configs.num_frames_sequence)
    print(len(events_infor))
    ttnet_dataset = TTNet_Dataset(events_infor, configs.events_dict, transformations=None)

    print(len(ttnet_dataset))
    example_index = 150
    imgs, event_name, ball_position_xy, seg_img = ttnet_dataset.__getitem__(example_index)
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
    fig.suptitle('Event: {}, ball_position_xy: (x= {}, y= {})'.format(event_name, ball_position_xy[0], ball_position_xy[1]), fontsize=16)
    plt.savefig(os.path.join(out_images_dir, 'img_example_{}.jpg'.format(example_index)))

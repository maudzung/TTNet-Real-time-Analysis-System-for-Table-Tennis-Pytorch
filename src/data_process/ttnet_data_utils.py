import os
import json

import cv2


def load_raw_img(img_path):
    """
    Load raw image
    :param img_path: The path to the image
    :return:
    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # BGR --> RGB
    return img


def get_events_infor(game_list, configs, dataset_type, num_frames_sequence=9):
    """

    :param game_list:
    :return:
    [
        each event: [[img_path_list], ball_position, event_name, segmentation_path]
    ]
    """
    # the paper mentioned 25, but used 9 frames only
    num_frames_from_event = int((num_frames_sequence - 1) / 2)

    annos_dir = os.path.join(configs.dataset_dir, dataset_type, 'annotations')
    images_dir = os.path.join(configs.dataset_dir, dataset_type, 'images')
    events_infor = []
    for game_name in game_list:
        ball_annos_path = os.path.join(annos_dir, game_name, 'ball_markup.json')
        events_annos_path = os.path.join(annos_dir, game_name, 'events_markup.json')
        # Load ball annotations
        json_ball = open(ball_annos_path)
        ball_annos = json.load(json_ball)

        # Load events annotations
        json_events = open(events_annos_path)
        events_annos = json.load(json_events)
        for event_frameidx, event_name in events_annos.items():
            img_path_list = []
            for f_idx in range(int(event_frameidx) - num_frames_from_event,
                               int(event_frameidx) + num_frames_from_event + 1):
                img_path = os.path.join(images_dir, game_name, 'img_{:06d}.jpg'.format(f_idx))
                img_path_list.append(img_path)
            last_f_idx = int(event_frameidx) + num_frames_from_event
            # Get ball position for the last frame in the sequence
            ball_position_xy = ball_annos['{}'.format(last_f_idx)]
            ball_position_xy = [int(ball_position_xy['x']), int(ball_position_xy['y'])]

            # Get segmentation path for the last frame in the sequence
            seg_path = os.path.join(annos_dir, game_name, 'segmentation_masks', '{}.png'.format(last_f_idx))
            assert os.path.isfile(seg_path) == True, "event_frameidx: {} The segmentation path {} is invalid".format(
                event_frameidx,
                seg_path)

            events_infor.append([img_path_list, ball_position_xy, event_name, seg_path])
    return events_infor


if __name__ == '__main__':
    from config.config import parse_configs

    configs = parse_configs()
    game_list = ['game_1']
    dataset_type = 'training'
    get_events_infor(game_list, configs, dataset_type, num_frames_sequence=9)

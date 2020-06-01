import os
from glob import glob
import json

import cv2


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def get_frame_indexes(events_annos_path):
    json_file = open(events_annos_path)
    events_annos = json.load(json_file)
    selected_indexes = []
    main_frames = sorted(events_annos.keys())
    for main_f_idx in main_frames:
        main_f_idx = int(main_f_idx)
        # take 9 or 25 videos frames with the event right in the middle frame
        for idx in range(main_f_idx - num_frames_from_event, main_f_idx + num_frames_from_event + 1):
            selected_indexes.append(idx)
    selected_indexes = set(selected_indexes)
    return selected_indexes


def extract_images_from_videos(video_path, events_annos_path, out_images_dir):
    # Get the selected frame indexes
    selected_indexes = get_frame_indexes(events_annos_path)

    video_fn = os.path.basename(video_path)[:-4]
    sub_images_dir = os.path.join(out_images_dir, video_fn)

    make_folder(sub_images_dir)

    video_cap = cv2.VideoCapture(video_path)
    n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    f_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('video_fn: {}.mp4, f_width: {}, f_height: {}'.format(video_fn, f_width, f_height))
    print('number of frames in the video: {}, number of selected frames: {}'.format(n_frames, len(selected_indexes)))
    frame_cnt = -1
    while True:
        ret, img = video_cap.read()
        if ret:
            frame_cnt += 1
            if frame_cnt in selected_indexes:
                image_path = os.path.join(sub_images_dir, 'img_{:06d}.jpg'.format(frame_cnt))
                if os.path.isfile(image_path):
                    print('video {} had been already extracted'.format(video_path))
                    break
                cv2.imwrite(image_path, img)
        else:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video_cap.release()
    print('done extraction: {}'.format(video_path))


if __name__ == '__main__':
    dataset_dir = '../dataset'
    num_frames_sequence = 9  # the paper mentioned 25, but used 9 frames only
    num_frames_from_event = int((num_frames_sequence - 1) / 2)
    for dataset_type in ['training', 'test']:
        video_dir = os.path.join(dataset_dir, dataset_type, 'videos')
        annos_dir = os.path.join(dataset_dir, dataset_type, 'annotations')

        out_images_dir = os.path.join(dataset_dir, dataset_type, 'images')

        video_paths = glob(os.path.join(video_dir, '*.mp4'))

        for video_idx, video_path in enumerate(video_paths):
            video_fn = os.path.basename(video_path)[:-4]
            events_annos_path = os.path.join(annos_dir, video_fn, 'events_markup.json')
            extract_images_from_videos(video_path, events_annos_path, out_images_dir)

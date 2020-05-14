import os
from glob import glob

import cv2


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def extract_images_from_videos(video_path, out_images_dir):
    video_fn = os.path.basename(video_path)[:-4]
    sub_images_dir = os.path.join(out_images_dir, video_fn)

    make_folder(sub_images_dir)

    video_cap = cv2.VideoCapture(video_path)
    n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    f_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('video_fn: {}.mp4, number of frames: {}, f_width: {}, f_height: {}'.format(video_fn, n_frames, f_width,
                                                                                     f_height))

    frame_cnt = 0
    while True:
        ret, img = video_cap.read()
        if ret:
            frame_cnt += 1
            image_path = os.path.join(sub_images_dir, 'img_{:06d}.jpg'.format(frame_cnt))
            if os.path.isfile(image_path):
                print('video {} had been already extracted'.format(video_path))
                break
            cv2.imwrite(image_path, img)
        else:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    print('done extraction: {}'.format(video_path))


if __name__ == '__main__':
    dataset_dir = '../../dataset'
    train_video_dir = os.path.join(dataset_dir, 'training', 'videos')
    train_images_dir = os.path.join(dataset_dir, 'training', 'images')

    train_video_paths = glob(os.path.join(train_video_dir, '*.mp4'))

    for video_idx, video_path in enumerate(train_video_paths):
        extract_images_from_videos(video_path, train_images_dir)

    test_video_dir = os.path.join(dataset_dir, 'test', 'videos')
    test_images_dir = os.path.join(dataset_dir, 'test', 'images')

    test_video_paths = glob(os.path.join(test_video_dir, '*.mp4'))

    for video_idx, video_path in enumerate(test_video_paths):
        extract_images_from_videos(video_path, test_images_dir)

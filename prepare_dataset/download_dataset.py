import os

import wget


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


if __name__ == '__main__':
    # Need checksums
    dataset_dir = '../dataset'
    train_video_dir = os.path.join(dataset_dir, 'training', 'videos')
    train_annotation_dir = os.path.join(dataset_dir, 'training', 'annotations')

    test_video_dir = os.path.join(dataset_dir, 'test', 'videos')
    test_annotation_dir = os.path.join(dataset_dir, 'test', 'annotations')

    make_folder(train_video_dir)
    make_folder(train_annotation_dir)
    make_folder(test_video_dir)
    make_folder(test_annotation_dir)

    common_url = 'https://lab.osai.ai/datasets/openttgames/data/'

    train_video_filenames = ['game_{}.mp4'.format(i) for i in range(1, 6)]  # 1 to 5
    train_annotation_filenames = ['game_{}.zip'.format(i) for i in range(1, 6)]  # 1 to 5

    test_video_filenames = ['test_{}.mp4'.format(i) for i in range(1, 8)]  # 1 to 7
    test_annotation_filenames = ['test_{}.zip'.format(i) for i in range(1, 8)]  # 1 to 7

    for video_fn, annos_fn in zip(train_video_filenames, train_annotation_filenames):
        if not os.path.isfile(os.path.join(train_video_dir, video_fn)):
            print('Downloading...{}'.format(common_url + video_fn))
            wget.download(common_url + video_fn, os.path.join(train_video_dir, video_fn))
        if not os.path.isfile(os.path.join(train_annotation_dir, annos_fn)):
            print('Downloading...{}'.format(common_url + annos_fn))
            wget.download(common_url + annos_fn, os.path.join(train_annotation_dir, annos_fn))

    for video_fn, annos_fn in zip(test_video_filenames, test_annotation_filenames):
        if not os.path.isfile(os.path.join(test_video_dir, video_fn)):
            print('Downloading...{}'.format(common_url + video_fn))
            wget.download(common_url + video_fn, os.path.join(test_video_dir, video_fn))
        if not os.path.isfile(os.path.join(test_annotation_dir, annos_fn)):
            print('Downloading...{}'.format(common_url + annos_fn))
            wget.download(common_url + annos_fn, os.path.join(test_annotation_dir, annos_fn))

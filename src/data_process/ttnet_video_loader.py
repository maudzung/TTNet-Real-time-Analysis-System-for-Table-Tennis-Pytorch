import os
from collections import deque

import cv2
import numpy as np


class TTNet_Video_Loader:  # for inference
    def __init__(self, video_path, input_size=(320, 128), num_frames_sequence=9):
        assert os.path.isfile(video_path), "No video at {}".format(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.video_fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = input_size[0]
        self.height = input_size[1]
        self.count = 0
        self.num_frames_sequence = num_frames_sequence
        print('Length of the video: {:d} frames'.format(self.video_num_frames))

        self.images_sequence = deque(maxlen=num_frames_sequence)
        self.get_first_images_sequence()

    def get_first_images_sequence(self):
        # Load (self.num_frames_sequence - 1) images
        while (self.count < self.num_frames_sequence):
            self.count += 1
            ret, frame = self.cap.read()  # BGR
            assert ret, 'Failed to load frame {:d}'.format(self.count)
            self.images_sequence.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image

        ret, frame = self.cap.read()  # BGR
        assert ret, 'Failed to load frame {:d}'.format(self.count)
        self.images_sequence.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        origin_imgs = np.array(self.images_sequence).transpose(1, 2, 0, 3)  # (9, 1080, 1920, 3) --> (1080, 1920, 9, 3)
        # Considering make the reshape step faster!
        origin_imgs = origin_imgs.reshape(origin_imgs.shape[0], origin_imgs.shape[1], -1)  # (1080, 1920, 27)
        resized_imgs = cv2.resize(origin_imgs, (self.width, self.height))

        # Transpose (H, W, C) to (C, H, W) --> fit input of TTNet model
        origin_imgs = origin_imgs.transpose(2, 0, 1)
        resized_imgs = resized_imgs.transpose(2, 0, 1)

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, origin_imgs, resized_imgs

    def __len__(self):
        return self.video_num_frames - self.num_frames_sequence + 1  # number of sequences

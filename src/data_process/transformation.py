import random

import cv2
import numpy as np


class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, imgs, ball_position_xy, seg_img):
        if random.random() <= self.p:
            for t in self.transforms:
                imgs, ball_position_xy, seg_img = t(imgs, ball_position_xy, seg_img)
        return imgs, ball_position_xy, seg_img


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, img):
        img = ((img / 255.) - self.mean) / self.std

        return img


class Denormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0):
        self.p = p
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)

    def __call__(self, img):
        img = (img * self.std + self.mean) * 255.
        img = img.astype(np.uint8)

        return img


class Resize(object):
    def __init__(self, new_size, p=0.5):
        self.new_size = new_size
        self.p = p

    def __call__(self, imgs, ball_position_xy, seg_img):
        if random.random() <= self.p:
            h, w, c = imgs.shape
            assert ((h == 1080.) and (w == 1920.) and (c == 27)), "The image need to be resized first"

            resize_imgs = None
            num_imgs = int(c / 3)
            for i in range(num_imgs):
                resize_img = cv2.resize(imgs[:, :, (3 * i): (3 * (i + 1))], self.new_size)
                if i == 0:
                    resize_imgs = resize_img
                else:
                    resize_imgs = np.concatenate((resize_imgs, resize_img), axis=-1)
            imgs = resize_imgs
            # Dont need to resize seg_img
            # Adjust ball position
            w_ratio = w / self.new_size[0]
            h_ratio = h / self.new_size[1]
            ball_position_xy = [int(ball_position_xy[0] / w_ratio), int(ball_position_xy[1] / h_ratio)]

        return imgs, ball_position_xy, seg_img


class Random_Crop(object):
    def __init__(self, max_height_reduction_percent=0.15, max_width_reduction_percent=0.15, p=0.5):
        self.max_height_reduction_percent = max_height_reduction_percent
        self.max_width_reduction_percent = max_width_reduction_percent
        self.p = p

    def __call__(self, imgs, ball_position_xy, seg_img):
        # imgs are before resizing
        if random.random() <= self.p:
            h, w, c = imgs.shape
            assert ((h == 1080.) and (w == 1920.) and (c == 27)), "The original images are needed"

            # Calculate min_x, max_x, min_y, max_y
            new_w = random.uniform(1. - self.max_width_reduction_percent, 1.) * w
            min_x = int(random.uniform(0, w - new_w))
            max_x = int(min_x + new_w)
            w_ratio = w / new_w

            new_h = random.uniform(1. - self.max_height_reduction_percent, 1.) * h
            min_y = int(random.uniform(0, h - new_h))
            max_y = int(new_h + min_y)
            h_ratio = h / new_h

            num_imgs = int(c / 3)
            for i in range(num_imgs):
                crop_img = imgs[:, :, (3 * i): (3 * (i + 1))][min_y:max_y, min_x:max_x, :]
                imgs[:, :, (3 * i): (3 * (i + 1))] = cv2.resize(crop_img, (w, h))

            # crop seg_img
            seg_img_h, seg_img_w, _ = seg_img.shape
            # 1. Resize to original
            seg_img = cv2.resize(seg_img, (w, h))
            # 2. Crop
            seg_img = seg_img[min_y:max_y, min_x:max_x, :]
            # 3. Resize to (128, 320, 3)
            seg_img = cv2.resize(seg_img, (seg_img_w, seg_img_h))

            # Adjust ball position
            ball_position_xy = [int((ball_position_xy[0] - min_x) * w_ratio),
                                int((ball_position_xy[1] - min_y) * h_ratio)]

        return imgs, ball_position_xy, seg_img


class Random_Rotate(object):
    def __init__(self, rotation_angle_limit=15, p=0.5):
        self.rotation_angle_limit = rotation_angle_limit
        self.p = p

    def __call__(self, imgs, ball_position_xy, seg_img):
        if random.random() <= self.p:
            h, w, c = imgs.shape
            assert ((h == 128.) and (w == 320.) and (c == 27)), "The image need to be resized first"

            center = (int(w / 2), int(h / 2))

            random_angle = random.uniform(-self.rotation_angle_limit, self.rotation_angle_limit)

            rotate_matrix = cv2.getRotationMatrix2D(center, random_angle, 1.)
            # Convert to radian
            # random_angle = (np.pi / 180.0) * random_angle
            num_imgs = int(c / 3)
            for i in range(num_imgs):
                imgs[:, :, (3 * i): (3 * (i + 1))] = cv2.warpAffine(imgs[:, :, (3 * i): (3 * (i + 1))], rotate_matrix,
                                                                    (w, h), flags=cv2.INTER_LINEAR)

            # Rotate seg_img
            seg_img = cv2.warpAffine(seg_img, rotate_matrix, (seg_img.shape[1], seg_img.shape[0]),
                                     flags=cv2.INTER_LINEAR)

            # Adjust ball position
            ball_position_xy = rotate_matrix.dot(np.array([ball_position_xy[0], ball_position_xy[1], 1.]).T)
            ball_position_xy = ball_position_xy.astype(np.int)

        return imgs, ball_position_xy, seg_img


class Random_HFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs, ball_position_xy, seg_img):
        if random.random() <= self.p:
            h, w, c = imgs.shape
            assert ((h == 128.) and (w == 320.) and (c == 27)), "The image need to be resized first"
            num_imgs = int(c / 3)
            for i in range(num_imgs):
                imgs[:, :, (3 * i): (3 * (i + 1))] = cv2.flip(imgs[:, :, (3 * i): (3 * (i + 1))], 1)

            # Horizontal flip seg_img
            seg_img = cv2.flip(seg_img, 1)

            # Adjust ball position: Same y, new x = w - x
            ball_position_xy = [w - ball_position_xy[0], ball_position_xy[1]]

        return imgs, ball_position_xy, seg_img

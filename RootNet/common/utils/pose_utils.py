# --------------------------------------------------------
# 3DMPPE_ROOTNET
# Copyright (c) 2019 Gyeongsik Moon
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import tensorflow as tf
from ...main import cfg


# Convert the camera coordinates to the pixel coordinates.
# f: focal length
# c: Optical center (the principal point)
def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate(x[:, None], y[:, None], z[:, None], dim=1)
    return img_coord


# Convert the pixel coordinates to the camera coordinates
def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate(x[:, None], y[:, None], z[:, None], dim=1)
    return cam_coord


# Convert the world coordinates to the camera coordinates
# capital_r (R): rotation in each axe
# t: translation in each axe
def world2cam(world_coord, capital_r, t):
    cam_coord = np.dot(capital_r, world_coord.transpose((1, 0))).transpose((1, 0)) + t.reshape(1, 3)
    return cam_coord


# extract the keypoint coordinates from the bbox
def get_bbox(joint_img):
    bbox = np.zeros(4)
    x_min = np.min(joint_img[:, 0])
    y_min = np.min(joint_img[:, 1])
    x_max = np.max(joint_img[:, 0])
    y_max = np.max(joint_img[:, 1])
    width = x_max - x_min - 1
    height = y_max - y_min - 1

    # make the bbox 1.2 times bigger
    # but remain the original central point
    bbox[0] = (x_min + x_max) / 2. - width / 2 * 1.2
    bbox[1] = (y_min + y_max) / 2. - height / 2 * 1.2
    bbox[2] = width * 1.2
    bbox[3] = height * 1.2

    return bbox


# sanitize bbox
def process_bbox(bbox, width, height):
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))

    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = cfg.input_shape[1] / cfg.input_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.
    return bbox


# Creates a meshgrid from possibly many
# elements (instead of only 2).
# Returns a nd tensor with as many dimensions
# as there are arguments
def multi_meshgrid(*args):
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [tf.range(tensor.shape[dim] - 1, -1, -1, tf.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    return flipped

"""
This file exports generators that outputs data to train and test on
"""
import os
import re
import random
from glob import glob
from itertools import zip_longest

import imageio
import numpy as np
import torch


def get_vis_png(train=True):
    pass


def _negate_image(image):
    """
    Takes a single value image and for each pixel, 255 - pixel
    """
    negate = lambda x: 255 - x
    vnegate = np.vectorize(negate)
    return vnegate(image)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_combined_epoch_png(data_path, train=True, batch_size=32):
    """
    Combine the VIS and IR images to a single one with the following color map (VIS, 255 - IR, 255 - IR)
    :param train: Should the images be from the Train data set or Eval dataset
    :return: Generator that output (Image, label)
    """
    folder = "Test"
    if train:
        folder = "Train"

    folder_fullpath = os.path.join(data_path, folder)
    glob_pattern = os.path.join(folder_fullpath, "*VIS.png")
    filenames = glob(glob_pattern)
    random.shuffle(filenames)

    # last one can have None and it will break everything
    epoch = list(grouper(filenames, batch_size))[:-1]

    for batch in epoch:
        images, labels = [], []
        for vis_name in batch:
            vis_path = os.path.join(folder_fullpath, vis_name)
            vis = imageio.imread(vis_path)

            ir_name = vis_name.replace("VIS", "IR")
            ir_path = os.path.join(folder_fullpath, ir_name)
            ir = imageio.imread(ir_path)

            image = np.stack([vis, _negate_image(ir), _negate_image(ir)], axis=0)
            images.append(image)

            label = re.search("label_(.*)_", vis_name).group(1)
            label = int(label)-1
            labels.append(label)

        images = torch.from_numpy(np.array(images, dtype="float32"))
        labels = torch.from_numpy(np.array(labels, dtype="int64"))
        yield images, labels

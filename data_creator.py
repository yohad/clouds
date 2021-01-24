"""
This file contains functions that orginize the raw images to data that we can train
neural networks on
"""
import os
from collections import namedtuple, Counter
import re

import errno
import imageio
from glob import glob

import numpy
import matplotlib.pyplot as plt
from tqdm import tqdm

Window = namedtuple("Window", "x, y, size, inner_window_size")


LABELS = {
    0: 1,  # BACKGROUND
    255: 2,  # CLOSED_CELL_CLOUDS
    129: 3,  # OPEN_CELL_CLOUDS
    78: 4,  # UNORGANIZED_CELLULAR_STRUCTURE
    186: 5  # OTHER
}


def _get_basename(filename):
    return re.match("...._.._.._.._.._", filename).group(0)

def _HOTPATCH_visual_filename(filename):
    """
    It seems that I'm missing some files. Fortunatly, I can hotpatch it with close enough samples
    """
    MAPPING = {
        r"D:\clouds\from_dropbox\labeled_ps\2017_10_11_02_45_VIS.png":
            r"D:\clouds\from_dropbox\labeled_ps\2017_10_11_02_30_VIS.png",
        r"D:\clouds\from_dropbox\labeled_ps\2018_09_13_08_45_IFR.png":
            r"D:\clouds\from_dropbox\labeled_ps\2018_09_13_09_00_IFR.png",
    }

    return MAPPING.get(filename, filename)


class Data(object):
    def __init__(self, mask_path, percent):
        visual, ir = self._get_satellite_images(mask_path)
        visual = _HOTPATCH_visual_filename(visual)
        ir = _HOTPATCH_visual_filename(ir)

        self._percent = percent

        self._mask = imageio.imread(mask_path)
        self._visual = imageio.imread(visual)
        self._ir = imageio.imread(ir)

    def get_tagged_data(self, window_size, inner_window_size):
        """
        Get a generator object that will output (Visual, IR, label, window) tagged data
        """
        hight, width = self._mask.shape
        hight = hight - window_size
        width = width - window_size

        while True:
            # Get random place in the image
            x = int(numpy.random.rand() * hight)
            y = int(numpy.random.rand() * width)
            window = Window(x, y, window_size, inner_window_size)

            label = self._get_label(window)
            if label is not None:
                output_vis = Data.get_window(self._visual, window)
                output_ir = Data.get_window(self._ir, window)
                output_mask = Data.get_window(self._mask, window)  # Just here for debugging purposes

                return (output_vis, output_ir, label, window)

    @staticmethod
    def get_window(image, window):
        """
        slicing helper for getting a window from image
        """
        return image[window.x:window.x+window.size, window.y:window.y+window.size]
    
    @staticmethod
    def get_inner_window(image, window):
        """
        slicing helper for getting the inner window from image for labeling
        """
        return image[window.x:window.x+window.inner_window_size, window.y:window.y+window.inner_window_size]

    def _get_label(self, window):
        """
        Gets a window from a mask and check to see if atleast %percent of the pixels are the same
        """
        w = Data.get_window(self._mask, window)
        histogram = Counter(w.flatten())
        total = sum(histogram.values())
        for p, n in histogram.items():
            if n / total > self._percent:
                return LABELS[p]

    def _get_satellite_images(self, masked_path):
        """
        Finds the Visual and IR images corresponding with the given mask
        :return: (VIS.png, IR.png)
        """
        dirpath = os.path.dirname(masked_path)
        filename = os.path.basename(masked_path)
        base = _get_basename(filename)
        vis_png = os.path.join(dirpath, base + "VIS.png")
        ir_png = os.path.join(dirpath, base + "IFR.png")

        return (vis_png, ir_png)


def _safe_create_dirs(path):
    """
    Create the path even if it is already exists
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def from_labled_ps(input_path,
                   output_path,
                   window_size=96,
                   inner_window_size=32,
                   output_size=1600,
                   mask_percentage=0.8,
                   test_percentage=3):
    # setup output folder
    train_path = os.path.join(output_path, "Train")
    _safe_create_dirs(train_path)

    test_path = os.path.join(output_path, "Test")
    _safe_create_dirs(test_path)
    
    # Get all the masks we currently have
    masks = glob(os.path.join(input_path, "*mask.tif"))
    samples_per_mask = int(output_size / len(masks))

    for i, mask in tqdm(enumerate(masks)):
        for _ in range(samples_per_mask):
            d = Data(mask, mask_percentage)
            vis, ir, label, window = d.get_tagged_data(window_size, inner_window_size)

            if i % test_percentage == 0:
                # add to test dataset
                output_dir = test_path
            else:
                # add to train dataset
                output_dir = train_path

            basename = _get_basename(os.path.basename(mask))
            basename = basename + f"{window.x}x{window.y}_label_{label}_"

            vis_png = os.path.join(output_dir, basename + "VIS.png")
            ir_png = os.path.join(output_dir, basename + "IR.png")

            imageio.imwrite(vis_png, vis)
            imageio.imwrite(ir_png, ir)

def main():
    from_labled_ps(r"D:\clouds\from_dropbox\labeled_ps", r"D:\clouds\YotamDataSet\96x96_32x32")


if __name__ == '__main__':
    main()

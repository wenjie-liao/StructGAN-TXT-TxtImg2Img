###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os

TXT_EXTENSIONS = ['.txt']


def is_text_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)


def make_dataset_txt(dir):
    texts = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_text_file(fname):
                path = os.path.join(root, fname)
                texts.append(path)

    return texts
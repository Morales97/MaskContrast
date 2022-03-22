# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
from PIL import Image

import pdb
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
#==========================dataset load==========================

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def pil_loader(_path, width, height):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        with Image.open(f) as _img:
            _img = _img.convert('RGB')
            #_img = _img.resize(_build_size(_img, width, height), Image.ANTIALIAS)
    return _img

class cityscapesDataset(Dataset):
    def __init__(
        self,
        image_path,
        split="train",
        n_samples= -1,        # Select only few samples for training
        size="tiny",
		transform=None
    ):
        self.image_path = image_path
        self.split = split
        if size == "small":
            self.img_size = (1024, 512) # w, h -- PIL uses (w, h) format
        elif size == "tiny":
            self.img_size = (512, 256)
        else:
            raise Exception('size not valid')

        self.n_samples = n_samples
        self.files = {}
        self.transforms = transform
        self.images_base = os.path.join(self.image_path, self.split)

        self.files[split] = sorted(recursive_glob(rootdir=self.images_base, suffix=".jpg"))
        if self.n_samples >= 0:
            self.files[split] = self.files[split][:self.n_samples]
    
        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def get_img_save_path(self, index):
        img_path = self.files[self.split][index]
        img_path = img_path.split(os.sep)
        return img_path[-2], img_path[-1].rstrip('.jpg')

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
            
        # Image
        img = pil_loader(img_path, self.img_size[0], self.img_size[1])
        img = self.transforms(img)

        lbl = np.zeros((512, 256))
        sample = {'image': img, 'label': lbl, 'index': index}

        return sample


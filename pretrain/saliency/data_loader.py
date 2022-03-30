# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF

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

def _build_size(orig_img, width, height):
    size = [width, height]
    if size[0] == -1: size[0] = orig_img.width
    if size[1] == -1: size[1] = orig_img.height
    return size

def pil_loader(_path, width, height, is_segmentation=False):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(_path, 'rb') as f:
        with Image.open(f) as _img:
            if is_segmentation:
                _img = _img.resize(_build_size(_img, width, height), Image.NEAREST)
            else:
                _img = _img.convert('RGB')
    return _img


class cityscapesDataset(Dataset):
    def __init__(
        self,
        image_path,
        label_path=None,
        split="train",
        n_samples= -1,        # Select only few samples for training
        size="tiny",
		transform=None
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.split = split
        if size == "small":
            self.img_size = (1024, 512) # w, h -- PIL uses (w, h) format
        elif size == "tiny":
            self.img_size = (512, 256)
        else:
            raise Exception('size not valid')

        self.n_samples = n_samples
        self.n_classes = 19
        self.files = {}
        self.transforms = transform
        self.images_base = os.path.join(self.image_path, self.split)
        if self.label_path is None:
            self.annotations_base = None
        else:
            self.annotations_base = os.path.join(self.label_path, self.split)


        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

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
        return os.path.join(img_path[-2], img_path[-1].rstrip('.jpg'))

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        # Image
        img_path = self.files[self.split][index].rstrip()
        img = pil_loader(img_path, self.img_size[0], self.img_size[1])
        img = self.transforms(img)

        # Label
        if self.label_path is None:
            lbl = np.zeros((512, 256))
        else:
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png")
            lbl = pil_loader(lbl_path, self.img_size[0], self.img_size[1], is_segmentation=True)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        sample = {'image': img, 'label': lbl, 'index': index}
        return sample

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        
        # sanity checks
        lbl = mask
        classes = np.unique(lbl)
        lbl = lbl.astype(int)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        lbl = torch.from_numpy(lbl).long()
        return lbl


class gtaDataset(Dataset):
    def __init__(
        self,
        image_path,
        label_path=None,
        split="train",
        n_samples= -1,        # Select only few samples for training
        size="tiny",
		transform=None
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.split = split
        if size == "small":
            self.img_size = (1280, 720) 
        elif size == "tiny":
            self.img_size = (640, 360)  # w, h -- PIL uses (w, h) format
            self.crop_size = (256, 512)
        else:
            raise Exception('size not valid')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.n_samples = n_samples
        self.n_classes = 19
        self.files = {}
        self.transforms = transform
        self.images_base = self.image_path
        self.annotations_base = self.label_path

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
        return img_path[-1].rstrip('.jpg')

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
            
        # Image
        img = pil_loader(img_path, self.img_size[0], self.img_size[1])
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, self.crop_size)
        img = TF.crop(img, i, j, h, w)
        img = self.transforms(img)

        if self.label_path is None:
            lbl = np.zeros(self.img_size)
        else:
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-1][:-4] + ".png")
            lbl = pil_loader(lbl_path, self.img_size[0], self.img_size[1], is_segmentation=True)
            lbl = TF.crop(lbl, i, j, h, w)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        sample = {'image': img, 'label': lbl, 'index': index}
        return sample

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        
        # sanity checks
        lbl = mask
        classes = np.unique(lbl)
        lbl = lbl.astype(int)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")
        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        lbl = torch.from_numpy(lbl).long()
        return lbl
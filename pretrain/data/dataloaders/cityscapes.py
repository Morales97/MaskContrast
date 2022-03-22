

import os
import sys
import errno
import hashlib
import glob
import tarfile
import numpy as np
import torch.utils.data as data
import pdb

# from data.util.mypath import Path
from PIL import Image

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

class Cityscapes(data.Dataset):

    def __init__(self, root='/home/danmoral/MaskContrast/pretrain/data/cityscapes',     #TODO change to use data.util.mypath as in VOCSegmentation
                 saliency='saliency_basenet_tiny', split='leftImg8bit_tiny/train', n_samples=-1,
                 transform=None, overfit=False):
        super(Cityscapes, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split

        self.images_dir = os.path.join(self.root, self.split)
        valid_saliency = ['saliency_basenet_tiny']
        assert(saliency in valid_saliency)
        self.saliency = saliency
        self.sal_dir = os.path.join(self.root, self.saliency)
    
        self.images = []
        self.sal = []

        self.n_samples = n_samples
        self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".jpg"))
        if self.n_samples >= 0:
            self.files = self.files[:self.n_samples]
    
        if not self.files:
            raise Exception("No files found in %s" % self.images_dir)
        print("Found %d %s images" % (len(self.files), split))

        pdb.set_trace()
        for f in self.files:
            _image = os.path.join(self.images_dir, f + ".jpg")
            _sal = os.path.join(self.sal_dir, f + ".png")
            if os.path.isfile(_image) and os.path.isfile(_sal):
                self.images.append(_image)
                self.sal.append(_sal)

        assert (len(self.images) == len(self.sal))

        if overfit:
            n_of = 32
            self.images = self.images[:n_of]
            self.sal = self.sal[:n_of]

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        sample['image'] = self._load_img(index)
        sample['sal'] = self._load_sal(index)

        if self.transform is not None:
            sample = self.transform(sample)
        
        sample['meta'] = {'image': str(self.images[index])}

        return sample

    def __len__(self):
            return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        return _img

    def _load_sal(self, index):
        _sal = Image.open(self.sal[index])
        return _sal

    def __str__(self):
        return 'VOCSegmentation(saliency=' + self.saliency + ')'

    def get_class_names(self):
        # Class names for sal
        return ['background', 'salient object']
    
        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(Path.db_root_dir())
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')


if __name__ == '__main__':
    """ For purpose of debugging """
    dataset = Cityscapes()
    '''
    from matplotlib import pyplot as plt
    # Sample from supervised saliency model
    dataset = VOCSegmentation(saliency='supervised_model')
    sample = dataset.__getitem__(5)
    fig, axes = plt.subplots(2)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['sal'])
    plt.show()
    plt.close()

    # Sample from unsupervised saliency model
    dataset = VOCSegmentation(saliency='unsupervised_model')
    sample = dataset.__getitem__(5)
    fig, axes = plt.subplots(2)
    axes[0].imshow(sample['image'])
    axes[1].imshow(sample['sal'])
    plt.show()
    plt.close()
    '''


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

def recursive_find_masks(rootdir=".", rootname=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.startswith(rootname)
    ]

class Gta(data.Dataset):

    def __init__(self, root='/home/danmoral/MaskContrast/pretrain/data/gta5',     #TODO change to use data.util.mypath as in VOCSegmentation
                 saliency='saliency_basnet_cropped', split='images_tiny_cropped', n_samples=-1,
                 transform=None, overfit=False, use_gt_masks=False):
        super(Gta, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split
        self.use_gt_masks = use_gt_masks

        self.images_dir = os.path.join(self.root, self.split)
        valid_saliency = ['saliency_basnet_cropped', 'saliency_mined_masks_cropped']
        assert(saliency in valid_saliency)
        self.saliency = saliency
        self.sal_dir = os.path.join(self.root, self.saliency)
    
        self.images = []
        self.sal = []

        self.n_samples = n_samples
        self.files = sorted(recursive_glob(rootdir=self.images_dir, suffix=".jpg"))
        if self.n_samples >= 0:
            self.files = self.files[:self.n_samples]
    
        if not self.files:
            raise Exception("No files found in %s" % self.images_dir)

        if not self.use_gt_masks:
            # Load image and corresponding estimated saliency mask
            for img_path in self.files:
                sal_name = img_path.split(os.sep)[-1].rstrip('.jpg') + '.png'
                sal_path = os.path.join(self.sal_dir, sal_name)
                if os.path.isfile(sal_path):
                    self.images.append(img_path)
                    self.sal.append(sal_path)
            print("Found %d images with saliency map, out of %d total images" % (len(self.images), len(self.files)))
        else:
            # Load image and all corresponding masks from the ground truth
            i = 0
            for img_path in self.files:
                sal_name = img_path.split(os.sep)[-1].rstrip('.jpg')
                masks = sorted(recursive_find_masks(self.sal_dir, sal_name))
                if len(masks) > 0:
                    i += 1
                    self.images = self.images + ([img_path] * len(masks)) # add the images once for each mask
                    self.sal = self.sal + masks
            print("Found %d images with %d ground-truth object masks" % (i, len(self.sal)))

        assert (len(self.images) == len(self.sal))

        if overfit:
            n_of = 32
            self.images = self.images[:n_of]
            self.sal = self.sal[:n_of]


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

import torchvision
class RandomResizedCrop(torchvision.transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        super(RandomResizedCrop, self).__init__(size, scale=scale, ratio=ratio)
        self.interpolation_img = Image.BILINEAR
        self.interpolation_sal = Image.NEAREST
    
    def __call__(self, sample):
        img = sample['image']
        sal = sample['sal']

        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        import torchvision.transforms.functional as F

        sample['image'] = F.resized_crop(img, i, j, h, w, self.size, self.interpolation_img)
        sample['sal'] = F.resized_crop(sal, i, j, h, w, self.size, self.interpolation_sal)
        return sample

if __name__ == '__main__':
    """ For purpose of debugging """
    dataset = Gta()
    sample = dataset.__getitem__(0)
    sample['image'].save('/home/danmoral/test0.jpg')
    sample['sal'].save('/home/danmoral/test0sal.png')
    dataset = Gta(transform=RandomResizedCrop(224, scale=(0.2, 1.)))
    sample = dataset.__getitem__(0)
    sample['image'].save('/home/danmoral/test0_.jpg')
    sample['sal'].save('/home/danmoral/test0sal_.png')
    
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
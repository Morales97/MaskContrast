

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

class Cityscapes(data.Dataset):

    def __init__(self, root='/home/danmoral/MaskContrast/pretrain/data/cityscapes',     #TODO change to use data.util.mypath as in VOCSegmentation
                 saliency='saliency_basnet_small', split='leftImg8bit_small/train', n_samples=-1,
                 transform=None, overfit=False):
        super(Cityscapes, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split

        self.images_dir = os.path.join(self.root, self.split)
        valid_saliency = ['saliency_basnet_tiny', 'saliency_basnet_small']
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

        for img_path in self.files:
            city = img_path.split(os.sep)[-2]
            sal_name = img_path.split(os.sep)[-1].rstrip('.jpg') + '.png'
            sal_path = os.path.join(self.sal_dir, city, sal_name)
            if os.path.isfile(sal_path):
                self.images.append(img_path)
                self.sal.append(sal_path)
        print("Found %d images with saliency map, out of %d total images" % (len(self.images), len(self.files)))

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



class Cityscapes_Mix(data.Dataset):
    '''
    Mix masks obtained from saliency estimation with masks obtained from ground truth segmentation labels
    '''

    def __init__(self, root='/home/danmoral/MaskContrast/pretrain/data/cityscapes',     #TODO change to use data.util.mypath as in VOCSegmentation
                 saliency='saliency_basnet_tiny', saliency_gt='saliency_mined_masks', split='leftImg8bit_tiny/train', n_samples_lbld=-1, sample_idxs_lbl=None, sample_idxs_unlbl=None,
                 transform=None, overfit=False, load_unsup=True):
        super(Cityscapes_Mix, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split

        self.images_dir = os.path.join(self.root, self.split)
        valid_saliency = ['saliency_basnet_tiny', 'saliency_mined_masks', 'saliency_mined_masks_all_classes', 'saliency_mined_masks_100_seed1', 'saliency_mined_masks_100_seed2', 'saliency_mined_masks_100_seed3']
        assert(saliency in valid_saliency) and (saliency_gt in valid_saliency)
        self.saliency = saliency
        self.sal_dir = os.path.join(self.root, self.saliency)
        self.masks_sup_dir = os.path.join(self.root, saliency_gt)
    
        self.images = []
        self.sal = []

        self.n_samples = n_samples_lbld
        self.files = sorted(recursive_glob(rootdir=self.images_dir, suffix=".jpg"))
        if not self.files:
            raise Exception("No files found in %s" % self.images_dir)
        
        if sample_idxs_lbl is not None:
            assert sample_idxs_unlbl is not None
            files = np.array(self.files)
            self.files_gt = files[sample_idxs_lbl].tolist() 
            self.files_est = files[sample_idxs_unlbl].tolist() 
        else:
            self.files_gt = self.files[:self.n_samples]
            self.files_est = self.files[self.n_samples:]

        i = 0
        for img_path in self.files_gt:
            city = img_path.split(os.sep)[-2]
            sal_name = img_path.split(os.sep)[-1].rstrip('.jpg')
            masks = sorted(recursive_find_masks(os.path.join(self.masks_sup_dir, city), sal_name))
            if len(masks) > 0:
                i += 1
                self.images = self.images + ([img_path] * len(masks)) # add the images once for each mask
                self.sal = self.sal + masks
        print("Step 1. Found %d images with %d ground-truth object masks" % (i, len(self.sal)))

        i = 0
        if load_unsup:
            for img_path in self.files_est:
                city = img_path.split(os.sep)[-2]
                sal_name = img_path.split(os.sep)[-1].rstrip('.jpg') + '.png'
                sal_path = os.path.join(self.sal_dir, city, sal_name)
                if os.path.isfile(sal_path):
                    i += 1
                    self.images.append(img_path)
                    self.sal.append(sal_path)
            print("Step 2. Found %d images with an estimated object mask, out of %d total images" % (i, len(self.files_est)))

        assert len(self.images) == len(self.sal)

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
    dataset = Cityscapes()
    sample = dataset.__getitem__(0)
    sample['image'].save('/home/danmoral/test0.jpg')
    sample['sal'].save('/home/danmoral/test0sal.png')
    dataset = Cityscapes(transform=RandomResizedCrop(224, scale=(0.2, 1.)))
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
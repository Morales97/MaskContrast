

import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
from PIL import Image
import glob
from tqdm import tqdm
from data_loader import cityscapesDataset, gtaDataset
import cv2
import numpy as np
from copy import deepcopy

from model import BASNet

def save_output(save_dir, save_name, mask):
	os.makedirs(save_dir, exist_ok=True)

	im = Image.fromarray(mask*255).convert('RGB')

	save_path = os.path.join(save_dir, save_name) + '.png'
	im.save(save_path)

def save_img(save_dir, save_name, img):
	os.makedirs(save_dir, exist_ok=True)

	save_path = os.path.join(save_dir, save_name) + '.jpg'
	torchvision.utils.save_image(img, save_path)

def postprocess(model_output: np.array, area_th=0.01) -> np.array:
	"""
	adapted from https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation/tree/main/saliency 
	We postprocess the predicted saliency mask to remove very small segments. 
	If the mask is too small overall, we skip the image.

	Args:
		model_output: The predicted saliency mask scaled between 0 and 1. 
					  Shape is (height, width). 
	Return:
			result: The postprocessed saliency mask.
	"""
	model_output = model_output.squeeze().cpu().data.numpy()
	mask = (model_output > 0.5).astype(np.uint8)
	contours, _ = cv2.findContours(deepcopy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	# Throw out small segments
	for contour in contours:
		segment_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
		segment_mask = cv2.drawContours(segment_mask, [contour], 0, 255, thickness=cv2.FILLED)
		area = (np.sum(segment_mask) / 255.0) / np.prod(segment_mask.shape)
		if area < area_th:
			mask[segment_mask == 255] = 0

	# If area of mask is too small, return None
	if np.sum(mask) / np.prod(mask.shape) < 0.01:
		return None

	return mask

if __name__ == '__main__':

	_gta = False
	_cityscapes = True
	ignore_index = 250
	top_k = 5 # -1
	seed = 3
	n_samples = 100

	# --------- 1. get image path and name ---------
	
	if _cityscapes:
		image_dir = '../data/cityscapes/leftImg8bit_tiny/'
		label_dir = '../data/cityscapes/gtFine'
		save_dir = '../data/cityscapes/saliency_mined_masks_100_seed3/'
	elif _gta:
		image_dir = '../data/gta5/images_tiny/'
		label_dir = '../data/gta5/labels/'
		img_save_dir = '../data/gta5/images_tiny_cropped_sup/'
		save_dir = '../data/gta5/saliency_mined_masks_cropped/'
	else:
		raise Exception('no dataset specified')
	model_dir = './basnet.pth'
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# --------- 2. dataloader ---------
	#1. dataload

	cs_samples = 2975
	np.random.seed(seed)

	idxs = np.arange(cs_samples)
	idxs = np.random.permutation(idxs)
	idxs_lbl = idxs[:n_samples]	# checked, this selects the same samples as in 'ssda' project train

	transform = transforms.Compose([transforms.ToTensor()])
	if _cityscapes:
		dataset = cityscapesDataset(image_path=image_dir, label_path=label_dir, transform=transform, sample_idxs=idxs_lbl)
	elif _gta:
		dataset = gtaDataset(image_path=image_dir, label_path=label_dir, transform=transform, n_samples=-1)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
	
	# --------- 3. mine object masks from segmentation labels ---------
	for data in tqdm(dataloader):
	
		lbl = data['label']

		lbl_class, lbl_count = lbl.unique(return_counts=True)   
		if top_k == -1:
			top_count, top_idxs = lbl_count.topk(len(lbl_class))   # get top k class counts and idx
		else:
			top_count, top_idxs = lbl_count.topk(min(top_k, len(lbl_class)))   # get top k class counts and idx
		top_classes = lbl_class[top_idxs]            # get class from idx

		masks = []
		for top_class in top_classes:
			if top_class != ignore_index:
				#mask = (lbl == top_class) * top_class   # generates mask with index of the class
				mask = (lbl == top_class) * 1            # generates mask with 0 and 1
				mask = postprocess(mask, area_th=0)		 # area_th set at 0 to not throw away small segments
				if mask is None:
					continue
				masks.append(mask)

		# save
		name = dataset.get_img_save_path(data['index'])
		if _cityscapes:
			city, name = name.split(os.sep)
			save_path = os.path.join(save_dir, city)
		if _gta:
			image = data['image'].type(torch.FloatTensor).to(device)
			save_img(img_save_dir, name, image)
			save_path = save_dir
		
		for i, mask in enumerate(masks):
			save_output(save_path, name + '_' + str(i), mask)

	

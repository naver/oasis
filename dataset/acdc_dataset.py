import os
import numpy as np
import random
import collections
import glob

import torch
import torchvision
from torch.utils import data

from PIL import Image
import cv2

class ACDC(data.Dataset):
	def __init__(self, root, scene_list, cond_list, crop_size=(321, 321),
			mean=(128, 128, 128), wct2_random_style_transfer=False,
			wct2_nn_style_transfer=False):
		"""
			params

				root : str
					Path to the data folder
		"""

		self.class_conversion_dict = {
				7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
				23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}

		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.scene_list = scene_list
		self.cond_list = cond_list

		self.files = []
		self.label_files = []

		assert not (wct2_random_style_transfer and wct2_nn_style_transfer)

		# NOTE if using style transfer, we assume images have been transformed 
		# before. Of course, in a practical application this will happen online.
		if wct2_random_style_transfer:
			self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
		elif wct2_nn_style_transfer:
			self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
		else:
			self.images_root = self.root

		for cond in cond_list:
			if cond not in ['clean', 'fog', 'night', 'rain', 'snow']:
				raise ValueError(
						'Unknown conditions [supported are clean, fog, night, rain, snow]')

		assert len(cond_list) == len(scene_list)

		self.num_imgs_per_seq = []

		for scene, cond in zip(self.scene_list, self.cond_list):

			self.img_paths = glob.glob(os.path.join(
					self.images_root, 'rgb_anon_trainvaltest/rgb_anon',
					cond, 'train', scene, '*.png'))
			self.img_paths += glob.glob(os.path.join(
					self.images_root, 'rgb_anon_trainvaltest/rgb_anon',
					cond, 'val', scene, '*.png'))
			self.img_paths = sorted(self.img_paths)

			self.label_img_paths = [os.path.join(self.root, 'gt_trainval/gt', cond,
										path.split('/')[-3], scene,
										path.split('/')[-1].rstrip('_rgb_anon.png')+'_gt_labelIds.png')
												for path in self.img_paths]

			print(f'{scene}/{cond}: {len(self.img_paths)},')

			self.num_imgs_per_seq.append(len(self.img_paths))

			for img_path in sorted(self.img_paths):
				name = img_path.split('/')[-1]
				self.files.append({
					'img': img_path, # used path
					'name': name # just the end of the path
				})

			for label_img_path in sorted(self.label_img_paths):
				name = label_img_path.split('/')[-1]
				self.label_files.append({
					'label_img': label_img_path, # used path
					'label_name': name # just the end of the path
				})


	def __len__(self):
		return len(self.files)


	def __getitem__(self, index):

		image = Image.open(self.files[index]['img']).convert('RGB')
		name = self.files[index]['name']

		label = Image.open(self.label_files[index]['label_img'])
		label_name = self.label_files[index]['label_name']

		# resize
		image = image.resize(self.crop_size, Image.BICUBIC)
		image = np.asarray(image, np.float32)

		label = cv2.resize(np.array(label), self.crop_size, interpolation=0)

		# re-assign labels to filter out non-used ones
		label_copy = 255 * np.ones(label.shape, dtype=np.float32)
		for k, v in self.class_conversion_dict.items():
			label_copy[label == k] = v

		size = image.shape
		image = image[:, :, ::-1]  # change to BGR
		image -= self.mean
		image = image.transpose((2, 0, 1))

		return image.copy(), label_copy.copy(), np.array(size), name

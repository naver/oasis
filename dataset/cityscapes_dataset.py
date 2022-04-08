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

class Cityscapes(data.Dataset):
	def __init__(self, root, city_list, cond_list, crop_size=(321, 321),
				mean=(128, 128, 128), set='val', alpha=0.02, beta=0.01,
				dropsize=0.005, pattern=3, wct2_random_style_transfer=False,
				wct2_nn_style_transfer=False):
		"""
			params

				root : str
					Path to the data folder'

		"""

		self.class_conversion_dict = {
				7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7, 21:8, 22:9,
				23: 10, 24:11, 25:12, 26:13, 27:14, 28:15, 31:16, 32:17, 33:18}

		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.city_list = city_list
		self.cond_list = cond_list
		self.alpha = alpha
		self.beta = beta
		self.dropsize = dropsize
		self.pattern = pattern

		self.files = []
		self.label_files = []

		for cond in cond_list:
			if cond not in ['clean', 'fog', 'rain']:
				raise ValueError(
						'Unknown conditions [supported are clean, rain, fog]')

		assert len(cond_list) == len(city_list)

		self.num_imgs_per_seq = []

		assert not (wct2_random_style_transfer and wct2_nn_style_transfer)

		# NOTE if using style transfer, we assume images have been transformed 
		# before. Of course, in a practical application this will happen online.
		if wct2_random_style_transfer:
			self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
		elif wct2_nn_style_transfer:
			self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
		else:
			self.images_root = self.root

		for city, cond in zip(self.city_list, self.cond_list):
			if city in ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']:
				self.set = 'test'
			elif city in ['frankfurt', 'lindau', 'munster']:
				self.set = 'val'
			else:
				self.set = 'train'

			# Path to the txt containing the relative paths
			# (with respect to root) of the images/labels to load
			list_of_images_file = f'./dataset/cityscapes_list/images_{city}.txt'
			list_of_label_images_file = f'./dataset/cityscapes_list/labels_{city}.txt'

			self.img_names = [i_id.strip() for i_id in open(list_of_images_file)]
			if cond == 'clean':
				pass
			elif cond == 'fog':
				self.img_names = [i_id.rstrip('.png')+f'_foggy_beta_0.02.png' for i_id in self.img_names]
			elif cond == 'rain':
				self.img_names = sorted(glob.glob(os.path.join(
						self.images_root,
						f'leftImg8bit_rain/{self.set}/{city}',
						f'*_alpha_{self.alpha}_beta_{self.beta}_dropsize_{self.dropsize}_pattern_{self.pattern}.png')))
			else:
				raise ValueError('Unknown conditions [supported are clean,rain,fog]')

			self.label_img_names = [i_id.strip() for i_id in open(list_of_label_images_file)]

			if cond == 'rain':
				img_names_ = [_.split(f'/{self.set}/')[1].split('_leftImg8bit_')[0] for _ in self.img_names]
				self.label_img_names = [_ for _ in self.label_img_names if _.rstrip('_gtFine_labelIds.png') in img_names_]

			print(f'\'{city}\': {len(self.img_names)},')

			self.num_imgs_per_seq.append(len(self.img_names))

			for name in sorted(self.img_names):
				if cond == 'clean':
					img_path = os.path.join(self.images_root, f'leftImg8bit/{self.set}/{name}')
				elif cond == 'fog':
					img_path = os.path.join(self.images_root, f'leftImg8bit_foggyDBF/{self.set}/{name}')
				elif cond == 'rain':
					img_path, name = name, name.split('/')[-1]
				else:
					raise ValueError('Unknown conditions [supported are clean,rain,fog]')

				self.files.append({
					'img': img_path, # used path
					'name': name # just the end of the path
				})

			for name in sorted(self.label_img_names):
				img_path = os.path.join(self.root, f'gtFine/{self.set}/{name}')
				self.label_files.append({
					'label_img': img_path, # used path
					'label_name': name # just the end of the path
				})


	def __len__(self):
		return len(self.files)


	def __getitem__(self, index):

		image = Image.open(self.files[index]['img']).convert('RGB')
		name = self.files[index]['name']

		label = Image.open(self.label_files[index]['label_img'])#.convert('RGB')
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

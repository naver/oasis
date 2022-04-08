import os
import numpy as np
import numpy.random as npr
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import copy


class GTA5(data.Dataset):
	def __init__(self, root, num_epochs=None, crop_size=(321, 321), mean=(128, 128, 128)):
		self.root = root
		self.list_path = os.path.join(root, 'gta5_list/train.txt')
		self.crop_size = crop_size
		self.mean = mean
		self.img_names_ = [i_id.strip() for i_id in open(self.list_path)]#[:10]
		self.img_names = []
		if num_epochs is not None:
			img_names_ = copy.deepcopy(self.img_names_)
			for _ in range(num_epochs):
				npr.shuffle(img_names_)
				self.img_names += img_names_

		self.files = []
		self.class_conversion_dict = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
							19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
							26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

		for name in self.img_names:
			img_path = os.path.join(self.root, f'images/images/{name}')
			label_path = os.path.join(self.root, f'labels/labels/{name}')
			self.files.append({
				'img': img_path,
				'label': label_path,
				'name': name
			})

		self.img_files = [_['img'] for _ in self.files]


	def __len__(self):
		return len(self.files)


	def __getitem__(self, index):

		image = Image.open(self.files[index]['img']).convert('RGB')
		label = Image.open(self.files[index]['label'])
		name = self.files[index]['name']

		# resize
		image = image.resize(self.crop_size, Image.BICUBIC)
		label = label.resize(self.crop_size, Image.NEAREST)

		image = np.asarray(image, np.float32)
		label = np.asarray(label, np.float32)

		# re-assign labels to match the format of Cityscapes
		label_copy = 255 * np.ones(label.shape, dtype=np.float32)
		for k, v in self.class_conversion_dict.items():
			label_copy[label == k] = v

		size = image.shape
		image = image[:, :, ::-1]  # change to BGR
		image -= self.mean
		image = image.transpose((2, 0, 1))

		return image.copy(), label_copy.copy(), np.array(size), name

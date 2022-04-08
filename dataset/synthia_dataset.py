import os
import glob
import numpy as np
import numpy.random as npr
import random
import collections
import torch
import torchvision
from torch.utils import data

from PIL import Image
import imageio
import cv2


class SYNTHIA(data.Dataset):
	def __init__(self, root, scene_list, cond_list, camera_id=0,
				crop_size=(321, 321), mean=(128, 128, 128), set='train',
				num_images=300, wct2_random_style_transfer=False,
				wct2_nn_style_transfer=False):
		"""
			params

				root : str
					Path to the data folder.

				scene : str
					Defines which SYNTHIA scenes to use
					Format: X_Y_Z, with X/Y/Z \in {01,02,04,05,06}

				cond : str
					Defines which weather/daylight condition to use
					Format: X_Y_Z, with X/Y/Z \in
					{DAWN, FOG, NIGHT, SPRING, SUMMER, WINTER, SUNSET}
		"""

		self.CLASSES = [
				'void', 'sky', 'building', 'road', 'sidewalk', 'fence',
				'vegetation', 'pole', 'car', 'traffic_sign', 'pedestrian',
				'bicycle', 'lanemarking', 'None', 'None', 'traffic_light']

		self.CLASSES_dict = {
				0: 'void', 1:'sky', 2:'building', 3:'road', 4:'sidewalk',
				5:'fence', 6:'vegetation', 7:'pole', 8:'car', 9:'traffic_sign',
				10:'pedestrian', 11:'bicycle', 12:'lanemarking', 13:'None',
				14:'None', 15:'traffic_light'}

		self.LABELS_RGB_VALUES = [
				'0_0_0', '128_128_128', '128_0_0', '128_64_128',
				'0_0_192', '64_64_128', '128_128_0', '192_192_128',
				'64_0_128', '192_128_128', '64_64_0', '0_128_192',
				'0_172_0', 'None' ,'None', '0_128_128']

		self.id_to_trainid = {
				1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13, 9:7, 10:11, 11:18, 15:6}

		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.files = []
		self.label_files = []

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

		for n, (scene_, cond_) in enumerate(zip(scene_list, cond_list)):
			img_regex = os.path.join(self.images_root, f'SYNTHIA-SEQS-{scene_}-{cond_}', 'RGB/Stereo_Left/Omni_F/*')
			label_regex = os.path.join(self.root, f'SYNTHIA-SEQS-{scene_}-{cond_}', 'GT/LABELS/Stereo_Left/Omni_F/*')

			img_files = sorted(glob.glob(img_regex))
			label_files = sorted(glob.glob(label_regex))


			npr.seed(len(img_files))
			n = npr.randint(0,len(img_files)-num_images)
			print(f'Random seed: {len(img_files)}, sampled value: {n}')
			img_files = img_files[n:n+num_images]
			label_files = label_files[n:n+num_images]


			if len(img_files) == 0:
				raise RuntimeError(f'No images in {img_regex}')

			print(f'Loaded {len(img_files)} from {img_regex}')

			self.num_imgs_per_seq.append(len(img_files))

			for img_file in img_files:
				self.files.append({
					"img": img_file, # used path
					"name": f'{n}_'+img_file.split('/')[-1] # just the end of the path
				})

			for label_file in label_files:
				self.label_files.append({
					"label_img": label_file, # used path
					"name": f'{n}_'+label_file.split('/')[-1] # just the end of the path
				})

		self.img_files = [_['img'] for _ in self.files]


	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]
		labelfiles = self.label_files[index]

		image = Image.open(datafiles["img"]).convert('RGB')
		name = datafiles["name"]


		label_image = np.asarray(imageio.imread(labelfiles["label_img"], format='PNG-FI'))[:,:,0]  # uint16
		label_name = labelfiles["name"]

		# resize
		image = image.resize(self.crop_size, Image.BICUBIC)

		# be careful not interpolating the label img, otherwise wrong labels
		label_image = cv2.resize(np.array(label_image), self.crop_size, interpolation=0)

		# re-assign labels to match the format of Cityscapes
		label_copy = 255 * np.ones(label_image.shape, dtype=np.float32)

		for k, v in self.id_to_trainid.items():
			label_copy[label_image == k] = v

		image = np.asarray(image, np.float32)
		size = image.shape
		image = image[:, :, ::-1]  # change to BGR
		image -= self.mean
		image = image.transpose((2, 0, 1))

		return image.copy(), label_copy.copy(), np.array(size), name

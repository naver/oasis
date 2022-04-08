"""
Note: the GitHub repo https://github.com/wasidennis/AdaptSegNet was used
as a starting point to develop the code contained in this file.
"""

import numpy as np
import argparse
import json
from PIL import Image
import imageio
from os.path import join
from sklearn import preprocessing

SYNTHIA_TO_CITYSCAPES_MAPPING = {1:10, 2:2, 3:0, 4:1, 5:4, 6:8, 7:5, 8:13, 9:7, 10:11, 11:18, 15:6}

def fast_hist(a_, b_, n_):
	k_ = (a_ >= 0) & (a_ < n_)
	return np.bincount(n_ * a_[k_].astype(int) + b_[k_], minlength=n_ ** 2).reshape(n_, n_)


def per_class_iu(hist):
	return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
	output = np.copy(input)
	for ind in range(len(mapping)):
		output[input == mapping[ind][0]] = mapping[ind][1]
	return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir='./dataset/cityscapes_list'):
	"""
	Compute IoU given the predicted colorized images and
	"""
	with open(join(devkit_dir, 'info.json'), 'r') as fp:
		info = json.load(fp)
	num_classes = np.int(info['classes'])
	print(f'Num classes {num_classes}')
	name_classes = np.array(info['label'], dtype=np.str)
	mapping = np.array(info['label2train'], dtype=np.int)
	hist = np.zeros((num_classes, num_classes))

	image_path_list = join(devkit_dir, 'val.txt')
	label_path_list = join(devkit_dir, 'label.txt')
	gt_imgs = open(label_path_list, 'r').read().splitlines()
	gt_imgs = [join(gt_dir, x) for x in gt_imgs]
	pred_imgs = open(image_path_list, 'r').read().splitlines()
	pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

	for ind in range(len(gt_imgs)):
		pred_image = Image.open(pred_imgs[ind])
		pred = np.array(pred_image)
		label_image = Image.open(gt_imgs[ind])
		label = np.array(label_image)
		label = label_mapping(label, mapping)
		if len(label.flatten()) != len(pred.flatten()):
			print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}')
			continue
		hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
		if ind > 0 and ind % 10 == 0:
			print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))

	mIoUs = per_class_iu(hist)
	for ind_class in range(num_classes):
		print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
	print(f'===> mIoU: {round(np.nanmean(mIoUs) * 100, 2)}')
	return mIoUs


def compute_acc_single_image(label, pred):
	"""
	Compute pixel and mean accuracy given a predicted colorized image and the GT
	(also GT in color format, not label format)
	"""

	assert len(label.flatten()) == len(pred.flatten())

	correct_pred = (label[label!=255] == pred[label!=255])
	pixel_acc = 100. * np.sum(correct_pred)/len(correct_pred)

	image_labels = np.unique(label)
	mean_acc = 0.
	for cl in image_labels:
		tmp_correct_pred = (label[label==cl] == pred[label==cl])
		mean_acc += (1./len(image_labels)) * np.sum(tmp_correct_pred)/len(tmp_correct_pred)

	return pixel_acc, mean_acc


def compute_mIoU_single_image(label, pred, num_classes, name_classes):
	"""
	Compute IoU given a predicted colorized image and the GT
	(also GT in color format, not label format)
	"""

	if len(label.flatten()) != len(pred.flatten()):
		print('Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(len(label.flatten()), len(pred.flatten())))
		return -1

	hist = fast_hist(label.flatten(), pred.flatten(), num_classes)
	sample_mIoU = 100*np.nanmean(per_class_iu(hist))

	mIoUs = per_class_iu(hist)

	return mIoUs


def compute_acc_fromlist(gt_imgs, pred_imgs, trg_dataset, devkit_dir='./dataset/cityscapes_list'):
	"""
	Compute pixel and mean accuracy given the predicted colorized images and
	"""

	assert len(gt_imgs) == len(pred_imgs)

	with open(join(devkit_dir, 'info.json'), 'r') as fp:
		info = json.load(fp)
	num_classes = np.int(info['classes'])
	print(f'Num classes {num_classes}')
	name_classes = np.array(info['label'], dtype=np.str)
	mapping = np.array(info['label2train'], dtype=np.int)

	gt_imgs = sorted(gt_imgs)
	pred_imgs = sorted(pred_imgs)

	gt_imgs_list = []
	pred_imgs_list = []

	le = preprocessing.LabelEncoder()

	for ind in range(len(gt_imgs)):
		pred_image = Image.open(pred_imgs[ind])
		pred = np.array(pred_image)

		if ('Cityscapes' in trg_dataset) or ('ACDC' in trg_dataset):
			label_image = Image.open(gt_imgs[ind])
			label = np.array(label_image)
			label = label_mapping(label, mapping)

		elif 'SYNTHIA' in trg_dataset:
			label_image = np.asarray(imageio.imread(gt_imgs[ind], format='PNG-FI'))[:,:,0]	# uint16
			label = 255 * np.ones(label_image.shape, dtype=np.float32)
			for k, v in SYNTHIA_TO_CITYSCAPES_MAPPING.items():
				label[label_image == k] = v

		else:
			raise NotImplementedError("Unknown target dataset")

		if len(label.flatten()) != len(pred.flatten()):
			print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}')
			continue

		pred_imgs_list.append(pred)
		gt_imgs_list.append(label)

	pred_imgs_stack = np.stack(pred_imgs_list)
	gt_imgs_stack = np.stack(gt_imgs_list)

	# only consider classes available in GT
	# for evaluation (not an issue for Cityscapes)
	gt_classes = np.unique(gt_imgs_stack)
	gt_classes = gt_classes[gt_classes!=255]
	le.fit(gt_classes)
	num_classes = len(gt_classes)

	pixel_acc = 0.
	mean_acc = 0.

	num_images = len(pred_imgs_list)

	for ind, (pred, label) in enumerate(zip(pred_imgs_stack, gt_imgs_stack)):
		label = label.flatten()
		pred = pred.flatten()

		if trg_dataset == 'SYNTHIA':
			idx = np.any((pred == gt_classes[:,None]),0)
			pred = pred[idx]
			label = label[idx]

			idx = np.any((label == gt_classes[:,None]),0)
			pred = pred[idx]
			label = label[idx]

			pred = le.transform(pred)
			label = le.transform(label)

		pixel_acc_tmp, mean_acc_tmp = compute_acc_single_image(label, pred)
		pixel_acc += (1./num_images) * pixel_acc_tmp
		mean_acc += (1./num_images) * mean_acc_tmp

	return pixel_acc, mean_acc



def compute_mIoU_fromlist(gt_imgs, pred_imgs, trg_dataset, devkit_dir='./dataset/cityscapes_list'):
	"""
	Compute IoU given the predicted colorized images and
	"""
	with open(join(devkit_dir, 'info.json'), 'r') as fp:
		info = json.load(fp)
	num_classes = np.int(info['classes'])
	print(f'Num classes {num_classes}')
	name_classes = np.array(info['label'], dtype=np.str)
	mapping = np.array(info['label2train'], dtype=np.int)

	gt_imgs = sorted(gt_imgs)
	pred_imgs = sorted(pred_imgs)

	gt_imgs_list = []
	pred_imgs_list = []

	le = preprocessing.LabelEncoder()

	for ind in range(len(gt_imgs)):
		pred_image = Image.open(pred_imgs[ind])
		pred = np.array(pred_image) # pred: (1024, 2048)

		if ('Cityscapes' in trg_dataset) or ('ACDC' in trg_dataset):
			label_image = Image.open(gt_imgs[ind])
			label = np.array(label_image) # label: (1024, 2048)
			label = label_mapping(label, mapping)

		elif 'SYNTHIA' in trg_dataset:
			label_image = np.asarray(imageio.imread(gt_imgs[ind], format='PNG-FI'))[:,:,0]	# uint16
			label = 255 * np.ones(label_image.shape, dtype=np.float32)
			for k, v in SYNTHIA_TO_CITYSCAPES_MAPPING.items():
				label[label_image == k] = v

		else:
			raise NotImplementedError("Unknown target dataset")

		if len(label.flatten()) != len(pred.flatten()):
			print(f'Skipping: len(gt) = {len(label.flatten())}, len(pred) = {len(pred.flatten())}, {gt_imgs[ind]}, {pred_imgs[ind]}')
			continue

		pred_imgs_list.append(pred)
		gt_imgs_list.append(label)

	pred_imgs_stack = np.stack(pred_imgs_list)
	gt_imgs_stack = np.stack(gt_imgs_list)

	# only consider classes available in GT for evaluation (not an issue for Cityscapes)
	gt_classes = np.unique(gt_imgs_stack)
	gt_classes = gt_classes[gt_classes!=255]
	le.fit(gt_classes)
	num_classes = 19#len(gt_classes)

	hist = np.zeros((num_classes, num_classes))

	import time
	time_0 = time.time()

	for ind, (pred, label) in enumerate(zip(pred_imgs_stack, gt_imgs_stack)):
		label = label.flatten()
		pred = pred.flatten()

		hist += fast_hist(label, pred, num_classes)

		if ind > 0 and ind % 10 == 0:
			print(f'Elapsed time: {time.time() - time_0}')
			time_0 = time.time()
			print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.nanmean(per_class_iu(hist))))

	mIoUs = per_class_iu(hist)

	for ind_class in range(num_classes):
		print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
	print(f'===> mIoU: {round(np.nanmean(mIoUs) * 100, 2)}')

	gt_freq = np.unique(gt_imgs_stack, return_counts=True)

	gt_classes_ = gt_freq[0].tolist()
	gt_counts_ = gt_freq[1].tolist()

	gt_classes = []
	gt_counts = []

	for i in range(num_classes):
		gt_classes.append(i)
		if i in gt_classes_:
			idx = gt_classes_.index(i)
			gt_counts.append(gt_counts_[idx])
		else:
			gt_counts.append(0)

	return mIoUs, (np.array(gt_classes), np.array(gt_counts))


def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')
	parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
	parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
	args = parser.parse_args()
	main(args)

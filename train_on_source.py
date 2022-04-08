"""
Note: the GitHub repo https://github.com/wasidennis/AdaptSegNet was used
as a starting point to train and test semantic segmetnation models on
the GTA5 and Cityscapes datasets. The reference model architectures are
also from the repo -- see models/*.py

In particularly, this file is a modification of
https://github.com/wasidennis/AdaptSegNet/blob/master/train_gta2cityscapes_multi.py
from which the adaptation part was removed.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import numpy.random as npr
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from model.deeplab import Res_Deeplab as Deeplab
from utils.loss import CrossEntropy2d

from dataset.gta5_dataset import GTA5
from dataset.gta52cityscapes_dataset import GTA52Cityscapes
from dataset.gta5_dataset_augm import GTA5Augm

IMG_MEAN = np.array(
		(104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

DATASET = 'GTA5'
MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/PATH/TO/GTA5/DATA'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0#.9
NUM_CLASSES = 19
NUM_EPOCHS = 5
POWER = 0.9
WEIGHT_DECAY = 0.0005

LAMBDA_SEG = 0.1


def get_arguments():
	"""Parse all the arguments provided from the CLI.

	Returns:
	A list of parsed arguments.
	"""
	parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
	parser.add_argument("--dataset", type=str, default=DATASET,
						help="available options : GTA5, GTA52Cityscapes")
	parser.add_argument("--model", type=str, default=MODEL,
						help="available options : DeepLab")
	parser.add_argument("--optimizer", type=str, default='SGD',
						help="available options : SGD/Adam/RMSprop")
	parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS,
						help="Number of training epochs.")
	parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
						help="Number of images sent to the network in one step.")
	parser.add_argument("--iter_size", type=int, default=ITER_SIZE,
						help="Accumulate gradients for ITER_SIZE iterations.")
	parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
						help="number of workers for multithread dataloading.")
	parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
						help="Path to the directory containing the source dataset.")
	parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
						help="Path to the file listing the images in the source dataset.")
	parser.add_argument("--ignore_label", type=int, default=IGNORE_LABEL,
						help="The index of the label to ignore during the training.")
	parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
						help="Comma-separated string with height and width of source images.")
	parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
						help="Base learning rate for training with polynomial decay.")
	parser.add_argument("--lambda_seg", type=float, default=LAMBDA_SEG,
						help="lambda_seg.")
	parser.add_argument("--momentum", type=float, default=MOMENTUM,
						help="Momentum component of the optimiser.")
	parser.add_argument("--not_restore_last", action="store_true",
						help="Whether to not restore last (FC) layers.")
	parser.add_argument("--num_classes", type=int, default=NUM_CLASSES,
						help="Number of classes to predict (including background).")
	parser.add_argument("--power", type=float, default=POWER,
						help="Decay parameter to compute the learning rate.")
	parser.add_argument("--random_mirror", action="store_true",
						help="Whether to randomly mirror the inputs during the training.")
	parser.add_argument("--random_scale", action="store_true",
						help="Whether to randomly scale the inputs during the training.")
	parser.add_argument("--seed", type=int, default=213,
						help="Random seed to have reproducible results.")
	parser.add_argument("--snapshot_dir", type=str, default='./snapshots',
						help="Where to save snapshots of the model.")
	parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
						help="Regularisation parameter for L2-loss.")
	parser.add_argument("--gpu", type=int, default=0,
						help="choose gpu device.")

	# for domain randomization (DR in the paper)
	parser.add_argument("--do_augm", type=int, default=0,
						help="Whether to perform data augmentation.")
	parser.add_argument("--augm_set", type=int, default=0,
						help="Which data augmentation set to use.")


	return parser.parse_args()

args = get_arguments()
args.do_augm = bool(args.do_augm)

if args.do_augm:
	raise NotImplementedError(
			"Code to perform data augmentation not available yet.")

if not args.do_augm:
	args.augm_set = 0

def loss_calc(pred, label, gpu):
	"""
	This function returns cross entropy loss for semantic segmentation
	"""
	label = Variable(label.long()).cuda(gpu)
	criterion = CrossEntropy2d().cuda(gpu)

	return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
	lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
	optimizer.param_groups[0]['lr'] = lr
	if len(optimizer.param_groups) > 1:
		optimizer.param_groups[1]['lr'] = lr * 10


def main():
	"""Create the model and start the training."""

	npr.seed(args.seed)

	w, h = map(int, args.input_size.split(','))
	input_size = (w, h)

	cudnn.enabled = True
	gpu = args.gpu

	# Create network
	model = Deeplab(num_classes=args.num_classes)

	args.restore_from = dict()
	args.restore_from['DeeplabMulti'] = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

	# --- loading pre-trained weights (on ImageNet + COCO) ------------
	print('Loading pre-trained model')
	if args.restore_from['DeeplabMulti'].startswith('http'):
		saved_state_dict = model_zoo.load_url(args.restore_from['DeeplabMulti'])
	else:
		saved_state_dict = torch.load(args.restore_from['DeeplabMulti'])

	saved_state_dict_original = model.state_dict().copy()
	for i in saved_state_dict:
		if (('bn' in i) or ('running_mean' in i) or ('running_var' in i)):
			continue
		# Scale.layer5.conv2d_list.3.weight
		i_parts = i.split('.')
		if not args.num_classes == 19 or not i_parts[1] == 'layer5':
			saved_state_dict_original['.'.join(i_parts[1:])] = saved_state_dict[i]


	# loading model
	model.load_state_dict(saved_state_dict_original)

	model.train()
	model.cuda(args.gpu)

	cudnn.benchmark = True

	summary_dict = {'loss':[], 'iter':[], 'lr':[]}

	args.snapshot_dir = f'./snapshots/{args.dataset}/arch_{args.model}_epochs_{args.num_epochs}_bs_{args.batch_size}_is_{args.iter_size}_' + \
						f'lr_{args.learning_rate}_mom_{args.momentum}_wd_{args.weight_decay}_opt_{args.optimizer}_' + \
						f'augm_{args.do_augm}_set_{args.augm_set}'

	print(f'exp = {args.snapshot_dir}')

	if not os.path.exists(args.snapshot_dir):
		os.makedirs(args.snapshot_dir)

	if args.dataset == 'GTA5':
		train_set = GTA5(root=args.data_dir, num_epochs=args.num_epochs,
						crop_size=input_size, mean=IMG_MEAN)
	elif args.dataset.upper() == 'GTA52CITYSCAPES':
		train_set = GTA52Cityscapes(root=args.data_dir, num_epochs=args.num_epochs,
								crop_size=input_size, mean=IMG_MEAN)

	trainloader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
								num_workers=args.num_workers, pin_memory=True)

	trainloader_iter = enumerate(trainloader)

	# implement model.optim_parameters(args) to handle different models' lr setting

	if args.optimizer == 'SGD':
		optimizer = optim.SGD(model.optim_parameters(args),
							lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer == 'Adam':
		optimizer = optim.Adam(model.optim_parameters(args), lr=args.learning_rate)

	elif args.optimizer == 'RMSprop':
		optimizer = optim.RMSprop(model.optim_parameters(args), lr=args.learning_rate)

	else:
		raise ValueError('Non-supported optimizer')

	optimizer.zero_grad()

	interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
	args.num_steps = args.num_epochs * (25000 // (args.batch_size * args.iter_size))

	for i_iter in range(args.num_steps):

		optimizer.zero_grad()

		if args.optimizer == 'SGD':
			adjust_learning_rate(optimizer, i_iter)

		for sub_i in range(args.iter_size):

			try:
				_, batch = next(trainloader_iter)
			except:
				print('End of training.')
				print('Saving model.')
				torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA.pth'))
				with open(osp.join(args.snapshot_dir, 'summary.pkl'),'wb') as f:
					pickle.dump(summary_dict, f, pickle.HIGHEST_PROTOCOL)
				exit()

			images, labels, _, _ = batch
			images = Variable(images).cuda(args.gpu)

			pred = model(images)
			pred = interp(pred)
			loss = loss_calc(pred, labels, args.gpu)

			# proper normalization
			loss = loss / args.iter_size
			loss.backward()

		optimizer.step()

		if i_iter%50==0:
			lr_ = optimizer.param_groups[0]['lr']
			print(f'iter = {i_iter:8d}/{args.num_steps:8d}, loss = {loss:.3f}, lr = {lr_:.5f}')

			summary_dict['iter'].append(i_iter)
			summary_dict['lr'].append(lr_)
			summary_dict['loss'].append(loss.detach().cpu().numpy())

		if (i_iter % 10000) == 0:
			print('Saving model.')
			torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA.pth'))
			with open(osp.join(args.snapshot_dir, 'summary.pkl'),'wb') as f:
				pickle.dump(summary_dict, f, pickle.HIGHEST_PROTOCOL)
		if (i_iter % 25000) == 0:
			print('Backing up model.')
			torch.save(model.state_dict(), osp.join(args.snapshot_dir, f'GTA_{i_iter}.pth'))


if __name__ == '__main__':
	main()

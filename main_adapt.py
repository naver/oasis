"""
Note: the GitHub repo https://github.com/wasidennis/AdaptSegNet was used
as a starting point to train and test semantic segmetnation models on
the GTA5 and Cityscapes datasets. The reference model architectures are
also from the repo -- see models/*.py
"""

import sys
import os
import glob
import random
import json
import copy
import argparse
import pickle

import torch
import torch.nn as nn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import numpy.random as npr

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# from AdaptSegNet repository
# (https://github.com/wasidennis/AdaptSegNet)
from deeplab import Res_Deeplab as Deeplab
from utils.loss import CrossEntropy2d

# ours
from dataset.cityscapes_dataset import Cityscapes
from dataset.acdc_dataset import ACDC
from dataset.synthia_dataset import SYNTHIA
from dataset.gta5_dataset import GTA5

from metrics_helpers import compute_mIoU_single_image, \
		compute_acc_single_image, compute_mIoU_fromlist, \
		compute_acc_fromlist
from image_helpers import ImageOps

IMG_MEAN = np.array(
		(104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

class SolverOps:

	"""
	Class used to carry out all train/test/load ops.
	"""

	def __init__(self, args):

		"""
		"args" contains all the input required to
		specify the experiments. See main() function.
		"""

		self.args = args

		# from the AdaptSegnet repo
		with open('./dataset/cityscapes_list/info.json', 'r') as f:
			info = json.load(f)

		self.num_classes = np.int(info['classes'])
		self.name_classes = np.array(info['label'], dtype=np.str)

		print('Num classes', self.num_classes)

		if len(self.args.cond.split('-')) != len(self.args.scene.split('-')):
			raise ValueError(
					'If using sequences, the number of conditions'+\
					f'must match the number of scenes.')

		# checking whether the adaptation mode is supported
		if self.args.adapt_mode not in [
				'batch-norm', 'naive-batch-norm', 'no-adaptation', 'tent',
				'naive-tent', 'pseudo-labels', 'naive-pseudo-labels',
				'class-reset-pseudo-labels', 'oracle-reset-pseudo-labels',
				'class-reset-batch-norm', 'class-reset-tent',
				'oracle-reset-batch-norm', 'oracle-reset-tent'
				]:
			raise ValueError(f'Unknown "adapt_mode" [{self.args.adapt_mode}]')

		self.image_ops = ImageOps()

		w_trg, h_trg = map(int, self.args.input_size_target.split(','))
		self.input_size_target = (w_trg, h_trg)

		w_src, h_src = map(int, self.args.input_size_source.split(','))
		self.input_size_source = (w_src, h_src)


	def adapt(self):

		"""
		Method to adapt a model sample by sample on a given
		sequence. All parameters setup by the user (args).
		"""

		cudnn.enabled = True
		gpu = self.args.gpu

		cudnn.benchmark = True

		summary_dict = {
						'loss':[],
						'iter':[],
						'lr':[],
						'pixel_acc_init':[],
						'pixel_acc_final':[],
						'mean_acc_init':[],
						'mean_acc_final':[],
						'mious_init':[],
						'mious_final':[],
						'pixel_acc_all':[],
						'mean_acc_all':[],
						'miou_all':[],
						'avg_class_miou':[],
						'avg_class_miou_bkp':[],
						'pred_count_all':[],
						'pred_count_all_ma':[],
						'wass_dist':[],
						'wass_dist_ma':[],
						'pred_count_all_bkp':[],
						'pred_count_all_ma_bkp':[],
						'wass_dist_bkp':[],
						'wass_dist_ma_bkp':[],
						'unique_classes':[],
						'unique_classes_bkp':[],
						'reset':[]
						}

		bn_stats_dict = {}

		optimizer = optim.SGD(self.model.optim_parameters(self.args), lr=self.args.learning_rate)

		interp_src = nn.Upsample(
				size=(self.input_size_source[1], self.input_size_source[0]), mode='bilinear')
		interp_trg = nn.Upsample(
				size=(self.input_size_target[1], self.input_size_target[0]), mode='bilinear')
		interp_trg_big = nn.Upsample(
				size=(self.input_size_target[1] * 2, self.input_size_target[0] * 2), mode='bilinear')

		self.args.num_steps = len(self.trg_train_loader)

		if self.args.adapt_mode == 'no-adaptation':
			print('NOT ADAPTING, JUST EVALUATING')
			self.model.eval()


		if np.any([_ in self.args.adapt_mode
				for _ in ['class-reset', 'oracle-reset']]):
			print('Model backup')
			self.model_bkp = copy.deepcopy(self.model)

		reset_bool = False

		for i_iter, trg_batch in enumerate(self.trg_train_loader):

			trg_image, trg_labels_ONLY_FOR_EVAL, _, trg_image_name = trg_batch
			trg_image = Variable(trg_image).cuda(self.args.gpu)

			# forward pass on eval - just to track initial metrics for this sample
			self.model.eval()
			trg_pred = self.model(trg_image)

			# computing initial miou
			output = interp_trg(trg_pred).cpu().data[0].numpy()
			output = output.transpose(1,2,0)
			output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

			if trg_labels_ONLY_FOR_EVAL is not None:
				trg_labels_ONLY_FOR_EVAL = np.squeeze(
						trg_labels_ONLY_FOR_EVAL.detach().cpu().numpy()).astype(np.uint8)
				mious_init = compute_mIoU_single_image(
						trg_labels_ONLY_FOR_EVAL, output, self.num_classes, self.name_classes)
				pixel_acc_init, mean_acc_init = compute_acc_single_image(
						trg_labels_ONLY_FOR_EVAL, output)
			else:
				mious_init = None
				pixel_acc_init = None
				mean_acc_init = None

			#output_ = trg_pred.argmax(1).cpu().numpy().squeeze()

			####### TRAINING/ADAPTING #########################################
			if self.args.adapt_mode != 'no-adaptation':
				self.model.train()
				if self.args.adapt_mode in [
							'tent', 'naive-tent', 'pseudo-labels',
							'naive-pseudo-labels', 'class-reset-pseudo-labels',
							'oracle-reset-pseudo-labels',
							'class-reset-tent', 'oracle-reset-tent'
							]:
					# updating model parameters in training iterations
					for _ in range(args.adapt_iters):
						# forward pass, BN statistics are updated
						trg_pred = self.model(trg_image)

						optimizer.zero_grad()

						if args.adapt_mode in [
									'pseudo-labels', 'naive-pseudo-labels',
									'oracle-reset-pseudo-labels',
									'class-reset-pseudo-labels'
									]:
							trg_psd_labels = self.compute_pseudo_labels(
									trg_pred, args.pseudo_labels_mode,
									args.pseudo_labels_thrs)
							trg_loss = self.loss_calc(
									trg_pred, trg_psd_labels, args.gpu)

						elif args.adapt_mode in [
									'tent', 'naive-tent', 'class-reset-tent',
									'oracle-reset-tent']:
							trg_loss = self.compute_output_entropy(
									trg_pred, avg_batch=True)

						else:
							raise RuntimeError(
									f'Unknown args.adapt_mode [{args.adapt_mode}]')

						# if also using source samples
						if args.src_iters > 0:
							src_loss = 0
							for _ in range(args.src_iters):
								try:
									_, src_batch = next(self.src_train_loader_iter)
								except:
									# we create another source iterator, if finished
									del(self.src_train_loader_iter)
									del(self.src_train_loader)
									self.src_train_loader = data.DataLoader(
											self.src_parent_set, batch_size=args.batch_size,
											shuffle=True, num_workers=args.num_workers,
											pin_memory=True)
									# ---
									self.src_train_loader_iter = enumerate(self.src_train_loader)
									_, src_batch = next(self.src_train_loader_iter)

								src_images, src_labels, _, _ = src_batch
								src_images = Variable(src_images).cuda(args.gpu)

								# computing source loss with ground truth
								src_pred = self.model(src_images)
								src_pred = interp_src(src_pred)

								src_loss += self.loss_calc(
										src_pred, src_labels, self.args.gpu)

						# summing everything up
						if args.src_iters == 0:
							loss = trg_loss

						else:
							loss = trg_loss + args.src_iters * src_loss
							loss = loss/(1 + args.src_iters)

						loss.backward()
						optimizer.step()

				elif self.args.adapt_mode in ['batch-norm', 'naive-batch-norm']:
					# simple forward pass, where BN stats are updated
					trg_pred = self.model(trg_image)

				else:
					raise ValueError(f'Unknown adapt_mode {self.args.adapt_mode}')


			####### EVALUATING ############################################
			# computing final metrics for the current image - if adaptation
			if self.args.adapt_mode != 'no-adaptation':
				self.model.eval()
				trg_pred = self.model(trg_image)
				trg_feat = self.model(trg_image, extract_features=True)

				output = interp_trg(trg_pred).cpu().data[0].numpy()
				output = output.transpose(1,2,0)
				output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

				# computing histogram of predictions made by the model
				pred_id, pred_count = np.unique(output, return_counts=True)
				pred_id = pred_id.tolist()
				pred_count = [pred_count[pred_id.index(n)] 
						if n in pred_id else 0 for n in range(19)]

				# computing classes present in the image - for mIoU computation
				image_gt = np.unique(trg_labels_ONLY_FOR_EVAL)
				image_gt = image_gt[image_gt!=255]

				if trg_labels_ONLY_FOR_EVAL is not None:
					mious_final = compute_mIoU_single_image(
							trg_labels_ONLY_FOR_EVAL, output,
							self.num_classes, self.name_classes)
					pixel_acc_final, mean_acc_final = compute_acc_single_image(
							trg_labels_ONLY_FOR_EVAL, output)
				else:
					mious_final = None
					pixel_acc_final = None
					mean_acc_final = None

				# computing the average mious for the last three samples
				avg_class_miou_ = np.copy(mious_final)
				avg_class_miou_[np.delete(np.arange(19), image_gt)] = np.nan
				avg_class_miou = np.nanmean(avg_class_miou_)

				if self.args.adapt_mode.startswith('class-reset') \
						or self.args.adapt_mode.startswith('oracle-reset'):

					trg_pred_bkp = self.model_bkp(trg_image)
					output_bkp = interp_trg(trg_pred_bkp).cpu().data[0].numpy()
					output_bkp = output_bkp.transpose(1,2,0)
					output_bkp = np.asarray(np.argmax(output_bkp, axis=2), dtype=np.uint8)
					mious_final_bkp = compute_mIoU_single_image(
							trg_labels_ONLY_FOR_EVAL, output_bkp,
							self.num_classes, self.name_classes)
					pixel_acc_final_bkp, mean_acc_final_bkp = compute_acc_single_image(
							trg_labels_ONLY_FOR_EVAL, output_bkp)

					# computing the average mious for the last three samples
					avg_class_miou_bkp_ = np.copy(mious_final_bkp)
					avg_class_miou_bkp_[np.delete(np.arange(19), image_gt)] = np.nan
					avg_class_miou_bkp = np.nanmean(avg_class_miou_bkp_)

					# computing histogram of predictions made by the model
					pred_id_bkp, pred_count_bkp = np.unique(output_bkp, return_counts=True)
					pred_id_bkp = pred_id_bkp.tolist()
					pred_count_bkp = [
							pred_count_bkp[pred_id_bkp.index(n)] if n in pred_id_bkp else 0
									for n in range(19)]

				else:
					avg_class_miou_bkp = np.nan
					pred_count_bkp = [np.nan]
					pred_id_bkp = [np.nan]

			else:
				# if 'no-adaptation', nothing has changed, so we avoid computations
				mious_final = mious_init
				pixel_acc_final = pixel_acc_init
				mean_acc_final = mean_acc_init
				avg_class_miou = np.nan
				avg_class_miou_bkp = np.nan
				pred_count = [np.nan]
				pred_count_bkp = [np.nan]
				pred_id_bkp = [np.nan]
				pred_id = [np.nan]

			# for saving predictions and re-compute mIoU at the end
			output_big = interp_trg_big(trg_pred).cpu().data[0].numpy()
			output_big = output_big.transpose(1,2,0)
			output_big = np.asarray(np.argmax(output_big, axis=2), dtype=np.uint8)

			# appending stuff to dict
			summary_dict['iter'].append(i_iter)
			summary_dict['reset'].append(reset_bool)
			summary_dict['mious_init'].append(mious_init)
			summary_dict['mious_final'].append(mious_final)
			summary_dict['pixel_acc_init'].append(pixel_acc_init)
			summary_dict['pixel_acc_final'].append(pixel_acc_final)
			summary_dict['mean_acc_init'].append(mean_acc_init)
			summary_dict['mean_acc_final'].append(mean_acc_final)
			summary_dict['unique_classes'].append(len(pred_id))
			summary_dict['unique_classes_bkp'].append(len(pred_id_bkp))
			summary_dict['avg_class_miou'].append(avg_class_miou)
			summary_dict['pred_count_all'].append(pred_count)
			summary_dict['avg_class_miou_bkp'].append(avg_class_miou_bkp)
			summary_dict['pred_count_all_bkp'].append(pred_count_bkp)

			# computing unique classes, to perform class-reset in case of forgetting
			avg_unique_classes = np.mean(summary_dict['unique_classes'][-self.args.buffer_size:])
			avg_unique_classes_bkp = np.mean(summary_dict['unique_classes_bkp'][-self.args.buffer_size:])

			# computing moving average mIoU performance and pred count
			avg_class_miou_ma = np.mean(summary_dict['avg_class_miou'][-self.args.buffer_size:])
			pred_count_ma = np.array(summary_dict['pred_count_all'][-self.args.buffer_size:]).mean(0)

			# computing same values for bkp model
			if self.args.adapt_mode.startswith('class-reset') \
					or self.args.adapt_mode.startswith('oracle-reset'):

				avg_class_miou_bkp_ma = np.mean(
						summary_dict['avg_class_miou_bkp'][-self.args.buffer_size:])
				pred_count_ma_bkp = np.array(
						summary_dict['pred_count_all_bkp'][-self.args.buffer_size:]).mean(0)
			else:
				avg_class_miou_bkp_ma = np.nan
				pred_count_ma_bkp = [np.nan]

			summary_dict['pred_count_all_ma'].append(pred_count_ma)
			summary_dict['pred_count_all_ma_bkp'].append(pred_count_ma_bkp)

			# difference between number of classes
			class_dist = avg_unique_classes_bkp - avg_unique_classes

			# if 'naive', we re-load the pre-trained model
			if self.args.adapt_mode in [
					'naive-batch-norm', 'naive-tent', 'naive-pseudo-labels']:
				self.load_model()

			# if 'reset' method &  trigger, we re-load the pre-trained model
			elif self.args.adapt_mode.startswith('class-reset'):
				if class_dist > self.args.reset_thrs:
					print('Smart reset [class forgetting]')
					self.model.load_state_dict(self.model_bkp.state_dict())
					avg_class_miou_ma = avg_class_miou_bkp_ma
					avg_class_miou = avg_class_miou_bkp
					summary_dict['avg_class_miou'][-1] = avg_class_miou_bkp
					mious_final = mious_final_bkp
					pixel_acc_final = pixel_acc_final_bkp
					mean_acc_final = mean_acc_final_bkp
					summary_dict['mious_final'][-1] = mious_final_bkp
					summary_dict['pixel_acc_final'][-1] = pixel_acc_final_bkp
					summary_dict['mean_acc_final'][-1] = mean_acc_final_bkp
				else:
					reset_bool = True

			elif self.args.adapt_mode.startswith('oracle-reset'):
				if avg_class_miou_bkp_ma > avg_class_miou_ma:
					print('Oracle reset [lower mIoU than baseline]')
					self.model.load_state_dict(self.model_bkp.state_dict())
					avg_class_miou_ma = avg_class_miou_bkp_ma
					avg_class_miou = avg_class_miou_bkp
					summary_dict['avg_class_miou'][-1] = avg_class_miou_bkp
					mious_final = mious_final_bkp
					pixel_acc_final = pixel_acc_final_bkp
					mean_acc_final = mean_acc_final_bkp
					summary_dict['mious_final'][-1] = mious_final_bkp
					summary_dict['pixel_acc_final'][-1] = pixel_acc_final_bkp
					summary_dict['mean_acc_final'][-1] = mean_acc_final_bkp
				else:
					reset_bool = True
			else:
				reset_bool = False

			mious_final_print = np.nanmean(mious_final)
			mious_init_print = np.nanmean(mious_init)
			pixel_acc_init_print = pixel_acc_init
			pixel_acc_final_print = pixel_acc_final
			mean_acc_init_print = mean_acc_init
			mean_acc_final_print = mean_acc_final

			if i_iter%1==0:
				print(f'iter = {i_iter:4d}/{self.args.num_steps:5d}, '+ \
						f'mIoU (init) = {mious_init_print:.3f},'+ \
						f'mIoU (final) = {mious_final_print:.3f}')

				# --- saving images ------
				trg_image_ = self.image_ops.process_image_for_saving(
						trg_image, interp_trg)
				output_ = self.image_ops.colorize_mask(output)
				output_big_col = self.image_ops.colorize_mask(output_big)
				output_big = Image.fromarray(output_big)
				trg_labels_ONLY_FOR_EVAL_ = self.image_ops.colorize_mask(
						trg_labels_ONLY_FOR_EVAL)
				self.image_ops.save_concat_image(
						trg_image_, trg_labels_ONLY_FOR_EVAL_, output_,
							self.input_size_target, self.result_dir,
							f'{i_iter:06d}')

				image_name = trg_image_name[0].split('/')[-1]
				output_big.save(os.path.join(
						self.result_dir, f'{i_iter:06d}_label.png'))
				output_big_col.save(os.path.join(
						self.result_dir, f'{i_iter:06d}_color.png'))
				# -------------------------

		print('End of training.')

		print('Saving model.')
		torch.save(self.model.state_dict(),
				os.path.join(self.model_dir, self.model_name))

		# dumping before final tests, in case something goes wrong with them
		with open(os.path.join(self.model_dir, self.summary_name),'wb') as f:
			pickle.dump(summary_dict, f, pickle.HIGHEST_PROTOCOL)

		with open(os.path.join(self.model_dir, self.DONE_name),'wb') as f:
			print('Saving end of training file')


	def setup_model(self):
		# Create network
		if self.args.model_arch == 'Deeplab':
			self.model = Deeplab(
					num_classes=self.num_classes)

		else:
			raise NotImplementedError(f'{self.args.model_arch}')

		self.model.train()
		self.model.cuda(self.args.gpu)


	def loss_calc(self, pred, label, gpu):

		"""
		This function returns cross entropy loss for semantic segmentation
		"""

		label = Variable(label.long()).cuda(gpu)
		criterion = CrossEntropy2d().cuda(gpu)

		return criterion(pred, label)


	def load_model(self, vanilla_load=False):

		"""
		Method to load a pre-trained model.
		"""

		# setting all to False
		if 'pseudo-labels' not in self.args.adapt_mode:
			# setting all to non-trainable
			for param in self.model.parameters():
				param.requires_grad = False

		if ('pseudo-labels' in self.args.adapt_mode) \
				and self.args.adapt_only_classifier:
			# we will only adapt the classifier
			for child in self.model.children():
				classname = child.__class__.__name__
				if classname == 'Classifier_Module':
					for param in child.parameters():
						param.requires_grad = True
				else:
					for param in child.parameters():
						param.requires_grad = False

		print('Setting BN params to \'trainable\'')
		for module in self.model.modules():
			classname = module.__class__.__name__
			if classname.find('BatchNorm') != -1:
				module.momentum = self.args.batch_norm_momentum
				if not self.args.adapt_only_classifier:
					# avoid learning the BN's params
					for param in module.parameters():
						param.requires_grad = True

		print('Load source model')
		saved_state_dict = torch.load(self.args.restore_from)
		self.model.load_state_dict(saved_state_dict, strict=True)


	def compute_output_entropy(
			self, predictions, avg_image=True, avg_batch=True):

		"""
		Given output predictions, performs softmax and computes entropy

		params:

			predictions : torch tensor (M,K,H,W)

				The output of the model: M batch size, K
				number of classes, (H,W) image size

			avg_image : bool

				If True, averages the different pixels'
				output predictions in a single number

			avg_batch : bool

				If True, averages everything (this is like
				reduce_mean + avg_image=True)

		returns:

			output_entropy : torch.tensor / torch.float

				The entropy associated with the prediction
				provided.

		"""


		predictions = torch.softmax(predictions, 1)
		output_entropy = torch.sum(
				-(torch.log(predictions) * predictions),1)

		if avg_batch:
			return torch.mean(output_entropy)
		else:
			if avg_image:
				return output_entropy.mean(-1).mean(-1)
			else:
				return output_entropy


	def compute_pseudo_labels(
			self, predictions, pseudo_labels_mode, pseudo_labels_thrs=None):

		"""
		Method to compute pseudo labels.

		params:

			predictions : torch.tensor (M,K,H,W)

				The output of the model: M batch size, K number of
				classes, (H,W) image size

			pseudo_labels_mode : str

				Which PL method to use. Supported values are
				'vanilla' or 'softmax'

			pseudo_labels_thrs : str

				Which threshold to use, if required by the
				PL method specified via pseudo_labels_mode


		returns:

			trg_psd_labels : torch.tensor

				Pseudo labels associted with the provided
				predictions and thresholding method.
		"""

		# generating pseudo-labels
		if pseudo_labels_mode == 'vanilla':
			# "hard" pseudo-labels: treating every pixel
			# prediction as ground truth,
			# regardless of any confidence metric.
			trg_psd_labels = torch.argmax(predictions, 1)

		elif pseudo_labels_mode == 'softmax':
			# "softmax-based" pseudo-labels: setting a threshold
			# and only using the predictions whose associated softmax
			# value is higher than that.
			trg_psd_labels = torch.argmax(predictions, 1)
			trg_softmax_vals = torch.max(torch.softmax(predictions, 1),1)[0]
			trg_psd_labels[trg_softmax_vals<pseudo_labels_thrs] = 255

		return trg_psd_labels


	def setup_experiment_folder(self):

		"""
		Method to define model folder's name and create it, and to
		define the name of the output files created at end of training.
		"""

		if self.args.adapt_mode in [
				'batch-norm', 'naive-batch-norm', 'oracle-reset-batch-norm']:
			self.args.adapt_mode_save = f'{self.args.adapt_mode}'+\
										f'_{self.args.batch_norm_momentum}'

		elif self.args.adapt_mode in ['class-reset-batch-norm']:
			self.args.adapt_mode_save = f'{self.args.adapt_mode}'+\
						f'_{self.args.batch_norm_momentum}-{self.args.reset_thrs}'

		elif self.args.adapt_mode in ['tent', 'naive-tent', 'oracle-reset-tent']:
			self.args.adapt_mode_save = f'{self.args.adapt_mode}_'+\
								f'{self.args.adapt_iters}-{self.args.learning_rate}-'+\
								f'{self.args.src_iters}-{self.args.batch_norm_momentum}'

		elif self.args.adapt_mode == 'class-reset-tent':
			self.args.adapt_mode_save = f'{self.args.adapt_mode}_'+\
								f'{self.args.adapt_iters}-{self.args.learning_rate}-'+\
								f'{self.args.src_iters}-{self.args.batch_norm_momentum}-'+\
								f'{self.args.reset_thrs}'

		elif self.args.adapt_mode in [
				'pseudo-labels', 'naive-pseudo-labels', 'oracle-reset-pseudo-labels']:
			self.args.adapt_mode_save = f'{self.args.adapt_mode}_'+\
								f'{self.args.adapt_iters}-{self.args.learning_rate}-'+\
								f'{self.args.src_iters}-{self.args.pseudo_labels_mode}-'+\
								f'{self.args.pseudo_labels_thrs}-'+\
								f'{self.args.batch_norm_momentum}'

		elif self.args.adapt_mode == 'class-reset-pseudo-labels':
			self.args.adapt_mode_save = f'{self.args.adapt_mode}_'+\
								f'{self.args.adapt_iters}-{self.args.learning_rate}-'+\
								f'{self.args.src_iters}-{self.args.pseudo_labels_mode}-'+\
								f'{self.args.pseudo_labels_thrs}-'+\
								f'{self.args.batch_norm_momentum}'+\
								f'{self.args.reset_thrs}'

		elif self.args.adapt_mode == 'no-adaptation':
			if args.wct2_random_style_transfer:
				self.args.adapt_mode_save  = 'style-transfer_wct2-random'
			elif args.wct2_nn_style_transfer:
				self.args.adapt_mode_save  = 'style-transfer_wct2-nn'
			else:
				self.args.adapt_mode_save = 'no-adaptation'

		else:
			raise NotImplementedError(f'Unknown "adapt_mode" [{self.args.adapt_mode}]')

		sub_folder0 = f'{self.args.trg_dataset}_{self.args.scene}_{self.args.cond}'
		self.model_name = f'{self.args.src_dataset}_to_{self.args.trg_dataset}_'+\
							f'{self.args.scene}_{self.args.cond}.pth'
		self.DONE_name = f'{self.args.src_dataset}_to_{self.args.trg_dataset}_'+\
							f'{self.args.scene}_{self.args.cond}.DONE'
		self.summary_name = f'{self.args.src_dataset}_to_{self.args.trg_dataset}_'+\
							f'{self.args.scene}_{self.args.cond}_summary.pkl'

		if self.args.adapt_mode == 'no-adaptation':
			sub_folder1 = 'no_adaptation'
		elif self.args.adapt_mode in [
				'pseudo-labels', 'naive-pseudo-labels',
				'class-reset-pseudo-labels', 'oracle-reset-pseudo-labels']:
			if self.args.adapt_only_classifier:
				sub_folder1 = 'only_classifier'
			else:
				sub_folder1 = 'whole_net'
		else:
			sub_folder1 = 'only_batch_norm_params'

		restore_from_name = self.args.restore_from.split("/")[-2]

		if self.args.seed == 111: # default
			self.model_dir = os.path.join(
					self.args.models_dir,
					f'{restore_from_name}/src2trg/{sub_folder0}/{sub_folder1}/'+\
					f'adapt_{self.args.adapt_mode_save}'
					)
		else:
			self.model_dir = os.path.join(
					self.args.models_dir,
					f'{restore_from_name}/src2trg/{sub_folder0}/{sub_folder1}/'+\
					f'adapt_{self.args.adapt_mode_save}/seed_{self.args.seed}'
					)


		# check if experiment/testing was done already
		if os.path.isfile(
				os.path.join(self.model_dir, self.DONE_name)) \
				and not self.args.force_retraining:
			print('DONE file present -- training was already carried out')
			print(os.path.join(self.model_dir, self.DONE_name))
			exit(0)

		self.result_dir = os.path.join(self.model_dir, 'results')

		print(f'Experiment directory: {self.model_dir}')

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		if not os.path.exists(self.result_dir):
			os.makedirs(self.result_dir)


	def setup_source_data_loader(self):

		"""
		Method to create pytorch dataloaders for the
		source domain. This is fixed to GTA5, used for
		pre-training our models.
		"""

		self.src_parent_set = GTA5(
				GTA5_ROOT, num_epochs=1,
				crop_size=self.input_size_source, mean=IMG_MEAN)


		self.src_train_loader = data.DataLoader(
				self.src_parent_set, batch_size=args.batch_size,
				shuffle=True, num_workers=args.num_workers, pin_memory=True)

		self.src_train_loader_iter = enumerate(self.src_train_loader)


	def setup_target_data_loader(self):

		"""
		Method to create pytorch dataloaders for the
		target domain selected by the user
		"""

		if len(self.args.scene.split('-')) > 1:
			scene_list = self.args.scene.split('-')
			cond_list = self.args.cond.split('-')
		else:
			scene_list = [self.args.scene]
			cond_list = [self.args.cond]

		if self.args.trg_dataset=='Cityscapes':
			print(f'Loading Cityscapes from {self.args.cityscapes_root}')
			self.trg_parent_set = Cityscapes(
					self.args.cityscapes_root, scene_list, cond_list,
					crop_size=self.input_size_target, mean=IMG_MEAN,
					alpha=0.02, beta=0.01, dropsize=0.005, pattern=3,
					wct2_random_style_transfer = self.args.wct2_random_style_transfer,
					wct2_nn_style_transfer = self.args.wct2_nn_style_transfer)

		elif self.args.trg_dataset=='SYNTHIA':
			self.trg_parent_set = SYNTHIA(
					self.args.synthia_root, scene_list, cond_list,
					camera_id='0', crop_size=self.input_size_target,
					mean=IMG_MEAN, set='all', num_images=300,
					wct2_random_style_transfer = self.args.wct2_random_style_transfer,
					wct2_nn_style_transfer = self.args.wct2_nn_style_transfer)

		elif self.args.trg_dataset=='ACDC':
			self.trg_parent_set = ACDC(
					self.args.acdc_root, scene_list, cond_list,
					crop_size=self.input_size_target, mean=IMG_MEAN,
					wct2_random_style_transfer = self.args.wct2_random_style_transfer,
					wct2_nn_style_transfer = self.args.wct2_nn_style_transfer)

		else:
			raise ValueError(f'Unknown dataset {self.args.dataset}')

		self.trg_train_loader = data.DataLoader(
				self.trg_parent_set, batch_size=1, shuffle=False, pin_memory=True)


if __name__ == '__main__':

	# Parse all the arguments provided from the CLI.
	parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

	# what to do with the script (training, testing, etc.)
	parser.add_argument("--mode", type=str, default='adapt',
						help="available options : adapt")

	# main experiment parameters
	parser.add_argument("--model_arch", type=str, default='Deeplab',
						help="available options : {DeepLab, Resnet50Dilated}")
	parser.add_argument("--src_dataset", type=str, default='GTA5',
						help="Which source dataset to start from [GTA5]")
	parser.add_argument("--batch_size", type=int, default=1,
						help="Number of images sent to the network in one step.")
	parser.add_argument("--num_workers", type=int, default=4,
						help="number of workers for multithread dataloading.")
	parser.add_argument("--seed", type=int, default=111,
						help="Random seed to have reproducible results.")
	parser.add_argument("--models_dir", type=str, default='./adapted-models',
						help="Where to save trained models.")
	parser.add_argument("--gpu", type=int, default=0,
						help="choose gpu device.")
	parser.add_argument("--restore_from", type=str,
						default='./pre-trained-models/GTA5_Deeplab_DR2/GTA5.pth',
						help="path to .pth model")
	parser.add_argument("--force_retraining", type=int, default=0,
						help="Whether to re-train even if exp already done")

	# data dirs
	parser.add_argument("--cityscapes_root", type=str, default='./data/Cityscapes',
						help="Directory which contains Cityscapes data.")
	parser.add_argument("--synthia_root", type=str, default='./data/SYNTHIA',
						help="Directory which contains SYNTHIA data.")
	parser.add_argument("--gta5_root", type=str, default='./data/GTA5',
						help="Directory which contains GTA5 data.")
	parser.add_argument("--acdc_root", type=str, default='./data/ACDC',
						help="Directory which contains ACDC data.")

	# for target
	parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
						help="Which target dataset to transfer to")
	parser.add_argument("--scene", type=str, default='aachen',
						help="Scene, depends on specific datasets")
	parser.add_argument("--cond", type=str, default='clean',
						help="Condition, depends on specific datasets")

	# Rain/Foggy Cityscapes parameters (beta only one for both)
	parser.add_argument("--alpha", type=float, default=0.02,
						help="Alpha [for Rain]")
	parser.add_argument("--beta", type=float, default=0.02,
						help="Beta [for Rain and Fog]")
	parser.add_argument("--dropsize", type=float, default=0.005,
						help="Rain's dropsize [for Rain]")
	parser.add_argument("--pattern", type=int, default=3,
						help="Rain's pattern [for Rain]")

	# for adapt method
	parser.add_argument("--adapt_mode", type=str, default='no-adaptation',
						help="Which method to use to adapt the network")

	# batch-norm parameters
	parser.add_argument("--batch_norm_momentum", type=float, default=0.1,
						help="momentum for the BN layers")
	parser.add_argument("--batch_norm_mean_inf_momentum", type=float, default=0.0,
						help="momentum for the BN layers - only at inference time")
	parser.add_argument("--batch_norm_var_inf_momentum", type=float, default=0.0,
						help="momentum for the BN layers - only at inference time")

	# for iterative adaptation algorithms (pseudo-labeling, test)
	parser.add_argument("--adapt_iters", type=int, default=3,
						help="How many staps to carry out for the adaptation process")
	parser.add_argument("--learning_rate", type=float, default=0.0001,
						help="Base learning rate for training with polynomial decay.")

	# parameters for pseudo-labeling algos
	parser.add_argument("--pseudo_labels_mode", type=str, default='vanilla',
						help="How to use pseudo-labels [vanilla, softmax]")
	parser.add_argument("--pseudo_labels_thrs", type=float, default=0.8,
						help="Threshold for filtering pseudo label")
	parser.add_argument('--adapt_only_classifier', action='store_true', default=False,
						help="Whether to only adapt the final classifier)")

	# for reset algos
	parser.add_argument("--reset_thrs", type=float, default=0.0,
						help="Threshold for resetting the model [0.0; 1.0]")
	parser.add_argument("--buffer_size", type=int, default=1,
						help="How many samples to consider for the buffer")

	# source-rehearse parameters
	parser.add_argument("--src_iters", type=int, default=0,
						help="How many source samples to mix with in the current update")

	# style transfer
	parser.add_argument('--wct2_random_style_transfer', action='store_true', default=False,
						help="Use images pre-processed with WCT2 - Random")
	parser.add_argument('--wct2_nn_style_transfer', action='store_true', default=False,
						help="Use images pre-processed with WCT2 - Nearest_samples")

	args = parser.parse_args()

	if args.pseudo_labels_mode == 'vanilla':
		args.pseudo_labels_thrs = 0.0

	args.force_retraining = bool(args.force_retraining)

	if 'Cityscapes' in args.trg_dataset:
		args.input_size_target = '1024,512'
	elif 'ACDC' in args.trg_dataset:
		args.input_size_target = '960,540'
	elif 'SYNTHIA' in args.trg_dataset:
		args.input_size_target = '640,380'
	else:
		raise NotImplementedError("Input size unknown")

	if args.src_dataset == 'GTA5':
		args.input_size_source = '1280,720'
	else:
		raise ValueError(f'Unknown source dataset {args.src_dataset}')

	npr.seed(args.seed)

	solver_ops = SolverOps(args)

	print('Setting up experiment folder')
	solver_ops.setup_experiment_folder()

	print('Setting up data target loader')
	solver_ops.setup_target_data_loader()

	if args.src_iters > 0:
		print('Setting up data source loader')
		solver_ops.setup_source_data_loader()

	print('Defining model')
	solver_ops.setup_model()

	if args.mode == 'adapt':
		print('Loading pre-trained model')
		solver_ops.load_model()
		print('Start adapting')
		solver_ops.adapt()

	else:
		raise ValueError(f'Unknown args.mode [{args.mode}]')

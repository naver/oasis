import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

trg_dataset_list, scene_list, cond_list = [], [], []

run_exps_helpers.update_synthia_lists('multi', trg_dataset_list, scene_list, cond_list)

run_exps_helpers.update_cityscapes_lists('multi-O', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cityscapes_lists('multi-AW', trg_dataset_list, scene_list, cond_list)

run_exps_helpers.update_acdc_lists('multi', trg_dataset_list, scene_list, cond_list)

#which models we want to start from
model_arch_ = 'Deeplab'
pth_list = ['/PATH/TO/PRETRAINED/SOURCE/MODELS/*.pth']

seed_list=[111]

adapt_mode_list = ['naive-pseudo-labels', 'pseudo-labels',
					'class-reset-pseudo-labels', 'oracle-reset-pseudo-labels']

# learning rate
learning_rate_list = [0.0001, 0.001]

# number of adaptation iterations
adapt_iters_list = [1, 3, 5]

# number of source iterations (-SR)
src_iters_list = [0,1,2]

# BN momentum while training
bn_mom_list = [0.1]

# PL mode, and threshold if softmax is used
pseudo_labels_mode_list = ['vanilla']#
pseudo_labels_thrs_list = [0.0]

# in case one desires not to train the whole net
param_to_train_list = ['whole_net']

models_dir_ = '/WHERE/TO/SAVE/MODELS'

for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list, scene_list, cond_list):
	for seed_ in seed_list:
		for adapt_mode_ in adapt_mode_list:
			for pseudo_labels_mode_ in pseudo_labels_mode_list:
					for pseudo_labels_thrs_ in pseudo_labels_thrs_list:
						for learning_rate_ in learning_rate_list:
							for adapt_iters_ in adapt_iters_list:
								for src_iters_ in src_iters_list:
									for bn_mom_ in bn_mom_list:
										for restore_from_ in pth_list:
											for param_to_train_ in param_to_train_list:

												if adapt_mode_ in ['naive-pseudo-labels', 'pseudo-labels', 'oracle-reset-pseudo-labels']:
													reset_thrs_ = 0.0
												elif adapt_mode_ == 'smart-reset-pseudo-labels':
													reset_thrs_ = 0.5
												else:
													raise RuntimeError(f'Unknown adapt_mode [adapt_mode_]')

												to_print = (f'python -u main_adapt.py'+
															f' --seed={seed_}'+
															f' --trg_dataset={trg_dataset_}' +
															f' --scene={scene_}' +
															f' --cond={cond_}' +
															f' --model_arch={model_arch_}' +
															f' --learning_rate={learning_rate_}'+
															f' --adapt_iters={adapt_iters_}'+
															f' --src_iters={src_iters_}'+
															f' --adapt_mode={adapt_mode_}'+
															f' --force_retraining={force_retraining}'+
															f' --pseudo_labels_mode={pseudo_labels_mode_}'+
															f' --pseudo_labels_thrs={pseudo_labels_thrs_}'+
															f' --batch_norm_momentum={bn_mom_}'+
															f' --reset_thrs={reset_thrs_}'+
															f' --restore_from={restore_from_}')

												if param_to_train_ == 'only_classifier':
													to_print += f' --adapt_only_classifier'

												print(to_print)

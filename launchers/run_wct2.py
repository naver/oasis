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

# we don't perform adaptation in addition to Style transfer,
# but it can be done.
adapt_mode_list = ['no-adaptation']

# different style transfer modes
style_transfer_list = ['WCT2Random', 'WCT2Nearest_samples']

models_dir_ = './adapted-models'

for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list, scene_list, cond_list):
	for seed_ in seed_list:
		for adapt_mode_ in adapt_mode_list:
			for style_transfer_ in style_transfer_list:
				for restore_from_ in pth_list:

					to_print = (f'python -u main_adapt.py'+
								f' --force_retraining={force_retraining}'+
								f' --mode={exp_mode_}'+
								f' --seed={seed_}'+
								f' --trg_dataset={trg_dataset_}' +
								f' --scene={scene_}' +
								f' --cond={cond_}' +
								f' --model_arch={model_arch_}' +
								f' --adapt_mode={adapt_mode_}'+
								f' --restore_from={restore_from_}')

					if style_transfer_ == 'WCT2Random':
						to_print += f' --wct2_random_style_transfer'

					elif style_transfer_ == 'WCT2Nearest_samples':
						to_print += f' --wct2_nn_style_transfer'

					else:
						raise RuntimeError(f'Unknown style_transfer_ method [{style_transfer_}')

					print(to_print)

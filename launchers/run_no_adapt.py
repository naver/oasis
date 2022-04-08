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
adapt_mode_ = 'no-adaptation'
models_dir_ = './adapted-models'

for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list, scene_list, cond_list):
	for seed_ in seed_list:
		for restore_from_ in pth_list:
			print(f'python -u main_adapt.py'+
				f' --force_retraining={force_retraining}'+
				f' --seed={seed_}'+
				f' --trg_dataset={trg_dataset_}' +
				f' --scene={scene_}' +
				f' --cond={cond_}' +
				f' --model_arch={model_arch_}' +
				f' --adapt_mode={adapt_mode_}'+
				f' --restore_from={restore_from_}')

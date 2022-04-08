import glob
import os
import sys

import numpy.random as npr
import numpy as np

import run_exps_helpers

mode_='train'

if mode_ == 're_train':
	force_retraining = 1
else:
	force_retraining = 0

trg_dataset_list, scene_list, cond_list = [], [], []

run_exps_helpers.update_synthia_lists('multi', trg_dataset_list, scene_list, cond_list)

run_exps_helpers.update_cityscapes_lists('multi-city', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cityscapes_lists('multi-all', trg_dataset_list, scene_list, cond_list)

run_exps_helpers.update_acdc_lists('multi', trg_dataset_list, scene_list, cond_list)

#which models we want to start from
model_arch_ = 'Deeplab'
pth_list = ['/PATH/TO/PRETRAINED/SOURCE/MODELS/*.pth']

seed_list=[111]

# N-BN or C-BN
adapt_mode_list = ['batch-norm', 'reset-batch-norm']

# BN momentum
batch_norm_momentum_list = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

models_dir_ = './trained-models'
for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list, scene_list, cond_list):
	for seed_ in seed_list:
		for adapt_mode_ in adapt_mode_list:
			for restore_from_ in pth_list:
				for bn_mom_ in batch_norm_momentum_list:

					if (adapt_mode_ == 'smart-reset-batch-norm') ^ (reset_thrs_ != 0.0):
						continue

					sub_folder1 = 'only_batch_norm_params'
					adapt_mode_save = f'{adapt_mode_}_{bn_mom_}'
					if adapt_mode_ == 'smart-reset-batch-norm' or adapt_mode_ == 'distance-reset-batch-norm':
						adapt_mode_save += f'-{reset_thrs_}'

					trg_dataset_name_ = trg_dataset_.lstrip('Full') if trg_dataset_=='FullSYNTHIA' else trg_dataset_

					sub_folder0 = f'{trg_dataset_name_}_{scene_}_{cond_}'

					if seed_ == 111:
						model_dir = os.path.join(
								models_dir_,
								f'{restore_from_.split("/")[-2]}/src2trg/{sub_folder0}/{sub_folder1}/adapt_{adapt_mode_save}'
								)
					else:
						model_dir = os.path.join(
								models_dir_,
								f'{restore_from_.split("/")[-2]}/src2trg/{sub_folder0}/{sub_folder1}/adapt_{adapt_mode_save}/seed_{seed_}'
								)

					DONE_name = f'GTA_to_{trg_dataset_name_}_{scene_}_{cond_}.DONE'
					if os.path.isfile(os.path.join(model_dir, DONE_name)) and not force_retraining:
						continue # training already done - abort training

					exp_mode_ = 'train' if mode_=='re_train' else mode_

					print(f'python -u main_adapt.py'+
									f' --mode={exp_mode_}'+
									f' --seed={seed_}'+
									f' --trg_dataset={trg_dataset_}' +
									f' --scene={scene_}' +
									f' --force_retraining={force_retraining}'+
									f' --cond={cond_}' +
									f' --model_arch={model_arch_}' +
									f' --adapt_mode={adapt_mode_}'+
									f' --reset_thrs={reset_thrs_}'+
									f' --batch_norm_momentum={bn_mom_}'+
									f' --restore_from={restore_from_}')

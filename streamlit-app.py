import os
import functools
import itertools
import glob
import time

import pickle
import numpy as np
import numpy.random as npr
import scipy.stats

import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import cv2

SYNTHIA_labels = [
		'road', 'sidewalk', 'building', 'fence', 'pole', 'light',
		'sign', 'vegetation', 'sky', 'person', 'car', 'bicycle']

GTA5_labels_dict = {
		0:"road", 1:"sidewalk", 2:"building", 3:"wall", 4:"fence",
		5:"pole", 6:"light", 7:"sign", 8:"vegetation", 9:"terrain",
		10:"sky", 11:"person", 12:"rider", 13:"car", 14:"truck",
		15:"bus", 16:"train", 17:"motocycle", 18:"bicycle"}


Cityscapes_labels_dict = GTA5_labels_dict
ACDC_labels_dict = Cityscapes_labels_dict

SYNTHIA_labels_dict = dict()
for k in GTA5_labels_dict.keys():
	if GTA5_labels_dict[k] in SYNTHIA_labels:
		SYNTHIA_labels_dict[k] = GTA5_labels_dict[k]

Cityscapes_img_per_seq_dict = {
		'aachen': 174, 'dusseldorf': 221, 'krefeld': 99,
		'ulm': 95, 'bochum': 96, 'erfurt': 109,
		'monchengladbach': 94, 'weimar': 142, 'bremen': 316,
		'hamburg': 248, 'strasbourg': 365, 'zurich': 122,
		'cologne': 154, 'hanover': 196, 'stuttgart': 196,
		'darmstadt': 85, 'jena': 119, 'tubingen': 144,
		'frankfurt':270, 'lindau':62, 'munster':177}

ACDC_img_per_seq_file = './dataset/ACDC_num_images_per_seq/ACDC_img_per_folder.pkl'
with open(ACDC_img_per_seq_file, 'rb') as f:
	ACDC_img_per_seq_dict = pickle.load(f)

Cityscapes_train_seqs = ['aachen', 'bremen', 'darmstadt',
						'erfurt', 'hanover', 'krefeld',
						'strasbourg', 'tubingen', 'weimar',
						'bochum', 'cologne', 'dusseldorf',
						'hamburg', 'jena', 'monchengladbach',
						'stuttgart', 'ulm', 'zurich']

Cityscapes_val_seqs = ['frankfurt', 'lindau', 'munster']

Cityscapes_seqs = Cityscapes_train_seqs + Cityscapes_val_seqs # we don't make any distinction

class StreamPlot():

	#@st.cache
	def __init__(self):

		print('__init__')

		self.model=0
		self.dataset=0
		self.frame=0
		self.results_dir = './adapted-models'

		with open('./dataset/available_benchmarks.pkl', 'rb') as f:
			self.available_benchmarks_list = pickle.load(f)

		self.src_models_list = sorted(
				glob.glob(os.path.join(self.results_dir, '*')))
		self.src_models_list = [i.split('/')[-1] for i in self.src_models_list if os.path.isdir(i)]

		self.src_models_list = [_ for _ in self.src_models_list if _ != '_UNUSED_'] # filtering out old experiments

		self.datasets_list_dict = dict()

		self.benchmarks_list = ['SYNTHIA', 'Cityscapes', 'ACDC']

		for benchmark in self.benchmarks_list:
			print(f'Processing benchmark {benchmark}')
			self.datasets_list_dict[benchmark] = sorted(glob.glob(os.path.join(self.results_dir, f'*/src2trg/{benchmark}*')))
			self.datasets_list_dict[benchmark] = [i for i in self.datasets_list_dict[benchmark] if os.path.isdir(i)]
			self.datasets_list_dict[benchmark] = [i.split('/')[-1] for i in self.datasets_list_dict[benchmark]]
			self.datasets_list_dict[benchmark] = np.unique(self.datasets_list_dict[benchmark])
			self.datasets_list_dict[benchmark] = [_ for _ in self.datasets_list_dict[benchmark] if _ in self.available_benchmarks_list]

		self.labels_per_img_dict = dict()
		self.num_imgs_per_seq_dict = dict()
		#all_datasets = [i for j in self.datasets_list_dict.values() for i in j]

		for benchmark in self.benchmarks_list:
			print(f'Processing benchmark {benchmark}')
			for dts in self.datasets_list_dict[benchmark]:
				try:
					pkl_path = os.path.join(
							f'./dataset/{benchmark}_labels_per_image',
							f'{dts}_labels_per_image.pkl')
					with open(pkl_path, 'rb') as f:
						self.labels_per_img_dict[dts] = pickle.load(f)
				except:
					print(f'Missing labels_per_img for {dts}')

				try:
					for n in range(len(self.labels_per_img_dict[dts])):
						self.labels_per_img_dict[dts][n] = self.labels_per_img_dict[dts][n][self.labels_per_img_dict[dts][n]!=255]
				except:
					print(f'Error in processing labels for {dts}')

				try:
					pkl_path = os.path.join(
							f'./dataset/{benchmark}_num_images_per_seq',
							f'{dts}_num_images_per_seq.pkl')
					with open(pkl_path, 'rb') as f:
						self.num_imgs_per_seq_dict[dts] = pickle.load(f)
				except:
					print(f'Missing num_imgs_per_seq for {dts}')

		self.src_models_list_ids = []

		# defining better looking model names
		for curr_id in self.src_models_list:
			# NOTE this is if we want to give different IDs
			self.src_models_list_ids.append(curr_id)

		self.setup_exp_choices()


	def setup_exp_choices(self):
		all_exps = np.unique([_.split('/')[-1]
				for _ in glob.glob(
						os.path.join(self.results_dir, '*/src2trg/*'))])
		all_exps = all_exps.tolist()

		all_exps = [_ for _ in all_exps if _ in self.available_benchmarks_list]

		self.exp_choices_list = [
				'SYNTHIA_multi', 'SYNTHIA_01', 'SYNTHIA_04', 'SYNTHIA_05',
				'SYNTHIA_daylight', 'SYNTHIA_night', 'SYNTHIA_rain', 'SYNTHIA_fog',
				'Cityscapes_single', 'Cityscapes_multi-AW', 'Cityscapes_multi-O',
				'ACDC_multi', 'ACDC_single-fog', 'ACDC_single-night',
				'ACDC_single-rain', 'ACDC_single-snow']

		self.exp_choices_dict = {k_:[] for k_ in self.exp_choices_list}

		for _, exp_ in enumerate(all_exps):
			scenes_list = exp_.split('_')[1].split('-')
			conds_list = exp_.split('_')[2].split('-')

			if 'SYNTHIA' in exp_:
				if (len(conds_list) > 1) and (len(scenes_list) > 1):
					self.exp_choices_dict['SYNTHIA_multi'].append(exp_)

				elif len(conds_list) == 1 and len(scenes_list) == 1:
					if conds_list[0] in ['SUMMER', 'SPRING', 'FALL', 'WINTER']:
						if conds_list[0] == 'SUMMER' and scenes_list[0] == '04':
							continue
						self.exp_choices_dict['SYNTHIA_daylight'].append(exp_)
					elif conds_list[0] in ['RAIN', 'SOFTRAIN']:
						self.exp_choices_dict['SYNTHIA_rain'].append(exp_)
					elif conds_list[0] in ['FOG']:
						self.exp_choices_dict['SYNTHIA_fog'].append(exp_)
					elif conds_list[0] in ['NIGHT']:
						self.exp_choices_dict['SYNTHIA_night'].append(exp_)
					self.exp_choices_dict[f'SYNTHIA_{scenes_list[0]}'].append(scenes_list[0])

			elif ('Cityscapes' in exp_) and ('FullCityscapes' not in exp_):
				if (len(np.unique(conds_list)) == 1) and (len(np.unique(scenes_list)) > 1):
					self.exp_choices_dict['Cityscapes_multi-O'].append(exp_)
				elif (len(np.unique(conds_list)) > 1) and (len(np.unique(scenes_list)) > 1):
					self.exp_choices_dict['Cityscapes_multi-AW'].append(exp_)
				elif len(conds_list) == 1 and len(scenes_list) == 1:
					self.exp_choices_dict['Cityscapes_single'].append(exp_)

			elif 'ACDC' in exp_:
				if ((len(np.unique(scenes_list)) > 1) and (len(np.unique(conds_list)) > 1)):
					self.exp_choices_dict['ACDC_multi'].append(exp_)
				elif ((len(np.unique(scenes_list)) == 1) and (len(np.unique(conds_list)) == 1)) and (conds_list[0] == 'fog'):
					self.exp_choices_dict['ACDC_single-fog'].append(exp_)
				elif ((len(np.unique(scenes_list)) == 1) and (len(np.unique(conds_list)) == 1)) and (conds_list[0] == 'night'):
					self.exp_choices_dict['ACDC_single-night'].append(exp_)
				elif ((len(np.unique(scenes_list)) == 1) and (len(np.unique(conds_list)) == 1)) and (conds_list[0] == 'rain'):
					self.exp_choices_dict['ACDC_single-rain'].append(exp_)
				elif ((len(np.unique(scenes_list)) == 1) and (len(np.unique(conds_list)) == 1)) and (conds_list[0] == 'snow'):
					self.exp_choices_dict['ACDC_single-snow'].append(exp_)

			else:
				# TODO
				pass


	def define_fig(self, n):
		if n==0:
			raise ValueError('n > 0')
		fig, axes = plt.subplots(n,1)
		if n==1:
			axes = [axes]
		return fig, axes


	def show_image(self, axes, verbose=False):

		img_path = self.saved_images[self.frame]
		if verbose:
			print(f'Loading {img_path}')
		img_to_plot = cv2.imread(img_path)
		axes[0].imshow(img_to_plot[:,:,::-1])
		axes[0].set_xticks([])
		axes[0].set_yticks([])


	def retrieve_adapt_method_data(self):

		self.trg_models_list = glob.glob(
				os.path.join(self.adapt_exps_root, f'adapt_{self.adapt_method_choice}*'))
		self.adapt_method_list = [_.split('/')[-1] for _ in self.trg_models_list]

		string_to_match = f'adapt_{self.adapt_method_choice}_'

		if self.adapt_method_choice in [
				'tent', 'naive-tent', 'class-reset-tent', 'oracle-reset-tent']:

			# exp folder's format: {self.args.adapt_mode}-{self.args.adapt_iters}-{self.args.learning_rate}-{self.src_iters}

			# retrieving args.adapt_iters and filtering
			adapt_iters_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.adapt_iters_choice = st.sidebar.selectbox('Number of steps', adapt_iters_list, 0)

			string_to_match += (self.adapt_iters_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			# retrieving args.learning_rate and filering
			lr_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.lr_choice = st.sidebar.selectbox("Learning rate", lr_list, 0)

			string_to_match += (self.lr_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			# retrieving args.src_iters and filering
			src_iters_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.src_iters_choice = st.sidebar.selectbox("Number of source iterations", src_iters_list, 0)

			string_to_match += (self.src_iters_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			if self.adapt_method_choice in ['class-reset-tent']:
				# retrieving args.batch_norm_momentum and filtering
				bn_momentum_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
				self.bn_momentum_choice = st.sidebar.selectbox("Batch norm momentum", bn_momentum_list, 0)

				string_to_match += (self.bn_momentum_choice+'-')
				self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

				# retrieving args.reset_threshold and filtering
				reset_thrs_list = np.unique([i.replace(string_to_match, '') for i in self.adapt_method_list]).tolist()
				self.reset_thrs_choice = st.sidebar.selectbox("Smart reset threshold", reset_thrs_list, 0)

				string_to_match += self.reset_thrs_choice
				self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match == _]

			else:
				# retrieving args.batch_norm_momentum and filtering
				bn_momentum_list = np.unique([i.replace(string_to_match, '') for i in self.adapt_method_list]).tolist()
				self.bn_momentum_choice = st.sidebar.selectbox("Batch norm momentum", bn_momentum_list, 0)

				string_to_match += self.bn_momentum_choice
				self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match == _]


		elif self.adapt_method_choice in ['pseudo-labels', 'naive-pseudo-labels',
										'class-reset-pseudo-labels', 'oracle-reset-pseudo-labels']:

			# retrieving args.adapt_iters and filtering
			adapt_iters_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.adapt_iters_choice = st.sidebar.selectbox('Number of steps', adapt_iters_list, 0)

			string_to_match += (self.adapt_iters_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			# retrieving args.learning rate and filering
			lr_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.lr_choice = st.sidebar.selectbox("Learning rate", lr_list, 0)

			string_to_match += (self.lr_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			# retrieving args.src_iters and filtering
			src_iters_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.src_iters_choice = st.sidebar.selectbox("Number of source iterations", src_iters_list, 0)

			string_to_match += (self.src_iters_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			# retrieving args.pseudo_labels_mode and filtering
			pl_mode_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.pl_mode_choice = st.sidebar.selectbox("Pseudo-labels mode", pl_mode_list, 0)

			string_to_match += (self.pl_mode_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			# retrieving args.src_iters and filtering
			pl_threshold_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.pl_threshold_choice = st.sidebar.selectbox("Pseudo-labels thresholds", pl_threshold_list, 0)

			string_to_match += (self.pl_threshold_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			if self.adapt_method_choice in ['class-reset-pseudo-labels']:
				# retrieving args.batch_norm_momentum and filtering
				bn_momentum_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
				self.bn_momentum_choice = st.sidebar.selectbox("Batch norm momentum", bn_momentum_list, 0)

				string_to_match += (self.bn_momentum_choice+'-')
				self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

				# retrieving args.reset_threshold and filtering
				reset_thrs_list = np.unique([i.replace(string_to_match, '') for i in self.adapt_method_list]).tolist()
				self.reset_thrs_choice = st.sidebar.selectbox("Smart reset threshold", reset_thrs_list, 0)

				string_to_match += self.reset_thrs_choice
				self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match == _]

			else:
				# retrieving args.batch_norm_momentum and filtering
				bn_momentum_list = np.unique([i.replace(string_to_match, '') for i in self.adapt_method_list]).tolist()
				self.bn_momentum_choice = st.sidebar.selectbox("Batch norm momentum", bn_momentum_list, 0)

				string_to_match += self.bn_momentum_choice
				self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match == _]

		elif self.adapt_method_choice in ['batch-norm', 'naive-batch-norm']:

			# exp folder's format: {self.args.adapt_mode}-{self.args.batch_norm_momentum}

			# retrieving args.batch_norm_momentum and filtering
			bn_momentum_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.bn_momentum_choice = st.sidebar.selectbox('BN momentum', bn_momentum_list, 0)

			# different format, otherwise issues with values that are contained in others, like 0.1 and 0.15 or 0.0 and 0.01
			self.adapt_method_list = [_ for _ in self.adapt_method_list if _.split('_')[-1] == self.bn_momentum_choice]


		elif 'no-adaptation' == self.adapt_method_choice:
			self.adapt_method_list = ['adapt_no-adaptation']

		elif 'style-transfer' == self.adapt_method_choice:

			# retrieving style transfer method and filtering
			style_transfer_method_list = np.unique([i.replace(string_to_match, '').split('-')[0] for i in self.adapt_method_list]).tolist()
			self.style_transfer_method_choice = st.sidebar.selectbox('Style transfer method', style_transfer_method_list, 0)

			string_to_match += (self.style_transfer_method_choice + '-')
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match in _]

			# retrieving sample selection method and filtering
			sample_sel_method_list = np.unique([i.replace(string_to_match, '') for i in self.adapt_method_list]).tolist()
			self.sample_sel_method_choice = st.sidebar.selectbox("Sample selection method", sample_sel_method_list, 0)

			string_to_match += self.sample_sel_method_choice
			self.adapt_method_list = [_ for _ in self.adapt_method_list if string_to_match == _]

		else:
			raise ValueError(f'Unknown adapt_method_choice [{self.adapt_method_choice}]')

		# now the list need to be a single element, otherwise we're missing some parameters
		assert len(self.adapt_method_list) == 1
		self.method_name = self.adapt_method_list[0]

		self.trg_model_dir = os.path.join(self.adapt_exps_root, self.method_name)

		# path of summary for chosen experiment 
		self.current_exp_summary = glob.glob(os.path.join(self.trg_model_dir, '*summary.pkl'))[0]


	def get_user_input_sidebar(self):

		self.src_model_id_choice = st.sidebar.selectbox("Models", self.src_models_list_ids, 0)
		src_model_index = self.src_models_list_ids.index(self.src_model_id_choice)
		self.src_model_choice = self.src_models_list[src_model_index]

		self.benchmark_choice = st.sidebar.selectbox("Benchmark", self.benchmarks_list, 0)

		self.datasets_list = self.datasets_list_dict[self.benchmark_choice]

		self.set = st.sidebar.selectbox("Set", ['single', 'multi'], 0)
		if self.set == 'single':
			self.datasets_list = [i for i in self.datasets_list if len(i.split('_')[1].split('-')) == 1]
		elif self.set == 'multi':
			self.datasets_list = [i for i in self.datasets_list if len(i.split('_')[1].split('-')) > 1]
		else:
			raise ValueError(f'Unknown experiment set [{self.set}]')

		if len(self.datasets_list) == 0:
			st.sidebar.write('No experiments for this configuration!')
			return False

		self.dataset_choice = st.sidebar.selectbox("Dataset", self.datasets_list, 0)

		train_list = [i.split('/')[-1] for i in glob.glob(
				os.path.join(self.results_dir, self.src_model_choice,
							'src2trg', self.dataset_choice, '*'))]

		self.train_choice = st.sidebar.selectbox("Parameters trained", train_list, 0)

		self.adapt_exps_root = os.path.join(
				self.results_dir, self.src_model_choice, 'src2trg',
				self.dataset_choice, self.train_choice)

		self.trg_models_list = glob.glob(
				os.path.join(self.adapt_exps_root, '*/*DONE*'))

		# specific folders are in format 'adapt_METHOD-NAME(_HYPER-PARAMS)'
		adapt_method_list = np.unique([i.split('/')[-2].split('adapt_')[-1].split('_')[0]
				for i in self.trg_models_list]).tolist()

		if len(adapt_method_list) == 0:
			st.sidebar.write('No models learned for this configuration!')
			return False

		self.adapt_method_choice = st.sidebar.selectbox("Adapt method", adapt_method_list, 0)

		######## retrieve directories and summaries from selected adapt method
		self.retrieve_adapt_method_data()

		return True


	def setup_image_slider(self):

		# retrieving saved images
		self.saved_images = sorted(
				glob.glob(os.path.join(self.trg_model_dir, f'results/*concat.png')))

		st.sidebar.write('----')
		st.sidebar.markdown('**Frame selector (for displaying)**')
		self.frame = st.sidebar.slider(f"Choose a frame for {self.dataset}", 0,
										len(self.saved_images)-1, 0)

		fig, axes = self.define_fig(1)
		self.show_image(axes)
		self.col1.pyplot(fig)


	def load_and_compute_results(self, summary_file):

		with open(summary_file, 'rb') as f:
			res_dict = pickle.load(f)

		# Computing metrics for avg-image based mesures
		miou_init_list = res_dict['mious_init']
		miou_final_list = res_dict['mious_final']
		pixel_acc_init_list = res_dict['pixel_acc_init']
		pixel_acc_final_list = res_dict['pixel_acc_final']
		mean_acc_init_list = res_dict['mean_acc_init']
		mean_acc_final_list = res_dict['mean_acc_final']

		dataset_choice = summary_file.split('/')[-4] # note, self.dataset_choice does not work well for cross_exp mode

		try:
			for i in range(len(miou_init_list)):
				img_gt = self.labels_per_img_dict[dataset_choice][i].astype(int)
				miou_init_list[i][np.delete(np.arange(19), img_gt)] = np.nan
				miou_final_list[i][np.delete(np.arange(19), img_gt)] = np.nan
			# averaging mious for different classes
			avg_miou_init_list = [np.nanmean(miou) for miou in miou_init_list]
			avg_miou_final_list = [np.nanmean(miou) for miou in miou_final_list]
		except:
			print('Error retrieving image GT classes - setting np.nan list')
			miou_init_list = [np.nan for miou in miou_init_list]
			miou_final_list = [np.nan for miou in miou_final_list]
			avg_miou_init_list = [np.nan for miou in miou_init_list]
			avg_miou_final_list = [np.nan for miou in miou_final_list]

		self.current_seq_pixel_acc_init = pixel_acc_init_list
		self.current_seq_pixel_acc_final = pixel_acc_final_list
		self.current_seq_mean_acc_init = mean_acc_init_list
		self.current_seq_mean_acc_final = mean_acc_final_list
		self.current_seq_miou_init = miou_init_list
		self.current_seq_miou_final = miou_final_list

		# this is for FullCityscapes and other datasets for which some frames don't have GT
		available_indices = res_dict['iter']

		return pixel_acc_init_list, pixel_acc_final_list, avg_miou_init_list, avg_miou_final_list, available_indices


	def print_current_exp_results(self):

		self.col2.write('NOTE: not computing whole dataset\'s numbers')

		outputs = self.load_and_compute_results(self.current_exp_summary)

		pixel_acc_init_list = outputs[0]
		pixel_acc_final_list = outputs[1]
		miou_init_list = outputs[2]
		miou_final_list = outputs[3]
		available_indices = outputs[4]

		self.col2.write('----')
		self.col2.markdown('**Metrics computed by averaging single images**')
		self.col2.write(f'Average mIoUs    : {round(np.mean(miou_init_list) * 100., 2)} '+\
					f' --> {round(np.mean(miou_final_list) * 100., 2)} | Selected image '+\
					f'{round(miou_init_list[self.frame]*100., 2)} --> '+\
					f'{round(miou_final_list[self.frame]*100., 2)}')

		self.col2.write(f'Average PixelAcc : {round(np.mean(self.current_seq_pixel_acc_init), 2)} '+\
					f' --> {round(np.mean(self.current_seq_pixel_acc_final), 2)} | Selected image '+\
					f'{round(self.current_seq_pixel_acc_init[self.frame], 2)} -->'+\
					f'{round(self.current_seq_pixel_acc_final[self.frame], 2)}')

		self.col2.write(f'Average MeanAcc  : {round(np.mean(self.current_seq_mean_acc_init)* 100., 2)} '+\
					f'--> {round(np.mean(self.current_seq_mean_acc_final) * 100., 2)} | Selected image '+\
					f'{round(self.current_seq_mean_acc_init[self.frame]*100., 2)} --> '+\
					f'{round(self.current_seq_mean_acc_final[self.frame]*100., 2)}')

		self.col2.write('----')
		for i in range(19):
			self.col2.write(f'**{GTA5_labels_dict[i]}**:'+\
						f'{self.current_seq_miou_init[self.frame][i]*100.:.2f}'+\
						f'--> {self.current_seq_miou_final[self.frame][i]*100.:.2f}')


	def plot_seq_results(self):

		st.sidebar.write('Plotting options')

		_, _, miou_init_list, miou_final_list, _, = self.load_and_compute_results(self.current_exp_summary)

		num_seqs = len(self.dataset_choice.split('_')[1].split('-'))
		seqs_len = len(miou_final_list)//num_seqs

		y_lim_min = st.sidebar.text_input("ylim (min)", "0.0")
		y_lim_max = st.sidebar.text_input("ylim (max)", "1.0")

		if st.sidebar.button("Init sequence plot sequence", key="A"):
			self.plt_dict['fig'], self.plt_dict['ax'] = plt.subplots(1,1)#,figsize=(5,5))
			self.plt_dict['ax'].set_title("mIoU (average over classes in images)")
			self.plt_dict['ax'].set_ylim([float(y_lim_min), float(y_lim_max)])
			self.plt_dict['ax'].set_yticks(np.arange(float(y_lim_min), float(y_lim_max), step=0.1))

			self.plt_dict['miou_final_ma']=[]
			self.plt_dict['method_name']=[]
			self.col1.pyplot(self.plt_dict['fig'])

		color_list = ['tab:blue', 'tab:orange', 'tab:green',
					'firebrick', 'orchid', 'tab:pink', 'tab:cyan'
					]
		color = st.sidebar.selectbox("Color", color_list, 0)
		l_width = st.sidebar.text_input("Line width", "1.0")
		ma_size = int(st.sidebar.text_input('Moving average size', '5'))
		if st.sidebar.button("Add method to plot", key="A"):
			miou_final_ma = [np.mean(miou_final_list[i-ma_size:i])
							for i in range(len(miou_final_list))]
			self.plt_dict['miou_final_ma'].append(miou_final_ma)
			line, = self.plt_dict['ax'].plot(miou_final_ma, linewidth=float(l_width), c=color)

			line.set_label(self.method_name)
			self.plt_dict['ax'].set_ylim([float(y_lim_min), float(y_lim_max)])
			self.plt_dict['ax'].set_yticks(np.arange(float(y_lim_min), float(y_lim_max), step=0.1))
			self.plt_dict['ax'].set_title("mIoU (average over classes in images)")

			self.plt_dict['ax'].legend(
					loc='lower center', fancybox=False, shadow=False, ncol=1,
					bbox_to_anchor=(0.5, -0.5))

			cum_subseq_len = 0
			for i in range(num_seqs-1):
				if self.benchmark_choice == 'SYNTHIA':
					self.plt_dict['ax'].axvline(x=seqs_len * (i+1), c='k', linewidth=0.25)
				elif self.benchmark_choice == 'Cityscapes':
					subseq = self.dataset_choice.lstrip('Cityscapes_').split('-')[i]
					subseq_len = Cityscapes_img_per_seq_dict[subseq]
					cum_subseq_len += subseq_len
					self.plt_dict['ax'].axvline(x=cum_subseq_len, c='k', linewidth=0.25)
				elif self.benchmark_choice == 'ACDC':
					subseq = self.dataset_choice.lstrip('ACDC_').split('-')[i]
					subseq_len = ACDC_img_per_seq_dict[subseq]
					print(f'Subseq {subseq} : {subseq_len}')
					cum_subseq_len += subseq_len
					self.plt_dict['ax'].axvline(x=cum_subseq_len, c='k', linewidth=0.25)
				else:
					raise ValueError('Unknown benchmark.')

			self.col1.pyplot(self.plt_dict['fig'])

		if st.sidebar.checkbox("Plot", key="A"):
			self.col1.pyplot(self.plt_dict['fig'])

		file_name = self.col1.text_input('File name', 'plot')
		if self.col1.button('Save plot as pdf'):
			print('Saving plot')
			plt.savefig(f'{file_name}.pdf')



	def compute_res_for_table(self):
		st.sidebar.write('------------')
		st.sidebar.write('**Retrieve and process results for all datasets (Table 1)**')
		self.miou_all_exps_dict = dict()
		if st.sidebar.button('Compute results',  key='A'):

			print('Retrieving all summaries (may take a while)')
			self.all_summaries = glob.glob(os.path.join(
					self.results_dir, self.src_model_choice,
					'src2trg/*/*/*/*_summary.pkl'))
			print('Done!')

			table_exp_choices = ['Cityscapes_multi-AW', 'Cityscapes_multi-O',
								'ACDC_multi', 'SYNTHIA_multi']

			for table_exp_choice in table_exp_choices:
				print(f'Computing results for {table_exp_choice}')
				self.table_exp_summaries = [_ for _ in self.all_summaries
						if _.split('/')[3] in self.exp_choices_dict[table_exp_choice]]

				self.miou_current_exp_dict = {k_:{}
						for k_ in np.unique([_.split('/')[-3] + '/' + _.split('/')[-2]
								for _ in self.table_exp_summaries])}
				print('Retrieving results')
				for n, summary in enumerate(self.table_exp_summaries):
					method_name_ = summary.split('/')[-3] + '/' + summary.split('/')[-2]
					dataset_name_ = summary.split('/')[3]
					if (n%100)==0:
						print(n,len(self.table_exp_summaries))
					outputs = self.load_and_compute_results(summary)
					_, _, _, miou_final_list, _ = outputs
					miou_all = np.mean(miou_final_list)
					if len(self.miou_current_exp_dict[method_name_]) == 0:
						print(method_name_, miou_all)
					self.miou_current_exp_dict[method_name_][dataset_name_] = miou_all
				self.miou_all_exps_dict[table_exp_choice] = self.miou_current_exp_dict
			with open(f"./tex_tables/new_results.pkl", "wb") as f:
				print('Saving')
				pickle.dump(self.miou_all_exps_dict, f)
			print('Done')


def main():

	print('\n---------------')
	st.set_page_config(
			layout='wide', page_title='OASIS')

	@st.cache(allow_output_mutation=True)
	def define_object():
		return StreamPlot()

	@st.cache(allow_output_mutation=True)
	def init_dict():
		return {'ax':None, 'fig':None, 'miou_final_ma':[], 'method_name':[],
				'acc_final':[]}

	sp = define_object()

	sp.plt_dict = init_dict()

	# page payout
	st.title('Online Unsupervised Domain Adaptation (OUDA)')
	sp.col1, sp.col2 = st.columns(2)

	# setup sidebar and get user input
	st.sidebar.markdown('**Model/Dataset specifics**')
	found_exps = sp.get_user_input_sidebar()

	if not found_exps:
		print('No experiment found!')
		return -1

	print(f'Source model: {sp.src_model_choice} | Adapt method: {sp.method_name}')
	print(f'{sp.trg_model_dir}')

	# setup slider to explore images
	print('Setting up image slider')
	sp.setup_image_slider()

	# compute and print average results for this experiment
	print('Computing and printing exp results')
	sp.print_current_exp_results()

	# plot sequence results
	print('Plotting')
	st.sidebar.write('----')
	st.sidebar.markdown('**Plot sequences**')
	if st.sidebar.checkbox('See panel', key='A'):
		sp.plot_seq_results()

	# computing all results associated with selected exp and making tex table
	sp.compute_res_for_table()



if __name__=='__main__':
	main()

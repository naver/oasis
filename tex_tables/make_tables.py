import pickle
import pdb

import numpy as np
import pandas as pd


def make_main_table(
		data_dict, selected_methods, selected_exps,
		include_avg_column=True):

	print('\n\n\nMaking table')

	validated_methods = [
						'No adaptation',
						'N-BN-0.1-None-None-only_batch_norm_params',
						'C-BN-0.1-None-None-only_batch_norm_params',
						'N-TENT-1-0-1.0-None-0.1-only_batch_norm_params',
						'C-TENT-1-0-0.01-None-0.1-only_batch_norm_params',
						'C-TENT-SR-1-1-0.01-None-0.1-only_batch_norm_params',
						'C-TENT-SR-1-2-0.01-None-0.1-only_batch_norm_params',
						'Class-N-TENT-1-0-0.1-1.0-0.1-only_batch_norm_params',
                        'Oracle-N-TENT-1-0-1.0-None-0.1-only_batch_norm_params',
						'N-PL-vanilla-0.0-1-0-0.0001-0.1-None-whole_net',
						'C-PL-vanilla-0.0-1-0-0.0001-0.1-None-whole_net',
						'C-PL-SR-vanilla-0.0-1-2-0.0001-0.1-None-whole_net',
						'Class-N-PL-vanilla-0.0-1-0-0.0001-0.1-0.5-whole_net',
						'Oracle-N-PL-vanilla-0.0-1-0-0.0001-0.1-None-whole_net',
						'ST_wct2-random',
						'ST_wct2-nn',
						]

	### TODO: validated_methods += YOUR_METHODS

	res_for_table_dict = dict()
	res_for_avg_column_dict = dict()

	sub_folders_list = []

	for exp_key in selected_exps:
		for sel_method_ in selected_methods:
			for n, method_ in enumerate(data_dict['method']):

				if data_dict['experiment'][n] != exp_key:
					continue

				if method_ != sel_method_:
					continue

				bn_mom = data_dict['batch_norm_momentum'][n]
				lr = data_dict['learning_rate'][n]
				reset_thrs = data_dict['reset_thrs'][n]
				pl_mode = data_dict['pseudo_labels_mode'][n]
				pl_thrs = data_dict['pseudo_labels_thrs'][n]
				src_iters = data_dict['src_iters'][n]
				adapt_iters = data_dict['adapt_iters'][n]
				params_trained = data_dict['params_trained'][n]

				if 'PL' in sel_method_:
					if pl_mode != 'vanilla':
						continue

				if 'BN' in sel_method_:
					key_ = f'{sel_method_}-{bn_mom}-{params_trained}'

				elif 'TENT' in sel_method_:
					key_ = f'{sel_method_}-{adapt_iters}-{src_iters}-{lr}-'+\
							f'{reset_thrs}-{bn_mom}-{params_trained}'

				elif 'PL' in sel_method_:
					key_ = f'{sel_method_}-{pl_mode}-{pl_thrs}-{adapt_iters}-'+\
							f'{src_iters}-{lr}-{bn_mom}-{reset_thrs}-'+\
							f'{params_trained}'

				elif sel_method_ == 'No adaptation':
					key_=f'{sel_method_}'

				elif 'ST' in sel_method_:
					key_=f'{sel_method_}'

				### TODO add your method here.

				else:
					raise RuntimeError(f'Unknown adapt_method [{sel_method_}]')

				if key_ not in validated_methods:
					continue

				if data_dict['sub_folder'][n] not in sub_folders_list:
					sub_folders_list.append(data_dict['sub_folder'][n])

				if key_ not in res_for_table_dict:
					res_for_table_dict[key_] = dict()
				if key_ not in res_for_avg_column_dict:
					res_for_avg_column_dict[key_] = []

				avg_results_type = 'avg_results'
				std_results_type = 'std_results'
				all_results_type = 'all_results'
				avg_res = float(data_dict[avg_results_type][n])
				std_res = float(data_dict[std_results_type][n])
				tex_line = f'$+ {avg_res*100.:.1f} \\% $ \\tiny {{$ \\pm  {std_res*100:.1f} $ }}'
				res_for_table_dict[key_][exp_key] = tex_line
				res_for_avg_column_dict[key_] += data_dict[all_results_type][n]

	available_exps = []

	for method_ in res_for_table_dict.keys():
		for exp_type_ in res_for_table_dict[method_].keys():
			if exp_type_ not in available_exps:
				available_exps.append(exp_type_)

	available_exps = sorted(available_exps)

	for method_ in res_for_table_dict.keys():
		for exp_type_ in available_exps:
			if exp_type_ not in res_for_table_dict[method_]:
				res_for_table_dict[method_][exp_type_] = '-'

	exp_counter_dict = dict()
	if include_avg_column:
		avg_column_dict = {
					k_: f'$+{np.nanmean(res_for_avg_column_dict[k_])*100.:.1f} '+\
						f'\\% $ \\tiny{{$\\pm {np.nanstd(res_for_avg_column_dict[k_])*100.:.1f}'+\
						f' $}} ({len(res_for_avg_column_dict[k_])})'
							for k_ in res_for_avg_column_dict.keys()}
		for method_ in res_for_table_dict.keys():
			res_for_table_dict[method_]['Avg.'] = avg_column_dict[method_]
			exp_counter_dict[method_] = len(res_for_avg_column_dict[method_])
		available_exps.append('Avg.')

	table_name = 'OASIS'

	print('\\begin{table*}[t]')
	print('\\begin{center}')
	print('{\scriptsize')
	print('\\setlength{\\tabcolsep}{3.5pt}')
	print('\\begin{tabular}{@{}l'+'c'*len(available_exps)+'@{}}')
	print('\\multicolumn {'+str(len(available_exps)+1)+'} {c}{\\textbf{' + table_name + '}} \\\\')
	print('\\toprule')
	print('& \\multicolumn{'+str(len(available_exps))+'}{c}{\\textbf{Sequence type}} \\\\')
	print('\\cmidrule(r){2-'+str(len(available_exps)+1)+'}')
	print('\\textbf{Method} & ' + ' & '.join(available_exps) + '\\\\')
	print('\\midrule')
	for method_ in sorted(res_for_table_dict.keys()):
		tex_string = f" {method_.replace('_', '-')}"
		for exp_type_ in available_exps:
			tex_string +=f' & {res_for_table_dict[method_][exp_type_]}'
		tex_string += '\\\\'
		print(tex_string)
		print('\\midrule')
	print('\\bottomrule')
	print('\\end{tabular}')
	print('}')
	print('\\end{center}')
	print('\\caption{ }')
	print('\\label{ }')
	print('\\end{table*}')


if __name__ == '__main__':

	with open('cvpr_table1.pkl', 'rb') as f:
		res_dict_all  = pickle.load(f)

	### TODO: uncomment when you have computed your results! ###
	#with open('new_results.pkl', 'rb') as f:
	#	res_dict_new  = pickle.load(f)
	###	

	no_adapt_seqs = [k_ for q_ in res_dict_all.keys()
			for k_ in res_dict_all[q_]['no_adaptation/adapt_no-adaptation'].keys()]

	### TODO: add here any new hyper-parameters from your method!
	data_dict = {
				'sub_folder':[],
				'experiment':[],
				'params_trained':[],
				'method':[],
				'batch_norm_momentum':[],
				'reset_thrs':[],
				'adapt_iters':[],
				'learning_rate':[],
				'src_iters':[],
				'reset_thrs':[],
				'pseudo_labels_mode':[],
				'pseudo_labels_thrs':[],
				'avg_results':[],
				'std_results':[],
				'all_results':[],
				}

	for exp_ in res_dict_all.keys():
		res_dict = res_dict_all[exp_]
		res_dict_keys = list(res_dict.keys())

		### TODO: also loop over your experiments in res_dict_new,
		### for example running res_dict.update(res_dict_new[exp_]),
		### and include the code to load results according to any
		### specific hyper-parameters you may have introduced.
		### It is important than all experiments have run.

		for k_ in res_dict_keys:
			params_trained, adapt_method = k_.split('/')

			try:
				_, method, details = adapt_method.split('_')
				details_list = details.split('-')
			except:
				_, method = adapt_method.split('_')


			if method in [
					'batch-norm', 'naive-batch-norm',
					'oracle-reset-batch-norm', 'smart-reset-batch-norm']:

				batch_norm_momentum = details_list[0]
				adapt_iters = None
				learning_rate = None
				src_iters = None
				pseudo_labels_mode = None
				pseudo_labels_thrs = None
				if method.startswith('smart-reset'):
					reset_thrs = details_list[-1]
				else:
					reset_thrs = None

				if method == 'batch-norm':
					method_ = 'C-BN'
				elif method == 'naive-batch-norm':
					method_ = 'N-BN'
				elif method == 'oracle-reset-batch-norm':
					method_ = 'Oracle-N-BN'
				elif method == 'smart-reset-batch-norm':
					method_ = 'Class-N-BN'
				else:
					raise RuntimeError('Unknown method')

			elif method in [
					'tent', 'naive-tent',
					'smart-reset-tent', 'oracle-reset-tent']:

				batch_norm_momentum = details_list[3]
				adapt_iters = details_list[0]
				learning_rate = details_list[1]
				src_iters = details_list[2]
				pseudo_labels_mode = None
				pseudo_labels_thrs = None
				if method.startswith('smart-reset'):
					reset_thrs = details_list[-1]
				else:
					reset_thrs = None

				if method == 'tent':
					method_ = 'C-TENT'
				elif method == 'naive-tent':
					method_ = 'N-TENT'
				elif method == 'oracle-reset-tent':
					method_ = 'Oracle-N-TENT'
				elif method == 'smart-reset-tent':
					method_ = 'Class-N-TENT'
				else:
					raise RuntimeError('Unknown method')
				if int(src_iters) > 0:
					method_ += '-SR'

			elif method in [
					'pseudo-labels', 'naive-pseudo-labels',
					'smart-reset-pseudo-labels', 'oracle-reset-pseudo-labels']:

				batch_norm_momentum = details_list[5]
				adapt_iters = details_list[0]
				learning_rate = details_list[1]
				src_iters = details_list[2]
				pseudo_labels_mode = details_list[3]
				pseudo_labels_thrs = details_list[4]

				if method.startswith('smart-reset'):
					reset_thrs = details_list[-1]
				else:
					reset_thrs = None

				if method == 'pseudo-labels':
					method_ = 'C-PL'
				elif method == 'naive-pseudo-labels':
					method_ = 'N-PL'
				elif method == 'oracle-reset-pseudo-labels':
					method_ = 'Oracle-N-PL'
				elif method == 'smart-reset-pseudo-labels':
					method_ = 'Class-N-PL'
				else:
					raise RuntimeError('Unknown method')
				if int(src_iters) > 0:
					method_ += '-SR'

			elif method == 'no-adaptation':
				batch_norm_momentum = None
				adapt_iters = None
				learning_rate = None
				src_iters = None
				pseudo_labels_mode = None
				pseudo_labels_thrs = None
				reset_thrs = None
				method_ = 'No adaptation'

			elif method == 'style-transfer':
				batch_norm_momentum = None
				adapt_iters = None
				learning_rate = None
				src_iters = None
				pseudo_labels_mode = None
				pseudo_labels_thrs = None
				reset_thrs = None
				method_ = 'ST_'+details

			data_dict['params_trained'].append(params_trained)
			data_dict['batch_norm_momentum'].append(batch_norm_momentum)
			data_dict['adapt_iters'].append(adapt_iters)
			data_dict['learning_rate'].append(learning_rate)
			data_dict['src_iters'].append(src_iters)
			data_dict['pseudo_labels_mode'].append(pseudo_labels_mode)
			data_dict['pseudo_labels_thrs'].append(pseudo_labels_thrs)
			data_dict['reset_thrs'].append(reset_thrs)
			data_dict['experiment'].append(exp_)
			data_dict['sub_folder'].append(k_)
			data_dict['method'].append(method_)

			res_list = []
			for q_ in res_dict[k_].keys():
				res = res_dict[k_][q_]
				adapt_res = res_dict['no_adaptation/adapt_no-adaptation'][q_]
				perc_gain_res = (res - adapt_res)/adapt_res
				res_list.append(perc_gain_res)

			avg_results = np.nanmean(res_list)
			std_results = np.nanstd(res_list)

			data_dict['all_results'].append(res_list)
			data_dict['avg_results'].append(avg_results)
			data_dict['std_results'].append(std_results)

	selected_methods = [
			'C-PL', 'N-PL', 'C-PL-SR', 'Oracle-N-PL', 'Class-N-PL',
			'Dist-N-PL', 'C-TENT', 'N-TENT', 'C-TENT-SR', 'Oracle-N-TENT',
			'Class-N-TENT', 'Dist-N-TENT', 'C-BN', 'N-BN', 'No adaptation',
			'ST_wct2-random', 'ST_wct2-nn']

	### TODO: selected_methods += YOUR_METHOD_FROM_RES_DICT_NEW

	selected_exps = [
			'SYNTHIA_multi','ACDC_multi',
			'Cityscapes_multi-O',	'Cityscapes_multi-AW']

	make_main_table(data_dict, selected_methods, selected_exps, False)

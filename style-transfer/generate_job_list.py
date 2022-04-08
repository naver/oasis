import os
import glob
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='SYNTHIA')
parser.add_argument('--style_folder', type=str, default='/PATH/TO/STYLE/FOLDER/')
parser.add_argument('--selection_mode', type=str, default='random', choices=['random', 'nearest'])
config = parser.parse_args()

if config.dataset.upper() == 'SYNTHIA':
	content_folders = glob.glob('/PATH/TO/*/ALL/*/DSET/FOLDERS/')
	content_folder = [x for x in content_folder if '/train_extra/' not in x]

elif config.dataset.upper() == 'ACDC':
	content_folders = []
	content_folders += glob.glob('/PATH/TO/*/ALL/*/DSET/FOLDERS/')

elif config.dataset.upper() == 'Cityscapes'.upper():
	content_folders = glob.glob('/PATH/TO/*/ALL/*/DSET/FOLDERS/')

elif config.dataset.upper() == 'WeatherCityscapes'.upper():
	content_folders = []
	content_folders += glob.glob('/PATH/TO/*/ALL/*/DSET/FOLDERS/')


# Create .sh file to launch jobs
launch_file = f'launch_style_transfer_jobs_{config.dataset.upper()}_{config.selection_mode}_samples.sh'
with open(launch_file, "w") as f:
	f.write('source ~/anaconda3/etc/profile.d/conda.sh' + '\n' )
	f.write('conda activate pt171' + '\n') # Pytorch 1.7.1 used (but should be compatible with other versions)
	f.write('' + '\n')
	info_folder = '/FOLDER/TO/STORE/stdout/stderr/FILES'
	for content_folder in content_folders:
		if config.selection_mode == 'random':
			output_folder = content_folder.replace('/data/', '/data/STYLE_TRANSFER_CVPR2022/WCT2/Random_samples/')
		elif config.selection_mode == 'nearest':
			output_folder = content_folder.replace('/data/', '/data/STYLE_TRANSFER_CVPR2022/WCT2/Nearest_samples/')
			style_features = '/FOLDER/OF/STYLE/FEATURES/SINGLEFILE'

		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		else:
			warnings.warn(f'Folder {output_folder} already exists!')

		# Change this to work for all dsets and create directory
		style_file_list = output_folder + '_style_file_list/'
		if not os.path.exists(style_file_list):
			os.makedirs(style_file_list)
		style_file_list = style_file_list + 'style_file_list.txt'
		f.write('sbatch -p gpu-be --wrap="')
		f.write(f'CUDA_VISIBLE_DEVICES=0 python dataset_style_transfer.py --content {content_folder} ' + 
				f'--style {config.style_folder} --style_file_list {style_file_list} ' +
				f'--selection_mode {config.selection_mode} --style_features {style_features} ' +
				f'--output {output_folder} -e -d -s --verbose" \\' + '\n')
		f.write(f'--gres=gpu:1 --mem=64000 --cpus-per-task=4 --job-name=style_transfer \\' + '\n')
		f.write(f'--output={info_folder}%j.stdout \\' + '\n')
		f.write(f'--error={info_folder}%j.stderr \\' + '\n')
		f.write(f'--constraint="gpu_v100"' + '\n')
		f.write('' + '\n')

print(f'Created {len(content_folders)} jobs')



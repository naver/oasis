import os
import argparse
import glob
import random
from transfer import *
import torch.nn as nn


IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG',
]


def select_style_img(style_files, content_file, config, style_features):
	if config.selection_mode == 'random':
		return random.choice(style_files)
	elif config.selection_mode == 'nearest':
		content_feature = extract_image_features(content_file, config)
		print(f'Finding nearest neighbour for image {content_file}')
		return find_nearest_neighbour(content_feature, style_features, style_files)


def find_nearest_neighbour(content_feature, style_features, style_files):
	# Define cosine similarity operation from pytorch
	cos = nn.CosineSimilarity()
	with torch.no_grad():
		dist = cos(content_feature.unsqueeze(0), style_features)

	return style_files[dist.argmax()]


def extract_image_features(image_file, config):
	device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
	device = torch.device(device)

	transfer_at = set()
	if config.transfer_at_encoder:
		transfer_at.add('encoder')
	if config.transfer_at_decoder:
		transfer_at.add('decoder')
	if config.transfer_at_skip:
		transfer_at.add('skip')

	wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool,
				device=device, verbose=config.verbose)

	image, _, _ = open_image(image_file)
	image_feat = image.to(device)
	image_skips = {}
	with torch.no_grad():
		for level in [1, 2, 3, 4]:
			image_feat = wct2.encode(image_feat, image_skips, level)

		image_feat = torch.mean(image_feat.view(image_feat.size(0), image_feat.size(1), -1), dim=2).flatten()
	return image_feat


def create_style_file_list(config):
	device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
	device = torch.device(device)

	folder_content = config.content
	folder_style = config.style
	style_file_list = config.style_file_list

	# Fix random seed when choosing images?
	assert os.path.exists(folder_content)
	assert os.path.exists(folder_style)

	content_files = sorted(glob.glob(f'{folder_content}/*'))
	content_files = [x for x in content_files if is_image_file(x)]
	style_files = sorted(glob.glob(f'{folder_style}/*'))
	style_files = [x for x in style_files if is_image_file(x)]
	style_features = None

	if os.path.exists(style_file_list):
		file_list, _ = parse_style_file_list(style_file_list)
		if len(content_files) == len(file_list):
			print(f'Style file {style_file_list} already exists. Skipping this step.')
			return
		else:
			print(f'Style file already exists, but is missing images. Remaking style file in {style_file_list}')
			os.remove(style_file_list)
	else:
		print(f'Creating style_file_list in {style_file_list}')

	if config.selection_mode == 'nearest':
		print('Loading style features...')
		style_features = torch.load(config.style_features).to(device)
		assert len(style_files) == style_features.shape[0]

	with open(style_file_list, "w") as f:
		for ii, content_file in enumerate(content_files):
			style_file = select_style_img(style_files, content_file, config, style_features)
			f.write(f'{content_file}   ----   {style_file}' + '\n' )



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--content', type=str, default=None)
	parser.add_argument('--content_segment', type=str, default=None)
	parser.add_argument('--style', type=str, default=None)
	parser.add_argument('--style_segment', type=str, default=None)
	parser.add_argument('--style_file_list', type=str, default=None)
	parser.add_argument('--output', type=str, default=None)
	parser.add_argument('--image_size', type=int, default=None)
	parser.add_argument('--alpha', type=float, default=1)
	parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
	parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
	parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
	parser.add_argument('-s', '--transfer_at_skip', action='store_true')
	parser.add_argument('-a', '--transfer_all', action='store_true')
	parser.add_argument('--cpu', action='store_true')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--selection_mode', type=str, default='random')
	parser.add_argument('--style_features', type=str, default=None,
						help='''File with pre-computed features.
						To be used with selection_mode = nearest''')
	config = parser.parse_args()

	print(config)

	create_style_file_list(config)

	if not os.path.exists(config.output):
		os.makedirs(config.output)
	else:
		# Check if stylized files have already been generated
		content_files, _ = parse_style_file_list(config.style_file_list)
		out_files = []
		for file in content_files:
			out_file = os.path.join(config.output, file.split('/')[-1])
			if os.path.exists(out_file):
				out_files.append(out_file)
		if len(out_files) == len(content_files):
			print('Stylized images have already been generated! Finishing here.')
			exit()


	print('Running style transfer')
	run_bulk(config)

	'''
	CUDA_VISIBLE_DEVICES=1 python dataset_style_transfer.py --content /PATH/TO/CONTENT/IMAGES --style /PATH/TO/STYLE/IMAGES --style_file_list /PATH/TO/STYLE/FILELIST --output /PATH/TO/SAVE/STYLIZED/IMAGES -e -d -s --verbose --style_features /PATH/TO/STYLE/FEATURES/SINGLEFILE --selection_mode nearest
	'''



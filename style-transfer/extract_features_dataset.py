import os
import argparse
import glob
from transfer import *


IMG_EXTENSIONS = [
	'.jpg', '.JPG', '.jpeg', '.JPEG',
	'.png', '.PNG',
]


def extract_features_dataset(config):
	image_folder = config.style
	output_folder = config.style_features

	device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
	device = torch.device(device)

	image_list = glob.glob(image_folder + '/*')
	image_list = [x for x in image_list if is_image_file(x)]

	assert os.path.exists(image_folder)
	if not os.path.exists(output_folder):
		print(f'Saving extracted features to {output_folder}')
		os.makedirs(output_folder)
	else:
		num_images = len(image_list)
		style_features = glob.glob(f'{config.style_features}/*.pth')
		if len(style_features) == len(image_list):
			print(f'Feature folder {output_folder} already exists. Skipping this step.')
			return

	transfer_at = set()
	if config.transfer_at_encoder:
		transfer_at.add('encoder')
	if config.transfer_at_decoder:
		transfer_at.add('decoder')
	if config.transfer_at_skip:
		transfer_at.add('skip')

	wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool,
				device=device, verbose=config.verbose)

	with torch.no_grad():
		for ii, image_file in enumerate(image_list):
			image, _, _ = open_image(image_file)
			image_feat = image.to(device)
			image_skips = {}
			for level in [1, 2, 3, 4]:
				image_feat = wct2.encode(image_feat, image_skips, level)

			image_feat = torch.mean(image_feat.view(image_feat.size(0), image_feat.size(1), -1), dim=2).flatten()

			fname = output_folder + '/' + image_file.split('/')[-1].split('.')[0] + '.pth'
			torch.save(image_feat, fname)

			if ii % 20 == 0:
				print(f'Processing image {ii} of {len(image_list)}')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--style', type=str, default=None,
						help='''Folder where to look for the style images.''')
	parser.add_argument('--alpha', type=float, default=1)
	parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
	parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
	parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
	parser.add_argument('-s', '--transfer_at_skip', action='store_true')
	parser.add_argument('-a', '--transfer_all', action='store_true')
	parser.add_argument('--cpu', action='store_true')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--style_features', type=str, default=None,
						help='''Folder where to save features of style images.''')
	config = parser.parse_args()

	print(config)

	extract_features_dataset(config)


	'''
	CUDA_VISIBLE_DEVICES=0 python extract_features_dataset.py --style PATH/TO/STYLE/FOLDER -e -d -s --verbose --style_features PATH/TO/SAVE/STYLE/FEATURES
	'''



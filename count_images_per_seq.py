import argparse
import torch
from torch.utils import data
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import sys
import os

from dataset.cityscapes_dataset import Cityscapes
from dataset.synthia_dataset import SYNTHIA
from dataset.gta5_dataset import GTA5
from dataset.acdc_dataset import ACDC

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

CITYSCAPES_ROOT = '/tmp-network/project/miai-meta/data/Cityscapes'
WEATHER_CITYSCAPES_ROOT = '/tmp-network/project/miai-meta/data/WeatherCityscapes'
ACDC_ROOT = '/tmp-network/project/miai-meta/data/ACDC'
SYNTHIA_ROOT = '/nfs/tmp/Synthia'

def get_arguments():

	parser = argparse.ArgumentParser(description="User input")

	parser.add_argument("--trg_dataset", type=str, default='Cityscapes',
						help="Which target dataset to transfer to {VirtualKITTY2, SYNTHIA, Cityscapes, ACDC}")
	parser.add_argument("--trg_data_dir", type=str, default='/tmp-network/data/vision/cityscapes/cityscapes-1/',
						help="Directory of target dataset {VirtualKITTY2, SYNTHIA, Cityscapes}")
	parser.add_argument("--city", type=str, default='aachen',
						help="City for CityScapes")
	parser.add_argument("--scene", type=str, default='01',
						help="Scene for KITTY (and SYNTHIA)")
	parser.add_argument("--cond", type=str, default='clone',
						help="Condition for KITTY (and SYNTHIA)")

	return parser.parse_args()

args = get_arguments()


def main():
	"""Create the model and start the training."""

	input_size = '1280,720'
	w, h = map(int, args.input_size.split(','))
	args.input_size = (w, h)

	cudnn.enabled = True
	cudnn.benchmark = True

	if len(args.scene.split('-')) > 1:
		scene_list = args.scene.split('-')
		cond_list = args.cond.split('-')
	else:
		scene_list = [args.scene]
		cond_list = [args.cond]

	if args.trg_dataset=='Cityscapes':
		trg_parent_set = Cityscapes(
				CITYSCAPES_ROOT, WEATHER_CITYSCAPES_ROOT, scene_list, cond_list,
				crop_size=args.input_size, mean=IMG_MEAN,
				alpha=0.02, beta=0.01, dropsize=0.005, pattern=3)

	elif args.trg_dataset=='SYNTHIA':
		trg_parent_set = SYNTHIA(
				SYNTHIA_ROOT, scene_list, cond_list,
				camera_id='0', crop_size=args.input_size, mean=IMG_MEAN,
				set='all', num_images=300)

	elif args.trg_dataset=='ACDC':
		trg_parent_set = ACDC(
				ACDC_ROOT, scene_list, cond_list, crop_size=args.input_size, mean=IMG_MEAN)

	trg_train_loader = data.DataLoader(trg_parent_set, batch_size=1, shuffle=False, pin_memory=True)

	summary_file_path = os.path.join(
			f'./dataset',f'{args.trg_dataset}_num_images_per_seq',
			f'{args.trg_dataset}_{args.scene}_{args.cond}_num_images_per_seq.pkl')

	with open(summary_file_path, 'wb') as f:
		pickle.dump(trg_parent_set.num_imgs_per_seq, f, pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
	main()

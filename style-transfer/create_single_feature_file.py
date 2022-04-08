import os
import glob
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--style_features', type=str, default=None,
					help='''Folder where to look for features of style images.''')
parser.add_argument('--style_features_singlefile', type=str, default=None,
					help='''Folder where to save a singlefile with all features.''')
config = parser.parse_args()


# Folder where features for every individual style image are saved
style_features_folder = config.style_features # Output folder from extract_features_dataset.py
style_feat_files = sorted(glob.glob(f'{style_features_folder}/*.pth'))


# Load features for all images and store them in matrix
print('Loading features...')
style_features = torch.zeros((len(style_feat_files), 512))
for ii, style_fname in enumerate(style_feat_files):
	style_feat = torch.load(style_fname)
	style_features[ii, :] = style_feat

# Folder where to store the file with all features
all_features = config.style_features_singlefile

if not os.path.exists(all_features):
	os.makedirs(all_features)

filename = all_features + 'all_feats_file.pth'
torch.save(style_features, filename)

print(f'Features saved in {filename}')

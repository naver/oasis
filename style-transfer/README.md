
# Style transfer for online UDA 
 
PyTorch implementation to generate a stylized dataset based on single-image style transfer.



### Pre-compute style dataset features (only needed for nearest selection)
The following code will generate a folder where features will be saved for each style image
```bash
python extract_features_dataset.py --style PATH/TO/STYLE/FOLDER -e -d -s --verbose --style_features PATH/TO/SAVE/STYLE/FEATURES
```

To speed up the nearest search, create one single tensor with all features 
```bash
python create_single_feature_file.py --style_features PATH/TO/STYLE/FEATURES --style_features_singlefile PATH/TO/SAVE/STYLE/FEATURES/SINGLEFILE
```

### Generate stylized dataset
Given a content dataset (e.g. a SYNTHIA image subfolder) and a style folder we can create a stylized dataset based on two criteria:

- *Random selection:* Sample any image from the style folder randomly for each content image.
```bash
python dataset_style_transfer.py --content /PATH/TO/CONTENT/IMAGES --style /PATH/TO/STYLE/IMAGES --style_file_list /PATH/TO/STYLE/FILELIST --output /PATH/TO/SAVE/STYLIZED/IMAGES -e -d -s --verbose --selection_mode random
```

- *Nearest selection:* For each content image, sample the closest style image.
```bash
python dataset_style_transfer.py --content /PATH/TO/CONTENT/IMAGES --style /PATH/TO/STYLE/IMAGES --style_file_list /PATH/TO/STYLE/FILELIST --output /PATH/TO/SAVE/STYLIZED/IMAGES -e -d -s --verbose --style_features /PATH/TO/STYLE/FEATURES/SINGLEFILE --selection_mode nearest
```

### Launch jobs in batch (slurm)
In case of using slurm, the script `generate_job_list.py` can be useful to generate a `.sh` file that will generate several jobs for a given dataset.
```bash
python generate_job_list.py --dataset DATASET_NAME --style_folder /PATH/TO/STYLE/IMAGES --selection_mode [nearest, random]
```
*NOTE* Before using this script, it needs to be modified to include the paths to the desired datasets.

<!---
### Results
<img src="./figures/results.png" width="1000" title="results"> 
-->

## Acknowledgement
This code is partially based on  Photorealistic Style Transfer via Wavelet Transforms | [paper](https://arxiv.org/abs/1903.09760) | [code](https://github.com/clovaai/WCT2/)
 

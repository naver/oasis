"""
Code to generate sequences used to replicate the experiments
presented in the main paper. All functions take in input lists
for datasets/conditions/scenes and append key words for specific
experiments to them.

For example, the function update_synthia_lists with mode = "single"
will fill scene_list with '01', '04', etc. (the environments used),
cond_list with 'DAWN', 'NIGHT', etc (the conditions used) and
trg_dataset_list simply with the 'SYNTHIA' keyword.
"""

import glob
import numpy as np
import numpy.random as npr

def update_cityscapes_lists(mode, trg_dataset_list, scene_list, cond_list):

	if mode == 'single':
		scene_list_ = ['aachen', 'dusseldorf', 'krefeld',
						'ulm', 'bochum', 'erfurt',
						'monchengladbach', 'weimar', 'bremen',
						'hamburg', 'strasbourg', 'zurich',
						'cologne', 'hanover', 'stuttgart',
						'darmstadt', 'jena', 'tubingen']

		cond_list_ = ['clean', 'clean', 'clean', 'clean',
					'clean', 'clean', 'clean', 'clean',
					'clean', 'clean', 'clean', 'clean',
					'clean', 'clean', 'clean', 'clean',
					'clean', 'clean']

	elif mode == 'multi-O':
		scene_list_ = ['aachen-hamburg-frankfurt-munster',
						'jena-hamburg-zurich-hanover',
						'hamburg-stuttgart-tubingen-darmstadt',
						'stuttgart-bochum-monchengladbach-bremen',
						'lindau-bochum-aachen-stuttgart',
						'monchengladbach-dusseldorf-jena-strasbourg',
						'jena-strasbourg-bochum-dusseldorf',
						'strasbourg-stuttgart-tubingen-monchengladbach',
						'krefeld-erfurt-tubingen-strasbourg',
						'monchengladbach-lindau-aachen-jena', ]

		cond_list_ = ['clean-clean-clean-clean']*10

	elif mode == 'multi-AW':
		scene_list_ = ['zurich-darmstadt-dusseldorf-jena',
						'munster-hamburg-cologne-erfurt',
						'bremen-stuttgart-aachen-tubingen',
						'dusseldorf-darmstadt-tubingen-bremen',
						'bremen-krefeld-lindau-bochum',
						'cologne-munster-hanover-bremen',
						'frankfurt-erfurt-zurich-cologne',
						'hanover-aachen-jena-munster',
						'bremen-ulm-zurich-darmstadt',
						'erfurt-ulm-aachen-lindau']

		cond_list_ = ['clean-fog-rain-fog', 'rain-fog-clean-clean',
					'clean-fog-rain-clean', 'rain-clean-fog-rain',
					'rain-clean-fog-clean', 'clean-rain-fog-clean',
					'fog-rain-clean-clean', 'clean-fog-fog-rain',
					'rain-fog-clean-fog', 'rain-clean-rain-fog']

	trg_dataset_list_ = ['Cityscapes'] * 10

	trg_dataset_list += trg_dataset_list_
	scene_list += scene_list_
	cond_list += cond_list_

	return None


def update_acdc_lists(mode, trg_dataset_list, scene_list,
					cond_list, seq_len=4, weather=None):

	if mode == 'single':
		scene_list_ = ['GOPR0475', 'GOPR0476', 'GOPR0477', 'GOPR0478',
					'GOPR0479', 'GOPR0351', 'GOPR0376', 'GP010376',
					'GP010397', 'GP020397', 'GOPR0400', 'GOPR0402',
					'GP010400', 'GP010402', 'GP020400', 'GOPR0122',
					'GOPR0604', 'GOPR0606', 'GOPR0607', 'GP010122']

		cond_list_ = ['fog', 'fog', 'fog', 'fog', 'fog',
					'night', 'night', 'night', 'night', 'night',
					'rain', 'rain', 'rain', 'rain', 'rain',
					'snow', 'snow', 'snow', 'snow', 'snow']

	if mode == 'multi':
		scene_list_ = ['GP010476-GP010402-GP030176-GP010376',
						'GP010476-GOPR0351-GOPR0122-GP020402',
						'GOPR0402-GP010376-GP010607-GOPR0478',
						'GP040176-GP020402-GP020397-GP010476',
						'GOPR0122-GP020475-GOPR0356-GP010402']

		cond_list_ = ['fog-rain-snow-night', 'fog-night-snow-rain',
						'rain-night-snow-fog', 'snow-rain-night-fog',
						'snow-fog-night-rain']

	trg_dataset_list_ = ['ACDC'] * len(scene_list_)

	trg_dataset_list += trg_dataset_list_
	scene_list += scene_list_
	cond_list += cond_list_

	return None


def update_synthia_lists(mode, trg_dataset_list, scene_list, cond_list):

	if mode == 'single':
		cond_list_ = ['FOG', 'DAWN', 'NIGHT', 'SUMMER', 'FALL',
					'SPRING', 'FOG', 'DAWN', 'NIGHT', 'SUMMER',
					'FALL', 'SPRING', 'FOG', 'DAWN', 'NIGHT',
					'FALL', 'SPRING', 'RAIN', 'SOFTRAIN']

		scene_list_ = ['01', '01', '01', '01', '01', '01',
					'05', '05', '05', '05', '05', '05',
					'04', '04', '04', '04', '04', '05', '04']


	elif mode == 'multi':
		cond_list_ = ['NIGHT-DAWN-WINTER-WINTERNIGHT-SOFTRAIN',
					'NIGHT-WINTER-SOFTRAIN-WINTERNIGHT-FOG',
					'WINTERNIGHT-WINTERNIGHT-NIGHT-SUNSET-WINTER',
					'WINTERNIGHT-DAWN-NIGHT-SUNSET-WINTER',
					'SOFTRAIN-NIGHT-FOG-WINTER-WINTERNIGHT',
					'NIGHT-FOG-FALL-FALL-RAIN',
					'SPRING-WINTER-NIGHT-DAWN-RAINNIGHT',
					'WINTER-SUNSET-SPRING-SPRING-FOG',
					'RAINNIGHT-SOFTRAIN-WINTER-FOG-DAWN']

		scene_list_ = ['05-01-01-05-04', '04-01-04-05-05', '01-05-05-04-04',
					'05-05-05-05-01', '05-01-04-05-01', '01-04-01-05-05',
					'04-05-04-01-04', '01-04-04-01-05', '04-04-05-05-01']

	trg_dataset_list_ = ['SYNTHIA'] * len(scene_list_)
	trg_dataset_list += trg_dataset_list_
	scene_list += scene_list_
	cond_list += cond_list_

	return None

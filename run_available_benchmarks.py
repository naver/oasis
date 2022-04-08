import pickle

import run_exps_helpers

trg_dataset_list, scene_list, cond_list = [], [], []

run_exps_helpers.update_synthia_lists('multi', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cityscapes_lists('multi-O', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_cityscapes_lists('multi-AW', trg_dataset_list, scene_list, cond_list)
run_exps_helpers.update_acdc_lists('multi', trg_dataset_list, scene_list, cond_list)

avail_bench_list = []

for (trg_dataset_, scene_, cond_) in zip(trg_dataset_list[:], scene_list[:], cond_list[:]):
	avail_bench_list.append(f'{trg_dataset_}_{scene_}_{cond_}')
	print(f'{trg_dataset_}_{scene_}_{cond_}')

with open('./dataset/available_benchmarks.pkl', 'wb') as f:
	pickle.dump(avail_bench_list, f, pickle.HIGHEST_PROTOCOL)

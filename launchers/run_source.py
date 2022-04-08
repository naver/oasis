dataset_ = 'GTA5'
seed=[111]
learning_rate = [0.0001]
weight_decay = [0.0005]
momentum = [0.9]
batch_size = [1]
iter_size = [1]
optimizer = ['SGD']
augm = [0,1]
augm_set = [0,1,2,3]

model = ['Deeplab']
num_epochs_ = 20

for seed_ in seed:
	for model_ in model:
		for optimizer_ in optimizer:
			for learning_rate_ in learning_rate:
				for weight_decay_ in weight_decay:
					for momentum_ in momentum:
						for batch_size_ in batch_size:
							for iter_size_ in iter_size:
								for augm_ in augm:
									for augm_set_ in augm_set:

										if (augm_==0) and (augm_set_!=0):
											continue
										if (augm_==1) and (augm_set_==0):
											continue
										print(f'python -u train_on_source.py'+
												f' --dataset={dataset_}'+
												f' --seed={seed_}'+
												f' --optimizer={optimizer_}'+
												f' --learning_rate={learning_rate_}'+
												f' --weight_decay={weight_decay_}'+
												f' --model={model_}'+
												f' --num_epochs={num_epochs_}'+
												f' --batch_size={batch_size_}'+
												f' --iter_size={iter_size_}'+
												f' --momentum={momentum_}'+
												f' --do_augm={augm_}'+
												f' --augm_set={augm_set_}')

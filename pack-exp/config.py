import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

##########################################
# Hyperparameters for input data
##########################################

hyperparams_input = cfg['hyperparams_input']
img_width_imagenet = hyperparams_input['img_width_imagenet']
img_height_imagenet = hyperparams_input['img_height_imagenet']
img_width_cifar10 = hyperparams_input['img_width_cifar10']
img_height_cifar10 = hyperparams_input['img_height_cifar10']
img_width_mnist = hyperparams_input['img_width_mnist']
img_height_mnist = hyperparams_input['img_height_mnist']
num_class_imagenet = hyperparams_input['num_class_imagenet']
num_class_cifar10 = hyperparams_input['num_class_cifar10']
num_class_mnist = hyperparams_input['num_class_mnist']
num_channels_rgb = hyperparams_input['num_channel_rgb']
num_channels_bw = hyperparams_input['num_channel_bw']

##########################################
# Hyperparameters for single training
##########################################

hyperparams_single_cfg = cfg['hyperparams_single_train']
single_num_epoch = hyperparams_single_cfg['num_epoch']
single_rand_seed = hyperparams_single_cfg['random_seed']
single_model_type = hyperparams_single_cfg['model_type']
single_activation = hyperparams_single_cfg['activation']
single_opt = hyperparams_single_cfg['optimizer']
single_batch_size = hyperparams_single_cfg['batch_size']
single_num_layer = hyperparams_single_cfg['num_model_layer']
single_learning_rate = hyperparams_single_cfg['learning_rate']
single_record_marker = hyperparams_single_cfg['record_marker']
single_use_cpu = hyperparams_single_cfg['use_cpu']
single_use_raw_image = hyperparams_single_cfg['use_raw_image']
single_measure_step = hyperparams_single_cfg['measure_step']
single_use_tb_timeline = hyperparams_single_cfg['use_tb_timeline']

##########################################
# Hyperparameters for pack training
##########################################

hyperparams_pack_cfg = cfg['hyperparams_pack_train']
pack_num_epoch = hyperparams_pack_cfg['num_epoch']
pack_rand_seed = hyperparams_pack_cfg['random_seed']
pack_model_type = hyperparams_pack_cfg['packed_model_type']
pack_activation = hyperparams_pack_cfg['activation']
pack_opt = hyperparams_pack_cfg['optimizer']
pack_batch_size = hyperparams_pack_cfg['batch_size']
pack_num_layer = hyperparams_pack_cfg['num_model_layer']
pack_learning_rate = hyperparams_pack_cfg['learning_rate']
pack_record_marker = hyperparams_pack_cfg['record_marker']
pack_batch_padding = hyperparams_pack_cfg['batch_padding']
pack_use_cpu = hyperparams_pack_cfg['use_cpu']
pack_use_raw_image = hyperparams_pack_cfg['use_raw_image']
pack_measure_step = hyperparams_pack_cfg['measure_step']
pack_same_input = hyperparams_pack_cfg['same_input']
pack_use_tb_timeline = hyperparams_pack_cfg['use_tb_timeline']

##########################################
# Path
##########################################

#path_cfg = cfg['local_path']
#path_cfg = cfg['roscoe_path']
path_cfg = cfg['hp_path']

mnist_train_img_path = path_cfg['mnist_train_img_path']
mnist_train_label_path = path_cfg['mnist_train_label_path']
mnist_t10k_img_path = path_cfg['mnist_t10k_img_path']
mnist_t10k_label_path = path_cfg['mnist_t10k_label_path']
cifar_10_path = path_cfg['cifar_10_path']
imagenet_t1k_img_path = path_cfg['imagenet_t1k_img_path']
imagenet_t1k_label_path = path_cfg['imagenet_t1k_label_path']
imagenet_t1k_bin_path = path_cfg['imagenet_t1k_bin_path']
imagenet_t10k_img_path = path_cfg['imagenet_t10k_img_path']
imagenet_t10k_label_path = path_cfg['imagenet_t10k_label_path']
imagenet_t10k_bin_path = path_cfg['imagenet_t10k_bin_path']
profile_path = path_cfg['profile_path']

import shutil


print('test')


import sys
sys.path.extend(['/home/omerl/rdnd', '/home/omerl/rdnd'])

from RapidBase.import_all import *
from torch.utils.data import DataLoader
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.callbacks import InferenceCallback_Denoising_Base
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.losses import Loss_Simple
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_RLSP_Recursive, PostProcessing_FastDVDNet
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.tensorboard_callbacks import TB_Denoising_Recursive
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import get_default_IO_dict

from os import path as osp
from pathlib import Path
from datetime import datetime
### Import Models: ###
from KAIR_light.models.vrt_models import VRT_denoise_6_frames_checkpoint

### Initialize Network State Dict: ###
Train_dict = EasyDict()
######################################################################################################################
### Paths: ###
#(1). General Paths:
home_path = '/home/omerl' #Path.home()
base_path = home_path
project_path = osp.join(base_path, 'rdnd')
today_date = datetime.today().strftime('%Y-%m-%d')
inference_path_master = osp.join('/raid/Pytorch_Checkpoints', today_date)
script_path = osp.join('/raid/scripts', today_date)
TensorBoard_path = osp.join(inference_path_master, 'TensorBoard')
Network_checkpoint_folder = osp.join(inference_path_master, 'Model_New_Checkpoints')
datasets_main_folder = '/raid/datasets'
model_denoise_path = osp.join(base_path, 'Checkpoints/VRT/008_VRT_videodenoising_DAVIS.pth')
# model_denoise_path = '/raid/Pytorch_Checkpoints/2022-06-06/Model_New_Checkpoints/REDS_denoise.py/REDS_denoise.py_TEST1_Step14250.tar'
model_deblur_path = '/raid/Pytorch_Checkpoints/2022-05-31/Model_New_Checkpoints/decomp_h264_0.75_0.75_2_REDS.py/decomp_h264_0.75_0.75_2_REDS.py_TEST1_Step10275.tar'
model_deblur_path = osp.join(base_path, 'Checkpoints/VRT/007_VRT_videodeblurring_REDS.pth')
model_sr_path = osp.join(base_path, 'Checkpoints/VRT/001_VRT_videosr_bi_REDS_6frames.pth')
model_checkpoints = [model_denoise_path, model_deblur_path]

#(2). Train Images:
Train_Images_Folder = osp.join(datasets_main_folder, 'drones/T/train')
# Train_Images_Folder = osp.join(base_path, 'save_folder')
# Train_Images_Folder = osp.join(datasets_main_folder, 'drones/T/train')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'DIV2K/DIV2K/DIV2K_train_HR_BW')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'REDS/val_blur/val/val_blur/004')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'GoPro/train')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'DIV2K/Image_SuperResolution/RealSR (Final)/Nikon/Train/2')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'Vidmeo90K/vimeo_septuplet/sequences/00001')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'DIV2K/real_RGB_noisy_images')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'DIV2K\Official_Test_Images\Original_Images')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'Example Videos/Beirut_BW_Original')
# Train_Images_Folder = os.path.join(datasets_main_folder, 'Example Videos/Beirut_BW_Original/Beirut.mp4')

#(3). Test Images
# Test_Images_Folder = osp.join(datasets_main_folder, 'REDS/val_blur')
# Test_Images_Folder = osp.join(base_path, 'save_folder')
# Test_Images_Folder = osp.join(datasets_main_folder, 'drones/T/test')
Test_Images_Folder = osp.join(datasets_main_folder, 'drones/T/test')
# Test_Images_Folder = os.path.join(datasets_main_folder, 'Example Videos/Beirut_BW_Original')
# Test_Images_Folder = os.path.join(datasets_main_folder, 'Example Videos/Beirut_BW_Original/Beirut.mp4')
Inference_folder_name_addon = ''
######################################################################################################################
### Train/Inference/Debug Parameters: ###
Train_dict.is_training = True
Train_dict.flag_do_validation = True
Train_dict.flag_movie_inference = False
Train_dict.no_GT = False
### Number of Epochs / Iterations: ###
Train_dict.number_of_epochs = 1e9
Train_dict.max_total_number_of_batches = 1e9
### Frequencies/TensorBoard: ###
Train_dict.debug_step = 75
Train_dict.save_models_frequency = 75
Train_dict.validation_set_frequency = 75
Train_dict.tensorboard_train_frequency = 75
Train_dict.number_of_validation_samples = 3
Train_dict.flag_write_TB = False
Train_dict.flag_plot_grad_flow = False
### Flags: ###
Train_dict.flag_residual_learning = False
Train_dict.flag_clip_noisy_tensor_after_noising = False
Train_dict.flag_learn_clean_or_running_average = 'clean'  # 0=clean, 1=running_average
## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=12344)
### Save Network: ###
# (1.1). Name to Save checkpoint as when training new network:
# don't use the string 'Step' at any point in the prefix!!@#!@$@#
Train_dict.Network_checkpoint_prefix = os.path.basename(__file__) +'_3'
# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Train_dict.flag_continue_checkpoint_naming_convention = False
Train_dict.network_free_flow_description = ''
### Load Network: ###
#######################
Train_dict.load_Network_filename = r''
### New Training Or From Checkpoint: ###
# 'new', 'checkpoint'  # Use New Network or Checkpoint
Train_dict.flag_use_new_or_checkpoint_network = 'new'
Train_dict.flag_use_new_or_checkpoint_optimizer = 'new'
Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False
##########################################################################################
### Adjust Network Checkpoint string according to script variables/prefixes/postfixes: ###
Train_dict.main_file_to_save_for_later = __file__
Train_dict = TrainingCore.training_utils.PreTraining_Aux(Train_dict, Network_checkpoint_folder, project_path)

os.makedirs(script_path, exist_ok=True)
shutil.copyfile(Train_dict.main_file_to_save_for_later, osp.join(script_path, Train_dict.Network_checkpoint_prefix+'.py'))
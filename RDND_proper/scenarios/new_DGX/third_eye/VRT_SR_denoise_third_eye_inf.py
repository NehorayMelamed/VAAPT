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
from KAIR_light.models.vrt_models import VRT_SR_6_frames_checkpoint, VRT_SR_6_frames_checkpoint_img_sz_160
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
TensorBoard_path = osp.join(inference_path_master, 'TensorBoard')
Network_checkpoint_folder = osp.join(inference_path_master, 'Model_New_Checkpoints')
datasets_main_folder = '/raid/datasets'
# model_denoise_path = osp.join(base_path, 'Checkpoints/VRT/008_VRT_videodenoising_DAVIS.pth')
model_denoise_path = '/raid/Pytorch_Checkpoints/2022-06-06/Model_New_Checkpoints/REDS_denoise.py/REDS_denoise.py_TEST1_Step14250.tar'
model_deblur_path = '/raid/Pytorch_Checkpoints/2022-05-31/Model_New_Checkpoints/decomp_h264_0.75_0.75_2_REDS.py/decomp_h264_0.75_0.75_2_REDS.py_TEST1_Step10275.tar'
model_deblur_path = osp.join(base_path, 'Checkpoints/VRT/007_VRT_videodeblurring_REDS.pth')
# model_sr_path = osp.join(base_path, 'Checkpoints/VRT/001_VRT_videosr_bi_REDS_6frames.pth')
# model_sr_path = '/raid/Pytorch_Checkpoints/2022-07-18/Model_New_Checkpoints/visDrone_SR_noise_8_to_2_bw.py/visDrone_SR_noise_8_to_2_bw.py_TEST1_Step37050.tar'
# model_sr_path = '/raid/Pytorch_Checkpoints/2022-07-27/Model_New_Checkpoints/' \
#                 'drones_SR_BW[160,160].py/drones_SR_BW[160,160].py_TEST1_Step10950.tar'
# model_sr_path = '/raid/Pytorch_Checkpoints/2022-08-28/Model_New_Checkpoints/' \
#                 'VRT_SR_denoise-deblur_third_eye[pretrained].py/VRT_SR_denoise-deblur_third_eye[pretrained].py' \
#                 '_TEST1_Step15450.tar'
# model_sr_path = '/raid/Pytorch_Checkpoints/2022-08-29/Model_New_Checkpoints/' \
#                 'VRT_SR_denoise-deblur_third_eye[pretrained]_18f.py/VRT_SR_denoise-deblur_third_eye[pretrained]_18f.py' \
#                 '_TEST1_Step131850.tar'
# model_sr_path = '/raid/Pytorch_Checkpoints/2022-08-29/Model_New_Checkpoints/' \
#                 'VRT_SR_denoise-deblur_third_eye[pretrained]_18f.py-higher_ppp_more_sigma/VRT_SR_denoise-deblur_third_eye[pretrained]_18f.py-higher_ppp_more_sigma' \
#                 '_TEST1_Step132000.tar'
model_sr_path = '/raid/Pytorch_Checkpoints/2022-08-29/Model_New_Checkpoints/' \
                'VRT_SR_denoise-deblur_third_eye[pretrained]_18f.py-higher_ppp_more_sigma/VRT_SR_denoise-deblur_third_eye[pretrained]_18f.py-higher_ppp_more_sigma' \
                '_TEST1_Step132000.tar'
# VRT_SR_denoise-deblur_third_eye[pretrained].py-higher_ppp_more_sigma

model_checkpoints = [model_denoise_path, model_deblur_path]

#(2). Train Images:
Train_Images_Folder = osp.join(datasets_main_folder, 'third_fixed_images')
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
Test_Images_Folder = osp.join(datasets_main_folder, 'third_fixed_images')
# Test_Images_Folder = os.path.join(datasets_main_folder, 'Example Videos/Beirut_BW_Original')
# Test_Images_Folder = os.path.join(datasets_main_folder, 'Example Videos/Beirut_BW_Original/Beirut.mp4')
Inference_folder_name_addon = ''
######################################################################################################################
### Train/Inference/Debug Parameters: ###
Train_dict.is_training = False
Train_dict.flag_do_validation = True
Train_dict.flag_movie_inference = False
Train_dict.no_GT = True
### Number of Epochs / Iterations: ###
Train_dict.number_of_epochs = 1e9
Train_dict.max_total_number_of_batches = 1e9
### Frequencies/TensorBoard: ###
Train_dict.debug_step = 75
Train_dict.save_models_frequency = 75
Train_dict.validation_set_frequency = 75
Train_dict.tensorboard_train_frequency = 75
Train_dict.number_of_validation_samples = 4
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
Train_dict.Network_checkpoint_prefix = os.path.basename(__file__) + '_2'
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
##########################################################################################
### IO: ###
IO_dict = get_default_IO_dict()
### Devices: ###
train_devices = [5]
test_devices = [6]
IO_dict.device = torch.device(train_devices[0])
print ('Starting to train on GPUs: {}'.format(train_devices+test_devices))
IO_dict.flag_use_train_dataset_split = False ### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_only_split_indices = False
IO_dict.train_dataset_split_to_test_factor = 0.1
IO_dict.test_args = TestArgs(tile=[20,256,256], nonblind_denoising=False, scale=4)
IO_dict.no_GT = True

### Loading Images: ###
#(1). Folder names:
#(1.1). Root Folder:
IO_dict.root_folder = Train_Images_Folder
IO_dict.corrupted_folder = Train_Images_Folder
#(2). how to search:
IO_dict.string_pattern_to_search = '*'
IO_dict.Corrupted_string_pattern_to_search = '*fixed_images*'
IO_dict.GT_string_pattern_to_search = '*fixed_images_X4*'
IO_dict.allowed_extentions = IMG_EXTENSIONS
IO_dict.flag_recursive = False
#(3). how to load:
IO_dict.flag_to_RAM = False
IO_dict.image_loader = ImageLoaderCV
IO_dict.max_number_of_images = np.inf  # max number of images to search for
IO_dict.max_number_of_noise_images = np.inf # max number of noise images to search for
IO_dict.max_number_of_videos = np.inf # max number of videos to search for
IO_dict.number_of_images_per_video_to_scan = np.inf # max number of images per video to search for / scan
IO_dict.number_of_image_frames_to_generate = 6*3
IO_dict.number_of_images_per_video_to_load = 6*3
### Miscellenous: ###
IO_dict.flag_normalize_images_to_PPP = False
IO_dict.PPP = 1.5 #[2, 10]  # Photons per Pixel.
### Noise Addition: ###
temp_factor = 12/255
IO_dict.noise_dict = get_default_noise_dict()
IO_dict.noise_dict.RRNU_sigmas_polynomial_params = [(3*temp_factor,10*temp_factor)]
IO_dict.noise_dict.CRNU_sigmas_polynomial_params = [(3*temp_factor,10*temp_factor)]
IO_dict.noise_dict.flag_add_row_NU = False
IO_dict.noise_dict.flag_add_col_NU = False
IO_dict.noise_dict.flag_add_blob_NU = False
IO_dict.noise_dict.readout_noise_external_image_path = osp.join(datasets_main_folder, 'KAYA_CAMERA_NOISE/noise')
IO_dict.noise_dict.readout_noise_search_pattern = '*'
IO_dict.noise_dict.flag_add_per_pixel_readout_noise = False
IO_dict.noise_dict.flag_add_external_readout_noise = False
IO_dict.noise_dict.flag_add_shot_noise = False
IO_dict.noise_dict.flag_quantize_images = False
IO_dict.noise_dict.per_pixel_readout_noise_sigma = (5*temp_factor,20*temp_factor)

IO_dict.flag_add_row_readout_noise = False
IO_dict.flag_add_col_readout_noise = False
IO_dict.row_readout_noise_sigma = (3*temp_factor,10*temp_factor)
IO_dict.col_readout_noise_sigma = (3*temp_factor,10*temp_factor)
### Post Loading Stuff: ###
#TODO: understand which of these are still relevant
IO_dict.flag_to_BW_before_noise = True
IO_dict.flag_to_RGB_before_noise = False
IO_dict.flag_to_BW_after_noise = False
IO_dict.flag_to_RGB_after_noise = False
IO_dict.flag_noise_to_BW = False
IO_dict.flag_noise_to_RGB = False
### Transforms / Warping: ###
### Shift / Directional-Blur Parameters: ###
IO_dict.transforms_dict.warp_method = 'bilinear'  # 'nearest', 'bilinear', 'bicubic', 'fft'
IO_dict.transforms_dict.shift_size = 0
IO_dict.transforms_dict.rotation_angle_size = 0
IO_dict.transforms_dict.scale_delta = 0
IO_dict.transforms_dict.blur_fraction = 0
IO_dict.transforms_dict.shift_mode = 'seperate'  # 'seperate'=each axis randomizes shift seperately,  'constant_size'=constant shift size, direction is randomized
IO_dict.transforms_dict.number_of_blur_steps_per_pixel = 0
### Super Resolution: ###
IO_dict.transforms_dict.flag_upsample_noisy_input_to_same_size_as_original = False
IO_dict.transforms_dict.upsample_method = 'bilinear'  # 'nearest', 'bilinear', 'bicubic', 'fft'
IO_dict.transforms_dict.downsample_method = 'binning'  # 'binning', 'nearest', 'bilinear', 'bicubic', 'fft'
IO_dict.transforms_dict.downsampling_factor = 1
### Training Flags: ###
IO_dict.non_valid_border_size = 10
#(1). Universal Training Parameters:
IO_dict.batch_size = 1
### Cropping: ###
IO_dict.flag_crop_mode = 'random'  # 'center', 'random'
w,h,offset = 1920,2560,0# 12,512,0
IO_dict.initial_crop_size = [h + offset, w + offset]
IO_dict.final_crop_size = [h, w]
IO_dict.noise_dict.final_crop_size = IO_dict.final_crop_size
### Test/Validation Dict: ###
w,h,offset = 1920,2560,0# 12,512,0
IO_dict_test = EasyDict(IO_dict)
IO_dict_test.root_folder = Test_Images_Folder
IO_dict_test.corrupted_folder = Test_Images_Folder
IO_dict_test.flag_crop_mode = 'center'
IO_dict_test.initial_crop_size = [w + offset, h + offset]
IO_dict_test.final_crop_size = [w,h]
IO_dict_test.mean_gray_level_per_pixel = [10,10]
IO_dict_test.number_of_image_frames_to_generate = 6*3
IO_dict_test.number_of_images_per_video_to_load = 6*3
#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
train_dataset = TrainingCore.datasets.DataSet_Videos_In_Folders_LoadCorruptedGT(Train_Images_Folder, IO_dict)
test_dataset = TrainingCore.datasets.DataSet_Videos_In_Folders_LoadCorruptedGT(Test_Images_Folder, IO_dict_test)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


### Number Of Elements: ###
IO_dict.num_mini_batches_trn = len(train_dataset) // IO_dict.batch_size
IO_dict.num_mini_batches_val = len(test_dataset) // 1
IO_dict.train_dataset_length = len(train_dataset)
IO_dict.test_dataset_length = len(test_dataset)
### Get Automatic Train/Test Split From Train Set: ###
if IO_dict.flag_use_train_dataset_split:
    train_dataset, test_dataset, train_dataloader, test_dataloader = TrainingCore.datasets.split_dataset(train_dataset, test_dataset, IO_dict)
######################################################################################################################################
### Show images: ###
bla = test_dataset[0]
a = bla.output_frames_noisy
b = bla.output_frames_original
imshow_torch(a[0]/255)
imshow_torch(b[0]/255)
img = a[0]/255
plt.show(block=True)
#####################################################################################################################################################################
### Model Itself: ###
model = VRT_SR_6_frames_checkpoint_img_sz_160(False, checkpoint_path=model_sr_path, train_device=train_devices[0],
                                       test_device=test_devices[0], test_args=IO_dict.test_args)
if Train_dict.flag_use_new_or_checkpoint_network == 'checkpoint':
    model, Train_dict = RapidBase.TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=model)
else:
    Train_dict.Network_checkpoint_step = 0

#####################################################################################################################################################################
### Get Network Preprocessing/Postprocessing: ###
# preprocessing_object = PreProcessing_VRT_Denoise()
# preprocessing_object.flag_BW2RGB = True
# preprocessing_object.flag_denoise = False
#
# postprocessing_object = PostProcessing_VRT_Denoise()
# postprocessing_object.flag_RGB2BW = True

preprocessing_object = PreProcessing_VRT_Denoise_X3_frames()
preprocessing_object.flag_BW2RGB = False
preprocessing_object.flag_denoise = False

postprocessing_object = PostProcessing_VRT_Denoise_X3_frames()
postprocessing_object.flag_RGB2BW = False
#####################################################################################################################################################################
### Optimizer: ###
Network_Optimizer_object = Optimizer_Base(model,
                                          intial_lr=4e-4,
                                          hypergrad_lr=0,
                                          optimizer_type='adam')
if Train_dict.flag_use_new_or_checkpoint_optimizer == 'checkpoint':
    Network_Optimizer_object.Network_Optimizer = TrainingCore.training_utils.load_optimizer_from_checkpoint(
        Network_Optimizer_object.Network_Optimizer, Train_dict)
optimization_dict = Network_Optimizer_object.final_dict
#####################################################################################################################################################################
### LR Scheduler: ###
optimization_dict.minimum_learning_rate = 1e-6
optimization_dict.learning_rate_lowering_factor = 0.8
optimization_dict.learning_rate_lowering_patience_before_lowering = 200000
optimization_dict.flag_momentum_cycling = False
optimization_dict.momentum_bounds = [0.85,0.95]
optimization_dict.flag_uptick_learning_rate_after_falling_below_minimum = False
LR_scheduler_object = LR_Scheduler_ReduceLROnPlateauCustom(Network_Optimizer_object.Network_Optimizer, input_dict=optimization_dict)
optimization_dict = LR_scheduler_object.final_dict
#####################################################################################################################################################################
### Gradient Clipping: ###
optimization_dict.flag_use_gradient_clipping = False
optimization_dict.flag_clip_grad_value_or_norm = 'norm'
optimization_dict.anomaly_factor_threshold = 3
optimization_dict.gradient_clip_norm = 20
optimization_dict.gradient_clip_value_factor = 6.6/34  # initial_grad_norm/initial_grad_max_value
Clip_gradient_object = Clip_Gradient_Base(input_dict=optimization_dict)
optimization_dict = Clip_gradient_object.final_dict
#####################################################################################################################################################################
### Get Loss Object: ###
Loss_object = RapidBase.TrainingCore.losses.Loss_Simple_VRT(inputs_dict=None) #L1 only
# Loss_object = core.losses.Loss_Recursive_Simple(inputs_dict=None)
losses_dict = Loss_object.final_dict
#####################################################################################################################################################################
### TensorBoard: ###
if Train_dict.flag_write_TB:
    TB_writer_train = SummaryWriter(os.path.join(TensorBoard_path, Train_dict.Network_checkpoint_prefix + '_Train'))
    TB_writer_test = SummaryWriter(os.path.join(TensorBoard_path, Train_dict.Network_checkpoint_prefix + '_Validation'))
else:
    TB_writer_train = None
    TB_writer_test = None
TB_object_train = TB_Denoising(TB_writer_train)
TB_object_test = TB_Denoising(TB_writer_test)
TB_objects = [TB_object_train, TB_object_test]
#####################################################################################################################################################################
### Combine dictionaries to Train_dict: ###
Train_dict.optimization_dict = optimization_dict
Train_dict.IO_dict = IO_dict
Train_dict.losses_dict = losses_dict
Train_dict.update(optimization_dict)
Train_dict.update(IO_dict)
Train_dict.update(losses_dict)
Train_dict.update(Train_dict)
#####################################################################################################################################################################
### Callbacks Names: ###
if IO_dict.flag_use_train_dataset_split:
    dataset_name = str(test_dataset.dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
else:
    dataset_name = str(test_dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
inference_path = os.path.join(inference_path_master, 'Inference', Train_dict.Network_checkpoint_prefix + '/' + dataset_name)
if Train_dict.is_training == False: #If i'm doing validation (and not just outputing stuff while training) - create a special folder to save results
    inference_path = os.path.join(inference_path, os.path.split(Test_Images_Folder)[-1] + '_' + Inference_folder_name_addon)

### Get Debug Callback: ###
debug_callback = InferenceCallback_Denoising_Base(inference_path + '_debug', Train_dict)  #TODO: isn't the Train_dict.is_training useless now?!!
### Get Validation Callback: ###
validation_callback = InferenceCallback_Denoising_Base(inference_path + '_validation', Train_dict)
### Get Inference Callback: ###
inference_callback = validation_callback
# debug_callback = None
# validation_callback = None
# inference_callback = None
#####################################################################################################################################################################
### Define General Trainer: ###
Trainer = GeneralTrainer(model=model,
                         preprocessing_object=preprocessing_object,
                         postprocessing_object=postprocessing_object,
                         Network_Optimizer_object=Network_Optimizer_object,
                         Loss_object=Loss_object,
                         TB_object=TB_objects,
                         LR_scheduler_object=LR_scheduler_object,
                         Clip_gradient_object=Clip_gradient_object,
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,
                         debug_callback=debug_callback,
                         validation_callback=validation_callback,
                         inference_callback=inference_callback,
                         Train_dict=Train_dict)

### Train/Inference: ###
if Train_dict.is_training:
    Trainer.Train()
else:
    # assert (Train_dict.flag_use_new_or_checkpoint_network == 'checkpoint')
    Trainer.inference()
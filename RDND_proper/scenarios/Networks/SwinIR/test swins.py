from NEW_TASHTIT.import_Tashtit import *

from torch.utils.data import DataLoader
from RapidBase.TrainingCore.training_utils import *
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.losses import Loss_Simple
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_RLSP_Recursive
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.tensorboard_callbacks import TB_Denoising_Recursive
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import get_default_IO_dict
import RapidBase.TrainingCore.training_utils
import RapidBase.TrainingCore
from RDND_proper.models.SwinIR.SwinIR import SwinIR

### Initialize Network State Dict: ###
Train_dict = EasyDict()

######################################################################################################################
### Paths: ###
#(1). General Paths
base_path = path_fix_path_for_linux('/media/mmm/DATADRIVE6/Omer Leibovitch')
project_path = path_fix_path_for_linux(os.path.join(base_path, 'IMOD'))
inference_path_master = path_fix_path_for_linux(os.path.join(base_path, 'Results', 'Inference'))
TensorBoard_path = path_fix_path_for_linux(os.path.join(base_path, 'Results', 'TensorBoard'))
Network_checkpoint_folder = path_fix_path_for_linux(os.path.join(base_path, 'Results', 'Model Checkpoints'))
#(2). Train Images:
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/test')
Train_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'Div2K/DIV2K_train_HR_BW'))
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Drones')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/test')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u/')

#(3). Test Images
# Test_Images_Folder = os.path.join(datasets_main_folder, '/Vidmeo90K/vimeo_septuplet/sequences/00001/0021')
Test_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'Div2K/DIV2K_train_HR_BW'))
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Drones')
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/test')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u/')
#RGB images
Inference_folder_name_addon = ''
######################################################################################################################


######################################################################################################################
### Train/Inference/Debug Parameters: ###
Train_dict.is_training = False
Train_dict.flag_do_validation = False
Train_dict.flag_movie_inference = False

### Number of Epochs / Iterations: ###
Train_dict.number_of_epochs = 1e9
Train_dict.max_total_number_of_batches = 1e9

### Frequencies/TensorBoard: ###
Train_dict.debug_step = 200
Train_dict.save_models_frequency = 200
Train_dict.validation_set_frequency = 100
Train_dict.tensorboard_train_frequency = 100
Train_dict.number_of_validation_samples = 10
Train_dict.flag_write_TB = True
Train_dict.flag_plot_grad_flow = False

### Flags: ###
Train_dict.flag_residual_learning = False
Train_dict.flag_clip_noisy_tensor_after_noising = False
Train_dict.flag_learn_clean_or_running_average = 'clean' # 0=clean, 1=running_average
###
######################################################################################################################



######################################################################################################################


######################################################################################################################
### Save Network: ###
#######################
# (1.1). Name to Save checkpoint as when training new network:
# don't use the string 'Step' at any point in the prefix!!@#!@$@#
Train_dict.Network_checkpoint_prefix = 'FastDVDNet_SuperResolution_1'

# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Train_dict.flag_continue_checkpoint_naming_convention = False

Train_dict.network_free_flow_description = ''
######################################################################################################################
## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=12322)

######################################################################################################################
### Load Network: ###
#######################
Train_dict.load_Network_filename = r''
######################################################################################################################


########################################
### New Training Or From Checkpoint: ###
# 'new', 'checkpoint'                # Use New Network or Checkpoint
Train_dict.flag_use_new_or_checkpoint_network = 'new'
Train_dict.flag_use_new_or_checkpoint_optimizer = 'new'

Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False
########################################


##########################################################################################
### Adjust Network Checkpoint string according to script variables/prefixes/postfixes: ###
Train_dict.main_file_to_save_for_later = os.path.realpath('__file__')
Train_dict = TrainingCore.training_utils.PreTraining_Aux(Train_dict, Network_checkpoint_folder, project_path)
##########################################################################################



#######################################################################################################################################
### IO: ###
IO_dict = get_default_IO_dict()
### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_use_train_dataset_split = True #TODO: add possibility of splitting the train set as far as images but still use the IO_dict_test instead o the IO_dict_train
IO_dict.flag_only_split_indices = True #test_dataset is still defined seperately with its own IO_dict_test but split on indices base
IO_dict.train_dataset_split_to_test_factor = 0.05
### Loading Images: ###
IO_dict.string_pattern_to_search = '*'
IO_dict.HR_string_pattern_to_search = '*HR*'
IO_dict.LR_string_pattern_to_search = '*LR*'
IO_dict.flag_recursive = True
IO_dict.max_number_of_images = np.inf
IO_dict.number_of_image_frames_to_generate = 3 #5
IO_dict.flag_to_RAM = False
IO_dict.flag_noise_images_to_RAM = False
### Post Loading Stuff: ###
IO_dict.flag_to_BW_before_noise = True
IO_dict.flag_to_RGB_before_noise = False
IO_dict.flag_to_BW_after_noise = False
IO_dict.flag_to_RGB_after_noise = False
IO_dict.flag_how_to_concat = 'T'
### Noise Parameters: ###
IO_dict.SNR = np.inf  # mean~10, std~2(readout)+sqrt(10)(shot)=5, effective_sigma~255/mean*5~127.5 -> SNR~4. this is only an approximation to better condition network
IO_dict.sigma_to_dataset = 1.5 #1 / np.sqrt(IO_dict.SNR) * 255
IO_dict.SNR_to_model = IO_dict.SNR
### Camera Parameters: ### (*). INSTEAD OF ABOVE NOISE PARAMETERS!!!!
IO_dict.mean_gray_level_per_pixel = [200, 200]
IO_dict.electrons_per_gray_level = 14.53
IO_dict.photons_per_electron = 1
IO_dict.gray_levels_per_electron = 1 / IO_dict.electrons_per_gray_level
IO_dict.electrons_per_photon = 1 / IO_dict.photons_per_electron
### Shift / Directional-Blur Parameters: ###
IO_dict.shift_size = 0.5*4 # * 1e-9
IO_dict.blur_size = 0
### Gaussian Blur Parameters: ###
IO_dict.blur_number_of_channels = 1
IO_dict.blur_kernel_size = 3
IO_dict.blur_sigma = 1
### Super Resolution: ###
IO_dict.downsampling_factor = 4
### Universal Training Parameters: ###
IO_dict.batch_size = 1
#### Decide whether to load all inputs to gpu immediately after dataloader: ###
IO_dict.flag_all_inputs_to_gpu = True
### Cropping: ###
IO_dict.flag_crop_mode = 'random'
scale = 10
IO_dict.initial_crop_size = [int(64*scale)+5,int(64*scale)+5]
IO_dict.final_crop_size = [int(64*scale),int(64*scale)]

### Test/Validation Dict: ###
IO_dict_test = EasyDict(IO_dict)
IO_dict_test.flag_crop_mode = 'center'
IO_dict_test.initial_crop_size = np.inf
IO_dict_test.final_crop_size = np.inf
IO_dict_test.mean_gray_level_per_pixel = [10,10]
#######################################################################################################################################



#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_LoadLRHR_1(Train_Images_Folder, IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_LoadLRHR_1(Test_Images_Folder, IO_dict_test)
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
### Number Of Elements: ###
IO_dict.num_mini_batches_trn = len(train_dataset) // IO_dict.batch_size
IO_dict.num_mini_batches_val = len(test_dataset) // 1
IO_dict.train_dataset_length = len(train_dataset)
IO_dict.test_dataset_length = len(test_dataset)
######################################################################################################################

### Get Batch: ###
for batch in train_dataloader:
    outputs_dict = batch
    break

batch_orig = outputs_dict['center_frame_original']
first_orig = batch_orig[0]

### Model path: ###
model_path_L = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
model_path_M = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
model_path_LW = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth')
model_path_div2k = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')

### Get Models: ###
model_L = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv').cuda()
model_L.load_state_dict(torch.load(model_path_L)['params_ema'])

### Get Batch: ###
batch_output_frames = outputs_dict['output_frames_noisy']
x = batch_output_frames/255
x = x.squeeze(dim=2)
# if x.shape[2] == 1: # x is single channel
#     x = x.expand(-1, -1, 3, -1, -1) # make x 3-channel
#
#
## Upsample Bicubic: ###
# upsample_layer = nn.Upsample(scale_factor=4, mode='bicubic')
# stacked_x = x.view(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
# upsampled_stacked_x = upsample_layer(stacked_x)
# x_upsampled_bicubic = upsampled_stacked_x.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3]*4,x.shape[4]*4)

### Test Models: ###

with torch.no_grad():
    x_sr = model_L(x.cuda())

# with torch.no_grad():
#     x_iconvsr = model_iconvsr(x.cuda())

### Show Plots: ###
num_frames = IO_dict.number_of_image_frames_to_generate
imshow_torch(x[0][num_frames//2], False, 'original downsampled')
imshow_torch(outputs_dict['output_frames_original'][0][num_frames//2]/255, False, 'original HR')
imshow_torch(x_sr[0][num_frames//2], False, 'SwinIR model')
# imshow_torch(x_sr[0].mean(axis=0), False, 'SwinIR model Mean')

# imshow_torch(x_SR, False, 'EDVR SR model')
# imshow_torch(x_basicvsr[0][5//2], False, 'BasicVSR model')
# imshow_torch(x_iconvsr[0][5//2], False, 'IconVSR model')

plt.show(block=True)


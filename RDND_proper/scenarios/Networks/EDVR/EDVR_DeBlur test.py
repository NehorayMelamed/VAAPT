from RapidBase.import_all import *

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
from RDND_proper.models.BasicSRVSR.basicsr.archs import basicvsr_arch
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import *
from RDND_proper.models.BasicSRVSR.basicsr.archs.basicvsr_arch import *


### Initialize Network State Dict: ###
Train_dict = EasyDict()

######################################################################################################################
### Paths: ###
#(1). General Paths:
project_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD')
inference_path_master = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints")
TensorBoard_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\TensorBoard")
Network_checkpoint_folder = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\Model_New_Checkpoints")
datasets_main_folder = path_fix_path_for_linux("/home/mafat/DataSets")
#(2). Train Images:
Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW')
# Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/Image_SuperResolution/RealSR (Final)/Nikon/Train/2')
#(3). Test Images
Test_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW')
# Test_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/real_RGB_noisy_images')  #RGB images
# Test_Images_Folder = os.path.join(datasets_main_folder, '/DataSets\DIV2K\Official_Test_Images\Original_Images')  #Official & Interesting Test Images
# Test_Images_Folder = os.path.join(datasets_main_folder, '/Example Videos/Beirut_BW_Original') #Beirut Drone Movie
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
Train_dict.Network_checkpoint_prefix = 'EDVR_deblur_1'

# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Train_dict.flag_continue_checkpoint_naming_convention = False

Train_dict.network_free_flow_description = ''
######################################################################################################################
## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=222)

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
Train_dict.main_file_to_save_for_later = __file__
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
IO_dict.number_of_image_frames_to_generate = 5 #5
IO_dict.flag_to_RAM = False
IO_dict.flag_noise_images_to_RAM = False
### Post Loading Stuff: ###
IO_dict.flag_to_BW_before_noise = False
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
IO_dict.shift_size = 0.5*4
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
IO_dict.initial_crop_size = [128*4+5,128*4+5]
IO_dict.final_crop_size = [128*4,128*4]

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
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolutionDeblur_DownSampleOnTheFly_Bicubic(Train_Images_Folder, IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolutionDeblur_DownSampleOnTheFly_Bicubic(Test_Images_Folder, IO_dict_test)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
### Number Of Elements: ###
IO_dict.num_mini_batches_trn = len(train_dataset) // IO_dict.batch_size
IO_dict.num_mini_batches_val = len(test_dataset) // 1
IO_dict.train_dataset_length = len(train_dataset)
IO_dict.test_dataset_length = len(test_dataset)
######################################################################################################################
# try ocr dataset
# try other edvr deblurs
# try real reds dataset sequence comparison
#TODO: add OCR metrics
#TODO: add finetuning

### Get Batch: ###
for batch in train_dataloader:
    outputs_dict = batch
    break

batch_orig = outputs_dict['center_frame_original']
first_orig = batch_orig[0]

### Model paths: ###
edvr_compressed_deblur_model_path = 'C:/Users/Omer Leibovitch/Desktop/Work/IMOD/checkpoints/EDVR/EDVR_L_deblurcomp_REDS_official-0e988e5c.pth' # motion blur + video compression artifacts.
edvr_deblur_model_path = 'C:/Users/Omer Leibovitch/Desktop/Work/IMOD/checkpoints/EDVR/EDVR_L_deblur_REDS_official-ca46bd8c.pth' # standard deblurring (motion blur).

edvr_sr_model_path = 'C:/Users/Omer Leibovitch/Desktop/Work/IMOD/checkpoints/EDVR/EDVR_L_x4_SR_REDS_official-9f5f5039.pth'
basicvsr_model_path = 'C:/Users/Omer Leibovitch/Desktop/Work/IMOD/checkpoints/BasicVSR/BasicVSR_REDS4-543c8261.pth'
iconvsr_model_path = 'C:/Users/Omer Leibovitch/Desktop/Work/IMOD/checkpoints/BasicVSR/IconVSR_Vimeo90K_BDx4-cfcb7e00.pth'

### Get Models: ###
model_edvr_deblur = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
model_edvr_deblur.load_state_dict(torch.load(edvr_deblur_model_path)['params'])

model_edvr_compressed_deblur = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
model_edvr_compressed_deblur.load_state_dict(torch.load(edvr_compressed_deblur_model_path)['params'])

model_edvr_deblur = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
model_edvr_deblur.load_state_dict(torch.load(edvr_deblur_model_path)['params'])

model_edvr_sr = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur= False, with_tsa=True, hr_in=False).cuda()
model_edvr_sr.load_state_dict(torch.load(edvr_sr_model_path)['params'])

model_basicvsr = BasicVSR(num_feat=64, num_block=30, spynet_path = None).cuda()
model_basicvsr.load_state_dict(torch.load(basicvsr_model_path)['params'])

model_iconvsr = IconVSR(num_feat=64, num_block=15, keyframe_stride=5, temporal_padding=3, spynet_path=None, edvr_path=None).cuda()
model_iconvsr.load_state_dict(torch.load(iconvsr_model_path)['params'], strict=False) # They left me no choice

### Get Batch: ###
batch_output_frames = outputs_dict['output_frames_noisy']
x = batch_output_frames/255
if x.shape[2] == 1: # x is single channel
    x = x.expand(-1, -1, 3, -1, -1) # make x 3-channel


### Upsample Bicubic: ###
upsample_layer = nn.Upsample(scale_factor=4, mode='bicubic')
stacked_x = x.view(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
upsampled_stacked_x = upsample_layer(stacked_x)
x_upsampled_bicubic = upsampled_stacked_x.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3]*4,x.shape[4]*4)

### Test Models: ###
with torch.no_grad():
    x_deblur = model_edvr_deblur(x_upsampled_bicubic.cuda())

with torch.no_grad():
    x_compressed_deblur = model_edvr_compressed_deblur(x_upsampled_bicubic.cuda())

with torch.no_grad():
    x_SR = model_edvr_sr(x.cuda())

with torch.no_grad():
    x_basicvsr = model_basicvsr(x.cuda())

# with torch.no_grad():
#     x_iconvsr = model_iconvsr(x.cuda())

### Show Plots: ###
imshow_torch(x[0][2], False, 'original downsampled')
imshow_torch(outputs_dict['output_frames_original'][0][2]/255, False, 'original HR')
imshow_torch(x_deblur, False, 'EDVR deblur model')
imshow_torch(x_compressed_deblur, False, 'EDVR compressed deblur model')
# imshow_torch(x_SR, False, 'EDVR SR model')
# imshow_torch(x_basicvsr[0][5//2], False, 'BasicVSR model')
# imshow_torch(x_iconvsr[0][5//2], False, 'IconVSR model')
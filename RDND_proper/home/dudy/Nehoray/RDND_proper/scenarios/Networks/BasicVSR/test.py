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
from RDND_proper.models.BasicSRVSR.basicsr.archs import basicvsr_arch

### Initialize Network State Dict: ###
Train_dict = EasyDict()

######################################################################################################################
### Paths: ###
#(1). General Paths

project_path = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/IMOD')
inference_path_master = path_fix_path_for_linux("C:/Users/Omer Leibovitch/Desktop/Work/ytorch_Checkpoints")
TensorBoard_path = path_fix_path_for_linux("C:/Users/Omer Leibovitch/Desktop/Work/home/mafat/Pytorch_Checkpoints/TensorBoard")
Network_checkpoint_folder = path_fix_path_for_linux("C:/Users/Omer Leibovitch/Desktop/Work/Pytorch_Checkpoints/Model_New_Checkpoints")
#(2). Train Images:
Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/REDS/val_sharp/val/val_sharp/029')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u')
#(3). Test Images
#Test_Images_Folder = os.path.join(datasets_main_folder, '/Vidmeo90K/vimeo_septuplet/sequences/00001/0021')
Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/REDS/val_sharp/val/val_sharp/029')  #RGB images
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u')
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
TrainingCore.training_utils.set_seed(seed=422)

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
IO_dict.number_of_image_frames_to_generate = 3 #5
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
IO_dict.sigma_to_dataset = 1.5#1 / np.sqrt(IO_dict.SNR) * 255
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
IO_dict.initial_crop_size = [128*1,128*1]
IO_dict.final_crop_size = [128*1,128*1]

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
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
### Number Of Elements: ###
IO_dict.num_mini_batches_trn = len(train_dataset) // IO_dict.batch_size
IO_dict.num_mini_batches_val = len(test_dataset) // 1
IO_dict.train_dataset_length = len(train_dataset)
IO_dict.test_dataset_length = len(test_dataset)
######################################################################################################################
'''
for a in train_dataloader:
    pic = a
    break

#model_path = 'checkpoints/BasicVSR/BasicVSR_REDS4-543c8261.pth'
model_path = 'checkpoints/BasicVSR/BasicVSR_Vimeo90K_BIx4-2a29695a.pth'

device = torch.device('cuda')
model = basicvsr_arch.BasicVSR(num_feat=64, num_block=30).to(device)
#list(model.parameters())[112]
model.load_state_dict(torch.load(model_path)['params'])

#x = torch.randn(1,5,3,64,64).to(device)
#xx = model(x)
#print ('finished')


'''
for a in train_dataloader:
    pic = a
    break

batch_orig = pic['center_frame_original']
first_orig = batch_orig[0]

imshow_torch(first_orig/255)

device = torch.device('cuda')

# model = RLSP()
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import *

model = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=3, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=False)
if Train_dict.flag_use_new_or_checkpoint_network == 'checkpoint':
    model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=model)
else:
    Train_dict.Network_checkpoint_step = 0
model.to(device)
x = pic['output_frames_original'].to(device)
xx = model(x)
xxx = xx[0][2]
imshow_torch(xxx/255)

'''
model2 = basicvsr_arch.BasicVSR(num_feat=64, num_block=30).to(device)
model2.load_state_dict(torch.load(model_path)['params'])
xx2 = model2(x)
xxx2 = xx2[0][2]
imshow_torch(xxx2/255)
'''

x=3
model=2
pic  = 4
xx = 2
torch.cuda.empty_cache()

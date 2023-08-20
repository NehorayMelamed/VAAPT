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
from RDND_proper.models.Restormer.Restormer import Restormer
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import *
from RDND_proper.models.BasicSRVSR.basicsr.archs.basicvsr_arch import *


### Initialize Network State Dict: ###
Train_dict = EasyDict()

######################################################################################################################
### Paths: ###
#(1). General Paths
base_path = path_fix_path_for_linux('/media/mmm/DATADRIVE6/Omer')
project_path = path_fix_path_for_linux(os.path.join(base_path, 'IMOD'))
inference_path_master = path_fix_path_for_linux(os.path.join(base_path, 'Inference'))
TensorBoard_path = path_fix_path_for_linux(os.path.join(base_path, 'TensorBoard'))
Network_checkpoint_folder = path_fix_path_for_linux(os.path.join(base_path, 'Model Checkpoints'))
#(2). Train Images:
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/test')
# Train_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data/Div2K/DIV2K_train_HR_BW'))
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Drones')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Thermal_Images')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u/')
Train_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'test'))
#(3). Test Images
# Test_Images_Folder = os.path.join(datasets_main_folder, '/Vidmeo90K/vimeo_septuplet/sequences/00001/0021')
# Test_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data/Div2K/DIV2K_train_HR_BW'))
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Drones')
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Thermal_Images')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u/')
Test_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'test'))
#RGB images
Inference_folder_name_addon = ''
######################################################################################################################


######################################################################################################################
### Train/Inference/Debug Parameters: ###
Train_dict.is_training = True
Train_dict.flag_do_validation = True
Train_dict.flag_movie_inference = False

### Number of Epochs / Iterations: ###
Train_dict.number_of_epochs = 1e9
Train_dict.max_total_number_of_batches = 1e9

### Frequencies/TensorBoard: ###
Train_dict.debug_step = 200
Train_dict.save_models_frequency = 1000
Train_dict.validation_set_frequency = 200
Train_dict.tensorboard_train_frequency = 200
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
Train_dict.Network_checkpoint_prefix = 'Restormer_finetune_L1_3e-4_yes-gradclip_overfit-single_image-scale=1_not_pretrained'

# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Train_dict.flag_continue_checkpoint_naming_convention = False

Train_dict.network_free_flow_description = ''
######################################################################################################################
## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=42)

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
IO_dict.flag_crop_mode = 'center'
scale = 1 # To make crop_size changes easier
offset = 5
IO_dict.initial_crop_size = [int(256*scale)+offset, int(256*scale)+offset]
IO_dict.final_crop_size = [int(256*scale),int(256*scale)]

### Test/Validation Dict: ###
IO_dict_test = EasyDict(IO_dict)
IO_dict_test.flag_crop_mode = 'center'
IO_dict_test.initial_crop_size = [int(256*scale)+offset, int(256*scale)+offset] #np.inf
IO_dict_test.final_crop_size = [int(256*scale),int(256*scale)] #np.inf
IO_dict_test.mean_gray_level_per_pixel = [10,10]
#######################################################################################################################################



#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolutionDeblur_DownSampleOnTheFly_Bicubic(Train_Images_Folder, IO_dict)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolutionDeblur_DownSampleOnTheFly_Bicubic(Test_Images_Folder, IO_dict_test)

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

### Get Batch: ###
batch_output_frames = outputs_dict['output_frames_noisy']
x = batch_output_frames/255
y = outputs_dict['center_frame_original']/255
#x = x.squeeze(dim=2)
### Model path: ###
model_path = os.path.join(base_path,'Pretrained Checkpoints/Restormer/motion_deblurring.pth')

### Get Models: ###
model = Restormer().cuda()
model.load_state_dict(torch.load(model_path)['params'])

model_path_edvr_compressed_deblur = os.path.join(base_path,'Pretrained Checkpoints/EDVR/EDVR_L_deblurcomp_REDS_official-0e988e5c.pth') # motion blur + video compression artifacts.

model = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
model.load_state_dict(torch.load(model_path_edvr_compressed_deblur)['params'])

### Get Batch: ###

Train_dict.Network_checkpoint_step = 0
#####################################################################################################################################################################
### Get Network Preprocessing/Postprocessing: ### #Change to swinIR
preprocessing_object = PreProcessing_EDVR()
preprocessing_object.flag_BW2RGB = True
postprocessing_object = PostProcessing_EDVR()
postprocessing_object.flag_RGB2BW = True
#####################################################################################################################################################################


#####################################################################################################################################################################
### Optimizer: ###
Network_Optimizer_object = Optimizer_Base(model,
                                          intial_lr=3e-4,
                                          hypergrad_lr=0)
if Train_dict.flag_use_new_or_checkpoint_optimizer == 'checkpoint':
    Network_Optimizer_object.Network_Optimizer = TrainingCore.training_utils.load_optimizer_from_checkpoint(
        Network_Optimizer_object.Network_Optimizer, Train_dict)
optimization_dict = Network_Optimizer_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### LR Scheduler: ###
optimization_dict.minimum_learning_rate = 1e-6
optimization_dict.learning_rate_lowering_factor = 0.8
optimization_dict.learning_rate_lowering_patience_before_lowering = 2000
optimization_dict.flag_momentum_cycling = False
optimization_dict.momentum_bounds = [0.85,0.95]
optimization_dict.flag_uptick_learning_rate_after_falling_below_minimum = False
LR_scheduler_object = LR_Scheduler_ReduceLROnPlateauCustom(Network_Optimizer_object.Network_Optimizer, input_dict=optimization_dict)
optimization_dict = LR_scheduler_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### Gradient Clipping: ###
optimization_dict.flag_use_gradient_clipping = True
optimization_dict.flag_clip_grad_value_or_norm = 'norm'
optimization_dict.anomaly_factor_threshold = 3
optimization_dict.gradient_clip_norm = 20
optimization_dict.gradient_clip_value_factor = 6.6/34  # initial_grad_norm/initial_grad_max_value
Clip_gradient_object = Clip_Gradient_Base(input_dict=optimization_dict)
optimization_dict = Clip_gradient_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### Get Loss Object: ###
Loss_object = TrainingCore.losses.Loss_Simple(inputs_dict=None)
losses_dict = Loss_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### TensorBoard: ###
if Train_dict.flag_write_TB:
    TB_writer_train = SummaryWriter(os.path.join(TensorBoard_path, Train_dict.Network_checkpoint_prefix + '_Train'))
    TB_writer_test = SummaryWriter(os.path.join(TensorBoard_path, Train_dict.Network_checkpoint_prefix + '_Validation'))
else:
    TB_writer = None
TB_object_train = TB_Denoising(TB_writer_train)
TB_object_test = TB_Denoising(TB_writer_test)
TB_objects = [TB_object_train, TB_object_test]
#####################################################################################################################################################################


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


#####################################################################################################################################################################
### Callbacks Names: ###

if IO_dict.flag_use_train_dataset_split:
    dataset_name = str(test_dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
else:
    dataset_name = str(test_dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
inference_path = os.path.join(inference_path_master, 'Inference', Train_dict.Network_checkpoint_prefix + '/' + dataset_name)
if Train_dict.is_training == False: #If i'm doing validation (and not just outputing stuff while training) - create a special folder to save results
    inference_path = os.path.join(inference_path, os.path.split(Test_Images_Folder)[-1] + '_' + Inference_folder_name_addon)

### Get Debug Callback: ###
debug_callback = InferenceCallback_DenoisingSuperResolution_Base(inference_path + '_debug', Train_dict)  #TODO: isn't the Train_dict.is_training useless now?!!
debug_callback.frame_index_to_predict = 0
### Get Validation Callback: ###
validation_callback = InferenceCallback_DenoisingSuperResolution_Base(inference_path + '_validation', Train_dict)
validation_callback.frame_index_to_predict = 0
### Get Inference Callback: ###
inference_callback = validation_callback
#####################################################################################################################################################################



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
    assert (Train_dict.flag_use_new_or_checkpoint_network == 'checkpoint')
    Trainer.inference()

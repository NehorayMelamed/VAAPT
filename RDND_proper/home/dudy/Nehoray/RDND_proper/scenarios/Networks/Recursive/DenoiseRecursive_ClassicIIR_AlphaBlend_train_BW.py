from RapidBase.import_all import *

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.callbacks import InferenceCallback
from RapidBase.TrainingCore.datasets import Dataset_MultipleImagesFromSingleImage
from RapidBase.TrainingCore.losses import Loss_DVDNet
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_Base, PreProcessing_DVDNet_BW, PreProcessing_FFDNet_Recursive_2
from RapidBase.TrainingCore.pre_post_callbacks import PostProcessing_DVDNet
from RapidBase.TrainingCore.tensorboard_callbacks import TB_Denoising, TB_Denoising_Recursive
from RapidBase.TrainingCore.lr import LR_Scheduler_Base
from RapidBase.TrainingCore.optimizers import Optimizer_Base
from RapidBase.TrainingCore.trainer import GeneralTrainer, GeneralTrainer_Recursive
from RapidBase.TrainingCore.clip_gradients import Clip_Gradient_Base
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_Base, PreProcessing_FFDNet, PreProcessing_FFDNet_Recursive, PostProcessing_Recursive_Base
from RapidBase.TrainingCore.pre_post_callbacks import PostProcessing_Base, PostProcessing_ClassicIIR, PreProcessing_ClassicIIR_AlphaBlend, PostProcessing_ClassicIIR_AlphaBlend

### Import Models: ###
from RDND_proper.models.DVDNet.DVDNet import *
# from RDND_proper.models.KAIR_FFDNet_and_more.utils import utils_logger
# from RDND_proper.models.KAIR_FFDNet_and_more.utils import utils_image as util
from RDND_proper.models.KAIR_FFDNet_and_more.models.network_ffdnet import FFDNet_dudy as FFDNet
from RDND_proper.models.KAIR_FFDNet_and_more.models.network_ffdnet import FFDNet_dudy2, FFDNet_dudy_Recursive
from RDND_proper.models.Classic_IIR.Classic_IIR import Classic_IIR_alpha_blend


### Initialize Network State Dict: ###
Network_save_load_dict = EasyDict()
Train_dict = EasyDict()


### Paths: ###
##############
project_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD')
inference_path_master = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints")
TensorBoard_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\TensorBoard")
Network_checkpoint_folder = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\Model_New_Checkpoints")
datasets_main_folder = path_fix_path_for_linux("/home/mafat/DataSets")
Train_Images_Folder = os.path.join(datasets_main_folder, '/DataSets\Div2K\DIV2K_train_HR_BW')
Test_Images_Folder = os.path.join(datasets_main_folder, '/DataSets\Div2K\DIV2K_train_HR_BW')

Inference_folder_name_addon = 'BW_SNR5_Checkpoint_Training'
###

### Train/Inference/Debug Parameters: ###
Train_dict.debug_step = 200
Train_dict.is_training = True
Train_dict.flag_do_validation = False
Train_dict.flag_movie_inference = False

### Frequencies/TensorBoard: ###
Train_dict.save_models_frequency = 200
Train_dict.validation_set_frequency = 100
Train_dict.flag_plot_grad_flow = False
Train_dict.tensorboard_train_frequency = 100
Network_save_load_dict.flag_write_TB = True
###

### Temporal/Recursive Parameters: ###
### Time Parametrs: ###
Train_dict.number_of_total_backward_steps_per_image = 5 #number of backward steps before changing images batch
Train_dict.number_of_time_steps_in_one_backward = 5 #number of time steps before doing a .backward()
Train_dict.total_number_of_time_steps = Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward

### Time Weights: ###
Train_dict.flag_use_all_frames = True # True=all frames (with below listed weights), False=only last frame
# Same Weight: #
Train_dict.time_steps_weights = [1] * (Train_dict.number_of_time_steps_in_one_backward * Train_dict.number_of_total_backward_steps_per_image)
# Linearly Increasing Weight: #
step = 0.05
if step != 0:
    Train_dict.time_steps_weights += np.arange(0, step * Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward, step)
# # Something Else: #
# Train_dict.time_steps_weights = [0,0,0,0,1] * 5

### Flags: ###
Train_dict.reset_every_batch = False
Train_dict.flag_initialize_hidden_with_zero_or_input = 'input' #'zero' / 'input'
Train_dict.flag_residual_learning = False
Train_dict.flag_clip_noisy_tensor_after_noising = False
Train_dict.flag_learn_clean_or_running_average = 'clean' # 0=clean, 1=running average
# Train_dict.flag_learn_clean_or_running_average = 'running_average' # 0=clean, 1=running average
###

## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=12344)

#### GPU Allocation Program: ####
# device0 = torch.device("cuda:0")
# device1 = torch.device("cuda:1")
# Network_device = device0
# other_device = device0
Network_device = torch.device('cuda')
other_device = torch.device('cuda')
###################################




######################################################################################################################
### Save Network: ###
#######################
# (1.1). Name to Save checkpoint as when training new network:
# don't use the string 'Step' at any point in the prefix!!@#!@$@#
Network_save_load_dict.Network_checkpoint_prefix = 'ClassicIIR_AlphaBlend_1'
Network_save_load_dict.Network_checkpoint_prefix = 'ClassicIIR_AlphaBlend_Gradient_1'

# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Network_save_load_dict.Network_checkpoint_prefix_addition = ''  # No Change In Name From Checkpoint
######################################################################################################################


######################################################################################################################
### Load Network: ###
#######################
Network_save_load_dict.load_Network_filename = path_fix_path_for_linux(r'/home/mafat\PycharmProjects\IMOD\models\DVDNet\dvdnet/model_combined.pth')
# Network_save_load_dict.load_Network_filename = r'ffdnet_gray_TEST1_Step50.tar'
######################################################################################################################


########################################
### New Training Or From Checkpoint: ###
# 'new', 'checkpoint'                # Use New Network or Checkpoint
Network_save_load_dict.flag_use_new_or_checkpoint_network = 'new'
Network_save_load_dict.flag_use_new_or_checkpoint_optimizer = 'new'
########################################


##########################################################################################
### Adjust Network Checkpoint string according to script variables/prefixes/postfixes: ###
Network_save_load_dict = TrainingCore.training_utils.PreTraining_Aux(Network_save_load_dict, Network_checkpoint_folder, project_path)
##########################################################################################


###################################
### Train Or Debug: ###
Train_dict.flag_1_train_iteration_to_debug = False
### Control Number of Training Iterations (and whether i'm in loop debug mode or not): ###
if Train_dict.flag_1_train_iteration_to_debug:
    Train_dict.number_of_epochs = 1
    Train_dict.max_total_number_of_batches = 1
    Train_dict.flag_limit_number_of_iterations = True
else:
    Train_dict.max_total_number_of_batches = 1e9  # 10000
    Train_dict.flag_limit_number_of_iterations = False
###################################



#######################################################################################################################################
### IO: ###
IO_dict = EasyDict()
### Crops: ##
if (Train_dict.is_training):
    IO_dict.flag_crop = True
else:
    IO_dict.flag_crop = False
### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_use_train_dataset_split = True
IO_dict.train_dataset_split_to_test_factor = 0.01
### Noise Parameters: ###
IO_dict.sigma_to_dataset = 1/sqrt(10)*255
IO_dict.SNR_to_model = 10
IO_dict.sigma_to_model = 1/sqrt(IO_dict.SNR_to_model)*255
### Blur Parameters: ###
IO_dict.blur_size = 0
### Number of frames to load: ###
IO_dict.NUM_IN_FRAMES = 5 # temporal size of patch
### Training Flags: ###
IO_dict.non_valid_border_size = 30
### Universal Training Parameters: ###
IO_dict.batch_size = 8
IO_dict.number_of_epochs = 60000
#### Assign Device (for instance, we might want to do things on the gpu in the dataloader): ###
IO_dict.device = Network_device
#(*). DataSet & DataLoader Parameters:
IO_dict.max_shift_in_pixels = 0
IO_dict.max_search_area = 11 #for algorithms like NCC
IO_dict.SNR = 5
IO_dict.crop_size_x = 1024
IO_dict.crop_size_y = 1024
IO_dict.inner_crop_size = 1000
IO_dict.noise_sigma = 1/sqrt(IO_dict.SNR)


### Model Dict: ###
#(*). Model Type:
IO_dict.flag_register_images_method = 'zero_shift'  # zero_shift, neural, classic
IO_dict.flag_estimate_noise_method = 'known_level'  # known_level, classical_iterative_residual
IO_dict.flag_estimate_reset_gate_or_alpha_blend = 'reset_gate'  # reset_gate, alpha_blend
IO_dict.flag_get_classical_combine_predictions = False
#(*). Model Flags & Parameters:
IO_dict.registration_algorithm = 'DeepFlow'  #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR, NormalizedCrossCorrelation
IO_dict.noise_estimation_algorithm = 'IterativeResidual'  #IterativeResidual, FullFrameHistogram
IO_dict.flag_do_initial_SAD = False
IO_dict.downsample_kernel_size_FrameCombine = 8
IO_dict.downsample_kernel_size_NoiseEstimation = 8
IO_dict.downsample_factor_registration = 2
IO_dict.inner_crop_size = np.inf
IO_dict.max_search_area = 10


#(*). General Auxiliary Parameters:
IO_dict.flag_save_uint16 = False
IO_dict.flag_normalized_dataloader_by_255 = False
# max_value_possible = 2**16-1
IO_dict.max_value_possible = 256
IO_dict.image_index_to_start_from = 10
IO_dict.image_index_to_stop_at = 140
IO_dict.number_of_images = 100
#######################################################################################################################################




#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AddNoise(Train_Images_Folder,
                                                                    number_of_image_frame_to_generate=Train_dict.total_number_of_time_steps,
                                                                    base_transform=None, batch_transform=None, image_loader=ImageLoaderCV,
                                                                    max_number_of_images=np.inf,
                                                                    crop_size=128*2,
                                                                    flag_to_RAM=True, flag_recursive=False, flag_normalize_by_255=False, flag_crop_mode='random',
                                                                    flag_explicitely_make_tensor=False, allowed_extentions=IMG_EXTENSIONS, flag_to_BW=False, flag_to_RGB=False,
                                                                    flag_base_tranform=False, flag_batch_transform=False, flag_how_to_concat='C', Train_dict=IO_dict)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AddNoise(Test_Images_Folder,
                                                                    number_of_image_frame_to_generate=Train_dict.total_number_of_time_steps,
                                                                    base_transform=None, batch_transform=None, image_loader=ImageLoaderCV,
                                                                    max_number_of_images=25,
                                                                    crop_size=128*2,
                                                                    flag_to_RAM=False, flag_recursive=False, flag_normalize_by_255=False, flag_crop_mode='random',
                                                                    flag_explicitely_make_tensor=False, allowed_extentions=IMG_EXTENSIONS, flag_to_BW=False, flag_to_RGB=False,
                                                                    flag_base_tranform=False, flag_batch_transform=False, flag_how_to_concat='C', Train_dict=IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AddNoise_Blur(Test_Images_Folder,
#                                                                     number_of_image_frame_to_generate=5,
#                                                                     base_transform=None, batch_transform=None, image_loader=ImageLoaderCV,
#                                                                     max_number_of_images=25,
#                                                                     crop_size=np.inf,
#                                                                     flag_to_RAM=False, flag_recursive=False, flag_normalize_by_255=False, flag_crop_mode='random',
#                                                                     flag_explicitely_make_tensor=False, allowed_extentions=IMG_EXTENSIONS, flag_to_BW=False, flag_to_RGB=True,
#                                                                     flag_base_tranform=False, flag_batch_transform=False, flag_how_to_concat='T', Train_dict=IO_dict)
# test_dataset = TrainingCore.datasets.DataSet_Videos_In_Folders_AddNoise(root_folder=Test_Images_Folder,
#                                                         transform_per_image=None,
#                                                          transform_on_all_images=None,
#                                                          image_loader=ImageLoaderCV,
#                                                          number_of_images_per_video=100,  #TODO: add possibility to randomize start index from a possibly long video
#                                                          max_number_of_videos=np.inf,
#                                                          crop_size=100,
#                                                          flag_to_RAM=True,
#                                                          flag_explicitely_make_tensor=True,
#                                                          flag_normalize_by_255=True,
#                                                          flag_how_to_concat='T',
#                                                          flag_return_torch_on_load=False,
#                                                          flag_to_BW=False,
#                                                          flag_to_RGB=True, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
#                                                          Train_dict=IO_dict)
# test_dataset = TrainingCore.datasets.DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=Test_Images_Folder,
#                                                         transform_per_image=None,
#                                                          transform_on_all_images=None,
#                                                          image_loader=ImageLoaderCV,
#                                                          number_of_images_per_video_to_scan=100,  #TODO: add possibility to randomize start index from a possibly long video
#                                                          number_of_images_per_video_to_load=5,
#                                                          max_number_of_videos=np.inf,
#                                                          crop_size=np.inf,
#                                                          flag_to_RAM=True,
#                                                          flag_explicitely_make_tensor=True,
#                                                          flag_normalize_by_255=True,
#                                                          flag_how_to_concat='T',
#                                                          flag_return_torch_on_load=False,
#                                                          flag_to_BW=False,
#                                                          flag_to_RGB=True, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
#                                                          Train_dict=IO_dict)
### DataLoaders: ###
train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
### Number Of Elements: ###
if Train_dict.is_training == False:
    IO_dict.batch_size = 1
IO_dict.num_mini_batches_trn = len(train_dataset) // IO_dict.batch_size
IO_dict.num_mini_batches_val = len(test_dataset) // 1
IO_dict.train_dataset_length = len(train_dataset)
IO_dict.test_dataset_length = len(test_dataset)

### Get Automatic Train/Test Split From Train Set: ###
if IO_dict.flag_use_train_dataset_split:
    test_dataset_length_new = int(np.round(len(train_dataset)*IO_dict.train_dataset_split_to_test_factor))
    train_dataset_length_new = len(train_dataset) - test_dataset_length_new
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_dataset_length_new, test_dataset_length_new])
    train_dataset.indices = np.arange(0, train_dataset_length_new)
    test_dataset.indices = np.arange(train_dataset_length_new, train_dataset_length_new+test_dataset_length_new)
    IO_dict.num_mini_batches_trn = train_dataset_length_new // IO_dict.batch_size
    IO_dict.num_mini_batches_val = test_dataset_length_new // IO_dict.batch_size
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
######################################################################################################################################



#####################################################################################################################################################################
### Model Itself: ###
model = Classic_IIR_alpha_blend(IO_dict, in_channels=3, out_channels=1)
if Network_save_load_dict.flag_use_new_or_checkpoint_network == 'checkpoint':
    model, Network_save_load_dict = TrainingCore.training_utils.load_model_from_checkpoint(Network_save_load_dict=Network_save_load_dict, model=model)
else:
    Network_save_load_dict.Network_checkpoint_step = 0
#####################################################################################################################################################################


#####################################################################################################################################################################
### Get Network Preprocessing/Postprocessing: ###
preprocessing_object = PreProcessing_ClassicIIR_AlphaBlend()
postprocessing_object = PostProcessing_ClassicIIR_AlphaBlend()
#####################################################################################################################################################################


#####################################################################################################################################################################
### Optimizer: ###
Network_Optimizer_object = Optimizer_Base(model,
                                          intial_lr=1e-3,
                                          hypergrad_lr=0)
if Network_save_load_dict.flag_use_new_or_checkpoint_optimizer == 'checkpoint':
    Network_Optimizer_object.Network_Optimizer = TrainingCore.training_utils.load_optimizer_from_checkpoint(
        Network_Optimizer_object.Network_Optimizer, Network_save_load_dict)
optimization_dict = Network_Optimizer_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### LR Scheduler: ###
optimization_dict.minimum_learning_rate=1e-5
optimization_dict.learning_rate_lowering_factor=0.8
optimization_dict.learning_rate_lowering_patience_before_lowering=5000
LR_scheduler_object = LR_Scheduler_Base(Network_Optimizer_object.Network_Optimizer, input_dict=optimization_dict)
optimization_dict = LR_scheduler_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### Gradient Clipping: ###
Clip_gradient_object = Clip_Gradient_Base(input_dict=optimization_dict)
optimization_dict = Clip_gradient_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### Get Loss Object: ###
Loss_object = TrainingCore.losses.Loss_Recursive_Simple(inputs_dict=None)
# Loss_object = TrainingCore.losses.Loss_Gradient(inputs_dict=None)
losses_dict = Loss_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### TensorBoard: ###
if Network_save_load_dict.flag_write_TB:
    TB_writer = SummaryWriter(os.path.join(TensorBoard_path, Network_save_load_dict.Network_checkpoint_prefix))
else:
    TB_writer = None
TB_object = TB_Denoising_Recursive(TB_writer)
#####################################################################################################################################################################


#####################################################################################################################################################################
### Combine dictionaries to Train_dict: ###
Train_dict.optimization_dict = optimization_dict
Train_dict.IO_dict = IO_dict
Train_dict.losses_dict = losses_dict
Train_dict.Network_save_load_dict = Network_save_load_dict
Train_dict.update(optimization_dict)
Train_dict.update(IO_dict)
Train_dict.update(losses_dict)
Train_dict.update(Network_save_load_dict)
#####################################################################################################################################################################



### Define General Trainer: ###
Trainer = GeneralTrainer_Recursive(model=model,
                         preprocessing_object=preprocessing_object,
                         postprocessing_object=postprocessing_object,
                         Network_Optimizer_object=Network_Optimizer_object,
                         Loss_object=Loss_object,
                         TB_object=TB_object,
                         LR_scheduler_object=LR_scheduler_object,
                         Clip_gradient_object=Clip_gradient_object,
                         train_dataloader=train_dataloader,
                         test_dataloader=test_dataloader,
                         Train_dict=Train_dict,
                         Network_save_load_dict=Network_save_load_dict,
                         Network_device=Network_device,
                         other_device=other_device,
                         Network_checkpoint_folder=Network_checkpoint_folder)


### Inference Callback: ###
if IO_dict.flag_use_train_dataset_split:
    dataset_name = str(test_dataset.dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
else:
    dataset_name = str(test_dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
inference_path = os.path.join(inference_path_master, 'Inference', Network_save_load_dict.Network_checkpoint_prefix + '/' + dataset_name)
if Train_dict.is_training==False: #If i'm doing validation (and not just outputing stuff while training) - create a special folder to save results
    inference_path = os.path.join(inference_path, os.path.split(Test_Images_Folder)[-1] + '_' + Inference_folder_name_addon)


### Get Debug Callback: ###
# debug_callback = TrainingCore.callbacks.InferenceCallback(inference_path, Train_dict.is_training)
# debug_callback = TrainingCore.callbacks.InferenceCallback_ValidationDataSet(inference_path, Train_dict.is_training)
# debug_callback = TrainingCore.callbacks.InferenceCallback_ExampleFolder(inference_path, Train_dict.is_training)
# debug_callback = TrainingCore.callbacks.InferenceCallback_ExampleFolder_WithRunningMean(inference_path, Train_dict.is_training)
# debug_callback = TrainingCore.callbacks.InferenceCallback_ExampleFolder_Recursive(inference_path, Train_dict.is_training)
debug_callback = TrainingCore.callbacks.InferenceCallback_ExampleFolder_Recursive_ClassicIIR(inference_path, Train_dict.is_training)
# debug_callback = TrainingCore.callbacks.InferenceCallback_Movie(inference_path, Train_dict.is_training)
# debug_callback = None


### Train/Inference: ###
if Train_dict.is_training:
    Trainer.Train(debug_callback=debug_callback, debug_step=Train_dict.debug_step)
else:
    assert (Network_save_load_dict.flag_use_new_or_checkpoint_network == 'checkpoint')
    Trainer.inference(debug_callback, test_dataloader)  #TODO: probably, at least philosophically, we should have a debug_callback, validation_callback, inference_callback?







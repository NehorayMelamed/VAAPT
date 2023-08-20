from RapidBase.import_all import *

from torch.utils.data import DataLoader
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.losses import Loss_Simple
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import get_default_IO_dict


### Import Models: ###
from RDND_proper.models.KAIR_FFDNet_and_more.models.network_ffdnet import FFDNet_dudy as FFDNet
from RDND_proper.models.MPRNet.Denoising.MPRNet import MPRNet


### Initialize Network State Dict: ###
Train_dict = EasyDict()


### Paths: ###
##############
#(1). General Paths:
project_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD')
inference_path_master = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints")
TensorBoard_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\TensorBoard")
Network_checkpoint_folder = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\Model_New_Checkpoints")
datasets_main_folder = path_fix_path_for_linux("/home/mafat/DataSets")
#(2). Train Images:
Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW')
#(3). Test Images
Test_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW')
# Test_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/real_RGB_noisy_images')  #RGB images
# Test_Images_Folder = os.path.join(datasets_main_folder, '/DataSets\DIV2K\Official_Test_Images\Original_Images')  #Official & Interesting Test Images
# Test_Images_Folder = os.path.join(datasets_main_folder, '/Example Videos/Beirut_BW_Original') #Beirut Drone Movie
Camera_Noise_Folder = os.path.join(datasets_main_folder, '/KAYA_CAMERA_NOISE/noise')

Inference_folder_name_addon = ''
###

######################################################################################################################
### Train/Inference/Debug Parameters: ###
Train_dict.is_training = True
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


## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=12344)

#### GPU Allocation Program: ####
Network_device = torch.device('cuda')
other_device = torch.device('cuda')
###################################



######################################################################################################################
### Save Network: ###
#######################
# (1.1). Name to Save checkpoint as when training new network:
# don't use the string 'Step' at any point in the prefix!!@#!@$@#
Train_dict.Network_checkpoint_prefix = 'MPRNet_1'

# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Train_dict.flag_continue_checkpoint_naming_convention = False

# (1.3) Network "Free-Flow" Description To Be Saved In Checkpoint:
Train_dict.network_free_flow_description = ''
######################################################################################################################


######################################################################################################################
### Load Network: ###
#######################
Train_dict.load_Network_filename = r'/home/mafat/PycharmProjects/IMOD/models/MPRNet/Denoising/pretrained_models/model_denoising.pth'
Train_dict.load_Network_filename = r'/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/MPRNet_Video_2/MPRNet_Video_2_TEST1_Step43000.tar'
######################################################################################################################


########################################
### New Training Or From Checkpoint: ###
# 'new', 'checkpoint'                # Use New Network or Checkpoint
Train_dict.flag_use_new_or_checkpoint_network = 'new'
Train_dict.flag_use_new_or_checkpoint_optimizer = 'new'

Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = True
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
IO_dict.flag_use_train_dataset_split = True
IO_dict.train_dataset_split_to_test_factor = 0.05
### Loading Images: ###
IO_dict.flag_recursive = True
IO_dict.max_number_of_images = np.inf
IO_dict.number_of_image_frames_to_generate = 5
IO_dict.flag_to_RAM = False
IO_dict.flag_noise_images_to_RAM = False
### Post Loading Stuff: ###
IO_dict.flag_to_BW_before_noise = False
IO_dict.flag_to_RGB_before_noise = False
IO_dict.flag_to_BW_after_noise = False
IO_dict.flag_to_RGB_after_noise = False
IO_dict.flag_how_to_concat = 'T'
### Noise Parameters: ###
IO_dict.SNR = 20  # mean~10, std~2(readout)+sqrt(10)(shot)=5, effective_sigma~255/mean*5~127.5 -> SNR~4. this is only an approximation to better condition network
IO_dict.sigma_to_dataset = 1 / np.sqrt(IO_dict.SNR) * 255
IO_dict.SNR_to_model = IO_dict.SNR
### Camera Parameters: ### (*). INSTEAD OF ABOVE NOISE PARAMETERS!!!!
IO_dict.mean_gray_level_per_pixel = [200, 200]
IO_dict.electrons_per_gray_level = 14.53
IO_dict.photons_per_electron = 1
IO_dict.gray_levels_per_electron = 1 / IO_dict.electrons_per_gray_level
IO_dict.electrons_per_photon = 1 / IO_dict.photons_per_electron
### Shift / Directional-Blur Parameters: ###
IO_dict.shift_size = 1
IO_dict.blur_size = 0
### Gaussian Blur Parameters: ###
IO_dict.blur_number_of_channels = 1
IO_dict.blur_kernel_size = 3
IO_dict.blur_sigma = 1
### Universal Training Parameters: ###
IO_dict.batch_size = 2
#### Assign Device (for instance, we might want to do things on the gpu in the dataloader): ###
IO_dict.device = Network_device
IO_dict.flag_all_inputs_to_gpu = True
### Cropping: ###
IO_dict.flag_crop_mode = 'random'
IO_dict.initial_crop_size = [128*2,128*2]
IO_dict.final_crop_size = [128*2,128*2]
IO_dict.final_crop_size = [128*2,128*2]

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
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_Shifts_AddExternalNoiseImage(Train_Images_Folder, IO_dict)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_Shifts_AddExternalNoiseImage(Test_Images_Folder, IO_dict_test)
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



#####################################################################################################################################################################
### Model Itself: ###
#(*). currently this is a RGB model which i will finetune for BW. later on i will train a model with 1 input and output channels
model = MPRNet(in_c=5, out_c=1, n_feat=80, scale_unetfeats=48, scale_orsnetfeats=32, num_cab=8, kernel_size=3, reduction=4, bias=False)
if Train_dict.flag_use_new_or_checkpoint_network == 'checkpoint':
    model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=model)
else:
    Train_dict.Network_checkpoint_step = 0
#####################################################################################################################################################################


#####################################################################################################################################################################
### Get Network Preprocessing/Postprocessing: ###
preprocessing_object = PreProcessing_MPRNet()
postprocessing_object = PostProcessing_MPRNet()
#####################################################################################################################################################################


#####################################################################################################################################################################
### Optimizer: ###
Network_Optimizer_object = Optimizer_Base(model,
                                          intial_lr=1e-4,
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
    dataset_name = str(test_dataset.dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
else:
    dataset_name = str(test_dataset.__class__).replace('.','__').replace('<class \'','').replace('\'>','').replace('core__datasets__','')
inference_path = os.path.join(inference_path_master, 'Inference', Train_dict.Network_checkpoint_prefix + '/' + dataset_name)
if Train_dict.is_training == False: #If i'm doing validation (and not just outputing stuff while training) - create a special folder to save results
    inference_path = os.path.join(inference_path, os.path.split(Test_Images_Folder)[-1] + '_' + Inference_folder_name_addon)

### Get Debug Callback: ###
debug_callback = InferenceCallback_Denoise(inference_path + '_debug', Train_dict)  #TODO: isn't the Train_dict.is_training useless now?!!
debug_callback.frame_index_to_predict = 0
### Get Validation Callback: ###
validation_callback = InferenceCallback_Denoise(inference_path + '_validation', Train_dict)
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






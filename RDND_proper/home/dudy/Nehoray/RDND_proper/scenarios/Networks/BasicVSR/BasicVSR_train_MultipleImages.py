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

### Import Models: ###
from RDND_proper.models.mmediting.mmedit.apis import init_model, restoration_video_inference
from RDND_proper.models.mmediting.mmedit.core import tensor2img
from RDND_proper.models.mmediting.demo import *

### Initialize Network State Dict: ###
Train_dict = EasyDict()

######################################################################################################################
### Paths: ###
# (1). General Paths:
project_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD')
inference_path_master = os.path.join(project_path, 'Pytorch_Checkpoints')
TensorBoard_path = os.path.join(inference_path_master, 'TensorBoard')
Network_checkpoint_folder = os.path.join(inference_path_master, 'Model_New_Checkpoints')

# (2). Train Images:
#Train_Images_Folder = path_fix_path_for_linux('/media/mmm/Elements/datasets/vimeo_sample')
Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW')
# (3). Test Images
#Test_Images_Folder = path_fix_path_for_linux('/media/mmm/Elements/datasets/vimeo_sample')
Test_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW')

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
Train_dict.save_models_frequency = 200
Train_dict.validation_set_frequency = 200
Train_dict.tensorboard_train_frequency = 200
Train_dict.number_of_validation_samples = 45
Train_dict.flag_write_TB = True
Train_dict.flag_plot_grad_flow = False

### Temporal/Recursive Parameters: ###
### Time Parametrs: ###
Train_dict.number_of_time_steps_in_one_backward = 10  # 5 #number of time steps before doing a .backward()
Train_dict.number_of_total_backward_steps_per_image = 2  # 4 #number of backward steps before changing images batch
Train_dict.total_number_of_time_steps = Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward

### Time Weights: ###
Train_dict.flag_use_all_frames = True  # True=all frames (with below listed weights), False=only last frame
# Same Weight: #
Train_dict.time_steps_weights = [1] * (Train_dict.number_of_time_steps_in_one_backward * Train_dict.number_of_total_backward_steps_per_image)
# Linearly Increasing Weight: #
step = 0.05
if step != 0:
    Train_dict.time_steps_weights += np.arange(0,step * Train_dict.number_of_total_backward_steps_per_image * Train_dict.number_of_time_steps_in_one_backward, step)
# # Something Else: #
# Train_dict.time_steps_weights = [0,0,0,0,1] * 5
# Something Else: #
Train_dict.time_steps_weights = [1, 1, 1, 1, 1] * 5

### Flags: ###
Train_dict.reset_every_batch = True
Train_dict.flag_initialize_hidden_with_zero_or_input = 'input'  # 'zero' / 'input'
Train_dict.flag_residual_learning = False
Train_dict.flag_clip_noisy_tensor_after_noising = False
Train_dict.flag_learn_clean_or_running_average = 'clean'  # 0=clean, 1=running_average
###
######################################################################################################################


######################################################################################################################
## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=12344)
######################################################################################################################


######################################################################################################################
### Save Network: ###
#######################
# (1.1). Name to Save checkpoint as when training new network:
# don't use the string 'Step' at any point in the prefix!!@#!@$@#
Train_dict.Network_checkpoint_prefix = 'BasicVSR'

# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Train_dict.flag_continue_checkpoint_naming_convention = False

Train_dict.network_free_flow_description = ''
######################################################################################################################


######################################################################################################################
### Load Network: ###
#######################
checkpoint_path = '/home/mafat/PycharmProjects/IMOD/models/mmediting/checkpoints/basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth'
Train_dict.load_Network_filename = checkpoint_path
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
Train_dict.main_file_to_save_for_later = os.path.dirname(os.path.abspath("__file__"))
Train_dict = TrainingCore.training_utils.PreTraining_Aux(Train_dict, Network_checkpoint_folder, project_path)
##########################################################################################


#######################################################################################################################################
### IO: ###
IO_dict = get_default_IO_dict()
### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_use_train_dataset_split = True  # TODO: add possibility of splitting the train set as far as images but still use the IO_dict_test instead o the IO_dict_train
IO_dict.flag_only_split_indices = True  # test_dataset is still defined seperately with its own IO_dict_test but split on indices base
IO_dict.train_dataset_split_to_test_factor = 0.0005  # 46 examples
### Loading Images: ###
IO_dict.string_pattern_to_search = '*'
IO_dict.HR_string_pattern_to_search = '*HR*'
IO_dict.LR_string_pattern_to_search = '*LR*'
IO_dict.flag_recursive = True
IO_dict.max_number_of_images = np.inf
IO_dict.number_of_image_frames_to_generate = 20
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
IO_dict.sigma_to_dataset = 0  # 1 / np.sqrt(IO_dict.SNR) * 255
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
IO_dict.blur_sigma = 0
### Super Resolution: ###
IO_dict.downsampling_factor = 4
### Universal Training Parameters: ###
IO_dict.batch_size = 4
#### Decide whether to load all inputs to gpu immediately after dataloader: ###
IO_dict.flag_all_inputs_to_gpu = False
### Cropping: ###
IO_dict.flag_crop_mode = 'random'
IO_dict.initial_crop_size = [128 * 2, 128 * 2]
IO_dict.final_crop_size = [128 * 2, 128 * 2]

### Test/Validation Dict: ###
IO_dict_test = EasyDict(IO_dict)
IO_dict_test.flag_crop_mode = 'center'
IO_dict_test.initial_crop_size = [256, 256]
IO_dict_test.final_crop_size = [256, 256]
IO_dict_test.mean_gray_level_per_pixel = [10, 10]
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
### Get Automatic Train/Test Split From Train Set: ###
core.datasets.split_dataset(train_dataset, test_dataset, IO_dict)
if IO_dict.flag_use_train_dataset_split:
    train_dataset, test_dataset, train_dataloader, test_dataloader = TrainingCore.datasets.split_dataset(train_dataset, test_dataset, IO_dict)
######################################################################################################################################


#####################################################################################################################################################################
### Model Itself: ###

model_config = 'models/mmediting/configs/restore/basicvsr/basicvsr_reds4.py'

model = init_model(os.path.join(project_path,model_config),None,device='cuda')


model = init_model(
            './configs/restorers/basicvsr/basicvsr_reds4.py',
            None,
            device='cuda')
from RDND_proper.models.mmediting.mmedit.models.backbones.sr_backbones.basicvsr_pp import BasicVSRPlusPlus
model = BasicVSRPlusPlus(
    mid_channels=64,
    num_blocks=7,
    is_low_res_input=True,
    spynet_pretrained=None,
    cpu_cache_length=100).cuda()
input_tensor = torch.rand(1, 5, 3, 64, 64).cuda()
output = model(input_tensor)

args_from_checkpoint = torch.load(Train_dict.load_Network_filename)
actual_state_dict = args_from_checkpoint['state_dict']
new_state_dict = EasyDict()
for key,value in actual_state_dict.items():
    if 'generator' in key:
        new_key = str.replace(key, 'generator.','')
    new_state_dict[new_key] = value
args_from_checkpoint = new_state_dict

if args_from_checkpoint is not None:
    pretrained_dict = args_from_checkpoint
    current_network_weights = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in current_network_weights}
    current_network_weights.update(pretrained_dict)
    model.load_state_dict(current_network_weights)  # pretrained_dict or model_dict?


if Train_dict.flag_use_new_or_checkpoint_network == 'checkpoint':
    model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=model)
else:
    Train_dict.Network_checkpoint_step = 0

#####################################################################################################################################################################
# gpu


#####################################################################################################################################################################
### Get Network Preprocessing/Postprocessing: ###
preprocessing_object = PreProcessing_EGVSR_Recursive()
preprocessing_object.flag_BW2RGB = False
postprocessing_object = PostProcessing_EGVSR_Recursive()
postprocessing_object.number_of_model_input_frames = 1
postprocessing_object.flag_RGB2BW = False
#####################################################################################################################################################################


#####################################################################################################################################################################
### Optimizer: ###
Network_Optimizer_object = Optimizer_Base(model,
                                          intial_lr=1e-5,
                                          hypergrad_lr=0,
                                          optimizer_type='adam')
if Train_dict.flag_use_new_or_checkpoint_optimizer == 'checkpoint':
    Network_Optimizer_object.Network_Optimizer = TrainingCore.training_utils.load_optimizer_from_checkpoint(
        Network_Optimizer_object.Network_Optimizer, Train_dict)
optimization_dict = Network_Optimizer_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### LR Scheduler: ###
optimization_dict.minimum_learning_rate = 1e-6
optimization_dict.learning_rate_lowering_factor = 0.8
optimization_dict.learning_rate_lowering_patience_before_lowering = 200000
optimization_dict.flag_momentum_cycling = False
optimization_dict.momentum_bounds = [0.85, 0.95]
optimization_dict.flag_uptick_learning_rate_after_falling_below_minimum = False
LR_scheduler_object = LR_Scheduler_ReduceLROnPlateauCustom(Network_Optimizer_object.Network_Optimizer,
                                                           input_dict=optimization_dict)
optimization_dict = LR_scheduler_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### Gradient Clipping: ###
optimization_dict.flag_use_gradient_clipping = True
optimization_dict.flag_clip_grad_value_or_norm = 'norm'
optimization_dict.anomaly_factor_threshold = 3
optimization_dict.gradient_clip_norm = 20
optimization_dict.gradient_clip_value_factor = 6.6 / 34  # initial_grad_norm/initial_grad_max_value
Clip_gradient_object = Clip_Gradient_Base(input_dict=optimization_dict)
optimization_dict = Clip_gradient_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### Get Loss Object: ###
Loss_object = TrainingCore.losses.Loss_Simple(inputs_dict=None) #L1 only
# Loss_object = TrainingCore.losses.Loss_Simple_RLSP(inputs_dict=None)
# Loss_object = TrainingCore.losses.Loss_Recursive_EGVSR(inputs_dict=None)
losses_dict = Loss_object.final_dict
#####################################################################################################################################################################


#####################################################################################################################################################################
### TensorBoard: ###
if Train_dict.flag_write_TB:
    TB_writer_train = SummaryWriter(os.path.join(TensorBoard_path, Train_dict.Network_checkpoint_prefix + '_Train'))
    TB_writer_test = SummaryWriter(os.path.join(TensorBoard_path, Train_dict.Network_checkpoint_prefix + '_Validation'))
else:
    TB_writer_train = None
    TB_writer_test = None
TB_object_train = TB_Denoising_Recursive(TB_writer_train)
TB_object_test = TB_Denoising_Recursive(TB_writer_test)
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
    dataset_name = str(test_dataset.dataset.__class__).replace('.', '__').replace('<class \'', '').replace('\'>',
                                                                                                           '').replace(
        'core__datasets__', '')
else:
    dataset_name = str(test_dataset.__class__).replace('.', '__').replace('<class \'', '').replace('\'>', '').replace(
        'core__datasets__', '')
inference_path = os.path.join(inference_path_master, 'Inference',
                              Train_dict.Network_checkpoint_prefix + '/' + dataset_name)
if Train_dict.is_training == False:  # If i'm doing validation (and not just outputing stuff while training) - create a special folder to save results
    inference_path = os.path.join(inference_path,
                                  os.path.split(Test_Images_Folder)[-1] + '_' + Inference_folder_name_addon)

### Get Debug Callback: ###
debug_callback = InferenceCallback_SuperResolution_Recursive(inference_path + '_debug',
                                                             Train_dict.is_training)  # TODO: isn't the Train_dict.is_training useless now?!!
debug_callback.frame_index_to_predict = 0
### Get Validation Callback: ###
validation_callback = InferenceCallback_SuperResolution_Recursive(inference_path + '_validation',
                                                                  Train_dict.is_training)  # TODO: pass in Train_dict instead of this only
validation_callback.frame_index_to_predict = 0
### Get Inference Callback: ###
inference_callback = validation_callback
#####################################################################################################################################################################


#####################################################################################################################################################################
### Define General Trainer: ###
Trainer = GeneralTrainer_Recursive(model=model,
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


with torch.no_grad():
    import torch.fft
    input_tensor = torch.randn((1,1,64,64))
    tic()
    for i in np.arange(100):
        output_tensor = torch_fft2(input_tensor)
    output_tensor[0,0,0,0] = 1
    toc()


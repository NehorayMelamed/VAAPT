from RapidBase.import_all import *

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import Dataset_MultipleImagesFromSingleImage
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils



### Initialize Network State Dict: ###
Train_dict = EasyDict()


### Paths: ###
##############
inference_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints/Inference/model_combined/test_for_bla_example_folder")
original_images_path = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_Videos_In_Folders_AddNoise/Div2k_validation_set_BW_video_TESTING')
noisy_images_path = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_Videos_In_Folders_AddNoise/Div2k_validation_set_BW_video_TESTING')
clean_images_path = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_Videos_In_Folders_AddNoise/Div2k_validation_set_BW_video_TESTING')

### VBM4D: ###
clean_images_path = path_fix_path_for_linux('/home/mafat/Downloads/VBM4D-20201202T100900Z-001/VBM4D/PSNR5')
original_images_path = path_fix_path_for_linux('/home/mafat/Downloads/VBM4D-20201202T100900Z-001/VBM4D/PSNR5')
noisy_images_path = path_fix_path_for_linux('/home/mafat/Downloads/VBM4D-20201202T100900Z-001/VBM4D/PSNR5')
###

### Train/Inference/Debug Parameters: ###
Train_dict.debug_step = 10
Train_dict.is_training = False
Train_dict.flag_do_validation = True
Train_dict.flag_movie_inference = False

### Frequencies/TensorBoard: ###
Train_dict.save_models_frequency = 10
Train_dict.validation_set_frequency = 500
Train_dict.flag_plot_grad_flow = False
Train_dict.tensorboard_train_frequency = 10
###


## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=12344)

#### GPU Allocation Program: ####
Network_device = torch.device('cuda')
other_device = torch.device('cuda')
###################################


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
### Crop Parameters: ###
IO_dict.crop_X = 340
IO_dict.crop_Y = 270
### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_use_train_dataset_split = False
IO_dict.train_dataset_split_to_test_factor = 0.1
### Noise Parameters: ###
IO_dict.sigma_to_dataset = 1/sqrt(np.inf)*255
IO_dict.SNR_to_model = np.inf
IO_dict.sigma_to_model = 1/sqrt(IO_dict.SNR_to_model)*255
# IO_dict.sigma_to_model = 80
### Blur Parameters: ###
IO_dict.blur_size = 20
### Number of frames to load: ###
IO_dict.NUM_IN_FRAMES = 5 # temporal size of patch
### Training Flags: ###
IO_dict.non_valid_border_size = 30
### Universal Training Parameters: ###
IO_dict.batch_size = 8
IO_dict.number_of_epochs = 60000
#### Assign Device (for instance, we might want to do things on the gpu in the dataloader): ###
IO_dict.device = Network_device
#######################################################################################################################################




#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
original_images, original_image_filenames = read_images_and_filenames_from_folder(original_images_path,
                                      flag_recursive=True,
                                      crop_size=np.inf,
                                      max_number_of_images=np.inf,
                                      allowed_extentions=IMG_EXTENSIONS,
                                      flag_return_numpy_or_list='list',
                                      flag_how_to_concat='C',
                                      crop_style='random',
                                      flag_return_torch=False,
                                      transform=None,
                                      flag_to_BW=False,
                                      flag_random_first_frame=False)
noisy_images, noisy_image_filenames = read_images_and_filenames_from_folder(noisy_images_path,
                                      flag_recursive=True,
                                      crop_size=np.inf,
                                      max_number_of_images=np.inf,
                                      allowed_extentions=IMG_EXTENSIONS,
                                      flag_return_numpy_or_list='list',
                                      flag_how_to_concat='C',
                                      crop_style='random',
                                      flag_return_torch=False,
                                      transform=None,
                                      flag_to_BW=False,
                                      flag_random_first_frame=False)
clean_images, clean_image_filenames = read_images_and_filenames_from_folder(original_images_path,
                                      flag_recursive=True,
                                      crop_size=np.inf,
                                      max_number_of_images=np.inf,
                                      allowed_extentions=IMG_EXTENSIONS,
                                      flag_return_numpy_or_list='list',
                                      flag_how_to_concat='C',
                                      crop_style='random',
                                      flag_return_torch=False,
                                      transform=None,
                                      flag_to_BW=False,
                                      flag_random_first_frame=False)
center_frame_original_image_filenames_and_indices = [(index,x) for index,x in enumerate(original_image_filenames) if 'center_frame_original' in x]
center_frame_noisy_image_filenames_and_indices = [(index,x) for index,x in enumerate(noisy_image_filenames) if 'center_frame_noisy.png' in x]
center_frame_clean_image_filenames_and_indices = [(index,x) for index,x in enumerate(clean_image_filenames) if 'clean_frame_estimate' in x]
center_frame_original_images = []
center_frame_noisy_images = []
center_frame_clean_images = []
center_frame_running_mean_images = []
for i in np.arange(len(center_frame_clean_image_filenames_and_indices)):
    center_frame_original_index, center_frame_original_filename = center_frame_original_image_filenames_and_indices[i]
    center_frame_noisy_index, center_frame_noisy_filename = center_frame_noisy_image_filenames_and_indices[i]
    center_frame_clean_index, center_frame_clean_filename = center_frame_clean_image_filenames_and_indices[i]
    center_frame_original_images.append(original_images[center_frame_original_index])
    center_frame_noisy_images.append(noisy_images[center_frame_noisy_index])
    center_frame_clean_images.append(clean_images[center_frame_clean_index])
######################################################################################################################################



#####################################################################################################################################################################
### Combine dictionaries to Train_dict: ###
Train_dict.IO_dict = IO_dict
Train_dict.update(IO_dict)
#####################################################################################################################################################################



#####################################################################################################################################################################
### Get Debug Callback: ###
# debug_callback = TrainingCore.callbacks.InferenceCallback(inference_path, Train_dict.is_training)
# debug_callback = TrainingCore.callbacks.InferenceCallback_ValidationDataSet(inference_path, Train_dict.is_training)
debug_callback = InferenceCallback_ExampleFolder(inference_path, Train_dict.is_training)
# debug_callback = TrainingCore.callbacks.InferenceCallback_Movie(inference_path, Train_dict.is_training)
# debug_callback = None
#####################################################################################################################################################################


Train_dict.test_dataset_length = len(center_frame_noisy_images)
for index in np.arange(len(center_frame_noisy_images)):
    ### Get Data From DataLoader: ###
    center_frame_original_torch = torch.Tensor(center_frame_original_images[index]).permute([2,0,1]).unsqueeze(0)
    center_frame_noisy_torch = torch.Tensor(center_frame_noisy_images[index]).permute([2,0,1]).unsqueeze(0)
    center_frame_clean_torch = torch.Tensor(center_frame_clean_images[index]).permute([2,0,1]).unsqueeze(0)

    inputs_dict = EasyDict()
    inputs_dict['center_frame_original'] = center_frame_original_torch/255
    inputs_dict['center_frame_noisy'] = center_frame_noisy_torch/255

    outputs_dict = EasyDict()
    outputs_dict['clean_frame_estimate'] = center_frame_clean_torch/255

    inputs_dict = data_from_dataloader_to_GPU(inputs_dict, other_device)
    outputs_dict = data_from_dataloader_to_GPU(outputs_dict, other_device)


    debug_callback.run(index, inputs_dict, outputs_dict, Train_dict)














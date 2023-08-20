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
inference_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints/Inference/model_combined/VBM4D_Beirut_RGBNoise_PSNR5")
original_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/original_frames')
noisy_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/noisy_frames')
clean_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/clean_frame_estimate')
###

### VBM4D: ###
original_frames_folder = path_fix_path_for_linux('/home/mafat/Downloads/Original-20201207T111847Z-001/Original')
noisy_frames_folder = path_fix_path_for_linux('/home/mafat/Downloads/Noisy-20201206T150253Z-001/Noisy')
clean_frames_folder = path_fix_path_for_linux('/home/mafat/Downloads/Clean-20201206T140036Z-001/Clean')


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
IO_dict.crop_X = 1000
IO_dict.crop_Y = 1000
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
original_frames_dataset = TrainingCore.datasets.DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=original_frames_folder,
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=100,  #TODO: add possibility to randomize start index from a possibly long video
                                                         number_of_images_per_video_to_load=1,
                                                         max_number_of_videos=np.inf,
                                                         crop_size=np.inf,
                                                         flag_to_RAM=True,
                                                         flag_explicitely_make_tensor=True,
                                                         flag_normalize_by_255=True,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict, allowed_extentions='.mat')
noisy_frames_dataset = TrainingCore.datasets.DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=noisy_frames_folder,
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=100,  #TODO: add possibility to randomize start index from a possibly long video
                                                         number_of_images_per_video_to_load=1,
                                                         max_number_of_videos=np.inf,
                                                         crop_size=np.inf,
                                                         flag_to_RAM=True,
                                                         flag_explicitely_make_tensor=True,
                                                         flag_normalize_by_255=True,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict,
                                                        allowed_extentions='.mat')
clean_frames_dataset = TrainingCore.datasets.DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=clean_frames_folder,
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=100,  #TODO: add possibility to randomize start index from a possibly long video
                                                         number_of_images_per_video_to_load=1,
                                                         max_number_of_videos=np.inf,
                                                         crop_size=np.inf,
                                                         flag_to_RAM=True,
                                                         flag_explicitely_make_tensor=True,
                                                         flag_normalize_by_255=True,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict, allowed_extentions='.mat')
### DataLoaders: ###
clean_frames_dataloader = DataLoader(dataset=clean_frames_dataset, batch_size=1, shuffle=False)
noisy_frames_dataloader = DataLoader(dataset=noisy_frames_dataset, batch_size=1, shuffle=False)
original_frames_dataloader = DataLoader(dataset=original_frames_dataset, batch_size=1, shuffle=False)
### Number Of Elements: ###
if Train_dict.is_training == False:
    IO_dict.batch_size = 1
IO_dict.num_mini_batches_trn = len(clean_frames_dataloader) // IO_dict.batch_size
IO_dict.num_mini_batches_val = len(clean_frames_dataloader) // IO_dict.batch_size
IO_dict.train_dataset_length = len(clean_frames_dataloader)
IO_dict.test_dataset_length = len(clean_frames_dataloader)

### Get Automatic Train/Test Split From Train Set: ###
if IO_dict.flag_use_train_dataset_split:
    test_dataset_length_new = int(np.round(len(clean_frames_dataset)*IO_dict.train_dataset_split_to_test_factor))
    train_dataset_length_new = len(clean_frames_dataset) - test_dataset_length_new
    train_dataset, test_dataset = torch.utils.data.random_split(clean_frames_dataset, [train_dataset_length_new, test_dataset_length_new])
    train_dataset.indices = np.arange(0, train_dataset_length_new)
    test_dataset.indices = np.arange(train_dataset_length_new, train_dataset_length_new+test_dataset_length_new)
    IO_dict.num_mini_batches_trn = train_dataset_length_new // IO_dict.batch_size
    IO_dict.num_mini_batches_val = test_dataset_length_new // IO_dict.batch_size
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
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
# debug_callback = TrainingCore.callbacks.InferenceCallback_ExampleFolder(inference_path, Train_dict.is_training)
debug_callback = TrainingCore.callbacks.InferenceCallback_Movie(inference_path, Train_dict.is_training)
# debug_callback = None
#####################################################################################################################################################################


original_frames_iterator = original_frames_dataloader.__iter__()
noisy_frames_iterator = noisy_frames_dataloader.__iter__()
clean_frames_iterator = clean_frames_dataloader.__iter__()
for batch_index_validation in np.arange(IO_dict.test_dataset_length):
    ### Get Data From DataLoader: ###
    data_from_dataloader_original_frames = original_frames_iterator.next()
    data_from_dataloader_noisy_frames = noisy_frames_iterator.next()
    data_from_dataloader_clean_estimate = clean_frames_iterator.next()

    data_from_dataloader_original_frames = data_from_dataloader_to_GPU(data_from_dataloader_original_frames, other_device)
    data_from_dataloader_noisy_frames = data_from_dataloader_to_GPU(data_from_dataloader_noisy_frames, other_device)
    data_from_dataloader_clean_estimate = data_from_dataloader_to_GPU(data_from_dataloader_clean_estimate, other_device)

    data_from_dataloader_original_frames = EasyDict(data_from_dataloader_original_frames)
    data_from_dataloader_noisy_frames = EasyDict(data_from_dataloader_noisy_frames)
    data_from_dataloader_clean_estimate = EasyDict(data_from_dataloader_clean_estimate)

    inputs_dict = data_from_dataloader_original_frames
    inputs_dict['output_frames_noisy'] = data_from_dataloader_noisy_frames['original_frames']
    inputs_dict['center_frame_noisy'] = data_from_dataloader_noisy_frames['center_frame_original']

    outputs_dict = EasyDict(data_from_dataloader_clean_estimate)
    outputs_dict['clean_frame_estimate'] = outputs_dict['center_frame_original']

    ### Upper-Left Crop if wanted: ###
    inputs_dict['center_frame_noisy'] = inputs_dict['center_frame_noisy'][:,:,0:1000,0:1000].clamp(0,1)
    inputs_dict['center_frame_original'] = inputs_dict['center_frame_original'][:,:,0:1000,0:1000].clamp(0,1)
    outputs_dict['clean_frame_estimate'] = outputs_dict['clean_frame_estimate'][:,:,0:1000,0:1000].clamp(0,1)


    # bla = inputs_dict['center_frame_original']
    # bla = torch.cat([bla,inputs_dict['center_frame_noisy']],-1)
    # bla = torch.cat([bla,outputs_dict['clean_frame_estimate']],-1)
    # imshow_torch(bla)
    # imshow_torch(inputs_dict['center_frame_original'])
    # imshow_torch(inputs_dict['center_frame_noisy'])
    # imshow_torch(outputs_dict['clean_frame_estimate'])

    debug_callback.run(batch_index_validation, inputs_dict, outputs_dict, Train_dict)











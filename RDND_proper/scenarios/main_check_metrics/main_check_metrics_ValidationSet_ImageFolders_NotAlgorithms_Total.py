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
inference_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints/Inference/DIV2K_ValidationSet/Blurred_Shift20_PSNR5")
original_images_path = os.path.join(datasets_main_folder, '/DataSets/Div2K/Shifted_Images_Blur/Div2k_validation_set_BW_video/Shift_0/PSNR_inf')
noisy_images_path = os.path.join(datasets_main_folder, '/DataSets/Div2K/Shifted_Images/Div2k_validation_set_BW_video')
clean_images_path = noisy_images_path
all_filenames_list = path_get_files_recursively(noisy_images_path)

all_folders_set = list()
for filename in all_filenames_list:
    current_folder = os.path.split(os.path.split(filename)[0])[0]
    PSNR = os.path.split(current_folder)[-1].split('_')[-1]
    Shift = os.path.split(os.path.split(current_folder)[0])[-1].split('_')[-1]
    all_folders_set.append(current_folder)
all_folders_set = unique(all_folders_set)

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


metrics_dict = EasyDict()
metrics_dict2 = EasyDict()
for folder_name in all_folders_set:
    #######################################################################################################################################
    ## DataSet & DataLoader Objects: ###
    ####################################
    PSNR = os.path.split(folder_name)[-1].split('_')[-1]
    Shift = os.path.split(os.path.split(folder_name)[0])[-1].split('_')[-1]
    dict_key = 'PSNR' + PSNR + '_' + 'Blur' + Shift


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
    noisy_images, noisy_image_filenames = read_images_and_filenames_from_folder(str(folder_name),
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
    clean_images, clean_image_filenames = read_images_and_filenames_from_folder(str(folder_name),
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

    counter = 0
    noisy_image_filenames_final = []
    while counter<len(noisy_image_filenames)-1:
        current_folder_name = os.path.split(os.path.split(noisy_image_filenames[counter])[0])[-1]
        noisy_image_filenames_final.append(noisy_image_filenames[counter])
        flag_still_same_folder = True
        while flag_still_same_folder and counter<=len(noisy_image_filenames)-1:
              current_folder_name2 = os.path.split(os.path.split(noisy_image_filenames[counter])[0])[-1]
              if not current_folder_name == current_folder_name2:
                flag_still_same_folder = False
              counter += 1
              print(counter)
        counter -= 1


    counter = 0
    original_image_filenames_final = []
    while counter<len(original_image_filenames)-1:
        current_folder_name = os.path.split(os.path.split(original_image_filenames[counter])[0])[-1]
        original_image_filenames_final.append(original_image_filenames[counter])
        flag_still_same_folder = True
        while flag_still_same_folder and counter<=len(original_image_filenames)-1:
              current_folder_name2 = os.path.split(os.path.split(original_image_filenames[counter])[0])[-1]
              if not current_folder_name == current_folder_name2:
                flag_still_same_folder = False
              counter += 1
              print(counter)
        counter -= 1


    counter = 0
    clean_image_filenames_final = []
    while counter<len(clean_image_filenames)-1:
        current_folder_name = os.path.split(os.path.split(clean_image_filenames[counter])[0])[-1]
        clean_image_filenames_final.append(clean_image_filenames[counter])
        flag_still_same_folder = True
        while flag_still_same_folder and counter<=len(clean_image_filenames)-1:
              current_folder_name2 = os.path.split(os.path.split(clean_image_filenames[counter])[0])[-1]
              if not current_folder_name == current_folder_name2:
                flag_still_same_folder = False
              counter += 1
              print(counter)
        counter -= 1

    center_frame_original_images = []
    center_frame_noisy_images = []
    center_frame_clean_images = []
    for i in np.arange(len(original_image_filenames_final)):
        center_frame_original_images.append(read_image_cv2(original_image_filenames_final[i]))
        center_frame_noisy_images.append(read_image_cv2(noisy_image_filenames_final[i]))
        center_frame_clean_images.append(read_image_cv2(clean_image_filenames_final[i]))
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
    debug_callback = InferenceCallback_ExampleFolder_OnlyStats(inference_path, Train_dict.is_training)
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


    ### Update dict with metrics: ###
    metrics_dict[dict_key] = debug_callback.original_noisy_average_metrics_dict
    metrics_dict2[dict_key] = debug_callback.original_noisy_history_metrics_dict
    1
1

arrays_folder = '/home/mafat/Pytorch_Checkpoints/Inference/DIV2K_ValidationSet/total_comparison'
numpy.save(os.path.join(arrays_folder, 'average_dict.npy'), metrics_dict, allow_pickle=True, fix_imports=True)
numpy.save(os.path.join(arrays_folder, 'history_dict.npy'), metrics_dict2, allow_pickle=True, fix_imports=True)

from RapidBase.import_all import *
arrays_folder = '/home/mafat/Pytorch_Checkpoints/Inference/DIV2K_ValidationSet/total_comparison'
metrics_dict = numpy.load(os.path.join(arrays_folder, 'average_dict.npy'),allow_pickle=True)
metrics_dict2 = numpy.load(os.path.join(arrays_folder, 'history_dict.npy'),allow_pickle=True)
metrics_dict = metrics_dict.tolist()
metrics_dict2 = metrics_dict2.tolist()

### Plot Images Showing Results: ###
#Blurs = 20,10,5,2,1,0
#PSNRs = inf,100,10,5,4,3,2,1
SSIM_vec = np.zeros((7,6))  #8 PSNRs, 6 Blurs
NMSE_vec = np.zeros((7,6))
MSE_vec = np.zeros((7,6))
SNR_linear_vec = np.zeros((7,6))
SNR_dB_vec = np.zeros((7,6))
PSNR_vec = np.zeros((7,6))
contrast_measure_delta_vec = np.zeros((7,6))
blur_measurement_vec = np.zeros((7,6))
eigen_focus_vec = np.zeros((7,6))
laplacian_variance_vec = np.zeros((7,6))
gradient_MSE_vec = np.zeros((7,6))
gradient_NMSE_vec = np.zeros((7,6))
gradient_SNR_linear_vec = np.zeros((7,6))
gradient_SNR_dB_vec = np.zeros((7,6))
gradient_PSNR_vec = np.zeros((7,6))
PSNR_dict = {'inf':0,'10':1,'5':2,'4':3,'3':4,'2':5,'1':6}
Blur_dict = {'0':0,'1':1,'2':2,'5':3,'10':4,'20':5}
for current_key in metrics_dict.keys():
    ### Get current PSNR and Blur: ###
    current_PSNR = current_key.split('_')[0].split('PSNR')[-1]
    current_Blur = current_key.split('_')[1].split('Blur')[-1]

    if not current_PSNR == '100':
        ### Get arrays (x,y) indices: ###
        current_x_index = PSNR_dict[current_PSNR]
        current_y_index = Blur_dict[current_Blur]

        ### Get current keys dict: ###
        current_key_dict = metrics_dict[current_key]

        ### Assign arrays: ###
        SSIM_vec[current_x_index, current_y_index] = current_key_dict['SSIM']
        NMSE_vec[current_x_index, current_y_index] = current_key_dict['NMSE']
        MSE_vec[current_x_index, current_y_index] = current_key_dict['MSE']
        SNR_linear_vec[current_x_index, current_y_index] = current_key_dict['SNR_linear']
        SNR_dB_vec[current_x_index, current_y_index] = current_key_dict['SNR_dB']
        PSNR_vec[current_x_index, current_y_index] = current_key_dict['PSNR']
        contrast_measure_delta_vec[current_x_index, current_y_index] = current_key_dict['contrast_measure_delta']
        blur_measurement_vec[current_x_index, current_y_index] = current_key_dict['blur_measurement']
        eigen_focus_vec[current_x_index, current_y_index] = current_key_dict['eigen_focus']
        laplacian_variance_vec[current_x_index, current_y_index] = current_key_dict['laplacian_variance']
        gradient_MSE_vec[current_x_index, current_y_index] = current_key_dict['gradient_MSE']
        gradient_NMSE_vec[current_x_index, current_y_index] = current_key_dict['gradient_NMSE']
        gradient_SNR_linear_vec[current_x_index, current_y_index] = current_key_dict['gradient_SNR_linear']
        gradient_SNR_dB_vec[current_x_index, current_y_index] = current_key_dict['gradient_SNR_dB']
        gradient_PSNR_vec[current_x_index, current_y_index] = current_key_dict['gradient_PSNR']

    ### SSIM: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(SSIM_vec.transpose([1,0]))
    x_label_list = ['inf','10','5','4','3','2','1']
    y_label_list = ['0','1','2','5','10','20']
    # y_label_list = np.flip(y_label_list)
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('SSIM')
    savefig(os.path.join(arrays_folder,'SSIM'))
    close()

    ### SNR_linear: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(SNR_linear_vec.transpose([1,0]).clip(0,100))
    # y_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # x_label_list = ['0', '1', '2', '5', '10', '20']
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('SNR linear')
    savefig(os.path.join(arrays_folder, 'SNR linear'))
    close()

    ### PSNR: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(PSNR_vec.transpose([1,0]))
    # y_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # x_label_list = ['0', '1', '2', '5', '10', '20']
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('PSNR')
    savefig(os.path.join(arrays_folder, 'PSNR'))
    close()

    ### contrast_measure_delta: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(contrast_measure_delta_vec.transpose([1,0]))
    # y_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # x_label_list = ['0', '1', '2', '5', '10', '20']
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('contrast measure delta')
    savefig(os.path.join(arrays_folder, 'contras measure delta'))
    close()

    ### blur_measurement: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(blur_measurement_vec.transpose([1,0]))
    # y_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # x_label_list = ['0', '1', '2', '5', '10', '20']
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('blur measurement')
    savefig(os.path.join(arrays_folder, 'blur measurement'))
    close()

    ### eigen_focus: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(eigen_focus_vec.transpose([1,0]))
    # y_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # x_label_list = ['0', '1', '2', '5', '10', '20']
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('eigen focus')
    savefig(os.path.join(arrays_folder, 'eigen focus'))
    close()

    ### laplacian_variance: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(laplacian_variance_vec.transpose([1,0]))
    # y_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # x_label_list = ['0', '1', '2', '5', '10', '20']
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('laplacian variance')
    savefig(os.path.join(arrays_folder, 'laplacian variance'))
    close()

    ### gradient_SNR_linear: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(gradient_SNR_linear_vec.clip(0,5).transpose([1,0]))
    # y_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # x_label_list = ['0', '1', '2', '5', '10', '20']
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Blur')
    fig.colorbar(img)
    title('gradient SNR linear')
    savefig(os.path.join(arrays_folder, 'gradient SNR linear'))
    close()

    # ### SSIM: ###
    # fig, ax = plt.subplots(1, 1)
    # img = ax.imshow(SSIM_vec)
    # x_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # y_label_list = ['0', '1', '2', '5', '10', '20']
    # ax.set_xticks(np.arange(8))
    # ax.set_yticks(np.arange(6))
    # ax.set_xticklabels(x_label_list)
    # ax.set_yticklabels(y_label_list)
    # fig.colorbar(img)
    #
    # ### SSIM: ###
    # fig, ax = plt.subplots(1, 1)
    # img = ax.imshow(SSIM_vec)
    # x_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # y_label_list = ['0', '1', '2', '5', '10', '20']
    # ax.set_xticks(np.arange(8))
    # ax.set_yticks(np.arange(6))
    # ax.set_xticklabels(x_label_list)
    # ax.set_yticklabels(y_label_list)
    # fig.colorbar(img)
    #
    # ### SSIM: ###
    # fig, ax = plt.subplots(1, 1)
    # img = ax.imshow(SSIM_vec)
    # x_label_list = ['inf', '100', '10', '5', '4', '3', '2', '1']
    # y_label_list = ['0', '1', '2', '5', '10', '20']
    # ax.set_xticks(np.arange(8))
    # ax.set_yticks(np.arange(6))
    # ax.set_xticklabels(x_label_list)
    # ax.set_yticklabels(y_label_list)
    # fig.colorbar(img)
    #
    #

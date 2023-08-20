
from RapidBase.Utils.IO.Path_and_Reading_utils import save_image_numpy, gray2color_torch, to_range

from RapidBase.import_all import *

from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Noise_Estimate_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration_misc import *


### Paths: ###
##############
### Movies Paths: ###
# original_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/original_frames')
# original_frames_folder = os.path.join(datasets_main_folder, '/Example Videos/Beirut_inf')
# original_frames_folder = os.path.join(datasets_main_folder, '/KAYA_EXPERIMENT/23')
# original_frames_folder = os.path.join(datasets_main_folder, '/DRONE_EXPERIMENTS/3ms_constant_height')
original_frames_folder = os.path.join(datasets_main_folder, '/Example Videos/LWIR_video_inf')
noisy_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/noisy_frames')
clean_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/clean_frame_estimate')
### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies'
# results_path_name_addition = 'KAYA_Drone'
results_path_name_addition = 'LWIRNEW'
# results_path_name_addition = 'KAYA23NEW'
path_make_path_if_none_exists(results_path)
###


### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Parameters: ###
flag_save_uint16 = False
flag_normalized_dataloader_by_255 = False
# max_value_possible = 2**16-1
max_value_possible = 256
image_index_to_start_from = 10
image_index_to_stop_at = 140
number_of_images = 100
registration_algorithm = 'DeepFlow'  #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR, NormalizedCrossCorrelation
noise_estimation_algorithm = 'IterativeResidual'  #IterativeResidual, FullFrameHistogram
downsample_kernel_size_FrameCombine = 8
downsample_kernel_size_NoiseEstimation = 8
downsample_factor_registration = 2
max_shift_in_pixels = 0
max_search_area = 11 #for algorithms like NCC
SNR = 5
crop_size_x = 1024
crop_size_y = 1024
inner_crop_size = 1000
noise_sigma = 1/sqrt(SNR)

#######################################################################################################################################
### IO: ###
IO_dict = EasyDict()
### Crop Parameters: ###
IO_dict.crop_X = 340
IO_dict.crop_Y = 270
### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_use_train_dataset_split = False
IO_dict.train_dataset_split_to_test_factor = 0.1
### Noise Parameters: ###
IO_dict.sigma_to_dataset = 1/sqrt(SNR)*255
IO_dict.SNR_to_model = SNR
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
IO_dict.device = torch.device('cuda')
#######################################################################################################################################

#######################################################################################################################################
### DataSets: ###
original_frames_dataset = DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=original_frames_folder,
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=100,  #TODO: add possibility to randomize start index from a possibly long video
                                                         number_of_images_per_video_to_load=1,
                                                         max_number_of_videos=np.inf,
                                                         crop_size=np.inf,
                                                         flag_to_RAM=True,
                                                         flag_explicitely_make_tensor=True,
                                                         flag_normalize_by_255=flag_normalized_dataloader_by_255,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=True,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict,
                                                         allowed_extentions=['png', 'tif'])

### Get Global Statistics: ###
images_list = []
quantile_list = []
shifts_x_sum = 0
shifts_y_sum = 0
# for i in np.arange(image_index_to_start_from, len(original_frames_dataset.image_filenames_list_of_lists[0])):
for i in np.arange(image_index_to_start_from, image_index_to_start_from+1):
    current_image = original_frames_dataset[i].center_frame_original[0].data.unsqueeze(0).unsqueeze(0)
    images_list.append(current_image)
    quantile_list.append(current_image.quantile(0.999))
images_list = torch.cat(images_list,0)
max_to_use = np.quantile(quantile_list,0.5)
stretch_factor = 255/max_to_use/1.0  # Stretch factor to "fill" a uint8 image
stretch_factor_from_scaling_by_255 = max_to_use/255/1
# imshow_torch(images_list[0].clamp(0,max_to_use))
# imshow_torch((images_list[0]*stretch_factor).clamp(0,255))


### Take care of initializations and such for first frame: ###
original_frames_output_dict = original_frames_dataset[image_index_to_start_from]
original_frame_current = original_frames_output_dict.center_frame_original[0].data
original_frame_current = original_frame_current.unsqueeze(0).unsqueeze(0) / max_value_possible
# noisy_frame_current = original_frame_current + noise_sigma*torch.randn_like(original_frame_current)
noisy_frame_current = original_frames_output_dict.center_frame_noisy[0].data
noisy_frame_current = noisy_frame_current.unsqueeze(0).unsqueeze(0) / max_value_possible
noisy_frame_current_cropped = crop_torch_batch(noisy_frame_current, crop_size_x, crop_style='center')
original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_x, crop_style='center')
original_frame_seed_cropped = torch.zeros_like(original_frame_current_cropped)
original_frame_seed_cropped.data = original_frame_current_cropped.data

### Initialize More Things: ###
shift_layer_torch = Shift_Layer_Torch()
clean_frame_previous = noisy_frame_current_cropped  #initialize internal "cleaned" image to simply be first frame, which is noisy
original_frame_previous = original_frame_current
noisy_frame_previous = noisy_frame_current
clean_frame_perfect_averaging = noisy_frame_current_cropped
clean_frame_simple_averaging = noisy_frame_current_cropped

B,C,H,W = noisy_frame_current.shape
crop_size_x = min(crop_size_x, W)
crop_size_y = min(crop_size_y, H)
shifts_array = torch.zeros((1, 1, crop_size_x, crop_size_x, 2))
noise_map_last = torch.zeros((1, 1, crop_size_x, crop_size_x))
pixel_counts_map_previous = torch.ones((1, 1, crop_size_x, crop_size_x))
noise_map_previous = None
clean_frames_list = []
noisy_frames_list = []
original_frames_list = []
perfect_average_frames_list = []
simple_average_frames_list = []
seed_frames_list = []

real_shifts_list = []
inferred_shifts_list = []

### Get Video Recorder: ###
experiment_name = results_path_name_addition + '_SNR' + str(SNR) + '_' + registration_algorithm + '_' + noise_estimation_algorithm + \
             '_DSCombine' + str(downsample_kernel_size_FrameCombine) + \
             '_DSNoise' + str(downsample_kernel_size_NoiseEstimation) + \
              '_DSReg' + str(downsample_factor_registration) + \
             '_NoiseSigma' + str(int(float(decimal_notation(noise_sigma*255,0)))) + \
             '_MaxShift' + str(max_shift_in_pixels)
save_path = os.path.join(results_path, experiment_name)
save_path1 = os.path.join(save_path, 'original_frames')
save_path2 = os.path.join(save_path, 'clean_frame_estimate')
save_path3 = os.path.join(save_path, 'noisy_frames')
save_path4 = os.path.join(save_path, 'original_frames_uint16')
save_path5 = os.path.join(save_path, 'clean_frame_estimate_uint16')
save_path6 = os.path.join(save_path, 'noisy_frames_uint16')
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(save_path3)
path_make_path_if_none_exists(save_path4)
path_make_path_if_none_exists(save_path5)
path_make_path_if_none_exists(save_path6)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name = os.path.join(save_path, 'Results', 'Results.avi')
video_name_NoiseEstimate = os.path.join(save_path, 'Results', 'NoiseEstimate.avi')
video_name_CleanVSAverage = os.path.join(save_path, 'Results', 'Clean_VS_SimpleAverage.avi')
video_name_OriginalVSClean = os.path.join(save_path, 'Results', 'Original_VS_Clean.avi')
video_name_ResetGate = os.path.join(save_path, 'Results', 'Reset_Gate.avi')
video_name_Residual = os.path.join(save_path, 'Results', 'Residual.avi')
video_name_Noise = os.path.join(save_path, 'Results', 'Noise.avi')
video_name_StabilizedVideo = os.path.join(save_path, 'Results', 'StabilizedVideo.avi')
video_width = crop_size_x * 2
video_height = crop_size_x * 1
video_object = cv2.VideoWriter(video_name, 0, 10, (crop_size_x * 2, crop_size_x))
video_object_NoiseEstimate = cv2.VideoWriter(video_name_NoiseEstimate, 0, 10, (crop_size_x, crop_size_x))
video_object_CleanEstimateVSAverage = cv2.VideoWriter(video_name_CleanVSAverage, 0, 10, (crop_size_x * 2, crop_size_x))
video_object_OriginalVSCleanEstimate = cv2.VideoWriter(video_name_OriginalVSClean, 0, 10, (crop_size_x * 2, crop_size_x))
video_object_ResetGate = cv2.VideoWriter(video_name_ResetGate, 0, 10, (crop_size_x, crop_size_x))
video_object_Residual = cv2.VideoWriter(video_name_Residual, 0, 10, (crop_size_x, crop_size_x))
video_object_Noise = cv2.VideoWriter(video_name_Noise, 0, 10, (crop_size_x, crop_size_x))
video_object_StabilizedVideo = cv2.VideoWriter(video_name_StabilizedVideo, 0, 10, (crop_size_x, crop_size_x))


#############################################################################################################################################
### Get Noise Profile Using Histogram Width Algorithm: ###
histogram_spatial_binning_factor = 3
# max_value_possible = 2**16-1
# max_value_possible = 255
crude_histogram_bins_vec = my_linspace(images_list[0].quantile(0.001),
                                       images_list[0].quantile(0.999), 10)
downsample_kernel_size = 1
noisy_frames_current = images_list * 1
# clean_frames_estimate = noisy_frames_current.median(0)[0].unsqueeze(0)
# plot_and_save_all_noise_estimates(save_path,
#                                       noisy_frames_current,
#                                       clean_frames_estimate,
#                                       histogram_spatial_binning_factor,
#                                       crude_histogram_bins_vec,
#                                       downsample_kernel_size, max_value_possible
#                                       )
#
#
# ### Align Entire Movie: ###
# shifts_array, aligned_frames_list = register_images_batch(images_list,
#                                                           'CrossCorrelation',
#                                                           max_search_area,
#                                                           inner_crop_size,
#                                                           downsample_factor_registration,
#                                                           flag_do_initial_SAD=False,
#                                                           initial_SAD_search_area=5,
#                                                           center_frame_index=images_list.shape[0]//2)
# ### Record Stabilized Movie: ###
# video_object_StabilizedVideo = cv2.VideoWriter(video_name_StabilizedVideo, 0, 10, (crop_size_x*2, crop_size_x))
# for i in np.arange(len(aligned_frames_list)):
#     print(i)
#     current_image = aligned_frames_list[i].squeeze()
#     current_image = np.atleast_3d(current_image)
#     current_image = crop_tensor(current_image, crop_size_x)
#     image_to_write = np.concatenate([current_image, crop_tensor(images_list[i].cpu().permute([1,2,0]).numpy(), crop_size_x)], 1)
#     image_to_write = np.concatenate([image_to_write,image_to_write,image_to_write],2)
#     image_to_write = (image_to_write*255).astype(np.uint8)
#     video_object_StabilizedVideo.write(image_to_write)
# video_object_StabilizedVideo.release()
# ### Get Shifts Stats: ###
# inferred_shifts_x = []
# inferred_shifts_y = []
# for inferred_shifts_current in shifts_array:
#     inferred_shifts_x.append(inferred_shifts_current[0])
#     inferred_shifts_y.append(inferred_shifts_current[1])
# inferred_shifts_x[inferred_shifts_x==0] = np.nan
# inferred_shifts_y[inferred_shifts_y==0] = np.nan
# print(np.nanmax(np.abs(inferred_shifts_x)))
# print(np.nanmax(np.abs(inferred_shifts_y)))


### Loop Over Frames: ###
image_index_to_stop_at = min(image_index_to_stop_at, len(original_frames_dataset.image_filenames_list_of_lists[0]))
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_stop_at):
# for frame_index in np.arange(image_index_to_start_from+1, 30):
    print(frame_index)
    ### Get Next Frame: ###
    original_frames_output_dict = original_frames_dataset[frame_index]
    original_frame_current = original_frames_output_dict.center_frame_original[0].data
    original_frame_current = original_frame_current.unsqueeze(0).unsqueeze(0) / max_value_possible
    noisy_frame_current = original_frames_output_dict.center_frame_noisy[0].data
    noisy_frame_current = noisy_frame_current.unsqueeze(0).unsqueeze(0) / max_value_possible

    # ### Add Noise: ###
    # noisy_frame_current = original_frame_current + 1/sqrt(SNR)*torch.randn_like(original_frame_current)

    ### Center Crop: ###
    noisy_frame_current_cropped = crop_torch_batch(noisy_frame_current, crop_size_x, crop_style='center')
    noisy_frame_previous_cropped = crop_torch_batch(noisy_frame_previous.data, crop_size_x, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_x, crop_style='center')
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, crop_size_x, crop_style='center')

    ### Keep Track Of Simple Averaging (No Tracking): ###
    clean_frame_simple_averaging = (1 - 1 / (frame_index+1)) * clean_frame_simple_averaging + \
                                   (1 / (frame_index+1)) * noisy_frame_current_cropped
    ### Keep Track Of Perfect Tracking + Averaging: ###
    clean_frame_perfect_averaging = (1 - 1 / (frame_index + 1)) * clean_frame_perfect_averaging + \
                                    (1 / (frame_index + 1)) * (original_frame_seed_cropped + 1 / sqrt(SNR) * torch.randn_like(noise_map_last))

    ### Register Images (return numpy arrays): ###
    # registration_algorithm = 'CrossCorrelation' #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR
    shifts_array, clean_frame_previous_warped = register_images(noisy_frame_current_cropped.data, clean_frame_previous.data,
                                                                registration_algorithm,
                                                                max_search_area,
                                                                inner_crop_size,
                                                                downsample_factor_registration,
                                                                flag_do_initial_SAD=False,
                                                                initial_SAD_search_area=max_search_area)

    # shifts_array, clean_frame_previous_warped = register_images(original_frame_current_cropped.data, original_frame_previous_cropped.data,
    #                                                             registration_algorithm,
    #                                                             max_search_area,
    #                                                             inner_crop_size,
    #                                                             downsample_factor_registration,
    #                                                             flag_do_initial_SAD=False,
    #                                                             initial_SAD_search_area=max_search_area)
    print('deduced shifts: ' + str(shifts_array))

    ### Keep Track of recorded shifts to compare with real shifts: ###
    # real_shifts_list.append((shiftx,shifty))
    inferred_shifts_list.append(shifts_array)

    ### Get both frames to be in Tensor / Numpy Type: ###
    clean_frame_previous_warped = torch.Tensor(clean_frame_previous_warped).unsqueeze(0).unsqueeze(0)

    ### Estimate noise: ###
    if noise_estimation_algorithm == 'IterativeResidual':
        noise_map_current, noise_pixel_counts_map_current = estimate_noise_IterativeResidual(noisy_frame_current_cropped.data, noise_map_previous,
                                                             pixel_counts_map_previous, clean_frame_previous_warped.data,  #TODO: clean_frame_previous_warped
                                                             downsample_kernel_size=downsample_kernel_size_NoiseEstimation) #original_frame_current_cropped
    elif noise_estimation_algorithm == 'FullFrameHistogram':  #TODO: use histograms
        noise_map_current = estimate_noise_FullFrameHistogram(noisy_frame_current_cropped.data, noise_map_previous,
                                                              pixel_counts_map_previous, clean_frame_previous_warped.data,
                                                              downsample_kernel_size=downsample_kernel_size_NoiseEstimation)
    # noise_map_current = torch.ones_like(noisy_frame_current_cropped)
    # noise_pixel_counts_map_current = 1

    # ### Combine Images: ###
    clean_frame_current, pixel_counts_map_current, reset_gates = combine_frames(noisy_frame_current_cropped.data,
                                         clean_frame_previous_warped.data,  # clean_frame_previous_warped
                                         noise_map_current = noise_map_current,
                                         noise_map_last_warped = None,
                                         pixel_counts_map_previous = pixel_counts_map_previous,
                                         downsample_kernel_size = downsample_kernel_size_FrameCombine)
    # clean_frame_current = noisy_frame_current_cropped
    # pixel_counts_map_current = 1
    # reset_gates = torch.ones_like(clean_frame_current)

    ### Keep variables for next frame: ###
    clean_frame_previous = clean_frame_current  #TODO: right now i simply keep the last noisy image, as i don't clean anything now
    original_frame_previous = original_frame_current
    noisy_frame_previous = noisy_frame_current
    pixel_counts_map_previous = pixel_counts_map_current
    noise_map_previous = noise_map_current
    shifts_x_sum += shifts_array[0]
    shifts_y_sum += shifts_array[1]

    # imshow_torch(noisy_frame_current_cropped.unsqueeze(0))
    # imshow_torch(clean_frame_current.unsqueeze(0))
    # imshow_torch(original_frame_current_cropped.unsqueeze(0))
    
    ### Save Wanted Images: ###
    scale_to_255 = 255
    original_frame_current_cropped_numpy = (original_frame_current_cropped[0] * scale_to_255 * stretch_factor_from_scaling_by_255).cpu().numpy().transpose(1, 2, 0)
    clean_frame_current_cropped_numpy = (clean_frame_current[0] * scale_to_255 * stretch_factor_from_scaling_by_255).cpu().numpy().transpose(1, 2, 0)
    noisy_frame_current_cropped_numpy = (noisy_frame_current_cropped[0] * scale_to_255 * stretch_factor_from_scaling_by_255).cpu().numpy().transpose(1, 2, 0)
    save_image_numpy(save_path1, str(frame_index) + '_center_frame_original.png', (original_frame_current_cropped_numpy).clip(0,255).astype(np.uint8), False, flag_scale=False)
    save_image_numpy(save_path2, str(frame_index) + '_clean_frame_estimate.png', (clean_frame_current_cropped_numpy).clip(0,255).astype(np.uint8) , False, flag_scale=False)
    save_image_numpy(save_path3, str(frame_index) + '_center_frame_noisy.png', (noisy_frame_current_cropped_numpy).clip(0,255).astype(np.uint8), False, flag_scale=False)

    if flag_save_uint16:
        save_image_numpy(save_path4, str(frame_index) + '_center_frame_original.png',
                         (original_frame_current_cropped_numpy*max_value_possible/scale_to_255).astype(np.uint16), False, flag_scale=False, flag_save_uint16=True)
        save_image_numpy(save_path5, str(frame_index) + '_clean_frame_estimate.png',
                         (clean_frame_current_cropped_numpy*max_value_possible/scale_to_255).astype(np.uint16), False, flag_scale=False, flag_save_uint16=True)
        save_image_numpy(save_path6, str(frame_index) + '_center_frame_noisy.png',
                         (noisy_frame_current_cropped_numpy*max_value_possible/scale_to_255).astype(np.uint16), False, flag_scale=False, flag_save_uint16=True)

    ### Record Stabilized Video: ###
    noisy_frame_current_croped_shifted = shift_matrix_subpixel_torch(noisy_frame_current_cropped, shifts_x_sum, shifts_y_sum)
    image_to_write = noisy_frame_current_croped_shifted
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_to_255 * stretch_factor_from_scaling_by_255).clip(0, 255).astype(np.uint8)
    video_object_StabilizedVideo.write(image_to_write)

    ### Record Noisy-Clean Video: ###
    image_to_write = torch.cat((noisy_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write,image_to_write,image_to_write),1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1,2,0)
    image_to_write = (image_to_write*scale_to_255*stretch_factor_from_scaling_by_255).clip(0,255).astype(np.uint8)
    video_object.write(image_to_write)

    ### Record Clean VS. SimpleAverage: ###
    image_to_write = torch.cat((clean_frame_current, clean_frame_simple_averaging), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_to_255*stretch_factor_from_scaling_by_255).clip(0, 255).astype(np.uint8)
    video_object_CleanEstimateVSAverage.write(image_to_write)

    ### Record Original VS. Clean: ###
    image_to_write = torch.cat((original_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_to_255*stretch_factor_from_scaling_by_255).clip(0, 255).astype(np.uint8)
    video_object_OriginalVSCleanEstimate.write(image_to_write)

    ### Record Reset Gate: ###
    image_to_write = torch.Tensor(reset_gates)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_to_255).clip(0, 255).astype(np.uint8)
    video_object_ResetGate.write(image_to_write)

    ### Record Residual From Clean Estimate: ###
    noise_residual = (noisy_frame_current_cropped - clean_frame_current)
    image_to_write = torch.Tensor(noise_residual).abs()
    fig = imshow_torch(image_to_write * max_value_possible)
    title('Residual From Clean Estimate')
    figure_image = video_get_mat_from_figure(fig, (1024, 1024))
    video_object_Residual.write(figure_image)
    plt.close()

    ### Record Noise Map: ###
    #TODO: if i want constant color map for noise AND actually see something i need smart scaling especially for this
    image_to_write = torch.Tensor(noise_map_current).abs()
    fig = imshow_torch(image_to_write * max_value_possible)
    title('Inferred Signal Mean Normalized: ' + str((float(decimal_notation((clean_frame_current * scale_to_255).mean(), 1)))) + ', ' +
          'Inferred Sigma Normalized: ' + str((float(decimal_notation((noise_map_current * scale_to_255).mean(), 1)))) + ', ' + '\n' +
          'Inferred Signal Mean: ' + str((float(decimal_notation((clean_frame_current * max_value_possible).mean(), 1)))) + ', ' +
          'Inferred Sigma: ' + str((float(decimal_notation((noise_map_current * max_value_possible).mean(), 1)))) +
          ', Inferred SNR: ' + decimal_notation(np.abs((clean_frame_current / (noise_map_current + 1e-7))).mean(), 1))
    figure_image = video_get_mat_from_figure(fig, (1024, 1024))
    video_object_Noise.write(figure_image)
    plt.close()


    ### Record Noisy-Clean-NoiseEstimate: ###
    image_to_write = torch.cat((noisy_frame_current_cropped, clean_frame_current, noise_map_current * clean_frame_current.max()/noise_map_current.max()), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_to_255 * stretch_factor).clip(0, 255).astype(np.uint8)
    fig = figure()
    imshow(image_to_write)
    title('Inferred Signal Mean Normalized: ' + str((float(decimal_notation((clean_frame_current*scale_to_255).mean(),1)))) + ', ' +
          'Inferred Sigma Normalized: ' + str((float(decimal_notation((noise_map_current*scale_to_255).mean(),1)))) + ', ' + '\n' +
          'Inferred Signal Mean: ' + str((float(decimal_notation((clean_frame_current*max_value_possible).mean(),1)))) + ', ' +
          'Inferred Sigma: ' + str((float(decimal_notation((noise_map_current*max_value_possible).mean(),1)))) + ', '
          'Inferred SNR: ' + decimal_notation(np.abs((clean_frame_current/(noise_map_current+1e-7))).mean(), 1) )
    figure_image = video_get_mat_from_figure(fig,(crop_size_x, crop_size_x))
    video_object_NoiseEstimate.write(figure_image)
    plt.close()

    ### Add Frame To History: ###
    (clean_frame_current - original_frame_current_cropped).std()
    (noisy_frame_current_cropped - original_frame_current_cropped).std()
    (clean_frame_perfect_averaging - original_frame_current_cropped).std()
    (clean_frame_perfect_averaging - original_frame_seed_cropped).std()
    (original_frame_current_cropped - original_frame_seed_cropped).std()

    ### Transfer torch images to numpy: ###
    clean_frame_current_numpy = ((clean_frame_current[0]*255*stretch_factor_from_scaling_by_255).clamp(0,255).cpu().numpy().transpose(1, 2, 0)).astype(uint8)
    noisy_frame_current_cropped_numpy = ((noisy_frame_current_cropped[0]*255*stretch_factor_from_scaling_by_255).clamp(0,255).cpu().numpy().transpose(1, 2, 0)).astype(uint8)
    original_frame_current_cropped_numpy = ((original_frame_current_cropped[0]*255*stretch_factor_from_scaling_by_255).clamp(0,255).cpu().numpy().transpose(1, 2, 0)).astype(uint8)
    perfect_average_frame_current_cropped_numpy = ((clean_frame_perfect_averaging[0]*255*stretch_factor_from_scaling_by_255).clamp(0,255).cpu().numpy().transpose(1, 2, 0)).astype(uint8)
    simple_average_frame_current_cropped_numpy = ((clean_frame_simple_averaging[0]*255*stretch_factor_from_scaling_by_255).clamp(0,255).cpu().numpy().transpose(1, 2, 0)).astype(uint8)
    original_frame_seed_cropped_numpy = ((original_frame_seed_cropped[0]*255*stretch_factor_from_scaling_by_255).clamp(0, 255).cpu().numpy().transpose(1, 2, 0) * 255).astype(uint8)

    ### Add images to lists for later handling: ###
    clean_frames_list.append(clean_frame_current_numpy)
    noisy_frames_list.append(noisy_frame_current_cropped_numpy)
    original_frames_list.append(original_frame_current_cropped_numpy)
    perfect_average_frames_list.append(perfect_average_frame_current_cropped_numpy)
    simple_average_frames_list.append(simple_average_frame_current_cropped_numpy)
    seed_frames_list.append(original_frame_seed_cropped_numpy)


### Stop Video Writer Object: ###
video_object.release()
video_object_NoiseEstimate.release()
video_object_CleanEstimateVSAverage.release()
video_object_OriginalVSCleanEstimate.release()


### Get Metrics For Shifts: ###
inferred_shifts_x = [x[0] for x in inferred_shifts_list]
inferred_shifts_y = [x[1] for x in inferred_shifts_list]
inferred_shifts_x = np.array(inferred_shifts_x)
inferred_shifts_y = np.array(inferred_shifts_y)
shift_x_std = (inferred_shifts_x).std()  #TODO: get std from smoothed mean graph?
shift_y_std = (inferred_shifts_y).std()

figure()
plot(inferred_shifts_x)
legend(['inferred_shift_x'])
title('Shift X STD: ' + str(shift_x_std))
plt.savefig(os.path.join(save_path, 'Results', 'Shifts_Graph_x.png'))
plt.close()

figure()
plot(inferred_shifts_y)
legend(['inferred_shift_y'])
title('Shift Y STD: ' + str(shift_y_std))
plt.savefig(os.path.join(save_path, 'Results', 'Shifts_Graph_y.png'))
plt.close()

# ### Get Metrics For Video: ###
# output_dict_CleanOriginal_average, output_dict_CleanOriginal_history = \
#     get_metrics_video_lists(original_frames_list, clean_frames_list, number_of_images=np.inf)
# output_dict_NoisyOriginal_average, output_dict_NoisyOriginal_history = \
#     get_metrics_video_lists(original_frames_list, noisy_frames_list, number_of_images=np.inf)
# output_dict_PerfectAverageOriginal_average, output_dict_PerfectAverageOriginal_history = \
#     get_metrics_video_lists(seed_frames_list, perfect_average_frames_list, number_of_images=np.inf)
# output_dict_SimpleAverageOriginal_average, output_dict_SimpleAverageOriginal_history = \
#     get_metrics_video_lists(original_frames_list, simple_average_frames_list, number_of_images=np.inf)
# path_make_path_if_none_exists(os.path.join(save_path,'Results'))
# for key in output_dict_CleanOriginal_history.keys():
#     try:
#         y1 = np.array(output_dict_NoisyOriginal_history.inner_dict[key])
#         y2 = np.array(output_dict_CleanOriginal_history.inner_dict[key])
#         y3 = np.array(output_dict_PerfectAverageOriginal_history.inner_dict[key])
#         y4 = np.array(output_dict_SimpleAverageOriginal_history.inner_dict[key])
#         plot_multiple([y1, y2, y3, y4],
#                       legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
#                                      'cleaned-noisy: ' + decimal_notation(y2.mean(), 2),
#                                      'cleaned-PerfectAverage: ' + decimal_notation(y3.mean(), 2),
#                                      'cleaned-SimpleAverage: ' + decimal_notation(y4.mean(), 2)],
#                       super_title=key + ' over time', x_label='frame-counter', y_label=key)
#         plt.savefig(os.path.join(save_path, 'Results', key + ' over time.png'))
#         plt.close()
#     except:
#         1
# plt.close()







from RapidBase.Utils.IO.Path_and_Reading_utils import save_image_numpy

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
original_frames_folder = path_fix_path_for_linux('/home/mafat/Insync/dudy_karl/Google Drive - Shared drives/GalGalaim/GalGalaim/Algo-Team/DataSets/Gimbaless/Experiment 09.03.2021/Illumination v= 30.1  a = 0.01 lux=  4/fps = 450 exposure = 2.2 ms')
### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies_ImageBatch'
results_path_name_addition = 'KAYA_OFIR_EXPERIMENT_FPS450_EXP2.2'
results_path = os.path.join(results_path, results_path_name_addition)
path_make_path_if_none_exists(results_path)
###
### Gain Offset Matrices Paths: ###
gain_offset_folder = r'/home/mafat/DataSets/KAYA_EXPERIMENT/Experiment 09.03.2021/Integration sphere'
gain_matrix_path = os.path.join(gain_offset_folder, 'fps__450_exposure__2.2_ms_gain.mat')
offset_matrix_path = os.path.join(gain_offset_folder, 'fps__450_exposure__2.2_ms_offset.mat')
###

### Read Gain/Offset Matrices: ###
gain_average = load_image_mat(gain_matrix_path).squeeze()
offset_average = load_image_mat(offset_matrix_path).squeeze()
gain_avg_minus_offset_avg = gain_average - offset_average

#######################################################################################################################################
### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Methos: ###
image_index_to_start_from = 10
number_of_images_to_generate_each_time = 5
number_of_images = 100
registration_algorithm = 'CrossCorrelation'  #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR, NormalizedCrossCorrelation
noise_estimation_algorithm = 'IterativeResidual'  #IterativeResidual, FullFrameHistogram
downsample_kernel_size_FrameCombine = 8
downsample_kernel_size_NoiseEstimation = 8
downsample_factor_registration = 2
max_shift_in_pixels = 0
max_search_area = 11 #for algorithms like NCC
SNR = np.inf
crop_size_x = np.inf
crop_size_y = np.inf
inner_crop_size = 1000
noise_sigma = 1/sqrt(SNR)

#######################################################################################################################################
### IO: ###
IO_dict = EasyDict()
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


####################################
### DataSets: ###
original_frames_dataset = DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=original_frames_folder,
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=100,  #TODO: add possibility to randomize start index from a possibly long video
                                                         number_of_images_per_video_to_load=number_of_images_to_generate_each_time,
                                                         max_number_of_videos=np.inf,
                                                         crop_size=np.inf,
                                                         flag_to_RAM=True,
                                                         flag_explicitely_make_tensor=True,
                                                         flag_normalize_by_255=False,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict,
                                                         allowed_extentions=['png', 'tif', '.raw'])


### Take care of initializations and such for first frame: ###
#TODO: create dataset object like DataSet_SingleVideo_RollingIndex_AddNoise but which does it not load superflously
normalization_factor_to_get_255_max = 256
original_frames_output_dict = original_frames_dataset[image_index_to_start_from]
original_frames_current = original_frames_output_dict.original_frames.data[:,0:1,:,:]
original_frames_current = original_frames_current / normalization_factor_to_get_255_max
noisy_frames_current = original_frames_current + noise_sigma*torch.randn_like(original_frames_current)
noisy_frames_current_cropped = crop_torch_batch(noisy_frames_current, crop_size_x, crop_style='center')
original_frames_current_cropped = crop_torch_batch(original_frames_current, crop_size_x, crop_style='center')

# ### Use All Frames To Find Noise Estimate (Not for Real-Time): ###
# histogram_spatial_binning_factor = 4
# crude_histogram_bins_vec = my_linspace(0, 1, 10)
# clean_frames_estimate = original_frames_current.median(0)[0].unsqueeze(0)
# noise_estimate = (clean_frames_estimate - noisy_frames_current).std(0).unsqueeze(0)
# SNR_estimate = clean_frames_estimate / (noise_estimate + 1e-4)
# SNR_estimate_median = SNR_estimate.median()
# SNR_estimate_simple = SNR_estimate.clamp(0,SNR_estimate_median*2)


### Initialize More Things: ###
shift_layer_torch = Shift_Layer_Torch()

B,C,H,W = original_frames_current.shape
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
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(save_path3)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name = os.path.join(save_path, 'Results', 'Results.avi')
video_name_NoiseEstimate = os.path.join(save_path, 'Results', 'NoiseEstimate.avi')
video_name_CleanVSAverage = os.path.join(save_path, 'Results', 'Clean_VS_SimpleAverage.avi')

video_width = crop_size_x * 2
video_height = crop_size_x * 1
video_object = cv2.VideoWriter(video_name, 0, 10, (crop_size_x * 2, crop_size_x))
video_object_NoiseEstimate = cv2.VideoWriter(video_name_NoiseEstimate, 0, 10, (crop_size_x, crop_size_x))
video_object_CleanEstimateVSAverage = cv2.VideoWriter(video_name_CleanVSAverage, 0, 10, (crop_size_x * 2, crop_size_x))
video_object_OriginalVSCleanEstimate = cv2.VideoWriter(video_name_CleanVSAverage, 0, 10, (crop_size_x * 2, crop_size_x))

for frame_index in np.arange(1, number_of_images-number_of_images_to_generate_each_time//2-1):
    print(frame_index)
    ### Get Next Frame BATCH: ###
    original_frames_output_dict = original_frames_dataset[frame_index]
    current_frames_original = original_frames_output_dict.original_frames[:,0:1,:,:]
    current_frames_noisy = original_frames_output_dict.output_frames_noisy[:,0:1,:,:]

    ### Center Frames
    center_frame_current_original = original_frames_output_dict.center_frame_original.unsqueeze(0)[:,0:1,:,:]
    center_frame_current_noisy = original_frames_output_dict.center_frame_noisy.unsqueeze(0)[:,0:1,:,:]

    ### Make Gain/Offset Correction: ### #TODO: insert into dataset object
    current_frames_original = (current_frames_original - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)
    current_frames_noisy = (current_frames_noisy - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)
    center_frame_current_original = (center_frame_current_original - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)
    center_frame_current_noisy = (center_frame_current_noisy - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)

    current_frames_original = current_frames_original / 12
    current_frames_noisy = current_frames_noisy / 12
    center_frame_current_original = center_frame_current_original / 12
    center_frame_current_noisy = center_frame_current_noisy / 12

    current_frames_original = current_frames_original.clamp(0,4096)
    current_frames_noisy = current_frames_noisy.clamp(0,4096)
    center_frame_current_original = center_frame_current_original.clamp(0,4096)
    center_frame_current_noisy = center_frame_current_noisy.clamp(0,4096)

    # imshow_torch_BW(current_frames_original/current_frames_original.max())
    # ### Add Noise: ###
    # current_frames_noisy = current_frames_original + 1/sqrt(SNR)*torch.randn_like(current_frames_original)
    # current_frame_noisy = current_frames_noisy[current_frames_noisy//2:current_frames_noisy//2+1,:,:,;]

    ### Center Crop: ###
    # original_frame_current_cropped = crop_torch_batch(center_frame_current_original, crop_size_x, crop_style='center')
    # noisy_frame_current_cropped = crop_torch_batch(center_frame_current_noisy, crop_size_x, crop_style='center')
    original_frame_current_cropped = center_frame_current_original
    noisy_frame_current_cropped = center_frame_current_noisy

    ### Keep Track Of Simple Averaging (No Tracking): ###
    clean_frame_simple_averaging = 0
    for i in np.arange(current_frames_noisy.shape[0]):
        clean_frame_simple_averaging += current_frames_noisy[i:i+1]
    clean_frame_perfect_averaging = clean_frame_simple_averaging #TODO: take care of this
    # clean_frame_simple_averaging = crop_torch_batch(clean_frame_simple_averaging, crop_size_x, crop_style='center')
    # clean_frame_perfect_averaging = crop_torch_batch(clean_frame_perfect_averaging, crop_size_x, crop_style='center')

    ### Register Images (return numpy arrays): ###
    # registration_algorithm = 'CrossCorrelation' #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR
    shifts_array, aligned_frames_list = register_images_batch(current_frames_noisy,
                                                              registration_algorithm,
                                                              max_search_area,
                                                              inner_crop_size,
                                                              downsample_factor_registration,
                                                              flag_do_initial_SAD=False,
                                                              initial_SAD_search_area=5,
                                                              center_frame_index=number_of_images_to_generate_each_time // 2)
    inferred_shifts_x = []
    inferred_shifts_y = []
    for inferred_shifts_current in shifts_array:
        inferred_shifts_x.append(inferred_shifts_current[0])
        inferred_shifts_y.append(inferred_shifts_current[1])

    ### Average Frames After Alignment: ###
    clean_frame_current = torch.zeros((1, 1, aligned_frames_list[0].shape[-2], aligned_frames_list[0].shape[-1]))
    for i in np.arange(len(aligned_frames_list)):
        clean_frame_current += aligned_frames_list[i]
    clean_frame_current = clean_frame_current / len(aligned_frames_list)
    clean_frame_current = crop_torch_batch(clean_frame_current, crop_size_x, crop_style='center')

    # imshow_torch(noisy_frame_current_cropped.unsqueeze(0))
    # imshow_torch(clean_frame_current.unsqueeze(0))
    # imshow_torch(original_frame_current_cropped.unsqueeze(0))

    ### Save Wanted Images: ###
    save_image_numpy(save_path1, str(frame_index) + '_center_frame_original.png', original_frame_current_cropped[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True, flag_convert_to_uint8=True)
    save_image_numpy(save_path2, str(frame_index) + '_clean_frame_estimate.png', clean_frame_current[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True, flag_convert_to_uint8=True)
    save_image_numpy(save_path3, str(frame_index) + '_center_frame_noisy.png', noisy_frame_current_cropped[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True, flag_convert_to_uint8=True)

    ### Record Noisy-Clean Video: ###
    stretch_factor = 256  #TODO: take care of this to make sense and be automatic
    image_to_write = torch.cat((noisy_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write,image_to_write,image_to_write),1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1,2,0)
    image_to_write = (image_to_write * stretch_factor).clip(0,255).astype(np.uint8)
    video_object.write(image_to_write)

    ### Record Clean VS. SimpleAverage: ###
    image_to_write = torch.cat((clean_frame_current, clean_frame_simple_averaging), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * stretch_factor).clip(0, 255).astype(np.uint8)
    video_object_CleanEstimateVSAverage.write(image_to_write)

    ### Record Original VS. Clean: ###
    image_to_write = torch.cat((original_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * stretch_factor).clip(0, 255).astype(np.uint8)
    video_object_OriginalVSCleanEstimate.write(image_to_write)

    ### Add Frame To History: ###
    (clean_frame_current - original_frame_current_cropped).std()
    (noisy_frame_current_cropped - original_frame_current_cropped).std()

    ### Transfer torch images to numpy: ###
    clean_frame_current_numpy = (clean_frame_current[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*1).astype(uint8)
    noisy_frame_current_cropped_numpy = (noisy_frame_current_cropped[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*1).astype(uint8)
    original_frame_current_cropped_numpy = (original_frame_current_cropped[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*1).astype(uint8)
    perfect_average_frame_current_cropped_numpy = (clean_frame_perfect_averaging[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*1).astype(uint8)
    simple_average_frame_current_cropped_numpy = (clean_frame_simple_averaging[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*1).astype(uint8)

    ### Add images to lists for later handling: ###
    clean_frames_list.append(clean_frame_current_numpy)
    noisy_frames_list.append(noisy_frame_current_cropped_numpy)
    original_frames_list.append(original_frame_current_cropped_numpy)
    perfect_average_frames_list.append(perfect_average_frame_current_cropped_numpy)
    simple_average_frames_list.append(simple_average_frame_current_cropped_numpy)


### Stop Video Writer Object: ###
video_object.release()
video_object_NoiseEstimate.release()
video_object_CleanEstimateVSAverage.release()
video_object_OriginalVSCleanEstimate.release()

### Get Metrics For Video: ###
output_dict_CleanOriginal_average, output_dict_CleanOriginal_history = \
    get_metrics_video_lists(original_frames_list, clean_frames_list, number_of_images=np.inf)
output_dict_NoisyOriginal_average, output_dict_NoisyOriginal_history = \
    get_metrics_video_lists(original_frames_list, noisy_frames_list, number_of_images=np.inf)
output_dict_PerfectAverageOriginal_average, output_dict_PerfectAverageOriginal_history = \
    get_metrics_video_lists(original_frames_list, perfect_average_frames_list, number_of_images=np.inf)
output_dict_SimpleAverageOriginal_average, output_dict_SimpleAverageOriginal_history = \
    get_metrics_video_lists(original_frames_list, simple_average_frames_list, number_of_images=np.inf)
path_make_path_if_none_exists(os.path.join(save_path,'Results'))
for key in output_dict_CleanOriginal_history.keys():
    try:
        y1 = np.array(output_dict_NoisyOriginal_history.inner_dict[key])
        y2 = np.array(output_dict_CleanOriginal_history.inner_dict[key])
        y3 = np.array(output_dict_PerfectAverageOriginal_history.inner_dict[key])
        y4 = np.array(output_dict_SimpleAverageOriginal_history.inner_dict[key])
        plot_multiple([y1, y2, y3, y4],
                      legend_labels=['original-noisy: ' + decimal_notation(y1.mean(), 2),
                                     'cleaned-noisy: ' + decimal_notation(y2.mean(), 2),
                                     'cleaned-PerfectAverage: ' + decimal_notation(y3.mean(), 2),
                                     'cleaned-SimpleAverage: ' + decimal_notation(y4.mean(), 2)],
                      super_title=key + ' over time', x_label='frame-counter', y_label=key)
        plt.savefig(os.path.join(save_path, 'Results', key + ' over time.png'))
        plt.close()
    except:
        1
plt.close()

# ### Get Metrics For Shifts: ###
# real_shifts_x = [x[0] for x in real_shifts_list]
# real_shifts_y = [x[1] for x in real_shifts_list]
# inferred_shifts_x = [x[0] for x in inferred_shifts_list]
# inferred_shifts_y = [x[1] for x in inferred_shifts_list]
# inferred_shifts_x = np.array(inferred_shifts_x)
# inferred_shifts_y = np.array(inferred_shifts_y)
# real_shifts_x = np.array(real_shifts_x).squeeze()
# real_shifts_y = np.array(real_shifts_y).squeeze()
# shift_x_std = (real_shifts_x - inferred_shifts_x).std()
# shift_y_std = (real_shifts_y - inferred_shifts_y).std()
#
# plot(real_shifts_x)
# plot(inferred_shifts_x)
# legend(['real_shift_x','inferred_shift_x'])
# title('Shift X STD: ' + str(shift_x_std))
# plt.savefig(os.path.join(save_path, 'Results', 'Shifts_Graph_x.png'))
# plt.close()
#
# plot(real_shifts_y)
# plot(inferred_shifts_y)
# legend(['real_shift_y','inferred_shift_y'])
# title('Shift Y STD: ' + str(shift_y_std))
# plt.savefig(os.path.join(save_path, 'Results', 'Shifts_Graph_y.png'))
# plt.close()





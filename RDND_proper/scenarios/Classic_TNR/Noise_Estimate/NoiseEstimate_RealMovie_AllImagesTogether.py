from RapidBase.Utils.IO.Imshow_and_Plots import save_image_torch

from RapidBase.Utils.IO.Path_and_Reading_utils import save_image_numpy

from RapidBase.import_all import *

from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.models.BasicSR.RapidBase.Utils.MISCELENEOUS import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Noise_Estimate_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration_misc import *


### Paths: ###
##############
### Movies Paths: ###
# original_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/original_frames')
original_frames_folder = os.path.join(datasets_main_folder, '/Example Videos/Beirut_inf')
original_frames_folder = os.path.join(datasets_main_folder, '/KAYA_EXPERIMENT/19')

### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies_ImageBatch'
results_path_name_addition = 'KAYA19'
results_path = os.path.join(results_path, results_path_name_addition)
path_make_path_if_none_exists(results_path)
###


#######################################################################################################################################
### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Methos: ###
image_index_to_start_from = 10
number_of_images_to_generate_each_time = 50
number_of_images = 50
registration_algorithm = 'NormalizedCrossCorrelation'  #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR, NormalizedCrossCorrelation
noise_estimation_algorithm = 'IterativeResidual'  #IterativeResidual, FullFrameHistogram
downsample_kernel_size_FrameCombine = 8
downsample_kernel_size_NoiseEstimation = 8
downsample_factor_registration = 2
max_shift_in_pixels = 0
max_search_area = 11 #for algorithms like NCC
SNR = np.inf
crop_size_x = 1024
crop_size_y = 1024
inner_crop_size = 1000
noise_sigma = 1/sqrt(SNR)

#######################################################################################################################################
### IO: ###
IO_dict = EasyDict()
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
                                                         flag_normalize_by_255=True,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict,
                                                         allowed_extentions=['png', 'tif'])


### Take care of initializations and such for first frame: ###
#TODO: create dataset object like DataSet_SingleVideo_RollingIndex_AddNoise but which does it not load superflously
normalization_factor_to_get_255_max = 256
original_frames_output_dict = original_frames_dataset[image_index_to_start_from]
original_frames_current = original_frames_output_dict.original_frames.data
original_frames_current = original_frames_current / normalization_factor_to_get_255_max
noisy_frames_current = original_frames_current + noise_sigma*torch.randn_like(original_frames_current)
noisy_frames_current_cropped = crop_torch_batch(noisy_frames_current, crop_size_x, crop_style='center')
original_frames_current_cropped = crop_torch_batch(original_frames_current, crop_size_x, crop_style='center')

### Align All images if wanted: ###
flag_align_images = False
if flag_align_images:
    registration_algorithm = 'CrossCorrelation' #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR
    downsample_factor_registration = 1
    max_search_area = 7 #for algorithms like NCC  (max_search_area > max_shift_in_pixels*2 + 2)
    inner_crop_size = 1000
    shifts_array, aligned_frames_list = register_images_batch(noisy_frames_current,
                                                              registration_algorithm,
                                                              max_search_area,
                                                              inner_crop_size,
                                                              downsample_factor_registration,
                                                              flag_do_initial_SAD=False,
                                                              initial_SAD_search_area=5,
                                                              center_frame_index=noisy_frames_current.shape[0]//2)
    aligned_frames_list = [torch.Tensor(x).unsqueeze(0).unsqueeze(0) for x in aligned_frames_list]
else:
    aligned_frames_tensor = noisy_frames_current


### Get Clean Frame Estimate: ###
# noisy_frames_current = 0.5 * torch.ones((50,1,2048,2048)) + 0.1 * torch.randn((50,1,2048,2048))  #TODO: TEMP- DELETE

clean_frames_estimate = noisy_frames_current.median(0)[0].unsqueeze(0)
save_image_torch(folder_path=results_path, filename='Clean_Frame_Estimate.png',
                 torch_tensor=clean_frames_estimate[0]*255**2,
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)

### Use Simple Calculation (Temporal Statistics Per Pixel): ###
noise_estimate = (clean_frames_estimate - noisy_frames_current).std(0).unsqueeze(0)
SNR_estimate = clean_frames_estimate / (noise_estimate + 1e-7)
SNR_estimate_median = SNR_estimate.median()
noise_estimate_median = noise_estimate.median()
SNR_estimate_simple = SNR_estimate.clamp(0,SNR_estimate_median*2)
save_image_torch(folder_path=results_path, filename='SNR_estimate_temporal.png',
                 torch_tensor=SNR_estimate_simple[0],
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)

### Spatio-Temporal STD Statistics Of Aligned Images: ###
downsample_kernel_size = 4
noise_map_estimate_spatio_temporal, SNR_estimate_spatio_temporal = estimate_noise_ImagesBatch(noisy_frames_current,
                              clean_frames_estimate,
                              'spatial_differences', #'spatial_differences', 'difference_histogram_peak', 'histogram_width', 'per_pixel_histogram'
                              histogram_spatial_binning_factor=None,
                              crude_histogram_bins_vec=None,
                              downsample_kernel_size=downsample_kernel_size)
SNR_estimate_spatio_temporal = SNR_estimate_spatio_temporal.clamp(0,SNR_estimate_median*2)
save_image_torch(folder_path=results_path, filename='SNR_estimate_spatio_temporal.png',
                 torch_tensor=SNR_estimate_spatio_temporal[0],
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)
save_image_torch(folder_path=results_path, filename='NoiseMap_estimate_spatio_temporal_interpolation.png',
                 torch_tensor=(noise_map_estimate_spatio_temporal[0]*255**2).clamp(0,(255**2*noise_estimate_median)*2),
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)


### Histogram Width Of Images Batch: ###
histogram_spatial_binning_factor = 3
crude_histogram_bins_vec = my_linspace(0, 0.2, 10)
downsample_kernel_size = 1
noise_map_estimate_width, SNR_estimate_width, noise_as_function_of_gray_level, histograms_bins_list, histograms_list = estimate_noise_ImagesBatch(noisy_frames_current,
                              clean_frames_estimate,
                              'histogram_width', #'spatial_differences', 'difference_histogram_peak', 'histogram_width', 'per_pixel_histogram'
                              histogram_spatial_binning_factor,
                              crude_histogram_bins_vec,
                               downsample_kernel_size=downsample_kernel_size)
SNR_estimate_width = SNR_estimate_width.clamp(0,SNR_estimate_median*2)
save_image_torch(folder_path=results_path, filename='SNR_estimate_histogram_width_interpolation.png',
                 torch_tensor=SNR_estimate_width[0],
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)
save_image_torch(folder_path=results_path, filename='NoiseMap_estimate_histogram_width_interpolation.png',
                 torch_tensor=(noise_map_estimate_width[0]*255**2).clamp(0,(255**2*noise_estimate_median)*2),
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)


### Histogram Of Differences Peaks Of Images Batch: ###
histogram_spatial_binning_factor = 3
crude_histogram_bins_vec = my_linspace(0, 1, 10)
downsample_kernel_size = 1
noise_map_estimate_peak, SNR_estimate_peak, noise_as_function_of_gray_level, histograms_bins_list, histograms_list = estimate_noise_ImagesBatch(noisy_frames_current,
                              clean_frames_estimate,
                              'difference_histogram_peak', #'spatial_differences', 'difference_histogram_peak', 'histogram_width', 'per_pixel_histogram'
                              histogram_spatial_binning_factor,
                              crude_histogram_bins_vec,
                               downsample_kernel_size=downsample_kernel_size)
SNR_estimate_peak = SNR_estimate_peak.clamp(0,SNR_estimate_median*2)
save_image_torch(folder_path=results_path, filename='SNR_estimate_histogram_peak_interpolation.png',
                 torch_tensor=SNR_estimate_peak[0],
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)
save_image_torch(folder_path=results_path, filename='NoiseMap_estimate_histogram_peak_interpolation.png',
                 torch_tensor=(noise_map_estimate_peak[0]*255**2).clamp(0,(255**2*noise_estimate_median)*2),
                 flag_convert_bgr2rgb=False,
                 flag_scale_by_255=False,
                 flag_array_to_uint8=False,
                 flag_imagesc=False,
                 flag_convert_grayscale_to_heatmap=False,
                 flag_save_figure=True, flag_colorbar=True)

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
original_frames_folder = os.path.join(datasets_main_folder, '/KAYA_EXPERIMENT/1-20210214T111908Z-001/1')
noisy_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/noisy_frames')
clean_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/clean_frame_estimate')
### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies'
results_path_name_addition = 'KAYA1'
path_make_path_if_none_exists(results_path)
###



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




#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
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
                                                         flag_normalize_by_255=True,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict,
                                                         allowed_extentions=['png', 'tif'])
noisy_frames_dataset = DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=noisy_frames_folder,
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
                                                         Train_dict=IO_dict)


### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Methos: ###
number_of_images = 100
registration_algorithm = 'NormalizedCrossCorrelation'  #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR, NormalizedCrossCorrelation
noise_estimation_algorithm = 'IterativeResidual'  #IterativeResidual, FullFrameHistogram
downsample_kernel_size_FrameCombine = 8
downsample_kernel_size_NoiseEstimation = 8
downsample_factor_registration = 2
max_shift_in_pixels = 1.5
max_search_area = 11 #for algorithms like NCC
SNR = 10
crop_size_x = 1024
crop_size_y = 1024
inner_crop_size = 1000
noise_sigma = 1/sqrt(SNR)

### Take care of initializations and such for first frame: ###
original_frames_output_dict = original_frames_dataset[0]
original_frame_current = original_frames_output_dict.center_frame_original[0].data
original_frame_current = original_frame_current.unsqueeze(0).unsqueeze(0) / 256
noisy_frame_current = original_frame_current + noise_sigma*torch.randn_like(original_frame_current)
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

for frame_index in np.arange(1, number_of_images):
    print(frame_index)
    ### Get Next Frame: ###
    original_frames_output_dict = original_frames_dataset[frame_index]
    original_frame_current = original_frames_output_dict.center_frame_original[0].data
    original_frame_current = original_frame_current.unsqueeze(0).unsqueeze(0) / 256

    # ### Shift: ###
    # shiftx = get_random_number_in_range(-max_shift_in_pixels, max_shift_in_pixels)
    # shifty = get_random_number_in_range(-max_shift_in_pixels, max_shift_in_pixels)
    # print('shifts: ' + str(shiftx) + ', ' + str(shifty))
    # original_frame_current = shift_layer_torch.forward(original_frame_previous, shiftx, shifty)

    ### Add Noise: ###
    noisy_frame_current = original_frame_current + 1/sqrt(SNR)*torch.randn_like(original_frame_current)

    ### Center Crop: ###
    noisy_frame_current_cropped = crop_torch_batch(noisy_frame_current, crop_size_x, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_x, crop_style='center')

    ### Keep Track Of Simple Averaging (No Tracking): ###
    clean_frame_simple_averaging = (1 - 1 / (frame_index+1)) * clean_frame_simple_averaging + \
                                   (1 / (frame_index+1)) * noisy_frame_current_cropped
    ### Keep Track Of Perfect Tracking + Averaging: ###
    clean_frame_perfect_averaging = (1 - 1 / (frame_index + 1)) * clean_frame_perfect_averaging + \
                                    (1 / (frame_index + 1)) * (original_frame_seed_cropped + 1 / sqrt(SNR) * torch.randn_like(noise_map_last))

    ### Register Images (return numpy arrays): ###
    # registration_algorithm = 'CrossCorrelation' #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR
    shifts_array, clean_frame_previous_warped = register_images(noisy_frame_current_cropped, clean_frame_previous,
                                                                registration_algorithm, max_search_area, inner_crop_size, downsample_factor_registration)
    # shifts_array, clean_frame_previous_warped = register_image_SAD(noisy_frame_current_cropped, clean_frame_previous, search_area=5)
    print('deduced shifts: ' + str(shifts_array))

    ### Keep Track of recorded shifts to compare with real shifts: ###
    # real_shifts_list.append((shiftx,shifty))
    inferred_shifts_list.append(shifts_array)

    ### Get both frames to be in Tensor / Numpy Type: ###
    clean_frame_previous_warped = torch.Tensor(clean_frame_previous_warped)

    ### Estimate noise: ###
    if noise_estimation_algorithm == 'IterativeResidual':
        noise_map_current, noise_pixel_counts_map_current = estimate_noise_IterativeResidual(noisy_frame_current_cropped, noise_map_previous,
                                                             pixel_counts_map_previous, clean_frame_previous_warped,
                                                             downsample_kernel_size=downsample_kernel_size_NoiseEstimation)
    elif noise_estimation_algorithm == 'FullFrameHistogram':
        noise_map_current = estimate_noise_FullFrameHistogram(noisy_frame_current_cropped, noise_map_previous,
                                                              pixel_counts_map_previous, clean_frame_previous_warped,
                                                              downsample_kernel_size=downsample_kernel_size_NoiseEstimation)

    ### Combine Images: ###
    clean_frame_current, pixel_counts_map_current = combine_frames(noisy_frame_current_cropped,
                                         clean_frame_previous_warped,
                                         noise_map_current = noise_map_current,
                                         noise_map_last_warped = None,
                                         pixel_counts_map_previous = pixel_counts_map_previous,
                                         downsample_kernel_size = downsample_kernel_size_FrameCombine)

    ### Keep variables for next frame: ###
    clean_frame_previous = clean_frame_current  #TODO: right now i simply keep the last noisy image, as i don't clean anything now
    original_frame_previous = original_frame_current
    noisy_frame_previous = noisy_frame_current
    pixel_counts_map_previous = pixel_counts_map_current
    noise_map_previous = noise_map_current

    # imshow_torch(noisy_frame_current_cropped.unsqueeze(0))
    # imshow_torch(clean_frame_current.unsqueeze(0))
    # imshow_torch(original_frame_current_cropped.unsqueeze(0))

    ### Save Wanted Images: ###
    save_image_numpy(save_path1, str(frame_index) + '_center_frame_original.png', original_frame_current_cropped[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True)
    save_image_numpy(save_path2, str(frame_index) + '_clean_frame_estimate.png', clean_frame_current[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True)
    save_image_numpy(save_path3, str(frame_index) + '_center_frame_noisy.png', noisy_frame_current_cropped[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True)

    ### Record Noisy-Clean Video: ###
    image_to_write = torch.cat((noisy_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write,image_to_write,image_to_write),1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1,2,0)
    image_to_write = (image_to_write*255).clip(0,255).astype(np.uint8)
    video_object.write(image_to_write)

    ### Record Clean VS. SimpleAverage: ###
    image_to_write = torch.cat((clean_frame_current, clean_frame_simple_averaging), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * 255).clip(0, 255).astype(np.uint8)
    video_object_CleanEstimateVSAverage.write(image_to_write)

    ### Record Original VS. Clean: ###
    image_to_write = torch.cat((original_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * 255).clip(0, 255).astype(np.uint8)
    video_object_OriginalVSCleanEstimate.write(image_to_write)


    ### Record Noisy-Clean-NoiseEstimate: ###
    image_to_write = torch.cat((noisy_frame_current_cropped, clean_frame_current, noise_map_current), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * 255).clip(0, 255).astype(np.uint8)
    fig = figure()
    imshow(image_to_write)
    title('Real Sigma: ' + str(int(float(decimal_notation(noise_sigma*255,0)))) + ', ' +
          'Inferred Sigma: ' + str(int(float(decimal_notation((noise_map_current*255).mean(),0)))) + ', ' +
          'SigmaDiff_STD: ' + decimal_notation(np.abs((noise_map_current-noise_sigma)*255).std(), 2) )
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
    clean_frame_current_numpy = (clean_frame_current[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)
    noisy_frame_current_cropped_numpy = (noisy_frame_current_cropped[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)
    original_frame_current_cropped_numpy = (original_frame_current_cropped[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)
    perfect_average_frame_current_cropped_numpy = (clean_frame_perfect_averaging[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)
    simple_average_frame_current_cropped_numpy = (clean_frame_simple_averaging[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)
    original_frame_seed_cropped_numpy = (original_frame_seed_cropped[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(uint8)

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

### Get Metrics For Video: ###
output_dict_CleanOriginal_average, output_dict_CleanOriginal_history = \
    get_metrics_video_lists(original_frames_list, clean_frames_list, number_of_images=np.inf)
output_dict_NoisyOriginal_average, output_dict_NoisyOriginal_history = \
    get_metrics_video_lists(original_frames_list, noisy_frames_list, number_of_images=np.inf)
output_dict_PerfectAverageOriginal_average, output_dict_PerfectAverageOriginal_history = \
    get_metrics_video_lists(seed_frames_list, perfect_average_frames_list, number_of_images=np.inf)
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





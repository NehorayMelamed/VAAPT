# from RapidBase.import_all import *
from RapidBase.TrainingCore.datasets import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *


### Paths: ###
##############
### Movies Paths: ###
original_frames_folder = os.path.join(datasets_main_folder, '/KAYA_EXPERIMENT/Experiment 09.03.2021/'
                                                 'Illumination v= 30.1  a = 0.01 lux=  4/fps = 450 exposure = 2.2 ms')

### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies_Kaya_FFTRegistration'
path_make_path_if_none_exists(results_path)
###

### Gain Offset Matrices Paths: ###
gain_offset_folder = r'/home/mafat/DataSets/KAYA_EXPERIMENT/Experiment 09.03.2021/Integration sphere'
gain_matrix_path = os.path.join(gain_offset_folder, 'fps__450_exposure__2.2_ms_gain.mat')
offset_matrix_path = os.path.join(gain_offset_folder, 'fps__450_exposure__2.2_ms_offset.mat')
# gain_matrix_path = os.path.join(gain_offset_folder, 'fps__630_exposure__1.5_ms_gain.mat')
# offset_matrix_path = os.path.join(gain_offset_folder, 'fps__630_exposure__1.5_ms_offset.mat')
### Read Gain/Offset Matrices: ###
gain_average = load_image_mat(gain_matrix_path).squeeze()
offset_average = load_image_mat(offset_matrix_path).squeeze()
gain_average = torch.Tensor(gain_average).unsqueeze(0).unsqueeze(0).cuda()
offset_average = torch.Tensor(offset_average).unsqueeze(0).unsqueeze(0).cuda()
gain_avg_minus_offset_avg = gain_average - offset_average
###


#######################################################################################################################################
### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Methos: ###
Algo_Dict = EasyDict()
image_index_to_start_from = 0
number_of_images_to_generate_each_time = 9
number_of_images = 100
flag_initial_shift_method = 'sub_pixel_fft' # 'integer_roll', 'sub_pixel_fft'
flag_CC_finetunning = 'none'  # 'none', 'manual_CC'
flag_use_clean_or_original_frame_for_reference = 'original'  # 'original', 'clean'
registration_algorithm = 'CrossCorrelation'  #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR, NormalizedCrossCorrelation
max_search_area = 11 #for algorithms like NCC
downsample_kernel_size_FrameCombine = 1
downsample_kernel_size_NoiseEstimation = 1
downsample_factor_Registration = 1
crop_size_initial_W = np.inf
crop_size_initial_H = np.inf
final_crop_size = (512, 512)

### Set Up Avg Pooling Layers: ###
downsample_layer_FrameCombine = nn.AvgPool2d(downsample_kernel_size_FrameCombine)
downsample_layer_NoiseEstimatione = nn.AvgPool2d(downsample_kernel_size_NoiseEstimation)
downsample_layer_Registration = nn.AvgPool2d(downsample_factor_Registration)
#######################################################################################################################################



#######################################################################################################################################
### Get Images: ###
image_filenames = read_image_filenames_from_folder(path=original_frames_folder,
                                     number_of_images=np.inf,
                                     allowed_extentions=['.raw'],
                                     flag_recursive=True,
                                     string_pattern_to_search='*')
#######################################################################################################################################


#######################################################################################################################################
### Take care of initializations and such for first frame: ###
normalization_factor_after_NUC = 20
original_frame_current = read_image_general(image_filenames[0])
original_frame_current = torch.Tensor(original_frame_current).permute([2,0,1]).unsqueeze(0).cuda()
original_frame_current = (original_frame_current - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)
original_frame_current = original_frame_current / normalization_factor_after_NUC
original_frame_current_cropped = crop_torch_batch(original_frame_current, (crop_size_initial_W, crop_size_initial_H), crop_style='center')
original_frame_current_cropped = downsample_layer_Registration(original_frame_current_cropped)
#######################################################################################################################################


#######################################################################################################################################
### Initialize More Things: ###
shift_layer_torch = Shift_Layer_Torch()

noise_map_previous = None
clean_frames_list = []
original_frames_list = []
aligned_frames_list = []
perfect_average_frames_list = []
simple_average_frames_list = []
seed_frames_list = []

real_shifts_list = []
inferred_shifts_list = []

### Get Video Recorder: ###
crop_size_x = final_crop_size[0]
experiment_name = 'first_try'
save_path = os.path.join(results_path, experiment_name)
save_path1 = os.path.join(save_path, 'original_frames')
save_path2 = os.path.join(save_path, 'clean_frame_estimate')
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name_OriginalAligned = os.path.join(save_path, 'Results', 'OriginalVsAligned_bla.avi')
video_name_OriginalClean = os.path.join(save_path, 'Results', 'OriginalVsClean_bla.avi')
video_width = crop_size_x * 2
video_height = crop_size_x * 1
video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (crop_size_x * 2, crop_size_x))
video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (crop_size_x * 2, crop_size_x))
#######################################################################################################################################



#######################################################################################################################################
affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
number_of_frames_to_average = 5
final_crop_size = final_crop_size[0]

### Populate Initial List: ###
original_frame_initial = read_image_general(image_filenames[0])
original_frame_initial = torch.Tensor(original_frame_initial).cuda().permute([2, 0, 1]).unsqueeze(0)
original_frame_initial = (original_frame_initial - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)
original_frame_initial = crop_torch_batch(original_frame_initial, final_crop_size, crop_style='center')
original_frames_list.append(original_frame_initial)
aligned_frames_list.append(original_frame_initial)
scale_factor = (210/original_frame_initial.max()).item()

### Loop Over Images: ###
for frame_index in np.arange(1, number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    original_frame_previous = aligned_frames_list[0]
    # original_frame_previous = original_frames_list[0]
    original_frame_current = read_image_general(image_filenames[frame_index])
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2,0,1]).unsqueeze(0)
    original_frame_current = (original_frame_current - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)

    ### Register Images: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, final_crop_size, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, final_crop_size, crop_style='center')
    original_frame_previous_cropped = nn.AvgPool2d(1)(original_frame_previous_cropped)
    original_frame_current_cropped = nn.AvgPool2d(1)(original_frame_current_cropped)
    original_frames_list.append(original_frame_current_cropped)

    ## Scale-Rotation-Translation Discovery: ###
    initial_scale_rotation_registration_downsample_factor = 1
    affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
    recovered_angle, recovered_scale, recovered_translation, original_frame_current_cropped_warped = affine_registration_layer.forward(
        original_frame_previous_cropped,
        original_frame_current_cropped,
        downsample_factor=initial_scale_rotation_registration_downsample_factor,
        flag_return_shifted_image=True)

    # cross_correlation, shifts, shifts_sub_pixel = get_cross_correlation_and_shifts_subpixel_torch(original_frame_previous_cropped, original_frame_current_cropped)
    # original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped, shifts_sub_pixel[1], shifts_sub_pixel[0])

    ### Update Aligned Frames Lists: ###
    aligned_frames_list.append(original_frame_current_cropped_warped)

    ### Average ALIGNED Frames: ###
    clean_frame_current = torch.zeros((1,1,original_frame_current_cropped_warped[0].shape[-2],original_frame_current_cropped_warped[0].shape[-1])).cuda()
    for i in np.arange(len(aligned_frames_list)):
        clean_frame_current += aligned_frames_list[i]
    clean_frame_current = clean_frame_current / len(aligned_frames_list)
    clean_frames_list.append(clean_frame_current)

    ### Update Lists: ###
    if frame_index > 9:
        aligned_frames_list.pop(0)
        original_frames_list.pop(0)
        clean_frames_list.pop(0)


    # ### Save Wanted Images: ###
    # save_image_numpy(save_path1, str(frame_index) + '_center_frame_original.png', original_frame_current_cropped[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True)
    # save_image_numpy(save_path2, str(frame_index) + '_clean_frame_estimate.png', clean_frame_current[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True)

    ### Record Original-Clean Video: ###
    image_to_write = torch.cat((original_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write,image_to_write,image_to_write),1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1,2,0)
    image_to_write = (image_to_write * scale_factor).clip(0,255).astype(np.uint8)
    video_object_OriginalClean.write(image_to_write)

    ### Record Original-Aligned: ###
    image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    video_object_OriginalAligned.write(image_to_write)


    ### Transfer torch images to numpy: ###
    clean_frame_current_numpy = (clean_frame_current[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)
    original_frame_current_cropped_numpy = (original_frame_current_cropped[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)

    # ### Add images to lists for later handling: ###
    # clean_frames_list.append(clean_frame_current_numpy)
    # original_frames_list.append(original_frame_current_cropped_numpy)


### Stop Video Writer Object: ###
video_object_OriginalAligned.release()
video_object_OriginalClean.release()
1
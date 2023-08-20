# from RapidBase.import_all import *
from RapidBase.TrainingCore.datasets import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *


### Paths: ###
##############
### Movies Paths: ###
original_frames_folder = '/home/mafat/DataSets/Aviram/2'

### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies_Kaya_FFTRegistration'
path_make_path_if_none_exists(results_path)
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
initial_crop_size = (1024, 1024)
final_crop_size = (800, 800)
final_final_crop_size = (700, 700)
# final_crop_size = (200,200)

### Set Up Avg Pooling Layers: ###
downsample_layer_FrameCombine = nn.AvgPool2d(downsample_kernel_size_FrameCombine)
downsample_layer_NoiseEstimatione = nn.AvgPool2d(downsample_kernel_size_NoiseEstimation)
downsample_layer_Registration = nn.AvgPool2d(downsample_factor_Registration)
#######################################################################################################################################



#######################################################################################################################################
### Get Images: ###
image_filenames = read_image_filenames_from_folder(path=original_frames_folder,
                                     number_of_images=np.inf,
                                     allowed_extentions=['.png'],
                                     flag_recursive=True,
                                     string_pattern_to_search='*')
#######################################################################################################################################


#######################################################################################################################################
### Take care of initializations and such for first frame: ###
original_frame_current = read_image_general(image_filenames[0])
original_frame_current = torch.Tensor(original_frame_current).permute([2,0,1]).unsqueeze(0).cuda()
original_frame_initial = original_frame_current
original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size, crop_style='center')
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
video_name_OriginalAligned = os.path.join(save_path, 'Results', 'OriginalVsAligned.avi')
video_name_OriginalClean = os.path.join(save_path, 'Results', 'OriginalVsClean.avi')
video_width = final_final_crop_size[1] * 2
video_height = final_final_crop_size[0] * 1
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 480))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 480))
video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (video_width, video_height))
video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (video_width, video_height))
#######################################################################################################################################



#######################################################################################################################################
affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
number_of_frames_to_average = 5
# final_crop_size = final_crop_size[0]

### Populate Initial List: ###
original_frame_initial = read_image_general(image_filenames[0])
original_frame_initial = torch.Tensor(original_frame_initial).cuda().permute([2, 0, 1]).unsqueeze(0)
original_frame_initial = crop_torch_batch(original_frame_initial, initial_crop_size, crop_style='center')
original_frame_initial = RGB2BW(original_frame_initial)
original_frame_initial_crop = crop_torch_batch(original_frame_initial, final_final_crop_size, crop_style='center')
original_frames_list.append(original_frame_initial_crop)
aligned_frames_list.append(original_frame_initial_crop)
scale_factor = (210/original_frame_initial_crop.max()).item()

### Loop Over Images: ###
affine_layer_torch = Warp_Tensors_Affine_Layer()
recovered_angles_list = []
real_angles_list = []
angles_diff_list = []
scale_diff_list = []
for frame_index in np.arange(1, number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    original_frame_previous = aligned_frames_list[0]
    # original_frame_previous = original_frames_list[0]

    original_frame_current = original_frame_initial
    GT_shift_x = np.float32(get_random_number_in_range(-30,30,[1]))
    GT_shift_y = np.float32(get_random_number_in_range(-30,30,[1]))
    GT_rotation_angle = np.float32(get_random_number_in_range(-5,5,[1]))
    GT_scale = np.float32(get_random_number_in_range(1-0.05,1+0.05,[1]))
    original_frame_current = affine_layer_torch.forward(original_frame_current, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bicubic')

    ### Register Images: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_initial, final_crop_size, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, final_crop_size, crop_style='center')
    original_frame_previous_cropped = nn.AvgPool2d(1)(original_frame_previous_cropped)
    original_frame_current_cropped = nn.AvgPool2d(1)(original_frame_current_cropped)
    original_frames_list.append(original_frame_current_cropped)

    ### Add Noise: ###
    SNR = 20
    noise_sigma = 255 * 1/sqrt(SNR)
    original_frame_previous_cropped += torch.randn_like(original_frame_previous_cropped) * noise_sigma
    original_frame_current_cropped += torch.randn_like(original_frame_current_cropped) * noise_sigma

    ## Scale-Rotation-Translation Discovery: ###
    initial_scale_rotation_registration_downsample_factor = 1
    affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
    recovered_angle, recovered_scale, recovered_translation, original_frame_current_cropped_warped = affine_registration_layer.forward(
        original_frame_previous_cropped,
        original_frame_current_cropped,
        downsample_factor=initial_scale_rotation_registration_downsample_factor,
        flag_return_shifted_image=True)
    # print(recovered_angle)
    # print(abs(GT_rotation_angle-recovered_angle.item()))
    print(abs(GT_scale-recovered_scale))
    recovered_angles_list.append(recovered_angle)
    real_angles_list.append(GT_rotation_angle)
    angles_diff_list.append(abs(GT_rotation_angle-recovered_angle.item()))
    scale_diff_list.append(abs(GT_scale-recovered_scale))

    ### Crop Again To Avoid InValid Regions: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_initial, final_final_crop_size, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, final_final_crop_size, crop_style='center')
    original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, final_final_crop_size, crop_style='center')

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
    # image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped-original_frame_current_cropped), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    video_object_OriginalAligned.write(image_to_write)


    # ### Transfer torch images to numpy: ###
    # clean_frame_current_numpy = (clean_frame_current[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)
    # original_frame_current_cropped_numpy = (original_frame_current_cropped[0].clamp(0,1).cpu().numpy().transpose(1, 2, 0)*255).astype(uint8)

    # ### Add images to lists for later handling: ###
    # clean_frames_list.append(clean_frame_current_numpy)
    # original_frames_list.append(original_frame_current_cropped_numpy)


### Stop Video Writer Object: ###
video_object_OriginalAligned.release()
video_object_OriginalClean.release()
1

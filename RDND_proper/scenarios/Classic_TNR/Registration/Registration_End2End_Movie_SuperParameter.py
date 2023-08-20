# from RapidBase.import_all import *
from RapidBase.TrainingCore.datasets import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *


### Paths: ###
##############
### Movies Paths: ###
# original_frames_folder = '/home/mafat/DataSets/Aviram/2'
original_frames_folder = '/home/mafat/DataSets/Drones_Random_Experiments/NINJAV_S001_S001_T015_8B'

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
image_index_to_start_from = 1000
number_of_images_to_generate_each_time = 9
number_of_images = 200
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

initial_crop_size = (1024*2, 1024*2)
crop_size_before_cross_correlation = (1024*2, 1024*2)  #perhapse to avoid worse case scenario of mismatch
crop_size_after_cross_correlation = (1024, 1024)  #perhapse to avoid worse case scenario of mismatch
# final_crop_size = (512, 512)
# final_final_crop_size = (480, 480)
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
### Initialize More Things: ###
shift_layer_torch = Shift_Layer_Torch()

noise_map_previous = None
clean_frames_list = []
original_frames_list = []
original_frames_cropped_list = []
aligned_frames_list = []
perfect_average_frames_list = []
simple_average_frames_list = []
seed_frames_list = []

real_shifts_list = []
inferred_shifts_list = []

### Get Video Recorder: ###
experiment_name = 'Sony2'
save_path = os.path.join(results_path, experiment_name)
save_path1 = os.path.join(save_path, 'original_frames')
save_path2 = os.path.join(save_path, 'clean_frame_estimate')
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name_OriginalAligned = os.path.join(save_path, 'Results', 'OriginalVsAligned.avi')
video_name_OriginalClean = os.path.join(save_path, 'Results', 'OriginalVsClean.avi')
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 480))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 480))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 512))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 512))
video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024*2, 1024))
video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024*2, 1024))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (400, 200))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (400, 200))
#######################################################################################################################################



#######################################################################################################################################
affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
number_of_frames_to_average = 17

### Populate Initial List: ###
original_frame_initial = read_image_general(image_filenames[image_index_to_start_from])
original_frame_initial = torch.Tensor(original_frame_initial).cuda().permute([2, 0, 1]).unsqueeze(0)
original_frame_initial = crop_torch_batch(original_frame_initial, initial_crop_size, crop_style='center')
original_frame_initial = RGB2BW(original_frame_initial)
original_frames_list.append(original_frame_initial)
aligned_frames_list.append(crop_torch_batch(original_frame_initial, crop_size_before_cross_correlation, crop_style='center'))
clean_frames_list.append(crop_torch_batch(original_frame_initial, crop_size_after_cross_correlation, crop_style='center'))  #TODO: in the previous script i didn't initialize it with the first frame...interesting
original_frames_cropped_list.append(crop_torch_batch(original_frame_initial, crop_size_after_cross_correlation, crop_style='center'))
# scale_factor = (210/original_frame_initial.max()).item()
scale_factor = (120/original_frame_initial.median()).item()


def cross_correlation_alignment_super_parameter(input_dict):
    ### Get Images To Align Together: ###
    aligned_frames_list = input_dict['aligned_frames_list']
    original_frames_list = input_dict['original_frames_list']
    original_frames_cropped_list = input_dict['original_frames_cropped_list']
    clean_frames_list = input_dict['clean_frames_list']
    number_of_frames_to_average = input_dict['number_of_frames_to_average']

    original_frames_reference = aligned_frames_list[0]
    # original_frames_reference = original_frames_list[0]
    original_frame_current = original_frames_list[-1]

    ### Crop Images: ###
    original_frames_reference_cropped = crop_torch_batch(original_frames_reference, crop_size_before_cross_correlation, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_before_cross_correlation, crop_style='center')

    ### Use Cross Correlation To Find Translation Between Images: ###
    cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(
        original_frames_reference_cropped, original_frame_current_cropped)
    print('shifts: ' + str(shifts))
    print('shifts_sub_pixel: ' + str(shifts_sub_pixel))
    # shifts_sub_pixel = (0,0)

    ### Align The Images Using FFT Translation: ###
    original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped,
                                                                        shifts_sub_pixel[1], shifts_sub_pixel[0])

    ### Update Aligned Frames Lists: ###
    aligned_frames_list.append(original_frame_current_cropped_warped)

    ### Crop Again To Avoid InValid Regions: ###
    original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, crop_size_after_cross_correlation, crop_style='center')
    original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, crop_size_after_cross_correlation, crop_style='center')

    ### Append Cropped Frames: ###
    original_frames_cropped_list.append(original_frame_current_cropped)

    ### Average Aligned Frames: ###
    clean_frame_current = torch.zeros((1, 1, original_frame_current_cropped_warped[0].shape[-2],
                                       original_frame_current_cropped_warped[0].shape[-1])).cuda()
    for i in np.arange(len(aligned_frames_list)):
        # clean_frame_current += aligned_frames_list[i]
        clean_frame_current += crop_torch_batch(aligned_frames_list[i], crop_size_after_cross_correlation, crop_style='center')
    clean_frame_current = clean_frame_current / len(aligned_frames_list)
    clean_frames_list.append(clean_frame_current)

    ### Update Lists: ###
    if len(aligned_frames_list) > number_of_frames_to_average:
        aligned_frames_list.pop(0)
        original_frames_list.pop(0)
        original_frames_cropped_list.pop(0)
        clean_frames_list.pop(0)

    ### Update Super Dictionary: ###
    input_dict['aligned_frames_list'] = aligned_frames_list
    input_dict['original_frames_list'] = original_frames_list
    input_dict['clean_frames_list'] = clean_frames_list
    input_dict['original_frames_cropped_list'] = original_frames_cropped_list

    return input_dict

### Loop Over Images: ###
input_dict = EasyDict()
input_dict.aligned_frames_list = aligned_frames_list
input_dict.original_frames_list = original_frames_list
input_dict.original_frames_cropped_list = original_frames_cropped_list
input_dict.clean_frames_list = clean_frames_list
input_dict.number_of_frames_to_average = number_of_frames_to_average
input_dict.initial_crop_size = initial_crop_size
input_dict.crop_size_before_cross_correlation = crop_size_before_cross_correlation
input_dict.crop_size_after_cross_correlation = crop_size_after_cross_correlation
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + 1 + number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    original_frame_current = read_image_general(image_filenames[frame_index]).astype('float32')
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2,0,1]).unsqueeze(0)
    original_frame_current = RGB2BW(original_frame_current)

    ### Register Images: ###
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size, crop_style='center')
    original_frame_current_cropped = nn.AvgPool2d(1)(original_frame_current_cropped)
    original_frames_list.append(original_frame_current_cropped)

    ### Update Dict: ###
    input_dict.original_frames_list = original_frames_list

    ### Perform Alignment: ###
    input_dict = cross_correlation_alignment_super_parameter(input_dict)

    ### Outputs From Dict: ###
    clean_frame_current = input_dict['clean_frames_list'][-1]
    original_frame_current_cropped = input_dict['original_frames_cropped_list'][-1]
    original_frame_current_cropped_warped = crop_torch_batch(input_dict['aligned_frames_list'][-1], input_dict.crop_size_after_cross_correlation)

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
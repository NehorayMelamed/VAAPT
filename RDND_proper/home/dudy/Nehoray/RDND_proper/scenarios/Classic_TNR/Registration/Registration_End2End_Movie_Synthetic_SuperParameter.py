# from RapidBase.import_all import *
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *


### Temp: ###
file_path = '/media/mafat/dudy/TIGER/8B/NINJAV_S001_S001_T015_8B.mov'
file_name = os.path.split(file_path)[-1]
final_save_path = '/home/mafat/DataSets/Drones_Random_Experiments/NINJAV_S001_S001_T015_8B'
path_make_path_if_none_exists(final_save_path)
# video_to_images()
video_stream = cv2.VideoCapture(file_path)
# video_stream.open()
counter = 0
while video_stream.isOpened():
    flag_frame_available, current_frame = video_stream.read()
    if flag_frame_available:
        # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        # current_frame = current_frame.astype('float32')
        # current_frame = RGB2BW(current_frame)
        # current_frame = current_frame * 1
        image_full_filename = str.replace(file_name, '.mov', '_' + string_rjust(counter, 4) + '.png')
        image_full_filename = str.replace(image_full_filename, '.mkv', '_' + string_rjust(counter, 4) + '.png')
        image_full_filename = str.replace(image_full_filename, '.avi', '_' + string_rjust(counter, 4) + '.png')
        image_full_filename = str.replace(image_full_filename, '.mp4', '_' + string_rjust(counter, 4) + '.png')
        current_image_final_save_path = os.path.join(final_save_path, image_full_filename)
        cv2.imwrite(
            current_image_final_save_path,
            current_frame.astype('uint8'))
        counter += 1
        print(counter)
video_stream.release()
# clahe_object = cv2.createCLAHE(5,(100,100))
# bla = clahe_object.apply(RGB2BW(current_frame).squeeze().astype('uint8'))
# imshow(bla)

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
image_index_to_start_from = 1000
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
crop_size_before_cross_correlation = (1024, 1024)  #perhapse to avoid worse case scenario of mismatch
crop_size_after_cross_correlation = (480, 480)  #perhapse to avoid worse case scenario of mismatch
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
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 480))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 480))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 512))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 512))
video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (480*2, 480))
video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (480*2, 480))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (400, 200))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (400, 200))
#######################################################################################################################################



#######################################################################################################################################
affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
number_of_frames_to_average = 5

### Populate Initial List: ###
original_frame_initial = read_image_general(image_filenames[image_index_to_start_from])
original_frame_initial = torch.Tensor(original_frame_initial).cuda().permute([2, 0, 1]).unsqueeze(0)
original_frame_initial = crop_torch_batch(original_frame_initial, initial_crop_size, crop_style='center')
original_frame_initial = RGB2BW(original_frame_initial)
original_frames_list.append(original_frame_initial)
aligned_frames_list.append(crop_torch_batch(original_frame_initial, crop_size_before_cross_correlation, crop_style='center'))
clean_frames_list.append(crop_torch_batch(original_frame_initial, crop_size_after_cross_correlation, crop_style='center'))  #TODO: in the previous script i didn't initialize it with the first frame...interesting
original_frames_cropped_list.append(crop_torch_batch(original_frame_initial, crop_size_after_cross_correlation, crop_style='center'))
scale_factor = (210/original_frame_initial.max()).item()


### Initialize Dict: ###
input_dict = EasyDict()
input_dict.number_of_frames_to_average = number_of_frames_to_average
input_dict.crop_size_before_cross_correlation = crop_size_before_cross_correlation
input_dict.crop_size_after_cross_correlation = crop_size_after_cross_correlation
input_dict.counter = 0

### Loop Over Images: ###
affine_layer_torch = Warp_Tensors_Affine_Layer()
counter = 0
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + 1 + number_of_images):
    print(frame_index)

    ### Get Current Frame: ###
    original_frame_current = original_frame_initial
    if counter < 50:
        GT_shift_x = np.float32(get_random_number_in_range(-30, 30, [1]))
        GT_shift_y = np.float32(get_random_number_in_range(-30, 30, [1]))
        GT_rotation_angle = np.float32(get_random_number_in_range(-0, 0, [1]))
        GT_scale = np.float32(get_random_number_in_range(1 - 0.0, 1 + 0.0, [1]))
    else:
        GT_shift_x = np.float32(get_random_number_in_range(-0, 0, [1]))
        GT_shift_y = np.float32(get_random_number_in_range(-0, 0, [1]))
        GT_rotation_angle = np.float32(get_random_number_in_range(-0, 0, [1]))
        GT_scale = np.float32(get_random_number_in_range(1 - 0.0, 1 + 0.0, [1]))
    original_frame_current = affine_layer_torch.forward(original_frame_current, GT_shift_x, GT_shift_y, GT_scale,
                                                        GT_rotation_angle, flag_interpolation_mode='bicubic')

    ### Register Images: ###
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size, crop_style='center')
    original_frame_current_cropped = nn.AvgPool2d(1)(original_frame_current_cropped)

    ### Perform Alignment: ###
    # input_dict = cross_correlation_alignment_sup

    input_dict = cross_correlation_alignment_super_parameter_efficient(input_dict, original_frame_current_cropped)

    ### Outputs From Dict: ###
    clean_frame_current = input_dict['clean_frames_list'][-1]
    original_frame_current_cropped = input_dict['original_frames_cropped_list'][-1]
    original_frame_current_cropped_warped = crop_torch_batch(input_dict['aligned_frames_list'][-1], input_dict.crop_size_after_cross_correlation, 'center')

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

    counter += 1

### Stop Video Writer Object: ###
video_object_OriginalAligned.release()
video_object_OriginalClean.release()
1
import torch

from RapidBase.import_all import *
from RapidBase.TrainingCore.datasets import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
# from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *
# from RDND_proper.scenarios.Classic_TNR.Registration.MinSAD_Registration import minSAD_FFT_Affine_Registration_Layer, minSAD_FFT_Affine_Registration_MemorySave_Layer

import kornia


### Paths: ###
##############
### Movies Paths: ###
original_frames_folder = '/home/mafat/DataSets/pngsh/pngs/2'

### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/Palantir_2'
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
final_crop_size = (512, 512)
final_final_crop_size = (512, 512)
# final_crop_size = (200,200)
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
original_frame_current_cropped = crop_torch_batch(original_frame_current, (crop_size_initial_W, crop_size_initial_H), crop_style='center')
#######################################################################################################################################


#######################################################################################################################################
### Take Care Of Crop Sizes - very important when doing fitting/stabilization: ###
original_frame_current = read_image_general(image_filenames[0])
H,W,C = original_frame_current.shape
initial_crop_size_first_stage = (H, W)
final_crop_size_for_movie_first_stage = (H, W)
# max_estimated_shift = max(max(abs(np.array(estimated_shifts_x_list))), max(abs(np.array(estimated_shifts_y_list)))).data  #the real world minimal result in the field
max_estimated_shift = 50
initial_crop_size_second_stage = (int(initial_crop_size_first_stage[0]-max_estimated_shift), int(initial_crop_size_first_stage[1]-max_estimated_shift)) #after cross correlation alignment
final_crop_size_for_movie_second_stage = initial_crop_size_second_stage
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
video_name_Original_Vs_Aligned = os.path.join(save_path, 'Results', 'Original_Vs_Aligned.avi')
video_name_Original_Vs_Aligned = os.path.join(save_path, 'Results', 'Original_Vs_Aligned_SecondStage.avi')
video_name_Original_Vs_BG = os.path.join(save_path, 'Results', 'Original_Vs_BG.avi')
video_name_Aligned_Vs_BG = os.path.join(save_path, 'Results', 'Aligned_Vs_BG.avi')
video_name_Aligned_Vs_BGCanny = os.path.join(save_path, 'Results', 'Aligned_Vs_BGCanny.avi')
video_name_Aligned_Vs_BGCanny_Dilated = os.path.join(save_path, 'Results', 'Aligned_Vs_BGCannyDilated.avi')
video_name_Aligned_Vs_BGSubstractedAligned = os.path.join(save_path, 'Results', 'Aligned_Vs_BGSubstractedAligned.avi')
video_name_Aligned_Vs_BGSubstractedAlignedMaskedByBGCanny = os.path.join(save_path, 'Results', 'Aligned_Vs_BGSubstractedAligned_MaskedByBGCanny.avi')
video_name_Aligned_Vs_BGSubstractedAlignedCanny = os.path.join(save_path, 'Results', 'Aligned_Vs_BGSubstractedAlignedCanny.avi')
video_name_Aligned_Vs_BGSubstractedAlignedBinary = os.path.join(save_path, 'Results', 'Aligned_Vs_BGSubstractedAlignedBinary.avi')
video_name_Aligned_Vs_BGSubstractedAlignedBinaryMaskedByCanny = os.path.join(save_path, 'Results', 'Aligned_Vs_BGSubstractedAlignedBinaryMaskedByCanny.avi')
video_name_Aligned_Vs_BGSubstractedAlignedBinaryMaskedByCanny_median = os.path.join(save_path, 'Results', 'Aligned_Vs_BGSubstractedAlignedBinaryMaskedByCanny_median.avi')
video_width = crop_size_x * 2
video_height = crop_size_x * 1
video_object_Original_Vs_Aligned = cv2.VideoWriter(video_name_Original_Vs_Aligned, 0, 10, (final_crop_size_for_movie_first_stage[1]*2, final_crop_size_for_movie_first_stage[0]))
video_object_Original_Vs_Aligned_second_stage = cv2.VideoWriter(video_name_Original_Vs_Aligned, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Original_Vs_BG = cv2.VideoWriter(video_name_Original_Vs_BG, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BG = cv2.VideoWriter(video_name_Aligned_Vs_BG, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGCanny = cv2.VideoWriter(video_name_Aligned_Vs_BGCanny, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGCanny_Dilated = cv2.VideoWriter(video_name_Aligned_Vs_BGCanny_Dilated, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGSubstractedAligned = cv2.VideoWriter(video_name_Aligned_Vs_BGSubstractedAligned, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGSubstractedAlignedMaskedByBGCanny = cv2.VideoWriter(video_name_Aligned_Vs_BGSubstractedAlignedMaskedByBGCanny, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGSubstractedAlignedCanny = cv2.VideoWriter(video_name_Aligned_Vs_BGSubstractedAlignedCanny, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGSubstractedGradientsBinary = cv2.VideoWriter(video_name_Aligned_Vs_BGSubstractedAlignedBinary, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGSubstractedGradientsBinaryMaskedByBGCanny = cv2.VideoWriter(video_name_Aligned_Vs_BGSubstractedAlignedBinaryMaskedByCanny, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
video_object_Aligned_Vs_BGSubstractedGradientsBinaryMaskedByBGCanny_median = cv2.VideoWriter(video_name_Aligned_Vs_BGSubstractedAlignedBinaryMaskedByCanny_median, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
#######################################################################################################################################



#######################################################################################################################################
affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
minSAD_fft_affine_registration_layer = minSAD_FFT_Affine_Registration_MemorySave_Layer()
fft_translation_rotation_layer = FFT_Translation_Rotation_Layer()
number_of_frames_to_average = 5
final_crop_size = final_crop_size[0]

### Populate Initial List: ###
#TODO: (H,W)=(400,1024), perhapse relax cropping a bit
original_frame_initial = read_image_general(image_filenames[image_index_to_start_from])
original_frame_initial = torch.Tensor(original_frame_initial).cuda().permute([2, 0, 1]).unsqueeze(0)
original_frame_initial = crop_torch_batch(original_frame_initial, final_crop_size, crop_style='center')
original_frame_initial_crop = crop_torch_batch(original_frame_initial, final_final_crop_size, crop_style='center')
original_frames_list.append(original_frame_initial)
aligned_frames_list.append(original_frame_initial)
scale_factor = (210/original_frame_initial.max()).item()


###############################################################################################################################################
### Loop Over Images And Align Them!: ###
estimated_shifts_x_list = []
estimated_shifts_y_list = []
estimated_angles_list = []

### Actually Loop Over The images and align them using cross correlation as first stage: ###
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + 1 + number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    # original_frame_previous = aligned_frames_list[0]
    # original_frame_previous = original_frames_list[0]
    #(1). Previous Image:
    original_frame_previous = read_image_general(image_filenames[image_index_to_start_from+1])
    # original_frame_previous = read_image_general(image_filenames[frame_index])
    original_frame_previous = torch.Tensor(original_frame_previous).cuda().permute([2, 0, 1]).unsqueeze(0)
    #(2). Current Image
    original_frame_current = read_image_general(image_filenames[frame_index+1])
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2,0,1]).unsqueeze(0)

    ### Crop Images: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, initial_crop_size_first_stage, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size_first_stage, crop_style='center')
    original_frames_list.append(original_frame_current_cropped.cpu())
    # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped)

    # ### Cross Correlation Translation Correction: ###
    cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(original_frame_previous_cropped, original_frame_current_cropped)
    rotation_sub_pixel = torch.Tensor([0]).data
    original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped, shifts_sub_pixel[1], shifts_sub_pixel[0])
    shifts_sub_pixel_initial_CC = shifts_sub_pixel
    shift_x_sub_pixel_initial_CC = shifts_sub_pixel[0]
    shift_y_sub_pixel_initial_CC = shifts_sub_pixel[1]
    # # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped_warped)
    # # imshow_torch(abs(original_frame_previous_cropped - original_frame_current_cropped_warped))
    # # imshow_torch(abs(original_frame_previous_cropped - original_frame_current_cropped))

    # ## Scale-Rotation-Translation Discovery Using FFT-Registration layer which uses LogPolar transform: ###
    # initial_scale_rotation_registration_downsample_factor = 1
    # affine_registration_layer = get_rotation_scaling_and_translation_object()
    # recovered_angle, recovered_scale, recovered_translation, original_frame_current_cropped_warped = affine_registration_layer.forward(
    #     original_frame_previous_cropped,
    #     original_frame_current_cropped,
    #     downsample_factor=initial_scale_rotation_registration_downsample_factor,
    #     flag_return_shifted_image=True, flag_interpolation_mode='bilinear')
    # shifts_sub_pixel = (recovered_translation[0], recovered_translation[1])
    # # print(recovered_angle)

    # ### minSAD FFT Layer for translation and rotation: ###
    # #(1). Crop images before registering to avoid edge effects:
    # B,C,H,W = original_frame_previous_cropped.shape
    # frame_size_to_cut = 0
    # H_new = H - frame_size_to_cut
    # W_new = W - frame_size_to_cut
    # original_frame_previous_cropped_new = crop_torch_batch(original_frame_previous_cropped, (W_new, H_new), crop_style='center')
    # original_frame_current_cropped_new = crop_torch_batch(original_frame_current_cropped_warped, (W_new, H_new), crop_style='center')
    # #(2). minSAD registration:
    # shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, original_frame_current_cropped_warped = minSAD_fft_affine_registration_layer.forward(original_frame_previous_cropped_new,
    #                                                                                                                            original_frame_current_cropped_new)
    # shifts_sub_pixel = (shift_x_sub_pixel, shift_y_sub_pixel)

    ### Translate and Rotate according to found rotations and translations: ###
    # rotation_sub_pixel = rotation_sub_pixel * 0
    #TODO: perhapse add to shift_x_sub_pixel_initial_CC the result from the upwards minSAD layer later when it's more accurate????
    # original_frame_current_cropped_warped = fft_translation_rotation_layer.forward(original_frame_current_cropped,
    #                                                                                shift_y_sub_pixel_initial_CC.unsqueeze(0).unsqueeze(0),
    #                                                                                shift_x_sub_pixel_initial_CC.unsqueeze(0).unsqueeze(0),
    #                                                                                rotation_sub_pixel.unsqueeze(0).unsqueeze(0) * np.pi/180)
    # original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, (W, H), crop_style='center')

    ### Update Shifts/Angles Lists: ###
    estimated_shifts_x_list.append(shifts_sub_pixel[0])
    estimated_shifts_y_list.append(shifts_sub_pixel[1])
    estimated_angles_list.append(rotation_sub_pixel)

    ### Update Aligned Frames Lists: ###
    aligned_frames_list.append(original_frame_current_cropped_warped.cpu())

    ### Crop Again To Avoid InValid Regions: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, final_crop_size_for_movie_first_stage, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, final_crop_size_for_movie_first_stage, crop_style='center')
    original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, final_crop_size_for_movie_first_stage, crop_style='center')

    ### Record Original-Aligned: ###
    image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped), -1)
    # image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped-original_frame_current_cropped), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    video_object_Original_Vs_Aligned.write(image_to_write)

### Stop Video Writer Object: ###
video_object_Original_Vs_Aligned.release()
1
###############################################################################################################################################


###############################################################################################################################################
### Go over the frames, and use the consecutive-pairs affine transforms estimated and align them all according to the original frame for total stability: ###
final_aligned_frames_list = []
current_shift_x = 0
current_shift_y = 0
current_rotation = 0

for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    # original_frame_previous = aligned_frames_list[0]
    # original_frame_previous = original_frames_list[0]
    #(1). Previous Image:
    original_frame_previous = read_image_general(image_filenames[image_index_to_start_from+1])
    # original_frame_previous = read_image_general(image_filenames[frame_index])
    original_frame_previous = torch.Tensor(original_frame_previous).cuda().permute([2, 0, 1]).unsqueeze(0)
    #(2). Current Image
    # original_frame_current = read_image_general(image_filenames[frame_index+1])
    original_frame_current = aligned_frames_list[frame_index+1].cuda()
    # original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2,0,1]).unsqueeze(0)
    # imshow_torch(original_frame_previous); imshow_torch(original_frame_current)

    ### Crop Images: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, initial_crop_size_second_stage, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size_second_stage, crop_style='center')
    original_frames_list.append(original_frame_current_cropped.cpu())
    # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped)

    # ### Warp second image to align with first according to first stage estimation: ###
    # #TODO: simply using cumsum isn't enough because of i'm effectively amplifying the low frequencies,
    # # i need to have a continuous high pass filter and then shift it towards original image and register it there
    # current_shift_x += estimated_shifts_x_list[frame_index].unsqueeze(0).unsqueeze(0)
    # current_shift_y += estimated_shifts_y_list[frame_index].unsqueeze(0).unsqueeze(0)
    # current_rotation += estimated_angles_list[frame_index].unsqueeze(0).unsqueeze(0)
    # original_frame_current_cropped_warped = fft_translation_rotation_layer.forward(original_frame_current_cropped,
    #                                                                                -current_shift_x,
    #                                                                                -current_shift_y,
    #                                                                                -current_rotation)
    # B,C,H,W = original_frame_current_cropped.shape
    # original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, (W,H), crop_style='center')

    ### minSAD FFT Layer for translation and rotation: ###
    #(1). Crop images before registering to avoid edge effects:
    B, C, H, W = original_frame_previous_cropped.shape
    #(2). minSAD registration:
    shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, original_frame_current_cropped_warped = minSAD_fft_affine_registration_layer.forward(
                                                                                                                                original_frame_previous_cropped,
                                                                                                                               original_frame_current_cropped)
    shifts_sub_pixel = (shift_x_sub_pixel, shift_y_sub_pixel)

    ### Update Aligned Frames Lists: ###
    final_aligned_frames_list.append(original_frame_current_cropped_warped.cpu())

    ### Crop Again To Avoid InValid Regions: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, final_crop_size_for_movie_second_stage, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, final_crop_size_for_movie_second_stage, crop_style='center')
    original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, final_crop_size_for_movie_second_stage, crop_style='center')
    # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped); imshow_torch(original_frame_current_cropped_warped)


    ### Record Original-Aligned: ###
    image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped), -1)
    # image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped-original_frame_current_cropped), -1)
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    video_object_Original_Vs_Aligned_second_stage.write(image_to_write)

### Stop Video Writer Object: ###
video_object_Original_Vs_Aligned_second_stage.release()
###############################################################################################################################################


###############################################################################################################################################
### Go Over Aligned Movie And Extract Stats and Save Aligned For Later: ###
number_of_frames = len(aligned_frames_list)
number_of_sub_frames = 5
canny_edge_detection_layer = canny_edge_detection(10, True)

### Initialize Lists: ###
BGCanny_list = []
Aligned_BGSubstracted_list = []
Aligned_BGSubstracted_Canny_list = []
Aligned_BGSubstracted_Binary_list = []
Aligned_BGSubstracted_Binary_MaskedByBGCanny_list = []

### Save Videos: ###
def turn_to_video_frame(input_image):
    image_to_write = torch.cat((input_image, input_image, input_image), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    return image_to_write

### Estimate Constant BG For Future Frames (not!!!!!! running bg estimation): ###
first_frame_to_estimate_bg = 0
final_frame_to_estimate_bg = first_frame_to_estimate_bg + 20
aligned_frames_for_bg_list = []
for frame_index in np.arange(first_frame_to_estimate_bg, final_frame_to_estimate_bg):
    ### Get Current Temporal Context For Median Calculation: ###
    current_original_frame = original_frames_list[frame_index + 1].cuda()
    current_aligned_frame = aligned_frames_list[frame_index + 1].cuda()
    aligned_frames_for_bg_list.append(current_aligned_frame)
input_tensor = torch.cat(aligned_frames_for_bg_list)
input_tensor_median = torch.median(input_tensor, 0)[0].unsqueeze(0)  #for using constant median/BG-estimation from the beginning of the movie

### Loop Over Aligned Frames & Get Stats and Stuff: ###
for frame_index, current_frame in enumerate(aligned_frames_list):
    print(frame_index)
    if frame_index < len(aligned_frames_list) - number_of_sub_frames:
        ### Get Current Temporal Context For Median Calculation: ###
        current_original_frame = original_frames_list[frame_index+1].cuda()
        current_aligned_frame = aligned_frames_list[frame_index+1].cuda()

        ### Estimate Running Median For BG Estimation: ###
        current_sub_list = []
        for i in np.arange(number_of_sub_frames):
            current_sub_list.append(aligned_frames_list[frame_index + i + 1].cuda())
            # imshow_torch(aligned_frames_list[frame_index + i])
        input_tensor = torch.cat(current_sub_list)
        input_tensor_median = torch.median(input_tensor, 0)[0].unsqueeze(0)
        # imshow_torch(input_tensor); imshow_torch(input_tensor_median/255)

        ### Perform Canny Edge Detection On Median Image: ###
        canny_threshold_BG = 10
        blurred_img_BG, grad_mag_BG, grad_orientation_BG, thin_edges_BG, thresholded_BG, early_threshold_BG = canny_edge_detection_layer(BW2RGB(input_tensor_median), canny_threshold_BG)
        binary_threshold_BG = 700
        early_threshold_binary_BG = (early_threshold_BG > binary_threshold_BG)
        thresholded_binary_BG = (thresholded_BG > binary_threshold_BG).float()

        ### Get BG-Substracted Image: ###
        input_tensor_BG_substracted = (current_aligned_frame - input_tensor_median).abs()
        # imshow_torch(input_tensor_BG_substracted)

        ### Perform Canny Edge Detection On BG-Substracted Image: ###
        canny_threshold_BGSubstracted = 10
        current_aligned_BGSubstracted = (current_aligned_frame - input_tensor_median).abs()
        blurred_img_BGSubstracted, grad_mag_BGSubstracted, grad_orientation_BGSubstracted, thin_edges_BGSubstracted, thresholded_BGSubstracted, early_threshold_BGSubstracted = canny_edge_detection_layer(BW2RGB(current_aligned_BGSubstracted), canny_threshold_BGSubstracted)
        binary_threshold_BGSubstracted = 70
        early_threshold_binary_BGSubstracted = (early_threshold_BGSubstracted > binary_threshold_BGSubstracted)
        thresholded_binary_AlignedBGSubstracted = (thresholded_BGSubstracted > binary_threshold_BGSubstracted).float()
        # imshow_torch(thresholded_binary_AlignedBGSubstracted)

        ### Perform Binary Thresholding On BG-Substracted Image: ###
        binary_threshold_BG_substraction = 4
        current_aligned_BG_substracted_binary = (input_tensor_BG_substracted > binary_threshold_BG_substraction).float()
        # imshow_torch(current_aligned_BG_substracted_binary)

        ### Perform Dilation On Canny Edge: ###
        thresholded_binary_BG_dilated = kornia.morphology.dilation(thresholded_binary_BG, torch.ones(3,3).cuda())

        ### Perform Masking Of The BG Substracted Image and Binary By The Canny Edges: ###
        current_aligned_BGSubstracted_MaskedByBGCanny = current_aligned_BGSubstracted * (1-thresholded_binary_BG_dilated)
        current_aligned_BGSubstractedBinary_MaskedByBGCanny = current_aligned_BG_substracted_binary * (1-thresholded_binary_BG_dilated)

        ### Save Results To Lists: ###
        BGCanny_list.append(thresholded_binary_BG.cpu())
        Aligned_BGSubstracted_list.append(input_tensor_BG_substracted.cpu())
        Aligned_BGSubstracted_Canny_list.append(thresholded_binary_AlignedBGSubstracted.cpu())
        Aligned_BGSubstracted_Binary_list.append(current_aligned_BG_substracted_binary.cpu())
        Aligned_BGSubstracted_Binary_MaskedByBGCanny_list.append(current_aligned_BGSubstractedBinary_MaskedByBGCanny.cpu())

        ### Save Videos: ###
        #(1). original vs. bg:
        image_to_write = turn_to_video_frame(torch.cat((current_original_frame, input_tensor_median), -1))
        video_object_Original_Vs_BG.write(image_to_write)
        #(2). aligned vs. bg:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, input_tensor_median), -1))
        video_object_Aligned_Vs_BG.write(image_to_write)
        #(3). aligned vs. bg canny:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, thresholded_binary_BG*255), -1))
        video_object_Aligned_Vs_BGCanny.write(image_to_write)
        #(4). aligned vs. bg canny dilated:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, thresholded_binary_BG_dilated * 255), -1))
        video_object_Aligned_Vs_BGCanny_Dilated.write(image_to_write)
        #(5). aligned vs. bg substracted aligned:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, input_tensor_BG_substracted*10), -1))
        video_object_Aligned_Vs_BGSubstractedAligned.write(image_to_write)
        #(6). aligned vs. bg substracted aligned masked by bg-canny:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, current_aligned_BGSubstracted_MaskedByBGCanny * 10), -1))
        video_object_Aligned_Vs_BGSubstractedAlignedMaskedByBGCanny.write(image_to_write)
        #(7). aligned vs. bg substracted aligned canny:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, thresholded_binary_AlignedBGSubstracted), -1))
        video_object_Aligned_Vs_BGSubstractedAlignedCanny.write(image_to_write)
        #(8). aligned vs. bg substracted aligned binary:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, current_aligned_BG_substracted_binary * 255), -1))
        video_object_Aligned_Vs_BGSubstractedGradientsBinary.write(image_to_write)
        # (8). aligned vs. bg substracted aligned binary masked by bg-canny:
        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, current_aligned_BGSubstractedBinary_MaskedByBGCanny * 255), -1))
        video_object_Aligned_Vs_BGSubstractedGradientsBinaryMaskedByBGCanny.write(image_to_write)

        # ### Perform Morphology On Binary Images (maybe on images themselves or gradients themselves?!?!): ###
        # #(1). Dilation Layer Custom:
        # # dilation_layer = Dilation2d(1, 1, kernel_size=3, soft_max=False, beta=20).cuda()
        # # thresholded_binary_dilated = dilation_layer.forward(thresholded_binary)
        # #(2). Kornia:
        # thresholded_binary_dilated = kornia.morphology.dilation(thresholded_binary, torch.ones(3,3).cuda())
        # thresholded_binary_eroded = kornia.morphology.erosion(thresholded_binary, torch.ones(3,3).cuda())
        # thresholded_binary_opening = kornia.morphology.opening(thresholded_binary, torch.ones(3,3).cuda())
        # thresholded_binary_closing = kornia.morphology.closing(thresholded_binary, torch.ones(3,3).cuda())
        # thresholded_binary_gradient = kornia.morphology.gradient(thresholded_binary, torch.ones(3,3).cuda())
        # thresholded_binary_top_hat = kornia.morphology.top_hat(thresholded_binary, torch.ones(3,3).cuda())
        # thresholded_binary_bottom_hat = kornia.morphology.bottom_hat(thresholded_binary, torch.ones(3,3).cuda())
        # # imshow_torch(thresholded_binary_dilated, title_str='kornia dilation')
        # # imshow_torch(thresholded_binary_eroded, title_str='kornia erosion')
        # # imshow_torch(thresholded_binary_opening, title_str='kornia opening')
        # # imshow_torch(thresholded_binary_closing, title_str='kornia closing')
        # # imshow_torch(thresholded_binary_gradient, title_str='kornia gradient')
        # # imshow_torch(thresholded_binary_top_hat, title_str='kornia top hat')
        # # imshow_torch(thresholded_binary_bottom_hat, title_str='kornia bottom hat')
        # #(3). Scipy:
        # thresholded_binary_numpy = thresholded_binary.cpu()[0,0].numpy()
        # thresholded_binary_dilated_numpy = torch.Tensor(skimage.morphology.dilation(thresholded_binary_numpy)).cuda()
        # thresholded_binary_eroded_numpy = torch.Tensor(skimage.morphology.erosion(thresholded_binary_numpy)).cuda()
        # thresholded_binary_area_closing_numpy = torch.Tensor(skimage.morphology.area_closing(thresholded_binary_numpy)).cuda()
        # thresholded_binary_area_opening_numpy = torch.Tensor(skimage.morphology.area_opening(thresholded_binary_numpy)).cuda()
        # thresholded_binary_binary_closing_numpy = torch.Tensor(skimage.morphology.binary_closing(thresholded_binary_numpy)).cuda()
        # thresholded_binary_binary_erosion_numpy = torch.Tensor(skimage.morphology.binary_erosion(thresholded_binary_numpy)).cuda()
        # thresholded_binary_binary_opening_numpy = torch.Tensor(skimage.morphology.binary_opening(thresholded_binary_numpy)).cuda()
        # thresholded_binary_convex_hull_image_numpy = torch.Tensor(skimage.morphology.convex_hull_image(thresholded_binary_numpy)).cuda()
        # thresholded_binary_convex_hull_object_numpy = torch.Tensor(skimage.morphology.convex_hull_object(thresholded_binary_numpy)).cuda()
        # # imshow_torch(thresholded_binary_dilated_numpy, title_str='scipy dilation')
        # # imshow_torch(thresholded_binary_eroded_numpy, title_str='scipy erosion')
        # # imshow_torch(thresholded_binary_area_closing_numpy, title_str='scipy area closing')
        # # imshow_torch(thresholded_binary_area_opening_numpy, title_str='scipy area opening')
        # # imshow_torch(thresholded_binary_binary_closing_numpy, title_str='scipy binary closing')
        # # imshow_torch(thresholded_binary_binary_erosion_numpy, title_str='scipy binary erosion')
        # # imshow_torch(thresholded_binary_binary_opening_numpy, title_str='scipy binary opening')
        # # imshow_torch(thresholded_binary_convex_hull_image_numpy, title_str='scipy convex hull image')
        # # imshow_torch(thresholded_binary_convex_hull_object_numpy, title_str='scipy convex hull object')

        ### Show Some Results: ###
        # imshow_torch(input_tensor_median/255, title_str='aligned median')
        # imshow_torch(abs(input_tensor[0] - input_tensor[1]))
        # imshow_torch(early_threshold, title_str='early thresholded')
        # imshow_torch(thresholded, title_str='thresholded')
        # imshow_torch(grad_mag, title_str='grad magnitude')
###############################################################################################################################################



###############################################################################################################################################
### Loop over binary skelatons of BG substracted frames and output their median: ###
number_of_sub_frames = 5
for frame_index, current_frame in enumerate(aligned_frames_list):
    print(frame_index)
    if frame_index < len(aligned_frames_list) - number_of_sub_frames:
        ### Get Current Frames: ###
        current_original_frame = original_frames_list[frame_index+1].cuda()
        current_aligned_frame = aligned_frames_list[frame_index+1].cuda()
        current_BGCanny = BGCanny_list[frame_index+1].cuda()
        current_BGSubstracted = Aligned_BGSubstracted_list[frame_index+1].cuda()
        current_BGCanny = Aligned_BGSubstracted_Canny_list[frame_index+1].cuda()
        current_BGBinary = Aligned_BGSubstracted_Binary_list[frame_index+1].cuda()
        current_BGBinary_MaskedByBGCanny = Aligned_BGSubstracted_Binary_MaskedByBGCanny_list[frame_index+1].cuda()

        ### Get Current Temporal Context Needed For Estimations: ###
        current_sub_list = []
        for i in np.arange(number_of_sub_frames):
            current_sub_list.append(BGCanny_list[frame_index + i + 1].cuda())
        BGCanny_frames = torch.cat(current_sub_list)
        BGCanny_frames_median = torch.median(BGCanny_frames, 0)[0].unsqueeze(0)

        current_sub_list = []
        for i in np.arange(number_of_sub_frames):
            current_sub_list.append(Aligned_BGSubstracted_Binary_MaskedByBGCanny_list[frame_index + i + 1].cuda())
        aligned_BGSubstracted_Binary_MaskedByBGCanny = torch.cat(current_sub_list)
        aligned_BGSubstracted_Binary_MaskedByBGCanny_median = torch.median(aligned_BGSubstracted_Binary_MaskedByBGCanny, 0)[0].unsqueeze(0)

        image_to_write = turn_to_video_frame(torch.cat((current_aligned_frame, aligned_BGSubstracted_Binary_MaskedByBGCanny_median * 255), -1))
        video_object_Aligned_Vs_BGSubstractedGradientsBinaryMaskedByBGCanny_median.write(image_to_write)
###############################################################################################################################################









# from RapidBase.import_all import *
import torch
import torchvision.transforms.functional

from RapidBase.Anvil.alignments_layers import minSAD_transforms_layer
from RapidBase.TrainingCore.datasets import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *
from RapidBase.Anvil.alignments_layers import *
from RapidBase.Anvil.alignments import classic_circular_cc_shifts_calc, align_to_center_frame_circular_cc
import kornia
from RapidBase.Anvil.Dudy import *
from RapidBase.Anvil import affine_warp_matrix
from RapidBase.Anvil.alignments import align_to_reference_frame_circular_cc

#TODO: make a "from Anvil.import_all import *" statement possible

### Paths: ###
##############
### Movies Paths: ###
original_frames_folder = '/home/mafat/Datasets/pngsh/pngs/2'

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
number_of_images = 200
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
### Take Care Of Crop Sizes - very important when doing fitting/stabilization: ###
original_frame_current = read_image_general(image_filenames[0])
H,W,C = original_frame_current.shape
#(1). First Stage:
initial_crop_size_first_stage = (H, W)
final_crop_size_for_movie_first_stage = (H, W)
# max_estimated_shift = max(max(abs(np.array(estimated_shifts_x_list))), max(abs(np.array(estimated_shifts_y_list)))).data  #the real world minimal result in the field
#(2). Second Stage:
max_estimated_shift = 50
initial_crop_size_second_stage = (int(initial_crop_size_first_stage[0]-max_estimated_shift), int(initial_crop_size_first_stage[1]-max_estimated_shift)) #after cross correlation alignment
final_crop_size_for_movie_second_stage = initial_crop_size_second_stage
#######################################################################################################################################


#######################################################################################################################################
### Get Video Recorder: ###
experiment_name = 'first_try'
save_path = os.path.join(results_path, experiment_name)
save_path1 = os.path.join(save_path, 'original_frames')
save_path2 = os.path.join(save_path, 'clean_frame_estimate')
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name_Original_Vs_Aligned = os.path.join(save_path, 'Results', 'Original_Vs_Aligned.avi')
video_name_Original_Vs_Aligned_second_stage = os.path.join(save_path, 'Results', 'Original_Vs_Aligned_SecondStage.avi')
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


video_object_Original_Vs_Aligned = cv2.VideoWriter(video_name_Original_Vs_Aligned, 0, 10, (final_crop_size_for_movie_first_stage[1]*2, final_crop_size_for_movie_first_stage[0]))
video_object_Original_Vs_Aligned_second_stage = cv2.VideoWriter(video_name_Original_Vs_Aligned_second_stage, 0, 10, (final_crop_size_for_movie_second_stage[1]*2, final_crop_size_for_movie_second_stage[0]))
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
### Initialize Lists: ###
clean_frames_list = []
original_frames_list = []
aligned_frames_list = []
perfect_average_frames_list = []
simple_average_frames_list = []
seed_frames_list = []

real_shifts_list = []
inferred_shifts_list = []
#######################################################################################################################################


#######################################################################################################################################
### Get scale factor for viewing purposes: ###
original_frame_initial = read_image_general(image_filenames[image_index_to_start_from])
scale_factor = (210/original_frame_initial.max()).item()
#######################################################################################################################################



# ###############################################################################################################################################
# ### Stabilize initial frame for BG estimation: ###
# # video_create_movie_from_images_in_folder(original_frames_folder)
# flag_plot = False
# image_concat_array = read_images_from_filenames_list(image_filenames,
#                                 flag_return_numpy_or_list='numpy',
#                                 crop_size=np.inf,
#                                 max_number_of_images=100,
#                                 flag_how_to_concat='C',
#                                 crop_style='center',
#                                 flag_return_torch=True,
#                                 transform=None,
#                                 first_frame_index=image_index_to_start_from)
# input_tensor = image_concat_array.unsqueeze(1).cuda()
# input_tensor = input_tensor.unsqueeze(0)  #to get 5D tensor until Anvil gets its shit together
# B,T,C,H,W = input_tensor.shape
#
#
# #(*). Show raw video:
# if flag_plot:
#     imshow_torch_video(input_tensor, FPS=50, video_title='raw video', frame_stride=5)
#
# ### Cross Correlation initial correction: ###
# #(1). compare to reference (center) frame:
# shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(input_tensor, input_tensor[:,T//2:T//2+1], None, None)
# # # # #(2). compare pair-wise:
# # shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(input_tensor[:,1:], input_tensor[:,0:-1], None, None)
# # shift_H = shift_H.cumsum(0)
# # shift_W = shift_W.cumsum(0)
# # shift_H = torch.cat([torch.tensor([0]).to(shift_H.device), shift_H], 0)
# # shift_W = torch.cat([torch.tensor([0]).to(shift_W.device), shift_W], 0)
# #(3). get max shifts:
# max_shift_H = shift_H.max()
# max_shift_W = shift_W.max()
#
#
# ### Shift input tensor to make aligned tensor: ###
# aligned_tensor = shift_matrix_subpixel(input_tensor, -shift_H, -shift_W, matrix_FFT=None, warp_method='fft')
# aligned_tensor = crop_torch_batch(aligned_tensor, [350,950])
# aligned_tensor = aligned_tensor.to('cuda')
# aligned_tensor = aligned_tensor.squeeze(0).squeeze(0).unsqueeze(1)
# if flag_plot:
#     imshow_torch_video(aligned_tensor, FPS=50)
#
# ### Crop both original and aligned to be the same size and correct for size disparities (TODO: will be corrected to be seemless): ###
# aligned_tensor = crop_torch_batch(aligned_tensor, [350,950])
# input_tensor = crop_torch_batch(input_tensor, [350,950])
# input_tensor = input_tensor.squeeze(0)
# concat_tensor = torch.cat([input_tensor, aligned_tensor], dim=-1)
#
# ### Present results after cross correlation: ###
# if flag_plot:
#     imshow_torch_video(input_tensor, FPS=50)
#     imshow_torch_video(aligned_tensor, FPS=50)
#     imshow_torch_video(concat_tensor, FPS=50)
#     imshow_torch_video(input_tensor-input_tensor[T//2:T//2+1], FPS=50)
#     imshow_torch_video(aligned_tensor-aligned_tensor[T//2:T//2+1], FPS=50)
#
#
# ### Min-SAD correction: ###
# from RapidBase.Anvil.transforms import affine_warp_matrix
# T, C, H, W = input_tensor.shape
# minSAD = minSAD_transforms_layer_elisheva(1, T-1, H, W)
#
# # #(1). Pair-Wise Comparison:
# # shifts_y, shifts_x, rotational_shifts, scale_shifts, min_sad_matrix = minSAD(
# #     matrix=aligned_tensor[1:],
# #     reference_matrix=aligned_tensor[0:-1],
# #     shift_h_vec=[0],
# #     shift_w_vec=[0],
# #     rotation_vec=[-0.3,-0.2,-0.1, 0, 0.1, 0.2,0.3],
# #     scale_vec=[1],
# #     warp_method='bicubic')
# # #(*). perform cumsum on min-SAD outputs because we checked for pairwise shifts and we want to shift all matrices to be aligned: ###
# # shifts_x_cumsum = shifts_x.cumsum(0)
# # shifts_y_cumsum = shifts_y.cumsum(0)
# # rotational_shifts_cumsum = rotational_shifts.cumsum(0)
#
# #(2). Reference Comparison:
# shifts_y, shifts_x, rotational_shifts, scale_shifts, min_sad_matrix = minSAD(
#     matrix=aligned_tensor,
#     reference_matrix=aligned_tensor[T//2:T//2+1],
#     shift_h_vec=[-1,0,1],
#     shift_w_vec=[-1,0,1],
#     rotation_vec=[-0.3,-0.2,-0.1, 0, 0.1, 0.2,0.3],
#     scale_vec=[1],
#     warp_method='bicubic')
#
# ### Make all shift vectors proper vector size: ###
# shifts_y = shifts_y*torch.ones_like(rotational_shifts)
# shifts_x = shifts_x*torch.ones_like(rotational_shifts)
# scale_shifts = scale_shifts*torch.ones_like(rotational_shifts)
#
# # ### Warp according to Pair-Wise min-SAD results: ###
# # warped_tensor = affine_warp_matrix(aligned_tensor[1:],
# #                         -shifts_y_cumsum,
# #                         -shifts_x_cumsum,
# #                         -rotational_shifts_cumsum*np.pi/180,
# #                         scale_shifts,
# #                         warp_method='bicubic')
#
# ### Warp according to Pair-Wise min-SAD results: ###
# warped_tensor = affine_warp_matrix(aligned_tensor,
#                         -shifts_y,
#                         -shifts_x,
#                         -rotational_shifts*np.pi/180,
#                         scale_shifts,
#                         warp_method='bicubic')
# warped_tensor = warped_tensor.squeeze(0).unsqueeze(1)
# warped_tensor = crop_torch_batch(warped_tensor, [350,950])
# warped_tensor = warped_tensor.cuda()
# concat_tensor_min_sad = torch.cat([aligned_tensor, warped_tensor], dim=-1)
#
# ### Plot: ###
# if flag_plot:
#     plot_torch(rotational_shifts); plt.show()
#     imshow_torch_video(warped_tensor, FPS=50)
#     imshow_torch_video(concat_tensor_min_sad, FPS=50)
#     imshow_torch_video(warped_tensor - warped_tensor[T//2:T//2+1], FPS=50)
#
#
# # ### Get min-SAD Bias term: ###
# # shifts_y_bias, shifts_x_bias, rotational_shifts_bias, scale_shifts_bias, min_sad_matrix_bias = minSAD(
# #     matrix=aligned_tensor[1:],
# #     reference_matrix=aligned_tensor[1:],
# #     shift_h_vec=[0],
# #     shift_w_vec=[0],
# #     rotation_vec=[-0.3,-0.2,-0.1, 0, 0.1, 0.2,0.3],
# #     scale_vec=[1],
# #     warp_method='bicubic')
# # bla = rotational_shifts - rotational_shifts_bias
# # plot_torch(bla); plt.show()
# # plot_torch(bla.cumsum(0)); plt.show()
#
# ### Get BG: ###
# input_tensor_BG = warped_tensor.median(0)[0].unsqueeze(0)
# BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
# torch.save(input_tensor_BG, os.path.join(BG_image_folder, 'palantir_BG.pt'))
#
# ### Free up GPU memory: ###
# del input_tensor
# del aligned_tensor
# del warped_tensor
# del concat_tensor
# del concat_tensor_min_sad
# del minSAD
# del min_sad_matrix
# del CC_tensor
# del image_concat_array
# torch.cuda.empty_cache()
# ###############################################################################################################################################



###############################################################################################################################################
### Stabilize Entire Movie: ###
BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
input_tensor_BG = torch.load(os.path.join(BG_image_folder, 'palantir_BG.pt'))

### Initialize Stuff: ###
flag_plot = False
number_of_frames_per_batch = 20
number_of_images_total = len(image_filenames)
number_of_batches = number_of_images_total // number_of_frames_per_batch
aligned_tensors_list = []
aligned_tensors_final_list = []

### Initialize Video: ####
H_final, W_final = [320, 880]
video_name_Original_Vs_Aligned = os.path.join(original_frames_folder, 'Original_Vs_Aligned.avi')
video_object_Original_Vs_Aligned = cv2.VideoWriter(video_name_Original_Vs_Aligned, 0, 10, (W_final*2, H_final))

### Loop Over Frames: ###
frame_counter = 0
for frame_index in np.arange(number_of_batches):
    print('FRAME COUNTER: ' + str(frame_counter))
    image_concat_array = read_images_from_filenames_list(image_filenames,
                                    flag_return_numpy_or_list='numpy',
                                    crop_size=np.inf,
                                    max_number_of_images=number_of_frames_per_batch,
                                    flag_how_to_concat='C',
                                    crop_style='center',
                                    flag_return_torch=True,
                                    transform=None,
                                    first_frame_index=frame_index*number_of_frames_per_batch)
    input_tensor = image_concat_array.unsqueeze(1).cuda()
    # input_tensor = input_tensor.unsqueeze(0)  #to get 5D tensor until Anvil gets its shit together
    # B,T,C,H,W = input_tensor.shape
    T,C,H,W = input_tensor.shape

    ### Center Crop Input Tensor To BG Size: ###
    H_BG, W_BG = input_tensor_BG.shape[-2:]
    input_tensor = crop_torch_batch(input_tensor, (H_BG, W_BG))

    #(*). Show raw video:
    if flag_plot:
        imshow_torch_video(input_tensor, FPS=50, video_title='raw video', frame_stride=5)

    ### Align Inter-Batch: ###
    # input_tensor, aligned_tensor, aligned_tensor_final = Align_Palantir_CC_Then_Classic(input_tensor[0:20], input_tensor[T//2:T//2+1])

    ### Align Batch To BG: ###
    input_tensor, aligned_tensor, aligned_tensor_final = Align_Palantir_CC_Then_Classic_Then_minSAD(input_tensor, input_tensor_BG)
    # imshow_torch_video(torch.cat([input_tensor, aligned_tensor_final],-1))

    ### Crop All To Same Size: ###
    input_tensor = crop_tensor(input_tensor, (H_final, W_final))
    aligned_tensor = crop_tensor(aligned_tensor, (H_final, W_final))
    aligned_tensor_final = crop_tensor(aligned_tensor_final, (H_final, W_final))
    concat_torch = torch.cat([input_tensor.cpu(), aligned_tensor_final.cpu()], -1)

    ### Make Frames Ready For Video: ###
    input_tensor = torch_to_numpy_video_ready(input_tensor)
    aligned_tensor = torch_to_numpy_video_ready(aligned_tensor)
    aligned_tensor_final = torch_to_numpy_video_ready(aligned_tensor_final)
    concat_torch = torch_to_numpy_video_ready(concat_torch)

    ### Save Results: ###
    original_folder = '/home/mafat/Datasets/pngsh/pngs/2'
    CC_aligned_folder = '/home/mafat/Datasets/pngsh/aligned_pngs/2'
    final_aligned_folder = '/home/mafat/Datasets/pngsh/final_aligned_pngs/2'
    path_create_path_if_none_exists(CC_aligned_folder)
    path_create_path_if_none_exists(final_aligned_folder)

    ### Save Results To Images: ###
    inter_batch_T = aligned_tensor_final.shape[0]
    for inter_frame_index in np.arange(inter_batch_T):
        current_filename = string_rjust(frame_counter, 5) + '.png'
        save_image_numpy(CC_aligned_folder, current_filename, aligned_tensor[inter_frame_index], flag_scale=False)
        save_image_numpy(final_aligned_folder, current_filename, aligned_tensor_final[inter_frame_index], flag_scale=False)
        frame_counter += 1


    ### Save Results To Video: ###
    inter_batch_T = aligned_tensor_final.shape[0]
    for inter_frame_index in np.arange(inter_batch_T):
        video_object_Original_Vs_Aligned.write(concat_torch[inter_frame_index])

video_object_Original_Vs_Aligned.release()
###############################################################################################################################################




####################################################
### Show Outliers: ###
original_folder = '/home/mafat/Datasets/pngsh/pngs/2'
CC_aligned_folder = '/home/mafat/Datasets/pngsh/aligned_pngs/2'
final_aligned_folder = '/home/mafat/Datasets/pngsh/final_aligned_pngs/2'

### Get BG: ###
BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
input_tensor_BG = torch.load(os.path.join(BG_image_folder, 'palantir_BG.pt'))

### Initialize Video: ####
H_final, W_final = [320, 880]
video_name_Diff = os.path.join(original_frames_folder, 'Diff.avi')
video_name_outliers_before_canny = os.path.join(original_frames_folder, 'Outliers_Before_Canny.avi')
video_name_outliers_after_canny = os.path.join(original_frames_folder, 'Outliers_After_Canny.avi')
video_name_aligned_vs_outliers_before_canny = os.path.join(original_frames_folder, 'Aligned_VS_Outliers_Before_Canny.avi')
video_name_aligned_vs_outliers_after_canny = os.path.join(original_frames_folder, 'Aligned_VS_Outliers_After_Canny.avi')

video_object_Diff = cv2.VideoWriter(video_name_Diff, 0, 10, (W_final*1, H_final))
video_object_Outliers_before_canny = cv2.VideoWriter(video_name_outliers_before_canny, 0, 10, (W_final*1, H_final))
video_object_Outliers_after_canny = cv2.VideoWriter(video_name_outliers_after_canny, 0, 10, (W_final*1, H_final))
video_object_aligned_vs_outliers_before_canny = cv2.VideoWriter(video_name_aligned_vs_outliers_before_canny, 0, 10, (W_final*2, H_final))
video_object_aligned_vs_outliers_after_canny = cv2.VideoWriter(video_name_aligned_vs_outliers_after_canny, 0, 10, (W_final*2, H_final))

### Initialize Stuff: ###
flag_plot = False
number_of_frames_per_batch = 20
number_of_images_total = len(image_filenames)
number_of_batches = number_of_images_total // number_of_frames_per_batch
aligned_tensors_list = []
aligned_tensors_final_list = []

### Get Reference Tensor: ###
image_filenames = read_image_filenames_from_folder(path=final_aligned_folder,
                                         number_of_images=np.inf,
                                         allowed_extentions=['.png'],
                                         flag_recursive=True,
                                         string_pattern_to_search='*')
image_concat_array = read_images_from_filenames_list(image_filenames,
                                        flag_return_numpy_or_list='numpy',
                                        crop_size=np.inf,
                                        max_number_of_images=1,
                                        flag_how_to_concat='C',
                                        crop_style='center',
                                        flag_return_torch=True,
                                        transform=None,
                                        first_frame_index=number_of_images_total//2)
reference_tensor = image_concat_array.unsqueeze(0).cuda()
reference_tensor = RGB2BW(reference_tensor)
reference_tensor = crop_tensor(reference_tensor, (H_final, W_final))

### Get Canny Edge Detection From Reference: ###
input_tensor_BG = crop_tensor(input_tensor_BG, (H_final,W_final))
BG_canny_magnitude, BG_canny_threshold = kornia.filters.canny(input_tensor_BG, 0.05*255, 0.2*255)
dilation_kernel = torch.zeros(3,3).to(BG_canny_threshold.device)
dilation_kernel[1,0] = 1; dilation_kernel[0,1] = 1; dilation_kernel[1,-1] = 1; dilation_kernel[-1,1] = 1;
BG_canny_threshold_dilation = kornia.morphology.dilation(BG_canny_threshold, dilation_kernel)
BG_canny_threshold_opening = kornia.morphology.opening(BG_canny_threshold, dilation_kernel)
BG_canny_threshold_closing = kornia.morphology.closing(BG_canny_threshold, dilation_kernel)
BG_canny_threshold = BG_canny_threshold_closing
# imshow_torch(torch.cat([BG_canny_threshold, BG_canny_threshold_dilation],-1))
# imshow_torch(torch.cat([BG_canny_threshold, BG_canny_threshold_closing],-1))
# imshow_torch(torch.cat([input_tensor_BG,BG_canny_threshold*255],-1))
percent_canny = BG_canny_threshold.sum() / (H_final*W_final)
percent_canny_dilated = BG_canny_threshold_dilation.sum() / (H_final*W_final)
percent_canny_closing = BG_canny_threshold_closing.sum() / (H_final*W_final)


### Loop Over Frames: ###
frame_counter = 0
for frame_index in np.arange(number_of_batches):
    print('Batch COUNTER: ' + str(frame_index))

    #(1). Get Original Images:
    image_filenames = read_image_filenames_from_folder(path=original_folder,
                                         number_of_images=np.inf,
                                         allowed_extentions=['.png'],
                                         flag_recursive=True,
                                         string_pattern_to_search='*')
    image_concat_array = read_images_from_filenames_list(image_filenames,
                                        flag_return_numpy_or_list='numpy',
                                        crop_size=np.inf,
                                        max_number_of_images=number_of_frames_per_batch,
                                        flag_how_to_concat='C',
                                        crop_style='center',
                                        flag_return_torch=True,
                                        transform=None,
                                        first_frame_index=frame_index*number_of_frames_per_batch)
    input_tensor = image_concat_array.unsqueeze(1).cuda()
    # input_tensor = input_tensor.unsqueeze(0)  #to get 5D tensor until Anvil gets its shit together
    # B,T,C,H,W = input_tensor.shape
    T,C,H,W = input_tensor.shape

    #(2). Get Final Aligned Images:
    final_image_filenames = read_image_filenames_from_folder(path=final_aligned_folder,
                                         number_of_images=np.inf,
                                         allowed_extentions=['.png'],
                                         flag_recursive=True,
                                         string_pattern_to_search='*')
    image_concat_array = read_images_from_filenames_list(final_image_filenames,
                                        flag_return_numpy_or_list='numpy',
                                        crop_size=np.inf,
                                        max_number_of_images=number_of_frames_per_batch,
                                        flag_how_to_concat='C',
                                        crop_style='center',
                                        flag_return_torch=True,
                                        transform=None, flag_to_BW=True,
                                        first_frame_index=frame_index*number_of_frames_per_batch)
    aligned_tensor = image_concat_array.unsqueeze(1).cuda()
    # aligned_tensor = aligned_tensor.unsqueeze(0)  #to get 5D tensor until Anvil gets its shit together
    # B,T,C,H,W = aligned_tensor.shape
    T,C,H,W = aligned_tensor.shape

    ### Get relevant tensors and crop themL: ###
    input_tensor = crop_tensor(input_tensor, (H_final,W_final))
    aligned_tensor = crop_tensor(aligned_tensor, (H_final,W_final))
    BG_canny_threshold = crop_tensor(BG_canny_threshold, (H_final,W_final))
    diff_tensor = (aligned_tensor - input_tensor_BG).abs()
    diff_tensor = crop_tensor(diff_tensor, (H_final, W_final))

    ### Get Outliers: ###
    if frame_index == 0:
        H_quantile, W_quantile = (250,750)
        outlier_threshold = torch.quantile(crop_tensor(diff_tensor[0], (H_quantile,W_quantile)), 0.93)
    outlier_tensor_before_canny = (diff_tensor > outlier_threshold).float()
    outlier_tensor_after_canny = (outlier_tensor_before_canny - BG_canny_threshold).clamp(0)

    ### Get Rid Of BG Edges Outliers: ###
    aligned_vs_outlier_tensor_before_canny = torch.cat([aligned_tensor, outlier_tensor_before_canny*255], -1)
    aligned_vs_outlier_tensor_after_canny = torch.cat([aligned_tensor, outlier_tensor_after_canny*255], -1)


    ### Turn Relevant Frames To Video Ready: ###
    frame_to_write_diff = torch_to_numpy_video_ready(diff_tensor)
    frame_to_write_outlier_before_canny = torch_to_numpy_video_ready(outlier_tensor_before_canny)
    frame_to_write_outlier_after_canny = torch_to_numpy_video_ready(outlier_tensor_after_canny)
    frame_to_write_aligned_vs_outlier_before_canny = torch_to_numpy_video_ready(aligned_vs_outlier_tensor_before_canny)
    frame_to_write_aligned_vs_outlier_after_canny = torch_to_numpy_video_ready(aligned_vs_outlier_tensor_after_canny)

    ### Write Videos: ###
    for inter_batch_index in np.arange(T):
        video_object_Diff.write(frame_to_write_diff[inter_batch_index])
        video_object_Outliers_before_canny.write(frame_to_write_outlier_before_canny[inter_batch_index])
        video_object_Outliers_after_canny.write(frame_to_write_outlier_after_canny[inter_batch_index])
        video_object_aligned_vs_outliers_before_canny.write(frame_to_write_aligned_vs_outlier_before_canny[inter_batch_index])
        video_object_aligned_vs_outliers_after_canny.write(frame_to_write_aligned_vs_outlier_after_canny[inter_batch_index])

video_object_Diff.release()
video_object_Outliers_before_canny.release()
video_object_Outliers_after_canny.release()
video_object_aligned_vs_outliers_before_canny.release()
video_object_aligned_vs_outliers_after_canny.release()
####################################################









### Get CURRENT-BATCH "BG" using median: ###
input_tensor_BGS = input_tensor - input_tensor[T//2:T//2+1]
outliers_without_alignment = (input_tensor_BGS).abs() > 20
outliers_without_alignment = outliers_without_alignment.float()
warped_tensor_median = warped_tensor.median(0)[0].unsqueeze(0)
warped_tensor_BGS = warped_tensor - warped_tensor_median
warped_tensor_BGS_outliers = (warped_tensor_BGS.abs() > 20).float()
if flag_plot:
    imshow_torch_video(warped_tensor_BGS, FPS=50, frame_stride=5)
    imshow_torch_video(warped_tensor_BGS_outliers, FPS=50, frame_stride=5)

### Perform Canny Edge Detection On Median Image: ###  #TODO: make the threshold's automatic
canny_edge_detection_layer = canny_edge_detection(10, True)
canny_threshold_BG = 10
blurred_img_BG, grad_mag_BG, grad_orientation_BG, thin_edges_BG, thresholded_BG, early_threshold_BG = \
    canny_edge_detection_layer(BW2RGB(warped_tensor_median), canny_threshold_BG)
binary_threshold_BG = 700  #TODO: i picked this out of my ass, automate
early_threshold_binary_BG = (early_threshold_BG > binary_threshold_BG)
thresholded_binary_BG = (thresholded_BG > binary_threshold_BG).float()

### Perform Dilation On Canny Edge: ###
thresholded_binary_BG_dilated = kornia.morphology.dilation(thresholded_binary_BG, torch.ones(3,3).cuda()).float()

### Reduce BG Edges from outlier detection: ###
# warped_tensor_BGS_outliers_final = warped_tensor_BGS_outliers - thresholded_binary_BG
warped_tensor_BGS_outliers_final = warped_tensor_BGS_outliers - thresholded_binary_BG_dilated
warped_tensor_BGS_outliers_final2 = convn_torch(warped_tensor_BGS_outliers_final.clamp(0), torch.ones(5)/5, 0)
if flag_plot:
    imshow_torch_video(warped_tensor_BGS_outliers_final.clamp(0), FPS=50, frame_stride=5)
    imshow_torch_video(outliers_without_alignment, FPS=50, frame_stride=5)
    imshow_torch_video(input_tensor_BGS.abs()*1, FPS=50, frame_stride=5)

    input_tensor_cropped_for_video = crop_torch_batch(outliers_without_alignment, tuple(torch.tensor(warped_tensor_BGS_outliers_final.shape[-2:]).cpu().numpy()))
    concat_tensor_for_video = torch.cat([input_tensor_cropped_for_video.cpu(), warped_tensor_BGS_outliers_final.cpu()], dim=-1)
    video_torch_array_to_video(warped_tensor_BGS_outliers_final.clamp(0), video_name=os.path.join('/home/mafat/PycharmProjects/TNR_Results', 'Palantir_interim3.avi'), FPS=10)
    video_torch_array_to_video(outliers_without_alignment.clamp(0), video_name=os.path.join('/home/mafat/PycharmProjects/TNR_Results', 'Palantir_initial_outliers.avi'), FPS=10)
    video_torch_array_to_video(concat_tensor_for_video.clamp(0), video_name=os.path.join('/home/mafat/PycharmProjects/TNR_Results', 'Palantir_concat_outliers.avi'), FPS=10)


### Delete stuff and free up GPU memory: ###
del warped_tensor_BGS_outliers
del warped_tensor_BGS_outliers_final
del warped_tensor_BGS
del min_sad_matrix
del minSAD
del concat_tensor_min_sad
del aligned_tensor
del input_tensor
torch.cuda.empty_cache()

### Align to BG tensor: ###
### Cross Correlation initial correction: ###
#(1). compare to reference (center) frame:
warped_tensor = warped_tensor.unsqueeze(0)
shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(warped_tensor, input_tensor_BG, None, None)
max_shift_H = shift_H.abs().max()
max_shift_W = shift_W.abs().max()
B,T,C,H,W = warped_tensor.shape
new_H = H - (max_shift_H.abs().int().cpu().numpy()*2 + 5)
new_W = W - (max_shift_W.abs().int().cpu().numpy()*2 + 5)
### Shift input tensor to make aligned tensor: ###
torch.cuda.empty_cache()
aligned_tensor = shift_matrix_subpixel(warped_tensor, -shift_H, -shift_W, matrix_FFT=None, warp_method='fft')
# aligned_tensor = crop_torch_batch(aligned_tensor, [new_H,new_W])
aligned_tensor = aligned_tensor.to('cuda')
aligned_tensor = aligned_tensor.squeeze(0).squeeze(0).unsqueeze(1)
### Crop: ###
# aligned_tensor = crop_torch_batch(aligned_tensor, (new_H, new_W))
# input_tensor_BG = crop_torch_batch(input_tensor_BG, (new_H, new_W))
aligned_tensor = aligned_tensor[:,:,0:H-max_shift_H.int(),0:W-max_shift_W.int()]
input_tensor_BG = input_tensor_BG[:,:,0:H-max_shift_H.int(),0:W-max_shift_W.int()]
if flag_plot:
    imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)

### Free up GPU memory: ###
del CC_tensor
del warped_tensor
torch.cuda.empty_cache()

#(2). Reference Comparison:
T, C, H, W = aligned_tensor.shape
minSAD = minSAD_transforms_layer_elisheva(1, T, H, W)
shifts_y, shifts_x, rotational_shifts, scale_shifts, min_sad_matrix = minSAD.forward(
    matrix=aligned_tensor,
    reference_matrix=aligned_tensor[T//2:T//2+1],
    shift_h_vec=[0,1,2,3],
    shift_w_vec=[0,1,2,3],
    rotation_vec=[-0.4,-0.2,0,0.2,0.4],
    scale_vec=[1],
    warp_method='bicubic')

### Make all shift vectors proper vector size: ###
shifts_y = shifts_y*torch.ones_like(rotational_shifts)
shifts_x = shifts_x*torch.ones_like(rotational_shifts)
scale_shifts = scale_shifts*torch.ones_like(rotational_shifts)

### Warp according to Pair-Wise min-SAD results: ###
warped_tensor = affine_warp_matrix(aligned_tensor,
                        -shifts_y,
                        -shifts_x,
                        -rotational_shifts*np.pi/180,
                        scale_shifts,
                        warp_method='bicubic')
warped_tensor = warped_tensor.squeeze(0).unsqueeze(1)
warped_tensor = crop_torch_batch(warped_tensor, [300,900])
aligned_tensor = crop_torch_batch(aligned_tensor, [300,900])
input_tensor_BG = crop_torch_batch(input_tensor_BG, [300,900])
warped_tensor = warped_tensor.cuda()
concat_tensor_min_sad = torch.cat([aligned_tensor, warped_tensor], dim=-1)
if flag_plot:
    imshow_torch_video(warped_tensor, FPS=50, frame_stride=5)

### Get Outliers from BG (as opposed to batch median, which is what we did beforehand): ###
### Get CURRENT-BATCH "BG" using median: ###
warped_tensor_BGS = warped_tensor - input_tensor_BG
warped_tensor_BGS_outliers = (warped_tensor_BGS > 20).float()
if flag_plot:
    imshow_torch_video(warped_tensor_BGS, FPS=50, frame_stride=5)
    imshow_torch_video(warped_tensor_BGS_outliers, FPS=50, frame_stride=5)

### Perform Canny Edge Detection On Median Image: ###  #TODO: make the threshold's automatic
canny_edge_detection_layer = canny_edge_detection(10, True)
canny_threshold_BG = 10
blurred_img_BG, grad_mag_BG, grad_orientation_BG, thin_edges_BG, thresholded_BG, early_threshold_BG = \
    canny_edge_detection_layer(BW2RGB(input_tensor_BG), canny_threshold_BG)
binary_threshold_BG = 700  #TODO: i picked this out of my ass, automate
early_threshold_binary_BG = (early_threshold_BG > binary_threshold_BG)
thresholded_binary_BG = (thresholded_BG > binary_threshold_BG).float()

### Perform Dilation On Canny Edge: ###
thresholded_binary_BG_dilated = kornia.morphology.dilation(thresholded_binary_BG, torch.ones(3,3).cuda())

### Reduce BG Edges from outlier detection: ###
warped_tensor_BGS_outliers_final = warped_tensor_BGS_outliers - thresholded_binary_BG
if flag_plot:
    imshow_torch_video(warped_tensor_BGS_outliers_final.clamp(0), FPS=50, frame_stride=5)

if flag_plot:
    imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)
    imshow_torch_video(concat_tensor_min_sad, FPS=50, frame_stride=5)
    imshow_torch_video(warped_tensor, FPS=50, frame_stride=5)


### Plot: ###
if flag_plot:
    warped_tensor_BGS_initial = warped_tensor - input_tensor_BG
    imshow_torch_video(warped_tensor_BGS_initial, FPS=50, frame_stride=5)

###############################################################################################################################################




### Initialize Lists: ###
estimated_shifts_x_list = []
estimated_shifts_y_list = []
estimated_angles_list = []

### Actually Loop Over The images and align them using cross correlation as first stage: ###
cumsum_shift_x = 0
cumsum_shift_y = 0
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + 1 + number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    # original_frame_previous = aligned_frames_list[0]
    # original_frame_previous = original_frames_list[0]
    #(1). Previous Image:
    # original_frame_previous = read_image_general(image_filenames[image_index_to_start_from+1])
    original_frame_previous = read_image_general(image_filenames[frame_index])
    original_frame_previous = torch.Tensor(original_frame_previous).cuda().permute([2, 0, 1]).unsqueeze(0)
    #(2). Current Image
    original_frame_current = read_image_general(image_filenames[frame_index+1])
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2,0,1]).unsqueeze(0)

    ### Crop Images: ###  #TODO: why crop at all?!?!
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, initial_crop_size_first_stage, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size_first_stage, crop_style='center')
    original_frames_list.append(original_frame_current_cropped.cpu())
    # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped)

    ### Cross Correlation Translation Correction: ###
    cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(original_frame_previous_cropped, original_frame_current_cropped)
    rotation_sub_pixel = torch.Tensor([0]).data
    cumsum_shift_x += shifts_sub_pixel[0]
    cumsum_shift_y += shifts_sub_pixel[1]
    # original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped, shifts_sub_pixel[1], shifts_sub_pixel[0])
    original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped, cumsum_shift_y, cumsum_shift_x)
    shifts_sub_pixel_initial_CC = shifts_sub_pixel
    shift_x_sub_pixel_initial_CC = shifts_sub_pixel[0]
    shift_y_sub_pixel_initial_CC = shifts_sub_pixel[1]
    # # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped_warped)
    # # imshow_torch(abs(original_frame_previous_cropped - original_frame_current_cropped_warped))
    # # imshow_torch(abs(original_frame_previous_cropped - original_frame_current_cropped))

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
bla = 1
###############################################################################################################################################






###############################################################################################################################################
#(*). REFERENCE-WISE COMPARISON, CROSS-CORRELATION ONLY

### Initialize Lists: ###
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

    ### Crop Images: ###  #TODO: why crop at all?!?!
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, initial_crop_size_first_stage, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size_first_stage, crop_style='center')
    original_frames_list.append(original_frame_current_cropped.cpu())
    # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped)

    ### Cross Correlation Translation Correction: ###
    cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(original_frame_previous_cropped, original_frame_current_cropped)
    rotation_sub_pixel = torch.Tensor([0]).data
    original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped, shifts_sub_pixel[1], shifts_sub_pixel[0])
    shifts_sub_pixel_initial_CC = shifts_sub_pixel
    shift_x_sub_pixel_initial_CC = shifts_sub_pixel[0]
    shift_y_sub_pixel_initial_CC = shifts_sub_pixel[1]
    # # imshow_torch(original_frame_previous_cropped); imshow_torch(original_frame_current_cropped_warped)
    # # imshow_torch(abs(original_frame_previous_cropped - original_frame_current_cropped_warped))
    # # imshow_torch(abs(original_frame_previous_cropped - original_frame_current_cropped))

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
### PAIR-WISE MIN-SAD: ###
final_aligned_frames_list = []
second_stage_aligned_frames_list = []
current_shift_x = 0
current_shift_y = 0
current_rotation = 0

for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    # original_frame_previous = aligned_frames_list[0]
    # original_frame_previous = original_frames_list[0]
    #(1). Previous Image:
    # original_frame_previous = read_image_general(image_filenames[image_index_to_start_from+1])
    original_frame_previous = read_image_general(image_filenames[frame_index])
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
    # final_aligned_frames_list.append(original_frame_current_cropped_warped.cpu())
    second_stage_aligned_frames_list.append(original_frame_current_cropped_warped.cpu())

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
import kornia
number_of_frames = len(aligned_frames_list)
number_of_sub_frames = 5


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


#######################################################################################################################################
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
input_tensor_median = torch.median(input_tensor, 0)[0].unsqueeze(0)
#######################################################################################################################################



#######################################################################################################################################
### Loop Over Aligned Frames & Get Stats and Stuff: ###
for frame_index, current_frame in enumerate(aligned_frames_list):
    print(frame_index)
    if frame_index < len(aligned_frames_list) - number_of_sub_frames:
        ### Get Current Temporal Context For Median Calculation: ###
        current_original_frame = original_frames_list[frame_index+1].cuda()
        current_aligned_frame = aligned_frames_list[frame_index+1].cuda()

        ### Estimate Running Median For BG Estimation: ###  #TODO: change this to constant BG instead of running estimation...which isn't smart
        current_sub_list = []
        for i in np.arange(number_of_sub_frames):
            current_sub_list.append(aligned_frames_list[frame_index + i + 1].cuda())
            # imshow_torch(aligned_frames_list[frame_index + i])
        input_tensor = torch.cat(current_sub_list)
        input_tensor_median = torch.median(input_tensor, 0)[0].unsqueeze(0)
        # imshow_torch(input_tensor); imshow_torch(input_tensor_median/255)

        ### Perform Canny Edge Detection On Median Image: ###  #TODO: make the threshold's automatic
        canny_threshold_BG = 10
        blurred_img_BG, grad_mag_BG, grad_orientation_BG, thin_edges_BG, thresholded_BG, early_threshold_BG = canny_edge_detection_layer(BW2RGB(input_tensor_median), canny_threshold_BG)
        binary_threshold_BG = 700
        early_threshold_binary_BG = (early_threshold_BG > binary_threshold_BG)
        thresholded_binary_BG = (thresholded_BG > binary_threshold_BG).float()

        ### Perform Canny Edge Detection On BG-Substracted Image: ###
        input_tensor_BG_substracted = (current_aligned_frame - input_tensor_median).abs()
        # imshow_torch(input_tensor_BG_substracted)

        ### Perform Canny Edge Detection On Median Image: ###
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









# from RapidBase.import_all import *
from RapidBase.TrainingCore.datasets import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *
from RDND_proper.models.RLSP.RLSP.pytorch.functions import shuffle_down, shuffle_up
from RDND_proper.models.RAFT.core.utils.flow_viz import flow_uv_to_colors


def put_optical_flow_arrows_on_image(image, optical_flow_image, threshold=2.0, skip_amount=30):
    # Don't affect original image
    image = image.copy()

    # Turn grayscale to rgb if needed
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=2)

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(np.meshgrid(range(optical_flow_image.shape[1]), range(optical_flow_image.shape[0])), 2)
    flow_end = (optical_flow_image[flow_start[:, :, 1], flow_start[:, :, 0], :1] * 3 + flow_start).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(flow_end - flow_start, axis=2)
    norm[norm < threshold] = 0

    # Draw all the nonzero values
    nz = np.nonzero(norm)
    for i in range(0, len(nz[0]), skip_amount):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(0, 255, 0),
                        thickness=1,
                        tipLength=.2)
    return image


### Paths: ###
##############
### Movies Paths: ###
original_frames_folder = '/home/mafat/DataSets/Aviram/2'
original_frames_folder = '/home/mafat/DataSets/stream_0/raw_image'


#######################################################################################################################################
### Neural Network Models: ###
#(*). For All:
import RapidBase.TrainingCore as core
Train_dict = EasyDict()
Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False

## Import Models: ###
#(1). FastDVDNet:
from RDND_proper.models.FastDVDNet.models import FastDVDnet, FastDVDnet_dudy, FastDVDnet_dudy_2
checkpoints_folder = r'/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/TEMP'
Train_dict.models_folder = checkpoints_folder
Train_dict.load_Network_filename = os.path.join(checkpoints_folder, r'FastDVDNet2_PreLU_Kaya_Video_New_1_TEST1_Step125400.tar')
# FastDVDNet_model = FastDVDnet_dudy(in_channels=3, num_input_frames=5)
FastDVDNet_model = FastDVDnet_dudy_2(in_channels=3, num_input_frames=5)
FastDVDNet_model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=FastDVDNet_model)
FastDVDNet_model = FastDVDNet_model.cuda()
FastDVDNet_model = FastDVDNet_model.eval()
# #(2). FFDnet:
# from RDND_proper.models.KAIR_FFDNet_and_more.models.network_ffdnet import FFDNet_dudy2, FFDNet_dudy, FFDNet_dudy_Recursive
# Train_dict.models_folder = r'/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/TEMP'
# Train_dict.load_Network_filename = r'/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/TEMP/ffdnet_gray_TEST1_Step0.tar'
# # Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_1/FFDNET_BW_1_TEST1_Step129400.tar'
# # Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_2/FFDNET_BW_2_TEST1_Step160800.tar'
# # Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_Gradient1/FFDNET_BW_Gradient1_TEST1_Step484600.tar'
# # Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_Residual_1/FFDNET_BW_Residual_1_TEST1_Step297200.tar'
# # Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_Residual_Shifts5/FFDNET_BW_Residual_Shifts5_TEST1_Step785200.tar'
# Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_Recursive_1/FFDNET_BW_Recursive_1_TEST1_Step9400.tar'
# # Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_Recursive_2/FFDNET_BW_Recursive_2_TEST1_Step7800.tar'
# # Train_dict.load_Network_filename = '/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/FFDNET_BW_Recursive2_1/FFDNET_BW_Recursive2_1_TEST1_Step19800.tar'
# # FFDNet_model = FFDNet_dudy(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
# # FFDNet_model = FFDNet_dudy2(in_nc=5, out_nc=1, nc=64, nb=15, act_mode='BL')
# FFDNet_model = FFDNet_dudy_Recursive(in_nc=5, out_nc=1, nc=64, nb=15, act_mode='BL')
# FFDNet_model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=FFDNet_model)
# FFDNet_model = FFDNet_model.cuda()
# # FFDNet_model = FFDNet_model.eval()

### IO dict: ###
IO_dict = EasyDict()
IO_dict.current_mean_gray_level_per_pixel = 2
IO_dict.electrons_per_gray_level = 14.53 * 1
IO_dict.photons_per_electron = 1 #TODO: i don't think this is the QE but whether we have a photo-multiplier or not....check this out
IO_dict.gray_levels_per_electron = 1/IO_dict.electrons_per_gray_level
IO_dict.electrons_per_photon = 1/IO_dict.photons_per_electron

normalization_constant_after_noise = 1
final_scaling_factor = 150/IO_dict.current_mean_gray_level_per_pixel
flag_quantize = True

#######################################################################################################################################
### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Methos: ###
Algo_Dict = EasyDict()
image_index_to_start_from = 0  #1000
number_of_images_to_generate_each_time = 9
number_of_images = np.inf #np.inf
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

initial_crop_size = (1800, 800)
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
                                     allowed_extentions=['.png', '.jpg'],
                                     flag_recursive=True,
                                     string_pattern_to_search='*')

### Get Noise Images: ###
Camera_Noise_Folder = os.path.join(datasets_main_folder, '/KAYA_CAMERA_NOISE/noise')
noise_filenames = read_image_filenames_from_folder(path=Camera_Noise_Folder,
                                     number_of_images=np.inf,
                                     allowed_extentions=['.tif'],
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
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies_Kaya_FFTRegistration'
path_make_path_if_none_exists(results_path)
# final_experiment_level = str(IO_dict.current_mean_gray_level_per_pixel)
final_experiment_level = 'FFDNet_try'
save_path = os.path.join(results_path, final_experiment_level)
save_path_SR = os.path.join(save_path, 'SR')
save_path1 = os.path.join(save_path, 'original_frames')
save_path2 = os.path.join(save_path, 'clean_frame_estimate')
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path_SR)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name_OriginalAligned = os.path.join(save_path, 'Results', 'OriginalVsAligned_bla2.avi')
video_name_OriginalClean = os.path.join(save_path, 'Results', 'OriginalVsClean_bla2.avi')
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 480))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 480))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 512))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 512))
video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1800*2, 800))
video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1800*2, 800))
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
original_frame_previous = original_frame_initial
original_frames_list.append(original_frame_initial)
aligned_frames_list.append(crop_torch_batch(original_frame_initial, initial_crop_size, crop_style='center'))
clean_frames_list.append(crop_torch_batch(original_frame_initial, initial_crop_size, crop_style='center'))  #TODO: in the previous script i didn't initialize it with the first frame...interesting
original_frames_cropped_list.append(crop_torch_batch(original_frame_initial, initial_crop_size, crop_style='center'))
scale_factor = (210/original_frame_initial.max()).item()


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

number_of_images = min(number_of_images, len(image_filenames) - 5)
counter = 0
affine_layer_torch = Warp_Tensors_Affine_Layer()

from RDND_proper.scenarios.Classic_TNR.Registration.Registration_End2End_Movie_Synthetic_SuperParameter import FFT_registration_layer
fft_registration_layer = FFT_registration_layer()
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + 1 + 300):
    print(frame_index)

    ### Get Last and Current Frame: ###
    # original_frame_current = read_image_general(image_filenames[frame_index])
    original_frame_current = read_image_general(image_filenames[image_index_to_start_from+1])
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2, 0, 1]).unsqueeze(0)
    original_frame_current = RGB2BW(original_frame_current)

    ### Shit For Synthetic Shifting Movie: ###
    original_frame_current = read_image_general(image_filenames[image_index_to_start_from + 1])
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2, 0, 1]).unsqueeze(0)
    original_frame_current = RGB2BW(original_frame_current)
    GT_shift_x = np.float32(get_random_number_in_range(-1, 1, [1]))
    GT_shift_y = np.float32(get_random_number_in_range(-1, 1, [1]))
    GT_rotation_angle = np.float32(get_random_number_in_range(-0,0, [1]))
    GT_scale = np.float32(get_random_number_in_range(1 - 0.0, 1 + 0.0, [1]))
    original_frame_current = affine_layer_torch.forward(original_frame_current, GT_shift_x, GT_shift_y, GT_scale,
                                                        GT_rotation_angle, flag_interpolation_mode='bicubic')

    ### Register Images: ###
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size, crop_style='center')
    original_frame_current_cropped = nn.AvgPool2d(1)(original_frame_current_cropped)

    ### Add Noise To Images: ###
    input_image_mean = original_frame_current_cropped.mean() + 1e-5
    original_frames_scaling_factor = (IO_dict.current_mean_gray_level_per_pixel) / input_image_mean
    original_frame_current_cropped = original_frame_current_cropped * original_frames_scaling_factor
    output_frames_original_photons = original_frame_current_cropped * IO_dict.electrons_per_gray_level * IO_dict.photons_per_electron
    output_frames_original_photons_shot_noise = torch.sqrt(output_frames_original_photons) * torch.randn(*original_frame_current_cropped.shape).cuda()
    output_frames_original_shot_noise = output_frames_original_photons_shot_noise * IO_dict.electrons_per_photon * IO_dict.gray_levels_per_electron
    shot_noise_sigma_map = torch.sqrt(output_frames_original_photons) * IO_dict.electrons_per_photon * IO_dict.gray_levels_per_electron
    current_readout_noise_image = read_image_general(noise_filenames[frame_index])
    current_readout_noise_image = torch.Tensor(current_readout_noise_image).cuda().permute([2,0,1]).unsqueeze(0)
    current_readout_noise_image = crop_torch_batch(current_readout_noise_image, initial_crop_size, crop_style='center')
    original_frame_current_cropped += output_frames_original_shot_noise
    original_frame_current_cropped += current_readout_noise_image
    # print(original_frame_current_cropped.max())
    if flag_quantize:
        original_frame_current_cropped = torch.round(original_frame_current_cropped).type(torch.int).type(torch.float32)
    # print(original_frame_current_cropped.max())

    ### Normalize Input After Adding Noise: ###
    # original_frame_current_cropped = original_frame_current_cropped / normalization_constant_after_noise
    # original_frame_current_cropped = original_frame_current_cropped / original_frame_current_cropped.mean() / 3

    ### Add Current Frame To List: ###
    input_dict.original_frames_list.append(original_frame_current_cropped)

    ### Perform Registration: ###
    frame_input = fft_registration_layer.forward(original_frame_current_cropped)
    # frame_input = original_frame_current_cropped

    ### Perform Denoise Using FastDVDNet: ###
    frames_list_length = len(input_dict.original_frames_list)
    if frames_list_length > 5:
        sigma_to_model = 0
        sigma_to_model = sigma_to_model * torch.ones_like(frame_input).to(original_frame_current_cropped.device)
        sigma_to_model = sigma_to_model[:, 0:1, :, :]
        current_frames_list = input_dict.original_frames_list[frames_list_length-5:frames_list_length]
        center_frame = current_frames_list[2]
        network_input = torch.cat(current_frames_list, 1).clone()
        network_input = BW2RGB_interleave_C(network_input)
        final_network_input = (network_input, sigma_to_model)
        with torch.no_grad():
            clean_frame_current = RGB2BW(FastDVDNet_model.forward(final_network_input))
    else:
        clean_frame_current = frame_input
        ### Perform Denoise Using FastDVDNet: ###
        frames_list_length = len(input_dict.original_frames_list)

    # ### Perform Denoise Using FFDNet: ###
    # frames_list_length = len(input_dict.original_frames_list)
    # if frames_list_length > 5:
    #     FFDNet_model = FFDNet_model.train()
    #     # FFDNet_model = FFDNet_model.eval()
    #     # sigma_to_model = 1/np.sqrt(5)*255
    #     sigma_to_model = 1/np.sqrt(np.inf)*255
    #     # sigma_to_model = sigma_to_model * torch.ones_like(original_frame_current_cropped).to(original_frame_current_cropped.device)
    #     # sigma_to_model = sigma_to_model[:, 0:1, :, :]
    #     current_frames_list = input_dict.original_frames_list[frames_list_length - 5:frames_list_length]
    #     center_frame = current_frames_list[2]
    #     network_input = torch.cat(current_frames_list, 1).clone()
    #     # network_input = BW2RGB_interleave_C(network_input)
    #     final_network_input = (network_input, sigma_to_model)
    #     # final_network_input = (original_frame_current_cropped, sigma_to_model)
    #     with torch.no_grad():
    #         clean_frame_current = FFDNet_model.forward(final_network_input)
    # else:
    #     clean_frame_current = original_frame_current_cropped


    # ### Update input dict: ###
    # input_dict.clean_frames_list.append(clean_frame_current)

    # ### Perform Alignment: ###
    # input_dict = cross_correlation_alignment_super_parameter(input_dict)

    # ### Outputs From Dict: ###
    # clean_frame_current = input_dict['clean_frames_list'][-1]

    ### Record Original-Clean Video: ###
    # image_to_write = torch.cat((original_frame_current_cropped, clean_frame_current), -1) * final_scaling_factor
    image_to_write = torch.cat((original_frame_current_cropped, clean_frame_current), -1) * 255
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    video_object_OriginalClean.write(image_to_write)

    # ### Save .png Image: ###
    # save_image_numpy(save_path, string_rjust(counter, 4) + '.png', image_to_write, flag_convert_bgr2rgb=False, flag_scale=False)
    # save_image_numpy(save_path_SR, string_rjust(counter, 4) + '.png', RGB2BW(clean_frame_current*256).cpu()[0].permute([1,2,0])[:,:,0].numpy(), flag_convert_bgr2rgb=False, flag_scale=False)
    counter += 1




### Stop Video Writer Object: ###
video_object_OriginalAligned.release()
video_object_OriginalClean.release()
1
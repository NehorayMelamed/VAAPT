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
# original_frames_folder = '/home/mafat/DataSets/Aviram/2'
original_frames_folder = '/home/mafat/DataSets/stream_0/raw_image'


#######################################################################################################################################
### Neural Network Models: ###
#(*). For All:
import RapidBase.TrainingCore as core
Train_dict = EasyDict()
Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False

### Import Models: ###
#(2). EGVSR:
from RDND_proper.models.EGVSR.codes.models.networks.egvsr_nets import *
Train_dict.models_folder = r'/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/'
# Train_dict.load_Network_filename = 'EGVSR_MultipleImages1Shift_Recursive_1_Checkpoint_TEST1_Step29400.tar'
# Train_dict.load_Network_filename = 'EGVSR_MultipleImages1Shift_Recursive_2_Checkpoint_TEST1_Step32600.tar'
# Train_dict.load_Network_filename = 'EGVSR_MultipleImages1Shift_Recursive_Denoise_1_Checkpoint_TEST1_Step64200.tar'
Train_dict.load_Network_filename = 'EGVSR_Videos_Recursive_1_Checkpoint_TEST1_Step17000.tar'
# EGVSR_model = FRNet_dudy(in_nc=3, out_nc=3, nf=64, nb=10, degradation='BI', scale=4)
EGVSR_model = FRNet_dudy_Jetson(in_nc=3, out_nc=3, nf=64, nb=10, degradation='BI', scale=4)

full_path = os.path.join('/home/mafat/Pytorch_Checkpoints/Model_New_Checkpoints/EGVSR_Videos_Recursive_1_Checkpoint', Train_dict.load_Network_filename)
args_from_checkpoint = torch.load(full_path)
network_weights_from_checkpoint = args_from_checkpoint['model_state_dict']
pretrained_dict = network_weights_from_checkpoint
current_network_weights = EGVSR_model.state_dict()
# print(network_weights_from_checkpoint.keys())
# print(current_network_weights.keys())
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in current_network_weights}
current_network_weights.update(pretrained_dict)
EGVSR_model.load_state_dict(current_network_weights)

# EGVSR_model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=EGVSR_model)
EGVSR_model = EGVSR_model.cuda()
EGVSR_model = EGVSR_model.eval()

### IO dict: ###
IO_dict = EasyDict()
IO_dict.current_mean_gray_level_per_pixel = 250
IO_dict.electrons_per_gray_level = 14.53 * 1
IO_dict.photons_per_electron = 1 #TODO: i don't think this is the QE but whether we have a photo-multiplier or not....check this out
IO_dict.gray_levels_per_electron = 1/IO_dict.electrons_per_gray_level
IO_dict.electrons_per_photon = 1/IO_dict.photons_per_electron

normalization_constant_after_noise = 1
final_scaling_factor = 150/IO_dict.current_mean_gray_level_per_pixel
sigma_to_model = 1/2*0
flag_quantize = True

#######################################################################################################################################
### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Methos: ###
Algo_Dict = EasyDict()
image_index_to_start_from = 0  #1000
number_of_images_to_generate_each_time = 9
number_of_images = 50 #np.inf
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
final_experiment_level = 'EGVSR_try'
save_path = os.path.join(results_path, final_experiment_level)
save_path_SR = os.path.join(save_path, 'SR')
save_path1 = os.path.join(save_path, 'original_frames')
save_path2 = os.path.join(save_path, 'clean_frame_estimate')
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path_SR)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name_OriginalAligned = os.path.join(save_path, 'Results', 'OriginalVsAligned.avi')
video_name_OriginalClean = os.path.join(save_path, 'Results', 'OriginalVsClean.avi')
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 480))
# video_object_OriginalOpticalFlow = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 480))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 512))
# video_object_OriginalOpticalFlow = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 512))
video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1800*2, 800))
video_object_OriginalOpticalFlow = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1800*2, 800))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (400, 200))
# video_object_OriginalOpticalFlow = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (400, 200))
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
hr_prev = BW2RGB(shuffle_up(torch.zeros_like(original_frame_previous).repeat(1, EGVSR_model.upsample_factor ** 2, 1, 1), EGVSR_model.upsample_factor).cuda())
lr_prev = BW2RGB(original_frame_previous)/255
affine_layer_torch = Warp_Tensors_Affine_Layer()
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + 1 + number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    original_frame_current = read_image_general(image_filenames[frame_index])
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2,0,1]).unsqueeze(0)
    original_frame_current = RGB2BW(original_frame_current)
    
    # ### Shit For Synthetic Shifting Movie: ###
    # original_frame_current = read_image_general(image_filenames[image_index_to_start_from + 1])
    # original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2, 0, 1]).unsqueeze(0)
    # original_frame_current = RGB2BW(original_frame_current)
    # GT_shift_x = np.float32(get_random_number_in_range(-1, 1, [1]))
    # GT_shift_y = np.float32(get_random_number_in_range(-1, 1, [1]))
    # GT_rotation_angle = np.float32(get_random_number_in_range(-0,0, [1]))
    # GT_scale = np.float32(get_random_number_in_range(1 - 0.0, 1 + 0.0, [1]))
    # original_frame_current = affine_layer_torch.forward(original_frame_current, GT_shift_x, GT_shift_y, GT_scale,
    #                                                     GT_rotation_angle, flag_interpolation_mode='bicubic')

    ### Register Images: ###
    original_frame_current_cropped = crop_torch_batch(original_frame_current, initial_crop_size, crop_style='center')
    original_frame_current_cropped = nn.AvgPool2d(1)(original_frame_current_cropped)

    # ### Add Current Frame To List: ###
    # input_dict.original_frames_list.append(original_frame_current_cropped)

    ### Perform Super-Resolution Using EGVSR: ###
    lr_curr = BW2RGB(original_frame_current_cropped)/255
    final_network_input = [lr_curr, lr_prev, hr_prev]
    # print_info(lr_curr, 'lr_curr')
    with torch.no_grad():
        # [clean_frame_current, lr_flow] = EGVSR_model.forward(final_network_input)
        [clean_frame_current, lr_flow] = EGVSR_model.forward(lr_curr)
    # print_info(clean_frame_current, 'network_output')
    u = lr_flow.cpu()[0,0].numpy()
    v = lr_flow.cpu()[0,1].numpy()
    flow_RGB = flow_uv_to_colors(u,v)
    flow_magnitude = torch.sqrt(lr_flow[:,0:1]**2+lr_flow[:,1:2]**2)
    lr_prev = lr_curr
    hr_prev = clean_frame_current

    # ### Update input dict: ###
    # input_dict.clean_frames_list.append(clean_frame_current)

    # ### Perform Alignment: ###
    # input_dict = cross_correlation_alignment_super_parameter(input_dict)

    # ### Outputs From Dict: ###
    # clean_frame_current = input_dict['clean_frames_list'][-1]

    ### Record Original-OpticalFlow Video: ###
    image_to_write = torch.cat((original_frame_current_cropped, flow_magnitude*20), -1) * final_scaling_factor
    # image_to_write = original_frame_current_cropped * final_scaling_factor
    image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    video_object_OriginalOpticalFlow.write(image_to_write)

    ### Save .png Image: ###
    # lr_flow_new = torch.flip(lr_flow, [1])
    # lr_flow_new = lr_flow mko0
    # lr_flow_new[:,1,:,:] = -lr_flow_new[:,1,:,:]
    # arrow_image = put_optical_flow_arrows_on_image(RGB2BW(lr_curr * 256).cpu()[0].permute([1, 2, 0])[:, :, 0].numpy(),
    #                                        (lr_flow_new * 2).cpu()[0].permute([1, 2, 0]).numpy(), threshold=2.0,
    #                                        skip_amount=30)
    #(1). Original + OpticalFlow .png:
    save_image_numpy(save_path, string_rjust(counter, 4) + '.png', image_to_write, flag_convert_bgr2rgb=False, flag_scale=False)
    #(2). Super Resolved Image .png:
    save_image_numpy(save_path_SR, string_rjust(counter, 4) + '.png', RGB2BW(clean_frame_current*256).cpu()[0].permute([1,2,0])[:,:,0].numpy(), flag_convert_bgr2rgb=False, flag_scale=False)
    counter += 1


### Stop Video Writer Object: ###
video_object_OriginalAligned.release()
video_object_OriginalOpticalFlow.release()
1
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
### Neural Network Models: ###
#(*). For All:
import RapidBase.TrainingCore as core
Train_dict = EasyDict()
Train_dict.models_folder = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints\Model_New_Checkpoints")
Train_dict.Network_checkpoint_step = 0
Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False

#(1). STARFlow:
from RDND_proper.models.star_flow.models.STAR import *
Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = True
Train_dict.load_Network_filename = r'/home/mafat/PycharmProjects/IMOD/models/star_flow/saved_checkpoint/StarFlow_things/checkpoint_best.ckpt'
args = EasyDict()
div_flow = 0.05
STARFlow_Net = StarFlow(args)
STARFlow_Net, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=STARFlow_Net)
STARFlow_Net = STARFlow_Net.cuda()
#(2). FastFlowNet:
from RDND_proper.models.FastFlowNet.models.FastFlowNet import *
Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False
Train_dict.load_Network_filename = r'/home/mafat/PycharmProjects/IMOD/models/FastFlowNet/checkpoints/fastflownet_things3d.pth'
FastFlowNet_model = FastFlowNet()
FastFlowNet_model, Train_dict = TrainingCore.training_utils.load_model_from_checkpoint(Train_dict=Train_dict, model=FastFlowNet_model)
FastFlowNet_model = FastFlowNet_model.cuda()
#(3). RAFT:
from RDND_proper.models.RAFT.core.raft import *
Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = True
Train_dict.load_Network_filename = '/home/mafat/PycharmProjects/IMOD/models/RAFT/Checkpoints/raft-small.pth'
args = EasyDict()
args.dropout = 0
args.alternate_corr = False
args.small = True
args.mixed_precision = False
args.number_of_iterations = 20
RAFT_model = RAFT_dudy(args).cuda()

#######################################################################################################################################

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
# final_crop_size = (512, 512)
final_crop_size = (1024, 1024)
final_final_crop_size = (1024, 1024)
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
normalization_factor_after_NUC = 20
original_frame_current = read_image_general(image_filenames[0])
original_frame_current = torch.Tensor(original_frame_current).permute([2,0,1]).unsqueeze(0).cuda()
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
video_name_OriginalAligned = os.path.join(save_path, 'Results', 'OriginalVsAligned.avi')
video_name_OriginalClean = os.path.join(save_path, 'Results', 'OriginalVsClean.avi')
video_width = crop_size_x * 2
video_height = crop_size_x * 1
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 480))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 480))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024, 512))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024, 512))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (480*2, 480))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (480*2, 480))
video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (1024*2, 1024))
video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (1024*2, 1024))
# video_object_OriginalAligned = cv2.VideoWriter(video_name_OriginalAligned, 0, 10, (400, 200))
# video_object_OriginalClean = cv2.VideoWriter(video_name_OriginalClean, 0, 10, (400, 200))
#######################################################################################################################################



#######################################################################################################################################
affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
number_of_frames_to_average = 5
final_crop_size = final_crop_size[0]

### Populate Initial List: ###
original_frame_initial = read_image_general(image_filenames[image_index_to_start_from])
original_frame_initial = torch.Tensor(original_frame_initial).cuda().permute([2, 0, 1]).unsqueeze(0)
original_frame_initial = crop_torch_batch(original_frame_initial, final_crop_size, crop_style='center')
# original_frame_initial = RGB2BW(original_frame_initial)
original_frame_initial_crop = crop_torch_batch(original_frame_initial, final_final_crop_size, crop_style='center')
original_frames_list.append(original_frame_initial)
aligned_frames_list.append(original_frame_initial)
scale_factor = (210/original_frame_initial.max()).item()

### Loop Over Images: ###
for frame_index in np.arange(image_index_to_start_from+1, image_index_to_start_from + 1 + number_of_images):
    print(frame_index)

    ### Get Last and Current Frame: ###
    # original_frame_previous = aligned_frames_list[0]
    # original_frame_previous = original_frames_list[0]
    original_frame_previous = read_image_cv2(image_filenames[frame_index-1])
    original_frame_current = read_image_cv2(image_filenames[frame_index])
    original_frame_current = torch.Tensor(original_frame_current).cuda().permute([2,0,1]).unsqueeze(0)
    original_frame_previous = torch.Tensor(original_frame_previous).cuda().permute([2,0,1]).unsqueeze(0)
    # original_frame_current = RGB2BW(original_frame_current)

    ### Register Images: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_previous, final_crop_size, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, final_crop_size, crop_style='center')
    original_frame_previous_cropped = nn.AvgPool2d(1)(original_frame_previous_cropped)
    original_frame_current_cropped = nn.AvgPool2d(1)(original_frame_current_cropped)
    original_frames_list.append(original_frame_current_cropped)

    # Get Optical Flow: ###
    #(1). STARFlow_Net:
    list_of_images = []
    list_of_images.append(original_frame_previous_cropped/255)
    list_of_images.append(original_frame_current_cropped/255)
    with torch.no_grad():
        output_dict = STARFlow_Net.forward(list_of_images)
    output_flow_list = output_dict['flow']
    output_occlusions_list = output_dict['occ']
    output_flow = output_flow_list[-1]
    output_occlusions = output_occlusions_list[-1]
    output_flow_magnitude = torch.sqrt(output_flow[:,0:1,:,:]**2 + output_flow[:,1:2,:,:,]**2)
    output_flow_magnitude = output_flow_magnitude * 20 * 2
    # #(2). FastFlowNet:
    # list_of_images = []
    # list_of_images.append(original_frame_previous_cropped / 255)
    # list_of_images.append(original_frame_current_cropped / 255)
    # # list_of_images.append(BW2RGB(RGB2BW(original_frame_previous_cropped)) / 255)
    # # list_of_images.append(BW2RGB(RGB2BW(original_frame_current_cropped)) / 255)
    # with torch.no_grad():
    #     FastFlowNet_model = FastFlowNet_model.eval()
    #     output_flow = FastFlowNet_model.forward(list_of_images)
    # output_flow_magnitude = torch.sqrt(output_flow[:, 0:1, :, :] ** 2 + output_flow[:, 1:2, :, :, ] ** 2)
    # output_flow_magnitude = output_flow_magnitude * 20 * 4
    # output_flow_magnitude = F.interpolate(output_flow_magnitude, scale_factor=4)
    # #(3). RAFT:
    # list_of_images = []
    # list_of_images.append(original_frame_previous_cropped / 255)
    # list_of_images.append(original_frame_current_cropped / 255)
    # with torch.no_grad():
    #     RAFT_model = RAFT_model.eval()
    #     model_input = (2 * list_of_images[0] - 1, 2 * list_of_images[1] - 1, RAFT_model.number_of_iterations, None, True, False)
    #     # output_flow = RAFT_model.forward(model_input)[1]
    #     output_flow = RAFT_model.forward(model_input)[-1]
    # output_flow_magnitude = torch.sqrt(output_flow[:, 0:1, :, :] ** 2 + output_flow[:, 1:2, :, :, ] ** 2)
    # output_flow_magnitude = output_flow_magnitude * 4

    # ### Update Aligned Frames Lists: ###
    # aligned_frames_list.append(original_frame_current_cropped_warped)

    ### Crop Again To Avoid InValid Regions: ###
    original_frame_previous_cropped = crop_torch_batch(original_frame_initial, final_final_crop_size, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, final_final_crop_size, crop_style='center')
    output_flow_magnitude = crop_torch_batch(output_flow_magnitude, final_final_crop_size, crop_style='center')
    original_frame_current_cropped = RGB2BW(original_frame_current_cropped)

    # ### Average ALIGNED Frames: ###
    # clean_frame_current = torch.zeros((1,1,original_frame_current_cropped_warped[0].shape[-2],original_frame_current_cropped_warped[0].shape[-1])).cuda()
    # for i in np.arange(len(aligned_frames_list)):
    #     # clean_frame_current += aligned_frames_list[i]
    #     clean_frame_current += crop_torch_batch(aligned_frames_list[i], final_final_crop_size, crop_style='center')
    # clean_frame_current = clean_frame_current / len(aligned_frames_list)
    # clean_frames_list.append(clean_frame_current)

    # ### Update Lists: ###
    # if frame_index > image_index_to_start_from+1+5:
    #     aligned_frames_list.pop(0)
    #     original_frames_list.pop(0)
    #     clean_frames_list.pop(0)


    # ### Save Wanted Images: ###
    # save_image_numpy(save_path1, str(frame_index) + '_center_frame_original.png', original_frame_current_cropped[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True)
    # save_image_numpy(save_path2, str(frame_index) + '_clean_frame_estimate.png', clean_frame_current[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True)

    ### Record Original-Clean Video: ###
    image_to_write = torch.cat((original_frame_current_cropped, output_flow_magnitude), -1)
    image_to_write = image_to_write
    image_to_write = torch.cat((image_to_write,image_to_write,image_to_write),1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1,2,0)
    image_to_write = (image_to_write * scale_factor).clip(0,255).astype(np.uint8)
    video_object_OriginalClean.write(image_to_write)

    # ### Record Original-Aligned: ###
    # image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped), -1)
    # # image_to_write = torch.cat((original_frame_current_cropped, original_frame_current_cropped_warped-original_frame_current_cropped), -1)
    # image_to_write = torch.cat((image_to_write, image_to_write, image_to_write), 1)
    # image_to_write = image_to_write[0].cpu().numpy().transpose(1, 2, 0)
    # image_to_write = (image_to_write * scale_factor).clip(0, 255).astype(np.uint8)
    # video_object_OriginalAligned.write(image_to_write)


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
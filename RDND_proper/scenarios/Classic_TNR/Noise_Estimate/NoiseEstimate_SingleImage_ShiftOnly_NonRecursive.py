from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_to_RGB

from RapidBase.import_all import *

from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Noise_Estimate_misc import *


### Paths: ###
##############
inference_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints/Inference/model_combined/test_for_bla")
original_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/original_frames')
noisy_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/noisy_frames')
clean_frames_folder = path_fix_path_for_linux('/home/mafat/Pytorch_Checkpoints/Inference/model_combined/DataSet_SingleVideo_RollingIndex_AddNoise/Beirut_BW_Original_SNR5/clean_frame_estimate')
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

# crop_size = 512
# bla = torch.randn((1,16,crop_size,crop_size)).cuda()
# bla_numpy = numpy.random.randn(16,crop_size,crop_size)
# tic()
# for i in np.arange(crop_size):
#     for j in np.arange(crop_size):
#         bli = numpy.histogram(bla_numpy[:,i,j])
# toc()
# tic()
# for i in np.arange(crop_size):
#     for j in np.arange(crop_size):
#         bli = torch.histc(bla[0,:,i,j], 16)
# toc()

#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
original_frames_dataset = DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=original_frames_folder,
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=10,  #TODO: add possibility to randomize start index from a possibly long video
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
noisy_frames_dataset = DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=noisy_frames_folder,
                                                         transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=10,  #TODO: add possibility to randomize start index from a possibly long video
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


### Initialize Shit: ###
clean_frame_last = 0
histograms_array = 0
SNR = 10
max_shift_in_pixels = 0.5
crop_size_x = 1024
crop_size_y = 1024
noise_sigma = 1/sqrt(SNR)

downsample_kernel_size = 16
average_pooling_layer = nn.AvgPool2d(downsample_kernel_size)
upsample_layer = nn.UpsamplingNearest2d(scale_factor=downsample_kernel_size)

### Take care of initializations and such for first frame: ###
original_frames_output_dict = original_frames_dataset[0]
original_frame_current = original_frames_output_dict.center_frame_original[0].data
original_frame_current = original_frame_current.unsqueeze(0).unsqueeze(0)
noisy_frame_current = original_frame_current + noise_sigma*torch.randn_like(original_frame_current)
noisy_frame_current_cropped = crop_torch_batch(noisy_frame_current, crop_size_x, crop_style='center')
original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_x, crop_style='center')

clean_frame_previous = noisy_frame_current_cropped  #initialize internal "cleaned" image to simply be first frame, which is noisy
original_frame_previous = original_frame_current
noisy_frame_previous = noisy_frame_current
B,C,H,W = noisy_frame_current.shape
crop_size_x = min(crop_size_x, W)
crop_size_y = min(crop_size_y, H)
shifts_array = torch.zeros((1, 1, crop_size_x, crop_size_x, 2))
noise_map_last = torch.zeros((1, 1, crop_size_x, crop_size_x))
pixel_counts_map_previous = torch.ones((1, 1, crop_size_x, crop_size_x))
noise_pixel_counts_map_previous = torch.ones((1, 1, crop_size_x, crop_size_x))
noise_map_previous = None

histogram_spatial_binning_factor = 4
histogram_number_of_images = 16
histogram_counter = 0
histogram_spatial_size = int(crop_size_x / histogram_spatial_binning_factor)
histogram_number_of_samples = histogram_number_of_images * histogram_spatial_binning_factor**2
max_histogram_counter = histogram_number_of_samples - 1
histogram_start = 0
histogram_stop = 1
histogram_number_of_steps = 255
histogram_bins_vec = my_linspace(histogram_start, histogram_stop, histogram_number_of_steps)
histogram_history_matrix = np.zeros((histogram_number_of_samples, histogram_spatial_size, histogram_spatial_size))
flag_get_hist = False

### Initialize video stream to record noise estimation results: ###
folder_path = '/home/mafat/PycharmProjects/IMOD/scenarios'
folder_path = os.path.join(folder_path, 'Noise_Estimation')
path_make_path_if_none_exists(folder_path)
video_name2 = os.path.join(folder_path,'ChangingNoiseLevel_UniformPattern_Frames.avi')
video_name = os.path.join(folder_path,'ChangingNoiseLevel_UniformPattern_NoiseGraph.avi')
video_object = cv2.VideoWriter(video_name, 0, 3, (crop_size_x, crop_size_x))
video_object2 = cv2.VideoWriter(video_name2, 0, 3, (crop_size_x, crop_size_x))

### Switch to tensorflow notation: ###1
shift_layer_torch = Shift_Layer_Torch()
for frame_index in np.arange(1, 16*5):
    print(frame_index)

    ### Repeat same image every time: ###
    original_frame_current = original_frame_previous
    B,C,H,W = original_frame_current.shape

    ### Creal Noise Map: ###
    if frame_index < 12:
        SNR = 10
    elif frame_index >=12 and frame_index <= 38:
        SNR = 10
    elif frame_index > 30:
        SNR = 6
    # SNR = 10
    noise_map = 1/sqrt(SNR) * torch.ones_like(original_frame_current)  #uniform noise
    # if frame_index == 1:
    #     noise_map, noise_map_field1, noise_map_field2 = create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels=200, N=max(H,W), polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0)
    #     noise_map = noise_map / noise_map.max()
    #     noise_map = torch.Tensor(noise_map).unsqueeze(0).unsqueeze(0)
    #     noise_map = noise_map[:,:,0:H,0:W]
    #     noise_map = 1/sqrt(SNR) * noise_map
    # if frame_index == 26:
    #     noise_map, noise_map_field1, noise_map_field2 = create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels=200, N=max(H,W), polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0)
    #     noise_map = noise_map / noise_map.max()
    #     noise_map = torch.Tensor(noise_map).unsqueeze(0).unsqueeze(0)
    #     noise_map = noise_map[:,:,0:H,0:W]
    #     noise_map = 1/sqrt(SNR) * noise_map

    ### Add Noise Map: ###
    noisy_frame_current = original_frame_current + noise_map * torch.randn_like(noise_map)

    ### Center Crop: ###
    noisy_frame_current_cropped = crop_torch_batch(noisy_frame_current, crop_size_x, crop_style='center')
    original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_x, crop_style='center')
    noise_map_real = crop_torch_batch(noise_map, crop_size_x, crop_style='center')

    ### Register Images (return numpy arrays): ###
    clean_frame_previous_warped = clean_frame_previous #no registration because we're using the same image, equivalent to perfect registration

    ### Get both frames to be in Tensor / Numpy Type: ###
    clean_frame_previous_warped = torch.Tensor(clean_frame_previous_warped)

    ### Estimate noise: ###
    noise_map_current, noise_pixel_counts_map_current = estimate_noise_IterativeResidual(noisy_frame_current_cropped,
                                                                         noise_map_previous,
                                                                         noise_pixel_counts_map_previous,
                                                                         clean_frame_previous_warped)
    # noise_map_current, noise_pixel_counts_map_current, flag_get_hist, histogram_counter = estimate_noise_FullFrameHistogram(noisy_frame_current_cropped,
    #                                                                      noise_map_previous,
    #                                                                      noise_pixel_counts_map_previous,
    #                                                                      clean_frame_previous_warped,
    #                                                                      histogram_history_matrix,
    #                                                                      histogram_counter,
    #                                                                      flag_get_hist)

    ### Combine Images: ###
    clean_frame_current, pixel_counts_map_current = combine_frames(noisy_frame_current_cropped,
                                         clean_frame_previous_warped,
                                         noise_map_current = noise_map_current,
                                         noise_map_last_warped = None,
                                         pixel_counts_map_previous = pixel_counts_map_previous)

    ### Keep variables for next frame: ###
    clean_frame_previous = clean_frame_current  #TODO: right now i simply keep the last noisy image, as i don't clean anything now
    original_frame_previous = original_frame_current
    noisy_frame_previous = noisy_frame_current
    pixel_counts_map_previous = pixel_counts_map_current
    noise_pixel_counts_map_previous = noise_pixel_counts_map_current
    noise_map_previous = noise_map_current

    ### Get Video Frames: ###
    noise_map_difference_mean = (noise_map_real - noise_map_current).mean()
    noise_map_difference_std = (noise_map_real - noise_map_current - noise_map_difference_mean).abs().std()
    fig = imshow_torch_multiple((noise_map_real, noise_map_current, (noise_map_real-noise_map_current).abs()))
    # fig = imshow_torch_multiple((noise_map_real, noise_map_current, noisy_frame_current_cropped))
    title('Frame: ' + str(frame_index) + ', Sigma ' + decimal_notation(1/sqrt(SNR),2) +
          ',\n diff mean = ' + decimal_notation(noise_map_difference_mean.numpy(), 3) +
          ',\n diff std = ' + decimal_notation(noise_map_difference_std.numpy(), 3))
    # Now we can save it to a numpy array.
    data = video_get_mat_from_figure(fig, (crop_size_x, crop_size_x))
    video_object.write(data)
    close()

    ### Get Video Frames Themselves: ###
    video_object2.write((torch_to_RGB(noisy_frame_current_cropped).squeeze(0).cpu().permute([1,2,0]).clamp(0,1).numpy()*255).astype(np.uint8))

cv2.destroyAllWindows()
video_object.release()

    # 1
    # imshow_torch(noisy_frame_current_cropped.unsqueeze(0))
    # imshow_torch(clean_frame_current.unsqueeze(0))
    # imshow_torch(original_frame_current_cropped.unsqueeze(0))


    # ### Estimate Noise: ###
    # noise_map_current = estimate_noise(noise_map_last, noisy_frame_current, clean_frame_last, histograms_array)
    # noise_map_last_warped = warp_image(noise_map_last_warped, shifts_array)
    #
    # ### Combine Images: ###
    # clean_frame_current = combine_frames(noisy_frame_current, clean_frame_last_warped, noise_map_current, noise_map_last_warped, pixel_counts_map_previous)












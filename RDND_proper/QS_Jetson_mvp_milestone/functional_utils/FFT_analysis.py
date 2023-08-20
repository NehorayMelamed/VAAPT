import numpy as np
import torch

from RapidBase.import_all import *
from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import *
import kornia

def Brute_Force_FFT(input_tensor, params):
    T, H, W = input_tensor.shape
    number_of_frames = T

    ### Perform FFT per pixel: ###
    FPS = params.FPS
    input_tensor = torch.Tensor(input_tensor).cuda()
    frequency_axis = FPS * np.linspace(-0.5, 0.5 - 1 / T, T)
    frequency_axis_delta = frequency_axis[2] - frequency_axis[1]

    ### Loop over the different angles: ###
    max_number_of_pixels_to_check = params.max_number_of_pixels_to_check
    angles_vec = my_linspace(0, 180, max_number_of_pixels_to_check)
    max_translation_vec = my_linspace(0, max_number_of_pixels_to_check, max_number_of_pixels_to_check)
    affine_warp_layer = Warp_Tensors_Affine_Layer()
    flow_grids_list = []  #should be len(angles_vec) X len(number_of_directional_paths), probably should make it X number_of_time_steps
    rotation_flow_grids_list = []
    translation_flow_grids_list = []
    output_tensors_list = []


    # ########################################################################################################################
    # ### Get Angles Grids: ###
    # for current_angle in angles_vec:
    #     ### Rotate Matrix: ###
    #     output_tensor, flow_grid = affine_warp_layer.forward(torch.zeros((1, 1, H, W)).cuda(),
    #                                                          shift_x=torch.Tensor([0]).cuda(),
    #                                                          shift_y=torch.Tensor([0]).cuda(),
    #                                                          scale=torch.Tensor([1]).cuda(),
    #                                                          rotation_angle=torch.Tensor([current_angle]).cuda(),
    #                                                          return_flow_grid=True,
    #                                                          flag_return_only_flow_grid=True,
    #                                                          flag_interpolation_mode='bilinear')
    #     # imshow_torch_video(output_tensor[0].unsqueeze(1), 50, 10, False)
    #     ### Append to flow grids list: ###
    #     rotation_flow_grids_list.append(flow_grid.cpu())
    #
    # ### Get Translations Vecs: ###
    # for current_max_translation in max_translation_vec:
    #     ### Calculate the translation per frame vec: ###
    #     translation_per_frame = current_max_translation / number_of_frames
    #     per_frame_translation_vec = my_linspace_step(0, translation_per_frame, number_of_frames)
    #
    #     ### Append to flow grids list: ###
    #     translation_flow_grids_list.append(torch.Tensor(per_frame_translation_vec).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    # ################################################################################################################################


    ################################################################################################################################
    ### Loop Over Possible Angles: ###
    for current_angle in angles_vec:
        ### Loop Over the different possible paths of the drone (different possible speeds): ###
        for current_max_translation in max_translation_vec:
            ### Calculate the translation per frame vec: ###
            # torch_get_4D()
            translation_per_frame = current_max_translation / number_of_frames
            per_frame_translation_vec = my_linspace_step(0, translation_per_frame, number_of_frames)

            ### Translate Matrices: ###
            #TODO: use affine_grid to take the grids themselves and add a shift to each frame grid (don't forget it's normalized to [-1,1]):
            output_tensor, flow_grid = affine_warp_layer.forward(torch_get_4D(input_tensor, 'THW'),
                                                                 shift_x=torch.Tensor(per_frame_translation_vec).cuda(),
                                                                 shift_y=torch.Tensor([0]).cuda(),
                                                                 scale=torch.Tensor([1]).cuda(),
                                                                 rotation_angle=torch.Tensor([current_angle]).cuda(),
                                                                 return_flow_grid=True,
                                                                 flag_interpolation_mode='bilinear')
            # output_tensors_list.append(output_tensor.cpu())
            # imshow_torch_video(output_tensor.cpu(), 500, 50, False, 5)

            ### FFT Analysis: ###
            suspects_list = FFT_Analysis_TimeBatch(output_tensor, params)



### TEMP - used to spot specific point in space: ###
def Total_FFT_Analysis(input_tensor, input_tensor_BG, params):
    T,H,W = input_tensor.shape

    ### Perform FFT per pixel: ###
    FPS = params.FPS
    input_tensor = torch.Tensor(input_tensor).cuda()
    frequency_axis = FPS * np.linspace(-0.5, 0.5 - 1 / T, T)
    frequency_axis_delta = frequency_axis[2] - frequency_axis[1]
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=0)
    input_tensor_fft = fftshift_torch_specific_dim(input_tensor_fft, 0)

    # ### Energy Between Frequency Bins: ###  #TODO: temp, delete and get most parameters from dict
    # frequency_range_for_drone = 1  #the peak size in Hz allocated for a drone (instead of just choosing the max value))
    # frequency_range_number_of_bins_for_drone = np.int(frequency_range_for_drone * frequency_axis_delta)
    # drone_start_bin = 90
    # drone_stop_bin = 120
    # noise_baseline_start_bin = 150
    # noise_baseline_stop_bin = 220

    ### Energy Between Frequency Bins: ###
    frequency_range_for_drone = params.frequency_range_for_drone  # the peak size in Hz allocated for a drone (instead of just choosing the max value))
    drone_start_bin = params.drone_start_bin
    drone_stop_bin = params.drone_stop_bin
    noise_baseline_start_bin = params.noise_baseline_start_bin
    noise_baseline_stop_bin = params.noise_baseline_stop_bin
    frequency_range_number_of_bins_for_drone = np.int(frequency_range_for_drone * frequency_axis_delta)

    ### Get Proper Indices: ###
    indices_arange = np.arange(len(frequency_axis))
    logical_mask_default = np.ones_like(indices_arange).astype(np.bool)
    drone_frequency_bins_logical_mask = ((frequency_axis>=drone_start_bin).astype(np.uint8) * (frequency_axis<=drone_stop_bin).astype(np.uint8)).astype(np.bool)
    non_drone_frequency_bins_logical_mask = (1-drone_frequency_bins_logical_mask.astype(np.uint8)).astype(np.bool)
    noise_baseline_frequency_bins_logical_mask = ((frequency_axis>=noise_baseline_start_bin).astype(np.uint8) * (frequency_axis<=noise_baseline_stop_bin).astype(np.uint8)).astype(np.bool)
    drone_bins_indices = indices_arange[drone_frequency_bins_logical_mask]
    non_drone_bins_indices = indices_arange[non_drone_frequency_bins_logical_mask]
    noise_baseline_bins_indices = indices_arange[noise_baseline_frequency_bins_logical_mask]

    ### Get SNR Stats: ###
    #(1). For each pixel in defined pixels
    #Movie 3
    indices_H = np.arange(179-2, 179+2)
    indices_W = np.arange(354-2, 354+2)
    # #Movie 14, 0=with flashlight, 4000=no flashlight
    indices_H = np.arange(169-3, 169+3)
    indices_W = np.arange(318-3, 318+3)
    # #Movie 25: 2500 with flashlight, 0 = no flashlight
    indices_H = np.arange(34 - 2, 34 + 2)
    indices_W = np.arange(349 - 2, 349 + 2)
    #Movie 38: 0 = with flashlight, 2500 = no flashlight
    indices_H = np.arange(36 - 3, 36 + 3)
    indices_W = np.arange(455 - 3, 455 + 3)
    for H_index in indices_H:
        for W_index in indices_W:
            pixel_fft = torch.clone(input_tensor_fft[:, H_index, W_index]).abs()
            figure()
            plot(frequency_axis, pixel_fft.cpu().numpy())
            plt.show()
            plt.title('[' + str(H_index) + ', ' + str(W_index) + ']')
            current_y_lim = pixel_fft[pixel_fft.shape[0]//2+5:-1].max() * 2
            plt.ylim([0,current_y_lim.cpu().numpy()])
    plt.close('all')
    #(2). For specific pixels
    # pixel_fft = torch.clone(input_tensor_fft[:, 97:98, 322:326])  # Drone
    pixel_fft = torch.clone(input_tensor_fft[:, 180:181, 354:355])  # Drone
    # pixel_fft = torch.clone(input_tensor_fft[:, 263:264, 295:296])
    #(1). Get max lobe (mean) energy and frequencies between drone search area (around 100Hz)
    pixel_fft_abs = pixel_fft.abs().sum([-1, -2])  #notice abs->sum or sum->abs
    figure(); plot(frequency_axis, pixel_fft_abs.cpu().numpy()); plt.show()
    pixel_fft_drone_bins = pixel_fft_abs[drone_frequency_bins_logical_mask]  #TODO: for clarity maybe first do a binning on entire signal and then search for max around drone bins
    pixel_fft_drone_lobe_energy = fast_binning_1D_overap_flexible(pixel_fft_drone_bins, frequency_range_number_of_bins_for_drone,
                                                                  frequency_range_number_of_bins_for_drone-1, False)
    pixel_fft_drone_lobe_energy_max_index = pixel_fft_drone_lobe_energy.argmax()
    pixel_fft_drone_lobe_energy_max_value = pixel_fft_drone_lobe_energy[pixel_fft_drone_lobe_energy_max_index]
    pixel_fft_drone_lobe_energy_max_index += drone_bins_indices[0] #correct for starting place on frequency axis
    max_energy_band_indices = torch.arange(pixel_fft_drone_lobe_energy_max_index,pixel_fft_drone_lobe_energy_max_index+frequency_range_number_of_bins_for_drone)
    max_energy_band_frequencies = np.atleast_1d(frequency_axis[max_energy_band_indices])
    max_energy_band_logical_mask = ((frequency_axis>=max_energy_band_frequencies[0]).astype(np.uint8) * (frequency_axis<=max_energy_band_frequencies[-1]).astype(np.uint8)).astype(np.bool)
    #(2). Get noise energy mean energy
    # (*). get the mean value of the coherent sum of the fft components over frequency_range_number_of_bins_for_drone number of bins:
    pixel_fft_noise_estimation_bins = pixel_fft[noise_baseline_frequency_bins_logical_mask].sum([-1,-2])  #sum only over the specific pixels where drone is at / or where we wish to explore and was defined above
    pixel_fft_noise_estimation_bins_binned_real = fast_binning_1D_overap_flexible(pixel_fft_noise_estimation_bins.real,
                                                                                  frequency_range_number_of_bins_for_drone,
                                                                                  frequency_range_number_of_bins_for_drone - 1,
                                                                                  False)
    pixel_fft_noise_estimation_bins_binned_imag = fast_binning_1D_overap_flexible(pixel_fft_noise_estimation_bins.imag,
                                                                                  frequency_range_number_of_bins_for_drone,
                                                                                  frequency_range_number_of_bins_for_drone - 1,
                                                                                  False)
    pixel_fft_noise_estimation_bins_binned = pixel_fft_noise_estimation_bins_binned_real + 1j*pixel_fft_noise_estimation_bins_binned_imag
    pixel_fft_noise_estimation_mean = pixel_fft_noise_estimation_bins_binned.abs().mean()
    #(3). Calculate SNR:
    pixel_fft_max_lobe_SNR_around_drone_bins = pixel_fft_drone_lobe_energy_max_value / pixel_fft_noise_estimation_mean
    print(pixel_fft_max_lobe_SNR_around_drone_bins)


    # ### Show Things On Graph: ###
    pixel_fft_drone = torch.clone(pixel_fft_abs)
    pixel_fft_drone[drone_frequency_bins_logical_mask] = 1
    pixel_fft_drone[~drone_frequency_bins_logical_mask] = 0
    pixel_fft_non_drone = torch.clone(pixel_fft_abs)
    pixel_fft_non_drone[non_drone_frequency_bins_logical_mask] = 1
    pixel_fft_non_drone[~non_drone_frequency_bins_logical_mask] = 0
    pixel_fft_noise_baseline = torch.clone(pixel_fft_abs)
    pixel_fft_noise_baseline[noise_baseline_frequency_bins_logical_mask] = 1
    pixel_fft_noise_baseline[~noise_baseline_frequency_bins_logical_mask] = 0
    plt.plot(frequency_axis, pixel_fft_abs.abs().cpu().numpy())
    plt.plot(frequency_axis, pixel_fft_drone.abs().cpu().numpy())
    plt.plot(frequency_axis, pixel_fft_non_drone.abs().cpu().numpy())
    plt.plot(frequency_axis, pixel_fft_noise_baseline.abs().cpu().numpy())
    plt.legend(['fft','drone','non-drone','noise'])
    plt.show()

    ### Present Diffferent Spatial Bins: ###
    input_tensor_fft_between_drone_bins = (input_tensor_fft[max_energy_band_logical_mask].abs()**2).sum(0,True).abs()
    input_tensor_fft_between_noise_bins = (input_tensor_fft[noise_baseline_frequency_bins_logical_mask].abs()**2).mean(0,True).abs()

    input_tensor_fft_1X1_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_drone_bins, (1,1), (0,0))
    input_tensor_fft_2X2_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_drone_bins, (2,2), (1,1))
    input_tensor_fft_4X4_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_drone_bins, (4,4), (3,3))
    input_tensor_fft_8X8_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_drone_bins, (8,8), (7,7))
    input_tensor_fft_16X16_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_drone_bins, (16,16), (15,15))

    input_tensor_fft_1X1_non_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_noise_bins, (1, 1),(0, 0))
    input_tensor_fft_2X2_non_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_noise_bins, (2, 2),(1, 1))
    input_tensor_fft_4X4_non_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_noise_bins, (4, 4),(3, 3))
    input_tensor_fft_8X8_non_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_noise_bins, (8, 8),(7, 7))
    input_tensor_fft_16X16_non_drone_bins = fast_binning_2D_overap_flexible(input_tensor_fft_between_noise_bins, (16, 16), (15, 15))

    plt.figure(); plt.imshow((input_tensor_fft_1X1_drone_bins/input_tensor_fft_1X1_non_drone_bins)[0].cpu().numpy())
    plt.figure(); plt.imshow((input_tensor_fft_2X2_drone_bins/input_tensor_fft_2X2_non_drone_bins)[0].cpu().numpy())
    plt.figure(); plt.imshow((input_tensor_fft_4X4_drone_bins/input_tensor_fft_4X4_non_drone_bins)[0].cpu().numpy())
    plt.figure(); plt.imshow((input_tensor_fft_8X8_drone_bins/input_tensor_fft_8X8_non_drone_bins)[0].cpu().numpy())
    plt.figure(); plt.imshow((input_tensor_fft_16X16_drone_bins/input_tensor_fft_16X16_non_drone_bins)[0].cpu().numpy())

    drone_SNR_threshold = 60
    plt.figure();
    plt.imshow((input_tensor_fft_1X1_drone_bins / input_tensor_fft_1X1_non_drone_bins)[0].cpu().numpy()>drone_SNR_threshold)
    plt.figure();
    plt.imshow((input_tensor_fft_2X2_drone_bins / input_tensor_fft_2X2_non_drone_bins)[0].cpu().numpy()>drone_SNR_threshold)
    plt.figure();
    plt.imshow((input_tensor_fft_4X4_drone_bins / input_tensor_fft_4X4_non_drone_bins)[0].cpu().numpy()>drone_SNR_threshold)
    plt.figure();
    plt.imshow((input_tensor_fft_8X8_drone_bins / input_tensor_fft_8X8_non_drone_bins)[0].cpu().numpy()>drone_SNR_threshold)
    plt.figure();
    plt.imshow((input_tensor_fft_16X16_drone_bins / input_tensor_fft_16X16_non_drone_bins)[0].cpu().numpy()>drone_SNR_threshold)

    plt.show()
    suspects_list = None
    plt.close('all')
    return suspects_list


def FFT_Analysis_TimeBatch(input_tensor, params):
    #(*). this function gets in advance frequecny_range_for_drone and frequency_range_for_noise!!!! if i want something more flexible i need to make it myself
    T,C,H,W = input_tensor.shape

    ### Perform FFT per pixel: ###
    FPS = params.FPS
    if type(input_tensor) == np.ndarray:
        input_tensor = torch.Tensor(input_tensor).cuda()
    frequency_axis = FPS * np.linspace(-0.5, 0.5 - 1 / T, T)
    frequency_axis_delta = frequency_axis[2] - frequency_axis[1]
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=0)
    input_tensor_fft = fftshift_torch_specific_dim(input_tensor_fft, 0)

    ### Get Params From Dictionary (Energy Between Frequency Bins Stuff): ###
    frequency_range_for_drone = params.frequency_range_for_drone  # the peak size in Hz allocated for a drone (instead of just choosing the max value))
    drone_start_bin = params.drone_start_bin
    drone_stop_bin = params.drone_stop_bin
    noise_baseline_start_bin = params.noise_baseline_start_bin
    noise_baseline_stop_bin = params.noise_baseline_stop_bin
    frequency_range_number_of_bins_for_drone = np.int(frequency_range_for_drone * frequency_axis_delta)

    ### Get Proper Indices For The Different "Search Locations": ###
    indices_arange = np.arange(len(frequency_axis))
    logical_mask_default = np.ones_like(indices_arange).astype(np.bool)
    #(1). Logical Masks:
    drone_frequency_bins_logical_mask = ((frequency_axis>=drone_start_bin).astype(np.uint8) * (frequency_axis<=drone_stop_bin).astype(np.uint8)).astype(np.bool)
    non_drone_frequency_bins_logical_mask = (1-drone_frequency_bins_logical_mask.astype(np.uint8)).astype(np.bool)
    noise_baseline_frequency_bins_logical_mask = ((frequency_axis>=noise_baseline_start_bin).astype(np.uint8) * (frequency_axis<=noise_baseline_stop_bin).astype(np.uint8)).astype(np.bool)
    #(2). Indices Themselves:
    drone_bins_indices = indices_arange[drone_frequency_bins_logical_mask]
    non_drone_bins_indices = indices_arange[non_drone_frequency_bins_logical_mask]
    noise_baseline_bins_indices = indices_arange[noise_baseline_frequency_bins_logical_mask]

    #(3). Get max lobe energy and frequencies between drone search area (around 100Hz)
    #(*). get max lobe energy per pixel within possible frequency range (about 90-120Hz):
    input_tensor_fft_possible_drone_bins_only = input_tensor_fft.abs()[drone_frequency_bins_logical_mask]  #TODO: for clarity maybe first do a binning on entire signal and then search for max around drone bins
    input_tensor_fft_possible_drone_bins_binned = fast_binning_1D_overap_flexible(input_tensor_fft_possible_drone_bins_only,
                                                                  frequency_range_number_of_bins_for_drone,
                                                                  frequency_range_number_of_bins_for_drone-1, False) #(*). returns SUM, not MEAN
    input_tensor_fft_possible_drone_bins_max_lobe_index = torch.argmax(input_tensor_fft_possible_drone_bins_binned, 0)
    input_tensor_fft_possible_drone_bine_max_lobe_value = torch.gather(input_tensor_fft_possible_drone_bins_binned, 0, input_tensor_fft_possible_drone_bins_max_lobe_index.unsqueeze(0))
    input_tensor_fft_possible_drone_bins_max_lobe_index += drone_bins_indices[0] #correct for starting place on frequency axis
    # #(4). get lobe energy per pixel between PREDEFINED, NARROW drone range:
    # input_tensor_fft_possible_drone_bins_max_lobe_index = 360
    #TODO: switch to input_tensor_fft.abs()[drone_bins_indices].sum(0,True) later on as the "predetermined frequency bins strategy"
    input_tensor_fft_possible_drone_bine_max_lobe_value = input_tensor_fft.abs()[input_tensor_fft_possible_drone_bins_max_lobe_index :
                                                                                 input_tensor_fft_possible_drone_bins_max_lobe_index + frequency_range_number_of_bins_for_drone].sum(0,True)

    #(4). Get noise energy mean energy
    # (*). get the mean value of the coherent sum of the fft components over frequency_range_number_of_bins_for_drone number of bins:
    #TODO: make this more robust and per pixel, currently uses coherent sums, shouuld i use that?
    input_tensor_fft_noise_estimation_bins = input_tensor_fft[noise_baseline_frequency_bins_logical_mask]
    input_tensor_fft_noise_estimation_bins_binned_real = fast_binning_1D_overap_flexible(input_tensor_fft_noise_estimation_bins.real,
                                                                                  frequency_range_number_of_bins_for_drone,
                                                                                  frequency_range_number_of_bins_for_drone - 1,
                                                                                  True, 0)  #return average, not sums!
    input_tensor_fft_noise_estimation_bins_binned_imag = fast_binning_1D_overap_flexible(input_tensor_fft_noise_estimation_bins.imag,
                                                                                  frequency_range_number_of_bins_for_drone,
                                                                                  frequency_range_number_of_bins_for_drone - 1,
                                                                                  True, 0) #return average, not sums!
    input_tensor_fft_noise_estimation_bins_binned = input_tensor_fft_noise_estimation_bins_binned_real + 1j*input_tensor_fft_noise_estimation_bins_binned_imag
    input_tensor_fft_noise_estimation_mean = input_tensor_fft_noise_estimation_bins_binned.abs().mean(0, True) #return the average of lobe-sized sums
    #(3). Calculate SNR:
    #TODO: change the calculation to use the entire drone lobe instead of just the max value
    input_tensor_fft_max_lobe_SNR_around_drone_bins = input_tensor_fft_possible_drone_bine_max_lobe_value / input_tensor_fft_noise_estimation_mean.cuda()

    # # ### Show Things On Graph: ###
    # input_tensor_fft_drone = torch.clone(input_tensor_fft)
    # input_tensor_fft_drone[drone_frequency_bins_logical_mask] = 1
    # input_tensor_fft_drone[~drone_frequency_bins_logical_mask] = 0
    # input_tensor_fft_non_drone = torch.clone(input_tensor_fft)
    # input_tensor_fft_non_drone[non_drone_frequency_bins_logical_mask] = 1
    # input_tensor_fft_non_drone[~non_drone_frequency_bins_logical_mask] = 0
    # input_tensor_fft_noise_baseline = torch.clone(input_tensor_fft)
    # input_tensor_fft_noise_baseline[noise_baseline_frequency_bins_logical_mask] = 1
    # input_tensor_fft_noise_baseline[~noise_baseline_frequency_bins_logical_mask] = 0
    # plt.plot(frequency_axis, input_tensor_fft.abs().cpu().numpy())
    # plt.plot(frequency_axis, input_tensor_fft_drone.abs().cpu().numpy())
    # plt.plot(frequency_axis, input_tensor_fft_non_drone.abs().cpu().numpy())
    # plt.plot(frequency_axis, input_tensor_fft_noise_baseline.abs().cpu().numpy())
    # plt.legend(['fft','drone','non-drone','noise'])
    # plt.show()

    # ### Get BG estimation and Canny Edge Detection To Get Rid Of Image Edges which can give frequencies similar to drone because of camera vibrations: ###
    # #TODO: understand the "canny threshold" better, perhapse make it so it will be automatic. the same goes for the simple binary threshold
    # canny_edge_detection_layer = canny_edge_detection(use_cuda=input_tensor.is_cuda)
    # #(1). Get BG Estimation:
    # input_tensor_median = torch.median(input_tensor, 0)[0].unsqueeze(0)
    # input_tensor_BG_substracted = (input_tensor - input_tensor_median).abs()
    # #(2). Perform Canny Edge Detection On Median Image:
    # #TODO: try using morphological closing or something else to avoid isolated, single pixel, "fake edges"
    # canny_threshold_BG = 10
    # blurred_img_BG, grad_mag_BG, grad_orientation_BG, thin_edges_BG, thresholded_BG, early_threshold_BG = \
    #     canny_edge_detection_layer(BW2RGB(input_tensor_median), canny_threshold_BG)
    # binary_threshold_BG = 2
    # early_threshold_binary_BG = (early_threshold_BG > binary_threshold_BG)
    # thresholded_binary_BG = (thresholded_BG > binary_threshold_BG).float()
    # imshow_torch(thresholded_binary_BG)
    # imshow_torch(input_tensor_median)
    # #(3). Perform Canny Edge Detection On BG-Substracted Image:
    # canny_threshold_BGSubstracted = 1.0
    # blurred_img_BGSubstracted, grad_mag_BGSubstracted, grad_orientation_BGSubstracted, thin_edges_BGSubstracted, thresholded_BGSubstracted, early_threshold_BGSubstracted = \
    #     canny_edge_detection_layer(BW2RGB(input_tensor_BG_substracted.median(0)[0].unsqueeze(0)), canny_threshold_BGSubstracted)
    # # imshow_torch(thresholded_BGSubstracted)
    # binary_threshold_BGSubstracted = 1
    # early_threshold_binary_BGSubstracted = (early_threshold_BGSubstracted > binary_threshold_BGSubstracted)
    # thresholded_binary_AlignedBGSubstracted = (thresholded_BGSubstracted > binary_threshold_BGSubstracted).float()
    # # imshow_torch(thresholded_binary_AlignedBGSubstracted)
    # #(4). Perform Binary Thresholding On BG-Substracted Image:
    # binary_threshold_BG_substraction = 0.2
    # current_aligned_BG_substracted_binary = (input_tensor_BG_substracted > binary_threshold_BG_substraction).float()
    # # imshow_torch(current_aligned_BG_substracted_binary)
    # #(5). Perform Dilation On Canny Edge:
    # thresholded_binary_BG_dilated = kornia.morphology.dilation(thresholded_binary_BG, torch.ones(6, 6).cuda())

    # ### Show Results: ###
    # imshow_torch(input_tensor_median)
    # imshow_torch(input_tensor_BG_substracted[0])
    # imshow_torch(input_tensor_BG_substracted.mean(0))
    # imshow_torch(input_tensor_BG_substracted.median(0)[0].unsqueeze(0))
    # imshow_torch(thresholded_binary_BG)
    # imshow_torch(thresholded_binary_BG_dilated)
    # imshow_torch(thresholded_BGSubstracted)
    # imshow_torch(thresholded_binary_AlignedBGSubstracted)


    ### Present Diffferent Spatial Bins: ###
    #(1). Get relevant figures for "signal" and "noise":
    #TODO:
    #(****). Non Coherent SUMS:
    input_tensor_fft_between_drone_bins = (input_tensor_fft[input_tensor_fft_possible_drone_bins_max_lobe_index].abs()**2).mean(0, True).abs()
    input_tensor_fft_between_noise_bins = (input_tensor_fft[noise_baseline_frequency_bins_logical_mask].abs()**2).mean(0, True).abs().squeeze(1)
    # #(****). Coherent SUMS:
    # input_tensor_fft_between_drone_bins = (input_tensor_fft[input_tensor_fft_possible_drone_bins_max_lobe_index].abs()).mean(0, True) ** 2
    # input_tensor_fft_between_noise_bins = (input_tensor_fft[noise_baseline_frequency_bins_logical_mask]).mean(0, True).abs().squeeze(1) ** 2
    #(2). perform binning:
    #(*). NOTICE: controllable binning on both axes!!!
    input_tensor_fft_1X1_drone_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_drone_bins, (1,1), (0,0))
    input_tensor_fft_2X2_drone_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_drone_bins, (2,2), (1,1))
    input_tensor_fft_4X4_drone_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_drone_bins, (4,4), (3,3))
    input_tensor_fft_1X1_noise_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_noise_bins, (1, 1),(0, 0))
    input_tensor_fft_2X2_noise_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_noise_bins, (2, 2),(1, 1))
    input_tensor_fft_4X4_noise_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_noise_bins, (4, 4),(3, 3))
    #(3). show images of the "Raw" "SNR" results:
    plt.figure(); plt.imshow((input_tensor_fft_1X1_drone_bins/input_tensor_fft_1X1_noise_bins)[0].cpu().numpy())
    plt.figure(); plt.imshow((input_tensor_fft_2X2_drone_bins/input_tensor_fft_2X2_noise_bins)[0].cpu().numpy())
    plt.figure(); plt.imshow((input_tensor_fft_4X4_drone_bins/input_tensor_fft_4X4_noise_bins)[0].cpu().numpy())
    #(4). show thresholded image for SNR:
    drone_SNR_threshold = 10
    input_tensor_fft_1X1_binning_SNR_above_threshold = (input_tensor_fft_1X1_drone_bins / input_tensor_fft_1X1_noise_bins)[0] > drone_SNR_threshold
    input_tensor_fft_2X2_binning_SNR_above_threshold = (input_tensor_fft_2X2_drone_bins / input_tensor_fft_2X2_noise_bins)[0] > drone_SNR_threshold
    input_tensor_fft_4X4_binning_SNR_above_threshold = (input_tensor_fft_4X4_drone_bins / input_tensor_fft_4X4_noise_bins)[0] > drone_SNR_threshold
    imshow_torch(input_tensor_fft_1X1_binning_SNR_above_threshold)
    imshow_torch(input_tensor_fft_2X2_binning_SNR_above_threshold)
    imshow_torch(input_tensor_fft_4X4_binning_SNR_above_threshold)
    #(5). Get rid of points which are suspected to be a part of the view and gave us a "false signal" due to vibration:
    input_tensor_fft_1X1_binning_SNR_above_threshold_minus_BG = input_tensor_fft_1X1_binning_SNR_above_threshold * (1 - thresholded_binary_BG_dilated)
    input_tensor_fft_2X2_binning_SNR_above_threshold_minus_BG = input_tensor_fft_2X2_binning_SNR_above_threshold * (1 - thresholded_binary_BG_dilated)
    input_tensor_fft_4X4_binning_SNR_above_threshold_minus_BG = input_tensor_fft_4X4_binning_SNR_above_threshold * (1 - thresholded_binary_BG_dilated)
    imshow_torch(input_tensor_fft_1X1_binning_SNR_above_threshold)
    imshow_torch(input_tensor_fft_1X1_binning_SNR_above_threshold_minus_BG)
    imshow_torch(input_tensor_fft_2X2_binning_SNR_above_threshold)
    imshow_torch(input_tensor_fft_2X2_binning_SNR_above_threshold_minus_BG)
    imshow_torch(input_tensor_median)
    imshow_torch(thresholded_binary_BG_dilated)
    #(6). get spectrum graphs of several points which passed threshold:
    matches_tensor = torch_get_where_condition_holds(input_tensor_fft_1X1_binning_SNR_above_threshold==True)
    number_of_matches, number_of_dims = matches_tensor.shape
    ffts_list = []
    for match_index in np.arange(number_of_matches):
        current_H = matches_tensor[match_index, 0]
        current_W = matches_tensor[match_index, 1]
        ffts_list.append(input_tensor_fft[:,:,current_H, current_W])


    ### Plot Spectrum Graphs of all the points which "survived" as potential suspects
    graphs_folder = os.path.join(params.results_folder_seq, 'FFT_Graphs')
    create_folder_if_doesnt_exist(graphs_folder)
    for i in np.arange(number_of_matches):
        figure()
        plt.plot(frequency_axis, ffts_list[i].abs().squeeze().cpu().numpy().clip(0,10))
        title_string = str(matches_tensor[i].cpu().numpy())
        plt.title(title_string)
        filename = os.path.join(graphs_folder, str(i).zfill(4) + '.png')
        plt.savefig(filename)
        plt.close()

    suspects_list = None
    return suspects_list

def unify_close_frequencies(final_frequency_peaks_list):
    ### Analyse Found Frequencies: ###
    sorted_frequencies_list = sort(final_frequency_peaks_list)
    sorted_frequencies_tensor = torch.tensor(sorted_frequencies_list)
    sorted_frequencies_tensor_diff = torch.diff(sorted_frequencies_tensor)
    sorted_frequencies_tensor_diff_below_threshold = sorted_frequencies_tensor_diff <= 3
    ### Average Out & Count Similar Frequencies: ###
    flag_big_loop_continue = True
    global_counter = 0
    global_indices_list = []
    global_frequencies_list = []
    global_average_frequencies_list = []
    while flag_big_loop_continue:
        current_flag_below_threshold = sorted_frequencies_tensor_diff_below_threshold[global_counter]
        flag_internal_loop_continue = current_flag_below_threshold
        indices_list = []
        frequencies_list = []
        while flag_internal_loop_continue:
            indices_list.append(global_counter)
            frequencies_list.append(sorted_frequencies_tensor[global_counter].item())
            global_counter += 1
            flag_internal_loop_continue = sorted_frequencies_tensor_diff_below_threshold[global_counter]
        indices_list.append(global_counter)
        frequencies_list.append(sorted_frequencies_tensor[global_counter].item())
        indices_tensor = torch.tensor(indices_list)
        frequencies_tensor = torch.tensor(frequencies_list)
        average_frequency = frequencies_tensor.mean()
        # sorted_frequencies_tensor_diff_below_threshold[indices_tensor] = average_frequency
        print(frequencies_tensor)

        ### Update Global/Average Lists: ###
        global_indices_list.append(indices_list)
        global_frequencies_list.append(frequencies_list)
        global_average_frequencies_list.append(average_frequency)

        ### Uptick global counter: ###
        global_counter += 1
        flag_big_loop_continue = (global_counter < len(sorted_frequencies_tensor_diff_below_threshold) - 1)


def plot_fft_graphs(input_tensor, H_center=None, W_center=None, area_around_center=None, specific_case_string='', params=None):
    # input_tensor = Movie_BGS[t_vec.type(torch.LongTensor)]
    # input_tensor = Movie_BGS
    # input_tensor = Movie[t_vec.type(torch.LongTensor)]
    # input_tensor = TrjMov.unsqueeze(1)
    # H_center = 279
    # W_center = 251
    # area_around_center = 3
    # specific_case_string = 'TrjMovie_BGS_1_pixel_right_constant'

    # ### Start from a certain frame (mostly for testing purposes if i don't want to start from the first frame): ###
    # frame_to_start_from = 500 * 5
    # f = initialize_binary_file_reader(f)
    # Movie_temp = read_frames_from_binary_file_stream(f, 1000, frame_to_start_from * 1, params)
    # Movie_temp = Movie_temp.astype(float)
    # # Movie_temp = scale_array_from_range(Movie_temp.clip(q1, q2),
    # #                                        min_max_values_to_clip=(q1, q2),
    # #                                        min_max_values_to_scale_to=(0, 1))
    # # Movie_temp = scale_array_stretch_hist(Movie_temp)
    # input_tensor = torch.tensor(Movie_temp).unsqueeze(1)

    # ### Read Certain Area: ###
    # frame_to_start_from = 500 * 5
    # f = initialize_binary_file_reader(f)
    # Movie_temp = read_frames_from_binary_file_stream_SpecificArea(f,
    #                                                                       number_of_frames_to_read=1010,
    #                                                                       number_of_frames_to_skip=frame_to_start_from * 1,
    #                                                                       params=params,
    #                                                                       center_HW=(279,251),
    #                                                                       area_around_center=10)
    # # Movie_temp = scale_array_from_range(Movie_temp.clip(q1, q2),
    # #                                        min_max_values_to_clip=(q1, q2),
    # #                                        min_max_values_to_scale_to=(0, 1))
    # # Movie_temp = scale_array_stretch_hist(Movie_temp)
    # input_tensor = torch.tensor(Movie_temp).unsqueeze(1)
    # imshow_torch_video(input_tensor, FPS=50, frame_stride=5)

    ### Get Drone Area: ###
    # Drone = [279, 251], Edge = [302, 221]
    if H_center is not None:
        i_vec = np.linspace(H_center - area_around_center, H_center + area_around_center, area_around_center * 2 + 1).astype(int)
        j_vec = np.linspace(W_center - area_around_center, W_center + area_around_center, area_around_center * 2 + 1).astype(int)
    else:
        i_vec = np.arange(input_tensor.shape[-2])
        j_vec = np.arange(input_tensor.shape[-1])

    ### Get FFT Over Entire Image: ###
    # imshow_torch_video(input_tensor, FPS=50)
    # input_tensor_fft = (torch.fft.rfftn(input_tensor, dim=0).abs())
    # input_tensor_fft = fftshift_torch(torch.fft.fftn(input_tensor, dim=0).abs(),0)
    # T,C,H,W = input_tensor_fft.shape

    ### Get FFT Over Specific Drone Area: ###
    input_tensor_drone = input_tensor[:, :, i_vec[0]:i_vec[-1] + 1, j_vec[0]:j_vec[-1] + 1].cuda()
    input_tensor_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_drone, dim=0).abs(), 0)
    T = input_tensor_fft.shape[0]

    ### Create Proper Folders: ###
    ffts_folder = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New')
    ffts_folder_before_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'Before_Conditions')
    ffts_folder_after_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'After_Conditions')
    path_create_path_if_none_exists(ffts_folder)
    path_create_path_if_none_exists(ffts_folder_before_conditions)
    path_create_path_if_none_exists(ffts_folder_after_conditions)

    ### Loop Over Individual Indices: ###
    final_frequency_peaks_list = []
    for i in np.arange(len(i_vec)):
        for j in np.arange(len(j_vec)):
            ### Initialize Tensor: ###
            # Drone = [279, 251], Edge = [302, 221]
            i = int(i)
            j = int(j)
            # input_tensor_fft_graph = input_tensor_fft[:, 0, int(i), int(j)]
            # input_tensor_fft_graph = convn_torch(input_tensor_fft_graph, torch.ones(3)/3, dim=0).squeeze()
            # input_tensor_fft_graph = input_tensor_fft_graph.clamp(0, 1)
            FPS = params.FPS
            frequency_axis = torch.tensor(FPS * np.linspace(-0.5, .5 - 1 / T, T))
            frequency_axis_numpy = frequency_axis.cpu().numpy()

            ### Initialize graphs location and string: ###
            graph_string = specific_case_string + '   ' + 'H=' + str(int(i_vec[i])) + '_W=' + str(int(j_vec[j]))

            ### Get Current 1D FFT-Vec: ###
            input_vec = input_tensor_fft[:, 0:1, int(i):int(i + 1), int(j):int(j + 1)].abs().cpu()
            # input_vec_save_full_filename = os.path.join(ffts_folder_before_conditions, graph_string + '.npy')
            # np.save(input_vec_save_full_filename, input_vec.cpu().numpy(), allow_pickle=True)

            ### Peak Detect: ###
            # input_vec = convn_torch(input_vec, torch.ones(3)/3, dim=0)
            maxima_peaks, arange_in_correct_dim_tensor = peak_detect_pytorch(input_vec,
                                                                             window_size=21,
                                                                             dim=0,
                                                                             flag_use_ratio_threshold=True,
                                                                             ratio_threshold=3,
                                                                             flag_plot=True)
            # (*). Only Keep Peaks Above Median Noise-Floor Enough:
            median_noise_floor = input_vec.median().item()
            SNR_threshold = median_noise_floor * 0
            logical_mask_above_noise_median = input_vec > SNR_threshold
            maxima_peaks = maxima_peaks * logical_mask_above_noise_median

            ######### Plot Graphs Between Harmonics Conditions: #########
            ### Get All Peak Frequencies: ###
            maxima_peaks_to_plot = (arange_in_correct_dim_tensor)[maxima_peaks]
            maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_to_plot] > 0
            peak_frequencies_list = frequency_axis[maxima_peaks_to_plot][maxima_frequency_peaks_logical_mask]
            peak_frequencies_list = peak_frequencies_list.tolist()
            peak_frequencies_array = np.array(peak_frequencies_list)
            ### Legends list: ###
            legends_list = copy.deepcopy(peak_frequencies_list)
            for legend_index in np.arange(len(peak_frequencies_list)):
                legends_list[legend_index] = decimal_notation(peak_frequencies_list[legend_index], 1)
            legends_list = ['FFT', 'peaks', 'Noise Floor', 'Threshold'] + legends_list
            ### Plot Results: this is a specific case of tchw tensor, chosen dim t: ###
            k1 = 0
            k2 = 0
            k3 = 0
            maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, k1, k2, k3]])[:, k1, k2, k3]
            plt.figure()
            plt.plot(frequency_axis_numpy, input_vec[:, k1, k2, k3].clamp(0, 1).numpy())
            # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
            plt.plot(frequency_axis[maxima_peaks_to_plot], input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0, 1).numpy(), '.')
            plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * median_noise_floor)
            plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold)
            plt.title(graph_string)
            for current_legend in legends_list:
                plt.plot(frequency_axis_numpy, frequency_axis_numpy * 0)
            plt.legend(legends_list)
            ### Set Y limits: ###
            plt.ylim([0,1])
            ### Save Graph: ###
            save_string = os.path.join(ffts_folder_before_conditions, graph_string + '.png')
            plt.savefig(save_string)
            plt.close('all')

            # ########### Plot Graphs After Harmonics Conditions: ###############
            # # (*). Get Rid Of Peaks Which Are Harmonies Of The Base 23[Hz]:
            # base_frequency = 23
            # base_frequency_harmonic_tolerance = 1.5
            # frequency_axis_remainder_from_base_frequency = torch.remainder(frequency_axis, base_frequency)
            # # Instead of creating an array of [23,23,23.....,46,46,46,....] and getting the diff i simply check by a different, maybe stupid way for close harmonics
            # frequency_axis_modulo_base_logical_mask = frequency_axis_remainder_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from above
            # frequency_axis_diff_from_base_frequency = (frequency_axis_remainder_from_base_frequency - base_frequency).abs()
            # frequency_axis_modulo_base_logical_mask *= frequency_axis_diff_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from below
            # frequency_axis_modulo_base_logical_mask = frequency_axis_modulo_base_logical_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # maxima_peaks = maxima_peaks * frequency_axis_modulo_base_logical_mask
            # # (*). Get Rid Of Negative Frequencies:
            # maxima_peaks[0:T // 2] = False
            #
            # ### Get All Peak Frequencies: ###
            # maxima_peaks_to_plot = (arange_in_correct_dim_tensor)[maxima_peaks]
            # maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_to_plot] > 0
            # peak_frequencies_list = frequency_axis[maxima_peaks_to_plot][maxima_frequency_peaks_logical_mask]
            # peak_frequencies_list = peak_frequencies_list.tolist()
            # peak_frequencies_array = np.array(peak_frequencies_list)
            # ### Record Peak Frequencies: ###
            # for peak_frequency in peak_frequencies_list:
            #     final_frequency_peaks_list.append(round_to_nearest_half(peak_frequency))
            # ### Legends list: ###
            # legends_list = copy.deepcopy(peak_frequencies_list)
            # for legend_index in np.arange(len(peak_frequencies_list)):
            #     legends_list[legend_index] = decimal_notation(peak_frequencies_list[legend_index], 1)
            # legends_list = ['FFT', 'peaks', 'Noise Floor', 'Threshold'] + legends_list
            # ### Plot Results: this is a specific case of tchw tensor, chosen dim t: ###
            # k1 = 0
            # k2 = 0
            # k3 = 0
            # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, k1, k2, k3]])[:, k1, k2, k3]
            # plt.figure()
            # plt.plot(frequency_axis_numpy, input_vec[:, k1, k2, k3].clamp(0, 1).numpy())
            # # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
            # plt.plot(frequency_axis[maxima_peaks_to_plot], input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0, 1).numpy(), '.')
            # plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * median_noise_floor)
            # plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold)
            # plt.title(graph_string)
            # for current_legend in legends_list:
            #     plt.plot(frequency_axis_numpy, frequency_axis_numpy * 0)
            # plt.legend(legends_list)
            # ### Save Graph: ###
            # save_string = os.path.join(ffts_folder_after_conditions, graph_string + '.png')
            # plt.savefig(save_string)
            # plt.close('all')

            # ### Save Array: ###
            # np.save(os.path.join(ffts_folder, graph_string + '.npy'), input_tensor_fft_graph.cpu().numpy())

            # ### Save Graph: ###
            # plt.figure()
            # plot_torch(frequency_axis, input_tensor_fft_graph)
            # plt.title(graph_string)
            # plt.show()
            # save_string = os.path.join(ffts_folder, graph_string + '.png')
            # plt.savefig(save_string)
            # plt.close('all')
    plt.close('all')


def plot_fft_graphs_original(flag_was_drone_found, params, trajectory_index,
                             input_tensor_fft,
                             input_tensor_fft_max_lobe_SNR_around_drone_bins,
                             input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough,
                             input_tensor_fft_noise_estimation_mean,
                             frequency_axis,
                             input_tensor_fft_possible_drone_bine_max_lobe_value,
                             drone_frequencies_axis,
                             noise_frequency_axis):
    flag_save = (flag_was_drone_found == True and params.was_there_drone == 'False')
    flag_save = flag_save or (flag_was_drone_found == False and params.was_there_drone == 'True')
    if params.flag_save_interim_graphs_and_movies:
        # TODO: add condition that if sum of low frequencies is very large, or that we can fit a lorenzian/gaussian on the low frequencies then it's probably a bird
        # (*). All Graphs:
        print('DRONE FOUND, SAVING FFT GRAPHS')
        flag_drone_found_but_no_drone_there = (flag_was_drone_found == True and params.was_there_drone == 'False')
        flag_drone_found_and_is_there = (flag_was_drone_found == True and params.was_there_drone == 'True')
        flag_drone_not_found_but_is_there = (flag_was_drone_found == False and params.was_there_drone == 'True')
        flag_drone_not_found_and_is_not_there = (flag_was_drone_found == False and params.was_there_drone == 'False')
        if flag_drone_not_found_but_is_there:
            post_string = '_Drone_Not_Found_But_Is_There(FT)'
        elif flag_drone_found_but_no_drone_there:
            post_string = '_Drone_Found_But_Is_Not_There(TF)'
        elif flag_drone_found_and_is_there:
            post_string = '_Drone_Found_And_Is_There(TT)'
        elif flag_drone_not_found_and_is_not_there:
            post_string = '_Drone_Not_Found_And_Is_Not_There(FF)'
        else:
            post_string = ''
        post_string = ''
        graphs_folder_unclipped = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_Unclipped')
        graphs_folder_clipped = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_clipped')
        graphs_folder_fft_binned = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_Binned_Clipped')
        path_make_path_if_none_exists(graphs_folder_fft_binned)
        path_make_path_if_none_exists(graphs_folder_unclipped)
        path_make_path_if_none_exists(graphs_folder_clipped)
        fft_counter = 0
        ROI_H, ROI_W = input_tensor_fft_max_lobe_SNR_around_drone_bins.shape[-2:]
        for roi_H in np.arange(ROI_H):
            for roi_W in np.arange(ROI_W):
                current_fft = input_tensor_fft[:, roi_H, roi_W].abs()
                current_SNR = input_tensor_fft_max_lobe_SNR_around_drone_bins[0, roi_H, roi_W].item()
                current_decision = input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough[0, roi_H, roi_W].item()

                # ### Unclipped Graph: ##
                # figure()
                # plt.plot(frequency_axis, current_fft.abs().squeeze().cpu().numpy())
                # plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(drone_frequencies_axis, (+0.003) * np.ones_like(drone_frequencies_axis))
                # plt.plot(noise_frequency_axis, (+0.003 * 2) * np.ones_like(noise_frequency_axis))
                # plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
                # title_string = str('ROI = [' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(np.round(current_SNR * 10) / 10) + ', Decision = ' + str(current_decision))
                # file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
                # plt.title(title_string)
                # filename = os.path.join(graphs_folder_unclipped, file_string + '.png')
                # plt.savefig(filename)
                # plt.close()

                ### clipped Graph: ##
                figure()
                plt.plot(frequency_axis, current_fft.abs().squeeze().cpu().numpy().clip(0, 10))
                plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                plt.plot(drone_frequencies_axis, (+0.003) * np.ones_like(drone_frequencies_axis))
                plt.plot(noise_frequency_axis, (+0.003 * 2) * np.ones_like(noise_frequency_axis))
                plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
                title_string = str('[' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(current_SNR) + ', Decision = ' + str(current_decision))
                file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
                plt.title(title_string)
                ### Set Y limits: ###
                # plt.ylim([0, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * 3])
                plt.ylim([0, 1])
                filename = os.path.join(graphs_folder_clipped, file_string + '.png')
                plt.savefig(filename)
                plt.close()

                # ### Binned clipped Graph: ##
                # figure()
                # plt.plot(frequency_axis, input_tensor_fft_binned[:,roi_H, roi_W].abs().squeeze().cpu().numpy().clip(0, 10))
                # plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
                # plt.plot(drone_frequencies_axis, 0 * np.ones_like(drone_frequencies_axis))
                # plt.plot(noise_frequency_axis, (-0.003) * np.ones_like(noise_frequency_axis))
                # plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
                # title_string = str('[' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(current_SNR) + ', Decision = ' + str(current_decision))
                # file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
                # plt.title(title_string)
                # plt.ylim([0, 1])
                # filename = os.path.join(graphs_folder_clipped, file_string + '.png')
                # plt.savefig(filename)
                # plt.close()

                fft_counter += 1

                plt.close('all')
                plt.pause(0.1)

def FFT_Analysis_Dudy_PredeterminedRanges(input_trajectories, params):
    ### Logistic Function: ###
    def logistic_torch(input_val, reference_val=5):
        # return (1. / (1 + torch.exp(-0.4 * (input_val - reference_val))) - 0.5).abs() * 2
        # return (1. / (1 + torch.exp(-1 * (torch.log(input_val)/reference_val)))).abs()
        return (1. / (1 + torch.exp(-5 * (torch.log(input_val / reference_val)))))  # TODO: why the -5???

    ### PreAllocate and initialize lists which hold decision results for each trajectory: ###
    #TODO: put into params dict instead of reinitializing
    number_of_trajectories = len(input_trajectories)
    TrjMovie_FFT_BinPartitioned_AfterScoreFunction = list(np.zeros(number_of_trajectories))
    pxl_scr = list(np.zeros(number_of_trajectories))
    DetectionConfLvl = list(np.zeros(number_of_trajectories))
    DetectionDec = list(np.zeros(number_of_trajectories))
    frequency_axis_per_trajectory = list(np.zeros(number_of_trajectories))

    ### Loop over the different trajectories (TrjMov)
    for trajectory_index in np.arange(len(input_trajectories)):
        input_tensor = input_trajectories[trajectory_index]

        ###########################################################################################################3
        #(*). this function gets in advance frequecny_range_for_drone and frequency_range_for_noise!!!!
        T, H, W = input_tensor.shape

        ### Perform FFT per pixel: ###
        FPS = params.FPS
        if type(input_tensor) == np.ndarray:
            input_tensor = torch.Tensor(input_tensor).cuda()

        frequency_axis = FPS * np.linspace(-0.5, 0.5 - 1 / T, T)
        frequency_axis_torch = torch.tensor(frequency_axis).to(input_tensor.device)
        frequency_axis_per_trajectory.append(frequency_axis_torch) #TODO: check this works
        frequency_axis_delta = frequency_axis[2] - frequency_axis[1]
        input_tensor_fft = torch.fft.fftn(input_tensor, dim=0)
        input_tensor_fft = fftshift_torch_specific_dim(input_tensor_fft, 0)  #TODO: if this is too slow i can simply use only half of values of course

        ### Get Params From Dictionary (Energy Between Frequency Bins Stuff): ###
        frequency_range_for_drone = params.frequency_range_for_drone  # the peak size in Hz allocated for a drone (instead of just choosing the max value))
        #(1). drone range:
        drone_start_bin = params.drone_start_bin
        drone_stop_bin = params.drone_stop_bin
        drone_frequencies_axis = np.arange(drone_start_bin, drone_stop_bin)
        #(2). noise floor range:
        noise_baseline_start_bin = params.noise_baseline_start_bin
        noise_baseline_stop_bin = params.noise_baseline_stop_bin
        noise_frequency_axis = np.arange(noise_baseline_start_bin, noise_baseline_stop_bin)
        frequency_range_number_of_bins_for_drone = np.int(frequency_range_for_drone * frequency_axis_delta)
        #(3). low frequencies range (mainly for noise which big birds tend to cause):
        low_frequencies_to_check_start_bin = params.low_frequency_start_bin
        low_frequencies_to_check_stop_bin = params.low_frequency_stop_bin
        low_frequencies_to_check_frequency_axis = np.arange(noise_baseline_start_bin, noise_baseline_stop_bin)

        ### Get Proper Indices For The Different "Search Locations": ###
        indices_arange = np.arange(len(frequency_axis))
        logical_mask_default = np.ones_like(indices_arange).astype(np.bool)
        #(1). Logical Masks:
        drone_frequency_bins_logical_mask = ((frequency_axis>=drone_start_bin).astype(np.uint8) * (frequency_axis<=drone_stop_bin).astype(np.uint8)).astype(np.bool)
        non_drone_frequency_bins_logical_mask = (1-drone_frequency_bins_logical_mask.astype(np.uint8)).astype(np.bool)
        noise_baseline_frequency_bins_logical_mask = ((frequency_axis>=noise_baseline_start_bin).astype(np.uint8) * (frequency_axis<=noise_baseline_stop_bin).astype(np.uint8)).astype(np.bool)
        low_frequencies_frequency_bins_logical_mask = ((frequency_axis>=low_frequencies_to_check_start_bin).astype(np.uint8) * (frequency_axis<=low_frequencies_to_check_stop_bin).astype(np.uint8)).astype(np.bool)
        #(2). Indices Themselves:
        drone_bins_indices = indices_arange[drone_frequency_bins_logical_mask]
        non_drone_bins_indices = indices_arange[non_drone_frequency_bins_logical_mask]
        noise_baseline_bins_indices = indices_arange[noise_baseline_frequency_bins_logical_mask]
        ###########################################################################################################3

        ###########################################################################################################3
        ### Get "Signal" Part of the SNR: ###
        #(3). Get max lobe energy and frequencies between drone search area (around 100Hz)
        ### get max lobe energy per pixel within possible frequency range (about 90-120Hz) / "Max-Lobe Strategy": ###
        input_tensor_fft_possible_drone_bins_only = input_tensor_fft.abs()[drone_frequency_bins_logical_mask]  #TODO: for clarity maybe first do a binning on entire signal and then search for max around drone bins
        input_tensor_fft_possible_drone_bins_binned = fast_binning_1D_overlap_flexible(input_tensor_fft_possible_drone_bins_only,
                                                                                      binning_size = frequency_range_number_of_bins_for_drone,
                                                                                      overlap_size = frequency_range_number_of_bins_for_drone-1,
                                                                                      flag_return_average = False,
                                                                                      dim = 0) #(*). returns SUM, not MEAN
        #(*). Get binned fft over entire frequencies range to be used later-on for presentation and lobe detection (as opposed to singular "thin" peak detection):
        input_tensor_fft_binned = fast_binning_1D_overlap_flexible(input_tensor_fft.abs(),   #TODO: only calculated for plotting basically for now. maybe shouuld use this apriori
                                                                   binning_size=frequency_range_number_of_bins_for_drone,
                                                                   overlap_size=frequency_range_number_of_bins_for_drone - 1,
                                                                   flag_return_average=False,
                                                                   dim=0)  # (*). returns SUM, not MEAN
        input_tensor_fft_binned_for_plot = fast_binning_1D_overlap_flexible(input_tensor_fft.abs(),  # TODO: only calculated for plotting basically for now. maybe shouuld use this apriori
                                                                   binning_size=10,
                                                                   overlap_size=0,
                                                                   flag_return_average=True,
                                                                   dim=0)
        frequency_axis_torch_binned_for_plot = fast_binning_1D_overlap_flexible(frequency_axis_torch,  # TODO: only calculated for plotting basically for now. maybe shouuld use this apriori
                                                                            binning_size=10,
                                                                            overlap_size=0,
                                                                            flag_return_average=True,
                                                                            dim=0).round()

        ### Get Max Value & Index from Binned FFT (which as of now is not binned at all as frequency_range_number_of_bins_for_drone=1): ###
        input_tensor_fft_possible_drone_bins_max_lobe_index = torch.argmax(input_tensor_fft_possible_drone_bins_binned, 0)
        input_tensor_fft_possible_drone_bine_max_lobe_value = torch.gather(input_tensor_fft_possible_drone_bins_binned, 0, input_tensor_fft_possible_drone_bins_max_lobe_index.unsqueeze(0))
        input_tensor_fft_possible_drone_bins_max_lobe_index += drone_bins_indices[0] #correct for starting place on frequency axis

        # ### "predetermined frequency bins strategy": ###
        # # #(*). get lobe energy per pixel between PREDEFINED, NARROW drone range:
        # input_tensor_fft_possible_drone_bins_max_lobe_index = 360
        # input_tensor_fft_possible_drone_bine_max_lobe_value = input_tensor_fft.abs()[input_tensor_fft_possible_drone_bins_max_lobe_index :
        #                                                                              input_tensor_fft_possible_drone_bins_max_lobe_index +
        #                                                                              frequency_range_number_of_bins_for_drone].sum(0,True)

        # # ### Peak Detection: ###
        # nice_peaks = peak_detect_pytorch(input_tensor_fft.abs(), window_size=11, flag_plot=True)
        ###########################################################################################################3

        ###########################################################################################################3
        ### Get "Noise" part of the SNR: ###
        # # (1). get the mean value of the COHERENT SUM of the fft components over frequency_range_number_of_bins_for_drone number of bins over predetermined range:
        # #TODO: make this more robust and per pixel, currently uses coherent sums, shouuld i use that?
        # input_tensor_fft_noise_estimation_bins = input_tensor_fft[noise_baseline_frequency_bins_logical_mask]
        # input_tensor_fft_noise_estimation_bins_binned_real = fast_binning_1D_overap_flexible(input_tensor_fft_noise_estimation_bins.real,
        #                                                                               binning_size = frequency_range_number_of_bins_for_drone,
        #                                                                               overlap_size = frequency_range_number_of_bins_for_drone - 1,
        #                                                                               flag_return_average = True,
        #                                                                                      dim = 0)  #return average, not sums!
        # input_tensor_fft_noise_estimation_bins_binned_imag = fast_binning_1D_overap_flexible(input_tensor_fft_noise_estimation_bins.imag,
        #                                                                               binning_size = frequency_range_number_of_bins_for_drone,
        #                                                                               overlap_size = frequency_range_number_of_bins_for_drone - 1,
        #                                                                               flag_return_average = True,
        #                                                                                      dim = 0) #return average, not sums!
        # input_tensor_fft_noise_estimation_bins_binned = input_tensor_fft_noise_estimation_bins_binned_real + 1j*input_tensor_fft_noise_estimation_bins_binned_imag
        # input_tensor_fft_noise_estimation_mean = input_tensor_fft_noise_estimation_bins_binned.abs().mean(0, True) #return the average of lobe-sized sums
        #
        # #(2). get the mean value of the abs() sum (INCOHERENT SUM) of the fft components over frequency_range_number_of_bins_for_drone of bins over predetermined range:
        # input_tensor_fft_noise_estimation_bins = input_tensor_fft[noise_baseline_frequency_bins_logical_mask].abs()
        # input_tensor_fft_noise_estimation_bins_binned = fast_binning_1D_overap_flexible(input_tensor_fft_noise_estimation_bins,
        #                                                                               binning_size = frequency_range_number_of_bins_for_drone,
        #                                                                               overlap_size = frequency_range_number_of_bins_for_drone - 1,
        #                                                                               flag_return_average = True,
        #                                                                                      dim = 0)
        # input_tensor_fft_noise_estimation_mean = input_tensor_fft_noise_estimation_bins_binned.abs().mean(0, True)  # return the average of lobe-sized sums

        #(3). get the MEDIAN value of the abs() of frequencies which are within a certain range of the median over a larger frequency range (which can include the drone, and median is used for robustness):
        input_tensor_fft_noise_estimation_bins = input_tensor_fft[noise_baseline_frequency_bins_logical_mask].abs()
        input_tensor_fft_noise_estimation_bins_median = input_tensor_fft_noise_estimation_bins.median(0)[0].unsqueeze(0)
        input_tensor_fft_noise_estimation_mean = input_tensor_fft_noise_estimation_bins_median
        ###########################################################################################################3

        ###########################################################################################################3
        ### Calculate SNR Per Pixel: ###
        input_tensor_fft_max_lobe_SNR_around_drone_bins = input_tensor_fft_possible_drone_bine_max_lobe_value / input_tensor_fft_noise_estimation_mean.cuda()
        ###########################################################################################################3

        ###########################################################################################################3
        ### Lower ~DC Lobe By Fitting To A Gaussian/Lorentzian: ###
        1
        ###########################################################################################################3


        ###########################################################################################################3
        ### Check Low Frequencies Condition: ###
        # #(1). "Brute-Force"/"Integral" condition, threshold over low frequencies "SNR"
        # input_tensor_fft_low_frequencies_bins_mean = input_tensor_fft.abs()[low_frequencies_frequency_bins_logical_mask].mean(0 ,True)
        # absolute_noise_floor_of_entire_ROI = input_tensor_fft_noise_estimation_mean.median()
        # input_tensor_fft_low_frequencies_mean_over_noise_floor_ratio = input_tensor_fft_low_frequencies_bins_mean / absolute_noise_floor_of_entire_ROI
        # input_tensor_fft_low_frequencies_sum_too_large = input_tensor_fft_low_frequencies_mean_over_noise_floor_ratio > params.low_frequency_lobe_over_noise_floor_threshold
        # input_tensor_fft_DC_lobe_small_enough = 1 - input_tensor_fft_low_frequencies_sum_too_large.float()

        #(2). Condition Over Numeric FWHM:
        #(*). Smooth (doesn't seem to work)
        # input_tensor_fft_smooth = smooth_tensor_loess(input_tensor_fft.abs()[input_tensor_fft.shape[0]//2+1:].unsqueeze(1), window=11, use_matrix=False)
        # plot_torch(input_tensor_fft.abs()[input_tensor_fft.shape[0] // 2 + 10:, 2, 2])
        # plot_torch(input_tensor_fft_smooth[:, 0, 2, 2])
        #(*). Check FWHM:
        input_tensor_fft_half = input_tensor_fft.abs()[input_tensor_fft.shape[0]//2:]
        input_tensor_fft_DC_component = input_tensor_fft_half[0:1]
        input_tensor_fft_DC_component_SNR = input_tensor_fft_DC_component / input_tensor_fft_noise_estimation_mean
        input_tensor_fft_smaller_then_DC_FWHM_fraction = input_tensor_fft_half <= input_tensor_fft_DC_component * params.FWHM_fraction
        any_nonz, idx_first_nonz = first_nonzero_torch(input_tensor_fft_smaller_then_DC_FWHM_fraction, axis=0)
        delta_f = frequency_axis_torch[1] - frequency_axis_torch[0]
        input_tensor_fft_DC_lobe_small_enough = idx_first_nonz.unsqueeze(0) < delta_f * params.max_DC_lobe_size
        input_tensor_fft_DC_component_exists = input_tensor_fft_DC_component_SNR > params.DC_value_to_noise_threshold
        input_tensor_fft_DC_lobe_small_enough *= input_tensor_fft_DC_component_exists #(*) just to make sure there even is a DC component
        input_tensor_fft_DC_lobe_small_enough = input_tensor_fft_DC_lobe_small_enough.float()
        # #(*). Fit Gaussian Curve:
        # sigma_gauss_mat = torch.zeros(H,W).to(input_tensor_fft.device)
        # point_around_DC_to_fit = 10
        # for i in np.arange(input_tensor_fft.shape[-2]):
        #     for j in np.arange(input_tensor_fft.shape[-1]):
        #         y_to_fit = input_tensor_fft.abs()[T // 2 - point_around_DC_to_fit:T // 2 + point_around_DC_to_fit + 1, i, j]
        #         x_to_fit = torch.arange(len(y_to_fit)) - len(y_to_fit) // 2
        #         if input_tensor_fft_DC_lobe_small_enough[0,i,j] == 1 and input_tensor_fft_DC_component_exists[0, i,j] == True:
        #             y_gaussian_fit, gaussian_fit_coefficients, gaussian_fit_covariance = fit_Gaussian(x_to_fit.cpu().numpy(), y_to_fit.cpu().numpy())
        #             a_gauss, x0_gauss, sigma_gauss = gaussian_fit_coefficients
        #             sigma_gauss_mat[i,j] = sigma_gauss
        #         else:
        #             sigma_gauss_mat[i,j] = 0
        # sigma_gauss_mat *= input_tensor_fft_DC_component_exists.squeeze()
        # if sigma_gauss_mat.max() > 30:
        #     input_tensor_fft_DC_lobe_small_enough = 0 * input_tensor_fft_DC_lobe_small_enough

        ### TEMP - try out specific gaussian fit on a specific pixel: ###
        # y_to_fit = input_tensor_fft.abs()[T//2-10:T//2+10+1, 0, 3]
        # x_to_fit = torch.arange(len(y_to_fit)) - len(y_to_fit)//2
        # y_gaussian_fit, gaussian_fit_coefficients, gaussian_fit_covariance = fit_Gaussian(x_to_fit.cpu().numpy(), y_to_fit.cpu().numpy())
        # a_gauss, x0_gauss, sigma_gauss = gaussian_fit_coefficients
        # plot_torch(torch.tensor(x_to_fit), torch.tensor(y_to_fit))
        # plot_torch(torch.tensor(x_to_fit), torch.tensor(y_gaussian_fit))
        # plot_torch(input_tensor_fft.abs()[:,0,3].clamp(0,1))

        #(3). Add Condition Per Pixel That DC Component even exists(!), and that it's not just random fluctuations (or at least lower the risk there of):
        input_tensor_fft_DC_component_exists = input_tensor_fft_DC_component_SNR > params.DC_value_to_noise_threshold
        input_tensor_fft_DC_component_largest = (torch.argmax(input_tensor_fft_half) == 0)
        ###########################################################################################################3


        ###########################################################################################################3
        ### Perform Peak Detection, Harmonics Detection, Cooler Harmonics Invalidation: ###
        input_tensor = input_tensor.unsqueeze(1)
        i_vec = np.arange(input_tensor.shape[-2])
        j_vec = np.arange(input_tensor.shape[-1])

        ### Get FFT Over Specific Drone Area: ###
        input_tensor_drone = input_tensor[:, :, i_vec[0]:i_vec[-1] + 1, j_vec[0]:j_vec[-1] + 1].cuda()
        input_tensor_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_drone, dim=0).abs(), 0)
        T = input_tensor_fft.shape[0]

        ### Create Proper Folders: ###
        specific_case_string = 'Trajectory_' + str(trajectory_index)
        ffts_folder = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New')
        ffts_folder_before_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'Before_Conditions')
        ffts_folder_after_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'After_Conditions')
        path_create_path_if_none_exists(ffts_folder)
        path_create_path_if_none_exists(ffts_folder_before_conditions)
        path_create_path_if_none_exists(ffts_folder_after_conditions)

        ### Initialize Peak Frequencies List Over All ROI: ###
        final_frequency_peaks_tuples_list = []
        final_frequency_peaks_list = []

        ### Loop Over Every Pixel In The ROI: ###
        for i in np.arange(len(i_vec)):
            for j in np.arange(len(j_vec)):
                ### Initialize Tensor: ###
                # Drone = [279, 251], Edge = [302, 221]
                i = int(i)
                j = int(j)
                # input_tensor_fft_graph = input_tensor_fft[:, 0, int(i), int(j)]
                # input_tensor_fft_graph = convn_torch(input_tensor_fft_graph, torch.ones(3)/3, dim=0).squeeze()
                # input_tensor_fft_graph = input_tensor_fft_graph.clamp(0, 1)
                FPS = params.FPS
                frequency_axis = torch.tensor(FPS * np.linspace(-0.5, .5 - 1 / T, T))
                frequency_axis_numpy = frequency_axis.cpu().numpy()

                ### Initialize graphs location and string: ###
                specific_case_string = ''
                graph_string = specific_case_string + '   ' + 'H=' + str(int(i_vec[i])) + '_W=' + str(int(j_vec[j]))


                ### Peak Detect: ###
                input_vec = input_tensor_fft[:, 0:1, int(i):int(i + 1), int(j):int(j + 1)].abs()  #TODO: why am i using the cpu here?!!?
                # input_vec = convn_torch(input_vec, torch.ones(3)/3, dim=0)
                maxima_peaks, arange_in_correct_dim_tensor = peak_detect_pytorch(input_vec,
                                                                                 window_size=params.peak_detection_window_size, #TODO: make this a parameter in the dict
                                                                                 dim=0,
                                                                                 flag_use_ratio_threshold=True,
                                                                                 ratio_threshold=0,  #TODO: threshold over median of window, decide whether i want to keep this
                                                                                 flag_plot=False)
                maxima_peaks_original = copy.deepcopy(maxima_peaks)

                ### Get All Peak Frequencies From Peak-Detection Algorithm: ###
                maxima_peaks_indices = (arange_in_correct_dim_tensor)[maxima_peaks]
                maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_indices] > 0
                peak_frequencies_list_original = frequency_axis[maxima_peaks_indices][maxima_frequency_peaks_logical_mask]
                peak_frequencies_list_original = peak_frequencies_list_original.tolist()
                peak_frequencies_array_original = np.array(peak_frequencies_list_original)

                ### Get Noise Floor: ###
                median_noise_floor = input_tensor_fft_noise_estimation_mean[0, int(i), int(j)]
                # median_noise_floor = input_vec.median().item()

                ### Only Keep Peaks Above Median Noise-Floor Enough: ###
                #(1). Initial SNR threshold to see which frequencies survive, then see which freqeucneis are close-by to add them up
                SNR_threshold_value = median_noise_floor * params.FFT_SNR_threshold_initial #TODO: again, use the noise floor calculated above, with the params defined SNR threshold
                SNR_threshold_value_final = median_noise_floor * params.FFT_SNR_threshold_final #TODO: again, use the noise floor calculated above, with the params defined SNR threshold
                logical_mask_above_noise_median = input_vec > SNR_threshold_value
                maxima_peaks = maxima_peaks * logical_mask_above_noise_median
                #(2). Lobe SNR threshold (sometimes the energy is not concentrated in a single frequency but in a lobe):
                #TODO: add condition over binned fft
                lobe_median_noise_floor = median_noise_floor * 2 # ~2 for lobe=5, ~3 for lobe=10, this is simply statistics of random sums
                SNR_threshold_lobe = median_noise_floor * params.FFT_SNR_threshold_initial * 3 # this is an ed-hok factor which should, in theory, depend on the size of the lobe dress and i should simply aggregate the energy
                input_vec_binned = 1
                #(2). Total SNR threshold after combining close-by frequencies in different pixels AND/OR different frequencies in the same pixel (TODO: which is smarter?):
                #TODO: add condition over the different pixels OUTSIDE THE (i,j) LOOP, which sums up the different peak frequencies and adds close ones
                SNR_threshold_total = median_noise_floor * params.FFT_SNR_threshold_initial
                input_vec_combined = 1


                ### Get All Peak Frequencies From Peak-Detection Algorithm: ###
                maxima_peaks_indices = (arange_in_correct_dim_tensor)[maxima_peaks]
                maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_indices] > 0
                peak_frequencies_list_after_SNR_threshold = frequency_axis[maxima_peaks_indices][maxima_frequency_peaks_logical_mask]
                peak_frequencies_list_after_SNR_threshold = peak_frequencies_list_after_SNR_threshold.tolist()
                peak_frequencies_array_after_SNR_threshold = np.array(peak_frequencies_list_after_SNR_threshold)

                ### Get Rid Of Peaks Which Are Harmonies Of The Base Cooler Frequency (usual ~23[Hz] or ~36[Hz]): ###
                base_frequency = float(params.cooler_base_harmonic)
                base_frequency_harmonic_tolerance = params.base_frequency_harmonic_tolerance
                frequency_axis_remainder_from_base_frequency = torch.remainder(frequency_axis, base_frequency)
                # Instead of creating an array of [23,23,23.....,46,46,46,....] and getting the diff i simply check by a different, maybe stupid way for close harmonics
                ### Keep Only Frequencies Which Are Far Enough Away From Base Frequency Harmonics: ###
                frequency_axis_modulo_base_logical_mask = frequency_axis_remainder_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from above
                frequency_axis_diff_from_base_frequency = (frequency_axis_remainder_from_base_frequency - base_frequency).abs()
                frequency_axis_modulo_base_logical_mask *= frequency_axis_diff_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from below
                frequency_axis_modulo_base_logical_mask = frequency_axis_modulo_base_logical_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                maxima_peaks = maxima_peaks * frequency_axis_modulo_base_logical_mask.to(maxima_peaks.device)
                # (*). Get Rid Of Negative Frequencies:
                maxima_peaks[0:T // 2] = False

                ### Get All Peak Frequencies (After Harminic Analysis): ###
                maxima_peaks_indices = (arange_in_correct_dim_tensor)[maxima_peaks]
                maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_indices] > 0
                peak_frequencies_list = frequency_axis[maxima_peaks_indices][maxima_frequency_peaks_logical_mask]
                peak_frequencies_list = peak_frequencies_list.tolist()
                peak_frequencies_array = np.array(peak_frequencies_list)

                ### Record Peak Frequencies: ###
                for peak_index in np.arange(len(peak_frequencies_list)):
                    peak_frequency = peak_frequencies_list[peak_index]
                    peak_index_in_fft = maxima_peaks_indices[peak_index]
                    peak_value = input_vec[peak_index_in_fft]
                    peak_SNR = peak_value / median_noise_floor
                    peak_pixel_within_ROI = (i, j)
                    input_tuple = (trajectory_index, peak_frequency, peak_value.item(), peak_SNR.item(), peak_pixel_within_ROI)
                    final_frequency_peaks_tuples_list.append(input_tuple)
                    final_frequency_peaks_list.append(peak_frequency)

                ######### Plot Graphs Between Harmonics Conditions: #########
                if params.flag_save_interim_graphs_and_movies:
                    ### Legends list: ###
                    legends_list = copy.deepcopy(peak_frequencies_list)
                    for legend_index in np.arange(len(peak_frequencies_list)):
                        legends_list[legend_index] = decimal_notation(peak_frequencies_list[legend_index], 1)
                    legends_list = ['FFT', 'Original Peaks', 'Final Peaks', 'Noise Floor', 'SNR Threshold Initial', 'SNR Threshold Final', 'Noise Floor'] + legends_list
                    ### Plot Results: this is a specific case of tchw tensor, chosen dim t: ###
                    k1 = 0
                    k2 = 0
                    k3 = 0
                    clamp_value = 3
                    maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, k1, k2, k3]])[:, k1, k2, k3]
                    maxima_peaks_to_plot_original = (arange_in_correct_dim_tensor[maxima_peaks_original[:, k1, k2, k3]])[:, k1, k2, k3]
                    plt.figure()
                    plt.plot(frequency_axis_numpy, input_vec[:, k1, k2, k3].clamp(0, clamp_value).cpu().numpy())
                    # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
                    plt.plot(frequency_axis[maxima_peaks_to_plot_original], input_vec[maxima_peaks_to_plot_original, k1, k2, k3].clamp(0, clamp_value).cpu().numpy(), '.')
                    plt.plot(frequency_axis[maxima_peaks_to_plot], input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0, clamp_value).cpu().numpy(), '.', linewidth=3)
                    plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * median_noise_floor.item())
                    plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold_value.item())
                    plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold_value_final.item())
                    plt.plot(drone_frequencies_axis, (+0.003 * 4) * np.ones_like(drone_frequencies_axis), linewidth=3)
                    plt.plot(noise_frequency_axis, (+0.003 * 8) * np.ones_like(noise_frequency_axis), linewidth=3)
                    plt.title(graph_string)
                    for current_legend in legends_list:
                        plt.plot(frequency_axis_numpy, frequency_axis_numpy * 0)
                    plt.legend(legends_list)
                    ### Set Y limits: ###
                    plt.ylim([0, clamp_value])
                    ### Save Graph: ###
                    save_string = os.path.join(ffts_folder_before_conditions, graph_string + '.png')
                    plt.savefig(save_string)
                    plt.close('all')
        ###########################################################################################################3

        # ###########################################################################################################3
        # ### Perform Peak Detection, Harmonics Detection, Cooler Harmonics Invalidation: ###
        # input_tensor = input_tensor.unsqueeze(1)
        # i_vec = np.arange(input_tensor.shape[-2])
        # j_vec = np.arange(input_tensor.shape[-1])
        #
        # ### Get FFT Over Specific Drone Area: ###
        # input_tensor_drone = input_tensor[:, :, i_vec[0]:i_vec[-1] + 1, j_vec[0]:j_vec[-1] + 1].cuda()
        # input_tensor_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_drone, dim=0).abs(), 0)
        # fft_averaging_window_size = 5
        # input_tensor_fft = convn_torch(input_tensor_fft, torch.ones(fft_averaging_window_size)/fft_averaging_window_size, 0)
        # T = input_tensor_fft.shape[0]
        #
        # ### Create Proper Folders: ###
        # specific_case_string = 'Trajectory_' + str(trajectory_index)
        # ffts_folder = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New')
        # ffts_folder_before_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'Before_Conditions')
        # ffts_folder_after_conditions = os.path.join(params.results_folder_seq, '' + specific_case_string, 'FFT_Analysis_New', 'After_Conditions')
        # path_create_path_if_none_exists(ffts_folder)
        # path_create_path_if_none_exists(ffts_folder_before_conditions)
        # path_create_path_if_none_exists(ffts_folder_after_conditions)
        #
        # ### Initialize Peak Frequencies List Over All ROI: ###
        # final_frequency_peaks_tuples_list = []
        # final_frequency_peaks_list = []
        #
        # ### Loop Over Every Pixel In The ROI: ###
        # for i in np.arange(len(i_vec)):
        #     for j in np.arange(len(j_vec)):
        #         ### Initialize Tensor: ###
        #         # Drone = [279, 251], Edge = [302, 221]
        #         i = int(i)
        #         j = int(j)
        #         # input_tensor_fft_graph = input_tensor_fft[:, 0, int(i), int(j)]
        #         # input_tensor_fft_graph = convn_torch(input_tensor_fft_graph, torch.ones(3)/3, dim=0).squeeze()
        #         # input_tensor_fft_graph = input_tensor_fft_graph.clamp(0, 1)
        #         FPS = params.FPS
        #         frequency_axis = torch.tensor(FPS * np.linspace(-0.5, .5 - 1 / T, T))
        #         frequency_axis_numpy = frequency_axis.cpu().numpy()
        #
        #         ### Initialize graphs location and string: ###
        #         specific_case_string = ''
        #         graph_string = specific_case_string + '   ' + 'H=' + str(int(i_vec[i])) + '_W=' + str(int(j_vec[j]))
        #
        #         ### Peak Detect: ###
        #         input_vec = input_tensor_fft[:, 0:1, int(i):int(i + 1), int(j):int(j + 1)].abs()  # TODO: why am i using the cpu here?!!?
        #         # input_vec = convn_torch(input_vec, torch.ones(3)/3, dim=0)
        #         maxima_peaks, arange_in_correct_dim_tensor = peak_detect_pytorch(input_vec,
        #                                                                          window_size=params.peak_detection_window_size,  # TODO: make this a parameter in the dict
        #                                                                          dim=0,
        #                                                                          flag_use_ratio_threshold=True,
        #                                                                          ratio_threshold=0,  # TODO: threshold over median of window, decide whether i want to keep this
        #                                                                          flag_plot=False)
        #
        #         ### Get Noise Floor: ###
        #         median_noise_floor = input_tensor_fft_noise_estimation_mean[0, int(i), int(j)]
        #         # median_noise_floor = input_vec.median().item()
        #
        #         ### Only Keep Peaks Above Median Noise-Floor Enough: ###
        #         # (1). Initial SNR threshold to see which frequencies survive, then see which freqeucneis are close-by to add them up
        #         SNR_threshold_value = median_noise_floor * params.FFT_SNR_threshold_initial  # TODO: again, use the noise floor calculated above, with the params defined SNR threshold
        #         SNR_threshold_value_final = median_noise_floor * params.FFT_SNR_threshold_final  # TODO: again, use the noise floor calculated above, with the params defined SNR threshold
        #         logical_mask_above_noise_median = input_vec > SNR_threshold_value
        #         maxima_peaks = maxima_peaks * logical_mask_above_noise_median
        #         # (2). Lobe SNR threshold (sometimes the energy is not concentrated in a single frequency but in a lobe):
        #         # TODO: add condition over binned fft
        #         lobe_median_noise_floor = median_noise_floor * 2  # ~2 for lobe=5, ~3 for lobe=10, this is simply statistics of random sums
        #         SNR_threshold_lobe = median_noise_floor * params.FFT_SNR_threshold_initial * 3  # this is an ed-hok factor which should, in theory, depend on the size of the lobe dress and i should simply aggregate the energy
        #         input_vec_binned = 1
        #         # (2). Total SNR threshold after combining close-by frequencies in different pixels AND/OR different frequencies in the same pixel (TODO: which is smarter?):
        #         # TODO: add condition over the different pixels OUTSIDE THE (i,j) LOOP, which sums up the different peak frequencies and adds close ones
        #         SNR_threshold_total = median_noise_floor * params.FFT_SNR_threshold_initial
        #         input_vec_combined = 1
        #
        #         ### Get All Peak Frequencies From Peak-Detection Algorithm: ###
        #         maxima_peaks_indices = (arange_in_correct_dim_tensor)[maxima_peaks]  # TODO: change to maxima peak INDICES
        #         maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_indices] > 0
        #         peak_frequencies_list = frequency_axis[maxima_peaks_indices][maxima_frequency_peaks_logical_mask]
        #         peak_frequencies_list = peak_frequencies_list.tolist()
        #         peak_frequencies_array = np.array(peak_frequencies_list)
        #
        #         ### Get Rid Of Peaks Which Are Harmonies Of The Base Cooler Frequency (usual ~23[Hz] or ~36[Hz]): ###
        #         base_frequency = float(params.cooler_base_harmonic)
        #         base_frequency_harmonic_tolerance = params.base_frequency_harmonic_tolerance
        #         frequency_axis_remainder_from_base_frequency = torch.remainder(frequency_axis, base_frequency)
        #         # Instead of creating an array of [23,23,23.....,46,46,46,....] and getting the diff i simply check by a different, maybe stupid way for close harmonics
        #         ### Keep Only Frequencies Which Are Far Enough Away From Base Frequency Harmonics: ###
        #         frequency_axis_modulo_base_logical_mask = frequency_axis_remainder_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from above
        #         frequency_axis_diff_from_base_frequency = (frequency_axis_remainder_from_base_frequency - base_frequency).abs()
        #         frequency_axis_modulo_base_logical_mask *= frequency_axis_diff_from_base_frequency >= base_frequency_harmonic_tolerance  # 23[Hz] harmonics from below
        #         frequency_axis_modulo_base_logical_mask = frequency_axis_modulo_base_logical_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #         maxima_peaks = maxima_peaks * frequency_axis_modulo_base_logical_mask.to(maxima_peaks.device)
        #         # (*). Get Rid Of Negative Frequencies:
        #         maxima_peaks[0:T // 2] = False
        #
        #         ### Get All Peak Frequencies (After Harminic Analysis): ###
        #         maxima_peaks_indices = (arange_in_correct_dim_tensor)[maxima_peaks]
        #         maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_indices] > 0
        #         peak_frequencies_list = frequency_axis[maxima_peaks_indices][maxima_frequency_peaks_logical_mask]
        #         peak_frequencies_list = peak_frequencies_list.tolist()
        #         peak_frequencies_array = np.array(peak_frequencies_list)
        #
        #         ### Record Peak Frequencies: ###
        #         for peak_index in np.arange(len(peak_frequencies_list)):
        #             peak_frequency = peak_frequencies_list[peak_index]
        #             peak_index_in_fft = maxima_peaks_indices[peak_index]
        #             peak_value = input_vec[peak_index_in_fft]
        #             peak_SNR = peak_value / median_noise_floor
        #             peak_pixel_within_ROI = (i, j)
        #             input_tuple = (trajectory_index, peak_frequency, peak_value.item(), peak_SNR.item(), peak_pixel_within_ROI)
        #             final_frequency_peaks_tuples_list.append(input_tuple)
        #             final_frequency_peaks_list.append(peak_frequency)
        #
        #         ######### Plot Graphs Between Harmonics Conditions: #########
        #         if params.flag_save_interim_graphs_and_movies:
        #             ### Legends list: ###
        #             legends_list = copy.deepcopy(peak_frequencies_list)
        #             for legend_index in np.arange(len(peak_frequencies_list)):
        #                 legends_list[legend_index] = decimal_notation(peak_frequencies_list[legend_index], 1)
        #             legends_list = ['FFT', 'peaks', 'Noise Floor', 'SNR Threshold Initial', 'SNR Threshold Final', 'Drone Frequencies', 'Noise Floor'] + legends_list
        #             ### Plot Results: this is a specific case of tchw tensor, chosen dim t: ###
        #             k1 = 0
        #             k2 = 0
        #             k3 = 0
        #             maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, k1, k2, k3]])[:, k1, k2, k3]
        #             plt.figure()
        #             plt.plot(frequency_axis_numpy, input_vec[:, k1, k2, k3].clamp(0, 1).cpu().numpy())
        #             # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
        #             plt.plot(frequency_axis[maxima_peaks_to_plot], input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0, 1).cpu().numpy(), '.')
        #             plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * median_noise_floor.item())
        #             plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold_value.item())
        #             plt.plot(frequency_axis_numpy, np.ones_like(frequency_axis_numpy) * SNR_threshold_value_final.item())
        #             plt.plot(drone_frequencies_axis, (+0.003 * 4) * np.ones_like(drone_frequencies_axis), linewidth=3)
        #             plt.plot(noise_frequency_axis, (+0.003 * 8) * np.ones_like(noise_frequency_axis), linewidth=3)
        #             plt.title(graph_string)
        #             for current_legend in legends_list:
        #                 plt.plot(frequency_axis_numpy, frequency_axis_numpy * 0)
        #             plt.legend(legends_list)
        #             ### Set Y limits: ###
        #             plt.ylim([0, 1])
        #             ### Save Graph: ###
        #             save_string = os.path.join(ffts_folder_before_conditions, graph_string + '.png')
        #             plt.savefig(save_string)
        #             plt.close('all')
        # ###########################################################################################################3


        ###########################################################################################################3
        ### Get Condition Over Max Peak, Make Sure It's Within Drone Range And Above Threshold (instead of previous method which simply seeks maximum): ###
        # input_tuple = (trajectory_index, peak_frequency, peak_value, peak_SNR, peak_pixel_within_ROI)
        # final_frequency_peaks_tuples_list.append(input_tuple)
        final_peak_decision_list = []
        best_frequency = 0
        best_SNR = 0
        peak_confidence = torch.tensor(0)
        victory_picture_tensor = torch.zeros((1, 1, input_tensor.shape[-2], input_tensor.shape[-1]))
        for tuple_index in np.arange(len(final_frequency_peaks_tuples_list)):
            trajectory_index, peak_frequency, peak_value, peak_SNR, peak_pixel_within_ROI = final_frequency_peaks_tuples_list[tuple_index]
            current_i, current_j = peak_pixel_within_ROI

            ### Check Conditions (later on will be checked in the above function built-in to the logical masks: ###
            flag_peak_frequency_within_range = peak_frequency >= drone_start_bin and peak_frequency <= drone_stop_bin
            flag_peak_SNR_above_threshold = peak_SNR > params.FFT_SNR_threshold_final
            if flag_peak_frequency_within_range * flag_peak_SNR_above_threshold:
                ### Update Victory Picture: ###
                victory_picture_tensor[:,:,peak_pixel_within_ROI[0], peak_pixel_within_ROI[1]] = 1

                ### Update Peak SNR & Frequency: ###
                if peak_SNR > best_SNR:
                    best_SNR = peak_SNR
                    best_frequency = peak_frequency

            ### Perform logistic function over the drone SNR (get probablity for drone from SNR): ###
            input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic = logistic_torch(peak_SNR,
                                                                                      reference_val=torch.tensor(params.FFT_SNR_threshold_final))

            ### Hard Decision - try and find it there's a single pixel which passed the decision threshold: ###
            # (1). Get SNR after logistic function
            input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough = float(flag_peak_frequency_within_range * flag_peak_SNR_above_threshold)
            # (2). Check where SNR is big enough and low frequencies aren't too large:
            #TODO: add another subtlty - center pixel / center of mass should probably be absolved from low frequencies condition
            input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified = (input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough *
                                                                                     input_tensor_fft_DC_lobe_small_enough[0, current_i, current_j])

            if input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified.item() == 1:
                peak_confidence = max(peak_confidence, input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic)

            ### Add Current Decision To List: ###
            final_peak_decision_list.append(input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified.item())

        ### Go Over All Peaks And See If Any Of Them Passed Thresholds: ###
        # flag_was_drone_found = input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified.sum() >= 0 #TODO: dudy changed, return!!!
        flag_was_drone_found = sum(final_peak_decision_list) > 1
        confidence_this_is_drone = peak_confidence
        print('Best Frequency: ' + str(best_frequency))
        # imshow_torch(victory_picture_tensor, title_str='Victory Image')

        ### Save Something Stupid Just So I See If It Passed Decision Or Not: ###
        # (1). peak detection result logging:
        if flag_was_drone_found == 1:
            folder_name = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index))
            filename_to_save = os.path.join(folder_name, 'PeakDetction_True.npy')
            np.save(filename_to_save, np.array([0]), allow_pickle=True)
        else:
            folder_name = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index))
            filename_to_save = os.path.join(folder_name, 'PeakDetection_False.npy')
            np.save(filename_to_save, np.array([0]), allow_pickle=True)


        ### Fill in according to old convention: ###
        pxl_scr[trajectory_index] = confidence_this_is_drone.item()
        # pxl_scr[trajectory_index] = input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic
        DetectionDec[trajectory_index] = flag_was_drone_found
        # DetectionDec[trajectory_index] = True  #TODO: dudy changed, return!!!
        DetectionConfLvl[trajectory_index] = confidence_this_is_drone.item()


        # ### Analyze The Different Peaks From Different Pixels And See Whether To Sum Them Up Coherently (with different threshold then a single peak): ###
        # final_frequency_peaks_tuples_list.append(input_tuple)
        # final_frequency_peaks_list.append(peak_frequency)
        # final_frequency_peaks_tensor = torch.tensor(final_frequency_peaks_list)
        # sorted_frequencies, sorted_indices = torch.sort(final_frequency_peaks_tensor)
        # sorted_frequencies_diff = sorted_frequencies.diff()
        # sorted_frequencies_diff = torch.cat([torch.tensor([params.FPS]), sorted_frequencies_diff], -1)  # the params.FPS is simply to put something above any frequency diff
        # sorted_frequencies_diff_within_threshold = sorted_frequencies_diff < params.frequency_peaks_distance_still_considered_same_drone

        # ### Sum Up SNR For The Different Peaks Close Enough Together: ###
        # sorted_frequencies_index = 0
        # flag_continue_looping = True
        # final_average_frequencies_list = []
        # final_SNR_list = []
        # #TODO: assuming i don't make SNR_threshold_final so incredibly large over SNR_threshold_initial, the moment i have two close frequencies above threshold i'm done no?
        # # i should, however, notice, the relative direction of the different peaks, if one is ABOVE the other, for instance, i expect them to be close,
        # # but only if they are apart i believe i can make the frequencies diff somewhat large and excuse that as different motor speeds
        # while flag_continue_looping:
        #     flag_diff_below_threshold = sorted_frequencies_diff_within_threshold[sorted_frequencies_index]
        #     while flag_diff_below_threshold:
        #         1
        ###########################################################################################################3



        # ###########################################################################################################3
        # ### Decide whether this is a drone by simply thresholding the above calculated SNR over the ROI, which takes into account peaks in the fft basically: ###
        # # ### Plots For Understanding: ###
        # # range_for_plot = torch.arange(0, 10, 0.1)
        # # output_logistic = logistic_torch(range_for_plot)
        # # plot_torch(range_for_plot, output_logistic)
        # # plot_torch(range_for_plot, torch.log(range_for_plot))
        #
        # ### Perform logistic function over the drone SNR (get probablity for drone from SNR): ###
        # input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic = logistic_torch(input_tensor_fft_max_lobe_SNR_around_drone_bins,
        #                                                                           reference_val=params.logistic_function_reference_value)
        #
        # ### Hard Decision - try and find it there's a single pixel which passed the decision threshold: ###
        # #(1). Get SNR after logistic function
        # input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough = input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic > 0.5
        # #(2). Check where SNR is big enough and low frequencies aren't too large:
        # input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified = (input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough *
        #                                                                         input_tensor_fft_DC_lobe_small_enough)
        # #(3). Check if any pixel has both large enough SNR and wasn't disqualified due to large low frequency lobe:
        # # flag_was_drone_found = input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified.sum() >= 0 #TODO: dudy changed, return!!!
        # flag_was_drone_found = input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified.sum() > 1
        # confidence_this_is_drone = (input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic * input_tensor_fft_DC_lobe_small_enough).max()
        # #(4). Fill in according to old convention:
        # pxl_scr[trajectory_index] = input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic
        # DetectionDec[trajectory_index] = flag_was_drone_found
        # # DetectionDec[trajectory_index] = True  #TODO: dudy changed, return!!!
        # DetectionConfLvl[trajectory_index] = confidence_this_is_drone
        #
        # #TODO: the later functions expect this to be in a certain binned format to be displayed, but that's not how i do it now....
        # # but this type of representation IS valuable, so get to a point where for each frequency bin i have a score (which can include, of course, priors like 23[Hz])
        # TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index] = input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough_modified
        #
        # # ### Soft Decision - sum logistic output over all pixels to decide whether or not there's a drone: ###
        # # soft_sum = (input_tensor_fft_max_lobe_SNR_around_drone_bins_logistic * input_tensor_fft_DC_lobe_small_enough.float()).sum([-1,-2])
        # # input_tensor_fft_drone_decision_passed = soft_sum > 0.1*input_tensor.shape[-1]*input_tensor.shape[-2]
        # ###########################################################################################################3

        ###########################################################################################################3
        # ### Plot Individual FFT Graphs For Each Individual Pixel: ###
        # if params.flag_save_interim_graphs_and_movies:
        #     plot_fft_graphs(input_tensor.unsqueeze(1), H_center=None, W_center=None, area_around_center=None, specific_case_string='Trajectory_' + str(trajectory_index), params=params)
        #     plt.pause(1)
        #     plt.close('all')
        #     plt.pause(1)
        #     plot_fft_graphs_original(flag_was_drone_found, params, trajectory_index,
        #                          input_tensor_fft,
        #                          input_tensor_fft_max_lobe_SNR_around_drone_bins,
        #                          input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough,
        #                          input_tensor_fft_noise_estimation_mean,
        #                          frequency_axis,
        #                          input_tensor_fft_possible_drone_bine_max_lobe_value,
        #                          drone_frequencies_axis,
        #                          noise_frequency_axis)
        #     plt.pause(1)
        #     plt.close('all')
        #     plt.pause(1)
        #     # imshow_torch_video(input_tensor.unsqueeze(1), FPS=50, frame_stride=5)

        # def plot_fft_graphs_original():
        #     flag_save = (flag_was_drone_found == True and params.was_there_drone == 'False')
        #     flag_save = flag_save or (flag_was_drone_found == False and params.was_there_drone == 'True')
        #     if params.flag_save_interim_graphs_and_movies:
        #         # TODO: add condition that if sum of low frequencies is very large, or that we can fit a lorenzian/gaussian on the low frequencies then it's probably a bird
        #         # (*). All Graphs:
        #         print('DRONE FOUND, SAVING FFT GRAPHS')
        #         flag_drone_found_but_no_drone_there = (flag_was_drone_found == True and params.was_there_drone == 'False')
        #         flag_drone_found_and_is_there = (flag_was_drone_found == True and params.was_there_drone == 'True')
        #         flag_drone_not_found_but_is_there = (flag_was_drone_found == False and params.was_there_drone == 'True')
        #         flag_drone_not_found_and_is_not_there = (flag_was_drone_found == False and params.was_there_drone == 'False')
        #         if flag_drone_not_found_but_is_there:
        #             post_string = '_Drone_Not_Found_But_Is_There(FT)'
        #         elif flag_drone_found_but_no_drone_there:
        #             post_string = '_Drone_Found_But_Is_Not_There(TF)'
        #         elif flag_drone_found_and_is_there:
        #             post_string = '_Drone_Found_And_Is_There(TT)'
        #         elif flag_drone_not_found_and_is_not_there:
        #             post_string = '_Drone_Not_Found_And_Is_Not_There(FF)'
        #         else:
        #             post_string = ''
        #         graphs_folder_unclipped = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_Unclipped')
        #         graphs_folder_clipped = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_clipped')
        #         graphs_folder_fft_binned = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + post_string, 'FFT_Graphs_Binned_Clipped')
        #         path_make_path_if_none_exists(graphs_folder_fft_binned)
        #         path_make_path_if_none_exists(graphs_folder_unclipped)
        #         path_make_path_if_none_exists(graphs_folder_clipped)
        #         fft_counter = 0
        #         ROI_H, ROI_W = input_tensor_fft_max_lobe_SNR_around_drone_bins.shape[-2:]
        #         for roi_H in np.arange(ROI_H):
        #             for roi_W in np.arange(ROI_W):
        #                 current_fft = input_tensor_fft[:, roi_H, roi_W].abs()
        #                 current_SNR = input_tensor_fft_max_lobe_SNR_around_drone_bins[0, roi_H, roi_W].item()
        #                 current_decision = input_tensor_fft_max_lobe_SNR_around_drone_bins_large_enough[0, roi_H, roi_W].item()
        #
        #                 ### Unclipped Graph: ##
        #                 figure()
        #                 plt.plot(frequency_axis, current_fft.abs().squeeze().cpu().numpy())
        #                 plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
        #                 plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
        #                 plt.plot(drone_frequencies_axis, (+0.003) * np.ones_like(drone_frequencies_axis))
        #                 plt.plot(noise_frequency_axis, (+0.003 * 2) * np.ones_like(noise_frequency_axis))
        #                 plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
        #                 title_string = str('ROI = [' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(np.round(current_SNR*10)/10) + ', Decision = ' + str(current_decision))
        #                 file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
        #                 plt.title(title_string)
        #                 filename = os.path.join(graphs_folder_unclipped, file_string + '.png')
        #                 plt.savefig(filename)
        #                 plt.close()
        #
        #                 ### clipped Graph: ##
        #                 figure()
        #                 plt.plot(frequency_axis, current_fft.abs().squeeze().cpu().numpy().clip(0, 10))
        #                 plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
        #                 plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
        #                 plt.plot(drone_frequencies_axis, (+0.003) * np.ones_like(drone_frequencies_axis))
        #                 plt.plot(noise_frequency_axis, (+0.003 * 2) * np.ones_like(noise_frequency_axis))
        #                 plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
        #                 title_string = str('[' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(current_SNR) + ', Decision = ' + str(current_decision))
        #                 file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
        #                 plt.title(title_string)
        #                 plt.ylim([0, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * 3])
        #                 filename = os.path.join(graphs_folder_clipped, file_string + '.png')
        #                 plt.savefig(filename)
        #                 plt.close()
        #
        #                 # ### Binned clipped Graph: ##
        #                 # figure()
        #                 # plt.plot(frequency_axis, input_tensor_fft_binned[:,roi_H, roi_W].abs().squeeze().cpu().numpy().clip(0, 10))
        #                 # plt.plot(frequency_axis, input_tensor_fft_possible_drone_bine_max_lobe_value.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
        #                 # plt.plot(frequency_axis, input_tensor_fft_noise_estimation_mean.cpu().numpy()[0, roi_H, roi_W] * np.ones_like(frequency_axis))
        #                 # plt.plot(drone_frequencies_axis, 0 * np.ones_like(drone_frequencies_axis))
        #                 # plt.plot(noise_frequency_axis, (-0.003) * np.ones_like(noise_frequency_axis))
        #                 # plt.legend(['FFT', 'Max_Lobe_Value', 'Noise_Floor', 'Drone Search Bins', 'Baseline Search Bins'])
        #                 # title_string = str('[' + str(roi_H) + ', ' + str(roi_W) + ']' + ', SNR = ' + str(current_SNR) + ', Decision = ' + str(current_decision))
        #                 # file_string = 'H' + str(roi_H) + 'W' + str(roi_W)
        #                 # plt.title(title_string)
        #                 # plt.ylim([0, 1])
        #                 # filename = os.path.join(graphs_folder_clipped, file_string + '.png')
        #                 # plt.savefig(filename)
        #                 # plt.close()
        #
        #                 fft_counter += 1
        ###########################################################################################################3


    suspects_list = None
    return DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_axis_per_trajectory, pxl_scr, DetectionConfLvl


def Frequency_Analysis_And_Detection(TrjMov, noise_std_FFT_over_drone_frequencies, noise_dc_FFT_over_drone_frequencies, params):
    ### Get variables from params dict: ###
    DetFrqInit = params['DetFrqInit']
    DetFrqStop = params['DetFrqStop']
    DetDFrq = params['DetDFrq']
    FrameRate = params['FPS']
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    DetNoiseThresh = params['DetNoiseThresh']
    DetScrThresh = params['DetScrThresh']
    DetScrThresh100Conf = params['DetScrThresh100Conf']
    SeqT = params['SeqT']

    ### PreAllocate and initialize lists which hold decision results for each trajectory: ###
    number_of_trajectories = len(TrjMov)
    dec = list(np.zeros(number_of_trajectories))
    TrjMovie_FFT_BinPartitioned_AfterScoreFunction = list(np.zeros(number_of_trajectories))
    pxl_scr = list(np.zeros(number_of_trajectories))
    DetectionConfLvl = list(np.zeros(number_of_trajectories))

    ### Get frequency vec and bins to be used for decision: ###  #TODO: not the "natural" bins
    frq_d = np.arange(DetFrqInit, DetFrqStop + 1, DetDFrq)[:-1]
    frq_u = frq_d + DetDFrq
    number_of_frequency_steps = len(frq_u)  # Todo: change name to something like number_of_frequency_bins_to_check
    frequency_vec = frq_u / 2 + frq_d / 2

    ### Loop over the different trajectories: ###
    for trajectory_index in range(number_of_trajectories):
        ### Perform FFT on the trajectory for further decision: ###   #TODO: are there any gaps in the trajectory, is the time domain continuous and without jumps?
        current_trajectory_number_of_points = len(TrjMov[trajectory_index])
        frequency_axis = FrameRate * np.linspace(-.5, .5 - 1 / current_trajectory_number_of_points, current_trajectory_number_of_points)
        FFT_normalization_constant = (current_trajectory_number_of_points / SeqT / FrameRate) ** .5
        TrjMovie_FFT = np.fft.fftshift(np.fft.fft(TrjMov[trajectory_index], axis=0), axes=0) * FFT_normalization_constant

        ### Sum frequency content between each predefined frequency bin defined using the above freq_d: ###
        TrjMovie_FFT_BinPartitioned = np.zeros((number_of_frequency_steps, ROI_allocated_around_suspect, ROI_allocated_around_suspect))
        for ii in range(number_of_frequency_steps):
            ind_frq_in = (frequency_axis > frq_d[ii]) & (frequency_axis < frq_u[ii])
            # TrjMovie_FFT_BinPartitioned[ii, :, :] = np.real(np.mean(TrjMovie_FFT[ind_frq_in], 0)) #TODO: why is he taking only the real part? what's going on?!?!?
            TrjMovie_FFT_BinPartitioned[ii, :, :] = np.real(np.mean(np.abs(TrjMovie_FFT[ind_frq_in]), 0)) #TODO: why is he taking only the real part? what's going on?!?!?

        ### Normalize fft by the noise estimated globally before to see if it's significant enough by passing it into the score function: ###
        TrjMovie_FFT_BinPartitioned_NoiseNormalized = (TrjMovie_FFT_BinPartitioned - noise_dc_FFT_over_drone_frequencies) / noise_std_FFT_over_drone_frequencies
        TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index] = ScrFunc(DetNoiseThresh, TrjMovie_FFT_BinPartitioned_NoiseNormalized)

        ### See at which bins the score was significant and normalize it to 100 for confidence measure: ###
        pxl_scr[trajectory_index] = np.sum(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index])  # sum over the entire bins and pixels (why over all bins)
        dec[trajectory_index] = pxl_scr[trajectory_index] > DetScrThresh
        DetectionConfLvl[trajectory_index] = 100 * np.minimum((pxl_scr[trajectory_index] - DetScrThresh) / (DetScrThresh100Conf - DetScrThresh), 1)

    return dec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl


def Frequency_Analysis_And_Detection_Torch(TrjMov, noise_std_FFT_over_drone_frequencies, noise_dc_FFT_over_drone_frequencies, params):
    ### Get variables from params dict: ###
    DetFrqInit = params['DetFrqInit']
    DetFrqStop = params['DetFrqStop']
    DetDFrq = params['DetDFrq']
    FrameRate = params['FPS']
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    DetNoiseThresh = params['DetNoiseThresh']
    DetScrThresh = params['DetScrThresh']
    DetScrThresh100Conf = params['DetScrThresh100Conf']
    SeqT = params['SeqT']

    ### PreAllocate and initialize lists which hold decision results for each trajectory: ###
    number_of_trajectories = len(TrjMov)
    dec = list(np.zeros(number_of_trajectories))
    TrjMovie_FFT_BinPartitioned_AfterScoreFunction = list(np.zeros(number_of_trajectories))
    pxl_scr = list(np.zeros(number_of_trajectories))
    DetectionConfLvl = list(np.zeros(number_of_trajectories))

    ### Get frequency vec and bins to be used for decision: ###  #TODO: not the "natural" bins
    frq_d = np.arange(DetFrqInit, DetFrqStop + 1, DetDFrq)[:-1]
    frq_u = frq_d + DetDFrq
    number_of_frequency_steps = len(frq_u)  # Todo: change name to something like number_of_frequency_bins_to_check
    frequency_vec = frq_u / 2 + frq_d / 2

    ### Loop over the different trajectories: ###
    for trajectory_index in range(number_of_trajectories):
        ### Perform FFT on the trajectory for further decision: ###
        current_trajectory_number_of_points = len(TrjMov[trajectory_index])
        frequency_axis = FrameRate * torch.linspace(-.5, .5 - 1 / current_trajectory_number_of_points, current_trajectory_number_of_points).to(TrjMov[trajectory_index].device)
        FFT_normalization_constant = (current_trajectory_number_of_points / SeqT / FrameRate) ** .5
        TrjMovie_FFT = fftshift_torch_specific_dim(torch.fft.fftn(TrjMov[trajectory_index], dim=0), dim=0) * FFT_normalization_constant

        ### Sum frequency content between each predefined frequency bin defined using the above freq_d: ###
        TrjMovie_FFT_BinPartitioned = torch.zeros((number_of_frequency_steps, ROI_allocated_around_suspect, ROI_allocated_around_suspect)).to(TrjMov[0].device)
        for ii in range(number_of_frequency_steps):
            ind_frq_in = (frequency_axis > frq_d[ii]) & (frequency_axis < frq_u[ii])
            TrjMovie_FFT_BinPartitioned[ii, :, :] = TrjMovie_FFT[ind_frq_in].abs().mean(0,True)

        ### Normalize fft by the noise estimated globally before to see if it's significant enough by passing it into the score function: ###
        TrjMovie_FFT_BinPartitioned_NoiseNormalized = (TrjMovie_FFT_BinPartitioned - noise_dc_FFT_over_drone_frequencies) / noise_std_FFT_over_drone_frequencies
        TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index] = ScrFunc_Torch(DetNoiseThresh, TrjMovie_FFT_BinPartitioned_NoiseNormalized)

        ### See at which bins the score was significant and normalize it to 100 for confidence measure: ###
        #(1). sum score over all spatial bins and all FFT bins...this is a bit weird...supposedly because we want to gather "all evidence" from all frequencies and pixels...but still....this can really add up. i don't see any real SNR ratio calculated and thresholded
        pxl_scr[trajectory_index] = (TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index]).sum()
        dec[trajectory_index] = pxl_scr[trajectory_index] > DetScrThresh
        DetectionConfLvl[trajectory_index] = 100 * torch.minimum((pxl_scr[trajectory_index] - DetScrThresh) / (DetScrThresh100Conf - DetScrThresh), torch.Tensor([1]).to(TrjMov[0].device))

    return dec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl






 # ###########################################################################################################3
# ### Sum Of Different Bins In Jumps Of 10 For Plotting: ###
# binned_frequency_axis_len = frequency_axis_torch_binned_for_plot.__len__()
# raw_SNR_per_bin = input_tensor_fft_binned_for_plot / input_tensor_fft_noise_estimation_mean
# raw_SNR_per_bin_logistic = logistic_torch(raw_SNR_per_bin)
# for current_frequency_index in np.arange(binned_frequency_axis_len//2+1,binned_frequency_axis_len):
#     current_title = str(frequency_axis_torch_binned_for_plot[current_frequency_index].int().item())
#     current_title += '[Hz]'
#     imshow_torch(raw_SNR_per_bin_logistic[current_frequency_index]>0.5, title_str=current_title, flag_maximize=False)
#
# TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index] = raw_SNR_per_bin_logistic
# pxl_scr[trajectory_index] = soft_sum.item()
# dec[trajectory_index] = input_tensor_fft_drone_decision_passed
# DetectionConfLvl[trajectory_index] =
#
# TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index] = ScrFunc_Torch(DetNoiseThresh, TrjMovie_FFT_BinPartitioned_NoiseNormalized)
# pxl_scr[trajectory_index] = (TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index]).sum()
# dec[trajectory_index] = pxl_scr[trajectory_index] > DetScrThresh
# DetectionConfLvl[trajectory_index] = 100 * torch.minimum((pxl_scr[trajectory_index] - DetScrThresh) / (DetScrThresh100Conf - DetScrThresh), torch.Tensor([1]).to(TrjMov[0].device))
# # TODO: Maor's function returns: DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl
# ###########################################################################################################3
###########################################################################################################3


# ###########################################################################################################3
# ### Sum Different Pixels (Binning) To Possibly Enhance SNR: ###
# #(1). Get relevant figures for "signal" and "noise":
# #TODO: Add perhapse 1X2 and 2X1 binning,
# # add possibility of counting harmonics, and possibility of returning the entire 11X11 ROI to view what's going on
# #(****). Non Coherent SUMS:
# input_tensor_fft_between_drone_bins = (input_tensor_fft[input_tensor_fft_possible_drone_bins_max_lobe_index].abs()**2).mean(0, True).abs()
# input_tensor_fft_between_noise_bins = (input_tensor_fft[noise_baseline_frequency_bins_logical_mask].abs()**2).mean(0, True).abs().squeeze(1)
# # #(****). Coherent SUMS:
# # input_tensor_fft_between_drone_bins = (input_tensor_fft[input_tensor_fft_possible_drone_bins_max_lobe_index].abs()).mean(0, True) ** 2
# # input_tensor_fft_between_noise_bins = (input_tensor_fft[noise_baseline_frequency_bins_logical_mask]).mean(0, True).abs().squeeze(1) ** 2
# #(2). perform binning:
# #(*). NOTICE: controllable binning on both axes!!!
# input_tensor_fft_1X1_drone_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_drone_bins, (1,1), (0,0))
# input_tensor_fft_2X2_drone_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_drone_bins, (2,2), (1,1))
# input_tensor_fft_4X4_drone_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_drone_bins, (4,4), (3,3))
# input_tensor_fft_1X1_noise_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_noise_bins, (1, 1),(0, 0))
# input_tensor_fft_2X2_noise_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_noise_bins, (2, 2),(1, 1))
# input_tensor_fft_4X4_noise_bins = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_fft_between_noise_bins, (4, 4),(3, 3))
# # #(3). show images of the "Raw" "SNR" results:
# # plt.figure(); plt.imshow((input_tensor_fft_1X1_drone_bins/input_tensor_fft_1X1_noise_bins)[0].cpu().numpy())
# # plt.figure(); plt.imshow((input_tensor_fft_2X2_drone_bins/input_tensor_fft_2X2_noise_bins)[0].cpu().numpy())
# # plt.figure(); plt.imshow((input_tensor_fft_4X4_drone_bins/input_tensor_fft_4X4_noise_bins)[0].cpu().numpy())
# #(4). show thresholded image for SNR:
# drone_SNR_threshold = 10
# input_tensor_fft_1X1_binning_SNR_above_threshold = (input_tensor_fft_1X1_drone_bins / input_tensor_fft_1X1_noise_bins)[0] > drone_SNR_threshold
# input_tensor_fft_2X2_binning_SNR_above_threshold = (input_tensor_fft_2X2_drone_bins / input_tensor_fft_2X2_noise_bins)[0] > drone_SNR_threshold
# input_tensor_fft_4X4_binning_SNR_above_threshold = (input_tensor_fft_4X4_drone_bins / input_tensor_fft_4X4_noise_bins)[0] > drone_SNR_threshold
# # imshow_torch(input_tensor_fft_1X1_binning_SNR_above_threshold)
# # imshow_torch(input_tensor_fft_2X2_binning_SNR_above_threshold)
# # imshow_torch(input_tensor_fft_4X4_binning_SNR_above_threshold)
#
# #(6). get spectrum graphs of several points which passed threshold:
# matches_tensor = torch_get_where_condition_holds(input_tensor_fft_1X1_binning_SNR_above_threshold==True)
# number_of_matches, number_of_dims = matches_tensor.shape
# ffts_list = []
# for match_index in np.arange(number_of_matches):
#     current_H = matches_tensor[match_index, 0]
#     current_W = matches_tensor[match_index, 1]
#     ffts_list.append(input_tensor_fft[:,:,current_H, current_W])
#
# ### Plot Spectrum Graphs of all the points which "survived" as potential suspects
# graphs_folder = os.path.join(params.results_folder_seq, 'FFT_Graphs_SNR_Above_Threshold')
# create_folder_if_doesnt_exist(graphs_folder)
# for i in np.arange(number_of_matches):
#     figure()
#     plt.plot(frequency_axis, ffts_list[i].abs().squeeze().cpu().numpy().clip(0, 10))
#     title_string = str(matches_tensor[i].cpu().numpy())
#     plt.title(title_string)
#     filename = os.path.join(graphs_folder, str(i).zfill(4) + '.png')
#     plt.savefig(filename)
#     plt.close()
# ###########################################################################################################3

####
#TODO: Maor's function returns: DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl
####



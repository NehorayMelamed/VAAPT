from RapidBase.MISC_REPOS.Layers.DownSample_UpSample import UnshufflePixels

from RapidBase.import_all import *

import numpy as np
import torch
import torch.nn.functional as F
# loading images
from PIL import Image
from os import listdir
# import image_registration



# import the necessary packages
import numpy as np
import imutils
import cv2
import torch.fft
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *





# number_of_experiments = 100
# std_list = []
# number_of_bins = 10
# for j in arange(number_of_experiments):
#     number_of_samples = 20
#     bla = np.random.randn(number_of_samples)
#     current_mean = 0
#     previous_mean = 0
#     current_std = 0
#     previous_std = 0
#     for i in arange(1,number_of_samples):
#         current_mean = bla[i] * (1/i) + previous_mean * (1 - 1/i)
#         current_std = np.sqrt((bla[i]-current_mean)**2 * (1/i) + previous_std**2 * (1-1/i))
#         previous_std = current_std
#         previous_mean = current_mean
#     std_list.append(current_std)
#     print(current_std)
# final_histogram = np.histogram(std_list, bins = number_of_bins)
# plot(final_histogram[1][1:], np.array(final_histogram[0]))





def estimate_noise_IterativeResidual(noisy_frame_current,
                                     noise_map_previous,
                                     noise_pixel_counts_map_previous,
                                     last_clean_frame_warped,
                                     downsample_kernel_size=8):
    ### Initialize Layers: ###
    average_pooling_layer = nn.AvgPool2d(downsample_kernel_size)
    upsample_layer = nn.UpsamplingNearest2d(scale_factor=downsample_kernel_size)

    # ### Get current noise estimation from last clean image (better estimation of mean value than spatial mean): ###
    # reference_image = upsample_layer(average_pooling_layer(noisy_frame_current))
    reference_image = upsample_layer(average_pooling_layer(last_clean_frame_warped))
    # reference_image = last_clean_frame_warped
    difference_map_squared = (noisy_frame_current - reference_image).abs() ** 2
    difference_map_squared_averaged = upsample_layer(average_pooling_layer(difference_map_squared).mean(0,False).unsqueeze(0))  #if we input image batch average over frames
    difference_map_averaged = torch.sqrt(difference_map_squared_averaged)

    ### Update Noise Map: ###
    if noise_map_previous is None:
        noise_map_current = difference_map_averaged
        noise_pixel_counts_map_current = noise_pixel_counts_map_previous
    else:
        # ### Get Reset Gate and Uptick Noise Pixel Counts Map: ###
        # noise_map_distance = torch.abs(difference_map_averaged - noise_map_previous)  # (0, 0.07)
        # noise_map_ratio = (difference_map_averaged / noise_map_previous)  # (0.8, 1.2)
        # min_value_distance_activation = 0.  # sigma difference below which i consider statistical fluctuations
        # max_value_distance_activation = 0.06  # sigma difference above which i consider us having a new noise level
        # noise_map_distance_slope = (1 - 0) / (max_value_distance_activation - min_value_distance_activation)
        # reset_gate = torch.clip(1 + min_value_distance_activation - noise_map_distance * noise_map_distance_slope, 0, 1)

        ### 0 -> 1 -> 0: ###
        normalized_distance = torch.abs(difference_map_averaged/(noise_map_previous+1e-7))
        min_value_distance_activation = 1 - 1 / downsample_kernel_size * 1
        max_value_distance_activation = 1 + 1 / downsample_kernel_size * 1
        normalized_distance_slope = (1 - 0) / (max_value_distance_activation - min_value_distance_activation)
        x_max = max_value_distance_activation * (1 + min_value_distance_activation)
        x_min = 0
        x_0 = (x_max + x_min) / 2
        Base = (x_max - x_min)
        y_max = Base * normalized_distance_slope / 2
        reset_gate = torch.clip(y_max - normalized_distance_slope * torch.abs(normalized_distance - x_0), 0, 1)

        ### Update pixel counts map: ###
        noise_pixel_counts_map_current = noise_pixel_counts_map_previous * reset_gate + 1

        ### Get Updated Noise Map: ###
        noise_map_current = torch.sqrt(difference_map_squared_averaged ** 1 * (1 / noise_pixel_counts_map_current) +
                                       noise_map_previous ** 2 * (1 - 1 / noise_pixel_counts_map_current))

    return noise_map_current, noise_pixel_counts_map_current


def plot_and_save_all_noise_estimates(save_path,
                                      noisy_frames_current,
                                      clean_frames_estimate,
                                      histogram_spatial_binning_factor,
                                      crude_histogram_bins_vec,
                                      downsample_kernel_size, max_value_possible
                                      ):
    ### Use Histogram Width Around Clean Frame Estimate To Estimate Noise: ###
    # TODO: there's a factor of ~1.31 of noise_as_function_of_gray_level_width/noise_as_function_of_gray_level_peak, see if this is also in gaussian case!!!!!
    noise_map_estimate_width, SNR_estimate_width, noise_as_function_of_gray_level_width, histograms_bins_list, histograms_list = \
        estimate_noise_ImagesBatch(noisy_frames_current,
                                   clean_frames_estimate,
                                   'histogram_width',
                                   # 'spatial_differences', 'difference_histogram_peak', 'histogram_width', 'per_pixel_histogram'
                                   histogram_spatial_binning_factor,
                                   crude_histogram_bins_vec,
                                   downsample_kernel_size=downsample_kernel_size, max_value_possible=max_value_possible)

    ### Save Noise_Map, SNR_Estimate and Clean_Frame_Estimate: ###
    path_histogram_width = os.path.join(save_path, 'Histograms_Width')
    path_make_path_if_none_exists(path_histogram_width)
    save_image_numpy(path_histogram_width, 'clean_frame_estimate.png',
                     (clean_frames_estimate.squeeze().cpu().numpy()).astype(np.uint16), False, flag_scale=False,
                     flag_save_uint16=True)
    save_image_numpy(path_histogram_width, 'noise_map_estimate.png',
                     (noise_map_estimate_width.squeeze().cpu().numpy()).astype(np.uint16), False, flag_scale=False,
                     flag_save_uint16=True)
    save_image_numpy(path_histogram_width, 'SNR_estimate.png',
                     (SNR_estimate_width.squeeze().cpu().numpy()).astype(np.uint8), False, flag_scale=False,
                     flag_save_uint16=True)

    save_image_numpy(path_histogram_width, 'clean_frame_estimate_stretch.png',
                     ((255 * gray2color_torch(stretch_tensor_quantiles(clean_frames_estimate, 0.00, 1),
                                              1)).squeeze().permute([1, 2, 0]).cpu().numpy()).astype(np.uint8), False,
                     flag_scale=False, flag_save_uint16=False)
    save_image_numpy(path_histogram_width, 'noise_map_estimate_stretch.png',
                     ((255 * gray2color_torch(stretch_tensor_quantiles(noise_map_estimate_width, 0.00, 1),
                                              1)).squeeze().permute([1, 2, 0]).cpu().numpy()).astype(np.uint8), False,
                     flag_scale=False, flag_save_uint16=False)
    save_image_numpy(path_histogram_width, 'SNR_estimate_stretch.png',
                     ((255 * gray2color_torch(stretch_tensor_quantiles(SNR_estimate_width, 0.00, 1),
                                              1)).squeeze().permute([1, 2, 0]).cpu().numpy()).astype(np.uint8), False,
                     flag_scale=False, flag_save_uint16=False)

    fig = imshow_torch(clean_frames_estimate)
    figure_image = video_get_mat_from_figure(fig, (1024 * 2, 1024 * 2))
    save_image_numpy(path_histogram_width, 'clean_frame_estimate_figure.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()
    fig = imshow_torch(noise_map_estimate_width)
    figure_image = video_get_mat_from_figure(fig, (1024 * 2, 1024 * 2))
    save_image_numpy(path_histogram_width, 'noise_map_estimate_width_figure.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()
    fig = imshow_torch(SNR_estimate_width)
    figure_image = video_get_mat_from_figure(fig, (1024 * 2, 1024 * 2))
    save_image_numpy(path_histogram_width, 'SNR_estimate_width_figure.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()

    ### Plot Noise Level As Function Of Gray-Level: ###
    noise_levels_vec = np.arange(max_value_possible)
    fig = figure()
    plot(noise_levels_vec, noise_as_function_of_gray_level_width)
    xlabel('gray level')
    ylabel('noise level [gray levels]')
    figure_image = video_get_mat_from_figure(fig, (1024, 1024))
    save_image_numpy(path_histogram_width, 'Noise_As_Function_Of_GrayLevel.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()

    ### Plot and Save Crudely Binned Histograms: ###
    for i in np.arange(len(histograms_bins_list)):
        current_histogram_bins = histograms_bins_list[i]
        current_histogram = histograms_list[i]
        if len(current_histogram_bins) > 1:
            print('yes')
            fig = figure()
            plot(current_histogram_bins, current_histogram)
            title('histogram of values between: ' + decimal_notation(crude_histogram_bins_vec[i],
                                                                     0) + '-' + decimal_notation(
                crude_histogram_bins_vec[i + 1], 0))
            xlabel('difference from average')
            figure_image = video_get_mat_from_figure(fig, (1024, 1024))
            save_image_numpy(path_histogram_width, 'Hist_' + str(i) + '.png',
                             figure_image, False, flag_scale=False, flag_save_uint16=False)
            plt.close()
    #############################################################################################################################################

    #############################################################################################################################################
    ### Get Noise Profile Using Histogram Peak Placing Algorithm: ###
    clean_frames_estimate = noisy_frames_current.median(0)[0].unsqueeze(0)

    ### Use Histogram Width Around Clean Frame Estimate To Estimate Noise: ###
    noise_map_estimate_peak, SNR_estimate_peak, noise_as_function_of_gray_level_peak, histograms_bins_list, histograms_list = \
        estimate_noise_ImagesBatch(noisy_frames_current,
                                   clean_frames_estimate,
                                   'difference_histogram_peak',
                                   # 'spatial_differences', 'difference_histogram_peak', 'histogram_width', 'per_pixel_histogram'
                                   histogram_spatial_binning_factor,
                                   crude_histogram_bins_vec,
                                   downsample_kernel_size=downsample_kernel_size, max_value_possible=max_value_possible)

    ### Save Noise_Map, SNR_Estimate and Clean_Frame_Estimate: ###
    path_histogram_peak = os.path.join(save_path, 'Histograms_Peak')
    path_make_path_if_none_exists(path_histogram_peak)
    save_image_numpy(path_histogram_peak, 'clean_frame_estimate.png',
                     (clean_frames_estimate.squeeze().cpu().numpy()).astype(np.uint16), False, flag_scale=False,
                     flag_save_uint16=True)
    save_image_numpy(path_histogram_peak, 'noise_map_estimate.png',
                     (noise_map_estimate_peak.squeeze().cpu().numpy()).astype(np.uint16), False, flag_scale=False,
                     flag_save_uint16=True)
    save_image_numpy(path_histogram_peak, 'SNR_estimate.png',
                     (SNR_estimate_peak.squeeze().cpu().numpy()).astype(np.uint8), False, flag_scale=False,
                     flag_save_uint16=True)

    save_image_numpy(path_histogram_peak, 'clean_frame_estimate_stretch.png',
                     ((255 * gray2color_torch(stretch_tensor_quantiles(clean_frames_estimate, 0.00, 1),
                                              1)).squeeze().permute([1, 2, 0]).cpu().numpy()).astype(np.uint8), False,
                     flag_scale=False, flag_save_uint16=False)
    save_image_numpy(path_histogram_peak, 'noise_map_estimate_stretch.png',
                     ((255 * gray2color_torch(stretch_tensor_quantiles(noise_map_estimate_peak, 0.00, 1),
                                              1)).squeeze().permute([1, 2, 0]).cpu().numpy()).astype(np.uint8), False,
                     flag_scale=False, flag_save_uint16=False)
    save_image_numpy(path_histogram_peak, 'SNR_estimate_stretch.png',
                     ((255 * gray2color_torch(stretch_tensor_quantiles(SNR_estimate_peak, 0.00, 1),
                                              1)).squeeze().permute([1, 2, 0]).cpu().numpy()).astype(np.uint8), False,
                     flag_scale=False, flag_save_uint16=False)

    fig = imshow_torch(clean_frames_estimate)
    figure_image = video_get_mat_from_figure(fig, (1024 * 2, 1024 * 2))
    save_image_numpy(path_histogram_peak, 'clean_frame_estimate_figure.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()
    fig = imshow_torch(noise_map_estimate_peak)
    figure_image = video_get_mat_from_figure(fig, (1024 * 2, 1024 * 2))
    save_image_numpy(path_histogram_peak, 'noise_map_estimate_peak_figure.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()
    fig = imshow_torch(SNR_estimate_peak)
    figure_image = video_get_mat_from_figure(fig, (1024 * 2, 1024 * 2))
    save_image_numpy(path_histogram_peak, 'SNR_estimate_peak_figure.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()

    ### Plot Noise Level As Function Of Gray-Level: ###
    noise_levels_vec = np.arange(max_value_possible)
    fig = figure()
    plot(noise_levels_vec, noise_as_function_of_gray_level_peak)
    xlabel('gray level')
    ylabel('noise level [gray levels]')
    figure_image = video_get_mat_from_figure(fig, (1024 * 2, 1024 * 2))
    save_image_numpy(path_histogram_peak, 'Noise_As_Function_Of_GrayLevel.png',
                     figure_image, False, flag_scale=False, flag_save_uint16=False)
    plt.close()

    ### Plot and Save Crudely Binned Histograms: ###
    for i in np.arange(len(histograms_bins_list)):
        current_histogram_bins = histograms_bins_list[i]
        current_histogram = histograms_list[i]
        if len(current_histogram_bins) > 1:
            print('yes')
            fig = figure()
            plot(current_histogram_bins, current_histogram)
            title('histogram of values between: ' + decimal_notation(crude_histogram_bins_vec[i],
                                                                     0) + '-' + decimal_notation(
                crude_histogram_bins_vec[i + 1], 0))
            xlabel('Spatial Mean Of Difference From Average')
            figure_image = video_get_mat_from_figure(fig, (1024, 1024))
            save_image_numpy(path_histogram_peak, 'Hist_' + str(i) + '.png',
                             figure_image, False, flag_scale=False, flag_save_uint16=False)
            plt.close()
    #############################################################################################################################################


def estimate_noise_ImagesBatch(noisy_frames,
                              clean_frames_estimate,
                              algorithm_type,
                              histogram_spatial_binning_factor,
                              crude_histogram_bins_vec,
                              downsample_kernel_size, max_value_possible=256):  #'spatial_differences', 'difference_histogram_peak', 'histogram_width', 'per_pixel_histogram'
    # ### Experimental Part - TEMP TO BE DELETED: ###
    # sigma_noise = 10
    # histogram_spatial_binning_factor = 4
    # clean_frames_estimate = get_random_number_in_range(0, 255, (50, 1, 1000, 1000))
    # noisy_frames = clean_frames_estimate + sigma_noise * np.random.randn(50, 1, 1000, 1000)
    # noisy_frames = noisy_frames.clip(0, 255)
    # crude_histogram_bins_vec = my_linspace(0, 255, 10)
    # clean_frames_estimate = torch.Tensor(clean_frames_estimate)
    # noisy_frames = torch.Tensor(noisy_frames)
    # imshow_torch((clean_frames_estimate / ((noisy_frames[:,:,:,:] - clean_frames_estimate).std(0).unsqueeze(0) + 1e-3)).clamp(0, 20))
    # imshow_torch(clean_frames_estimate)

    ### Do binning: ###
    histogram_spatial_size_y = noisy_frames.shape[-2]
    histogram_spatial_size_x = noisy_frames.shape[-1]


    # ### Initialize Layers: ###
    average_pooling_layer = nn.AvgPool2d(downsample_kernel_size)
    upsample_layer = nn.UpsamplingNearest2d(scale_factor=downsample_kernel_size)

    if algorithm_type == 'spatial_differences':
        # reference_image = upsample_layer(average_pooling_layer(noisy_frames))
        reference_image = clean_frames_estimate
        difference_map_squared = (noisy_frames - reference_image).abs() ** 2
        difference_map_squared_averaged = upsample_layer(average_pooling_layer(difference_map_squared).mean(0, True))  # if we input image batch average over frames
        noise_estimate = torch.sqrt(difference_map_squared_averaged)
        SNR_estimate = clean_frames_estimate / (noise_estimate + 1e-4)
        return noise_estimate, SNR_estimate

    elif algorithm_type == 'difference_histogram_peak':
        ############################################################################################################################################################
        ### Whole Image Approach - DIFFERENCE HISTOGRAM PEAK!!!!: ###
        ### Get Histogram with predetermined bins: ###
        crude_histogram_number_of_bins = len(crude_histogram_bins_vec) - 1
        crude_histogram_bin_centers = (crude_histogram_bins_vec[0:-1] + crude_histogram_bins_vec[1:]) / 2
        ### Loop over the "crude" bins of the entire image: ###
        crude_bins_std_list = []
        histograms_list = []
        histograms_bins_list = []
        for bin_counter in np.arange(crude_histogram_number_of_bins):
            # bin_counter = 2
            ### Get logical array of pixels within the predetermined bins: ###
            clean_frame_estimate_average_pooling = nn.AvgPool2d(histogram_spatial_binning_factor)(clean_frames_estimate)
            logical_array = (clean_frame_estimate_average_pooling > crude_histogram_bins_vec[bin_counter]) * \
                            (clean_frame_estimate_average_pooling < crude_histogram_bins_vec[bin_counter + 1])
            logical_array = torch.cat([logical_array] * noisy_frames.shape[0], 0)
            # print(logical_array.sum())

            minimum_number_of_samples = 800
            if logical_array.sum() > minimum_number_of_samples:
                ### Get differences between currently relevant pixels of noisy frame and reference frame: ###
                differences_between_relevant_pixels = ((noisy_frames - clean_frames_estimate) ** 1).abs()

                ### Do Spatial Binning & Addition: ###
                differences_between_relevant_pixels_average_pooling = nn.AvgPool2d(histogram_spatial_binning_factor)(differences_between_relevant_pixels)[logical_array]

                ### Get Histogram of the values in the current "crude bin" differences sum: ###
                current_std = differences_between_relevant_pixels_average_pooling.std()
                current_mean = differences_between_relevant_pixels_average_pooling.mean()
                difference_histogram_bins_vec = my_linspace(max(0,current_mean - 2 * current_std), current_mean + 2 * current_std, 20)
                histogram_counts, x_bins = np.histogram(differences_between_relevant_pixels_average_pooling,
                                                        bins=difference_histogram_bins_vec)
                histogram_bin_centers = (difference_histogram_bins_vec[0:-1] + difference_histogram_bins_vec[1:]) / 2
                histograms_list.append(histogram_counts)
                histograms_bins_list.append(histogram_bin_centers)
                histogram_statistical_distribution = histogram_counts / histogram_counts.sum()
                # figure()
                # plot(histogram_bin_centers, histogram_statistical_distribution)

                ### Use Built histogram and fit gaussian over it: ###
                x_max, z_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(histogram_statistical_distribution,histogram_bin_centers)
                crude_bins_std_list.append(x_max)  # std
            else:
                crude_bins_std_list.append(np.inf)  # std
                histograms_list.append([0])
                histograms_bins_list.append([0])

        ### Interpolate to all Gray-Levels: ###
        from scipy.interpolate import interp1d
        crude_bins_std_list = np.array(crude_bins_std_list)
        valid_indices = crude_bins_std_list != np.inf
        crude_bins_std_list = crude_bins_std_list[valid_indices]
        crude_histogram_bin_centers = crude_histogram_bin_centers[valid_indices]
        crude_histogram_bin_centers = (crude_histogram_bin_centers * 1).astype(np.int)
        bin_start = crude_histogram_bin_centers[0]
        bin_stop = crude_histogram_bin_centers[-1]
        gray_levels_vec_to_interpolate_at = np.arange(bin_start, bin_stop, 1)
        all_gray_levels_vec = my_linspace(0, max_value_possible, max_value_possible)
        y_function = interp1d(crude_histogram_bin_centers.astype(np.int), crude_bins_std_list)
        noise_per_gray_level_list = list(y_function(gray_levels_vec_to_interpolate_at))
        noise_per_gray_level_list = [noise_per_gray_level_list[0]] * (int(crude_histogram_bin_centers[0])) + \
                                    noise_per_gray_level_list + \
                                    [noise_per_gray_level_list[-1]] * (max_value_possible - int(crude_histogram_bin_centers[-1]))
        # plot(all_gray_levels_vec, noise_per_gray_level_list)
        y_function_all = interp1d(all_gray_levels_vec, noise_per_gray_level_list)
        clean_frames_estimate_average_pooling = upsample_layer(average_pooling_layer(clean_frames_estimate))
        noise_estimate = torch.Tensor(y_function_all(clean_frames_estimate_average_pooling.clamp(0, max(all_gray_levels_vec))))

        ### Correct Noise Estimatino By Correction Factor From Peak Position To STD: ###
        noise_estimation_factor_dict = {'2':0.71, '3':0.75, '4':0.79}  # 2 -> 0.71  , 3 -> 0.75 ,  4 -> 0.79
        noise_estimate = noise_estimate / noise_estimation_factor_dict[str(histogram_spatial_binning_factor)]
        SNR_estimate = clean_frames_estimate_average_pooling / (noise_estimate + 1e-7)
        return noise_estimate, SNR_estimate, noise_per_gray_level_list, histograms_bins_list, histograms_list
        ############################################################################################################################################################

    elif algorithm_type == 'histogram_width':
        ############################################################################################################################################################
        ### Whole Image Approach - Histogram WIDTH!!!!: ###
        ### Get Histogram with predetermined bins: ###
        crude_histogram_number_of_bins = len(crude_histogram_bins_vec) - 1
        crude_histogram_bin_centers = (crude_histogram_bins_vec[0:-1] + crude_histogram_bins_vec[1:]) / 2
        ### Loop over the "crude" bins of the entire image: ###
        crude_bins_mean_list = []
        crude_bins_std_list = []
        histograms_list = []
        histograms_bins_list = []
        for bin_counter in np.arange(crude_histogram_number_of_bins):
            # bin_counter = 2
            ### Get logical array of pixels within the predetermined bins: ###
            logical_array = (clean_frames_estimate > crude_histogram_bins_vec[bin_counter]) * \
                            (clean_frames_estimate < crude_histogram_bins_vec[bin_counter + 1])
            logical_array = torch.cat([logical_array]*noisy_frames.shape[0],0)
            print(logical_array.sum())

            minimum_number_of_samples = 800
            if logical_array.sum() > minimum_number_of_samples:
                ### Get differences between currently relevant pixels of noisy frame and reference frame: ###
                differences_between_relevant_pixels = ((noisy_frames - clean_frames_estimate) ** 1)[logical_array]

                ### Get Histogram of the values in the current "crude bin" differences sum: ###
                current_std = differences_between_relevant_pixels.std()
                current_mean = differences_between_relevant_pixels.mean()
                difference_histogram_bins_vec = my_linspace(current_mean - 3*current_std,current_mean + 3*current_std,30)
                histogram_counts, x_bins = np.histogram(differences_between_relevant_pixels, bins=difference_histogram_bins_vec)
                histogram_bin_centers = (difference_histogram_bins_vec[0:-1] + difference_histogram_bins_vec[1:]) / 2
                histograms_list.append(histogram_counts)
                histograms_bins_list.append(histogram_bin_centers)
                histogram_statistical_distribution = histogram_counts/histogram_counts.sum()
                # figure()
                # plot(histogram_bin_centers, histogram_statistical_distribution)

                ### Use Built histogram and fit gaussian over it: ###
                #(*). TODO: this loops over different x_vec values trying to find a good fit. put all of this into one robust function!
                i = 0
                RMSE_list = []
                pops_list = []
                y_fitted_list = []
                while i<4:
                    try:
                        y_fitted, pops, pcov = fit_Gaussian(histogram_bin_centers*(255**i), histogram_statistical_distribution)
                        pops[1] = pops[1] / (255**i)
                        pops[2] = pops[2] / (255**i)
                        RMSE_list.append(np.abs(y_fitted-histogram_statistical_distribution).mean())
                        pops_list.append(pops)
                        y_fitted_list.append(y_fitted)
                    except:
                        1
                    i += 1
                min_RMSE_index = np.argmin(RMSE_list)
                pops = pops_list[min_RMSE_index]
                y_fitted = y_fitted_list[min_RMSE_index]
                crude_bins_mean_list.append(pops[1])  # mean
                crude_bins_std_list.append(abs(pops[2]))  # std
                # plot(histogram_bin_centers, y_fitted)
            else:
                crude_bins_mean_list.append(np.inf)  # mean
                crude_bins_std_list.append(np.inf)  # std
                histograms_list.append([0])
                histograms_bins_list.append([0])


        ### Interpolate to all Gray-Levels: ###
        from scipy.interpolate import interp1d
        crude_bins_mean_list = np.array(crude_bins_mean_list)
        crude_bins_std_list = np.array(crude_bins_std_list)
        valid_indices = crude_bins_mean_list != np.inf
        crude_bins_mean_list = crude_bins_mean_list[valid_indices]
        crude_bins_std_list = crude_bins_std_list[valid_indices]
        crude_histogram_bin_centers = crude_histogram_bin_centers[valid_indices]
        crude_histogram_bin_centers = (crude_histogram_bin_centers * 1).astype(np.int)
        bin_start = crude_histogram_bin_centers[0]
        bin_stop = crude_histogram_bin_centers[-1]
        gray_levels_vec_to_interpolate_at = np.arange(bin_start,bin_stop,1)
        all_gray_levels_vec = my_linspace(0,max_value_possible,max_value_possible)
        y_function = interp1d(crude_histogram_bin_centers.astype(np.int), crude_bins_std_list)
        noise_per_gray_level_list = list(y_function(gray_levels_vec_to_interpolate_at))
        noise_per_gray_level_list = [noise_per_gray_level_list[0]]*(int(crude_histogram_bin_centers[0])) + \
                                    noise_per_gray_level_list + \
                                    [noise_per_gray_level_list[-1]]*(max_value_possible-int(crude_histogram_bin_centers[-1]))
        # plot(all_gray_levels_vec, noise_per_gray_level_list)
        y_function_all = interp1d(all_gray_levels_vec, noise_per_gray_level_list)
        clean_frames_estimate_average_pooling = upsample_layer(average_pooling_layer(clean_frames_estimate))
        noise_estimate = torch.Tensor(y_function_all(clean_frames_estimate_average_pooling.clamp(0,max(all_gray_levels_vec))))
        SNR_estimate = clean_frames_estimate_average_pooling / (noise_estimate + 1e-7)
        return noise_estimate, SNR_estimate, noise_per_gray_level_list, histograms_bins_list, histograms_list
        ############################################################################################################################################################

    elif algorithm_type == 'per_pixel_histogram':
        #TODO: complete this
        ############################################################################################################################################################v
        ### Per Pixel Approach: ###
        ### Do binning: ###
        shuffle_layer = UnshufflePixels(histogram_spatial_binning_factor)
        noisy_frames_shuffled = shuffle_layer(noisy_frames)
        number_of_bins_around_mean_value = 10

        histogram_matrix = np.zeros((number_of_bins_around_mean_value - 1, histogram_spatial_size_y, histogram_spatial_size_x))
        mean_mat = np.zeros((histogram_spatial_size_x, histogram_spatial_size_y))
        std_mat = np.zeros((histogram_spatial_size_x, histogram_spatial_size_y))
        for i in np.arange(histogram_spatial_size_y):
            for j in np.arange(histogram_spatial_size_x):
                ### Build Histogram over predefined space-time chunk: ###
                mean_value = noisy_frames_shuffled[:, i, j].mean()
                std_value = noisy_frames_shuffled[:, i, j].std()
                histogram_bins_vec = my_linspace(mean_value - std_value * 3, mean_value + std_value * 3,
                                                 number_of_bins_around_mean_value)
                histogram_matrix[:, i, j], x_bins = np.histogram(noisy_frames_shuffled[:, i, j],
                                                                 bins=histogram_bins_vec)
                histogram_bin_centers = (histogram_bins_vec[0:-1] + histogram_bins_vec[1:]) / 2
                plot(histogram_bin_centers, histogram_matrix[:, i, j])

                ### Use Built histogram and fit gaussian over it: ###
                y_fitted, pops = fit_Gaussian(histogram_bin_centers, histogram_matrix[:, i, j])
                mean_mat[i, j] = pops[1]  # mean
                std_mat[i, j] = pops[2]  # std

        # ### Test on pure white noise: ###
        # number_of_experiments = 30
        # area_in_pixels_to_average_out = 4  # 2 -> 0.71  , 3 -> 0.75 ,  4 -> 0.79
        # list_of_max_bin = []
        # list_of_mean_bin = []
        # for i in np.arange(number_of_experiments):
        #     bla = 0
        #     for j in np.arange(area_in_pixels_to_average_out**2):
        #         bla += np.abs(np.random.randn(2000000))**1 / area_in_pixels_to_average_out**2
        #     # bla = (np.random.randn(10000))**1 - (np.random.randn(10000))**1
        #     bins_vec = my_linspace(0, 5, 300)
        #     bins_vec_centers = (bins_vec[0:-1] + bins_vec[1:]) / 2
        #     histogram_counts, x_bins = np.histogram(bla, bins=bins_vec)
        #     # figure()
        #     # plot(bins_vec_centers, histogram_counts)
        #     max_index = np.argmax(histogram_counts)
        #     max_bin = x_bins[max_index]
        #     # mean_bin = (x_bins[1:]*histogram_counts).sum()/(histogram_counts).sum()
        #     mean_bin = (x_bins[0:-1]*histogram_counts).sum()/(histogram_counts).sum()
        #     # print('dist mean: ' + str(bla.mean()))
        #     print('hist max bin: ' + str(max_bin))
        #     list_of_max_bin.append(max_bin)
        #     list_of_mean_bin.append(mean_bin)
        # print('dist max: ' + str(mean(list_of_max_bin)))   # 1.24
        # print('current distribution simple mean / average: ' + str(bla.mean()))  # 1.59
        # print('distribution expection (mean): ' + str(mean(list_of_mean_bin)))  # 1.54
        ############################################################################################################################################################v


def estimate_noise_WholeImageHistogram_Rolling(noisy_frame_current,
                                     noisy_frame_previous,
                                     noise_map_previous,
                                     noise_pixel_counts_map_previous,
                                     last_clean_frame_warped,
                                     histogram_history_matrix,  #Previous histogram / histograms
                                     histogram_counter,
                                     crude_histogram_bins_vec,
                                     difference_histogram_bins_vec,
                                     downsample_kernel_size=8):
    average_pooling_layer = nn.AvgPool2d(downsample_kernel_size)

    # ### Experimental Part - TEMP TO BE DELETED: ###
    # average_pooling_layer = nn.AvgPool2d(1)
    # upsample_layer = nn.UpsamplingNearest2d(scale_factor=downsample_kernel_size)
    # sigma_noise = 10
    # last_clean_frame_warped = get_random_number_in_range(0, 255, (1000, 1000))
    # noisy_frame_current = last_clean_frame_warped + sigma_noise * np.random.randn(1000, 1000)
    # noisy_frame_current = noisy_frame_current.clip(0, 255)
    # noisy_frame_current2 = last_clean_frame_warped + sigma_noise * np.random.randn(1000, 1000)
    # noisy_frame_current2 = noisy_frame_current2.clip(0, 255)
    # crude_histogram_bins_vec = my_linspace(0, 255, 10)
    # difference_histogram_bins_vec = my_linspace(0, 20, 20)

    ### Do downsampling if wanted: ###
    noisy_frame_current_downsampled = average_pooling_layer(noisy_frame_current)
    noisy_frame_previous_downsampled = average_pooling_layer(noisy_frame_previous)
    last_clean_frame_warped_downsampled = average_pooling_layer(last_clean_frame_warped)

    ### Get Histogram with predetermined bins: ###
    crude_histogram_number_of_bins = len(crude_histogram_bins_vec) - 1
    crude_histogram_bin_centers = (crude_histogram_bins_vec[0:-1] + crude_histogram_bins_vec[1:]) / 2

    difference_histogram_number_of_bins = len(difference_histogram_bins_vec) - 1
    difference_histogram_bin_centers = (difference_histogram_bins_vec[0:-1] + difference_histogram_bins_vec[1:]) / 2

    ### Loop over the "crude" bins of the entire image: ###
    crude_bins_max_list = []
    for bin_counter in np.arange(crude_histogram_number_of_bins):
        bin_counter = 2
        ### Get logical array of pixels within the predetermined bins: ###
        logical_array = (last_clean_frame_warped_downsampled > crude_histogram_bins_vec[bin_counter]) * \
                        (last_clean_frame_warped_downsampled < crude_histogram_bins_vec[bin_counter + 1])

        ### Get differences between currently relevant pixels of noisy frame and reference frame: ###
        differences_between_relevant_pixels = (np.abs(noisy_frame_current_downsampled - last_clean_frame_warped_downsampled) ** 1)[logical_array]
        differences_between_relevant_pixels2 = (np.abs(noisy_frame_previous_downsampled - last_clean_frame_warped_downsampled) ** 1)[logical_array]
        differences_between_relevant_pixels += differences_between_relevant_pixels2

        ### Get Histogram of the values in the current "crude bin" differences sum: ###
        histogram_counts, x_bins = np.histogram(differences_between_relevant_pixels, bins=difference_histogram_bins_vec)
        # plot(difference_histogram_bin_centers, histogram_counts)

        ### Get Histogram max: ###
        x_max, y_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(histogram_counts, x_vec=x_bins)
        crude_bins_max_list.append(x_max)

    ### Interpolate to all gray-levels: ###





def estimate_noise_PerPixelHistogram_Rolling(noisy_frame_current,
                                      noise_map_previous,
                                      noise_pixel_counts_map_previous,
                                      last_clean_frame_warped,
                                      histogram_history_matrix,
                                      histogram_counter,
                                      histogram_spatial_binning_factor,
                                      histogram_number_of_images,
                                      downsample_kernel_size,
                                      flag_get_hist):
    ### This function uses histograms / MaximumLikelihood: ###
    ### Do binning: ###
    shuffle_layer = UnshufflePixels(histogram_spatial_binning_factor)
    noisy_frame_current_shuffled = shuffle_layer(noisy_frame_current)

    ### Add current binned/shuffled input mat to histogram mat: ###
    histogram_counter_rolling = (histogram_counter) % histogram_number_of_images
    histogram_history_matrix[histogram_counter_rolling * histogram_spatial_binning_factor ** 2:
                             (histogram_counter_rolling + 1) * histogram_spatial_binning_factor ** 2, :, :] = noisy_frame_current_shuffled.squeeze(0).numpy()
    histogram_counter = histogram_counter + 1
    flag_enter = (histogram_counter > histogram_number_of_images)
    histogram_counter_rolling = (histogram_counter) % histogram_number_of_images
    histogram_spatial_size_x = noisy_frame_current_shuffled.shape[-1]
    histogram_spatial_size_y = noisy_frame_current_shuffled.shape[-2]

    ### Get Histogram: ###
    if flag_get_hist or flag_enter:
        flag_get_hist = True

        ### Loop over each pixel and gather histogram over time (and space if there is binning), for the first time: ###
        number_of_bins_around_mean_value = 10
        histogram_matrix = np.zeros((number_of_bins_around_mean_value - 1, histogram_spatial_size_y, histogram_spatial_size_x))
        mean_mat = np.zeros((histogram_spatial_size_x, histogram_spatial_size_y))
        std_mat = np.zeros((histogram_spatial_size_x, histogram_spatial_size_y))
        for i in np.arange(histogram_spatial_size_y):
            for j in np.arange(histogram_spatial_size_x):
                ### Build Histogram over predefined space-time chunk: ###
                # TODO: current way is problematic because sometimes there isn't enough "meat" for the gaussian fit
                mean_value = histogram_history_matrix[:, i, j].mean()
                std_value = histogram_history_matrix[:, i, j].std()
                histogram_bins_vec = my_linspace(mean_value-std_value*3, mean_value+std_value*3, number_of_bins_around_mean_value)
                histogram_matrix[:, i, j], x_bins = np.histogram(histogram_history_matrix[:, i, j],
                                                                 bins=histogram_bins_vec)
                histogram_bin_centers = (histogram_bins_vec[0:-1] + histogram_bins_vec[1:]) / 2
                plot(histogram_bin_centers, histogram_matrix[:, i, j])

                ### Use Built histogram and fit gaussian over it: ###
                y_fitted, pops = fit_Gaussian(histogram_bin_centers, histogram_matrix[:, i, j])
                mean_mat[i, j] = pops[1]  # mean
                std_mat[i, j] = pops[2]  # std

    else: #if i don't use histogram, then use the iterative algorithm
        average_pooling_layer = nn.AvgPool2d(1)
        upsample_layer = nn.UpsamplingNearest2d(scale_factor=downsample_kernel_size)

        reference_image = last_clean_frame_warped
        difference_map_squared = (noisy_frame_current - reference_image).abs() ** 2
        difference_map_squared_averaged = upsample_layer(average_pooling_layer(difference_map_squared))
        difference_map_averaged = torch.sqrt(difference_map_squared_averaged)

        ### Update Noise Map: ###
        if noise_map_previous is None:
            noise_map_current = difference_map_averaged
            noise_pixel_counts_map_current = noise_pixel_counts_map_previous
        else:
            ### Get Reset Gate and Uptick Noise Pixel Counts Map: ###
            noise_map_distance = torch.abs(difference_map_averaged - noise_map_previous)  # (0, 0.07)
            noise_map_ratio = (difference_map_averaged / noise_map_previous)  # (0.8, 1.2)
            min_value_distance_activation = 0.  # sigma difference below which i consider statistical fluctuations
            max_value_distance_activation = 0.06  # sigma difference above which i consider us having a new noise level
            noise_map_distance_slope = (1 - 0) / (max_value_distance_activation - min_value_distance_activation)
            reset_gate = torch.clip(1 + min_value_distance_activation - noise_map_distance * noise_map_distance_slope,
                                    0, 1)
            noise_pixel_counts_map_current = noise_pixel_counts_map_previous * reset_gate + 1

            ### Get Updated Noise Map: ###
            noise_map_current = torch.sqrt(difference_map_squared_averaged ** 1 * (1 / noise_pixel_counts_map_current) +
                                           noise_map_previous ** 2 * (1 - 1 / noise_pixel_counts_map_current))

    return noise_map_current, noise_pixel_counts_map_current, flag_get_hist, histogram_counter




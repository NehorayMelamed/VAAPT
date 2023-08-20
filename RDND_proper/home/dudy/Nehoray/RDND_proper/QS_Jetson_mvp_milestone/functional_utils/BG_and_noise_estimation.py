import cv2
import torch

from RapidBase.import_all import *
from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import *
import kornia

def get_BG_over_entire_movie(f, params):
    #TODO: enable entering quantiles from outside to be consistent throughout all algorithm aprrt
    roi = params['roi']
    utype = params['utype']

    flag_keep_going = True
    count = 0
    median_images_list = []
    f = initialize_binary_file_reader(f)
    number_of_frame_batches_for_BG_estimation = np.inf
    number_of_frames_per_batch = 500
    max_number_of_counts = 10
    while flag_keep_going and count < number_of_frame_batches_for_BG_estimation and count<max_number_of_counts:
        print(count)
        ### Read frame: ###
        current_frames = read_frames_from_binary_file_stream(f,
                                                            number_of_frames_to_read=number_of_frames_per_batch,
                                                            number_of_frames_to_skip=0,
                                                            params=params)  # TODO: turn this into a general function which accepts dtype, length, roi_size etc'
        T, H, W = current_frames.shape

        ### If done reading then stop: ###
        if current_frames.shape[0] < number_of_frames_per_batch:
            flag_keep_going = False
        else:
            ### Scale Array To Get Rid Of Total Outliers In The FPA: ###
            #Maybe i should only get the Q1,Q2 but not scale to 1 to avoid confusions
            if count == 0:
                current_frames_stretched, (q1, q2) = scale_array_stretch_hist(current_frames, flag_return_quantile=True)
            else:
                current_frames_stretched = scale_array_from_range(current_frames.clip(q1, q2),
                                                       min_max_values_to_clip=(q1, q2),
                                                       min_max_values_to_scale_to=(0, 1))  #notice that when the values to clip and the values to scale to are constant, the scaling is constant!!!
                1

            ### Get Median For Current Frames Batch: ###
            #TODO: perhapse add a spatial filter with 0 at the center before the temporal median to avoid the need for a moving drone?
            current_frames_median = np.median(current_frames, 0)
            # non_center_filter = np.ones((3,3))
            # non_center_filter[1,1] = 0
            # non_center_filter = non_center_filter/np.sum(non_center_filter)
            # current_frames_median_filtered = cv2.filter2D(current_frames_median, -1, non_center_filter)
            median_images_list.append(numpy_unsqueeze(current_frames_median,0))

        ### Advance Counter: ###
        count = count + 1

    ### Get Median Of Medians as an efficient "in-between" solution to finding a median over the entire giant movie: ###
    median_images_numpy = np.concatenate(median_images_list, 0)
    # Movie_BG = np.median(median_images_numpy, 0)  #TODO: perhapse change to lower quantile to avoid flashlight being there too much
    Movie_BG = np.quantile(median_images_numpy, 0.2, 0)

    ### Save Mov BG for later use (so i don't have to calculate it each time): ###
    # save_image_mat(folder_path=params.results_folder, filename='Movie_BG.mat', numpy_array=Movie_BG, variable_name='Movie_BG')
    np.save(os.path.join(params.results_folder, 'Mov_BG.npy'), Movie_BG, allow_pickle=True)
    np.save(os.path.join(params.results_folder, 'quantiles.npy'), (q1,q2), allow_pickle=True)
    return Movie_BG, (q1,q2)


def get_BGS_and_RealSpaceNoiseEstimation_torch(Movie, Movie_BG):
    # ### With Global Gain Estimation: ###
    # ratio = Mov / (Movie_BG + 1e-3)
    # gain = ratio[0].quantile(0.5)
    # Movie_BGS = Movie - Movie_BG * gain

    ### WithOut Global Gain Estimation: ###
    Movie_BGS = Movie - Movie_BG

    Movie_BGS_std = Movie_BGS.std(0, True).unsqueeze(0)
    Movie_BGS_std = Movie_BGS_std.clip(None, 10)
    real_space_noise_estimation = Movie_BGS_std
    return Movie_BGS, real_space_noise_estimation


def get_connected_components_logical_mask(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0):
    labels_out = kornia.contrib.connected_components(input_tensor, num_iterations=number_of_iterations)
    final_CC_logical_mask = torch.zeros_like(labels_out)  # f there's only one component -> leave it
    unique_labels = labels_out.unique()
    number_of_components_list = []
    flag_label_allowed_list = []
    ### If there's more then one component there....: ###
    for label_index in np.arange(len(unique_labels)):
        label_value = unique_labels[label_index]
        current_label_logical_mask = (labels_out == label_value)
        number_of_components = current_label_logical_mask.float().sum().item()  # TODO: this sum is over all possible images, and i want per image!!!
        number_of_components_list.append((label_value, number_of_components))
        if label_value == 0:
            final_CC_logical_mask[current_label_logical_mask] = label_zero_value
        else:
            flag_current_label_allowed = (number_of_components >= number_of_components_tuple[0] and number_of_components < number_of_components_tuple[1])
            flag_label_allowed_list.append(flag_current_label_allowed)
            if flag_current_label_allowed:
                final_CC_logical_mask[current_label_logical_mask] = 1

    return final_CC_logical_mask

def get_connected_components_logical_mask_batch(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0):
    T, C, H, W = input_tensor.shape
    final_CC_logical_mask = [get_connected_components_logical_mask(input_tensor[i:i + 1],
                                                                   number_of_iterations=number_of_iterations,
                                                                   number_of_components=number_of_components_tuple) for i in np.arange(T)]
    final_CC_logical_mask = torch.cat(final_CC_logical_mask, 0)
    return final_CC_logical_mask

def mask_according_to_connected_components_stats(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0):
    final_CC_logical_mask = get_connected_components_logical_mask(input_tensor,
                                                                  number_of_iterations=number_of_iterations,
                                                                  number_of_components=number_of_components_tuple)
    ### Multiple Final Outliers By CC Logical Mask: ###
    input_tensor = input_tensor * final_CC_logical_mask
    return input_tensor

def mask_according_to_connected_components_stats_batch(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0):
    logical_mask = get_connected_components_logical_mask_batch(input_tensor,
                                                               number_of_iterations=number_of_iterations,
                                                               number_of_components=number_of_components_tuple)
    input_tensor = input_tensor * logical_mask
    return input_tensor


def get_logical_mask_from_BB_list(input_tensor_reference, BB_list):
    # filepath = r'F:\Movies\12.4.2022 - natznatz experiments\15_night_500fps_20000frames_640x320\Results\Flashlight/flashlight_BB_list.npy'
    # BB_list = np.load(filepath, allow_pickle=True)
    logical_mask = torch.ones_like(input_tensor_reference)
    for BB_index in np.arange(len(BB_list)):
        current_BB = BB_list[BB_index][0]
        W1, H1 = current_BB[0]
        W2, H2 = current_BB[1]
        W3, H3 = current_BB[2]
        W4, H4 = current_BB[3]
        start_H = min(H1, H2, H3, H4)
        stop_H = max(H1, H2, H3, H4)
        start_W = min(W1, W2, W3, W4)
        stop_W = max(W1, W2, W3, W4)
        logical_mask[:, :, start_H:stop_H, start_W:stop_W] = 0
    return logical_mask


def Initialize_BG_Stats(Movie, params):
    ### Initialize outlier_sequence_counter_map: ###
    T, C, H, W = Movie.shape
    params.outlier_sequence_counter_map = torch.zeros((1, C, H, W)).to(Movie.device)

    ### Initialize First BG As The Median Of The First Batch: ###
    Movie_BG = Movie.median(0)[0].unsqueeze(0).cuda().float()

    # ### Estimate Noise By Robust STD: ###
    # # Movie_BG_std_torch = torch_robust_std_1(Movie, dim=0)
    # Movie_BG_std_torch = torch_robust_std_2(Movie, dim=0)

    # ### Estimate Noise Using Diff_Abs_Sum Histogram: ###
    # crude_histogram_bins_vec = np.linspace(0,256,20)
    # noise_estimate, SNR_estimate, noise_per_gray_level_list, histograms_bins_list, histograms_list = \
    #     estimate_noise_DifferenceHistogramPeak_step(Movie,
    #                                                 Movie.median(0)[0].unsqueeze(0),
    #                                                 histogram_spatial_binning_factor=1,
    #                                                 crude_histogram_bins_vec=crude_histogram_bins_vec,
    #                                                 downsample_kernel_size=1,
    #                                                 minimum_number_of_samples_for_valid_bin=800,
    #                                                 max_value_possible=256)

    ### Estimate Noise Using Clean Image and EMVA Report: ###
    Movie_BG_std_torch = estimate_noise_in_image_EMVA(Movie_BG,
                                                     QE=params.QE,
                                                     G_DL_to_e_EMVA=params.G_DL_to_e_EMVA,
                                                     G_e_to_DL_EMVA=params.G_e_to_DL_EMVA,
                                                     readout_noise_electrons=params.readout_noise_electrons,
                                                     full_well_electrons=params.full_well_electrons,
                                                     N_bits_EMVA=params.N_bits_EMVA,
                                                     N_bits_final=params.N_bits_final)

    ### Update dictionary: ###
    params.Movie_BG_previous = Movie_BG
    params.Movie_BG_std_torch_previous = Movie_BG_std_torch

    return Movie_BG, Movie_BG_std_torch, params


def Update_BG(Movie, Movie_BG, params):
    ### Get pixels with "stationary" outliers ###
    T,C,H,W = Movie.shape
    initial_outliers_sums = params.current_initial_outlier_sums  #Got this from outlier detection block previously
    logical_mask_current_outlier_stationary = initial_outliers_sums > T * 0.5  # threshold over which the outlier is considered ~stationary

    ### Update outliers counter map: ###
    params.outlier_sequence_counter_map += logical_mask_current_outlier_stationary.float()

    ### Where there WASN'T a stationary drone --> zero out that pixel's counter: ###
    params.outlier_sequence_counter_map[~logical_mask_current_outlier_stationary] = 0

    ### Whenever outlier sequence counter is too much, that means that basically this is a part of the view now and i just mixed up: ###
    logical_mask_outlier_sequence_counter_long_enough_to_be_BG = params.outlier_sequence_counter_map > params.number_of_batches_before_considered_BG
    params.outlier_sequence_counter_map[logical_mask_outlier_sequence_counter_long_enough_to_be_BG] = 0  # zero out counter
    logical_mask_current_outlier_stationary[logical_mask_outlier_sequence_counter_long_enough_to_be_BG] = 0  # places with permanent stationary outlier no longer considered outlier, but BG itself
    
    ### Get Masked Median Over Non-Outlier Pixels: ###
    Movie_BG_current = torch_masked_median(Movie, ~logical_mask_current_outlier_stationary)

    ### Where there IS a stionary outlier use previous BG estimation: ###
    Movie_BG_current[logical_mask_current_outlier_stationary] = params.Movie_BG_previous[logical_mask_current_outlier_stationary]

    ### Get STD/Noise-Estimate: ###
    # Movie_BG_STD_torch_current = torch_masked_std(Movie, ~logical_mask_current_outlier_stationary) #TODO: now i have the EMVA based STD....maybe i shouldn't take the STD from here right now...in any case it doesn't seem very consistent
    Movie_BG_STD_torch_current = estimate_noise_in_image_EMVA(Movie_BG_current,
                                                              QE=params.QE,
                                                              G_DL_to_e_EMVA=params.G_DL_to_e_EMVA,
                                                              G_e_to_DL_EMVA=params.G_e_to_DL_EMVA,
                                                              readout_noise_electrons=params.readout_noise_electrons,
                                                              full_well_electrons=params.full_well_electrons,
                                                              N_bits_EMVA=params.N_bits_EMVA,
                                                              N_bits_final=params.N_bits_final)

    ### Assign new BG estimation in params for next run: ###
    params.Movie_BG_previous = Movie_BG_current

    return params, Movie_BG_current, Movie_BG_STD_torch_current


def Update_BG_Long_Sequence(Movie, params):
    ### Get Parameters: ###
    T,C,H,W = Movie.shape
    mini_sequence_size = params.mini_sequence_size_for_BG_estimation
    number_of_mini_sequences = T//mini_sequence_size

    ### Initilize Lists: ###
    Movie_BG_list = []
    Movie_BG_STD_torch_list = []
    Movie_BG_list.append(params.Movie_BG_previous)
    Movie_BG_STD_torch_list.append(params.Movie_BG_std_torch_previous)

    ### Loop over mini sequences and update BG statistics: ###
    for mini_sequence_counter in np.arange(number_of_mini_sequences):
        ### Update BG: ###
        params, Movie_BG_current, Movie_BG_STD_torch_current = Update_BG(Movie, params)

        ### Update lists: ###
        Movie_BG_list.append(Movie_BG_current)
        Movie_BG_STD_torch_list.append(Movie_BG_STD_torch_current)


    ### Update Params: ###
    params.Movie_BG_list = Movie_BG_list
    params.Movie_BG_STD_torch_list = Movie_BG_STD_torch_list

    return params

# def Update_BG(Movie, Movie_BG, xyz_line, trajectory_tuple, params):
#     # ### Turn XYZ_line into logical mask: ###
#     # # (1). Using torch.index_put_()
#     # logical_mask = torch.ones_like(Movie).bool()
#     # for trajectory_index in np.arange(len(xyz_line)):
#     #     # ### Using xyz_line: ###
#     #     # flag_staionary_x_direction = (xyz_line[trajectory_index][-1, 1] - xyz_line[trajectory_index][0, 1]) < 5  #TODO: should probably be the size of the drone itself
#     #     # flag_staionary_y_direction = (xyz_line[trajectory_index][-1, 2] - xyz_line[trajectory_index][0, 2]) < 5
#     #     ### Using (Smooth) trajectory_tuple: ###
#     #     flag_staionary_x_direction = (trajectory_tuple[1][trajectory_index][-1] - trajectory_tuple[1][trajectory_index][0]).abs() < 5  # TODO: should probably be the size of the drone itself
#     #     flag_staionary_y_direction = (trajectory_tuple[2][trajectory_index][-1] - trajectory_tuple[2][trajectory_index][0]).abs() < 5
#     #
#     #     ### Is drone ~stationary: ###
#     #     flag_drone_stationary = flag_staionary_x_direction * flag_staionary_y_direction
#     #
#     #     ### Update Logical Mask Where Drone Is Stationary: ###
#     #     if flag_drone_stationary:
#     #         logical_mask = indices_to_logical_mask_torch(input_logical_mask=logical_mask, indices=xyz_line[trajectory_index], flag_one_based=False)
#     #
#     # ### Wherever There Was A Drone Trajectory Which Is Stationary Use The Previous BG Estimation Or Spatial Median: ###
#     # Movie_BG_new = params.Movie_BG_current
#     # Movie_BG_new[logical_mask] = params.Movie_BG_previous[logical_mask]
#
#
#     ### Use Previous initial outliers as-is and update BG per pixel value: ###
#     initial_outliers_sums = params.current_initial_outlier_sums
#     logical_mask_current_outlier_stationary = initial_outliers_sums > Movie.shape[0] * 0.5  #threshold over which the outlier is considered ~stationary
#     params.outlier_sequence_counter_map += logical_mask_current_outlier_stationary.float()
#
#     ### Where there wasn't a stationary drone --> zero out that pixel's counter: ###
#     params.outlier_sequence_counter_map[~logical_mask_current_outlier_stationary] = 0
#
#     ### Whenever outlier sequence counter is too much, that means that basically this is a part of the view now and i just mixed up: ###
#     outlier_sequence_counter_long_enough_to_be_BG = params.outlier_sequence_counter_map > params.number_of_batches_before_considered_BG
#     params.outlier_sequence_counter_map[outlier_sequence_counter_long_enough_to_be_BG] = 0  #zero out counter
#     logical_mask_current_outlier_stationary[outlier_sequence_counter_long_enough_to_be_BG] = 0  #places with permanent stationary outlier no longer considered outlier, but BG itself
#
#     ### Get Masked Median Over Non-Outlier Pixels: ###
#     # Movie_BG_current = torch.nanmedian(Movie, dim=0, keepdim=True)[0]
#     Movie_BG_current = torch_masked_median(Movie, ~logical_mask_current_outlier_stationary)
#     # Movie_BG_STD_torch_current = torch_masked_std(Movie, ~logical_mask_current_outlier_stationary)
#
#     ### Update BG estimation wherever valid: ###
#     Movie_BG_current[logical_mask_current_outlier_stationary] = params.Movie_BG_previous[logical_mask_current_outlier_stationary]
#
#     ### Assign new BG estimation in params for next run: ###
#     params.Movie_BG_previous = Movie_BG_current
#
#     return params


def Filter_Out_Outlier_Blobs(Movie_outliers_QuantileFiltered_LogicalMap):
    ####################################################################################################
    ### Filter Out Large Blobs Using (Connected-Components / Blob Detection): ###
    T, C, H, W = Movie_outliers_QuantileFiltered_LogicalMap.shape

    ### Connected-Componenets Get Rid Of Big Connected Chunks, Which Could Be Because Of Clouds or improper BG: ###
    # (1). Filter Connected-Components Over Each Frame:
    # CC_logical_mask = get_connected_components_logical_mask_batch(Movie_outliers_QuantileFiltered_LogicalMap,
    #                                                               number_of_iterations=10,
    #                                                               number_of_components=10)
    # Movie_outliers_QuantileFiltered_LogicalMap *= (1-CC_logical_mask)
    # #(2). Filter Connected-Components Over Outlier Map Temporal Mean:
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean = (Movie_outliers_QuantileFiltered_LogicalMap.mean(0, True) > 0.2).float()
    CC_logical_mask = get_connected_components_logical_mask(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean,
                                                            number_of_iterations=40,
                                                            number_of_components_tuple=(40, np.inf),
                                                            label_zero_value=0)
    Movie_outliers_QuantileFiltered_LogicalMap *= 1 - CC_logical_mask
    # imshow_torch(CC_logical_mask)
    # imshow_torch(1-CC_logical_mask)
    # imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean)

    ### Try Blob Detection: ###
    Movie_outliers_QuantileFiltered_LogicalMap *= torch_get_valid_center(Movie_outliers_QuantileFiltered_LogicalMap, 25)
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean *= torch_get_valid_center(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean, 25)
    # image_with_blobs, blobs_list = blob_detection_scipy_DifferenceOfGaussian_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean*255,
    #                                                                               min_sigma=10,
    #                                                                               max_sigma=150,
    #                                                                               sigma_ratio=1.5,
    #                                                                                overlap=0.9,
    #                                                                               threshold=0.1)
    # image_with_blobs, blobs_list = blob_detection_scipy_DeterminantOfHessian_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean * 255,
    #                                                                                min_sigma=5,
    #                                                                                max_sigma=25,
    #                                                                                number_of_sigmas=20,
    #                                                                                threshold=0.2)
    # image_with_blobs, blobs_list = blob_detection_scipy_LaplacianOfGaussian_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean * 255,
    #                                                                                min_sigma=10,
    #                                                                                max_sigma=35,
    #                                                                                number_of_sigmas=20,
    #                                                                                threshold=0.5)
    final_images, final_centers = blob_detection_scipy_LaplacianOfGaussian_torch_batch(Movie_outliers_QuantileFiltered_LogicalMap[0:5] * 255,
                                                                                       min_sigma=10,
                                                                                       max_sigma=50,
                                                                                       number_of_sigmas=30,
                                                                                       threshold=0.3, overlap=0.97)
    final_images, final_centers = blob_detection_scipy_DifferenceOfGaussian_torch_batch(fast_binning_2D_AvgPool2d(Movie_outliers_QuantileFiltered_LogicalMap[0:5], (2, 2), (0, 0)) * 255,
                                                                                        min_sigma=5,
                                                                                        max_sigma=35,
                                                                                        sigma_ratio=1.6,
                                                                                        threshold=0.1, overlap=0.97)
    final_images_torch = torch.tensor(final_images).permute([2, 0, 1]).unsqueeze(1)
    imshow_torch_video(final_images_torch, FPS=0.6)
    # plt.imshow(image_with_blobs)
    # plt.show()

    ###
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X2 = fast_binning_2D_AvgPool2d(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean,
                                                                                                                     (2, 2), (0, 0))
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X4 = fast_binning_2D_AvgPool2d(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean,
                                                                                                                     (4, 4), (0, 0))
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X8 = fast_binning_2D_AvgPool2d(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean,
                                                                                                                     (8, 8), (0, 0))
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X2 = torch.nn.functional.interpolate(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X2,
                                                                                                          scale_factor=2, mode='bilinear')
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X4 = torch.nn.functional.interpolate(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X4,
                                                                                                          scale_factor=4, mode='bilinear')
    Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X8 = torch.nn.functional.interpolate(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X8,
                                                                                                          scale_factor=8, mode='bilinear')
    imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean)
    imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X2)
    imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X4)
    imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean_X8)

    ### Save Connected Components Logical Mask: ###

    # # imshow_torch_video(Movie_outliers_QuantileFiltered_LogicalMap, FPS=50, frame_stride=2)
    # # imshow_torch_video(Movie, FPS=50, frame_stride=2)
    # # imshow_torch_video(CC_logical_mask, FPS=50, frame_stride=2)
    # imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap.mean(0))
    # imshow_torch(CC_logical_mask)
    # imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap_temporal_mean)
    # imshow_torch(Movie_outliers_QuantileFiltered_LogicalMap)

    return Movie_outliers_QuantileFiltered_LogicalMap


def Get_Outliers_Above_BG_2_torch(Movie, Movie_BGS, Movie_BG, params):

    ####################################################################################################
    ### Use BGS-TO-STD RATIO threshold betwen Difference and STD (noise): ###
    if params.flag_perform_BGS_running_mean:
        temporal_mean_kernel = torch.ones(params.BGS_runnning_mean_size).to(Movie.device) / params.BGS_runnning_mean_size
        Movie_BGS_running_mean = convn_torch(Movie_BGS, temporal_mean_kernel, 0)
        ratio_outlier_threshold = params.difference_to_STD_ratio_threshold / np.sqrt(params.BGS_runnning_mean_size)  #since we're averaging frames i can lower the threshold
        Movie_BGS_to_STD_ratio = Movie_BGS_running_mean / (params.Movie_BG_std_torch_previous + 1e-3)
        Movie_outliers = (Movie_BGS_to_STD_ratio > ratio_outlier_threshold).type(torch.float)
    else:
        ratio_outlier_threshold = params.difference_to_STD_ratio_threshold
        Movie_BGS_to_STD_ratio = Movie_BGS / (params.Movie_BG_std_torch_previous + 1e-3)
        Movie_outliers = (Movie_BGS_to_STD_ratio > ratio_outlier_threshold).type(torch.float)

    ### Record current movie outliers sums to be used for next batch: ###
    params.current_initial_outlier_sums = Movie_outliers.sum(0, True)

    ### Save interim outputs of BGS Itself if wanted: ###
    if params.flag_save_interim_graphs_and_movies:
        if params.flag_perform_BGS_running_mean:
            video_torch_array_to_video(scale_array_to_range(Movie_BGS_running_mean), video_name=os.path.join(params.results_folder_seq, 'BGS_running_mean.avi'), FPS=50)
        video_torch_array_to_video(scale_array_to_range(Movie_BGS), video_name=os.path.join(params.results_folder_seq, 'BGS.avi'), FPS=50)
        video_torch_array_to_video(Movie_outliers, video_name=os.path.join(params.results_folder_seq, 'initial_outliers.avi'), FPS=50)

    ### Save BGS/STD Ratio if wanted: ###
    if params.flag_save_interim_graphs_and_movies:
        if params.flag_perform_BGS_running_mean:
            video_torch_array_to_video(Movie_BGS_to_STD_ratio, video_name=os.path.join(params.results_folder_seq, 'STD_BGS_Over_std.avi'), FPS=50)
        video_torch_array_to_video(Movie_outliers, video_name=os.path.join(params.results_folder_seq, 'STD_initial_outliers.avi'), FPS=50)
    ####################################################################################################


    # ####################################################################################################
    # ### Use simple GLOBAL THRESHOLD over Running-Mean BGS (should allow you to use a lower threshold): ###
    # if params.flag_perform_BGS_running_mean:
    #     temporal_mean_kernel = torch.ones(params.BGS_runnning_mean_size).to(Movie.device)/params.BGS_runnning_mean_size
    #     Movie_BGS_running_mean = convn_torch(Movie_BGS, temporal_mean_kernel, 0)
    #     Movie_outliers = (Movie_BGS_running_mean > params.global_outlier_threshold).type(torch.float)
    # else:
    #     Movie_outliers = (Movie_BGS > params.global_outlier_threshold).type(torch.float)
    #
    # ### Record current movie outliers sums to be used for next batch: ###
    # params.current_initial_outlier_sums = Movie_outliers.sum(0, True)
    #
    # ### Save interim outputs of BGS if wanted: ###
    # if params.flag_save_interim_graphs_and_movies:
    #     if params.flag_perform_BGS_running_mean:
    #         video_torch_array_to_video(scale_array_to_range(Movie_BGS_running_mean), video_name=os.path.join(params.results_folder_seq, 'BGS_running_mean.avi'), FPS=50)
    #     video_torch_array_to_video(scale_array_to_range(Movie_BGS), video_name=os.path.join(params.results_folder_seq, 'BGS.avi'), FPS=50)
    #     video_torch_array_to_video(Movie_outliers, video_name=os.path.join(params.results_folder_seq, 'initial_outliers.avi'), FPS=50)
    #
    # ### Save BGS/STD Ratio: ###
    # if params.flag_perform_BGS_running_mean:
    #     temporal_mean_kernel = torch.ones(params.BGS_runnning_mean_size).to(Movie.device) / params.BGS_runnning_mean_size
    #     Movie_BGS_running_mean = convn_torch(Movie_BGS, temporal_mean_kernel, 0)
    #     ratio_outlier_threshold = params.difference_to_STD_threshold / torch.sqrt(params.BGS_runnning_mean_size)
    #     Movie_BGS_to_STD_ratio = Movie_BGS_running_mean / (params.Movie_BG_std_torch_previous + 1e-3)
    #     Movie_outliers = (Movie_BGS_to_STD_ratio > ratio_outlier_threshold).type(torch.float)
    # else:
    #     ratio_outlier_threshold = params.difference_to_STD_threshold
    #     Movie_BGS_to_STD_ratio = Movie_BGS / (params.Movie_BG_std_torch_previous + 1e-3)
    #     Movie_outliers = (Movie_BGS_to_STD_ratio > ratio_outlier_threshold).type(torch.float)
    # if params.flag_save_interim_graphs_and_movies:
    #     if params.flag_perform_BGS_running_mean:
    #         video_torch_array_to_video(Movie_BGS_to_STD_ratio, video_name=os.path.join(params.results_folder_seq, 'STD_BGS_Over_std.avi'), FPS=50)
    #     else:
    #         video_torch_array_to_video(Movie_BGS_to_STD_ratio, video_name=os.path.join(params.results_folder_seq, 'STD_BGS_Over_std.avi'), FPS=50)
    #     video_torch_array_to_video(Movie_outliers, video_name=os.path.join(params.results_folder_seq, 'STD_initial_outliers.avi'), FPS=50)
    # ####################################################################################################


    ####################################################################################################
    ### Decide whether to perform outlier running mean: ###
    if params.flag_perform_outlier_running_mean:
        outlier_temporal_mean_kernel = torch.ones(params.outlier_detection_running_mean_size).to(Movie.device)
        Movie_outliers_QuantileFiltered_LogicalMap = convn_torch(Movie_outliers, outlier_temporal_mean_kernel, 0) >= params.outlier_after_temporal_mean_threshold
    else:
        Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers
    ####################################################################################################

    ####################################################################################################
    ### Perform Canny Edge Detection On Median Image: ###
    if params.flag_use_canny_in_outlier_detection:
        thresholded_binary_BG = kornia.filters.canny(BW2RGB(Movie_BG))[1] #Get edge map of BG itself
        # thresholded_binary_BG = kornia.filters.canny(BW2RGB(Movie_BGS.abs().sum(0,True))/500)[1]  #Get edge map of BGS!

        ### Perform Dilation On Canny Edge: ###
        thresholded_binary_BG_dilated = kornia.morphology.dilation(thresholded_binary_BG, torch.ones(3, 3).cuda()).float()

        ### Substract BG Edge From Outliers: ###
        Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap * (1-thresholded_binary_BG)

        ### Save interim outputs if wanted: ###
        if params.flag_save_interim_graphs_and_movies:
            video_torch_array_to_video(Movie_outliers_QuantileFiltered_LogicalMap, video_name=os.path.join(params.results_folder_seq, 'outliers_after_canny.avi'), FPS=50)
    ####################################################################################################

    ####################################################################################################
    ## Filter Out Points From Wherever It The User Says So: ###
    Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap * params.user_defined_outlier_logical_mask
    ####################################################################################################

    ####################################################################################################
    ### Get Rid Of External Frame From Palantir Stabilization: ###
    Movie_outliers_QuantileFiltered_LogicalMap *= torch_get_valid_center(Movie_outliers_QuantileFiltered_LogicalMap,
                                                                                               non_valid_border_size=15) #TODO: maybe get this from stabilization before this block
    ####################################################################################################


    ####################################################################################################
    ### Filter Out Large Contiguous Blobs: ###
    # Movie_outliers_QuantileFiltered_LogicalMap = Filter_Out_Outlier_Blobs(Movie_outliers_QuantileFiltered_LogicalMap)
    blob_detection_layer = Blob_Detection_Layer(number_of_components_tuple=(0,np.inf))
    list_of_BB, BB_tensor, Movie_outliers_QuantileFiltered_LogicalMap_Reduced, output_tensor_with_BB,\
    sublob_center_per_frame_per_label_list, sublob_covariance_per_frame_per_label_list,\
    number_of_BBs_per_frame, number_of_pixels_per_frame_per_label_list = \
        blob_detection_layer.forward(Movie_outliers_QuantileFiltered_LogicalMap, flag_plot_BB_on_output_tensor=False)
    # imshow_torch_video(Movie_outliers_QuantileFiltered_LogicalMap_Reduced, FPS=10, frame_stride=1)
    # imshow_torch_video(Movie_outliers_QuantileFiltered_LogicalMap, FPS=10, frame_stride=1)
    # imshow_torch_video(torch.cat([Movie_outliers_QuantileFiltered_LogicalMap, Movie_outliers_QuantileFiltered_LogicalMap_Reduced], -1), FPS=10, frame_stride=1)

    # Movie_outliers_QuantileFiltered_LogicalMap_Reduced = Movie_outliers_QuantileFiltered_LogicalMap
    ####################################################################################################

    # ####################################################################################################
    # #TODO: should probably perform canny edge detection on remaining outliers after reduction, should probably get rid of remaining edge cases
    # ### Perform Canny Edge Detection On Median Image: ###
    # if params.flag_use_canny_in_outlier_detection:
    #     thresholded_binary_BG = kornia.filters.canny(BW2RGB(Movie_outliers_QuantileFiltered_LogicalMap_Reduced))[1]  # Get edge map of BG itself
    #     # thresholded_binary_BG = kornia.filters.canny(BW2RGB(Movie_BGS.abs().sum(0,True))/500)[1]  #Get edge map of BGS!
    #
    #     ### Perform Dilation On Canny Edge: ###
    #     thresholded_binary_BG_dilated = kornia.morphology.dilation(thresholded_binary_BG, torch.ones(3, 3).cuda()).float()
    #
    #     ### Substract BG Edge From Outliers: ###
    #     Movie_outliers_QuantileFiltered_LogicalMap_Reduced = Movie_outliers_QuantileFiltered_LogicalMap_Reduced * (1 - thresholded_binary_BG)
    #
    #     ### Save interim outputs if wanted: ###
    #     if params.flag_save_interim_graphs_and_movies:
    #         video_torch_array_to_video(Movie_outliers_QuantileFiltered_LogicalMap, video_name=os.path.join(params.results_folder_seq, 'outliers_after_canny.avi'), FPS=50)
    # ####################################################################################################

    ####################################################################################################
    ### Save interim outputs if wanted: ###
    if params.flag_save_interim_graphs_and_movies:
        video_torch_array_to_video(Movie_outliers_QuantileFiltered_LogicalMap, video_name=os.path.join(params.results_folder_seq, 'final_outliers_RAW.avi'), FPS=50)
        video_torch_array_to_video(Movie_outliers_QuantileFiltered_LogicalMap_Reduced, video_name=os.path.join(params.results_folder_seq, 'final_outliers_Reduced.avi'), FPS=50)

    ### Reshape Outlier Indices To Proper Numpy Array: ###
    #TODO: this, again, is probably slow and can be avoided!. also, if i decide to use THW indices instead of TXY this whole thing can be speed-up i think.
    # also, i should probably reconsider the way i'm using the TXY indices in general and tuples/lists in general in QuickShot
    outliers_indices_tuple = torch.where(Movie_outliers_QuantileFiltered_LogicalMap_Reduced.squeeze(1))  #TODO: the tuple received is a tuple of indices
    outliers_indices_tuple = permute_tuple_indices(outliers_indices_tuple, [0,2,1])  #TODO: remember why is permute the tuple indices!!!
    movie_current_outliers_TXY_indices = torch.vstack(outliers_indices_tuple).T.type(torch.float)
    ####################################################################################################

    return movie_current_outliers_TXY_indices, params




def Get_Outliers_Above_BG_Multiple_BG_Estimations_torch_1(Movie, Movie_BGS, Movie_BG, params):
    ### Simple Version: take the median of each sub-sequence as the BG for the sub-sequence and perform outlier detection: ###
    number_of_sub_sequences = 5
    T,C,H,W = Movie.shape
    number_of_frames_per_sub_sequence = T//number_of_sub_sequences
    for sub_sequence_index in np.arange(number_of_sub_sequences):
        ### Get Current Sub-Sequence & Estimate BG: ###
        start_frame_index = sub_sequence_index * number_of_frames_per_sub_sequence
        stop_frame_index = start_frame_index + number_of_frames_per_sub_sequence
        Movie_BG_current_torch = Movie[start_frame_index:stop_frame_index].median(0,True)[0]
        Movie_BGS_current = Movie - Movie_BG_current_torch

        ### Get Outliers: ###
        movie_current_outliers_TXY_indices_current, params = Get_Outliers_Above_BG_2_torch(Movie, Movie_BGS, Movie_BG_current_torch, params)

    return movie_current_outliers_TXY_indices, params


def Get_Outliers_Above_BG_Multiple_BG_Estimations_torch_2(Movie, Movie_BGS, Movie_BG, params):
    ### Simple Version: take the median of each sub-sequence as the BG for the sub-sequence and perform outlier detection: ###
    number_of_sub_sequences = 5
    T,C,H,W = Movie.shape
    number_of_frames_per_sub_sequence = T//number_of_sub_sequences
    ### Initialize Outlier/Event Indices: ###
    # #(1). as tensor:
    # movie_current_outliers_TXY_indices = torch.empty([0,3]).to(Movie.device)
    #(2). as list:
    movie_current_outliers_TXY_indices = []
    for sub_sequence_index in np.arange(number_of_sub_sequences):
        ### Get Current Sub-Sequence & Estimate BG: ###
        start_frame_index = sub_sequence_index * number_of_frames_per_sub_sequence
        stop_frame_index = start_frame_index + number_of_frames_per_sub_sequence
        Movie_current = Movie[start_frame_index:stop_frame_index]   #TODO: maybe even add strides here or within the function itself

        ### Get Current BGS: ###   #TODO: transfer BGS to function itself maybe?!!?!
        Movie_BG_current = params.Movie_BG_previous
        Movie_BGS = Movie_current - Movie_BG_current

        ### Get Outliers: ###
        movie_current_outliers_TXY_indices_current, params = Get_Outliers_Above_BG_2_torch(Movie_current, Movie_BGS, Movie_BG_current, params)
        movie_current_outliers_TXY_indices_current[:, 0] += number_of_frames_per_sub_sequence * sub_sequence_index

        ### Append To Final Outliers/Events Tensor/List: ###
        # #(1). Tensor:
        # movie_current_outliers_TXY_indices = torch.cat([movie_current_outliers_TXY_indices, movie_current_outliers_TXY_indices_current], 0)
        #(2). List:
        movie_current_outliers_TXY_indices.append(movie_current_outliers_TXY_indices_current)

        ### Update BG: ###
        params, Movie_BG_current, Movie_BG_STD_torch_current = Update_BG(Movie, Movie_BG, params)

    return movie_current_outliers_TXY_indices, params


def create_random_labels_map(classes: int):
    labels_map: dict[int, Tuple[int, int, int]] = {}
    for i in classes:
        labels_map[i] = torch.randint(0, 255, (3,))
    labels_map[0] = torch.zeros(3)
    return labels_map


def labels_to_image(img_labels, labels_map) -> torch.Tensor:
    """Function that given an image with labels ids and their pixels intrensity mapping, creates a RGB
    representation for visualisation purposes."""
    assert len(img_labels.shape) == 2, img_labels.shape
    H, W = img_labels.shape
    out = torch.empty(3, H, W, dtype=torch.uint8)
    for label_id, label_val in labels_map.items():
        mask = (img_labels == label_id)
        for i in range(3):
            out[i].masked_fill_(mask, label_val[i])
    return out


def show_components(img, labels):
    color_ids = torch.unique(labels)
    labels_map = create_random_labels_map(color_ids)
    labels_img = labels_to_image(labels, labels_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))

    # Showing Original Image
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
    ax1.set_title("Orginal Image")

    # Showing Image after Component Labeling
    ax2.imshow(labels_img.permute(1, 2, 0).squeeze().numpy())
    ax2.axis('off')
    ax2.set_title("Component Labeling")




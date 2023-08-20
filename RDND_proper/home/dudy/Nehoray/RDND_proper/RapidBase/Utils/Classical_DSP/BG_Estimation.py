

import kornia
import torch
import numpy as np


def mask_according_to_connected_components_stats(input_tensor, number_of_iterations=10, maximum_number_of_components=5):
    labels_out = kornia.contrib.connected_components(input_tensor, num_iterations=number_of_iterations)
    final_CC_logical_mask = torch.zeros_like(labels_out)  #f there's only one component -> leave it
    unique_labels = labels_out.unique()
    number_of_components_list = []
    flag_label_allowed_list = []
    ### If there's more then one component there....: ###
    for label_index in np.arange(len(unique_labels)):
        label_value = unique_labels[label_index]
        current_label_logical_mask = (labels_out == label_value)
        number_of_components = current_label_logical_mask.float().sum().item()  #TODO: this sum is over all possible images, and i want per image!!!
        number_of_components_list.append((label_value,number_of_components))
        flag_current_label_allowed = (number_of_components <= maximum_number_of_components)
        flag_label_allowed_list.append(flag_current_label_allowed)
        if flag_current_label_allowed:
            final_CC_logical_mask[current_label_logical_mask] = 1

    ### Multiple Final Outliers By CC Logical Mask: ###
    input_tensor = input_tensor * final_CC_logical_mask
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


def Update_BG(Movie, Movie_BG, xyz_line, trajectory_tuple, params):
    # ### Turn XYZ_line into logical mask: ###
    # # (1). Using torch.index_put_()
    # logical_mask = torch.ones_like(Movie).bool()
    # for trajectory_index in np.arange(len(xyz_line)):
    #     # ### Using xyz_line: ###
    #     # flag_staionary_x_direction = (xyz_line[trajectory_index][-1, 1] - xyz_line[trajectory_index][0, 1]) < 5  #TODO: should probably be the size of the drone itself
    #     # flag_staionary_y_direction = (xyz_line[trajectory_index][-1, 2] - xyz_line[trajectory_index][0, 2]) < 5
    #     ### Using (Smooth) trajectory_tuple: ###
    #     flag_staionary_x_direction = (trajectory_tuple[1][trajectory_index][-1] - trajectory_tuple[1][trajectory_index][0]).abs() < 5  # TODO: should probably be the size of the drone itself
    #     flag_staionary_y_direction = (trajectory_tuple[2][trajectory_index][-1] - trajectory_tuple[2][trajectory_index][0]).abs() < 5
    #
    #     ### Is drone ~stationary: ###
    #     flag_drone_stationary = flag_staionary_x_direction * flag_staionary_y_direction
    #
    #     ### Update Logical Mask Where Drone Is Stationary: ###
    #     if flag_drone_stationary:
    #         logical_mask = indices_to_logical_mask_torch(input_logical_mask=logical_mask, indices=xyz_line[trajectory_index], flag_one_based=False)
    #
    # ### Wherever There Was A Drone Trajectory Which Is Stationary Use The Previous BG Estimation Or Spatial Median: ###
    # Movie_BG_new = params.Movie_BG_current
    # Movie_BG_new[logical_mask] = params.Movie_BG_previous[logical_mask]


    ### Use Previous initial outliers as-is and update BG per pixel value: ###
    initial_outliers_sums = params.current_initial_outlier_sums
    logical_mask_current_outlier_stationary = initial_outliers_sums > Movie.shape[0] * 0.5  #threshold over which the outlier is considered ~stationary
    params.outlier_sequence_counter_map += logical_mask_current_outlier_stationary.float()

    ### Where there wasn't a stationary drone --> zero out that pixel's counter: ###
    params.outlier_sequence_counter_map[~logical_mask_current_outlier_stationary] = 0

    ### Whenever outlier sequence counter is too much, that means that basically this is a part of the view now and i just mixed up: ###
    outlier_sequence_counter_long_enough_to_be_BG = params.outlier_sequence_counter_map > params.number_of_batches_before_considered_BG
    params.outlier_sequence_counter_map[outlier_sequence_counter_long_enough_to_be_BG] = 0  #zero out counter
    logical_mask_current_outlier_stationary[outlier_sequence_counter_long_enough_to_be_BG] = 0  #places with permanent stationary outlier no longer considered outlier, but BG itself

    ### Get Masked Median Over Non-Outlier Pixels: ###
    # Movie_BG_current = torch.nanmedian(Movie, dim=0, keepdim=True)[0]
    Movie_BG_current = torch_masked_median(Movie, ~logical_mask_current_outlier_stationary)
    # Movie_BG_STD_torch_current = torch_masked_std(Movie, ~logical_mask_current_outlier_stationary)

    ### Update BG estimation wherever valid: ###
    Movie_BG_current[logical_mask_current_outlier_stationary] = params.Movie_BG_previous[logical_mask_current_outlier_stationary]

    ### Assign new BG estimation in params for next run: ###
    params.Movie_BG_previous = Movie_BG_current

    return params


def Get_Outliers_Above_BG_2_torch(Movie, Movie_BGS, Movie_BG, Movie_BGS_std, params):
    ####################################################################################################
    # ### Use ratio over BGS_std to find outliers - however!!! BGS_std includes the drone itself!!!!!...so that's stupid: ###
    # Movie_BGS_over_std = Movie_BGS / (Movie_BGS_std + 1e-6)
    # Movie_outliers = torch.abs(Movie_BGS_over_std) > params['BGSSigma']
    # Movie_outliers = Movie_outliers.type(torch.float)
    # Movie_outliers_QuantileFiltered_LogicalMap = convn_torch(Movie_outliers, torch.ones((5)).to(Movie.device), 0) >= 3
    # # imshow_torch_video(torch.Tensor(Movie_outliers).unsqueeze(1), None, 50)
    # # imshow_torch_video(torch.Tensor(Movie_outliers_QuantileFiltered_LogicalMap).unsqueeze(1), None, 50)
    ####################################################################################################

    ####################################################################################################
    # ### Use simple global threshold over BGS: ###
    # outlier_threshold = 0.2
    # Movie_outliers = Movie_BGS > outlier_threshold
    # Movie_outliers_QuantileFiltered_LogicalMap = signal.convolve(Movie_outliers, np.ones((5, 1, 1)), mode='same') >= 3
    # imshow_torch_video(torch.Tensor(Movie_outliers).unsqueeze(1), None, 50)
    # imshow_torch_video(torch.Tensor(Movie_outliers_QuantileFiltered_LogicalMap).unsqueeze(1), None, 50)
    ####################################################################################################

    ####################################################################################################
    ### Use simple global threshold over Running-Mean BGS (should allow you to use a lower threshold): ###
    running_mean_size = 5
    outlier_threshold = 0.03 #TODO: this is very ed-hok and obviously should change with distance etc. this should be relative to movie std per pixel
    #TODO: if i'm using a running mean on the images in the time domain...
    # why even have the entire prediction on the full 500 frames t_vec???....
    # why not simply use the reduced number of time steps and that's it?!!?...

    ### Decide whether to perform BGS running mean: ###
    if params.flag_perform_BGS_running_mean:
        Movie_BGS_running_mean = convn_torch(Movie_BGS, torch.ones(running_mean_size).to(Movie.device), 0) / running_mean_size
        Movie_outliers = (Movie_BGS_running_mean > outlier_threshold).type(torch.float)
    else:
        Movie_outliers = (Movie_BGS > outlier_threshold).type(torch.float)


    # ### Try and use BGS with threshold over std (instead of a global threshold): ###
    # #TODO: finalize this!!!!!
    # if params.flag_perform_BGS_running_mean:
    #     Movie_BGS_over_STD = Movie_BGS_running_mean / (params.Movie_BG_std_torch_previous + 1e-3)
    #     Movie_outliers_STD = (Movie_BGS_running_mean > params.Movie_BG_std_torch_previous * 3/np.sqrt(running_mean_size)).type(torch.float)
    # if params.flag_save_interim_graphs_and_movies:
    #     video_torch_array_to_video(Movie_BGS_over_STD, video_name=os.path.join(params.results_folder_seq, 'STD_BGS_Over_std.avi'), FPS=50)
    #     video_torch_array_to_video(Movie_outliers_STD, video_name=os.path.join(params.results_folder_seq, 'STD_initial_outliers.avi'), FPS=50)

    ### Record current movie outliers sums to be used for next batch: ###
    params.current_initial_outlier_sums = Movie_outliers.sum(0, True)

    ### Save interim outputs if wanted: ###
    if params.flag_save_interim_graphs_and_movies:
        if params.flag_perform_BGS_running_mean:
            video_torch_array_to_video(scale_array_to_range(Movie_BGS_running_mean), video_name=os.path.join(params.results_folder_seq, 'BGS_running_mean.avi'), FPS=50)
        video_torch_array_to_video(Movie_outliers, video_name=os.path.join(params.results_folder_seq, 'initial_outliers.avi'), FPS=50)
    ####################################################################################################

    # ####################################################################################################
    # ### Use ratio over MinPool2D/MedianPool2D/AvgPool2D OF THE ABS to still be able to use videos with
    # # the drone present in the BG/BGS/BGS_std estimation: ### (this assumes the drone is small enough)
    # min_pool_size = 7
    # max_pooling_layer = nn.MaxPool2d(min_pool_size, 1)
    # Movie_BGS_min_pool = fast_binning_2D_MinPool2d(Movie_BGS.abs(), min_pool_size, min_pool_size-1)  #TODO: notice the abs()!!!!
    # # Movie_BGS_min_pool = fast_binning_2D_overap_flexible_MedianPool2d(torch.Tensor(Movie_BGS).unsqueeze(1).abs(), min_pool_size, min_pool_size-1).squeeze(1).cpu().numpy()  #TODO: notice the abs()!!!!
    # current_ratio = Movie_BGS / (Movie_BGS_min_pool + 1e-3)
    # Movie_outliers = current_ratio > 30
    # Movie_outliers_QuantileFiltered_LogicalMap = convn_torch(Movie_outliers.float(), torch.ones(5).to(Movie.device), dim=0) >= 3
    # # imshow(current_ratio[0]); plt.show()
    # # imshow_torch_video(current_ratio, None, 50)
    # # imshow_torch_video(Movie_outliers, None, 50)
    # # imshow_torch_video(Movie_outliers_QuantileFiltered_LogicalMap, None, 50)
    # ###################################################################################################

    ####################################################################################################
    # ### Use threshold above MindPool2D/AvgPool2D over space of the running mean: ###
    # convn_layer = convn_layer_torch()
    # running_mean_size = 5
    # outlier_ratio_threshold = 200
    # Movie_BGS_running_mean = convn_layer.forward(torch.Tensor(Movie_BGS).unsqueeze(1), torch.ones(running_mean_size), 0) / running_mean_size
    # min_pool_size = 7
    # max_pooling_layer = nn.MaxPool2d(min_pool_size, 1)
    # Movie_BGS_min_pool = fast_binning_2D_overap_flexible_MinPool2d(torch.Tensor(Movie_BGS_running_mean).abs(), min_pool_size, min_pool_size - 1).cpu().numpy()  # TODO: notice the abs()!!!!
    # # Movie_BGS_min_pool = fast_binning_2D_overap_flexible_MedianPool2d(torch.Tensor(Movie_BGS).unsqueeze(1).abs(), min_pool_size, min_pool_size-1).squeeze(1).cpu().numpy()  #TODO: notice the abs()!!!!
    # current_ratio = Movie_BGS / (Movie_BGS_min_pool + 1e-3)
    # Movie_outliers = current_ratio > outlier_ratio_threshold
    # Movie_outliers_QuantileFiltered_LogicalMap = signal.convolve(Movie_outliers, np.ones((5, 1, 1)), mode='same') >= 3
    # imshow(current_ratio[0]);
    # plt.show()
    # imshow_torch_video(torch.Tensor(current_ratio).unsqueeze(1), None, 50)
    # imshow_torch_video(torch.Tensor(Movie_outliers).unsqueeze(1), None, 50)
    # imshow_torch_video(torch.Tensor(Movie_outliers_QuantileFiltered_LogicalMap).unsqueeze(1), None, 50)
    ####################################################################################################

    ####################################################################################################
    # ### Use threshold above MindPool3D/AvgPool3D over space-time: ###
    # outlier_ratio_threshold = 200
    # min_pool_size = 7
    # max_pooling_layer = nn.MaxPool3d(min_pool_size, 1)
    # Movie_BGS_min_pool = -1 * max_pooling_layer(-1 * torch.Tensor(Movie_BGS).unsqueeze(0)).squeeze(0)
    # # Movie_BGS_min_pool = fast_binning_2D_overap_flexible_MinPool2d(torch.Tensor(Movie_BGS_running_mean).abs(), min_pool_size, min_pool_size - 1).cpu().numpy()  # TODO: notice the abs()!!!!
    # # Movie_BGS_min_pool = fast_binning_2D_overap_flexible_MedianPool2d(torch.Tensor(Movie_BGS).unsqueeze(1).abs(), min_pool_size, min_pool_size-1).squeeze(1).cpu().numpy()  #TODO: notice the abs()!!!!
    # current_ratio = Movie_BGS / (Movie_BGS_min_pool + 1e-3)
    # Movie_outliers = current_ratio > outlier_ratio_threshold
    # Movie_outliers_QuantileFiltered_LogicalMap = signal.convolve(Movie_outliers, np.ones((5, 1, 1)), mode='same') >= 3
    # imshow(current_ratio[0]); plt.show()
    # imshow_torch_video(torch.Tensor(current_ratio).unsqueeze(1), None, 50)
    # imshow_torch_video(torch.Tensor(Movie_outliers).unsqueeze(1), None, 50)
    # imshow_torch_video(torch.Tensor(Movie_outliers_QuantileFiltered_LogicalMap).unsqueeze(1), None, 50)
    ####################################################################################################

    ####################################################################################################
    # ### Use global Quantile statistics to get only the strongest outliers above BGS: ###
    # min_quantile_BGS = 0.99
    # Movie_BGS_clip_stretch, (q1,q2) = scale_array_stretch_hist(np.abs(Movie_BGS), (min_quantile_BGS,1), (0,1), True)
    # Movie_outliers = Movie_BGS > q1
    # Movie_outliers_QuantileFiltered_LogicalMap = signal.convolve(Movie_outliers, np.ones((5, 1, 1)), mode='same') >= 3
    # imshow_torch_video(torch.Tensor(Movie_outliers).unsqueeze(1), None, 50)
    # imshow_torch_video(torch.Tensor(Movie_outliers_QuantileFiltered_LogicalMap).unsqueeze(1), None, 50)
    # imshow(Movie_BGS[0]); plt.show()
    # imshow(Movie_BGS_clip_stretch[0]); plt.show()
    ####################################################################################################


    ####################################################################################################
    ### Decide whether to perform outlier running mean: ###
    if params.flag_perform_outlier_running_mean:
        Movie_outliers_QuantileFiltered_LogicalMap = convn_torch(Movie_outliers, torch.ones(5).to(Movie.device), 0) >= 3
    else:
        Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers
    ####################################################################################################

    ####################################################################################################
    ### Perform Canny Edge Detection On Median Image: ###
    if params.flag_use_canny_in_outlier_detection:
        thresholded_binary_BG = kornia.filters.canny(BW2RGB(Movie_BG))[1]
        # thresholded_binary_BG = kornia.filters.canny(BW2RGB(Movie_BGS.abs().sum(0,True))/500)[1]

        ### Perform Dilation On Canny Edge: ###
        thresholded_binary_BG_dilated = kornia.morphology.dilation(thresholded_binary_BG, torch.ones(3, 3).cuda()).float()

        ### Substract BG Edge From Outliers: ###
        Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap * (1-thresholded_binary_BG)

        # ### Save interim outputs if wanted: ###
        # if params.flag_save_interim_graphs_and_movies:
        #     video_torch_array_to_video(Movie_outliers_QuantileFiltered_LogicalMap, video_name=os.path.join(params.results_folder_seq, 'outliers_after_canny.avi'), FPS=50)
    ####################################################################################################

    ####################################################################################################
    ### Filter Out Points From The Bottom Part Of The Image: ###
    #TODO: again, this can be initialized before to save time!!!!
    T,C,H,W = Movie_outliers_QuantileFiltered_LogicalMap.shape
    # logical_mask = torch.zeros((1,1,H,W)).to(Movie.device)
    # fractions_W = (0,1)
    # fractions_H = (0,0.7)
    # start_H = np.int(fractions_H[0] * H)
    # stop_H = np.int(fractions_H[1] * H)
    # start_W = np.int(fractions_W[0] * W)
    # stop_W = np.int(fractions_W[1] * W)
    # logical_mask[:, :, start_H:stop_H, start_W:stop_W] = 1
    # Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap * logical_mask
    # # imshow_torch_video(Movie_outliers_QuantileFiltered_LogicalMap, FPS=50, frame_stride=5)

    ## Filter Out Points From Wherever It The User Says So: ###
    Movie_outliers_QuantileFiltered_LogicalMap = Movie_outliers_QuantileFiltered_LogicalMap * params.user_defined_outlier_logical_mask

    # ### Connected-Componenets Get Rid Of Big Connected Chunks, Which Could Be Because Of Clouds or improper BG: ###
    # Movie_outliers_QuantileFiltered_LogicalMap = [mask_according_to_connected_components_stats(Movie_outliers_QuantileFiltered_LogicalMap[i:i+1], 20, 10) for i in np.arange(T)]
    # Movie_outliers_QuantileFiltered_LogicalMap = torch.cat(Movie_outliers_QuantileFiltered_LogicalMap, 0)
    # # imshow_torch_video(Movie_outliers_QuantileFiltered_LogicalMap, FPS=50, frame_stride=5)

    ### Save interim outputs if wanted: ###
    if params.flag_save_interim_graphs_and_movies:
        video_torch_array_to_video(Movie_outliers_QuantileFiltered_LogicalMap, video_name=os.path.join(params.results_folder_seq, 'final_outliers.avi'), FPS=50)

    ### Get Conditions On Chunk Size and Threshold: ###
    max_BGS_threshold = 3


    # ### Save interim outputs if wanted: ###
    # if params.flag_save_interim_graphs_and_movies:
    #     video_torch_array_to_video(Movie_outliers_QuantileFiltered_LogicalMap, video_name=os.path.join(params.results_folder_seq, 'final_outliers_after_connected_components.avi'), FPS=50)
    # # imshow_torch_video(Movie_outliers, None, 50, frame_stride=5)
    # # imshow_torch_video(Movie_outliers_QuantileFiltered_LogicalMap.float().cpu(), None, 50, frame_stride=5)

    ### Reshape Outlier Indices To Proper Numpy Array: ###
    #TODO: this, again, is probably slow and can be avoided!. also, if i decide to use THW indices instead of TXY this whole thing can be speed-up i think.
    # also, i should probably reconsider the way i'm using the TXY indices in general and tuples/lists in general in QuickShot
    outliers_indices_tuple = torch.where(Movie_outliers_QuantileFiltered_LogicalMap.squeeze(1))  #TODO: the tuple received is a tuple of indices
    outliers_indices_tuple = permute_tuple_indices(outliers_indices_tuple, [0,2,1])  #TODO: remember why is permute the tuple indices!!!
    movie_current_outliers_TXY_indices = torch.vstack(outliers_indices_tuple).T.type(torch.float)
    # movie_current_outliers_TXY_indices = np.vstack(np.where(Movie_outliers_QuantileFiltered_LogicalMap)).T
    # movie_current_outliers_TXY_indices = np.1
    ####################################################################################################


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








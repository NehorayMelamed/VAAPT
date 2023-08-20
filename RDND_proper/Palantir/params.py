import os

from easydict import EasyDict


def algorithm_params():
    # all algorithm specific parameters go here
    algo_params = EasyDict()
    algo_params.Height = 540  # possibly should be made into list for experiments of differing sizes
    algo_params.Width = 8192
    algo_params.number_of_images_for_bg_estimation = 50  # debug = 10, undebug = 50
    algo_params.image_index_to_start_from_bg_estimation = 29  # debug = 0, undebug = 29
    algo_params.flag_calculate_separate_shifts_for_bg = True
    # algo_params.crop_H_homog_matrix = 300
    # algo_params.crop_W_homog_matrix = 300  # if used


    # general parameters
    algo_params.number_of_frames_per_bg = 500  # debug = 40, undebug = 1000
    # plotting parameters
    algo_params.alignment_crop_w_diff = 10
    algo_params.alignment_crop_h_diff = 10
    algo_params.alignment_crop_w = algo_params.Width - algo_params.alignment_crop_w_diff
    algo_params.alignment_crop_h = algo_params.Height - algo_params.alignment_crop_h_diff
    algo_params.final_crop_w = algo_params.Width - 50
    algo_params.final_crop_h = algo_params.Height - 50
    # flags
    # experiment_dict.flag_plot = False
    # experiment_dict.flag_estimate_bg = True
    # experiment_dict.flag_video_from_bin_files = True
    return algo_params


def saving_params():
    # all parameters having to do with
    results_params = EasyDict()
    results_params.save_video_fps = 10

    results_params.save_input_frames_as_avi = False
    results_params.save_input_frames_as_png = True

    results_params.save_aligned_frames_as_Bin = False
    results_params.save_aligned_frames_as_avi = False
    results_params.save_aligned_frames_as_png = True

    results_params.save_outliers_as_Bin = False
    results_params.save_outliers_as_avi = True
    results_params.save_outliers_as_png = True

    results_params.save_diff_tensor_as_avi = False
    results_params.save_dif_tensor_as_png = False

    results_params.save_old_bgs = False
    results_params.save_h_matrices = False

    # following params are so we know if a certain metric is used at all. A product of bad code needed for debugging
    results_params.use_input_frames = results_params.save_input_frames_as_png or results_params.save_input_frames_as_avi
    results_params.use_aligned_frames = results_params.save_aligned_frames_as_Bin or results_params.save_aligned_frames_as_png or results_params.save_aligned_frames_as_avi
    results_params.use_diff_tensor = results_params.save_dif_tensor_as_png or results_params.save_diff_tensor_as_avi
    results_params.use_outliers = results_params.save_outliers_as_png or results_params.save_outliers_as_avi or results_params.save_outliers_as_Bin
    return results_params


def config_params():
    # all parameters having to do with the specific machine go here.
    # Perhaps reasonable defaults (ex. working_dir/results/exp_name) should be added
    local_params = EasyDict()
    local_params.experiment_directories = ["/home/avraham/mafat/QS/PalantirData/p1/"]  # must be list
    local_params.results_directory = "/home/avraham/mafat/QS/PalantirResults/"
    local_params.frames_per_batch = 5  # debug = 4, undebug = 20
    local_params.gpu_idx = 0
    local_params.batches_per_saving = 4
    return local_params


def parameters():
    all_params = EasyDict()
    all_params.algo = algorithm_params()
    all_params.results = saving_params()
    all_params.local = config_params()
    return all_params

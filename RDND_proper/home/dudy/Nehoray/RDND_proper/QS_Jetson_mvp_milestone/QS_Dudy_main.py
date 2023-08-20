# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 09:20:06 2022

@author: Lenovo
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import create_folder_if_doesnt_exist
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import decimal_notation
from RapidBase.import_all import *
from QS_Jetson_mvp_milestone.functional_utils.functional_utils import *
from QS_Jetson_mvp_milestone.functional_utils.trajectory_utils import *
from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import *
from QS_Jetson_mvp_milestone.functional_utils.plotting_utils import *
from QS_Jetson_mvp_milestone.functional_utils.Flashlight_utils import *
from QS_Jetson_mvp_milestone.functional_utils.BG_and_noise_estimation import *
from QS_Jetson_mvp_milestone.functional_utils.FFT_analysis import *
import pandas as pd

params = {
    # General params
    'TrjNumFind': 8,  # num of trj. to find. (max number of trajectories to find within frame)
    'TrjNumFind_Flashlight':2,
    'ROI_allocated_around_suspect': 5,
    # The frame size of the sus. close up. (after finding trajectory cut this size ROI around suspect)
    'SeqT': 1,  # Seq. time in sec.
    'SaveResFlg': 1,
    'FrameRate': 500,
    'roi': [512, 640],
    # 'roi': [320, 640],
    'utype': np.uint16,
    'DistFold': r"C:\Users\dudyk\Desktop\dudy_karl\QS_experiments",  # where the files are at. TODO: what is this even used for?
    # 'ExperimentsFold': '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/12.4.2022 - natznatz experiments',
    'ExperimentsFold': r"C:\Users\dudyk\Desktop\dudy_karl\QS_experiments",

    'RunOverAllFilesFlg': 0,  #TODO: change this to be able to toggle between file_name, filenames_list, or experiments_folder
    'flag_run_mode': 'filenames_list',  #(*). 'filename', 'filenames_list', 'experiments_folder'

    'File_Names_List': [
        # r'E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies\8-1000m_background_night_flir_640x512_500fps_20000frames_converted/8-900m_matrice300_night_flir_640x512_800fps_26000frames_converted.Bin',
        # r'E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies\9-900m_matrice600_night_flir_640x512_800fps_26000frames_converted/9-900m_matrice600_night_flir_640x512_800fps_26000frames_converted.Bin',
        # "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_static_3000frames_500fps_drone_at244_90_w640_h320/1200m_matrice300_mavic2_rural_static_3000frames_500fps_drone_at244_90_w640_h320.Bin",
    # "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_cloaser_further_3000frames_500fps_drone_at324_65_w640_h320/1200m_matrice300_mavic2_rural_cloaser_further_3000frames_500fps_drone_at324_65_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_up+down_3000frames_500fps_drone_at324_65_w640_h320/1200m_matrice300_mavic2_rural_up+down_3000frames_500fps_drone_at324_65_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/3000m_matrice300_mavic2_rural_left_right_3000frames_500fps_w640_h320/3000m_matrice300_mavic2_rural_left_right_3000frames_500fps_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/3000m_matrice300_mavic2_rural_static_3000frames_500fps_w640_h320/3000m_matrice300_mavic2_rural_static_3000frames_500fps_w640_h320.Bin",
    # New Experiments: ###
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/3_night_500fps_20000frames_640x320/3_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/6_night_500fps_20000frames_640x320/6_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/12_night_500fps_20000frames_640x320/12_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/14_night_500fps_20000frames_640x320/14_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/15_night_500fps_20000frames_640x320/15_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/21_night_500fps_20000frames_640x320/21_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/22_night_500fps_20000frames_640x320/22_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/23_night_500fps_20000frames_640x320/23_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/24_night_500fps_20000frames_640x320/24_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/25_day_500fps_20000frames_640x320/25_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/26_day_500fps_20000frames_640x320/26_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/27_day_500fps_20000frames_640x320/27_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/28_day_500fps_20000frames_640x320/28_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/31_day_500fps_20000frames_640x320/31_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/32_day_500fps_20000frames_640x320/32_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/33_day_500fps_20000frames_640x320/33_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/34_night_500fps_20000frames_640x320/34_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/35_night_500fps_20000frames_640x320/35_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/36_night_500fps_20000frames_640x320/36_night_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/38_day_500fps_20000frames_640x320/38_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/43_day_500fps_20000frames_640x320/43_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/44_day_500fps_20000frames_640x320/44_day_500fps_20000frames_640x320.Bin',
    # r'E:\Quickshot/12.4.2022 - natznatz experiments/45_day_500fps_20000frames_640x320/45_day_500fps_20000frames_640x320.Bin',
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/QS_fars/birds/third_set/10/OriginalRawMovie.Bin',
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/QS_fars/birds/third_set/0/OriginalRawMovie.pt'
    ### New PelicanD+Flir Experiment: ###
    #(1). PelicanD
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/PelicanD/640x160 500fps 20000frames - 1000m static sky background - scd/640x160 500fps 20000frames - 1000m static sky background - scd.Bin'
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/PelicanD/640x160 500fps 20000frames - 1000m static urban background - scd/640x160 500fps 20000frames - 1000m static urban background - scd.Bin'
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/PelicanD/640x160 800fps 10000frames - 1000m no drone sky background - scd/640x160 800fps 10000frames - 1000m no drone sky background - scd.Bin'
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/PelicanD/640x160 800fps 20000frames -1000m static sky background - scd/640x160 800fps 20000frames -1000m static sky background - scd.Bin'
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/PelicanD/640x160 800fps 20000frames - 1000m left right rural background - scd/640x160 800fps 20000frames - 1000m left right rural background - scd.Bin'
    # '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/PelicanD/640x160 800fps 20000frames - 1000m static rural background - scd/640x160 800fps 20000frames - 1000m static rural background - scd.Bin'
    #(2). FLIR:
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/0_640_512_800fps_10000frames_1000_meter_flir/0_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/1_640_512_800fps_10000frames_1000_meter_flir/1_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/2_640_512_800fps_10000frames_1000_meter_flir/2_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/3_640_512_800fps_10000frames_1000_meter_flir/3_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/4_640_512_800fps_10000frames_1000_meter_flir/4_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/5_640_512_800fps_10000frames_1000_meter_flir/5_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/6_640_512_800fps_10000frames_1000_meter_flir/6_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/7_640_512_800fps_10000frames_1000_meter_flir/7_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/8_640_512_800fps_10000frames_1000_meter_flir/8_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/9_640_512_800fps_10000frames_1000_meter_flir/9_640_512_800fps_10000frames_1000_meter_flir.Bin',
    # r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments/10_640_512_800fps_10000frames_1000_meter_flirr/10_640_512_800fps_10000frames_1000_meter_flir.Bin',

     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\13-2700m_background_night_flir_640x512_500fps_26000frames_converted\\13-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\14-2700m_background_night_flir_640x512_500fps_26000frames_converted\\14-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\15-2700m_background_night_flir_640x512_500fps_26000frames_converted\\15-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\16-2700m_background_night_flir_640x512_500fps_26000frames_converted\\16-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\17-2700m_background_night_flir_640x512_500fps_26000frames_converted\\17-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\18-2700m_background_night_flir_640x512_500fps_26000frames_converted\\18-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\19-2700m_background_night_flir_640x512_500fps_26000frames_converted\\19-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\2-background_daylight_flir_640x512_500fps_20000frames_converted\\2-background_daylight_flir_640x512_500fps_20000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\20-2700m_background_night_flir_640x512_500fps_26000frames_converted\\20-2700m_background_night_flir_640x512_500fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\3-background_daylight_flir_640x512_500fps_20000frames_converted\\3-background_daylight_flir_640x512_500fps_20000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\4-background_daylight_flir_640x512_500fps_20000frames_converted\\4-background_daylight_flir_640x512_500fps_20000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\5-background_daylight_flir_640x512_500fps_20000frames_converted\\5-1000m_background_night_flir_640x512_500fps_20000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\6-background_daylight_flir_640x512_500fps_20000frames_converted\\6-1000m_background_night_flir_640x512_500fps_20000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\7-1000m_background_night_flir_640x512_500fps_20000frames_converted\\7-1000m_background_night_flir_640x512_500fps_20000frames_converted.Bin',
     r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\8-900m_matrice300_night_flir_640x512_800fps_26000frames_converted\\8-900m_matrice300_night_flir_640x512_800fps_26000frames_converted.Bin',
     # r'E:\\Quickshot\\06_07_2022_FLIR\\converted_bin\\large_bin_files_with_avi_movies\\9-900m_matrice600_night_flir_640x512_800fps_26000frames_converted\\9-900m_matrice600_night_flir_640x512_800fps_26000frames_converted.Bin']

    ],


    # TODO: temp! delete!: #
    'number_of_frames_to_read': 2500,
    'frame_to_start_from': 3000,

    # System Parameters:
    'z': 750,  #TODO: change to get distance from description file!!!!!
    'f': 35e-3,
    'pixel_size':15e-6,

    # BGS params
    'BGSSigma': 2.5,  # BGS alg. threshold

    # Drone RANSAC Line est. params: #
    'DroneLineEst_polyfit_degree': 2,  # line est. fit: polynum deg.
    'DroneLineEst_RANSAC_D': 6,  # Ransac diameter: cylinder in which pnts are est. as a line
    'DroneLineEst_RANSAC_max_trials': 100,
    'DroneLineEst_minimum_number_of_samples_before_polyfit': 20,
    'DroneLineEst_minimum_number_of_samples_after_polyfit': 10,
    'DroneLineEst_TrjNumFind': 8,
    'DroneLineEst_ROI_allocated_around_suspect': 5,
    ### Trj. decision params: ###
    'DroneTrjDec_minimum_fraction_of_points_inside_RANSAC_D': 0.15,
    'DroneTrjDec_minimum_fraction_of_frames_where_drone_is_visible': 0.5,
    'DroneTrjDec_allowed_BB_within_frame_H_by_fraction': [0,0.7],
    'DroneTrjDec_allowed_BB_within_frame_W_by_fraction': [0,1],
    'DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible': .15,  # min percentage of points out of the point cloud which are under D. changed to: DroneTrjDec_minimum_fraction_of_points_inside_RANSAC_D
    'DroneTrjDec_minimum_projection_onto_XY_plane': .000,  #0.005,  # max. vel. for sus. units ???.  np.tan(np.arcsin(TrjDecMaxVel)) is num. of pxls per frame
    'DroneTrjDec_maximum_projection_onto_XY_plane': .5,
    'DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible': .5,  # min percentage of frames in which suspect is visible
    'DroneTrjDec_max_time_gap_in_trajectory': 100,  # max. time of sus. disappear in pnt cloud


    ### Flashlight RANSAC Line Estimation Params: ###
    'FlashlightLineEst_polyfit_degree': 2,  # line est. fit: polynum deg.
    'FlashlightLineEst_RANSAC_D': 6,  # Ransac diameter: cylinder in which pnts are est. as a line
    'FlashlightLineEst_RANSAC_max_trials': 100,
    'FlashlightLineEst_minimum_number_of_samples_before_polyfit': 50,
    'FlashlightLineEst_minimum_number_of_samples_after_polyfit': 30,
    'FlashlightLineEst_TrjNumFind': 2,
    'FlashlightLineEst_ROI_allocated_around_suspect': 5,
    'FlashlightTrjDec_minimum_fraction_of_points_inside_RANSAC_D': 0.15,
    'FlashlightTrjDec_minimum_fraction_of_frames_where_flashlight_is_visible': 0.5,
    'FlashlightTrjDec_allowed_BB_within_frame_H_by_fraction': [0,0.7],
    'FlashlightTrjDec_allowed_BB_within_frame_W_by_fraction': [0,1],


    ### Trajectory FFT Decision Parameters: ###
    'DetDFrq': 10,  # for frq. comb binning
    'DetFrqInit': 90,  # for frq. comb binning
    'DetFrqStop': 250,  # for frq. comb binning
    'DetScrThresh': 1,  # min. score for sus.
    'DetScrThresh100Conf': 5,  # sus. score for 100% confidence
    'DetNoiseThresh': 2.5,
    # For FFT pxl of certain frq. range to be considered
    # it should be larger then the noise lvl. times this number

    ### Frequency Analysis Parameters: ###
    'frequency_range_for_drone': 1,
    # the peak size in Hz allocated for a drone (instead of just choosing the max value))
    'drone_start_bin': 90,
    'drone_stop_bin': 240,
    'noise_baseline_start_bin': 170,
    'noise_baseline_stop_bin': 220,
    'low_frequency_start_bin': 15,
    'low_frequency_stop_bin': 60,

    # Brute Force FFT:
    'max_number_of_pixels_to_check': 50,

    # Flags:
    'flag_save_interim_graphs_and_movies': True,
    'flag_perform_BGS_running_mean': True,
    'flag_perform_outlier_running_mean': True,
}



def QS_flow_run_Torch(params):

    ######################################################################################################################
    ### get files to run over list (which can be made up of several experiments within a folder or a single FileName): ###
    main_folder = params['DistFold']
    params.experiments_folders_list = os.listdir(params['ExperimentsFold'])
    if params.flag_run_mode == 'filenames_list':
        experiment_filenames_list = params['File_Names_List']
    elif params.flag_run_mode == 'experiments_folder':
        experiment_filenames_list = get_filenames_from_folder(params['ExperimentsFold'],
                                                              flag_recursive=True,
                                                              flag_full_filename=True,
                                                              string_pattern_to_search='*OriginalRawMovie.Bin',
                                                              flag_sort=False)
    ######################################################################################################################



    ######################################################################################################################
    ### Run over the different experiments and analyze: ###
    for experiment_index in range(len(experiment_filenames_list)):
        ### Get Auxiliary Parameters For Current Experiment: ###
        number_of_frames_per_sequence = np.int(params['FPS'] * params['SeqT'])

        ### Initialize Experiment Params: ###
        flag_bin_file, current_experiment_filename, results_folder, experiment_folder, \
        filename_itself, params, results_summary_string = initialize_experiment(experiment_filenames_list, experiment_index, params)
        if flag_bin_file == False:
            continue

        ### Setup QS_stats DataFrame For Current(!) Experiment: ###
        QS_stats_pd_Experiment = pd.DataFrame()
        save_DataFrame_to_csv(QS_stats_pd_Experiment, os.path.join(params.results_folder, 'QS_stats_pd_Experiment.csv'))

        ### Get Experiment Relevant Variables/Description From Text File: ###
        params = get_experiment_description_from_txt_file(params)

        ### Get binary file reader: ###
        f = get_binary_file_reader(params)


        # #######################################################################################################
        # # ### Start from a certain frame (mostly for testing purposes if i don't want to start from the first frame): ###
        # frame_to_start_from = 500 * 0
        # f = initialize_binary_file_reader(f)
        # Movie_temp = read_frames_from_binary_file_stream(f, 100, frame_to_start_from * 1, params)
        # Movie_temp = Movie_temp.astype(float)
        # # Movie_temp = scale_array_from_range(Movie_temp.clip(q1, q2),
        # #                                        min_max_values_to_clip=(q1, q2),
        # #                                        min_max_values_to_scale_to=(0, 1))
        # Movie_temp = scale_array_stretch_hist(Movie_temp)
        # input_tensor = torch.tensor(Movie_temp).unsqueeze(1)
        # imshow_torch_video(input_tensor, FPS=5)
        # #
        # ### Get Drone Area: ###
        # # Drone = [279, 251], Edge = [302, 221]
        # H_drone = 279
        # W_drone = 251
        # area_around = 3
        # i_vec = np.linspace(H_drone - area_around, H_drone + area_around, area_around * 2 + 1).astype(int)
        # j_vec = np.linspace(W_drone - area_around, W_drone + area_around, area_around * 2 + 1).astype(int)
        #
        # # ### Lower BG If Wanted: ###
        # Movie_BG = np.load(os.path.join(params.results_folder, 'Mov_BG.npy'), allow_pickle=True)
        # (q1, q2) = np.load(os.path.join(params.results_folder, 'quantiles.npy'), allow_pickle=True)
        # input_tensor_BG = torch.tensor(Movie_BG).unsqueeze(0).unsqueeze(0)
        # # gain_estimate = input_tensor[0:1]/(input_tensor_BG + 1e-3)
        # # gain_estimate = gain_estimate[0].median()
        # # input_tensor_BGS = input_tensor - input_tensor_BG * 0
        # # # input_tensor_BGS = input_tensor_BGS / q2
        # # input_tensor_BGS = scale_array_stretch_hist(input_tensor_BGS)
        # # # imshow_torch(scale_array_stretch_hist(input_tensor_BG))
        # # # imshow_torch_video(scale_array_stretch_hist(input_tensor), FPS=50, frame_stride=5)
        # # # imshow_torch_video(scale_array_stretch_hist(input_tensor_BGS), FPS=50, frame_stride=1)
        #
        #
        # ### Get Scale: ###
        # (q1, q2) = np.load(os.path.join(params.results_folder, 'quantiles.npy'), allow_pickle=True)
        # #(1).
        # # input_tensor = scale_array_stretch_hist(input_tensor)
        # #(2).
        # input_tensor = scale_array_from_range(input_tensor.clamp(q1,q2), (q1,q2), (0,1))
        # input_tensor_BG = scale_array_from_range(input_tensor_BG.clamp(q1,q2), (q1,q2), (0,1))
        # input_tensor = input_tensor - input_tensor_BG
        # #(3).
        # # input_tensor = scale_array_to_range(input_tensor)
        #
        # ### Get FFT Over Entire Image: ###
        # # imshow_torch_video(input_tensor, FPS=50)
        # # input_tensor_fft = (torch.fft.rfftn(input_tensor, dim=0).abs())
        # # input_tensor_fft = fftshift_torch(torch.fft.fftn(input_tensor, dim=0).abs(),0)
        # # T,C,H,W = input_tensor_fft.shape
        #
        #
        # ### Get FFT Over Specific Drone Area: ###
        # input_tensor_drone = input_tensor[:, :, i_vec[0]:i_vec[-1]+1, j_vec[0]:j_vec[-1]+1].cuda()
        # input_tensor_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_drone, dim=0).abs(), 0)
        # T = input_tensor_fft.shape[0]
        #
        #
        # for i in np.arange(len(i_vec)):
        #     for j in np.arange(len(j_vec)):
        #         ### Initialize Tensor: ###
        #         # Drone = [279, 251], Edge = [302, 221]
        #         i = int(i)
        #         j = int(j)
        #         input_tensor_fft_graph = input_tensor_fft[:, 0, int(i), int(j)]
        #         input_tensor_fft_graph = input_tensor_fft_graph.clamp(0, 1)
        #         FPS = 500
        #         frequency_axis = torch.tensor(FPS * np.linspace(-0.5, .5 - 1 / T, T))
        #         frequency_axis_numpy = frequency_axis.cpu().numpy()
        #
        #         ### Initialize graphs location and string: ###
        #         ffts_folder = os.path.join(params.experiment_folder, 'FFT_temp')
        #         path_create_path_if_none_exists(ffts_folder)
        #         graph_string = 'H=' + str(int(i_vec[i])) + '_W=' + str(int(j_vec[j]))
        #
        #         ### Peak Detect: ###
        #         input_vec = input_tensor_fft[:, 0:1, int(i):int(i+1), int(j):int(j+1)].abs().cpu()
        #         maxima_peaks, arange_in_correct_dim_tensor = peak_detect_pytorch(input_vec,
        #                                                                          window_size=31,
        #                                                                          dim=0,
        #                                                                          flag_use_ratio_threshold=True,
        #                                                                          ratio_threshold=0,
        #                                                                          flag_plot=True)
        #         # (*). Only Keep Peaks Above Median Noise-Floor Enough:
        #         logical_mask_above_noise_median = input_vec > input_vec.median() * 3
        #         maxima_peaks = maxima_peaks * logical_mask_above_noise_median
        #         # (*). Get Rid Of Peaks Which Are Harmonies Of The Base 23[Hz]:
        #         base_frequency = 23
        #         base_frequency_harmonic_tolerance = 2
        #         frequency_axis_modulo_base = torch.remainder(frequency_axis, base_frequency)
        #         frequency_axis_modulo_base_logical_mask = frequency_axis_modulo_base.abs() > base_frequency_harmonic_tolerance
        #         frequency_axis_modulo_base_logical_mask = frequency_axis_modulo_base_logical_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        #         maxima_peaks = maxima_peaks * frequency_axis_modulo_base_logical_mask
        #         # (*). Get Rid Of Negative Frequencies:
        #         maxima_peaks[0:T//2] = False
        #
        #         #(*). Get All Peak Frequencies:
        #         maxima_peaks_to_plot = (arange_in_correct_dim_tensor)[maxima_peaks]
        #         maxima_frequency_peaks_logical_mask = frequency_axis[maxima_peaks_to_plot] > 0
        #         peak_frequencies_list = frequency_axis[maxima_peaks_to_plot][maxima_frequency_peaks_logical_mask]
        #         peak_frequencies_list = peak_frequencies_list.tolist()
        #         peak_frequencies_array = np.array(peak_frequencies_list)
        #
        #         ### Legends list: ###
        #         legends_list = copy.deepcopy(peak_frequencies_list)
        #         for legend_index in np.arange(len(peak_frequencies_list)):
        #             legends_list[legend_index] = decimal_notation(peak_frequencies_list[legend_index], 1)
        #         legends_list = ['FFT', 'peaks'] + legends_list
        #
        #         ### Plot Results: this is a specific case of tchw tensor, chosen dim t: ###
        #         k1 = 0
        #         k2 = 0
        #         k3 = 0
        #         maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, k1, k2, k3]])[:, k1, k2, k3]
        #         plt.figure()
        #         plt.plot(frequency_axis_numpy, input_vec[:, k1, k2, k3].clamp(0,1).numpy())
        #         # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
        #         plt.plot(frequency_axis[maxima_peaks_to_plot], input_vec[maxima_peaks_to_plot, k1, k2, k3].clamp(0,1).numpy(), '.')
        #         plt.title(graph_string)
        #         for current_legend in legends_list:
        #             plt.plot(frequency_axis_numpy, frequency_axis_numpy*0)
        #         plt.legend(legends_list)
        #         ### Save Graph: ###
        #         save_string = os.path.join(ffts_folder, graph_string + '.png')
        #         plt.savefig(save_string)
        #         plt.close('all')
        #
        #         # ### Save Array: ###
        #         # np.save(os.path.join(ffts_folder, graph_string + '.npy'), input_tensor_fft_graph.cpu().numpy())
        #
        #         # ### Save Graph: ###
        #         # plt.figure()
        #         # plot_torch(frequency_axis, input_tensor_fft_graph)
        #         # plt.title(graph_string)
        #         # plt.show()
        #         # save_string = os.path.join(ffts_folder, graph_string + '.png')
        #         # plt.savefig(save_string)
        #         # plt.close('all')
        # plt.close('all')
        #
        # # # Movie_temp = Movie_temp * 0.8
        # # current_frames_for_video_writer = np.copy(Movie_temp)  # create copy to be used for informative movie to disk
        # # current_frames_for_video_writer = torch_to_numpy_video_ready(torch_get_4D(torch.Tensor(current_frames_for_video_writer), 'THW'))
        # # for i in np.arange(len(Movie_temp)):
        # #     current_frame = current_frames_for_video_writer[i]
        # #     temp_folder_path = os.path.join(params.results_folder, 'TEMP_OMER')
        # #     create_folder_if_doesnt_exist(temp_folder_path)
        # #     save_image_numpy(temp_folder_path, string_rjust(i, 4) + '.png', current_frame, False, flag_scale=False)
        # #######################################################################################################


        #######################################################################################################
        ### Estimate BG Over Entire Movie (because we can't know exactly at which frames there is a flashlight): ###
        if os.path.isfile(os.path.join(params.results_folder, 'Mov_BG.npy')) == False:
            Movie_BG, (q1, q2) = get_BG_over_entire_movie(f, params)
        else:
            Movie_BG = np.load(os.path.join(params.results_folder, 'Mov_BG.npy'), allow_pickle=True)
            (q1, q2) = np.load(os.path.join(params.results_folder, 'quantiles.npy'), allow_pickle=True)
            if Movie_BG.max() < 2:
                Movie_BG = Movie_BG * (q2-q1) + q1
        Movie_BG = torch_get_4D(torch.Tensor(Movie_BG), 'HW')
        save_image_numpy(params.results_folder, 'Movie_BG.png', BW2RGB(scale_array_from_range(Movie_BG.clip(q1,q2),(q1,q2), (0,255))).astype(np.uint8), flag_scale=False)
        (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(Movie_BG)
        params.H = H
        params.W = W

        ### Scale BG: ###
        Movie_BG = scale_array_from_range(Movie_BG.clip(q1, q2),  # TODO: before Efcom this was already in the correct "stretched/[0,1]" format, now i need to redo this
                                              min_max_values_to_clip=(q1, q2),
                                              min_max_values_to_scale_to=(0, 1))
        Movie_BG = Movie_BG.cuda()

        ### Save entire movie of entire experiment for viewing purposes: ###
        flag_original_movie_exists = os.path.isfile(os.path.join(results_folder, 'Original_Movie.avi')) == False and os.path.isfile(os.path.join(results_folder, 'original_movie.avi')) == False
        if flag_original_movie_exists and params.flag_save_interim_graphs_and_movies:
            long_movie_name = os.path.join(results_folder, 'Original_Movie.avi')
            f = initialize_binary_file_reader(f)
            read_long_movie_and_save_to_avi(f, long_movie_name, 25, params)
        ###
        #######################################################################################################

        #######################################################################################################
        # # ### Find Flashlight In Movie & Get Sequence Occurense + Flashlight Polygon Points: ###
        # # #TODO: notice, if there IS a flashlight it MUST be found for the statistics to work
        # if params.was_there_flashlight == 'True':
        #     flashlight_results_list_filename = os.path.join(params.results_folder, 'Flashlight', 'flag_flashlight_found_list.npy')
        #     flag_flaghlight_results_list_exists = os.path.isfile(flashlight_results_list_filename)
        #     if flag_flaghlight_results_list_exists == False:
        #         flag_flashlight_found_list, polygon_points_list, flashlight_BB_list, robust_flashlight_polygon_points = \
        #             Find_Thermal_Flashlight_In_Movie(f, Movie_BG, params)
        #     else:
        #         flag_flashlight_found_list = np.load(flashlight_results_list_filename, allow_pickle=True)
        #         flashlight_BB_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'flashlight_BB_list.npy'), allow_pickle=True)
        #         robust_flashlight_polygon_points = np.load(os.path.join(params.results_folder, 'Flashlight', 'robust_flashlight_polygon_points.npy'), allow_pickle=True)
        # else:
        #     flag_flashlight_found_list = np.array([False])
        #     flashlight_BB_list = np.array([None])
        #     robust_flashlight_polygon_points = np.array([None])

        flag_flashlight_found_list = np.array([False])
        flashlight_BB_list = np.array([None])
        robust_flashlight_polygon_points = np.array([None])
        #######################################################################################################


        #######################################################################################################
        ### Initialize video with information (flashlight, drone, etc') on it of the entire experiment: ###
        if params.flag_save_interim_graphs_and_movies:
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            informative_movie_full_file_name = os.path.join(params.results_folder, 'Full_Informative_Movie.avi')
            informative_movie_video_writer = cv2.VideoWriter(informative_movie_full_file_name, fourcc, 50, (W, H))
        #######################################################################################################

        #######################################################################################################
        ### ### Go Over Sequences Of Current Experiment: ### ###
        ### ### ### ### ### ### ### ### ### ### ### ###  ### ###

        ### Initialize current flashlight polygon points with the first polygon found (in case the movie doesn't start with it): ###
        current_polygon_points = robust_flashlight_polygon_points

        ### Loop over the different sequences and analyze: ###
        f = initialize_binary_file_reader(f)
        flag_enough_frames_to_analyze = True
        sequence_index = -1
        flag_drone_trajectory_inside_BB_list = []
        ### Start From Non-Zero Frame: ###
        movie_temp_for_fast_forward = read_frames_from_binary_file_stream(f, number_of_frames_per_sequence, 500*4, params)
        while flag_enough_frames_to_analyze:
            sequence_index += 1

            ### Make results path for current sequence: ###
            results_folder_seq = os.path.join(results_folder, 'Sequences', "seq" + str(sequence_index))
            print(params['FileName'] + ": seq" + str(sequence_index))
            create_folder_if_doesnt_exist(results_folder_seq)
            params.results_folder_seq = results_folder_seq

            ### Get relevant frames from entire movie: ###
            Movie = read_frames_from_binary_file_stream(f, number_of_frames_per_sequence, 0, params)

            ### Only Continue If We Have Enough Frames: ###
            flag_enough_frames_to_analyze = (Movie.shape[0] == number_of_frames_per_sequence)
            results_summary_string = ''
            if flag_enough_frames_to_analyze:
                ### Stretch movie range according to previously found quantiles: ###
                Movie = scale_array_from_range(Movie.clip(q1, q2),
                                                       min_max_values_to_clip=(q1, q2),
                                                       min_max_values_to_scale_to=(0, 1))
                Movie = torch.Tensor(Movie).cuda().unsqueeze(1)
                current_frames_for_video_writer = np.copy(Movie)  # create copy to be used for informative movie to disk
                current_frames_for_video_writer = torch_to_numpy_video_ready(torch_get_4D(torch.Tensor(current_frames_for_video_writer), 'THW'))

                ### Save single image and Temporal Average of current batch: ###
                if os.path.isfile(os.path.join(results_folder_seq, 'temporal_average.png')):
                    cv2.imwrite(os.path.join(results_folder_seq, 'temporal_average.png'), (Movie * 255).mean(0).clip(0, 255))
                    cv2.imwrite(os.path.join(results_folder_seq, 'single_image.png'), (Movie[0] * 255).clip(0, 255))

                ### Check if flashlight found, if not -> search for drone: ###
                if flag_flashlight_found_list[sequence_index]:
                    # (*). if there's a flashlight, don't try and find a drone, simply use the flashlight polygon points as future GT bounding box:
                    print('seq: ' + str(sequence_index) + ', Flashlight ON')

                    ### Write results down to txt file: ###
                    results_summary_string += 'Sequence ' + str(sequence_index) + ', '
                    results_summary_string += 'Flashlight On, No Drone Search Algorithm Done'
                    results_summary_string += '\n'
                    open(os.path.join(results_folder, 'res_summ.txt'), 'a').write(results_summary_string)

                    ### Get Flashlight Polygon and Locatoin (BB): ###
                    current_polygon_points = robust_flashlight_polygon_points
                    current_flashlight_BB = flashlight_BB_list[sequence_index]

                    ### Plot Polygon and Flashlight On Large Movie: ###
                    if params.flag_save_interim_graphs_and_movies:
                        for inter_frame_index in np.arange(current_frames_for_video_writer.shape[0]):
                            current_movie_frame = current_frames_for_video_writer[inter_frame_index]
                            current_movie_frame = draw_polygon_points_on_image(current_movie_frame, current_polygon_points)
                            current_movie_string = 'Sequence ' + str(sequence_index) + ',    '
                            current_movie_string += 'Flashlight On        '
                            current_movie_string += 'No Drone Search Algorithm Done        '
                            current_movie_frame = draw_text_on_image(current_movie_frame, text_for_image=current_movie_string, fontScale=0.3, thickness=1)
                            current_movie_frame = Plot_BoundingBox_On_Frame(current_movie_frame, BoundingBox_list=[current_flashlight_BB[inter_frame_index]])
                            current_frames_for_video_writer[inter_frame_index] = current_movie_frame
                            informative_movie_video_writer.write(current_movie_frame)


                elif flag_flashlight_found_list[sequence_index] == False:
                    print('seq: ' + str(sequence_index) + ', No Flashlight, Searching For Drone')
                    # (*). if there is NO flashlight, search for a drone:

                    ##################################################################################
                    ### Maor Outlier Trajectory Finding Method: ###
                    tic()
                    Movie_BGS, Movie_BGS_std, \
                    trajectory_tuple, trajectory_tuple_BeforeFFTDec,\
                    t_vec, res_points, NonLinePoints, xyz_line, number_of_drone_trajectories_found, \
                    TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec_per_trajectory, TrjMov, \
                    DetectionDec, DetectionConfLvl, params = Dudy_Analysis_torch(Movie, Movie_BG, params)
                    toc('   Maor Analysis')
                    ### Unpack trajectory_tuple: ###
                    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y = trajectory_tuple
                    ##################################################################################

                    ##################################################################################
                    ### Plot Results: ###
                    if params.flag_save_interim_graphs_and_movies:
                        # (0). Turn Tensors To Numpy Arrays because we're dealing with plotting which is numpy anyway: ###
                        res_points, NonLinePoints, xyz_line, frequency_vec_per_trajectory, \
                        Movie, Movie_BGS, Movie_BGS_std, \
                        trajectory_tuple_BeforeFFTDec, trajectory_tuple, TrjMovie_FFT_BinPartitioned_AfterScoreFunction =\
                            tuple_to_numpy((res_points, NonLinePoints, xyz_line, frequency_vec_per_trajectory,
                                            Movie, Movie_BGS, Movie_BGS_std,
                                            trajectory_tuple_BeforeFFTDec, trajectory_tuple, TrjMovie_FFT_BinPartitioned_AfterScoreFunction))
                        Movie_BGS = Movie_BGS.squeeze(1)
                        Movie_BGS_std = Movie_BGS_std.squeeze(1)

                        # (1). Plot 3D CLoud and Outline Trajectories (of ALL trajectories found by RANSAC):
                        Plot_3D_PointCloud_With_Trajectories_Demonstration(res_points, NonLinePoints, xyz_line, number_of_drone_trajectories_found, params,
                                                             "Plt3DPntCloudTrj_AllTrajectoriesFromRANSAC", results_folder_seq)
                        _, Movie_with_BB = Plot_Bounding_Box_Demonstration(Movie, prestring='', results_folder='', trajectory_tuple=trajectory_tuple)
                        # imshow_torch_video(Movie_with_BB.unsqueeze(1), FPS=50)

                        # (2). Save Movie With ALL Trajectories Found From RANSAC (before going through frequency analysis):
                        tic()
                        Save_Movies_With_Trajectory_BoundingBoxes(Movie, Movie_BGS, Movie_BGS_std,
                                                                  prestring='All_Trajectories_RANSAC_Found',
                                                                  results_folder=results_folder_seq,
                                                                  trajectory_tuple=trajectory_tuple_BeforeFFTDec)
                        # (3). Save Movie With Only Valid Trajectories:
                        Save_Movies_With_Trajectory_BoundingBoxes(Movie, Movie_BGS, Movie_BGS_std,
                                                                  prestring='Only_Valid_Trajectories',
                                                                  results_folder=results_folder_seq,
                                                                  trajectory_tuple=trajectory_tuple)
                        toc('   save Movie with Bounding Boxes')
                        # (3). Plot The Detection Over different frequency bands:
                        tic()
                        Plot_FFT_Bins_Detection_SubPlots(number_of_drone_trajectories_found,
                                                         TrjMovie_FFT_BinPartitioned_AfterScoreFunction,
                                                         frequency_vec_per_trajectory, params, results_folder_seq, TrjMov, DetectionDec, DetectionConfLvl)
                        toc('   show and save FFT bins plots`')
                    ##################################################################################

                    ##################################################################################
                    ### Automatic Location Labeling - was the drone that was found inside the "GT" label, as defined by the flashlight polygon:
                    # (*). Check if trajectory found is within flashlight polygon:
                    #TODO: add a wrapper function which works for multiple/swarm flashlights!!!!
                    flag_drone_trajectory_inside_BB_list, current_frames_for_video_writer = \
                        check_for_each_trajectory_if_inside_flashlight_polygon(Movie, current_frames_for_video_writer,
                                                                               trajectory_tuple, current_polygon_points)
                    ##################################################################################

                    ##################################################################################
                    ### Loop over all frames intended for video writer (with all the information drawn on them) and write the video file: ###
                    if params.flag_save_interim_graphs_and_movies:
                        current_frames_for_video_writer = draw_trajectories_on_images(current_frames_for_video_writer, trajectory_tuple)
                        tic()
                        full_t_vec = np.arange(Movie.shape[0])
                        for inter_frame_index in full_t_vec:
                            current_movie_frame = current_frames_for_video_writer[inter_frame_index]
                            current_movie_string = 'Sequence ' + str(sequence_index) + ',    '
                            current_movie_string += 'Flashlight OFF        '
                            current_movie_string += 'Preseting Drone Trajectories'
                            current_movie_frame = draw_text_on_image(current_movie_frame, text_for_image=current_movie_string, fontScale=0.3, thickness=1)
                            if current_polygon_points.all() != None:
                                current_movie_frame = draw_polygon_points_on_image(current_movie_frame, current_polygon_points)
                            informative_movie_video_writer.write(current_movie_frame)
                        toc('   draw stuff on images and write them to .avi file')
                    #####################################################################################

                    ##################################################################################
                    ### Update Experiment DataFrame With Current Sequence: ###
                    QS_stats_pd_Experiment = Update_QS_DataFrame(QS_stats_pd_Experiment, params, flag_drone_trajectory_inside_BB_list)

                    ### Save/Load From Disk Using CSV Using Excel (to this sequence specifically!...later on grab all of the them and analyze): ###
                    save_DataFrame_to_csv(QS_stats_pd_Experiment, os.path.join(params.results_folder_seq, 'QS_stats_pd_Experiment.csv'))
                    ##################################################################################

                    ##################################################################################
                    ### Write Results To Summary Text File: ###
                    if number_of_drone_trajectories_found > 0:
                        for trajectory_index in np.arange(number_of_drone_trajectories_found):
                            current_drone_detection_confidence = DetectionConfLvl[trajectory_index]

                            ### Write results down to txt file: ###
                            results_summary_string += 'Sequence ' + str(sequence_index) + ': \n'
                            results_summary_string += '             Trajectory ' + str(trajectory_index) + \
                                                      ', Confidence: ' + str(current_drone_detection_confidence.cpu().item())

                            ### If there was no flashlight say so, and if there was then check if it's inside flashlight BB: ###
                            if flag_drone_trajectory_inside_BB_list == []:
                                results_summary_string += '             No Flashlight!!!: '
                            else:
                                flag_current_trajectory_inside_flashlight_BB = flag_drone_trajectory_inside_BB_list[trajectory_index]
                                results_summary_string += '             Inside Flashlight BB: ' + str(flag_current_trajectory_inside_flashlight_BB)

                            results_summary_string += '\n'
                    else:
                        results_summary_string += 'Sequence ' + str(sequence_index) + ': \n'
                        results_summary_string += 'No Drones Found!'
                        results_summary_string += '\n'
                    open(os.path.join(results_folder, 'res_summ.txt'), 'a').write(results_summary_string)
                    ##################################################################################

        ### Release When Current Experiment Is Done: ###
        informative_movie_video_writer.release()



    ### Get Global Statistics Folder: ###
    statistics_folder = os.path.join(params.ExperimentsFold, 'Global_Statistics')
    params.statistics_folder = statistics_folder
    create_folder_if_doesnt_exist(statistics_folder)

    ### Gather and Present Statistics For Current Experiment: ###
    if os.path.isfile(os.path.join(params.statistics_folder, 'QS_stats_pd_global.csv')):
        # (*). if already exists -> load it:
        QS_stats_pd_global = load_DataFrame_from_csv(os.path.join(params.statistics_folder, 'QS_stats_pd_global.csv'))
    else:
        # (*). if it doesn't exist -> create it:
        QS_stats_pd_global = pd.DataFrame()
        save_DataFrame_to_csv(QS_stats_pd_global, os.path.join(params.statistics_folder, 'QS_stats_pd_global.csv'))


    ### Loop over all experiment sub-folders and get all the statistics from everywhere: ###
    experiment_filenames_list = get_filenames_from_folder(params['ExperimentsFold'],
                                                          flag_recursive=True,
                                                          flag_full_filename=True,
                                                          string_pattern_to_search='*QS_stats_pd_Experiment.csv',
                                                          flag_sort=False)
    for experiment_csv_filename in experiment_filenames_list:
        QS_stats_pd_Current_Experiment = load_DataFrame_from_csv(experiment_csv_filename)

        ### Add Current Experiment Stats To Global Stats DataFrame: ###
        QS_stats_pd_global = pd.concat([QS_stats_pd_global, QS_stats_pd_Current_Experiment])

    ### Get unique Vecs: ###
    distances_vec = np.unique(QS_stats_pd_global.distance.unique().astype(np.float))
    scenes_vec = np.unique(QS_stats_pd_global.background_scene.unique())

    # ### Analyze Results To Give PD, FAR, Confusion Matrix: ###
    # # (*). get only those from distance 1500:
    # QS_stats_pd_global[QS_stats_pd_global['distance'] == 1500]
    # # (*). get only those with urban background:
    # QS_stats_pd_global[QS_stats_pd_global['background_scene'] == 'urban']
    # # (*). get only those with urban background and distance from 1500-2000:
    # QS_stats_pd_global[QS_stats_pd_global['background_scene'] == 'urban'][QS_stats_pd['distance'] == 1500]
    # # (*). get only those where no drone was present (for 24/7 FAR statistics):
    # QS_stats_pd_global[QS_stats_pd_global['flag_was_there_drone'] == True]
    # # (*). get only those where no drone was detected but drone was present:
    # QS_stats_pd_global[QS_stats_pd_global['flag_was_there_drone'] == True][QS_stats_pd['flag_was_drone_detected'] == True]

    ### Loop over distances and get confusion matrix: ###
    FF_list = []
    FT_list = []
    TF_list = []
    TT_list = []
    for current_distance in distances_vec:
        ### Get Confusion Matrix: ###
        df_current_distance = QS_stats_pd_global[QS_stats_pd_global['distance'].astype(np.float) == current_distance]
        # (1). There was no drone and if wasn't detection (FF):
        FF_records = df_current_distance[df_current_distance['flag_was_there_drone'] == False]
        FF_records = FF_records[FF_records['flag_was_drone_detected'] == False]
        FF = len(FF_records)
        # (2). There was no drone and drone was detection (FT), False Alarm (FAR):
        FT_records = df_current_distance[df_current_distance['flag_was_there_drone'] == False]
        FT_records = FT_records[FT_records['flag_was_drone_detected'] == True]
        FT = len(FT_records)
        # (3). There WAS a drone but no drone was detected (TF):
        TF_records = df_current_distance[df_current_distance['flag_was_there_drone'] == True]
        TF_records = TF_records[TF_records['flag_was_drone_detected'] == False]
        TF = len(TF_records)
        # (4). There WAS a drone and it WAS detected (TT):
        TT_records = df_current_distance[df_current_distance['flag_was_there_drone'] == True]
        TT_records = TT_records[TT_records['flag_was_drone_detected'] == True]
        TT = len(TT_records)

        ### Update Lists Per Distance: ###
        total_sum = FF + FT + TF + TT
        FF_list.append(FF / total_sum)
        FT_list.append(FT / total_sum)
        TF_list.append(TF / total_sum)
        TT_list.append(TT / total_sum)

    ### Plot Confusion Matrix Values As A Function Of Distance: ###
    #(1). FF:
    plt.figure()
    plot(distances_vec, FF_list, 'ro');
    plt.title('There was no drone and it was not detected (FF)')
    plt.xlabel('Distance[m]')
    plt.ylabel('Probablity[%]')
    plt.savefig(os.path.join(params.statistics_folder, 'FF_plot.png'))
    #(2). FT:
    plt.figure()
    plot(distances_vec, FT_list, 'ro');
    plt.title('There was no drone and drone was detection (FT), False Alarm (FAR)')
    plt.xlabel('Distance[m]')
    plt.ylabel('Probablity[%]')
    plt.savefig(os.path.join(params.statistics_folder, 'FT_plot.png'))
    #(3). TF:
    plt.figure()
    plot(distances_vec, TF_list, 'ro');
    plt.title('There WAS a drone but no drone was detected (TF)')
    plt.xlabel('Distance[m]')
    plt.ylabel('Probablity[%]')
    plt.savefig(os.path.join(params.statistics_folder, 'TF_plot.png'))
    #(4). TT:
    plt.figure()
    plot(distances_vec, TT_list, 'ro');
    plt.title('There WAS a drone and it WAS detected (TT)')
    plt.xlabel('Distance[m]')
    plt.ylabel('Probablity[%]')
    plt.savefig(os.path.join(params.statistics_folder, 'TT_plot.png'))
    ### Close All Figures: ###
    plt.close('all')

    ### Save Global Results (after adding current experiments) To Statistics File: ###
    save_DataFrame_to_csv(QS_stats_pd_global, os.path.join(params.statistics_folder, 'QS_stats_pd_global.csv'))









params = EasyDict(params)
# QS_flow_run(params)
QS_flow_run_Torch(params)
# QS_flow_run_Torch_NoFlashlight(params)


# #######################################################################################################
# ### Get Movie Sample For Viewing and (q1,q2) Quantiles: ###
# # (1). read movie samples:
# Mov = read_frames_from_binary_file_stream(f,
#                                           number_of_frames_to_read=250,
#                                           number_of_frames_to_skip=0,
#                                           params=params)
# # (2). get numpy and torch versions:
# Mov = Mov.astype(np.float)
# Movie = torch_get_4D(torch.Tensor(Mov), 'THW')  # transform to [T,C,H,W]
# # (3). find (q1,q2) quantiles:
# Movie, (q1, q2) = scale_array_stretch_hist(Movie, flag_return_quantile=True)
# # (4). show movie sample using imshow_torch_video:
# # imshow_torch_video(Movie, 2500, 50, frame_stride=5)
#
# ### TODO: Temp for stationary drone signal testing, delete later: ###
# f = initialize_binary_file_reader(f)
# Movie = read_frames_from_binary_file_stream(f, 200, 2500*1, params)
# Movie = Movie.astype(np.float)
# Movie, (q1, q2) = scale_array_stretch_hist(Movie, flag_return_quantile=True)
# Movie = torch.Tensor(Movie)
# # Movie = Movie - Movie.mean(0)
# # imshow_torch_video(torch.Tensor(Movie[0:500]), 2500, 50, frame_stride=4)

#### Test video alignment: ###
# Movie_aligned = Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(Movie[0:200,0])
# Movie_aligned = Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation(Movie[0:200,0])
# imshow_torch_video(Movie[0:500], 2500, 50, frame_stride=4)
# imshow_torch_video(Movie_aligned[0:500], 2500, 50, frame_stride=4)
# imshow_torch_video(Movie_aligned[0:500] - Movie_BG , 2500, 50, frame_stride=4)
# imshow_torch_video(Movie[0:500] - Movie_BG , 2500, 50, frame_stride=4)

# ## Test Align Movie After RCC: ###
# #(*). Clone Tensor to keep the original for later:
# Movie = Movie[0:200]
# Movie_before_RCC = torch.clone(Movie).cpu()
# #(1). RCC per frame:
# Movie, residual_tensor = perform_RCC_on_tensor(Movie_before_RCC.unsqueeze(1), lambda_weight=50, number_of_iterations=1)
# imshow_torch_video(residual_tensor, 2500, 50, frame_stride=4)
# imshow_torch_video(Movie, 2500, 50, frame_stride=4)
# imshow_torch_video(Movie - Movie.mean(0,True), 2500, 50, frame_stride=4)
# std_per_pixel = (Movie - Movie.mean(0,True)).std([0])
# video_torch_array_to_video(Movie_before_RCC.clip(0,1), os.path.join(params.results_folder, 'blabla_before_RCC.avi'), 50)
# #(2). RCC from mean frame:
# Movie_mean_RCC, residual_tensor = perform_RCC_on_tensor(Movie_before_RCC.mean(0,True), lambda_weight=50, number_of_iterations=1)
# Movie = Movie_before_RCC - residual_tensor
# #(3). RCC from mean frame and then RCC on what's left:
# Movie_mean_RCC, residual_tensor_mean_RCC = perform_RCC_on_tensor(Movie_before_RCC.mean(0, True), lambda_weight=50, number_of_iterations=1)
# Movie_minus_mean_RCC_Residual = Movie_before_RCC - residual_tensor_mean_RCC
# Movie, residual_tensor = perform_RCC_on_tensor(Movie_minus_mean_RCC_Residual, lambda_weight=50, number_of_iterations=1)
# #(4). Use CrossCorrelation or WeightedCrossCorrelation For Alignment
# Movie = Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(Movie)
# Movie = Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation(Movie_before_RCC)
# print('done weighted cross correlation alignment')
# #(5). Show Movie after alignment:
# imshow_torch_video(Movie_before_RCC[0:500], 2500, 50, frame_stride=4)
# imshow_torch_video(torch_get_4D(Movie[0:500],'THW'), 2500, 50, frame_stride=4)
# #(6). Save aligned Movie to disk:
# numpy_array_to_video(BW2RGB(np.transpose(np.expand_dims(Movie, 0), (1, 2, 3, 0))),
#                      os.path.join(results_folder_seq, 'Aligned_Movie.avi'), 25.0)
# #######################################################################################################




###########################################################################################################
### Another Attemp At Stabilization: ###
### Stabilize Frame Using Cross-Correlation Alignment Functions: ###
# # aligned_tensor, shifts_h, shifts_w, CC_matrix = align_to_reference_frame_circular_cc(matrix=input_tensor,
# #                                                                                      reference_matrix=input_tensor_BG,
# #                                                                                      matrix_fft=None,
# #                                                                                      normalize_over_matrix=False,
# #                                                                                      warp_method='bicubic',
# #                                                                                      crop_warped_matrix=False)
# # aligned_tensor, shifts_h, shifts_w, CC_matrix = align_to_center_frame_circular_cc(matrix=input_tensor.cuda(),
# #                                                                                      matrix_fft=None,
# #                                                                                      normalize_over_matrix=False,
# #                                                                                      warp_method='bicubic',
# #                                                                                      crop_warped_matrix=False)
#
# ### Stabilize Using Weighted Cross Correlation: ###
# T, C, H, W = input_tensor.shape
# input_tensor = input_tensor.cuda()
# input_tensor_BG = input_tensor[T//2:T//2+1].cuda()
# input_tensor_canny = kornia.filters.canny(scale_array_stretch_hist(input_tensor_BG))[1]
# input_tensor_canny = kornia.morphology.dilation(input_tensor_canny, torch.ones(3,3).to(input_tensor_canny.device))
# input_tensor_canny[:,:,0:H//2,:] = 0
# input_tensor_canny[:,:,:,0:170] = 0
# input_tensor_canny[:,:,-1,:] = 0
# input_tensor_canny[:,:,:,-1] = 0
# input_tensor_canny[:,:,0,:] = 0
# input_tensor_canny[:,:,:,0] = 0
# aligned_tensor, shifts_x, shifts_y = Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation(input_tensor.squeeze(1),
#                                                                                                      input_tensor_BG.squeeze(1),
#                                                                                                      input_tensor_canny)
#
# shifts_x_bias, shifts_y_bias = Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation_MultipleReference(input_tensor.squeeze(1),
#                                                                                                                      input_tensor.squeeze(1),
#                                                                                                                      input_tensor_canny)
# shifts_x -= shifts_x_bias
# shifts_y -= shifts_y_bias
#
# # ### Interpolate According to shifts after bias removal: ###
# # input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0),
# #                                                     construct_tensor(-shifts_y),
# #                                                     construct_tensor(-shifts_x),
# #                                                     construct_tensor(0 * np.pi / 180),
# #                                                     construct_tensor(1),
# #                                                     warp_method='bicubic',
# #                                                     expand=False).squeeze(0)
# # imshow_torch_video(scale_array_stretch_hist(input_tensor_warped - input_tensor_BG.to(aligned_tensor.device)), FPS=50, frame_stride=1)
# # # imshow_torch_video(scale_array_stretch_hist(aligned_tensor), FPS=50, frame_stride=5)
# # # imshow_torch_video(scale_array_stretch_hist(aligned_tensor - input_tensor_BG.to(aligned_tensor.device)), FPS=50, frame_stride=1)
# # # imshow_torch_video(scale_array_stretch_hist(aligned_tensor - input_tensor[T // 2:T // 2 + 1].to(aligned_tensor.device)), FPS=50, frame_stride=1)
#
# ### Try Different Gain Factors On The Shifts: ###
# gain_factors_vec = torch.tensor(np.linspace(0, 2, 20))
# diff_list = []
# for g in np.arange(len(gain_factors_vec)):
#     gain_factor_current = gain_factors_vec[g]
#     input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0),
#                                                         construct_tensor(-shifts_y * gain_factor_current),
#                                                         construct_tensor(-shifts_x * gain_factor_current),
#                                                         construct_tensor(0 * np.pi / 180),
#                                                         construct_tensor(1),
#                                                         warp_method='bicubic',
#                                                         expand=False).squeeze(0)
#     logical_mask = (input_tensor_canny==1).repeat(T,1,1,1)
#     diff_tensor = (input_tensor_warped - input_tensor_BG).abs()[logical_mask].sum().item()  #TODO: maybe switch to only where canny edge detection applies
#     diff_list.append(diff_tensor)
# diff_list = torch.tensor(np.array(diff_list))
# plot_torch(gain_factors_vec, diff_list)
# min_gain_factor = gain_factors_vec[diff_list.argmin()]
# min_gain_factor = 0.7
#
# ### Warp According To Min Gain Factor: ###
# input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0),
#                                                     construct_tensor(-shifts_y * min_gain_factor),
#                                                     construct_tensor(-shifts_x * min_gain_factor),
#                                                     construct_tensor(0 * np.pi / 180),
#                                                     construct_tensor(1),
#                                                     warp_method='bicubic',
#                                                     expand=False).squeeze(0)
# imshow_torch_video(scale_array_stretch_hist(input_tensor_warped - input_tensor_BG.to(aligned_tensor.device)), FPS=50, frame_stride=1)
# # imshow_torch_video(scale_array_stretch_hist(aligned_tensor), FPS=50, frame_stride=5)
# # imshow_torch_video(scale_array_stretch_hist(aligned_tensor - input_tensor_BG.to(aligned_tensor.device)), FPS=50, frame_stride=1)
# # imshow_torch_video(scale_array_stretch_hist(aligned_tensor - input_tensor[T // 2:T // 2 + 1].to(aligned_tensor.device)), FPS=50, frame_stride=1)
#
# # ### Use Gimbaless To Get Pixel-Level Flow Prediction: ###
# # gimbaless_layer = Gimbaless_Layer_Torch(H, 0)
# # delta_x, delta_y = gimbaless_layer.forward(input_tensor.unsqueeze(0), input_tensor_BG.unsqueeze(0))
# # flow_magnitude = torch.sqrt(delta_x ** 2 + delta_y ** 2)
# # warp_object = Warp_Object()
# # warped_tensor = warp_object.forward(input_tensor, -delta_x.squeeze(0), -delta_y.squeeze(0))
# # imshow_torch_video(scale_array_stretch_hist(warped_tensor-input_tensor_BG), FPS=50, frame_stride=1)
# # imshow_torch_video(flow_magnitude, FPS=50, frame_stride=1)


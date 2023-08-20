# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 09:20:06 2022

@author: Lenovo
"""
import copy
import os

import kornia.filters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kornia.filters import MedianBlur

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

# filepath = r'E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies\8-1000m_background_night_flir_640x512_500fps_20000frames_converted/8-900m_matrice300_night_flir_640x512_800fps_26000frames_converted.Bin'

params = {
    # General params
    'TrjNumFind': 32,  # num of trj. to find. (max number of trajectories to find within frame)
    'TrjNumFind_Flashlight': 2,
    'ROI_allocated_around_suspect': 7,
    # The frame size of the sus. close up. (after finding trajectory cut this size ROI around suspect)
    'SeqT': 1,  # Seq. time in sec.
    'SaveResFlg': 1,
    'FrameRate': 800,
    'mini_sequence_size_for_BG_estimation': 100,
    # 'roi': [512, 640],
    # 'roi': [320, 640],
    'utype': np.uint16,
    'DistFold': r"C:\Users\dudyk\Desktop\dudy_karl\QS_experiments",  # where the files are at. TODO: what is this even used for?
    # 'ExperimentsFold': '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/12.4.2022 - natznatz experiments',
    'ExperimentsFold': r"E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies",

    'RunOverAllFilesFlg': 0,  #TODO: change this to be able to toggle between file_name, filenames_list, or experiments_folder
    'flag_run_mode': 'filenames_list',  #(*). 'filename', 'filenames_list', 'experiments_folder'

    'File_Names_List': [

    # #(3). FLIR E-Drive:
        r'E:\Palantir\famous_movie\one_big_bin_file\2/2.Bin'
        # r'E:\Quickshot\29.06.22_exp_bin\one_big_bin_file/0-Background_Flir_640x512_500fps_20000frames_converted/0-Background_Flir_640x512_500fps_20000frames_converted.Bin',
        # r'E:\Quickshot\29.06.22_exp_bin\one_big_bin_file/1-2natztz_1km_Flir_640x512_500fps_20000frames_converted/1-2natztz_1km_Flir_640x512_500fps_20000frames_converted.Bin',
        # r'E:\Quickshot\29.06.22_exp_bin\one_big_bin_file/2-2natztz_Take2_1km_Flir_640x512_500fps_20000frames_converted/2-2natztz_Take2_1km_Flir_640x512_500fps_20000frames_converted.Bin',
        # r'E:\Quickshot\29.06.22_exp_bin\one_big_bin_file/3-Mavic2_1km_Left_Right_Flir_640x512_500fps_20000frames_converted/3-Mavic2_1km_Left_Right_Flir_640x512_500fps_20000frames_converted.Bin',
        # r'E:\Quickshot\29.06.22_exp_bin\one_big_bin_file/4-Mavic2_Take2_1km_Left_Right_Flir_640x512_500fps_20000frames_converted/4-Mavic2_Take2_1km_Left_Right_Flir_640x512_500fps_20000frames_converted.Bin',
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
    'DroneTrjDec_minimum_fraction_of_points_inside_RANSAC_D': 0.5,
    'DroneTrjDec_allowed_BB_within_frame_H_by_fraction': [0, 0.7],
    'DroneTrjDec_allowed_BB_within_frame_W_by_fraction': [0, 1],
    'DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible': 0.75,  # min percentage of points out of the point cloud which are under D. changed to: DroneTrjDec_minimum_fraction_of_points_inside_RANSAC_D
    'DroneTrjDec_minimum_projection_onto_XY_plane': 0.000,  #0.005,  # max. vel. for sus. units ???.  np.tan(np.arcsin(TrjDecMaxVel)) is num. of pxls per frame
    'DroneTrjDec_maximum_projection_onto_XY_plane': 0.5,
    'DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible': 0.75,
    'DroneTrjDec_max_time_gap_in_trajectory': 50,
    'DroneTrjDec_max_BGS_threshold': 0.2,  # to delete effect default value = np.inf

    ### Flashlight RANSAC Line Estimation Params: ###  #TODO: will probably get rid of this for manual flashlight. it is interesting to understand how to do it robustly but not now
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

    ### Outlier Detection Params: ###
    'BGS_runnning_mean_size': 5,
    'outlier_detection_running_mean_size': 5,
    'outlier_detection_BGS_threshold': 0.03,
    'number_of_batches_before_considered_BG': 25,
    'global_outlier_threshold': 6,
    'difference_to_STD_ratio_threshold': 8,
    'outlier_after_temporal_mean_threshold': 3,

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
    #(*). Frequencies:
    'frequency_range_for_drone': 1, # the peak size in Hz allocated for a drone (instead of just choosing the max value))
    'drone_start_bin': 80,
    'drone_stop_bin': 170,
    'noise_baseline_start_bin': 170,
    'noise_baseline_stop_bin': 245,
    'low_frequency_start_bin': 15,
    'low_frequency_stop_bin': 60,
    ### FFT Decision Parameters: ###
    'logistic_function_reference_value': 5,  #to pass all peaks - 1(???)
    'logistic_function_sigma_value': 5,
    'low_frequency_lobe_over_noise_floor_threshold': 7,  #to pass all peaks - np.inf
    'frequency_peaks_distance_still_considered_same_drone': 15,
    'peak_detection_window_size': 15,
    'cooler_base_harmonic': 37,
    'FFT_SNR_threshold_initial': 3,  #to pass all peaks - 0
    'FFT_SNR_threshold_final': 3.2,  #to pass all peaks - 0
    'base_frequency_harmonic_tolerance': 2.5, #[Hz]
    #(*). DC Lobe Size:
    'FWHM_fraction': 0.2,  #to pass all peaks - 1
    'max_DC_lobe_size': 70, #[Hz]
    'DC_value_to_noise_threshold': 7,  #to pass all peaks - np.inf

    # Brute Force FFT:
    'max_number_of_pixels_to_check': 50,

    # Flags:
    'flag_save_interim_graphs_and_movies': False,
    'flag_perform_BGS_running_mean': True,
    'flag_perform_outlier_running_mean': True,
    'flag_use_canny_in_outlier_detection': True,
    'flag_MaxLobeSNR_or_PeakDetection': 'max_lobe_snr',  #'max_lobe_snr', 'peak_detection'
    'flag_calculate_RunningBG_estimation': True,
    'flag_statistics_mode': False,

    # EMVA Camera Report:
    'QE': 0.57,
    'G_DL_to_e_EMVA': 0.468,
    'G_e_to_DL_EMVA': 2.417,
    'readout_noise_electrons': 4.53,
    'full_well_electrons': 7948,
    'N_bits_EMVA': 12,
    'N_bits_final': 8,
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
                                                              string_pattern_to_search='*.Bin',
                                                              flag_sort=False)
    ######################################################################################################################



    ######################################################################################################################
    ### Run over the different experiments and analyze: ###
    for experiment_index in range(len(experiment_filenames_list)):
        #######################################################################################################

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
        params.FPS = float(params.FPS)
        params.FrameRate = float(params.FrameRate)
        params.distance = float(params.distance)
        params.roi = (int(params.ROI[1:params.ROI.find(',')]), int(params.ROI[1+params.ROI.find(','):-1]))
        params.noise_baseline_stop_bin = params.FPS/2 -1
        if 'cooler_base_harmonic' not in params.keys():
            params.cooler_base_harmonic = 23


        ### Get binary file reader: ###
        f = get_binary_file_reader(params)
        #######################################################################################################


        #######################################################################################################
        ### Estimate BG Over Entire Movie (because we can't know exactly at which frames there is a flashlight): ###
        if os.path.isfile(os.path.join(params.results_folder, 'Mov_BG.npy')) == False:
            Movie_BG_numpy, (q1, q2) = get_BG_over_entire_movie(f, params)
        else:
            Movie_BG_numpy = np.load(os.path.join(params.results_folder, 'Mov_BG.npy'), allow_pickle=True)
            (q1, q2) = np.load(os.path.join(params.results_folder, 'quantiles.npy'), allow_pickle=True)
            if Movie_BG_numpy.max() < 2:
                Movie_BG_numpy = Movie_BG_numpy * (q2-q1) + q1
        Movie_BG = torch_get_4D(torch.Tensor(Movie_BG_numpy), 'HW')
        save_image_numpy(params.results_folder, 'Movie_BG.png', BW2RGB(scale_array_from_range(Movie_BG_numpy.clip(q1,q2),(q1,q2), (0,255))).astype(np.uint8), flag_scale=False)
        (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(Movie_BG)
        params.H = H
        params.W = W
        ### Scale BG: ###  #TODO: i now only work with unscaled inputs for consistency sake, understand whether i still need this.
        Movie_BG = scale_array_from_range(Movie_BG.clip(q1, q2),
                                              min_max_values_to_clip=(q1, q2),
                                              min_max_values_to_scale_to=(0, 1)).cuda()


        ### Save entire movie of entire experiment for viewing purposes: ###
        flag_original_movie_exists = os.path.isfile(os.path.join(results_folder, 'Original_Movie.avi')) == False and os.path.isfile(os.path.join(results_folder, 'original_movie.avi')) == False
        if flag_original_movie_exists and params.flag_save_interim_graphs_and_movies:
            long_movie_name = os.path.join(results_folder, 'Original_Movie.avi')
            f = initialize_binary_file_reader(f)
            read_long_movie_and_save_to_avi(f, long_movie_name, 25, params)
        ###
        #######################################################################################################


        #######################################################################################################
        ### Find Flashlight In Movie & Get Sequence Occurense + Flashlight Polygon Points: ###
        if params.flag_statistics_mode:
            if params.was_there_flashlight == 'True':
                ### If There Is Flashlight & It's Statistics Mode --> Either Get Existing Flashlight Information Or Search For It In Movie: ###
                flashlight_results_list_filename = os.path.join(params.results_folder, 'Flashlight', 'flag_flashlight_found_list.npy')
                flag_flaghlight_results_list_exists = os.path.isfile(flashlight_results_list_filename)
                if flag_flaghlight_results_list_exists == False:
                    ### If File Doesn't Already Exist --> Search For It In Movie (later on will delete this and only use manual tagging maybe): ###
                    flag_flashlight_found_list, polygon_points_list, flashlight_BB_list, robust_flashlight_polygon_points = \
                        Find_Thermal_Flashlight_In_Movie(f, Movie_BG, params)
                else:
                    ### If File Does Exist --> Load It: ###
                    flag_flashlight_found_list = np.load(flashlight_results_list_filename, allow_pickle=True)
                    flashlight_BB_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'flashlight_BB_list.npy'), allow_pickle=True)

                ### Get Initial Flashlight Polygon and Location (BB), in case movie doesn't start with flashlight but i still want to compare with something: ###
                current_flashlight_BB = flashlight_BB_list[0]
                current_flashlight_BB = keep_only_valid_flashlight_BB_list_in_sequence(current_flashlight_BB).tolist()
                current_polygon_center = get_BB_center_from_vertices(current_flashlight_BB)

            else:
                flag_flashlight_found_list = np.array([False]*1000)
                flashlight_BB_list = np.array([None]*1000)
                current_flashlight_BB = []
        else:
            ## When I Don't Have Something Tagged And I Simply Want It To Run: ###
            flag_flashlight_found_list = np.array([False]*1000)
            flashlight_BB_list = np.array([None]*1000)
            current_flashlight_BB = []
        #######################################################################################################

        #######################################################################################################
        ### Get User Defined Outlier Logical Mask (areas in the image to avoid): ###
        user_defined_logical_mask_BB_file = os.path.join(params.results_folder, 'BB_regions_to_disregard.npy')
        flag_user_outlier_logical_mask_exists = os.path.isfile(user_defined_logical_mask_BB_file)
        if flag_user_outlier_logical_mask_exists:
            user_defined_BB_list = np.load(user_defined_logical_mask_BB_file, allow_pickle=True)
        else:
            user_defined_BB_list = []
        params.user_defined_outlier_logical_mask = get_logical_mask_from_BB_list(Movie_BG, user_defined_BB_list)
        #######################################################################################################

        #######################################################################################################
        ### Initialize video with information (flashlight, drone, etc') on it of the entire experiment: ###
        if params.flag_save_interim_graphs_and_movies:
            # fourcc = cv2.VideoWriter_fourcc(*'MP42')
            informative_movie_full_file_name = os.path.join(params.results_folder, 'Full_Informative_Movie.avi')
            all_trajectories_full_file_name = os.path.join(params.results_folder, 'All_Trajectories_Movie.avi')
            informative_movie_video_writer = cv2.VideoWriter(informative_movie_full_file_name, 0, 50, (W, H))
            all_trajectories_movie_video_writer = cv2.VideoWriter(all_trajectories_full_file_name, 0, 50, (W, H))
        #######################################################################################################


        #######################################################################################################
        ### ### Go Over Sequences Of Current Experiment: ### ###
        ### ### ### ### ### ### ### ### ### ### ### ###  ### ###

        ### Initialize params Stuff: ###
        params.outlier_sequence_counter_map = 0
        params.Movie_BG_previous = None

        ### Loop over the different sequences and analyze: ###
        params.global_counter = 0
        flag_drone_trajectory_inside_BB_list = []
        number_of_frames_per_sequence = int(params['FPS'] * params['SeqT'])
        f = initialize_binary_file_reader(f)
        #(1). start from different sequence from start sequence:
        flag_enough_frames_to_analyze = True
        sequence_index = 0  #TODO: if i wanna start from another sequence
        temp = read_frames_from_binary_file_stream(f, 0, number_of_frames_per_sequence * sequence_index, params)
        #(2). Re-initialize sequence index:
        sequence_index -= 1

        ### Actually Loop Over Sequences: ###
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
                #####################################################################################################################################################
                # ### Stretch movie range according to previously found quantiles: ###
                # Movie = scale_array_from_range(Movie.clip(q1, q2),
                #                                        min_max_values_to_clip=(q1, q2),
                #                                        min_max_values_to_scale_to=(0, 1))

                ### Initialize Numpy Frames For Movie Writter: ###
                Movie = Movie.astype(float)
                current_frames_for_video_writer = np.copy(Movie)  # create copy to be used for informative movie to disk
                current_frames_for_video_writer = torch_to_numpy_video_ready(torch_get_4D(torch.Tensor(current_frames_for_video_writer), 'THW'))
                Movie = torch.Tensor(Movie).cuda().unsqueeze(1)
                #####################################################################################################################################################


                #####################################################################################################################################################
                ### Initialize BG If At First Stage (global_counter==0): ###
                if params.global_counter == 0:
                    ### Initialize outlier_sequence_counter_map: ###
                    Movie_BG, Movie_BG_std_torch, params = Initialize_BG_Stats(Movie[0:20], params)

                    ### Maybe Load From Memory Or Save: ###
                    running_BG_estimation_filename = os.path.join(params.results_folder_seq, 'Mov_BG.npy')
                    if os.path.isfile(running_BG_estimation_filename):
                        Movie_BG = torch.tensor(np.load(running_BG_estimation_filename, allow_pickle=True)).unsqueeze(0).unsqueeze(0).cuda().float()
                        params.Movie_BG_previous = Movie_BG
                    else:
                        # ### If file doesn't exists the save it to disk, maybe after replacing a patch with a stationary dron - FOR CASES WHERE THE DRONE IS STATIC AT THE BEGINNING!!!!: ###
                        1
                        # ## Switch BG Patch If Wanted: ###
                        # ROI_size = 4
                        # # drone_point = (279, 251)
                        # # drone_point = (206, 196)
                        # # drone_point = (238, 380)
                        # drone_point = (212, 218)
                        # shift_pixels_XY = (5, 0)
                        # new_point = (drone_point[0] + shift_pixels_XY[1], drone_point[1] + shift_pixels_XY[0])
                        # Movie_BG[:, :, drone_point[0] - ROI_size:drone_point[0] + ROI_size, drone_point[1] - ROI_size:drone_point[1] + ROI_size] =\
                        #     Movie_BG[:, :, new_point[0] - ROI_size:new_point[0] + ROI_size, new_point[1] - ROI_size:new_point[1] + ROI_size]

                        ### Save BG Estimation: ###
                        #(*). notice this is within the condition of global_counter==0, so only works for the first batch, which is what i want!!!
                        np.save(running_BG_estimation_filename, Movie_BG.squeeze().cpu().numpy(), allow_pickle=True)
                        # np.save(running_BG_STD_estimation_filename, Movie_BG_std_torch_previous.squeeze().cpu().numpy(), allow_pickle=True)


                    ### Uptick global_counter: ###
                    params.global_counter += 1

                else:
                    #(*). After the first batch now start the lag of 1 batch in BG and update BG estimation AFTER each batch
                    Movie_BG = params.Movie_BG_previous
                #####################################################################################################################################################

                #####################################################################################################################################################
                ### Save single image and Temporal Average of current batch: ###
                if os.path.isfile(os.path.join(results_folder_seq, 'temporal_average.png')):
                    cv2.imwrite(os.path.join(results_folder_seq, 'temporal_average.png'), (Movie * 255).mean(0).clip(0, 255))
                    cv2.imwrite(os.path.join(results_folder_seq, 'single_image.png'), (Movie[0] * 255).clip(0, 255))
                #####################################################################################################################################################


                #####################################################################################################################################################
                ### Check if flashlight found, if not -> search for drone: ###
                if flag_flashlight_found_list[sequence_index]:
                    # (*). if there's a flashlight, don't try and find a drone, simply use the flashlight polygon points as future GT bounding box:
                    print('seq: ' + str(sequence_index) + ', Flashlight ON')

                    ### Write results down to txt file: ###
                    results_summary_string += 'Sequence ' + str(sequence_index) + ', '
                    results_summary_string += 'Flashlight On, No Drone Search Algorithm Done'
                    results_summary_string += '\n'
                    open(os.path.join(results_folder, 'res_summ.txt'), 'a').write(results_summary_string)

                    ### Get Flashlight Polygon and Location (BB): ###
                    current_flashlight_BB = flashlight_BB_list[sequence_index]
                    current_flashlight_BB = keep_only_valid_flashlight_BB_list_in_sequence(current_flashlight_BB).tolist()
                    current_flashlight_BB_centers = get_BB_center_from_vertices(current_flashlight_BB)

                    ### Plot Polygon and Flashlight On Large Movie: ###
                    if params.flag_save_interim_graphs_and_movies:
                        for inter_frame_index in np.arange(current_frames_for_video_writer.shape[0]):
                            current_movie_frame = current_frames_for_video_writer[inter_frame_index]
                            current_movie_frame = draw_polygons_on_image(current_movie_frame, current_flashlight_BB)
                            current_movie_string = 'Sequence ' + str(sequence_index) + ',    '
                            current_movie_string += 'Flashlight On        '
                            current_movie_string += 'No Drone Search Algorithm Done        '
                            current_movie_frame = draw_text_on_image(current_movie_frame, text_for_image=current_movie_string, fontScale=0.3, thickness=1)
                            current_frames_for_video_writer[inter_frame_index] = current_movie_frame
                            informative_movie_video_writer.write(current_movie_frame)
                            all_trajectories_movie_video_writer.write(current_movie_frame)


                elif flag_flashlight_found_list[sequence_index] == False:
                    print('seq: ' + str(sequence_index) + ', No Flashlight, Searching For Drone')
                    # (*). if there is NO flashlight, search for a drone:

                    ##################################################################################
                    ### Maor Outlier Trajectory Finding Method: ###
                    tic()
                    Movie_BGS, Movie_BGS_std, \
                    trajectory_tuple, trajectory_tuple_BeforeFFTDec,\
                    t_vec, res_points, NonLinePoints, xyz_line, xyz_line_after_FFT, number_of_RANSAC_valid_trajectories, number_of_FFT_valid_trajectories, \
                    TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec_per_trajectory, TrjMov, \
                    DetectionDec, DetectionConfLvl, params = Dudy_Analysis_torch(Movie, Movie_BG, params)
                    toc('   Maor Analysis')
                    ### Unpack trajectory_tuple: ###
                    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y = trajectory_tuple


                    ### Turn Tensors To Numpy Arrays because we're dealing with plotting which is numpy anyway: ###
                    #TODO: i should probably turn this whole thing upside down into a tracking system. i should use a small sequence size of about ~100 frames,
                    # i should keep a list of trajectories, and whenever a trajectory is long enough in time i should continuously check whether it's a drone,
                    # and that way i can also keep adding more and more information about each suspect.
                    # but that way it's simply a tracking system which RAFAEL can handle easily... since they won't cooperate i need a video/radar tracking expert
                    res_points, NonLinePoints, xyz_line, frequency_vec_per_trajectory, \
                    Movie, Movie_BGS, Movie_BGS_std, \
                    trajectory_tuple_BeforeFFTDec, trajectory_tuple, TrjMovie_FFT_BinPartitioned_AfterScoreFunction = \
                        tuple_to_numpy((res_points, NonLinePoints, xyz_line, frequency_vec_per_trajectory,
                                        Movie, Movie_BGS, Movie_BGS_std,
                                        trajectory_tuple_BeforeFFTDec, trajectory_tuple, TrjMovie_FFT_BinPartitioned_AfterScoreFunction))
                    Movie_BGS = Movie_BGS.squeeze(1)
                    Movie_BGS_std = Movie_BGS_std.squeeze(1)
                    torch.cuda.empty_cache()
                    ##################################################################################

                    ##################################################################################
                    ### Plot Results: ###
                    if params.flag_save_interim_graphs_and_movies:
                        # (1). Plot 3D CLoud and Outline Trajectories (of ALL trajectories found by RANSAC):
                        Plot_3D_PointCloud_With_Trajectories_Demonstration(res_points, NonLinePoints, xyz_line, number_of_RANSAC_valid_trajectories, params,
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
                        Plot_FFT_Bins_Detection_SubPlots(number_of_FFT_valid_trajectories,
                                                         TrjMovie_FFT_BinPartitioned_AfterScoreFunction,
                                                         frequency_vec_per_trajectory, params, results_folder_seq, TrjMov, DetectionDec, DetectionConfLvl)
                        toc('   show and save FFT bins plots`')
                    ##################################################################################

                    ##################################################################################
                    ### Automatic Location Labeling - was the drone that was found inside the "GT" label, as defined by the flashlight polygon:
                    # (*). Check if trajectory found is within flashlight polygon:
                    if current_flashlight_BB is not None and current_flashlight_BB != []:
                        ### With Flashlight Information - Gather Statistics Mode: ###
                        flag_drone_trajectory_inside_BB_list, current_frames_for_video_writer = \
                            check_for_each_trajectory_if_inside_flashlight_polygons(Movie,
                                                                                   current_frames_for_video_writer,
                                                                                   trajectory_tuple,
                                                                                   current_flashlight_BB)
                    else:
                        ### Without Flashlight Information - Test Mode: ###
                        flag_drone_trajectory_inside_BB_list = [False] * number_of_FFT_valid_trajectories
                    ##################################################################################

                    ##################################################################################
                    ### Loop over all frames intended for video writer (with all the information drawn on them) and write the video file: ###
                    if params.flag_save_interim_graphs_and_movies:
                        ### Draw Drone/Suspect Trajectories On Movie: ###
                        current_frames_for_video_writer_2 = draw_trajectories_on_images(copy.deepcopy(current_frames_for_video_writer), trajectory_tuple_BeforeFFTDec)
                        current_frames_for_video_writer = draw_trajectories_on_images(current_frames_for_video_writer, trajectory_tuple)
                        tic()
                        full_t_vec = np.arange(Movie.shape[0])
                        for inter_frame_index in full_t_vec:
                            current_movie_string = 'Sequence ' + str(sequence_index) + ',    '
                            current_movie_string += 'Flashlight OFF        '
                            current_movie_string += 'Preseting Drone Trajectories'
                            #(1). Movie After Frequency Decision:
                            current_movie_frame = current_frames_for_video_writer[inter_frame_index]
                            current_movie_frame = draw_text_on_image(current_movie_frame, text_for_image=current_movie_string, fontScale=0.3, thickness=1)
                            current_movie_frame = draw_polygons_on_image(current_movie_frame, current_flashlight_BB)
                            informative_movie_video_writer.write(current_movie_frame)
                            #(2). Movie All Trajectories:
                            current_movie_frame_2 = current_frames_for_video_writer_2[inter_frame_index]
                            current_movie_frame_2 = draw_text_on_image(current_movie_frame_2, text_for_image=current_movie_string, fontScale=0.3, thickness=1)
                            current_movie_frame_2 = draw_polygons_on_image(current_movie_frame_2, current_flashlight_BB)
                            all_trajectories_movie_video_writer.write(current_movie_frame_2)
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
                    if number_of_FFT_valid_trajectories > 0:
                        for trajectory_index in np.arange(number_of_FFT_valid_trajectories):
                            current_drone_detection_confidence = DetectionConfLvl[trajectory_index]

                            ### Write results down to txt file: ###
                            results_summary_string += 'Sequence ' + str(sequence_index) + ': \n'
                            results_summary_string += '             Trajectory ' + str(trajectory_index) + \
                                                      ', Confidence: ' + str(current_drone_detection_confidence)

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
        all_trajectories_movie_video_writer.release()


    # ### Plot Ed-Hok Pd: ###
    # x_vec = [9,6,4.5,3,2.5,1.5]
    # y_vec = [1,1,1,1,1,28/30]
    # x_vec_tensor = torch.tensor(x_vec)
    # y_vec_tensor = torch.tensor(y_vec)
    # # x_vec_tensor = nn.Upsample(scale_factor=8, mode='linear')(x_vec_tensor.unsqueeze(0).unsqueeze(0)).squeeze()
    # # y_vec_tensor = nn.Upsample(scale_factor=8, mode='linear')(y_vec_tensor.unsqueeze(0).unsqueeze(0)).squeeze()
    # plt.scatter(x_vec_tensor.cpu().numpy(), 100*y_vec_tensor.cpu().numpy())
    # plt.plot(x_vec_tensor.cpu().numpy(), 100*y_vec_tensor.cpu().numpy(), 'g')
    # plt.xlim([10,1])
    # plt.ylim([0,110])
    # plt.title('Pd (Drone Size)')
    # plt.xlabel('Drone Size [pixels]')
    # plt.ylabel('Pd[%]')

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


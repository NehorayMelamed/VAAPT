# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 09:20:06 2022

@author: Lenovo
"""
from venv import create

import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
# from QS_alg_flow_main.ShowVid import Plot_BoundingBoxes_On_Video
from skimage.measure import LineModelND, ransac
from sklearn.cluster import KMeans
import scipy
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.signal import medfilt
from scipy import signal
from easydict import EasyDict
# from NEW_TASHTIT.import_Tashtit import *

from QS_Jetson_mvp_milestone.functional_utils.functional_utils import *

params = {
    # General params
    'TrjNumFind': 8,  # num of trj. to find. (max number of trajectories to find within frame)
    'TrjNumFind_Flashlight': 2,
    'ROI_allocated_around_suspect': 11,
    # The frame size of the sus. close up. (after finding trajectory cut this size ROI around suspect)
    'SeqT': 1,  # Seq. time in sec.
    'SaveResFlg': 1,
    'FrameRate': 500,
    'roi': [320, 640],
    'utype': np.uint16,
    'DistFold': "/home/mafat/PycharmProjects/QS_alg_flow_main",  # where the files are at
    'ExperimentsFold': '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments',
    # where all the experiments are at when you want to process many videos together
    'ResultsFold': '/home/mafat/PycharmProjects/QS_alg_flow_main/Res',  # where to put all the results
    # 'ExpName': "mavic2_1p5km_static_urban",  #specific experiment if wanted

    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/mavic2_1p5km_static_urban/mavic2_1p5km_static_urban.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_static_3000frames_500fps_drone_at244_90_w640_h320/1200m_matrice300_mavic2_rural_static_3000frames_500fps_drone_at244_90_w640_h320.Bin",
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/boris_with_fleshlight_integration_time_800/boris_with_fleshlight_integration_time_800.Bin",
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/boris_with_fleshlight_integration_time_200/boris_with_fleshlight_integration_time_200.Bin",
    'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/boris_with_fleshlight_integration_time_50/boris_with_fleshlight_integration_time_50.Bin",
    # specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_cloaser_further_3000frames_500fps_drone_at324_65_w640_h320/1200m_matrice300_mavic2_rural_cloaser_further_3000frames_500fps_drone_at324_65_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_up+down_3000frames_500fps_drone_at324_65_w640_h320/1200m_matrice300_mavic2_rural_up+down_3000frames_500fps_drone_at324_65_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320/1200m_matrice300_mavic2_rural_left+right_3000frames_500fps_drone_at308_91_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/3000m_matrice300_mavic2_rural_left_right_3000frames_500fps_w640_h320/3000m_matrice300_mavic2_rural_left_right_3000frames_500fps_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/3000m_matrice300_mavic2_rural_static_3000frames_500fps_w640_h320/3000m_matrice300_mavic2_rural_static_3000frames_500fps_w640_h320.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/flash_600m_640_160_1000fps_14bit_frames5000/flash_600m_640_160_1000fps_14bit_frames5000.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/flash_test_640_160_1000fps_14bit_frames5000/flash_test_640_160_1000fps_14bit_frames5000.Bin",  #specific filename - the movie comes in the form of .bin files
    # 'FileName': "/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/flash_test_640_320_500fps_14bit_3822frames/flash_test_640_320_500fps_14bit_3822frames.Bin",  #specific filename - the movie comes in the form of .bin files
    'RunOverAllFilesFlg': 0,

    'number_of_frames_to_read': 2500,
    'frame_to_start_from': 3000,

    # BGS params
    'BGSSigma': 2.5,  # BGS alg. threshold

    # Line est. params
    'DroneLineEst_polyfit_degree': 2,  # line est. fit: polynum deg.
    'DroneLineEst_RANSAC_D': 6,  # Ransac diameter: cylinder in which pnts are est. as a line
    'DroneLineEst_RANSAC_max_trials': 100,

    # Trj. decision params
    'TrjDecMinNumOfPnts': .15,  # min percentage of points out of the point cloud which are under D
    'TrjDecMaxVel': .5,  # max. vel. for sus. units: units: np.tan(np.arcsin(TrjDecMaxVel)) is num. of pxls per frame
    'TrjDecMinVel': .005,  # TODO: perhapse change this to maximum change in pixels per frame or something
    'TrjDecMinT': .5,  # min percentage of frames in which suspect is visible
    'TrjDecMaxDisappear': 100,  # max. time of sus. disappear in pnt cloud

    # Trj. detection params
    'DetDFrq': 10,  # for frq. comb binning
    'DetFrqInit': 90,  # for frq. comb binning
    'DetFrqStop': 250,  # for frq. comb binning
    'DetScrThresh': 1,  # min. score for sus.
    'DetScrThresh100Conf': 5,  # sus. score for 100% confidence
    'DetNoiseThresh': 2.5,
    # For FFT pxl of certain frq. range to be considered
    # it should be larger then the noise lvl. times this number

    # Frequency Analysis Parameters:
    'frequency_range_for_drone': 1,
    # the peak size in Hz allocated for a drone (instead of just choosing the max value))
    'drone_start_bin': 90,
    'drone_stop_bin': 120,
    'noise_baseline_start_bin': 150,
    'noise_baseline_stop_bin': 220,

    # Brute Force FFT:
    'max_number_of_pixels_to_check': 50,
}


def QS_flow_run(params):
    ### Get a list of the files in the current directory: ###
    main_folder = params['DistFold']
    list_of_files_in_path = os.listdir(main_folder)

    ### get files to run over list (which can be made up of several experiments within a folder or a single FileName): ###
    if params['RunOverAllFilesFlg']:
        experiment_filenames_list = os.listdir(
            params['ExperimentsFold'])  # all the filenames must be in the form of a .bin file currently
    else:
        experiment_filenames_list = [params['FileName']]

    ### Run over the different experiments and analyze: ###
    for experiment_index in range(len(experiment_filenames_list)):
        ### Make sure filename is a .bin file: ###
        current_experiment_filename = experiment_filenames_list[experiment_index]
        if not current_experiment_filename[-3:] == 'Bin':
            continue
        params['FileName'] = experiment_filenames_list[experiment_index]

        ### Get parameters from params dict: ###
        FileName = params['FileName']
        filename_itself = os.path.split(FileName)[-1]
        experiment_folder = os.path.split(FileName)[0]
        results_folder = os.path.join(experiment_folder, 'Results')
        create_folder_if_doesnt_exist(results_folder)

        ### initialize a .txt file with a summation of the results: ###
        if os.path.isfile(os.path.join(results_folder, 'res_summ.txt')):
            res_summ_txt = open(os.path.join(results_folder, 'res_summ.txt'), 'r').read()
        else:
            res_summ_txt = ''

        ### Read full movie to analyze: ###
        Mov, Res_dir = read_movie_sample(params)
        Mov = Mov.astype(np.float)

        ### TODO: make sure to stick it in the correct place         the pipeline: ###
        Movie = torch.Tensor(Mov).unsqueeze(1)[:, :, 2:-1, 2:-1]  # transform to [T,C,H,W]
        Movie = scale_array_stretch_hist(Movie)
        # imshow_torch_video(Movie, 2500, 50, frame_stride=3)
        Find_Thermal_Flashlight(Movie, params)

        ### Save Entire Movie: ###
        if os.path.isfile(os.path.join(results_folder, 'Original_Movie.avi')) == False:
            Stretched_Movie, params = PreProcess_Movie(Mov, params)
            numpy_array_to_video(BW2RGB(np.transpose(np.expand_dims(Stretched_Movie, 0), (1, 2, 3, 0))),
                                 os.path.join(results_folder, 'Original_Movie.avi'), 25.0)

        ### Get relevant variables from movie: ###
        number_of_frames_per_sequence = np.int(params['FPS'] * params['SeqT'])
        total_number_of_sequences = np.int(np.floor(Mov.shape[0] / number_of_frames_per_sequence))

        ### Loop over the different sequences and analyze: ###
        for sequence_index in range(total_number_of_sequences):
            ### Make results path for current sequence: ###
            Res_dir_seq = os.path.join(results_folder, "seq" + str(sequence_index))
            print(params['FileName'] + ": seq" + str(sequence_index))
            create_folder_if_doesnt_exist(Res_dir_seq)

            ### Get relevant frames from entire movie: ###
            frame_start_index = sequence_index * number_of_frames_per_sequence
            frame_stop_index = (sequence_index + 1) * number_of_frames_per_sequence
            Movie = Mov[frame_start_index:frame_stop_index, :, :]

            ### get rid of bad outer frame, and normalize: ###
            Movie, params = PreProcess_Movie(Movie, params)

            ### Save single image and Temporal Average of current batch: ###
            cv2.imwrite(os.path.join(Res_dir_seq, 'temporal_average.png'), (Movie * 255).mean(0).clip(0, 255))
            cv2.imwrite(os.path.join(Res_dir_seq, 'single_image.png'), (Movie[0] * 255).clip(0, 255))

            # ### TODO: temp: ###
            # Movie_aligned = get_image_translation_and_translate_batch(Movie)
            # numpy_array_to_video(BW2RGB(np.transpose(np.expand_dims(Movie_aligned, 0), (1, 2, 3, 0))),
            #                      os.path.join(results_folder, 'Aligned_Movie_Boris.avi'), 25.0)

            ### Align Movie: ###
            # Movie = Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(Movie)
            # Movie = Ctypi_align(Movie)
            # numpy_array_to_video(BW2RGB(np.transpose(np.expand_dims(Movie, 0), (1, 2, 3, 0))),
            #                      os.path.join(results_folder, 'Aligned_Movie.avi'), 25.0)

            # ### FFT Analysis: ###
            # Total_FFT_Analysis(Movie, 500)

            # ### Brute Force FFT: ###
            # Brute_Force_FFT(Movie, params)

            ### Estimate and substract background: ###
            BG_estimation, Movie_BGS, Movie_BGS_std = BG_estimation_and_substraction_on_aligned_movie(
                Movie)

            ### Estimate Noise From BG substracted movie (noise estimate per pixel over time): ### #TODO: right now this estimates one noise dc and std for the entire image! return estimation per pixel
            noise_std, noise_dc = NoiseEst(Movie_BGS, params)

            ### use quantile filter to get pixels which are sufficiently above noise for more then a certain quantile of slots in time: ###
            events_points_TXY_indices = Get_Outliers_Above_BG_1(Movie_BGS, Movie_BGS_std, params)

            ### Find trajectories using RANSAC and certain heuristics: ###
            res_points, NonLinePoints, direction_vec, holding_point, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, TrjMov, num_of_trj_found = \
                Find_Valid_Trajectories_RANSAC_And_Align(events_points_TXY_indices, Movie_BGS, params)

            ### Test whether the trajectory frequency content agrees with the heuristics and is a drone candidate: ###
            DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
                Frequency_Analysis_And_Detection(TrjMov, noise_std, noise_dc, params)

            ### Plot 3D CLoud and Outline Trajectories: ###
            Plot_3D_PointCloud_With_Trajectories(res_points, NonLinePoints, xyz_line, num_of_trj_found, params,
                                                 "Plt3DPntCloudTrj", Res_dir_seq)

            ### Plot The Detection Over different frequency bands: ###
            Plot_FFT_Bins_Detection_SubPlots(num_of_trj_found, TrjMovie_FFT_BinPartitioned_AfterScoreFunction,
                                             frequency_vec, params, Res_dir_seq, TrjMov, DetectionDec, DetectionConfLvl)

            ### Show Raw and BG_Substracted_Normalized Videos with proper bounding boxes: ###
            trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)
            BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(Movie, fps=50, tit='Mov',
                                                                            Res_dir=Res_dir_seq, flag_save_movie=1,
                                                                            trajectory_tuple=trajectory_tuple)
            Plot_BoundingBox_On_Movie(Movie, fps=50, tit="Mov", Res_dir=Res_dir_seq, flag_save_movie=1,
                                      trajectory_tuple=trajectory_tuple,
                                      BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)
            Plot_BoundingBox_On_Movie(Movie_BGS / Movie_BGS_std, fps=50, tit="Movie_BGS",
                                      Res_dir=Res_dir_seq, flag_save_movie=1,
                                      trajectory_tuple=trajectory_tuple,
                                      BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)

            res_summ_txt = res_summ_txt + params['FileName'][:-4] + ": seq" + str(sequence_index) + '. ' + (
                'drone has found, conf = ' + str(np.int(max(DetectionConfLvl))) + '%' if np.any(
                    DetectionDec) else 'drone has not found') + '\n'
            open(results_folder + '\\res_summ.txt', 'w').write(res_summ_txt)


def QS_flow_run_Torch(params):
    ### Get a list of the files in the current directory: ###
    main_folder = params['DistFold']
    list_of_files_in_path = os.listdir(main_folder)

    ### create results folder for current experiment if none exists: ###
    results_folder = params['ResultsFold']
    create_folder_if_doesnt_exist(params['ResultsFold'])

    ### get files to run over list (which can be made up of several experiments within a folder or a single FileName): ###
    if params['RunOverAllFilesFlg']:
        experiment_filenames_list = os.listdir(
            params['ExperimentsFold'])  # all the filenames must be in the form of a .bin file currently
    else:
        experiment_filenames_list = [params['FileName']]

    ### initialize a .txt file with a summation of the results: ###
    if os.path.isfile(os.path.join(results_folder, 'res_summ.txt')):
        res_summ_txt = open(os.path.join(results_folder, 'res_summ.txt'), 'r').read()
    else:
        res_summ_txt = ''

    ### Run over the different experiments and analyze: ###
    for experiment_index in range(len(experiment_filenames_list)):
        ### Make sure filename is a .bin file: ###
        current_experiment_filename = experiment_filenames_list[experiment_index]
        if not current_experiment_filename[-3:] == 'Bin':
            continue
        params['FileName'] = experiment_filenames_list[experiment_index]

        ### Read full movie to analyze: ###
        Mov, Res_dir = read_movie_sample(params)
        Mov = Mov.astype(np.float)
        Mov = torch.Tensor(Mov)

        ### Get relevant variables from movie: ###
        number_of_frames_per_sequence = np.int(params['FPS'] * params['SeqT'])
        total_number_of_sequences = np.int(np.floor(Mov.shape[0] / number_of_frames_per_sequence))

        ### Loop over the different sequences and analyze: ###
        for sequence_index in range(total_number_of_sequences):
            ### Make results path for current sequence: ###
            Res_dir_seq = os.path.join(Res_dir, "seq" + str(sequence_index))
            print(params['FileName'] + ": seq" + str(sequence_index))
            create_folder_if_doesnt_exist(Res_dir_seq)

            ### Get relevant frames from entire movie: ###
            frame_start_index = sequence_index * number_of_frames_per_sequence
            frame_stop_index = (sequence_index + 1) * number_of_frames_per_sequence
            Movie = Mov[frame_start_index:frame_stop_index, :, :]

            ### get rid of bad outer frame, and normalize: ###
            Movie, params = PreProcess_Movie_Torch(Movie, params)

            ### Align Movie: ###
            Movie = Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(Movie,
                                                                                    reference_tensor=None)

            ### Estimate and substract background: ###
            BG_estimation, Movie_BGS, Movie_BGS_std = BG_estimation_and_substraction_on_aligned_movie_torch(
                Movie)

            ### Estimate Noise From BG substracted movie (noise estimate per pixel over time): ### #TODO: right now this estimates one noise dc and std for the entire image! return estimation per pixel
            noise_std, noise_dc = Noise_Estimation_FourierSpace_PerPixel_Torch(Movie_BGS, params)

            ### use quantile filter to get pixels which are sufficiently above noise for more then a certain quantile of slots in time: ###
            events_points_TXY_indices = Get_Outliers_Above_BG_1_Torch(Movie_BGS, Movie_BGS_std, params)

            ### Find trajectories using RANSAC and certain heuristics: ###
            res_points, NonLinePoints, direction_vec, holding_point, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, TrjMov, num_of_trj_found = \
                Find_Valid_Trajectories_RANSAC_And_Align(events_points_TXY_indices, Movie_BGS, params)

            ### Test whether the trajectory frequency content agrees with the heuristics and is a drone candidate: ###
            DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
                Frequency_Analysis_And_Detection(TrjMov, noise_std, noise_dc, params)

            ### Plot 3D CLoud and Outline Trajectories: ###
            Plot_3D_PointCloud_With_Trajectories(res_points, NonLinePoints, xyz_line, num_of_trj_found, params,
                                                 "Plt3DPntCloudTrj", Res_dir_seq)

            ### Plot The Detection Over different frequency bands: ###
            Plot_FFT_Bins_Detection_SubPlots(num_of_trj_found, TrjMovie_FFT_BinPartitioned_AfterScoreFunction,
                                             frequency_vec, params,
                                             Res_dir_seq, TrjMov, DetectionDec, DetectionConfLvl)

            ### Show Raw and BG_Substracted_Normalized Videos with proper bounding boxes: ###
            Plot_BoundingBoxes_On_Video(Movie, fps=50, tit="Mov", flag_transpose_movie=False,
                                        frame_skip_steps=3, resize_factor=2, flag_save_movie=1,
                                        Res_dir=Res_dir_seq,
                                        histogram_limits_to_stretch=0.01,
                                        trajectory_tuple=(
                                            t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y))
            Plot_BoundingBoxes_On_Video(Movie_BGS / Movie_BGS_std, fps=50, tit="Movie_BGS",
                                        flag_transpose_movie=False, frame_skip_steps=3, resize_factor=2,
                                        flag_save_movie=1,
                                        Res_dir=Res_dir_seq,
                                        histogram_limits_to_stretch=0.01,
                                        trajectory_tuple=(
                                            t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y))
            res_summ_txt = res_summ_txt + params['FileName'][:-4] + ": seq" + str(sequence_index) + '. ' + \
                           ('drone has found, conf = ' + str(np.int(max(DetectionConfLvl))) + '%' if np.any(
                               DetectionDec) else 'drone has not found') + '\n'
            open(results_folder + '\\res_summ.txt', 'w').write(res_summ_txt)


params = EasyDict(params)
QS_flow_run(params)
# QS_flow_run_Torch(params)





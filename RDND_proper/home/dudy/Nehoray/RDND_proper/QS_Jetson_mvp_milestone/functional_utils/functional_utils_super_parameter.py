
import numpy as np
import torch
import torch.nn as nn
import scipy
from scipy import signal
from skimage.measure import LineModelND, ransac
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import cv2

from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from skimage.measure import LineModelND, ransac
from sklearn.cluster import KMeans
import scipy
import os
from matplotlib import pyplot as plt
import cv2
from scipy import signal
import torch.fft

def create_folder_if_doesnt_exist(folder_full_path):
    if not os.path.exists(folder_full_path):
        os.makedirs(folder_full_path)


def read_movie_sample(params):
    ### Get parameters from params dict: ###
    ExpName = params['ExpName']
    FileName = params['FileName']
    DistFold = params['DistFold']
    ResultsFold = params['ResultsFold']
    roi = params['roi']
    utype = params['utype']

    ### make resuls path if none exists: ###
    filename_itself = os.path.split(FileName)[-1]
    Res_dir = os.path.join(ResultsFold, ExpName, filename_itself[0:-4])
    text_full_filename = os.path.join(Res_dir, 'params.txt')
    create_folder_if_doesnt_exist(Res_dir)

    ### Open movie, which is in binary format: ###
    f = open(FileName, "rb")  #TODO: switched from complicated nested stuff to simple full filename

    ### write down current parameters into params.txt format: ###
    open(text_full_filename, 'w+').write(str(params))

    ### Read entire movie from binary file into numpy array: ###
    Mov = np.fromfile(f, dtype=utype)
    Movie_len = np.int(len(Mov)/roi[0]/roi[1])
    Mov = np.reshape(Mov, [Movie_len, roi[0], roi[1]])

    return Mov, Res_dir

def BG_estimation_and_substraction_on_aligned_movie(super_dict):
    ### Get Parameters From super_dict: ###
    Mov = super_dict.Mov

    BG_estimation = np.mean(Mov, 0)
    Movie_BGS = Mov - BG_estimation
    Movie_BGS_std = np.std(Movie_BGS, axis=0)
    Movie_BGS_std = Movie_BGS_std.clip(None, 10)
    return BG_estimation, Movie_BGS, Movie_BGS_std

def BG_estimation_and_substraction_on_aligned_movie_torch(super_dict):
    ### Get Parameters From super_dict: ###
    Mov = super_dict.Mov

    BG_estimation = torch.mean(Mov, 0)
    Movie_BGS = Mov - BG_estimation
    Movie_BGS_std = torch.std(Movie_BGS, dim=0)
    Movie_BGS_std = Movie_BGS_std.clamp(None, 10)
    return BG_estimation, Movie_BGS, Movie_BGS_std

def PreProcess_Movie(input_tensor, params=None):
    ### Stretch & Change Movie For Proper Viewing & Analysis: ###

    ### Get min and max values to stretch image for proper viewing: ###
    if 'min_value' in params.keys():
        min_value = params.min_value
        max_value = params.max_value
    else:
        min_value = input_tensor[0:10].quantile(0.01)
        max_value = input_tensor[0:10].quantile(0.99)
        params.min_value = min_value
        params.max_value = max_value

    ### Actually stretch image: ###
    input_tensor = input_tensor.clamp(min_value, max_value)  #clamp the image first to allow for effective stretching for viewing
    input_tensor = scale_array_to_range(input_tensor, (0, 1))

    ### Cut the edges of the frame which are non valid for our pelican-D: ###
    input_tensor = input_tensor[:, 2:, 2:]

    return input_tensor, params


def NoiseEst(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    params = super_dict.params

    ### Get variables from params dict: ##
    FrameRate = params['FPS']
    DetFrqInit = params['DetFrqInit']
    DetFrqStop = params['DetFrqStop']
    T, H, W = Movie_BGS.shape

    ### Perform FFT per pixel: ###
    frq_ax = FrameRate * np.linspace(-0.5, 0.5 - 1 / T, T)
    MovFFT = np.fft.fftshift(np.fft.fft(Movie_BGS, axis=0), axes=0)

    ### Estimate noise between relevant frequency bins on the entire image: ###
    # TODO: this assumes each pixel has the same dynamics, use robust noise statistics
    # TODO: return noise per pixel instead of one noise figure
    noise_std = 0.5 * np.std(MovFFT[(frq_ax > DetFrqInit) & (frq_ax < DetFrqStop)])
    noise_dc = 0
    return noise_std, noise_dc

def Get_Outliers_Above_BG_1(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    Movie_BGS_std = super_dict.Movie_BGS_std
    params = super_dict.params

    Movie_BGS_over_std = Movie_BGS / (Movie_BGS_std + 1e-6)
    Movie_outliers = np.abs(Movie_BGS_over_std) > params['BGSSigma']
    Movie_outliers = Movie_outliers.astype(np.float)

    ### Perform a quantile filter over time per pixel (if the outlier is CONSISTENT, even though it may be small, it's probably good enough): ###
    quantile_filter_window_size = 5
    quantile_filter = np.ones((quantile_filter_window_size,1,1))
    number_of_events_inside_window = 3
    Movie_outliers_QuantileFiltered_LogicalMap = signal.convolve(Movie_outliers, quantile_filter, mode='same') >= number_of_events_inside_window
    movie_current_outliers_TXY_indices = np.vstack(np.where(Movie_outliers_QuantileFiltered_LogicalMap)).T
    return movie_current_outliers_TXY_indices

def Get_Outliers_Above_BG_1_Torch(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    Movie_BGS_std = super_dict.Movie_BGS_std
    params = super_dict.params

    Movie_BGS_over_std = Movie_BGS / (Movie_BGS_std + 1e-6)
    Movie_outliers = torch.abs(Movie_BGS_over_std) > params['BGSSigma']
    Movie_outliers = Movie_outliers.astype(torch.float)

    ### Perform a quantile filter over time per pixel (if the outlier is CONSISTENT, even though it may be small, it's probably good enough): ###
    quantile_filter_window_size = 5
    quantile_filter = torch.ones((1,quantile_filter_window_size,1,1))
    number_of_events_inside_window = 3
    Movie_outliers_QuantileFiltered_LogicalMap = torch_convolve(Movie_outliers, quantile_filter) >= number_of_events_inside_window
    movie_current_outliers_TXY_indices = torch.vstack(torch.where(Movie_outliers_QuantileFiltered_LogicalMap)).T
    return movie_current_outliers_TXY_indices

def get_Distance_From_Points_To_DirectionVec(xyz, b, p):
    #(*). b is a unit vector!
    ### calculates total distance between xyz trajectory (centered by p holding point) and unit vector b: ###
    number_of_points, number_of_dimensions = xyz.shape
    xyz_centered = xyz - p
    xyz_centered_dot_product_with_b = np.dot(xyz_centered, b).reshape(number_of_points, 1)    # SCALAR = project xyz_centered on b, which is a UNIT VECTOR!
    xyz_centered_projection_on_b = np.matmul(xyz_centered_dot_product_with_b, b.T)   # multiply above scalar projection by b vec
    dist = np.sum((xyz_centered_projection_on_b - xyz_centered)**2, 1)**.5  #distance from each point on the trajectory points to the line projection on b
    return dist

def get_Distance_From_Points_To_DirectionVec_Torch(xyz, b, p):
    #(*). b is a unit vector!
    ### calculates total distance between xyz trajectory (centered by p holding point) and unit vector b: ###
    number_of_points, number_of_dimensions = xyz.shape
    xyz_centered = xyz - p
    xyz_centered_dot_product_with_b = torch.dot(xyz_centered, b).reshape(number_of_points, 1)    # SCALAR = project xyz_centered on b, which is a UNIT VECTOR!
    xyz_centered_projection_on_b = torch.matmul(xyz_centered_dot_product_with_b, b.T)   # multiply above scalar projection by b vec
    dist = torch.sum((xyz_centered_projection_on_b - xyz_centered)**2, 1)**.5  #distance from each point on the trajectory points to the line projection on b
    return dist

def Cut_And_Align_ROIs_Around_Trajectory(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    t_vec = super_dict.t_vec
    trajectory_smoothed_polynom_X = super_dict.trajectory_smoothed_polynom_X
    trajectory_smoothed_polynom_Y = super_dict.trajectory_smoothed_polynom_Y
    params = super_dict.params

    ### Get basic parameters: ###
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    H = np.size(Movie_BGS,1)
    W = np.size(Movie_BGS,2)
    number_of_time_steps = len(t_vec)

    ### Allocated large frame grid: ###
    xM = np.arange(H)
    yM = np.arange(W)
    original_grid = (t_vec, xM, yM)

    ### Allocate suspect ROI grid: ###
    xl = np.arange(ROI_allocated_around_suspect) - np.int(ROI_allocated_around_suspect/2)
    xg = np.array(np.meshgrid(t_vec, xl, xl)).T  #just like meshgrid(x,y) creates two matrices X,Y --> this create 3 matrices of three dimensions which are then concatated
    number_of_pixels_per_ROI = ROI_allocated_around_suspect**2

    ### loop over suspect ROI grid and add the trajectory to each point on the grid: ###
    for ii in range(ROI_allocated_around_suspect):
        for jj in range(ROI_allocated_around_suspect):
            xg[ii, :, jj, 2] = xg[ii, :, jj, 2] + trajectory_smoothed_polynom_Y
            xg[ii, :, jj, 1] = xg[ii, :, jj, 1] + trajectory_smoothed_polynom_X
    xg = xg.reshape((number_of_pixels_per_ROI*number_of_time_steps, 3))

    ### "align" ROIs which contain suspect on top of each other by multi-dimensional interpolation): ###
    TrjMov = scipy.interpolate.interpn(original_grid, Movie_BGS[t_vec,:,:], xg, method='linear')  #TODO: maybe nearest neighbor???? to avoid problems
    TrjMov = TrjMov.reshape([ROI_allocated_around_suspect, number_of_time_steps, ROI_allocated_around_suspect]).transpose([1, 0, 2])

    #TODO: there's probably a faster way of doing it: ###
    return TrjMov


def Find_Valid_Trajectories_RANSAC_And_Align(super_dict):
    ### Get Parameters From super_dict: ###
    events_points_TXY_indices = super_dict.events_points_TXY_indices
    Movie_BGS = super_dict.Movie_BGS
    params = super_dict.params

    ### Get variables from params dict: ###
    TrjNumFind = params['TrjNumFind']  # max number of trajectories to look at

    ### Get input movie shape: ###
    H = np.size(Movie_BGS, 1)
    W = np.size(Movie_BGS, 2)
    number_of_frames = np.size(Movie_BGS, 0)

    ### Initialize points_not_assigned_to_lines_yet with all the points: ###
    points_not_assigned_to_lines_yet = events_points_TXY_indices

    ### PreAllocate and initialize lists which hold decision results for each trajectory: ###
    direction_vec = list(np.zeros(TrjNumFind))
    holding_point = list(np.zeros(TrjNumFind))
    t_vec = list(np.zeros(TrjNumFind))
    trajectory_smoothed_polynom_X = list(np.zeros(TrjNumFind))
    trajectory_smoothed_polynom_Y = list(np.zeros(TrjNumFind))
    TrjMov = list(np.zeros(TrjNumFind))
    xyz_line = list(np.zeros(TrjNumFind))
    flag_is_trajectory_valid_vec = np.ones(TrjNumFind) < 0
    flag_enough_valid_samples = np.ones(TrjNumFind) < 0  # TODO: right it as a vector of false flags for god sake
    NonLinePoints = np.zeros([1, 3])

    ### Loop over the max number of trajectories to find (each time we find a line candidate, get rid of the : ###
    for ii in range(TrjNumFind):
        ### if there are less then 10 points just ignore and continue: ###
        if points_not_assigned_to_lines_yet.shape[0] < 10:
            continue

        ### Estimate line using RANSAC: ###
        direction_vec[ii], holding_point[ii], t_vec[ii], \
        trajectory_smoothed_polynom_X[ii], trajectory_smoothed_polynom_Y[ii], \
        points_not_assigned_to_lines_yet, xyz_line[ii], flag_enough_valid_samples[ii] = \
            Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit(points_not_assigned_to_lines_yet, params, H, W)

        ### if the flag "line_out_of_range" is True, meaning there aren't enough valid points after RANSAC, then ignore and continue: ###
        if flag_enough_valid_samples[ii] == False:
            continue

        ### Perform Some Heuristics on trajectory to see if there's enough "meat" to bite into: ###
        flag_is_trajectory_valid_vec[ii] = Decide_If_Trajectory_Valid_Drone(xyz_line[ii], t_vec[ii], direction_vec[ii],
                                                                      number_of_frames, params)
        if ~flag_is_trajectory_valid_vec[ii]:
            TrjMov[ii] = []
            # (*). NonLinePoints are points which RANSAC found to be a part of a line but after some heuristics in the above function i decided to ignore!
            NonLinePoints = np.concatenate((NonLinePoints, xyz_line[ii]))
            continue

        ### "Straighten out" and align frames where suspect is at to allow for proper frequency analysis: ###
        TrjMov[ii] = Cut_And_Align_ROIs_Around_Trajectory(Movie_BGS, t_vec[ii],
                                                          trajectory_smoothed_polynom_X[ii],
                                                          trajectory_smoothed_polynom_Y[ii], params)

    ### Only keep the "valid" trajectories which passed through all the heuristics: ###
    direction_vec = list(np.array(direction_vec)[np.where(flag_is_trajectory_valid_vec)])
    holding_point = list(np.array(holding_point)[np.where(flag_is_trajectory_valid_vec)])
    t_vec = list(np.array(t_vec)[np.where(flag_is_trajectory_valid_vec)])
    trajectory_smoothed_polynom_X = list( np.array(trajectory_smoothed_polynom_X)[np.where(flag_is_trajectory_valid_vec)])
    trajectory_smoothed_polynom_Y = list(np.array(trajectory_smoothed_polynom_Y)[np.where(flag_is_trajectory_valid_vec)])
    xyz_line = list(np.array(xyz_line)[np.where(flag_is_trajectory_valid_vec)])
    TrjMov = list(np.array(TrjMov)[np.where(flag_is_trajectory_valid_vec)])
    num_of_trj_found = np.sum(flag_is_trajectory_valid_vec)

    return points_not_assigned_to_lines_yet, NonLinePoints, direction_vec, holding_point, t_vec, \
           trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, TrjMov, num_of_trj_found


def Find_Valid_Trajectories_RANSAC_And_Align_Torch(super_dict):
    ### Get Parameters From super_dict: ###
    events_points_TXY_indices = super_dict.events_points_TXY_indices
    Movie_BGS = super_dict.Movie_BGS
    params = super_dict.params

    ### Get variables from params dict: ###
    TrjNumFind = params['TrjNumFind']  # max number of trajectories to look at

    ### Get input movie shape: ###
    H = Movie_BGS.shape[1]
    W = Movie_BGS.shape[2]
    number_of_frames = Movie_BGS.shape[0]

    ### Initialize points_not_assigned_to_lines_yet with all the points: ###
    points_not_assigned_to_lines_yet = events_points_TXY_indices

    ### PreAllocate and initialize lists which hold decision results for each trajectory: ###
    direction_vec = list(np.zeros(TrjNumFind))
    holding_point = list(np.zeros(TrjNumFind))
    t_vec = list(np.zeros(TrjNumFind))
    trajectory_smoothed_polynom_X = list(np.zeros(TrjNumFind))
    trajectory_smoothed_polynom_Y = list(np.zeros(TrjNumFind))
    TrjMov = list(np.zeros(TrjNumFind))
    xyz_line = list(np.zeros(TrjNumFind))
    flag_is_trajectory_valid_vec = np.ones(TrjNumFind) < 0
    flag_enough_valid_samples = np.ones(TrjNumFind) < 0  # TODO: right it as a vector of false flags for god sake
    NonLinePoints = np.zeros([1, 3])

    ### Loop over the max number of trajectories to find (each time we find a line candidate, get rid of the : ###
    for ii in range(TrjNumFind):
        ### if there are less then 10 points just ignore and continue: ###
        if points_not_assigned_to_lines_yet.shape[0] < 10:
            continue

        ### Estimate line using RANSAC: ###
        direction_vec[ii], holding_point[ii], t_vec[ii], \
        trajectory_smoothed_polynom_X[ii], trajectory_smoothed_polynom_Y[ii], \
        points_not_assigned_to_lines_yet, xyz_line[ii], flag_enough_valid_samples[ii] = \
            Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Torch(points_not_assigned_to_lines_yet, params, H,
                                                                            W)

        ### if the flag "line_out_of_range" is True, meaning there aren't enough valid points after RANSAC, then ignore and continue: ###
        if flag_enough_valid_samples[ii] == False:
            continue

        ### Perform Some Heuristics on trajectory to see if there's enough "meat" to bite into: ###
        flag_is_trajectory_valid_vec[ii] = Decide_If_Trajectory_Valid_Drone_Torch(xyz_line[ii], t_vec[ii], direction_vec[ii],
                                                                            number_of_frames, params)
        if ~flag_is_trajectory_valid_vec[ii]:
            TrjMov[ii] = []
            # (*). NonLinePoints are points which RANSAC found to be a part of a line but after some heuristics in the above function i decided to ignore!
            NonLinePoints = np.concatenate((NonLinePoints, xyz_line[ii]))
            continue

        ### "Straighten out" and align frames where suspect is at to allow for proper frequency analysis: ###
        TrjMov[ii] = Cut_And_Align_ROIs_Around_Trajectory_Torch(Movie_BGS, t_vec[ii],
                                                                trajectory_smoothed_polynom_X[ii],
                                                                trajectory_smoothed_polynom_Y[ii], params)

    ### Only keep the "valid" trajectories which passed through all the heuristics: ###
    direction_vec = list(np.array(direction_vec)[np.where(flag_is_trajectory_valid_vec)])
    holding_point = list(np.array(holding_point)[np.where(flag_is_trajectory_valid_vec)])
    t_vec = list(np.array(t_vec)[np.where(flag_is_trajectory_valid_vec)])
    trajectory_smoothed_polynom_X = list(np.array(trajectory_smoothed_polynom_X)[np.where(flag_is_trajectory_valid_vec)])
    trajectory_smoothed_polynom_Y = list(np.array(trajectory_smoothed_polynom_Y)[np.where(flag_is_trajectory_valid_vec)])
    xyz_line = list(np.array(xyz_line)[np.where(flag_is_trajectory_valid_vec)])
    TrjMov = list(np.array(TrjMov)[np.where(flag_is_trajectory_valid_vec)])
    num_of_trj_found = np.sum(flag_is_trajectory_valid_vec)

    return points_not_assigned_to_lines_yet, NonLinePoints, direction_vec, holding_point, t_vec, \
           trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, TrjMov, num_of_trj_found


def Frequency_Analysis_And_Detection(super_dict):
    ### Get Parameters From super_dict: ###
    TrjMov = super_dict.TrjMov
    noise_std = super_dict.noise_std
    noise_dc = super_dict.noise_dc
    params = super_dict.params

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
        frq_ax = FrameRate * np.linspace(-.5, .5 - 1 / current_trajectory_number_of_points,
                                         current_trajectory_number_of_points)
        TrjMovie_FFT = np.fft.fftshift(np.fft.fft(TrjMov[trajectory_index], axis=0), axes=0) * (
                    current_trajectory_number_of_points / SeqT / FrameRate) ** .5

        ### Sum frequency content between each predefined frequency bin defined using the above freq_d: ###
        TrjMovie_FFT_BinPartitioned = np.zeros(
            (number_of_frequency_steps, ROI_allocated_around_suspect, ROI_allocated_around_suspect))
        for ii in range(number_of_frequency_steps):
            ind_frq_in = (frq_ax > frq_d[ii]) & (frq_ax < frq_u[ii])
            TrjMovie_FFT_BinPartitioned[ii, :, :] = np.real(np.mean(TrjMovie_FFT[ind_frq_in], 0))

        ### Normalize fft by the noise estimated globally before to see if it's significant enough by passing it into the score function: ###
        TrjMovie_FFT_BinPartitioned_NoiseNormalized = (TrjMovie_FFT_BinPartitioned - noise_dc) / noise_std
        TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index] = ScrFunc(DetNoiseThresh,
                                                                                 TrjMovie_FFT_BinPartitioned_NoiseNormalized)

        ### See at which bins the score was significant and normalize it to 100 for confidence measure: ###
        pxl_scr[trajectory_index] = np.sum(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[
                                               trajectory_index])  # sum over the entire bins and pixels (why over all bins)
        dec[trajectory_index] = pxl_scr[trajectory_index] > DetScrThresh
        DetectionConfLvl[trajectory_index] = 100 * np.minimum((pxl_scr[trajectory_index] - DetScrThresh) / (DetScrThresh100Conf - DetScrThresh), 1)

    return dec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl


def Noise_Estimation_FourierSpace_PerPixel_Torch(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    params = super_dict.params

    ### Get variables from params dict: ##
    FrameRate = params['FPS']
    DetFrqInit = params['DetFrqInit']
    DetFrqStop = params['DetFrqStop']
    T, H, W = Movie_BGS.shape

    ### Perform FFT per pixel: ###
    frq_ax = torch.Tensor(FrameRate * np.linspace(-0.5, 0.5 - 1 / T, T))
    MovFFT = fftshift_torch_specific_dim(torch.fft.fftn(Movie_BGS, 0), 0)

    ### Estimate noise between relevant frequency bins on the entire image: ###
    # TODO: this assumes each pixel has the same dynamics, use robust noise statistics
    # TODO: return noise per pixel instead of one noise figure
    noise_std = 0.5 * torch.std(MovFFT[(frq_ax > DetFrqInit) & (frq_ax < DetFrqStop)])
    noise_dc = 0
    return noise_std, noise_dc


def Noise_Estimation_FourierSpace_PerPixel(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    params = super_dict.params

    ### Get variables from params dict: ##
    FrameRate = params['FPS']
    DetFrqInit = params['DetFrqInit']
    DetFrqStop = params['DetFrqStop']
    T, H, W = Movie_BGS.shape

    ### Perform FFT per pixel: ###
    frq_ax = FrameRate * np.linspace(-0.5, 0.5 - 1 / T, T)
    MovFFT = np.fft.fftshift(np.fft.fft(Movie_BGS, axis=0), axes=0)

    ### Estimate noise between relevant frequency bins on the entire image: ###
    #TODO: instead of simply doing a mean over the frequency bands per pixel, use a robust noise model which fits statistics for the same mean pixel values
    noise_std = 0.5 * np.mean(MovFFT[(frq_ax > DetFrqInit) & (frq_ax < DetFrqStop)], 0)
    noise_dc = 0
    return noise_std, noise_dc

def NoiseEstOld(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    params = super_dict.params

    ### Get variables from params dict: ###
    DetFrqInit = params['DetFrqInit']
    DetFrqStop = params['DetFrqStop']
    DetDFrq = params['DetDFrq']
    FrameRate = params['FPS']
    number_of_frames, H, W = Movie_BGS.shape

    ### Get frequency vec to estimate on: ###
    frq_d = np.arange(DetFrqInit, DetFrqStop + 1, DetDFrq)[:-1]
    frq_u = frq_d + DetDFrq
    number_of_frequency_steps = len(frq_u)

    ### Get the movie's fft per pixel and "natural" frequency axis from the fft: ###
    frq_ax = FrameRate * np.linspace(-.5, .5 - 1 / number_of_frames, number_of_frames)
    MovFFT = np.fft.fftshift(np.fft.fft(Movie_BGS, axis=0), axes=0)

    ### Sum frequency content between each predefined frequency bin defined using the above freq_d: ###
    MovFFTBP = np.zeros((number_of_frequency_steps, H, W))
    for ii in range(number_of_frequency_steps):
        indices_within_current_frequency_bins = (frq_ax > frq_d[ii]) & (frq_ax < frq_u[ii])
        MovFFTBP[ii, :, :] = np.real(np.mean(MovFFT[indices_within_current_frequency_bins], 0))  # coherent sum

    ### Get the histogram of the entire fft_sums_in_bins and fit a gaussian function which will give us the mean and std of the expected values: ###
    # TODO: this assumes a constant noise throughout the image. this should be transferred to noise per pixel using robust per-pixel noise estimates
    y = np.histogram(MovFFTBP.flat, 1000)
    x0 = [np.max(y[0]), y[1][1:][y[0] == np.max(y[0])][0], 20]
    popt, pcov = curve_fit(gaussian_function, y[1][1:], y[0], p0=x0)
    noise_dc = popt[1]
    noise_std = 2 * popt[2]

    return noise_std, noise_dc


def Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit(super_dict):
    ### Get Parameters From super_dict: ###
    res_points = super_dict.res_points
    params = super_dict.params
    H = super_dict.H
    W = super_dict.W

    ### Get variables from params dict: ###
    DroneLineEst_RANSAC_D = params['DroneLineEst_RANSAC_D']
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    DroneLineEst_polyfit_degree = params['DroneLineEst_polyfit_degree']
    DroneLineEst_RANSAC_max_trials = params['DroneLineEst_RANSAC_max_trials']

    ### Perform RANSAC to estimate line: ###
    # (*). Remember, this accepts ALL the residual points coordinates and outputs a line. shouldn't we use something like clustering???
    model_robust, indices_within_distance = ransac(res_points, LineModelND, min_samples=2,
                                                   residual_threshold=DroneLineEst_RANSAC_D, max_trials=DroneLineEst_RANSAC_max_trials)
    holding_point = model_robust.params[0]
    direction_vec = model_robust.params[1].reshape(3, 1)

    ### Calculate distance between points and the line found by RANSAC, which hopefully gets close enough to a real trajectory, and only get those within a certain radius!: ###
    indices_within_distance = get_Distance_From_Points_To_DirectionVec(res_points, direction_vec, holding_point) < (
                2 * DroneLineEst_RANSAC_D)

    ### Use the above found "valid" points to estimate a new line/vector which is more robust: ###
    q = res_points[indices_within_distance]
    holding_point1 = np.mean(q,
                             0)  # the holding point is the temporal mean of the "valid" indices? is there a better alternative?
    _, _, direction_vec1 = np.linalg.svd(q - holding_point)
    direction_vec1 = direction_vec1[0].reshape(3, 1)

    ### Use these new points to see if the new line estimate and the residuals from it are still within the predefined radius: ###
    indices_within_distance = get_Distance_From_Points_To_DirectionVec(res_points, direction_vec1,
                                                                       holding_point1) < DroneLineEst_RANSAC_D
    valid_trajectory_points = res_points[indices_within_distance]
    t_vec = np.arange(min(valid_trajectory_points[:, 0]), max(valid_trajectory_points[:, 0]))

    ### if there are enough valid points then do a polynomial fit for their trajectory: ###
    if valid_trajectory_points.shape[0] > 20:

        ### Polyfit trajectory to a polynomial of wanted degree: ###
        fitx = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 1],
                                    DroneLineEst_polyfit_degree))  # fit (t,x) polynomial
        fity = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 2],
                                    DroneLineEst_polyfit_degree))  # fit (t,y) polynomial

        ### Use found polynomial to get "smoothed" trajectory from the polyfit over the relevant, discrete time_vec: ###
        trajectory_smoothed_polynom_X = fitx(t_vec)
        trajectory_smoothed_polynom_Y = fity(t_vec)

        ### Get (t,x,y) points vec: ###
        smoothed_trajectory_over_time_vec = np.array(
            (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)).T

        ### Test if the x&y trajectories are within certain boundaries: ###
        indices_where_x_trajectory_within_frame_boundaries = (
                                                                         trajectory_smoothed_polynom_X >= ROI_allocated_around_suspect / 2 + 1) & (
                                                                         trajectory_smoothed_polynom_X <= H - ROI_allocated_around_suspect / 2 - 1)
        indices_where_y_trajectory_within_frame_boundaries = (
                                                                         trajectory_smoothed_polynom_Y >= ROI_allocated_around_suspect / 2 + 1) & (
                                                                         trajectory_smoothed_polynom_Y <= W - ROI_allocated_around_suspect / 2 - 1)

        ### Get indices_where_trajectory_is_within_frame_boundaries: ###
        indices_1 = (
                    indices_where_x_trajectory_within_frame_boundaries & indices_where_y_trajectory_within_frame_boundaries)

        ### Get indices where_trajectory_is_close_enough_to_line_estimate: ###
        indices_2 = get_Distance_From_Points_To_DirectionVec(smoothed_trajectory_over_time_vec, direction_vec1,
                                                             holding_point1) < (DroneLineEst_RANSAC_D)

        ### Final valid indices are those which satisfy both conditions: ###
        t_vec_valid = indices_1 * indices_2
    else:
        # TODO: ?????
        t_vec_valid = t_vec < 0
        trajectory_smoothed_polynom_X = t_vec
        trajectory_smoothed_polynom_Y = t_vec

    ### Get valid parts of the trajectory: ###
    t_vec = t_vec[t_vec_valid]
    trajectory_smoothed_polynom_X = trajectory_smoothed_polynom_X[t_vec_valid]
    trajectory_smoothed_polynom_Y = trajectory_smoothed_polynom_Y[t_vec_valid]

    ### make sure there are at least 10 valid points: ###
    if len(t_vec) > 10:
        flag_enough_valid_samples = True
    else:
        flag_enough_valid_samples = False

    ### Get points within the predefined distance and those above it: ###   #TODO: what do we do with points_new. i should really name points_new to be points_far_from_line or points_off_line
    points_off_line_found = res_points[indices_within_distance == False]
    points_on_line_found = res_points[indices_within_distance]

    return direction_vec1, holding_point1, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, points_off_line_found, points_on_line_found, flag_enough_valid_samples


def Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Torch(super_dict):
    ### Get Parameters From super_dict: ###
    res_points = super_dict.res_points
    params = super_dict.params
    H = super_dict.H
    W = super_dict.W

    ### Get variables from params dict: ###
    DroneLineEst_RANSAC_D = params['DroneLineEst_RANSAC_D']
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    DroneLineEst_polyfit_degree = params['DroneLineEst_polyfit_degree']
    DroneLineEst_RANSAC_max_trials = params['DroneLineEst_RANSAC_max_trials']

    ### Perform RANSAC to estimate line: ###
    # (*). Remember, this accepts ALL the residual points coordinates and outputs a line. shouldn't we use something like clustering???
    # TODO: switch to pytorch RANSAC from RANSAC-Flow
    model_robust, indices_within_distance = ransac(res_points, LineModelND, min_samples=2,
                                                   residual_threshold=DroneLineEst_RANSAC_D, max_trials=DroneLineEst_RANSAC_max_trials)
    holding_point = model_robust.params[0]
    direction_vec = model_robust.params[1].reshape(3, 1)

    ### Calculate distance between points and the line found by RANSAC, which hopefully gets close enough to a real trajectory, and only get those within a certain radius!: ###
    indices_within_distance = get_Distance_From_Points_To_DirectionVec_Torch(res_points, direction_vec,
                                                                             holding_point) < (2 * DroneLineEst_RANSAC_D)

    ### Use the above found "valid" points to estimate a new line/vector which is more robust: ###
    q = res_points[indices_within_distance]
    holding_point1 = torch.mean(q,
                                0)  # the holding point is the temporal mean of the "valid" indices? is there a better alternative?
    _, _, direction_vec1 = torch.svd(q - holding_point)
    direction_vec1 = direction_vec1[0].reshape(3, 1)

    ### Use these new points to see if the new line estimate and the residuals from it are still within the predefined radius: ###
    indices_within_distance = get_Distance_From_Points_To_DirectionVec_Torch(res_points, direction_vec1,
                                                                             holding_point1) < DroneLineEst_RANSAC_D
    valid_trajectory_points = res_points[indices_within_distance]
    t_vec = np.arange(min(valid_trajectory_points[:, 0]), max(valid_trajectory_points[:, 0]))

    ### if there are enough valid points then do a polynomial fit for their trajectory: ###
    if valid_trajectory_points.shape[0] > 20:

        ### Polyfit trajectory to a polynomial of wanted degree: ###
        fitx = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 1],
                                    DroneLineEst_polyfit_degree))  # fit (t,x) polynomial
        fity = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 2],
                                    DroneLineEst_polyfit_degree))  # fit (t,y) polynomial

        ### Use found polynomial to get "smoothed" trajectory from the polyfit over the relevant, discrete time_vec: ###
        trajectory_smoothed_polynom_X = fitx(t_vec)
        trajectory_smoothed_polynom_Y = fity(t_vec)

        ### Get (t,x,y) points vec: ###
        smoothed_trajectory_over_time_vec = np.array(
            (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)).T

        ### Test if the x&y trajectories are within certain boundaries: ###
        indices_where_x_trajectory_within_frame_boundaries = (
                                                                         trajectory_smoothed_polynom_X >= ROI_allocated_around_suspect / 2 + 1) & (
                                                                         trajectory_smoothed_polynom_X <= H - ROI_allocated_around_suspect / 2 - 1)
        indices_where_y_trajectory_within_frame_boundaries = (
                                                                         trajectory_smoothed_polynom_Y >= ROI_allocated_around_suspect / 2 + 1) & (
                                                                         trajectory_smoothed_polynom_Y <= W - ROI_allocated_around_suspect / 2 - 1)

        ### Get indices_where_trajectory_is_within_frame_boundaries: ###
        indices_1 = (
                    indices_where_x_trajectory_within_frame_boundaries & indices_where_y_trajectory_within_frame_boundaries)

        ### Get indices where_trajectory_is_close_enough_to_line_estimate: ###
        indices_2 = get_Distance_From_Points_To_DirectionVec_Torch(smoothed_trajectory_over_time_vec, direction_vec1,
                                                                   holding_point1) < (DroneLineEst_RANSAC_D)

        ### Final valid indices are those which satisfy both conditions: ###
        t_vec_valid = indices_1 * indices_2
    else:
        # TODO: ?????
        t_vec_valid = t_vec < 0
        trajectory_smoothed_polynom_X = t_vec
        trajectory_smoothed_polynom_Y = t_vec

    ### Get valid parts of the trajectory: ###
    t_vec = t_vec[t_vec_valid]
    trajectory_smoothed_polynom_X = trajectory_smoothed_polynom_X[t_vec_valid]
    trajectory_smoothed_polynom_Y = trajectory_smoothed_polynom_Y[t_vec_valid]

    ### make sure there are at least 10 valid points: ###
    if len(t_vec) > 10:
        flag_enough_valid_samples = True
    else:
        flag_enough_valid_samples = False

    ### Get points within the predefined distance and those above it: ###   #TODO: what do we do with points_new. i should really name points_new to be points_far_from_line or points_off_line
    points_off_line_found = res_points[indices_within_distance == False]
    points_on_line_found = res_points[indices_within_distance]

    return direction_vec1, holding_point1, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, points_off_line_found, points_on_line_found, flag_enough_valid_samples


def gaussian_function(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def ScrFunc(DetNoiseThresh, TrjMovFFTBP1):
    return 1./(1+np.exp(5*(DetNoiseThresh-TrjMovFFTBP1)))


def Decide_If_Trajectory_Valid_Drone(super_dict):
    ### Get Parameters From super_dict: ###
    xyz_line = super_dict.xyz_line
    t_vec = super_dict.t_vec
    direction_vec = super_dict.direction_vec
    number_of_frames = super_dict.number_of_frames
    params = super_dict.params

    ### Get variables from params dictionary: ###
    TrjDecMinNumOfPnts = params['TrjDecMinNumOfPnts']
    TrjDecMaxVel = params['TrjDecMaxVel']
    TrjDecMinVel = params['TrjDecMinVel']
    TrjDecMaxDisappear = params['TrjDecMaxDisappear']
    TrjDecMinT = params['TrjDecMinT']
    SeqT = params['SeqT']
    FrameRate = params['FPS']

    ### make sure there are enough time stamps to even look into trajectory: ###
    flag_trajectory_valid = len(t_vec) > (number_of_frames * TrjDecMinNumOfPnts)

    if flag_trajectory_valid:
        ### make sure velocity is between min and max values: ###
        b_abs = np.abs(direction_vec)[:, 0]
        v = (b_abs[1] ** 2 + b_abs[
            2] ** 2) ** 0.5  # the portion of the unit (t,x,y) direction vec's projection on the (x,y) plane
        flag_trajectory_valid = ((v < TrjDecMaxVel) & (
                    v > TrjDecMinVel))  # make sure the drone's "velocity" in the (x,y) is between some values, if it's too stationary it's weird and if it's too fast it's weird
        if flag_trajectory_valid:

            ### make sure max difference between time stamps, which is also the maximum consecutive frames suspect disappeared, is not too high: ###
            flag_trajectory_valid = (np.max(np.diff(xyz_line[:, 0])) < TrjDecMaxDisappear)
            if flag_trajectory_valid:
                ### Make sure total time for valid points is above the minimum we decided upon: ###
                minimum_total_valid_time_range_for_suspect = (TrjDecMinT * SeqT * FrameRate)
                flag_trajectory_valid = minimum_total_valid_time_range_for_suspect < (np.max(t_vec) - np.min(t_vec))
    return flag_trajectory_valid


def Decide_If_Trajectory_Valid_Drone_Torch(super_dict):
    ### Get Parameters From super_dict: ###
    xyz_line = super_dict.xyz_line
    t_vec = super_dict.t_vec
    direction_vec = super_dict.direction_vec
    number_of_frames = super_dict.number_of_frames
    params = super_dict.params

    ### Get variables from params dictionary: ###
    TrjDecMinNumOfPnts = params['TrjDecMinNumOfPnts']
    TrjDecMaxVel = params['TrjDecMaxVel']
    TrjDecMinVel = params['TrjDecMinVel']
    TrjDecMaxDisappear = params['TrjDecMaxDisappear']
    TrjDecMinT = params['TrjDecMinT']
    SeqT = params['SeqT']
    FrameRate = params['FPS']

    ### make sure there are enough time stamps to even look into trajectory: ###
    flag_trajectory_valid = len(t_vec) > (number_of_frames * TrjDecMinNumOfPnts)

    if flag_trajectory_valid:
        ### make sure velocity is between min and max values: ###
        b_abs = np.abs(direction_vec)[:, 0]
        v = (b_abs[1] ** 2 + b_abs[
            2] ** 2) ** 0.5  # the portion of the unit (t,x,y) direction vec's projection on the (x,y) plane
        flag_trajectory_valid = ((v < TrjDecMaxVel) & (
                    v > TrjDecMinVel))  # make sure the drone's "velocity" in the (x,y) is between some values, if it's too stationary it's weird and if it's too fast it's weird
        if flag_trajectory_valid:

            ### make sure max difference between time stamps, which is also the maximum consecutive frames suspect disappeared, is not too high: ###
            flag_trajectory_valid = (np.max(np.diff(xyz_line[:, 0])) < TrjDecMaxDisappear)
            if flag_trajectory_valid:
                ### Make sure total time for valid points is above the minimum we decided upon: ###
                minimum_total_valid_time_range_for_suspect = (TrjDecMinT * SeqT * FrameRate)
                flag_trajectory_valid = minimum_total_valid_time_range_for_suspect < (np.max(t_vec) - np.min(t_vec))
    return flag_trajectory_valid


def Cut_And_Align_ROIs_Around_Trajectory(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    t_vec = super_dict.t_vec
    trajectory_smoothed_polynom_X = super_dict.trajectory_smoothed_polynom_X
    trajectory_smoothed_polynom_Y = super_dict.trajectory_smoothed_polynom_Y
    params = super_dict.params

    ### Get basic parameters: ###
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    H = np.size(Movie_BGS,1)
    W = np.size(Movie_BGS,2)
    number_of_time_steps = len(t_vec)

    ### Allocated large frame grid: ###
    xM = np.arange(H)
    yM = np.arange(W)
    original_grid = (t_vec, xM, yM)

    ### Allocate suspect ROI grid: ###
    xl = np.arange(ROI_allocated_around_suspect) - np.int(ROI_allocated_around_suspect/2)
    xg = np.array(np.meshgrid(t_vec, xl, xl)).T  #just like meshgrid(x,y) creates two matrices X,Y --> this create 3 matrices of three dimensions which are then concatated
    number_of_pixels_per_ROI = ROI_allocated_around_suspect**2

    ### loop over suspect ROI grid and add the trajectory to each point on the grid: ###
    for ii in range(ROI_allocated_around_suspect):
        for jj in range(ROI_allocated_around_suspect):
            xg[ii, :, jj, 2] = xg[ii, :, jj, 2] + trajectory_smoothed_polynom_Y
            xg[ii, :, jj, 1] = xg[ii, :, jj, 1] + trajectory_smoothed_polynom_X
    xg = xg.reshape((number_of_pixels_per_ROI*number_of_time_steps, 3))

    ### "align" ROIs which contain suspect on top of each other by multi-dimensional interpolation): ###
    TrjMov = scipy.interpolate.interpn(original_grid, Movie_BGS[t_vec,:,:], xg, method='linear')  #TODO: maybe nearest neighbor???? to avoid problems
    TrjMov = TrjMov.reshape([ROI_allocated_around_suspect, number_of_time_steps, ROI_allocated_around_suspect]).transpose([1, 0, 2])

    #TODO: there's probably a faster way of doing it: ###
    return TrjMov

def Cut_And_Align_ROIs_Around_Trajectory_Torch(super_dict):
    ### Get Parameters From super_dict: ###
    Movie_BGS = super_dict.Movie_BGS
    t_vec = super_dict.t_vec
    trajectory_smoothed_polynom_X = super_dict.trajectory_smoothed_polynom_X
    trajectory_smoothed_polynom_Y = super_dict.trajectory_smoothed_polynom_Y
    params = super_dict.params

    ### Get basic parameters: ###
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    H = Movie_BGS.shape[1]
    W = Movie_BGS.shape[2]
    number_of_time_steps = len(t_vec)

    ### Allocated large frame grid: ###
    xM = np.arange(H)
    yM = np.arange(W)
    original_grid = (t_vec, xM, yM)

    ### Allocate suspect ROI grid: ###
    xl = np.arange(ROI_allocated_around_suspect) - np.int(ROI_allocated_around_suspect/2)
    xg = np.array(np.meshgrid(t_vec, xl, xl)).T  #just like meshgrid(x,y) creates two matrices X,Y --> this create 3 matrices of three dimensions which are then concatated
    number_of_pixels_per_ROI = ROI_allocated_around_suspect**2

    ### loop over suspect ROI grid and add the trajectory to each point on the grid: ###
    for ii in range(ROI_allocated_around_suspect):
        for jj in range(ROI_allocated_around_suspect):
            xg[ii, :, jj, 2] = xg[ii, :, jj, 2] + trajectory_smoothed_polynom_Y
            xg[ii, :, jj, 1] = xg[ii, :, jj, 1] + trajectory_smoothed_polynom_X
    xg = xg.reshape((number_of_pixels_per_ROI * number_of_time_steps, 3))

    ### "align" ROIs which contain suspect on top of each other by multi-dimensional interpolation): ###
    #TODO: change to pytorch grid sample
    TrjMov = scipy.interpolate.interpn(original_grid, Movie_BGS[t_vec,:,:], xg, method='linear')  #TODO: maybe nearest neighbor???? to avoid problems
    TrjMov = TrjMov.reshape([ROI_allocated_around_suspect, number_of_time_steps, ROI_allocated_around_suspect]).transpose([1, 0, 2])

    #TODO: there's probably a faster way of doing it: ###
    return TrjMov


def scale_array_to_range(mat_in, min_max_values=(0,1)):
    mat_in_normalized = (mat_in - mat_in.min()) / (mat_in.max()-mat_in.min()) * (min_max_values[1]-min_max_values[0]) + min_max_values[0]
    return mat_in_normalized

def PreProcess_Movie(super_dict):
    ### Get Parameters From super_dict: ###
    Mov = super_dict.Mov
    params = super_dict.params

    ### Stretch & Change Movie For Proper Viewing & Analysis: ###

    ### Get min and max values to stretch image for proper viewing: ###
    if 'min_value' in params.keys():
        if params.min_value is not None:
            min_value = params.min_value
            max_value = params.max_value
            flag_normalize = False
        else:
            flag_normalize = True
    else:
        flag_normalize = True
    if flag_normalize:
        min_value = np.quantile(Mov, 0.01)
        max_value = np.quantile(Mov, 0.99)
        params.min_value = min_value
        params.max_value = max_value

    ### Actually stretch image: ###
    Mov = Mov.clip(min_value, max_value)  #clamp the image first to allow for effective stretching for viewing
    Mov = scale_array_to_range(Mov, (0, 1))

    ### Cut the edges of the frame which are non valid for our pelican-D: ###
    Mov = Mov[:, 2:, 2:]

    return Mov, params

def PreProcess_Movie_Torch(super_dict):
    ### Get Parameters From super_dict: ###
    Mov = super_dict.Mov
    params = super_dict.params

    ### Stretch & Change Movie For Proper Viewing & Analysis: ###

    ### Get min and max values to stretch image for proper viewing: ###
    if 'min_value' in params.keys():
        if params.min_value is not None:
            min_value = params.min_value
            max_value = params.max_value
            flag_normalize = False
        else:
            flag_normalize = True
    else:
        flag_normalize = True
    if flag_normalize:
        min_value = Mov[0:10].quantile(0.01)
        max_value = Mov[0:10].quantile(0.99)
        params.min_value = min_value
        params.max_value = max_value

    ### Actually stretch image: ###
    Mov = Mov.clip(min_value, max_value)  #clamp the image first to allow for effective stretching for viewing
    Mov = scale_array_to_range(Mov, (0, 1))

    ### Cut the edges of the frame which are non valid for our pelican-D: ###
    Mov = Mov[:, 2:, 2:]

    return Mov, params

def get_shape(input_array):
    if type(input_array) == np.ndarray:
        if len(input_array.shape)==2:
            H,W = input_array.shape
            C = 1
        elif len(input_array.shape)==3:
            H,W,C = input_array.shape
    elif type(input_array) == torch.Tensor:
        if len(input_array.shape) == 2:
            H,W = input_array.shape
            C = 1
        if len(input_array.shape)==3:
            C,H,W = input_array.shape
        elif len(input_array.shape)==4:
            B,H,W,C = input_array.shape
    return H,W,C


def shift_matrix_subpixel_numpy(original_image, shiftx, shifty):

    H,W,C = get_shape(original_image)

    # Get tilt phases k-space:
    x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
    y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
    delta_f1 = 1 / W
    delta_f2 = 1 / H
    f_x = x * delta_f1
    f_y = y * delta_f2
    # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_x = np.fft.fftshift(f_x)
    f_y = np.fft.fftshift(f_y)
    # Build k-space meshgrid:
    [kx, ky] = np.meshgrid(f_x, f_y)
    # get displacement matrix:
    displacement_matrix = np.exp(-(1j * 2 * np.pi * ky * shifty + 1j * 2 * np.pi * kx * shiftx))

    # displacement_matrix = np.atleast_3d(displacement_matrix)
    # original_image = np.atleast_3d(original_image)
    displacement_matrix = displacement_matrix.squeeze()
    original_image = original_image.squeeze()

    # get displaced speckles:
    fft_image = np.fft.fft2(original_image)
    fft_image_displaced = fft_image * displacement_matrix  # the shift or from one image to another, not for the original phase screen
    original_image_displaced = np.fft.ifft2(fft_image_displaced).real

    original_image_displaced = np.atleast_3d(original_image_displaced)
    return original_image_displaced


def Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(input_tensor, reference_tensor=None, device='cuda'):
    ### the function expects input_tensor to be a numpy array of shape [T,H,W]: ###

    ### Initialization: ###
    if type(input_tensor) == np.ndarray:
        input_tensor = torch.Tensor(input_tensor)
        if device is not None:
            input_tensor = input_tensor.to(device)
        flag_type = 'numpy'
    else:
        flag_type = 'torch'
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    shift_layer = Shift_Layer_Torch()
    B, C, H, W = input_tensor.shape
    number_of_samples = B

    def fit_polynomial_torch(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[:, 2] + y[:, 0] - 2 * y[:, 1]) / 2
        b = -(y[:, 0] + 2 * a * x[1] - y[:, 1] - a)
        c = y[:, 1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    #########################################################################################################
    ### Get CC: ###
    # input tensor [T,H,W]
    B, T, H, W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1, -2])

    ### Get reference tensor fft: ###
    if reference_tensor is None:
        reference_tensor_fft = input_tensor_fft[:, T // 2:T//2+1, :, :]
    else:
        if device is not None:
            reference_tensor = reference_tensor.to(device)
        reference_tensor_fft = torch.fft.fftn(reference_tensor, dim=[-1,-2])

    # (*). Circular Cross Corerlation Using FFT:
    output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * reference_tensor_fft.conj(), dim=[-1, -2]).real
    # output_CC = torch.cat()

    #########################################################################################################
    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    # (1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    # (2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > H] -= W
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * W
    output_CC_flattened_indices_i0 = i1 + i0 * W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, H * W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    #########################################################################################################

    #########################################################################################################
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # print(shiftx)
    # print(real_shifts_to_center_frame)
    #########################################################################################################

    #########################################################################################################
    ### Align Images: ###
    shifted_tensors = shift_layer.forward(input_tensor.permute([1, 0, 2, 3]), -shiftx, -shifty)  # TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0, True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    if flag_type == 'numpy':
        return shifted_tensors[:, 0, :, :].cpu().numpy()
    else:
        return shifted_tensors[:,0,:,:]


class Shift_Layer_Torch(nn.Module):
    # Initialize this with a module
    def __init__(self):
        super(Shift_Layer_Torch, self).__init__()
        self.kx = None
        self.ky = None

    def forward(self, input_image, shiftx, shifty, fft_image=None):
        ### Check if we need to create a new kvec: ###
        if self.kx is None:
            flag_create_new_kvec = True
        else:
            if self.kx.shape[-1] != input_image.shape[-1] or self.kx.shape[-2] != input_image.shape[-2]:
                flag_create_new_kvec = True
            else:
                flag_create_new_kvec = False

        ### If we do, create it: ###
        if flag_create_new_kvec:
            # Get Input Dimensions:
            self.ndims = len(input_image.shape)
            if self.ndims == 4:
                B,C,H,W = input_image.shape
            elif self.ndims == 3:
                C,H,W = input_image.shape
                B = 1
            elif self.ndims == 2:
                H,W = input_image.shape
                B = 1
                C = 1
            self.B = B
            self.C = C
            self.H = H
            self.W = W
            # Get tilt phases k-space:
            x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
            y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
            delta_f1 = 1 / W
            delta_f2 = 1 / H
            f_x = x * delta_f1
            f_y = y * delta_f2
            # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
            f_x = np.fft.fftshift(f_x)
            f_y = np.fft.fftshift(f_y)
            # Build k-space meshgrid:
            [kx, ky] = np.meshgrid(f_x, f_y)
            # Frequency vec to tensor:
            self.kx = torch.Tensor(kx)
            self.ky = torch.Tensor(ky)
            # Expand to original image dimensions:
            if self.ndims == 3:
                self.kx = self.kx.unsqueeze(0)
                self.ky = self.ky.unsqueeze(0)
            if self.ndims == 4:
                self.kx = self.kx.unsqueeze(0).unsqueeze(0)
                self.ky = self.ky.unsqueeze(0).unsqueeze(0)
            # K-vecs to tensor device:
            self.kx = self.kx.to(input_image.device)
            self.ky = self.ky.to(input_image.device)

        ### Expand shiftx & shifty to match input image shape (and if shifts are more than a scalar i assume you want to multiply B or C): ###
        #TODO: enable accepting pytorch tensors/vectors etc'
        if type(shiftx) != list and type(shiftx) != tuple and type(shiftx) != np.ndarray and type(shiftx) != torch.Tensor:
            shiftx = [shiftx]
            shifty = [shifty]
        if self.ndims == 3:
            if type(shiftx) != torch.Tensor:
                shiftx = torch.Tensor(shiftx).to(input_image.device).unsqueeze(-1).unsqueeze(-1)
                shifty = torch.Tensor(shifty).to(input_image.device).unsqueeze(-1).unsqueeze(-1)
            else:
                shiftx = shiftx.to(input_image.device).unsqueeze(-1).unsqueeze(-1)
                shifty = shifty.to(input_image.device).unsqueeze(-1).unsqueeze(-1)
            shiftx = shiftx.to(input_image.device)
            shifty = shifty.to(input_image.device)
        elif self.ndims == 4:
            if type(shiftx) != torch.Tensor:
                shiftx = torch.Tensor(shiftx).to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                shifty = torch.Tensor(shifty).to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                shiftx = shiftx.to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                shifty = shifty.to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shiftx = shiftx.to(input_image.device)
            shifty = shifty.to(input_image.device)

        ### Displace: ###
        # print(shiftx)
        # print(self.kx.shape)
        displacement_matrix = torch.exp(-(1j * 2 * np.pi * self.ky * shifty + 1j * 2 * np.pi * self.kx * shiftx))
        if fft_image is None:
            fft_image = torch.fft.fftn(input_image, dim=[-1, -2])
        fft_image_displaced = fft_image * displacement_matrix
        # input_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()
        input_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).real

        return input_image_displaced


def Plot_3D_PointCloud_With_Trajectories(super_dict):
    ### Get Parameters From super_dict: ###
    res_points = super_dict.res_points
    NonLinePoints = super_dict.NonLinePoints
    xyz_line = super_dict.xyz_line
    num_of_trj_found = super_dict.num_of_trj_found
    params = super_dict.params
    save_name = super_dict.save_name
    Res_dir = super_dict.Res_dir
    auto_close_flg = super_dict.auto_close_flg

    ### Get variables from params dict: ###
    SaveResFlg = params['SaveResFlg']
    roi = params['roi']
    T = params['FPS'] * params['SeqT']

    ### Stake All The Found Lines On Top Of Each Other: ###
    x = np.zeros((0, 3))
    c = np.zeros((0, 1))
    for ii in range(num_of_trj_found):
        x = np.vstack((x, xyz_line[ii]))
        c = np.vstack((c, ii * np.ones((xyz_line[ii].shape[0], 1))))
    c = c.reshape((len(c),))

    ### Create 3D Plot: ###
    fig = plt.figure(210)
    ax = plt.axes(projection='3d')

    ### Add all the outlier points found: ###
    ax.scatter(res_points[:, 0], res_points[:, 1], res_points[:, 2], c='k',
               marker='*', label='Outlier', s=1)

    ### Add the "NonLinePoints" which were found by RANSAC but were disqualified after heuristics: ###
    ax.scatter(NonLinePoints[:, 0], NonLinePoints[:, 1], NonLinePoints[:, 2], c='k',
               marker='*', label='NonLinePoints', s=20)

    ### Add the proper trajectory: ###
    scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c, marker='o', s=20)

    ### Add Legends: ###
    if x.shape[0] != 0:
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper left", title="Trj")
        ax.add_artist(legend1)

    ### Maximize plot window etc': ###
    ax.legend(loc='lower left')
    ax.view_init(elev=20, azim=60)
    plt.get_current_fig_manager().full_screen_toggle()
    ax.set_xlim(0, T)
    ax.set_zlim(0, roi[1])
    ax.set_ylim(0, roi[0])
    plt.pause(.000001)
    plt.show()

    ### Save Figure If Wanted: ###
    if SaveResFlg:
        plt.savefig(Res_dir + "\\" + save_name + ".png")

    ### Close Plot: ###
    if auto_close_flg:
        plt.close(fig)


def Plot_FFT_Bins_Detection_SubPlots(super_dict):
    ### Get Parameters From super_dict: ###
    num_of_trj_found = super_dict.num_of_trj_found
    TrjMovie_FFT_BinPartitioned_AfterScoreFunction = super_dict.TrjMovie_FFT_BinPartitioned_AfterScoreFunction
    frequency_vec = super_dict.frequency_vec
    params = super_dict.params
    Res_dir = super_dict.Res_dir
    TrjMov = super_dict.TrjMov
    DetectionDec = super_dict.DetectionDec
    DetectionConfLvl = super_dict.DetectionConfLvl

    ### FFT Bins Plot: ###

    SaveResFlg = params['SaveResFlg']

    ### Loop over the number of trajectories found: ###
    for trajectory_index in range(num_of_trj_found):
        fig = plt.figure(21 + trajectory_index)
        clim_min = 0
        clim_max = 1
        number_of_bins_sqrt = np.ceil(np.sqrt(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[0].shape[0]))

        ### Loop over the different frequency bins and present them all in different subplots on the same big plot: ###
        for ii in range(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index].shape[0]):
            plt.subplot(number_of_bins_sqrt, number_of_bins_sqrt, 1 + ii)
            plt.imshow(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index][ii, :, :].T, cmap='hot')
            plt.title("frq. = " + str(frequency_vec[ii]))
            plt.clim(clim_min, clim_max)
            plt.axis('off')
        plt.colorbar()

        fig.text(0.5, 0.95, "Suspect signature in Fourier domain - Trj. no." + str(trajectory_index) + (
            " (quad. conf. " + str(int(DetectionConfLvl[trajectory_index])) + "%)" if DetectionDec[
                trajectory_index] else " (not a quad.)"), size=50, ha="center", va="center",
                 bbox=dict(boxstyle="round", ec=(0.2, 1., 0.2), fc=(0.8, 1., 0.8), ))
        plt.pause(.01)

        if SaveResFlg:
            plt.get_current_fig_manager().full_screen_toggle()
            plt.pause(.01)
            plt.savefig(Res_dir + "\\" + "Detection_trj" + str(trajectory_index) + ".png")
            Plot_BoundingBoxes_On_Video(TrjMov[trajectory_index], fps=100, tit="Trj" + str(trajectory_index), flag_transpose_movie=True,
                    frame_skip_steps=1, resize_factor=30, flag_save_movie=1, Res_dir=Res_dir + "\\")

        plt.close(fig)


def Plot_BoundingBoxes_On_Video(super_dict):
    ### Get Parameters From super_dict: ###
    Mov = super_dict.Mov
    fps = super_dict.fps
    tit = super_dict.tit
    flag_transpose_movie = super_dict.flag_transpose_movie
    frame_skip_steps = super_dict.frame_skip_steps
    flag_save_movie = super_dict.flag_save_movie
    resize_factor = super_dict.resize_factor
    histogram_limits_to_stretch = super_dict.histogram_limits_to_stretch
    Res_dir = super_dict.Res_dir
    trajectory_tuple = super_dict.trajectory_tuple

    # trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)

    ### Initialize Things: ###
    flag_found_trajectories = len(trajectory_tuple) == 3
    if flag_found_trajectories == False:
        return
    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    number_of_frames_total,H,W = Mov.shape
    number_of_frames_to_view = number_of_frames_total // frame_skip_steps
    Mov = np.float32(Mov)
    wait1 = 1 / fps
    if flag_transpose_movie:
        iax0 = 1
        iax1 = 2
        Mov = np.transpose(Mov, (0,2,1))
        trajectory_smoothed_polynom_XY = (trajectory_tuple[2], trajectory_tuple[1])
    else:
        iax0 = 2
        iax1 = 1
        trajectory_smoothed_polynom_XY = (trajectory_tuple[1], trajectory_tuple[2])
    if flag_save_movie:
        OutVideoWriter = cv2.VideoWriter(Res_dir + tit + ".avi", cv2.VideoWriter_fourcc(*'XVID'), fps,
                                         (resize_factor * Mov.shape[iax0], resize_factor * Mov.shape[iax1]), 0)

    ### Stretch Movie Histogram & Leave out the histogram_limits_to_stretch distribution ends on both sides: ###
    # histogram_limits_to_stretch_2 = 1 - histogram_limits_to_stretch
    # a, b = np.histogram(Mov[0], np.int(np.max(Mov[0])-np.min(Mov[0])))
    # a = np.cumsum(a)
    # a = a/np.max(a)
    # amn = b[np.where(a>histogram_limits_to_stretch)[0][0]]
    # amx = b[np.where(a>histogram_limits_to_stretch_2)[0][0]]
    # Mov[Mov>amx] = amx
    # Mov[Mov<amn] = amn
    # Mov = np.uint8(255*(Mov - amn)/(amx - amn))
    Mov = np.uint8(255*Mov.clip(0,1))

    ### Loop over movie frames and paint circle over discovered trajectories: ###
    BoundingBox_PerFrame_list = []
    for i in np.arange(number_of_trajectories):
        BoundingBox_PerFrame_list.append([])

    current_frame_index = 0
    for frame_index in np.arange(0,number_of_frames_total,frame_skip_steps):
        ### Get current movie frame: ###
        image_frame = Mov[current_frame_index, :, :]

        ### Resize Frame: ###
        new_size = tuple((np.array(image_frame.shape) * resize_factor).astype(int))
        image_frame = cv2.resize(image_frame, (new_size[1], new_size[0]), cv2.INTER_NEAREST)

        ### Loop Over Found Trajectories & Paint Circles On Them: ###
        for trajectory_index in range(number_of_trajectories):
            if (current_frame_index in t_vec[trajectory_index]):
                ### Find the index within current trajectory which includes current frame: ###
                current_frame_index_within_trajectory = np.where(current_frame_index == t_vec[trajectory_index])[0][0]

                ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
                target_spatial_coordinates = (np.int(trajectory_smoothed_polynom_XY[0][trajectory_index][current_frame_index_within_trajectory]*resize_factor),
                                              np.int(trajectory_smoothed_polynom_XY[1][trajectory_index][current_frame_index_within_trajectory]*resize_factor))

                ### Keep Track Of Bounding-Boxes (H_index, W_index): ###
                BoundingBox_PerFrame_list[trajectory_index].append((target_spatial_coordinates[1],target_spatial_coordinates[0]))

                ### Draw Circle On Screen: ###   #TODO: draw rectangle instead and register/output coordinates per frame cleanly for later analysis
                circle_radius_in_pixels = 10
                image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (0,0,255), 1)
                image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (target_spatial_coordinates[0]-circle_radius_in_pixels, target_spatial_coordinates[1]-circle_radius_in_pixels), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

        ### Actually Show Image With Circles On It: ###
        cv2.imshow(tit, image_frame)

        ### Save Movie With Circles On It If Wanted: ###
        if flag_save_movie:
            OutVideoWriter.write(image_frame)

        ### Skip Frame/s: ###
        current_frame_index += frame_skip_steps
        if ((cv2.waitKey(np.int(1000*wait1)) == ord('q')) | (current_frame_index>np.size(Mov,0)-1)):
            break

    ### Finish Movie Recording: ###
    if flag_save_movie:
        OutVideoWriter.release()
    cv2.destroyWindow(tit)

    return BoundingBox_PerFrame_list


def Get_BoundingBox_List_For_Each_Frame(super_dict):
    ### Get Parameters From super_dict: ###
    Mov = super_dict.Mov
    fps = super_dict.fps
    tit = super_dict.tit
    Res_dir = super_dict.Res_dir
    flag_save_movie = super_dict.flag_save_movie
    trajectory_tuple = super_dict.trajectory_tuple

    # trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)

    ### Initialize Things: ###
    flag_found_trajectories = len(trajectory_tuple) == 3
    if flag_found_trajectories == False:
        return
    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    number_of_frames_total, H, W = Mov.shape
    number_of_frames_to_view = number_of_frames_total
    Mov = np.float32(Mov)
    wait1 = 1 / fps
    iax0 = 2
    iax1 = 1
    trajectory_smoothed_polynom_XY = (trajectory_tuple[1], trajectory_tuple[2])


    ### Stretch Movie Histogram & Leave out the histogram_limits_to_stretch distribution ends on both sides: ###
    Mov = np.uint8(255*Mov.clip(0,1))

    ### Loop over movie frames and paint circle over discovered trajectories: ###
    BoundingBox_PerFrame_list = []
    for i in np.arange(number_of_trajectories):
        BoundingBox_PerFrame_list.append([])

    ### Loop over movie frames and see if there are trajectories found in these frames: ###
    current_frame_index = 0
    for frame_index in np.arange(0,number_of_frames_total,1):
        ### Loop Over Found Trajectories & Paint Circles On Them: ###
        for trajectory_index in range(number_of_trajectories):
            if (current_frame_index in t_vec[trajectory_index]):
                ### Find the index within current trajectory which includes current frame: ###
                current_frame_index_within_trajectory = np.where(current_frame_index == t_vec[trajectory_index])[0][0]

                ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
                target_spatial_coordinates = (np.int(trajectory_smoothed_polynom_XY[0][trajectory_index][current_frame_index_within_trajectory]),
                                              np.int(trajectory_smoothed_polynom_XY[1][trajectory_index][current_frame_index_within_trajectory]))

                ### Keep Track Of Bounding-Boxes (H_index, W_index): ###
                BoundingBox_PerFrame_list[trajectory_index].append((target_spatial_coordinates[1],target_spatial_coordinates[0]))
            else:
                ### If there isn't a drone trajectory found in current frame then simply put -1 instead: ###
                BoundingBox_PerFrame_list[trajectory_index].append(None)

        ### Skip Frame: ###
        current_frame_index += 1

    return BoundingBox_PerFrame_list


def Plot_BoundingBox_On_Movie(super_dict):
    ### Get Parameters From super_dict: ###
    Mov = super_dict.Mov
    fps = super_dict.fps
    tit = super_dict.tit
    Res_dir = super_dict.Res_dir
    flag_save_movie = super_dict.flag_save_movie
    trajectory_tuple = super_dict.trajectory_tuple
    BoundingBox_PerFrame_list = super_dict.BoundingBox_PerFrame_list

    ### Initialize Things: ###
    flag_found_trajectories = len(trajectory_tuple) == 3
    if flag_found_trajectories == False:
        return
    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    number_of_frames_total, H, W = Mov.shape
    number_of_frames_to_view = number_of_frames_total
    iax0 = 2
    iax1 = 1
    if flag_save_movie:
        OutVideoWriter = cv2.VideoWriter(Res_dir + tit + ".avi", cv2.VideoWriter_fourcc(*'XVID'), fps,
                                         (Mov.shape[iax0], Mov.shape[iax1]), 0)

    ### Loop Over Video Frames & Paint Them On Screen: ###
    for frame_index in np.arange(0, number_of_frames_total, 1):
        ### Get current movie frame: ###
        image_frame = Mov[frame_index, :, :]

        ### Loop over trajectories and for each trajectory see if there exists a valid trajectory in current frame and if so then draw circle: ###
        for trajectory_index in range(number_of_trajectories):
            ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
            target_spatial_coordinates = BoundingBox_PerFrame_list[trajectory_index][frame_index]

            if target_spatial_coordinates is not None:
                circle_radius_in_pixels = 10
                image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (0, 0, 255),
                                         1)
                image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (
                target_spatial_coordinates[0] - circle_radius_in_pixels,
                target_spatial_coordinates[1] - circle_radius_in_pixels), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                          cv2.LINE_AA)

        ### Actually Show Image With Circles On It: ###
        # cv2.imshow(tit, image_frame)
        # plt.imshow(image_frame)
        # plt.pause(0.001)

        ### Save Movie With Circles On It If Wanted: ###
        if flag_save_movie:
            OutVideoWriter.write(image_frame)

    ### Finish Movie Recording: ###
    if flag_save_movie:
        OutVideoWriter.release()

def numpy_array_to_video(input_tensor, video_name='my_movie.avi', FPS=25.0):
    T, H, W, C = input_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))
    for frame_counter in np.arange(T):
        current_frame = input_tensor[frame_counter]
        current_frame = (current_frame*255).astype(np.uint8)
        video_writer.write(current_frame)
    video_writer.release()

def BW2RGB(input_image):
    ### For Both Torch Tensors and Numpy Arrays!: ###
    # Actually... we're not restricted to RGB....
    if len(input_image.shape) == 2:
        if type(input_image) == torch.Tensor:
            RGB_image = input_image.unsqueeze(0)
            RGB_image = torch.cat([RGB_image,RGB_image,RGB_image], 0)
        elif type(input_image) == np.ndarray:
            RGB_image = np.atleast_3d(input_image)
            RGB_image = np.concatenate([RGB_image, RGB_image, RGB_image], -1)
        return RGB_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 0)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 1)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 2)
        else:
            RGB_image = input_image

    return RGB_image


def torch_array_to_video(input_tensor, video_name='my_movie.avi', FPS=25.0):
    T, C, H, W = input_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))
    for frame_counter in np.arange(T):
        current_frame = input_tensor[frame_counter]
        current_frame = current_frame.permute([1,2,0]).numpy()
        current_frame = BW2RGB(current_frame)
        current_frame = (current_frame * 255).astype(np.uint8)
        video_writer.write(current_frame)
    video_writer.release()

def Ctypi(Movie):
    full_frame_flg = True
    ctypi_fr_sz = 64
    ctypi_fr_sz_2 = np.int(ctypi_fr_sz / 2)
    if not full_frame_flg:
        feature_pts = cv2.goodFeaturesToTrack(np.uint8(np.float32(Movie[0]) / 3), maxCorners=200,
                                              qualityLevel=0.001, minDistance=3, blockSize=3)
        x = feature_pts.reshape((feature_pts.shape[0], 2))
        y_pred = KMeans(n_clusters=10, random_state=170).fit_predict(x)
        yind = np.where(y_pred == np.where(np.bincount(y_pred) == max(np.bincount(y_pred)))[0][0])[0]
        cnt_ = np.mean(x[yind, :], axis=0).astype('int')
        Movie = Movie[:, cnt_[1] - ctypi_fr_sz_2:cnt_[1] + ctypi_fr_sz_2,
                        cnt_[0] - ctypi_fr_sz_2:cnt_[0] + ctypi_fr_sz_2]
    Mov0 = Movie[0]

    c_shift_lp_for, c_shift_lp_back, c_shift_d_for, c_shift_d_back, reg_lst = creat_filt()

    sig = np.zeros((len(Movie), 2))
    for i in range(len(Movie)):
        # print('Ctypi iter num.' + str(i) + 'out of ' + str(len(Movie)))
        Movi = Movie[i]

        d0, d1, At, Bt = est0_corr(Mov0, Movi, d_max=6)
        sig[i] = d0 + d1
        # sig[i] = ctypi_1fr(At,Bt,d1,d0,reg_lst,c_shift_lp_for,c_shift_lp_back,c_shift_d_for,c_shift_d_back)
    return sig


def Ctypi_align(Movie):
    sig = Ctypi(Movie)
    number_of_frames = np.size(Movie, 0)
    xl = np.arange(np.size(Movie, 1))
    yl = np.arange(np.size(Movie, 2))

    Movie_stabilized = np.zeros(Movie.shape)
    for i in range(number_of_frames):
        f = scipy.interpolate.interp2d(yl, xl, Movie[i])
        Movie_stabilized[i] = f(yl + sig[i, 1], xl + sig[i, 0])
    return Movie_stabilized


def creat_filt():
    filt_len = 7
    reg_step = .1 / 4
    reg_lst = np.arange(0, 0.5, reg_step)

    L = 4096
    L_2 = np.int(L / 2)
    lv = np.arange(- np.int(np.floor(filt_len / 2)), np.int(np.floor(filt_len / 2)) + 1)
    lv_inv = np.arange(filt_len, 0, -1) - 1
    k = np.fft.fftshift(np.dot(np.dot(2j, np.pi), (np.arange(- 0.5, 0.5 - 1 / L, 1 / L))))
    mov_k = lambda d=None: np.exp(np.dot(k, d))
    mov = lambda d=None: np.fft.fftshift(np.fft.ifft(mov_k(d)))
    y = np.zeros(L - 1);
    y[1] = 1;
    y[-1] = -1
    mov_d = lambda d=None: np.fft.fftshift(np.fft.ifft(np.multiply(mov_k(d), np.fft.fft(y))))
    mov_lp = lambda d=None: np.fft.fftshift(
        np.fft.ifft(np.multiply(mov_k(d), np.sinc(np.fft.fftshift(np.arange(- 1, 1 - 2 / L, 2 / L))))))

    c_shift_lp_for = list(np.zeros(len(reg_lst))))
    c_shift_lp_back = list(np.zeros(len(reg_lst))))
    c_shift_d_for = list(np.zeros(len(reg_lst))))
    c_shift_d_back = list(np.zeros(len(reg_lst))))

    for ii in range(len(reg_lst)):
        ymov = np.real(mov(reg_lst[ii] / 2))
        yd = np.real(mov_d(reg_lst[ii] / 2))
        ylp = np.real(mov_lp(reg_lst[ii] / 2))
        clp_for = ylp[L_2 - 1 + lv] / sum(ylp[L_2 - 1 + lv])
        clp_back = clp_for[lv_inv]

        c_shift_lp_for[ii] = clp_for
        c_shift_lp_back[ii] = clp_back

        cd_for = yd[L_2 - 1 + lv] / sum(ymov[L_2 - 1 + lv])
        cd_back = - cd_for[lv_inv]
        c_shift_d_for[ii] = cd_for
        c_shift_d_back[ii] = cd_back

    return c_shift_lp_for, c_shift_lp_back, c_shift_d_for, c_shift_d_back, reg_lst


def est0_corr(A, B, d_max=6):
    A = np.float64(A)
    B = np.float64(B)
    sz1 = np.size(A, 0)
    sz2 = np.size(A, 1)
    spn_ax = np.arange(-d_max, d_max + 1)
    i_ax1 = np.arange(d_max, sz1 - d_max)
    i_ax2 = np.arange(d_max, sz2 - d_max)
    shft_arr = 2 * d_max + 1
    At = A[d_max:-d_max, d_max:-d_max]
    d_shft_pxl = list(np.zeros(shft_arr ** 2)))
    ABdiff_abs = np.zeros(shft_arr ** 2)
    cc = 0
    for ii1 in spn_ax:
        for jj1 in spn_ax:
            Bt = B[i_ax1 + ii1, :]
            Bt = Bt[:, i_ax2 + jj1]
            ABdiff_abs[cc] = np.sum(At * Bt) / (np.sum(At * At) * np.sum(Bt * Bt)) ** 0.5
            d_shft_pxl[cc] = [jj1, ii1]
            cc = cc + 1

    ABdiff_abs = ABdiff_abs.reshape((shft_arr, shft_arr))
    s1 = np.where(ABdiff_abs == np.max(ABdiff_abs))
    i1 = s1[0][0]
    j1 = s1[1][0]
    d0 = np.array([spn_ax[i1], spn_ax[j1]])

    d1_2 = - (ABdiff_abs[i1, j1 + 1] - ABdiff_abs[i1, j1 - 1]) / 2 / (
                - 2 * ABdiff_abs[i1, j1] + ABdiff_abs[i1, j1 + 1] + ABdiff_abs[i1, j1 - 1])
    d1_1 = - (ABdiff_abs[i1 + 1, j1] - ABdiff_abs[i1 - 1, j1]) / 2 / (
                - 2 * ABdiff_abs[i1, j1] + ABdiff_abs[i1 + 1, j1] + ABdiff_abs[i1 - 1, j1])
    d1 = np.array([d1_1, d1_2])

    i0 = max(abs(d0))
    i_ax1 = np.arange(i0, sz1 - i0)
    i_ax2 = np.arange(i0, sz2 - i0)

    Bt = np.float32(B[i_ax1, :])
    Bt = Bt[:, i_ax2]
    At = np.float32(A[i_ax1 - d0[0], :])
    At = At[:, i_ax2 - d0[1]]
    return d0, d1, At, Bt


def ctypi_1fr(At, Bt, d1, d0, reg_lst, c_shift_lp_for, c_shift_lp_back, c_shift_d_for, c_shift_d_back):
    L = c_shift_lp_for[0].shape[0]

    d0_reg = np.abs(d1[0]) - reg_lst
    d1_reg = np.abs(d1[1]) - reg_lst
    ilow0 = np.where(np.abs(d0_reg) == np.min(np.abs(d0_reg)))[0][0]
    ilow1 = np.where(np.abs(d1_reg) == np.min(np.abs(d1_reg)))[0][0]

    if d1[0] < 0:
        clpA0 = c_shift_lp_for[ilow0]
        clpB0 = c_shift_lp_back[ilow0]
        cdA0 = c_shift_d_for[ilow0]
        cdB0 = c_shift_d_back[ilow0]
    else:
        clpA0 = c_shift_lp_back[ilow0]
        clpB0 = c_shift_lp_for[ilow0]
        cdA0 = c_shift_d_back[ilow0]
        cdB0 = c_shift_d_for[ilow0]

    if d1[1] < 0:
        clpA1 = c_shift_lp_for[ilow1]
        clpB1 = c_shift_lp_back[ilow1]
        cdA1 = c_shift_d_for[ilow1]
        cdB1 = c_shift_d_back[ilow1]
    else:
        clpA1 = c_shift_lp_back[ilow1]
        clpB1 = c_shift_lp_for[ilow1]
        cdA1 = c_shift_d_back[ilow1]
        cdB1 = c_shift_d_for[ilow1]

    clpA0 = clpA0.reshape(L, 1)
    clpA1 = clpA1.reshape(L, 1)
    clpB0 = clpB0.reshape(L, 1)
    clpB1 = clpB1.reshape(L, 1)
    cdA0 = cdA0.reshape(L, 1)
    cdA1 = cdA1.reshape(L, 1)
    cdB0 = cdB0.reshape(L, 1)
    cdB1 = cdB1.reshape(L, 1)

    Alp = signal.convolve2d(signal.convolve2d(At, clpA0, 'valid'), clpA1.T, 'valid')
    Blp = signal.convolve2d(signal.convolve2d(Bt, clpB0, 'valid'), clpB1.T, 'valid')
    Adx = signal.convolve2d(signal.convolve2d(At, cdA0, 'valid'), clpA1.T, 'valid')
    Bdx = signal.convolve2d(signal.convolve2d(Bt, cdB0, 'valid'), clpB1.T, 'valid')
    Ady = signal.convolve2d(signal.convolve2d(At, clpA0, 'valid'), cdA1.T, 'valid')
    Bdy = signal.convolve2d(signal.convolve2d(Bt, clpB0, 'valid'), cdB1.T, 'valid')

    dshcr = np.array([reg_lst[ilow0] * np.sign(d1[0]), reg_lst[ilow1] * np.sign(d1[1])])

    px = Bdx / 4 + Adx / 4
    py = Bdy / 4 + Ady / 4
    ABtx = Alp - Blp
    p = np.array([np.array(px.flat), np.array(py.flat)]).T
    x = np.array(ABtx.flat).reshape(len(ABtx.flat), 1)

    d2 = np.array(np.linalg.lstsq(- p, x, rcond=None)[0].flat) + dshcr
    d = d0 + d2
    return d









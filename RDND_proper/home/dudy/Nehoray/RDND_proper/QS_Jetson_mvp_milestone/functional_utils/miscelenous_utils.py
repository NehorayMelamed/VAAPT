import matplotlib.path
import numpy as np
import torch
import torch.nn as nn
import scipy
from PIL.ImageOps import scale
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
# import torch.fft
import kornia

import torch
import torch.fft
from tqdm.asyncio import tarange

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
import pandas as pd

from RapidBase.import_all import *


##################################################################################################################################################################
### Some Auxiliaries: ###
def create_folder_if_doesnt_exist(folder_full_path):
    if not os.path.exists(folder_full_path):
        os.makedirs(folder_full_path)

def write_params_to_txt_file(params):
    ### Get parameters from params dict: ###
    FileName = params['FileName']
    DistFold = params['DistFold']
    ResultsFold = params['ResultsFold']
    roi = params['roi']
    utype = params['utype']

    ### make resuls path if none exists: ###
    filename_itself = os.path.split(FileName)[-1]
    experiment_folder = os.path.split(FileName)[0]
    Res_dir = os.path.join(experiment_folder, 'Results')
    text_full_filename = os.path.join(Res_dir, 'params.txt')
    create_folder_if_doesnt_exist(Res_dir)

    ### write down current parameters into params.txt format: ###
    open(text_full_filename, 'w+').write(str(params))

def get_binary_file_reader(params):
    ### Get parameters from params dict: ###
    FileName = params['FileName']
    DistFold = params['DistFold']
    roi = params['roi']
    utype = params['utype']

    ### Open movie, which is in binary format: ###
    f = open(FileName, "rb")

    return f


def initialize_experiment(experiment_filenames_list, experiment_index, params):
    ### Make sure filename is a .bin file: ###
    current_experiment_filename = experiment_filenames_list[experiment_index]
    if not current_experiment_filename[-3:] == 'Bin':
        flag_bin_file = False
    else:
        flag_bin_file = True

    if flag_bin_file:
        params['FileName'] = experiment_filenames_list[experiment_index]

        ### Make Results Path: ##
        results_folder, experiment_folder, filename_itself, params = take_care_of_paths(params)

        ### initialize a .txt file with a summation of the results: ###
        results_summary_text_full_filename = os.path.join(results_folder, 'res_summ.txt')
        if os.path.isfile(results_summary_text_full_filename):
            os.remove(results_summary_text_full_filename)

        ### Initialize results_summary_string: ###
        results_summary_string = ''

        return flag_bin_file, current_experiment_filename, results_folder, experiment_folder, filename_itself, params, results_summary_string
    else:
        return flag_bin_file, None, None, None, None, None, None

def initialize_binary_file_reader(f):
    f.close()
    f = open(f.name, 'rb')
    return f

def take_care_of_paths(params):
    ### Get parameters from params dict: ###
    FileName = params['FileName']
    DistFold = params['DistFold']
    # roi = params['roi']
    utype = params['utype']

    ### make resuls path if none exists: ###
    filename_itself = os.path.split(FileName)[-1]
    experiment_folder = os.path.split(FileName)[0]
    results_folder = os.path.join(experiment_folder, 'Results')
    create_folder_if_doesnt_exist(results_folder)

    params.results_folder = results_folder
    params.experiment_folder = experiment_folder

    return results_folder, experiment_folder, filename_itself, params


def get_shifted_intersection_points(p_intersection_point, delta, H, W):
    if p_intersection_point[0] == W or p_intersection_point[0] == 0:
        delta_x = 0
        delta_y = delta
    elif p_intersection_point[1] == H or p_intersection_point[1] == 0:
        delta_x = delta
        delta_y = 0
    ### Shift X: ###
    intersection_point_plus_delta = (p_intersection_point[0] + delta_x, p_intersection_point[1])
    intersection_point_minus_delta = (p_intersection_point[0] - delta_x, p_intersection_point[1])
    ### Shift Y: ###
    intersection_point_plus_delta = (intersection_point_plus_delta[0], intersection_point_plus_delta[1] + delta_y)
    intersection_point_minus_delta = (intersection_point_minus_delta[0], intersection_point_minus_delta[1] - delta_y)
    ### Clip To Frame Sizes: ###
    intersection_point_plus_delta = (min(intersection_point_plus_delta[0], W), intersection_point_plus_delta[1])
    intersection_point_minus_delta = (max(intersection_point_minus_delta[0], 0), intersection_point_minus_delta[1])
    intersection_point_plus_delta = (intersection_point_plus_delta[0], min(intersection_point_plus_delta[1], H))
    intersection_point_minus_delta = (intersection_point_minus_delta[0], max(intersection_point_minus_delta[1], 0))
    ### Round and make int for later processing with OpenCV functions expecting integers: ###
    intersection_point_plus_delta = (np.int(intersection_point_plus_delta[0]), np.int(intersection_point_plus_delta[1]))
    intersection_point_minus_delta = (np.int(intersection_point_minus_delta[0]), np.int(intersection_point_minus_delta[1]))
    p_intersection_point = (np.int(p_intersection_point[0]), np.int(p_intersection_point[1]))
    ### Get Line from shifted intersection points: ###
    new_intersection_points_line = get_line_ABC_parameters_from_points(intersection_point_plus_delta,
                                                     intersection_point_minus_delta)
    ### Get intersection points tuple for ease of use later on: ###
    intersection_points_tuple = (intersection_point_plus_delta, intersection_point_minus_delta)
    return intersection_point_plus_delta, intersection_point_minus_delta, p_intersection_point, new_intersection_points_line, intersection_points_tuple


def make_tuple_int(input_tuple):
    input_tuple = (np.int(np.round(input_tuple[0])), np.int(np.round(input_tuple[1])))
    return input_tuple


def split_string_to_name_and_value(input_string):
    split_index = str.find(input_string, ':')
    variable_name = input_string[0:split_index].replace(' ', '')
    variable_value = input_string[split_index + 1:].replace(' ', '')
    return variable_name, variable_value


def get_experiment_description_from_txt_file(params):
    ### Get Lines From Description File: ###
    experiment_description_text_file_full_filename = os.path.join(params.experiment_folder, 'Description.txt')
    experiment_description_string_list = open(experiment_description_text_file_full_filename).readlines()
    for line_index in np.arange(len(experiment_description_string_list)):
        if '\n' in experiment_description_string_list[line_index]:
            experiment_description_string_list[line_index] = experiment_description_string_list[line_index][0:-1]  # get rid of \n (line break)
        else:
            experiment_description_string_list[line_index] = experiment_description_string_list[line_index]
        key, value = split_string_to_name_and_value(experiment_description_string_list[line_index])
        print(key + ':    ' + str(value))
        params[key] = value

    return params


def Update_QS_DataFrame(QS_stats_pd, params, flag_drone_trajectory_inside_BB_list):
    # TODO: at the end it will need to loop on all actual drones present...so i guess on all flashlight trajectories found!
    # TODO: perhaps instead of "flag_drone_in_BB" which only check one large bounding box, i should first squeeze the bounding box or check direction vec???
    # TODO: i shuold really understand what to do about drone swarms
    if len(flag_drone_trajectory_inside_BB_list) > 0:
        #(*). Trajectories were found, fill stuff accordingly:
        for trajectory_index in np.arange(len(flag_drone_trajectory_inside_BB_list)):
            current_pd = {"distance": [np.float(params.distance)],
                          "drone_type": [params.drone_type],
                          "drone_movement": [params.drone_movement],
                          "background_scene": [params.background_scene],
                          "flag_drone_in_BB": [flag_drone_trajectory_inside_BB_list[trajectory_index]],
                          'flag_was_there_drone': (params.was_there_drone == 'True'),
                          'flag_was_drone_detected': True,
                          'experiment_name': os.path.split(params.experiment_folder)[-1],
                          'full_filename_sequence': params.results_folder_seq
                          }
            current_pd = pd.DataFrame.from_dict(current_pd)
            if len(QS_stats_pd) == 0:
                QS_stats_pd = pd.DataFrame.from_dict(current_pd)
            else:
                QS_stats_pd = QS_stats_pd.append(current_pd, ignore_index=True)
    else:
        #(*). No Trajectories were found, fill stuff accordingly:
        current_pd = {"distance": [params.distance],
                      "drone_type": [params.drone_type],
                      "drone_movement": [params.drone_movement],
                      "background_scene": [params.background_scene],
                      "flag_drone_in_BB": False,
                      'flag_was_there_drone': (params.was_there_drone == 'True'),
                      'flag_was_drone_detected': False,
                      'experiment_name': os.path.split(params.experiment_folder)[-1],
                      'full_filename_sequence': params.results_folder_seq
                      }
        current_pd = pd.DataFrame.from_dict(current_pd)
        if len(QS_stats_pd) == 0:
            QS_stats_pd = pd.DataFrame.from_dict(current_pd)
        else:
            QS_stats_pd = QS_stats_pd.append(current_pd, ignore_index=True)

    return QS_stats_pd


def play_with_DataFrame(QS_stats_pd):
    QS_stats_pd.head()
    QS_stats_pd.tail()
    QS_stats_pd.index  # display indices
    QS_stats_pd.columns
    QS_stats_pd.to_numpy()
    QS_stats_pd.describe()
    QS_stats_pd.sort_index(axis=1, ascending=False)
    QS_stats_pd.sort_values(by='distance')
    QS_stats_pd['distance']
    QS_stats_pd[0:1]  # select
    QS_stats_pd.loc[:, ['distance', 'drone_type']]  # select by label
    QS_stats_pd.iloc[1]  # row selection
    QS_stats_pd.iloc[[0], [0, 1]]  # row and column selection
    QS_stats_pd[QS_stats_pd['distance'] == '1500']  # select by condition
    QS_stats_pd.mean()
    QS_stats_pd.columns = [x.lower() for x in QS_stats_pd.columns]  # change column names
    QS_stats_pd['distance'] = QS_stats_pd['distance'].astype(np.float)  # transform to float
    # (*). get only those from distance 1500:
    QS_stats_pd[QS_stats_pd['distance'] == 1500]
    # (*). get only those with urban background:
    QS_stats_pd[QS_stats_pd['background_scene'] == 'urban']
    # (*). get only those with urban background and distance from 1500-2000:
    QS_stats_pd[QS_stats_pd['background_scene'] == 'urban'][QS_stats_pd['distance'] == 1500]
    # (*). get only those where no drone was present (for 24/7 FAR statistics):
    QS_stats_pd[QS_stats_pd['flag_was_there_drone'] == 'True']  # TODO: turn this into boolean value for fuck sake
    # (*). get only those where no drone was detected but drone was present:
    QS_stats_pd[QS_stats_pd['flag_was_there_drone'] == 'True'][QS_stats_pd['flag_was_drone_detected'] == True]  # TODO: why is it boolean in one place and string on the other
    # (*). for those with distance 1500 get confusion matrix
    QS_stats_pd == 'True'


def save_DataFrame_to_csv(input_df, full_filename):
    input_df.to_csv(full_filename)


def load_DataFrame_from_csv(full_filename):
    return pd.read_csv(full_filename).iloc[:, 1:]


def save_DataFrame_DifferentMethods():
    QS_stats_pd.to_csv(os.path.join(params.ExperimentsFold, 'QS_stats_pd.csv'))
    pd.read_csv(os.path.join(params.ExperimentsFold, 'QS_stats_pd.csv')).iloc[:, 1:]

    ### Save/Load From Disk Using Jason: ###
    QS_stats_pd.to_pickle(os.path.join(params.ExperimentsFold, 'QS_stats_pd.json'))
    QS_stats_pd = pd.read_pickle(os.path.join(params.ExperimentsFold, 'QS_stats_pd.json'))

    ### Save/Load From Disk Using HDF: ###
    QS_stats_pd.to_hdf(os.path.join(params.ExperimentsFold, 'QS_stats_pd.h5'), 'table', append=True)
    pd.HDFStore(os.path.join(params.ExperimentsFold, 'QS_stats_pd.h5')).append('Table', QS_stats_pd)

    ### Save Updated DataFrame To Disk Using numpy: ###
    np.save(os.path.join(params.ExperimentsFold, 'QS_stats_pd.npy'), QS_stats_pd, allow_pickle=True)
    np.save(os.path.join(params.ExperimentsFold, 'QS_stats_pd_columns.npy'), QS_stats_pd.columns, allow_pickle=True)
    QS_stats_pd = np.load(os.path.join(params.ExperimentsFold, 'QS_stats_pd.npy'), allow_pickle=True)
    QS_stats_pd_columns = np.load(os.path.join(params.ExperimentsFold, 'QS_stats_pd_columns.npy'), allow_pickle=True)
    QS_stats_pd = pd.DataFrame(QS_stats_pd)
    QS_stats_pd.columns = QS_stats_pd_columns


def get_flashlight_line_and_intersection_with_frame_parameters(input_image, flashlight_p1, flashlight_p2,
                                                               delta_pixel_flashlight_area_side=20):
    ### Get Image Parameters: ###w
    H, W = input_image.shape

    ### Add a tiny tiny little offset between points to avoid a rare problem where the two points form a completly parallel line to one of the frames: ###
    flashlight_p2 = list(flashlight_p2)
    flashlight_p2[0] = flashlight_p2[0] + 1e-3
    flashlight_p2[1] = flashlight_p2[1] + 1e-3
    flashlight_p2 = tuple(flashlight_p2) #TODO: understand why am i even dealing with tuples instead of lists!?!?!?

    # (*). Image Frame Line Parameters:
    upper_left_point = (0, 0)
    upper_right_point = (W, 0)
    bottom_left_point = (0, H)
    bottom_right_point = (W, H)
    bottom_frame_line = get_line_ABC_parameters_from_points(bottom_left_point, bottom_right_point)
    left_frame_line = get_line_ABC_parameters_from_points(upper_left_point, bottom_left_point)
    right_frame_line = get_line_ABC_parameters_from_points(upper_right_point, bottom_right_point)
    upper_frame_line = get_line_ABC_parameters_from_points(upper_left_point, upper_right_point)

    # (*). Flashlight Direction Line Parameters
    A, B, C = get_line_ABC_parameters_from_points(flashlight_p1, flashlight_p2)
    flashlight_line = (A, B, C)
    flashlight_line_slop_m = get_line_slop_from_points(flashlight_p1, flashlight_p2)

    # (*). Find Intersection Between Flashlight Line And Image Frame:
    #TODO: instead of stretching the flashlight line indefinitely, have a variable "flashlight_line_length" in pixels and use THAT to get the polygon
    #TODO: instead of the image frame i can use any other line i want...for instance a line in the middle of the image...or something defined by the flashlight line extended itself
    #(*). intersection points of flashlight with frame lines (not necessarily frame itself, the lines are infinite):
    p_intersection_left = get_intersection_point_between_two_lines_defined_by_ABC(flashlight_line, left_frame_line)
    p_intersection_right = get_intersection_point_between_two_lines_defined_by_ABC(flashlight_line, right_frame_line)
    p_intersection_upper = get_intersection_point_between_two_lines_defined_by_ABC(flashlight_line, upper_frame_line)
    p_intersection_bottom = get_intersection_point_between_two_lines_defined_by_ABC(flashlight_line, bottom_frame_line)
    #(*). Round Points To Nearest Int:
    p_intersection_left = make_tuple_int(p_intersection_left)
    p_intersection_right = make_tuple_int(p_intersection_right)
    p_intersection_upper = make_tuple_int(p_intersection_upper)
    p_intersection_bottom = make_tuple_int(p_intersection_bottom)
    #(*). understand where in the frame the flashlight line actually hits:
    flag_p_intersection_left_inside_image = p_intersection_left[1] >= 0 and p_intersection_left[1] <= H
    flag_p_intersection_right_inside_image = p_intersection_right[1] >= 0 and p_intersection_right[1] <= H
    flag_p_intersection_upper_inside_image = p_intersection_upper[0] >= 0 and p_intersection_upper[0] <= W
    flag_p_intersection_bottom_inside_image = p_intersection_bottom[0] >= 0 and p_intersection_bottom[0] <= W
    flag_p_intersection_with_image_up_right = flag_p_intersection_upper_inside_image and flag_p_intersection_right_inside_image
    flag_p_intersection_with_image_up_left = flag_p_intersection_upper_inside_image and flag_p_intersection_left_inside_image
    flag_p_intersection_with_image_bottom_right = flag_p_intersection_bottom_inside_image and flag_p_intersection_right_inside_image
    flag_p_intersection_with_image_bottom_left = flag_p_intersection_bottom_inside_image and flag_p_intersection_left_inside_image
    flag_p_intersection_with_image_left_right = flag_p_intersection_left_inside_image and flag_p_intersection_right_inside_image
    flag_p_intersection_with_image_up_bottom = flag_p_intersection_upper_inside_image and flag_p_intersection_bottom_inside_image
    #(*). Exapnd flashlight line to the sides according to where it hit:
    p_intersection_upper_plus_delta, p_intersection_upper_minus_delta, p_intersection_upper, intersection_line_upper, intersection_points_tuple_upper =\
        get_shifted_intersection_points(p_intersection_upper, delta_pixel_flashlight_area_side, H, W)
    p_intersection_bottom_plus_delta, p_intersection_bottom_minus_delta, p_intersection_bottom, intersection_line_bottom, intersection_points_tuple_bottom = \
        get_shifted_intersection_points(p_intersection_bottom, delta_pixel_flashlight_area_side, H, W)
    p_intersection_right_plus_delta, p_intersection_right_minus_delta, p_intersection_right, intersection_line_right, intersection_points_tuple_right = \
        get_shifted_intersection_points(p_intersection_right, delta_pixel_flashlight_area_side, H, W)
    p_intersection_left_plus_delta, p_intersection_left_minus_delta, p_intersection_left, intersection_line_left, intersection_points_tuple_left = \
        get_shifted_intersection_points(p_intersection_left, delta_pixel_flashlight_area_side, H, W)
    #(*). Get Polygon Inside Defined By flashlight expanded line and frame:
    if flag_p_intersection_with_image_up_right:
        polygon_points_tuple = (intersection_points_tuple_upper[0], intersection_points_tuple_upper[1], intersection_points_tuple_right[1], intersection_points_tuple_right[0])
    elif flag_p_intersection_with_image_up_left:
        polygon_points_tuple = (intersection_points_tuple_upper[0], intersection_points_tuple_upper[1], intersection_points_tuple_left[0], intersection_points_tuple_left[1])
    elif flag_p_intersection_with_image_bottom_right:
        polygon_points_tuple = (intersection_points_tuple_bottom[0], intersection_points_tuple_bottom[1], intersection_points_tuple_right[0], intersection_points_tuple_right[1])
    elif flag_p_intersection_with_image_bottom_left:
        polygon_points_tuple = (intersection_points_tuple_bottom[0], intersection_points_tuple_bottom[1], intersection_points_tuple_left[1], intersection_points_tuple_left[0])
    elif flag_p_intersection_with_image_left_right:
        polygon_points_tuple = (intersection_points_tuple_left[0], intersection_points_tuple_left[1], intersection_points_tuple_right[1], intersection_points_tuple_right[0])
    elif flag_p_intersection_with_image_up_bottom:
        polygon_points_tuple = (intersection_points_tuple_upper[0], intersection_points_tuple_upper[1], intersection_points_tuple_bottom[1], intersection_points_tuple_bottom[0])
    # (*). Define Polygon Using The Vertices of the image frame intersection points:
    polygon_points_list = list(polygon_points_tuple)
    polygon_points = np.atleast_2d(polygon_points_tuple)
    polygon_path = matplotlib.path.Path(polygon_points)

    return flashlight_line, flashlight_line_slop_m, \
           intersection_points_tuple_upper, intersection_points_tuple_left, intersection_points_tuple_right, intersection_points_tuple_bottom,\
           polygon_path, polygon_points_list


def check_if_line_is_in_polygon(polygon, drone_locations_t_list, drone_BoundingBox_PerFrame_list):
    if type(polygon) == list or type(polygon) == tuple:
        polygon = np.atleast_2d(polygon)
        polygon = matplotlib.path.Path(polygon)
    else:
        1  # entered matplotlib.path.Path object!!!

    ### Loop over the different frames, and for each image check if line point is inside predifined polygon: ###
    flag_is_drone_in_polygon_vec = []
    for frame_index in drone_locations_t_list:
        current_point = drone_BoundingBox_PerFrame_list[frame_index]
        current_point = np.atleast_2d(current_point)
        flag_is_drone_in_polygon = polygon.contains_points(current_point)[0]
        flag_is_drone_in_polygon_vec.append(flag_is_drone_in_polygon)

    ### Decide If drone is considered inside "GT" Bounding-Box as derived by the flashlight: ###
    is_drone_inside_polygon = sum(flag_is_drone_in_polygon_vec) > len(flag_is_drone_in_polygon_vec) * 1/3

    return flag_is_drone_in_polygon_vec, is_drone_inside_polygon


def get_only_trajectories_which_passed_frequency_decision(t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, frequency_decision_vec):
    t_vec_temp = []
    trajectory_smoothed_polynom_X_temp = []
    trajectory_smoothed_polynom_Y_temp = []
    xyz_line_temp = []
    for trajectory_index in np.arange(len(frequency_decision_vec)):
        flag_current_trajectory_valid = frequency_decision_vec[trajectory_index]
        if flag_current_trajectory_valid:
            t_vec_temp.append(t_vec[trajectory_index])
            trajectory_smoothed_polynom_X_temp.append(trajectory_smoothed_polynom_X[trajectory_index])
            trajectory_smoothed_polynom_Y_temp.append(trajectory_smoothed_polynom_Y[trajectory_index])
            xyz_line_temp.append(xyz_line[trajectory_index])
    return t_vec_temp, trajectory_smoothed_polynom_X_temp, trajectory_smoothed_polynom_Y_temp, xyz_line_temp

def check_if_drone_trajectory_is_inside_flashlight_GT_BB(flashlight_frame_polygon_points, drone_t_vec, drone_BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list):
    #TODO: transfer this to an outside function which, if there's no flashlight in the image, checks if the drone (if one was found) is inside the flashlight polygon
    #(*). Check if drone estimated location is inside polygon:
    flag_is_drone_in_polygon_vec, is_drone_inside_polygon = check_if_line_is_in_polygon(flashlight_frame_polygon_points,
                                                               drone_t_vec,  #TODO: should be a long vec, not the entire big multidimensional list, which should be changed to something else!!!!
                                                               drone_BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list)  #TODO: should be a long vec, not the entire big multidimensional list, which should be changed to something else!!!!
    return flag_is_drone_in_polygon_vec, is_drone_inside_polygon

def numpy_array_to_video(input_tensor, video_name='my_movie.avi', FPS=25.0):
    T, H, W, C = input_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))
    for frame_counter in np.arange(T):
        current_frame = input_tensor[frame_counter]
        current_frame = (current_frame*255).clip(0,255).astype(np.uint8)
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


def gaussian_function(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def ScrFunc(DetNoiseThresh, TrjMovFFTBP1):
    return 1./(1+np.exp(5*(DetNoiseThresh-TrjMovFFTBP1)))

def ScrFunc_Torch(DetNoiseThresh, TrjMovFFTBP1):
    return 1./(1+torch.exp(5*(DetNoiseThresh-TrjMovFFTBP1)))  #why the factor 5??!?!? what's going on here...why not simply use the logistic function


def scale_array_to_range(mat_in, min_max_values=(0,1)):
    mat_in_normalized = (mat_in - mat_in.min()) / (mat_in.max()-mat_in.min()) * (min_max_values[1]-min_max_values[0]) + min_max_values[0]
    return mat_in_normalized

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


# def Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(input_tensor, reference_tensor=None, device='cuda'):
#     ### the function expects input_tensor to be a numpy array of shape [T,H,W]: ###
#
#     ### Initialization: ###
#     if type(input_tensor) == np.ndarray:
#         input_tensor = torch.Tensor(input_tensor)
#         if device is not None:
#             input_tensor = input_tensor.to(device)
#         flag_type = 'numpy'
#     else:
#         flag_type = 'torch'
#     if len(input_tensor.shape) == 3:
#         input_tensor = input_tensor.unsqueeze(0)
#     shift_layer = Shift_Layer_Torch()
#     B, C, H, W = input_tensor.shape
#     number_of_samples = B
#
#     def fit_polynomial_torch(x, y):
#         # solve for 2nd degree polynomial deterministically using three points
#         a = (y[:, 2] + y[:, 0] - 2 * y[:, 1]) / 2
#         b = -(y[:, 0] + 2 * a * x[1] - y[:, 1] - a)
#         c = y[:, 1] - b * x[1] - a * x[1] ** 2
#         return [c, b, a]
#
#     #########################################################################################################
#     ### Get CC: ###
#     # input tensor [T,H,W]
#     B, T, H, W = input_tensor.shape
#     cross_correlation_shape = (H, W)
#     midpoints = (np.fix(H / 2), np.fix(W / 2))
#     input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1, -2])
#
#     ### Get reference tensor fft: ###
#     if reference_tensor is None:
#         reference_tensor_fft = input_tensor_fft[:, T // 2:T//2+1, :, :]
#     else:
#         if device is not None:
#             reference_tensor = reference_tensor.to(device)
#         reference_tensor_fft = torch.fft.fftn(reference_tensor, dim=[-1,-2])
#
#     # (*). Circular Cross Corerlation Using FFT:
#     output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * reference_tensor_fft.conj(), dim=[-1, -2]).real
#     # output_CC = torch.cat()
#
#     #########################################################################################################
#     ### Get Correct Max Indices For Fit: ###
#     output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H * W), dim=-1).unsqueeze(-1)
#     i0 = output_CC_flattened_indices // W
#     i1 = output_CC_flattened_indices - i0 * W
#     i0[i0 > midpoints[0]] -= H
#     i1[i1 > midpoints[1]] -= W
#     i0_original = i0 + 0
#     i1_original = i1 + 0
#     i0_minus1 = i0 - 1
#     i0_plus1 = i0 + 1
#     i1_minus1 = i1 - 1
#     i1_plus1 = i1 + 1
#     ### Correct For Wrap-Arounds: ###
#     # (1). Below Zero:
#     i0_minus1[i0_minus1 < 0] += H
#     i1_minus1[i1_minus1 < 0] += W
#     i0_plus1[i0_plus1 < 0] += H
#     i1_plus1[i1_plus1 < 0] += W
#     i0[i0 < 0] += H
#     i1[i1 < 0] += W
#     # (2). Above Max:
#     i0[i0 > H] -= H
#     i1[i1 > H] -= W
#     i0_plus1[i0_plus1 > H] -= H
#     i1_plus1[i1_plus1 > W] -= W
#     i0_minus1[i0_minus1 > W] -= H
#     i1_minus1[i1_minus1 > W] -= W
#     ### Get Flattened Indices From Row/Col Indices: ###
#     output_CC_flattened_indices_i1 = i1 + i0 * W
#     output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * W
#     output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * W
#     output_CC_flattened_indices_i0 = i1 + i0 * W
#     output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
#     output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
#     #########################################################################################################
#
#     #########################################################################################################
#     ### Get Proper Values For Fit: ###
#     # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
#     output_CC = output_CC.contiguous().view(-1, H * W)
#     output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
#     output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
#     output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
#     output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
#     output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
#     output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
#     #########################################################################################################
#
#     #########################################################################################################
#     ### Get Sub Pixel Shifts: ###
#     fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
#     fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
#     x_vec = [-1, 0, 1]
#     y_vec = [-1, 0, 1]
#     # fit a parabola over the CC values: #
#     [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
#     [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
#     # find the sub-pixel max value and location using the parabola coefficients: #
#     delta_shiftx = -b_x / (2 * a_x)
#     x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
#     delta_shifty = -b_y / (2 * a_y)
#     y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
#     # Add integer shift:
#     shiftx = i1_original.squeeze() + delta_shiftx
#     shifty = i0_original.squeeze() + delta_shifty
#     # print(shiftx)
#     # print(real_shifts_to_center_frame)
#     #########################################################################################################
#
#     #########################################################################################################
#     ### Align Images: ###
#     shifted_tensors = shift_layer.forward(input_tensor.permute([1, 0, 2, 3]), -shiftx, -shifty)  # TODO: instead of permuting make shift_layer accept [1,T,H,W]
#     mean_frame_averaged = shifted_tensors.mean(0, True)
#     # print(input_tensor.shape)
#     # imshow_torch(input_tensor[:,number_of_samples//2])
#     # imshow_torch(mean_frame_averaged[number_of_samples//2])
#     #########################################################################################################
#
#     if flag_type == 'numpy':
#         return shifted_tensors[:, 0, :, :].cpu().numpy()
#     else:
#         return shifted_tensors[:,0,:,:]


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

    c_shift_lp_for = list(np.zeros(len(reg_lst)))
    c_shift_lp_back = list(np.zeros(len(reg_lst)))
    c_shift_d_for = list(np.zeros(len(reg_lst)))
    c_shift_d_back = list(np.zeros(len(reg_lst)))

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
    d_shft_pxl = list(np.zeros(shft_arr ** 2))
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



def z_score_filter_avg_BORIS(nparray, harris_thresholdold=3.5):
    #z-score filter
    mean_int = np.mean(nparray)
    if (mean_int == 0):
        return 0
    std_int = np.std(nparray)
    z_scores = (nparray - mean_int) / std_int

    count = 0
    avg = 0
    for k in range(0,len(z_scores)):
        if abs(z_scores[k]) < harris_thresholdold:
            avg += z_scores[k]
            count += 1
    return int(avg/count)

def calc_images_translation_BORIS(img1, img2, numX, numY, stp):
#nimg1 and nimg2 are images to compare. numX, and numY are the number of search ROIs per image. step is the size of the ROI.
    tX = img1.shape[0]
    tY = img1.shape[1]
    # logging.debug("Dividing image to " + str(numX*numY) +" parts. Croping a " + str(2*stp) + "*" + str(2*stp) + " search pattern") #debug
    horizontal = np.zeros(numX*numY, int)
    vertical = np.zeros(numX*numY, int)
    for i in range(0,numX):
        for j in range(0,numY):
            #divide image into segments and calc positions of ROIs to search
            imTMP = img1[int((tX/numX)*i) + int(tX/(2*numX))-stp:int((tX/numX)*i) + int(tX/(2*numX))+stp, int((tY/numY)*j) + int(tY/(2*numY))-stp:int((tY/numY)*j) + int(tY/(2*numY))+stp]
            #do the work
            result = cv2.matchTemplate((img2*255).clip(0,255).astype(np.uint8), (imTMP*255).clip(0,255).astype(np.uint8), cv2.TM_CCORR_NORMED)
            _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
            yshift = maxLoc[0] - (int((tY/numY)*j) + int(tY/(2*numY))-stp)
            xshift = maxLoc[1] - (int((tX/numX)*i) + int(tX/(2*numX))-stp)
            vertical = np.append(vertical, yshift)
            horizontal = np.append(horizontal, xshift)
            # logging.debug ("Crop i:" + str(i) + " j:" + str(j) + " is shifted by x:" + str(xshift) + " y:"+str(yshift)) #debug

    #select the best x,y values from all ROIs
    return (z_score_filter_avg_BORIS(horizontal), z_score_filter_avg_BORIS(vertical))

def translate_BORIS(image, x, y):
    # move and image, define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted

def get_images_translation_and_translate_BORIS(image1, image2):
    transx, transy = calc_images_translation_BORIS(image1, image2, 4, 4, 10)
    image2_shifted = translate_BORIS(image2, transx, transy)
    return image2_shifted

def get_image_translation_and_translate_batch_BORIS(image_batch):
    T,H,W = image_batch.shape
    reference_tensor = image_batch[T//2,:,:]
    aligned_tensors = []
    for time_index in np.arange(T):
        current_tensor = image_batch[time_index,:,:]
        current_tensor_shifted = get_images_translation_and_translate_BORIS(reference_tensor, current_tensor)
        aligned_tensors.append(np.expand_dims(current_tensor_shifted,0))

    aligned_tensors = np.concatenate(aligned_tensors, 0)
    return aligned_tensors

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift_torch_specific_dim(X, dim):
    return roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim]/2)))



########################################################################################################################################################################
### Binary file reader auxiliaries: ###
def read_movie_sample(params, number_of_frames_to_read=None, frame_to_start_from=None):
    ### Get parameters from params dict: ###
    FileName = params['FileName']
    DistFold = params['DistFold']
    ResultsFold = params['ResultsFold']
    roi = params['roi']
    utype = params['utype']

    ### make resuls path if none exists: ###
    filename_itself = os.path.split(FileName)[-1]
    experiment_folder = os.path.split(FileName)[0]
    Res_dir = os.path.join(experiment_folder, 'Results')
    text_full_filename = os.path.join(Res_dir, 'params.txt')
    create_folder_if_doesnt_exist(Res_dir)

    ### Open movie, which is in binary format: ###
    f = open(FileName, "rb")

    ### Read entire movie from binary file into numpy array: ###
    #(1). how many frames to read:
    if number_of_frames_to_read is not None:
        total_number_of_frames = number_of_frames_to_read
    elif 'number_of_frames_to_read' in params.keys():
        total_number_of_frames = params['number_of_frames_to_read']
    else:
        total_number_of_frames = -1
    #(2). which frame to start from:
    if frame_to_start_from is not None:
        frame_to_start_from = frame_to_start_from
    if 'frame_to_start_from' in params.keys():
        frame_to_start_from = params['frame_to_start_from']
    else:
        frame_to_start_from = 0

    ### Assign binary reader to dictionary for later use: ###
    params.f = f

    Mov = np.fromfile(f, dtype=utype, count=total_number_of_frames*roi[0]*roi[1], offset=frame_to_start_from*roi[0]*roi[1]*2)
    Movie_len = np.int(len(Mov)/roi[0]/roi[1])
    number_of_elements = Movie_len * roi[0] * roi[1]
    Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])

    Movie = torch.Tensor(Mov.astype(np.float))
    # imagesc_hist_torch(Movie[0].unsqueeze(0))
    # Movie_bla = Movie[2:-1,2:-1].clamp(Movie[0][2:-1,2:-1].quantile(0.01), Movie[0][2:-1,2:-1].quantile(0.99))
    # Movie_bla = scale_array_to_range(Movie_bla)
    # Movie_bla = (Movie_bla*255).type(torch.uint8)
    # Movie_bla = BW2RGB(Movie_bla.unsqueeze(1))
    # imshow_torch_video(Movie_bla, 50)
    # plt.show()
    return Mov, Res_dir


def read_frames_from_binary_file_stream(f, number_of_frames_to_read=-1, number_of_frames_to_skip=0, params=None):
    ### Get parameters from params dict: ###
    FileName = params['FileName']
    DistFold = params['DistFold']
    roi = params['roi']
    utype = params['utype']

    ### Read Frames: ###
    # roi = [160,640]
    # number_of_frames_to_read = 700
    if utype == np.uint8:
        number_of_bytes = 1
    elif utype == np.uint16:
        number_of_bytes = 2
    elif utype == np.float:
        number_of_bytes = 4
    Mov = np.fromfile(f, dtype=utype, count=number_of_frames_to_read * roi[0] * roi[1], offset=number_of_frames_to_skip*roi[0]*roi[1]*number_of_bytes)
    Movie_len = np.int(len(Mov) / roi[0] / roi[1])
    number_of_elements = Movie_len * roi[0] * roi[1]
    Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])
    Mov = Mov[:,2:,2:]  #Get rid of bad frame

    # bla = torch.tensor(Mov.astype(np.float)).unsqueeze(1).float()
    # bla = scale_array_stretch_hist(bla)
    # imshow_torch_video(bla, FPS=50, frame_stride=5)
    return Mov

def read_frames_from_binary_file_stream_SpecificArea(f, number_of_frames_to_read=-1, number_of_frames_to_skip=0, params=None, center_HW=None, area_around_center=None):
    ### Get parameters from params dict: ###
    FileName = params['FileName']
    DistFold = params['DistFold']
    roi = params['roi']
    utype = params['utype']

    ### Read Frames: ###
    # roi = [160,640]
    min_batch_size = 250
    number_of_batches = int(np.ceil(number_of_frames_to_read/min_batch_size))
    left_over_frames_in_final_batch = number_of_frames_to_read - (number_of_frames_to_read//min_batch_size) * min_batch_size
    #(1). Skip Initial Frames:
    Mov = np.fromfile(f, dtype=utype, count=0, offset=number_of_frames_to_skip * roi[0] * roi[1] * 2)
    #(2). Loop and Read Frames:
    if center_HW is None:
        center_HW = (roi[0]//2, roi[1]//2)
    if area_around_center is None:
        area_around_center = np.inf
    final_movie = None
    for i in np.arange(number_of_batches-1):
        ### Get Current Batch: ###
        Mov = np.fromfile(f, dtype=utype, count=min_batch_size * roi[0] * roi[1], offset=0)
        Movie_len = np.int(len(Mov) / roi[0] / roi[1])
        number_of_elements = Movie_len * roi[0] * roi[1]
        Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])
        Mov = Mov[:,2:,2:]  #Get rid of bad frame
        ### Crop Around Wanted Area: ###
        if center_HW is not None:
            center_H, center_W = center_HW
            H_start = max(center_H - area_around_center + 1, 0)
            H_stop = min(center_H + area_around_center + 1, roi[0])
            W_start = max(center_W - area_around_center + 1, 0)
            W_stop = min(center_W + area_around_center + 1, roi[1])
            Mov = Mov[:, H_start:H_stop, W_start:W_stop]
        if final_movie is None:
            final_movie = np.zeros([0, Mov.shape[-2], Mov.shape[-1]])
        final_movie = np.concatenate([final_movie, Mov], axis=0)
    #(3). Get "left_over" frames:
    Mov = np.fromfile(f, dtype=utype, count=left_over_frames_in_final_batch * roi[0] * roi[1], offset=0)
    Movie_len = np.int(len(Mov) / roi[0] / roi[1])
    number_of_elements = Movie_len * roi[0] * roi[1]
    Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])
    Mov = Mov[:, 2:, 2:]
    if left_over_frames_in_final_batch > 0:
        if final_movie is None:
            final_movie = np.zeros([0, Mov.shape[-2], Mov.shape[-1]])
        if center_HW is not None:
            center_H, center_W = center_HW
            H_start = max(center_H - area_around_center + 1, 0)
            H_stop = min(center_H + area_around_center + 1, roi[0])
            W_start = max(center_W - area_around_center + 1, 0)
            W_stop = min(center_W + area_around_center + 1, roi[1])
            Mov = Mov[:, H_start:H_stop, W_start:W_stop]
        final_movie = np.concatenate([final_movie, Mov], axis=0)

    # bla = torch.tensor(Mov.astype(np.float)).unsqueeze(1).float()
    # bla = scale_array_stretch_hist(bla)
    # imshow_torch_video(bla, FPS=50, frame_stride=5)
    return final_movie

def read_long_movie_and_save_to_avi(f, video_name, FPS, params, max_number_of_images=np.inf):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    flag_keep_going = True
    count = 0
    while flag_keep_going and count<max_number_of_images:
        ### Read frame: ###
        current_frame = read_frames_from_binary_file_stream(f, 1, 0, params)   #TODO: turn this into a general function which accepts dtype, length, roi_size etc'
        T,H,W = current_frame.shape

        ### Scale Array: ###
        if count == 0:
            video_writer = cv2.VideoWriter(video_name, fourcc, FPS, (W, H))
            current_frame, (q1, q2) = scale_array_stretch_hist(current_frame, flag_return_quantile=True)
        else:
            current_frame = scale_array_from_range(current_frame.clip(q1, q2),
                                                   min_max_values_to_clip=(q1, q2),
                                                   min_max_values_to_scale_to=(0, 1))
        current_frame = numpy_array_to_video_ready(current_frame)
        current_frame = cv2.putText(img=current_frame,
                                    text=str(count),
                                    org=(W//2, 30),
                                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                    fontScale=1,
                                    color=(0, 255, 0),
                                    thickness=3)
        ### If done reading then stop: ###
        if current_frame.shape[0] == 0:
            flag_keep_going = False

        ### Write frame down: ###
        video_writer.write(current_frame)
        count = count + 1
        print(count)
    video_writer.release()
##################################################################################################################################################################





################################################################################################################################
def PreProcess_Movie(input_tensor, params=None):
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
        max_number_of_frames_to_use_for_quantile_calculation = 2
        number_of_frames_to_use = min(input_tensor.shape[0], max_number_of_frames_to_use_for_quantile_calculation)
        min_value = np.quantile(input_tensor[0:number_of_frames_to_use], 0.01)
        max_value = np.quantile(input_tensor[0:number_of_frames_to_use], 0.99)
        params.min_value = min_value
        params.max_value = max_value

    ### Actually stretch image: ###
    input_tensor = input_tensor.clip(min_value, max_value)  #clamp the image first to allow for effective stretching for viewing
    input_tensor = scale_array_to_range(input_tensor, (0, 1))

    ### Cut the edges of the frame which are non valid for our pelican-D: ###
    input_tensor = input_tensor[:, 2:, 2:]

    return input_tensor, params

def PreProcess_Movie_Torch(input_tensor, params=None):
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
        max_number_of_frames_to_use_for_quantile_calculation = 2
        number_of_frames_to_use = min(input_tensor.shape[0], max_number_of_frames_to_use_for_quantile_calculation)
        min_value = input_tensor[0:number_of_frames_to_use].quantile(0.01)
        max_value = input_tensor[0:number_of_frames_to_use].quantile(0.99)
        params.min_value = min_value
        params.max_value = max_value

    ### Actually stretch image: ###
    input_tensor = input_tensor.clip(min_value, max_value)  #clamp the image first to allow for effective stretching for viewing
    input_tensor = scale_array_to_range(input_tensor, (0, 1))

    ### Cut the edges of the frame which are non valid for our pelican-D: ###
    input_tensor = input_tensor[:, 2:, 2:]

    return input_tensor, params
################################################################################################################################





import numpy as np
import torch
import matplotlib
import os
from RapidBase.Utils.IO.Path_and_Reading_utils import get_filenames_from_folder_string_pattern
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import scale_array_to_range, scale_array_stretch_hist, scale_array_from_range
import cv2
import kornia
from RapidBase.Utils.IO.Path_and_Reading_utils import *
from RapidBase.Utils.IO.Imshow_and_Plots import draw_polygon_points_on_image, Get_BoundingBox_List_For_Each_Frame, draw_trajectories_on_images, draw_polygons_on_image, draw_text_on_image, draw_bounding_box_with_label_on_image_XYHW, plot_torch, imshow_torch, Plot_BoundingBox_On_Frame, Plot_BoundingBox_On_Movie, Plot_BoundingBox_And_Polygon_On_Movie, Plot_Bounding_Box_Demonstration, Plot_BoundingBoxes_On_Video
from RapidBase.Utils.Classical_DSP.FFT_utils import *
from RapidBase.Utils.Classical_DSP.Fitting_And_Optimization import polyfit_torch_parallel, polyval_torch, polyfit_torch, polyfit_torch_FitSimplestDegree_parallel, polyfit_torch_FitSimplestDegree
from RapidBase.Utils.Tensor_Manipulation.Pytorch_Numpy_Utils import get_circle_kernel
from RapidBase.Utils.MISCELENEOUS import get_elements_from_list_by_indices, create_empty_list_of_lists
from RapidBase.Utils.Classical_DSP.Fitting_And_Optimization import ransac_Torch
from RapidBase.Utils.IO.Binary_Files import initialize_binary_file_reader
import copy


def make_tuple_int(input_tuple):
    input_tuple = (np.int(np.round(input_tuple[0])), np.int(np.round(input_tuple[1])))
    return input_tuple


def get_line_ABC_parameters_from_points(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C

def get_line_slop_from_points(p1, p2):
    # dy/dx
    # (y2 - y1) / (x2 - x1)
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def get_intersection_point_between_two_lines_defined_by_ABC(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False  #todo: deal with this bringing back False!!!

def get_intersection_point_between_two_lines_defined_by_point_and_slope(m1, b1, m2, b2):
    if m1 == m2:
        print("These lines are parallel!!!")
        return None
    # y = mx + b
    # Set both lines equal to find the intersection point in the x direction
    # m1 * x + b1 = m2 * x + b2
    # m1 * x - m2 * x = b2 - b1
    # x * (m1 - m2) = b2 - b1
    # x = (b2 - b1) / (m1 - m2)
    x = (b2 - b1) / (m1 - m2)
    # Now solve for y -- use either line, because they are equal here
    # y = mx + b
    y = m1 * x + b1
    return x, y



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


def keep_only_valid_flashlight_BB_list_in_sequence(current_flashlight_BB):
    counter = 0
    for i in np.arange(len(current_flashlight_BB)):
        if current_flashlight_BB[i] != []:
            counter += 1
    return current_flashlight_BB[0:counter]


def get_BB_center_from_vertices(current_flashlight_BB):
    polygon_center_points_list = []
    for i in np.arange(len(current_flashlight_BB)):
        current_BB_vertices = current_flashlight_BB[i]
        vertex_0 = current_BB_vertices[0]
        vertex_1 = current_BB_vertices[1]
        vertex_2 = current_BB_vertices[2]
        vertex_3 = current_BB_vertices[3]
        x_center = (vertex_0[0] + vertex_1[0]) // 2
        y_center = (vertex_1[1] + vertex_2[1]) // 2
        center_tuple = (x_center, y_center)
        polygon_center_points_list.append(center_tuple)
    return polygon_center_points_list



from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA, SparsePCA, TruncatedSVD, FactorAnalysis, FastICA

def Trajectory_RandomForest_Fit():
    # super_folder = r'E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies\9-900m_matrice600_night_flir_640x512_800fps_26000frames_converted\Results\Sequences'
    super_folder = r'E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies'
    drone_filenames_list = get_filenames_from_folder_string_pattern(super_folder, True, string_pattern_to_search='*Trajectory_Drone.txt')
    not_drone_filenames_list = get_filenames_from_folder_string_pattern(super_folder, True, string_pattern_to_search='*Trajectory_NotDrone.txt')
    flashlight_filenames_list = get_filenames_from_folder_string_pattern(super_folder, True, string_pattern_to_search='*Trajectory_Flashlight.txt')
    total_filenames = get_filenames_from_folder_string_pattern(super_folder, True, string_pattern_to_search='*Trajectory*.txt')


    trajectory_total_features_torch_list = []
    labels_torch_list = []
    for current_filename_index in np.arange(len(total_filenames)):
        ### Get Filename Parts: ###
        current_filename = total_filenames[current_filename_index]
        folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(current_filename)

        ### Get Current Label: ###
        if filename_without_extension == 'Trajectory_Drone':
            current_label = 0
        elif filename_without_extension == 'Trajectory_NotDrone':
            current_label = 1
        elif filename_without_extension == 'Trajectory_Flashlight':
            current_label = 2

        ### Get Numpy Vec For Feature Extraction: ###
        COM_X_filename = os.path.join(folder, 'COM_X.npy')
        COM_Y_filename = os.path.join(folder, 'COM_Y.npy')
        MOI_X_filename = os.path.join(folder, 'MOI_X.npy')
        MOI_Y_filename = os.path.join(folder, 'MOI_Y.npy')
        TrjMovie_filename = os.path.join(folder, 'TrjMov.npy')
        cx = np.load(COM_X_filename, allow_pickle=True)
        cy = np.load(COM_Y_filename, allow_pickle=True)
        cx2 = np.load(MOI_X_filename, allow_pickle=True)
        cy2 = np.load(MOI_Y_filename, allow_pickle=True)
        TrjMov = np.load(TrjMovie_filename, allow_pickle=True)

        ### Switch to tensors: ###
        cx = torch.tensor(cx)
        cy = torch.tensor(cy)
        cx2 = torch.tensor(cx2)
        cy2 = torch.tensor(cy2)
        TrjMov = torch.tensor(TrjMov)
        t_vec = torch.arange(cx.shape[0])

        ### Get More Auxiliary Variables: ###
        trajectory_max_over_time = TrjMov.flatten(-2,-1).max(-1)[0]

        ### Break Down Into Non-OverLapping Parts: ###
        patch_size = 100
        stride_size = 100
        cx_unfolded = cx.unfold(dimension=0, size=patch_size, step=stride_size)
        cy_unfolded = cy.unfold(dimension=0, size=patch_size, step=stride_size)
        cx2_unfolded = cx2.unfold(dimension=0, size=patch_size, step=stride_size)
        cy2_unfolded = cy2.unfold(dimension=0, size=patch_size, step=stride_size)
        t_vec_unfolded = t_vec.unfold(dimension=0, size=patch_size, step=stride_size)
        trajectory_max_over_time_unfolded = trajectory_max_over_time.unfold(dimension=0, size=patch_size, step=stride_size)
        ### Perform Polyfit: ###
        cx_coefficients, cx_prediction, cx_residual_values, cx_residual_std = polyfit_torch_parallel(t_vec_unfolded, cx_unfolded, 4, True, True)
        cy_coefficients, cy_prediction, cy_residual_values, cy_residual_std = polyfit_torch_parallel(t_vec_unfolded, cy_unfolded, 4, True, True)
        cx2_coefficients, cx2_prediction, cx2_residual_values, cx2_residual_std = polyfit_torch_parallel(t_vec_unfolded, cx2_unfolded, 4, True, True)
        cy2_coefficients, cy2_prediction, cy2_residual_values, cy2_residual_std = polyfit_torch_parallel(t_vec_unfolded, cy2_unfolded, 4, True, True)
        TrjMax_coefficients, TrjMax_prediction, TrjMax_residual_values, TrjMax_residual_std = polyfit_torch_parallel(t_vec_unfolded, trajectory_max_over_time_unfolded, 4, True, True)
        ### Unify Into Feature Vec: ###
        cx_features = torch.cat([cx_coefficients.flatten(), cx_residual_std])
        cy_features = torch.cat([cy_coefficients.flatten(), cy_residual_std])
        cx2_features = torch.cat([cx2_coefficients.flatten(), cx2_residual_std])
        cy2_features = torch.cat([cy2_coefficients.flatten(), cy2_residual_std])
        TrjMax_features = torch.cat([TrjMax_coefficients.flatten(), TrjMax_residual_std])
        trajectory_total_features_torch = torch.cat([cx_features, cy_features, cx2_features, cy2_features, TrjMax_features])
        trajectory_total_features_numpy = trajectory_total_features_torch.cpu().numpy()

        ### Append To Lists: ###
        trajectory_total_features_torch_list.append(trajectory_total_features_torch.unsqueeze(0))
        labels_torch_list.append(current_label)

        ### Save Features Into Proper Place In Disk: ###
        final_features_filename = os.path.join(folder, 'trajectory_total_features.npy')
        np.save(final_features_filename, trajectory_total_features_numpy, allow_pickle=True)


    ### Get All Features Into Proper ML Format: ###
    trajectory_total_features_torch = torch.cat(trajectory_total_features_torch_list)
    labels_torch = torch.tensor(labels_torch_list)
    X = trajectory_total_features_torch.cpu().numpy()
    Y = labels_torch.cpu().numpy()
    standard_scaler_object = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    ### Only Get Certain Data If Wanted: ###
    features_1_len = len(cx_features)
    features_2_len = len(cy_features)
    features_3_len = len(cx2_features)
    features_4_len = len(cy2_features)
    features_5_len = len(TrjMax_features)
    features_len_cumsum = np.array([0, features_1_len, features_2_len, features_3_len, features_4_len, features_5_len]).cumsum()
    # #(1). Only Use COM:
    # X = X[:, 0:features_1_len+features_2_len]
    # #(2). Use Only COM-Y & MOI-Y:
    # X = np.concatenate([X[:, features_len_cumsum[1]:features_len_cumsum[1]+features_2_len], X[:, features_len_cumsum[3]:features_len_cumsum[3]+features_4_len]], -1)
    #(3). Use Only COM-Y:
    X = np.concatenate([X[:, features_len_cumsum[1]:features_len_cumsum[1] + features_2_len]], -1)

    ### Scale Data: ###
    X_train = standard_scaler_object.fit_transform(X_train)
    X_test = standard_scaler_object.fit_transform(X_test)
    y_train_OneHot = OneHotEncoder

    ### T-SNE Embedding: ###
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(X)
    X_embedded_Drone = X_embedded[Y==0]
    X_embedded_NotDrone = X_embedded[Y==1]
    X_embedded_Flashlight = X_embedded[Y==2]
    plt.scatter(X_embedded_Drone[:,0], X_embedded_Drone[:,1], c='blue')
    plt.scatter(X_embedded_NotDrone[:,0], X_embedded_NotDrone[:,1], c='green')
    plt.scatter(X_embedded_Flashlight[:,0], X_embedded_Flashlight[:,1], c='yellow')
    plt.legend(['Drone', 'Not Drone', 'Flashlight'])
    # plt.scatter(X_embedded[:,0], X_embedded[:,1], c='blue')

    ### Define Random Forest Regressor: ###
    RF_classifier = RandomForestRegressor(n_estimators=1000)
    RF_classifier.fit(X_train, y_train, sample_weight=None)
    y_pred_test = RF_classifier.predict(X_test)
    y_pred_train = RF_classifier.predict(X_train)

    ### Define Random Forest Classifier: ###
    RF_classifier = RandomForestClassifier(n_estimators=1000)
    RF_classifier.fit(X_train, y_train, sample_weight=None)
    y_pred_test = RF_classifier.predict(X_test)
    y_pred_test_probability = RF_classifier.predict_proba(X_test)
    y_pred_train = RF_classifier.predict(X_train)
    y_pred_train_probability = RF_classifier.predict_proba(X_train)

    ### Print Metrics Train: ###
    print(confusion_matrix(y_train, y_pred_train))
    print(classification_report(y_train, y_pred_train))
    print(accuracy_score(y_train, y_pred_train))
    ### Print Metrics Test: ###
    print(confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))
    print(accuracy_score(y_test, y_pred_test))

    ### Print Train Results: ###
    print('Train Prediction: ' + str(y_pred_train))
    print('Train GT: ' + str(y_train))
    ### Print Test Results: ###
    print('Test Prediction: ' + str(y_pred_test))
    print('Test GT: ' + str(y_test))

# Trajectory_RandomForest_Fit()




def get_flashlight_from_BGS_in_circle_shape_1(Mov, Movie_BG, frame_index, flag_save_interim=False):
    ### Try and equalize gain if needed: ###
    bla = Mov / (Movie_BG + 1e-3)
    gain = bla[0].quantile(0.5)

    ### BG substraction and scaling to uint8 image: ###
    input_tensor = Mov[frame_index] - Movie_BG[0]*gain  # TODO: understand whether to use a constant BG or a running BG, something like Mov[i]-Mov.median(0)[0]
    input_tensor = input_tensor.clip(0)
    input_tensor = input_tensor - input_tensor.min()  # maybe clip to zero before going further to avoid big minus signs?
    input_tensor = input_tensor.cpu().numpy()[0]
    input_tensor = scale_array_to_range(input_tensor)
    input_tensor = (input_tensor * 255).astype(np.uint8)
    if flag_save_interim:
        interim_list_1.append(input_tensor)

    ### Maybe i can simply assume that the flashlight is the strongest thing in the image and would therefor have 255 values?
    ### maybe max blur?

    ### PreProcessing Spatial Median-Blur (to avoid pixel-sized "fake" outliers): ###
    # TODO: be careful it doesn't "wipe-out" flashlight itself...i assume it's "aura" is big enough
    # TODO: this DOES "soften" the flashlight....maybe i should use some other strategy instead?
    #  maybe instead of median choose "second largest" or something?
    #  OR maybe just cut off small quantiles right here and that's it?!?!
    input_tensor = cv2.medianBlur(input_tensor, 3)
    if flag_save_interim:
        interim_list_2.append(input_tensor)

    ### Clamp To Get Rid Of BG which can be circles: ###
    # input_tensor = (scale_array_stretch_hist(torch.Tensor(input_tensor), (0.998,1)).numpy() * 255).astype(np.uint8)
    # TODO: perhapse switch to predefined limits??!?!? this would assumes the flashlight is at ~255,
    #  this isn't the case always!!!! maybe i should do something else like...i don't know
    q1 = np.quantile(input_tensor, 0.9994)
    # q1 = 200
    input_tensor = input_tensor.clip(q1).astype(np.uint8)
    if flag_save_interim:
        interim_list_3.append(input_tensor)

    ### small errosion filtering to avoid pixel sized "fake" outliers: ###
    # TODO: why isn't the median blur above enough?
    input_tensor = cv2.erode(input_tensor, get_circle_kernel(23, 1))
    if flag_save_interim:
        interim_list_4.append(input_tensor)

    ### Dilate in order to expand the "True" Flashlight to be big enough: ###
    input_tensor = cv2.dilate(input_tensor, get_circle_kernel(23, 11))
    if flag_save_interim:
        interim_list_5.append(input_tensor)

    ### Scale / Stretch: ###
    input_tensor = scale_array_to_range(input_tensor, (0, 255)).astype(np.uint8)
    if flag_save_interim:
        interim_list_6.append(input_tensor)

    ### Again clip and stretch to avoid small gradients which were dilated from being identified as the circles: ###
    input_tensor = input_tensor.clip(input_tensor.max() - 10, input_tensor.max())
    if flag_save_interim:
        interim_list_7.append(input_tensor)
    input_tensor = scale_array_to_range(input_tensor, (0, 255)).astype(np.uint8)
    if flag_save_interim:
        interim_list_8.append(input_tensor)

    return input_tensor


def get_flashlight_from_BGS_in_circle_shape_2(Mov, Movie_BG, frame_index, flag_save_interim=False):
    ### Try and equalize gain if needed: ###
    bla = Mov / (Movie_BG + 1e-3)
    gain = bla[0].quantile(0.5)

    ### BG substraction and scaling to uint8 image: ###
    input_tensor = Mov[frame_index] - Movie_BG[0] * gain  # TODO: understand whether to use a constant BG or a running BG, something like Mov[i]-Mov.median(0)[0]
    input_tensor = input_tensor.clip(0)
    input_tensor = input_tensor - input_tensor.min()  # maybe clip to zero before going further to avoid big minus signs?
    input_tensor = input_tensor[0]
    input_tensor = scale_array_to_range(input_tensor)
    input_tensor = (input_tensor * 255).type(torch.uint8)

    ### PreProcessing Spatial Median-Blur (to avoid pixel-sized "fake" outliers): ###
    # TODO: be careful it doesn't "wipe-out" flashlight itself...i assume it's "aura" is big enough
    # TODO: this DOES "soften" the flashlight....maybe i should use some other strategy instead?
    #  maybe instead of median choose "second largest" or something?
    #  OR maybe just cut off small quantiles right here and that's it?!?!
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).float()
    input_tensor = kornia.filters.median_blur(input_tensor, (3,3))
    # input_tensor = cv2.medianBlur(input_tensor, 3)

    ### Clamp To Get Rid Of BG which can be circles: ###
    # input_tensor = (scale_array_stretch_hist(torch.Tensor(input_tensor), (0.998,1)).numpy() * 255).astype(np.uint8)
    # TODO: perhapse switch to predefined limits??!?!? this would assumes the flashlight is at ~255
    q1 = input_tensor.quantile(0.9994)
    input_tensor = input_tensor.clamp(q1)
    # q1 = np.quantile(input_tensor, 0.9994)
    # input_tensor = input_tensor.clip(q1).astype(np.uint8)

    ### small errosion filtering to avoid pixel sized "fake" outliers: ###
    # TODO: why isn't the median blur above enough?
    input_tensor = kornia.morphology.erosion(input_tensor, torch.tensor(get_circle_kernel(23,1)).float().to(input_tensor.device))
    # input_tensor = cv2.erode(input_tensor, get_circle_kernel(23, 1))

    ### Dilate in order to expand the "True" Flashlight to be big enough: ###
    input_tensor = kornia.morphology.dilation(input_tensor, torch.tensor(get_circle_kernel(23, 1)).float().to(input_tensor.device))
    # input_tensor = cv2.dilate(input_tensor, get_circle_kernel(23, 11))
    ### Scale / Stretch: ###
    input_tensor = scale_array_to_range(input_tensor, (0, 255))

    ### Again clip and stretch to avoid small gradients which were dilated from being identified as the circles: ###
    input_tensor = input_tensor.clamp(input_tensor.max() - 10, input_tensor.max())

    input_tensor = scale_array_to_range(input_tensor, (0, 255))


    return input_tensor


def get_locations_and_velocity(BoundingBox_PerFrame_list, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, t_vec, flag_flashlight_found, params):
    if flag_flashlight_found:
        flashlight_BB_index = 0  # TODO: assuming that i found only one "flashlight" or that the first index is the flashlight itself
        flashlight_BB_per_frame_list = BoundingBox_PerFrame_list[flashlight_BB_index]
        locations_x_list = trajectory_smoothed_polynom_X[flashlight_BB_index]
        locations_y_list = trajectory_smoothed_polynom_Y[flashlight_BB_index]
        locations_t_list = t_vec[flashlight_BB_index]
        velocity_x_pixels_per_frame = (locations_x_list[-1] - locations_x_list[0]) / (locations_t_list[-1] - locations_t_list[0])
        velocity_y_pixels_per_frame = (locations_y_list[-1] - locations_y_list[0]) / (locations_t_list[-1] - locations_t_list[0])
        z = np.float(params.distance)
        f = np.float(params.f)
        pixel_size = params.pixel_size
        delta_t_per_frame = 1/ params.FPS  # [seconds]
        delta_x_per_pixel = pixel_size * (z / f)
        velocity_x_meters_per_second = velocity_x_pixels_per_frame * delta_x_per_pixel / delta_t_per_frame
        velocity_y_meters_per_second = velocity_y_pixels_per_frame * delta_x_per_pixel / delta_t_per_frame
    else:
        velocity_x_pixels_per_frame = None
        velocity_y_pixels_per_frame = None
        velocity_x_meters_per_second = None
        velocity_y_meters_per_second = None
        locations_x_list = None
        locations_y_list = None
        locations_t_list = None
    return velocity_x_pixels_per_frame, velocity_y_pixels_per_frame,\
           velocity_x_meters_per_second, velocity_y_meters_per_second, \
           locations_x_list, locations_y_list, locations_t_list

def Find_Thermal_Flashlight_In_Sequence(Mov, Movie_BG, params, sequence_index):
    ### Parameters: ###
    total_number_of_frames, C, H, W = Mov.shape

    ### Get Movie_BGS: ###
    Movie_BGS = Mov - Movie_BG
    # imshow_torch(Movie_BG)
    # imshow_torch_video(Mov-Movie_BG, number_of_frames=2500, FPS=50, frame_stride=5)
    # imshow_torch_video(Mov, number_of_frames=2500, FPS=50, frame_stride=5)

    ### Loop over frames and get positions of the flashlight: ###
    images_with_circles = []
    circles_TXY_centers_list = []

    interim_list_1 = []
    interim_list_2 = []
    interim_list_3 = []
    interim_list_4 = []
    interim_list_5 = []
    interim_list_6 = []
    interim_list_7 = []
    interim_list_8 = []
    flag_save_interim = True
    # imshow_numpy_list_video(interim_list_1, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow_numpy_list_video(interim_list_2, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow_numpy_list_video(interim_list_3, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow_numpy_list_video(interim_list_4, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow_numpy_list_video(interim_list_5, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow_numpy_list_video(interim_list_6, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow_numpy_list_video(interim_list_7, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow_numpy_list_video(interim_list_8, number_of_frames=2500, FPS=50, frame_stride=1)
    for frame_index in np.arange(total_number_of_frames):
        #################################################################################################################################
        input_tensor = get_flashlight_from_BGS_in_circle_shape_1(Mov, Movie_BG, frame_index)
        # input_tensor = get_flashlight_from_BGS_in_circle_shape_2(Mov, Movie_BG, frame_index)
        #################################################################################################################################

        #################################################################################################################################
        ### Find Circles Using Hough Transform: ###
        minDist = 50
        param1 = 30  # 500
        param2 = 10  # 200 #smaller value-> more false circles
        minRadius = 1
        maxRadius = 15  # 10
        # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
        circles = cv2.HoughCircles(input_tensor, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)
        #################################################################################################################################

        #################################################################################################################################
        ### Draw circles on the image: ###
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle_parameters in circles[0, :]:
                x ,y ,r = circle_parameters
                cv2.circle(input_tensor, (circle_parameters[0], circle_parameters[1]), circle_parameters[2], (122, 122, 122), 2)
                input_tensor_zeros = np.ones_like(input_tensor)
                cv2.circle(input_tensor_zeros, (circle_parameters[0], circle_parameters[1]), circle_parameters[2], (122, 122, 122), 2)
                ### Add circle parameters as events to list: ###
                circles_TXY_centers_list.append((frame_index, circle_parameters[0], circle_parameters[1]))
        else:
            # print('did not find any circles!!!')
            1

        ### Add current image WITH CIRCLES ON IT to list: ###
        images_with_circles.append(torch.Tensor(input_tensor).unsqueeze(0))
        # images_with_circles.append(torch.Tensor(input_tensor_zeros).unsqueeze(0))
        #################################################################################################################################

    ### Get images with circles tensor: ###
    # images_with_circles_tensor = torch.cat(images_with_circles).unsqueeze(1)
    # imshow_torch_video(images_with_circles_tensor, number_of_frames=2500, FPS=50, frame_stride=1)
    # imshow(input_tensor); plt.show()

    ### Find Lines Using Ransac: ###
    # TODO: to make things more specific rename t_vec -> t_vec_PerTrajectory
    # TODO: i don't really need the Movie_BGS here at all, in fact the only reason i'm passing any movie in is to get the (T,H,W) shape!!!. get rid of this!!!
    points_not_assigned_to_lines_yet, NonLinePoints, direction_vec, holding_point, t_vec, \
    trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, flashlight_trajectory_tuple, TrjMov, num_of_trj_found = \
        Find_Flashlight_Trajectory(np.atleast_2d(circles_TXY_centers_list), Movie_BGS.cpu().squeeze(1).numpy(), params)

    ### Create Flashlight Folder If Needed: ###
    flashlight_folder = os.path.join(params.results_folder, 'Flashlight')
    create_folder_if_doesnt_exist(flashlight_folder)

    ### Plot 3D CLoud and Outline Trajectories: ###
    Plot_3D_PointCloud_With_Trajectories(points_not_assigned_to_lines_yet,
                                         NonLinePoints,
                                         xyz_line,
                                         num_of_trj_found,
                                         params,
                                         "Flashlight_Plt3DPntCloudTrj_" + str(sequence_index),
                                         flashlight_folder)

    ### Assign to a variable whether we've found a flashlight: ###
    flag_flashlight_found = len(trajectory_smoothed_polynom_X) > 0

    ### Show Raw and BG_Substracted_Normalized Videos with proper bounding boxes: ###
    BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(Mov.squeeze(1).cpu().numpy(),
                                                                    fps=50,
                                                                    tit='Movie_' + str(sequence_index),
                                                                    Res_dir=flashlight_folder,
                                                                    flag_save_movie=False,
                                                                    trajectory_tuple=flashlight_trajectory_tuple)
    Plot_BoundingBox_On_Movie(Mov.squeeze(1).cpu().numpy(),
                              fps=50,
                              tit="BB_only_where_flashlight_was_predifined_" + str(sequence_index),
                              Res_dir=flashlight_folder,
                              flag_save_movie=flag_flashlight_found,
                              trajectory_tuple=flashlight_trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)


    ### Get Velocity Of Flashlight: ###
    flashlight_BB_index = 0   #
    velocity_x_pixels_per_frame, velocity_y_pixels_per_frame, \
    velocity_x_meters_per_second, velocity_y_meters_per_second, \
    locations_x_list, locations_y_list, locations_t_list = \
        get_locations_and_velocity(BoundingBox_PerFrame_list, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, t_vec, flag_flashlight_found, params)

    ### Define Bounding Box for the entire movie (not just for samples which have the flashlight on): ####
    # TODO: turn into function
    BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list = copy.deepcopy(BoundingBox_PerFrame_list)
    if flag_flashlight_found:
        # (1). complete [0, t_vec[0]] with bounding box (assuming constant velocity):
        for t_index in np.arange(0, t_vec[flashlight_BB_index][0]):
            current_location_x = locations_x_list[0] - velocity_x_pixels_per_frame * (t_vec[flashlight_BB_index][0] - t_index)
            current_location_y = locations_y_list[0] - velocity_y_pixels_per_frame * (t_vec[flashlight_BB_index][0] - t_index)
            BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list[flashlight_BB_index][t_index] = (np.int(current_location_x), np.int(current_location_y))
        # (2). complete [t_vec[-1], end] with bounding box (assuming constant velocity):
        for t_index in np.arange(t_vec[flashlight_BB_index][-1], total_number_of_frames):
            current_location_x = locations_x_list[-1] + velocity_x_pixels_per_frame * (t_index - t_vec[flashlight_BB_index][-1])
            current_location_y = locations_y_list[-1] + velocity_y_pixels_per_frame * (t_index - t_vec[flashlight_BB_index][-1])
            BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list[flashlight_BB_index][t_index] = (np.int(current_location_x), np.int(current_location_y))

    #TODO: i need to add a sequence_index variable to add to movie title, otherwise it means nothing
    Plot_BoundingBox_On_Movie(Mov.squeeze(1).cpu().numpy(),
                              fps=50,
                              tit="BB_in_entire_movie",
                              Res_dir=flashlight_folder,
                              flag_save_movie=flag_flashlight_found,
                              trajectory_tuple=flashlight_trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list)

    ### Get flashlight line and intersection with frame parameters: ###
    if flag_flashlight_found:
        # TODO: use the locations list to get the middle of trajectory and extend that line in both directions by a set amount. maybe it intersects the frame and maybe not, but that will define the polygon instead of all the other shit
        flashlight_p1 = (locations_x_list[0], locations_y_list[0])
        flashlight_p2 = (locations_x_list[-1], locations_y_list[-1])
        flashlight_line, flashlight_line_slop_m, \
        upper_line_points, left_line_points, right_line_points, bottom_line_points, \
        polygon_path, polygon_points = \
            get_flashlight_line_and_intersection_with_frame_parameters(input_tensor, flashlight_p1, flashlight_p2,
                                                                       delta_pixel_flashlight_area_side=20)

        # (*). Draw Lines Which Define Restricted Area On Image:
        #TODO: same here, add sequence_index or don't show anything at all because i already have "full informative movie"
        Plot_BoundingBox_And_Polygon_On_Movie(Mov.squeeze(1).cpu().numpy(),
                                              fps=50,
                                              tit="BB_and_Line_in_entire_movie",
                                              Res_dir=flashlight_folder,
                                              flag_save_movie=1,
                                              trajectory_tuple=flashlight_trajectory_tuple,
                                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list,
                                              polygon_points_list=polygon_points)
    else:
        flashlight_p1 = None
        flashlight_p2 = None
        flashlight_line = None
        polygon_lines_list = None
        polygon_points = None

    return flag_flashlight_found, flashlight_trajectory_tuple, \
           BoundingBox_PerFrame_list, BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list, \
           velocity_x_pixels_per_frame, velocity_y_pixels_per_frame, \
           velocity_x_meters_per_second, velocity_y_meters_per_second, \
           flashlight_p1, flashlight_p2, flashlight_line, \
           polygon_points


def Find_Thermal_Flashlight_In_Movie(f, Movie_BG, params):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    flag_keep_going = True
    count = 0
    video_name = 'Movie_With_Flashlight_YesNo_Indication.avi'
    FPS = 50.0
    H, W = Movie_BG.shape[-2:]
    f = initialize_binary_file_reader(f)
    video_writer = cv2.VideoWriter(os.path.join(params.results_folder, 'Flashlight', video_name), fourcc, FPS, (W, H))
    batch_size = int(params['FPS'] * params['SeqT'])
    max_number_of_batches = 50  # TODO: change this later to np.inf or something to go over the entire movie. this is just to speed debugging up

    flag_flashlight_found_list = []
    polygon_points_list = []
    flashlight_BB_list = []
    velocity_per_frame_list = []
    velocity_meters_per_second_list = []
    flashlight_line_list = []
    flashlight_smooth_trajectory_points_list = []
    Velocity_TXY_list = []
    while flag_keep_going and count < max_number_of_batches:
        print(count)
        ### Read frame: ###
        Mov = read_frames_from_binary_file_stream(f,
                                                  number_of_frames_to_read=batch_size,
                                                  number_of_frames_to_skip=0,
                                                  params=params)  # TODO: turn this into a general function which accepts dtype, length, roi_size etc'
        T, H, W = Mov.shape

        ### Scale Array: ###
        if count == 0:
            Mov, (q1, q2) = scale_array_stretch_hist(Mov, flag_return_quantile=True)
        else:
            Mov = scale_array_from_range(Mov.clip(q1, q2),
                                         min_max_values_to_clip=(q1, q2),
                                         min_max_values_to_scale_to=(0, 1))

        ### If done reading then stop: ###
        if Mov.shape[0] < batch_size:
            flag_keep_going = False

        if flag_keep_going:
            ### Make current frames a tensor: ###
            Movie = torch_get_4D(torch.Tensor(Mov), 'THW').to(Movie_BG.device)
            # imshow_torch_video(Movie, number_of_frames=2500, FPS=50, frame_stride=5)

            ### Check Whether Flashlight Was Found: ###
            flag_flashlight_found, flashlight_trajectory_tuple, \
            BoundingBox_PerFrame_list, BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list, \
            velocity_x_pixels_per_frame, velocity_y_pixels_per_frame, \
            velocity_x_meters_per_second, velocity_y_meters_per_second, \
            flashlight_p1, flashlight_p2, flashlight_line, \
            flashlight_polygon_points = \
                Find_Thermal_Flashlight_In_Sequence(Movie, Movie_BG, params, sequence_index=count)

            ### Print Whether Flashlight Was Found: ###
            print('Flashlight Found: ' + str(flag_flashlight_found))

            ### Save Auxiliary Results To Disk: ###
            velocity_per_frame_list.append([velocity_x_pixels_per_frame, velocity_y_pixels_per_frame])
            velocity_meters_per_second_list.append([velocity_x_meters_per_second, velocity_y_meters_per_second])
            Velocity_TXY_list.append([1, velocity_x_pixels_per_frame, velocity_y_pixels_per_frame])
            flashlight_line_list.append(flashlight_line)
            flashlight_smooth_trajectory_points_list.append((flashlight_p1, flashlight_p2))

            ### Update Tracking Lists: ###
            flag_flashlight_found_list.append(flag_flashlight_found)
            polygon_points_list.append(flashlight_polygon_points)
            if flag_flashlight_found:
                flashlight_BB_list.append(BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list[0])  # assuming the flashlight is the only one here, in the 0 index
            else:
                flashlight_BB_list.append(None)

            ### Loop over inter-batch frames and add them to video: ###
            # TODO: insert this into a function, it's too much clutter
            if params.flag_save_interim_graphs_and_movies:
                for inter_frame_index in np.arange(T):
                    current_frame = Mov[inter_frame_index]

                    ### Get frame video ready: ###
                    current_frame = numpy_array_to_video_ready(current_frame)

                    ### Put proper title according to whether a flashlight was found: ###
                    string_for_image = 'Batch: ' + str(count) + ',\n Frame: ' + str(count * batch_size + inter_frame_index) + ',\n Interframe Index: ' + str(inter_frame_index)
                    if flag_flashlight_found:
                        string_for_image += ', Flashlight On'
                    else:
                        string_for_image += ', Flashlight Off'
                    current_frame = cv2.putText(img=current_frame,
                                                text=string_for_image,
                                                org=(0, 30),
                                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                                fontScale=0.7,
                                                color=(0, 255, 0),
                                                thickness=2)

                    ### Plot Bounding-Box, Polyon On Frame: ###
                    # TODO: the below code shows how i need to rethink how i represent trajectories and bounding boxes, perhapse represent as ndarray
                    if flag_flashlight_found:
                        BoundingBoxes_For_Current_Frame = []
                        number_of_trajectories = len(BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list)
                        # (1). get all the trajectories found as bounding boxes:
                        for trajectory_index in np.arange(number_of_trajectories):
                            BoundingBoxes_For_Current_Frame.append(BoundingBox_PerFrame_Extrapolate_On_Entire_Movie_list[trajectory_index][inter_frame_index])
                        # (2). add long bounding box of while trajectory on image frame:
                        current_frame = draw_polygon_points_on_image(current_frame, polygon_points=flashlight_polygon_points)
                        # (3). add flashlight bounding box (circle) on current frame:
                        current_frame = Plot_BoundingBox_On_Frame(current_frame, BoundingBoxes_For_Current_Frame)

                    ### Write frame down: ###
                    video_writer.write(current_frame)

        ### Advance Batch Counter: ###
        count = count + 1

    video_writer.release()

    ######################################################################################################
    ### Write down where flashlight is located to disk for future use: ###
    np.save(os.path.join(params.results_folder, 'Flashlight', 'flashlight_polygon_points_list.npy'), polygon_points_list, allow_pickle=True)
    np.save(os.path.join(params.results_folder, 'Flashlight', 'flag_flashlight_found_list.npy'), flag_flashlight_found_list, allow_pickle=True)
    np.save(os.path.join(params.results_folder, 'Flashlight', 'flashlight_BB_list.npy'), flashlight_BB_list, allow_pickle=True)

    np.save(os.path.join(params.results_folder, 'Flashlight', 'flashlight_smooth_trajectory_points_list.npy'), flashlight_smooth_trajectory_points_list, allow_pickle=True)
    np.save(os.path.join(params.results_folder, 'Flashlight', 'flashlight_line_list.npy'), flashlight_line_list, allow_pickle=True)
    np.save(os.path.join(params.results_folder, 'Flashlight', 'velocity_meters_per_second_list.npy'), velocity_meters_per_second_list, allow_pickle=True)
    np.save(os.path.join(params.results_folder, 'Flashlight', 'velocity_per_frame_list.npy'), velocity_per_frame_list, allow_pickle=True)
    np.save(os.path.join(params.results_folder, 'Flashlight', 'Velocity_TXY_list.npy'), Velocity_TXY_list, allow_pickle=True)
    ######################################################################################################

    #######################################################################################################
    ### Make one, robust, flashlight Bounding-Box from all the sub-sequences found: ###
    flag_flashlight_found_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'flag_flashlight_found_list.npy'), allow_pickle=True)
    polygon_points_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'flashlight_polygon_points_list.npy'), allow_pickle=True)
    flashlight_BB_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'flashlight_BB_list.npy'), allow_pickle=True)
    flashlight_smooth_trajectory_points_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'flashlight_smooth_trajectory_points_list.npy'), allow_pickle=True)
    flashlight_line_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'flashlight_line_list.npy'), allow_pickle=True)
    velocity_meters_per_second_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'velocity_meters_per_second_list.npy'), allow_pickle=True)
    velocity_per_frame_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'velocity_per_frame_list.npy'), allow_pickle=True)
    Velocity_TXY_list = np.load(os.path.join(params.results_folder, 'Flashlight', 'Velocity_TXY_list.npy'), allow_pickle=True)

    ### Get Robust Estimation Of Flashlight Velocity & Bounding-Box: ###
    flag_flashlight_found_array = np.array(flag_flashlight_found_list)
    if np.sum(flag_flashlight_found_array) > 0:
        velocity_TXY_array = Velocity_TXY_list[flag_flashlight_found_array]
        velocity_per_frame_array = velocity_per_frame_list[flag_flashlight_found_array]
        flashlight_BB_array = flashlight_BB_list[flag_flashlight_found_array]
        weights_list = []
        for i in np.arange(velocity_TXY_array.shape[0]):
            Vt, Vx, Vy = velocity_TXY_array[i]
            Vx, Vy = velocity_per_frame_array[i]
            current_weight = ((Vx ** 2 + Vy ** 2) / Vt ** 2) ** 1
            weights_list.append(current_weight)
        weights_array = np.array(weights_list)
        weights_array = weights_array / weights_array.sum()
        weights_array = numpy_unsqueeze(weights_array, -1)
        total_velocity = (weights_array * velocity_per_frame_array).sum(0)
        total_velocity_x = total_velocity[0]
        total_velocity_y = total_velocity[1]
        initial_location_x = flashlight_BB_array[0][0][0]
        initial_location_y = flashlight_BB_array[0][0][1]
        final_location_x = initial_location_x + total_velocity_x * 1
        final_location_y = initial_location_y + total_velocity_y * 1
        initial_location = (initial_location_x, initial_location_y)
        final_location = (final_location_x, final_location_y)

        ### Get Robust Flashlight Bounding Box and Polygon: ###
        flashlight_BB_array = flashlight_BB_list[flag_flashlight_found_array]

        ### Get Intersection Of Robust Flashlight Bounding Box with frame: ###
        flashlight_line, flashlight_line_slop_m, \
        upper_line_points, left_line_points, right_line_points, bottom_line_points, \
        polygon_path, robust_flashlight_polygon_points = \
            get_flashlight_line_and_intersection_with_frame_parameters(Movie_BG.cpu().numpy()[0, 0],
                                                                       initial_location,
                                                                       final_location,
                                                                       delta_pixel_flashlight_area_side=20)

        ### Draw Flashlight Polygon On Frame Just To Be Sure: ###
        # current_frame = draw_polygon_points_on_image(Movie_BG.cpu().numpy()[0,0], polygon_points=polygon_points)

        ### Save Robust Polygon: ###
        np.save(os.path.join(params.results_folder, 'Flashlight', 'robust_flashlight_polygon_points.npy'), robust_flashlight_polygon_points, allow_pickle=True)
    else:
        np.save(os.path.join(params.results_folder, 'Flashlight', 'robust_flashlight_polygon_points.npy'), None, allow_pickle=True)
        robust_flashlight_polygon_points = None
    ######################################################################################################


    return flag_flashlight_found_list, polygon_points_list, flashlight_BB_list, robust_flashlight_polygon_points





def Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Drone_Torch(res_points, params, H, W):
    ### Get variables from params dict: ###
    DroneLineEst_RANSAC_D = params['DroneLineEst_RANSAC_D']
    DroneLineEst_polyfit_degree = params['DroneLineEst_polyfit_degree']
    DroneLineEst_RANSAC_max_trials = params['DroneLineEst_RANSAC_max_trials']
    minimum_number_of_samples_after_polyfit = params['DroneLineEst_minimum_number_of_samples_after_polyfit']
    minimum_number_of_samples_before_polyfit = params['DroneLineEst_minimum_number_of_samples_before_polyfit']
    ROI_allocated_around_suspect = params['DroneLineEst_ROI_allocated_around_suspect']

    ### Perform RANSAC to estimate line: ###
    # (*). Remember, this accepts ALL the residual points coordinates and outputs a line. shouldn't we use something like clustering???
    model_robust, indices_within_distance = ransac_Torch(res_points, LineModelND_Torch, min_samples_for_model=2,
                                                         residual_threshold=DroneLineEst_RANSAC_D, max_trials=DroneLineEst_RANSAC_max_trials)
    holding_point = model_robust.params[0]
    direction_vec = model_robust.params[1]
    valid_trajectory_points = res_points[indices_within_distance]

    ### Get points within the predefined distance and those above it: ###
    points_off_line_found = res_points[~indices_within_distance]
    points_on_line_found = valid_trajectory_points

    ### if there are enough valid points then do a polynomial fit for their trajectory: ###
    if valid_trajectory_points.shape[0] > minimum_number_of_samples_before_polyfit:
        ### Get Linearly increasing and "full" t_vec: ###
        t_vec = torch.arange(valid_trajectory_points[:, 0][0], valid_trajectory_points[:, 0][-1]).to(res_points.device)

        ########################################################################
        ### Pytorch Polyfit: ###
        coefficients_x, prediction_x, residuals_x = polyfit_torch(valid_trajectory_points[:, 0], valid_trajectory_points[:, 1], DroneLineEst_polyfit_degree)
        coefficients_y, prediction_y, residuals_y = polyfit_torch(valid_trajectory_points[:, 0], valid_trajectory_points[:, 2], DroneLineEst_polyfit_degree)
        ### Pytorch polyval over full t_vec between the first and last time elements of the trajectory: ###
        trajectory_smoothed_polynom_X = polyval_torch(coefficients_x, t_vec)
        trajectory_smoothed_polynom_Y = polyval_torch(coefficients_y, t_vec)
        ########################################################################

        # ########################################################################
        # ### Get (t,x,y) points vec: ###
        # smoothed_trajectory_over_time_vec = torch.cat( (t_vec.unsqueeze(-1),
        #                                                 trajectory_smoothed_polynom_X.unsqueeze(-1),
        #                                                 trajectory_smoothed_polynom_Y.unsqueeze(-1)), -1)
        # ########################################################################

        # ########################################################################
        # ### Test if the x&y trajectories are within certain boundaries: ###
        # #TODO: is this really necessary? there are a lot of calculations being done for nothing! simply check the first and last elements!!!
        # #TODO: i can get rid of this in the first place when accepting outliers probably but probably doesn't take a long time anyway
        # indices_where_x_trajectory_within_frame_boundaries = (trajectory_smoothed_polynom_X >= ROI_allocated_around_suspect / 2 + 1) &\
        #                                                      (trajectory_smoothed_polynom_X <= W - ROI_allocated_around_suspect / 2 - 1)
        # indices_where_y_trajectory_within_frame_boundaries = (trajectory_smoothed_polynom_Y >= ROI_allocated_around_suspect / 2 + 1) &\
        #                                                      (trajectory_smoothed_polynom_Y <= H - ROI_allocated_around_suspect / 2 - 1)
        #
        # ### Get indices_where_trajectory_is_within_frame_boundaries: ###
        # indices_1 = (indices_where_x_trajectory_within_frame_boundaries & indices_where_y_trajectory_within_frame_boundaries)
        #
        # ### Get indices where_trajectory_is_close_enough_to_line_estimate: ###
        # #TODO: is it really necessary to do this again? i mean, the smoothed line will probably contain the same indices as before!!!
        # # it's pretty much a straight line!!! no need to get over our heads!!!!
        # indices_2 = get_Distance_From_Points_To_DirectionVec_Torch(smoothed_trajectory_over_time_vec, direction_vec, holding_point) < (DroneLineEst_RANSAC_D)
        #
        # ### Final valid indices are those which satisfy both conditions: ###
        # t_vec_valid = indices_1 * indices_2
        #
        # ### Get valid parts of the trajectory: ###
        # t_vec = t_vec[t_vec_valid]
        # trajectory_smoothed_polynom_X = trajectory_smoothed_polynom_X[t_vec_valid]
        # trajectory_smoothed_polynom_Y = trajectory_smoothed_polynom_Y[t_vec_valid]
        # ########################################################################

    else:
        ### Get valid parts of the trajectory: ###
        t_vec = []
        trajectory_smoothed_polynom_X = []
        trajectory_smoothed_polynom_Y = []

    ### make sure there are enough valid points: ###
    if len(t_vec) > minimum_number_of_samples_after_polyfit:
        flag_enough_valid_samples = True
    else:
        flag_enough_valid_samples = False

    return direction_vec, holding_point, t_vec,\
           trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, \
           points_off_line_found, points_on_line_found, flag_enough_valid_samples



def Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Flashlight(res_points, params, H, W):
    ### Get variables from params dict: ###
    DroneLineEst_RANSAC_D = params['FlashlightLineEst_RANSAC_D']
    DroneLineEst_polyfit_degree = params['FlashlightLineEst_polyfit_degree']
    DroneLineEst_RANSAC_max_trials = params['FlashlightLineEst_RANSAC_max_trials']
    minimum_number_of_samples_after_polyfit = params['FlashlightLineEst_minimum_number_of_samples_after_polyfit']
    minimum_number_of_samples_before_polyfit = params['FlashlightLineEst_minimum_number_of_samples_before_polyfit']
    ROI_allocated_around_suspect = params['FlashlightLineEst_ROI_allocated_around_suspect']

    ### Perform RANSAC to estimate line: ###
    # (*). Remember, this accepts ALL the residual points coordinates and outputs a line. shouldn't we use something like clustering???
    model_robust, indices_within_distance = ransac(res_points, LineModelND, min_samples_for_model=2,
                                                   residual_threshold=DroneLineEst_RANSAC_D, max_trials=DroneLineEst_RANSAC_max_trials)
    holding_point = model_robust.params[0]
    direction_vec = model_robust.params[1].reshape(3, 1)

    ### Calculate distance between points and the line found by RANSAC, which hopefully gets close enough to a real trajectory, and only get those within a certain radius!: ###
    indices_within_distance = get_Distance_From_Points_To_DirectionVec(res_points, direction_vec, holding_point) < (2 * DroneLineEst_RANSAC_D)

    ### Use the above found "valid" points to estimate a new line/vector which is more robust: ###
    q = res_points[indices_within_distance]
    holding_point1 = np.mean(q, 0)  # the holding point is the temporal mean of the "valid" indices? is there a better alternative?
    _, _, direction_vec1 = np.linalg.svd(q - holding_point)
    direction_vec1 = direction_vec1[0].reshape(3, 1)

    ### Use these new points to see if the new line estimate and the residuals from it are still within the predefined radius: ###
    indices_within_distance = get_Distance_From_Points_To_DirectionVec(res_points, direction_vec1, holding_point1) < DroneLineEst_RANSAC_D
    valid_trajectory_points = res_points[indices_within_distance]
    t_vec = np.arange(min(valid_trajectory_points[:, 0]), max(valid_trajectory_points[:, 0]))

    ### if there are enough valid points then do a polynomial fit for their trajectory: ###
    if valid_trajectory_points.shape[0] > minimum_number_of_samples_before_polyfit:

        ### Polyfit trajectory to a polynomial of wanted degree: ###
        fitx = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 1],
                                    DroneLineEst_polyfit_degree))  # fit (t,x) polynomial
        fity = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 2],
                                    DroneLineEst_polyfit_degree))  # fit (t,y) polynomial

        ### Use found polynomial to get "smoothed" trajectory from the polyfit over the relevant, discrete time_vec: ###
        trajectory_smoothed_polynom_X = fitx(t_vec)
        trajectory_smoothed_polynom_Y = fity(t_vec)

        ### Get (t,x,y) points vec: ###
        smoothed_trajectory_over_time_vec = np.array((t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)).T

        ### Test if the x&y trajectories are WITHIN FRAME BOUNDARIES: ###
        indices_where_x_trajectory_within_frame_boundaries = (trajectory_smoothed_polynom_X >= ROI_allocated_around_suspect / 2 + 1) \
                                                             & (trajectory_smoothed_polynom_X <= W - ROI_allocated_around_suspect / 2 - 1)
        indices_where_y_trajectory_within_frame_boundaries = (trajectory_smoothed_polynom_Y >= ROI_allocated_around_suspect / 2 + 1) \
                                                             & (trajectory_smoothed_polynom_Y <= H - ROI_allocated_around_suspect / 2 - 1)

        ### Get indices_where_trajectory_is_within_frame_boundaries: ###
        indices_1 = (indices_where_x_trajectory_within_frame_boundaries & indices_where_y_trajectory_within_frame_boundaries)

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
    #TODO: turn this number into a parameter!
    if len(t_vec) > minimum_number_of_samples_after_polyfit:
        flag_enough_valid_samples = True
    else:
        flag_enough_valid_samples = False

    ### Get points within the predefined distance and those above it: ###   #TODO: what do we do with points_new. i should really name points_new to be points_far_from_line or points_off_line
    points_off_line_found = res_points[indices_within_distance == False]
    points_on_line_found = res_points[indices_within_distance]

    return direction_vec1, holding_point1, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, points_off_line_found, points_on_line_found, flag_enough_valid_samples



def Decide_If_Trajectory_Valid_Drone_Torch(xyz_line, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, t_vec, direction_vec, number_of_frames, params):
    ### Get variables from params dictionary: ###
    DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible = params['DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible']
    DroneTrjDec_maximum_projection_onto_XY_plane = params['DroneTrjDec_maximum_projection_onto_XY_plane']
    DroneTrjDec_minimum_projection_onto_XY_plane = params['DroneTrjDec_minimum_projection_onto_XY_plane']
    DroneTrjDec_max_time_gap_in_trajectory = params['DroneTrjDec_max_time_gap_in_trajectory']
    DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible = params['DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible']
    SeqT = params['SeqT']
    FrameRate = params['FPS']
    DroneTrjDec_allowed_BB_within_frame_H_by_fraction = params['DroneTrjDec_allowed_BB_within_frame_H_by_fraction']
    DroneTrjDec_allowed_BB_within_frame_W_by_fraction = params['DroneTrjDec_allowed_BB_within_frame_W_by_fraction']
    H = params.H
    W = params.W

    ### make sure there are enough time stamps to even look into trajectory: ###
    flag_trajectory_valid = len(t_vec) > (number_of_frames * DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible)

    if flag_trajectory_valid:
        ### make sure velocity is between min and max values: ###
        b_abs = direction_vec.abs()

        ### the portion of the unit (t,x,y) direction vec's projection on the (x,y) plane: ###
        v = (b_abs[1] ** 2 + b_abs[2] ** 2) ** 0.5

        ### make sure the drone's "velocity" in the (x,y) is between some values, if it's too stationary it's weird and if it's too fast it's weird: ###
        #TODO: for experiments DroneTrjDec_minimum_projection_onto_XY_plane should be 0, IN REAL LIFE it should be >0 because drones usually move
        flag_trajectory_valid = ((v < DroneTrjDec_maximum_projection_onto_XY_plane) & (v >= DroneTrjDec_minimum_projection_onto_XY_plane))
        if flag_trajectory_valid:

            ### make sure max difference between time stamps, which is also the maximum consecutive frames suspect disappeared, is not too high: ###
            original_t_vec = xyz_line[:,0]
            original_t_vec_unique = original_t_vec.unique()
            flag_trajectory_valid = original_t_vec_unique.diff().max() < DroneTrjDec_max_time_gap_in_trajectory
            if flag_trajectory_valid:

                ### Make sure total time for valid points is above the minimum we decided upon: ###
                total_amount_of_valid_original_t_vec_samples = len(original_t_vec_unique)
                minimum_total_valid_time_range_for_suspect = (DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible * SeqT * FrameRate)  # this is stupid, SeqT*FrameRate = number of frames
                flag_trajectory_valid = minimum_total_valid_time_range_for_suspect < total_amount_of_valid_original_t_vec_samples

                ### Make sure the trajectory boundaries are within predefined range within the image frame (basically not to low to avoid cars): ###
                if flag_trajectory_valid:
                    # ### Using Actual Trajectory Points: ###
                    # trajectory_min_H = min(xyz_line[:, 2])
                    # trajectory_max_H = max(xyz_line[:, 2])
                    # trajectory_min_W = min(xyz_line[:, 1])
                    # trajectory_max_W = max(xyz_line[:, 1])

                    ### Using Smoothed Trajectory: ###
                    trajectory_min_H = min(trajectory_smoothed_polynom_Y[0], trajectory_smoothed_polynom_Y[-1])
                    trajectory_max_H = max(trajectory_smoothed_polynom_Y[0], trajectory_smoothed_polynom_Y[-1])
                    trajectory_min_W = min(trajectory_smoothed_polynom_X[0], trajectory_smoothed_polynom_X[-1])
                    trajectory_max_W = max(trajectory_smoothed_polynom_X[0], trajectory_smoothed_polynom_X[-1])

                    frame_min_W = W * DroneTrjDec_allowed_BB_within_frame_W_by_fraction[0]
                    frame_max_W = W * DroneTrjDec_allowed_BB_within_frame_W_by_fraction[1]
                    frame_min_H = H * DroneTrjDec_allowed_BB_within_frame_H_by_fraction[0]
                    frame_max_H = H * DroneTrjDec_allowed_BB_within_frame_H_by_fraction[1]

                    flag_trajectory_valid = (frame_min_W < trajectory_min_W) and (frame_max_W > trajectory_max_W) and \
                                            (frame_min_H < trajectory_min_H) and (frame_max_H > trajectory_max_H)

    # flag_trajectory_valid = True  #TODO: temp!, delete
    return flag_trajectory_valid



def get_trajectory_statistics_and_graphs(input_tensor, params, t_vec):
    ### Expects [input_tensor] = [T,H,W]: ###

    ### Max Over Time Statistics & FFT: ###
    input_tensor_spatial_max_over_time = input_tensor.max(-1)[0].max(-1)[0]
    input_Tensor_spatial_max_over_time_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_spatial_max_over_time, dim=0).abs(), 0)
    frequency_vec = get_frequency_vec_torch(input_tensor.shape[0], params.FPS)

    ### Mean Over Time Statistics & FFT: ###
    input_tensor_spatial_mean_over_time = input_tensor.mean([-1, -2])
    input_tensor_spatial_mean_over_time_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_spatial_mean_over_time, dim=0).abs(), 0)

    ### Contrast Over Time Statistics & FFT: ###
    input_tensor_spatial_contrast_over_time = input_tensor_spatial_max_over_time / (input_tensor_spatial_mean_over_time + 1e-4)
    input_tensor_spatial_contrast_over_time_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_spatial_contrast_over_time, dim=0).abs(), 0)

    ### Normalize Input Tensor Before Getting Statistics: ###
    input_tensor_sum = input_tensor.sum([-1, -2], True)
    input_tensor_normalized = input_tensor / (input_tensor_sum + 1e-6)
    H, W = input_tensor_normalized.shape[-2:]
    x_vec = torch.arange(W)
    y_vec = torch.arange(H)
    x_vec = x_vec - W // 2
    y_vec = y_vec - H // 2
    x_vec = x_vec.to(input_tensor_normalized.device)
    y_vec = y_vec.to(input_tensor_normalized.device)
    x_grid = x_vec.unsqueeze(0).repeat(H, 1)
    y_grid = y_vec.unsqueeze(1).repeat(1, W)
    cx = (input_tensor_normalized * x_grid).sum([-1, -2], True).squeeze()
    cy = (input_tensor_normalized * y_grid).sum([-1, -2], True).squeeze()
    cx2 = (input_tensor_normalized * x_grid ** 2).sum([-1, -2], True).squeeze().sqrt()
    cy2 = (input_tensor_normalized * y_grid ** 2).sum([-1, -2], True).squeeze().sqrt()
    x_grid_batch = x_grid.unsqueeze(0).repeat(len(cx), 1, 1)
    y_grid_batch = y_grid.unsqueeze(0).repeat(len(cy), 1, 1)
    x_grid_batch_modified = (x_grid_batch - cx.unsqueeze(-1).unsqueeze(-1))
    y_grid_batch_modified = (y_grid_batch - cy.unsqueeze(-1).unsqueeze(-1))
    cx2_modified = (input_tensor_normalized * x_grid_batch_modified ** 2).sum([-1, -2], True).squeeze().sqrt()
    cy2_modified = (input_tensor_normalized * y_grid_batch_modified ** 2).sum([-1, -2], True).squeeze().sqrt()

    ### Get Smooth COM Trajectory: ###
    # #(1). Running Mean:
    # K = 21
    # cx_smooth = convn_torch(cx, torch.ones(K)/K, dim=0)
    # (2). Polyfit:
    # coefficients, cx_smooth, residual_values = polyfit_torch(t_vec, cx, polynom_degree=2, flag_get_prediction=True, flag_get_residuals=False)
    # coefficients, cy_smooth, residual_values = polyfit_torch(t_vec, cy, polynom_degree=2, flag_get_prediction=True, flag_get_residuals=False)
    # (3). Robust Polyfit:
    coefficients_x, cx_smooth, residual_values_x, poly_residual_torch_x, best_polynomial_degree_x = \
        polyfit_torch_FitSimplestDegree(torch.arange(cx.shape[0]).to(cx.device), cx, max_degree_to_check=10, flag_get_prediction=True, flag_get_residuals=True)
    coefficients_y, cy_smooth, residual_values_y, poly_residual_torch_y, best_polynomial_degree_y = \
        polyfit_torch_FitSimplestDegree(torch.arange(cy.shape[0]).to(cy.device), cy, max_degree_to_check=10, flag_get_prediction=True, flag_get_residuals=True)



    return cx, cy, cx2_modified, cy2_modified, cx_smooth, cy_smooth, \
           input_tensor_spatial_max_over_time, input_tensor_spatial_mean_over_time, input_tensor_spatial_contrast_over_time, \
           input_Tensor_spatial_max_over_time_fft, input_tensor_spatial_mean_over_time_fft, input_tensor_spatial_contrast_over_time_fft


def get_COM_plots(input_tensor, params, t_vec, trajectory_index):
    ### Expects [input_tensor] = [T,H,W]: ###

    ### Max Over Time Statistics & FFT: ###
    input_tensor_spatial_max_over_time = input_tensor.max(-1)[0].max(-1)[0]
    input_Tensor_spatial_max_over_time_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_spatial_max_over_time, dim=0).abs(), 0)
    frequency_vec = get_frequency_vec_torch(input_tensor.shape[0], params.FPS)

    ### Mean Over Time Statistics & FFT: ###
    input_tensor_spatial_mean_over_time = input_tensor.mean([-1,-2])
    input_tensor_spatial_mean_over_time_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_spatial_mean_over_time, dim=0).abs(), 0)

    ### Contrast Over Time Statistics & FFT: ###
    input_tensor_spatial_contrast_over_time = input_tensor_spatial_max_over_time / (input_tensor_spatial_mean_over_time + 1e-4)
    input_tensor_spatial_contrast_over_time_fft = fftshift_torch_specific_dim(torch.fft.fftn(input_tensor_spatial_contrast_over_time, dim=0).abs(), 0)

    ### Normalize Input Tensor Before Getting Statistics: ###
    input_tensor_sum = input_tensor.sum([-1, -2], True)
    input_tensor_normalized = input_tensor / (input_tensor_sum + 1e-6)
    H, W = input_tensor_normalized.shape[-2:]
    x_vec = torch.arange(W)
    y_vec = torch.arange(H)
    x_vec = x_vec - W//2
    y_vec = y_vec - H//2
    x_vec = x_vec.to(input_tensor_normalized.device)
    y_vec = y_vec.to(input_tensor_normalized.device)
    x_grid = x_vec.unsqueeze(0).repeat(H,1)
    y_grid = y_vec.unsqueeze(1).repeat(1,W)
    cx = (input_tensor_normalized * x_grid).sum([-1,-2],True).squeeze()
    cy = (input_tensor_normalized * y_grid).sum([-1,-2],True).squeeze()
    cx2 = (input_tensor_normalized * x_grid**2).sum([-1,-2],True).squeeze().sqrt()
    cy2 = (input_tensor_normalized * y_grid**2).sum([-1,-2],True).squeeze().sqrt()
    x_grid_batch = x_grid.unsqueeze(0).repeat(len(cx),1,1)
    y_grid_batch = y_grid.unsqueeze(0).repeat(len(cy),1,1)
    x_grid_batch_modified = (x_grid_batch - cx.unsqueeze(-1).unsqueeze(-1))
    y_grid_batch_modified = (y_grid_batch - cy.unsqueeze(-1).unsqueeze(-1))
    cx2_modified = (input_tensor_normalized * x_grid_batch_modified**2).sum([-1,-2], True).squeeze().sqrt()
    cy2_modified = (input_tensor_normalized * y_grid_batch_modified**2).sum([-1,-2], True).squeeze().sqrt()

    ### Get Smooth COM Trajectory: ###
    # #(1). Running Mean:
    # K = 21
    # cx_smooth = convn_torch(cx, torch.ones(K)/K, dim=0)
    # #(2). Polyfit:
    # coefficients, cx_smooth, residual_values = polyfit_torch(t_vec, cx, 2, flag_get_prediction=True, flag_get_residuals=False)
    # coefficients, cy_smooth, residual_values = polyfit_torch(t_vec, cy, 2, flag_get_prediction=True, flag_get_residuals=False)
    #(3). Polyfit find:
    coefficients_x, cx_smooth, residual_values_x, poly_residual_torch_x, best_polynomial_degree_x = \
        polyfit_torch_FitSimplestDegree(torch.arange(cx.shape[0]).to(cx.device), cx, max_degree_to_check=10, flag_get_prediction=True, flag_get_residuals=True)
    coefficients_y, cy_smooth, residual_values_y, poly_residual_torch_y, best_polynomial_degree_y = \
        polyfit_torch_FitSimplestDegree(torch.arange(cy.shape[0]).to(cy.device), cy, max_degree_to_check=10, flag_get_prediction=True, flag_get_residuals=True)

    ### Write Video File: ###
    # video_name = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index) + '.avi')
    # video_torch_array_to_video(nn.Upsample(scale_factor=10)(scale_array_to_range(input_tensor.unsqueeze(1))), video_name)
    path_create_path_if_none_exists(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index)))
    video_name = os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'Trajectory_' + str(trajectory_index) + '.avi')
    if params.flag_save_interim_graphs_and_movies:
        video_torch_array_to_video(nn.Upsample(scale_factor=10)(scale_array_to_range(input_tensor.unsqueeze(1))), video_name)

    ### Save Trajectories: ###
    np.save(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'COM_X.npy'), cx.squeeze().cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'COM_Y.npy'), cy.squeeze().cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'MOI_X.npy'), cx2_modified.squeeze().cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'MOI_Y.npy'), cy2_modified.squeeze().cpu().numpy(), allow_pickle=True)
    np.save(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'TrjMov.npy'), input_tensor.cpu().numpy(), allow_pickle=True)

    ### Plot Things For Checking: ###
    path_create_path_if_none_exists(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index)))
    if params.flag_save_interim_graphs_and_movies:
        ### Plot Max Over Time: ###
        plt.close('all')
        plot_torch(input_tensor_spatial_max_over_time)
        # plt.show()
        plt.title('BGS Max Over Time')
        plt.xlabel('time')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'BGS_max_over_time.png'))
        plt.close('all')
        plt.pause(0.1)
        ### Plot Max Over Time: ###
        plt.close('all')
        plot_torch(input_tensor_spatial_max_over_time)
        # plt.show()
        plt.title('BGS Max Over Time')
        plt.xlabel('time')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'BGS_max_over_time.png'))
        plt.close('all')
        plt.pause(0.1)

        ### Plot Mean Over Time: ###
        plot_torch(input_tensor_spatial_mean_over_time)
        # plt.show()
        plt.title('BGS Mean Over Time')
        plt.xlabel('time')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'BGS_mean_over_time.png'))
        plt.close('all')
        plt.pause(0.1)
        ### Plot Contrast Over Time: ###
        plot_torch(input_tensor_spatial_contrast_over_time)
        # plt.show()
        plt.title('BGS Contrast Over Time')
        plt.xlabel('time')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'BGS_contrast_over_time.png'))
        plt.close('all')
        plt.pause(0.1)

        #(1). COM x:
        plot_torch(cx)
        plot_torch(cx_smooth)
        # plt.show()
        plt.title('center of mass location - x, polyfit degree = ' + str(best_polynomial_degree_x.item()))
        plt.xlabel('time')
        plt.ylabel('location [pixels]')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'center_of_mass_location_X.png'))
        plt.close('all')
        plt.pause(0.1)
        #(2). COM y:
        plot_torch(cy)
        plot_torch(cy_smooth)
        # plt.show()
        plt.title('center of mass location - y, polyfit degree = ' + str(best_polynomial_degree_y.item()))
        plt.xlabel('time')
        plt.ylabel('location [pixels]')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'center_of_mass_location_Y.png'))
        plt.close('all')
        plt.pause(0.1)
        #(3). Moment Of Interia (MOI) x:
        plot_torch(cx2.clamp(-7,7))
        # plt.show()
        plt.title(' moment of interia - x')
        plt.xlabel('time')
        plt.ylabel('MOI [pixels]')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'moment_of_inertia_X.png'))
        plt.close('all')
        plt.pause(0.1)
        #(4). Moment Of Interia (MOI) y:
        plot_torch(cy2.clamp(-7,7))
        # plt.show()
        plt.title(' moment of interia - y')
        plt.xlabel('time')
        plt.ylabel('MOI [pixels]')
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'moment_of_inertia_Y.png'))
        plt.close('all')
        plt.pause(0.1)

        ### FFT Graphs: ###
        # (5). Max Over Time FFT:
        plot_torch(frequency_vec, input_Tensor_spatial_max_over_time_fft)
        # plt.show()
        plt.title(' Max Over Time FFT')
        plt.xlabel('frequency[Hz]')
        plt.ylabel('')
        plt.ylim([0, 1])
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'FFT_BGS_max_over_time.png'))
        plt.close('all')
        plt.pause(0.1)
        # (6). Meam Over Time FFT:
        plot_torch(frequency_vec, input_tensor_spatial_mean_over_time_fft)
        # plt.show()
        plt.title(' Mean Over Time FFT ')
        plt.xlabel('frequency[Hz]')
        plt.ylabel('')
        plt.ylim([0, 1])
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'FFT_BGS_mean_over_time.png'))
        plt.close('all')
        plt.pause(0.1)
        # (6). Contrast Over Time FFT:
        plot_torch(frequency_vec, input_tensor_spatial_contrast_over_time_fft)
        # plt.show()
        plt.title(' Contrast Over Time FFT ')
        plt.xlabel('frequency[Hz]')
        plt.ylabel('')
        plt.ylim([0, 1])
        plt.savefig(os.path.join(params.results_folder_seq, 'Trajectory_' + str(trajectory_index), 'FFT_BGS_contrast_over_time.png'))
        plt.close('all')
        plt.pause(0.1)
        # ### Get Stats Over Variables: ###
        # max_value = input_tensor.max()
        # std_value = input_tensor.std()
        # mean_value = input_tensor.mean()
        # median_value = input_tensor.median()
        ### Fit Polynom Over cx,cx2 etc' and get std/error-bars size to get better understanding: ###
        1


    #######################################################################################



def Cut_And_Align_ROIs_Around_Trajectory_Torch(Movie, Movie_BGS, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, params):
    ### Get basic parameters: ###
    ROI_allocated_around_suspect = params['ROI_allocated_around_suspect']
    H = Movie_BGS.shape[-2]
    W = Movie_BGS.shape[-1]
    number_of_time_steps = len(t_vec)

    # # TODO: delete and replace with pure pytorch
    # t_vec_numpy = t_vec.cpu().numpy()
    # trajectory_smoothed_polynom_Y_numpy = trajectory_smoothed_polynom_Y.cpu().numpy()
    # trajectory_smoothed_polynom_X_numpy = trajectory_smoothed_polynom_X.cpu().numpy()
    # Movie_BGS_numpy = Movie_BGS.cpu().numpy()
    #
    # ### Allocated large frame grid: ###
    # xM = np.arange(H)
    # yM = np.arange(W)
    # original_grid = (t_vec_numpy, xM, yM)
    #
    # ### Allocate suspect ROI grid: ###
    # xl = np.arange(ROI_allocated_around_suspect) - np.int(ROI_allocated_around_suspect/2)
    # xg = np.array(np.meshgrid(t_vec_numpy, xl, xl)).T  #just like meshgrid(x,y) creates two matrices X,Y --> this create 3 matrices of three dimensions which are then concatated
    # number_of_pixels_per_ROI = ROI_allocated_around_suspect**2
    #
    # ### loop over suspect ROI grid and add the trajectory to each point on the grid: ###
    # for ii in range(ROI_allocated_around_suspect):
    #     for jj in range(ROI_allocated_around_suspect):
    #         xg[ii, :, jj, 1] = xg[ii, :, jj, 1] + trajectory_smoothed_polynom_Y_numpy
    #         xg[ii, :, jj, 2] = xg[ii, :, jj, 2] + trajectory_smoothed_polynom_X_numpy
    # xg = xg.reshape((number_of_pixels_per_ROI * number_of_time_steps, 3))
    #
    # ### "align" ROIs which contain suspect on top of each other by multi-dimensional interpolation): ###
    # #TODO: change to pytorch grid sample interpolation for 3D volumetric data. i'll need to turn xg into a [-1,1] tensor. i think i did it before!!!!
    # TrjMovie_numpy = scipy.interpolate.interpn(original_grid, Movie_BGS_numpy[t_vec_numpy.astype(np.int),0,:,:], xg, method='linear')  #TODO: maybe nearest neighbor???? to avoid problems
    # TrjMovie_numpy = TrjMovie_numpy.reshape([ROI_allocated_around_suspect, number_of_time_steps, ROI_allocated_around_suspect]).transpose([1, 0, 2])
    # TrjMov = torch.Tensor(TrjMovie_numpy).to(trajectory_smoothed_polynom_X.device)

    ### since the interpolation is only spatial (not interpolating between t indices) i can do a 2D interpolation: ###
    # trajectory_smoothed_polynom_X = torch.linspace(251, 251, len(trajectory_smoothed_polynom_X)).to(trajectory_smoothed_polynom_X.device)
    # trajectory_smoothed_polynom_Y = torch.linspace(279, 279, len(trajectory_smoothed_polynom_Y)).to(trajectory_smoothed_polynom_Y.device)
    # ROI_allocated_around_suspect = 51

    ROI_grid = torch.arange(ROI_allocated_around_suspect) - np.int(ROI_allocated_around_suspect/2) # [grid_shape] = [B,H,W,2] = [499, H, W, 2]    /    [input_shape] = [B,C,H,W] = [499, 1, H, W]
    ROI_grid = ROI_grid.to(Movie_BGS.device).type(torch.float)
    ROI_grid_Y, ROI_grid_X = torch.meshgrid(ROI_grid, ROI_grid)
    ROI_grid_X = torch.cat([ROI_grid_X.unsqueeze(0)]*len(trajectory_smoothed_polynom_X), 0).unsqueeze(-1)
    ROI_grid_Y = torch.cat([ROI_grid_Y.unsqueeze(0)]*len(trajectory_smoothed_polynom_Y), 0).unsqueeze(-1)
    ROI_grid_X += trajectory_smoothed_polynom_X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    ROI_grid_Y += trajectory_smoothed_polynom_Y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    ROI_grid_X_normalized = 2 * ROI_grid_X / max(params.W - 1, 1) - 1
    ROI_grid_Y_normalized = 2 * ROI_grid_Y / max(params.H - 1, 1) - 1
    output_grid = torch.cat([ROI_grid_X_normalized, ROI_grid_Y_normalized], -1)
    # output_grid = torch.cat([ROI_grid_Y, ROI_grid_X], -1)

    ### Get ROI Around Trajectory: ###
    TrjMov = torch.nn.functional.grid_sample(Movie_BGS[t_vec.type(torch.LongTensor)].float(), output_grid, 'bilinear')
    # TrjMov = torch.nn.functional.grid_sample(Movie[t_vec.type(torch.LongTensor)], output_grid, 'bilinear')
    TrjMov = TrjMov.squeeze(1)  #[T,1,H,W] -> [T,H,W]. to be changed later to [T,1,H,W] throughout the entire script
    # imshow_torch_video(TrjMov.unsqueeze(1), FPS=50, frame_stride=5)

    ### Get Trajectory Statistics And Graphs: ###
    cx, cy, cx2_modified, cy2_modified, cx_smooth, cy_smooth, \
    input_tensor_spatial_max_over_time, input_tensor_spatial_mean_over_time, input_tensor_spatial_contrast_over_time, \
    input_Tensor_spatial_max_over_time_fft, input_tensor_spatial_mean_over_time_fft, input_tensor_spatial_contrast_over_time_fft = \
        get_trajectory_statistics_and_graphs(TrjMov.abs(), params, t_vec)

    # ### Correct For Trajectory Drift & Get More Precise TrjMov: ###
    # ROI_grid_X += cx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # ROI_grid_Y += cy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # ROI_grid_X_normalized = 2 * ROI_grid_X / max(params.W - 1, 1) - 1
    # ROI_grid_Y_normalized = 2 * ROI_grid_Y / max(params.H - 1, 1) - 1
    # output_grid = torch.cat([ROI_grid_X_normalized, ROI_grid_Y_normalized], -1)
    # ### Get ROI Around Trajectory: ###
    # TrjMov = torch.nn.functional.grid_sample(Movie_BGS[t_vec.type(torch.LongTensor)].float(), output_grid, 'bilinear')
    # # TrjMov = torch.nn.functional.grid_sample(Movie[t_vec.type(torch.LongTensor)], output_grid, 'bilinear')
    # TrjMov = TrjMov.squeeze(1)  # [T,1,H,W] -> [T,H,W]. to be changed later to [T,1,H,W] throughout the entire script
    # # imshow_torch_video(TrjMov.unsqueeze(1), FPS=50, frame_stride=5)


    # #######################################################################################
    # ### Get Stats Over trajectory_smoothed_polynom_X, center_of_mass_trajectory etc': ###
    # max_velocity = 1
    # max_acceleration = 1 # can be derived numerically or from polynom coefficients (for instance, if there's a parabola peak, meaning sharp turn, it doesn't make sense...on the other hand maybe the RANSAC and polyfit got it wrong)
    # #######################################################################################


    # ### TODO: test FFT of Movie_BGS, Movie, and TrjMov: ###
    # #######################################################################################################
    # # input_tensor = Movie_BGS[t_vec.type(torch.LongTensor)]
    # # input_tensor = Movie_BGS
    # # input_tensor = Movie[t_vec.type(torch.LongTensor)]
    # input_tensor = TrjMov.unsqueeze(1)
    # H_center = None
    # W_center = None
    # area_around_center = 3
    # specific_case_string = 'TrjMovie_BGS_1_pixel_right_constant'
    # # plot_fft_graphs(input_tensor, H_center=None, W_center=None, area_around_center=None, specific_case_string='', params=params)
    #######################################################


    return TrjMov, \
           cx, cy, cx2_modified, cy2_modified, cx_smooth, cy_smooth, \
           input_tensor_spatial_max_over_time, input_tensor_spatial_mean_over_time, input_tensor_spatial_contrast_over_time




def Find_Valid_Trajectories_RANSAC_And_Align_Drone_Torch(events_points_TXY_indices, Movie, Movie_BGS, params):
    ### Get variables from params dict: ###
    TrjNumFind = params['TrjNumFind']  # max number of trajectories to look at

    ### Get input movie shape: ###
    H = Movie_BGS.shape[-2]
    W = Movie_BGS.shape[-1]
    number_of_frames = Movie_BGS.shape[0]

    ### Initialize points_not_assigned_to_lines_yet with all the points: ###
    points_not_assigned_to_lines_yet = events_points_TXY_indices

    ### PreAllocate and initialize lists which hold decision results for each trajectory: ###
    #TODO: make a part of the params dict instead of constantly re-initializing lists from numpy, which can be slow
    direction_vec = list(np.zeros(TrjNumFind))
    holding_point = list(np.zeros(TrjNumFind))
    t_vec = list(np.zeros(TrjNumFind))
    trajectory_smoothed_polynom_X = list(np.zeros(TrjNumFind))
    trajectory_smoothed_polynom_Y = list(np.zeros(TrjNumFind))
    TrjMov = list(np.zeros(TrjNumFind))
    xyz_line = list(np.zeros(TrjNumFind))
    flag_is_trajectory_valid_vec = np.ones(TrjNumFind) < 0
    flag_enough_valid_samples = np.ones(TrjNumFind) < 0  # TODO: right it as a vector of false flags for god sake
    NonLinePoints = torch.zeros([1, 3]).to(Movie_BGS.device)

    ### Loop over the max number of trajectories to find (each time we find a line candidate, get rid of the : ###
    for ii in range(TrjNumFind):
        ### if there are less then 10 points just ignore and continue: ###
        if points_not_assigned_to_lines_yet.shape[0] < 10:
            continue

        ### Estimate line using RANSAC: ###
        #TODO: when using RANSAC it can bring multiple points of the drone, and so for each t_vec there will be multiple (x,y) points.
        # understand how that effects polynom fit. maybe use the intensity center of mass or something???
        direction_vec[ii], holding_point[ii], t_vec[ii], \
        trajectory_smoothed_polynom_X[ii], trajectory_smoothed_polynom_Y[ii], \
        points_not_assigned_to_lines_yet, xyz_line[ii], flag_enough_valid_samples[ii] = \
            Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Drone_Torch(points_not_assigned_to_lines_yet, params, H, W)

        ### if the flag "line_out_of_range" is True, meaning there aren't enough valid points after RANSAC, then ignore and continue: ###
        if flag_enough_valid_samples[ii] == False:
            continue

        ### Perform Some Heuristics on trajectory to see if there's enough "meat" to bite into: ###
        flag_is_trajectory_valid_vec[ii] = Decide_If_Trajectory_Valid_Drone_Torch(xyz_line[ii],
                                                                                  trajectory_smoothed_polynom_X[ii],
                                                                                  trajectory_smoothed_polynom_Y[ii],
                                                                                  t_vec[ii],
                                                                                  direction_vec[ii],
                                                                                  number_of_frames, params)

        ### "Straighten out" and align frames where suspect is at to allow for proper frequency analysis: ###
        #TODO: both TrjMov of the BGS and of the movie itself are needed. i think at the end the TrjMov of the movie itself will be used
        TrjMov[ii], \
        cx, cy, cx2_modified, cy2_modified, cx_smooth, cy_smooth, \
        input_tensor_spatial_max_over_time, input_tensor_spatial_mean_over_time, input_tensor_spatial_contrast_over_time = \
            Cut_And_Align_ROIs_Around_Trajectory_Torch(Movie,
                                                        Movie_BGS,
                                                        t_vec[ii],
                                                        trajectory_smoothed_polynom_X[ii],
                                                        trajectory_smoothed_polynom_Y[ii], params)

        ### Check Conditions Over TrjMov: ###
        # #(1). Maximum BGS (too hot --> it's not a drone):
        # flag_trajectory_too_hot = input_tensor_spatial_max_over_time.max() > params.DroneTrjDec_max_BGS_threshold
        # if flag_trajectory_too_hot:
        #     flag_is_trajectory_valid_vec[ii] = False

        # #(2). Number Of Peaks / Polynomial Fit:
        # coefficients_x, prediction_x, residual_values_x, poly_residual_torch_x, best_polynomial_degree_x = \
        #     polyfit_torch_FitSimplestDegree(torch.arange(cx.shape[0]).to(cx.device), cx, max_degree_to_check=10, flag_get_prediction=True, flag_get_residuals=True)
        # coefficients_y, prediction_y, residual_values_y, poly_residual_torch_y, best_polynomial_degree_y = \
        #     polyfit_torch_FitSimplestDegree(torch.arange(cy.shape[0]).to(cy.device), cy, max_degree_to_check=10, flag_get_prediction=True, flag_get_residuals=True)
        # if best_polynomial_degree_y >= 4 or best_polynomial_degree_x >= 4:
        #     flag_is_trajectory_valid_vec[ii] = False

        # plot_torch(cy); plot_torch(prediction_y)
        # plot_torch(cx); plot_torch(prediction_x)
        # plot_torch(poly_residual_torch_y)
        # plot_torch(poly_residual_torch_x)

        # cy_smooth_loess = smooth_vec_loess(cy, window=5, use_matrix=False)
        # cy_smooth_convn = convn_torch(cy, torch.ones(11) / 11, 0).squeeze()
        # poly_residual = []
        # for poly_degree in np.arange(10):
        #     _, cy_smooth_poly, residual_values = polyfit_torch(torch.arange(len(cy)).to(cy.device), cy, polynom_degree=poly_degree, flag_get_prediction=True, flag_get_residuals=True)
        #     poly_residual.append(residual_values.abs().mean().item() / (len(residual_values) - poly_degree - 1))
        # poly_residual_tensor = torch.tensor(poly_residual)
        # laplace_b = ((poly_residual_tensor).abs().sum() / len(poly_residual_tensor))
        # laplace_fit, coeffs, covariance_laplace = fit_Laplace(np.arange(10), scale_array_to_range(poly_residual_tensor).cpu().numpy())
        # a, x0, laplace_b = coeffs
        # poly_residual_tensor_stretched = scale_array_to_range(poly_residual_tensor)
        # laplace_y = a / (2 * laplace_b) * torch.exp(-(torch.arange(10) - x0).abs() / laplace_b)
        # plot_torch(poly_residual_tensor_stretched);
        # plot_torch(laplace_y);
        # plot_torch(scale_array_to_range(poly_residual_tensor))
        # plt.plot(laplace_fit)
        # plot_torch(cy);
        # plot_torch(cy_smooth_loess);
        # plot_torch(cy_smooth_poly);
        # plot_torch(cy_smooth_convn);
        # plt.legend(['cy', 'cy_loess', 'cy_poly', 'cy_convn']);
        # plt.show()


        ### Decide Whether Trajectory Is Valid: ###
        if ~flag_is_trajectory_valid_vec[ii]:
            TrjMov[ii] = []
            # (*). NonLinePoints are points which RANSAC found to be a part of a line but after some heuristics in the above function i decided to ignore!
            #TODO: perhapse add to NonLinePoints some points around current trajectory ?
            NonLinePoints = torch.cat((NonLinePoints, xyz_line[ii]), 0)
            continue



    ### Only keep the "valid" trajectories which passed through all the heuristics: ###
    valid_indices = np.where(flag_is_trajectory_valid_vec)[0]
    direction_vec = get_elements_from_list_by_indices(direction_vec, valid_indices)
    holding_point = get_elements_from_list_by_indices(holding_point, valid_indices)
    t_vec = get_elements_from_list_by_indices(t_vec, valid_indices)
    trajectory_smoothed_polynom_X = get_elements_from_list_by_indices(trajectory_smoothed_polynom_X, valid_indices)
    trajectory_smoothed_polynom_Y = get_elements_from_list_by_indices(trajectory_smoothed_polynom_Y, valid_indices)
    xyz_line = get_elements_from_list_by_indices(xyz_line, valid_indices)
    TrjMov = get_elements_from_list_by_indices(TrjMov, valid_indices)
    num_of_trj_found = np.sum(flag_is_trajectory_valid_vec)

    ### TODO: put into main RANSAC loop later when you have decision thresholds: ###
    ### Get Plots Of COM and MOM: ###
    if params.flag_save_interim_graphs_and_movies:
        for trajectory_index in np.arange(len(TrjMov)):
            get_COM_plots(TrjMov[trajectory_index].abs(), params, t_vec[trajectory_index], trajectory_index)

    # direction_vec = list(np.array(direction_vec)[valid_indices])
    # holding_point = list(np.array(holding_point)[valid_indices])
    # t_vec = list(np.array(t_vec)[valid_indices])
    # trajectory_smoothed_polynom_X = list(np.array(trajectory_smoothed_polynom_X)[valid_indices])
    # trajectory_smoothed_polynom_Y = list(np.array(trajectory_smoothed_polynom_Y)[valid_indices])
    # xyz_line = list(np.array(xyz_line)[valid_indices])
    # TrjMov = list(np.array(TrjMov)[valid_indices])

    return points_not_assigned_to_lines_yet, NonLinePoints, direction_vec, holding_point, t_vec, \
           trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, TrjMov, num_of_trj_found



def check_for_each_trajectory_if_inside_flashlight_polygons(Movie, current_frames_for_video_writer, trajectory_tuple, total_polygon_points):
    number_of_polygons = len(total_polygon_points)
    flag_drone_trajectory_inside_BB_list_of_lists = []
    for polygon_index in np.arange(number_of_polygons):
        current_polygon_points = np.array(total_polygon_points[polygon_index])
        flag_drone_trajectory_inside_BB_list, current_frames_for_video_writer =\
            check_for_each_trajectory_if_inside_flashlight_polygon(Movie, current_frames_for_video_writer, trajectory_tuple, current_polygon_points)
        flag_drone_trajectory_inside_BB_list_of_lists.append(flag_drone_trajectory_inside_BB_list)

    ### For Each Trajectory Check Whether It's In ANY (Flashlight) Polygon Bounding-Box By Summing Over BB dim: ###
    final_results = np.array(flag_drone_trajectory_inside_BB_list_of_lists).sum(0) >= 1
    final_results = final_results.tolist()

    return final_results, current_frames_for_video_writer

def check_for_each_trajectory_if_inside_flashlight_polygon(Movie, current_frames_for_video_writer, trajectory_tuple, current_polygon_points):
    ### Unpack trajectory_tuple: ###
    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y = trajectory_tuple

    ### Initialize Parameters: ###
    full_t_vec = np.arange(Movie.shape[0])
    number_of_drone_trajectories_found = len(t_vec)
    trajectory_inside_flashlight_polygon_list = create_empty_list_of_lists(number_of_drone_trajectories_found)
    flag_drone_trajectory_inside_BB_list = []

    if current_polygon_points.all() != None:
        flashlight_polygon_path_object = matplotlib.path.Path(current_polygon_points)

        # (1). Loop over all trajectories:
        for trajectory_index in np.arange(number_of_drone_trajectories_found):
            current_trajectory_t_vec = t_vec[trajectory_index]
            in_trajectory_t_vec_counter = -1  # since trajectory_tuple is not "full", meaning, it isn't filled with None when a trajectory doesn't include a certain frame, it's size is smaller then full_t_vec, and therefore must have it's own counter
            # (2). Loop over all time steps:
            for inter_frame_index in full_t_vec:
                ### Get current movie frame: ###
                current_movie_frame = current_frames_for_video_writer[inter_frame_index]  # TODO: remove that to outside the function!!!!! plot the trajectories in a seperate function

                # (*). if current inter_frame_index is in trajectory_t_vec then check if the trajectory is inside the flashlight polygon:
                if inter_frame_index in current_trajectory_t_vec:
                    in_trajectory_t_vec_counter += 1
                    ### Get current trajectory point on image: ###
                    current_x = trajectory_smoothed_polynom_X[trajectory_index][in_trajectory_t_vec_counter]
                    current_y = trajectory_smoothed_polynom_Y[trajectory_index][in_trajectory_t_vec_counter]
                    current_t = t_vec[trajectory_index][in_trajectory_t_vec_counter]
                    current_trajectory_point = (np.int(current_x), np.int(current_y))

                    ### Check if current trajectory point is inside polygon: ###
                    flag_current_point_inside_flashlight_polygon = flashlight_polygon_path_object.contains_point(current_trajectory_point)
                    trajectory_inside_flashlight_polygon_list[trajectory_index].append(flag_current_point_inside_flashlight_polygon)

                    ### Add Trajectory Bounding Box To Frame: ###
                    current_movie_frame = draw_circle_with_label_on_image(current_movie_frame, current_trajectory_point, circle_label='Trj' + str(trajectory_index))

                else:
                    # (*). if current inter_frame_index is NOT inside trajectory_t_vec,
                    # that means the current trajectory has no valid points in this frame
                    # (*). therefore nothing to draw
                    1

                ### In any case draw the flashlight polygon: ###
                current_movie_frame = current_frames_for_video_writer[inter_frame_index]

            ### Decide Whether the statistics of the current trajectory being inside the flashlight polygon are good enough to say "we got it": ###
            trajectory_inside_flashlight_polygon_fraction = sum(trajectory_inside_flashlight_polygon_list[trajectory_index]) / \
                                                            len(trajectory_inside_flashlight_polygon_list[trajectory_index])
            if trajectory_inside_flashlight_polygon_fraction > 0.5:
                flag_drone_inside_BB = True
            else:
                flag_drone_inside_BB = False  # no drone was detected, so it doesn't really matter
            flag_drone_trajectory_inside_BB_list.append(flag_drone_inside_BB)

    return flag_drone_trajectory_inside_BB_list, current_frames_for_video_writer




def get_Distance_From_Points_To_DirectionVec_Torch(xyz, b, p):
    #(*). b is a unit vector!
    ### calculates total distance between xyz trajectory (centered by p holding point) and unit vector b: ###
    number_of_points, number_of_dimensions = xyz.shape
    xyz_centered = xyz - p
    xyz_centered_dot_product_with_b = torch.matmul(xyz_centered, b.unsqueeze(1)).reshape(number_of_points, 1)    # SCALAR = project xyz_centered on b, which is a UNIT VECTOR!
    xyz_centered_projection_on_b = torch.matmul(xyz_centered_dot_product_with_b, b.unsqueeze(1).T)   # multiply above scalar projection by b vec
    dist = torch.sum((xyz_centered_projection_on_b - xyz_centered)**2, 1)**.5  #distance from each point on the trajectory points to the line projection on b
    return dist









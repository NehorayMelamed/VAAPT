import torch

from RapidBase.import_all import *
from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import *
from QS_Jetson_mvp_milestone.functional_utils.plotting_utils import *
from QS_Jetson_mvp_milestone.functional_utils.Optimization_utils import *
from QS_Jetson_mvp_milestone.functional_utils.FFT_analysis import *


def Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Drone(res_points, params, H, W):
    ### Get variables from params dict: ###
    DroneLineEst_RANSAC_D = params['DroneLineEst_RANSAC_D']
    DroneLineEst_polyfit_degree = params['DroneLineEst_polyfit_degree']
    DroneLineEst_RANSAC_max_trials = params['DroneLineEst_RANSAC_max_trials']
    minimum_number_of_samples_after_polyfit = params['DroneLineEst_minimum_number_of_samples_after_polyfit']
    minimum_number_of_samples_before_polyfit = params['DroneLineEst_minimum_number_of_samples_before_polyfit']
    ROI_allocated_around_suspect = params['DroneLineEst_ROI_allocated_around_suspect']

    ### Perform RANSAC to estimate line: ###
    # (*). Remember, this accepts ALL the residual points coordinates and outputs a line. shouldn't we use something like clustering???
    model_robust, indices_within_distance = ransac(res_points, LineModelND, min_samples=2,
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
        fitx = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 1], DroneLineEst_polyfit_degree))  # fit (t,x) polynomial
        fity = np.poly1d(np.polyfit(valid_trajectory_points[:, 0], valid_trajectory_points[:, 2], DroneLineEst_polyfit_degree))  # fit (t,y) polynomial

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
        indices_2 = get_Distance_From_Points_To_DirectionVec(smoothed_trajectory_over_time_vec, direction_vec1, holding_point1) < (DroneLineEst_RANSAC_D)

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

    ### make sure there are enough valid points: ###
    if len(t_vec) > minimum_number_of_samples_after_polyfit:
        flag_enough_valid_samples = True
    else:
        flag_enough_valid_samples = False

    ### Get points within the predefined distance and those above it: ###   #TODO: what do we do with points_new. i should really name points_new to be points_far_from_line or points_off_line
    points_off_line_found = res_points[indices_within_distance == False]
    points_on_line_found = res_points[indices_within_distance]

    return direction_vec1, holding_point1, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, points_off_line_found, points_on_line_found, flag_enough_valid_samples


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



def Decide_If_Trajectory_Valid_Flashlight(xyz_line, t_vec, direction_vec, number_of_frames, params):
    ### Get variables from params dictionary: ###
    FlashlightTrjDec_minimum_fraction_of_points_inside_RANSAC_D = params['FlashlightTrjDec_minimum_fraction_of_points_inside_RANSAC_D']
    FlashlightTrjDec_minimum_fraction_of_frames_where_flashlight_is_visible = params['FlashlightTrjDec_minimum_fraction_of_frames_where_flashlight_is_visible']
    FlashlightTrjDec_allowed_BB_within_frame_H_by_fraction = params['FlashlightTrjDec_allowed_BB_within_frame_H_by_fraction']
    FlashlightTrjDec_allowed_BB_within_frame_W_by_fraction = params['FlashlightTrjDec_allowed_BB_within_frame_W_by_fraction']
    H = params.H
    W = params.W

    ### make sure there are enough time stamps to even look into trajectory: ###
    flag_trajectory_valid = len(t_vec) > (number_of_frames * FlashlightTrjDec_minimum_fraction_of_frames_where_flashlight_is_visible)

    if flag_trajectory_valid:
        ### Make sure flashlight trajectory found is within the predefined boundaries within the frame (basically to avoid cars below): ###
        #TODO: xyz_line should be called txy_line or TWH_line or something...
        trajectory_min_H = min(xyz_line[:,2])
        trajectory_max_H = max(xyz_line[:,2])
        trajectory_min_W = min(xyz_line[:, 1])
        trajectory_max_W = max(xyz_line[:, 1])

        frame_min_W = W * FlashlightTrjDec_allowed_BB_within_frame_W_by_fraction[0]
        frame_max_W = W * FlashlightTrjDec_allowed_BB_within_frame_W_by_fraction[1]
        frame_min_H = H * FlashlightTrjDec_allowed_BB_within_frame_H_by_fraction[0]
        frame_max_H = H * FlashlightTrjDec_allowed_BB_within_frame_H_by_fraction[1]

        flag_trajectory_valid = (frame_min_W < trajectory_min_W) and (frame_max_W > trajectory_max_W) and \
                                (frame_min_H < trajectory_min_H) and (frame_max_H > trajectory_max_H)

    return flag_trajectory_valid


def Decide_If_Trajectory_Valid_Drone(xyz_line, t_vec, direction_vec, number_of_frames, params):
    ### Get variables from params dictionary: ###
    #TODO: change all the names! TrjDecMaxVel -> TrjDec_max_velocity_fraction_on_XY_plane
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
        b_abs = np.abs(direction_vec)[:, 0]

        ### the portion of the unit (t,x,y) direction vec's projection on the (x,y) plane: ###
        v = (b_abs[1] ** 2 + b_abs[2] ** 2) ** 0.5

        ### make sure the drone's "velocity" in the (x,y) is between some values, if it's too stationary it's weird and if it's too fast it's weird: ###
        flag_trajectory_valid = ((v < DroneTrjDec_maximum_projection_onto_XY_plane) & (v > DroneTrjDec_minimum_projection_onto_XY_plane))
        if flag_trajectory_valid:

            ### make sure max difference between time stamps, which is also the maximum consecutive frames suspect disappeared, is not too high: ###
            flag_trajectory_valid = (np.max(np.diff(xyz_line[:, 0])) < DroneTrjDec_max_time_gap_in_trajectory)
            if flag_trajectory_valid:

                ### Make sure total time for valid points is above the minimum we decided upon: ###
                minimum_total_valid_time_range_for_suspect = (DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible * SeqT * FrameRate)  #this is stupid, SeqT*FrameRate = number of frames
                flag_trajectory_valid = minimum_total_valid_time_range_for_suspect < (np.max(t_vec) - np.min(t_vec))

                ### Make sure the trajectory boundaries are within predefined range within the image frame (basically not to low to avoid cars): ###
                if flag_trajectory_valid:
                    trajectory_min_H = min(xyz_line[:, 2])
                    trajectory_max_H = max(xyz_line[:, 2])
                    trajectory_min_W = min(xyz_line[:, 1])
                    trajectory_max_W = max(xyz_line[:, 1])

                    frame_min_W = W * DroneTrjDec_allowed_BB_within_frame_W_by_fraction[0]
                    frame_max_W = W * DroneTrjDec_allowed_BB_within_frame_W_by_fraction[1]
                    frame_min_H = H * DroneTrjDec_allowed_BB_within_frame_H_by_fraction[0]
                    frame_max_H = H * DroneTrjDec_allowed_BB_within_frame_H_by_fraction[1]

                    flag_trajectory_valid = (frame_min_W < trajectory_min_W) and (frame_max_W > trajectory_max_W) and \
                                            (frame_min_H < trajectory_min_H) and (frame_max_H > trajectory_max_H)

    return flag_trajectory_valid


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
    # t_vec is the smooth, incrementally increasing, t_vec.   t_vec = np.arange(min(xyz_line[:,0]), max(xyz_line[:,0])))
    flag_trajectory_valid = len(t_vec) > (number_of_frames * DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible)
    output_message = ''
    if flag_trajectory_valid == False:
        output_message = 'number of trajectory time steps too little. len(t_vec) = ' + str(len(t_vec)) + ', minimum required: ' + str(number_of_frames * DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible)
    if flag_trajectory_valid:
        ### make sure velocity is between min and max values: ###
        b_abs = direction_vec.abs()

        ### the portion of the unit (t,x,y) direction vec's projection on the (x,y) plane: ###
        v = (b_abs[1] ** 2 + b_abs[2] ** 2) ** 0.5

        ### make sure the drone's "velocity" in the (x,y) is between some values, if it's too stationary it's weird and if it's too fast it's weird: ###
        #TODO: for experiments DroneTrjDec_minimum_projection_onto_XY_plane should be 0, IN REAL LIFE it should be >0 because drones usually move
        flag_trajectory_valid = ((v < DroneTrjDec_maximum_projection_onto_XY_plane) & (v >= DroneTrjDec_minimum_projection_onto_XY_plane))
        if flag_trajectory_valid == False:
            output_message = 'drone trajectory projection onto XY plane not in the appropriate range'
        if flag_trajectory_valid:

            ### make sure max difference between time stamps, which is also the maximum consecutive frames suspect disappeared, is not too high: ###
            original_t_vec = xyz_line[:, 0]
            original_t_vec_unique = original_t_vec.unique()   #SLIGHTLY different from t_vec....t_vec is built using np.arange(min_events_T, max_events_T), this is simply number of unique values
            flag_trajectory_valid = original_t_vec_unique.diff().max() < DroneTrjDec_max_time_gap_in_trajectory
            if flag_trajectory_valid == False:
                output_message = 'max time different between time stamps is too large'
            if flag_trajectory_valid:

                ### Make sure total time for valid points is above the minimum we decided upon: ###
                total_amount_of_valid_original_t_vec_samples = len(original_t_vec_unique)
                minimum_total_valid_time_range_for_suspect = (DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible * number_of_frames)  # this is stupid, SeqT*FrameRate = number of frames
                flag_trajectory_valid = minimum_total_valid_time_range_for_suspect < total_amount_of_valid_original_t_vec_samples

                if flag_trajectory_valid == False:
                    output_message = 'total valid time for suspect is not enough ' +\
                                     ', original len(t_vec): ' +\
                                     str(total_amount_of_valid_original_t_vec_samples) +\
                                     ', whereas minimum amount of time wanted is: ' \
                                     + str(minimum_total_valid_time_range_for_suspect)

    # flag_trajectory_valid = True  #TODO: temp!, delete
    return flag_trajectory_valid, output_message


def Cut_And_Align_ROIs_Around_Trajectory(Movie_BGS, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, params):
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
    xg = np.array(np.meshgrid(t_vec, xl, xl)).T  #just like meshgrid(x,y) creates two matrices X,Y --> this create 3 matrices of three dimensions which are then concatated to create a 4D array
    number_of_pixels_per_ROI = ROI_allocated_around_suspect**2
    # imshow_torch_video(torch.Tensor(Movie_BGS).unsqueeze(1), None, 50)
    # imshow(Movie_BGS[0]); plt.show()

    ### loop over suspect ROI grid and add the trajectory to each point on the grid: ###
    for ii in range(ROI_allocated_around_suspect):
        for jj in range(ROI_allocated_around_suspect):
            # xg[ii, :, jj, 1] = xg[ii, :, jj, 1] + trajectory_smoothed_polynom_X  #TODO: this is the way it was before....i wonder how it ever worked
            # xg[ii, :, jj, 2] = xg[ii, :, jj, 2] + trajectory_smoothed_polynom_Y
            xg[ii, :, jj, 1] = xg[ii, :, jj, 1] + trajectory_smoothed_polynom_Y
            xg[ii, :, jj, 2] = xg[ii, :, jj, 2] + trajectory_smoothed_polynom_X
    xg = xg.reshape((number_of_pixels_per_ROI*number_of_time_steps, 3))

    ### "align" ROIs which contain suspect on top of each other by multi-dimensional interpolation): ###
    #TODO: i really think there's a mixup in the way we construct xg....otherwise everything's weird
    # original_grid = (t,H,W) , Movie_BGS.shape = (t,H,W), xg = (t,W,H) !!!! how did anything work so far??!?!?!?
    TrjMov = scipy.interpolate.interpn(original_grid, Movie_BGS[t_vec,:,:], xg, method='linear')  #TODO: maybe nearest neighbor???? to avoid problems
    TrjMov = TrjMov.reshape([ROI_allocated_around_suspect, number_of_time_steps, ROI_allocated_around_suspect]).transpose([1, 0, 2])

    #TODO: there's probably a faster way of doing it: ###
    return TrjMov

# from pykeops.torch import LazyTensor
from RapidBase.MISC_REPOS.torchcubicspline.torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

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

    return TrjMov


def Cut_And_Align_ROIs_Around_Trajectory_And_Get_Auxiliary_Trajectories_Torch(Movie, Movie_BGS, t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, params):
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


def Find_Valid_Trajectories_RANSAC_And_Align_Drone(events_points_TXY_indices, Movie_BGS, params):
    ### Get variables from params dict: ###
    TrjNumFind = params['TrjNumFind']  # max number of trajectories to look at

    ### Get input movie shape: ###
    H = Movie_BGS.shape[-2]
    W = Movie_BGS.shape[-1]
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
            Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Drone(points_not_assigned_to_lines_yet, params, H, W)

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
    trajectory_smoothed_polynom_X = list(np.array(trajectory_smoothed_polynom_X)[np.where(flag_is_trajectory_valid_vec)])
    trajectory_smoothed_polynom_Y = list(np.array(trajectory_smoothed_polynom_Y)[np.where(flag_is_trajectory_valid_vec)])
    xyz_line = list(np.array(xyz_line)[np.where(flag_is_trajectory_valid_vec)])
    TrjMov = list(np.array(TrjMov)[np.where(flag_is_trajectory_valid_vec)])
    num_of_trj_found = np.sum(flag_is_trajectory_valid_vec)

    return points_not_assigned_to_lines_yet, NonLinePoints, direction_vec, holding_point, t_vec, \
           trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, TrjMov, num_of_trj_found


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
        #TODO!!!!!!: what i should use is either connected-components analysis on the line fit by RANSAC or use different radii to try and kill all the points around the line!!!!!!!
        # THIS IS EXTERMELY NEEDED TO AVOID THE RANSAC MISSING POINTS AND RUNNING WILD. ANOTHER THING I CAN TRY AND DO IS PERFORM MORPHOLOGICAL CLOSING EXCEPT FOR
        # CENTER OF MASS WHICH CAN BE APPROXIMATELY FOUND USING CONVOLUTIONS

        #TODO!!!!!: in the future, to avoid missing close drone swarms, we need to segment and follow BLOBS/CHUNKS/CLUSTERS of outliers!!!!, and balance between RANSAC load and missing positives
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



def Find_Valid_Trajectories_OnlyRANSAC_And_Align_Drone_Torch(events_points_TXY_indices, Movie, Movie_BGS, params):
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
        #TODO!!!!!!: what i should use is either connected-components analysis on the line fit by RANSAC or use different radii to try and kill all the points around the line!!!!!!!
        # THIS IS EXTERMELY NEEDED TO AVOID THE RANSAC MISSING POINTS AND RUNNING WILD. ANOTHER THING I CAN TRY AND DO IS PERFORM MORPHOLOGICAL CLOSING EXCEPT FOR
        # CENTER OF MASS WHICH CAN BE APPROXIMATELY FOUND USING CONVOLUTIONS

        #TODO!!!!!: in the future, to avoid missing close drone swarms, we need to segment and follow BLOBS/CHUNKS/CLUSTERS of outliers!!!!, and balance between RANSAC load and missing positives
        direction_vec[ii], holding_point[ii], t_vec[ii], \
        trajectory_smoothed_polynom_X[ii], trajectory_smoothed_polynom_Y[ii], \
        points_not_assigned_to_lines_yet, xyz_line[ii], flag_enough_valid_samples[ii] = \
            Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Drone_Torch(points_not_assigned_to_lines_yet, params, H, W)

        ### if the flag "line_out_of_range" is True, meaning there aren't enough valid points after RANSAC, then ignore and continue: ###
        if flag_enough_valid_samples[ii] == False:
            continue

        ### Perform Some Heuristics on trajectory to see if there's enough "meat" to bite into: ###
        flag_is_trajectory_valid_vec[ii], output_message = Decide_If_Trajectory_Valid_Drone_Torch(xyz_line[ii],
                                                                                  trajectory_smoothed_polynom_X[ii],
                                                                                  trajectory_smoothed_polynom_Y[ii],
                                                                                  t_vec[ii],
                                                                                  direction_vec[ii],
                                                                                  number_of_frames, params)

        # ### Print Results: ###
        # print('X: ' + str(trajectory_smoothed_polynom_X[ii][0].item()) + ', Y: ' + str(trajectory_smoothed_polynom_Y[ii][0].item()))
        # if flag_is_trajectory_valid_vec[ii]:
        #     print('Trajectory Valid')
        # else:
        #     print(output_message)

        ### "Straighten out" and align frames where suspect is at to allow for proper frequency analysis: ###
        #TODO: both TrjMov of the BGS and of the movie itself are needed. i think at the end the TrjMov of the movie itself will be used
        TrjMov[ii] = \
            Cut_And_Align_ROIs_Around_Trajectory_Torch(Movie,
                                                        Movie_BGS,
                                                        t_vec[ii],
                                                        trajectory_smoothed_polynom_X[ii],
                                                        trajectory_smoothed_polynom_Y[ii], params)


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
    xyz_centered_dot_product_with_b = torch.matmul(xyz_centered, b.unsqueeze(1)).reshape(number_of_points, 1)    # SCALAR = project xyz_centered on b, which is a UNIT VECTOR!
    xyz_centered_projection_on_b = torch.matmul(xyz_centered_dot_product_with_b, b.unsqueeze(1).T)   # multiply above scalar projection by b vec
    dist = torch.sum((xyz_centered_projection_on_b - xyz_centered)**2, 1)**.5  #distance from each point on the trajectory points to the line projection on b
    return dist


def Find_Flashlight_Trajectory(events_points_TXY_indices, Movie_BGS, params):
    ### Get variables from params dict: ###
    TrjNumFind = params['TrjNumFind_Flashlight']  # max number of trajectories to look at

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
        minimum_number_of_event_points_to_work_with = 100 #(*). the flashlight should be ON AND VISIBLE for a significant part of the frame
        if points_not_assigned_to_lines_yet.shape[0] < minimum_number_of_event_points_to_work_with:
            continue

        ### Estimate line using RANSAC: ###
        direction_vec[ii], holding_point[ii], t_vec[ii], \
        trajectory_smoothed_polynom_X[ii], trajectory_smoothed_polynom_Y[ii], \
        points_not_assigned_to_lines_yet, xyz_line[ii], flag_enough_valid_samples[ii] = \
            Estimate_Line_From_Event_Indices_Using_RANSAC_And_LineFit_Flashlight(points_not_assigned_to_lines_yet, params, H, W)

        ### if the flag "line_out_of_range" is True, meaning there aren't enough valid points after RANSAC, then ignore and continue: ###
        if flag_enough_valid_samples[ii] == False:
            continue

        ### Perform Some Heuristics on trajectory to see if there's enough "meat" to bite into: ###
        flag_is_trajectory_valid_vec[ii] = Decide_If_Trajectory_Valid_Flashlight(xyz_line[ii],
                                                                      t_vec[ii],
                                                                      direction_vec[ii],
                                                                      number_of_frames, params)
        # flag_is_trajectory_valid_vec[ii] = True

        ### If trajectory is not a valid flashlight then dismiss them: ###
        if ~flag_is_trajectory_valid_vec[ii]:
            TrjMov[ii] = []
            # (*). NonLinePoints are points which RANSAC found to be a part of a line but after some heuristics in the above function i decided to ignore!
            NonLinePoints = np.concatenate((NonLinePoints, xyz_line[ii]))
            continue


    ### Only keep the "valid" trajectories which passed through all the heuristics: ###
    direction_vec = list(np.array(direction_vec)[np.where(flag_is_trajectory_valid_vec)])
    holding_point = list(np.array(holding_point)[np.where(flag_is_trajectory_valid_vec)])
    t_vec = list(np.array(t_vec)[np.where(flag_is_trajectory_valid_vec)])
    trajectory_smoothed_polynom_X = list(np.array(trajectory_smoothed_polynom_X)[np.where(flag_is_trajectory_valid_vec)])
    trajectory_smoothed_polynom_Y = list(np.array(trajectory_smoothed_polynom_Y)[np.where(flag_is_trajectory_valid_vec)])
    xyz_line = list(np.array(xyz_line)[np.where(flag_is_trajectory_valid_vec)])
    TrjMov = list(np.array(TrjMov)[np.where(flag_is_trajectory_valid_vec)])
    num_of_trj_found = np.sum(flag_is_trajectory_valid_vec)

    flashlight_trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)

    return points_not_assigned_to_lines_yet, NonLinePoints, direction_vec, holding_point, t_vec, \
           trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line, flashlight_trajectory_tuple, TrjMov, num_of_trj_found





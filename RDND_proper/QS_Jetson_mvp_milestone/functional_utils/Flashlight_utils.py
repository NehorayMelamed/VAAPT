
from RapidBase.import_all import *
from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import *
from QS_Jetson_mvp_milestone.functional_utils.trajectory_utils import *
from QS_Jetson_mvp_milestone.functional_utils.plotting_utils import *


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





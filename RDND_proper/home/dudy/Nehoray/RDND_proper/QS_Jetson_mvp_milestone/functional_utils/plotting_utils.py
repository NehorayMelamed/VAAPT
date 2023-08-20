
from RapidBase.import_all import *

def draw_polygon_points_on_image(image_frame, polygon_points):
    for i in np.arange(len(polygon_points)-1):
        if type(polygon_points[i]) != tuple:
            cv2.line(image_frame, tuple(polygon_points[i]), tuple(polygon_points[i + 1]), (255, 255, 255), 2)
        else:
            cv2.line(image_frame, polygon_points[i], polygon_points[i + 1], (255, 255, 255), 2)

    if type(polygon_points[-1]) != tuple:
        cv2.line(image_frame, tuple(polygon_points[-1]), tuple(polygon_points[0]), (255,255,255), 2)  #close the polygon
    else:
        cv2.line(image_frame, polygon_points[-1], polygon_points[0], (255, 255, 255), 2)

    return image_frame


def draw_polygons_on_image(image_frame, polygon_points):
    if polygon_points is None:
        return image_frame
    for i in np.arange(len(polygon_points)):
        image_frame = draw_polygon_points_on_image(image_frame, polygon_points[i])
    return image_frame


def draw_circles_with_trajectory_labels_on_image(image_frame, circle_points, circle_radius_in_pixels=5):
    for i in np.arange(len(circle_points)):
        current_circle_point = circle_points[i]
        cv2.circle(image_frame, current_circle_point, circle_radius_in_pixels, (255, 255, 255), 3)
        image_frame = cv2.putText(image_frame, 'Trj' + str(i), (
            current_circle_point[0] - circle_radius_in_pixels,
            current_circle_point[1] - circle_radius_in_pixels),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                  cv2.LINE_AA)
    return image_frame

def draw_circle_with_label_on_image(image_frame, circle_point, circle_label='Trj0', circle_radius_in_pixels=5):
    cv2.circle(image_frame, circle_point, circle_radius_in_pixels, (255, 255, 255), 3)
    image_frame = cv2.putText(image_frame, circle_label, (
        circle_point[0] - circle_radius_in_pixels,
        circle_point[1] - circle_radius_in_pixels),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                              cv2.LINE_AA)
    return image_frame

def draw_text_on_image(image_frame, text_for_image='', org=(0,30), fontScale=0.7, color=(0,255,0), thickness=2):
    image_frame = cv2.putText(img=image_frame,
                                            text=text_for_image,
                                            org=org,
                                            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                            fontScale=fontScale,
                                            color=color,
                                            thickness=thickness)
    return image_frame

def draw_trajectories_on_images(input_frames, trajectory_tuple):
    BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(input_frames,
                                                                    fps=50,
                                                                    tit='',
                                                                    Res_dir='',
                                                                    flag_save_movie=0,
                                                                    trajectory_tuple=trajectory_tuple)
    input_frames, frames_list = Plot_BoundingBox_On_Movie(input_frames,
                              fps=50,
                              tit='',
                              Res_dir='',
                              flag_save_movie=0,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)

    return frames_list


def Plot_BoundingBoxes_On_Video(Mov, fps=1000, tit="tit", flag_transpose_movie=0, frame_skip_steps=1, resize_factor=1, flag_save_movie=0, Res_dir='Res_dir', histogram_limits_to_stretch =0.001, trajectory_tuple = []):
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
        # OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir,tit + ".avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (resize_factor * Mov.shape[iax0], resize_factor * Mov.shape[iax1]), 0)
        OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir,tit + ".avi"), 0, fps, (resize_factor * Mov.shape[iax0], resize_factor * Mov.shape[iax1]), 0)

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


def Get_BoundingBox_List_For_Each_Frame(Mov, fps=1000, tit="tit", Res_dir='Res_dir', flag_save_movie=1, trajectory_tuple=[]):
    # trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)

    ### Initialize Things: ###
    # flag_found_trajectories = len(trajectory_tuple) == 3
    flag_found_trajectories = trajectory_tuple[0] != []
    if flag_found_trajectories == False:
        return
    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    if len(Mov.shape) == 3:
        number_of_frames_total, H, W = Mov.shape
    elif len(Mov.shape) == 4:
        number_of_frames_total, H, W, C = Mov.shape
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

                # ### Keep Track Of Bounding-Boxes (H_index, W_index): ###
                #TODO: make the entire script be (H,W) instead of (X,Y)
                # BoundingBox_PerFrame_list[trajectory_index].append((target_spatial_coordinates[1],target_spatial_coordinates[0]))
                ### Keep Track Of Bounding-Boxes (Xindex, Yindex): ###
                BoundingBox_PerFrame_list[trajectory_index].append((target_spatial_coordinates[0], target_spatial_coordinates[1]))
            else:
                ### If there isn't a drone trajectory found in current frame then simply put -1 instead: ###
                BoundingBox_PerFrame_list[trajectory_index].append(None)

        ### Skip Frame: ###
        current_frame_index += 1

    return BoundingBox_PerFrame_list

def Plot_BoundingBox_On_Frame(image_frame, BoundingBox_list):
    ### Loop over trajectories and for each trajectory see if there exists a valid trajectory in current frame and if so then draw circle: ###
    number_of_trajectories = len(BoundingBox_list)
    for trajectory_index in range(number_of_trajectories):
        ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
        target_spatial_coordinates = BoundingBox_list[trajectory_index]

        if target_spatial_coordinates is not None:
            circle_radius_in_pixels = 10
            # TODO: stop mixing (X,Y) and (H,W). make the whole script be (H,W)!!!!
            # image_frame = cv2.circle(image_frame, (target_spatial_coordinates[1],target_spatial_coordinates[0]), circle_radius_in_pixels, (0, 0, 255), 1)
            if type(target_spatial_coordinates) != tuple:
                image_frame = cv2.circle(image_frame, tuple(target_spatial_coordinates), circle_radius_in_pixels, (0, 0, 255), 1)
            else:
                image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (0, 0, 255), 1)
            image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (
                target_spatial_coordinates[0] - circle_radius_in_pixels,
                target_spatial_coordinates[1] - circle_radius_in_pixels), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                      cv2.LINE_AA)

    return image_frame


def Plot_BoundingBox_On_Movie(Mov, fps=1000, tit="tit", Res_dir='Res_dir', flag_save_movie=1, trajectory_tuple=[], BoundingBox_PerFrame_list=[]):
    ### Initialize Things: ###
    # flag_found_trajectories = len(trajectory_tuple) == 3
    flag_found_trajectories = trajectory_tuple[0] != []
    if flag_found_trajectories == False:
        return Mov, Mov

    t_vec = trajectory_tuple[0]
    number_of_trajectories = len(t_vec)
    if len(Mov.shape) == 3:
        number_of_frames_total, H, W = Mov.shape
    elif len(Mov.shape) == 4:
        number_of_frames_total, H, W, C = Mov.shape
    number_of_frames_to_view = number_of_frames_total
    iax0 = 2
    iax1 = 1
    if flag_save_movie:
        # OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)
        OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), cv2.VideoWriter_fourcc(*'MP42'), fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)
        # OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), 0, fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)

    ### Loop Over Video Frames & Paint Them On Screen: ###
    frames_list_with_BB = []
    for frame_index in np.arange(0, number_of_frames_total, 1):
        ### Get current movie frame: ###
        image_frame = np.copy(Mov[frame_index, :, :])

        ### Loop over trajectories and for each trajectory see if there exists a valid trajectory in current frame and if so then draw circle: ###
        for trajectory_index in range(number_of_trajectories):

            ### If There's Any Bounding Box Draw It: ###
            if BoundingBox_PerFrame_list != None:
                ### Get Spatial Coordinates Of Drone On Current Frame According To Found Trajectory: ###
                target_spatial_coordinates = BoundingBox_PerFrame_list[trajectory_index][frame_index]

                if target_spatial_coordinates is not None:
                    circle_radius_in_pixels = 10
                    #TODO: stop mixing (X,Y) and (H,W). make the whole script be (H,W)!!!!
                    # image_frame = cv2.circle(image_frame, (target_spatial_coordinates[1],target_spatial_coordinates[0]), circle_radius_in_pixels, (0, 0, 255), 1)
                    image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (255, 0, 0), 1)
                    image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (
                    target_spatial_coordinates[0] - circle_radius_in_pixels,
                    target_spatial_coordinates[1] - circle_radius_in_pixels), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                              cv2.LINE_AA)

        ### Add frame to frames list: ###
        frames_list_with_BB.append(image_frame)
        ### Actually Show Image With Circles On It: ###
        # cv2.imshow(tit, image_frame)
        # plt.imshow(image_frame)
        # plt.pause(0.001)

        ### Save Movie With Circles On It If Wanted: ###
        if flag_save_movie:
            OutVideoWriter.write((image_frame*255).clip(0,255).astype(np.uint8))

    ### Finish Movie Recording: ###
    if flag_save_movie:
        OutVideoWriter.release()

    return Mov, frames_list_with_BB


def Plot_BoundingBox_And_Polygon_On_Movie(Mov, fps=1000, tit="tit", Res_dir='Res_dir', flag_save_movie=1, trajectory_tuple=[],
                                          BoundingBox_PerFrame_list=[],
                                          polygon_points_list=[]):
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
        # OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), cv2.VideoWriter_fourcc(*'XVID'), fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)
        OutVideoWriter = cv2.VideoWriter(os.path.join(Res_dir, tit + ".avi"), 0, fps, (Mov.shape[iax0], Mov.shape[iax1]), 0)

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
                #TODO: stop mixing (X,Y) and (H,W). make the whole script be (H,W)!!!!
                # image_frame = cv2.circle(image_frame, (target_spatial_coordinates[1],target_spatial_coordinates[0]), circle_radius_in_pixels, (0, 0, 255), 1)
                image_frame = cv2.circle(image_frame, target_spatial_coordinates, circle_radius_in_pixels, (0, 0, 255), 1)
                image_frame = cv2.putText(image_frame, 'Trj' + str(trajectory_index), (
                                            target_spatial_coordinates[0] - circle_radius_in_pixels,
                                            target_spatial_coordinates[1] - circle_radius_in_pixels),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                          cv2.LINE_AA)

            ### Add Lines To Image: ###
            image_frame = draw_polygon_points_on_image(image_frame, polygon_points_list)

        ### Actually Show Image With Circles On It: ###
        # cv2.imshow(tit, image_frame)
        # plt.imshow(image_frame)
        # plt.pause(0.001)

        ### Save Movie With Circles On It If Wanted: ###
        if flag_save_movie:
            OutVideoWriter.write((image_frame*255).clip(0,255).astype(np.uint8))

    ### Finish Movie Recording: ###
    if flag_save_movie:
        OutVideoWriter.release()

def Plot_Bounding_Box_Demonstration(Movie, prestring='', results_folder='', trajectory_tuple=[]):
    BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(Movie,
                                                                    fps=50,
                                                                    tit=prestring + 'Movie_With_Drone_BB',
                                                                    Res_dir=results_folder,
                                                                    flag_save_movie=0,
                                                                    trajectory_tuple=trajectory_tuple)

    Movie_with_BB, frames_list = Plot_BoundingBox_On_Movie(Movie,
                              fps=50,
                              tit=prestring + "Movie_With_Drone_BB",
                              Res_dir=results_folder,
                              flag_save_movie=0,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)

    return Movie_with_BB, torch.Tensor(frames_list)




def Save_Movies_With_Trajectory_BoundingBoxes(Movie, Movie_BGS, Movie_BGS_std, prestring='', results_folder='', trajectory_tuple=[]):
    # TODO: make sure it's only saved to disk and not on arrays themsleves!!!!
    BoundingBox_PerFrame_list = Get_BoundingBox_List_For_Each_Frame(Movie,
                                                                    fps=50,
                                                                    tit=prestring + 'Movie_With_Drone_BB',
                                                                    Res_dir=results_folder,
                                                                    flag_save_movie=0,
                                                                    trajectory_tuple=trajectory_tuple)
    Plot_BoundingBox_On_Movie(Movie,
                              fps=50,
                              tit=prestring + "Movie_With_Drone_BB",
                              Res_dir=results_folder,
                              flag_save_movie=1,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)

    Plot_BoundingBox_On_Movie((Movie_BGS * 5),  # notice this is the "normalized" BGS video
                              fps=50,
                              tit=prestring + "Movie_BGS_With_Drone_BB",
                              Res_dir=results_folder,
                              flag_save_movie=1,
                              trajectory_tuple=trajectory_tuple,
                              BoundingBox_PerFrame_list=BoundingBox_PerFrame_list)



def Plot_3D_PointCloud_With_Trajectories(res_points, NonLinePoints, xyz_line, num_of_trj_found, params, save_name,
                                         Res_dir, auto_close_flg=True):
    ### Get variables from params dict: ###
    SaveResFlg = params['SaveResFlg']
    roi = params['roi']
    T = params['FPS'] * params['SeqT']

    ### Stake All The Found Lines On Top Of Each Other: ###
    trajectory_TXY_stacked_numpy = np.zeros((0, 3))
    c = np.zeros((0, 1))
    for ii in range(num_of_trj_found):
        trajectory_TXY_stacked_numpy = np.vstack((trajectory_TXY_stacked_numpy, xyz_line[ii]))  #xyz_line are the points which belong to a trajectory!!!!
        c = np.vstack((c, ii * np.ones((xyz_line[ii].shape[0], 1))))
    c = c.reshape((len(c),))

    ### Create 3D Plot: ###
    fig = plt.figure(210)
    ax = plt.axes(projection='3d')

    ### Add all the outlier points found: ###
    if res_points.shape[1] > 0:
        ax.scatter(res_points[:, 0], res_points[:, 1], res_points[:, 2], c='k',
                   marker='*', label='Outlier', s=1)

    ### Add the "NonLinePoints" which were found by RANSAC but were disqualified after heuristics: ###
    ax.scatter(NonLinePoints[:, 0], NonLinePoints[:, 1], NonLinePoints[:, 2], c='k',
               marker='x', label='NonLinePoints', s=10)

    ### Add the proper trajectory: ###
    t_indices = trajectory_TXY_stacked_numpy[:, 0]
    x_indices = trajectory_TXY_stacked_numpy[:, 1]
    y_indices = trajectory_TXY_stacked_numpy[:, 2]
    scatter = ax.scatter(t_indices, x_indices, y_indices, c=c, marker='o', s=20)

    ### Add Legends: ###
    if trajectory_TXY_stacked_numpy.shape[0] != 0:
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper left", title="Trj")
        ax.add_artist(legend1)

    # ### Maximize plot window etc': ###
    # ax.legend(loc='lower left')
    # ax.view_init(elev=20, azim=60)
    # plt.get_current_fig_manager().full_screen_toggle()
    # ax.set_xlim(0, T)
    # ax.set_zlim(0, roi[1])
    # ax.set_ylim(0, roi[0])
    # plt.pause(.000001)
    # plt.show()

    ### Save Figure If Wanted: ###
    if SaveResFlg:
        plt.savefig(os.path.join(Res_dir, save_name + ".png"))

    ### Close Plot: ###
    # if auto_close_flg:
    #     plt.close(fig)
    plt.pause(0.1)
    plt.close(fig)
    plt.close('all')
    plt.clf()
    plt.cla()


def Plot_3D_PointCloud_With_Trajectories_Demonstration(res_points, NonLinePoints, xyz_line, num_of_trj_found, params, save_name,
                                         Res_dir, auto_close_flg=True):
    ### Get variables from params dict: ###
    SaveResFlg = params['SaveResFlg']
    roi = params['roi']
    T = params['FPS'] * params['SeqT']

    ### Stake All The Found Lines On Top Of Each Other: ###
    trajectory_TXY_stacked_numpy = np.zeros((0, 3))
    c = np.zeros((0, 1))
    for ii in range(num_of_trj_found):
        trajectory_TXY_stacked_numpy = np.vstack((trajectory_TXY_stacked_numpy, xyz_line[ii]))  #xyz_line are the points which belong to a trajectory!!!!
        c = np.vstack((c, ii * np.ones((xyz_line[ii].shape[0], 1))))
    c = c.reshape((len(c),))

    ### Create 3D Plot: ###
    fig = plt.figure(210)
    ax = plt.axes(projection='3d')

    ### Add all the outlier points found: ###
    if res_points.shape[1] > 0:
        ax.scatter(res_points[:, 0], res_points[:, 1], res_points[:, 2], c='k',
                   marker='*', label='Outlier', s=1)

    ### Add the "NonLinePoints" which were found by RANSAC but were disqualified after heuristics: ###
    ax.scatter(NonLinePoints[:, 0], NonLinePoints[:, 1], NonLinePoints[:, 2], c='k',
               marker='x', label='NonLinePoints', s=10)

    ### Add the proper trajectory: ###
    t_indices = trajectory_TXY_stacked_numpy[:, 0]
    x_indices = trajectory_TXY_stacked_numpy[:, 1]
    y_indices = trajectory_TXY_stacked_numpy[:, 2]
    scatter = ax.scatter(t_indices, x_indices, y_indices, c=c, marker='o', s=20)

    ### Add Legends: ###
    if trajectory_TXY_stacked_numpy.shape[0] != 0:
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper left", title="Trj")
        ax.add_artist(legend1)

    plt.xlabel('Time[sec]')
    plt.ylabel('X[pixels]')
    ax.set_zlabel('Y[pixels]')




    # ### Maximize plot window etc': ###
    # ax.legend(loc='lower left')
    # ax.view_init(elev=20, azim=60)
    # plt.get_current_fig_manager().full_screen_toggle()
    # ax.set_xlim(0, T)
    # ax.set_zlim(0, roi[1])
    # ax.set_ylim(0, roi[0])
    # plt.pause(.000001)
    # plt.show()


def Plot_FFT_Bins_Detection_SubPlots(num_of_trj_found, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec_per_trajectory,
                                     params, Res_dir, TrjMov, DetectionDec, DetectionConfLvl):
    ### Save Zoon-Im Trajectory Movie: ###
    for trajectory_index in range(num_of_trj_found):
        video_name = os.path.join(params.results_folder_seq, 'Trajectory_ROI_' + str(trajectory_index) + '.avi')
        video_torch_array_to_video(torch.nn.Upsample(scale_factor=10)(scale_array_to_range(TrjMov[trajectory_index]).unsqueeze(1)), video_name, FPS=25)

    #TODO: return to showing FFT-Bins Plot after taking care of shit
    # ### FFT Bins Plot: ###
    # SaveResFlg = params['SaveResFlg']
    #
    # ### Loop over the number of trajectories found: ###
    # for trajectory_index in range(num_of_trj_found):
    #     fig = plt.figure(21 + trajectory_index)
    #     clim_min = 0
    #     clim_max = 1
    #     number_of_bins_sqrt = np.ceil(np.sqrt(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[0].shape[0]))
    #
    #     ### Loop over the different frequency bins and present them all in different subplots on the same big plot: ###
    #     for ii in range(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index].shape[0]):
    #         plt.subplot(number_of_bins_sqrt, number_of_bins_sqrt, 1 + ii)
    #         plt.imshow(TrjMovie_FFT_BinPartitioned_AfterScoreFunction[trajectory_index][ii, :, :].T, cmap='hot')
    #         plt.title("frq. = " + str(frequency_vec_per_trajectory[trajectory_index][ii]))
    #         plt.clim(clim_min, clim_max)
    #         plt.axis('off')
    #     plt.colorbar()
    #
    #     fig.text(0.5, 0.95, "Suspect signature in Fourier domain - Trj. no." + str(trajectory_index) + (
    #         " (quad. conf. " + str(int(DetectionConfLvl[trajectory_index])) + "%)" if DetectionDec[
    #             trajectory_index] else " (not a quad.)"), size=50, ha="center", va="center",
    #              bbox=dict(boxstyle="round", ec=(0.2, 1., 0.2), fc=(0.8, 1., 0.8), ))
    #     plt.pause(.01)
    #
    #     if SaveResFlg:
    #         plt.get_current_fig_manager().full_screen_toggle()
    #         plt.pause(.01)
    #         plt.savefig(os.path.join(Res_dir, "Detection_trj" + str(trajectory_index) + ".png"))
    #         video_name = os.path.join(params.results_folder_seq, 'Trajectory_ROI_' + str(trajectory_index) + '.avi')
    #         video_torch_array_to_video(torch.nn.Upsample(scale_factor=10)(scale_array_to_range(TrjMov[trajectory_index]).unsqueeze(1)), video_name, FPS=25)
    #
    #     plt.close(fig)






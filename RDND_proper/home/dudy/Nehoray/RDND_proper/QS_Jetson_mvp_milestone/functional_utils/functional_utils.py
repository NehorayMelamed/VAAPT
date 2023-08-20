import torch

from RapidBase.import_all import *

from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import *
from QS_Jetson_mvp_milestone.functional_utils.plotting_utils import *
from QS_Jetson_mvp_milestone.functional_utils.trajectory_utils import *
from QS_Jetson_mvp_milestone.functional_utils.Flashlight_utils import *
from QS_Jetson_mvp_milestone.functional_utils.FFT_analysis import *
from QS_Jetson_mvp_milestone.functional_utils.BG_and_noise_estimation import *
########################################################################################################################################################################

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


def change_BG_image_which_includes_drone():
    ##########################################################################################
    ### TODO: temp to allow for manual handling of static drones without a "clean" BG: ###
    Movie_BG, (q1, q2) = get_BG_over_entire_movie(f, params)
    frame_to_start_from = 500 * 10
    f = initialize_binary_file_reader(f)
    Movie_temp = read_frames_from_binary_file_stream(f, 500, frame_to_start_from * 1, params).astype(np.float)
    imshow_torch_video(scale_array_stretch_hist(torch.tensor(Movie_temp).unsqueeze(1)), FPS=50, frame_stride=5)

    ROI_size = 2
    drone_point = (279, 251)
    new_point = (drone_point[0], drone_point[1]+5)
    interim = copy.deepcopy(Movie_BG)
    original_slice = np.s_[drone_point[0]-ROI_size:drone_point[0]+ROI_size, drone_point[1]-ROI_size:drone_point[1]+ROI_size]
    new_slice = np.s_[new_point[0]-ROI_size:new_point[0]+ROI_size, new_point[1]-ROI_size:new_point[1]+ROI_size]
    interim[original_slice] = interim[new_slice]

    figure(); imshow(scale_array_stretch_hist(Movie_BG))
    figure(); imshow(scale_array_stretch_hist(interim))
    np.save(os.path.join(params.results_folder, 'Mov_BG.npy'), interim, allow_pickle=True)
    ##########################################################################################


def take_pelicanD_bin_files_and_put_them_into_folders():
    super_folder = r'E:\Quickshot\12.4.2022 - natznatz experiments'
    filenames_list = path_get_all_filenames_from_folder(super_folder, flag_recursive=False)
    for full_filename in filenames_list:
        folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(full_filename)
        new_folder = os.path.join(folder, filename_without_extension)
        path_create_path_if_none_exists(new_folder)
        new_path = os.path.join(new_folder, filename)
        shutil.move(full_filename, new_path)


def put_description_file_in_all_folders():
    super_folder_to_put_description_txt_file_into = r'E:\Quickshot\12.4.2022 - natznatz experiments'
    original_description_file_path = r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments\1_640_512_800fps_10000frames_1000_meter_flir/Description.txt'
    folder, original_filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(original_description_file_path)
    folder_list = path_get_folder_names(super_folder_to_put_description_txt_file_into)
    for folder_name in folder_list:
        new_path = os.path.join(folder_name, original_filename)
        shutil.copy(original_description_file_path, new_path)


def turn_JAI_files_into_Bin_and_avi_files():
    jai_filenames_list = []
    # jai_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/Palantir_6_28_22/FAR_videos/0_train_station_highway.raw')
    # jai_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/Palantir_6_28_22/FAR_videos/non_far.raw')
    jai_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/Palantir_6_28_22/FAR_videos/nonfar1_dynamic.raw')

    ### Get (H,W) For The Different Files: ###
    H = 352
    W = 8192

    ### Write Down Movie .avi File: ###
    for jai_experiment_index in np.arange(len(jai_filenames_list)):
        ### Get Current Filenames: ###
        jai_files_path = jai_filenames_list[jai_experiment_index]

        ### Initialize Movie: ###
        movie_filename = os.path.splitext(os.path.split(jai_files_path)[-1])[0] + '.avi'
        movie_filename = os.path.join(os.path.split(jai_files_path)[0], movie_filename)
        video_object = cv2.VideoWriter(movie_filename, 0, 50, (W, H))

        ### Initialize Binary Reader: ###
        scene_infile = open(jai_files_path, 'rb')

        ### Loop Over Frames while there are still frames left: ###
        flag_continue = True
        counter = 0
        while flag_continue:
            print(counter)
            scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=W * H * 1)
            if scene_image_array.nbytes == 0:
                flag_continue = False

            ### Enough frames so continue: ###
            if flag_continue:
                ### Get Image: ###
                image = Image.frombuffer("I", [W, H],
                                         scene_image_array.astype('I'),
                                         'raw', 'I', 0, 1)
                image = np.array(image).astype(np.uint8)
                input_tensor = BW2RGB(torch.tensor(image).float())

                # ### Center Crop It (so i can see something from all this strip: ###
                # input_tensor = crop_tensor(input_tensor, (H,H))

                ### Stretch the image so i can see clearly: ###
                if counter == 0:
                    input_tensor, (q1,q2) = scale_array_stretch_hist(input_tensor, quantiles=(0.05,0.95), flag_return_quantile=True)
                else:
                    input_tensor = scale_array_from_range(input_tensor.clamp(q1,q2), min_max_values_to_clip=(q1,q2))

                ### Write Frame To Video: ###
                input_tensor_to_video = torch_to_numpy_video_ready(input_tensor)
                video_object.write(input_tensor_to_video)

                ### Uptick counter: ###
                counter += 1
        # imshow_torch(input_tensor)



    # ### Read Binary File Images For Testing: ###
    # f = open(new_filename, 'rb')
    # H, W = 512, 640
    # T = 1000
    # scene_image_array = np.fromfile(f, dtype=np.uint16, count=W * H * T)
    # scene_image_array = np.reshape(scene_image_array, [T, H, W]).astype(np.float)
    # input_tensor = torch.tensor(scene_image_array).unsqueeze(1)
    # bla, (q1, q2) = scale_array_stretch_hist(input_tensor[0], (0.01, 0.999), flag_return_quantile=True)
    # input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
    # imshow_torch_video(input_tensor, FPS=50, frame_stride=15)

# turn_JAI_files_into_Bin_and_avi_files()


def turn_pelicanD_files_into_avi_files():
    pelicanD_filenames_list = get_filenames_from_folder_string_pattern(r'E:\Quickshot\22-06-2022 - SCD + FLIR\scd',
                                                                       flag_recursive=True,
                                                                       max_number_of_filenames=np.inf,
                                                                       flag_full_filename=True,
                                                                       string_pattern_to_search='*.Bin',
                                                                       flag_sort=False)

    ### Get (H,W) For The Different Files: ###
    H = 513
    H_real = 160
    W = 640

    ### Write Down Movie .avi File: ###
    for pelicanD_experiment_index in np.arange(0, len(pelicanD_filenames_list)):
        ### Get Current Filenames: ###
        pelicanD_files_path = pelicanD_filenames_list[pelicanD_experiment_index]

        ### Initialize Movie: ###
        movie_filename = os.path.splitext(os.path.split(pelicanD_files_path)[-1])[0] + '.avi'
        # movie_filename = os.path.join(os.path.split(pelicanD_files_path)[0], movie_filename)
        results_folder = os.path.join(os.path.split(pelicanD_files_path)[0], 'Results')
        path_create_path_if_none_exists(results_folder)
        movie_filename = os.path.join(results_folder, 'Original_Movie.avi')
        video_object = cv2.VideoWriter(movie_filename, 0, 50, (W, H_real))

        ### Initialize Binary Reader: ###
        scene_infile = open(pelicanD_files_path, 'rb')

        ### Loop Over Frames while there are still frames left: ###
        flag_continue = True
        counter = 0
        while flag_continue:
            print(counter)
            scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H * 1)
            if scene_image_array.nbytes < H * W * 2:  # uint16 = 2 bytes
                flag_continue = False

            ### Enough frames so continue: ###
            if flag_continue:
                ### Get Image: ###
                image = Image.frombuffer("I", [W, H],
                                         scene_image_array.astype('I'),
                                         'raw', 'I', 0, 1)
                image = np.array(image).astype(np.uint16).astype(np.float)
                input_tensor = BW2RGB(torch.tensor(image).float())

                # ### Center Crop It (so i can see something from all this strip: ###
                # input_tensor = crop_tensor(input_tensor, (H_real, W))
                input_tensor = input_tensor[:, 0:H_real]

                ### Stretch the image so i can see clearly: ###
                if counter == 0:
                    input_tensor, (q1, q2) = scale_array_stretch_hist(input_tensor, quantiles=(0.02, 0.99), flag_return_quantile=True)
                else:
                    input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), min_max_values_to_clip=(q1, q2))

                ### Write Frame To Video: ###
                input_tensor_to_video = torch_to_numpy_video_ready(input_tensor)
                video_object.write(input_tensor_to_video)

                ### Uptick counter: ###
                counter += 1
        # imshow_torch(input_tensor)


def turn_pelicanD_files_into_Bin_and_avi_files():
    pelicanD_filenames_list = []
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/Palantir_6_28_22/FAR_videos/0_train_station_highway.raw')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/Palantir_6_28_22/FAR_videos/non_far.raw')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 500fps 20000frames - 1000m static sky background - scd/640x160 500fps 20000frames - 1000m static sky background - scd.Bin')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 500fps 20000frames - 1000m left right rural background - scd/640x160 500fps 20000frames - 1000m left right rural background - scd.Bin')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 500fps 20000frames - 1000m static urban background - scd/640x160 500fps 20000frames - 1000m static urban background - scd.Bin')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 800fps 10000frames - 1000m no drone sky background - scd/640x160 800fps 10000frames - 1000m no drone sky background - scd.Bin')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 800fps 20000frames -1000m static sky background - scd/640x160 800fps 20000frames -1000m static sky background - scd.Bin')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 800fps 20000frames - 1000m left right rural background - scd/640x160 800fps 20000frames - 1000m left right rural background - scd.Bin')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 800fps 20000frames - 1000m static rural background - scd/640x160 800fps 20000frames - 1000m static rural background - scd.Bin')
    # pelicanD_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/640x160 800fps background test - scd/640x160 800fps background test - scd.Bin')

    # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\12.4.2022 - natznatz experiments')
    pelicanD_filenames_list = get_filenames_from_folder_string_pattern(r'E:\Quickshot\12.4.2022 - natznatz experiments',
                                                                       flag_recursive=True,
                                                                       max_number_of_filenames=np.inf,
                                                                       flag_full_filename=True,
                                                                       string_pattern_to_search='*.Bin',
                                                                       flag_sort=False)

    ### Get (H,W) For The Different Files: ###
    H = 320
    H_real = 320
    W = 640

    ### Write Down Movie .avi File: ###
    for pelicanD_experiment_index in np.arange(len(pelicanD_filenames_list)):
        ### Get Current Filenames: ###
        pelicanD_files_path = pelicanD_filenames_list[pelicanD_experiment_index]

        ### Initialize Movie: ###
        movie_filename = os.path.splitext(os.path.split(pelicanD_files_path)[-1])[0] + '.avi'
        movie_filename = os.path.join(os.path.split(pelicanD_files_path)[0], movie_filename)
        video_object = cv2.VideoWriter(movie_filename, 0, 50, (W, H_real))

        ### Initialize Binary Reader: ###
        scene_infile = open(pelicanD_files_path, 'rb')

        ### Initialize Binary Writter: ###
        new_filename = os.path.splitext(pelicanD_files_path)[0] + '_new.Bin'
        f = open(new_filename, 'a')

        ### Loop Over Frames while there are still frames left: ###
        flag_continue = True
        counter = 0
        while flag_continue:
            print(counter)
            scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H * 1)
            if scene_image_array.nbytes == 0:
                flag_continue = False

            ### Enough frames so continue: ###
            if flag_continue:
                ### Get Image: ###
                image = Image.frombuffer("I", [W, H],
                                         scene_image_array.astype('I'),
                                         'raw', 'I', 0, 1)
                image = np.array(image).astype(np.uint16).astype(np.float)
                input_tensor = BW2RGB(torch.tensor(image).float())

                # ### Center Crop It (so i can see something from all this strip: ###
                # input_tensor = crop_tensor(input_tensor, (H_real, W))
                input_tensor = input_tensor[:,0:H_real]

                ### Stretch the image so i can see clearly: ###
                if counter == 0:
                    input_tensor, (q1,q2) = scale_array_stretch_hist(input_tensor, quantiles=(0.02,0.99), flag_return_quantile=True)
                else:
                    input_tensor = scale_array_from_range(input_tensor.clamp(q1,q2), min_max_values_to_clip=(q1,q2))

                ### Write Frame To Video: ###
                input_tensor_to_video = torch_to_numpy_video_ready(input_tensor)
                video_object.write(input_tensor_to_video)

                ### Write To .Bin File: ###
                image.tofile(f)

                ### Uptick counter: ###
                counter += 1
        # imshow_torch(input_tensor)

# turn_pelicanD_files_into_Bin_and_avi_files()


def turn_flir_files_into_Bin_and_avi_files():
    flir_filenames_list = []
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/0_640_512_800fps_10000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/1_640_512_800fps_10000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/2_640_512_800fps_10000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/3_640_512_800fps_10000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/4_640_512_800fps_10000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/5_640_512_800fps_10000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/6_640_512_500fps_20000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/7_640_512_500fps_20000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/8_640_512_500fps_20000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/9_640_512_500fps_20000frames_1000_meter_flir')
    # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/10_640_512_500fps_20000frames_1000_meter_flir')

    # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\06_07_2022_FLIR\converted_bin')
    # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\22-06-2022 - SCD + FLIR\flir\6.22_exp')
    # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\22-06-2022 - SCD + FLIR\flir\6.22_exp\individual_images_bin_files')
    # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\29.06.22_exp_bin\individual_images_bin_files')
    # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\13.07.22_bin\individual_images_bin_files')
    flir_filenames_list = path_get_folder_names(r'E:\Quickshot\19_7_22_flir_exp_Bin\individual_images_bin_files')

    ### Get (H,W) For The Different Files: ###
    H, W = 512, 640

    # ### Change Filenames rjust: ###
    # for flir_experiment_index in np.arange(len(flir_filenames_list)):
    #     ### Get Current Filenames: ###
    #     flir_files_path = flir_filenames_list[flir_experiment_index]
    #     filenames_list = get_filenames_from_folder(flir_files_path)
    #     ### Loop Over Individual Images: ###
    #     for i in np.arange(len(filenames_list)):
    #         current_full_filename = filenames_list[i]
    #         current_folder = os.path.split(current_full_filename)[0]
    #         current_filename = os.path.split(current_full_filename)[1]
    #         current_number_string = os.path.splitext(current_filename)[0]
    #         new_number_string = string_rjust(np.int(current_number_string), 6)
    #         new_filename = new_number_string + '.Bin'
    #         new_full_filename = os.path.join(current_folder, new_filename)
    #         os.rename(current_full_filename, new_full_filename)

    ### Write Down Movie .avi File: ###
    for flir_experiment_index in np.arange(0, len(flir_filenames_list)):
        ### Get Current Filenames: ###
        flir_files_path = flir_filenames_list[flir_experiment_index]
        filenames_list = get_filenames_from_folder(flir_files_path)

        ### Initialize 1 Big .Bin File: ###
        new_folder = flir_files_path.replace('individual_images_bin_files', 'one_big_bin_file')
        path_create_path_if_none_exists(new_folder)
        folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(filenames_list[0])
        new_folder = folder.replace('individual_images_bin_files', 'one_big_bin_file')
        path_create_path_if_none_exists(new_folder)
        specific_folder_name = os.path.split(folder)[1]
        new_filename = os.path.join(new_folder, specific_folder_name + '.Bin')

        # ### Initialize 1 Big .Bin File ###
        # new_filename = os.path.split(flir_files_path)[-1] + '.Bin'
        # new_filename = os.path.join(os.path.split(flir_files_path)[0], new_filename)
        # ### TEMP: ###
        # new_filename = new_filename.replace('individual_images_bin_files', 'one_big_bin_file')
        # folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(new_filename)
        # new_folder = os.path.join(folder, filename_without_extension)
        # new_filename = os.path.join(new_folder, filename)
        # path_create_path_if_none_exists(new_folder)

        ### Initialize Binary File Writter: ###
        f = open(new_filename, 'a')

        ### Loop Over Movie Filenames/Frames: ###
        # movie_filename = os.path.split(flir_files_path)[-1] + '.avi'
        # movie_filename = os.path.join(os.path.split(flir_files_path)[0], movie_filename)
        # results_folder = os.path.join(new_folder, 'Results')
        # path_create_path_if_none_exists(results_folder)

        results_folder = os.path.join(new_folder, 'Results')
        path_create_path_if_none_exists(results_folder)
        movie_filename = os.path.join(results_folder, 'Original_Movie.avi')
        video_object = cv2.VideoWriter(movie_filename, 0, 10, (W, H))
        for i in np.arange(len(filenames_list)):
            print(i)
            ### Get Current Image: ###
            current_full_filename = filenames_list[i]
            scene_infile = open(current_full_filename, 'rb')
            scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
            # scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
            image = Image.frombuffer("I", [W, H],
                                     scene_image_array.astype('I'),
                                     'raw', 'I', 0, 1)
            image = np.array(image).astype(np.uint16)

            ### Stretch Image And Write To .avi Movie: ###
            input_tensor = torch.tensor(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            if i == 0:
                input_tensor, (q1, q2) = scale_array_stretch_hist(input_tensor, (0.01, 0.999), flag_return_quantile=True)
            input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
            input_tensor_to_video = torch_to_numpy_video_ready(input_tensor).squeeze(0)
            video_object.write(input_tensor_to_video)

            ### Write Image To .Bin File: ###
            image.tofile(f)

    # ### Read Binary File Images For Testing: ###
    # f = open(new_filename, 'rb')
    # H, W = 512, 640
    # T = 1000
    # scene_image_array = np.fromfile(f, dtype=np.uint16, count=W * H * T)
    # scene_image_array = np.reshape(scene_image_array, [T, H, W]).astype(np.float)
    # input_tensor = torch.tensor(scene_image_array).unsqueeze(1)
    # bla, (q1, q2) = scale_array_stretch_hist(input_tensor[0], (0.01, 0.999), flag_return_quantile=True)
    # input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
    # imshow_torch_video(input_tensor, FPS=50, frame_stride=15)

# turn_flir_files_into_Bin_and_avi_files()

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

def move_avi_files_into_proper_folders():
    super_folder = r'E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies'
    filenames_list = get_filenames_from_folder_string_pattern(super_folder, True, string_pattern_to_search='*.avi')
    for filename_index in np.arange(len(filenames_list)):
        current_filename = filenames_list[filename_index]
        folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(current_filename)
        results_folder = os.path.join(folder, 'Results')
        path_create_path_if_none_exists(results_folder)
        new_name = os.path.join(folder, 'Original_Movie.avi')
        os.rename(current_filename, new_name)
        new_final_name = os.path.join(results_folder, 'Original_Movie.avi')
        shutil.move(new_name, new_final_name)

def open_single_Bin_image():
    image_path = r'C:\Users\dudyk\Desktop\dudy_karl/0.Bin'
    H, W = 540, 8192
    scene_infile = open(image_path, 'rb')
    scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=W * H)
    # scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
    image = Image.frombuffer("I", [W, H],
                             scene_image_array.astype('I'),
                             'raw', 'I', 0, 1)
    image = np.array(image).astype(np.uint8)
    imshow(BW2RGB(image))


def turn_JAI_files_into_Bin_and_avi_files_2():
    JAI_filenames_list = []
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\3')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\4')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\5')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\6')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\7')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\8')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\9')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\10')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\11')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\12')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\13')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\14')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\15')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\16')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\17')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\18')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\19')
    JAI_filenames_list.append(r'E:\Palantir\Palantir - Experiment - 06.07.22 - Zichron Yaakov - M600 M300 MVC2\20')


    ### Get (H,W) For The Different Files: ###
    H, W = 424, 8192

    # ### Change Filenames rjust: ###
    # for JAI_experiment_index in np.arange(16, len(JAI_filenames_list)):
    #     ### Get Current Filenames: ###
    #     JAI_files_path = JAI_filenames_list[JAI_experiment_index]
    #     filenames_list = get_filenames_from_folder(JAI_files_path)
    #     ### Loop Over Individual Images: ###
    #     for i in np.arange(len(filenames_list)):
    #         current_full_filename = filenames_list[i]
    #         current_folder = os.path.split(current_full_filename)[0]
    #         current_filename = os.path.split(current_full_filename)[1]
    #         current_number_string = os.path.splitext(current_filename)[0]
    #         new_number_string = string_rjust(np.int(current_number_string), 6)
    #         new_filename = new_number_string + '.Bin'
    #         new_full_filename = os.path.join(current_folder, new_filename)
    #         os.rename(current_full_filename, new_full_filename)

    ### Write Down Movie .avi File: ###
    for JAI_experiment_index in np.arange(len(JAI_filenames_list)):
        print(JAI_experiment_index)
        ### Get Current Filenames: ###
        JAI_files_path = JAI_filenames_list[JAI_experiment_index]
        filenames_list = get_filenames_from_folder(JAI_files_path)

        ### Initialize 1 Big .Bin File ###
        new_filename = os.path.split(JAI_files_path)[-1] + '.Bin'
        new_filename = os.path.join(os.path.split(JAI_files_path)[0], new_filename)
        f = open(new_filename, 'a')

        ### Loop Over Movie Filenames/Frames: ###
        crop_start_W = 2200
        crop_stop_W = 4400
        crop_total_W = crop_stop_W - crop_start_W + 1
        movie_filename = os.path.join(JAI_files_path, 'Original_Movie.avi')
        cropped_movie_filename = os.path.join(JAI_files_path, 'Cropped_Movie.avi')
        video_object = cv2.VideoWriter(movie_filename, 0, 10, (W, H))
        cropped_video_object = cv2.VideoWriter(cropped_movie_filename, 0, 10, (crop_total_W, H))
        for i in np.arange(len(filenames_list)):
            print(i)
            ### Get Current Image: ###
            current_full_filename = filenames_list[i]
            scene_infile = open(current_full_filename, 'rb')
            scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=W * H)
            # scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
            image = Image.frombuffer("I", [W, H],
                                     scene_image_array.astype('I'),
                                     'raw', 'I', 0, 1)
            image = np.array(image).astype(np.uint8)

            ### Stretch Image And Write To .avi Movie: ###
            input_tensor = torch.tensor(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            if i == 0:
                input_tensor, (q1, q2) = scale_array_stretch_hist(input_tensor, (0.01, 0.999), flag_return_quantile=True)
            input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
            input_tensor_to_video = torch_to_numpy_video_ready(input_tensor).squeeze(0)
            video_object.write(input_tensor_to_video)

            ### Get Cropped Movie: ###
            input_tensor_to_video_cropped = input_tensor_to_video[:,crop_start_W:crop_stop_W+1,:]
            cropped_video_object.write(input_tensor_to_video_cropped)

            # ### Write Image To .Bin File: ###
            # image.tofile(f)

    # ### Read Binary File Images For Testing: ###
    # f = open(new_filename, 'rb')
    # H, W = 512, 640
    # T = 1000
    # scene_image_array = np.fromfile(f, dtype=np.uint16, count=W * H * T)
    # scene_image_array = np.reshape(scene_image_array, [T, H, W]).astype(np.float)
    # input_tensor = torch.tensor(scene_image_array).unsqueeze(1)
    # bla, (q1, q2) = scale_array_stretch_hist(input_tensor[0], (0.01, 0.999), flag_return_quantile=True)
    # input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
    # imshow_torch_video(input_tensor, FPS=50, frame_stride=15)

# turn_JAI_files_into_Bin_and_avi_files_2()


def turn_pt_files_into_Bin_files(params):
    ################################
    ### TODO: temp, turn .pt files into .Bin files: ###
    experiment_filenames_list = get_filenames_from_folder(params['ExperimentsFold'],
                                                          flag_recursive=True,
                                                          flag_full_filename=True,
                                                          string_pattern_to_search='*OriginalRawMovie.pt',
                                                          flag_sort=False)

    count = 0
    for filename in experiment_filenames_list:
        print(count)
        current_tensor = torch.load(filename)
        current_tensor = current_tensor * 256
        current_directory = os.path.split(filename)[0]
        current_filename = os.path.split(filename)[1]
        new_filename = str.replace(current_filename, '.pt', '.Bin')
        new_full_filename = os.path.join(current_directory, new_filename)
        current_tensor.type(torch.int16).cpu().numpy().tofile(new_full_filename)
        count += 1

        # ### Make Sure You Can Read It Right: ###
        # f = open(new_full_filename, "rb")
        # ### Read Frames: ###
        # H,W = current_tensor.shape[-2:]
        # roi = (H,W)
        # Mov = np.fromfile(f, dtype=np.uint16, count=100 * roi[0] * roi[1], offset=0 * roi[0] * roi[1] * 2)
        # Movie_len = np.int(len(Mov) / roi[0] / roi[1])
        # number_of_elements = Movie_len * roi[0] * roi[1]
        # Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])
    ################################

    ################################
    ### Convert Movie BG + Calculate Quantiles: ###
    experiment_filenames_list = get_filenames_from_folder(params['ExperimentsFold'],
                                                          flag_recursive=True,
                                                          flag_full_filename=True,
                                                          string_pattern_to_search='*bg.pt',
                                                          flag_sort=False)

    count = 0
    for filename in experiment_filenames_list:
        print(count)
        current_tensor = torch.load(filename)
        current_tensor = current_tensor * 256
        current_directory = os.path.split(filename)[0]
        current_filename = os.path.split(filename)[1]
        new_filename = str.replace(current_filename, 'bg.pt', 'Mov_BG.npy')
        new_full_filename = os.path.join(current_directory, new_filename)
        np.save(new_full_filename, current_tensor, allow_pickle=True)
        # Movie_BG = torch_get_4D(torch.Tensor(current_tensor), 'HW')
        png_full_filename = os.path.join(current_directory, 'Movie_BG.png')
        save_image_numpy(current_directory, 'Movie_BG.png', BW2RGB(current_tensor.cpu().numpy() * 200 / 256).astype(np.uint8), flag_scale=False)

        q1 = current_tensor.quantile(0.01).cpu().numpy()
        q2 = current_tensor.quantile(0.99).cpu().numpy()
        np.save(os.path.join(current_directory, 'quantiles.npy'), (q1, q2), allow_pickle=True)
        count += 1
    ################################

    ################################
    # TODO: quantiles are really weird...i'll recalculate them myself!
    ### Convert Movie Quantiles: ###
    experiment_filenames_list = get_filenames_from_folder(params['ExperimentsFold'],
                                                          flag_recursive=True,
                                                          flag_full_filename=True,
                                                          string_pattern_to_search='*quantiles.pt',
                                                          flag_sort=False)

    count = 0
    for filename in experiment_filenames_list:
        print(count)
        current_tensor = torch.load(filename)
        current_tensor = current_tensor * 1
        current_directory = os.path.split(filename)[0]
        current_filename = os.path.split(filename)[1]
        new_filename = str.replace(current_filename, 'quantiles.pt', 'quantiles.npy')
        new_full_filename = os.path.join(current_directory, new_filename)
        np.save(os.path.join(params.results_folder, 'quantiles.npy'), current_tensor, allow_pickle=True)
        count += 1
    ################################


########################################################################################################################################################################
### Read entire movie or movie parts to get some stats or something: ###
########################################################################################################################################################################

def Maor_Analysis(Movie, Movie_BG, params):
    ### BGS, RANSAC, Alignment, FFT Analysis (Maor method): ###
    ### Get BG, BGS and BGS_std: ###

    tic()
    Movie_BGS, real_space_noise_estimation = get_BGS_and_RealSpaceNoiseEstimation(Movie, Movie_BG)
    toc('BGS')

    ### Show Some Interim Results: ###
    # imshow_torch_video(torch.Tensor(Movie_BGS).unsqueeze(1), None, 50)
    # imshow_torch_video_running_mean(torch.Tensor(Movie_BGS).unsqueeze(1), None, 50, running_mean_size=5)
    # imshow(Movie_BGS_std); plt.show()
    # imshow(np.mean(Movie_BGS,0)); plt.show()

    ### Estimate Noise From BG substracted movie (noise estimate per pixel over time): ### #TODO: right now this estimates one noise dc and std for the entire image! return estimation per pixel
    tic()
    noise_std_FFT_over_drone_frequencies, noise_dc_FFT_over_drone_frequencies = Noise_Estimation_FourierSpace_PerPixel(Movie_BGS, params)
    toc('FFT noise estimation')

    ### use quantile filter to get pixels which are sufficiently above noise for more then a certain quantile of slots in time: ###
    # events_points_TXY_indices = Get_Outliers_Above_BG_1(Movie_BGS, Movie_BGS_std, params)
    tic()
    events_points_TXY_indices = Get_Outliers_Above_BG_2(Movie, Movie_BGS, real_space_noise_estimation, params)
    toc('outlier detection')

    ### Find trajectories using RANSAC and certain heuristics: ###
    tic()
    res_points, NonLinePoints, direction_vec, holding_point, t_vec_BeforeFFTDec, \
    trajectory_smoothed_polynom_X_BeforeFFTDec, trajectory_smoothed_polynom_Y_BeforeFFTDec, xyz_line, TrjMov, num_of_trj_found = \
        Find_Valid_Trajectories_RANSAC_And_Align_Drone(events_points_TXY_indices, Movie_BGS, params)
    toc('RANSAC')

    ### Test whether the trajectory frequency content agrees with the heuristics and is a drone candidate: ###
    tic()
    DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
        Frequency_Analysis_And_Detection(TrjMov, noise_std_FFT_over_drone_frequencies, noise_dc_FFT_over_drone_frequencies, params)  # TODO; why TrjMov instead of simply Movie
    toc('Frequency Analysis')
    # imshow_torch_video(torch.Tensor(TrjMov[0]).unsqueeze(1))

    ### Only Leave The Trajectories Which Passed The Frequency Analysis Stage Successfully: ####
    tic()
    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y = \
        get_only_trajectories_which_passed_frequency_decision(t_vec_BeforeFFTDec,
                                                              trajectory_smoothed_polynom_X_BeforeFFTDec,
                                                              trajectory_smoothed_polynom_Y_BeforeFFTDec,
                                                              DetectionDec)
    trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)
    trajectory_tuple_BeforeFFTDec = (t_vec_BeforeFFTDec, trajectory_smoothed_polynom_X_BeforeFFTDec, trajectory_smoothed_polynom_Y_BeforeFFTDec)
    toc('get only valid trajectories')

    return Movie_BGS, real_space_noise_estimation, trajectory_tuple, trajectory_tuple_BeforeFFTDec, t_vec, res_points, NonLinePoints, xyz_line, num_of_trj_found, \
           TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, TrjMov, DetectionDec, DetectionConfLvl, params


def Maor_Analysis_torch(Movie, Movie_BG, params):
    ### BGS, RANSAC, Alignment, FFT Analysis (Maor method): ###

    ### To GPUs: ###
    Movie = torch.Tensor(Movie).cuda().unsqueeze(1)
    Movie_BG = Movie_BG.cuda()
    params.H = Movie.shape[-2]
    params.W = Movie.shape[-1]
    params.T = Movie.shape[0]

    ### Get BG, BGS and BGS_std: ###
    tic()
    Movie_BGS, real_space_noise_estimation = get_BGS_and_RealSpaceNoiseEstimation_torch(Movie, Movie_BG)
    toc('           BGS')

    ### Show Some Interim Results: ###
    # imshow_torch_video(torch.Tensor(Movie_BGS).unsqueeze(1), None, 50)
    # imshow_torch_video_running_mean(torch.Tensor(Movie_BGS).unsqueeze(1), None, 50, running_mean_size=5)
    # imshow(Movie_BGS_std); plt.show()
    # imshow(np.mean(Movie_BGS,0)); plt.show()

    # ### Estimate Noise From BG substracted movie (noise estimate per pixel over time): ### #TODO: right now this estimates one noise dc and std for the entire image! return estimation per pixel
    # tic()
    # #TODO: this should be gotten rid of! if i accept a ~500 frames of BG then i should calculate this once and that's it!
    # noise_std_FFT_over_drone_frequencies, noise_dc_FFT_over_drone_frequencies = Noise_Estimation_FourierSpace_PerPixel_Torch(Movie_BGS, params)
    # toc('           FFT noise estimation')

    ### use quantile filter to get pixels which are sufficiently above noise for more then a certain quantile of slots in time: ###
    tic()
    #TODO: i'm not using real_space_noise_estimation (and for good reason for now, because i'm only getting a single BG image...to be changed)
    events_points_TXY_indices = Get_Outliers_Above_BG_2_torch(Movie, Movie_BGS, Movie_BG, params)
    toc('           outlier detection')

    ### Find trajectories using RANSAC and certain heuristics: ###
    tic()
    res_points, NonLinePoints, direction_vec, holding_point, t_vec_BeforeFFTDec, \
    trajectory_smoothed_polynom_X_BeforeFFTDec, trajectory_smoothed_polynom_Y_BeforeFFTDec, xyz_line, TrjMov, num_of_trj_found_RASNAC = \
        Find_Valid_Trajectories_RANSAC_And_Align_Drone_Torch(events_points_TXY_indices, Movie, Movie_BGS, params)
    toc('           RANSAC')

    ### Test whether the trajectory frequency content agrees with the heuristics and is a drone candidate: ###
    tic()
    # DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
    #     Frequency_Analysis_And_Detection_Torch(TrjMov, noise_std_FFT_over_drone_frequencies, noise_dc_FFT_over_drone_frequencies, params)  # TODO; why TrjMov instead of simply Movie

    DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
        FFT_Analysis_Dudy_PredeterminedRanges(TrjMov, params)

    toc('           Frequency Analysis')
    # imshow_torch_video(torch.Tensor(TrjMov[0]).unsqueeze(1))

    ### Only Leave The Trajectories Which Passed The Frequency Analysis Stage Successfully: ####
    tic()
    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y = \
        get_only_trajectories_which_passed_frequency_decision(t_vec_BeforeFFTDec,
                                                              trajectory_smoothed_polynom_X_BeforeFFTDec,
                                                              trajectory_smoothed_polynom_Y_BeforeFFTDec,
                                                              DetectionDec)
    trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)
    trajectory_tuple_BeforeFFTDec = (t_vec_BeforeFFTDec, trajectory_smoothed_polynom_X_BeforeFFTDec, trajectory_smoothed_polynom_Y_BeforeFFTDec)
    toc('           get only valid trajectories')

    ### Get number of valid trajectories after FFT analysis: ###
    num_of_valid_trj_found = len(t_vec)

    return Movie_BGS, real_space_noise_estimation, trajectory_tuple, trajectory_tuple_BeforeFFTDec, t_vec, res_points, NonLinePoints, xyz_line, num_of_valid_trj_found, \
           TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, TrjMov, DetectionDec, DetectionConfLvl, params


def get_outlier_movie_from_event_points(Movie, events_points_TXY_indices):
    if type(events_points_TXY_indices) == list:
        events_points_TXY_indices_final = torch.cat(events_points_TXY_indices, 0)
    else:
        events_points_TXY_indices_final = events_points_TXY_indices

    # (1). transfer event points to outlier video:
    final_outlier_movie = torch.zeros_like(Movie)
    final_outlier_movie = final_outlier_movie[:, 0]
    T, H, W = final_outlier_movie.shape
    proper_form_indices = torch.cat([events_points_TXY_indices_final[:, 0:1], events_points_TXY_indices_final[:, 2:3], events_points_TXY_indices_final[:, 1:2]], -1).long()
    final_outlier_movie = indices_to_logical_mask_torch(input_logical_mask=None, input_tensor_reference=final_outlier_movie, indices=proper_form_indices, flag_one_based=False)

    # (2). present/save movie:
    # imshow_torch_video(final_outlier_movie.unsqueeze(1), FPS=10, frame_stride=3)
    # imshow_torch_video(Movie, FPS=10, frame_stride=3)
    # video_torch_array_to_video(final_outlier_movie.unsqueeze(1), video_name=os.path.join(params.results_folder_seq, 'final_outliers.avi'), FPS=50)
    # video_torch_array_to_video(Movie/255, video_name=os.path.join(params.results_folder_seq, 'Movie.avi'), FPS=50)

    return final_outlier_movie


def Dudy_Analysis_torch(Movie, Movie_BG, params):
    ### BGS, RANSAC, Alignment, FFT Analysis (Maor method): ###

    ### Get Parameters: ###
    params.H = Movie.shape[-2]
    params.W = Movie.shape[-1]
    params.T = Movie.shape[0]


    ### Get BG, BGS and BGS_std: ###
    tic()
    Movie_BGS = Movie - Movie_BG
    Movie_BGS_STD = params.Movie_BG_std_torch_previous  #(*). initially updated by the initial BG estimation stage and later on to be updated in the below functions
    toc('           BGS')


    ### Get Outliers Above BG: ###
    tic()
    # events_points_TXY_indices, params = Get_Outliers_Above_BG_2_torch(Movie, Movie_BGS, Movie_BG, params)
    events_points_TXY_indices, params = Get_Outliers_Above_BG_Multiple_BG_Estimations_torch_2(Movie, Movie_BGS, Movie_BG, params)
    toc('           outlier detection')

    ### Present Final Indices As Outlier Movie: ###
    final_outlier_movie = get_outlier_movie_from_event_points(Movie, events_points_TXY_indices)
    video_torch_array_to_video(final_outlier_movie.unsqueeze(1), video_name=os.path.join(params.results_folder_seq, 'final_outliers.avi'), FPS=50)
    video_torch_array_to_video(Movie / 255, video_name=os.path.join(params.results_folder_seq, 'Movie.avi'), FPS=50)

    ### Loop over sub-sequences, Fit straight line using RANSAC for each sub-sequence, then fuse them together: ###
    trajectory_smoothed_polynom_X_BeforeFFTDec_list = []
    trajectory_smoothed_polynom_Y_BeforeFFTDec_list = []
    t_vec_BeforeFFTDec_list = []
    number_of_trajectories_per_sub_sequence = []
    xyz_line_list = []
    number_of_sub_sequences = 5
    T, C, H, W = Movie.shape
    number_of_frames_per_sub_sequence = T // number_of_sub_sequences
    for sub_sequence_index in np.arange(len(events_points_TXY_indices)):
        ### Get Current Sub-Sequence Events: ###
        current_sub_sequence_events_points_TXY_indices = events_points_TXY_indices[sub_sequence_index]
        current_sub_sequence_events_points_TXY_indices[:, 0] -= number_of_frames_per_sub_sequence * sub_sequence_index

        ### Use RANSAC For Straight Line Fit ###
        movie_start_index = sub_sequence_index * number_of_frames_per_sub_sequence
        movie_stop_index = (sub_sequence_index+1) * number_of_frames_per_sub_sequence
        res_points, NonLinePoints, direction_vec, holding_point, t_vec_BeforeFFTDec, \
        trajectory_smoothed_polynom_X_BeforeFFTDec, trajectory_smoothed_polynom_Y_BeforeFFTDec, xyz_line, TrjMov, num_of_trj_found_RANSAC = \
            Find_Valid_Trajectories_OnlyRANSAC_And_Align_Drone_Torch(current_sub_sequence_events_points_TXY_indices,
                                                                     Movie[movie_start_index:movie_stop_index],
                                                                     Movie_BGS[movie_start_index:movie_stop_index],
                                                                     params)
        print(num_of_trj_found_RANSAC)

        ### Loop Over Found Trajectories And Keep Them: ###
        trajectory_smoothed_polynom_X_BeforeFFTDec_list.append(trajectory_smoothed_polynom_X_BeforeFFTDec)
        trajectory_smoothed_polynom_Y_BeforeFFTDec_list.append(trajectory_smoothed_polynom_Y_BeforeFFTDec)
        t_vec_BeforeFFTDec_list.append(t_vec_BeforeFFTDec)
        xyz_line_list.append(xyz_line)
        number_of_trajectories_per_sub_sequence.append(num_of_trj_found_RANSAC)

    ### Loop Over Sub-Sequences And Link Trajectories Together: ###
    ### Form Current Trajectories: ###
    previous_trajectories_X = 1.0 * trajectory_smoothed_polynom_X_BeforeFFTDec_list[0]
    previous_trajectories_Y = 1.0 * trajectory_smoothed_polynom_Y_BeforeFFTDec_list[0]
    previous_trajectories_T = 1.0 * t_vec_BeforeFFTDec_list[0]
    linked_trajectories_list = []
    for sub_sequence_index in np.arange(1, len(events_points_TXY_indices)):
        ### Loop Over Currently Active Trajectories To Be Fitted: ###
        for previous_trajectory_index in np.arange(len(previous_trajectories_X)):
            previous_trajectory_X_end_point = previous_trajectories_X[previous_trajectory_index][-1]
            previous_trajectory_Y_end_point = previous_trajectories_Y[previous_trajectory_index][-1]
            previous_trajectory_T_end_point = previous_trajectories_T[previous_trajectory_index][-1]

            ### Loop Over New Trajectories: ###
            min_distance = np.inf
            min_distance_index = np.inf
            for new_trajectory_index in np.arange(0, number_of_trajectories_per_sub_sequence[sub_sequence_index]):
                current_trajectories_X = trajectory_smoothed_polynom_X_BeforeFFTDec_list[sub_sequence_index]
                current_trajectories_Y = trajectory_smoothed_polynom_Y_BeforeFFTDec_list[sub_sequence_index]
                current_trajectories_T = t_vec_BeforeFFTDec_list[sub_sequence_index]

                new_trajectory_X_start_point = current_trajectories_X[new_trajectory_index][0]
                new_trajectory_Y_start_point = current_trajectories_Y[new_trajectory_index][0]
                trajectory_end_points_distance = ((new_trajectory_X_start_point - previous_trajectory_X_end_point).abs()**2 + (new_trajectory_Y_start_point - previous_trajectory_Y_end_point).abs()**2)**0.5

                if trajectory_end_points_distance < min_distance:
                    min_distance = trajectory_end_points_distance
                    min_distance_index = new_trajectory_index

            ### If Distance Is Small Enough --> Link Trajectories To One Longer Trajectory: ###
            min_end_points_distance_to_link = 5
            if min_distance <= min_end_points_distance_to_link:
                previous_trajectories_X = torch.cat([previous_trajectories_X, current_trajectories_X])
                previous_trajectories_Y = torch.cat([previous_trajectories_Y, current_trajectories_Y])
                previous_trajectories_T = torch.cat([previous_trajectories_T, current_trajectories_T])


    ### Find trajectories using RANSAC and certain heuristics: ###
    tic()
    res_points, NonLinePoints, direction_vec, holding_point, t_vec_BeforeFFTDec, \
    trajectory_smoothed_polynom_X_BeforeFFTDec, trajectory_smoothed_polynom_Y_BeforeFFTDec, xyz_line, TrjMov, num_of_trj_found_RANSAC = \
        Find_Valid_Trajectories_OnlyRANSAC_And_Align_Drone_Torch(events_points_TXY_indices[0], Movie[0:100], Movie_BGS[0:100], params)
    toc('           RANSAC')

    ### Present Trajectories Found On Outlier Movie: ###
    #(1). get smooth trajectory to list:
    xyz_line_numpy_list = []
    for trajectory_index in np.arange(num_of_trj_found_RANSAC):
        current_smooth_trajectory_X = trajectory_smoothed_polynom_X_BeforeFFTDec[trajectory_index]
        current_smooth_trajectory_Y = trajectory_smoothed_polynom_Y_BeforeFFTDec[trajectory_index]
        current_smooth_t_vec = t_vec_BeforeFFTDec[trajectory_index]
        current_trajectory = torch.cat((current_smooth_t_vec.unsqueeze(-1), current_smooth_trajectory_X.unsqueeze(-1), current_smooth_trajectory_Y.unsqueeze(-1)), -1)
        current_trajectory_numpy = current_trajectory.cpu().numpy()
        xyz_line_numpy_list.append(current_trajectory_numpy)
    #(2). present smooth trajectory on movie as circles:
    final_outlier_movie_numpy = (BW2RGB(final_outlier_movie.unsqueeze(1)).permute([0,2,3,1]).float()*255).cpu().numpy()
    final_outlier_movie_numpy = draw_trajectories_as_circle_on_images(final_outlier_movie_numpy,
                                                xyz_line_numpy_list,
                                                circle_radius_in_pixels=5,
                                                line_thickness=3)
    final_outlier_movie_torch = torch.tensor(final_outlier_movie_numpy).permute([0,3,1,2])
    # imshow_torch_video(final_outlier_movie_torch/255, FPS=10, frame_stride=5)
    video_torch_array_to_video(final_outlier_movie_torch/255, video_name=os.path.join(params.results_folder_seq, 'final_outliers_with_BB.avi'), FPS=50)


    ### Test whether the trajectory frequency content agrees with the heuristics and is a drone candidate: ###
    tic()
    # DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
    #     Frequency_Analysis_And_Detection_Torch(TrjMov, noise_std_FFT_over_drone_frequencies, noise_dc_FFT_over_drone_frequencies, params)  # TODO; why TrjMov instead of simply Movie
    DetectionDec, TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, pxl_scr, DetectionConfLvl = \
        FFT_Analysis_Dudy_PredeterminedRanges(TrjMov, params)
    toc('           Frequency Analysis')


    ### Only Leave The Trajectories Which Passed The Frequency Analysis Stage Successfully: ####
    tic()
    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y, xyz_line_after_FFT = \
        get_only_trajectories_which_passed_frequency_decision(t_vec_BeforeFFTDec,
                                                              trajectory_smoothed_polynom_X_BeforeFFTDec,
                                                              trajectory_smoothed_polynom_Y_BeforeFFTDec,
                                                              xyz_line,
                                                              DetectionDec)
    trajectory_tuple = (t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y)
    trajectory_tuple_BeforeFFTDec = (t_vec_BeforeFFTDec, trajectory_smoothed_polynom_X_BeforeFFTDec, trajectory_smoothed_polynom_Y_BeforeFFTDec)
    toc('           get only valid trajectories')

    ### Update BG Estimation For Next Run (running BG estimation): ###
    # params = Update_BG_Long_Sequence(Movie, params)
    # params = Update_BG(Movie, Movie_BG, xyz_line, trajectory_tuple, params)

    ### Get number of valid trajectories after FFT analysis: ###
    number_of_RANSAC_valid_trajectories = len(t_vec_BeforeFFTDec)
    number_of_FFT_valid_trajectories = len(t_vec)

    return Movie_BGS, Movie_BGS_STD, trajectory_tuple, trajectory_tuple_BeforeFFTDec, t_vec, res_points, NonLinePoints, xyz_line, xyz_line_after_FFT,  \
           number_of_RANSAC_valid_trajectories, number_of_FFT_valid_trajectories, \
           TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, TrjMov, DetectionDec, DetectionConfLvl, params




def QS_flow_run(params):
    ### Get a list of the files in the current directory: ###
    main_folder = params['DistFold']
    list_of_files_in_path = os.listdir(main_folder)

    ### Get QS_stats DataFrame For All Experiments: ###
    if os.path.isfile(os.path.join(params.ExperimentsFold, 'QS_stats_pd.npy')):
        QS_stats_pd = np.load(os.path.join(params.ExperimentsFold, 'QS_stats_pd.npy'), allow_pickle=True)
    else:
        QS_stats_pd = pd.DataFrame()
        np.save(os.path.join(params.ExperimentsFold, 'QS_stats_pd.npy'), QS_stats_pd, allow_pickle=True)

    ### get files to run over list (which can be made up of several experiments within a folder or a single FileName): ###
    if params['RunOverAllFilesFlg']:
        experiment_filenames_list = os.listdir(params['ExperimentsFold'])  # all the filenames must be in the form of a .bin file currently
    else:
        experiment_filenames_list = [params['FileName']]

    ### Run over the different experiments and analyze: ###
    for experiment_index in range(len(experiment_filenames_list)):
        ### Get Auxiliary Parameters: ###
        number_of_frames_per_sequence = np.int(params['FPS'] * params['SeqT'])

        ### Initialize Experiment Params: ###
        flag_bin_file, current_experiment_filename, results_folder, experiment_folder,\
            filename_itself, params, results_summary_string = initialize_experiment(experiment_filenames_list, experiment_index, params)
        if flag_bin_file == False:
            continue

        ### Get binary file reader: ###
        f = get_binary_file_reader(params)

        ### Start from a certain frame: ###
        frame_to_start_from = 0
        Movie_temp = read_frames_from_binary_file_stream(f, 1, frame_to_start_from * 1, params)

        #######################################################################################################
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

        # ### TODO: Temp for stationary drone signal testing, delete later: ###
        # f = initialize_binary_file_reader(f)
        # Movie = read_frames_from_binary_file_stream(f, 1500, 2500*1, params)
        # Movie = Movie.astype(np.float)
        # # Movie = scale_array_from_range(Movie.clip(q1, q2),
        # #                                        min_max_values_to_clip=(q1, q2),
        # #                                        min_max_values_to_scale_to=(0, 1))
        # # Movie = Movie - Movie.mean(0)
        # # imshow_torch_video(torch.Tensor(Movie[0:500]), 2500, 50, frame_stride=4)
        # Total_FFT_Analysis(Movie, Movie_BG.cuda(), params)

        #### Test video alignment: ###
        # Movie_aligned = Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(Movie[0:200,0])
        # Movie_aligned = Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation(Movie[0:200,0])
        # imshow_torch_video(Movie[0:500], 2500, 50, frame_stride=4)
        # imshow_torch_video(Movie_aligned[0:500], 2500, 50, frame_stride=4)
        # imshow_torch_video(Movie_aligned[0:500] - Movie_BG , 2500, 50, frame_stride=4)
        # imshow_torch_video(Movie[0:500] - Movie_BG , 2500, 50, frame_stride=4)

        # ## Test Align Movie After RCC: ###
        # #(*). Clone Tensor to keep the original for later:
        # Movie = Movie
        # Movie_before_RCC = torch.clone(Movie)
        # #(1). RCC per frame:
        # Movie, residual_tensor = perform_RCC_on_tensor(Movie_before_RCC, lambda_weight=50, number_of_iterations=1)
        # imshow_torch_video(Movie_before_RCC, 2500, 50, frame_stride=4)
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
        #######################################################################################################


        #######################################################################################################
        ### Estimate BG Over Entire Movie (because we can't know exactly at which frames there is a flashlight): ###
        if os.path.isfile(os.path.join(params.results_folder, 'Mov_BG.npy')) == False:
            Movie_BG, (q1,q2) = get_BG_over_entire_movie(f, params)
        else:
            Movie_BG = np.load(os.path.join(params.results_folder, 'Mov_BG.npy'), allow_pickle=True)
            (q1,q2) = np.load(os.path.join(params.results_folder, 'quantiles.npy'), allow_pickle=True)
        Movie_BG = torch_get_4D(torch.Tensor(Movie_BG), 'HW')
        (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(Movie_BG)

        ### Save entire movie of entire experiment: ###
        if os.path.isfile(os.path.join(results_folder, 'Original_Movie.avi')) == False and params.flag_save_interim_graphs_and_movies:
            long_movie_name = os.path.join(results_folder, 'Original_Movie.avi')
            f = initialize_binary_file_reader(f)
            read_long_movie_and_save_to_avi(f, long_movie_name, 25, params)
        #######################################################################################################

        #######################################################################################################
        ### Find Flashlight In Movie & Get Sequence Occurense + Flashlight Polygon Points: ###
        if os.path.isfile(os.path.join(params.results_folder, 'flag_flashlight_found_list.npy')) == False:
            flag_flashlight_found_list, polygon_points_list, flashlight_BB_list = Find_Thermal_Flashlight_In_Movie(f, Movie_BG, params)
        else:
            flag_flashlight_found_list = np.load(os.path.join(params.results_folder, 'flag_flashlight_found_list.npy'), allow_pickle=True)
            polygon_points_list = np.load(os.path.join(params.results_folder, 'flashlight_polygon_points_list.npy'), allow_pickle=True)
            flashlight_BB_list = np.load(os.path.join(params.results_folder, 'flashlight_BB_list.npy'), allow_pickle=True)
        #######################################################################################################

        #######################################################################################################
        ### Initialize video with information (flashlight, drone, etc') on it of the entire experiment: ###
        if params.flag_save_interim_graphs_and_movies:
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            informative_movie_full_file_name = os.path.join(params.results_folder, 'Full_Informative_Movie.avi')
            informative_movie_video_writer = cv2.VideoWriter(informative_movie_full_file_name, fourcc, 50, (W, H))
        #######################################################################################################

        #######################################################################################################
        ### Initialize current flashlight polygon points with the first polygon found (in case the movie doesn't start with it): ###
        current_polygon_points = polygon_points_list[0]

        ### Loop over the different sequences and analyze: ###
        f = initialize_binary_file_reader(f)
        flag_enough_frames_to_analyze = True
        sequence_index = -1
        flag_drone_trajectory_inside_BB_list = []
        while flag_enough_frames_to_analyze:
            sequence_index += 1

            ### Make results path for current sequence: ###
            results_folder_seq = os.path.join(results_folder, "seq" + str(sequence_index))
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
                                                       min_max_values_to_scale_to=(0,1))
                if params.flag_save_interim_graphs_and_movies:
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
                    current_polygon_points = polygon_points_list[sequence_index]
                    current_flashlight_BB = flashlight_BB_list[sequence_index]

                    ### Plot Polygon and Flashlight On Large Movie: ###
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
                    #(*). if there is NO flashlight, search for a drone:

                    # ##################################################################################
                    # ### Brute Force FFT (dudy method): ###
                    # Brute_Force_FFT(Movie, params)
                    # ##################################################################################

                    ##################################################################################
                    ### Maor Outlier Trajectory Finding Method: ###
                    Movie_BGS, Movie_BGS_std, trajectory_tuple, trajectory_tuple_BeforeFFTDec, t_vec, res_points, NonLinePoints, xyz_line, num_of_trj_found, \
                    TrjMovie_FFT_BinPartitioned_AfterScoreFunction, frequency_vec, TrjMov, \
                    DetectionDec, DetectionConfLvl, params = Maor_Analysis(Movie, Movie_BG, params)
                    ### Unpack trajectory_tuple: ###
                    t_vec, trajectory_smoothed_polynom_X, trajectory_smoothed_polynom_Y = trajectory_tuple
                    number_of_drone_trajectories_found = len(t_vec)
                    ##################################################################################


                    ##################################################################################
                    ### Plot Results:
                    #(1). Plot 3D CLoud and Outline Trajectories (of ALL trajectories found by RANSAC):
                    Plot_3D_PointCloud_With_Trajectories(res_points, NonLinePoints, xyz_line, num_of_trj_found, params,
                                                         "Plt3DPntCloudTrj", results_folder_seq)
                    #(2). Save Movie With ALL Trajectories Found From RANSAC (before going through frequency analysis):
                    Save_Movies_With_Trajectory_BoundingBoxes(Movie, Movie_BGS, Movie_BGS_std,
                                                              prestring='All_Trajectories_RANSAC_Found',
                                                              results_folder=results_folder_seq,
                                                              trajectory_tuple=trajectory_tuple_BeforeFFTDec)
                    #(3). Save Movie With ALL Valid Trajectories:
                    Save_Movies_With_Trajectory_BoundingBoxes(Movie, Movie_BGS, Movie_BGS_std,
                                                              prestring='Only_Valid_Trajectories',
                                                              results_folder=results_folder_seq,
                                                              trajectory_tuple=trajectory_tuple)
                    #(3). Plot The Detection Over different frequency bands:
                    Plot_FFT_Bins_Detection_SubPlots(num_of_trj_found, TrjMovie_FFT_BinPartitioned_AfterScoreFunction,
                                                     frequency_vec, params, results_folder_seq, TrjMov, DetectionDec, DetectionConfLvl)
                    ##################################################################################


                    ##################################################################################

                    ### Automatic Location Labeling - was the drone that was found inside the "GT" label, as defined by the flashlight polygon:
                    #(*). Check if trajectory found is within flashlight polygon:
                    flag_drone_trajectory_inside_BB_list, current_frames_for_video_writer = \
                        check_for_each_trajectory_if_inside_flashlight_polygon(Movie, current_frames_for_video_writer,
                                                                               trajectory_tuple, current_polygon_points)

                    ### Plot Trajectories On Images: ###
                    current_frames_for_video_writer = draw_trajectories_on_images(current_frames_for_video_writer, trajectory_tuple)

                    ### Loop over all frames intended for video writer (with all the information drawn on them) and write the video file: ###
                    full_t_vec = np.arange(Movie.shape[0])
                    for inter_frame_index in full_t_vec:
                        current_movie_frame = current_frames_for_video_writer[inter_frame_index]
                        current_movie_string = 'Sequence ' + str(sequence_index) + ',    '
                        current_movie_string += 'Flashlight OFF        '
                        current_movie_string += 'Preseting Drone Trajectories'
                        current_movie_frame = draw_text_on_image(current_movie_frame, text_for_image=current_movie_string, fontScale=0.3, thickness=1)
                        current_movie_frame = draw_polygon_points_on_image(current_movie_frame, current_polygon_points)
                        informative_movie_video_writer.write(current_movie_frame)
                    #####################################################################################


                    ##################################################################################
                    ### Loop over all found trajectories and see if there's anyone who's inside the flashlight polygon: ###
                    if number_of_drone_trajectories_found > 0:
                        for trajectory_index in np.arange(number_of_drone_trajectories_found):
                            flag_current_trajectory_inside_flashlight_BB = flag_drone_trajectory_inside_BB_list[trajectory_index]
                            current_drone_detection_confidence = DetectionConfLvl[trajectory_index]

                            ### Write results down to txt file: ###
                            results_summary_string += 'Sequence ' + str(sequence_index) + ': \n'
                            results_summary_string += '             Trajectory ' + str(trajectory_index) + \
                                                      ', Confidence: ' + str(current_drone_detection_confidence.cpu().item())
                            results_summary_string += '             Inside Flashlight BB: ' + str(flag_current_trajectory_inside_flashlight_BB)
                            results_summary_string += '\n'
                    else:
                        results_summary_string += 'Sequence ' + str(sequence_index) + ': \n'
                        results_summary_string += 'No Drones Found!'
                        results_summary_string += '\n'
                    open(os.path.join(results_folder, 'res_summ.txt'), 'a').write(results_summary_string)
                    ##################################################################################

        ### Gather and Present Statistics For Current Experiment: ###
        #(*). TODO: add current results to a proper dataset/array/panda for proper analysis of ALL experiments and also add distance variable etc'

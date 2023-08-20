


from RapidBase.import_all import *



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


def turn_list_of_JAI_large_bin_files_and_turn_them_into_avi_files():
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


def turn_list_of_PelicanD_large_bin_files_and_turn_them_into_avi_files():
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


def turn_list_of_PelicanD_large_bin_files_and_turn_them_into_large_Bin_and_avi_files():
    pelicanD_filenames_list = []
    pelicanD_filenames_list = get_filenames_from_folder_string_pattern(r'E:\Quickshot\12.4.2022 - natznatz experiments',
                                                                       flag_recursive=True,
                                                                       max_number_of_filenames=np.inf,
                                                                       flag_full_filename=True,
                                                                       string_pattern_to_search='*.Bin',
                                                                       flag_sort=False)
    pelicanD_filenames_list = get_filenames_from_folder_string_pattern(r'E:\JAI_cars_moving', flag_recursive=True)

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


def turn_list_of_JAI_large_bin_files_and_turn_them_into_large_Bin_and_avi_files():
    original_full_filenames_list = []
    master_folder = r'E:\JAI_cars_moving'
    original_full_filenames_list = get_filenames_from_folder_string_pattern(master_folder, flag_recursive=True)

    ### Get (H,W) For The Different Files: ###
    H = 1024
    W = 2048

    ### Write Down Movie .avi File: ###
    for pelicanD_experiment_index in np.arange(len(original_full_filenames_list)):
        ### Get Current Filenames: ###
        current_full_filename = original_full_filenames_list[pelicanD_experiment_index]

        ### Get Filename Parts: ###
        folder, filename, filename_without_extension, filename_extension = path_get_all_filename_parts(current_full_filename)
        ### Check if there are more files in the same folder: ###
        filenames_in_current_folder = get_filenames_from_folder_string_pattern(folder, flag_recursive=False, string_pattern_to_search='*.raw')
        if len(filenames_in_current_folder) > 1:
            ### Put current filename into a new folder for better order: ###
            new_folder = os.path.join(folder, filename_without_extension)
            path_create_path_if_none_exists(new_folder)
            ### Move .raw file into new folder: ###
            new_full_filename = os.path.join(new_folder, filename)
            shutil.move(current_full_filename, new_full_filename)
            ### Rename folder: ###
            folder = new_folder
            current_full_filename = os.path.join(new_folder, filename)

        ### Initialize Movie: ###
        movie_filename = os.path.join(folder, filename_without_extension + '.avi')
        video_object = cv2.VideoWriter(movie_filename, 0, 50, (W, H))

        ### Initialize Binary Reader: ###
        scene_infile = open(current_full_filename, 'rb')

        ### Initialize Binary Writter: ###
        new_filename = os.path.join(folder, filename_without_extension + '.Bin')
        f = open(new_filename, 'a')

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
                image = np.array(image).astype(np.uint16).astype(float)
                input_tensor = BW2RGB(torch.tensor(image).float())

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

# turn_list_of_JAI_large_bin_files_and_turn_them_into_large_Bin_and_avi_files()


def turn_list_of_folders_with_individual_bin_images_into_large_Bin_and_avi_files_under_a_different_super_folder():

    individual_images_folders_list = path_get_folder_names(r'E:\Quickshot\19_7_22_flir_exp_Bin\individual_images_bin_files')

    ### Get (H,W) For The Different Files: ###
    H, W = 512, 640

    ### Change Filenames rjust: ###
    for flir_experiment_index in np.arange(len(individual_images_folders_list)):
        ### Get Current Filenames: ###
        flir_files_path = individual_images_folders_list[flir_experiment_index]
        current_folder_filenames_list = get_filenames_from_folder(flir_files_path)
        ### Loop Over Individual Images: ###
        for i in np.arange(len(current_folder_filenames_list)):
            current_full_filename = current_folder_filenames_list[i]
            current_folder = os.path.split(current_full_filename)[0]
            current_filename = os.path.split(current_full_filename)[1]
            current_number_string = os.path.splitext(current_filename)[0]
            new_number_string = string_rjust(np.int(current_number_string), 6)
            new_filename = new_number_string + '.Bin'
            new_full_filename = os.path.join(current_folder, new_filename)
            os.rename(current_full_filename, new_full_filename)

    ### Loop Over The Different Folders: ###
    for current_folder_index in np.arange(0, len(individual_images_folders_list)):
        ### Get Current Filenames: ###
        current_folder_path = individual_images_folders_list[current_folder_index]
        current_folder_filenames_list = get_filenames_from_folder(current_folder_path)

        ### Initialize 1 Big .Bin File: ###
        new_folder = current_folder_path.replace('individual_images_bin_files', 'one_big_bin_file')
        path_create_path_if_none_exists(new_folder)
        folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(current_folder_filenames_list[0])
        new_folder = folder.replace('individual_images_bin_files', 'one_big_bin_file')
        path_create_path_if_none_exists(new_folder)
        specific_folder_name = os.path.split(folder)[1]
        new_filename = os.path.join(new_folder, specific_folder_name + '.Bin')

        # ### Initialize 1 Big .Bin File ###
        # new_filename = os.path.split(current_folder_path)[-1] + '.Bin'
        # new_filename = os.path.join(os.path.split(current_folder_path)[0], new_filename)
        # ### TEMP: ###
        # new_filename = new_filename.replace('individual_images_bin_files', 'one_big_bin_file')
        # folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(new_filename)
        # new_folder = os.path.join(folder, filename_without_extension)
        # new_filename = os.path.join(new_folder, filename)
        # path_create_path_if_none_exists(new_folder)

        ### Initialize Binary File Writter: ###
        f = open(new_filename, 'a')

        ### Loop Over Movie Filenames/Frames: ###
        # movie_filename = os.path.split(current_folder_path)[-1] + '.avi'
        # movie_filename = os.path.join(os.path.split(current_folder_path)[0], movie_filename)
        # results_folder = os.path.join(new_folder, 'Results')
        # path_create_path_if_none_exists(results_folder)

        ### Add "Results" Folder To Match QS-Palantir Folders: ###
        results_folder = os.path.join(new_folder, 'Results')
        path_create_path_if_none_exists(results_folder)
        movie_filename = os.path.join(results_folder, 'Original_Movie.avi')
        video_object = cv2.VideoWriter(movie_filename, 0, 10, (W, H))

        ### Loop Over Individual Filenames In This Specific Folder: ###
        for i in np.arange(len(current_folder_filenames_list)):
            print(i)
            ### Get Current Image: ###
            current_full_filename = current_folder_filenames_list[i]
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




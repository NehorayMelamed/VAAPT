from RapidBase.import_all import *


def split_video_into_constant_time_chunks(movie_full_filenames=''):
    ### Original Movie: ###
    movie_full_filenames = ['/home/mafat/Videos/4K Video Downloader/Walking and flying in Toride, Ibaraki・4K HDR.mkv']

    ### Where To Save Results: ###
    super_folder_to_save_everything_at = '/home/mafat/DataSets/Aviram/1'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)

    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9
    max_number_of_sequences = 1e9
    max_number_of_images = 1e9
    #(*). Sequene/Time Parameters:
    number_of_frames_per_sequence = int(60*1)  # 30[fps] * #[number of seconds]
    number_of_frames_to_skip_between_sequeces_of_the_same_video = int(30*(60*2))  # 30[fps] * #[number of seconds]
    #(*). Crops and decimation:
    flag_all_crops = False
    # crop_sizes = [256]
    crop_size = 256*4
    number_of_random_crops_per_crop_size = 50
    min_width_start = 0
    max_width_stop = 4000
    min_height_start = 0
    max_height_stop = 4000
    decimation = 1

    # (1). Images In Folders:
    flag_bayer_or_rgb = 'rgb'
    flag_BW_or_rgb = 'BW' # 'BW', 'RGB'


    movie_filename_counter = 0
    for movie_full_filename in movie_full_filenames:
        movie_filename_counter += 1

        ### Open Movie Stream: ###
        video_stream = cv2.VideoCapture(movie_full_filename)
        movie_filename = movie_full_filename.split('/')[-1].split('.')[0]

        ### Read first 3 frames because of a BUG in opencv: ###
        flag_frame_available, frame = video_stream.read()
        flag_frame_available, frame = video_stream.read()
        flag_frame_available, frame = video_stream.read()

        ### Loop Over Video Stream: ###
        number_of_sequences_so_far = 0
        number_of_total_writes_so_far = 0
        number_of_frames_so_far = 0
        flag_frame_available = True
        flag_continue_reading_movie = video_stream.isOpened() and number_of_sequences_so_far<max_number_of_sequences
        flag_continue_reading_movie = flag_continue_reading_movie and number_of_total_writes_so_far<max_number_of_images
        flag_continue_reading_movie = flag_continue_reading_movie and number_of_frames_so_far<max_number_of_frames
        flag_continue_reading_movie = flag_continue_reading_movie and flag_frame_available
        while flag_continue_reading_movie:
            # tic()
            number_of_sequences_so_far += 1
            crops_indices = []

            ### Loop Over all the different crops and write the down: ###
            total_crop_counter = 0
            frame_height, frame_width, number_of_channels = frame.shape
            number_of_width_crops = max(frame_width // crop_size, 1)
            number_of_height_crops = max(frame_height // crop_size, 1)
            current_crop_size_height = min(crop_size, frame_height)
            current_crop_size_width = min(crop_size, frame_width)

            ### Pick random crops indices: ###
            if flag_all_crops:
                random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
            else:
                random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size]

            ### Loop over current sequence frames: ###
            for mini_frame_number in arange(number_of_frames_per_sequence + number_of_frames_to_skip_between_sequeces_of_the_same_video):  #loop over number_of_frame_to_write+number_of_frames_to_wait
                tic()
                ### Read Frame If Available: ###
                if flag_frame_available:
                    number_of_frames_so_far += 1
                    flag_frame_available, frame = video_stream.read()

                ### Loop Over Different Crops: ###
                if flag_frame_available and mini_frame_number < number_of_frames_per_sequence:
                    frame_height, frame_width, number_of_channels = frame.shape

                    if flag_BW_or_rgb == 'BW':
                        number_of_channels = 1

                    current_crop_counter = 0
                    ### Loop over the different crops: ###
                    for width_crop_index in arange(number_of_width_crops):
                        for height_crop_index in arange(number_of_height_crops):
                            if current_crop_counter in random_crop_indices:
                                ### Get Current Crop: ###
                                current_frame_crop = frame[height_crop_index*current_crop_size_height : (height_crop_index+1)*current_crop_size_height:decimation,
                                                           width_crop_index*current_crop_size_width : (width_crop_index+1)*current_crop_size_width, :]

                                ### Save Image: ####
                                current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)))
                                new_sub_folder = os.path.join(current_crop_folder, 'Movie2_' + str(movie_filename_counter) +
                                                              '_Sequence_' + str(number_of_sequences_so_far) +
                                                              '_Crop_' + str(current_crop_counter))
                                if not os.path.exists(new_sub_folder):
                                    os.makedirs(new_sub_folder)
                                save_image_numpy(folder_path=new_sub_folder, filename=string_rjust(mini_frame_number,3) + '.png',
                                                 numpy_array=current_frame_crop, flag_convert_bgr2rgb=False,
                                                 flag_scale=False)

                            ### Uptick crop number: ###
                            current_crop_counter += 1
                            total_crop_counter += 1
                            # pause(0.01)

                toc('Sequence no. ' + str(number_of_sequences_so_far) + ', In Sequence Frame no. ' + str(mini_frame_number))

            ### Whether to continue: ###
            flag_continue_reading_movie = video_stream.isOpened() and number_of_sequences_so_far < max_number_of_sequences
            flag_continue_reading_movie = flag_continue_reading_movie and number_of_total_writes_so_far < max_number_of_images
            flag_continue_reading_movie = flag_continue_reading_movie and number_of_frames_so_far < max_number_of_frames
            flag_continue_reading_movie = flag_continue_reading_movie and flag_frame_available



def split_videos_in_folders_into_proper_form_crops():
    ### Original Movie: ###
    super_folder_to_get_images_from = r'/home/mafat/DataSets/Davis/DAVIS-2017-Unsupervised-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution'
    sub_folders = path_get_folder_names(super_folder_to_get_images_from)

    ### Where To Save Results: ###
    super_folder_to_save_everything_at = '/home/mafat/DataSets/DAVIS_CROPS'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)

    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9
    max_number_of_sequences = 1e9
    max_number_of_images = 1e9
    #(*). Sequene/Time Parameters:
    number_of_frames_per_sequence = int(200*1)  # 30[fps] * #[number of seconds]
    number_of_frames_to_skip_between_sequeces_of_the_same_video = int(30*(60*2))  # 30[fps] * #[number of seconds]
    #(*). Crops and decimation:
    flag_all_crops = True
    # crop_sizes = [256]
    crop_size = 256
    number_of_random_crops_per_crop_size = 50
    min_width_start = 0
    max_width_stop = 4000
    min_height_start = 0
    max_height_stop = 4000
    decimation = 1

    # (1). Images In Folders:
    flag_bayer_or_rgb = 'rgb'
    flag_BW_or_rgb = 'BW' # 'BW', 'RGB'

    ### Initialize Counters: ###
    number_of_sequences_so_far = 0
    number_of_total_writes_so_far = 0
    number_of_frames_so_far = 0

    movie_filename_counter = 0
    for current_sub_folder in sub_folders:
        ### Get current sub folder filenames: ###
        current_sub_folder_image_filenames = get_image_filenames_from_folder(current_sub_folder, number_of_images=np.inf, flag_recursive=False)
        flag_frame_available = len(current_sub_folder_image_filenames) <= number_of_frames_per_sequence

        ### Loop Over Video Stream: ###
        flag_continue_reading_movie = number_of_sequences_so_far<max_number_of_sequences
        flag_continue_reading_movie = flag_continue_reading_movie and number_of_total_writes_so_far<max_number_of_images
        flag_continue_reading_movie = flag_continue_reading_movie and number_of_frames_so_far<max_number_of_frames
        flag_continue_reading_movie = flag_continue_reading_movie and flag_frame_available

        sub_folder_frames_counter = 0

        while flag_continue_reading_movie: #loop over sequences
            # tic()
            number_of_sequences_so_far += 1
            crops_indices = []

            ### Read Frame: ###
            frame = read_image_general(current_sub_folder_image_filenames[sub_folder_frames_counter])

            ### Loop Over all the different crops and write the down: ###
            total_crop_counter = 0
            frame_height, frame_width, number_of_channels = frame.shape
            number_of_width_crops = max(frame_width // crop_size, 1)
            number_of_height_crops = max(frame_height // crop_size, 1)
            current_crop_size_height = min(crop_size, frame_height)
            current_crop_size_width = min(crop_size, frame_width)

            ### Pick random crops indices: ###
            if flag_all_crops:
                random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
            else:
                random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size]

            ### Loop over current sequence frames: ###
            current_number_of_frames_per_sequence = min(number_of_frames_per_sequence, len(current_sub_folder_image_filenames))
            for mini_frame_number in arange(current_number_of_frames_per_sequence):
                tic()
                ### Read Frame If Available: ###
                frame = read_image_general(current_sub_folder_image_filenames[sub_folder_frames_counter])
                number_of_frames_so_far += 1
                sub_folder_frames_counter += 1

                ### Loop Over Different Crops: ###
                if sub_folder_frames_counter < number_of_frames_per_sequence:
                    frame_height, frame_width, number_of_channels = frame.shape

                    if flag_BW_or_rgb == 'BW':
                        number_of_channels = 1

                    current_crop_counter = 0
                    ### Loop over the different crops: ###
                    for width_crop_index in arange(number_of_width_crops):
                        for height_crop_index in arange(number_of_height_crops):
                            if current_crop_counter in random_crop_indices:
                                ### Get Current Crop: ###
                                current_frame_crop = frame[height_crop_index*current_crop_size_height : (height_crop_index+1)*current_crop_size_height:decimation,
                                                           width_crop_index*current_crop_size_width : (width_crop_index+1)*current_crop_size_width, :]
                                if flag_BW_or_rgb == 'BW':
                                    current_frame_crop = RGB2BW(current_frame_crop)
                                    current_frame_crop = current_frame_crop.astype(np.uint8)

                                ### Save Image: ####
                                current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)))
                                new_sub_folder = os.path.join(current_crop_folder, 'DAVIS_' + str(movie_filename_counter) +
                                                              '_Sequence_' + str(number_of_sequences_so_far) +
                                                              '_Crop_' + str(current_crop_counter))
                                if not os.path.exists(new_sub_folder):
                                    os.makedirs(new_sub_folder)
                                save_image_numpy(folder_path=new_sub_folder, filename=string_rjust(mini_frame_number,3) + '.png',
                                                 numpy_array=current_frame_crop, flag_convert_bgr2rgb=False,
                                                 flag_scale=False)

                            ### Uptick crop number: ###
                            current_crop_counter += 1
                            total_crop_counter += 1
                            # pause(0.01)

                toc('Sequence no. ' + str(number_of_sequences_so_far) + ', In Sequence Frame no. ' + str(mini_frame_number))

            ### Whether to continue: ###
            flag_continue_reading_movie = number_of_sequences_so_far < max_number_of_sequences
            flag_continue_reading_movie = flag_continue_reading_movie and number_of_total_writes_so_far < max_number_of_images
            flag_continue_reading_movie = flag_continue_reading_movie and number_of_frames_so_far < max_number_of_frames
            flag_continue_reading_movie = flag_continue_reading_movie and flag_frame_available
            flag_continue_reading_movie = flag_continue_reading_movie and sub_folder_frames_counter < len(current_sub_folder_image_filenames) - current_number_of_frames_per_sequence



def transform_videos_dataset_from_RGB_to_BW():
    ### Original Movie: ###
    super_folder_to_get_images_from = r'/home/mafat/DataSets/Movies_DataSets/CropHeight_256_CropWidth_256_2'
    sub_folders = path_get_folder_names(super_folder_to_get_images_from)

    ### Where To Save Results: ###
    super_folder_to_save_everything_at = '/home/mafat/DataSets/Movies_DataSets/CropHeight_256_CropWidth_256_2_BW'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)

    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9
    max_number_of_sequences = 1e9
    max_number_of_images = 1e9
    #(*). Sequene/Time Parameters:
    number_of_frames_per_sequence = int(200*1)  # 30[fps] * #[number of seconds]
    #(*). Crops and decimation:
    flag_all_crops = True
    crop_size = 256
    number_of_random_crops_per_crop_size = 50
    min_width_start = 0
    max_width_stop = 4000
    min_height_start = 0
    max_height_stop = 4000
    decimation = 1

    # (1). Images In Folders:
    flag_bayer_or_rgb = 'rgb'
    flag_BW_or_rgb = 'BW' # 'BW', 'RGB'

    ### Initialize Counters: ###
    number_of_sequences_so_far = 0
    number_of_total_writes_so_far = 0
    number_of_frames_so_far = 0

    movie_filename_counter = 0
    for current_sub_folder in sub_folders:
        ### Get current sub folder filenames: ###
        current_sub_folder_image_filenames = get_image_filenames_from_folder(current_sub_folder, number_of_images=np.inf, flag_recursive=False)
        flag_frame_available = len(current_sub_folder_image_filenames) <= number_of_frames_per_sequence

        ### Loop Over Video Stream: ###
        flag_continue_reading_movie = number_of_sequences_so_far<max_number_of_sequences
        flag_continue_reading_movie = flag_continue_reading_movie and number_of_total_writes_so_far<max_number_of_images
        flag_continue_reading_movie = flag_continue_reading_movie and number_of_frames_so_far<max_number_of_frames
        flag_continue_reading_movie = flag_continue_reading_movie and flag_frame_available

        sub_folder_frames_counter = 0

        while flag_continue_reading_movie: #loop over sequences
            # tic()
            number_of_sequences_so_far += 1
            crops_indices = []

            ### Read Frame: ###
            frame = read_image_general(current_sub_folder_image_filenames[sub_folder_frames_counter])

            ### Loop Over all the different crops and write the down: ###
            total_crop_counter = 0
            frame_height, frame_width, number_of_channels = frame.shape
            number_of_width_crops = max(frame_width // crop_size, 1)
            number_of_height_crops = max(frame_height // crop_size, 1)
            current_crop_size_height = min(crop_size, frame_height)
            current_crop_size_width = min(crop_size, frame_width)

            ### Pick random crops indices: ###
            if flag_all_crops:
                random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
            else:
                random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size]

            ### Loop over current sequence frames: ###
            current_number_of_frames_per_sequence = min(number_of_frames_per_sequence, len(current_sub_folder_image_filenames))
            for mini_frame_number in arange(current_number_of_frames_per_sequence):
                tic()
                ### Read Frame If Available: ###
                frame = read_image_general(current_sub_folder_image_filenames[sub_folder_frames_counter])
                number_of_frames_so_far += 1
                sub_folder_frames_counter += 1

                ### Loop Over Different Crops: ###
                if sub_folder_frames_counter < number_of_frames_per_sequence:
                    frame_height, frame_width, number_of_channels = frame.shape

                    if flag_BW_or_rgb == 'BW':
                        number_of_channels = 1

                    current_crop_counter = 0
                    ### Loop over the different crops: ###
                    for width_crop_index in arange(number_of_width_crops):
                        for height_crop_index in arange(number_of_height_crops):
                            if current_crop_counter in random_crop_indices:
                                ### Get Current Crop: ###
                                current_frame_crop = frame[height_crop_index*current_crop_size_height : (height_crop_index+1)*current_crop_size_height:decimation,
                                                           width_crop_index*current_crop_size_width : (width_crop_index+1)*current_crop_size_width, :]
                                if flag_BW_or_rgb == 'BW':
                                    current_frame_crop = RGB2BW(current_frame_crop)
                                    current_frame_crop = current_frame_crop.astype(np.uint8)

                                ### Save Image: ####
                                current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)))
                                new_sub_folder = os.path.join(current_crop_folder, 'Movie3_' + str(movie_filename_counter) +
                                                              '_Sequence_' + str(number_of_sequences_so_far) +
                                                              '_Crop_' + str(current_crop_counter))
                                if not os.path.exists(new_sub_folder):
                                    os.makedirs(new_sub_folder)
                                save_image_numpy(folder_path=new_sub_folder, filename=string_rjust(mini_frame_number,3) + '.png',
                                                 numpy_array=current_frame_crop, flag_convert_bgr2rgb=False,
                                                 flag_scale=False)

                            ### Uptick crop number: ###
                            current_crop_counter += 1
                            total_crop_counter += 1
                            # pause(0.01)

                toc('Sequence no. ' + str(number_of_sequences_so_far) + ', In Sequence Frame no. ' + str(mini_frame_number))

            ### Whether to continue: ###
            flag_continue_reading_movie = number_of_sequences_so_far < max_number_of_sequences
            flag_continue_reading_movie = flag_continue_reading_movie and number_of_total_writes_so_far < max_number_of_images
            flag_continue_reading_movie = flag_continue_reading_movie and number_of_frames_so_far < max_number_of_frames
            flag_continue_reading_movie = flag_continue_reading_movie and flag_frame_available
            flag_continue_reading_movie = flag_continue_reading_movie and sub_folder_frames_counter < len(current_sub_folder_image_filenames) - current_number_of_frames_per_sequence



def create_binary_files_from_seperated_scene_folders_of_images(movie_full_filenames):
    ### Original Movie: ###
    #(1). Walk Through sub folders and get all .mkv (!!!!) video files:
    # movie_full_filename_filenames = []
    # master_folder_path = 'F:\Movies\Movie_scenes_seperated_into_different_videos'
    # for dirpath, _, fnames in sorted(os.walk(master_folder_path)):  # os.walk!!!!
    #     for fname in sorted(fnames):
    #         if fname.split('.')[-1] == 'mkv':
    #             movie_full_filename_filenames.append(os.path.join(dirpath, fname))
    # #(2). Specific filenames:
    # movie_full_filenames = ['F:\Movies\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']


    ### Seperated Scene Folders: ###
    super_folder_containing_sub_folders_of_scenes = 'F:\TrackingNet';
    scene_folders_list = []
    for dirpath, _, fnames in sorted(os.walk(super_folder_containing_sub_folders_of_scenes)):  # os.walk!!!!
        scene_folders_list.append(dirpath);


    ### Where To Save Results: ###
    super_folder_to_save_everything_at = 'G:\Movie_Scenes_bin_files'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)


    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9;
    max_number_of_sequences = 1e9;
    max_number_of_images = 1e9;
    #(*). Sequene/Time Parameters:
    number_of_frames_per_sequence = int(30*5);  # 30[fps] * #[number of seconds]
    number_of_frames_to_skip_between_sequeces_of_the_same_video = int(30*15);  # 30[fps] * #[number of seconds]
    #(*). Crops and decimation:
    flag_all_crops = False
    crop_sizes = [250]
    number_of_random_crops_per_crop_size = 5;
    crop_size = 250;
    min_width_start = 0;
    max_width_stop = 4000;
    min_height_start = 0;
    max_height_stop = 4000;
    decimation = 1;

    # (1). Images In Folders:
    # flag_binary_or_images = 'images';
    flag_binary_or_images = 'binary';
    flag_bayer_or_rgb = 'rgb';
    flag_random_gain_and_gamma = False;
    # (2). Binary File:
    number_of_images_per_binary_file = 10000000000;
    binary_dtype = 'uint8';


    movie_filename_counter = 0;
    for scene_folder in scene_folders_list:
        movie_filename_counter += 1;

        ### Get all images in scene folder in correct order: ###
        filenames_in_folder = get_image_filenames_from_folder(scene_folder)
        image_filenames_list = []
        for filenames in filenames_in_folder:
            image_filenames_list.append(int32(filenames.split('.')[0].split('\\')[-1]))
        sorted_indices = argsort(image_filenames_list);
        sorted_image_filenames_list = []
        for index in sorted_indices:
            sorted_image_filenames_list.append(filenames_in_folder[index])
        movie_filename = scene_folder.split('\\')


        ### Loop Over Video Stream: ###
        number_of_sequences_of_current_scene_so_far = 0;
        number_of_total_writes_of_current_scene_so_far = 0
        number_of_frames_of_current_scene_so_far = 0
        flag_frame_available = True
        while flag_frame_available and number_of_sequences_of_current_scene_so_far<max_number_of_sequences and number_of_total_writes_of_current_scene_so_far<max_number_of_images and number_of_frames_of_current_scene_so_far<max_number_of_frames:
            tic()
            ### Read Current Frames: ###
            number_of_sequences_of_current_scene_so_far += 1;
            crops_indices = []
            for mini_frame_number in arange(number_of_frames_per_sequence + number_of_frames_to_skip_between_sequeces_of_the_same_video):  #loop over number_of_frame_to_write + number_of_frames_to_wait
                if number_of_frames_of_current_scene_so_far < number_of_images_in_current_folder:
                    frame = read_image_cv2(sorted_image_filenames_list[number_of_frames_of_current_scene_so_far],flag_convert_to_rgb=1,flag_normalize_to_float=0)
                    frame_height, frame_width, number_of_channels = frame.shape
                else:
                    flag_frame_available = False;
                number_of_frames_of_current_scene_so_far += 1

                ### Loop Over Different Crops: ###
                if flag_frame_available and mini_frame_number<number_of_frames_per_sequence:
                    frame_height, frame_width, number_of_channels = frame.shape;
                    data_type_padded = str(frame.dtype).rjust(10,'0'); #get the data type and right fill with zeros to account for the different types: uint8, int8, float32....
                    max_image_value = np.array([np.iinfo(frame.dtype).max]).astype('float32')[0] # !!!!!!! max_value to be later used, if so wanted, to normalize the image. used for generality sake !!!!!!!!!!!!!!!!!!!!!!!!!!

                    if flag_bayer_or_rgb:
                        number_of_channels = 1; #Bayer


                    ### Start off by creating all the binary files writers in a list: ###
                    if flag_binary_or_images == 'binary':
                        binary_fid_writers_list = []
                        for crop_size_counter, crop_size in enumerate(crop_sizes):
                            current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropSize_' + str(crop_size))
                            create_folder_if_needed(current_crop_folder);
                            number_of_width_crops = max(frame_width // crop_size, 1)
                            number_of_height_crops = max(frame_height // crop_size, 1)
                            ### Pick random crops indices: ###
                            if flag_all_crops:
                                random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
                            else:
                                random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                                random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size[crop_size_counter]]
                                # rows_start = randint(min_height_start, min(max_height_stop, frame_height - crop_size))
                                # rows_stop = rows_start + crop_size;
                                # cols_start = randint(min_width_start, min(max_width_stop, frame_width - crop_size))
                                # cols_stop = cols_start + crop_size;
                                # crops_indices.append([rows_start, rows_stop, cols_start, cols_stop])
                            current_crop_counter = 0;
                            for width_crop_index in arange(number_of_width_crops):
                                for height_crop_index in arange(number_of_height_crops):
                                    fid_filename_X = os.path.join(current_crop_folder, 'MovieFilename_' + movie_filename +
                                                                  '_SequenceNo_' + str(number_of_sequences_of_current_scene_so_far) +
                                                                  '_CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)) +
                                                                  '_CropCounter_' + str(current_crop_counter) + '.bin')
                                    fid_Write_X = open(fid_filename_X, 'ab')
                                    binary_fid_writers_list.append(fid_Write_X)
                                    print('crop_size: ' + str(crop_size) + ', written height: ' + str(min(frame_height,crop_size)) + ', written width: ' + str(min(frame_width,crop_size)))
                                    np.array(min(frame_height,crop_size)).tofile(fid_Write_X)
                                    np.array(min(frame_width,crop_size)).tofile(fid_Write_X)
                                    np.array(number_of_channels).tofile(fid_Write_X)
                                    np.array(data_type_padded).tofile(fid_Write_X)
                                    np.array(max_image_value).tofile(fid_Write_X);
                                    current_crop_counter += 1;



                    ### Loop Over all the different crops and write the down: ###
                    total_crop_counter = 0;
                    for crop_size in crop_sizes:
                        number_of_width_crops = max(frame_width // crop_size, 1)
                        number_of_height_crops = max(frame_height // crop_size, 1)
                        current_crop_size_height = min(crop_size, frame_height);
                        current_crop_size_width = min(crop_size, frame_width);

                        current_crop_number = 0;
                        ### Loop over the different crops: ###
                        for width_crop_index in arange(number_of_width_crops):
                            for height_crop_index in arange(number_of_height_crops):
                                if current_crop_number in random_crop_indices:
                                    ### Get Current Crop: ###
                                    current_frame_crop = frame[height_crop_index*current_crop_size_height:(height_crop_index+1)*current_crop_size_height:decimation, width_crop_index*current_crop_size_width:(width_crop_index+1)*current_crop_size_width, :]

                                    ### Save Image: ####
                                    if flag_binary_or_images == 'images':
                                        #(1). Image In Folders:
                                        current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)))
                                        new_sub_folder = os.path.join(current_crop_folder, 'Movie_' + str(movie_filename_counter) + '_Sequence_' + str(number_of_sequences_of_current_scene_so_far) + '_Crop_' + str(current_crop_number))
                                        if not os.path.exists(new_sub_folder):
                                            os.makedirs(new_sub_folder)
                                        save_image_numpy(folder_path=new_sub_folder, filename=str(mini_frame_number) + '.png',
                                                         numpy_array=current_frame_crop, flag_convert_bgr2rgb=False,
                                                         flag_scale=False)

                                    elif flag_binary_or_images == 'binary':
                                        #(2). Binary File:
                                        current_fid_Write_X = binary_fid_writers_list[total_crop_counter]
                                        if flag_bayer_or_rgb == 'bayer':
                                            if flag_random_gain_and_gamma:
                                                current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=True)
                                                current_frame_crop_bayer.tofile(current_fid_Write_X)
                                            else:
                                                current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=False)
                                                current_frame_crop_bayer.tofile(current_fid_Write_X)
                                        elif flag_bayer_or_rgb == 'rgb':
                                            #TODO: implement random gain and gamma from saved RGB images!
                                            if flag_random_gain_and_gamma:
                                                current_frame_crop.tofile(current_fid_Write_X)
                                            else:
                                                # current_frame_crop_tensor_form = np.transpose(current_frame_crop,[])
                                                current_frame_crop.tofile(current_fid_Write_X)


                                ### Uptick crop number: ###
                                current_crop_number += 1;
                                total_crop_counter += 1;


            ### After writing things down close all fid writers: ###
            if flag_binary_or_images == 'binary':
                for fid in binary_fid_writers_list:
                    fid.close()
            toc('Number of Seconds So Far: ' + str(number_of_frames_of_current_scene_so_far/24))

    if flag_binary_or_images == 'binary':
        for fid in binary_fid_writers_list:
            fid.close()


# ### Test Binary Files Writers: ###
# binary_full_filename = 'F:\Movie Scenes Temp\CropSize_100/CropHeight_100_CropWidth_100_CropCounter_0.bin'
# fid_Read_X = open(binary_full_filename, 'rb')
# frame_height = np.fromfile(fid_Read_X, 'int32', count=1)[0];
# frame_width = np.fromfile(fid_Read_X, 'int32', count=1)[0];
# number_of_channels = np.fromfile(fid_Read_X, 'int32', count=1)[0];
# data_type = np.fromfile(fid_Read_X, dtype='<U10', count=1)[0]
# data_type = data_type.split('0')[-1]
# max_value = np.fromfile(fid_Read_X, 'float32', count=1)[0];
#
# number_of_elements_per_image = frame_height*frame_width*number_of_channels
# mat_in = np.fromfile(fid_Read_X, data_type, count=number_of_elements_per_image)
# mat_in = mat_in.reshape((frame_height,frame_width,number_of_channels))
# imshow(mat_in)
#
# # fid_Read_X.close()


def split_video_into_number_of_parts():
    ### Original Movie: ###
    movie_full_filenames = ['C:/Users\dkarl/Downloads/Full Movie 4k ULtra HD MSC MUSICA Cruise Tour Medi.mkv']
    # movie_full_filename = 'C:/Users\dkarl/Downloads/welcome back full movie hd.mp4'



    total_number_of_frames = 97500;
    number_of_parts = 25
    frames_per_part = 97500//number_of_parts;

    frame_height = 3840
    frame_width = 2160

    movie_filename_counter = 0;
    for movie_full_filename in movie_full_filenames:
        ### Open Movie Stream: ###
        video_stream = cv2.VideoCapture(movie_full_filename)
        movie_filename_counter += 1;

        for part_counter in arange(number_of_parts):
            ### Create New File For This Movie's Part: ###
            created_video_filename = 'C:/Users\dkarl/Downloads//' + movie_full_filename.split('.')[0].split('/')[-1] + ' Part' + str(part_counter) + '.mkv'
            fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
            video_writer = cv2.VideoWriter(created_video_filename, fourcc, 25.0, (frame_height, frame_width))

            ### Loop Over Frames For This Part: ###
            for current_part_frame_counter in arange(frames_per_part):
                flag_frame_available, frame = video_stream.read()
                video_writer.write(frame);
                print('part number ' + str(part_counter) + ', frame number ' + str(current_part_frame_counter))

            ### After Finishing Writing This Part To Disk Release The Video Writer: ###
            cv2.destroyAllWindows()
            video_writer.release()


    cv2.destroyAllWindows()
    video_writer.release()


def split_video_into_all_image_frames():
    ### Get original movie filename and folder to save to: ###
    # movie_full_filename = '/home/mafat/DataSets/Drones_Random_Experiments/RGB_drone_WhiteNight2.MP4'
    movie_full_filename = '/home/mafat/DataSets/Aviram/7/Vayu HD aerial demo over Los Angeles.mp4'
    folder_to_save_images_at = '/home/mafat/DataSets/Aviram/7'
    if not os.path.exists(folder_to_save_images_at):
        os.makedirs(folder_to_save_images_at)

    ### Start Video Capture Object: ###
    video_stream = cv2.VideoCapture(movie_full_filename)
    max_number_of_frames = 100000000
    current_step = 0

    ### Loop over number of images: ###
    while current_step < max_number_of_frames:
        tic()
        ### Read Image: ###
        flag_frame_available, current_frame = video_stream.read()
        if flag_frame_available==False:
            break

        ### Save Image: ###
        save_image_numpy(folder_path=folder_to_save_images_at,
                         filename=str(current_step).rjust(8, '0') + '.png',
                         numpy_array=current_frame,
                         flag_convert_bgr2rgb=False,
                         flag_scale=False)
        current_step += 1
        toc()




def split_video_into_image_frames_and_different_noises():
    ### Set Up Decices: ###
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device_cpu = torch.device('cpu')
    Generator_device = device1
    ### Get Raw File and Video_Stream Object: ###
    # movie_full_filenames = ['G:\Raw Films/Walking_in_Tokyo_Shibuya_at_night.mkv',
    #                        'G:\Raw Films/Driving_Downtown_Chicago_4K_USA.mkv',
    #                        'G:\Raw Films/3HRS_Stunning_Underwater_Footage_Relaxing_Music.mkv',
    #                        'G:\Raw Films/Night_videowalk_in_East_Shinjuku_Tokyo.mkv',
    #                        'G:\Raw Films/Rome_Virtual_Walking_Tour_in_4K_Rome_City_Travel.mkv',
    #                        'G:\Raw Films/Walk_to_the_Kaohsiung_Main_Public_Library_at_night.mkv',
    #                        'G:\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']
    movie_full_filenames = ['G:\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']
    saved_images_super_folder = 'F:\TNR - OFFICIAL TEST IMAGES\Dynamic_Videos'
    if not os.path.exists(saved_images_super_folder):
        os.makedirs(saved_images_super_folder)

    ### Noise Gains To Save: ###
    noise_gains = [0,30,65,100]
    # noise_gains = [100]

    ### Frames to Capture: ###
    # (1). Initial Frame:
    flag_use_initial_frame_or_second = 'second'  # 'frame'/'second'
    initial_frames = 10;
    # initial_seconds = [0*60, 10*60, 20*60, 30*60, 40*60, 50*60, 60*60];
    initial_seconds = [7*60+15];
    # (2). Number of Frams:
    flag_use_final_frame_or_second = 'frame'
    # number_of_frames = fps * 2;
    number_of_seconds = 10
    final_second = 0 * 3600 + 0 * 60 + 5;

    ### Crop Size: ###
    crop_size = 2000;


    ### Loop over movie files and create "dataset" (long sequences when i want to qualitatively compare videos and many short sequences when i want to gather statistics): ###
    for movie_full_filename in movie_full_filenames:
        video_stream = cv2.VideoCapture(movie_full_filename)
        frame_width = int(video_stream.get(3))
        frame_height = int(video_stream.get(4))
        fps = video_stream.get(5)
        total_number_of_frames = int(video_stream.get(7))
        ### Read a few frames for voodoo good luck: ###
        video_stream.read();
        video_stream.read();
        video_stream.read();

        # initial_frames = [0 * 60 * fps, 10 * 60 * fps, 20 * 60 * fps, 30 * 60 * fps, 40 * 60 * fps, 50 * 60 * fps, 60 * 60 * fps]

        for initial_second in initial_seconds:
            initial_frame = int(initial_second * fps)
            number_of_frames = int(number_of_seconds * fps)
            # ### Frames to Capture: ###
            # #(1). Initial Frame:
            # flag_use_initial_frame_or_second = 'second' #'frame'/'second'
            # initial_frame = 10;
            # initial_second = 0*3600 + 30*60 + 2;
            # if flag_use_initial_frame_or_second == 'second':
            #     initial_frame = int(fps * initial_second);
            # #(2). Number of Frams:
            # flag_use_final_frame_or_second = 'frame'
            # number_of_frames = fps*10;
            # final_second = 0*3600 + 0*60 + 5;
            # if flag_use_final_frame_or_second == 'second':
            #     number_of_frames = int(fps * (final_second-initial_second))
            #(3). Final Frame:
            final_frame = min(initial_frame+number_of_frames, total_number_of_frames)

            ### Loop Over Images And Save What Is Needed: ###
            current_step = 0
            frames_saved_so_far = 0;
            while current_step < initial_frame + number_of_frames:
                tic()
                ### Read Frame: ###
                flag_frame_available, current_frame = video_stream.read();
                if flag_frame_available==False:
                    break;


                ### Save Frames If It' Time: ###
                if current_step >= initial_frame:
                    current_frame = crop_tensor(current_frame, crop_size, crop_size)
                    ### Loop Over The Different Noises We Need To Save: ###
                    for noise_index, noise_gain in enumerate(noise_gains):
                        #(1). Add Noise To Frame:
                        current_frame_tensor = torch.Tensor(current_frame).permute([2,0,1]).unsqueeze(0)/255;
                        current_frame_tensor_noisy = noise_RGB_LinearShotNoise_Torch(current_frame_tensor, noise_gain, flag_output_uint8_or_float='float', flag_clip=False)
                        current_frame_noisy = current_frame_tensor_noisy.squeeze(0).permute([1,2,0]).cpu().numpy()
                        current_frame_noisy = current_frame_noisy * 255
                        current_frame_noisy = np.clip(current_frame_noisy,0,255)
                        current_frame_noisy = uint8(current_frame_noisy)
                        #(2). Save Noisy Frame (Format: super_folder -> noise_gain -> specific_movie
                        noisy_image_full_folder = os.path.join(saved_images_super_folder, 'NoiseGain' + str(noise_gain), movie_full_filename.split('.')[0].split('/')[-1] + '_CropSize' + str(crop_size) + '_Frames' + str(initial_frame) + '-' + str(final_frame))
                        path_make_path_if_none_exists(noisy_image_full_folder)
                        save_image_numpy(folder_path=noisy_image_full_folder,
                                         filename=str(frames_saved_so_far).rjust(8,'0') + '.png',
                                         numpy_array=current_frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
                    ### After Adding Noise at different amounts -> we're finished with this frame so uptick counter: ###
                    frames_saved_so_far += 1;

                ### Uptick current_step by 1: ###
                current_step += 1
                toc()





def create_video_from_image_sequence():

    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO\SID_Network\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V30_V1_CenterCrop_DirContex_NoDeform_Video_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR LONG VIDEO RESULTS/results'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO\SID_Network\ALL_DETAILS\clean'

    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V31_V1_OnlyIntensity_DirContexTime_NoDeform_Video_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V31_V1_OnlyIntensity_DirContexTime_NoDeform_Video_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V51_V2_VideoCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V51_V4_VideoCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V51_V3_VideoCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V51_V6_AllDeformable_VideoCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V59_StillsCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V51_V3_VideoCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V51_V6_AllDeformable_VideoCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V50_Still_Images_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V51_Still_Images_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V52_Still_Images_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V53_Still_Images_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V54_Still_Images_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V56_Still_Images_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_0\LONG_VIDEO/UNET_V57_Still_Images_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V59_StillsCorrect_OnlyL1_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V59_StillsCorrect_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V70_V2_StillsCorrect_TWICE\ALL_DETAILS\Concat'

    folder_name_to_place_video = os.path.join(images_folder_name, 'Clean_Movie')
    movie_file_name = os.path.split(images_folder_name)[-1] + '.avi'
    path_make_path_if_none_exists(folder_name_to_place_video)
    network_name = os.path.split(os.path.split(os.path.split(images_folder_name)[0])[0])[-1]
    movie_full_filename = os.path.join(folder_name_to_place_video, network_name + '_' + movie_file_name)

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    fps = 30;
    sample_image_numpy = read_single_image_from_folder(images_folder_name)
    frame_height, frame_width, number_of_channels = sample_image_numpy.shape
    video_writer = cv2.VideoWriter(movie_full_filename, fourcc, fps, (frame_width, frame_height))

    images_list = read_images_from_folder_to_list(images_folder_name, IMG_EXTENSIONS, flag_recursive=False, flag_convert_to_rgb=False)
    for i,current_image in enumerate(images_list):
        video_writer.write(current_image)
        print(i)

    cv2.destroyAllWindows()
    video_writer.release()



def create_concatenated_video_from_several_image_sequences():

    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO\SID_Network\ALL_DETAILS\Concat'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V30_V1_CenterCrop_DirContex_NoDeform_Video_TWICE\ALL_DETAILS\Concat'
    # images_folder_name = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR LONG VIDEO RESULTS/results'
    # images_folder_name = 'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO\SID_Network\ALL_DETAILS\clean'
    images_folder_names = [
        'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO/UNET_V31_V1_OnlyIntensity_DirContexTime_NoDeform_Video_TWICE\ALL_DETAILS\clean',
        'F:\TNR - OFFICIAL TEST IMAGES\Long_Video/NoiseGain_100\LONG_VIDEO\SID_Network\ALL_DETAILS\clean',
        'C:/Users\dkarl\PycharmProjects\dudykarl\TNR LONG VIDEO RESULTS\Original TNR Results',
        ]
    title_list = ['Deep','SID','TNR']

    ### Folder and Movie name of Super-Movie: ###
    super_movie_folder_path = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR LONG VIDEO RESULTS\Concatenated Results'
    # movie_file_name = 'Concatenated_Results' + '.avi'
    movie_file_name = 'Concatenated_Results_Zoom' + '.avi'
    path_make_path_if_none_exists(super_movie_folder_path)
    movie_full_filename = os.path.join(super_movie_folder_path, movie_file_name)
    ### Get Filenames Lists: ###
    image_filenames_lists_list = []
    min_number_of_images = 1e9;
    # min_image_size = 1e9
    min_image_size = 200
    for folder_index, images_folder_name in enumerate(images_folder_names):
        current_folder_image_filenames_list = read_number_of_image_filenames_from_folder(images_folder_name, 400, 100, True, False)
        if min_number_of_images > len(current_folder_image_filenames_list):
            min_number_of_images = len(current_folder_image_filenames_list);
        current_folder_sample_image = read_single_image_from_folder(images_folder_name)
        if min_image_size > min(current_folder_sample_image.shape[0:2]):
            min_image_size = min(current_folder_sample_image.shape[0:2])
        image_filenames_lists_list.append(current_folder_image_filenames_list)
    ### Create Movie Writer: ###
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    fps = 30;
    sample_image_numpy = read_single_image_from_folder(images_folder_names[0])
    frame_height, frame_width, number_of_channels = sample_image_numpy.shape
    video_writer = cv2.VideoWriter(movie_full_filename, fourcc, fps, (min_image_size * len(images_folder_names), min_image_size))

    ### Build Movie: ###
    number_of_folders = len(images_folder_names);
    for image_index in arange(min_number_of_images):
        for folder_index in arange(number_of_folders):
            current_image = read_image_cv2(image_filenames_lists_list[folder_index][image_index], flag_convert_to_rgb=0, flag_normalize_to_float=0)
            current_image = crop_tensor(current_image, min_image_size, min_image_size)
            ### Put Label On Image: ###
            current_title = title_list[folder_index]
            text_size = cv2.getTextSize(current_title, cv2.FONT_HERSHEY_PLAIN, 15, 15)[0]
            c1 = [int(min_image_size / 2), int(0 / 2)]
            # cv2.putText(current_image, current_title, (c1[0], c1[1] + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 12, [225, 255, 255], 18);
            ### Concatenate Images Together: ###
            if folder_index == 0:
                concat_image = current_image;
            else:
                concat_image = np.concatenate((concat_image,current_image),axis=1);
        video_writer.write(concat_image)
        print(image_index)


    # label = 'TNR'
    # current_image = read_image_cv2(image_filenames_lists_list[folder_index][image_index], flag_convert_to_rgb=0, flag_normalize_to_float=0)
    # text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 15, 15)[0]
    # c1 = [int(min_image_size/2), int(0/2)]
    # c2 = [0, min_image_size]
    # c2 = c1[0] + text_size[0] + 3, c1[1] + text_size[1] + 4
    # cv2.putText(current_image, label, (c1[0], c1[1] + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 12, [225, 255, 255], 18);
    # imshow(current_image)


    cv2.destroyAllWindows()
    video_writer.release()



def split_video_into_image_frames():
    ### Set Up Decices: ###
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device_cpu = torch.device('cpu')
    Generator_device = device1
    ### Get Raw File and Video_Stream Object: ###
    # movie_full_filenames = ['G:\Raw Films/Walking_in_Tokyo_Shibuya_at_night.mkv',
    #                        'G:\Raw Films/Driving_Downtown_Chicago_4K_USA.mkv',
    #                        'G:\Raw Films/3HRS_Stunning_Underwater_Footage_Relaxing_Music.mkv',
    #                        'G:\Raw Films/Night_videowalk_in_East_Shinjuku_Tokyo.mkv',
    #                        'G:\Raw Films/Rome_Virtual_Walking_Tour_in_4K_Rome_City_Travel.mkv',
    #                        'G:\Raw Films/Walk_to_the_Kaohsiung_Main_Public_Library_at_night.mkv',
    #                        'G:\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']
    movie_full_filenames = ['G:\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']
    saved_images_super_folder = 'F:\TNR - OFFICIAL TEST IMAGES\Dynamic_Videos'
    if not os.path.exists(saved_images_super_folder):
        os.makedirs(saved_images_super_folder)

    ### Noise Gains To Save: ###
    noise_gains = [0,30,65,100]
    # noise_gains = [100]

    ### Frames to Capture: ###
    # (1). Initial Frame:
    flag_use_initial_frame_or_second = 'second'  # 'frame'/'second'
    initial_frames = 10;
    # initial_seconds = [0*60, 10*60, 20*60, 30*60, 40*60, 50*60, 60*60];
    initial_seconds = [7*60+15];
    # (2). Number of Frams:
    flag_use_final_frame_or_second = 'frame'
    # number_of_frames = fps * 2;
    number_of_seconds = 10
    final_second = 0 * 3600 + 0 * 60 + 5;

    ### Crop Size: ###
    crop_size = 2000;


    ### Loop over movie files and create "dataset" (long sequences when i want to qualitatively compare videos and many short sequences when i want to gather statistics): ###
    for movie_full_filename in movie_full_filenames:
        video_stream = cv2.VideoCapture(movie_full_filename)
        frame_width = int(video_stream.get(3))
        frame_height = int(video_stream.get(4))
        fps = video_stream.get(5)
        total_number_of_frames = int(video_stream.get(7))
        ### Read a few frames for voodoo good luck: ###
        video_stream.read();
        video_stream.read();
        video_stream.read();

        # initial_frames = [0 * 60 * fps, 10 * 60 * fps, 20 * 60 * fps, 30 * 60 * fps, 40 * 60 * fps, 50 * 60 * fps, 60 * 60 * fps]

        for initial_second in initial_seconds:
            initial_frame = int(initial_second * fps)
            number_of_frames = int(number_of_seconds * fps)
            # ### Frames to Capture: ###
            # #(1). Initial Frame:
            # flag_use_initial_frame_or_second = 'second' #'frame'/'second'
            # initial_frame = 10;
            # initial_second = 0*3600 + 30*60 + 2;
            # if flag_use_initial_frame_or_second == 'second':
            #     initial_frame = int(fps * initial_second);
            # #(2). Number of Frams:
            # flag_use_final_frame_or_second = 'frame'
            # number_of_frames = fps*10;
            # final_second = 0*3600 + 0*60 + 5;
            # if flag_use_final_frame_or_second == 'second':
            #     number_of_frames = int(fps * (final_second-initial_second))
            #(3). Final Frame:
            final_frame = min(initial_frame+number_of_frames, total_number_of_frames)

            ### Loop Over Images And Save What Is Needed: ###
            current_step = 0
            frames_saved_so_far = 0;
            while current_step < initial_frame + number_of_frames:
                tic()
                ### Read Frame: ###
                flag_frame_available, current_frame = video_stream.read();
                if flag_frame_available==False:
                    break;


                ### Save Frames If It' Time: ###
                if current_step >= initial_frame:
                    current_frame = crop_tensor(current_frame, crop_size, crop_size)
                    ### Loop Over The Different Noises We Need To Save: ###
                    for noise_index, noise_gain in enumerate(noise_gains):
                        #(1). Add Noise To Frame:
                        current_frame_tensor = torch.Tensor(current_frame).permute([2,0,1]).unsqueeze(0)/255;
                        current_frame_tensor_noisy = noise_RGB_LinearShotNoise_Torch(current_frame_tensor, noise_gain, flag_output_uint8_or_float='float', flag_clip=False)
                        current_frame_noisy = current_frame_tensor_noisy.squeeze(0).permute([1,2,0]).cpu().numpy()
                        current_frame_noisy = current_frame_noisy * 255
                        current_frame_noisy = np.clip(current_frame_noisy,0,255)
                        current_frame_noisy = uint8(current_frame_noisy)
                        #(2). Save Noisy Frame (Format: super_folder -> noise_gain -> specific_movie
                        noisy_image_full_folder = os.path.join(saved_images_super_folder, 'NoiseGain' + str(noise_gain), movie_full_filename.split('.')[0].split('/')[-1] + '_CropSize' + str(crop_size) + '_Frames' + str(initial_frame) + '-' + str(final_frame))
                        path_make_path_if_none_exists(noisy_image_full_folder)
                        save_image_numpy(folder_path=noisy_image_full_folder,
                                         filename=str(frames_saved_so_far).rjust(8,'0') + '.png',
                                         numpy_array=current_frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
                    ### After Adding Noise at different amounts -> we're finished with this frame so uptick counter: ###
                    frames_saved_so_far += 1;

                ### Uptick current_step by 1: ###
                current_step += 1
                toc()
###################################################################################################################################################################################################################################################################################################



def create_dirty_movie_for_presentation():
    movie_full_filename = '/home/mafat/Videos/4K Video Downloader/Walking and flying in Toride, Ibaraki・4K HDR.mkv'
    movie_full_filename = '/home/mafat/DataSets/Example Videos/Beirut.mp4'
    movie_save_filename = 'Berlin_ForPresentation2.avi'

    ### Where To Save Results: ###
    super_folder_to_save_everything_at = '/home/mafat/DataSets/Synthetic_Movies'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)
    movie_save_full_filename = os.path.join(super_folder_to_save_everything_at, movie_save_filename)

    ### Auxiliary Layers: ###
    blur_number_of_channels = 1
    blur_kernel_size = 7
    blur_sigma = 1
    gaussian_blur_layer = Gaussian_Blur_Layer(blur_number_of_channels, blur_kernel_size, blur_sigma)
    average_pooling_layer = nn.AvgPool2d(16, 1)
    downsample_layer = nn.AvgPool2d(2, 2)
    crop_size_vec = [1000, 1000]

    ### Camera Parameters: ### (*). INSTEAD OF ABOVE NOISE PARAMETERS!!!!
    mean_gray_level_per_pixel = [1.5, 1.5]
    electrons_per_gray_level = 14.53
    photons_per_electron = 1  # TODO: i don't think this is the QE but whether we have a photo-multiplier or not....check this out
    gray_levels_per_electron = 1 / electrons_per_gray_level
    electrons_per_photon = 1 / photons_per_electron

    ### Kaya Camera Noise: ###
    Camera_Noise_Folder = os.path.join(datasets_main_folder, '/KAYA_CAMERA_NOISE/noise')
    noise_image_filenames = path_get_files_recursively(Camera_Noise_Folder, '', True)

    ### Video Reader: ###
    video_stream = cv2.VideoCapture(movie_full_filename)
    flag_frame_available, frame = video_stream.read()

    ### Video Writer: ###
    H, W, C = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(movie_save_full_filename, fourcc, 25.0, (crop_size_vec[0] * 2, crop_size_vec[0]))

    number_of_images = 200
    start_frame = 1
    for frame_index in np.arange(start_frame):
        print(frame_index)
        flag_frame_available, original_frame = video_stream.read()

    for frame_index in np.arange(number_of_images):
        print(frame_index)

        ### Read Movie Frame: ###
        flag_frame_available, original_frame = video_stream.read()
        original_frame = original_frame / 255
        original_frame = BW2RGB(RGB2BW(original_frame))
        noisy_frame = original_frame
        noisy_frame = RGB2BW(noisy_frame)

        ### Read Camera Readout-Noise: ###
        camera_noise_frame = read_image_general(noise_image_filenames[frame_index])

        ### Get Non-Uniformity Map: ###
        speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(
            400, crop_size_vec[0], 0, 1, 1)
        NU_map = speckle_pattern_total_intensity
        # NU_map = NU_map.clip(0,1)
        NU_map = to_range(NU_map, 0.7, 1)

        ### Numpy to Torch: ###
        original_frame = numpy_to_torch(original_frame, True)
        noisy_frame = numpy_to_torch(noisy_frame, True)
        NU_map = numpy_to_torch(NU_map, True)
        camera_noise_frame = numpy_to_torch(camera_noise_frame, True)

        ### Blur: ###
        noisy_frame = average_pooling_layer(noisy_frame)

        ### Crop Frames To Same Size: ###
        original_frame = crop_torch_batch(original_frame, crop_size_vec, 'center')
        noisy_frame = crop_torch_batch(noisy_frame, crop_size_vec, 'center')
        camera_noise_frame = crop_torch_batch(camera_noise_frame, crop_size_vec, 'center')
        NU_map = crop_torch_batch(NU_map, crop_size_vec, 'center')

        ### Add Noise: ###
        # (1). Shot Noise:
        input_image_mean = noisy_frame.mean() + 1e-5
        current_mean_gray_level_per_pixel = 2
        noisy_frame = noisy_frame * (current_mean_gray_level_per_pixel) / input_image_mean
        noisy_frame_photons = noisy_frame * electrons_per_gray_level * photons_per_electron
        shot_noise = torch.sqrt(noisy_frame_photons) * torch.randn(*noisy_frame_photons.shape)
        shot_noise = shot_noise * electrons_per_photon * gray_levels_per_electron
        noisy_frame = noisy_frame + shot_noise
        # (2). Kaya Camera Noise:
        noisy_frame = noisy_frame + camera_noise_frame
        # (3). General Synthetic Noise:

        # (4). Normalize:
        noisy_frame = noisy_frame / 10

        ### Add NU: ###
        noisy_frame = noisy_frame * NU_map

        ### Concatenate Original and Noisy: ###
        concatenated_frame = torch.cat([BW2RGB(noisy_frame), original_frame], -1)
        concatenated_frame_numpy = torch_to_numpy(concatenated_frame).squeeze(0)
        concatenated_frame_numpy = (concatenated_frame_numpy * 255).clip(0, 255).astype(np.uint8)

        ### Make Final Video: ###
        video_writer.write(concatenated_frame_numpy)

    video_writer.release()


def create_dataset_with_OpticalFlow_from_existing_dataset():
    ### TVNET IMPORTS: ###
    from TVnet_simple.data.frame_dataset import frame_dataset
    from TVnet_simple.train_options import arguments
    import torch.utils.data as data
    from TVnet_simple.model.network_tvnet import model
    import scipy.io as sio
    from TVnet_simple.utils import *
    import easydict
    from easydict import EasyDict

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device_cpu = torch.device('cpu')

    ### Function Returning TVNet Module Instance: ###
    def get_TVNet_instance(x_input, number_of_iterations=60, number_of_pyramid_scales=3, device=device1,
                           flag_trainable=False):
        args = EasyDict()
        args.frame_dir = 'blabla'  # path to frames
        args.img_save_dir = 'blabla'  # path to storage generated feature maps if needed
        args.n_epocs = 100
        args.n_threads = 1
        args.batch_size = 1;
        args.learning_rate = 1e-4
        args.is_shuffle = False
        args.visualize = False
        # args.data_size = (100,100); #TODO: understand the needed format here!$#@%
        args.zfactor = 0.5  # factor for building the image pyramid
        args.max_nscale = number_of_pyramid_scales;  # max number of scales for image pyramid
        args.n_warps = 1;  # number of warping per scale
        args.n_iters = number_of_iterations;  # max number of iterations for optimization
        args.demo = False;
        ## Don't Change according to moran: ###
        args.tau = 0.25;  # time step
        args.lbda = 0.1;  # weight parameter for the data term
        args.theta = 0.3;  # weight parameter for (u-v)^2

        ### Get default arguments: ###
        B, C, H, W = x_input.shape
        args.batch_size = B  # KILL
        args.demo = False
        args.data_size = [B, C, H, W]
        args.device = x_input.device

        ### Initialize TVNet from model function: ###
        Network = model(args).to(args.device)
        if flag_trainable == False:
            Network = Network.eval()
            for param in Network.parameters():
                param.requires_grad = False

        return Network

    Generator_device = device0
    # Generator_device = device_cpu #GET THIS!!!....ON CPU For batch_size=1 Images IT'S MUCH FASTER!!!!...probably because of the sequential nature of the algorithm!

    ### Optical Flow Module: ###
    TVNet_number_of_iterations = 60;
    TVNet_number_of_pyramid_scales = 5;
    TVNet_layer = None
    delta_x = None
    delta_y = None
    confidence = None  # ==RA update gate?

    def create_noisy_dataset_with_optical_flow_from_existing_dataset_PARALLEL():
        ### Get all image filenames: ###
        images_super_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'
        max_counter = 1;

        ### Initialize TVNet_layer to None: ###
        TVNet_layer = None;

        ### Loop over sub-folders and for each sub-folder: ###
        counter = 0;
        for dirpath, sub_folders_list, filenames in os.walk(images_super_folder):
            if counter > max_counter:
                break;

            ### Read ALL IMAGES for current sub-folder: ###
            for folder_index, current_subfolder in enumerate(sub_folders_list):
                if counter > max_counter:
                    break;
                images = read_images_from_folder_to_numpy(os.path.join(dirpath, current_subfolder), flag_recursive=True)

                ### Turn Images to Tensor and send to wanted device: ###
                tic()
                images = torch.Tensor(np.transpose(images, [0, 3, 1, 2])).to(Generator_device)

                ### get Optical Flow : ###
                if TVNet_layer is None:
                    TVNet_layer = get_TVNet_instance(images, number_of_iterations=TVNet_number_of_iterations,
                                                     number_of_pyramid_scales=TVNet_number_of_pyramid_scales,
                                                     device=Generator_device, flag_trainable=False)
                delta_x, delta_y, x2_warped = TVNet_layer.forward(images[0:-1], images[1:], need_result=True)
                toc('Optical Flow')

                ### Add Zero Shift Maps as start of map because usually i assume the for the first input image i assume the "previous_image" is simply the same firs input image: ###
                delta_x = torch.cat(
                    [torch.zeros(1, 1, delta_x.shape[2], delta_x.shape[3]).to(Generator_device), delta_x], dim=0)
                delta_y = torch.cat(
                    [torch.zeros(1, 1, delta_y.shape[2], delta_y.shape[3]).to(Generator_device), delta_y], dim=0)

                ### Write down Opical Flow Maps into Binary File: ###
                # (1). Initialize Binary File FID:
                fid_filename_X = os.path.join(os.path.join(dirpath, current_subfolder), 'Optical_Flow' + '.bin')
                fid_Write_X = open(fid_filename_X, 'ab')
                np.array(delta_x.shape[2]).tofile(fid_Write_X)  # Height
                np.array(delta_x.shape[3]).tofile(fid_Write_X)  # Width
                np.array(2).tofile(fid_Write_X)  # Number_of_channels=2 (u,v)
                # (2). Write Maps to FID:
                #   (2.1). Write in form [B,2,H,W] all at once:
                numpy_array_to_write = torch.cat([delta_x, delta_y], dim=1).cpu().numpy()
                #   (2.2). Write in form [1,2,H,W] sequentially:
                # TODO: IMPLEMENT!
                #   (2.3). Write Optical Flow Maps To Binary File:
                numpy_array_to_write.tofile(fid_Write_X)
                # (3). Release FID:
                fid_Write_X.close()
                ### Uptick counter by 1: ###
                counter += 1;

    def create_noisy_dataset_with_optical_flow_from_existing_dataset():
        ### Get all image filenames: ###
        images_super_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'
        max_counter = 10;
        # image_filenames_list = get_image_filenames_from_folder(images_super_folder, flag_recursive=True)

        ### Initialize TVNet_layer to None: ###
        TVNet_layer = None;

        ### Loop over sub-folders and for each sub-folder: ###
        counter = 0;
        for dirpath, dn, filenames in os.walk(images_super_folder):
            if counter > max_counter:
                break
            for image_index, current_filename in enumerate(sorted(filenames)):
                if counter > max_counter:
                    break
                ### Get image full filename & Read Image: ###
                full_filename = os.path.join(dirpath, current_filename);
                print(full_filename)
                current_image = read_image_torch(full_filename, flag_convert_to_rgb=1, flag_normalize_to_float=1)

                tic()
                ### Put current image in wanted device (maybe going to gpu will be worth it because it will be that much faster?...maybe i should load all images together into a batch format...send them to gpu and then do it?): ###
                current_image = current_image.to(Generator_device)

                ### If this is not the first image then we can compare it to the previous one using Optical Flow: ###
                if image_index > 0:
                    if TVNet_layer is None:
                        TVNet_layer = get_TVNet_instance(current_image, number_of_iterations=TVNet_number_of_iterations,
                                                         number_of_pyramid_scales=TVNet_number_of_pyramid_scales,
                                                         device=Generator_device, flag_trainable=False)
                    delta_x, delta_y, x2_warped = TVNet_layer.forward(current_image, previous_image, need_result=True)
                toc('Optical Flow')

                ### Assign previous image with current image: ###
                previous_image = current_image;
                counter += 1;








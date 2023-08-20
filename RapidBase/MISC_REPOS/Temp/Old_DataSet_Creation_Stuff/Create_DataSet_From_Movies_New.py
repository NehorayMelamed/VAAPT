from RapidBase.import_all import *


def split_video_into_constant_time_chunks(movie_full_filenames=''):
    ### Original Movie: ###
    movie_full_filenames = ['/home/mafat/Videos/4K Video Downloader/Walking and flying in Toride, Ibaraki・4K HDR.mkv']

    ### Where To Save Results: ###
    super_folder_to_save_everything_at = '/home/mafat/DataSets/Youtube_Movie_DataSets'
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



# split_video_into_constant_time_chunks()
# split_videos_in_folders_into_proper_form_crops()
# transform_videos_dataset_from_RGB_to_BW()


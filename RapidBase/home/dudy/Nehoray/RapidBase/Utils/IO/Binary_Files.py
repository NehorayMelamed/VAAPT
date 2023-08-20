
import numpy as np

def initialize_binary_file_reader(f):
    f.close()
    f = open(f.name, 'rb')
    return f

def read_frames_from_binary_file_stream(f, number_of_frames_to_read=-1, number_of_frames_to_skip=0, params=None):
    ### Get parameters from params dict: ###
    roi = params['roi']
    utype = params['utype']

    ### Read Frames: ###
    # roi = [160,640]
    Mov = np.fromfile(f, dtype=utype, count=number_of_frames_to_read * roi[0] * roi[1], offset=number_of_frames_to_skip*roi[0]*roi[1]*2)
    Movie_len = np.int(len(Mov) / roi[0] / roi[1])
    number_of_elements = Movie_len * roi[0] * roi[1]
    Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])
    Mov = Mov[:,2:,2:]  #Get rid of bad frame

    # bla = torch.tensor(Mov.astype(np.float)).unsqueeze(1).float()
    # bla = scale_array_stretch_hist(bla)
    # imshow_torch_video(bla, FPS=50, frame_stride=5)
    return Mov

import numpy as np
import cv2
def read_frames_from_binary_file_stream2(f, number_of_frames_to_read=-1, number_of_frames_to_skip=0, roi=(512,8192), utype=np.uint16):

    ### Read Frames: ###
    # roi = [160,640]
    Mov = np.fromfile(f, dtype=utype, count=number_of_frames_to_read * roi[0] * roi[1], offset=number_of_frames_to_skip*roi[0]*roi[1]*2)
    Movie_len = np.int(len(Mov) / roi[0] / roi[1])
    number_of_elements = Movie_len * roi[0] * roi[1]
    Mov = np.reshape(Mov[0:number_of_elements], [Movie_len, roi[0], roi[1]])
    Mov = Mov[:, 2:, 2:]  #Get rid of bad frame

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
            video_writer = cv2.VideoWriter(video_name, 0, FPS, (W, H))
            current_frame, (q1, q2) = scale_array_stretch_hist(current_frame, flag_return_quantile=True)
        else:
            current_frame = scale_array_from_range(current_frame.clip(q1, q2),
                                                   min_max_values_to_clip=(q1, q2),
                                                   min_max_values_to_scale_to=(0, 1))
        current_frame = numpy_array_to_video_ready(current_frame)
        # current_frame = cv2.putText(img=current_frame,
        #                             text=str(count),
        #                             org=(W//2, 30),
        #                             fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        #                             fontScale=1,
        #                             color=(0, 255, 0),
        #                             thickness=3)
        ### If done reading then stop: ###
        if current_frame.shape[0] == 0:
            flag_keep_going = False

        ### Write frame down: ###
        video_writer.write(current_frame)
        count = count + 1
        print(count)
    video_writer.release()

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




# def turn_flir_files_into_Bin_and_avi_files():
#     flir_filenames_list = []
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/0_640_512_800fps_10000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/1_640_512_800fps_10000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/2_640_512_800fps_10000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/3_640_512_800fps_10000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/4_640_512_800fps_10000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/5_640_512_800fps_10000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/6_640_512_500fps_20000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/7_640_512_500fps_20000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/8_640_512_500fps_20000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/9_640_512_500fps_20000frames_1000_meter_flir')
#     # flir_filenames_list.append('/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/22-06-2022 - SCD + FLIR/6.22_exp/10_640_512_500fps_20000frames_1000_meter_flir')
#
#     # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\06_07_2022_FLIR\converted_bin')
#     # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\22-06-2022 - SCD + FLIR\flir\6.22_exp')
#     # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\22-06-2022 - SCD + FLIR\flir\6.22_exp\individual_images_bin_files')
#     # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\29.06.22_exp_bin\individual_images_bin_files')
#     # flir_filenames_list = path_get_folder_names(r'E:\Quickshot\13.07.22_bin\individual_images_bin_files')
#     flir_filenames_list = path_get_folder_names(r'E:\Quickshot\19_7_22_flir_exp_Bin\individual_images_bin_files')
#
#     ### Get (H,W) For The Different Files: ###
#     H, W = 512, 640
#
#     # ### Change Filenames rjust: ###
#     # for flir_experiment_index in np.arange(len(flir_filenames_list)):
#     #     ### Get Current Filenames: ###
#     #     flir_files_path = flir_filenames_list[flir_experiment_index]
#     #     filenames_list = get_filenames_from_folder(flir_files_path)
#     #     ### Loop Over Individual Images: ###
#     #     for i in np.arange(len(filenames_list)):
#     #         current_full_filename = filenames_list[i]
#     #         current_folder = os.path.split(current_full_filename)[0]
#     #         current_filename = os.path.split(current_full_filename)[1]
#     #         current_number_string = os.path.splitext(current_filename)[0]
#     #         new_number_string = string_rjust(np.int(current_number_string), 6)
#     #         new_filename = new_number_string + '.Bin'
#     #         new_full_filename = os.path.join(current_folder, new_filename)
#     #         os.rename(current_full_filename, new_full_filename)
#
#     ### Write Down Movie .avi File: ###
#     for flir_experiment_index in np.arange(0, len(flir_filenames_list)):
#         ### Get Current Filenames: ###
#         flir_files_path = flir_filenames_list[flir_experiment_index]
#         filenames_list = get_filenames_from_folder(flir_files_path)
#
#         ### Initialize 1 Big .Bin File: ###
#         new_folder = flir_files_path.replace('individual_images_bin_files', 'one_big_bin_file')
#         path_create_path_if_none_exists(new_folder)
#         folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(filenames_list[0])
#         new_folder = folder.replace('individual_images_bin_files', 'one_big_bin_file')
#         path_create_path_if_none_exists(new_folder)
#         specific_folder_name = os.path.split(folder)[1]
#         new_filename = os.path.join(new_folder, specific_folder_name + '.Bin')
#
#         # ### Initialize 1 Big .Bin File ###
#         # new_filename = os.path.split(flir_files_path)[-1] + '.Bin'
#         # new_filename = os.path.join(os.path.split(flir_files_path)[0], new_filename)
#         # ### TEMP: ###
#         # new_filename = new_filename.replace('individual_images_bin_files', 'one_big_bin_file')
#         # folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(new_filename)
#         # new_folder = os.path.join(folder, filename_without_extension)
#         # new_filename = os.path.join(new_folder, filename)
#         # path_create_path_if_none_exists(new_folder)
#
#         ### Initialize Binary File Writter: ###
#         f = open(new_filename, 'a')
#
#         ### Loop Over Movie Filenames/Frames: ###
#         # movie_filename = os.path.split(flir_files_path)[-1] + '.avi'
#         # movie_filename = os.path.join(os.path.split(flir_files_path)[0], movie_filename)
#         # results_folder = os.path.join(new_folder, 'Results')
#         # path_create_path_if_none_exists(results_folder)
#
#         results_folder = os.path.join(new_folder, 'Results')
#         path_create_path_if_none_exists(results_folder)
#         movie_filename = os.path.join(results_folder, 'Original_Movie.avi')
#         video_object = cv2.VideoWriter(movie_filename, 0, 10, (W, H))
#         for i in np.arange(len(filenames_list)):
#             print(i)
#             ### Get Current Image: ###
#             current_full_filename = filenames_list[i]
#             scene_infile = open(current_full_filename, 'rb')
#             scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
#             # scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
#             image = Image.frombuffer("I", [W, H],
#                                      scene_image_array.astype('I'),
#                                      'raw', 'I', 0, 1)
#             image = np.array(image).astype(np.uint16)
#
#             ### Stretch Image And Write To .avi Movie: ###
#             input_tensor = torch.tensor(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
#             if i == 0:
#                 input_tensor, (q1, q2) = scale_array_stretch_hist(input_tensor, (0.01, 0.999), flag_return_quantile=True)
#             input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
#             input_tensor_to_video = torch_to_numpy_video_ready(input_tensor).squeeze(0)
#             video_object.write(input_tensor_to_video)
#
#             ### Write Image To .Bin File: ###
#             image.tofile(f)
#
#     # ### Read Binary File Images For Testing: ###
#     # f = open(new_filename, 'rb')
#     # H, W = 512, 640
#     # T = 1000
#     # scene_image_array = np.fromfile(f, dtype=np.uint16, count=W * H * T)
#     # scene_image_array = np.reshape(scene_image_array, [T, H, W]).astype(np.float)
#     # input_tensor = torch.tensor(scene_image_array).unsqueeze(1)
#     # bla, (q1, q2) = scale_array_stretch_hist(input_tensor[0], (0.01, 0.999), flag_return_quantile=True)
#     # input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
#     # imshow_torch_video(input_tensor, FPS=50, frame_stride=15)
#
#
# from RapidBase.import_all import *
# def turn_png_files_into_big_bin_file():
#     png_folders_list = path_get_folder_names(r'E:\Palantir\famous_movie\individual_images')
#
#     # ### Change Filenames rjust: ###
#     # for flir_experiment_index in np.arange(len(flir_filenames_list)):
#     #     ### Get Current Filenames: ###
#     #     flir_files_path = flir_filenames_list[flir_experiment_index]
#     #     filenames_list = get_filenames_from_folder(flir_files_path)
#     #     ### Loop Over Individual Images: ###
#     #     for i in np.arange(len(filenames_list)):
#     #         current_full_filename = filenames_list[i]
#     #         current_folder = os.path.split(current_full_filename)[0]
#     #         current_filename = os.path.split(current_full_filename)[1]
#     #         current_number_string = os.path.splitext(current_filename)[0]
#     #         new_number_string = string_rjust(np.int(current_number_string), 6)
#     #         new_filename = new_number_string + '.Bin'
#     #         new_full_filename = os.path.join(current_folder, new_filename)
#     #         os.rename(current_full_filename, new_full_filename)
#
#     ### Write Down Movie .avi File: ###
#     for png_folder_index in np.arange(0, len(png_folders_list)):
#         ### Get Current Filenames: ###
#         png_files_path = png_folders_list[png_folder_index]
#         filenames_list = get_filenames_from_folder(png_files_path)
#
#         ### Initialize 1 Big .Bin File: ###
#         new_folder = png_files_path.replace('individual_images', 'one_big_bin_file')
#         path_create_path_if_none_exists(new_folder)
#         folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(filenames_list[0])
#         new_folder = folder.replace('individual_images', 'one_big_bin_file')
#         path_create_path_if_none_exists(new_folder)
#         specific_folder_name = os.path.split(folder)[1]
#         new_filename = os.path.join(new_folder, specific_folder_name + '.Bin')
#
#         ### Initialize Binary File Writter: ###
#         f = open(new_filename, 'a')
#
#         ### Initialize Movie: ###
#         #(1). Get sample image for dimensions:
#         image_sample = read_image_cv2(filenames_list[0])
#         H,W,C = image_sample.shape
#         #(2). initial video writer in the appropriate file/folder format for QS-Palantir:
#         results_folder = os.path.join(new_folder, 'Results')
#         path_create_path_if_none_exists(results_folder)
#         movie_filename = os.path.join(results_folder, 'Original_Movie.avi')
#         video_object = cv2.VideoWriter(movie_filename, 0, 10, (880, 320))
#
#         ### Loop over the different images and write to .bin and .avi files: ###
#         for i in np.arange(len(filenames_list)):
#         # for i in np.arange(10):
#             print(i)
#             ### Get Current Image: ###
#             current_full_filename = filenames_list[i]
#             image = RGB2BW(read_image_cv2(current_full_filename).astype(np.uint8))
#
#             ### Center crop because these images are from video stabilization and therefore outer crop frame size changes and i want consistency: ###
#             image = crop_tensor(image, (320, 880))
#
#
#             # ### Stretch Image And Write To .avi Movie: ###
#             # input_tensor = torch.tensor(image.astype(np.float32)).permute([2,0,1]).unsqueeze(0)
#             # if i == 0:
#             #     _, (q1, q2) = scale_array_stretch_hist(input_tensor, (0.01, 0.999), flag_return_quantile=True)
#             # if i == 500:
#             #     1
#             # input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2), (0,255))
#             # input_tensor_to_video = torch_to_numpy_video_ready(input_tensor).squeeze(0)
#             # video_object.write(input_tensor_to_video)
#
#             ### Write Image To .Bin File: ###
#             image.astype(np.uint16).tofile(f)
#
#     # ### Read Binary File Images For Testing: ###
#     # f = open(new_filename, 'rb')
#     # H, W = 320, 920
#     # T = 200
#     # scene_image_array = np.fromfile(f, dtype=np.uint8, count=W * H * T)
#     # scene_image_array = np.reshape(scene_image_array, [T, H, W]).astype(float)
#     # input_tensor = torch.tensor(scene_image_array).unsqueeze(1)
#     # bla, (q1, q2) = scale_array_stretch_hist(input_tensor[0], (0.01, 0.999), flag_return_quantile=True)
#     # input_tensor = scale_array_from_range(input_tensor.clamp(q1, q2), (q1, q2))
#     # imshow_torch_video(input_tensor, FPS=50, frame_stride=5)
#     # # imshow_torch_video(input_tensor, FPS=1, frame_stride=1)
#
# # turn_png_files_into_big_bin_file()
#




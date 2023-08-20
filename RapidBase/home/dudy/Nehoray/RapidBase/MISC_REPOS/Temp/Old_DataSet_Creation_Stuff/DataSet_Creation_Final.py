from RapidBase.import_all import *
from RapidBase.Utils.Classical_DSP.Add_Noise import *

movie_full_filename = r'/home/mafat/DataSets/Example Videos/Beirut.mp4'
movie_name = 'Beirut'
video_reader = cv2.VideoCapture(movie_full_filename)
flag_frame_available, current_frame = video_reader.read()
H, W, C = current_frame.shape
final_noisy_images_folder_name_start = r'/home/mafat/DataSets/Example Videos/Beirut_RGB'


#########################   Script to create noisy movie from clean RGB movie: ##################
additive_noise_PSNR = [np.inf]

fourcc = cv2.VideoWriter_fourcc(*'MP42')
# cv2.VideoWriter_fourcc('M','J','P','G')
image_brightness_reduction_factor = 0.9
image_brightness_reduction_factor_delta = 1 - image_brightness_reduction_factor

for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):

    video_reader = cv2.VideoCapture(movie_full_filename)
    flag_frame_available, current_frame = video_reader.read()
    H, W, C = current_frame.shape
    final_movie_full_filename = os.path.join(final_noisy_images_folder_name_start + '_' + str(PSNR), movie_name + '.avi')
    final_noisy_images_folder = os.path.join(final_noisy_images_folder_name_start + '_' + str(PSNR))
    path_make_path_if_none_exists(final_noisy_images_folder)
    video_writer = cv2.VideoWriter(final_movie_full_filename, fourcc, 25.0, (W, H))
    frame_counter = 0
    while flag_frame_available:
        flag_frame_available, current_frame = video_reader.read()
        if flag_frame_available==False:
            break
        ### Convert to Appropriate Colors: ###
        # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

        ### Convert Range from [0,1] to [0.1,0.9]: ###
        current_frame = current_frame/255
        # current_frame = current_frame * (1-2*image_brightness_reduction_factor_delta) + image_brightness_reduction_factor_delta
        # current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0,
        #                                          additive_noise_SNR=additive_noise_PSNR[PSNR_counter],
        #                                          flag_input_normalized=True)
        current_frame_noisy = current_frame + 1 / sqrt(additive_noise_PSNR[PSNR_counter]) * np.random.randn(H, W, C)
        current_frame_noisy = current_frame_noisy * 255
        current_frame_noisy = current_frame_noisy.clip(0, 255)
        current_frame_noisy = current_frame_noisy.astype('uint8')

        # current_frame_noisy = np.expand_dims(current_frame_noisy, -1)
        # current_frame_noisy = np.concatenate((current_frame_noisy,current_frame_noisy,current_frame_noisy), -1)
        print(current_frame.shape)
        video_writer.write(current_frame_noisy)
        save_image_numpy(final_noisy_images_folder, string_rjust(frame_counter, 4) + '.png', current_frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
        frame_counter += 1
### Release Video Writer: ###
video_writer.release()
video_reader.release()





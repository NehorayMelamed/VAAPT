from RapidBase.import_all import *
from RapidBase.Utils.Classical_DSP.Add_Noise import *

#########################   Make Still Images For Official Test DataSet: ##################
### Still Images: ###
clean_still_images_folder = r'/home/mafat/DataSets/DataSets/Div2K/DIV2K_train_HR_BW'
super_shifts_folder = r'/home/mafat/DataSets/DataSets/Div2K/Shifted_Images/Div2k_validation_set_BW_video'
super_blur_folder = r'/home/mafat/DataSets/DataSets/Div2K/Shifted_Images_Blur/Div2k_validation_set_BW_video'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
min_crop_size = 1300
crop_size_x = 1000
crop_size_y = 1000
number_of_images_to_generate = 10
number_of_images_to_create_blur_per_pixel = 3
image_brightness_reduction_factor = 0.9
crop_size = np.inf
noise_gains = [0, 30, 65, 100]
additive_noise_PSNR = [np.inf, 10, 5, 4, 3, 2, 1]
max_shifts_in_pixels = [20, 10, 5, 2, 1, 0]
number_of_time_steps = 10

for max_shift_in_pixels in max_shifts_in_pixels:
    with_blur_shift_folder = os.path.join(super_shifts_folder, 'Shift_' + str(max_shift_in_pixels))
    no_blur_shift_folder = os.path.join(super_blur_folder, 'Shift_' + str(max_shift_in_pixels))

    ### Define the dataset object with the new relevant imgaug_parameters: ###
    train_dataset = Dataset_Images_From_Folder(clean_still_images_folder,
                                               transform=None,
                                               image_loader=ImageLoaderCV,
                                               max_number_of_images=number_of_images_to_generate,
                                               crop_size=np.inf,
                                               flag_to_RAM=True,
                                               flag_recursive=False,
                                               flag_normalize_by_255=False,
                                               flag_crop_mode='center',
                                               flag_explicitely_make_tensor=False,
                                               allowed_extentions=IMG_EXTENSIONS,
                                               flag_base_transform=True,
                                               flag_turbulence_transform=False,
                                               Cn2=5e-13)

    for image_index in arange(len(train_dataset)):

        for time_step in arange(number_of_time_steps):
            ### Read the image (and randomise new shift!): ###
            current_frame = train_dataset[image_index] / 255
            current_frame = current_frame * image_brightness_reduction_factor

            ### Blur: ###
            shiftx = get_random_number_in_range(-max_shift_in_pixels, max_shift_in_pixels)
            shifty = get_random_number_in_range(-max_shift_in_pixels, max_shift_in_pixels)
            current_frame_with_blur = blur_image_motion_blur(current_frame, shiftx, shifty, int(max_shift_in_pixels * number_of_images_to_create_blur_per_pixel + 1))
            current_frame_no_blur = shift_matrix_subpixel(current_frame, shiftx, shifty)

            ### For Every Image Get Noisy Image (For Every Noise Gain): ###
            for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
                ### Create new folder for this particular Noise: ###
                noise_gain_folder_with_blur = os.path.join(with_blur_shift_folder, 'PSNR_' + str(PSNR))
                noise_gain_folder_no_blur = os.path.join(no_blur_shift_folder, 'PSNR_' + str(PSNR))
                # image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames_list[image_index].split('.')[0].split('\\')[-1])
                image_within_noise_folder_with_blur = os.path.join(noise_gain_folder_with_blur,
                                                                   string_rjust(image_index, 4))
                image_within_noise_folder_no_blur = os.path.join(noise_gain_folder_no_blur,
                                                                 string_rjust(image_index, 4))
                path_make_directory_from_path(noise_gain_folder_with_blur)
                path_make_directory_from_path(noise_gain_folder_no_blur)
                path_make_directory_from_path(image_within_noise_folder_with_blur)
                path_make_directory_from_path(image_within_noise_folder_no_blur)

                current_frame_noisy_with_blur = add_noise_to_image(current_frame_with_blur, shot_noise_adhok_gain=0,
                                                                   additive_noise_SNR=additive_noise_PSNR[PSNR_counter],
                                                                   flag_input_normalized=True)
                numpy_array_with_blur = current_frame_noisy_with_blur * 255
                numpy_array_with_blur = numpy_array_with_blur.clip(0, 255)
                numpy_array_with_blur = numpy_array_with_blur.astype('uint8')
                numpy_array_with_blur = numpy_array_with_blur[0:crop_size_y, 0:crop_size_x, :]
                save_image_numpy(image_within_noise_folder_with_blur, str(time_step).rjust(4, '0') + '.png',
                                 numpy_array_with_blur, flag_convert_bgr2rgb=False, flag_scale=False)
                save_image_mat(image_within_noise_folder_with_blur, str(time_step).rjust(4, '0') + '.mat',
                               current_frame_noisy_with_blur, 'center_frame_noisy')

                current_frame_noisy_no_blur = add_noise_to_image(current_frame_no_blur, shot_noise_adhok_gain=0,
                                                                 additive_noise_SNR=additive_noise_PSNR[PSNR_counter],
                                                                 flag_input_normalized=True)
                numpy_array_no_blur = current_frame_noisy_no_blur * 255
                numpy_array_no_blur = numpy_array_no_blur.clip(0, 255)
                numpy_array_no_blur = numpy_array_no_blur.astype('uint8')
                numpy_array_no_blur = numpy_array_no_blur[0:crop_size_y, 0:crop_size_x, :]
                save_image_numpy(image_within_noise_folder_no_blur, str(time_step).rjust(4, '0') + '.png',
                                 numpy_array_no_blur, flag_convert_bgr2rgb=False, flag_scale=False)
                save_image_mat(image_within_noise_folder_no_blur, str(time_step).rjust(4, '0') + '.mat',
                               current_frame_noisy_no_blur, 'center_frame_noisy')

#
# ### For Every Image Get Noisy Image (For Every Noise Gain): ###
# for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
#     ### Create new folder for this particular Noise: ###
#     PSNR_folder = os.path.join(still_images_folder, 'Still_Images', 'PSNR_' + str(PSNR))
#     path_make_directory_from_path(PSNR_folder)
#
#     for image_index in arange(len(train_dataset)):
#         ### Create a new folder for this particular image: ###
#         # image_within_noise_folder = os.path.join(PSNR_folder, train_dataset.image_filenames_list[image_index].split('.')[0].split('\\')[-1])
#         image_within_noise_folder = os.path.join(PSNR_folder, validation_set_super_name, string_rjust(image_index, 4))
#         path_make_directory_from_path(image_within_noise_folder)
#
#         ### Read the image: ###
#         current_frame = train_dataset[image_index]
#         current_frame = current_frame/255
#         # current_frame = numpy_from_torch_to_numpy_convention(current_frame)
#         current_frame = current_frame * image_brightness_reduction_factor
#         # current_frame = auxiliary_take_care_of_input_image(current_frame, flag_input_normalized=False)
#         print('image index ' + str(image_index))
#         for time_step in arange(number_of_time_steps):
#             current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0, additive_noise_SNR=additive_noise_PSNR[PSNR_counter], flag_input_normalized=True)
#             numpy_array = current_frame_noisy * 255
#             numpy_array = numpy_array.clip(0, 255)
#             numpy_array = numpy_array.astype('uint8')
#             save_image_numpy(image_within_noise_folder, str(time_step).rjust(4,'0') + '.png', numpy_array, flag_convert_bgr2rgb=False, flag_scale=False)
#             save_image_mat(image_within_noise_folder, str(time_step).rjust(4,'0') + '.mat', current_frame_noisy, 'center_frame_noisy')


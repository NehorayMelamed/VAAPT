from RapidBase.import_all import *
from RapidBase.Utils.Classical_DSP.Add_Noise import *


#########################   Make Semi-Dynamic Affine Deformation Images For Official Test DataSet: ##################
### Still Images: ###
clean_still_images_folder = r'/home/mafat/DataSets/DataSets/Div2K/DIV2K_train_HR_BW'
validation_set_super_name = 'Div2k_validation_set_BW_video'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
max_shift_in_pixels = 0
min_crop_size = 1300
crop_size = np.inf
additive_noise_PSNR = [np.inf,100,10,5,4,3,2,1]
number_of_time_steps = 10

### Loop Variables: ###
additive_noise_PSNR = [100,10,5,4,3,2,1]
max_shifts_in_pixels = [0,5,20]
max_scales_deltas = [0,0.05]
max_degrees = [0,5]
max_perspective_change_factors = [0,0.05]

image_brightness_reduction_factor = 0.9


for max_shift_in_pixels in max_shifts_in_pixels:
    for max_scale in max_scales_deltas:
        for max_degree in max_degrees:
            for max_perspective_change_factor in max_perspective_change_factors:

                dot_change_char = '#'
                current_sub_folder_name = '_Shift_' + str(max_shift_in_pixels).replace('.',dot_change_char) + \
                                          '_Degrees_' + str(max_degree).replace('.',dot_change_char) + \
                                          '_Scale_' + str(max_scale).replace('.',dot_change_char) + \
                                          '_Perspective_' + str(max_perspective_change_factor).replace('.',dot_change_char)
                current_affine_folder = os.path.join(validation_set_super_name, current_sub_folder_name)
                path_make_directory_from_path(current_affine_folder)

                ### IMGAUG: ###
                imgaug_parameters = get_IMGAUG_parameters()
                # Affine:
                imgaug_parameters['flag_affine_transform'] = True
                imgaug_parameters['affine_scale'] = (1-max_scale,1+max_scale)
                imgaug_parameters['affine_translation_percent'] = None
                imgaug_parameters['affine_translation_number_of_pixels'] = {"x": (-max_shift_in_pixels, max_shift_in_pixels), "y": (-max_shift_in_pixels, max_shift_in_pixels)}
                imgaug_parameters['affine_rotation_degrees'] = (-max_degree,max_degree)
                imgaug_parameters['affine_shear_degrees'] = 0
                imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
                imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
                imgaug_parameters['probability_of_affine_transform'] = 1
                #Perspective:
                imgaug_parameters['flag_perspective_transform'] = True
                imgaug_parameters['flag_perspective_transform_keep_size'] = True
                imgaug_parameters['perspective_transform_scale'] = (0.0, max_perspective_change_factor)
                imgaug_parameters['probability_of_perspective_transform'] = 1
                ### Get Augmenter: ###
                imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)


                ### Define the dataset object with the new relevant imgaug_parameters: ###
                train_dataset = Dataset_Images_From_Folder(clean_still_images_folder,
                                                           transform=imgaug_transforms,
                                                           image_loader=ImageLoaderCV,
                                                           max_number_of_images=10,
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


                tic()
                for image_index in arange(len(train_dataset)):

                    for time_step in arange(number_of_time_steps):
                        ### Read the image (and randomise new shift!): ###
                        current_frame = train_dataset[image_index]
                        current_frame = numpy_from_torch_to_numpy_convention(current_frame)
                        current_frame = current_frame * image_brightness_reduction_factor
                        current_frame = auxiliary_take_care_of_input_image(current_frame, flag_input_normalized=False)

                        ### For Every Image Get Noisy Image (For Every Noise Gain): ###
                        for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
                            ### Create new folder for this particular Noise: ###
                            noise_gain_folder = os.path.join(current_affine_folder, 'PSNR_' + str(PSNR))
                            image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames_list[image_index].split('.')[0].split('\\')[-1])
                            path_make_directory_from_path(image_within_noise_folder)

                            current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0,
                                                                     additive_noise_SNR=additive_noise_PSNR[
                                                                         PSNR_counter],
                                                                     flag_input_normalized=True)
                            numpy_array = current_frame_noisy * 255
                            numpy_array = numpy_array.clip(0, 255)
                            numpy_array = numpy_array.astype('uint8')
                            save_image_numpy(image_within_noise_folder, str(time_step).rjust(3, '0') + '.png', numpy_array, flag_convert_bgr2rgb=True, flag_scale=False)
                toc()






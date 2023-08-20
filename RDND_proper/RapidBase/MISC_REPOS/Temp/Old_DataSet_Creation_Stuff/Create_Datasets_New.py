from RapidBase.import_all import *

from torch.utils.data import Dataset, DataLoader


import RapidBase.TrainingCore
import numpy as np
import RapidBase.TrainingCore.Basic_DataSets
from RapidBase.TrainingCore.Basic_DataSets import *
from RapidBase.Utils.Classical_DSP.Add_Noise import *



#########################   Script to create BW & noisy movie from clean RGB movie: ##################
additive_noise_PSNR = [100,10,5,4,3,2,1]

fourcc = cv2.VideoWriter_fourcc(*'MP42')
# cv2.VideoWriter_fourcc('M','J','P','G')
image_brightness_reduction_factor = 0.9
image_brightness_reduction_factor_delta = 1 - image_brightness_reduction_factor

for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
    movie_full_filename = r'C:\Users\dudyk\Pictures\Video Projects/Shibuya.mp4'
    video_reader = cv2.VideoCapture(movie_full_filename)
    flag_frame_available, current_frame = video_reader.read()
    H, W, C = current_frame.shape
    final_movie_full_filename = r'C:\Users\dudyk\Pictures\Video Projects/Shibuya_BW' + '_' + str(PSNR) + '.mkv'
    final_noisy_images_folder = r'C:\Users\dudyk\Pictures\Video Projects/Shibuya_BW' + '_' + str(PSNR)
    path_make_path_if_none_exists(final_noisy_images_folder)
    video_writer = cv2.VideoWriter(final_movie_full_filename, fourcc, 25.0, (W, H))
    frame_counter = 0
    while flag_frame_available:
        flag_frame_available, current_frame = video_reader.read()
        if flag_frame_available==False:
            break
        ### Convert to BW: ###
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

        ### Convert Range from [0,1] to [0.1,0.9]: ###
        current_frame = auxiliary_take_care_of_input_image(current_frame, flag_input_normalized=False)
        # current_frame = current_frame * (1-2*image_brightness_reduction_factor_delta) + image_brightness_reduction_factor_delta
        current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0,
                                                 additive_noise_SNR=additive_noise_PSNR[PSNR_counter],
                                                 flag_input_normalized=True)
        current_frame_noisy = current_frame_noisy * 255
        current_frame_noisy = current_frame_noisy.clip(0, 255)
        current_frame_noisy = current_frame_noisy.astype('uint8')

        current_frame_noisy = np.expand_dims(current_frame_noisy, -1)
        current_frame_noisy = np.concatenate((current_frame_noisy,current_frame_noisy,current_frame_noisy), -1)
        print(current_frame.shape)
        video_writer.write(current_frame_noisy)
        save_image_numpy(final_noisy_images_folder, string_rjust(frame_counter, 4) + '.png', current_frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
        frame_counter += 1
### Release Video Writer: ###
video_writer.release()
video_reader.release()







#########################   Script to create BW datasets for denoising FROM IMAGES: ##################
main_path = r'C:\DataSets\Div2K\DIV2K_train_HR'
final_path = r'C:\DataSets\Div2K\DIV2K_train_HR_BW'
final_path_crops = r'C:\DataSets\Div2K\DIV2K_train_HR_BW_Crops'
# dataset_object = ImageFolderRecursive_MaxNumberOfImages(main_path, transform=None)
dataset_object = Dataset_Images_From_Folder(final_path_crops, transform=None)
for i, full_filename in enumerate(dataset_object.imgs):
    ### Write converted RGB->BW image to disk: ###
    original_image = read_image_cv2(full_filename, 1, flag_normalize_to_float=0)
    BW_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(os.path.join(final_path, string_rjust(i, 4)+'.png'), BW_image)

    ### Write all crops of BW image to disk: ###
    crops_so_far = len(path_get_all_filenames_from_folder(final_path_crops))
    crops_list = get_list_of_crops_numpy(BW_image, flag_get_all_crops=True, number_of_random_crops=None, crop_size_x=128, crop_size_y=128)
    for j in np.arange(len(crops_list)):
        cv2.imwrite(os.path.join(final_path_crops, string_rjust(j+crops_so_far, 7)+'.png'), crops_list[j])











#########################   Example of train dataset which loads single image and affine shifts it and returns multiple images: ##################
# main_path = r'C:\DataSets\Div2K\DIV2K_train_HR_BW'
# ### IMGAUG: ###
# imgaug_parameters = get_IMGAUG_parameters()
# max_shift_in_pixels = 100
# # Affine:
# imgaug_parameters['flag_affine_transform'] = True
# imgaug_parameters['affine_scale'] = 1
# imgaug_parameters['affine_translation_percent'] = None
# imgaug_parameters['affine_translation_number_of_pixels'] = {"x": (-max_shift_in_pixels, max_shift_in_pixels), "y": (-max_shift_in_pixels, max_shift_in_pixels)}
# imgaug_parameters['affine_rotation_degrees'] = 0
# imgaug_parameters['affine_shear_degrees'] = 0
# imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
# imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
# imgaug_parameters['probability_of_affine_transform'] = 1
# #Perspective:
# imgaug_parameters['flag_perspective_transform'] = False
# imgaug_parameters['flag_perspective_transform_keep_size'] = True
# imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
# imgaug_parameters['probability_of_perspective_transform'] = 1
# ### Get Augmenter: ###
# imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)
# train_dataset = Dataset_MultipleImagesFromSingleImage(main_path,
#                                                       number_of_image_frame_to_generate=5,
#                                                       transform=imgaug_transforms,
#                                                       image_loader=ImageLoaderCV,
#                                                       max_number_of_images=np.inf,
#                                                       crop_size=np.inf,
#                                                       flag_to_RAM=False,
#                                                       flag_recursive=False,
#                                                       flag_normalize_by_255=False,
#                                                       flag_crop_mode='center',
#                                                       flag_explicitely_make_tensor=False,
#                                                       allowed_extentions=IMG_EXTENSIONS_PNG,
#                                                       flag_base_tranform=True,
#                                                       flag_turbulence_transform=False,
#                                                       Cn2=5e-13,
#                                                       flag_how_to_concat='C')



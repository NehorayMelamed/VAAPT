
import cv2
from albumentations import BasicTransform
from albumentations import DualTransform
from albumentations import ImageOnlyTransform
from albumentations import ImageOnlyIAATransform
# def new_transform_targets(self):
#     return {
#         'image': self.apply,
#         'image2': self.apply
#     }
# DualTransform.targets = property(new_transform_targets)
# ImageOnlyTransform.targets = property(new_transform_targets)
# ImageOnlyIAATransform.targets = property(new_transform_targets)
import albumentations
from albumentations import *


def get_Albumentations_parameters():
    ### Color Augmentations: ###
    flag_RGB_shift = False;
    flag_HSV_shift = False;
    flag_brightness_shift = False;
    flag_gamma_shift = False;
    flag_channel_shuffle = False;  # Shuffle functional is random so it's not able to be transfered for now to multiple image augmentations - it's pretty easy to take care of this actually
    flag_convert_to_gray = False;
    flag_invert_image_brightness = False;  # Only 255-Intensity, so doesn't fit numpy i think
    ### Flip / Transpose: ###
    flag_horizontal_flip = False;
    flag_vertical_flip = False;
    flag_random_rotation_multiple_of_90_degrees = False;
    ### Geometric Transforms: ###
    flag_Affine_transform = False;
    flag_perspective_transform = False;
    flag_optical_distortion = False;
    flag_elastic_distortion = False;  # functional contains random so this doesn't work for multiple image augmentations
    flag_grid_distortion = False;
    flag_piecewise_affine_transform = False;
    ### CutOut/Holes: ###
    flag_cutout = False;
    ###### Blur/Sharpen/Contrast: ######
    flag_simple_blur = False;
    flag_motion_blur = False;  # functional contains random so this doesn't work for multiple image augmentations - easy to fix
    flag_median_blur = False;
    flag_sharpen = False;
    flag_contrast_shift = False;  # Doesn't Work I Think
    flag_emboss = False;
    flag_CLAHE = False;  # Only Supports uint8 inputs!
    ### Additive Gaussian Noise: ###
    flag_additive_BW_gaussian_noise = False;  # Only for uint8 so i will do this myself or use imgaug if wanted
    ### JPEG Compression Degredation: ###
    flag_JPEG_compression_degredation = False;


    ###################################  At What Probabilities to make the Augmentations:  #####################################
    ### Color Augmentations: ###
    probability_of_RGB_shift = 0.5;
    probability_of_HSV_shift = 0.5;
    probability_of_brightness_shift = 0.5;
    probability_of_gamma_shift = 0.5;
    probability_of_channel_shuffle = 0.5;
    probability_of_convert_to_gray = 0.5;
    probability_of_invert_image_brightness = 0.5;
    ###### Flip/Transpose (Dihedral) Augmentations: ######
    probability_of_horizontal_flip = 0.5;
    probability_of_vertical_flip = 0.5;
    probability_of_random_rotation_multiple_of_90_degrees = 0.5;
    ###### Geometric Augmentations: ######
    probability_of_Affine_transform = 0.5;
    probability_of_perspective_transform = 0.5;
    probability_of_optical_distortion = 0.5;
    probability_of_elastic_distortion = 0.5;
    probability_of_grid_distortion = 0.5;
    probability_of_piecewise_affine_transform = 0.5;
    ###### Cutout: ######
    probability_of_cutout = 0.5
    ###### Blur/Sharpen/Contrast: ######
    probability_of_simple_blur = 0.5;
    probability_of_motion_blur = 0.5
    probability_of_median_blur = 0.5;
    probability_of_sharpen = 0.5;
    probability_of_contrast_shift = 0.5;
    probability_of_emboss = 0.5;
    probability_of_CLAHE = 0.5;
    ###### JPEG Compression: ######
    probability_of_JPEG_compression_degredation = 0.5;



    ###################################  Augmentation Parameters HQ:  #####################################
    # (*). Note: some of this functions receive shift values which are purely int, and are applicable only to uint8.
    #           this means that if i want to apply these functions to float32 i must change the functions in albumentations or make these functions myself.
    ###### Color Augmentations: ######
    # (1). RGB shift:
    RGB_channels_shift_range = (-20, 20)
    # (2). HSV shift:
    HSV_h_shift_range = 20;
    HSV_s_shift_range = 20;
    HSV_v_shift_range = 20;
    # (3). Brightness Shift:
    brightness_factor_range = 0.2;
    # (4). Gamma Shift:
    gamma_factor_range = (80, 120);
    # (5). Channel Shuffle:
    # (6). Convert To Gray:
    # (7). Invert Image Brightness:


    ###### Flip/Transpose (Dihedral) Augmentations: ######
    1;


    ###### Geometric Augmentations: ######
    # (1). Affine Transform:
    Affine_shift_range = 0;
    Affine_scale_without_resizing_range = (-0.5, 0.5);
    Affine_rotation_range = 45;
    Affine_transform_interpolation_method = cv2.INTER_CUBIC
    Affine_transform_border_mode = cv2.BORDER_REFLECT_101
    # (2). Pespective Transform:
    flag_keep_size_after_perspective_transform = True;
    perspective_transform_factor_range = (0.05, 0.1)
    # (3). Optical Distortion:
    optical_distortion_factor_range = 0.05;
    optical_distortion_shift_factor_range = 0.05;
    optical_distortion_interpolation_method = cv2.INTER_CUBIC;
    optical_distortion_border_mode = cv2.BORDER_REFLECT_101
    # (4). Elastic Distortion:
    elastic_distortion_alpha = 1;
    elastic_distortion_sigma = 50;
    elastic_distortion_alpha_affine = 50;
    elastic_distortion_interpolation_method = cv2.INTER_CUBIC
    elastic_distortion_border_mode = cv2.BORDER_REFLECT_101
    # (5). Grid Distortion:
    grid_distortion_number_of_steps = 2;
    grid_distortion_factor_range = 0.3;
    grid_distortion_interpolation_method = cv2.INTER_CUBIC
    grid_distortion_border_mode = cv2.BORDER_REFLECT_101
    # (6). PieceWise Affine:
    piecewise_affine_scale_factor = 0.05
    piecewise_affine_number_of_rows = 4;
    piecewise_affine_number_of_cols = 4;
    piecewise_affine_order = 2;
    piecewise_affine_constant_value_outside_valid_region = 0;
    piecewise_affine_extrapolation_mode = 'reflect'

    ###### Cutout / Effective Occlusions (super-pixels): ######
    # (1). CutOut/Holes:
    max_cutout_hole_sizes = [5, 10, 15]
    number_of_holes_per_hole_size = [4, 4, 4];


    ###### Blur/Sharpen/Contrast: ######
    # (1). Simple Blur:
    max_simple_blur_kernel_size = 5;
    # (2). Motion Blur:
    max_motion_blur_kernel_size = 5;
    # (3). Median Blur:
    max_median_blur_kernel_size = 5;
    # (4). Sharpen:
    sharpen_alpha_range = (0.0, 0.5);
    sharpen_lightness_factor_range = (0.5, 1.5);
    # (5). Contrast: DOESN'T WORK - FIX!
    contrast_shift_factor_range = (0, 0.2)
    # (6). Emboss:
    emboss_alpha_factor_range = (0.2, 0.5);
    emboss_strength_factor_range = (0.2, 0.7)
    # (7). CLAHE (Contrast Limited Adaptive Histogram Equalization):
    CLAHE_upper_contrast_limit_for_activation = 3;
    CLAHE_local_grid_size_tuple = (5, 5);


    ###### Additive Noise: ######
    # (1). Additive Gaussian Noise (For Now The Additive Gaussian Noise is Only for uint8 images and so te var_limit is between [0,255])
    additive_BW_gaussian_noise_sigma_range = 20;


    ###### JPEG Compression: ######
    # (1). JPEG Compression Degredation:
    JPEG_compression_quality_range = (95, 99)


    ######################################################
    ######################################################
    ####  Get Dictionary Of Above Defined Parameters: ####
    scope_dictionary = locals();
    return scope_dictionary;


def get_Albumentations_Augmenter(parameters_dictionary):
    #Initialize Transformation List:
    Transformations_List = []

    ###### Color Augmentations: ######
    # (7). Example Of Use:
    if parameters_dictionary['flag_RGB_shift']:
        Transformations_List.append(
            RGBShift(r_shift_limit=parameters_dictionary['RGB_channels_shift_range'], g_shift_limit=parameters_dictionary['RGB_channels_shift_range'],
                     b_shift_limit=parameters_dictionary['RGB_channels_shift_range'], p=parameters_dictionary['probability_of_RGB_shift']))
    if parameters_dictionary['flag_HSV_shift']:
        Transformations_List.append(
            HueSaturationValue(hue_shift_limit=parameters_dictionary['HSV_h_shift_range'], sat_shift_limit=parameters_dictionary['HSV_s_shift_range'],
                               val_shift_limit=parameters_dictionary['HSV_v_shift_range'], p=parameters_dictionary['probability_of_HSV_shift']))
    if parameters_dictionary['flag_brightness_shift']:
        Transformations_List.append(RandomBrightness(limit=parameters_dictionary['brightness_factor_range'], p=parameters_dictionary['probability_of_brightness_shift']))
    if parameters_dictionary['flag_gamma_shift']:
        Transformations_List.append(RandomGamma(gamma_limit=parameters_dictionary['gamma_factor_range'], p=parameters_dictionary['probability_of_gamma_shift']))
    if parameters_dictionary['flag_channel_shuffle']:
        Transformations_List.append(ChannelShuffle(p=parameters_dictionary['probability_of_channel_shuffle']))
    if parameters_dictionary['flag_convert_to_gray']:
        Transformations_List.append(ToGray(p=parameters_dictionary['probability_of_convert_to_gray']))
    if parameters_dictionary['flag_invert_image_brightness']:
        Transformations_List.append(InvertImg(p=parameters_dictionary['probability_of_invert_image_brightness']))


    ###### Flip/Transpose (Dihedral) Augmentations: ######
    # (7). Example Of Use:
    if parameters_dictionary['flag_vertical_flip']:
        Transformations_List.append(VerticalFlip(p=parameters_dictionary['probability_of_vertical_flip']))
    if parameters_dictionary['flag_horizontal_flip']:
        Transformations_List.append(HorizontalFlip(p=parameters_dictionary['probability_of_horizontal_flip']))
    if parameters_dictionary['flag_random_rotation_multiple_of_90_degrees']:
        Transformations_List.append(RandomRotate90(p=parameters_dictionary['probability_of_random_rotation_multiple_of_90_degrees']))

    ###### Geometric Augmentations: ######
    # (7). Example Of Use:
    if parameters_dictionary['flag_Affine_transform']:
        Transformations_List.append(
            ShiftScaleRotate(shift_limit=parameters_dictionary['Affine_shift_range'], scale_limit=parameters_dictionary['Affine_scale_without_resizing_range'],
                             rotate_limit=parameters_dictionary['Affine_rotation_range'], interpolation=parameters_dictionary['Affine_transform_interpolation_method'],
                             border_mode=parameters_dictionary['Affine_transform_border_mode'], p=parameters_dictionary['probability_of_Affine_transform']))
    if parameters_dictionary['flag_perspective_transform']:
        Transformations_List.append(IAAPerspective(scale=parameters_dictionary['perspective_transform_factor_range'],
                                                   keep_size=parameters_dictionary['flag_keep_size_after_perspective_transform'],
                                                   p=parameters_dictionary['probability_of_perspective_transform']))
    if parameters_dictionary['flag_optical_distortion']:
        Transformations_List.append(OpticalDistortion(distort_limit=parameters_dictionary['optical_distortion_factor_range'],
                                                      shift_limit=parameters_dictionary['optical_distortion_shift_factor_range'],
                                                      interpolation=parameters_dictionary['optical_distortion_interpolation_method'], border_mode=parameters_dictionary['optical_distortion_border_mode'],
                                                      p=parameters_dictionary['probability_of_optical_distortion']))
    if parameters_dictionary['flag_elastic_distortion']:
        Transformations_List.append(ElasticTransform(alpha=parameters_dictionary['elastic_distortion_alpha'], sigma=parameters_dictionary['elastic_distortion_sigma'],
                                                     alpha_affine=parameters_dictionary['elastic_distortion_alpha_affine'],
                                                     interpolation=parameters_dictionary['elastic_distortion_interpolation_method'],
                                                     border_mode=parameters_dictionary['elastic_distortion_border_mode'],
                                                     p=parameters_dictionary['probability_of_elastic_distortion']))  # understand alpha'], sigma'], alpha_affine
    if parameters_dictionary['flag_grid_distortion']:
        Transformations_List.append(
            GridDistortion(num_steps=parameters_dictionary['grid_distortion_number_of_steps'], distort_limit=parameters_dictionary['grid_distortion_factor_range'],
                           interpolation=parameters_dictionary['grid_distortion_interpolation_method'], border_mode=parameters_dictionary['grid_distortion_border_mode'],
                           p=parameters_dictionary['probability_of_grid_distortion']))
    if parameters_dictionary['flag_piecewise_affine_transform']:
        Transformations_List.append(
            IAAPiecewiseAffine(scale=parameters_dictionary['piecewise_affine_scale_factor'], nb_rows=parameters_dictionary['piecewise_affine_number_of_rows'],
                               nb_cols=parameters_dictionary['piecewise_affine_number_of_cols'], order=parameters_dictionary['piecewise_affine_order'], cval=parameters_dictionary['piecewise_affine_constant_value_outside_valid_region'],
                               mode=parameters_dictionary['piecewise_affine_extrapolation_mode'], p=parameters_dictionary['probability_of_piecewise_affine_transform']))

    ###### Cutout / Effective Occlusions (super-pixels): #####
    # (3). Example Of Use:
    if parameters_dictionary['flag_cutout']:
        for i, max_cutout_hole_size in enumerate(max_cutout_hole_sizes):
            Transformations_List.append(
                Cutout(num_holes=parameters_dictionary['number_of_holes_per_hole_size'][i], max_h_size=parameters_dictionary['max_cutout_hole_sizes'][i][0],
                       max_w_size=parameters_dictionary['max_cutout_hole_sizes'][i][1], p=parameters_dictionary['probability_of_cutout']))


    ###### Blur/Sharpen/Contrast: ######
    # (8). Example Of Use:
    if parameters_dictionary['flag_simple_blur']:
        Transformations_List.append(Blur(blur_limit=parameters_dictionary['max_simple_blur_kernel_size'], p=parameters_dictionary['probability_of_simple_blur']))
    if parameters_dictionary['flag_motion_blur']:
        Transformations_List.append(MotionBlur(blur_limit=parameters_dictionary['max_motion_blur_kernel_size'], p=parameters_dictionary['probability_of_motion_blur']))
    if parameters_dictionary['flag_median_blur']:
        Transformations_List.append(MedianBlur(blur_limit=parameters_dictionary['max_median_blur_kernel_size'], p=parameters_dictionary['probability_of_median_blur']))
    if parameters_dictionary['flag_sharpen']:
        Transformations_List.append(
            IAASharpen(alpha=parameters_dictionary['sharpen_alpha_range'], lightness=parameters_dictionary['sharpen_lightness_factor_range'], p=parameters_dictionary['probability_of_sharpen']))
    if parameters_dictionary['flag_contrast_shift']:
        Transformations_List.append(RandomContrast(limit=parameters_dictionary['contrast_shift_factor_range'], p=parameters_dictionary['probability_of_contrast_shift']))
    if parameters_dictionary['flag_emboss']:
        Transformations_List.append(
            IAAEmboss(alpha=parameters_dictionary['emboss_alpha_factor_range'], strength=parameters_dictionary['emboss_strength_factor_range'], p=parameters_dictionary['probability_of_emboss']))
    if parameters_dictionary['flag_CLAHE']:
        Transformations_List.append(
            CLAHE(clip_limit=parameters_dictionary['CLAHE_upper_contrast_limit_for_activation'], tile_grid_size=parameters_dictionary['CLAHE_local_grid_size_tuple'],
                  p=parameters_dictionary['probability_of_CLAHE']))


    ###### Additive Noise: ######
    # (2). Example of Use:
    if parameters_dictionary['flag_additive_BW_gaussian_noise']:
        Transformations_List.append(
            GaussNoise(var_limit=parameters_dictionary['additive_BW_gaussian_noise_sigma_range'], p=parameters_dictionary['probability_of_additive_BW_gaussian_noise']))

    ###### JPEG Compression: ######
    # (2). Example of Use:
    if parameters_dictionary['flag_JPEG_compression_degredation']:
        Transformations_List.append(JpegCompression(quality_lower=parameters_dictionary['JPEG_compression_quality_range'][0],
                                                    quality_upper=parameters_dictionary['JPEG_compression_quality_range'][1],
                                                    p=parameters_dictionary['probability_of_JPEG_compression_degredation']))


    ####### Compose Transformations_List to Transformations_Composed: ####
    Transformations_Composed = Compose(Transformations_List, preprocessing_transforms=[], postprocessing_transforms=[])
    return Transformations_Composed



# # ### Albumentations Use Case: ####
# import ESRGAN_utils
# from ESRGAN_utils import *
#
# alb_parameters = get_Albumentations_parameters()
# # # Elastic Transform:
# alb_parameters['flag_elastic_distortion'] = True
# alb_parameters['elastic_distortion_alpha'] = 100
# alb_parameters['elastic_distortion_sigma'] = 100
# alb_parameters['elastic_distortion_alpha_affine'] = 20
# alb_parameters['probability_of_elastic_distortion'] = 1
# #Affine:
# # alb_parameters['flag_Affine_transform'] = True
# # alb_parameters['Affine_shift_range'] = (-0.1, 0.1)
# # alb_parameters['Affine_scale_without_resizing_range'] = (-0.1, 0.1)
# # alb_parameters['Affine_rotation_range'] = 10
# # alb_parameters['probability_of_Affine_transform'] = 1
# # #Perspective:
# alb_parameters['flag_perspective_transform'] = True
# alb_parameters['flag_keep_size_after_perspective_transform'] = 0
# alb_parameters['perspective_transform_factor_range'] = (0.05, 0.1)
# alb_parameters['probability_of_perspective_transform'] = 1
# #Optical Distortion:
# alb_parameters['flag_optical_distortion'] = True
# alb_parameters['optical_distortion_factor_range'] = 0.05
# alb_parameters['optical_distortion_shift_factor_range'] = 0.05
# alb_parameters['probability_of_optical_distortion'] = 1
#Grid Distortion:
# alb_parameters['flag_grid_distortion'] = True
# alb_parameters['grid_distortion_number_of_steps'] = 2
# alb_parameters['grid_distortion_factor_range'] = 0.1
# alb_parameters['probability_of_grid_distortion'] = 1
# # #PieceWise Affine:
# alb_parameters['flag_piecewise_affine_transform'] = True
# alb_parameters['piecewise_affine_scale_factor'] = 0.01
# alb_parameters['piecewise_affine_number_of_rows'] = 5
# alb_parameters['piecewise_affine_number_of_cols'] = 5
# alb_parameters['piecewise_affine_order'] = 1
# alb_parameters['piecewise_affine_constant_value_outside_valid_region'] = 0
# alb_parameters['probability_of_piecewise_affine_transform'] = 1
#
#
# # # 1
# alb_transforms = get_Albumentations_Augmenter(alb_parameters)
#
# image_full_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet/n07565083/n07565083_207.jpg'
# current_frame = read_image_default()
#
# # current_frame = crop_tensor(current_frame,150,150)
# # current_frame.shape
#
# for i in arange(10):
#     tic()
#     images_transformed_dictionary = alb_transforms(image=current_frame)
#     toc()
#     current_frame_transformed = images_transformed_dictionary['image']
#
#     subplot(3,1,1)
#     imshow(current_frame);
#     subplot(3,1,2)
#     imshow(current_frame_transformed)
#     subplot(3,1,3)
#     imshow(current_frame-current_frame_transformed)
#     pause(0.5)





















# #### Albumentations Use Case: ####
# import ESRGAN_utils
# from ESRGAN_utils import *
# import IMGAUG_utils
# # from IMGAUG_utils import *
# imgaug_parameters = get_IMGAUG_parameters()
#
# # # Elastic Transform:
# # imgaug_parameters['flag_elastic_jitter_transform'] = True
# # imgaug_parameters['elastic_jitter_alpha'] = 0.5
# # imgaug_parameters['elastic_jitter_sigma'] = 10
# # imgaug_parameters['elastic_jitter_order'] = 3
# # imgaug_parameters['elastic_jitter_mode'] = 'reflect'
# # imgaug_parameters['probability_of_elastic_jitter_transform'] = 1
# # # Affine:
# imgaug_parameters['flag_affine_transform'] = True
# imgaug_parameters['affine_scale'] = (0.9,1.1)
# imgaug_parameters['affine_translation_percent'] = None
# imgaug_parameters['affine_translation_number_of_pixels'] = (0,10)
# imgaug_parameters['affine_rotation_degrees'] = (0,10)
# imgaug_parameters['affine_shear_degrees'] = (0,10)
# imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
# imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
# imgaug_parameters['probability_of_affine_transform'] = 1
# # #Perspective:
# # imgaug_parameters['flag_perspective_transform'] = True
# # imgaug_parameters['flag_perspective_transform_keep_size'] = True
# # imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
# # imgaug_parameters['probability_of_perspective_transform'] = 1
# # # #PieceWise Affine:
# # imgaug_parameters['flag_piecewise_affine_transform'] = True
# # imgaug_parameters['piecewise_affine_scale'] = 0.01
# # imgaug_parameters['piecewise_affine_number_of_rows'] = 5
# # imgaug_parameters['piecewise_affine_number_of_cols'] = 5
# # imgaug_parameters['piecewise_affine_order'] = 1
# # imgaug_parameters['piecewise_affine_mode'] = 'reflect'
# # imgaug_parameters['flag_piecewise_affine_absolute_scale'] = False #?
# # imgaug_parameters['probability_of_piecewise_affine_transform'] = 1
# #
#
# imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)
# image_full_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet/n07565083/n07565083_207.jpg'
# current_frame = read_image_default()
#
# # current_frame = crop_tensor(current_frame,150,150)
# # current_frame.shape
#
# for i in arange(10):
#     imgaug_transforms = imgaug_transforms.to_deterministic()
#     tic()
#     current_frame_transformed = imgaug_transforms.augment_images([current_frame])[0];
#     toc()
#
#     subplot(3,1,1)
#     imshow(current_frame);
#     subplot(3,1,2)
#     imshow(current_frame_transformed)
#     subplot(3,1,3)
#     imshow(current_frame-current_frame_transformed)
#     pause(0.5)
#
#









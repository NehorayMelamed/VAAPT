


############################################################################################################################################################################################################################################################
###################################  ALBUMENTATIONS:  #####################################
# IMPORTANT NOTE: i think albumentations is the only library which can handle float32 numpy arrays and probably the fastest library.
# IMPORTANT NOTE: there is imgaug which, according to albumentations benchmarks, is a littl bit slower the albumentations, but it does have more additive noise and cutout functions, it does only work with uint8 images,
#                 and it has it own batch loaders / batch operators functions which do the same operations on a batch of images (which might make things faster because you're acting in parallel on a batch of images (sort of like matlab's multi-dimensional operations)
import PIL
import albumentations
from albumentations import *
# import imgaug
# from imgaug import *


#Read Image:
image_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\HR_images\Set14';
image_name = 'lenna.png'
image_full_filename = os.path.join(image_folder, image_name)
input_image_PIL = PIL.Image.open(image_full_filename)
input_image_cv2 = read_image_cv2(image_full_filename);
input_image_height = input_image_PIL.height
input_image_width = input_image_PIL.width;
#PIL to Numpy:
input_image_numpy = numpy.array(input_image_PIL)
# tic()
# for i in arange(10):
#     bla = PIL.Image.fromarray(input_image_numpy)
# toc()


# for i in arange(10):
# albumentations.Compose(transforms_list,preprocessing_transforms=[], postprocessing_transforms=[], bbox_params={}, p=probability_of_compose)
figure()
imshow(input_image_cv2)
# input_image_cv2 = input_image_cv2.astype(float)/255
transforms_composed = Compose([GaussNoise(var_limit=(200,200), p=1)]);
images_transformed_dictionary = transforms_composed(image=input_image_cv2)
input_image_cv2_transformed = images_transformed_dictionary['image']
input_image_cv2_transformed.shape
figure()
imshow(input_image_cv2_transformed)


#Multi-Channel:
flag_initial_random_or_preset_crop_size = 'preset';  # 'random'=choose random crop size for each batch from the below defined range/ 'preset'=choose the below defined preset_crop_HR_size when choosing crop size
preset_final_crop_HR_size = double_tuple(100)
random_crop_initial_crop_height_range = double_tuple((70, 130));
random_crop_initial_aspect_ratio = 1;
# # (2). Resizing After Crop:
# flag_resize_to_preset_size_after_crop = False;
# final_resize_HR_size = 100;
# (4). Actually Create The Pre-Processing Augmenter:

random_crop_and_zoom_transform = RandomSizedCrop(min_max_height=random_crop_initial_crop_height_range,
                                                 height=preset_final_crop_HR_size[0],
                                                 width=preset_final_crop_HR_size[1],
                                                 w2h_ratio=random_crop_initial_aspect_ratio,
                                                 interpolation=cv2.INTER_CUBIC)  # Random Crop Size Deterministic Final Size after Resize
albumentations_transforms = albumentations.Compose([random_crop_and_zoom_transform])

input_image = np.random.randn(300,300,5)
transforms_composed = Compose([albumentations_transforms]);
input_image_transformed_dictionary = transforms_composed(image=input_image);
input_image_transformed = input_image_transformed_dictionary['image'];
input_image_transformed.shape
plot_multiple_images([convert_to_grayscale(input_image), convert_to_grayscale(input_image_transformed)], flag_common_colorbar=0)


############################################################################################################################################################################################################################################################
###################################  Crop/Resize/Padding HQ:  #####################################
###### Crop, Resize and Pad: ######
#(1). Random Crop Location, Deterministic in Size:
final_crop_size = double_tuple((100,100))
RandomCrop(height=final_crop_size[0], width=final_crop_size[1]) #Random Crop Location, Deterministic in Size
#(2). Absolutely Deterministic Crop:
crop_top_left_start_tuple = (0,0);
crop_size_tuple = double_tuple(200)
Crop(x_min=crop_width_start, y_min=crop_height_start, x_max=crop_width_size, y_max=crop_height_size) #Deterministic Crop
#(3). Deterministic Center Crop:
crop_size_tuple = double_tuple(200)
CenterCrop(height=center_crop_height, width=center_crop_width) #Deterministic Center Crop
#(4). Random Initial Crop Size, Deterministic Final Size After Resize:
random_crop_initial_crop_height_range = (100,200);
random_crop_intiial_aspect_ratio = 1;
final_crop_size_after_resize_tuple = double_tuple(200);
RandomSizedCrop(min_max_height=random_crop_height_range, height=height_after_crop_and_resize, width=width_after_crop_and_resize, w2h_ratio=width_to_height_aspect_ratio, interpolation=cv2.INTER_CUBIC) #Random Crop Size Deterministic Final Size after Resize
#(5). Deterministic Resizing Of Whole Image:
final_size_after_resize = double_tuple((150,200))
Resize(height=final_height_after_resize, width=final_width_after_resize, interpolation=cv2.INTER_CUBIC)  #possible interpolations: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4
#(6). Center Crop Specified in Terms of Frame Size Around Center Which is Cropped:
number_of_pixels_to_crop = 100;
percent_of_pixels_to_crop = 0.5;
pad_mode = 'constant'
pad_constant_value_outsize_valid_pixels = 0;
flag_resize_such_remained_center_is_the_size_of_the_original_image = True;
IAACropAndPad(px=number_of_pixels_to_crop, percent=None, pad_mode=pad_mode, pad_cval=pad_constant_value_outsize_valid_pixels, keep_size=flag_resize_such_remained_center_is_the_size_of_the_original_image)
#(7). Pad to Predetermined Size:
min_size_to_pad_to = double_tuple((200))
PadIfNeeded(min_height=min_size_to_pad_to[0], min_width=min_size_to_pad_to[1], border_mode=cv2.BORDER_REFLECT_101, value=[0, 0, 0])




#Try Custom Targets with albumentations:
import PIL
from albumentations import BasicTransform
from albumentations import DualTransform
from albumentations import ImageOnlyTransform
from albumentations import ImageOnlyIAATransform
def new_transform_targets(self):
    return {
        'image': self.apply,
        'image2': self.apply
    }
DualTransform.targets = property(new_transform_targets)
ImageOnlyTransform.targets = property(new_transform_targets)
ImageOnlyIAATransform.targets = property(new_transform_targets)
import albumentations
from albumentations import *
from imgaug import augmenters as iaa
import imgaug
from imgaug import *

###################################  Which Augmentations To Do HQ:  #####################################
### Color Augmentations: ###
flag_RGB_shift = True;
flag_HSV_shift = True;
flag_brightness_shift = True;
flag_gamma_shift = True;
flag_channel_shuffle = False; #Shuffle functional is random so it's not able to be transfered for now to multiple image augmentations - it's pretty easy to take care of this actually
flag_convert_to_gray = True;
flag_invert_image_brightness = False; #Only 255-Intensity, so doesn't fit numpy i think
### Flip / Transpose: ###
flag_horizontal_flip = True;
flag_vertical_flip = True;
flag_random_rotation_multiple_of_90_degrees = True;
### Geometric Transforms: ###
flag_Affine_transform = True;
flag_perspective_transform = True;
flag_optical_distortion = True;
flag_elastic_distortion = False; #functional contains random so this doesn't work for multiple image augmentations
flag_grid_distortion = True;
flag_piecewise_affine_transform = True;
### CutOut/Holes: ###
flag_cutout = False;
###### Blur/Sharpen/Contrast: ######
flag_simple_blur = True;
flag_motion_blur = False; #functional contains random so this doesn't work for multiple image augmentations - easy to fix
flag_median_blur = True;
flag_sharpen = True;
flag_contrast_shift = False; #Doesn't Work I Think
flag_emboss = True;
flag_CLAHE = False; #Only Supports uint8 inputs!
### Additive Gaussian Noise: ###
flag_additive_BW_gaussian_noise = False; #Only for uint8 so i will do this myself or use imgaug if wanted
### JPEG Compression Degredation: ###
flag_JPEG_compression_degredation = False;



###################################  Augmentation Parameters HQ:  #####################################
#(*). Note: some of this functions receive shift values which are purely int, and are applicable only to uint8.
#           this means that if i want to apply these functions to float32 i must change the functions in albumentations or make these functions myself.
Transformations_List = []
###### Color Augmentations: ######
#(1). RGB shift:
# flag_RGB_shift = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_RGB_shift = 0.5;
RGB_channels_shift_range = (-20,20)
#(2). HSV shift:
# flag_HSV_shift = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_HSV_shift = 0.5;
HSV_h_shift_range = 20;
HSV_s_shift_range = 20;
HSV_v_shift_range = 20;
#(3). Brightness Shift:
# flag_brightness_shift = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_brightness_shift = 0.5;
brightness_factor_range = 0.2;
#(4). Gamma Shift:
# flag_gamma_shift = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_gamma_shift = 0.5;
gamma_factor_range = (80,120);
#(5). Channel Shuffle:
# flag_channel_shuffle = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_channel_shuffle = 0.5;
#(5). Convert To Gray:
# flag_convert_to_gray = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_convert_to_gray = 0.5;
#(6). Invert Image Brightness:
# flag_invert_image_brightness = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_invert_image_brightness = 0.5;
#(7). Example Of Use:
if flag_RGB_shift:
    Transformations_List.append(RGBShift(r_shift_limit=RGB_channels_shift_range, g_shift_limit=RGB_channels_shift_range, b_shift_limit=RGB_channels_shift_range, p=probability_of_RGB_shift))
if flag_HSV_shift:
    Transformations_List.append(HueSaturationValue(hue_shift_limit=HSV_h_shift_range, sat_shift_limit=HSV_s_shift_range, val_shift_limit=HSV_v_shift_range, p=probability_of_HSV_shift))
if flag_brightness_shift:
    Transformations_List.append(RandomBrightness(limit=brightness_factor_range, p=probability_of_brightness_shift))
if flag_gamma_shift:
    Transformations_List.append(RandomGamma(gamma_limit=gamma_factor_range,p=probability_of_gamma_shift))
if flag_channel_shuffle:
    Transformations_List.append(ChannelShuffle(p=probability_of_channel_shuffle))
if flag_convert_to_gray:
    Transformations_List.append(ToGray(p=probability_of_convert_to_gray))
if flag_invert_image_brightness:
    Transformations_List.append(InvertImg(p=probability_of_invert_image_brightness))


###### Flip/Transpose (Dihedral) Augmentations: ######
# flag_horizontal_flip = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
# flag_vertical_flip = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
# flag_random_rotation_multiple_of_90_degrees = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_horizontal_flip = 0.5;
probability_of_vertical_flip = 0.5;
probability_of_random_rotation_multiple_of_90_degrees = 0.5;
#(7). Example Of Use:
if flag_vertical_flip:
    Transformations_List.append(VerticalFlip(p=probability_of_vertical_flip))
if flag_horizontal_flip:
    Transformations_List.append(HorizontalFlip(p=probability_of_horizontal_flip))
if flag_random_rotation_multiple_of_90_degrees:
    Transformations_List.append(RandomRotate90(p=probability_of_random_rotation_multiple_of_90_degrees))




###### Geometric Augmentations: ######
#(1). Affine Transform:
# flag_Affine_transform = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_Affine_transform = 0.5;
Affine_shift_range = 0;
Affine_scale_without_resizing_range = (-0.5,0.5);
Affine_rotation_range = 45;
#(2). Pespective Transform:
# flag_perspective_transform = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_perspective_transform = 0.5;
flag_keep_size_after_perspective_transform = True;
perspective_transform_factor_range = (0.05,0.1)
#(3). Optical Distortion:
# flag_optical_distortion = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_optical_distortion = 0.5;
optical_distortion_factor_range = 0.05;
optical_distortion_shift_factor_range = 0.05;
#(4). Elastic Distortion:
# flag_elastic_distortion = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_elastic_distortion = 0.5;
elastic_distortion_alpha = 1;
elastic_distortion_sigma = 50;
elastic_distortion_alpha_affine = 50;
#(5). Grid Distortion:
# flag_grid_distortion = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_grid_distortion = 0.5;
grid_distortion_number_of_steps = 2;
grid_distortion_factor_range = 0.3;
#(6). PieceWise Affine:
# flag_piecewise_affine_transform = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_piecewise_affine_transform = 0.5;
piecewise_affine_scale_factor = 0.05
piecewise_affine_number_of_rows = 4;
piecewise_affine_number_of_cols = 4;
piecewise_affine_order = 2;
#(7). Example Of Use:
if flag_Affine_transform:
    Transformations_List.append(ShiftScaleRotate(shift_limit=Affine_shift_range, scale_limit=Affine_scale_without_resizing_range,
                     rotate_limit=Affine_rotation_range, interpolation=cv2.INTER_CUBIC,
                     border_mode=cv2.BORDER_REFLECT_101, p=probability_of_Affine_transform))
if flag_perspective_transform:
    Transformations_List.append(IAAPerspective(scale=perspective_transform_factor_range, keep_size=flag_keep_size_after_perspective_transform,
                   p=probability_of_perspective_transform))
if flag_optical_distortion:
    Transformations_List.append(OpticalDistortion(distort_limit=optical_distortion_factor_range, shift_limit=optical_distortion_shift_factor_range,
                      interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101,
                      p=probability_of_optical_distortion))
if flag_elastic_distortion:
    Transformations_List.append(ElasticTransform(alpha=elastic_distortion_alpha, sigma=elastic_distortion_sigma,
                     alpha_affine=elastic_distortion_alpha_affine, interpolation=cv2.INTER_CUBIC,
                     border_mode=cv2.BORDER_REFLECT_101,
                     p=probability_of_elastic_distortion))  # understand alpha, sigma, alpha_affine
if flag_grid_distortion:
    Transformations_List.append(GridDistortion(num_steps=grid_distortion_number_of_steps, distort_limit=grid_distortion_factor_range,
                   interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=probability_of_grid_distortion))
if flag_piecewise_affine_transform:
    Transformations_List.append(IAAPiecewiseAffine(piecewise_affine_scale_factor, nb_rows=piecewise_affine_number_of_rows, nb_cols=piecewise_affine_number_of_cols, order=piecewise_affine_order,cval=0, mode='reflect', p=probability_of_piecewise_affine_transform))


###### Cutout / Effective Occlusions (super-pixels): ######
#(1). CutOut/Holes:
# flag_cutout = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_cutout = 0.5
max_cutout_hole_sizes = [5,10,15]
number_of_holes_per_hole_size = [4,4,4];
#(2). imaug functions: dropout, dropout per channel, coarse droupout, coarse dropout per channel, salt, pepper, salt and pepper, coarse salt and pepper, coarse salt, coarse pepper,
#TODO: add those
#(3). Example Of Use:
if flag_cutout:
    for i,max_cutout_hole_size in enumerate(max_cutout_hole_sizes):
        Transformations_List.append(Cutout(num_holes=number_of_holes_per_hole_size[i], max_h_size=max_cutout_hole_sizes[i], max_w_size=max_cutout_hole_sizes[i], p=probability_of_cutout))



###### Blur/Sharpen/Contrast: ######
#(1). Simple Blur:
# flag_simple_blur = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_simple_blur = 0.5;
max_simple_blur_kernel_size = 5;
#(2). Motion Blur:
# flag_motion_blur = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_motion_blur = 0.5
max_motion_blur_kernel_size = 5;
#(3). Median Blur:
# flag_median_blur = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_median_blur = 0.5;
max_median_blur_kernel_size = 5;
#(4). Sharpen:
# flag_sharpen = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_sharpen = 0.5;
sharpen_alpha_range = (0.0,0.5);
sharpen_lightness_factor_range = (0.5,1.5);
#(5). Contrast: DOESN'T WORK - FIX!
# flag_contrast_shift = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_contrast_shift = 0.5;
contrast_shift_factor_range = (0,0.2)
#(6). Emboss:
# flag_emboss = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_emboss = 0.5;
emboss_alpha_factor_range = (0.2,0.5);
emboss_strength_factor_range = (0.2,0.7)
#(7). CLAHE (Contrast Limited Adaptive Histogram Equalization):
# flag_CLAHE = True;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_CLAHE = 0.5;
CLAHE_upper_contrast_limit_for_activation = 3;
CLAHE_local_grid_size_tuple = (5,5);
#(8). Example Of Use:
if flag_simple_blur:
    Transformations_List.append(Blur(blur_limit=max_simple_blur_kernel_size, p=probability_of_simple_blur))
if flag_motion_blur:
    Transformations_List.append(MotionBlur(blur_limit=max_motion_blur_kernel_size, p=probability_of_motion_blur))
if flag_median_blur:
    Transformations_List.append(MedianBlur(blur_limit=max_median_blur_kernel_size, p=probability_of_median_blur))
if flag_sharpen:
    Transformations_List.append(IAASharpen(alpha=sharpen_alpha_range, lightness=sharpen_lightness_factor_range, p=probability_of_sharpen))
if flag_contrast_shift:
    Transformations_List.append(RandomContrast(limit=contrast_shift_factor_range, p=probability_of_contrast_shift))
if flag_emboss:
    Transformations_List.append(IAAEmboss(alpha=emboss_alpha_factor_range, strength=emboss_strength_factor_range, p=probability_of_emboss))
if flag_CLAHE:
    Transformations_List.append(CLAHE(clip_limit=CLAHE_upper_contrast_limit_for_activation, tile_grid_size=CLAHE_local_grid_size_tuple, p=probability_of_CLAHE))


###### Additive Noise: ######
#(1). Additive Gaussian Noise (For Now The Additive Gaussian Noise is Only for uint8 images and so te var_limit is between [0,255])
# flag_additive_BW_gaussian_noise = 1;    #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_additive_BW_gaussian_noise = 0.5;
additive_BW_gaussian_noise_sigma_range = 20;
#(2). Example of Use:
if flag_additive_BW_gaussian_noise:
    Transformations_List.append(GaussNoise(var_limit=additive_BW_gaussian_noise_sigma_range, p=probability_of_additive_BW_gaussian_noise))

###### JPEG Compression: ######
#(1). JPEG Compression Degredation:
# flag_JPEG_compression_degredation = 1;  #Uncomment To OverRide "Which Augmentations To Do HQ"
probability_of_JPEG_compression_degredation = 0.5;
JPEG_compression_quality_range = (95,99)
#(2). Example of Use:
if flag_JPEG_compression_degredation:
    Transformations_List.append(JpegCompression(quality_lower=JPEG_compression_quality_range[0], quality_upper=JPEG_compression_quality_range[1], p=probability_of_JPEG_compression_degredation))



####### Compose Transformations_List to Transformations_Composed: ####
Transformations_Composed = Compose(Transformations_List, preprocessing_transforms=[], postprocessing_transforms=[])








#### Perform Entire Transformations_List: ####
#Read Image:
image_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\HR_images\Set14';
image_name = 'lenna.png'
image_full_filename = os.path.join(image_folder, image_name)
input_image_PIL = PIL.Image.open(image_full_filename)
input_image_cv2 = read_image_cv2(image_full_filename);
input_image_cv2_copy = input_image_cv2.copy()
input_image_height = input_image_PIL.height
input_image_width = input_image_PIL.width;
#PIL to Numpy:
input_image_numpy = numpy.array(input_image_PIL)
# tic()
# for i in arange(10):
#     bla = PIL.Image.fromarray(input_image_numpy)
# toc()

#
# #Try Custom Targets with albumentations:
# import PIL
# from albumentations import BasicTransform
# from albumentations import DualTransform
# from albumentations import ImageOnlyTransform
# from albumentations import ImageOnlyIAATransform
# def new_transform_targets(self):
#     return {
#         'image': self.apply,
#         'image2': self.apply
#     }
# DualTransform.targets = property(new_transform_targets)
# ImageOnlyTransform.targets = property(new_transform_targets)
# ImageOnlyIAATransform.targets = property(new_transform_targets)
# import albumentations
# from albumentations import *
# import matplotlib.pyplot as plt
# import imgaug
# from imgaug import augmenters as iaa
# #Try Efficient Crop & Zoom in/out:
# random_crop_initial_crop_height_range = (70,130);
# random_crop_initial_aspect_ratio = 1;
# final_crop_size_after_resize_tuple = double_tuple(100);
# random_crop_and_zoom_transform = RandomSizedCrop(min_max_height=random_crop_initial_crop_height_range, height=final_crop_size_after_resize_tuple[0], width=final_crop_size_after_resize_tuple[1], w2h_ratio=random_crop_initial_aspect_ratio, interpolation=cv2.INTER_CUBIC) #Random Crop Size Deterministic Final Size after Resize
# albumentations_transforms = albumentations.Compose([random_crop_and_zoom_transform])
# #Try IMGAUG:
# imgaug_transforms = iaa.Sequential([iaa.CropToFixedSize(width=70,height=70), iaa.Scale(size=(100,100))])
#
#
# #Use albumentations built in solution for zooming and cropping:
# tic()
# for i in arange(100):
#     bli = albumentations_transforms(image=input_image_cv2)['image']
# toc()
# #Use IMGAUG my frankenstein solution:
# tic()
# for i in arange(100):
#     bli = imgaug_transforms.augment_image(input_image_cv2)
# toc()
#
# input_image_cv2.shape
# bli.shape
# imshow(bli)
# plt.show()


# albumentations.Compose(transforms_list,preprocessing_transforms=[], postprocessing_transforms=[], bbox_params={}, p=probability_of_compose)
# figure()
# imshow(input_image_cv2)
input_image_cv2_to_transform = input_image_cv2.copy().astype(float32)/255
input_image_cv2_copy_to_transform = input_image_cv2_copy.copy().astype(float32)/255
# input_image_cv2_to_transform = input_image_cv2.copy()
# input_image_cv2_copy_to_transform = input_image_cv2_copy.copy()

#Takes all of the transforms defined above and puts them together:
images_transformed_dictionary = Transformations_Composed(image=input_image_cv2_to_transform,image2=input_image_cv2_copy_to_transform)
input_image_cv2_transformed = images_transformed_dictionary['image']
input_image_cv2_transformed_2 = images_transformed_dictionary['image2']
input_image_cv2_transformed.shape
plot_images_mat([input_image_cv2_transformed,input_image_cv2_transformed_2,input_image_cv2_transformed-input_image_cv2_transformed_2], filter_indices=None, flag_common_colorbar=0, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                    super_title=str(np.array(input_image_cv2_transformed-input_image_cv2_transformed_2).max()), titles_string_list=None)


figure()
imshow(input_image_cv2_transformed)
figure()
imshow(input_image_cv2_transformed_2)
figure()

bla = cv2.cvtColor(input_image_cv2.astype(float32)/255, cv2.COLOR_BGR2HSV)
############################################################################################################################################################################################################################################################











# ############################################################################################################################################################################################################################################################
# ### albumentations ####
# #######################
# #(1). Type Conversion:
# FromFloat
# ToFloat
# #(2). Pre-Procesing Channel Normalization:
# Normalize # """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel. - i would like to control this super low level stuff by myself
# #(3). Padding:
# PadIfNeeded(min_height=height_to_pad_to, min_width=width_to_pad_to, border_mode=cv2.BORDER_REFLECT_101, value=[0, 0, 0], p=probability_of_padding)
# #(4). Resizing and Scaling:
# Resize(height=final_height_after_resize, width=final_width_after_resize, interpolation=cv2.INTER_LINEAR, p=probability_of_resizing)  #possible interpolations: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4
# RandomScale(scale_limit=scaling_range_tuple, p=probability_of_scaling, interpolation=cv2.INTER_CUBIC) #random RESIZING of image
# #(5). Rotation and Affine (shift+scale+rotation):
# Rotate(limit=rotation_range_tuple, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=probability_of_rotation)
# ShiftScaleRotate(shift_limit=shift_rage_tuple, scale_limit=scale_range_tuple, rotate_limit=rotation_range_tuple, interpolation=cv2.INTER_CUBIC,
#                  border_mode=cv2.BORDER_REFLECT_101, p=probability_of_affine_transform)
# #(6). Cropping:
# RandomCrop(height=final_crop_height, width=final_crop_width, p=probability_of_cropping) #Random Crop Location, Deterministic in Size
# Crop(x_min=crop_width_start, y_min=crop_height_start, x_max=crop_width_size, y_max=crop_height_size, p=probability_of_cropping) #Deterministic Crop
# CenterCrop(height=center_crop_height, width=center_crop_width, p=probability_of_center_crop) #Deterministic Center Crop
# RandomSizedCrop(min_max_height=random_crop_height_range, height=height_after_crop_and_resize, width=width_after_crop_and_resize, w2h_ratio=width_to_height_aspect_ratio, interpolation=cv2.INTER_CUBIC, p=probability_of_random_sized_crop) #Random Crop Size Deterministic Final Size after Resize
# #(7). Flipping and Transposing:
# VerticalFlip(p=probability_of_vertical_flip)
# HorizontalFlip(p=probability_of_horizontal_flip)
# Flip(p=probability_of_general_flip)
# Transpose(p=probability_of_transpose)
# RandomRotate90(p=probability_of_90_degree_rotation)
# #(8). Non Rigid Transform:
# OpticalDistortion(distort_limit=distortion_factor_range, shift_limit=shift_factor_range, interpolation=cv2.INTER_CUBIC,border_mode=cv2.BORDER_REFLECT_101, p=probability_of_optical_distortion)
# ElasticTransform(alpha=elastic_transform_alpha, sigma=elastic_transform_sigma, alpha_affine=elastic_transform_alpha_affine, interpolation=cv2.INTER_CUBIC,border_mode=cv2.BORDER_REFLECT_101, p=probability_of_elastic_transform) #understand alpha, sigma, alpha_affine
# GridDistortion(num_steps=number_of_grid_distortion_steps, distort_limit=distortion_factor_range, interpolation=cv2.INTER_CUBIC,border_mode=cv2.BORDER_REFLECT_101, p=probability_of_grid_distortion)
# IAAPerspective(scale=perspective_transform_factor_range, keep_size=flag_keep_size_after_perspective_transform, p=probability_of_perspective_transform)
# IAAPiecewiseAffine(scale=piecewise_affine_scale_factor, nb_rows=piecewise_affine_number_of_rows, nb_cols=piecewise_affine_number_of_cols, order=piecewise_affine_order,cval=0, mode='constant', p=probability_of_piecewise_affine)
# IAASuperpixels(p_replace=super_pixels_probability_of_replacement, n_segments=super_pixels_number_of_segments, p=probability_of_super_pixels)
# IAAEmboss(alpha=emboss_alpha_factor_range, strength=emboss_strength_factor_range, p=probability_of_emboss);
# IAAAdditiveGaussianNoise(loc=0,scale=additive_gaussian_noise_sigma_range, per_channel=flag_additive_gaussian_noise_per_channel, p=probability_of_additive_gaussian_noise)
# IAAAffine(scale=1, translate_percent=None, translate_px=None, rotate=0, shear=0, order=1,cval=0,mode='reflect',p=0.5)
# IAACropAndPad(px=None, percent=None, pad_mode='constant', pad_cval=0, keep_size=True, p=1)
# #(9). Color Augmentations:
# RGBShift(r_shift_limit=rgb_r_channel_shift_range, g_shift_limit=rgb_g_channel_shift_range, b_shift_limit=rgb_b_channel_shift_range, p=probability_of_rgb_shift)
# HueSaturationValue(hue_shift_limit=hsv_h_shift_range, sat_shift_limit=hsv_s_shift_range, val_shift_limit=hsv_v_shift_range, p=probability_of_hsv_shift)
# RandomBrightness(limit=brightness_factor_range, p=probability_of_brightness)
# RandomContrast(limit=contrast_factor_range, p=probability_of_contrast)
# RandomGamma(gamma_limit=gamma_factor_range,p=probability_of_gamma)
# ChannelShuffle(p=probability_of_channel_shuffle)
# ToGray(p=probability_of_grayscale)
# #(10). Filter Based Augmentations:
# GaussNoise(var_limit=noise_sigma_range, p=probability_of_gaussian_noise)
# Blur(blur_limit=max_blur_kernel_size, p=probability_of_blur)
# MotionBlur(blur_limit=max_motion_blur_kernel_size, p=probability_of_motion_blur)
# MedianBlur(blur_limit=max_median_blur_kernel_size, p=probability_of_median_blur)
# IAASharpen(alpha=sharpen_alpha_range, lightness=sharpen_lightness_factor_range, p=probability_of_sharpen)
# #(11). DropOut/Cutout:
# Cutout(num_holes=number_of_cutout_holes, max_h_size=max_cutout_hole_height, max_w_size=max_cutout_hole_width, p=probability_of_cutout)
# #(12). JPEG Compression Corrupt:
# JpegCompression(quality_lower=jpeg_compression_lower_quality, quality_upper=jpeg_compression_uper_quality, p=probability_of_jpeg_compression_corruption)
# #(13). Invert Image(???):
# InvertImg(p=probability_of_invert_image_brightness) #invert by transforming light to dark and dark to light: value->255-value
# #(14). CLAHE (Contrast Limited Adaptive Histogram Equalization):
# CLAHE(clip_limit=CLAHE_upper_contrast_limit_for_action, tile_grid_size=CLAHE_local_grid_size_tuple, p=probability_of_CLAHE)
# ############################################################################################################################################################################################################################################################
#
#
#
#
#
#
#
# ############################################################################################################################################################################################################################################################
# ### TorchVision ####
#
# #Transforms:
# #(1). Crop Strategy:
# flag_RandomCrop_or_RandomResizedCrop = 'RandomResizedCrop'
# final_crop_size = numpy.array([200,200]);
# initial_crop_size_from_final_crop_size = final_crop_size*sqrt(2); #
#
#
# if flag_RandomCrop_or_RandomResizedCrop == 'RandomCrop' or flag_RandomCrop_or_RandomResizedCrop==1:
#     #(1.1). RandomCrop:
#     # crop_size = final_final_crop_size*1;
#     1
# #(1.2). RandomResizedCrop:
# if flag_RandomCrop_or_RandomResizedCrop == 'RandomResizedCrop' or flag_RandomCrop_or_RandomResizedCrop==2:
#     final_crop_size_after_resize = int(input_image_height*0.5) #Apparently only square
#     crop_size_fraction_from_original_size_range = (0.35,0.5);
#     crop_aspect_ratio_range = (0.75,1.25);
#     crop_resize_interpolation_method = 2;
#
#
# #Probably better to first crop and the use other transforms:
# transforms_list = [];
# if flag_RandomCrop_or_RandomResizedCrop == 'RandomCrop':
#     #(1.1). simple random crop of given size -> resize
#     #TODO: in pytorch 1.0 the arguments are different!
#     transforms_list.append([torchvision.transforms.RandomCrop(size=crop_size, pad_if_needed=True, padding=0)])
# if flag_RandomCrop_or_RandomResizedCrop == 'RandomResizedCrop':
#     #(1.2). RandomResizedCrop:
#     transforms_list.append([torchvision.transforms.RandomResizedCrop(size=final_crop_size_after_resize,
#                                                                      scale=crop_size_fraction_from_original_size_range,
#                                                                      ratio=crop_aspect_ratio_range,
#                                                                      interpolation=crop_resize_interpolation_method)
#                             ]);
#
#
#
# #(2). Random Augmentations:
# #(2.1). RandomAffine:
# random_degrees_range = (-22,22);
# flag_expand_image_after_rotation_to_fill_result = True;
# random_translation_fraction_range = (0,0);
# random_shear_degrees_range = (0,20);
# random_scale_factor_range = (1,1);
# affine_resample_method = PIL.Image.BICUBIC
# #(2.2). ColorJitter:
# brightness_factor_range = 0.1;
# contrast_factor_range = 0.1;
# saturation_factor_range = 0.1;
# hue_factor_range = 0.1;
# #Build Transform:
# transforms_list.append([
#     torchvision.transforms.ColorJitter(brightness=brightness_factor_range, contrast=contrast_factor_range, saturation=saturation_factor_range, hue=hue_factor_range),
#     torchvision.transforms.RandomRotation(random_degrees_range, resample=False, expand=flag_expand_image_after_rotation_to_fill_result, center=None), #i seperate RandomRotation and RandomAffine because in RandomRotation you can state expand=True to avoid black areas due to rotation
#     torchvision.transforms.RandomGrayscale(p=0.1),
#     torchvision.transforms.RandomHorizontalFlip(p=0.5),
#     torchvision.transforms.RandomVerticalFlip(p=0.5),
# ])
# if (random_shear_degrees_range==(0,0) and random_scale_factor_range==(1,1) and random_translation_fraction_range==(0,0)) == False:
#     transforms_list.append([torchvision.transforms.RandomAffine(0, translate=random_translation_fraction_range, scale=random_scale_factor_range, shear=random_shear_degrees_range, resample=affine_resample_method, fillcolor=0)]);
#
#
#
# #(3). Crop Again After Affine Transforms If Wanted:
# flag_RandomCrop_or_RandomResizedCrop = 'RandomCrop'
# crop_factor = 1/sqrt(2);
# crop_size = final_crop_size_after_resize;
# if type(final_crop_size_after_resize) == tuple:
#     crop_size = list(crop_size)
#     crop_size[0] = int(crop_size[0]*crop_factor);
#     crop_size[1] = int(crop_size[1]*crop_factor);
# else:
#     crop_size = int(crop_size*crop_factor)
# final_crop_size_after_resize = (200,200);
# if flag_RandomCrop_or_RandomResizedCrop == 'RandomCrop':
#     #(1.1). simple random crop of given size -> resize
#     transforms_list.append([torchvision.transforms.CenterCrop(size=crop_size),\
#                             torchvision.transforms.Resize(size=final_crop_size_after_resize, interpolation=2)])
# if flag_RandomCrop_or_RandomResizedCrop == 'RandomResizedCrop':
#     #(1.2). RandomResizedCrop:
#     transforms_list.append([torchvision.transforms.RandomResizedCrop(size=final_crop_size_after_resize,
#                                                                      scale=crop_size_fraction_from_original_size_range,
#                                                                      ratio=crop_aspect_ratio_range,
#                                                                      interpolation=crop_resize_interpolation_method)
#                             ]);
#
#
# #TODO: change white balance augmentation!!!!
#
# transforms_list = flatten_list(transforms_list)
# transforms_function = torchvision.transforms.Compose(transforms_list)
# # tic()
# # for i in arange(10):
# #     input_image_transformed = transforms_list(input_image_PIL)
# # toc()
# input_image_transformed = transforms_function(input_image_PIL)
# imshow(input_image_transformed)
# title('(' + str(input_image_transformed.height) + ',' + str(input_image_transformed.width) + ')')
#
#
#
#
# bla = torchvision.transforms.functional.adjust_brightness(input_image_PIL, 2)
# imshow(bla)
#
#
#
# #Transforms List (uninterupted without explanations):
# #Basic:
# torchvision.transforms.CenterCrop(size=final_crop_size_at_center);
# torchvision.transforms.FiveCrop(size=final_crop_size_of_each_crop)
# torchvision.transforms.TenCrop(size, vertical_flip=False)
# torchvision.transforms.Grayscale(num_output_channels=1)
# torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
# torchvision.transforms.Normalize(mean=per_channel_mean_tuple, std=per_channel_std_tuple)
# torchvision.transforms.Resize(size, interpolation=2)
# #Random:
# torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
# torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
# torchvision.transforms.RandomApply(transforms, p=0.5)
# torchvision.transforms.RandomChoice #- Apply single transformation randomly picked from a list
# torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
# torchvision.transforms.RandomGrayscale(p=0.1)
# torchvision.transforms.RandomHorizontalFlip(p=0.5)
# torchvision.transforms.RandomOrder(transforms)
# torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
# torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
# torchvision.transforms.RandomVerticalFlip(p=0.5)
# #Tread Carefully with these - i think i should write these myself:
# pix = numpy.array(pic) #From PIL to Numpy
# torchvision.transforms.ToPILImage
# torchvision.transforms.ToTensor
# torchvision.transforms.Lambda(lambd)
# #Functional:
# torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
# torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
# torchvision.transforms.functional.adjust_gamma(img, gamma, gain=1)
# torchvision.transforms.functional.adjust_hue(img, hue_factor)
# torchvision.transforms.functional.adjust_saturation(img, saturation_factor)
# torchvision.transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)
# torchvision.transforms.functional.center_crop(img,output_size)
# torchvision.transforms.functional.crop(img, i, j, h, w)
# torchvision.transforms.functional.hflip(img)
# torchvision.transforms.functional.vflip(img)
# torchvision.transforms.functional.normalize(tensor, mean, std) # This transform acts in-place, i.e., it mutates the input tensor.
# torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
# torchvision.transforms.functional.resize(img, size, interpolation=2)
# torchvision.transforms.functional.resized_crop(img, i, j, h, w, size, interpolation=2)
# torchvision.transforms.functional.rotate(img, angle, resample=False, expand=False, center=None)
# torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)
# torchvision.transforms.functional.to_pil_image(pic, mode=None)
# #Warp with Flow:
# torch.nn.functional.grid_sample(input,grid,mode='bilinear', padding_mode='zeros')
#
#
#
#
#
# #TorchVision Transforms:
##########################################################################################################################################################
#(1). Deterministic Operators (not really augmentations!):
# torchvision.transforms.Compose
# torchvision.transforms.CenterCrop(size=final_crop_size_at_center);
#
# torchvision.transforms.FiveCrop(size=final_crop_size_of_each_crop)
# transform = Compose([
#    FiveCrop(size), # this is a list of PIL Images
#    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
# ])
# #In your test loop you can do the following:
# input, target = batch # input is a 5d tensor, target is 2d
# bs, ncrops, c, h, w = input.size()
# result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
# result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

# torchvision.transforms.TenCrop(size, vertical_flip=False)
# Crop the given PIL Image into four corners and the central crop plus the flipped version of these (horizontal flipping is used by default)

# torchvision.transforms.Grayscale(num_output_channels=1)
# # num_output_channels (int) – (1 or 3) number of channels desired for output image
#
# torchvision.transforms.Pad(padding, fill=0, padding_mode='constant')
# # Parameters:
# # padding (int or tuple) – If a single int is provided this is used to pad all borders.
# #                          If tuple of length 2 is provided this is the padding on left/right and top/bottom respectively.
# #                          If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
# # fill (int or tuple) – Pixel fill value for constant fill. Default is 0.
# #                       If a tuple of length 3, it is used to fill R, G, B channels respectively.
# #                       This value is only used when the padding_mode is constant
# #
# # padding_mode (str) – Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
# #     constant: pads with a constant value, this value is specified with fill
# #     edge: pads with the last value at the edge of the image
# #     reflect: pads with reflection of image without repeating the last value on the edge
# #     For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode will result in [3, 2, 1, 2, 3, 4, 3, 2]
# #     symmetric: pads with reflection of image repeating the last value on the edge
# #     For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode will result in [2, 1, 1, 2, 3, 4, 4, 3]
#
#
# torchvision.transforms.Normalize(mean=per_channel_mean_tuple, std=per_channel_std_tuple)
#
# torchvision.transforms.Resize(size, interpolation=2)
#
# #Tread Carefully with these - i think i should write these myself:
# torchvision.transforms.ToPILImage
# torchvision.transforms.ToTensor
#
# torchvision.transforms.Lambda(lambd)
# ##################################################################################################################################################################################################################################################
#
#
#
# ########################################################################################################################################################################################################################################################################
# #(2). Random Augmentations:
# torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
# # Parameters:
# # brightness (float) – brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
# # contrast (float) – contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
# # saturation (float) – saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
# # hue (float) – hue_factor is chosen uniformly from [-hue, hue]. Should be >=0 and <= 0.5.
#
#
# torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0)
# # Parameters:
# # degrees– If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
# # translate (tuple, optional) – tuple of maximum absolute fraction for horizontal and vertical translations.
# #                               For example translate=(a, b), then horizontal shift is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
# # scale (tuple, optional) – scaling factor interval, e.g (a, b), then scale is randomly sampled from the range a <= scale <= b. Will keep original scale by default.
# # shear- Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees). Will not apply shear by default
# # resample- ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional) – An optional resampling filter. See filters for more information.
# #            If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
# # fillcolor (int) – Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
#
#
# torchvision.transforms.RandomApply(transforms, p=0.5)
#
# torchvision.transforms.RandomChoice #- Apply single transformation randomly picked from a list
#
# torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
# # Crop the given PIL Image at a random location.
# # Parameters:
# # size (sequence or int) – Desired output size of the crop. If size is an int instead of sequence like (h, w), a square crop (size, size) is made.
# # padding (int or sequence, optional) – Optional padding on each border of the image. Default is None, i.e no padding. If a sequence of length 4 is provided, it is used to pad left, top, right, bottom borders respectively. If a sequence of length 2 is provided, it is used to pad left/right, top/bottom borders, respectively.
# # pad_if_needed (boolean) – It will pad the image if smaller than the desired size to avoid raising an exception.
# # fill – Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B channels respectively. This value is only used when the padding_mode is constant
# # padding_mode –
# # Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
# #
# # constant: pads with a constant value, this value is specified with fill
# # edge: pads with the last value on the edge of the image
# # reflect: pads with reflection of image (without repeating the last value on the edge)
# # padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode will result in [3, 2, 1, 2, 3, 4, 3, 2]
# # symmetric: pads with reflection of image (repeating the last value on the edge)
# # padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode will result in [2, 1, 1, 2, 3, 4, 4, 3]
#
# torchvision.transforms.RandomGrayscale(p=0.1)
#
# torchvision.transforms.RandomHorizontalFlip(p=0.5)
#
# torchvision.transforms.RandomOrder(transforms)
#
# torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
# # Crop the given PIL Image to random size and aspect ratio.
# #
# # A crop of random size (default: of 0.08 to 1.0) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made.
# # This crop is finally resized to given size.
# # This is popularly used to train the Inception networks.
# #
# # Parameters:
# # size – expected output size of each edge
# # scale – range of size of the origin size cropped
# # ratio – range of aspect ratio of the origin aspect ratio cropped
# # interpolation – Default: PIL.Image.BILINEAR
#
#
# torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None)
# # Parameters:
# # degrees (sequence or float or int) – Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
# # resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional) – An optional resampling filter. See filters for more information. If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
# # expand (bool, optional) – Optional expansion flag. If true, expands the output to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
# # center (2-tuple, optional) – Optional center of rotation. Origin is the upper left corner. Default is the center of the image.
#
# torchvision.transforms.RandomVerticalFlip(p=0.5)
# ########################################################################################################################################################################################################################################################################
#
#
#
#
# ########################################################################################################################################################################################################################################################################
# #(2). Deterministic/Functional Augmentations (can be used to change image pairs in a deterministic manner)
# #Note - the transforms.functional library is extremely Diverse and worth studying as it may save time in the future
# torchvision.transforms.functional.adjust_brightness(img, brightness_factor)
# torchvision.transforms.functional.adjust_contrast(img, contrast_factor)
# torchvision.transforms.functional.adjust_gamma(img, gamma, gain=1)
# # Also known as Power Law Transform. Intensities in RGB mode are adjusted based on the following equation:
# # Iout=255×gain×(Iin255)γ
# # See Gamma Correction for more details.
# #
# # Parameters:
# # img (PIL Image) – PIL Image to be adjusted.
# # gamma (float) – Non negative real number, same as γ in the equation. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
# # gain (float) – The constant multiplier.
#
# torchvision.transforms.functional.adjust_hue(img, hue_factor)
# # Parameters:
# # img (PIL Image) – PIL Image to be adjusted.
# # hue_factor (float) – How much to shift the hue channel. Should be in [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative direction respectively. 0 means no shift.
# # Therefore, both -0.5 and 0.5 will give an image with complementary colors while 0 gives the original image.
#
# torchvision.transforms.functional.adjust_saturation(img, saturation_factor)
#
#
# torchvision.transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)
# # Apply affine transformation on the image keeping image center invariant
# # Parameters:
# # img (PIL Image) – PIL Image to be rotated.
# # angle (float or int) – rotation angle in degrees between -180 and 180, clockwise direction.
# # translate (list or tuple of python:integers) – horizontal and vertical translations (post-rotation translation)
# # scale (float) – overall scale
# # shear (float) – shear angle value in degrees between -180 to 180, clockwise direction.
# # resample (PIL.Image.NEAREST or PIL.Image.BILINEAR or PIL.Image.BICUBIC, optional) – An optional resampling filter. See filters for more information. If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
# # fillcolor (int) – Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
#
# torchvision.transforms.functional.center_crop(img,output_size)
# torchvision.transforms.functional.crop(img, i, j, h, w)
# # Parameters:
# # img (PIL Image) – Image to be cropped.
# # i – Upper pixel coordinate.
# # j – Left pixel coordinate.
# # h – Height of the cropped image.
# # w – Width of the cropped image.
#
#
# torchvision.transforms.functional.hflip(img)
# torchvision.transforms.functional.vflip(img)
#
# torchvision.transforms.functional.normalize(tensor, mean, std)
# # This transform acts in-place, i.e., it mutates the input tensor.
#
# torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
#
# torchvision.transforms.functional.resize(img, size, interpolation=2)
#
# torchvision.transforms.functional.resized_crop(img, i, j, h, w, size, interpolation=2)
#
# torchvision.transforms.functional.rotate(img, angle, resample=False, expand=False, center=None)
# # img (PIL Image) – PIL Image to be rotated.
# # angle (float or int) – In degrees degrees counter clockwise order.
# # resample (PIL.Image.NEAREST or PIL.Image.BILINEAR or PIL.Image.BICUBIC, optional) – An optional resampling filter. See filters for more information. If omitted, or if the image has mode “1” or “P”, it is set to PIL.Image.NEAREST.
# # expand (bool, optional) – Optional expansion flag. If true, expands the output image to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
# # center (2-tuple, optional) – Optional center of rotation. Origin is the upper left corner. Default is the center of the image.
#
# torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)
# torchvision.transforms.functional.to_pil_image(pic, mode=None)
#
# #Warp with Flow:
# torch.nn.functional.grid_sample(input,grid,mode='bilinear', padding_mode='zeros')
# # img = Image.open('frame1.jpg')
# # img = np.array(img)
# # img = np.array([img])
# # b = torch.from_numpy(img)
# # b = b.permute(0, 3, 1, 2)
# # b = float(b));
# # print(b.size())
# # # b has the size (1, 3, 360, 640)
# # flow = torch.rand(1, 360, 640 , 2)
# # b = Variable(b)
# # flow = Variable(flow)
# # ohno = F.grid_sample(b, flow)
# ########################################################################################################################################################################################################################################################################






















#####################
#FastAI (if you only input x=input_image it will make the transform's parameter random, can also explicitly input a random function with your own parameters to set final parameter bounds):
brightness(x=input_image, change=brightness_change) #  0<=brightness_change<=1 , 0=all black, 0.5=no change, 1=all white
contrast(x=input_image, scale=contrast_change) #   contrast_change>=0,    0=no contrast all grey, 1=no change, >1 super contrast
crop(x=input_image, size=final_crop_size, row_pct=row_percent, col_pct=col_percent)  # deterministic/random CROP:    0<(row_focal_point,col_focal_point)<1,
                                                                                                           #         (0,0)=start from top-left, (0.5,0.5)=center,  (1,1)=end at bottom right
crop_pad(x=input_image, size=final_crop_size, padding_mode='reflection', row_pct=row_percent, col_pct=col_percent) #deterministic/random crop but can have a final_crop_size larger then the image in which case there will be padding
                                                                                                                   #padding_mode='reflection', 'zeros', 'border'
dihedral(x=input_image, k=combination_index)   # 1<=k<=9 , represents a combination of flips and 90 degree rotations
prespective_warp(c=input_image, magnitude=eight_vertices_list, invert=False)  #  Example: eight_vertices_list=torch.tensor(np.zeros(8)),  -1<eight_vertices_list[i]<1
rotate(degrees=degrees_of_rotation)  # obvious
skew(c=input_image, direction=direction_number_int, magnitude=skew_magnitude, invert=False) #  direction_number_int=[0...8], 0<magnitude<1
squish(scale=squish_stretch_factor, row_pct=row_percent, col_pct=col_percent)  # squish_stretch_factor>0, squish_stretch_factor=1-> no change
tilt(c=input_image, direction=direction_of_camera_shift_int, magnitude=shift_magnitude)  #direction_of_camera_shift_int=[0...4],  -1<shift_magnitude<1
zoom(scale=zoom_facotr, row_pct=row_percent, col_pct=col_percent)   # zoom_factor>1


#Deterministic (not random):
flip_lr(x=input_image) #flip...no random

#Random:
rand_crop()
rand_zoom()  #example: rand_zoom=(1.0, 1.5)


#Possibility to randomize "deterministic" transforms:
#(1). probability to do or not to do:
rotate(degrees=30, p=0.5);
#(2). specify random function parameter bounds:
rotate(degrees=(-30,30))
#(3). both:
rotate(degrees(-30,30), p=0.5)


#Special Compound Transform Function:
get_transforms(
               do_flip=True, #if True, a random flip is applied with probability 0.5
               flip_vert=False, #If True, the image can be flipped vertically or rotated of 90 degrees, otherwise only an horizontal flip is applied
               max_rotate=10, #a random rotation between -max_rotate and max_rotate degrees is applied with probability p_affine
               max_zoom=1.1, # a random zoom betweem 1. and max_zoom is applied with probability p_affine
               max_lighting=0.2, # a random lightning and contrast change controlled by max_lighting is applied with probability p_lighting
               max_warp=0.2, # a random symmetric warp of magnitude between -max_warp and maw_warp is applied with probability p_affine
               p_affine=0.75, # the probability that each affine transform and symmetric warp is applied
               p_lighting=0.75,  # the probability that each lighting transform is applied
               xtra_tfms=None
               )

#More Auxiliary:
rand_resize_crop(size:int, max_scale:float=2.0, ratios:Point=(0.75, 1.33))
zoom_crop(scale:float, do_rand:bool=False, p:float=1.0)

















############################################################################################################################################################################################################################################################
###################################  IMGAUG:  #####################################
import PIL
import albumentations
from albumentations import *
import imgaug
# from imgaug import *
from imgaug import augmenters as iaa

#Read Image:
image_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\HR_images\Set14';
image_name = 'lenna.png'
image_full_filename = os.path.join(image_folder, image_name)
input_image1 = read_image_cv2(image_full_filename);
input_image1 = input_image1.astype(float32)/255;
#Batch of 3 identical images:
input_image_batch1 = np.stack([input_image1.copy(),input_image1.copy(),input_image1.copy()])
input_image_batch1.shape
#Batch of 5 identical images:
input_image_batch2 = input_image_batch1.copy();
input_image_batch2.shape
input_image_batch3 = np.stack([input_image1.copy(),input_image1.copy(),input_image1.copy(),input_image1.copy(),input_image1.copy(),input_image1.copy()])
input_image_batch3.shape
#Batch of 3 images of lena artificially stacked to 6 channels:
input_image_batch1_stacked = np.append(input_image_batch1,input_image_batch1,axis=3)
input_image_batch1_stacked.shape
#Batch of 3 images of lena artificially stacked to 5 channels:
input_image_batch1_stacked = np.append(input_image_batch1,input_image_batch1,axis=3)
input_image_batch1_stacked.shape
#Batch of 4 random images of 5 channels:
random_input_image_batch1 = np.random.randn(4,100,100,5);



##################
# Lambda: #

def func_images(images, random_state, parents, hooks):
    images[:, ::2, :, :] = 0
    return images

def func_heatmaps(heatmaps, random_state, parents, hooks):
    for heatmaps_i in heatmaps:
        heatmaps.arr_0to1[::2, :, :] = 0
    return heatmaps

def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


aug = iaa.Lambda(func_images = func_images,
                 func_heatmaps = func_heatmaps,
                 func_keypoints = func_keypoints)


##################







#Check Functionality on image batches
# sequential = iaa.Sequential([iaa.Fliplr(p=0.5)]);
sequential = iaa.Sequential([iaa.CropToFixedSize(width=100,height=100,name=None,deterministic=False)]);
# (1). Using augment_images:
# input_image_batch1_augment_images = sequential.augment_images(input_image_batch1)
# plot_multiple_images(input_image_batch1_augment_images)
#(2). Using Deterministic augment_images to augment both image batches the same way:
sequential = sequential.to_deterministic()
sequential.deterministic #--> returns false
input_image_batch1_augment_images = sequential.augment_images(input_image_batch1) #Returns a List!!!....that's weird other augmentors have returned numpy arrays...
input_image_batch2_augment_images = sequential.augment_images(input_image_batch2) #Does this mean that for any sake (whether consistency or speed) i would have to immediately convert it to numpy?
input_image_batch3_augment_images = sequential.augment_images(input_image_batch3)
print(type(input_image_batch1_augment_images))
print(type(input_image_batch2_augment_images))
print(type(input_image_batch3_augment_images))
# input_image_batch1_augment_images.shape
# input_image_batch2_augment_images.shape
# input_image_batch3_augment_images.shape
plot_multiple_images(input_image_batch1_augment_images,flag_common_colorbar=0);
plot_multiple_images(input_image_batch2_augment_images,flag_common_colorbar=0);
plot_multiple_images(input_image_batch3_augment_images,flag_common_colorbar=0);


#Check Functionality on images and "heatmaps" (images with number of channels difffering from 3):
# sequential = iaa.Sequential([iaa.Fliplr(p=0.5)]);
sequential = iaa.Sequential([iaa.CropToFixedSize(width=100,height=100,name=None,deterministic=False)]);
sequential = sequential.to_deterministic()
input_image_batch1_augment_images = sequential.augment_images(input_image_batch1);
input_image_batch1_stacked_augment_images = sequential.augment_images(input_image_batch1_stacked);
random_input_image_batch1_augment_images = sequential.augment_images(random_input_image_batch1);

input_image_batch1_augment_images = np.array(input_image_batch1_augment_images);
input_image_batch1_stacked_augment_images = np.array(input_image_batch1_stacked_augment_images);
random_input_image_batch1_augment_images = np.array(random_input_image_batch1_augment_images)

input_image_batch1_augment_images_grayscale = convert_to_grayscale(input_image_batch1_augment_images);
input_image_batch1_stacked_augment_images_grayscale = convert_to_grayscale(input_image_batch1_stacked_augment_images);
random_input_image_batch1_augment_images_grayscale = convert_to_grayscale(random_input_image_batch1_augment_images);

input_image_batch1_augment_images_grayscale.shape
random_input_image_batch1_augment_images_grayscale.shape
input_image_batch1_stacked_augment_images_grayscale.shape

plot_multiple_images(input_image_batch1_augment_images_grayscale,flag_common_colorbar=0);
plot_multiple_images(input_image_batch1_stacked_augment_images_grayscale,flag_common_colorbar=0);
plot_multiple_images(random_input_image_batch1_augment_images_grayscale,flag_common_colorbar=0);
plot_multiple_images(input_image_batch1_augment_images_grayscale-input_image_batch1_stacked_augment_images_grayscale,flag_common_colorbar=0)





# input_image = input_image.astype(np.float32)/255
seq = iaa.Sequential([iaa.CropAndPad(px=None,percent=0.1,pad_mode='constant',pad_cval=0,keep_size=True,sample_independently=True,name=None,deterministic=False)])
input_image_iaa = seq.augment_image(input_image);
input_image_iaa.shape
input_image.shape
plot_multiple_images([input_image,input_image_iaa,input_image-input_image_iaa], filter_indices=None, flag_common_colorbar=0, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                    super_title=None, titles_string_list=None)


# I Think Here I Prefer The albumentations API... however perhapse i will need to implement some of the functions myself to fit this api
iaa.Scale(size=size,interpolation='cubic',name=None,deterministic=False) #Deterministic/Random Resize (depending on input variables) - only works on type uint8!
iaa.CropAndPad(px=None,percent=None,pad_mode='constant',pad_cval=0,keep_size=True,sample_independently=True,name=None,deterministic=False) #Deterministic/Random Center(!) Cropping/Padding (depending on input variables)
iaa.Crop(px=None,percent=None,keep_size=True,sample_independently=True,name=None,deterministic=False) #Deterministic/Random Center Crop
iaa.Pad(px=None,percent=None,pad_mode='constant',pad_cval=0,keep_size=True,sample_independently=True,name=None,deterministic=False) #Deterministic/Random Center Pad
iaa.PadToFixedSize(width=width,height=height,pad_mode='constant',pad_cval=0,name=None,deterministic=False) #Expand/Pad if needed to specific size
iaa.CropToFixedSize(width=width,height=height,name=None,deterministic=False) #Random location but Deterministic Crop Size
#Always return a specific size no metter the original size of the image:
iaa.Sequential[iaa.PadToFixedSize(100,100,'reflect'),
               iaa.CropToFixedSize(100,100)]



############################################################################################################################################
#### Things i would like to be able to achieve with my augmentation library: ####
#(1). be able to use as many functionalities on both uint8 and float32 images
#(2). be able to use as many functionalities on images with number of channels differing from 3
#(3). be able to augment two images of the same size the same way
#(4). be able to augment two images of different size the same approximate way
#(5). be able to augment an image and a segmentaion-map/bounding-box/etc' the same way in a manner appropriate to the kind of image
#(6). have the library be readable enough to be able to make changes and add new functionalities the way i want
#(7). have as much functionality as possible (including perhapse non-rigid geometric transforms)
#(8). be able to augment an arbitrary amount of images at once / augment an arbitrary amount of images the same way
#(9). have the library be fast in as many operations as are possible and especially in operations which are bottle-necks
#(10). have comfortable and efficient cropping and random cropping functionalities
#############################################################################################################################################











#############################################################################################################################################
###################################  Which Augmentations To Do HQ:  #####################################
#(1). BLUR:
flag_gaussian_blur = True;
flag_average_blur = True
flag_median_blur = True
flag_bilateral_blur = True
#(2). Arithmatic:
flag_add_value_all_pixels = True;
flag_add_value_per_pixel = True;
flag_multiply_all_pixels = True;
flag_multiple_per_pixel = True;
#(3). Noise:
flag_additive_gaussian_noise = True
flag_salt_and_pepper = True;
flag_coarse_salt_and_pepper = True
flag_salt = True
flag_coarse_salt = True
flag_pepper = True
flag_coarse_pepper = True
#TODO: add speckle noise
#(4). Dropout:
flag_dropout = True
flag_coarse_dropout = True
flag_replace_elementwise = True;
#(5). Overlay Images:
flag_overlay_alpha = True
flag_overlay_alpha_elementwise = True
flag_overlay_simplex_noise_alpha = True
flag_overlap_frequency_noise_alpha = True
#(6). Flip/Transpose:
flag_horizontal_flip = True;
flag_vertical_flip = True;
flag_transpose = True; #Not impelemented - make a Lambda function
#(7). Geometric:
flag_affine_transform = True
flag_piecewise_affine_transform = True
flag_perspective_transform = True;
flag_elastic_jitter_transform = True;
#TODO: add optical and "true" elastic transform
#(8). Color:
flag_HSV_transform = True
flag_colorspace_transform = True
flag_to_grayscale = True
flag_invert_intensity = True
#TODO: add RGB shift
#(9). Contrast (and gamma):
flag_gamma_contrast = True
flag_sigmoid_contrast = True
flag_log_contrast = True
flag_linear_contrast = True
flag_contrast_normalization = True
#(10). Low/High Frequency Emphasis:
flag_sharpen = True
flag_emboss = True
flag_edge_detect = True
flag_directed_edge_detect = True
#(11). Convolve:
flag_convolve = True
#(12). JPEG Compression:
flag_JPEF_compression = True





###################################  Augmentation Parameters HQ:  #####################################
#### CROP AND PAD: #####
# I Think Here I Prefer The albumentations API... however perhapse i will need to implement some of the functions myself to fit this api
iaa.Scale(size=size,interpolation='cubic',name=None,deterministic=False) #Deterministic/Random Resize (depending on input variables) - only works on type uint8!
iaa.CropAndPad(px=None,percent=None,pad_mode='constant',pad_cval=0,keep_size=True,sample_independently=True,name=None,deterministic=False) #Deterministic/Random Center(!) Cropping/Padding (depending on input variables)
iaa.Crop(px=None,percent=None,keep_size=True,sample_independently=True,name=None,deterministic=False) #Deterministic/Random Center Crop
iaa.Pad(px=None,percent=None,pad_mode='constant',pad_cval=0,keep_size=True,sample_independently=True,name=None,deterministic=False) #Deterministic/Random Center Pad
iaa.PadToFixedSize(width=width,height=height,pad_mode='constant',pad_cval=0,name=None,deterministic=False) #Expand/Pad if needed to specific size
iaa.CropToFixedSize(width=width,height=height,name=None,deterministic=False) #Random location but Deterministic Crop Size
#Always return a specific size no metter the original size of the image:
iaa.Sequential[iaa.PadToFixedSize(100,100,'reflect'),
               iaa.CropToFixedSize(100,100)]
########################



#### BLUR: ####
#(1). Gaussian Blur:
gaussian_blur_kernel_size = 5;
#(2). Average Blur:
average_blur_kernel_size = 5;
#(3). Median Blur:
median_blur_kernel_size = 5
#(4). Bilateral Blur:
bilateral_blur_kernel_size = 5;
sigma_color = (10,250);
sigma_space = (10,250);


#### Arithmatic: ####
#(1). Add value all pixels:
value_to_add_all_pixels = 0;
flag_add_value_all_pixels_per_channel = 0;
#(2). Add value per pixel:
value_to_add_per_pixel = 0;
flag_add_value_per_pixel_per_channel = 0;
#(3). Multiply value all pixels:
value_to_multiply_all_pixels = 0;
flag_multiply_value_all_pixels_per_channel = 0
#(4). Multiply value per pixel:
value_to_multiply_per_pixel = 0;
flag_multiply_value_per_pixel_per_channel = 0;


#### Noise: ####
#(1). Additive Gaussian Noise:
additive_gaussian_noise_mu = 0;
additive_gaussian_noise_sigma = 10;
flag_additive_gaussian_noise_per_channel = 1;
#(2). Salt and Pepper:
salt_and_pepper_noise_percent_of_pixels = 0.05;
flag_salt_and_pepper_noise_per_channel = 0;
#(3). Coarse Salt and Pepper:
coarse_salt_and_pepper_noise_percent_of_pixels = 0.05;
coarse_salt_and_pepper_noise_pixel_sizes = None;
coarse_salt_and_pepper_noise_percent_sizes = (0,0.1);
coarse_salt_and_pepper_noise_minimum_size = 4;
flag_coarse_salt_and_pepper_noise_per_channel = 0;
#(4). Salt Noise:
salt_noise_percent_of_pixels = 0;
flag_salt_noise_per_channel = 0;
#(5). Coarse Salt Noise:
coarse_salt_noise_percent_of_pixels = 0.05;
coarse_salt_noise_pixel_sizes = None;
coarse_salt_noise_percent_sizes = (0,0.1);
coarse_salt_noise_minimum_size = 4;
flag_coarse_salt_noise_per_channel = 0;
#(6). Pepper Noise:
pepper_noise_percent_of_pixels = 0;
flag_pepper_noise_per_channel = 0;
#(7). Coarse Pepper Noise:
coarse_pepper_noise_percent_of_pixels = 0.05;
coarse_pepper_noise_pixel_sizes = None;
coarse_pepper_noise_percent_sizes = (0,0.1);
coarse_pepper_noise_minimum_size = 4;
flag_coarse_pepper_noise_per_channel = 0;


#### Dropout/Replacement: ####
#(1). Dropout:
dropout_percent_of_pixels = 0;
flag_dropout_per_channel = 0;
#(2). Coarse Dropout:
coarse_dropout_percent_of_pixels = 0;
coarse_dropout_pixel_sizes = None
coarse_dropout_percent_sizes = 0;
coarse_dropout_minimum_size = 4
flag_coarse_dropout_per_channel = 0;
#(3). Replace ElementWise:
replace_elementwise_indices_to_change_mask = 0;
replace_elementwise_replacement_mask = 0;
flag_replace_elementwise_per_channel = 0;


#### Overlay Images: ####
#(1). Overlay Alpha:
overlay_factor = 0;
overlay_first = None;
overlay_second = None;
flag_overlay_per_channel = 0;
#(2). Overlay Alpha ElementWise:
overlay_elementwise_factor = 0;
overlay_elementwise_first = None
overlay_elementwise_second = None;
flag_overlay_elementwise_per_channel = 0;
#(3). Simplex Noise Alpha:
simplex_noise_first = None;
simplex_noise_second = None;
simplex_noise_max_pixel_sizes = (2,16);
simplex_noise_upscale_method = None;
simplex_noise_number_of_iterations = (1,3);
simplex_noise_aggregation_method = 'max';
simplex_noise_sigmoid = True;
simplex_noise_sigmoid_threshold = None;
flag_simplex_noise_per_channel = 0;
#(4). Frequency Noise Alpha:
frequency_noise_exponent = (-4,4);
frequency_noise_first = None
frequency_noise_second = None
frequency_noise_max_pixel_sizes = (4,16);
frequency_noise_upscale_method = None
frequency_noise_number_of_iterations = (1,3);
frequency_noise_aggregation_method = ['avg','max'];
frequency_noise_sigma = 0.5;
frequency_noise_sigma_threshold = None;
flag_frequency_noise_per_channel = 0;


#### Flip/Transpose: ####
#(1). Horizontal Flip:
horizontal_flip_probability = 0.5;
#(2). Vertical Flip:
vertical_flip_probability = 0.5;


#### Geometric: ####
#(1). Affine:
affine_scale = 1;
affine_translation_percent = 0;
affine_translation_number_of_pixels = 0;
affine_rotation_degrees = 30;
affine_shear_degrees = 30;
affine_order = cv2.INTER_CUBIC;
affine_mode = cv2.BORDER_REFLECT_101;
affine_constant_value_outside_valid_range = 0;
#(2). PieceWise Affine:
piecewise_affine_scale = 0;
piecewise_affine_number_of_rows = 10;
piecewise_affine_number_of_cols = 10;
piecewise_affine_order = 1;
piecewise_affine_mode = 'reflect';
piecewise_affine_constant_value_outside_valid_range = 0;
flag_piecewise_affine_absolute_scale = False;
#(3). Perspective Transform:
perspective_transform_scale = 0;
flag_perspective_transform_keep_size = True;
#(4). Elastic Jitter Transform:
elastic_jitter_alpha = 0;
elastic_jitter_sigma = 0;
elastic_jitter_order = 3;
elastic_jitter_mode = 'reflect'
elastic_jitter_constant_value_outside_valid_range = 0


#### Color: ####
#(1). Shift HSV:
shift_HSV_hs_value = 0;
shift_HSV_from_colorspace = 'RGB';
shift_HSV_channels = [0,1]
flag_shift_HSV_per_channel = False;
#(2). Invert Intensity:
invert_intensity_probability = 0;
invert_intensity_min_value = 0;
invert_intensity_max_value = 1;
flag_invert_intensity_per_channel = False;
#(3). To GrayScale (bottle-neck):
to_grayscale_alpha = 0;
to_grayscale_from_colorspace = 'RGB';


#### Contrast (and gamma): ####
#(1). gamma contrast:
gamma_contrast = 1;
flag_gamma_per_channel = False;
#(2). sigmoid contrast:
sigmoid_contrast_gain = 10;
sigmoid_contrast_cutoff = 0.5;
flag_sigmoid_contrast_per_channel = False;
#(3). Log Contrast:
log_contrast_gain = 1;
flag_log_contrast_per_channel = False;
#(4). Linear Contrast:
linear_contrast_gain = 1;
flag_linear_contrast_per_channel = False;
#(5). Contrast Normalization:
contrast_normalization_alpha = 1;
flag_contrast_normalization_per_channel = False;


#### Low/High Frequency Emphasis: ####
#(1). Sharpen:
sharpen_alpha = 1;
sharpen_lightness = 1;
#(2). Emboss:
emboss_alpha = 0;
emboss_strength = 1;
#(3). Edge Detect:
edge_detect_alpha = 0;
#(4). Directed Edge Detect:
directed_edge_detect_alpha = 0;
directed_edge_detect_direction = (0,1);


#### Convolve: ####
convolve_matrix_to_convolve = 0;


#### JPEG Compression Degredation: ####
JPEG_compression_level_of_compression = 90;






###################################################################################################################################################





#IMGAUG Operations:
#(1). BLUR:
iaa.GaussianBlur(sigma=0, name=None, deterministic=False)
iaa.AverageBlur(k=1, name=None, deterministic=False)
iaa.MedianBlur(k=1,name=None, deterministic=False) #k must be an odd number
iaa.BilateralBlur(d=1,sigma_color=(10,250), sigma_space=(10,250), name=None, deterministic=False)
#(2). ARITHMATIC:
iaa.Add(value=0, per_channel=False,name=None, deterministic=False)
iaa.AddElementwise(value=0, per_channel=False,name=None, deterministic=False)
iaa.AdditiveGaussianNoise(loc=0,scale=0,per_channel=False,name=None,deterministic=False)
iaa.Multiply(mul=1,per_channel=False,name=None,deterministic=False)
iaa.MultiplyElementwise(mul=1,per_channel=False,name=None,deterministic=False)
iaa.Dropout(p=0,per_channel=False,name=None,deterministic=False)
iaa.CoarseDropout(p=0,size_px=None,size_percent=None,per_channel=False,min_size=4,name=None,deterministic=False)
iaa.ReplaceElementwise(mask=mask,replacement=repelacement,per_channel=False,name=None,deterministic=False)
iaa.SaltAndPepper(p=0,per_channel=False,name=None,deterministic=False)
iaa.CoarseSaltAndPepper(p=0,size_px=None,size_percent=None,per_channel=False,min_size=4,name=None,deterministic=False)
iaa.Salt(p=0,per_channel=False,name=None,deterministic=False)
iaa.CoarseSalt(p=0,size_px=None,size_percent=None,per_channel=False,min_size=4,name=None,deterministic=False)
iaa.Pepper(p=0,per_channel=False,name=None,deterministic=False)
iaa.CoarsePepper(p=0,size_px=None,size_percent=None,per_channel=False,min_size=4,name=None,deterministic=False)
iaa.Invert(p=0,per_channel=False,min_value=0,max_value=255,name=None,deterministic=False)
iaa.ContrastNormalization(alpha=1,per_channel=False,name=None,deterministic=False)
iaa.JpegCompression(compression=50,name=None,deterministic=False)
#(3). COLOR:
iaa.WithColorspace(to_colorspace=colorspace_string, from_colorspace='RGB',name=None,deterministic=False)
iaa.AddToHueAndSaturation(value=0,per_channel=False,from_colorspace='RGB',channels=[0,1]) #it seems that HSV could be a bottleneck in terms of performance
iaa.ChangeColorspace(to_colorspace=colorspace_String,from_colorspace='RGB',alpha=1,name=None,deterministic=False)
iaa.Grayscale(alpha=0,from_colorspace='RGB',name=None,deterministic=False) #seems to be a bottle-neck operation and is very slow with IMGAUG
#(4). CONTRAST:
iaa.GammaContrast(gamma=1,per_channel=False,name=None,deterministic=False)
iaa.SigmoidContrast(gain=10,cutoff=0.5,per_channel=False,name=None,deterministic=False)
iaa.LogContrast(gain=1,per_channel=False,name=None,deterministic=False)
iaa.LinearContrast(alpha=1,per_channel=False,name=None,deterministic=False)
#(5). CONVOLVE:
iaa.Convolve(matrix=matrix_to_convolve, name=None, deterministic=False)
iaa.Sharpen(alpha=1,lightness=1,name=None, deterministic=False)
iaa.Emboss(alpha=0,strength=1,name=None,deterministic=False)
iaa.EdgeDetect(alpha=0,name=None,deterministic=False)
iaa.DirectedEdgeDetect(alpha=0,direction=(0,1), name=None, deterministic=False)
#(6). FLIP:
iaa.Fliplr(p=0,name=None,deterministic=False)
iaa.Flipud(p=0,name=None,deterministic=False)
#(7). GEOMETRIC:
iaa.Affine(scale=1,translate_percent=None,translate_px=None,rotate=0,shear=0,order=1,cval=0,mode='constant',fit_output=False,backend='auto',name=None,deterministic=False)
iaa.AffineCv2(scale=1,translate_percent=None,translate_px=None,rotate=0,shear=0,order=cv2.INTER_CUBIC,cval=0,mode=cv2.BORDER_REFLECT_101,name=None,deterministic=False)
iaa.PiecewiseAffine(scale=0,nb_rows=4,nb_cols=4,order=1,cval=0,mode='constant',absolute_scale=False,name=None,deterministic=False);
iaa.PerspectiveTransform(scale=0,keep_size=True,name=None,deterministic=False)
iaa.ElasticTransformation(alpha=0,sigma=0,order=3,cval=0,mode='constant',name=None,deterministic=False)
#(8). META/WRAPPER:
iaa.Sequential(children=None,random_order=False,name=None,deterministic=False)
iaa.SomeOf(n=None,children=None,random_order=False,name=None,deterministic=False)
iaa.OneOf(children=children_list, name=None,deterministic=False)
iaa.Sometimes(p=0,then_list=None,else_list=None,name=None,deterministic=False)
iaa.WithChannels(channels=None,children=None,name=None,deterministic=False)
iaa.Noop(name=None,deterministic=False)
iaa.Lambda(func_images=func_images,func_heatmaps=func_heatmaps,func_keypoints=func_keypoints,name=None,deterministic=False)
iaa.AssertLambda
iaa.AssertShape
#(9). OVERLAP- augmenters that overlay two images with each other:
iaa.Alpha(factor=0,first=None,second=None,per_channel=False,name=None,deterministic=False)
iaa.AlphaElementwise(factor=0,first=None,second=None,per_channel=False,name=None,deterministic=False)
iaa.SimplexNoiseAlpha(first=None,second=None,per_channel=False,size_px_max=(2,16),upscale_method=None,iterations=(1,3),aggregation_method='max',sigmoid=True,sigmoid_thresh=None,name=None,deterministic=False)
iaa.FrequencyNoiseAlpha(exponent=(-4,4), first=None,second=None,per_channel=False,size_px_max=(4,16),upscale_method=None,iterations=(1,3),aggregation_method=['avg','max'],sigmoid=0.5,sigmoid_thresh=None,name=None,deterministic=False)
#(10). CROP AND PAD:
iaa.Scale(size=size,interpolation='cubic',name=None,deterministic=False)
iaa.CropAndPad(px=None,percent=None,pad_mode='constant',pad_cval=0,keep_size=True,sample_independently=True,name=None,deterministic=False)
iaa.Crop(px=None,percent=None,keep_size=True,sample_independently=True,name=None,deterministic=False)
iaa.Pad(px=None,percent=None,pad_mode='constant',pad_cval=0,keep_size=True,sample_independently=True,name=None,deterministic=False)
iaa.PadToFixedSize(width=width,height=height,pad_mode='constant',pad_cval=0,name=None,deterministic=False)
iaa.CropToFixedSize(width=width,height=height,name=None,deterministic=False)



##################################################################################################################################################################################################################
# Augment two batches of images in exactly the same way (e.g. horizontally flip 1st, 2nd and 5th images in both batches, but do not alter 3rd and 4th images):

from imgaug import augmenters as iaa

# Standard scenario: You have N RGB-images and additionally 21 heatmaps per image.
# You want to augment each image and its heatmaps identically.
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
heatmaps = np.random.randint(0, 255, (16, 128, 128, 21), dtype=np.uint8) #i didn't get an error changing heatmap image size to (200,200). view results to see what's going on.

image_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\HR_images\Set14';
image_name = 'lenna.png'
image_full_filename = os.path.join(image_folder, image_name)
input_image_cv2 = read_image_cv2(image_full_filename);
input_image_cv2_float = input_image_cv2.astype(float32);
# seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(translate_px={"x": (-40, 40)})])
# seq = iaa.Sequential([iaa.PiecewiseAffine(scale=1/10/7,nb_rows=10,nb_cols=10,order=1,cval=0,mode='reflect',absolute_scale=False,name=None,deterministic=False)])
seq = iaa.AffineCv2(scale=(0.5,1.5),translate_percent=None,translate_px=None,rotate=0,shear=0,order=cv2.INTER_CUBIC,cval=0,mode=cv2.BORDER_REFLECT_101,name=None,deterministic=False);
seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the star
image_aug = seq_det.augment_image(input_image_cv2_float)
image_aug/=255
imshow(image_aug)
input_image_cv2_float.shape
image_aug.shape

seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(translate_px={"x": (-40, 40)})])

# Convert the stochastic sequence of augmenters to a deterministic one.
# The deterministic sequence will always apply the exactly same effects to the images.
#(*). Note: so images and heatmaps have the same number of batch elements (16) and size (128,128) but differing number of channels. so i presume that any augmentation
#           which acts on each channel independently is simply replacated so the corresponding batch element of the heatmaps.
#           the question is - what about color transforms, what about different sizes, and what about if i want to activate the save transforms on all batch elements?
#(*). Note: i suppose i can activate the same augmentations on each individual image hard coded or in a loop structure but there's got to be an easier way
seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
images_aug = seq_det.augment_images(images)
heatmaps_aug = seq_det.augment_images(heatmaps)
##################################################################################################################################################################################################################







############################################################################################################################################################################################################################################################
# A standard machine learning situation. Train on batches of images and augment each batch via crop, horizontal flip ("Fliplr") and gaussian blur:

from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

for batch_idx in range(1000):
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    images = load_batch(batch_idx)  # you have to implement this function
    images_aug = seq.augment_images(images)  # done by the library
    train_on_images(images_aug)  # you have to implement this function
#############################################################################################################################################################################################




#############################################################################################################################################################################################
# Apply heavy augmentations to images (used to create the image at the very top of this readme):

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# random example images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization((0.5, 2.0))
                    )
                ]),
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0)),
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)

images_aug = seq.augment_images(images)
##################################################################################################################################################################################################################







##################################################################################################################################################################################################################
# Quickly show example results of your augmentation sequence:

from imgaug import augmenters as iaa
import numpy as np

images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])

# show an image with 8*8 augmented versions of image 0
seq.show_grid(images[0], cols=8, rows=8)

# Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
# versions of image 1. The identical augmentations will be applied to
# image 0 and 1.
seq.show_grid([images[0], images[1]], cols=8, rows=8)
##################################################################################################################################################################################################################








##################################################################################################################################################################################################################
# Augment images and landmarks/keypoints on these images:

import imgaug as ia
from imgaug import augmenters as iaa
import random
import numpy as np
images = np.random.randint(0, 50, (4, 128, 128, 3), dtype=np.uint8)

# Generate random keypoints.
# The augmenters expect a list of imgaug.KeypointsOnImage.
keypoints_on_images = []
for image in images:
    height, width = image.shape[0:2]
    keypoints = []
    for _ in range(4):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        keypoints.append(ia.Keypoint(x=x, y=y))
    keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))

seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(scale=(0.5, 0.7))])
seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

# augment keypoints and images
images_aug = seq_det.augment_images(images)
keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)

# Example code to show each image and print the new keypoints coordinates
for img_idx, (image_before, image_after, keypoints_before, keypoints_after) in enumerate(zip(images, images_aug, keypoints_on_images, keypoints_aug)):
    image_before = keypoints_before.draw_on_image(image_before)
    image_after = keypoints_after.draw_on_image(image_after)
    ia.imshow(np.concatenate((image_before, image_after), axis=1)) # before and after
    for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
        keypoint_old = keypoints_on_images[img_idx].keypoints[kp_idx]
        x_old, y_old = keypoint_old.x, keypoint_old.y
        x_new, y_new = keypoint.x, keypoint.y
        print("[Keypoints for image #%d] before aug: x=%d y=%d | after aug: x=%d y=%d" % (img_idx, x_old, y_old, x_new, y_new))

##################################################################################################################################################################################################################








##################################################################################################################################################################################################################
# Apply single augmentations to images:

from imgaug import augmenters as iaa
import numpy as np
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

flipper = iaa.Fliplr(1.0) # always horizontally flip each input image
images[0] = flipper.augment_image(images[0]) # horizontally flip image 0

vflipper = iaa.Flipud(0.9) # vertically flip each input image with 90% probability
images[1] = vflipper.augment_image(images[1]) # probably vertically flip image 1

blurer = iaa.GaussianBlur(3.0)
images[2] = blurer.augment_image(images[2]) # blur image 2 by a sigma of 3.0
images[3] = blurer.augment_image(images[3]) # blur image 3 by a sigma of 3.0 too

translater = iaa.Affine(translate_px={"x": -16}) # move each input image by 16px to the left
images[4] = translater.augment_image(images[4]) # move image 4 to the left

scaler = iaa.Affine(scale={"y": (0.8, 1.2)}) # scale each input image to 80-120% on the y axis
images[5] = scaler.augment_image(images[5]) # scale image 5 by 80-120% on the y axis

##################################################################################################################################################################################################################




##################################################################################################################################################################################################################
# Apply an augmenter to only specific image channels:

from imgaug import augmenters as iaa
import numpy as np

# fake RGB images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# add a random value from the range (-30, 30) to the first two channels of
# input images (e.g. to the R and G channels)
aug = iaa.WithChannels(
  channels=[0, 1],
  children=iaa.Add((-30, 30))
)

images_aug = aug.augment_images(images)

##################################################################################################################################################################################################################




##################################################################################################################################################################################################################
# You can use more unusual distributions for the stochastic parameters of each augmenter:

from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Blur by a value sigma which is sampled from a uniform distribution
# of range 0.1 <= x < 3.0.
# The convenience shortcut for this is: iaa.GaussianBlur((0.1, 3.0))
blurer = iaa.GaussianBlur(iap.Uniform(0.1, 3.0))
images_aug = blurer.augment_images(images)

# Blur by a value sigma which is sampled from a normal distribution N(1.0, 0.1),
# i.e. sample a value that is usually around 1.0.
# Clip the resulting value so that it never gets below 0.1 or above 3.0.
blurer = iaa.GaussianBlur(iap.Clip(iap.Normal(1.0, 0.1), 0.1, 3.0))
images_aug = blurer.augment_images(images)

# Same again, but this time the mean of the normal distribution is not constant,
# but comes itself from a uniform distribution between 0.5 and 1.5.
blurer = iaa.GaussianBlur(iap.Clip(iap.Normal(iap.Uniform(0.5, 1.5), 0.1), 0.1, 3.0))
images_aug = blurer.augment_images(images)

# Use for sigma one of exactly three allowed values: 0.5, 1.0 or 1.5.
blurer = iaa.GaussianBlur(iap.Choice([0.5, 1.0, 1.5]))
images_aug = blurer.augment_images(images)

# Sample sigma from a discrete uniform distribution of range 1 <= sigma <= 5,
# i.e. sigma will have any of the following values: 1, 2, 3, 4, 5.
blurer = iaa.GaussianBlur(iap.DiscreteUniform(1, 5))
images_aug = blurer.augment_images(images)

##################################################################################################################################################################################################################





##################################################################################################################################################################################################################
# You can dynamically deactivate augmenters in an already defined sequence:

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# images and heatmaps, just arrays filled with value 30
images = np.ones((16, 128, 128, 3), dtype=np.uint8) * 30
heatmaps = np.ones((16, 128, 128, 21), dtype=np.uint8) * 30

# add vertical lines to see the effect of flip
images[:, 16:128-16, 120:124, :] = 120
heatmaps[:, 16:128-16, 120:124, :] = 120

seq = iaa.Sequential([
  iaa.Fliplr(0.5, name="Flipper"),
  iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
  iaa.Dropout(0.02, name="Dropout"),
  iaa.AdditiveGaussianNoise(scale=0.01*255, name="MyLittleNoise"),
  iaa.AdditiveGaussianNoise(loc=32, scale=0.0001*255, name="SomeOtherNoise"),
  iaa.Affine(translate_px={"x": (-40, 40)}, name="Affine")
])

# change the activated augmenters for heatmaps,
# we only want to execute horizontal flip, affine transformation and one of
# the gaussian noises
def activator_heatmaps(images, augmenter, parents, default):
    if augmenter.name in ["GaussianBlur", "Dropout", "MyLittleNoise"]:
        return False
    else:
        # default value for all other augmenters
        return default
hooks_heatmaps = ia.HooksImages(activator=activator_heatmaps)

seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
images_aug = seq_det.augment_images(images)
heatmaps_aug = seq_det.augment_images(heatmaps, hooks=hooks_heatmaps)

##################################################################################################################################################################################################################






##################################################################################################################################################################################################################
# Images can be augmented in background processes using the method augment_batches(batches, background=True),
# where batches is expected to be a list of image batches or a list of batches/lists of imgaug.KeypointsOnImage or a list of imgaug.Batch.
# The following example augments a list of image batches in the background:

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data

# Number of batches and batch size for this example
nb_batches = 10
batch_size = 32

# Example augmentation sequence to run in the background
augseq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CoarseDropout(p=0.1, size_percent=0.1)
])

# For simplicity, we use the same image here many times
astronaut = data.astronaut()
astronaut = ia.imresize_single_image(astronaut, (64, 64))

# Make batches out of the example image (here: 10 batches, each 32 times
# the example image)
batches = []
for _ in range(nb_batches):
    batches.append(
        np.array(
            [astronaut for _ in range(batch_size)],
            dtype=np.uint8
        )
    )

# Show the augmented images.
# Note that augment_batches() returns a generator.
for images_aug in augseq.augment_batches(batches, background=True):
    ia.imshow(ia.draw_grid(images_aug, cols=8))

##################################################################################################################################################################################################################





##################################################################################################################################################################################################################
# Images can also be augmented in background processes using the classes imgaug.BatchLoader and imgaug.BackgroundAugmenter, which offer a bit more flexibility.
# (augment_batches() is a wrapper around these.)
# Using these classes is good practice, when you have a lot of images that you don't want to load at the same time.

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from skimage import data

# Example augmentation sequence to run in the background.
augseq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CoarseDropout(p=0.1, size_percent=0.1)
])

# A generator that loads batches from the hard drive.
def load_batches():
    # Here, load 10 batches of size 4 each.
    # You can also load an infinite amount of batches, if you don't train
    # in epochs.
    batch_size = 4
    nb_batches = 10

    # Here, for simplicity we just always use the same image.
    astronaut = data.astronaut()
    astronaut = ia.imresize_single_image(astronaut, (64, 64))

    for i in range(nb_batches):
        # A list containing all images of the batch.
        batch_images = []
        # A list containing IDs of images in the batch. This is not necessary
        # for the background augmentation and here just used to showcase that
        # you can transfer additional information.
        batch_data = []

        # Add some images to the batch.
        for b in range(batch_size):
            batch_images.append(astronaut)
            batch_data.append((i, b))

        # Create the batch object to send to the background processes.
        batch = ia.Batch(
            images=np.array(batch_images, dtype=np.uint8),
            data=batch_data
        )

        yield batch

# background augmentation consists of two components:
#  (1) BatchLoader, which runs in a Thread and calls repeatedly a user-defined
#      function (here: load_batches) to load batches (optionally with keypoints
#      and additional information) and sends them to a queue of batches.
#  (2) BackgroundAugmenter, which runs several background processes (on other
#      CPU cores). Each process takes batches from the queue defined by (1),
#      augments images/keypoints and sends them to another queue.
# The main process can then read augmented batches from the queue defined
# by (2).
batch_loader = ia.BatchLoader(load_batches)
bg_augmenter = ia.BackgroundAugmenter(batch_loader, augseq)

# Run until load_batches() returns nothing anymore. This also allows infinite
# training.
while True:
    print("Next batch...")
    batch = bg_augmenter.get_batch()
    if batch is None:
        print("Finished epoch.")
        break
    images_aug = batch.images_aug

    print("Image IDs: ", batch.data)

    ia.imshow(np.hstack(list(images_aug)))

batch_loader.terminate()
bg_augmenter.terminate()



##################################################################################################################################################################################################################







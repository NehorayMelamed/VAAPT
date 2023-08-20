

from datetime import datetime
import cv2 as cv
import numpy as np
import os
#Pylab imshow stuff:
import pylab
from pylab import imshow, pause, draw, title, axes, ylabel, ylim, yticks, xlabel, xlim, xticks
from pylab import colorbar, colormaps, subplot, suptitle, plot

from colour_demosaicing import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Menon2007
import torch
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

def load_image(resize=None, indx = 1):
    # load image index from predetermined database_path
    #database_path = 'E:/learning_to_see_in_dark_Rony/dataset/test/'

    database_path = 'D:/Work/SeeInDark/datasets/test/'
    all_files = os.listdir(database_path)
    current_image = all_files[indx]


    img = cv.cvtColor(cv.imread(database_path + current_image), cv.COLOR_BGR2RGB)

    if resize is not None:
        h, w, _ = img.shape

        dx = resize if w > h else int(round(w * resize / h/2)*2)
        dy = resize if w < h else int(round(h * resize / w/2)*2)

        img = cv.resize(img, (dx, dy), interpolation=cv.INTER_AREA)

    return (img / np.iinfo(np.uint8).max).astype(np.float32)



def to_linear(img):
    """
    Apply reverse sRGB curve on input image as defined at
    https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation.
    """
    return np.where(img >= 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)


def compute_noise_stddev(linear, noise_gain, use_mean=True):
    """
    Noise model is defined by its signal-dependent noise variance given as a * x^2 + b * x + c.
    """
    mean = linear
    if use_mean: #The "use_mean" flag is something ronny added because he adds information about the estimated std per pixel , not relevant for me unless i do the same i think
        kernel = np.zeros((9, 9), np.float32)
        kernel[0:9:2,0:9:2]=1/25
        mean = cv.filter2D(linear, -1, kernel)

    p1 = 8.481e-5
    p2 = 4.222e-5

    c1 = 3.816e-8
    c2 = 2.957e-7
    c3 = -2.961e-6

    min_val = 5e-7

    a = p1 * noise_gain + p2
    b = max(c1 * noise_gain * noise_gain + c2 * noise_gain + c3, min_val)

    return np.sqrt(a * mean + b) * 1.3



def apply_gain(input_bayer, gain):
    #apply gain to BAYER PATTERN
    h, w = input_bayer.shape
    output = np.zeros((h, w))

    output[0:h:2, 0:w:2] = input_bayer[0:h:2, 0:w:2] * gain[0]
    output[0:h:2, 1:w:2] = input_bayer[0:h:2, 1:w:2] * gain[1]
    output[1:h:2, 1:w:2] = input_bayer[1:h:2, 1:w:2] * gain[2]
    output[1:h:2, 0:w:2] = input_bayer[1:h:2, 0:w:2] * gain[3]
    return output

def apply_gain_RGB(input_RGB, gain):
    #apply gain to RGB PATTERN
    h, w, c = input_RGB.shape
    output = np.zeros((h, w, c))

    output[:,:,0] = input_RGB[:,:,0] * gain[0]
    output[:,:,1] = input_RGB[:,:,1] * gain[1]
    output[:,:,2] = input_RGB[:,:,2] * gain[2]
    return output


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = np.expand_dims(raw, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out









def read_image_cv2(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
    image = cv.imread(path, cv.IMREAD_COLOR);
    if flag_convert_to_rgb == 1:
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB);
    if flag_normalize_to_float == 1:
        image = image.astype(np.float32) / 255; #normalize to range [0,1] instead of [0,255]
    if image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        image = np.expand_dims(image, axis=2);
    if image.shape[2] > 4:
        # In the weird case where the image has more then 3 channels only take the first 3 channels to make it conform with 3-channels image format:
        image = image[:,:,:3];

    return image





def get_noisy_RGB_using_bayer(bayer, std_image, flag_clip=True, gain=32):
    #########
    #(*). Calculating the std_image in the bayer domain, transforming it using demosaicing,
    #########
    h, w = bayer.shape

    ### Apply Additive random noise: ###
    rand_noise = np.random.randn(h, w)
    noise_image = std_image * rand_noise  # actually generate the noise

    ### Add noise to the bayer image: ###
    bayer = bayer + noise_image
    bayer = np.maximum(bayer, 0)
    bayer = np.minimum(bayer, 1)

    ### demosaic noisy bayer: ###
    RGB_image = demosaicing_CFA_Bayer_Menon2007(bayer, 'GRBG')

    ### return to rgb domain: ###
    if flag_clip:
        RGB_image = np.maximum(RGB_image, 0)
        RGB_image = np.minimum(RGB_image, 1)

    return RGB_image




def get_bayer_and_std_image(original_image, flag_clip=True, flag_random_analog_gain=False, gain=32):
    # Get Bayer image from RGB, compute noise std in bayer domain and return noisy BAYER #

    bayer = mosaicing_CFA_Bayer(original_image, 'GRBG')  # to_linear = de-gamma
    h, w = bayer.shape

    ### Compute noisy part of the image using analog gain: ###
    analog_gain = gain
    std_image = compute_noise_stddev(bayer, analog_gain, use_mean=False)  ### Difference is use_mean=False  ???

    ### Add noise to the bayer image: ###
    bayer = np.maximum(bayer, 0)
    bayer = np.minimum(bayer, 1)

    return bayer, std_image




def noise_RGB_through_bayer_no_rand(original_image, flag_clip=True, flag_random_analog_gain=False, gain=None):
    # Get RGB image, compute gain and std in bayer, noise in bayer, then transform back to RGB: #

    bayer = mosaicing_CFA_Bayer(original_image, 'GRBG')  # to_linear = de-gamma
    h, w = bayer.shape

    ### Compute noisy part of the image using analog gain: ###
    if flag_random_analog_gain:
        analog_gain = np.random.random(1) * 92 + 32
    else:
        analog_gain = 32 + 92/2
        if gain:
            analog_gain = gain
    std_image = compute_noise_stddev(bayer, analog_gain, use_mean=False)  ### Difference is use_mean=False  ???

    ### Apply Multiplicative random noise: ###
    rand_noise = np.random.randn(h, w)
    noise_image = std_image * rand_noise  # actually generate the noise

    ### Add noise to the bayer image: ###
    bayer = bayer + noise_image
    bayer = np.maximum(bayer, 0)
    bayer = np.minimum(bayer, 1)

    ### return to rgb domain: ###
    noisy_rgb = demosaicing_CFA_Bayer_Menon2007(bayer, 'GRBG')
    if flag_clip:
        noisy_rgb = np.maximum(noisy_rgb, 0)
        noisy_rgb = np.minimum(noisy_rgb, 1)

    return noisy_rgb




def noise_RGB_through_bayer(original_img, flag_clip=True, flag_random_gamma=False, flag_random_gain=False, flag_random_RGB_gain=False, flag_random_white_balance=False, flag_random_analog_gain=False):
    # the whole thing - get RGB, transform to bayer, put gain noises and std noise, transform to RGB and put color noises: #

    bayer = to_linear(mosaicing_CFA_Bayer(original_img, 'GRBG'))  # to_linear = de-gamma
    h, w = bayer.shape

    ### Get random gamma and gain (detector gain): ###
    if flag_random_gamma:
        gamma_rand = np.random.random(1) * 1.5 + 1
    else:
        gamma_rand = 1 + 1.5/2
    if flag_random_gain:
        gain_rand = np.random.random(1) * 0.8 + 0.2  # THIS DOESN'T GO BEYOND 1 !!!! .... WHAT IF I WANT TO PUT MORE EMPHASIS ON HIGH ALL-GAINS
    else:
        gain_rand = 0.2 + 0.8/2;
    # print(gain_rand)
    bayer = (bayer ** gamma_rand) * gain_rand

    ### Get random R,B,G gains: ### color augmentation
    if flag_random_RGB_gain:
        R_rand = np.random.random(1) * 2 + 0.5
        G_rand = np.random.random(1) * 2 + 0.5
        B_rand = np.random.random(1) * 2 + 0.5
    else:
        R_rand = 0.5 + 2/2
        G_rand = 0.5 + 2/2
        B_rand = 0.5 + 2/2
    gain_array = [G_rand, R_rand, G_rand, B_rand]
    bayer = apply_gain(bayer, gain_array)

    ### Accumulate clean-bayer and "clean" (demoasiced again(!)) RGB images: ###
    clean_bayer_images = bayer
    gt_rgb_image = np.clip(demosaicing_CFA_Bayer_Menon2007(bayer, 'GRBG'), 0, 1)

    ### Get more (White-Balance???) small random gains for the gr,gb and large gains for the R,B bayer channels: ###
    if flag_random_white_balance:
        gain_gr = np.random.random(1) * 0.05 + 1
        gain_gb = np.random.random(1) * 0.05 + 1
        gain_r = np.random.random(1) * 3 + 1
        gain_b = np.random.random(1) * 3 + 1
    else:
        gain_gr = 0.05/2 + 1
        gain_gb = 0.05/2 + 1
        gain_r = 3/2 + 1
        gain_b = 3/2 + 1
    gain_array = [gain_gr, gain_r, gain_gb, gain_b]
    inv_gain_array = [1 / gain_gr, 1 / gain_r, 1 / gain_gb, 1 / gain_b]

    ### Apply INVERSE GAIN to the bayer image: ###
    bayer_before_wb = apply_gain(bayer, inv_gain_array)

    ### Compute noisy part of the image using analog gain: ###
    if flag_random_analog_gain:
        analog_gain = np.random.random(1) * 92 + 32
    else:
        analog_gain = 32 + 92/2
    std_image = compute_noise_stddev(bayer_before_wb, analog_gain, use_mean=False)  ### Difference is use_mean=False  ???

    ### Apply Gain to the Noise itself: ###
    std_image = apply_gain(std_image, gain_array)  # This is basically what needs to change to accomodate pure RGB images, which would avoid changing to bayer and back and speed things up considerably

    ### Apply Multiplicative random noise: ###
    rand_noise = np.random.randn(h, w)
    noise_image = std_image * rand_noise  # actually generate the noise

    ### Add noise to the bayer image: ###
    bayer = bayer + noise_image
    bayer = np.maximum(bayer, 0)
    bayer = np.minimum(bayer, 1)

    ### return to rgb domain: ###
    noisy_rgb = demosaicing_CFA_Bayer_Menon2007(bayer, 'GRBG')
    if flag_clip:
        noisy_rgb = np.maximum(noisy_rgb, 0)
        noisy_rgb = np.minimum(noisy_rgb, 1)

    return noisy_rgb, gt_rgb_image





def noise_RGB_image_RonnyModelOnRGB(original_image, gain, flag_output_uint8_or_float='uint8'):
    if original_image.dtype == np.uint8:
        original_image = original_image / 255

    std_image = compute_noise_stddev(original_image, gain, False)
    noisy_image = original_image + std_image * np.random.randn(*original_image.shape);

    if flag_output_uint8_or_float==np.uint8 or flag_output_uint8_or_float=='uint8':
        noisy_image *= 255;
        noisy_image = noisy_image.clip(0,255)
        noisy_image = np.uint8(noisy_image)
    else:
        noisy_image = noisy_image.clip(0,1)

    return noisy_image





###########################################################################################################################################################
def noise_RGB_LinearShotNoise(original_image, gain, flag_output_uint8_or_float='float'):
    #Input is either uint8 between [0,255]  or float between [0,1]
    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        original_image *= 255;
    original_image = np.float32(original_image)

    default_gain = 20;
    electrons_per_pixel = original_image * default_gain / gain;
    std_shot_noise = np.sqrt(electrons_per_pixel);
    noise_image = np.random.randn(*original_image.shape)*std_shot_noise;
    noisy_image = electrons_per_pixel + noise_image
    noisy_image = noisy_image * gain / default_gain;

    if flag_output_uint8_or_float==np.uint8 or flag_output_uint8_or_float=='uint8':
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = np.uint8(noisy_image)
    else:
        noisy_image = noisy_image/255;
        noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image



def noise_RGB_LinearShotNoise_Torch(original_images, gain, flag_output_uint8_or_float='float', flag_clip=False):

    original_tensors = original_images.copy()
    #Input is either uint8 between [0,255]  or float between [0,1]
    if original_tensors.dtype == torch.float32 or original_tensors.dtype == torch.float64:
        original_tensors *= 255;
    original_tensors = original_tensors.type(torch.float32)

    default_gain = 20;
    electrons_per_pixel = original_tensors * default_gain / gain;
    std_shot_noise = torch.sqrt(electrons_per_pixel);
    noise_image = torch.randn(*original_tensors.shape).to(original_tensors.device)*std_shot_noise;
    noisy_image = electrons_per_pixel + noise_image
    noisy_image = noisy_image * gain / default_gain;

    if flag_output_uint8_or_float==torch.uint8 or flag_output_uint8_or_float=='uint8':
        noisy_image = torch.clamp(noisy_image, 0,255)
        noisy_image = noisy_image.type(torch.uint8)
    else:
        noisy_image = noisy_image/255;
        if flag_clip:
            noisy_image = torch.clamp(noisy_image, 0,1)

    return noisy_image





def noise_RGB_through_Bayer_NoDarkBias_SIDFormat(original_image_RGB, gain=None, flag_clip=True, flag_output_uint8_or_float='float'):
    # Input is either uint8 between [0,255]  or float between [0,1]
    if original_image_RGB.dtype == np.uint8:
        original_image_RGB /= 255;
        original_image_RGB = np.float32(original_image_RGB)


    ### Turn input RGB image into the needed Bayer Pattern: ###
    bayer = mosaicing_CFA_Bayer(original_image_RGB, 'GRBG')  # to_linear = de-gamma
    h, w = bayer.shape

    ### Compute noisy part of the image using analog gain: ###
    std_image = compute_noise_stddev(bayer, gain, use_mean=False)  ### Difference is use_mean=False  ???

    ### Apply Additive random noise: ###
    rand_noise = np.random.randn(h, w)
    noise_image = std_image * rand_noise  # actually generate the noise

    ### Add noise to the bayer image: ###
    bayer_noisy = bayer + noise_image
    bayer_noisy = np.maximum(bayer_noisy, 0)
    bayer_noisy = np.minimum(bayer_noisy, 1)

    ### Get STD Estimate For Rony's Model: ###
    std_image_estimated = compute_noise_stddev(bayer_noisy, gain, use_mean=True)


    ### Output Image in the Correct Format: ###
    if flag_output_uint8_or_float==np.uint8 or flag_output_uint8_or_float=='uint8':
        #uint8:
        bayer_noisy = bayer_noisy*255
        if flag_clip:
            bayer_noisy = np.clip(bayer_noisy, 0, 255)
        bayer_noisy = np.uint8(bayer_noisy)
    else:
        #float32:
        if flag_clip:
            bayer_noisy = np.clip(bayer_noisy, 0, 1)


    # ### Temp - Get Noisy RGB: ###
    # RGB_noisy = demosaicing_CFA_Bayer_Menon2007(bayer_noisy,'GRBG')


    ### Pack Bayer + STD into the correct format to feed rony's model: ###
    packed_bayer = pack_raw(bayer)
    packed_std = pack_raw(std_image_estimated)
    bayer_std_packed = np.concatenate((packed_bayer, packed_std), axis=2)
    img_noise_extended = np.expand_dims(bayer_std_packed, axis=0)  # ->[1, H,W,C]
    img_gt_extended = np.expand_dims(original_image_RGB, axis=0)

    return bayer_std_packed, img_gt_extended




def get_noisy_RGB_image_and_output_SIDFormat(RGB_noisy, gain=None, flag_clip=True, flag_output_uint8_or_float='float'):
    # Input is either uint8 between [0,255]  or float between [0,1]
    if RGB_noisy.dtype == np.uint8:
        RGB_noisy /= 255;
        RGB_noisy = np.float32(RGB_noisy)


    ### Turn input RGB image into the needed Bayer Pattern: ###
    bayer_noisy = mosaicing_CFA_Bayer(RGB_noisy, 'GRBG')  # to_linear = de-gamma
    h, w = bayer_noisy.shape

    ### Get STD Estimate For Rony's Model: ###
    std_image_estimated = compute_noise_stddev(bayer_noisy, gain, use_mean=True)


    ### Output Image in the Correct Format: ###
    if flag_output_uint8_or_float==np.uint8 or flag_output_uint8_or_float=='uint8':
        #uint8:
        bayer_noisy = bayer_noisy*255
        if flag_clip:
            bayer_noisy = np.clip(bayer_noisy, 0, 255)
        bayer_noisy = np.uint8(bayer_noisy)
    else:
        #float32:
        if flag_clip:
            bayer_noisy = np.clip(bayer_noisy, 0, 1)


    # ### Temp - Get Noisy RGB: ###
    # RGB_noisy = demosaicing_CFA_Bayer_Menon2007(bayer_noisy,'GRBG')


    ### Pack Bayer + STD into the correct format to feed rony's model: ###
    packed_bayer = pack_raw(bayer_noisy)
    packed_std = pack_raw(std_image_estimated)
    bayer_std_packed = np.concatenate((packed_bayer, packed_std), axis=2)
    img_noise_extended = np.expand_dims(bayer_std_packed, axis=0)  # ->[1, H,W,C]
    RGB_noisy_original_extended = np.expand_dims(RGB_noisy, axis=0)

    return bayer_std_packed, RGB_noisy_original_extended, bayer_noisy
###########################################################################################################################################################









def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(pre_string='', verbose=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if verbose:
        if pre_string != '':
            print(pre_string + ": %f seconds." % tempTimeInterval)
        else:
            print("Elapsed time: %f seconds." %tempTimeInterval )


def get_toc(pre_string='', verbose=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if verbose:
        if pre_string != '':
            print(pre_string + ": %f seconds." % tempTimeInterval)
        else:
            print("Elapsed time: %f seconds." %tempTimeInterval )
    return tempTimeInterval

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc('',False)


def crop_tensor(img,cropX=None,cropY=None):
    y,x,c = img.shape
    if cropX==None:
        print('Error!!'); #TODO raise official error:
    if cropY==None:
        cropY = cropX;
    startx = x//2-(cropX//2)
    starty = y//2-(cropY//2)
    return img[starty:starty+cropY,startx:startx+cropX]



# ####################################################################################################################################################################################################
# #### Noise Bayer Image: ####
# original_image = read_image_cv2('F:\GOPRO_Large/test\GOPR0384_11_00\sharp/000011.png', flag_convert_to_rgb=1, flag_normalize_to_float=1)
# noisy_image = noise_RGB_through_bayer(original_image)
# figure(1)
# imshow(original_image)
# figure(2)
# imshow(noisy_image)
#
#
#
#
# original_image = read_image_cv2('F:\GOPRO_Large/test\GOPR0384_11_00\sharp/000011.png', flag_convert_to_rgb=1, flag_normalize_to_float=1)
# noisy_image = noise_RGB_through_bayer_no_rand(original_image,gain=32)
# figure(1, figsize=(17,10))
# imshow(original_image)
# figure(2, figsize=(17,10))
# imshow(noisy_image)








####################################################################################################################################################################################################
Image_path = 'F:\TNR - OFFICIAL TEST IMAGES\Dynamic_Videos/NoiseGain100/Night_videowalk_in_East_Shinjuku_Tokyo_CropSize2000_Frames35964-36023/00000000.png'
####################################################################################################################################################################################################




####################################################################################################################################################################################################
#### #Ronny's Function with LARGE BIAS TOWARDS DARK IMAGES: ####
original_image = read_image_cv2(Image_path, flag_convert_to_rgb=1, flag_normalize_to_float=1)
# original_image = crop_tensor(original_image,100,100)
tic()
# noisy_image, gt_rgb_image = noise_RGB_through_bayer(original_image,
#                                       flag_random_gamma = False,
#                                       flag_random_gain = False,
#                                       flag_random_RGB_gain = False,
#                                       flag_random_white_balance = False,
#                                       flag_random_analog_gain = False)

noisy_image, gt_rgb_image = noise_RGB_through_bayer(original_image,
                                      floag_random_gamma = True,
                                      flag_random_gain = True,
                                      flag_random_RGB_gain = True,
                                      flag_random_white_balance = True,
                                      flag_random_analog_gain = True)
toc('noise RGB through bayer with bias towards dark')
figure(1)
imshow(original_image)
figure(2)
imshow(noisy_image)
figure(3)
imshow(gt_rgb_image)


input_numpy = gt_rgb_image
input_numpy_noisy = noisy_image
SSIM_numpy = compare_ssim(input_numpy,input_numpy_noisy,win_size=5,gradient=False,multichannel=True,gaussian_weights=False)
PSNR_numpy = compare_psnr(input_numpy,input_numpy_noisy)
L2_numpy = ( ((input_numpy-input_numpy_noisy)**2).mean() )
STD_numpy = sqrt( ((input_numpy-input_numpy_noisy)**2).mean() )
L1_numpy = (abs(input_numpy-input_numpy_noisy)).mean()
print('SSIM: ' + str(SSIM_numpy) + ', PSNR: ' + str(PSNR_numpy) + ', L1: ' + str(L1_numpy))

####################################################################################################################################################################################################





####################################################################################################################################################################################################
### Correct Way to noise image going through bayer but without random gain/gamma ###
original_image = read_image_cv2(Image_path, flag_convert_to_rgb=1, flag_normalize_to_float=1)
# original_image = crop_tensor(original_image,100,100)
tic()
noisy_image = noise_RGB_through_bayer_no_rand(original_image,gain=32)
toc('noise RGB through bay')
figure(1, figsize=(17,10))
imshow(original_image)
figure(2, figsize=(17,10))
imshow(noisy_image)
####################################################################################################################################################################################################





####################################################################################################################################################################################################
### Correct Way to noise image going through bayer but without random gain/gamma & OUTPUT IS IN RONY'S MODEL FORMAT (BAYER+STD ESTIMATE)###
original_image = read_image_cv2(Image_path, flag_convert_to_rgb=1, flag_normalize_to_float=1)
# original_image = crop_tensor(original_image,100,100)
tic()
noisy_image = noise_RGB_through_Bayer_NoDarkBias_SIDFormat(original_image, gain=100, flag_clip=True)
toc('noise RGB through bay')
figure(1, figsize=(17,10))
imshow(original_image)
figure(2, figsize=(17,10))
imshow(noisy_image)
####################################################################################################################################################################################################






### Correct Way to noise image going through bayer but without random gain/gamma & OUTPUT IS IN RONY'S MODEL FORMAT (BAYER+STD ESTIMATE)###
noisy_path = 'F:\TNR - OFFICIAL TEST IMAGES\Still Images/NoiseGain_100/000005/024.png'
RGB_noisy = read_image_cv2(noisy_path, flag_convert_to_rgb=1, flag_normalize_to_float=1)
# original_image = crop_tensor(original_image,100,100)
tic()
bayer_std_packed, img_gt_extended, noisy_bayer = get_noisy_RGB_image_and_output_SIDFormat(RGB_noisy, gain=100, flag_clip=True, flag_output_uint8_or_float='float')

RGB_noisy_reconstructed_from_bayer = demosaicing_CFA_Bayer_Menon2007(noisy_bayer,'GRBG')
toc('noise RGB through bay')
figure(1, figsize=(17,10))
imshow(RGB_noisy)
figure(2, figsize=(17,10))
imshow(RGB_noisy_reconstructed_from_bayer)
####################################################################################################################################################################################################







####################################################################################################################################################################################################
### Most simple "stupid" way to add pseudo-shot-noise: ###
original_image = read_image_cv2(Image_path, flag_convert_to_rgb=1, flag_normalize_to_float=0)
# original_image = crop_tensor(original_image,100,100)
tic()
noisy_image = noise_RGB_LinearShotNoise(original_image, gain=0.000000000000000001, flag_output_uint8_or_float='uint8')
toc()
figure(1)
imshow(original_image)
figure(2)
imshow(noisy_image)
np.std((np.float32(original_image)-np.float32(noisy_image))/255)
print('Images Residual STD: ' + str(np.std((float32(noisy_image)-float32(original_image))/255)) )

# import os
# cv.imwrite(os.path.join('C:/Users\dkarl\PycharmProjects\dudykarl','clean.png'), cv.cvtColor(noisy_image.squeeze() * 1, cv.COLOR_BGR2RGB))


# save_image_numpy(folder_path='C:/Users\dkarl\PycharmProjects\dudykarl',filename='100.png',numpy_array=noisy_image,flag_convert_bgr2rgb=True, flag_scale=True)
####################################################################################################################################################################################################








####################################################################################################################################################################################################
### Most simple "stupid" way to add pseudo-shot-noise <-> Testing ON BATCHES!: ###
original_image = read_image_cv2(Image_path, flag_convert_to_rgb=1, flag_normalize_to_float=0)
original_image_expanded = np.expand_dims(original_image,0)
original_images = np.concatenate((original_image_expanded,original_image_expanded), axis=0)
# original_image = crop_tensor(original_image,100,100)
tic()
noisy_image = noise_RGB_LinearShotNoise(original_images, gain=30, flag_output_uint8_or_float='uint8')
toc()
figure(1)
imshow(original_image)
figure(2)
imshow(noisy_image[1,:,:,:])

print('Images Residual STD: ' + str(np.std((np.float32(noisy_image)-np.float32(original_image))/255)) )
####################################################################################################################################################################################################






####################################################################################################################################################################################################
### Most simple "stupid" way to add pseudo-shot-noise ON TENSORS <-> Testing ON TENSOR BATCHES!: ###
original_image = read_image_cv2(Image_path, flag_convert_to_rgb=1, flag_normalize_to_float=0)
original_image_expanded = np.expand_dims(original_image,0)
original_images = np.concatenate((original_image_expanded,original_image_expanded), axis=0)
original_images = torch.Tensor(original_images).type(torch.uint8)
# original_image = crop_tensor(original_image,100,100)
tic()
noisy_image = noise_RGB_LinearShotNoise_Torch(original_images, gain=30, flag_output_uint8_or_float='uint8')
toc()
figure(1)
imshow(original_image)
figure(2)
imshow(noisy_image[1,:,:,:])

print('Images Residual STD: ' + str(np.std((float32(noisy_image)-float32(original_image))/255)) )
####################################################################################################################################################################################################








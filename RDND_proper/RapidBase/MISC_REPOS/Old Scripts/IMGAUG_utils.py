#Imports:
#(1). Auxiliary:
from __future__ import print_function
import PIL
import argparse
import os
import numpy as np
import math
from PIL import Image
import glob
import random
import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int,importlib
from int_range import *
from int_arange import *
from mat_range import *
from matlab_arange import *
from my_linspace import *
from my_linspace_int import *
import get_center_number_of_pixels
from get_center_number_of_pixels import *
import get_speckle_sequences
from get_speckle_sequences import *
import tic_toc
from tic_toc import *
import search_file
from search_file import *
import show_matrices_video
from show_matrices_video import *
import klepto_functions
from klepto_functions import *
from collections import OrderedDict
import sys
import math
from datetime import datetime
import cv2
from skimage.measure import compare_ssim
from shutil import get_terminal_size
import math
import pickle
import random
import numpy as np
import lmdb
import ctypes  # An included library with Python install.
from numpy import power as power
from numpy import exp as exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import time #time.time()
import nltk
import collections
import re
from csv import reader
import tarfile
from pandas import read_csv
from pandas import Series
import collections
#counter = collections.Counter()
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.ar_model import AR
import argparse
import pydot
import psutil
# import graphviz
# import pydot_ng
length = len #use length instead of len.... make sure it doesn't cause problems

#Pylab imshow stuff:
import pylab
from pylab import imshow, pause, draw, title, axes, ylabel, ylim, yticks, xlabel, xlim, xticks
from pylab import colorbar, colormaps, subplot, suptitle, plot
#Message Box:
def message_box(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
import pymsgbox
from pymsgbox import alert as alert_box
from pymsgbox import confirm as confirm_box
from pymsgbox import prompt as prompt_box

#(2). TorchVision (add FastAi stuff which are much faster and more intuitive as far as augmentations)
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import vgg19
from torchvision.utils import make_grid
#(3). Torch Utils:
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
#(4). Torch NN:
import torch.nn as nn
import torch.nn.functional as F
import torch
#(5). More Torch Stuff:
import torch
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import torch.utils.data as data

#TensorFlow:
import tensorflow as tf
import tensorboard as TB
import tensorboardX as TBX
from tensorboardX import SummaryWriter

#ESRGAN:
from ESRGAN_utils import *
from ESRGAN_deep_utils import *
# from ESRGAN_basic_Blocks_and_Layers import *
# from ESRGAN_Models import *
# from ESRGAN_Losses import *
# from ESRGAN_Optimizers import *
# from ESRGAN_Visualizers import *
# from ESRGAN_dataset import *
# from ESRGAN_OPT import *

import imgaug
from imgaug import augmenters as iaa


def get_IMGAUG_parameters():
    #############################################################################################################################################
    ###################################  Which Augmentations To Do HQ:  #####################################
    #(*). Note: I think a better thing would be to define the initial and final cropping operations (should one wishes them to exist) exhaugenously to this script

    #(1). BLUR:
    flag_gaussian_blur = False;
    flag_average_blur = False
    flag_median_blur = False
    flag_bilateral_blur = False
    #(2). Arithmatic:
    flag_add_value_all_pixels = False;
    flag_add_value_per_pixel = False;
    flag_multiply_all_pixels = False;
    flag_multiple_per_pixel = False;
    #(3). Noise:
    flag_additive_gaussian_noise = False
    flag_salt_and_pepper = False;
    flag_coarse_salt_and_pepper = False
    flag_salt = False
    flag_coarse_salt = False
    flag_pepper = False
    flag_coarse_pepper = False
    #TODO: add speckle noise
    #(4). Dropout:
    flag_dropout = False
    flag_coarse_dropout = False
    flag_replace_elementwise = False;
    #(5). Overlay Images:
    flag_overlay_alpha = False
    flag_overlay_alpha_elementwise = False
    flag_overlay_simplex_noise_alpha = False
    flag_overlap_frequency_noise_alpha = False
    #(6). Flip/Transpose:
    flag_horizontal_flip = False;
    flag_vertical_flip = False;
    flag_transpose = False; #Not impelemented - make a Lambda function
    #(7). Geometric:
    flag_affine_transform = False #Scale, Translation, Rotation, Shear,
    flag_piecewise_affine_transform = False
    flag_perspective_transform = False;
    flag_elastic_jitter_transform = False;
    #TODO: add optical and "true" elastic transform
    #(8). Color:
    flag_HSV_transform = False
    flag_colorspace_transform = False
    flag_to_grayscale = False
    flag_invert_intensity = False
    #TODO: add RGB shift
    #(9). Contrast (and gamma):
    flag_gamma_contrast = False
    flag_sigmoid_contrast = False
    flag_log_contrast = False
    flag_linear_contrast = False
    flag_contrast_normalization = False
    #(10). Low/High Frequency Emphasis:
    flag_sharpen = False
    flag_emboss = False
    flag_edge_detect = False
    flag_directed_edge_detect = False
    #(11). Convolve:
    flag_convolve = False
    #(12). JPEG Compression:
    flag_JPEF_compression = False
    #############################################################################################################################################





    #############################################################################################################################################
    ###################################  At What Probabilities to make the Augmentations::  #####################################
    # (1). BLUR:
    probability_of_gaussian_blur = 0.5
    probability_of_average_blur = 0.5
    probability_of_median_blur = 0.5
    probability_of_bilateral_blur = 0.5
    # (2). Arithmatic:
    probability_of_add_value_all_pixels = 0.5;
    probability_of_add_value_per_pixel = 0.5;
    probability_of_multiply_all_pixels = 0.5;
    probability_of_multiple_per_pixel = 0.5;
    # (3). Noise:
    probability_of_additive_gaussian_noise = 0.5
    probability_of_salt_and_pepper = 0.5;
    probability_of_coarse_salt_and_pepper = 0.5
    probability_of_salt = 0.5
    probability_of_coarse_salt = 0.5
    probability_of_pepper = 0.5
    probability_of_coarse_pepper = 0.5
    # (4). Dropout:
    probability_of_dropout = 0.5
    probability_of_coarse_dropout = 0.5
    probability_of_replace_elementwise = 0.5;
    # (5). Overlay Images:
    probability_of_overlay_alpha = 0.5
    probability_of_overlay_alpha_elementwise = 0.5
    probability_of_overlay_simplex_noise_alpha = 0.5
    probability_of_overlap_frequency_noise_alpha = 0.5
    # (6). Flip/Transpose:
    probability_of_horizontal_flip = 0.5;
    probability_of_vertical_flip = 0.5;
    probability_of_transpose = 0.5;
    # (7). Geometric:
    probability_of_affine_transform = 0.5
    probability_of_piecewise_affine_transform = 0.5
    probability_of_perspective_transform = 0.5;
    probability_of_elastic_jitter_transform = 0.5;
    # (8). Color:
    probability_of_HSV_transform = 0.5
    probability_of_colorspace_transform = 0.5
    probability_of_to_grayscale = 0.5
    probability_of_invert_intensity = 0.5
    # (9). Contrast (and gamma):
    probability_of_gamma_contrast = 0.5
    probability_of_sigmoid_contrast = 0.5
    probability_of_log_contrast = 0.5
    probability_of_linear_contrast = 0.5
    probability_of_contrast_normalization = 0.5
    # (10). Low/High Frequency Emphasis:
    probability_of_sharpen = 0.5
    probability_of_emboss = 0.5
    probability_of_edge_detect = 0.5
    probability_of_directed_edge_detect = 0.5
    # (11). Convolve:
    probability_of_convolve = 0.5
    # (12). JPEG Compression:
    probability_of_JPEF_compression = 0.5
    #############################################################################################################################################





    #############################################################################################################################################
    ###################################  Augmentation Parameters HQ:  #####################################
    #### BLUR: ####
    # (1). Gaussian Blur:
    gaussian_blur_kernel_size = 5;
    # (2). Average Blur:
    average_blur_kernel_size = 5;
    # (3). Median Blur:
    median_blur_kernel_size = 5  # k must be an odd number
    # (4). Bilateral Blur:
    bilateral_blur_kernel_size = 5;
    sigma_color = (10, 250);
    sigma_space = (10, 250);

    #### Arithmatic: ####
    # (1). Add value all pixels:
    value_to_add_all_pixels = 0;
    flag_add_value_all_pixels_per_channel = False;
    # (2). Add value per pixel:
    value_to_add_per_pixel = 0;
    flag_add_value_per_pixel_per_channel = False;
    # (3). Multiply value all pixels:
    value_to_multiply_all_pixels = 1;
    flag_multiply_value_all_pixels_per_channel = False
    # (4). Multiply value per pixel:
    value_to_multiply_per_pixel = 1;
    flag_multiply_value_per_pixel_per_channel = False;

    #### Noise: ####
    # (1). Additive Gaussian Noise:
    additive_gaussian_noise_mu = 0;
    additive_gaussian_noise_sigma = 10;
    flag_additive_gaussian_noise_per_channel = 1;
    # (2). Salt and Pepper:
    salt_and_pepper_noise_percent_of_pixels = 0.05;
    flag_salt_and_pepper_noise_per_channel = 0;
    # (3). Coarse Salt and Pepper:
    coarse_salt_and_pepper_noise_percent_of_pixels = 0.05;
    coarse_salt_and_pepper_noise_pixel_sizes = None;
    coarse_salt_and_pepper_noise_percent_sizes = (0, 0.1);
    coarse_salt_and_pepper_noise_minimum_size = 4;
    flag_coarse_salt_and_pepper_noise_per_channel = False;
    # (4). Salt Noise:
    salt_noise_percent_of_pixels = 0;
    flag_salt_noise_per_channel = False;
    # (5). Coarse Salt Noise:
    coarse_salt_noise_percent_of_pixels = 0.05;
    coarse_salt_noise_pixel_sizes = None;
    coarse_salt_noise_percent_sizes = (0, 0.1);
    coarse_salt_noise_minimum_size = 4;
    flag_coarse_salt_noise_per_channel = False;
    # (6). Pepper Noise:
    pepper_noise_percent_of_pixels = 0;
    flag_pepper_noise_per_channel = False;
    # (7). Coarse Pepper Noise:
    coarse_pepper_noise_percent_of_pixels = 0.05;
    coarse_pepper_noise_pixel_sizes = None;
    coarse_pepper_noise_percent_sizes = (0, 0.1);
    coarse_pepper_noise_minimum_size = 4;
    flag_coarse_pepper_noise_per_channel = False;

    #### Dropout/Replacement: ####
    # (1). Dropout:
    dropout_percent_of_pixels = 0;
    flag_dropout_per_channel = False;
    # (2). Coarse Dropout:
    coarse_dropout_percent_of_pixels = 0;
    coarse_dropout_pixel_sizes = None
    coarse_dropout_percent_sizes = 0;
    coarse_dropout_minimum_size = 4
    flag_coarse_dropout_per_channel = False;
    # (3). Replace ElementWise:
    replace_elementwise_indices_to_change_mask = 0;
    replace_elementwise_replacement_mask = 0;
    flag_replace_elementwise_per_channel = False;

    #### Overlay Images: ####
    # (1). Overlay Alpha:
    overlay_factor = 0;
    overlay_first = None;
    overlay_second = None;
    flag_overlay_per_channel = False;
    # (2). Overlay Alpha ElementWise:
    overlay_elementwise_factor = 0;
    overlay_elementwise_first = None
    overlay_elementwise_second = None;
    flag_overlay_elementwise_per_channel = False;
    # (3). Simplex Noise Alpha:
    simplex_noise_first = None;
    simplex_noise_second = None;
    simplex_noise_max_pixel_sizes = (2, 16);
    simplex_noise_upscale_method = None;
    simplex_noise_number_of_iterations = (1, 3);
    simplex_noise_aggregation_method = 'max';
    simplex_noise_sigmoid = True;
    simplex_noise_sigmoid_threshold = None;
    flag_simplex_noise_per_channel = False;
    # (4). Frequency Noise Alpha:
    frequency_noise_exponent = (-4, 4);
    frequency_noise_first = None
    frequency_noise_second = None
    frequency_noise_max_pixel_sizes = (4, 16);
    frequency_noise_upscale_method = None
    frequency_noise_number_of_iterations = (1, 3);
    frequency_noise_aggregation_method = ['avg', 'max'];
    frequency_noise_sigma = 0.5;
    frequency_noise_sigma_threshold = None;
    flag_frequency_noise_per_channel = False;

    #### Flip/Transpose: ####
    # (1). Horizontal Flip:
    horizontal_flip_probability = 0.5;
    # (2). Vertical Flip:
    vertical_flip_probability = 0.5;

    #### Geometric: ####
    # (1). Affine:
    affine_scale = 1;
    affine_translation_percent = 0;
    affine_translation_number_of_pixels = 0;
    affine_rotation_degrees = 30;
    affine_shear_degrees = 30;
    affine_order = cv2.INTER_CUBIC;
    affine_mode = cv2.BORDER_REFLECT_101;
    affine_constant_value_outside_valid_range = 0;
    # (2). PieceWise Affine:
    piecewise_affine_scale = 0;
    piecewise_affine_number_of_rows = 10;
    piecewise_affine_number_of_cols = 10;
    piecewise_affine_order = 1;
    piecewise_affine_mode = 'reflect';
    piecewise_affine_constant_value_outside_valid_range = 0;
    flag_piecewise_affine_absolute_scale = False;
    # (3). Perspective Transform:
    perspective_transform_scale = 0;
    flag_perspective_transform_keep_size = True;
    # (4). Elastic Jitter Transform:
    elastic_jitter_alpha = 0;
    elastic_jitter_sigma = 0;
    elastic_jitter_order = 3;
    elastic_jitter_mode = 'reflect'
    elastic_jitter_constant_value_outside_valid_range = 0

    #### Color: ####
    # (1). Shift HSV:
    shift_HSV_hs_value = 0;
    shift_HSV_from_colorspace = 'RGB';
    shift_HSV_channels = [0, 1]
    flag_shift_HSV_per_channel = False;
    # (2). Invert Intensity:
    invert_intensity_probability = 0;
    invert_intensity_min_value = 0;
    invert_intensity_max_value = 1;
    flag_invert_intensity_per_channel = False;
    # (3). To GrayScale (bottle-neck):
    to_grayscale_alpha = 0;
    to_grayscale_from_colorspace = 'RGB';

    #### Contrast (and gamma): ####
    # (1). gamma contrast:
    gamma_contrast = 1;
    flag_gamma_per_channel = False;
    # (2). sigmoid contrast:
    sigmoid_contrast_gain = 10;
    sigmoid_contrast_cutoff = 0.5;
    flag_sigmoid_contrast_per_channel = False;
    # (3). Log Contrast:
    log_contrast_gain = 1;
    flag_log_contrast_per_channel = False;
    # (4). Linear Contrast:
    linear_contrast_gain = 1;
    flag_linear_contrast_per_channel = False;
    # (5). Contrast Normalization:
    contrast_normalization_alpha = 1;
    flag_contrast_normalization_per_channel = False;

    #### Low/High Frequency Emphasis: ####
    # (1). Sharpen:
    sharpen_alpha = 1;
    sharpen_lightness = 1;
    # (2). Emboss:
    emboss_alpha = 0;
    emboss_strength = 1;
    # (3). Edge Detect:
    edge_detect_alpha = 0;
    # (4). Directed Edge Detect:
    directed_edge_detect_alpha = 0;
    directed_edge_detect_direction = (0, 1);

    #### Convolve: ####
    convolve_matrix_to_convolve = 0;

    #### JPEG Compression Degredation: ####
    JPEG_compression_level_of_compression = 90;
    #############################################################################################################################################





    ######################################################
    ######################################################
    ####  Get Dictionary Of Above Defined Parameters: ####
    scope_dictionary = locals();
    return scope_dictionary;





####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
def get_IMGAUG_Random_Crop_Location_Augmenter(crop_size):
    crop_size = double_tuple(crop_size)
    return iaa.CropToFixedSize(width=crop_size[0], height=crop_size[1], name=None, deterministic=False); #Don't forget to later change deterministic properties if you want to augment image and label the same way:

def get_IMGAUG_Random_Stretch_Random_Crop_Location_Augmenter(stretch_factor_range, crop_size):
    crop_size = double_tuple(crop_size);
    #Crop --> Resize --> Crop (a little):
    #(1). if we want to stretch/expand then we need to crop to initial_crop_size = final_crop_size*(1-stretch_factor), then simply resize up (stretch)
    #(2). if we want to squish/shrink then we need to crop to initial_crop_size = final_crop_size*(1+squish_factor), then simply resize down (shrink)
    #(3). General Solution: crop to initial_crop_size = final_crop_size*(1+max_squish_factor) , then zoom-in/out, then
    if stretch_factor>1:
        1;



def get_IMGAUG_Crop_Scale_Pad_Parameters(): #Reference Function For Scale/Size Changing Augmentations
    ########  Random Location Crop:  #########
    # Random location but Deterministic Crop Size:
    iaa.CropToFixedSize(width=width, height=height, name=None, deterministic=False)

    ########  Deterministic/Random Resize:  #########
    # Deterministic/Random Resize (depending on input variables)
    iaa.Scale(size=size, interpolation='cubic', name=None, deterministic=False)

    ####### ZOOM in/out, Stretch/Squish Using Affine Transform (Doesn't Change Number Of Pixels!!!!): ##########
    iaa.AffineCv2(scale=(0.5, 1.5), translate_percent=None, translate_px=None, rotate=0, shear=0, order=cv2.INTER_CUBIC,
                  cval=0, mode=cv2.BORDER_REFLECT_101, name=None, deterministic=False);

    ########  Center Crop/Pad:  #########
    # Deterministic/Random Center(!) Cropping/Padding (depending on input variables):
    iaa.CropAndPad(px=None, percent=None, pad_mode='constant', pad_cval=0, keep_size=True, sample_independently=True,
                   name=None, deterministic=False)
    # Deterministic/Random Center Crop:
    iaa.Crop(px=None, percent=None, keep_size=True, sample_independently=True, name=None, deterministic=False)
    # Deterministic/Random Center Pad:
    iaa.Pad(px=None, percent=None, pad_mode='constant', pad_cval=0, keep_size=True, sample_independently=True,
            name=None, deterministic=False)

    ########  Padding:  #########
    # Expand/Pad if needed to specific size:
    iaa.PadToFixedSize(width=width, height=height, pad_mode='constant', pad_cval=0, name=None, deterministic=False)
    # Always return a specific size no metter the original size of the image:
    iaa.Sequential[iaa.PadToFixedSize(100, 100, 'reflect'), iaa.CropToFixedSize(100, 100)]
#######################################################################################################################################################################################################################################################################################################################################################################################################################################







def get_IMGAUG_Augmenter(parameters_dictionary):
    ######################### Augmenters to remember: ######################################
    # #(*). META/WRAPPER:
    # iaa.Sequential(children=None, random_order=False, name=None, deterministic=False)
    # iaa.SomeOf(n=None, children=None, random_order=False, name=None, deterministic=False)
    # iaa.OneOf(children=children_list, name=None, deterministic=False)
    # iaa.Sometimes(p=0, then_list=None, else_list=None, name=None, deterministic=False)
    # iaa.WithChannels(channels=None, children=None, name=None, deterministic=False)
    # iaa.Noop(name=None, deterministic=False)
    # iaa.Lambda(func_images=func_images, func_heatmaps=func_heatmaps, func_keypoints=func_keypoints, name=None,
    #            deterministic=False)
    # iaa.AssertLambda
    # iaa.AssertShape
    #



    #Initialize Transformations List:
    Transformations_List = [];


    #### Initial Crop: ######




    #########################
    # (1). BLUR:
    if parameters_dictionary['flag_gaussian_blur']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_gaussian_blur'],iaa.GaussianBlur(sigma=parameters_dictionary['gaussian_blur_kernel_size'], name=None, deterministic=False)))
    if parameters_dictionary['flag_average_blur']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_average_blur'],iaa.AverageBlur(k=parameters_dictionary['average_blur_kernel_size'], name=None, deterministic=False)))
    if parameters_dictionary['flag_median_blur']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_median_blur'],iaa.MedianBlur(k=parameters_dictionary['median_blur_kernel_size'], name=None, deterministic=False)))
    if parameters_dictionary['flag_bilateral_blur']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_bilateral_blur'],iaa.BilateralBlur(d=parameters_dictionary['bilateral_blur_kernel_size'],
                                                      sigma_color=parameters_dictionary['sigma_color'],
                                                      sigma_space=parameters_dictionary['sigma_space'], name=None,
                                                      deterministic=False)))

    # (2). Arithmatic:
    if parameters_dictionary['flag_add_value_all_pixels']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_add_value_all_pixels'],iaa.Add(value=parameters_dictionary['value_to_add_all_pixels'],
                                            per_channel=parameters_dictionary['flag_add_value_all_pixels_per_channel'],
                                            name=None, deterministic=False)))
    if parameters_dictionary['flag_add_value_per_pixel']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_add_value_per_pixel'],iaa.AddElementwise(value=parameters_dictionary['value_to_add_per_pixel'],
                                                       per_channel=parameters_dictionary[
                                                           'flag_add_value_per_pixel_per_channel'], name=None,
                                                       deterministic=False)))
    if parameters_dictionary['flag_multiply_all_pixels']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_multiply_all_pixels'],iaa.Multiply(mul=parameters_dictionary['value_to_multiply_all_pixels'],
                                                 per_channel=parameters_dictionary[
                                                     'flag_multiply_value_all_pixels_per_channel'], name=None,
                                                 deterministic=False)))
    if parameters_dictionary['flag_multiple_per_pixel']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_multiple_per_pixel'],iaa.MultiplyElementwise(mul=parameters_dictionary['value_to_multiply_per_pixel'],
                                                            per_channel=parameters_dictionary[
                                                                'flag_multiply_value_per_pixel_per_channel'], name=None,
                                                            deterministic=False)))

    # (3). Noise:
    if parameters_dictionary['flag_additive_gaussian_noise']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_additive_gaussian_noise'],iaa.AdditiveGaussianNoise(loc=parameters_dictionary['additive_gaussian_noise_mu'],
                                                              scale=parameters_dictionary[
                                                                  'additive_gaussian_noise_sigma'],
                                                              per_channel=parameters_dictionary[
                                                                  'flag_additive_gaussian_noise_per_channel'],
                                                              name=None, deterministic=False)))
    if parameters_dictionary['flag_salt_and_pepper']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_salt_and_pepper'],iaa.SaltAndPepper(p=parameters_dictionary['salt_and_pepper_noise_percent_of_pixels'],
                              per_channel=parameters_dictionary['flag_salt_and_pepper_noise_per_channel'], name=None,
                              deterministic=False)))
    if parameters_dictionary['flag_coarse_salt_and_pepper']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_coarse_salt_and_pepper'],iaa.CoarseSaltAndPepper(p=parameters_dictionary['coarse_salt_and_pepper_noise_percent_of_pixels'],
                                    size_px=parameters_dictionary['coarse_salt_and_pepper_noise_pixel_sizes'],
                                    size_percent=parameters_dictionary['coarse_salt_and_pepper_noise_percent_sizes'],
                                    per_channel=parameters_dictionary['flag_coarse_salt_and_pepper_noise_per_channel'],
                                    min_size=parameters_dictionary['coarse_salt_and_pepper_noise_minimum_size'],
                                    name=None, deterministic=False)))
    if parameters_dictionary['flag_salt']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_salt'],iaa.Salt(p=parameters_dictionary['salt_noise_percent_of_pixels'],
                                             per_channel=parameters_dictionary['flag_salt_noise_per_channel'],
                                             name=None, deterministic=False)))
    if parameters_dictionary['flag_coarse_salt']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_coarse_salt'],iaa.CoarseSalt(p=parameters_dictionary['coarse_salt_noise_percent_of_pixels'],
                                                   size_px=parameters_dictionary['coarse_salt_noise_pixel_sizes'],
                                                   size_percent=parameters_dictionary[
                                                       'coarse_salt_noise_percent_sizes'],
                                                   per_channel=parameters_dictionary[
                                                       'flag_coarse_salt_noise_per_channel'],
                                                   min_size=parameters_dictionary['coarse_salt_noise_minimum_size'],
                                                   name=None, deterministic=False)))
    if parameters_dictionary['flag_pepper']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_pepper'],iaa.Pepper(p=parameters_dictionary['pepper_noise_percent_of_pixels'],
                                               per_channel=parameters_dictionary['flag_pepper_noise_per_channel'],
                                               name=None, deterministic=False)))
    if parameters_dictionary['flag_coarse_pepper']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_coarse_pepper'],iaa.CoarsePepper(p=parameters_dictionary['coarse_pepper_noise_percent_of_pixels'],
                                                     size_px=parameters_dictionary['coarse_pepper_noise_pixel_sizes'],
                                                     size_percent=parameters_dictionary[
                                                         'coarse_pepper_noise_percent_sizes'],
                                                     per_channel=parameters_dictionary[
                                                         'flag_coarse_pepper_noise_per_channel'],
                                                     min_size=parameters_dictionary['coarse_pepper_noise_minimum_size'],
                                                     name=None, deterministic=False)))

    # (4). Dropout:
    if parameters_dictionary['flag_dropout']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_dropout'],iaa.Dropout(p=parameters_dictionary['dropout_percent_of_pixels'],
                                                per_channel=parameters_dictionary['flag_dropout_per_channel'],
                                                name=None, deterministic=False)))
    if parameters_dictionary['flag_coarse_dropout']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_coarse_dropout'],iaa.CoarseDropout(p=parameters_dictionary['coarse_dropout_percent_of_pixels'],
                                                      size_px=parameters_dictionary['coarse_dropout_pixel_sizes'],
                                                      size_percent=parameters_dictionary[
                                                          'coarse_dropout_percent_sizes'],
                                                      per_channel=parameters_dictionary[
                                                          'flag_coarse_dropout_per_channel'],
                                                      min_size=parameters_dictionary['coarse_dropout_minimum_size'],
                                                      name=None, deterministic=False)))
    if parameters_dictionary['flag_replace_elementwise']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_replace_elementwise'],iaa.ReplaceElementwise(mask=parameters_dictionary['replace_elementwise_indices_to_change_mask'],
                                   replacement=parameters_dictionary['replace_elementwise_replacement_mask'],
                                   per_channel=parameters_dictionary['flag_replace_elementwise_per_channel'], name=None,
                                   deterministic=False)))

    # (5). Overlay Images:
    if parameters_dictionary['flag_overlay_alpha']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_overlay_alpha'],iaa.Alpha(factor=parameters_dictionary['overlay_factor'], first=parameters_dictionary['overlay_first'],
                      second=parameters_dictionary['overlay_second'],
                      per_channel=parameters_dictionary['flag_overlay_per_channel'], name=None, deterministic=False)))
    if parameters_dictionary['flag_overlay_alpha_elementwise']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_overlay_alpha_elementwise'],iaa.AlphaElementwise(factor=parameters_dictionary['overlay_elementwise_factor'],
                                                         first=parameters_dictionary['overlay_elementwise_first'],
                                                         second=parameters_dictionary['overlay_elementwise_second'],
                                                         per_channel=parameters_dictionary[
                                                             'flag_overlay_elementwise_per_channel'], name=None,
                                                         deterministic=False)))
    if parameters_dictionary['flag_overlay_simplex_noise_alpha']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_overlay_simplex_noise_alpha'],iaa.SimplexNoiseAlpha(first=parameters_dictionary['simplex_noise_first'],
                                                          second=parameters_dictionary['simplex_noise_second'],
                                                          per_channel=parameters_dictionary[
                                                              'flag_simplex_noise_per_channel'],
                                                          size_px_max=parameters_dictionary[
                                                              'simplex_noise_max_pixel_sizes'],
                                                          upscale_method=parameters_dictionary[
                                                              'simplex_noise_upscale_method'],
                                                          iterations=parameters_dictionary[
                                                              'simplex_noise_number_of_iterations'],
                                                          aggregation_method=parameters_dictionary[
                                                              'simplex_noise_aggregation_method'],
                                                          sigmoid=parameters_dictionary['simplex_noise_sigmoid'],
                                                          sigmoid_thresh=parameters_dictionary[
                                                              'simplex_noise_sigmoid_threshold'], name=None,
                                                          deterministic=False)))
    if parameters_dictionary['flag_overlap_frequency_noise_alpha']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_overlap_frequency_noise_alpha'],iaa.FrequencyNoiseAlpha(exponent=parameters_dictionary['frequency_noise_exponent'],
                                                            first=parameters_dictionary['frequency_noise_first'],
                                                            second=parameters_dictionary['frequency_noise_second'],
                                                            per_channel=parameters_dictionary[
                                                                'flag_frequency_noise_per_channel'],
                                                            size_px_max=parameters_dictionary[
                                                                'frequency_noise_max_pixel_sizes'],
                                                            upscale_method=parameters_dictionary[
                                                                'frequency_noise_upscale_method'],
                                                            iterations=parameters_dictionary[
                                                                'frequency_noise_number_of_iterations'],
                                                            aggregation_method=parameters_dictionary[
                                                                'frequency_noise_aggregation_method'],
                                                            sigmoid=parameters_dictionary['frequency_noise_sigma'],
                                                            sigmoid_thresh=parameters_dictionary[
                                                                'frequency_noise_sigma_threshold'], name=None,
                                                            deterministic=False)))

    # (6). Flip/Transpose:
    if parameters_dictionary['flag_horizontal_flip']:
        Transformations_List.append(
            iaa.Fliplr(p=parameters_dictionary['horizontal_flip_probability'], name=None, deterministic=False))
    if parameters_dictionary['flag_vertical_flip']:
        Transformations_List.append(
            iaa.Flipud(p=parameters_dictionary['vertical_flip_probability'], name=None, deterministic=False))

    # (7). Geometric:
    if parameters_dictionary['flag_affine_transform']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_affine_transform'],iaa.AffineCv2(scale=parameters_dictionary['affine_scale'],
                                                  translate_percent=parameters_dictionary['affine_translation_percent'],
                                                  translate_px=parameters_dictionary[
                                                      'affine_translation_number_of_pixels'],
                                                  rotate=parameters_dictionary['affine_rotation_degrees'],
                                                  shear=parameters_dictionary['affine_shear_degrees'],
                                                  order=parameters_dictionary['affine_order'],
                                                  cval=parameters_dictionary[
                                                      'affine_constant_value_outside_valid_range'],
                                                  mode=parameters_dictionary['affine_mode'], name=None,
                                                  deterministic=False)))
    if parameters_dictionary['flag_piecewise_affine_transform']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_piecewise_affine_transform'],iaa.PiecewiseAffine(scale=parameters_dictionary['piecewise_affine_scale'],
                                                        nb_rows=parameters_dictionary[
                                                            'piecewise_affine_number_of_rows'],
                                                        nb_cols=parameters_dictionary[
                                                            'piecewise_affine_number_of_cols'],
                                                        order=parameters_dictionary['piecewise_affine_order'],
                                                        cval=parameters_dictionary[
                                                            'piecewise_affine_constant_value_outside_valid_range'],
                                                        mode=parameters_dictionary['piecewise_affine_mode'],
                                                        absolute_scale=parameters_dictionary[
                                                            'flag_piecewise_affine_absolute_scale'],
                                                        name=None, deterministic=False)))
    if parameters_dictionary['flag_perspective_transform']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_perspective_transform'],iaa.PerspectiveTransform(scale=parameters_dictionary['perspective_transform_scale'],
                                                             keep_size=parameters_dictionary[
                                                                 'flag_perspective_transform_keep_size'], name=None,
                                                             deterministic=False)))
    if parameters_dictionary['flag_elastic_jitter_transform']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_elastic_jitter_transform'],iaa.ElasticTransformation(alpha=parameters_dictionary['elastic_jitter_alpha'],
                                                              sigma=parameters_dictionary['elastic_jitter_sigma'],
                                                              order=parameters_dictionary['elastic_jitter_order'],
                                                              cval=parameters_dictionary[
                                                                  'elastic_jitter_constant_value_outside_valid_range'],
                                                              mode=parameters_dictionary['elastic_jitter_mode'],
                                                              name=None, deterministic=False)))

    # (8). Color:
    if parameters_dictionary['flag_HSV_transform']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_HSV_transform'],iaa.AddToHueAndSaturation(value=parameters_dictionary['shift_HSV_hs_value'],
                                                              per_channel=parameters_dictionary[
                                                                  'flag_shift_HSV_per_channel'],
                                                              from_colorspace=parameters_dictionary[
                                                                  'shift_HSV_from_colorspace'],
                                                              channels=parameters_dictionary['shift_HSV_channels'])))
    if parameters_dictionary['flag_to_grayscale']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_to_grayscale'],iaa.Grayscale(alpha=parameters_dictionary['to_grayscale_alpha'],
                                                  from_colorspace=parameters_dictionary['to_grayscale_from_colorspace'],
                                                  name=None, deterministic=False)))
    if parameters_dictionary['flag_invert_intensity']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_invert_intensity'],iaa.Invert(p=parameters_dictionary['invert_intensity_probability'],
                                               per_channel=parameters_dictionary['flag_invert_intensity_per_channel'],
                                               min_value=parameters_dictionary['invert_intensity_min_value'],
                                               max_value=parameters_dictionary['invert_intensity_max_value'], name=None,
                                               deterministic=False)))

    # (9). Contrast (and gamma):
    if parameters_dictionary['flag_gamma_contrast']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_gamma_contrast'],iaa.GammaContrast(gamma=parameters_dictionary['gamma_contrast'],
                                                      per_channel=parameters_dictionary['flag_gamma_per_channel'],
                                                      name=None, deterministic=False)))
    if parameters_dictionary['flag_sigmoid_contrast']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_sigmoid_contrast'],iaa.SigmoidContrast(gain=parameters_dictionary['sigmoid_contrast_gain'],
                                                        cutoff=parameters_dictionary['sigmoid_contrast_cutoff'],
                                                        per_channel=parameters_dictionary[
                                                            'flag_sigmoid_contrast_per_channel'], name=None,
                                                        deterministic=False)))
    if parameters_dictionary['flag_log_contrast']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_log_contrast'],iaa.LogContrast(gain=parameters_dictionary['log_contrast_gain'],
                                                    per_channel=parameters_dictionary['flag_log_contrast_per_channel'],
                                                    name=None, deterministic=False)))
    if parameters_dictionary['flag_linear_contrast']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_linear_contrast'],iaa.LinearContrast(alpha=parameters_dictionary['linear_contrast_gain'],
                                                       per_channel=parameters_dictionary[
                                                           'flag_linear_contrast_per_channel'], name=None,
                                                       deterministic=False)))
    if parameters_dictionary['flag_contrast_normalization']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_contrast_normalization'],iaa.ContrastNormalization(alpha=parameters_dictionary['contrast_normalization_alpha'],
                                      per_channel=parameters_dictionary['flag_contrast_normalization_per_channel'],
                                      name=None, deterministic=False)))

    # (10). Low/High Frequency Emphasis:
    if parameters_dictionary['flag_sharpen']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_sharpen'],iaa.Sharpen(alpha=parameters_dictionary['sharpen_alpha'],
                                                lightness=parameters_dictionary['sharpen_lightness'], name=None,
                                                deterministic=False)))
    if parameters_dictionary['flag_emboss']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_emboss'],iaa.Emboss(alpha=parameters_dictionary['emboss_alpha'], strength=parameters_dictionary['emboss_strength'],
                       name=None, deterministic=False)))
    if parameters_dictionary['flag_edge_detect']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_edge_detect'],iaa.EdgeDetect(alpha=parameters_dictionary['edge_detect_alpha'], name=None, deterministic=False)))
    if parameters_dictionary['flag_directed_edge_detect']:
        Transformations_List.append(iaa.Sometimes(parameters_dictionary['probability_of_directed_edge_detect'],iaa.DirectedEdgeDetect(alpha=parameters_dictionary['directed_edge_detect_alpha'],
                                                           direction=parameters_dictionary[
                                                               'directed_edge_detect_direction'], name=None,
                                                           deterministic=False)))

    # (11). Convolve:
    if parameters_dictionary['flag_convolve']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_convolve'],iaa.Convolve(matrix=parameters_dictionary['convolve_matrix_to_convolve'], name=None, deterministic=False)))

    # (12). JPEG Compression:
    if parameters_dictionary['flag_JPEF_compression']:
        Transformations_List.append(
            iaa.Sometimes(parameters_dictionary['probability_of_JPEF_compression'],iaa.JpegCompression(compression=parameters_dictionary['JPEG_compression_level_of_compression'], name=None,
                                deterministic=False)))



    #### Final Crop: ######

    #########################




    ### Compose Transformation Object from Transformations List: ###
    Transformations_Composed = iaa.Sequential(Transformations_List);
    return Transformations_Composed;























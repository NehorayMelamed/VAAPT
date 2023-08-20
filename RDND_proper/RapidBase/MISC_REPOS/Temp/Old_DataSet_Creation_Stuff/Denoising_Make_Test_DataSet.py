
from __future__ import print_function
#Imports:
#(1). Auxiliary:
import PIL
import argparse
import os
import numpy as np
import math
from PIL import Image
import glob
import random
import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int,importlib
from Contextual_Shlomi import get_network_layers_list_flat
from int_range import *
from int_arange import *
from mat_range import *
from matlab_arange import *
from my_linspace import *
from my_linspace_int import *
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
# import torch.nn.functional as F
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

# Our libs
from Semantic_Segmentation.SemanticSegmentation_CVAIL.dataset import TestDataset
from Semantic_Segmentation.SemanticSegmentation_CVAIL.models import ModelBuilder, SegmentationModule
from Semantic_Segmentation.SemanticSegmentation_CVAIL.utils import colorEncode
from Semantic_Segmentation.SemanticSegmentation_CVAIL.lib.nn import user_scattered_collate, async_copy_to
from Semantic_Segmentation.SemanticSegmentation_CVAIL.lib.utils import as_numpy, mark_volatile
import Semantic_Segmentation.SemanticSegmentation_CVAIL.lib.utils.data as torchdata

#ESRGAN:
import ESRGAN_dataset
import ESRGAN_Visualizers
import ESRGAN_Optimizers
import ESRGAN_Losses
import ESRGAN_deep_utils
import ESRGAN_utils
import ESRGAN_Models
import ESRGAN_basic_Blocks_and_Layers
import ESRGAN_OPT
from ESRGAN_utils import *
from ESRGAN_deep_utils import *
from ESRGAN_basic_Blocks_and_Layers import *
from ESRGAN_Models import *
from ESRGAN_dataset import *
from ESRGAN_Losses import *
from ESRGAN_Optimizers import *
from ESRGAN_Visualizers import *
from ESRGAN_dataset import *
from ESRGAN_OPT import *


#Augmentations:
import PIL
import albumentations
from albumentations import *
import Albumentations_utils
import IMGAUG_utils
from IMGAUG_utils import *

#GPU utilizations:
import gpu_utils
gpu_utils.showUtilization(True)


import time
import argparse
import numpy as np
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn
from torchvision import transforms
from torchvision_x.transforms import functional as torchvisionX_F
from tensorboardX import SummaryWriter


import torch_fusion
from torch_fusion import gan
from torch_fusion import learners
from torch_fusion import datasets
from torch_fusion import metrics
from torch_fusion import layers
from torch_fusion import initializers
from torch_fusion import utils
from torch_fusion import transforms
from torch_fusion import fp16_utils

from torch_fusion import *
from torch_fusion.learners import *
from torch_fusion.datasets import *
from torch_fusion.metrics import *
from torch_fusion.layers import *
from torch_fusion.initializers import *
from torch_fusion.utils import *
from torch_fusion.transforms import *
from torch_fusion.fp16_utils import *
from torch_fusion.gan import *

from torch_fusion.gan.learners import *
from torch_fusion.gan.applications import *
from torch_fusion.gan.layers import *

from torch.optim import Adam
import torch.cuda as cuda
import torch.nn as nn
import torch
import numpy as np
from easydict import EasyDict as edict
from scipy.io import loadmat

import cv2
from tqdm import tqdm
# from ptsemseg.models.utils import *

import Memory_Networks
from Memory_Networks import *


import Denoising_Layers_And_Networks
from Denoising_Layers_And_Networks import *

from ESRGAN_Losses import *
from ESRGAN_utils import *
from dataset import *



### Set Up Decices: ###
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device_cpu = torch.device('cpu')
Generator_device = device1
def noise_RGB_LinearShotNoise_Torch(original_images_input, gain, flag_output_uint8_or_float='float', flag_clip=False):
    original_images = torch.zeros(original_images_input.shape).to(Generator_device)
    original_images = original_images.type(torch.float32)
    original_images.copy_(original_images_input)

    # Input is either uint8 between [0,255]  or float between [0,1]
    if original_images.dtype == torch.float32 or original_images.dtype == torch.float64:
        original_images *= 255;
        original_images = original_images.type(torch.float32)

    if gain != 0:
        default_gain = 20;
        electrons_per_pixel = original_images * default_gain / gain;
        std_shot_noise = torch.sqrt(electrons_per_pixel);
        noise_image = torch.randn(*original_images.shape).to(original_images.device) * std_shot_noise;
        noisy_image = electrons_per_pixel + noise_image
        noisy_image = noisy_image * gain / default_gain;
    else:
        noisy_image = original_images;

    if flag_output_uint8_or_float == torch.uint8 or flag_output_uint8_or_float == 'uint8':
        noisy_image = torch.clamp(noisy_image, 0, 255)
        noisy_image = noisy_image.type(torch.uint8)
    else:
        noisy_image = noisy_image / 255;
        if flag_clip:
            noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image








#########################   Make Still Images For Official Test DataSet: ##################
### Still Images: ###
clean_still_images_folder = 'F:\OFFICIAL TEST IMAGES\Still Images\Clean Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
max_shift_in_pixels = 0;
min_crop_size = 1300
crop_size = 1000
noise_gains = [0,30,65,100]
number_of_time_steps = 25;

### IMGAUG: ###
imgaug_parameters = get_IMGAUG_parameters()
# Affine:
imgaug_parameters['flag_affine_transform'] = True
imgaug_parameters['affine_scale'] = 1
imgaug_parameters['affine_translation_percent'] = None
imgaug_parameters['affine_translation_number_of_pixels'] = (0,max_shift_in_pixels)
imgaug_parameters['affine_rotation_degrees'] = 0
imgaug_parameters['affine_shear_degrees'] = 0
imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
imgaug_parameters['probability_of_affine_transform'] = 1
#Perspective:
imgaug_parameters['flag_perspective_transform'] = False
imgaug_parameters['flag_perspective_transform_keep_size'] = True
imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
imgaug_parameters['probability_of_perspective_transform'] = 1
### Get Augmenter: ###
imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)


### IMGAUG Cropping: ###
train_dataset = ImageFolderRecursive_MaxNumberOfImages_Deformations(clean_still_images_folder,
                                                                    transform=imgaug_transforms,
                                                                    flag_base_transform=False,
                                                                    flag_turbulence_transform = False,
                                                                    max_number_of_images = -1,
                                                                    min_crop_size=min_crop_size,
                                                                    crop_size=crop_size,
                                                                    extention='png',loader='CV')
image_brightness_reduction_factor = 0.9


### For Every Image Get Noisy Image (For Every Noise Gain): ###
for noise_gain in noise_gains:
    ### Create new folder for this particular Noise: ###
    noise_gain_folder = os.path.join(still_images_folder, 'NoiseGain_' + str(noise_gain))
    path_make_directory_from_path(noise_gain_folder)

    for image_index in arange(len(train_dataset)):
        ### Creatae new folder for this particular image: ###
        image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames[image_index].split('.')[0].split('\\')[-1])
        path_make_directory_from_path(image_within_noise_folder)

        ### Read the image: ###
        current_frame = torch.Tensor(train_dataset[image_index])
        # imshow_torch(current_frame)
        current_frame_tensor = current_frame.unsqueeze(0)
        current_frame_tensor = current_frame_tensor * image_brightness_reduction_factor

        for time_step in arange(number_of_time_steps):
            current_frame_tensor_noisy = noise_RGB_LinearShotNoise_Torch(current_frame_tensor, gain=noise_gain, flag_output_uint8_or_float='float32')
            numpy_array = np.transpose(current_frame_tensor_noisy.numpy().squeeze(), [1,2,0])
            numpy_array *= 255;
            numpy_array = numpy_array.clip(0,255)
            numpy_array = numpy_array.astype('uint8')
            save_image_numpy(image_within_noise_folder, str(time_step).rjust(3,'0') + '.png', numpy_array, flag_convert_bgr2rgb=True, flag_scale=False)
















#########################   Make Semi-Dynamic Shifting Images For Official Test DataSet: ##################
### Still Images: ###
super_shifts_folder = 'F:\OFFICIAL TEST IMAGES\Shifted Images'
clean_still_images_folder = 'F:\OFFICIAL TEST IMAGES\Clean Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
max_shifts_in_pixels = [1,2,5,10,20,50,100];
min_crop_size = 1300
crop_size = 1000
noise_gains = [0,30,65,100]
number_of_time_steps = 25;



for max_shift_in_pixels in max_shifts_in_pixels:
    shift_folder = os.path.join(super_shifts_folder, 'Shift_' + str(max_shift_in_pixels))

    ### IMGAUG: ###
    imgaug_parameters = get_IMGAUG_parameters()
    # Affine:
    imgaug_parameters['flag_affine_transform'] = True
    imgaug_parameters['affine_scale'] = 1
    imgaug_parameters['affine_translation_percent'] = None
    imgaug_parameters['affine_translation_number_of_pixels'] = {"x": (-max_shift_in_pixels, max_shift_in_pixels), "y": (-max_shift_in_pixels, max_shift_in_pixels)}
    imgaug_parameters['affine_rotation_degrees'] = 0
    imgaug_parameters['affine_shear_degrees'] = 0
    imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
    imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
    imgaug_parameters['probability_of_affine_transform'] = 1
    #Perspective:
    imgaug_parameters['flag_perspective_transform'] = False
    imgaug_parameters['flag_perspective_transform_keep_size'] = True
    imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
    imgaug_parameters['probability_of_perspective_transform'] = 1
    ### Get Augmenter: ###
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)


    ### IMGAUG Cropping: ###
    # iaa.Affine()
    train_dataset = ImageFolderRecursive_MaxNumberOfImages_Deformations(clean_still_images_folder,
                                                                        transform=imgaug_transforms,
                                                                        flag_base_transform=True,
                                                                        flag_turbulence_transform = False,
                                                                        max_number_of_images = -1,
                                                                        min_crop_size=min_crop_size,
                                                                        crop_size=crop_size,
                                                                        extention='png',loader='CV')


    for image_index in arange(len(train_dataset)):

        for time_step in arange(number_of_time_steps):
            ### Read the image: ###
            current_frame = torch.Tensor(train_dataset[image_index])
            # imshow_torch(current_frame)
            ### Reduce Image Brightness To Avoid Unrealistic constant Clipping: ###
            curent_frame = current_frame * image_brightness_reduction_factor
            current_frame_tensor = current_frame.unsqueeze(0)

            ### For Every Image Get Noisy Image (For Every Noise Gain): ###
            for noise_gain in noise_gains:
                ### Create new folder for this particular Noise: ###
                noise_gain_folder = os.path.join(shift_folder, 'NoiseGain_' + str(noise_gain))
                image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames[image_index].split('.')[0].split('\\')[-1])
                path_make_directory_from_path(image_within_noise_folder)

                current_frame_tensor_noisy = noise_RGB_LinearShotNoise_Torch(current_frame_tensor, gain=noise_gain, flag_output_uint8_or_float='float32')
                numpy_array = np.transpose(current_frame_tensor_noisy.cpu().numpy().squeeze(), [1,2,0])
                numpy_array *= 255;
                numpy_array = numpy_array.clip(0, 255)
                numpy_array = numpy_array.astype('uint8')
                save_image_numpy(image_within_noise_folder, str(time_step).rjust(3,'0') + '.png', numpy_array, flag_convert_bgr2rgb=True, flag_scale=False)













#########################   Make Semi-Dynamic Affine Deformation Images For Official Test DataSet: ##################
### Still Images: ###
super_affines_folder = 'F:\OFFICIAL TEST IMAGES\Affine Changes Images'
clean_still_images_folder = 'F:\OFFICIAL TEST IMAGES\Clean Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
min_crop_size = 1300
crop_size = 1000
noise_gains = [0,30,65,100]
number_of_time_steps = 25;

max_shifts_in_pixels = [0,5,20,50,100];
max_scales_deltas = [0,0.05]
max_degrees = [0,5]
max_perspective_change_factors = [0,0.05]

image_brightness_reduction_factor = 0.9;


for max_shift_in_pixels in max_shifts_in_pixels:
    for max_scale in max_scales_deltas:
        for max_degree in max_degrees:
            for max_perspective_change_factor in max_perspective_change_factors:

                dot_change_char = '#'
                current_sub_folder_name = '_Shift_' + str(max_shift_in_pixels).replace('.',dot_change_char) + \
                                          '_Degrees_' + str(max_degree).replace('.',dot_change_char) + \
                                          '_Scale_' + str(max_scale).replace('.',dot_change_char) + \
                                          '_Perspective_' + str(max_perspective_change_factor).replace('.',dot_change_char);
                current_affine_folder = os.path.join(super_affines_folder, current_sub_folder_name)
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


                ### IMGAUG Cropping: ###
                # iaa.Affine()
                train_dataset = ImageFolderRecursive_MaxNumberOfImages_Deformations(clean_still_images_folder,
                                                                                    transform=imgaug_transforms,
                                                                                    flag_base_transform=True,
                                                                                    flag_turbulence_transform = False,
                                                                                    max_number_of_images = -1,
                                                                                    min_crop_size=min_crop_size,
                                                                                    crop_size=crop_size,
                                                                                    extention='png',loader='CV')

                # for i in arange(10):
                #     imshow(np.transpose(train_dataset[0],[1,2,0]))
                #     pause(0.3)

                tic()
                for image_index in arange(len(train_dataset)):

                    for time_step in arange(number_of_time_steps):
                        ### Read the image: ###
                        current_frame = torch.Tensor(train_dataset[image_index])
                        # imshow_torch(current_frame)
                        ### Reduce Image Brightness To Avoid Unrealistic constant Clipping: ###
                        curent_frame = current_frame * image_brightness_reduction_factor
                        current_frame_tensor = current_frame.unsqueeze(0)

                        ### For Every Image Get Noisy Image (For Every Noise Gain): ###
                        for noise_gain in noise_gains:
                            ### Create new folder for this particular Noise: ###
                            noise_gain_folder = os.path.join(current_affine_folder, 'NoiseGain_' + str(noise_gain))
                            image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames[image_index].split('.')[0].split('\\')[-1])
                            path_make_directory_from_path(image_within_noise_folder)

                            current_frame_tensor_noisy = noise_RGB_LinearShotNoise_Torch(current_frame_tensor, gain=noise_gain, flag_output_uint8_or_float='float32')
                            numpy_array = np.transpose(current_frame_tensor_noisy.cpu().numpy().squeeze(), [1, 2, 0])
                            numpy_array *= 255;
                            numpy_array = numpy_array.clip(0, 255)
                            numpy_array = numpy_array.astype('uint8')
                            save_image_numpy(image_within_noise_folder, str(time_step).rjust(3, '0') + '.png', numpy_array, flag_convert_bgr2rgb=True, flag_scale=False)
                toc()
















#########################   Make Semi-Dynamic "Turbulence" Only Deformation Images For Official Test DataSet: ##################
### Still Images: ###
super_affines_folder = 'F:\OFFICIAL TEST IMAGES\Turbulence Changes Images'
clean_still_images_folder = 'F:\OFFICIAL TEST IMAGES\Still Images\Clean Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
min_crop_size = 1300
crop_size = 1000
noise_gains = [1e-8,30,65,100]
number_of_time_steps = 25;

max_shifts_in_pixels = [0,5,20,50,100];
max_scales_deltas = [0,0.05]
max_degrees = [0,10]
max_perspective_change_factors = [0,0.05]


Cn2_vec = [3e-15,5e-15,1e-14,3e-14,1e-13]
image_brightness_reduction_factor = 0.9;

def scientific_notation(input_number,number_of_digits_after_point=2):
    format_string = '{:.' + str(number_of_digits_after_point) + 'e}'
    return format_string.format(input_number)


for current_Cn2 in Cn2_vec:

    dot_change_char = '#'
    current_sub_folder_name = 'Cn2_' + str(current_Cn2)
    current_affine_folder = os.path.join(super_affines_folder, current_sub_folder_name)
    path_make_directory_from_path(current_affine_folder)

    ### IMGAUG: ###
    imgaug_parameters = get_IMGAUG_parameters()
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)


    ### IMGAUG Cropping: ###
    # iaa.Affine()
    train_dataset = ImageFolderRecursive_MaxNumberOfImages_Deformations(clean_still_images_folder,
                                                                        transform=imgaug_transforms,
                                                                        flag_base_transform=False,
                                                                        flag_turbulence_transform = True,
                                                                        Cn2=current_Cn2,
                                                                        max_number_of_images = -1,
                                                                        min_crop_size=min_crop_size,
                                                                        crop_size=crop_size,
                                                                        extention='png',loader='CV')

    # for i in arange(10):
    #     imshow(np.transpose(train_dataset[0],[1,2,0]))
    #     pause(0.3)

    tic()
    for image_index in arange(len(train_dataset)):

        for time_step in arange(number_of_time_steps):
            ### Read the image: ###
            current_frame = torch.Tensor(train_dataset[image_index])
            # imshow_torch(current_frame)
            ### Reduce Image Brightness To Avoid Unrealistic constant Clipping: ###
            curent_frame = current_frame * image_brightness_reduction_factor
            current_frame_tensor = current_frame.unsqueeze(0)

            ### For Every Image Get Noisy Image (For Every Noise Gain): ###
            for noise_gain in noise_gains:
                ### Create new folder for this particular Noise: ###
                noise_gain_folder = os.path.join(current_affine_folder, 'NoiseGain_' + str(noise_gain))
                image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames[image_index].split('.')[0].split('\\')[-1])
                path_make_directory_from_path(image_within_noise_folder)

                current_frame_tensor_noisy = noise_RGB_LinearShotNoise_Torch(current_frame_tensor, gain=noise_gain, flag_output_uint8_or_float='float32')
                numpy_array = np.transpose(current_frame_tensor_noisy.cpu().numpy().squeeze(), [1, 2, 0])
                numpy_array *= 255;
                numpy_array = numpy_array.clip(0, 255)
                numpy_array = numpy_array.astype('uint8')
                save_image_numpy(image_within_noise_folder, str(time_step).rjust(3, '0') + '.png', numpy_array, flag_convert_bgr2rgb=True, flag_scale=False)
    toc()















#
# ### IMGAUG: ###
# imgaug_parameters = get_IMGAUG_parameters()
# # Affine:
# imgaug_parameters['flag_affine_transform'] = True
# imgaug_parameters['affine_scale'] = (0.9,1.1)
# imgaug_parameters['affine_translation_percent'] = None
# imgaug_parameters['affine_translation_number_of_pixels'] = (0,10)
# imgaug_parameters['affine_rotation_degrees'] = (0,10)
# imgaug_parameters['affine_shear_degrees'] = (0,10)
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
#
#
# ### IMGAUG Cropping: ###
# #(*). Note - this method probably isn't smart for generality sake because it requires to reinitialize the iaa.Crop object for different sizes images....
# min_crop_size = 140
# crop_size = 100
# root = 'F:/NON JPEG IMAGES\HR_images\General100'
# batch_size = 16;
# train_dataset = ImageFolderRecursive_MaxNumberOfImages_Deformations(root,
#                                                                     transform=imgaug_transforms,
#                                                                     flag_base_transform=True,
#                                                                     flag_turbulence_transform = True,
#                                                                     max_number_of_images = -1,
#                                                                     min_crop_size=min_crop_size,
#                                                                     crop_size=crop_size,
#                                                                     extention='png',loader='CV')
# train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, pin_memory=True);  ### Mind the shuffle variable!@#!#%@#%@$%
# tic()
# for i,data_from_dataloader in enumerate(train_dataloader):
#     bla = data_from_dataloader;
#     # for image_counter in arange(len(data_from_dataloader)):
#     #     imshow_torch(data_from_dataloader[image_counter,:,:,:])
#     #     pause(0.2)
#     if i==0:
#         break
# toc()











# #### Affine Transform On GPU: ###
# crop_size = 100;
# current_image = read_image_default()
# current_image = crop_tensor(current_image, crop_size,crop_size)
# current_image = np.transpose(current_image, [2,0,1])
# current_image_tensor = torch.Tensor(current_image).unsqueeze(0)
#
# #(1). Initial Grid:
# turbulence_object = get_turbulence_flow_field_object(crop_size,crop_size,batch_size,Cn2=7e-13)
# [X0, Y0] = np.meshgrid(np.arange(crop_size), np.arange(crop_size))
# X0 = float32(X0)
# Y0 = float32(Y0)
# X0 = torch.Tensor(X0)
# Y0 = torch.Tensor(Y0)
# X0 = X0.unsqueeze(0)
# Y0 = Y0.unsqueeze(0)
# X0 = torch.cat([X0]*batch_size,0)
# Y0 = torch.cat([Y0]*batch_size,0)
# #(3). Scale:
# scale_x = 1.2;
# scale_y = 1;
# scale_x = 1/scale_x
# scale_y = 1/scale_y
# X0 *= scale_x
# Y0 *= scale_y
# #(4). Rotate:
# max_rotation_angle = 45
# rotation_angles = ( torch.rand((batch_size,1))*max_rotation_angle-0.5*max_rotation_angle  ) * 2
# # rotation_matrix = cv2.getRotationMatrix2D((crop_size/2,crop_size/2), rotation_angle, 1)
# # flow_field = torch.nn.functional.affine_grid(torch.Tensor(rotation_matrix).unsqueeze(0), torch.Size([1,3,crop_size,crop_size]) )
#
# X0_centered = X0-X0.max()/2
# Y0_centered = Y0-Y0.max()/2
# X0_new = cos(rotation_angles*pi/180).unsqueeze(-1)*X0_centered - sin(rotation_angles*pi/180).unsqueeze(-1)*Y0_centered
# Y0_new = sin(rotation_angles*pi/180).unsqueeze(-1)*X0_centered + cos(rotation_angles*pi/180).unsqueeze(-1)*Y0_centered
# X0 = X0_new
# Y0 = Y0_new
#
# #(2). Shift:
# shift_x = 0
# shift_y = 0
# X0 += shift_x*scale_x
# Y0 += shift_y*scale_y
#
# ### Add Turbulence: ###
# # turbulence_flow_field_X, turbulence_flow_field_Y = get_turbulence_flow_field(crop_size,crop_size,batch_size,Cn2=6e-13)
# turbulence_flow_field_X, turbulence_flow_field_Y = turbulence_object.get_flow_field()
# turbulence_flow_field_X = turbulence_flow_field_X/crop_size
# turbulence_flow_field_Y = turbulence_flow_field_Y/crop_size
#
# X0 = X0 + turbulence_flow_field_X
# Y0 = Y0 + turbulence_flow_field_X
# X0 = X0.unsqueeze(-1)
# Y0 = Y0.unsqueeze(-1)
#
# #(*). Normalize To 1:
# X0 = X0/((crop_size-1)/2)
# Y0 = Y0/((crop_size-1)/2)
# flow_grid = torch.cat([X0,Y0],3)
#
# tic()
# # bli = torch.nn.functional.grid_sample(current_image_tensor,flow_grid,mode='bilinear')
# bli = torch.nn.functional.grid_sample(data_from_dataloader,flow_grid,mode='bilinear',padding_mode='reflection')
# toc()
#
# figure(1)
# imshow_torch(data_from_dataloader[0,:,:,:])
# figure(2)
# imshow_torch(bli[0,:,:,:])
#
# figure(1);
# imshow_torch(current_image_tensor[0,:,:,:])
# figure(2)
# imshow_torch(bli[0,:,:,:])

# imgaug_transforms = imgaug_transforms.to_deterministic()
# current_frame_transformed = imgaug_transforms.augment_images([current_frame])[0];


















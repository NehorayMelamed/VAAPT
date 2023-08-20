

import os
import cv2
import tic_toc
from tic_toc import *
import numpy as np
from numpy import *
from numpy.random import randint


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

from colour_demosaicing import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Menon2007
import colour


import ESRGAN_utils
from ESRGAN_utils import *



def read_image_cv2(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
    image = cv2.imread(path, cv2.IMREAD_COLOR);
    if flag_convert_to_rgb == 1:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB);
    if flag_normalize_to_float == 1:
        image = image.astype(np.float32) / 255; #normalize to range [0,1] instead of [0,255]
    if image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        image = np.expand_dims(image, axis=2);
    if image.shape[2] > 4:
        # In the weird case where the image has more then 3 channels only take the first 3 channels to make it conform with 3-channels image format:
        image = image[:,:,:3];

    return image




def save_image_numpy(folder_path=None,filename=None,numpy_array=None,flag_convert_bgr2rgb=True, flag_scale=True):
    if flag_scale:
        scale = 255;
    else:
        scale = 1;
    if flag_convert_bgr2rgb:
        cv2.imwrite(os.path.join(folder_path,filename), cv2.cvtColor(numpy_array.squeeze() * scale, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(os.path.join(folder_path, filename), numpy_array.squeeze() * scale)

def create_folder_if_needed(folder_full_path):
    if not os.path.exists(folder_full_path):
        os.makedirs(folder_full_path)


def to_linear(img):
    """
    Apply reverse sRGB curve on input image as defined at
    https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation.
    """
    return np.where(img >= 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)


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


def compute_noise_stddev(linear, noise_gain, use_mean=True):
    """
    Noise model is defined by its signal-dependent noise variance given as a * x^2 + b * x + c.
    """
    mean = linear
    if use_mean:
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

    return np.sqrt(a * mean + b) *1.3


def apply_gain(input_bayer, gain):
    h, w = input_bayer.shape
    output = np.zeros((h, w))

    output[0:h:2, 0:w:2] = input_bayer[0:h:2, 0:w:2] * gain[0]
    output[0:h:2, 1:w:2] = input_bayer[0:h:2, 1:w:2] * gain[1]
    output[1:h:2, 1:w:2] = input_bayer[1:h:2, 1:w:2] * gain[2]
    output[1:h:2, 0:w:2] = input_bayer[1:h:2, 0:w:2] * gain[3]
    return output


def create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(img, flag_random_gamma_and_gain=False):
    ### Load image and use mosaicing to get the bayer image: ###
    bayer = to_linear(mosaicing_CFA_Bayer(img, 'GRBG'))  #to_linear = de-gamma
    h, w = bayer.shape

    ### Get random gamma and gain (detector gain): ###
    if flag_random_gamma_and_gain:
        gamma_rand = np.random.random(1)*1.5+1
        gain_rand = np.random.random(1)*0.8+0.2
        bayer = (bayer**gamma_rand)*gain_rand

    ### Get random R,B,G gains: ### color augmentation
    g_rand = np.random.random(1)*2+0.5
    gain_array = [g_rand, (np.random.random(1) * 2 + 0.5), g_rand, (np.random.random(1) * 2 + 0.5)]
    bayer = apply_gain(bayer, gain_array)
    rgb = np.clip(demosaicing_CFA_Bayer_Menon2007(bayer, 'GRBG'), 0, 1);

    ### Accumulate clean-bayer and "clean" (demoasiced again(!)) RGB images: ###
    return bayer, rgb



def add_random_WB_and_noise_to_bayer(bayer):
    ### Get current bayer and RGB (demoasiced from bayer) images: ###
    bayer = clean_bayer_images[ind]
    h, w = bayer.shape

    ### Get more (White-Balance) small random gains for the gr,gb and large gains for the R,B bayer channels: ###
    gain_gr = np.random.random(1)*0.05 + 1
    gain_gb = np.random.random(1)*0.05 + 1
    gain_r = np.random.random(1)*3 + 1
    gain_b = np.random.random(1)*3 + 1
    gain_array = [gain_gr, gain_r, gain_gb, gain_b]
    inv_gain_array = [1/gain_gr, 1/gain_r, 1/gain_gb, 1/gain_b]

    ### Apply INVERSE GAIN to the bayer image: ###
    bayer_before_wb = apply_gain(bayer, inv_gain_array)

    ### Compute noisy part of the image using analog gain: ###
    analog_gain = np.random.random(1) * 92 + 32
    noise_std_per_pixel_array = compute_noise_stddev(bayer_before_wb, analog_gain, use_mean=False)  ### Difference is use_mean=False  ???

    ### Apply Gain to the Noise itself: ###
    noise_std_per_pixel_array = apply_gain(noise_std_per_pixel_array, gain_array)

    ### Apply Multiplicative random noise: ###
    rand_noise = np.random.randn(h, w)
    noise_image = noise_std_per_pixel_array * rand_noise #actually generate the noise

    ### Add noise to the bayer image: ###
    #TODO: what is decided??? should i clip at this point or not?????????????????????????????????????????????????
    noisy_bayer = bayer + noise_image
    noisy_bayer = np.maximum(noisy_bayer, 0)
    noisy_bayer = np.minimum(noisy_bayer, 1)

    return noisy_bayer;

import os



### TVNET IMPORTS: ###
from TVnet_simple.data.frame_dataset import frame_dataset
from TVnet_simple.train_options import arguments
import torch.utils.data as data
from TVnet_simple.model.network_tvnet import model
import scipy.io as sio
from TVnet_simple.utils import *
import easydict
from easydict import EasyDict

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")
device_cpu = torch.device('cpu')

### Function Returning TVNet Module Instance: ###
def get_TVNet_instance(x_input, number_of_iterations=60, number_of_pyramid_scales=3, device=device1, flag_trainable=False):
    args = EasyDict()
    args.frame_dir = 'blabla' #path to frames
    args.img_save_dir = 'blabla' #path to storage generated feature maps if needed
    args.n_epocs = 100
    args.n_threads = 1
    args.batch_size = 1;
    args.learning_rate = 1e-4
    args.is_shuffle = False
    args.visualize = False
    # args.data_size = (100,100); #TODO: understand the needed format here!$#@%
    args.zfactor = 0.5 #factor for building the image pyramid
    args.max_nscale = number_of_pyramid_scales; #max number of scales for image pyramid
    args.n_warps = 1; #number of warping per scale
    args.n_iters = number_of_iterations; #max number of iterations for optimization
    args.demo = False;
    ## Don't Change according to moran: ###
    args.tau = 0.25; #time step
    args.lbda = 0.1; #weight parameter for the data term
    args.theta = 0.3; #weight parameter for (u-v)^2

    ### Get default arguments: ###
    B,C,H,W = x_input.shape
    args.batch_size = B # KILL
    args.demo = False
    args.data_size = [B, C, H, W]
    args.device = x_input.device

    ### Initialize TVNet from model function: ###
    Network = model(args).to(args.device)
    if flag_trainable==False:
        Network = Network.eval()
        for param in Network.parameters():
            param.requires_grad = False

    return Network



Generator_device = device0
# Generator_device = device_cpu #GET THIS!!!....ON CPU For batch_size=1 Images IT'S MUCH FASTER!!!!...probably because of the sequential nature of the algorithm!

### Optical Flow Module: ###
TVNet_number_of_iterations = 60;
TVNet_number_of_pyramid_scales = 5;
TVNet_layer = None
delta_x = None
delta_y = None
confidence = None #==RA update gate?





def create_noisy_dataset_with_optical_flow_from_existing_dataset_PARALLEL():
    ### Get all image filenames: ###
    images_super_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'
    max_counter = 1;

    ### Initialize TVNet_layer to None: ###
    TVNet_layer = None;

    ### Loop over sub-folders and for each sub-folder: ###
    counter = 0;
    for dirpath, sub_folders_list, filenames in os.walk(images_super_folder):
        if counter>max_counter:
            break;

        ### Read ALL IMAGES for current sub-folder: ###
        for folder_index, current_subfolder in enumerate(sub_folders_list):
            if counter>max_counter:
                break;
            images = read_images_from_folder_to_numpy(os.path.join(dirpath,current_subfolder),flag_recursive=True)

            ### Turn Images to Tensor and send to wanted device: ###
            tic()
            images = torch.Tensor(np.transpose(images,[0,3,1,2])).to(Generator_device)

            ### get Optical Flow : ###
            if TVNet_layer is None:
                TVNet_layer = get_TVNet_instance(images, number_of_iterations=TVNet_number_of_iterations, number_of_pyramid_scales=TVNet_number_of_pyramid_scales, device=Generator_device, flag_trainable=False)
            delta_x, delta_y, x2_warped = TVNet_layer.forward(images[0:-1], images[1:], need_result=True)
            toc('Optical Flow')

            ### Add Zero Shift Maps as start of map because usually i assume the for the first input image i assume the "previous_image" is simply the same firs input image: ###
            delta_x = torch.cat([torch.zeros(1,1,delta_x.shape[2],delta_x.shape[3]).to(Generator_device), delta_x], dim=0)
            delta_y = torch.cat([torch.zeros(1,1,delta_y.shape[2],delta_y.shape[3]).to(Generator_device), delta_y], dim=0)

            ### Write down Opical Flow Maps into Binary File: ###
            #(1). Initialize Binary File FID:
            fid_filename_X = os.path.join(os.path.join(dirpath,current_subfolder), 'Optical_Flow' + '.bin')
            fid_Write_X = open(fid_filename_X, 'ab')
            np.array(delta_x.shape[2]).tofile(fid_Write_X) #Height
            np.array(delta_x.shape[3]).tofile(fid_Write_X) #Width
            np.array(2).tofile(fid_Write_X) #Number_of_channels=2 (u,v)
            #(2). Write Maps to FID:
            #   (2.1). Write in form [B,2,H,W] all at once:
            numpy_array_to_write = torch.cat([delta_x,delta_y], dim=1).cpu().numpy()
            #   (2.2). Write in form [1,2,H,W] sequentially:
            #TODO: IMPLEMENT!
            #   (2.3). Write Optical Flow Maps To Binary File:
            numpy_array_to_write.tofile(fid_Write_X)
            #(3). Release FID:
            fid_Write_X.close()
            ### Uptick counter by 1: ###
            counter += 1;








def create_noisy_dataset_with_optical_flow_from_existing_dataset():
    ### Get all image filenames: ###
    images_super_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'
    max_counter = 10;
    # image_filenames_list = get_image_filenames_from_folder(images_super_folder, flag_recursive=True)

    ### Initialize TVNet_layer to None: ###
    TVNet_layer = None;

    ### Loop over sub-folders and for each sub-folder: ###
    counter = 0;
    for dirpath, dn, filenames in os.walk(images_super_folder):
        if counter>max_counter:
            break
        for image_index, current_filename in enumerate(sorted(filenames)):
            if counter>max_counter:
                break
            ### Get image full filename & Read Image: ###
            full_filename = os.path.join(dirpath, current_filename);
            print(full_filename)
            current_image = read_image_torch(full_filename, flag_convert_to_rgb=1, flag_normalize_to_float=1)

            tic()
            ### Put current image in wanted device (maybe going to gpu will be worth it because it will be that much faster?...maybe i should load all images together into a batch format...send them to gpu and then do it?): ###
            current_image = current_image.to(Generator_device)

            ### If this is not the first image then we can compare it to the previous one using Optical Flow: ###
            if image_index>0:
                if TVNet_layer is None:
                    TVNet_layer = get_TVNet_instance(current_image, number_of_iterations=TVNet_number_of_iterations, number_of_pyramid_scales=TVNet_number_of_pyramid_scales, device=Generator_device, flag_trainable=False)
                delta_x, delta_y, x2_warped = TVNet_layer.forward(current_image, previous_image, need_result=True)
            toc('Optical Flow')

            ### Assign previous image with current image: ###
            previous_image = current_image;
            counter += 1;



















def split_video_into_image_frames():
    ### Set Up Decices: ###
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device_cpu = torch.device('cpu')
    Generator_device = device1
    ### Get Raw File and Video_Stream Object: ###
    # movie_full_filenames = ['G:\Raw Films/Walking_in_Tokyo_Shibuya_at_night.mkv',
    #                        'G:\Raw Films/Driving_Downtown_Chicago_4K_USA.mkv',
    #                        'G:\Raw Films/3HRS_Stunning_Underwater_Footage_Relaxing_Music.mkv',
    #                        'G:\Raw Films/Night_videowalk_in_East_Shinjuku_Tokyo.mkv',
    #                        'G:\Raw Films/Rome_Virtual_Walking_Tour_in_4K_Rome_City_Travel.mkv',
    #                        'G:\Raw Films/Walk_to_the_Kaohsiung_Main_Public_Library_at_night.mkv',
    #                        'G:\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']
    movie_full_filenames = ['G:\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']
    saved_images_super_folder = 'F:\TNR - OFFICIAL TEST IMAGES\Dynamic_Videos'
    if not os.path.exists(saved_images_super_folder):
        os.makedirs(saved_images_super_folder)

    ### Noise Gains To Save: ###
    noise_gains = [0,30,65,100]
    # noise_gains = [100]

    ### Frames to Capture: ###
    # (1). Initial Frame:
    flag_use_initial_frame_or_second = 'second'  # 'frame'/'second'
    initial_frames = 10;
    # initial_seconds = [0*60, 10*60, 20*60, 30*60, 40*60, 50*60, 60*60];
    initial_seconds = [7*60+15];
    # (2). Number of Frams:
    flag_use_final_frame_or_second = 'frame'
    # number_of_frames = fps * 2;
    number_of_seconds = 10
    final_second = 0 * 3600 + 0 * 60 + 5;

    ### Crop Size: ###
    crop_size = 2000;


    ### Loop over movie files and create "dataset" (long sequences when i want to qualitatively compare videos and many short sequences when i want to gather statistics): ###
    for movie_full_filename in movie_full_filenames:
        video_stream = cv2.VideoCapture(movie_full_filename)
        frame_width = int(video_stream.get(3))
        frame_height = int(video_stream.get(4))
        fps = video_stream.get(5)
        total_number_of_frames = int(video_stream.get(7))
        ### Read a few frames for voodoo good luck: ###
        video_stream.read();
        video_stream.read();
        video_stream.read();

        # initial_frames = [0 * 60 * fps, 10 * 60 * fps, 20 * 60 * fps, 30 * 60 * fps, 40 * 60 * fps, 50 * 60 * fps, 60 * 60 * fps]

        for initial_second in initial_seconds:
            initial_frame = int(initial_second * fps)
            number_of_frames = int(number_of_seconds * fps)
            # ### Frames to Capture: ###
            # #(1). Initial Frame:
            # flag_use_initial_frame_or_second = 'second' #'frame'/'second'
            # initial_frame = 10;
            # initial_second = 0*3600 + 30*60 + 2;
            # if flag_use_initial_frame_or_second == 'second':
            #     initial_frame = int(fps * initial_second);
            # #(2). Number of Frams:
            # flag_use_final_frame_or_second = 'frame'
            # number_of_frames = fps*10;
            # final_second = 0*3600 + 0*60 + 5;
            # if flag_use_final_frame_or_second == 'second':
            #     number_of_frames = int(fps * (final_second-initial_second))
            #(3). Final Frame:
            final_frame = min(initial_frame+number_of_frames, total_number_of_frames)

            ### Loop Over Images And Save What Is Needed: ###
            current_step = 0
            frames_saved_so_far = 0;
            while current_step < initial_frame + number_of_frames:
                tic()
                ### Read Frame: ###
                flag_frame_available, current_frame = video_stream.read();
                if flag_frame_available==False:
                    break;


                ### Save Frames If It' Time: ###
                if current_step >= initial_frame:
                    current_frame = crop_tensor(current_frame, crop_size, crop_size)
                    ### Loop Over The Different Noises We Need To Save: ###
                    for noise_index, noise_gain in enumerate(noise_gains):
                        #(1). Add Noise To Frame:
                        current_frame_tensor = torch.Tensor(current_frame).permute([2,0,1]).unsqueeze(0)/255;
                        current_frame_tensor_noisy = noise_RGB_LinearShotNoise_Torch(current_frame_tensor, noise_gain, flag_output_uint8_or_float='float', flag_clip=False)
                        current_frame_noisy = current_frame_tensor_noisy.squeeze(0).permute([1,2,0]).cpu().numpy()
                        current_frame_noisy = current_frame_noisy * 255
                        current_frame_noisy = np.clip(current_frame_noisy,0,255)
                        current_frame_noisy = uint8(current_frame_noisy)
                        #(2). Save Noisy Frame (Format: super_folder -> noise_gain -> specific_movie
                        noisy_image_full_folder = os.path.join(saved_images_super_folder, 'NoiseGain' + str(noise_gain), movie_full_filename.split('.')[0].split('/')[-1] + '_CropSize' + str(crop_size) + '_Frames' + str(initial_frame) + '-' + str(final_frame))
                        path_make_path_if_none_exists(noisy_image_full_folder)
                        save_image_numpy(folder_path=noisy_image_full_folder,
                                         filename=str(frames_saved_so_far).rjust(8,'0') + '.png',
                                         numpy_array=current_frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
                    ### After Adding Noise at different amounts -> we're finished with this frame so uptick counter: ###
                    frames_saved_so_far += 1;

                ### Uptick current_step by 1: ###
                current_step += 1
                toc()
###################################################################################################################################################################################################################################################################################################










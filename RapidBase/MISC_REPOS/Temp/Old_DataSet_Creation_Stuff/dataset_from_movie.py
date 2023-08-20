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
import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


###################################################################################################################################################################################################################################################################################################
def split_video_into_scenes_using_SceneDetect():
    ### NOT a very strong automatic scene detector. so one of two options: (1). get a very "nice" video with "good" semi-constant scenes with a single change period between them (no jitter between scenes). (2). maybe get a professional tool


    # scenedetect --input welcome_back_full_movie_hd.mp4 --stats new_video.stats.csv detect-content split-video -c
    ################################################################################################################################################################################################################################################################################
    STATS_FILE_PATH = 'api_test_statsfile.csv'
    start_time_float = 60.0*5
    stop_time_float = 60.0*100

    # Create a video_manager point to video file testvideo.mp4. Note that multiple
    # videos can be appended by simply specifying more file paths in the list
    # passed to the VideoManager constructor. Note that appending multiple videos
    # requires that they all have the same frame size, and optionally, framerate.
    # Full_Movie_4k_ULtra_HD_MSC_MUSICA_Cruise_Tour_Medi_Part0.mkv
    movie_full_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl/welcome_back_full_movie_hd.mp4'
    video_manager = VideoManager([movie_full_filename])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    # Add ContentDetector algorithm (constructor takes detector options like threshold).
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    # If stats file exists, load it.
    if os.path.exists(STATS_FILE_PATH):
        # Read stats from CSV file opened in read mode:
        with open(STATS_FILE_PATH, 'r') as stats_file:
            stats_manager.load_from_csv(stats_file, base_timecode)

    start_time = base_timecode + start_time_float  # 00:00:00.667
    end_time = base_timecode + stop_time_float  # 00:00:20.000
    # Set video_manager duration to read frames from 00:00:00 to 00:00:20.
    video_manager.set_duration(start_time=start_time, end_time=end_time)

    # Set downscale factor to improve processing speed.
    video_manager.set_downscale_factor()

    # Start video_manager.
    video_manager.start()

    # Perform scene detection on video_manager.
    scene_manager.detect_scenes(frame_source=video_manager)

    # Obtain list of detected scenes.
    scenes_list = scene_manager.get_scene_list(base_timecode)
    # Like FrameTimecodes, each scene in the scene_list can be sorted if the
    # list of scenes becomes unsorted.

    print('List of scenes obtained:')
    for i, scene in enumerate(scene_list):
        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (i + 1, scene[0].get_timecode(), scene[0].get_frames(), scene[1].get_timecode(), scene[1].get_frames(),))

    # We only write to the stats file if a save is required:
    if stats_manager.is_save_required():
        with open(STATS_FILE_PATH, 'w') as stats_file:
            stats_manager.save_to_csv(stats_file, base_timecode)

    # Finally, release the video_manager:
    video_manager.release()
    ##############################################################################################################################################################################################################################################



    ##############################################################################################################################################################################################################################################
    ### Loop Over Film And Seperate It To Scenes: ###
    save_scenes_folder = 'F:\Movie Scenes Temp'
    video_stream = cv2.VideoCapture(movie_full_filename)
    max_number_of_frames = 1000;
    current_step = 0;
    current_frame_counter = 0;
    flag_frame_available = True;
    flag_continue = True;
    for current_scene_counter in arange(len(scenes_list)):
        ### Get Scene Time-Line: ###
        current_scene_times = scenes_list[current_scene_counter];
        current_scene_start, current_scene_stop = current_scene_times[0].get_frames(), current_scene_times[1].get_frames();

        ### Read Frame In Time-Line: ###
        current_in_scene_frame_step = 0;
        while current_in_scene_frame_step < (current_scene_stop-current_scene_start) and flag_continue:
            #(1). Get Frame:
            flag_frame_available, current_frame = video_stream.read();
            flag_continue = flag_frame_available and current_frame_counter<max_number_of_frames;

            #(2). Update in-scene step and if it's the first step create the video writer object:
            if current_in_scene_frame_step == 0:
                ### Open Video Writer: ###
                fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
                frame_height, frame_width, number_of_channels = current_frame.shape
                current_scene_full_filename = os.path.join(save_scenes_folder, str(current_scene_counter) + '.avi')
                video_writer = cv2.VideoWriter(current_scene_full_filename, fourcc, 25.0, (frame_width, frame_height))
            current_in_scene_frame_step += 1;

            #(3). Write Frame To Video-Writer:
            video_writer.write(current_frame)


        ### Release Video Writer: ###
        video_writer.release();
 ##############################################################################################################################################################################################################################################








###################################################################################################################################################################################################################################################################################################
def split_video_into_constant_time_chunks(movie_full_filenames):
    ### Original Movie: ###
    #(1). Walk Through sub folders and get all .mkv (!!!!) video files:
    # movie_full_filename_filenames = []
    # master_folder_path = 'F:\Movies\Movie_scenes_seperated_into_different_videos'
    # for dirpath, _, fnames in sorted(os.walk(master_folder_path)):  # os.walk!!!!
    #     for fname in sorted(fnames):
    #         if fname.split('.')[-1] == 'mkv':
    #             movie_full_filename_filenames.append(os.path.join(dirpath, fname))
    #(2). Specific filenames:
    movie_full_filenames = ['F:\Movies\Raw Films/Walking_in_Tokyo_Shibuya_at_night.mkv']

    ### Where To Save Results: ###
    super_folder_to_save_everything_at = 'G:\Movie_Scenes_bin_files'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)


    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9;
    max_number_of_sequences = 1e9;
    max_number_of_images = 1e9;
    #(*). Sequene/Time Parameters:
    number_of_frames_per_sequence = int(30*5);  # 30[fps] * #[number of seconds]
    number_of_frames_to_skip_between_sequeces_of_the_same_video = int(30*60);  # 30[fps] * #[number of seconds]
    #(*). Crops and decimation:
    flag_all_crops = False
    crop_sizes = [100]
    number_of_random_crops_per_crop_size = [100];
    min_width_start = 0;
    max_width_stop = 4000;
    min_height_start = 0;
    max_height_stop = 4000;
    decimation = 1;

    # (1). Images In Folders:
    # flag_binary_or_images = 'images';
    flag_binary_or_images = 'binary';
    flag_bayer_or_rgb = 'rgb';
    flag_random_gain_and_gamma = False;
    # (2). Binary File:
    number_of_images_per_binary_file = 10000000000;
    binary_dtype = 'uint8';


    movie_filename_counter = 0;
    for movie_full_filename in movie_full_filenames:
        movie_filename_counter += 1;

        ### Open Movie Stream: ###
        video_stream = cv2.VideoCapture(movie_full_filename)
        movie_filename = movie_full_filename.split('/')[-1].split('.')[0]

        ### Loop Over Video Stream: ###
        number_of_sequences_so_far = 0;
        number_of_total_writes_so_far = 0
        number_of_frames_so_far = 0
        flag_frame_available = True
        while video_stream.isOpened() and number_of_sequences_so_far<max_number_of_sequences and number_of_total_writes_so_far<max_number_of_images and number_of_frames_so_far<max_number_of_frames and flag_frame_available:
            # tic()
            number_of_sequences_so_far += 1;
            crops_indices = []

            ### Loop over current squence frames: ###
            for mini_frame_number in arange(number_of_frames_per_sequence + number_of_frames_to_skip_between_sequeces_of_the_same_video):  #loop over number_of_frame_to_write+number_of_frames_to_wait
                tic()
                if flag_frame_available:
                    number_of_frames_so_far += 1
                    flag_frame_available, frame = video_stream.read()


                ### Loop Over Different Crops: ###
                if flag_frame_available and mini_frame_number<number_of_frames_per_sequence:
                    frame_height, frame_width, number_of_channels = frame.shape;
                    data_type_padded = str(frame.dtype).rjust(10,'0'); #get the data type and right fill with zeros to account for the different types: uint8, int8, float32....
                    max_image_value = np.array([np.iinfo(frame.dtype).max]).astype('float32')[0] # !!!!!!! max_value to be later used, if so wanted, to normalize the image. used for generality sake !!!!!!!!!!!!!!!!!!!!!!!!!!

                    if flag_bayer_or_rgb=='bayer':
                        number_of_channels = 1; #Bayer


                    ### Start off by creating all the binary files writers in a list (once per sequence): ###
                    if flag_binary_or_images == 'binary' and mini_frame_number==0:
                        binary_fid_writers_list = []
                        fid_writer_counter = 0;
                        random_crop_indices_per_crop_size = []
                        for crop_size_counter, current_crop_size in enumerate(crop_sizes):
                            current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropSize_' + str(current_crop_size))
                            create_folder_if_needed(current_crop_folder);
                            number_of_width_crops = max(frame_width // current_crop_size, 1)
                            number_of_height_crops = max(frame_height // current_crop_size, 1)
                            width_crops_indices_arange = arange(number_of_width_crops);
                            height_crops_indices_arange = arange(number_of_height_crops);

                            ### Pick random crops indices: ###
                            if flag_all_crops:
                                random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
                            else:
                                random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                                random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size[crop_size_counter]]
                                # rows_start = randint(min_height_start, min(max_height_stop, frame_height - current_crop_size))
                                # rows_stop = rows_start + current_crop_size;
                                # cols_start = randint(min_width_start, min(max_width_stop, frame_width - current_crop_size))
                                # cols_stop = cols_start + current_crop_size;
                                # crops_indices.append([rows_start, rows_stop, cols_start, cols_stop])
                            random_crop_indices_per_crop_size.append(random_crop_indices);
                            current_crop_counter = 0;
                            for width_crop_index in arange(number_of_width_crops):
                                for height_crop_index in arange(number_of_height_crops):
                                    if current_crop_counter in random_crop_indices:
                                        fid_filename_X = os.path.join(current_crop_folder, 'MovieFilename_' + movie_filename +
                                                                      '_SequenceNo_' + str(number_of_sequences_so_far) +
                                                                      '_CropHeight_' + str(min(current_crop_size, frame_height)) + '_CropWidth_' + str(min(current_crop_size, frame_width)) +
                                                                      '_CropCounter_' + str(current_crop_counter) + '.bin')
                                        fid_Write_X = open(fid_filename_X, 'ab')
                                        binary_fid_writers_list.append(fid_Write_X)
                                        # print('current_crop_size: ' + str(current_crop_size) + ', written height: ' + str(min(frame_height,current_crop_size)) + ', written width: ' + str(min(frame_width,current_crop_size)))
                                        np.array(min(frame_height,current_crop_size)).tofile(fid_Write_X)
                                        np.array(min(frame_width,current_crop_size)).tofile(fid_Write_X)
                                        np.array(number_of_channels).tofile(fid_Write_X)
                                        np.array(data_type_padded).tofile(fid_Write_X)
                                        np.array(max_image_value).tofile(fid_Write_X);
                                        fid_writer_counter += 1;
                                    current_crop_counter += 1;




                    ### Loop Over all the different crops and write the down: ###
                    total_crop_counter = 0;
                    fid_writer_counter = 0;
                    for crop_size_counter, current_crop_size in enumerate(crop_sizes):
                        number_of_width_crops = max(frame_width // current_crop_size, 1)
                        number_of_height_crops = max(frame_height // current_crop_size, 1)
                        current_crop_size_height = min(current_crop_size, frame_height);
                        current_crop_size_width = min(current_crop_size, frame_width);

                        current_crop_counter = 0;
                        ### Loop over the different crops: ###
                        for width_crop_index in arange(number_of_width_crops):
                            for height_crop_index in arange(number_of_height_crops):
                                if current_crop_counter in random_crop_indices_per_crop_size[crop_size_counter]:
                                    ### Get Current Crop: ###
                                    current_frame_crop = frame[height_crop_index*current_crop_size_height:(height_crop_index+1)*current_crop_size_height:decimation, width_crop_index*current_crop_size_width:(width_crop_index+1)*current_crop_size_width, :]

                                    ### Save Image: ####
                                    if flag_binary_or_images == 'images':
                                        #(1). Image In Folders:
                                        current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropHeight_' + str(min(current_crop_size, frame_height)) + '_CropWidth_' + str(min(current_crop_size, frame_width)))
                                        new_sub_folder = os.path.join(current_crop_folder, 'Movie_' + str(movie_filename_counter) + '_Sequence_' + str(number_of_sequences_so_far) + '_Crop_' + str(current_crop_counter))
                                        if not os.path.exists(new_sub_folder):
                                            os.makedirs(new_sub_folder)
                                        save_image_numpy(folder_path=new_sub_folder, filename=str(mini_frame_number) + '.png',
                                                         numpy_array=current_frame_crop, flag_convert_bgr2rgb=False,
                                                         flag_scale=False)

                                    elif flag_binary_or_images == 'binary':
                                        #(2). Binary File:
                                        current_fid_Write_X = binary_fid_writers_list[fid_writer_counter]
                                        fid_writer_counter += 1;
                                        if flag_bayer_or_rgb == 'bayer':
                                            if flag_random_gain_and_gamma:
                                                current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=True)
                                                current_frame_crop_bayer.tofile(current_fid_Write_X)
                                            else:
                                                current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=False)
                                                current_frame_crop_bayer.tofile(current_fid_Write_X)
                                        elif flag_bayer_or_rgb == 'rgb':
                                            #TODO: implement random gain and gamma from saved RGB images!
                                            if flag_random_gain_and_gamma:
                                                current_frame_crop.tofile(current_fid_Write_X)
                                            else:
                                                #(1). Old Method - only save image itself and decide upon noising process within the training loop:
                                                current_frame_crop.tofile(current_fid_Write_X)
                                                #(2). New Method - get std image from ronny:




                                ### Uptick crop number: ###
                                current_crop_counter += 1;
                                total_crop_counter += 1;

                toc('Sequence no. ' + str(number_of_sequences_so_far) + ', In Sequence Frame no. ' + str(mini_frame_number))

            ### After writing things down close all fid writers: ###
            if flag_binary_or_images == 'binary':
                for fid in binary_fid_writers_list:
                    fid.close()
            # toc('Number of Seconds So Far: ' + str(number_of_frames_so_far/24))

    if flag_binary_or_images == 'binary':
        for fid in binary_fid_writers_list:
            fid.close()


    ### Test Binary Files Writers: ###
    super_folder = 'G:\Movie_Scenes_bin_files\CropSize_250';
    all_filenames = get_all_filenames_from_folder(super_folder);
    number_of_files = 5;
    number_of_frames_per_file_to_show = 1;
    for bin_file_counter in arange(len(all_filenames)):
        # current_sequence_bin_file_number = np.random.randint(number_of_files);
        binary_full_filename = all_filenames[bin_file_counter]
        print(binary_full_filename)
        fid_Read_X = open(binary_full_filename, 'rb')
        frame_height = np.fromfile(fid_Read_X, 'int32', count=1)[0];
        frame_width = np.fromfile(fid_Read_X, 'int32', count=1)[0];
        number_of_channels = np.fromfile(fid_Read_X, 'int32', count=1)[0];
        data_type = np.fromfile(fid_Read_X, dtype='<U10', count=1)[0]
        data_type = data_type.split('0')[-1]
        max_value = np.fromfile(fid_Read_X, 'float32', count=1)[0];
        number_of_elements_per_image = frame_height*frame_width*number_of_channels

        for i in arange(number_of_frames_per_file_to_show):
            mat_in = np.fromfile(fid_Read_X, data_type, count=number_of_elements_per_image)
            mat_in = mat_in.reshape((frame_height,frame_width,number_of_channels))
            imshow(mat_in)
            pause(0.1)

    fid_Read_X.close()
###################################################################################################################################################################################################################################################################################################



# bla = randn(100,100,3);
# tic()
# for i in arange(100):
#     blo = bla.reshape((3,100,100))
# toc()



# movie_full_filename = 'F:\Movies\Raw Films/Rome Virtual Walking Tour in 4K - Rome City Travel.mkv'
# video_stream = cv2.VideoCapture(movie_full_filename)
# start_frame = 20;
# crop_size = 250;
# number_of_consecutive_frame_to_show = 50
# row_start = randint(2160-crop_size)
# col_start = randint(3840-crop_size)
# flag_different_crop_every_time = True
# for i in arange(start_frame+number_of_consecutive_frame_to_show):
#     flag_frame_available, frame = video_stream.read()
#     if i>start_frame:
#         if flag_different_crop_every_time:
#             row_start = randint(2160 - crop_size)
#             col_start = randint(3840 - crop_size)
#         frame_crop = frame[row_start:row_start+crop_size, col_start:col_start+crop_size, :]
#         imshow(frame_crop)
#         pause(0.01);
#     print(i)






###################################################################################################################################################################################################################################################################################################
def create_binary_files_from_seperated_scene_folders_of_images(movie_full_filenames):
    ### Original Movie: ###
    #(1). Walk Through sub folders and get all .mkv (!!!!) video files:
    # movie_full_filename_filenames = []
    # master_folder_path = 'F:\Movies\Movie_scenes_seperated_into_different_videos'
    # for dirpath, _, fnames in sorted(os.walk(master_folder_path)):  # os.walk!!!!
    #     for fname in sorted(fnames):
    #         if fname.split('.')[-1] == 'mkv':
    #             movie_full_filename_filenames.append(os.path.join(dirpath, fname))
    # #(2). Specific filenames:
    # movie_full_filenames = ['F:\Movies\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']


    ### Seperated Scene Folders: ###
    super_folder_containing_sub_folders_of_scenes = 'F:\TrackingNet';
    scene_folders_list = []
    for dirpath, _, fnames in sorted(os.walk(super_folder_containing_sub_folders_of_scenes)):  # os.walk!!!!
        scene_folders_list.append(dirpath);


    ### Where To Save Results: ###
    super_folder_to_save_everything_at = 'G:\Movie_Scenes_bin_files'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)


    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9;
    max_number_of_sequences = 1e9;
    max_number_of_images = 1e9;
    #(*). Sequene/Time Parameters:
    number_of_frames_per_sequence = int(30*5);  # 30[fps] * #[number of seconds]
    number_of_frames_to_skip_between_sequeces_of_the_same_video = int(30*15);  # 30[fps] * #[number of seconds]
    #(*). Crops and decimation:
    flag_all_crops = False
    crop_sizes = [250]
    number_of_random_crops_per_crop_size = 5;
    crop_size = 250;
    min_width_start = 0;
    max_width_stop = 4000;
    min_height_start = 0;
    max_height_stop = 4000;
    decimation = 1;

    # (1). Images In Folders:
    # flag_binary_or_images = 'images';
    flag_binary_or_images = 'binary';
    flag_bayer_or_rgb = 'rgb';
    flag_random_gain_and_gamma = False;
    # (2). Binary File:
    number_of_images_per_binary_file = 10000000000;
    binary_dtype = 'uint8';


    movie_filename_counter = 0;
    for scene_folder in scene_folders_list:
        movie_filename_counter += 1;

        ### Get all images in scene folder in correct order: ###
        filenames_in_folder = get_image_filenames_from_folder(scene_folder)
        image_filenames_list = []
        for filenames in filenames_in_folder:
            image_filenames_list.append(int32(filenames.split('.')[0].split('\\')[-1]))
        sorted_indices = argsort(image_filenames_list);
        sorted_image_filenames_list = []
        for index in sorted_indices:
            sorted_image_filenames_list.append(filenames_in_folder[index])
        movie_filename = scene_folder.split('\\')


        ### Loop Over Video Stream: ###
        number_of_sequences_of_current_scene_so_far = 0;
        number_of_total_writes_of_current_scene_so_far = 0
        number_of_frames_of_current_scene_so_far = 0
        flag_frame_available = True
        while flag_frame_available and number_of_sequences_of_current_scene_so_far<max_number_of_sequences and number_of_total_writes_of_current_scene_so_far<max_number_of_images and number_of_frames_of_current_scene_so_far<max_number_of_frames:
            tic()
            ### Read Current Frames: ###
            number_of_sequences_of_current_scene_so_far += 1;
            crops_indices = []
            for mini_frame_number in arange(number_of_frames_per_sequence + number_of_frames_to_skip_between_sequeces_of_the_same_video):  #loop over number_of_frame_to_write + number_of_frames_to_wait
                if number_of_frames_of_current_scene_so_far < number_of_images_in_current_folder:
                    frame = read_image_cv2(sorted_image_filenames_list[number_of_frames_of_current_scene_so_far],flag_convert_to_rgb=1,flag_normalize_to_float=0)
                    frame_height, frame_width, number_of_channels = frame.shape
                else:
                    flag_frame_available = False;
                number_of_frames_of_current_scene_so_far += 1

                ### Loop Over Different Crops: ###
                if flag_frame_available and mini_frame_number<number_of_frames_per_sequence:
                    frame_height, frame_width, number_of_channels = frame.shape;
                    data_type_padded = str(frame.dtype).rjust(10,'0'); #get the data type and right fill with zeros to account for the different types: uint8, int8, float32....
                    max_image_value = np.array([np.iinfo(frame.dtype).max]).astype('float32')[0] # !!!!!!! max_value to be later used, if so wanted, to normalize the image. used for generality sake !!!!!!!!!!!!!!!!!!!!!!!!!!

                    if flag_bayer_or_rgb:
                        number_of_channels = 1; #Bayer


                    ### Start off by creating all the binary files writers in a list: ###
                    if flag_binary_or_images == 'binary':
                        binary_fid_writers_list = []
                        for crop_size_counter, crop_size in enumerate(crop_sizes):
                            current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropSize_' + str(crop_size))
                            create_folder_if_needed(current_crop_folder);
                            number_of_width_crops = max(frame_width // crop_size, 1)
                            number_of_height_crops = max(frame_height // crop_size, 1)
                            ### Pick random crops indices: ###
                            if flag_all_crops:
                                random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
                            else:
                                random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                                random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size[crop_size_counter]]
                                # rows_start = randint(min_height_start, min(max_height_stop, frame_height - crop_size))
                                # rows_stop = rows_start + crop_size;
                                # cols_start = randint(min_width_start, min(max_width_stop, frame_width - crop_size))
                                # cols_stop = cols_start + crop_size;
                                # crops_indices.append([rows_start, rows_stop, cols_start, cols_stop])
                            current_crop_counter = 0;
                            for width_crop_index in arange(number_of_width_crops):
                                for height_crop_index in arange(number_of_height_crops):
                                    fid_filename_X = os.path.join(current_crop_folder, 'MovieFilename_' + movie_filename +
                                                                  '_SequenceNo_' + str(number_of_sequences_of_current_scene_so_far) +
                                                                  '_CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)) +
                                                                  '_CropCounter_' + str(current_crop_counter) + '.bin')
                                    fid_Write_X = open(fid_filename_X, 'ab')
                                    binary_fid_writers_list.append(fid_Write_X)
                                    print('crop_size: ' + str(crop_size) + ', written height: ' + str(min(frame_height,crop_size)) + ', written width: ' + str(min(frame_width,crop_size)))
                                    np.array(min(frame_height,crop_size)).tofile(fid_Write_X)
                                    np.array(min(frame_width,crop_size)).tofile(fid_Write_X)
                                    np.array(number_of_channels).tofile(fid_Write_X)
                                    np.array(data_type_padded).tofile(fid_Write_X)
                                    np.array(max_image_value).tofile(fid_Write_X);
                                    current_crop_counter += 1;



                    ### Loop Over all the different crops and write the down: ###
                    total_crop_counter = 0;
                    for crop_size in crop_sizes:
                        number_of_width_crops = max(frame_width // crop_size, 1)
                        number_of_height_crops = max(frame_height // crop_size, 1)
                        current_crop_size_height = min(crop_size, frame_height);
                        current_crop_size_width = min(crop_size, frame_width);

                        current_crop_number = 0;
                        ### Loop over the different crops: ###
                        for width_crop_index in arange(number_of_width_crops):
                            for height_crop_index in arange(number_of_height_crops):
                                if current_crop_number in random_crop_indices:
                                    ### Get Current Crop: ###
                                    current_frame_crop = frame[height_crop_index*current_crop_size_height:(height_crop_index+1)*current_crop_size_height:decimation, width_crop_index*current_crop_size_width:(width_crop_index+1)*current_crop_size_width, :]

                                    ### Save Image: ####
                                    if flag_binary_or_images == 'images':
                                        #(1). Image In Folders:
                                        current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)))
                                        new_sub_folder = os.path.join(current_crop_folder, 'Movie_' + str(movie_filename_counter) + '_Sequence_' + str(number_of_sequences_of_current_scene_so_far) + '_Crop_' + str(current_crop_number))
                                        if not os.path.exists(new_sub_folder):
                                            os.makedirs(new_sub_folder)
                                        save_image_numpy(folder_path=new_sub_folder, filename=str(mini_frame_number) + '.png',
                                                         numpy_array=current_frame_crop, flag_convert_bgr2rgb=False,
                                                         flag_scale=False)

                                    elif flag_binary_or_images == 'binary':
                                        #(2). Binary File:
                                        current_fid_Write_X = binary_fid_writers_list[total_crop_counter]
                                        if flag_bayer_or_rgb == 'bayer':
                                            if flag_random_gain_and_gamma:
                                                current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=True)
                                                current_frame_crop_bayer.tofile(current_fid_Write_X)
                                            else:
                                                current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=False)
                                                current_frame_crop_bayer.tofile(current_fid_Write_X)
                                        elif flag_bayer_or_rgb == 'rgb':
                                            #TODO: implement random gain and gamma from saved RGB images!
                                            if flag_random_gain_and_gamma:
                                                current_frame_crop.tofile(current_fid_Write_X)
                                            else:
                                                # current_frame_crop_tensor_form = np.transpose(current_frame_crop,[])
                                                current_frame_crop.tofile(current_fid_Write_X)


                                ### Uptick crop number: ###
                                current_crop_number += 1;
                                total_crop_counter += 1;


            ### After writing things down close all fid writers: ###
            if flag_binary_or_images == 'binary':
                for fid in binary_fid_writers_list:
                    fid.close()
            toc('Number of Seconds So Far: ' + str(number_of_frames_of_current_scene_so_far/24))

    if flag_binary_or_images == 'binary':
        for fid in binary_fid_writers_list:
            fid.close()


### Test Binary Files Writers: ###
binary_full_filename = 'F:\Movie Scenes Temp\CropSize_100/CropHeight_100_CropWidth_100_CropCounter_0.bin'
fid_Read_X = open(binary_full_filename, 'rb')
frame_height = np.fromfile(fid_Read_X, 'int32', count=1)[0];
frame_width = np.fromfile(fid_Read_X, 'int32', count=1)[0];
number_of_channels = np.fromfile(fid_Read_X, 'int32', count=1)[0];
data_type = np.fromfile(fid_Read_X, dtype='<U10', count=1)[0]
data_type = data_type.split('0')[-1]
max_value = np.fromfile(fid_Read_X, 'float32', count=1)[0];

number_of_elements_per_image = frame_height*frame_width*number_of_channels
mat_in = np.fromfile(fid_Read_X, data_type, count=number_of_elements_per_image)
mat_in = mat_in.reshape((frame_height,frame_width,number_of_channels))
imshow(mat_in)

# fid_Read_X.close()
###################################################################################################################################################################################################################################################################################################












###################################################################################################################################################################################################################################################################################################
def create_new_images_dataset_from_heterogenous_images(movie_full_filenames):

    ### Seperated Scene Folders: ###
    # super_folder_containing_sub_folders_of_scenes = 'F:/NON JPEG IMAGES';
    super_folder_containing_sub_folders_of_scenes = 'C:/Users\dkarl\PycharmProjects\dudykarl\SR_GAN\HR_images\DIV2K_test_HR';
    scene_folders_list = []
    filenames_list = []
    for dirpath, _, fnames in sorted(os.walk(super_folder_containing_sub_folders_of_scenes)):  # os.walk!!!!
        scene_folders_list.append(dirpath);
        for filename in fnames:
            filenames_list.append(os.path.join(dirpath,filename))

    ### Where To Save Results: ###
    super_folder_to_save_everything_at = 'F:/NON_JPEG_IMAGES_BROKEN_INTO_CROPS'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)


    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9;
    max_number_of_sequences = 1e9;
    max_number_of_images = 1e9;
    #(*). Crops and decimation:
    flag_all_crops = True
    crop_sizes = [150]
    number_of_random_crops_per_crop_size = 5;
    min_width_start = 0;
    max_width_stop = 4000;
    min_height_start = 0;
    max_height_stop = 4000;
    decimation = 1;

    # (1). Images In Folders:
    flag_binary_or_images = 'images';
    # flag_binary_or_images = 'binary';
    flag_bayer_or_rgb = 'rgb';
    flag_random_gain_and_gamma = False;
    # (2). Binary File:
    number_of_images_per_binary_file = 10000000000;
    binary_dtype = 'uint8';



    ### Loop Over Images: ###
    # frame_counter = 0;
    # number_of_written_images_so_far = 0;
    for full_filename in filenames_list:
        ### Read Image: ###
        frame = read_image_cv2(full_filename, flag_convert_to_rgb=1, flag_normalize_to_float=0)
        print(full_filename)
        frame_height, frame_width, number_of_channels = frame.shape
        data_type_padded = str(frame.dtype).rjust(10, '0');  # get the data type and right fill with zeros to account for the different types: uint8, int8, float32....
        max_image_value = np.array([np.iinfo(frame.dtype).max]).astype('float32')[0]  # !!!!!!! max_value to be later used, if so wanted, to normalize the image. used for generality sake !!!!!!!!!!!!!!!!!!!!!!!!!!
        frame_counter += 1;

        ### Loop Over Different Crops: ###
        if frame_counter > max_number_of_frames:
            break;

        if frame_counter < max_number_of_frames:
            if flag_bayer_or_rgb=='bayer':
                number_of_channels = 1;  # Bayer


            ### Start off by creating all the binary files writers in a list: ###
            random_crop_indices_per_crop_size = []
            binary_fid_writers_list = []
            for crop_size_counter, crop_size in enumerate(crop_sizes):
                current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'Binary_CropSize_' + str(crop_size))
                create_folder_if_needed(current_crop_folder);
                number_of_width_crops = max(frame_width // crop_size, 1)
                number_of_height_crops = max(frame_height // crop_size, 1)
                ### Pick random crops indices: ###
                if flag_all_crops:
                    random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
                else:
                    random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                    random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size]  # rows_start = randint(min_height_start, min(max_height_stop, frame_height - crop_size))  # rows_stop = rows_start + crop_size;  # cols_start = randint(min_width_start, min(max_width_stop, frame_width - crop_size))  # cols_stop = cols_start + crop_size;  # crops_indices.append([rows_start, rows_stop, cols_start, cols_stop])
                random_crop_indices_per_crop_size.append(random_crop_indices)
                ###################################
                current_crop_counter = 0;

                if flag_binary_or_images == 'binary':
                    for width_crop_index in arange(number_of_width_crops):
                        for height_crop_index in arange(number_of_height_crops):
                            fid_filename_X = os.path.join(current_crop_folder, '_CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)) + '_CropCounter_' + str(current_crop_counter) + '.bin')
                            fid_Write_X = open(fid_filename_X, 'ab')
                            binary_fid_writers_list.append(fid_Write_X)
                            print('crop_size: ' + str(crop_size) + ', written height: ' + str(min(frame_height, crop_size)) + ', written width: ' + str(min(frame_width, crop_size)))
                            np.array(min(frame_height, crop_size)).tofile(fid_Write_X)
                            np.array(min(frame_width, crop_size)).tofile(fid_Write_X)
                            np.array(number_of_channels).tofile(fid_Write_X)
                            np.array(data_type_padded).tofile(fid_Write_X)
                            np.array(max_image_value).tofile(fid_Write_X);
                            current_crop_counter += 1;


            ### Loop Over all the different crops and write the down: ###
            total_crop_counter = 0;
            for crop_size_index, crop_size in enumerate(crop_sizes):
                number_of_width_crops = max(frame_width // crop_size, 1)
                number_of_height_crops = max(frame_height // crop_size, 1)
                current_crop_size_height = min(crop_size, frame_height);
                current_crop_size_width = min(crop_size, frame_width);

                current_crop_number = 0;
                ### Loop over the different crops: ###
                random_crop_indices = random_crop_indices_per_crop_size[crop_size_index]
                for width_crop_index in arange(number_of_width_crops):
                    for height_crop_index in arange(number_of_height_crops):
                        if current_crop_number in random_crop_indices:
                            ### Get Current Crop: ###
                            current_frame_crop = frame[height_crop_index * current_crop_size_height:(height_crop_index + 1) * current_crop_size_height:decimation, width_crop_index * current_crop_size_width:(width_crop_index + 1) * current_crop_size_width, :]

                            ### Save Image: ####
                            if flag_binary_or_images == 'images':
                                current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'Images_CropSize_' + str(crop_size))
                                create_folder_if_needed(current_crop_folder);
                                save_image_numpy(folder_path=current_crop_folder, filename=str(number_of_written_images_so_far) + '.png', numpy_array=current_frame_crop, flag_convert_bgr2rgb=True, flag_scale=False)

                            elif flag_binary_or_images == 'binary':
                                # (2). Binary File:
                                current_fid_Write_X = binary_fid_writers_list[total_crop_counter]
                                if flag_bayer_or_rgb == 'bayer':
                                    if flag_random_gain_and_gamma:
                                        current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=True)
                                        current_frame_crop_bayer.tofile(current_fid_Write_X)
                                    else:
                                        current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=False)
                                        current_frame_crop_bayer.tofile(current_fid_Write_X)
                                elif flag_bayer_or_rgb == 'rgb':
                                    # TODO: implement random gain and gamma from saved RGB images!
                                    if flag_random_gain_and_gamma:
                                        current_frame_crop.tofile(current_fid_Write_X)
                                    else:
                                        # current_frame_crop_tensor_form = np.transpose(current_frame_crop,[])
                                        current_frame_crop.tofile(current_fid_Write_X)

                        ### Uptick crop number: ###
                        current_crop_number += 1;
                        total_crop_counter += 1;
                        number_of_written_images_so_far += 1;


    ### After writing things down close all fid writers: ###
    if flag_binary_or_images == 'binary':
        for fid in binary_fid_writers_list:
            fid.close()




### Test Binary Files Writers: ###
binary_full_filename = 'F:\Movie Scenes Temp\CropSize_100/CropHeight_100_CropWidth_100_CropCounter_0.bin'
fid_Read_X = open(binary_full_filename, 'rb')
frame_height = np.fromfile(fid_Read_X, 'int32', count=1)[0];
frame_width = np.fromfile(fid_Read_X, 'int32', count=1)[0];
number_of_channels = np.fromfile(fid_Read_X, 'int32', count=1)[0];
data_type = np.fromfile(fid_Read_X, dtype='<U10', count=1)[0]
data_type = data_type.split('0')[-1]
max_value = np.fromfile(fid_Read_X, 'float32', count=1)[0];

number_of_elements_per_image = frame_height*frame_width*number_of_channels
mat_in = np.fromfile(fid_Read_X, data_type, count=number_of_elements_per_image)
mat_in = mat_in.reshape((frame_height,frame_width,number_of_channels))
imshow(mat_in)

# fid_Read_X.close()
###################################################################################################################################################################################################################################################################################################



























###################################################################################################################################################################################################################################################################################################
def create_binary_files_from_non_video_images(movie_full_filenames):
    ### Original Movie: ###
    #(1). Walk Through sub folders and get all .mkv (!!!!) video files:
    # movie_full_filename_filenames = []
    # master_folder_path = 'F:\Movies\Movie_scenes_seperated_into_different_videos'
    # for dirpath, _, fnames in sorted(os.walk(master_folder_path)):  # os.walk!!!!
    #     for fname in sorted(fnames):
    #         if fname.split('.')[-1] == 'mkv':
    #             movie_full_filename_filenames.append(os.path.join(dirpath, fname))
    # #(2). Specific filenames:
    # movie_full_filenames = ['F:\Movies\Raw Films/Fast_walking_from_Shinjuku_to_Shibuya.mkv']


    ### Seperated Scene Folders: ###
    super_folder_containing_sub_folders_of_scenes = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet';
    scene_folders_list = []
    for dirpath, _, fnames in sorted(os.walk(super_folder_containing_sub_folders_of_scenes)):  # os.walk!!!!
        scene_folders_list.append(dirpath);


    ### Where To Save Results: ###
    super_folder_to_save_everything_at = 'G:\Movie_Scenes_bin_files/Imagenet'
    if not os.path.exists(super_folder_to_save_everything_at):
        os.makedirs(super_folder_to_save_everything_at)


    ### Parameters: ###
    #(*). Max numbers of images/videos/sequences
    max_number_of_frames = 1e9;
    max_number_of_sequences = 1e9;
    max_number_of_images = 100;
    number_of_different_sequences_per_image = 1;
    #(*). Sequene/Time Parameters:
    number_of_frames_per_sequence = int(1);  # 30[fps] * #[number of seconds]
    number_of_frames_to_skip_between_sequeces_of_the_same_video = int(0);  # 30[fps] * #[number of seconds]
    #(*). Crops and decimation:
    flag_all_crops = False
    crop_sizes = [250]
    number_of_random_crops_per_crop_size = 5;
    crop_size = 250;
    min_width_start = 0;
    max_width_stop = 4000;
    min_height_start = 0;
    max_height_stop = 4000;
    decimation = 1;

    # (1). Images In Folders:
    # flag_binary_or_images = 'images';
    flag_binary_or_images = 'binary';
    flag_bayer_or_rgb = 'rgb';
    flag_random_gain_and_gamma = False;
    # (2). Binary File:
    number_of_images_per_binary_file = 10000000000;
    binary_dtype = 'uint8';


    movie_filename_counter = 0;
    for scene_folder in scene_folders_list:
        movie_filename_counter += 1;

        ### Get all images in scene folder in correct order: ###
        filenames_in_folder = get_image_filenames_from_folder(scene_folder)
        image_filenames_list = []
        for filenames in filenames_in_folder:
            image_filenames_list.append(int32(filenames.split('.')[0].split('\\')[-1]))
        sorted_indices = argsort(image_filenames_list);
        sorted_image_filenames_list = image_filenames_list
        for index in sorted_indices:
            sorted_image_filenames_list.append(filenames_in_folder[index])
        subfolder_names = scene_folder.split('\\')


        ### Loop Over Video Stream: ###
        number_of_sequences_of_current_scene_so_far = 0;
        number_of_total_writes_of_current_scene_so_far = 0
        number_of_images_from_current_folder_so_far = 0
        while number_of_sequences_of_current_scene_so_far<max_number_of_sequences and number_of_total_writes_of_current_scene_so_far<max_number_of_images and number_of_images_from_current_folder_so_far<max_number_of_frames:
            tic()
            ### Read Current Image: ###
            number_of_sequences_of_current_scene_so_far += 1;
            crops_indices = []
            if number_of_images_from_current_folder_so_far < number_of_images_in_current_folder:
                frame = read_image_cv2(sorted_image_filenames_list[number_of_images_from_current_folder_so_far], flag_convert_to_rgb=1, flag_normalize_to_float=0)
                frame_height, frame_width, number_of_channels = frame.shape
            else:
                flag_frame_available = False;
            number_of_images_from_current_folder_so_far += 1


            ### For the same image create number_of_different_sequences_per_image sequences: ###
            for mini_sequence_number in arange(number_of_different_sequences_per_image):
                ### For each sequence loop over the different "frames" to be created: ###
                for mini_frame_number in arange(number_of_frames_per_sequence):  #loop over number_of_frame_to_write + number_of_frames_to_wait

                    ### Loop Over Different Crops: ###
                    if mini_frame_number < number_of_frames_per_sequence:
                        frame_height, frame_width, number_of_channels = frame.shape;
                        data_type_padded = str(frame.dtype).rjust(10,'0'); #get the data type and right fill with zeros to account for the different types: uint8, int8, float32....
                        max_image_value = np.array([np.iinfo(frame.dtype).max]).astype('float32')[0] # !!!!!!! max_value to be later used, if so wanted, to normalize the image. used for generality sake !!!!!!!!!!!!!!!!!!!!!!!!!!

                        if flag_bayer_or_rgb:
                            number_of_channels = 1; #Bayer


                        ### Start off by creating all the binary files writers in a list: ###
                        if flag_binary_or_images == 'binary':
                            binary_fid_writers_list = []
                            for crop_size_counter, crop_size in enumerate(crop_sizes):
                                current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropSize_' + str(crop_size))
                                create_folder_if_needed(current_crop_folder);
                                number_of_width_crops = max(frame_width // crop_size, 1)
                                number_of_height_crops = max(frame_height // crop_size, 1)
                                ### Pick random crops indices: ###
                                if flag_all_crops:
                                    random_crop_indices = arange(number_of_width_crops * number_of_height_crops)
                                else:
                                    random_crop_indices = np.random.permutation(arange(number_of_width_crops * number_of_height_crops))
                                    random_crop_indices = random_crop_indices[0:number_of_random_crops_per_crop_size[crop_size_counter]]

                                current_crop_counter = 0;
                                for width_crop_index in arange(number_of_width_crops):
                                    for height_crop_index in arange(number_of_height_crops):
                                        fid_filename_X = os.path.join(current_crop_folder, 'SubFolderName_' + subfolder_names +
                                                                      '_SequenceNo_' + str(mini_sequence_number) +
                                                                      '_CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)) +
                                                                      '_CropCounter_' + str(current_crop_counter) + '.bin')
                                        fid_Write_X = open(fid_filename_X, 'ab')
                                        binary_fid_writers_list.append(fid_Write_X)
                                        print('crop_size: ' + str(crop_size) + ', written height: ' + str(min(frame_height,crop_size)) + ', written width: ' + str(min(frame_width,crop_size)))
                                        np.array(min(frame_height,crop_size)).tofile(fid_Write_X)
                                        np.array(min(frame_width,crop_size)).tofile(fid_Write_X)
                                        np.array(number_of_channels).tofile(fid_Write_X)
                                        np.array(data_type_padded).tofile(fid_Write_X)
                                        np.array(max_image_value).tofile(fid_Write_X);
                                        current_crop_counter += 1;



                        ### Loop Over all the different crops and write the down: ###
                        total_crop_counter = 0;
                        for crop_size in crop_sizes:
                            number_of_width_crops = max(frame_width // crop_size, 1)
                            number_of_height_crops = max(frame_height // crop_size, 1)
                            current_crop_size_height = min(crop_size, frame_height);
                            current_crop_size_width = min(crop_size, frame_width);

                            current_crop_number = 0;
                            ### Loop over the different crops: ###
                            for width_crop_index in arange(number_of_width_crops):
                                for height_crop_index in arange(number_of_height_crops):
                                    if current_crop_number in random_crop_indices:
                                        ### Get Current Crop: ###
                                        current_frame_crop = frame[height_crop_index*current_crop_size_height:(height_crop_index+1)*current_crop_size_height:decimation, width_crop_index*current_crop_size_width:(width_crop_index+1)*current_crop_size_width, :]

                                        ### Save Image: ####
                                        if flag_binary_or_images == 'images':
                                            #(1). Image In Folders:
                                            current_crop_folder = os.path.join(super_folder_to_save_everything_at, 'CropHeight_' + str(min(crop_size, frame_height)) + '_CropWidth_' + str(min(crop_size, frame_width)))
                                            new_sub_folder = os.path.join(current_crop_folder, 'Movie_' + str(movie_filename_counter) + '_Sequence_' + str(number_of_sequences_of_current_scene_so_far) + '_Crop_' + str(current_crop_number))
                                            if not os.path.exists(new_sub_folder):
                                                os.makedirs(new_sub_folder)
                                            save_image_numpy(folder_path=new_sub_folder, filename=str(mini_frame_number) + '.png',
                                                             numpy_array=current_frame_crop, flag_convert_bgr2rgb=False,
                                                             flag_scale=False)

                                        elif flag_binary_or_images == 'binary':
                                            #(2). Binary File:
                                            current_fid_Write_X = binary_fid_writers_list[total_crop_counter]
                                            if flag_bayer_or_rgb == 'bayer':
                                                if flag_random_gain_and_gamma:
                                                    current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=True)
                                                    current_frame_crop_bayer.tofile(current_fid_Write_X)
                                                else:
                                                    current_frame_crop_bayer = create_ground_truth_Bayer_and_RGB_with_random_gain_and_gamma(current_frame_crop, flag_random_gamma_and_gain=False)
                                                    current_frame_crop_bayer.tofile(current_fid_Write_X)
                                            elif flag_bayer_or_rgb == 'rgb':
                                                if flag_random_gain_and_gamma:
                                                    current_frame_crop.tofile(current_fid_Write_X)
                                                else:
                                                    # current_frame_crop_tensor_form = np.transpose(current_frame_crop,[])
                                                    current_frame_crop.tofile(current_fid_Write_X)


                                    ### Uptick crop number: ###
                                    current_crop_number += 1;
                                    total_crop_counter += 1;


                ### After writing things down close all fid writers: ###
                if flag_binary_or_images == 'binary':
                    for fid in binary_fid_writers_list:
                        fid.close()
                toc('Number of Seconds So Far: ' + str(number_of_images_from_current_folder_so_far/24))

    if flag_binary_or_images == 'binary':
        for fid in binary_fid_writers_list:
            fid.close()


### Test Binary Files Writers: ###
binary_full_filename = 'F:\Movie Scenes Temp\CropSize_100/CropHeight_100_CropWidth_100_CropCounter_0.bin'
fid_Read_X = open(binary_full_filename, 'rb')
frame_height = np.fromfile(fid_Read_X, 'int32', count=1)[0];
frame_width = np.fromfile(fid_Read_X, 'int32', count=1)[0];
number_of_channels = np.fromfile(fid_Read_X, 'int32', count=1)[0];
data_type = np.fromfile(fid_Read_X, dtype='<U10', count=1)[0]
data_type = data_type.split('0')[-1]
max_value = np.fromfile(fid_Read_X, 'float32', count=1)[0];

number_of_elements_per_image = frame_height*frame_width*number_of_channels
mat_in = np.fromfile(fid_Read_X, data_type, count=number_of_elements_per_image)
mat_in = mat_in.reshape((frame_height,frame_width,number_of_channels))
imshow(mat_in)

# fid_Read_X.close()
###################################################################################################################################################################################################################################################################################################























###################################################################################################################################################################################################################################################################################################
def create_binary_dataset_from_seperated_scenes_images():
    super_folder_containing_scene_folders = 'C:/Users\dkarl\PycharmProjects\dudykarl\Movies/Unprocessed_Scene_Folders'
    max_number_of_videos = 3;
    min_number_of_images_per_video = 10;
    crop_sizes = [100,200]

    # ### Create New Folders: ###
    # for i in arange(150):
    #     if not os.path.exists(os.path.join(super_folder_containing_scene_folders, str(i).rjust(3))):
    #         os.makedirs(os.path.join(super_folder_containing_scene_folders, str(i).rjust(3)))




    tic()
    folder_counter = 0;
    for directory_path, directory_name, file_names_in_sub_directory in os.walk(root_folder):
        ### If number of videos/directories so far exceeds maximum then break: ###
        if folder_counter > max_number_of_videos and max_number_of_videos != -1:
            break;

        ### If number of images in directory exceed minimum then keep going: ###
        if len(file_names_in_sub_directory) >= min_number_of_images_per_video:
            folder_counter += 1;
            current_folder_filenames_list = get_image_filenames_from_folder(super_folder_containing_scene_folders)

            ### Loop Over Crop Sizes Wanted And Create Binary Files With Sequential Images Of Size Crop: ###
            for current_crop_size in crop_sizes:

                ### Go over images in the folder and add them to the filenames list: ###
                images_counter = 0;
                for current_filename in file_names_in_sub_directory:
                    full_filename = os.path.join(directory_path, current_filename);
                    if images_counter == number_of_images_perl_video:
                        break;
                    elif is_image_file(full_filename):
                        images_counter += 1
                        image_filenames_list_of_lists[folder_counter - 1].append('{}'.format(full_filename))
                        if flag_load_to_RAM:
                            images.append(read_image_cv2(full_filename))

    toc('End Of DataSet')
###################################################################################################################################################################################################################################################################################################










###################################################################################################################################################################################################################################################################################################
def split_video_into_number_of_parts():
    ### Original Movie: ###
    movie_full_filenames = ['C:/Users\dkarl/Downloads/Full Movie 4k ULtra HD MSC MUSICA Cruise Tour Medi.mkv']
    # movie_full_filename = 'C:/Users\dkarl/Downloads/welcome back full movie hd.mp4'



    total_number_of_frames = 97500;
    number_of_parts = 25
    frames_per_part = 97500//number_of_parts;

    frame_height = 3840
    frame_width = 2160

    movie_filename_counter = 0;
    for movie_full_filename in movie_full_filenames:
        ### Open Movie Stream: ###
        video_stream = cv2.VideoCapture(movie_full_filename)
        movie_filename_counter += 1;

        for part_counter in arange(number_of_parts):
            ### Create New File For This Movie's Part: ###
            created_video_filename = 'C:/Users\dkarl/Downloads//' + movie_full_filename.split('.')[0].split('/')[-1] + ' Part' + str(part_counter) + '.mkv'
            fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
            video_writer = cv2.VideoWriter(created_video_filename, fourcc, 25.0, (frame_height, frame_width))

            ### Loop Over Frames For This Part: ###
            for current_part_frame_counter in arange(frames_per_part):
                flag_frame_available, frame = video_stream.read()
                video_writer.write(frame);
                print('part number ' + str(part_counter) + ', frame number ' + str(current_part_frame_counter))

            ### After Finishing Writing This Part To Disk Release The Video Writer: ###
            cv2.destroyAllWindows()
            video_writer.release()


    cv2.destroyAllWindows()
    video_writer.release()
###################################################################################################################################################################################################################################################################################################






###################################################################################################################################################################################################################################################################################################
def split_video_into_scenes_ContextualLoss():
    movie_full_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl/Full_Movie_4k_ULtra_HD_MSC_MUSICA_Cruise_Tour_Medi_Part0.mkv'
    video_stream = cv2.VideoCapture(movie_full_filename)
    max_number_of_frames = 25*100;
    current_step = 0

    losses_list = []
    Contextual_Loss_Function = Contextual_Loss(2)
    loss_threshold = 2;
    direct_pixels_Contextual_patch_factor = 5;

    new_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl\Movie_Scenes\Movie1'
    scene_counter = 0;
    created_video_filename = os.path.join(new_folder, str(scene_counter) + '.mkv')
    fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
    video_writer = cv2.VideoWriter(created_video_filename, fourcc, 25.0, (200, 200))

    while current_step < max_number_of_frames:
        tic()

        flag_frame_available, current_frame = video_stream.read();
        current_frame = cv2.resize(current_frame, dsize=(200,200))
        current_frame_tensor = torch.Tensor(np.transpose(current_frame,[2,0,1])).unsqueeze(0).cuda()
        current_frame_tensor = extract_patches_2D(current_frame_tensor, [int(current_frame_tensor.shape[2] / direct_pixels_Contextual_patch_factor), int(current_frame_tensor.shape[3] / direct_pixels_Contextual_patch_factor)])

        ### Compare Current To Previous Frame: ###
        if current_step > 1:
            contextual_loss = Contextual_Loss_Function(current_frame_tensor, previous_frame_tensor);
            current_contextual_loss = contextual_loss.item();

            if current_contextual_loss > loss_threshold:
                ### Create new scene writer: ###
                video_writer.release()
                scene_counter += 1
                created_video_filename = os.path.join(new_folder ,str(scene_counter) + '.mkv')
                fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
                video_writer = cv2.VideoWriter(created_video_filename, fourcc, 25.0, (200, 200))
            else:
                video_writer.write(current_frame);

            losses_list.append(contextual_loss.item())
            print(contextual_loss.item())

        ### Assign Current Frame To Previous Frame: ###
        previous_frame_tensor = torch.Tensor(np.transpose(current_frame, [2, 0, 1])).unsqueeze(0).cuda()
        previous_frame_tensor = extract_patches_2D(previous_frame_tensor, [int(previous_frame_tensor.shape[2] / direct_pixels_Contextual_patch_factor), int(previous_frame_tensor.shape[3] / direct_pixels_Contextual_patch_factor)])
        previous_frame_tensor = current_frame_tensor

        current_step += 1
        toc()

    losses_numpy = np.array(losses_list)
###################################################################################################################################################################################################################################################################################################





###################################################################################################################################################################################################################################################################################################
def split_video_into_all_image_frames():
    movie_full_filename = 'C:/Users\dkarl/Downloads/Full Movie 4k ULtra HD MSC MUSICA Cruise Tour Medi.mkv'
    saved_images_filename = 'C:/Users\dkarl/Downloads/Movie_Images'
    if not os.path.exists(saved_images_filename):
        os.makedirs(saved_images_filename)

    video_stream = cv2.VideoCapture(movie_full_filename)
    max_number_of_frames = 100000000;
    current_step = 0

    while current_step < max_number_of_frames:
        tic()

        flag_frame_available, current_frame = video_stream.read();
        if flag_frame_available==False:
            break;
        save_image_numpy(folder_path=saved_images_filename,
                         filename=str(current_step).rjust(8,'0') + '.png',
                         numpy_array=current_frame, flag_convert_bgr2rgb=False, flag_scale=False)
        current_step += 1
        toc()
###################################################################################################################################################################################################################################################################################################










###################################################################################################################################################################################################################################################################################################
def compare_videos():
    movie_full_filename1 = 'C:/Users\dkarl/Downloads/Full Movie 4k ULtra HD MSC MUSICA Cruise Tour Medi.mkv'
    movie_full_filename2 = 'C:/Users\dkarl/Downloads/Full Movie 4k ULtra HD MSC MUSICA Cruise Tour Medi Part0.mkv'
    video_stream1 = cv2.VideoCapture(movie_full_filename1)
    video_stream2 = cv2.VideoCapture(movie_full_filename2)
    flag_frame_available2, frame2 = video_stream1.read()
    number_of_images = 100000000000;
    for i in arange(number_of_images):
        flag_frame_available1, frame1 = video_stream1.read()
        flag_frame_available2, frame2 = video_stream2.read()
        print(sum(frame1-frame2))
        imshow(np.concatenate((frame1,frame2,frame1-frame2), axis=1))
        pause(0.1)


    movie_full_filename = 'C:/Users\dkarl/Downloads/Full Movie 4k ULtra HD MSC MUSICA Cruise Tour Medi.mkv'
    video_stream = cv2.VideoCapture(movie_full_filename)
    number_of_images = 10;
    for i in arange(number_of_images):
        #read frame:
        flag_frame_available, frame_from_video = video_stream.read();
        if flag_frame_available==False:
            break;
        frame_from_video = cv2.cvtColor(frame_from_video,cv2.COLOR_BGR2RGB)
        #save frame:
        save_image_numpy(folder_path='C:/Users\dkarl/Downloads/', filename='bla.png',
                         numpy_array=frame_from_video, flag_convert_bgr2rgb=False, flag_scale=False)
        #read frame:
        frame_from_image = read_image_cv2(path='C:/Users\dkarl/Downloads/bla.png', flag_convert_to_rgb=0, flag_normalize_to_float=0)
        #compare read and saved frames:
        print(sum(frame_from_image-frame_from_video))

        imshow(np.concatenate((frame_from_video,frame_from_image), 1))
###################################################################################################################################################################################################################################################################################################













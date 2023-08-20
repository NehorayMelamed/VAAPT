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
# from ESRGAN_utils import *
# from ESRGAN_deep_utils import *
# from ESRGAN_basic_Blocks_and_Layers import *
# from ESRGAN_Models import *
# from ESRGAN_Losses import *
# from ESRGAN_Optimizers import *
# from ESRGAN_Visualizers import *
# from ESRGAN_dataset import *
# from ESRGAN_OPT import *


#TODO: i need to fix this more generally, perhapse even delete some files, but gif and png files in the imagenet i got from or don't work so now TEMPORARILY i'm not including png in the IMG_EXTENTIONS
IMG_EXTENSIONS = ['.jpg',
                  '.JPG',
                  '.jpeg',
                  '.JPEG',
                  '.png',
                  '.PNG',
                  '.ppm',
                  '.PPM',
                  '.bmp',
                  '.BMP']
IMG_EXTENSIONS_NO_PNG = ['.jpg',
                  '.JPG',
                  '.jpeg',
                  '.JPEG',
                  '.ppm',
                  '.PPM',
                  '.bmp',
                  '.BMP']
IMG_EXTENSIONS_PNG = ['.png','.PNG']


########################################################################################################################################
####  Auxiliaries: ####
#######################
def get_torch_parents_list(input_object):
    return inspect.getmro(input_object.__class__)

#(*). Dictionary to Class Attributes Class:
class dictionary_to_class_attributes:
    def __init__(self, dictionary):
        #Set all dictionary keys and values as properties:
        for k, v in dictionary.items():
            setattr(self, k, v)
        #Set the dictionary itself as a property called variables_dictionary:
        setattr(self,'variables_dictionary',dictionary);

    def update_object_with_dictionary(self,dictionary):
        #Go through all dictionary items, if the key exists just update the value, if it doesn't then create it:
        for k,v in dictionary.items():
            self.variables_dictionary[k] = v;
            setattr(self,k,v);

    def update_object_with_dictionary_no_overwrite(self,dictionary):
        for k,v in dictionary.items():
            if k not in self.variables_dictionary.keys():
                self.variables_dictionary[k] = v;
                setattr(self,k,v);



#(*). Dictionary functions:
def merge_dictionaries(main_dictionary,changes_dictionary):
    for k,v in changes_dictionary.items():
        main_dictionary[k] = changes_dictionary[k];

def merge_dictionaries_if_keys_exist(main_dictionary,changes_dictionary):
    for k,v in changes_dictionary.items():
        if k in main_dictionary.keys():
            main_dictionary[k] = changes_dictionary[k];
    return main_dictionary

def merge_dictionaries_where_keys_dont_exist(main_dictionary,defaults_dictionary):
    for k,v in defaults_dictionary.items():
        if k not in main_dictionary.keys():
            main_dictionary[k] = v;
    return main_dictionary



def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(float(int(i[:2], 16))/256, float(int(i[2:4], 16))/256, float(int(i[4:], 16))/256) for i in colors]


def get_available_RAM_in_bytes(percent_of_RAM_to_return=0.8):
    return psutil.virtual_memory().free * percent_of_RAM_to_return;


def double_tuple(input_number):
    if type(input_number)==tuple or type(input_number)==list:
        if len(input_number)==1:
            return (input_number[0],input_number[0]);
        else:
            return input_number;
    else:
        return (input_number,input_number)

def double_tuple_antisymmetric(input_number):
    if type(input_number)==tuple or type(input_number)==list:
        if len(input_number)==1:
            return (input_number[0],-input_number[0]);
        else:
            return input_number;
    else:
        return (input_number,-input_number)


def upscale_number_modified(input_number, upscale_factor, final_number_base_multiple, upper_clip_threshold):
    # Multiply input number by upscale_factor:
    final_number = int(input_number * upscale_factor)
    # Round it downwards if necesasry to make it a multiple of scale:
    final_number = (final_number // final_number_base_multiple) * final_number_base_multiple
    return upper_clip_threshold if final_number < upper_clip_threshold else final_number


def round_number_to_whole_multiple_of_basenum(input_numbers, base_number):
    flag_is_tuple = (type(input_numbers)==tuple);
    if flag_is_tuple:
        input_numbers = list(input_numbers)
    if np.isscalar(input_numbers):
        return input_numbers // base_number * base_number;
    else:
        for i,input_number in enumerate(input_numbers):
            input_numbers[i] = input_number//base_number * base_number;
        if flag_is_tuple:
            input_numbers = tuple(input_numbers);
        return input_numbers


#Useful for if you want to only take those elements where a condition is satisfied:
def indices_from_logical(logical_array):
    logical_array = np.asarray(logical_array*1);
    return np.reshape(np.argwhere(logical_array==1),-1);


def find(condition_logical_array, mat_in=None):
    """" Like Matlab's Find:
         condition_logical_array is something like mat_in>1, which is a logical (bolean) array 
    """""
    #Use: good_elements = find(mat_in>5, mat_in)
    indices = np.asarray(condition_logical_array.nonzero())
    if mat_in==None:
        values = None;
    else:
        values = mat_in(condition_logical_array);
    return indices,values


def scientific_notation(input_number,number_of_digits_after_point=2):
    format_string = '{:.' + str(number_of_digits_after_point) + 'e}'
    return format_string.format(input_number)

def decimal_notation(input_number, number_of_digits_after_point=2):
    output_number = int(input_number*(10**number_of_digits_after_point))/(10**number_of_digits_after_point)
    return str(output_number)
########################################################################################################################################









########################################################################################################################################
####  Numpy/Array Functions: ####
#################################
def create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels=10, N=512, polarization=0, flag_gauss_circ=1,
                                              flag_normalize=1, flag_imshow=0):
    #    import numpy
    #    from numpy import arange
    #    from numpy.random import *

    # #Parameters:
    # speckle_size_in_pixels = 10;
    # N = 512;
    # polarization = 0;
    # flag_gauss_circ = 0;

    # Calculations:
    wf = (N / speckle_size_in_pixels);

    if flag_gauss_circ == 1:
        x = arange(-N / 2, N / 2, 1);
        distance_between_the_two_beams_x = 0;
        distance_between_the_two_beams_y = 0;
        [X, Y] = meshgrid(x, x);
        beam_one = exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 +
                          (Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2);
        beam_two = exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 +
                          (Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2);
        total_beam = beam_one + beam_two;
        total_beam = total_beam / sqrt(sum(sum(abs(total_beam) ** 2)));
    else:
        x = arange(-N / 2, N / 2, 1) * 1;
        y = x;
        [X, Y] = meshgrid(x, y);
        c = 0;
        distance_between_the_two_beams_y = 0;
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (
                    Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2);
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (
                    Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2);
        total_beam = beam_one + beam_two;

    # Polarization:
    if (polarization > 0 & polarization < 1):
        beam_one = total_beam * exp(2 * pi * 1j * 10 * randn(N, N));
        beam_two = total_beam * exp(2 * pi * 1j * 10 * randn(N, N));
        speckle_pattern1 = fftshift(fft2(fftshift(beam_one)));
        speckle_pattern2 = fftshift(fft2(fftshift(beam_two)));
        speckle_pattern_total_intensity = (1 - polarization) * abs(speckle_pattern1) ** 2 + \
                                          polarization * abs(speckle_pattern2) ** 2;
    else:
        total_beam = total_beam * exp(2 * pi * 1j * (10 * randn(N, N)));
        speckle_pattern1 = fftshift(fft2(fftshift(total_beam)));
        speckle_pattern2 = numpy.empty_like(speckle_pattern1);
        speckle_pattern_total_intensity = np.abs(speckle_pattern1) ** 2;

    if flag_normalize == 1: speckle_pattern_total_intensity = normalize_array(speckle_pattern_total_intensity,(0,1))
    if flag_imshow == 1: imshow(speckle_pattern_total_intensity); colorbar()

    return speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2;



def create_speckles_of_certain_size_in_pixels_multichannel(speckle_size_in_pixels=10, image_shape=(224,224,3), min_max_values=(0,1)):
    mat_out = np.random.uniform(min_max_values[0], min_max_values[1], image_shape)
    max_size = max(image_shape[0:2]);
    for channel_counter in arange(0,image_shape[2]):
        current_channel,_,_ = create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels, max_size, 0,1,0,0);
        current_channel = normalize_array(current_channel,min_max_values);
        mat_out[:, :, channel_counter] = current_channel[0:image_shape[0]+1, 0:image_shape[1]+1];
    return mat_out



start = -1
end = -1
def get_center_number_of_pixels(mat_in,number_of_center_pixels):
    mat_in_shape = numpy.shape(mat_in);
    mat_in_rows = mat_in_shape[0];
    mat_in_cols = mat_in_shape[1];
    mat_in_rows_excess = mat_in_rows - number_of_center_pixels;
    mat_in_cols_excess = mat_in_cols - number_of_center_pixels;
    return mat_in[int(start + mat_in_rows_excess/2) : int(end-mat_in_rows_excess/2) , \
                  int(start + mat_in_cols_excess/2) : int(end-mat_in_cols_excess/2)];




def normalize_array(mat_in, min_max_values=(0,1)):
    mat_in_normalized = (mat_in - mat_in.min()) / (mat_in.max()-mat_in.min()) * (min_max_values[1]-min_max_values[0]) + min_max_values[0]
    return mat_in_normalized

def scale_array_to_range(mat_in, min_max_values=(0,1)):
    mat_in_normalized = (mat_in - mat_in.min()) / (mat_in.max()-mat_in.min()) * (min_max_values[1]-min_max_values[0]) + min_max_values[0]
    return mat_in_normalized
########################################################################################################################################










########################################################################################################################################
####  Path Functions: ####
################################
def path_get_current_working_directory():
    return os.getcwd();

def path_get_mother_folder_path(path):
    return os.path.dirname(os.path.realpath(path))

def path_get_folder_name(path):
    return os.path.basename(path)

def path_get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def path_change_current_working_directory(path):
    return os.chdir(path)

def path_make_directory_from_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_make_directories_from_paths(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def path_make_directory_and_rename_if_needed(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def path_make_path_if_none_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def path_is_valid_directory(path):
    return os.path.isdir(path)


def get_path_size_in_MB(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size/1e6


#########################################################################################################################################









#########################################################################################################################################
####  Reading Images: ####
##########################
def assert_image_okay_v1():
    1;

def is_image_file(filename,allowed_endings=IMG_EXTENSIONS_NO_PNG):
    return any(list((filename.endswith(extension) for extension in allowed_endings)))

def get_image_filenames_from_folder(path,allowed_endings=IMG_EXTENSIONS_PNG, flag_recursive=False):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    image_filenames = []
    #Look recursively at all current and children path's in given path and get all image files
    for dirpath, _, fnames in sorted(os.walk(path)): #os.walk!!!!
        for fname in sorted(fnames):
            if is_image_file(fname,allowed_endings=allowed_endings):
                img_path = os.path.join(dirpath, fname)
                image_filenames.append(img_path)
        if flag_recursive == False:
            break;
    assert image_filenames, '{:s} has no valid image file'.format(path) #assert images is not an empty list
    return image_filenames


def get_binary_filenames_from_folder(path, flag_recursive=False):
    binary_filenames_list = []
    # Look recursively at all current and children path's in given path and get all binary files
    for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
        for fname in sorted(fnames):
            if fname.split('.')[-1] == 'bin':
                binary_file_full_filename = os.path.join(dirpath, fname)
                binary_filenames_list.append(binary_file_full_filename)
        if flag_recursive == False:
            break;
    return binary_filenames_list;


def get_all_filenames_from_folder(path, flag_recursive=False):
    filenames_list = []
    # Look recursively at all current and children path's in given path and get all binary files
    for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
        for fname in sorted(fnames):
            file_full_filename = os.path.join(dirpath, fname)
            filenames_list.append(file_full_filename)
        if flag_recursive == False:
            break;
    return filenames_list;




def get_image_filenames_from_lmdb(dataroot): #lmdb is apparently a file type which lands itself to be read by multiple workers... but there are many other options to implement!!!
    #Create an lmdb object/env with given dataroot, now i can read from the lmdb object the filenames inside that dataroot:
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')

    #if lmdb keys file exists use it, and if not Create it:
    if os.path.isfile(keys_cache_file):
        print('reading lmdb keys from cache: {}'.format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            print('creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))

    #Get paths from lmdb file:
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths



#Should probably depricate... need to think about it.
def get_image_filenames_according_to_filenames_source_type(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = get_image_filenames_from_lmdb(dataroot)
        elif data_type == 'images':
            paths = sorted(get_image_filenames_from_folder(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths



def read_image_from_lmdb_object(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii') #encode and then decode? is this really efficient?
    image_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    image = image_flat.reshape(H, W, C)
    # Make Sure That We Get the Image in the correct format [H,W,C]:
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2);
    return image


def imshow2(image):
    pylab.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));



def imshow_torch(image, flag_colorbar=True):
    # plt.cla()
    plt.clf()
    # plt.close()

    pylab.imshow(np.transpose(image.detach().cpu().numpy(),(1,2,0)).squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    if flag_colorbar:
        colorbar()  #TODO: fix the bug of having multiple colorbars when calling this multiple times


def plot_torch(input_signal):
    # plt.cla()
    plt.clf()
    # plt.close()
    plot(input_signal.detach().cpu().numpy().squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]


def read_image_cv2(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
    # image = cv2.imread(path, cv2.IMREAD_COLOR);
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED);
    if flag_convert_to_rgb == 1 and len(image.shape)==3:
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


def read_image_torch(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
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

    image = np.transpose(image,[2,0,1])
    image = torch.Tensor(image);
    image = image.unsqueeze(0)
    return image


def read_image_default_torch():
    return read_image_torch('F:\Imagenet/n01496331/n01496331_490.jpg')/255

def read_image_default():
    return read_image_cv2('F:\Imagenet/n01496331/n01496331_490.jpg')/255


def read_image_stack_default_torch():
    image1 = read_image_torch('F:\Imagenet/n01496331/n01496331_490.jpg')/255
    image2 = read_image_torch('F:\Imagenet/n01496331/n01496331_2720.jpg')/255
    image3 = read_image_torch('F:\Imagenet/n01496331/n01496331_3283.jpg')/255
    image1 = image1[:,:,0:400,0:400]
    image2 = image2[:,:,0:400,0:400]
    image3 = image3[:,:,0:400,0:400]
    images_total = torch.cat([image1,image2,image3],dim=0)
    return images_total

def read_image_stack_default():
    image1 = read_image_cv2('F:\Imagenet/n01496331/n01496331_490.jpg')/255
    image2 = read_image_cv2('F:\Imagenet/n01496331/n01496331_2720.jpg')/255
    image3 = read_image_cv2('F:\Imagenet/n01496331/n01496331_3283.jpg')/255
    image1 = image1[0:400,0:400,:]
    image2 = image2[0:400,0:400,:]
    image3 = image3[0:400,0:400,:]
    image1 = np.expand_dims(image1, 0)
    image2 = np.expand_dims(image2, 0)
    image3 = np.expand_dims(image3, 0)
    images_total = np.concatenate((image1,image2,image3), axis=0)
    return images_total


from skimage.color.colorlabel import label2rgb
def save_image_torch(folder_path=None,filename=None,torch_tensor=None,flag_convert_bgr2rgb=True, flag_scale=True):
    if flag_scale:
        scale = 255;
    else:
        scale = 1;

    if flag_convert_bgr2rgb:
        saved_array = cv2.cvtColor(torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale, cv2.COLOR_BGR2RGB)
    else:
        saved_array = torch_tensor.cpu().data.numpy().transpose([1, 2, 0]) * scale

    cv2.imwrite(os.path.join(folder_path, filename), saved_array)

    print(os.path.join(folder_path, filename))

def save_image_numpy(folder_path=None,filename=None,numpy_array=None,flag_convert_bgr2rgb=True, flag_scale=True):
    if flag_scale:
        scale = 255;
    else:
        scale = 1;
    if flag_convert_bgr2rgb:
        cv2.imwrite(os.path.join(folder_path,filename), cv2.cvtColor(numpy_array * scale, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(os.path.join(folder_path, filename), numpy_array * scale)


def read_images_from_folder_to_list(path, allowed_extentions=IMG_EXTENSIONS, flag_recursive=False):
    #read all images in path folder and return them in a LIST!
    #Note: maybe i should change the name to "read_images_from_folder_to_list" or something...
    image_filenames = get_image_filenames_from_folder(path,allowed_endings=allowed_extentions, flag_recursive=flag_recursive);
    images_list = [];
    for image_filename in image_filenames:
        if is_image_file(image_filename):
            images_list.append(read_image_cv2(image_filename))
    return images_list;


def read_images_from_folder_to_numpy(folder_path, allowed_extentions=IMG_EXTENSIONS, flag_recursive=False):
    #Note: images must be the same size!!!!!
    image_filenames = get_image_filenames_from_folder(folder_path,allowed_endings=allowed_extentions, flag_recursive=flag_recursive);
    images_mat = np.empty(shape=get_image_shape_from_path(folder_path));
    images_mat = np.expand_dims(images_mat,0)
    for image_filename in image_filenames:
        if is_image_file(image_filename,allowed_extentions):
            images_mat = np.concatenate((images_mat, np.expand_dims(read_image_cv2(image_filename),0)), axis=0)
    ### "Shave Off" unwanted first element... TODO: correct this programatically ###
    images_mat = images_mat[1:]
    return images_mat;


def read_number_of_images_from_folder_to_numpy_with_crop(folder_path, number_of_images, crop_size=200, number_of_channels=3,allowed_extentions=IMG_EXTENSIONS, flag_recursive=False):
    image_filenames = get_image_filenames_from_folder(folder_path,allowed_endings=allowed_extentions, flag_recursive=flag_recursive);
    images_mat = np.empty(shape=(1,crop_size,crop_size,number_of_channels));
    current_step = 0;
    for image_filename in image_filenames:
        if is_image_file(image_filename):
            current_image = read_image_cv2(image_filename);
            if current_image.shape[0]>crop_size and current_image.shape[1]>crop_size and current_image.shape[2]==number_of_channels:
                current_step += 1;
                # print(images_mat.shape)
                if current_step==1:
                    images_mat = np.expand_dims(current_image[0:crop_size, 0:crop_size, :],0)
                else:
                    images_mat = np.concatenate( (images_mat, np.expand_dims(current_image[0:crop_size,0:crop_size,:], 0)), axis=0)
                if current_step == number_of_images:
                    break;
    return images_mat, image_filenames;


def read_images_from_filenames_list_to_numpy_with_crop(image_filenames, crop_size=200, number_of_channels=3, max_number_of_images=10):
    images_mat = np.empty(shape=(1, crop_size, crop_size, number_of_channels));
    current_step = 0;
    for image_filename in image_filenames:
        current_image = read_image_cv2(image_filename);
        if current_image.shape[0] >= crop_size and current_image.shape[1] >= crop_size and current_image.shape[2] == number_of_channels:
            #TODO: deal with images being smaller than crop size (reflect? resize?);
            current_step += 1;
            # print(images_mat.shape)
            if current_step == 1:
                images_mat = np.expand_dims(current_image[0:crop_size, 0:crop_size, :], 0)
            else:
                images_mat = np.concatenate((images_mat, np.expand_dims(current_image[0:crop_size, 0:crop_size, :], 0)), axis=0)
            if current_step == max_number_of_images:
                break;
    return images_mat, image_filenames;


def read_images_from_folder_to_list(folder_path, allowed_extentions=IMG_EXTENSIONS, flag_recursive=False, flag_convert_to_rgb=True):
    image_filenames = get_image_filenames_from_folder(folder_path, allowed_endings=allowed_extentions, flag_recursive=flag_recursive);
    images_list = []
    for image_filename in image_filenames:
        if is_image_file(image_filename,allowed_extentions):
            images_list.append(read_image_cv2(image_filename,flag_convert_to_rgb))
    return images_list;


def read_single_image_from_folder(path):
    #assuming folder is all images!!!
    for dirpath, _, fnames in sorted(os.walk(path)):
        for filename in fnames:
            return read_image_cv2(os.path.join(dirpath,filename))


def read_number_of_images_from_folder(path,number_of_images,min_image_size,allowed_extentions=IMG_EXTENSIONS, flag_recursive=False):
    #TODO: add optional crop size variable
    #TODO: add possibility to return as numpy array
    count = 0;
    images_list = []
    for dirpath, _, fnames  in sorted(os.walk(path)):
        if count >= number_of_images and number_of_images!=-1:
            break;
        for fname in sorted(fnames):
            if count>=number_of_images  and number_of_images!=-1:
                break;
            elif is_image_file(fname,allowed_endings=allowed_extentions):
                img_path = os.path.join(dirpath, fname)
                current_image = read_image_cv2(img_path)
                if current_image.shape[0] > min_image_size and current_image.shape[1] > min_image_size:
                    images_list.append(current_image)
                    count += 1
        if flag_recursive == False:
            break;
    return images_list



def read_number_of_image_filenames_from_folder(path,number_of_images,min_image_size,flag_include_png=True, flag_recursive=False):
    count = 0;
    image_filenames_list = []
    if flag_include_png:
        IMG_EXTENSIONS1 = IMG_EXTENSIONS
    else:
        IMG_EXTENSIONS1 = IMG_EXTENSIONS_NO_PNG
    for dirpath, _, fnames  in sorted(os.walk(path)):
        if count >= number_of_images and number_of_images!=-1:
            break;
        for fname in sorted(fnames):
            if count>=number_of_images and number_of_images!=-1:
                break;
            elif is_image_file(fname,allowed_endings=IMG_EXTENSIONS1):
                img_path = os.path.join(dirpath, fname)
                current_image = read_image_cv2(img_path);
                if current_image.shape[0] > min_image_size and current_image.shape[1] > min_image_size:
                    image_filenames_list.append(img_path)
                    count += 1
        if flag_recursive == False:
            break;
    return image_filenames_list


def get_image_shape_from_path(path):
    if is_image_file(path):
        #path=image file
        image = read_image_cv2(path);
        shape = image.shape;
    else:
        #path=folder:
        image = read_single_image_from_folder(path);
        shape = image.shape;
    return shape;


def create_dataset_from_video(movie_full_filename):
    ### Original Movie: ###
    movie_full_filename = 'C:/Users\dkarl\PycharmProjects\dudykarl/anomaly_detection.avi'
    frames_super_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl/Movie1'
    if not os.path.exists(frames_super_folder):
        os.makedirs(frames_super_folder)
    video_stream = cv2.VideoCapture(movie_full_filename)

    ### Parameters: ###
    #(1). Images In Folders:
    number_of_images_per_folder = 10;
    number_of_crops = 5;
    crop_size = 200;
    min_width_start = 0;
    max_width_stop = 1000;
    min_height_start = 0;
    max_height_stop = 1000;
    #(2). Binary File:


    ### Loop Over Video Stream: ###
    current_step = 0;
    global_step = 0
    frame_step = 0
    max_number_of_frames = 1e9;
    max_number_of_sequences = 100*1000;
    max_number_of_images = 1e9
    while video_stream.isOpened() and current_step<max_number_of_sequences and global_step<max_number_of_images and frame_step<max_number_of_frames:
        tic()
        ### Read Current Frames: ###
        current_step += 1;
        crops_indices = []
        for mini_frame_number in arange(number_of_images_per_folder):
            frame_step += 1
            flag_frame_available, frame = video_stream.read()

            ### Loop Over Different Crops: ###
            if flag_frame_available:
                frame_height, frame_width, number_of_channels = frame.shape;
                for crop_number in arange(0,number_of_crops):
                    global_step += 1;
                    new_sub_folder = os.path.join(frames_super_folder, 'Sequence_' + str(current_step) + 'Crop_' + str(crop_number))
                    if not os.path.exists(new_sub_folder):
                        os.makedirs(new_sub_folder)

                    ### Get Crop Indices: ###
                    if len(crops_indices) <= crop_number:
                        rows_start = randint(min_height_start, frame_height - crop_size)
                        rows_stop = rows_start + crop_size;
                        cols_start = randint(min_width_start, frame_width - crop_size)
                        cols_stop = cols_start + crop_size;
                        crops_indices.append([rows_start,rows_stop,cols_start,cols_stop])

                    ### Get Crop And Save Images: ###
                    current_frame_crop = frame[crops_indices[crop_number][0]:crops_indices[crop_number][1], crops_indices[crop_number][2]:crops_indices[crop_number][3], :]
                    save_image_numpy(folder_path=new_sub_folder, filename=str(mini_frame_number)+'.png', numpy_array=current_frame_crop, flag_convert_bgr2rgb=False, flag_scale=False)

        toc('Sequence Creation: ')



def create_lmdb_dataset(images_folder, lmdb_save_path, allowed_endings=IMG_EXTENSIONS_NO_PNG):
    #?
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


    # # configurations
    # images_folder = 'C:/Users\dkarl\.fastai\data\DIV2K_test_HR'
    # lmdb_save_path = images_folder;
    image_list = get_image_filenames_from_folder(images_folder,allowed_endings)
    # images_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800/*'  # glob matching pattern
    # lmdb_save_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800.lmdb'  # must end with .lmdb

    # image_list = sorted(glob.glob(images_folder))
    image_dataset = []
    data_size = 0

    #Read Images into the image_dataset list object
    # (TODO: right there we're putting all the images into RAM... instead i need to unite both loops by: (1). reading a single image, (2). inserting it into the lmdb file!) !!!!!
    print('Read images...')
    pbar = ProgressBar(len(image_list))
    for i, v in enumerate(image_list):
        # pbar.update('Read {}'.format(v))
        img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        image_dataset.append(img)
        data_size += img.nbytes

    #Create lmdb object which will hold {key,image} pairs:
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    print('Finish reading {} images.\nWrite lmdb...'.format(len(image_list)))

    #Put {key,image} pairs into the lmdb object:
    pbar = ProgressBar(len(image_list))
    with env.begin(write=True) as txn:  # txn is a Transaction object
        for i, v in enumerate(image_list):
            # pbar.update('Write {}'.format(v))
            base_name = os.path.splitext(os.path.basename(v))[0]
            key = base_name.encode('ascii')
            data = image_dataset[i]
            if image_dataset[i].ndim == 2:
                H, W = image_dataset[i].shape
                C = 1
            else:
                H, W, C = image_dataset[i].shape
            meta_key = (base_name + '.meta').encode('ascii')
            meta = '{:d}, {:d}, {:d}'.format(H, W, C)
            # The encode is only essential in Python 3
            txn.put(key, data)
            txn.put(meta_key, meta.encode('ascii'))
    print('Finish writing lmdb.')

    # create keys cache
    keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
    env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        print('Create lmdb keys cache: {}'.format(keys_cache_file))
        keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, "wb"))
    print('Finish creating lmdb keys cache.')
#########################################################################################################################################









#########################################################################################################################################
####  Image Cropping: ####
##########################
def create_enlarged_patches_images_and_save_them(path):
    current_path = path_get_folder_name(path);

    #Configurations (TODO: turn to input variables into the function)
    h_start, h_len = 170, 64
    w_start, w_len = 232, 100
    enlarge_ratio = 3
    line_width = 2
    color = 'yellow'

    folder = os.path.join(current_path, './ori/*')
    save_patch_folder = os.path.join(current_path, './patch')
    save_rect_folder = os.path.join(current_path, './rect')

    color_tb = {}
    color_tb['yellow'] = (0, 255, 255)
    color_tb['green'] = (0, 255, 0)
    color_tb['red'] = (0, 0, 255)
    color_tb['magenta'] = (255, 0, 255)
    color_tb['matlab_blue'] = (189, 114, 0)
    color_tb['matlab_orange'] = (25, 83, 217)
    color_tb['matlab_yellow'] = (32, 177, 237)
    color_tb['matlab_purple'] = (142, 47, 126)
    color_tb['matlab_green'] = (48, 172, 119)
    color_tb['matlab_liblue'] = (238, 190, 77)
    color_tb['matlab_brown'] = (47, 20, 162)
    color = color_tb[color]
    image_list = glob.glob(folder)
    images = []

    # make temp folder
    if not os.path.exists(save_patch_folder):
        os.makedirs(save_patch_folder)
        print('mkdir [{}] ...'.format(save_patch_folder))
    if not os.path.exists(save_rect_folder):
        os.makedirs(save_rect_folder)
        print('mkdir [{}] ...'.format(save_rect_folder))

    for i, path in enumerate(image_list):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_UNCHANGED -> never heard of that
        base_name = os.path.splitext(os.path.basename(path))[0]
        print(i, base_name)
        # crop patch
        if img.ndim == 2:
            patch = img[h_start:h_start + h_len, w_start:w_start + w_len]
        elif img.ndim == 3:
            patch = img[h_start:h_start + h_len, w_start:w_start + w_len, :]
        else:
            raise ValueError('Wrong image dim [{:d}]'.format(img.ndim))

        # enlarge patch if necessary
        if enlarge_ratio > 1:
            H, W, _ = patch.shape
            patch = cv2.resize(patch, (W * enlarge_ratio, H * enlarge_ratio), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(save_patch_folder, base_name + '_patch.png'), patch)

        # draw rectangle
        img_rect = cv2.rectangle(img, (w_start, h_start), (w_start + w_len, h_start + h_len), color, line_width)
        cv2.imwrite(os.path.join(save_rect_folder, base_name + '_rect.png'), img_rect)




def extract_cropped_sub_images_from_folder_and_save_cropped_images(HR_images_path,
                                                                   HR_cropped_images_save_path,
                                                                   LR_cropped_images_save_path,
                                                                   cropped_image_size,
                                                                   LR_downsample_factor,
                                                                   number_of_threads,
                                                                   step_size_between_crop_windows,
                                                                   threshold_size,
                                                                   compression_level):
    """A multi-thread tool to crop sub imags."""
    # #Parameters:
    # HR_images_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800'
    # HR_cropped_images_save_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_cropped'
    # LR_cropped_images_save_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_LR_cropped'
    # number_of_threads = 20
    # cropped_image_size = 480
    # LR_downsample_factor = 4;
    # step_size_between_crop_windows = 240
    # threshold_size = 48
    # compression_level = 3  # 3 is the default value in cv2
    # # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer compression time. If read raw images during training, use 0 for faster IO speed.

    assert cropped_image_size % LR_downsample_factor == 0, 'cropped image size must be a whole multiple of LR_downsample_factor'

    ################################################################################
    #Save cropped images to HR_cropped_images_save_path:
    def crop_downsample_and_save_function(HR_image_path, HR_cropped_image_save_path, LR_cropped_image_save_path, cropped_image_size, step_size_between_crop_windows, threshold_size, compression_level):
        #Read Image:
        image_name = os.path.basename(HR_image_path)
        current_image = cv2.imread(HR_image_path, cv2.IMREAD_UNCHANGED) #cv2.IMREAD_UNCHANGED
        number_of_channels = len(current_image.shape)
        if number_of_channels == 2:
            image_height, image_width = current_image.shape
        elif number_of_channels == 3:
            image_height, image_width, number_of_channels = current_image.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(number_of_channels))


        #Get all possible Upper-Left starting points for cropping:
        height_starting_points_list = np.arange(0, image_height - cropped_image_size + 1, step_size_between_crop_windows)
        if image_height - (height_starting_points_list[-1] + cropped_image_size) > threshold_size:
            height_starting_points_list = np.append(height_starting_points_list, image_height - cropped_image_size)
        width_startpoing_points_list = np.arange(0, image_width - cropped_image_size + 1, step_size_between_crop_windows)
        if image_width - (width_startpoing_points_list[-1] + cropped_image_size) > threshold_size:
            width_startpoing_points_list = np.append(width_startpoing_points_list, image_width - cropped_image_size)


        #Loop over all possible Upper-Left starting points for cropping and actually crop the image and save the result:
        index = 0
        for x in height_starting_points_list:
            for y in width_startpoing_points_list:
                index += 1

                #Crop HR image:
                if number_of_channels == 2:
                    HR_cropped_image = current_image[x:x + cropped_image_size, y:y + cropped_image_size]
                else:
                    HR_cropped_image = current_image[x:x + cropped_image_size, y:y + cropped_image_size, :]
                    HR_cropped_image = np.ascontiguousarray(HR_cropped_image) #why is this good?

                #Create LR image from Cropped HR image:
                LR_cropped_image = cv2.resize(HR_cropped_image, (cropped_image_size % LR_downsample_factor, cropped_image_size % LR_downsample_factor), interpolation=cv2.INTER_CUBIC)

                #Save Cropped HR and LR images:
                cv2.imwrite(os.path.join(HR_cropped_image_save_path, image_name.replace('.png', '_s{:03d}.png'.format(index))),
                            HR_cropped_image,
                            [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
                cv2.imwrite(os.path.join(LR_cropped_image_save_path, image_name.replace('.png', '_s{:03d}.png'.format(index))),
                            LR_cropped_image,
                           [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        return 'Processing {:s} ...'.format(image_name)
    ################################################################################

    #If LR images save path doesn't exist then create it:
    if not os.path.exists(HR_cropped_images_save_path):
        os.makedirs(HR_cropped_images_save_path)
        print('mkdir [{:s}] ...'.format(HR_cropped_images_save_path))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(HR_cropped_images_save_path))
        sys.exit(1)

    #Go over images/folders and create a list of image filenames:
    image_filenames_list = []
    for root, _, file_list in sorted(os.walk(HR_images_path)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the HR_images_folder
        image_filenames_list.extend(path)

    #Create progress barUpdage Progress-Bar callback:
    pbar = ProgressBar(len(image_filenames_list))
    def update(arg):
        pbar.update(arg)

    #Create a pool of cpu threads (workers) and asynchronously apply the worker_function (in this case cropping):
    pool = Pool(number_of_threads)
    for image_filename_path in image_filenames_list:
        pool.apply_async(crop_downsample_and_save_function,
                         args=(image_filename_path, HR_cropped_images_save_path, cropped_image_size, step_size_between_crop_windows, threshold_size, compression_level),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')





def crop_image_modified(input_image, crop_factor_from_the_end): #crop_factor is a weird on... if crop_factor=3 then we CUT a third of the image
    #Crop from image image_size/crop_factor.
    #I think that the cropping is from the top left to the bottom right- check!

    # img_in: Numpy, HWC or HW
    input_image_numpy = np.copy(input_image)

    #If input image is a simple Matrix (1 channel):
    if input_image_numpy.ndim == 2:
        image_height, image_width = input_image_numpy.shape
        image_height_crop, image_width_crop = image_height % crop_factor_from_the_end, image_width % crop_factor_from_the_end
        input_image_numpy = input_image_numpy[:image_height - image_height_crop, :image_width - image_width_crop]
    #If input image is a multi channel image (>1 channels):
    elif input_image_numpy.ndim == 3:
        image_height, image_width, image_number_of_channels = input_image_numpy.shape
        image_height_crop, image_width_crop = image_height % crop_factor_from_the_end, image_width % crop_factor_from_the_end
        input_image_numpy = input_image_numpy[:image_height - image_height_crop, :image_width - image_width_crop, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(input_image_numpy.ndim))
    return input_image_numpy


def crop_tensor(img,cropX=None,cropY=None):
    y,x,c = img.shape
    if cropX==None:
        print('Error!!'); #TODO raise official error:
    if cropY==None:
        cropY = cropX;
    startx = x//2-(cropX//2)
    starty = y//2-(cropY//2)
    return img[starty:starty+cropY,startx:startx+cropX]

def crop_center_percent(img,cropX_percent=None,cropY_percent=None):

    y,x, *rest = img.shape
    if cropY_percent==None:
        cropY_percent = cropX_percent;
    cropX = int(floor(x*cropX_percent));
    cropY = int(floor(y*cropY_percent));
    startx = int(x//2-(cropX//2))
    starty = int(y//2-(cropY//2))
    if img.ndim==3: return img[starty:starty+cropY,startx:startx+cropX,:]
    if img.ndim==2: return img[starty:starty+cropY,startx:startx+cropX]


class Crop_Center_Layer(nn.Module):
    def __init__(self, crop_H, crop_W, *args):
        super(Crop_Center_Layer, self).__init__()
        self.crop_x = crop_W
        self.crop_y = crop_H
    def forward(self, x):
        B,C,H,W = x.shape
        startx = W // 2 - (self.crop_x // 2)
        starty = H // 2 - (self.crop_y // 2)
        return x[:,:,starty:starty + self.crop_y, startx:startx + self.crop_x]


class Crop_Center_Layer_3Values(nn.Module):
    def __init__(self, crop_H, crop_W, *args):
        super(Crop_Center_Layer_3Values, self).__init__()
        self.crop_x = crop_W
        self.crop_y = crop_H
    def forward(self, x):
        C,H,W = x.shape
        startx = W // 2 - (self.crop_x // 2)
        starty = H // 2 - (self.crop_y // 2)
        return x[:,starty:starty + self.crop_y, startx:startx + self.crop_x]


# input_tensor = torch.Tensor(randn(1,3,100,100))
# layer = Crop_Center_Layer(50,60);
# layer(input_tensor).shape
#########################################################################################################################################







#########################################################################################################################################
####  Image Comparison Critrions: ####
######################################
def calculate_PSNR(img1, img2):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

#SSIM Criterion!!!
def ssim(img1, img2, multichannel=False):
    assert img1.dtype == img2.dtype == np.uint8, 'np.uint8 is supposed.'
    #compare_ssim is an skimage.measure function!!
    return compare_ssim(img1, img2, multichannel=multichannel)
#########################################################################################################################################








#########################################################################################################################################
####  Multiple Image Visualization Functions: ####
##################################################

class VizList(list):
    """Extended List class which can be binded to an matplotlib's pyplot axis
    and, when being appended a value, automatically update the figure.

    Originally designed to be used in a jupyter notebook with activated
    %matplotlib notebook mode.

    Example of usage:

    %matplotlib notebook
    from matplotlib import pyplot as plt
    f, (loss_axis, validation_axis) = plt.subplots(2, 1)
    loss_axis.set_title('Training loss')
    validation_axis.set_title('MIoU on validation dataset')
    plt.tight_layout()

    loss_list = VizList()
    validation_accuracy_res = VizList()
    train_accuracy_res = VizList()
    loss_axis.plot([], [])
    validation_axis.plot([], [], 'b',
                         [], [], 'r')
    loss_list.bind_to_axis(loss_axis)
    validation_accuracy_res.bind_to_axis(validation_axis, 0)
    train_accuracy_res.bind_to_axis(validation_axis, 1)

    Now everytime the list are updated, the figure are updated
    automatically:

    # Run multiple times
    loss_list.append(1)
    loss_list.append(2)


    Attributes
    ----------
    axis : pyplot axis object
        Axis object that is being binded with a list
    axis_index : int
        Index of the plot in the axis object to bind to

    """

    def __init__(self, *args):
        super(VizList, self).__init__(*args)
        self.object_count = 0
        self.object_count_history = []
        self.axis = None
        self.axis_index = None

    def append(self, object):
        self.object_count += 1
        self.object_count_history.append(self.object_count)
        super(VizList, self).append(object)
        self.update_axis()

    def bind_to_axis(self, axis, axis_index=0):
        self.axis = axis
        self.axis_index = axis_index

    def update_axis(self):
        self.axis.lines[self.axis_index].set_xdata(self.object_count_history)
        self.axis.lines[self.axis_index].set_ydata(self)
        self.axis.relim()
        self.axis.autoscale_view()
        self.axis.figure.canvas.draw()




def plot_multiple_images(X_images, filter_indices=None, flag_common_colorbar=1, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                    super_title=None, titles_string_list=None):
    #This assumes that the images within the image batch are plottable (i.e. with 1 channel as in grayscale or with 3 channels as in rgb)

    if type(X_images) == torch.Tensor:
        # transform to numpy:
        X_images = X_images.detach().numpy();
        if X_images.ndim == 4: #[N,C,H,W]:
            X_images = np.transpose(X_images,[0,2,3,1]); #--> [N,H,W,C]
        if X_images.ndim == 3: #[C,H,W] (only 1 image received)
            X_images = np.transpose(X_images,[1,2,0]); #-->[H,W,C]


    # Parameters:
    if filter_indices == None:
        filter_indices = arange(0, len(X_images));
    number_of_images_to_show = len(filter_indices);
    number_of_images_to_show_cols = int(ceil(sqrt(number_of_images_to_show)));
    number_of_images_to_show_rows = int(ceil(number_of_images_to_show / number_of_images_to_show_cols))

    if flag_common_colorbar == 0:
        # Simple SubPlots:
        fig = plt.figure()
        plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
        if super_title is not None:
            plt.suptitle(super_title);

        plot_counter = 0;
        for filter_index in filter_indices:
            a = fig.add_subplot(number_of_images_to_show_rows, number_of_images_to_show_cols,
                                plot_counter + 1)
            current_image = X_images[filter_index];
            current_image = crop_center_percent(current_image,
                                                crop_percent)  # the "non-valid" convolution regions around the image overshadow the dynamic range in the image display so it can be a good idea to cut them out
            plt.imshow(current_image)
            if flag_colorbar == 1:
                plt.colorbar();
            if titles_string_list is not None:
                plt.title(titles_string_list[plot_counter]);
            plot_counter += 1

    elif flag_common_colorbar == 1:
        # Advanced SubPlots:
        fig, axes_array = plt.subplots(number_of_images_to_show_rows, number_of_images_to_show_cols)
        axes_array = np.atleast_2d(axes_array)
        fig.suptitle(super_title)
        print(shape(axes_array))
        plot_counter = 0;
        image_plots_list = []
        cmap = "cool"
        for i in range(number_of_images_to_show_rows):
            for j in range(number_of_images_to_show_cols):
                if plot_counter >= number_of_images_to_show:
                    break;
                # Generate data with a range that varies from one plot to the next.
                filter_index = filter_indices[plot_counter];
                current_image = X_images[filter_index];
                current_image = crop_center_percent(current_image, crop_percent)
                image_plots_list.append(axes_array[i, j].imshow(current_image, cmap=cmap))
                axes_array[i, j].label_outer()
                # if titles_string_list is not None:
                #     image_plots_list[plot_counter].title(titles_string_list[plot_counter]);
                plot_counter += 1;

        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in image_plots_list)
        vmax = max(image.get_array().max() for image in image_plots_list)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for image in image_plots_list:
            image.set_norm(norm)

        fig.colorbar(image_plots_list[0], ax=axes_array, orientation='horizontal', fraction=.1)





def plot_multiple_batches_of_images(X_images, filter_indices=None, flag_common_colorbar=1, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                                    super_title=None, titles_string_list=None):
    #Assumes X_images is in the form of: [N,H,W] or [N,H,W,C]
    for i in arange(len(X_images)):
        plot_multiple_images(X_images[i], filter_indices, flag_common_colorbar, crop_percent, delta, flag_colorbar, super_title, titles_string_list)



def plot_multiple_image_channels(X_images, filter_indices=None, flag_common_colorbar=1, crop_percent=0.8, delta=0.1, flag_colorbar=1,
                                    super_title=None, titles_string_list=None):
    #Assumes X_images is in the form of: [N,H,W] or [N,H,W,C]
    if X_images.ndim() == 4:
        number_of_images = X_images.shape[0];
    else:
        number_of_images = 1;

    #Plot each image's channels in different figures:
    for i in arange(number_of_images):
        plot_multiple_images(X_images[i], filter_indices, flag_common_colorbar, crop_percent, delta, flag_colorbar, super_title, titles_string_list)


##############
def get_image_differences(imageA, imageB, flag_plot=1, crop_percent=0.8, flag_normalize_by_number_of_channels=1):
    #Remember, input image ranges can be [0,1] and [0,255] depending on whether float or uint8
    grayA = convert_to_grayscale(imageA,flag_scale_to_normal_range=0);
    grayB = convert_to_grayscale(imageB,flag_scale_to_normal_range=0);
    if flag_normalize_by_number_of_channels == 1:
        grayA = grayA / imageA.shape[2];
        grayB = grayB / imageB.shape[2];

    imageA = np.atleast_3d(imageA);
    flag_input_image_plotable = (imageA.shape[2] == 3 or imageA.shape[2] == 1)
    if flag_input_image_plotable:  # BW and RGB style images can be visualized... otherwise put the rectangle on the synthetic grayscale summed image
        imageA_with_rectangle = imageA.copy();  # Should i use .copy() or not??
        imageB_with_rectangle = imageB.copy();
        raw_diff = imageA - imageB;
    else:
        imageA_with_rectangle = grayA.copy();
        imageB_with_rectangle = grayB.copy();
        raw_diff = grayA - grayB;

    #Use compare_ssim and get overall score and pixel-wise diff which is NOT the RAW diff:
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff_scaled = (diff * 255).astype("uint8")
    # print(diff.shape)
    thresh = cv2.threshold(diff_scaled, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = cnts[1]

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA_with_rectangle, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB_with_rectangle, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if flag_plot == 1:
        if flag_input_image_plotable==False:
            plot_multiple_images([raw_diff, grayA, grayB, imageA_with_rectangle, imageB_with_rectangle, diff, thresh], flag_common_colorbar=0,
                            crop_percent=crop_percent, titles_string_list=['raw_diff', 'imageA_gray', 'imageB_gray', 'imageA_gray_rect', 'imageB_gray_rect', 'SSIM_map', 'threshold'])
        if flag_input_image_plotable==True:
            plot_multiple_images([raw_diff, imageA, imageB, imageA_with_rectangle, imageB_with_rectangle, diff, thresh], flag_common_colorbar=0,
                            crop_percent=crop_percent, titles_string_list=['raw_diff', 'imageA', 'imageB', 'imageA_rect', 'imageB_rect', 'SSIM_map', 'threshold'])

    return raw_diff, diff, thresh, imageA_with_rectangle, imageB_with_rectangle
################



def spot_check_images_from_batch_matrix(X_images, number_of_images_to_show, flag_rgb_or_gray=1, flag_colorbar=0, super_title=None, titles_string_list=None):
    number_of_images_to_show_each_axis = int(ceil(sqrt(number_of_images_to_show)));
    image_indices_to_show = np.random.choice(arange(0, shape(X_images)[0]), number_of_images_to_show, replace=False);

    fig = plt.figure()
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    if super_title is not None:
        plt.suptitle(super_title);
    for x_counter in arange(0, number_of_images_to_show_each_axis):
        for y_counter in arange(0, number_of_images_to_show_each_axis):
            current_index = x_counter * number_of_images_to_show_each_axis + y_counter
            if current_index >= number_of_images_to_show:
                return;
            a = fig.add_subplot(number_of_images_to_show_each_axis, number_of_images_to_show_each_axis,
                                current_index + 1)
            if flag_rgb_or_gray == 1:
                plt.imshow(X_images[current_index,:])
            else:
                plt.imshow(skimage.color.rgb2gray(X_images[current_index, :]));
            if flag_colorbar == 1:
                plt.colorbar();
            if titles_string_list is not None:
                plt.title(titles_string_list[current_index]);
#########################################################################################################################################





#########################################################################################################################################
####  Multiple Graphs Visualization Functions: ####
##################################################
def plot_subplots(y_plots, flag_plot_or_histogram=1, number_of_samples_to_show=None, legend_labels=None,
                  super_title=None, x_label=None,y_label=None,title_strings=None):

    if length(shape(y_plots)) == 2:
        y_plots = np.expand_dims(y_plots,-1);
    number_of_plots_each_figure_axes = shape(y_plots)[2];
    number_of_images_to_show = shape(y_predictions)[1];
    number_of_images_to_show_x = int(ceil(sqrt(number_of_images_to_show)));
    number_of_images_to_show_y = int(ceil(number_of_images_to_show / number_of_images_to_show_x))
    if number_of_samples_to_show == None:
        number_of_samples_to_show = shape(y_predictions)[0];

    fig, axes_array = plt.subplots(number_of_images_to_show_x, number_of_images_to_show_y);
    delta = 0.02;
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    plt.figlegend(legend_labels, loc='lower center', ncol=5, labelspacing=0.)

    if super_title is not None:
        plt.suptitle(super_title);
    for x_counter in arange(0, number_of_images_to_show_x):
        for y_counter in arange(0, number_of_images_to_show_y):
            current_subplot_index = x_counter * number_of_images_to_show_y + y_counter
            if current_subplot_index >= number_of_images_to_show:
                axes_array[x_counter, y_counter].set_visible(False)
            else:
                current_axes = axes_array[x_counter, y_counter]

                #Plot simple plot or histogram:
                if flag_plot_or_histogram == 1:
                    for plot_counter in arange(0, number_of_plots_each_figure_axes):
                        current_axes.plot(y_plots[0:number_of_samples_to_show, current_subplot_index,plot_counter])
                elif flag_plot_or_histogram == 2:
                    current_axes.hist(y_plots[0:number_of_samples_to_show,current_subplot_index,:], alpha=0.5)


                if current_subplot_index == 0:
                    current_axes.legend(loc=2, bbox_to_anchor=[0, 1], labels=legend_labels, ncol=2, shadow=True, fancybox=True)
                if x_label is not None:
                    current_axes.set_xlabel(x_label);
                if y_label is not None:
                    current_axes.set_ylabel(y_label);
                if title_strings is not None:
                    current_axes.set_title(title_strings[current_subplot_index])
    return 1;
############################################################################################################################################################






############################################################################################################################################################
##### Feature Distribution Visualization Functions: ######
def plot_bar_graph(x_positions=None, y_heights=None, x_labels=None, y_label=None, title_str=None):
    number_of_bars = length(x_positions);
    color_array = get_spaced_colors(number_of_bars)
    fig, ax = plt.subplots()
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    #plt.show(block=False)
    bar_array = bar(x_positions, y_heights, width=0.3, align='center')
    for counter in arange(0, length(bar_array)):
        bar_array[counter].set_facecolor(color_array[counter])
    ax.set_xticks(np.arange(0, number_of_bars))

    if 'x_labels' in locals().keys():
        ax.set_xticklabels(x_labels);
    if 'y_label' in locals().keys():
        ax.set_ylabel(y_label);
    if 'title_str' in locals().keys():
        ax.set_title(title_str)


def plot_pair_plot(df,super_title):
    pair_plot = sns.pairplot(df);
    pair_plot_fig = pair_plot.fig;
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    if 'super_title' in locals().keys():
        pair_plot_fig.suptitle(super_title, fontsize=14)



def plot_box_plot(df,x_labels,y_label,title_str,**kwargs):
    fig, ax = plt.subplots()
    plt.show(block=False)
    plt.boxplot(np.asarray(df),**kwargs)
    if 'x_labels' in locals().keys():
        ax.set_xticklabels(x_labels);
    if 'y_label' in locals().keys():
        ax.set_ylabel(y_label);
    if 'title_str' in locals().keys():
        ax.set_title(title_str)


# ###### Example Of Use: #######
# different_classes_vec = np.unique(y_class_labels_train).astype(int);
# number_of_classes = length(different_classes_vec);
# [unique_class_values_sorted_vec, original_array_with_classes_instead_of_values, class_occurence_vec] = \
#                       np.unique(y_class_labels_train,return_index=False, return_inverse=True, return_counts=True, axis=None);
# plot_bar_graph(unique_class_values_sorted_vec,class_occurence_vec,classification_output_name_list,'Class Occurence','Class Imbalance Bar Graph');
#
# plot_pair_plot(df_y_geometric_labels_train_only_ellipse,'Pair Plots of Geometric Labels (Ellipse Only)');
# plot_box_plot(df_y_geometric_labels_train_only_ellipse,y_total_labels_line_format_train_split[1:],'Distribution','Geometric Labels Box Plots, Train Data - Before PreProcessing',
#                   notch=True, sym=None,patch_artist=True, meanline=True, showmeans=True, showcaps=True, showbox=True, manage_xticks=True)
############################################################################################################################################################



############################################################################################################################################################
##### Classification Model Evaluation Functions: ######
def plot_multi_class_ROC(y_class_labels_test, y_prediction_labels):
    #Take Care of Input:
    if np.asarray(shape(y_class_labels_test)).prod() == length(y_class_labels_test):
        #If 1D verctor this means 2 classes -> turn to categorical
        y_class_labels_test = keras.utils.to_categorical(y_class_labels_test,2);
        y_prediction_labels = keras.utils.to_categorical(y_prediction_labels,2);

    # Compute ROC and AUC for each class:
    false_positive_rates = dict();
    true_positive_rates = dict();
    roc_auc = dict();
    for i in arange(0, number_of_classes):
        false_positive_rates[i], true_positive_rates[i], _ = roc_curve(y_class_labels_test[:, i],
                                                                       y_prediction_labels[:, i]);
        roc_auc[i] = auc(false_positive_rates[i], true_positive_rates[i]);
    # Compute micro-average ROC curve & AUC:
    false_positive_rates['micro'], true_positive_rates['micro'], _ = roc_curve(y_class_labels_test.ravel(),
                                                                               y_prediction_labels.ravel());
    roc_auc['micro'] = auc(false_positive_rates['micro'], true_positive_rates['micro']);
    # Compute macro-average ROC curve & AUC:
    #   (*). Aggregate all false positive rates:
    all_false_positive_rates = np.unique(
        np.concatenate([false_positive_rates[i] for i in arange(0, number_of_classes)]))
    #   (*). Interpolate all ROC curves at these points:
    mean_true_positive_rates = np.zeros_like(all_false_positive_rates);
    for i in arange(0, number_of_classes):
        mean_true_positive_rates += interp(all_false_positive_rates, false_positive_rates[i], true_positive_rates[i]);
    #   (*). Average it and compute AUC:
    mean_true_positive_rates /= number_of_classes;
    #   (*). Assign macro {key,value} to false_positive_rates dictionary:
    false_positive_rates['macro'] = all_false_positive_rates;
    true_positive_rates['macro'] = mean_true_positive_rates;
    roc_auc['macro'] = auc(false_positive_rates['macro'], true_positive_rates['macro']);
    # Plot all ROC curves
    lw = 2
    plt.figure(1)
    plt.plot(false_positive_rates["micro"], true_positive_rates["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(false_positive_rates["macro"], true_positive_rates["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(number_of_classes), colors):
        plt.plot(false_positive_rates[i], true_positive_rates[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC to multi-class')
    plt.legend(loc="lower right")
    delta = 0.1
    plt.tight_layout(rect=[delta, delta, 1 - delta, 1 - delta])
    plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    # import matplotlib.pyplot as plt
    # import numpy as np
    # import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2: (len(lines) - 3)]:
        # print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')


# #########      Example Of Use:     ############
# #Get Prediction Results
# tic_toc.tic()
# K.set_learning_phase(0);
# y_prediction_class_labels = final_classification_model.predict(X_test);
# K.set_learning_phase(1);
# y_prediction_class_labels_uncategorical = argmax(y_prediction_class_labels,axis=1);
# y_class_labels_test_uncategorical = argmax(y_class_labels_test,axis=1);
# tic_toc.toc(True);
# #Visualize Classification Results:
# #(1). Confusion Matrix:
# plt.figure()
# confusion_matrix_mat = confusion_matrix(y_class_labels_test_uncategorical,y_prediction_class_labels_uncategorical);
# plot_confusion_matrix(confusion_matrix_mat,classification_output_name_list,title='Confusion matrix',cmap=None,normalize=True);
# #(2). Classification Report:
# print(classification_report(y_class_labels_test_uncategorical,y_prediction_class_labels_uncategorical,target_names=['Line','Ellipse']))
# plt.figure()
# plot_classification_report(classification_report(y_class_labels_test_uncategorical,y_prediction_class_labels_uncategorical,target_names=['Line','Ellipse']))
# #(3). ROC:
# plot_multi_class_ROC(y_class_labels_test,y_prediction_class_labels)
# #(4). Spot Check Images Misclassified:
# for class_counter in arange(0,number_of_classes):
#     #Get Indices of instances Which are of the current class but were MISCLASSIFIED as another class:
#     indices_current_class_misclassified = indices_from_logical(
#                                                 (y_prediction_class_labels_uncategorical!=class_counter) &
#                                                 (y_class_labels_test_uncategorical==class_counter))
#     #Show Images Misclassified:
#     number_of_images_to_show = 10;
#     number_of_images_to_show = np.minimum(number_of_images_to_show,length(indices_current_class_misclassified));
#     if number_of_images_to_show > 0:
#         image_indices_to_show = np.random.choice(indices_current_class_misclassified, number_of_images_to_show, replace=False);
#         current_super_title = 'Images of Class ' + classification_output_name_list[class_counter] + ' Misclassified as another class'
#         spot_check_images_from_batch_matrix((X_test[image_indices_to_show,:]*255+155).astype(int16),number_of_images_to_show,1,0,current_super_title)
#########################################################################################################################################









#########################################################################################################################################
####  Logger: ####
##################
# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self, log_files_directory):
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_files_directory, 'print_log.txt'), 'a')

    #"Write" Message to terminal/console and log file:
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Logger_to_Console_and_TXT(object):
    def __init__(self, OPT):
        #Parameters:
        self.model_name = OPT.model_name #experiment name
        self.log_files_directory = OPT.log_files_directory #log directory where all the log .txt files will be at
        self.train_log_filename = OPT.train_log_txt_filename; #.txt file
        self.validation_log_filename = OPT.validation_log_txt_filename; #.txt file


        #Write Header At Start of Training Log File:
        self.train_log_path = os.path.join(self.log_files_directory, self.train_log_filename)
        with open(self.train_log_path, 'a') as log_file:
            log_file.write('=============== Time: ' + get_timestamp() + ' =============\n')
            log_file.write('================ Training Losses ================\n')

        #Write Header At Start of Validation Log File:
        self.validation_log_path = os.path.join(self.log_files_directory, self.validation_log_filename)
        with open(self.validation_log_path, 'a') as log_file:
            log_file.write('================ Time: ' + get_timestamp() + ' ===============\n')
            log_file.write('================ Validation Results ================\n')



    def print_results_to_console_and_logfile(self, mode, realtime_logger):
        #Get Last Logger Basic Training Loop Values & Start Forming Message to be Printed in Console:
        epoch = realtime_logger.pop('epoch')
        iters = realtime_logger.pop('iters')
        time = realtime_logger.pop('time')
        model = realtime_logger.pop('model')
        if 'lr' in realtime_logger:
            lr = realtime_logger.pop('lr')
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}, lr:{:.1e}> '.format(epoch, iters, time, lr)
        else:
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}> '.format(epoch, iters, time)


        #Add General {label,value} Pairs That The Logger Keeps Track Of & Add To Message:
        for label, value in realtime_logger.items():
            if type(value) == torch.Tensor:
                current_value = value.item()
            elif type(value) == list:
                current_value = value[0];
            else:
                current_value = value
            if mode == 'train':
                message += ',  {:s}: {:.2e}  ,'.format(label, current_value)
            elif mode == 'val':
                message += ',  {:s}: {:.4e}  ,'.format(label, current_value)

        #Print in Console:
        print(message)

        #Write Current Message in Appropriate .TXT Log File
        if mode == 'train':
            with open(self.train_log_path, 'a') as log_file:
                log_file.write(message + '\n')
        elif mode == 'validation':
            with open(self.validation_log_path, 'a') as log_file:
                log_file.write(message + '\n')






class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widening the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '█' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
#########################################################################################################################################





#########################################################################################################################################
####  Image ColorSpace Transforms: ####
#######################################
def convert_to_grayscale(input_image, flag_average=1, flag_scale_to_normal_range=0):
    #Actually... we're not restricted to RGB....
    """
        Converts 3d image to grayscale
    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)
        flag_format_numpy_or_pytorch = 1 for numpy and 2 for pytorch
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    if input_image.ndim==2:
        return input_image;

    if input_image.ndim==3:
        axis_to_average_over = 2; #[H,W,C]
    elif input_image.ndim==4:
        axis_to_average_over = 3; #[N,H,W,C]

    if flag_average == 0:
        grayscale_image = np.sum(np.abs(input_image), axis=axis_to_average_over);
    elif flag_average == 1:
        grayscale_image = np.average(np.abs(input_image),axis=axis_to_average_over);


    #Normalize to between [0,1]:
    if flag_scale_to_normal_range == 1:
        im_max = np.percentile(grayscale_image, 99)
        im_min = np.min(grayscale_image)
        grayscale_image = (np.clip((grayscale_image - im_min) / (im_max - im_min), 0, 1))

    #Output a 3d image of shape (1,H,W) instead of the above created (H,W):
    return grayscale_image



def convert_image_list_color_space(image_list, number_of_input_channels, target_type):
    # conversion among BGR, gray and y
    if number_of_input_channels == 3 and target_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in image_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif number_of_input_channels == 3 and target_type == 'y':  # BGR to y?????
        y_list = [bgr2ycbcr(img, only_y=True) for img in image_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif number_of_input_channels == 1 and target_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in image_list]
    else:
        return image_list


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
#########################################################################################################################################








#########################################################################################################################################
####  Image Resizing: ####
##########################
# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()
#########################################################################################################################################



















#########################################################################################################################################
# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

### Running Scores and Averages: ###
class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#########################################################################################################################################









### Colors: ###
def get_color_formula(color_formula_triplet):
    formula_triplet = []
    if 0 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 1 in color_formula_triplet:
        formula_triplet.append(lambda x: 0.5)
    if 2 in color_formula_triplet:
        formula_triplet.append(lambda x: 1)
    if 3 in color_formula_triplet:
        formula_triplet.append(lambda x: x)
    if 4 in color_formula_triplet:
        formula_triplet.append(lambda x: x**2)
    if 5 in color_formula_triplet:
        formula_triplet.append(lambda x: x**3)
    if 6 in color_formula_triplet:
        formula_triplet.append(lambda x: x**4)
    if 7 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.5)
    if 8 in color_formula_triplet:
        formula_triplet.append(lambda x: x**0.25)
    if 9 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(pi/2*x))
    if 10 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(pi/2*x))
    if 11 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.abs(x-0.5))
    if 12 in color_formula_triplet:
        formula_triplet.append(lambda x: (2*x-1)**2)
    if 13 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(pi*x))
    if 14 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(pi*x)))
    if 15 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.sin(2*pi*x))
    if 16 in color_formula_triplet:
        formula_triplet.append(lambda x: torch.cos(2*pi*x))
    if 17 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(2*pi*x)))
    if 18 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(2*pi*x)))
    if 19 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.sin(4*pi*x)))
    if 20 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(torch.cos(4*pi*x)))
    if 21 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x)
    if 22 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-1)
    if 23 in color_formula_triplet:
        formula_triplet.append(lambda x: 3*x-2)
    if 24 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-1))
    if 25 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(3*x-2))
    if 26 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-1)/2)
    if 27 in color_formula_triplet:
        formula_triplet.append(lambda x: (3*x-2)/2)
    if 28 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-1)/2))
    if 29 in color_formula_triplet:
        formula_triplet.append(lambda x: abs((3*x-2)/2))
    if 30 in color_formula_triplet:
        formula_triplet.append(lambda x: x/0.32-0.78125)
    if 31 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.84)
    if 32 in color_formula_triplet:
        formula_triplet.append(lambda x: 0)
    if 33 in color_formula_triplet:
        formula_triplet.append(lambda x: abs(2*x-0.5))
    if 34 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x)
    if 35 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-0.5)
    if 36 in color_formula_triplet:
        formula_triplet.append(lambda x: 2*x-1)
    return formula_triplet

def gray2color(input_array, type_id=0):
    if type_id == 0:
        formula_id_triplet = [7,5,15]
    elif type_id == 1:
        formula_id_triplet = [3,11,6]
    elif type_id == 2:
        formula_id_triplet = [23,28,3]
    elif type_id == 3:
        formula_id_triplet = [21,22,23]
    elif type_id == 4:
        formula_id_triplet = [30,31,32]
    elif type_id == 5:
        formula_id_triplet = [31,13,10]
    elif type_id == 6:
        formula_id_triplet = [34,35,36]
    formula_triplet = get_color_formula(formula_id_triplet);
    R = formula_triplet[0](input_array)
    G = formula_triplet[1](input_array)
    B = formula_triplet[2](input_array)
    R = R.clamp(0,1)
    G = G.clamp(0,1)
    B = B.clamp(0,1)
    color_array = torch.cat([R,G,B], dim=1)
    return color_array

# ### Use Example: ###
# input_tensor = torch.Tensor(randn(1,1,100,100)).abs().clamp(0,1)
# color_tensor = gray2color(input_tensor,0)
# figure(1)
# imshow_torch(input_tensor[0].repeat((3,1,1)),0)
# figure(2)
# imshow_torch(color_tensor[0],0)




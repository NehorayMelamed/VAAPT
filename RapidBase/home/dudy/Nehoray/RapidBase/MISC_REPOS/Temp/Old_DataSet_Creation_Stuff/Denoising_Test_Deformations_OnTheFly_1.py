
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





#################################################################################################################################################################################################################################################################################
def to_list_of_certain_size(input_number, number_of_elements):
    if type(input_number)==tuple or type(input_number)==list:
        if len(input_number)==1:
            return input_number*number_of_elements;
        else:
            return input_number;
    else:
        return [input_number]*number_of_elements


def make_list_of_certain_size(n_layers, *args):
    args_list = []
    for current_arg in args:
        if type(current_arg) != list:
            args_list.append([current_arg] * n_layers)
        else:
            assert len(current_arg) == n_layers, str(current_arg) + ' must have the same length as n_layers'
            args_list.append(current_arg)
    return args_list


def decimal_notation(input_number, number_of_points_after_decimal=1):
    number_string = int(input_number*(10**(number_of_points_after_decimal)))/(10**(number_of_points_after_decimal))
    return str(number_string)



########################################################################################################################
### Turbulence Elastic Deformation: ###
def turbulence_deformation_single_image(I, Cn2=5e-13, flag_clip=False):
    ### TODO: why the .copy() ??? ###
    # I = I.copy()

    # I = read_image_default()
    # I = crop_tensor(I,150,150)
    # imshow(I)

    ### Parameters: ###
    h = 100
    # Cn2 = 2e-13
    # Cn2 = 7e-17
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(I.shape[0] / PatchSize))
    PatchNumCol = int(np.round(I.shape[1] / PatchSize))
    shape = I.shape
    [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    if I.dtype == 'uint8':
        mv = 255
    else:
        mv = 1

    ### Get Random Motion Field: ###
    ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)

    ### Resize (small) Random Motion Field To Image Size: ###
    ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    ### Add Rescaled Flow Field To Meshgrid: ###
    X = X0 + ShiftMatX
    Y = Y0 + ShiftMatY

    ### Resample According To Motion Field: ###
    I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)

    ### Clip Result: ###
    if flag_clip:
        I = np.minimum(I, mv)
        I = np.maximum(I, 0)

    # imshow(I)

    return I


# def turbulence_deformation_single_image(I, flag_clip=False):
#     ### TODO: why the .copy() ??? ###
#     # I = I.copy()
#
#     ### Parameters: ###
#     h = 100
#     Cn2 = 7e-14
#     # Cn2 = 7e-17
#     wvl = 5.3e-7
#     IFOV = 4e-7
#     R = 1000
#     VarTiltX = 3.34e-6
#     VarTiltY = 3.21e-6
#     k = 2 * np.pi / wvl
#     r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
#     PixSize = IFOV * R
#     PatchSize = 2 * r0 / PixSize
#
#     ### Get Current Image Shape And Appropriate Meshgrid: ###
#     PatchNumRow = int(np.round(I.shape[0] / PatchSize))
#     PatchNumCol = int(np.round(I.shape[1] / PatchSize))
#     shape = I.shape
#     [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#     if I.dtype == 'uint8':
#         mv = 255
#     else:
#         mv = 1
#
#     ### Get Random Motion Field: ###
#     ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
#     ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)
#
#     ### Resize (small) Random Motion Field To Image Size: ###
#     ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
#     ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
#
#     ### Add Rescaled Flow Field To Meshgrid: ###
#     X = X0 + ShiftMatX
#     Y = Y0 + ShiftMatY
#
#     ### Resample According To Motion Field: ###
#     I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)
#
#     # # TODO: pytorch implementation of resampling - probably more effective:
#     # torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='reflection')
#
#     ### Clip Result: ###
#     if flag_clip:
#         I = np.minimum(I, mv)
#         I = np.maximum(I, 0)
#
#     return I




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
##################################################################################################################################


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


def Generator_and_Discriminator_to_GPU(Generator_Network, Discriminator_Network, netF):
    global Generator_device, netF_device, discriminator_device
    Generator_Network = Generator_Network.to(Generator_device);
    if netF:
        netF = netF.to(netF_device);
    Discriminator_Network = Discriminator_Network.to(discriminator_device);
    return Generator_Network, Discriminator_Network


def Generator_to_GPU(Generator_Network,netF=None,device=None):
    global Generator_device, netF_device, discriminator_device
    if device:
        current_device = device
    else:
        current_device = Generator_device

    Generator_Network = Generator_Network.to(Generator_device);
    Generator_Network.hidden_states_to_device(Generator_device);
    if netF:
        netF = netF.to(netF_device);
    return Generator_Network;

def Discriminator_to_GPU(Discriminator_Network):
    global Generator_device, netF_device, discriminator_device
    Discriminator_Network = Discriminator_Network.to(discriminator_device);
    return Discriminator_Network;

def Generator_Loss_Functions_to_GPU():
    global DirectPixels_Loss_Function, FeatureExtractor_Loss_Function, Contextual_Loss_Function, GradientSensitive_Loss_Function, Gram_Loss_Function
    DirectPixels_Loss_Function.to(Generator_device);
    FeatureExtractor_Loss_Function.to(netF_device);
    Contextual_Loss_Function.to(Generator_device);
    GradientSensitive_Loss_Function.to(Generator_device)
    Gram_Loss_Function.to(Generator_device);

def Discriminator_Loss_Functions_to_GPU():
    global GAN_Validity_Loss_Function, GradientPenalty_Loss_Function, Relativistic_GAN_Validity_Loss_Function
    GAN_Validity_Loss_Function.to(discriminator_device)
    GradientPenalty_Loss_Function.to(discriminator_device)
    Relativistic_GAN_Validity_Loss_Function.to(discriminator_device)

def Generator_to_train_mode(Generator_Network):
    Generator_Network.train();
    Generator_Network = unfreeze_gradients(Generator_Network)
    return Generator_Network;

def Generator_to_eval_mode(Generator_Network):
    Generator_Network.eval();
    Generator_Network = freeze_gradients(Generator_Network);
    return Generator_Network;

def Discriminator_to_train_mode(Discriminator_Network):
    Discriminator_Network.train();
    Discriminator_Network = unfreeze_gradients(Discriminator_Network)
    return Discriminator_Network

def Discriminator_to_eval_mode(Discriminator_Network):
    Discriminator_Network.eval()
    Discriminator_Network = freeze_gradients(Discriminator_Network);
    return Discriminator_Network

def load_Generator_from_checkpoint(models_folder='C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Model New Checkpoints',
                                   load_Generator_filename='Generator_Network_TEST1_Step1020.pth',
                                   Generator_Network=None):
    # Get filename and postfix (.pt / .pth):
    if len(load_Generator_filename.split('.'))==1: #file doesn't have a suffix (.pth or .pt)
        load_Generator_filename += '.pth'
        filename_Generator_type = '.pth'
    else:
        filename_Generator_type = '.' + load_Generator_filename.split('.')[-1]

    # Get Generator Full Path:
    # path_Generator = os.path.join(models_folder , str(load_Generator_filename))
    path_Generator = os.path.join(models_folder , load_Generator_filename.split('_TEST1')[0])
    path_Generator = os.path.join(path_Generator , str(load_Generator_filename) )


    # If We Inserted a Network Then Load It There:
    if not Generator_Network:
        Generator_Network = get_Original_Generator();
    # Otherwise, Load New Generator:
    if path_Generator.split('.')[1] == 'pth':
        Generator_Network.load_state_dict(torch.load(path_Generator));
    elif path_Generator.split('.')[1] == 'pt':
        Generator_Network = torch.load(path_Generator);
    return Generator_Network;


def save_Generator_parts_to_checkpoint(Generator_Network,folder,basic_filename,flag_save_dict_or_whole='dict'):
    # Save Generator:
    if not os.path.exists(folder):
        os.makedirs(folder)
    basic_filename = basic_filename.split('.')[0] #in case i pass in filename.pth for generality and easiness sake
    if flag_save_dict_or_whole == 'dict' or 'both':
        path_Generator = os.path.join(folder,str(basic_filename))  + '.pth'
        #to get state_dict() it must be on cpu...so to do things straightforwardly - pass network to cpu and then back to gpu:
        Generator_Network = Generator_to_CPU(Generator_Network);
        torch.save(Generator_Network.state_dict(), path_Generator);
        Generator_Network = Generator_to_GPU(Generator_Network);
    if flag_save_dict_or_whole == 'whole' or 'both':
        path_Generator = os.path.join(folder,str(basic_filename))  + '.pt'
        Geneator_Network = Generator_to_CPU(Generator_Network);
        torch.save(Generator_Network, path_Generator);
        Generator_Network = Generator_to_GPU(Generator_Network);

def load_Discriminator_from_checkpoint(folder,filename,Discriminator_Network):
    Discriminator_load_path = os.path.join(folder,filename)
    if filename.split('.')[1] == 'pt':
        Discriminator_Network = torch.load(Discriminator_load_path)
    elif filename.split('.')[1] == 'pth':
        Discriminator_Network.load_state_dict(torch.load(Discriminator_load_path))
    return Discriminator_Network

def save_Discriminator_to_checkpoint(Discriminator_Network,folder,filename,flag_save_dict_or_whole='dict'):
    Discriminator_save_path = os.path.join(folder,filename);
    Discriminator_Network = Discriminator_to_CPU(Discriminator_Network);
    if flag_save_dict_or_whole == 'dict' or 'both':
        torch.save(Discriminator_Network.state_dict(), Discriminator_save_path+'.pth');
    if flag_save_dict_or_whole == 'whole' or 'both':
        torch.save(Discriminator_Network, Discriminator_save_path+'.pt');
    Discriminator_Network = Discriminator_to_GPU(Discriminator_Network);

def Generator_to_CPU(Generator_Network, netF=None):
    Generator_Network = Generator_Network.to('cpu')
    Generator_Network.hidden_states_to_device(device_cpu);
    # Generator_Network.reset_hidden_states()
    if netF:
        netF = netF.to('cpu');
    return Generator_Network;

def Generator_Loss_Functions_to_CPU():
    global DirectPixels_Loss_Function, FeatureExtractor_Loss_Function, Contextual_Loss_Function, GradientSensitive_Loss_Function, Gram_Loss_Function
    DirectPixels_Loss_Function.to('cpu');
    FeatureExtractor_Loss_Function.to('cpu');
    Contextual_Loss_Function.to('cpu');
    GradientSensitive_Loss_Function.to('cpu')
    Gram_Loss_Function.to('cpu')

def Discriminator_to_CPU(Discriminator_Network):
    Discriminator_Network = Discriminator_Network.cpu()
    return Discriminator_Network

def Discriminator_Loss_Functions_to_CPU():
    global GAN_Validity_Loss_Function, GradientPenalty_Loss_Function, Relativistic_GAN_Validity_Loss_Function
    GAN_Validity_Loss_Function.to('cpu')
    GradientPenalty_Loss_Function.to('cpu')
    Relativistic_GAN_Validity_Loss_Function.to('cpu')

def get_netF(FeatureExtractor_name='resnet', number_of_conv_layers_to_return=16):
    netF = define_F2(number_of_conv_layers_to_return=number_of_conv_layers_to_return, use_bn=False, FeatureExtractor_name=FeatureExtractor_name)
    return netF;



def get_Generator_Optimizer(Generator_Network, learning_rate=1e-3*(0.5**6), hypergrad_lr=5e-9):
    #TODO: i could have a different optimizer and lr for each part of the network
    parameters_to_optimize = [
        {
            'params': Generator_Network.parameters()
        },
    ]
    # optimizer_G = AdamHD(parameters_to_optimize, lr=learning_rate, hypergrad_lr=hypergrad_lr)
    optimizer_G = Adam(parameters_to_optimize, lr=learning_rate)
    return optimizer_G

def get_Discriminator_Optimizer(Discriminator_Network, learning_rate=1e-3*(0.5**6), hypergrad_lr=5e-9):
    parameters_to_optimize = [
        {
            'params': Discriminator_Network.parameters()
        }
                              ]
    optimizer_D = AdamHD(parameters_to_optimize, lr=learning_rate, hypergrad_lr=hypergrad_lr)
    return optimizer_D

def get_Network_Layers_Devices(Network):
    Generator_Network_layers_list = get_network_layers_list_flat(Generator_Network);
    for layer in Generator_Network_layers_list:
        if len(list(layer.parameters()))>0:
            print(layer.parameters().__next__().device)

def freeze_non_temporal_layers(Network):
    for name, value in Generator_Network.named_parameters():
        if 'cell' not in str.lower(name):
            value.requires_grad = False;
    return Network;

def plot_grad_flow(Generator_Network):
    #Use this function AFTER doing a .backward() which updates the .grad property of the layers
    average_abs_gradients_list = []
    layers_list = []
    for parameter_name, parameter in Generator_Network.named_parameters():
        if (parameter.requires_grad) and ("bias" not in parameter_name) and type(parameter.grad)!=type(None): #we usually don't really care about the bias term
            layers_list.append(parameter_name.split('.weight'))
            average_abs_gradients_list.append(parameter.grad.abs().mean())
    plt.plot(average_abs_gradients_list, alpha=0.3, color="b")
    # plt.scatter(arange(len(average_abs_gradients_list)),average_abs_gradients_list)
    plt.hlines(0, 0, len(average_abs_gradients_list)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(average_abs_gradients_list), 1), layers_list, rotation="horizontal")
    plt.xlim(xmin=0, xmax=len(average_abs_gradients_list))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    x_vec = arange(0,len(average_abs_gradients_list),1)
    for i,layer_name in enumerate(layers_list):
        text(x_vec[i], average_abs_gradients_list[i], layers_list[i], fontdict=None, withdash=False, rotation='vertical', verticalalignment ='bottom')
    plt.title("Gradient flow")
    plt.grid(True)
# #Use Example:










#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################
#############################################################################################################################################################################################################################
### Show Still Image Results: ###

#### GPU Allocation Program: ####
default_device = device0
Generator_device = default_device
netF_device = default_device
discriminator_device = default_device


#####################################################################################################################################################################
### Get Original Networks and New Optimizer Optimizers: ###
flag_use_new_or_checkpoint_generator = 'checkpoint';  # 'new', 'checkpoint'
netF_name = 'resnet'  # 'vgg', 'resnet', 'resnet50dilated-ppm_deepsup'
number_of_conv_layers = 10
flag_use_fresh_discriminator_or_checkpoint_discriminator = 'fresh'
reference_image_load_paths_list = ['C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim01.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim02.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim03.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim04.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim05.png', 'C:/Users\dkarl\PycharmProjects\dudykarl/4dudi\generated images/test images/kodim15.png']
flag_initialize_hidden_with_zero_or_input = 'zero'; #'zero' / 'input'
flag_write_TB = True;
# flag_write_TB = False;


### Flags: ###
flag_train_or_not = True;
flag_residual_learning = False;
flag_show_image_or_highpass_or_difference = True; # 0=image, 1=highpass details
flag_learn_clean_or_running_average = 0;
# noise_gain = 30;
number_of_time_steps = 25; #number of time steps before doing a .backward()
noise_gains = [100];

flag_residual_learning = False;  #### MAKE SURE THIS FITS THE WAY THE ORIGINAL MODEL WAS TRAINED (TODO: maybe use a klepto file to save all those stuff i want to remember!!@#@?!#?!)
flag_show_image_or_highpass_or_difference = 0; # 0=image, 1=highpass details
flag_learn_clean_or_running_average = 0;
flag_use_all_frames = True


max_total_number_of_batches = 1;
flag_limit_number_of_iterations = True;

### Whether To Show Images Whilst Training: ###
flag_show_generated_images_while_Generator_training = False
# flag_show_generated_images_while_Generator_training = True


### LOAD filenames and folders: ###
models_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Model New Checkpoints'
# load_Generator_filename = 'UNETV12_V3_alpha_estimator_NoiseGain30_TEST1_Step65900_GenLR_4d00em05.pt';
# load_Generator_filename = 'UNET_V20_V2_Try1LSTM_flags_Noise30_TEST1_Step4800_GenLR_5d00em05.pt';
# load_Generator_filename = 'UNET_V22_V2__Shifts1_LSTM_flags_Noise30_TEST1_Step21700_GenLR_3d20em05.pt';
# load_Generator_filename = 'UNET_V21_V1LSTM_flags_Noise30_TEST1_Step9400_GenLR_2d00em05.pt';
# load_Generator_filename = 'UNET_V21_V2LSTM_flags_Noise30_TEST1_Step7900_GenLR_8d00em05.pt';
# load_Generator_filename = 'UNET_V22_V2_TWICE_Shifts0_LSTM_flags_Noise100_TEST1_Step1_GenLR_4d00em04.pt';
load_Generator_filename = 'UNET_V26_V1__IntegerShifts1_LSTM_flags_Noise30_TEST1_Step47100_GenLR_2d00em05.pt';
# load_Generator_filename = 'UNET_V26_V1__IntegerShifts0_LSTM_flags_Noise30_TEST1_Step29600_GenLR_1d28em249.pt';


### Save filenames and folders: ###
save_models_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Official Testing'
save_Generator_filename = load_Generator_filename.split('_TEST1')[0]
path_make_path_if_none_exists(save_Generator_filename)


### LOAD Network From Checkpoint If Wanted: ###
Generator_checkpoint_step = int(load_Generator_filename.split('Step')[1].split('_')[0])   #Assumes only one Step substring in the Generator load filename
Generator_Network = torch.load(os.path.join(models_folder,load_Generator_filename.split('_TEST1')[0], load_Generator_filename))
netF = get_netF(FeatureExtractor_name=netF_name, number_of_conv_layers_to_return=number_of_conv_layers)
Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device);
Generator_Network = Generator_to_eval_mode(Generator_Network)
# Generator_Network = Generator_to_train_mode(Generator_Network)
netF = freeze_gradients(netF)


### Save Models/Images Frequency ([iterations]): ###
save_models_frequency = 100;
save_images_frequency = 30;


### Flags and Thresholds: ###
# (1). High level flags:
flag_train_Generator = True
flag_train_Discriminator = False
# (2). Initial Steps:
number_of_initial_Generator_steps = 0
number_of_initial_Discriminator_steps = 0


################################
### Generator Loss Function: ###
################################
L1_Loss_Function = nn.L1Loss()
SSIM_Loss_Function = SSIM()
PSNR_Loss_Function = PSNR_Loss()
### Generator Loss Functions to GPU: ###
# Generator_Loss_Functions_to_GPU()

########################################################################################################################################







########################################################################################################################################
### Initialize Some Things For The Training Loop: ###
### Residual Learning: ###
gaussian_blur_layer = get_gaussian_kernel(kernel_size=5, sigma=5, channels=3)  ####  Use the parameters relevant for the experiment!!@##@  ####
gaussian_blur_layer.to(Generator_device)
#############################################################################################################################################################################################################################








#######################################################################################################################################
#(*). Shift Images
# batch_size = 100; #TODO: lowered!
min_crop_size = 140
crop_size = 140
final_crop_size = 100
max_number_of_images = 1000

import Albumentations_utils
from Albumentations_utils import *
import IMGAUG_utils
from IMGAUG_utils import *

### Ways

### IMGAUG: ###
imgaug_parameters = get_IMGAUG_parameters()
# Affine:
max_shift = 0;
max_rotation_angle = 0;
max_scale_delta = 0;
imgaug_parameters['flag_affine_transform'] = True
imgaug_parameters['affine_scale'] = 1
imgaug_parameters['affine_translation_percent'] = None
imgaug_parameters['affine_translation_number_of_pixels'] = {"x": (-max_shift,max_shift), "y": (-max_shift,max_shift)}
imgaug_parameters['affine_rotation_degrees'] = 0
imgaug_parameters['affine_shear_degrees'] = 0
imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
imgaug_parameters['probability_of_affine_transform'] = 1
#Perspective:
imgaug_parameters['flag_perspective_transform'] = False
imgaug_parameters['flag_perspective_transform_keep_size'] = True
imgaug_parameters['perspective_transform_scale'] = 0
imgaug_parameters['probability_of_perspective_transform'] = 1
### Get Augmenter: ###
imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)



# images_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl/Imagenet'
images_folder = 'F:\TNR - OFFICIAL TEST IMAGES\Multiple_Test_Images_To_Change_On_The_Fly'
Cn2_value = 5e-13;
train_dataset = ImageFolderRecursive_MaxNumberOfImages_Deformations(images_folder,
                                                                    transform=imgaug_transforms,
                                                                    flag_base_transform = False,
                                                                    flag_turbulence_transform = False,
                                                                    max_number_of_images = max_number_of_images,
                                                                    Cn2 = Cn2_value,
                                                                    min_crop_size=min_crop_size,
                                                                    crop_size=min_crop_size,
                                                                    loader='CV',
                                                                    allowed_extensions = IMG_EXTENSIONS_PNG,
                                                                    flag_output_YUV = False,
                                                                    flag_output_HSV = False,
                                                                    flag_output_channel_average = False,
                                                                    flag_recursive = True,
                                                                    flag_explicitely_make_tensor = False,
                                                                    flag_normalize_by_255 = True
)
batch_size = len(train_dataset)
# batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False, pin_memory=True);  ### Mind the shuffle variable!@#!#%@#%@$%
# train_dataloader = torch.utils.data.DataLoader(train_dataset, pin_memory=False,
#                                                batch_sampler=torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.SequentialSampler(range(batch_size)), batch_size=batch_size, drop_last=False));  ### Mind the shuffle variable!@#!#%@#%@$%
# bla = train_dataloader.__iter__().__next__()

# tic()
# for i, data_from_dataloader in enumerate(train_dataloader):
#     # imshow_torch(data_from_dataloader[i,:,:,:])
#     # pause(0.2)
#     if i>=0:
#         break;
# toc()

######################################################################################################################################










#(1). Start Recording Video:
image_index = 1;
fps = 1.0
flag_save_videos = False;
flag_show_results = True;
flag_save_video_or_images = 'images' #'images' / 'videos' / 'both'

crop_size = 20000;
created_video_filename_prefix = load_Generator_filename.split('.')[0] + '_Shift' + str(max_shift) + '_Rot' + str(max_rotation_angle) + '_Scale' + str(max_scale_delta)
super_folder_to_save_result_videos = 'F:\TNR - OFFICIAL TEST IMAGES\Ensamble_Images_Results'
sub_folder_to_save_current_results_videos_ALL_DETAILS = super_folder_to_save_result_videos + created_video_filename_prefix + '/ALL_DETAILS';
sub_folder_to_save_current_results_videos_HIGHPASS = super_folder_to_save_result_videos + created_video_filename_prefix + '/HIGHPASS';
sub_folder_to_save_current_results_videos_DIFFERENCE = super_folder_to_save_result_videos + created_video_filename_prefix + '/DIFFERENCE';
if not os.path.exists(sub_folder_to_save_current_results_videos_ALL_DETAILS):
    os.makedirs(sub_folder_to_save_current_results_videos_ALL_DETAILS)
if not os.path.exists(sub_folder_to_save_current_results_videos_HIGHPASS):
    os.makedirs(sub_folder_to_save_current_results_videos_HIGHPASS)
if not os.path.exists(sub_folder_to_save_current_results_videos_DIFFERENCE):
    os.makedirs(sub_folder_to_save_current_results_videos_DIFFERENCE)
save_figures_folder = os.path.join(super_folder_to_save_result_videos, created_video_filename_prefix)
path_make_path_if_none_exists(save_figures_folder)



#########################
#### VIDEOS: ####
video_postfix_type = '.mpeg'
#ALL DETAILS:
created_video_filename_original_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'original' + video_postfix_type)
created_video_filename_noisy_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'noisy' + video_postfix_type)
created_video_filename_clean_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'clean' + video_postfix_type)
created_video_filename_Concat_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'Concat' + video_postfix_type)
#HIGHPASS:
created_video_filename_original_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'original' + video_postfix_type)
created_video_filename_noisy_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'noisy' + video_postfix_type)
created_video_filename_clean_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'clean' + video_postfix_type)
created_video_filename_Concat_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'Concat' + video_postfix_type)
fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # Be sure to use lower case  MP42
########################


if flag_save_video_or_images == 'images' or flag_save_video_or_images == 'both':
    #ALL DETAILS:
    original_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'original')
    noisy_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'noisy')
    clean_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'clean')
    Concat_video_images_folder_ALL_DETAILS = os.path.join(sub_folder_to_save_current_results_videos_ALL_DETAILS, 'Concat')
    if not os.path.exists(original_video_images_folder_ALL_DETAILS):
        os.makedirs(original_video_images_folder_ALL_DETAILS)
    if not os.path.exists(noisy_video_images_folder_ALL_DETAILS):
        os.makedirs(noisy_video_images_folder_ALL_DETAILS)
    if not os.path.exists(clean_video_images_folder_ALL_DETAILS):
        os.makedirs(clean_video_images_folder_ALL_DETAILS)
    if not os.path.exists(Concat_video_images_folder_ALL_DETAILS):
        os.makedirs(Concat_video_images_folder_ALL_DETAILS)
    #HIGHPASS:
    original_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'original')
    noisy_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'noisy')
    clean_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'clean')
    Concat_video_images_folder_HIGHPASS = os.path.join(sub_folder_to_save_current_results_videos_HIGHPASS, 'Concat')
    if not os.path.exists(original_video_images_folder_HIGHPASS):
        os.makedirs(original_video_images_folder_HIGHPASS)
    if not os.path.exists(noisy_video_images_folder_HIGHPASS):
        os.makedirs(noisy_video_images_folder_HIGHPASS)
    if not os.path.exists(clean_video_images_folder_HIGHPASS):
        os.makedirs(clean_video_images_folder_HIGHPASS)
    if not os.path.exists(Concat_video_images_folder_HIGHPASS):
        os.makedirs(Concat_video_images_folder_HIGHPASS)
    #DIFFERENCE:
    original_clean_video_images_folder_DIFFERENCE = os.path.join(sub_folder_to_save_current_results_videos_DIFFERENCE, 'original-clean')
    noisy_clean_video_images_folder_DIFFERENCE = os.path.join(sub_folder_to_save_current_results_videos_DIFFERENCE, 'noisy-clean')
    original_noisy_video_images_folder_DIFFERENCE = os.path.join(sub_folder_to_save_current_results_videos_DIFFERENCE, 'original-noisy')
    concat_video_images_folder_DIFFERENCE = os.path.join(sub_folder_to_save_current_results_videos_DIFFERENCE, 'Concat')
    if not os.path.exists(original_clean_video_images_folder_DIFFERENCE):
        os.makedirs(original_clean_video_images_folder_DIFFERENCE)
    if not os.path.exists(noisy_clean_video_images_folder_DIFFERENCE):
        os.makedirs(noisy_clean_video_images_folder_DIFFERENCE)
    if not os.path.exists(original_noisy_video_images_folder_DIFFERENCE):
        os.makedirs(original_noisy_video_images_folder_DIFFERENCE)
    if not os.path.exists(concat_video_images_folder_DIFFERENCE):
        os.makedirs(concat_video_images_folder_DIFFERENCE)


### Amount Of Training: ###
number_of_epochs = 1;
flag_stop_iterating = False;


### Initialize Loop Steps: ###
global_batch_step = 0;
number_of_time_steps_so_far = 0
current_step = 0;
# Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device);
# Generator_Network = Generator_to_eval_mode(Generator_Network)
# Generator_Network = Generator_to_train_mode(Generator_Network)



### Initialize Lists to keep track of Metrics: ###
fake_to_average_L1_list = []
fake_to_clean_L1_list = []
average_to_clean_L1_list = []
fake_to_clean_PSNR_list = []
average_to_clean_PSNR_list = []
fake_to_clean_SSIM_list = []
average_to_clean_SSIM_list = []


### Noise and Deformations On-The-Fly: ###
video_crop_size = 20000;



### Initialize Lists to keep track of Metrics For Different Gains: ###
fake_to_average_L1_list_by_noise = []
fake_to_clean_L1_list_by_noise = []
average_to_clean_L1_list_by_noise = []
fake_to_clean_PSNR_list_by_noise = []
average_to_clean_PSNR_list_by_noise = []
fake_to_clean_SSIM_list_by_noise = []
average_to_clean_SSIM_list_by_noise = []


flag_stop_iterating = False
max_total_number_of_batches = 100
for noise_gain_index, noise_gain in enumerate(noise_gains):
    ### Initialize Lists to keep track of Metrics For Current Gains: ###
    fake_to_average_L1_list = []
    fake_to_clean_L1_list = []
    average_to_clean_L1_list = []
    fake_to_clean_PSNR_list = []
    average_to_clean_PSNR_list = []
    fake_to_clean_SSIM_list = []
    average_to_clean_SSIM_list = []

    for batch_index, data_from_dataloader in enumerate(train_dataloader):

        ### Limit number of batches: ###
        if (global_batch_step == max_total_number_of_batches or flag_stop_iterating==True) and flag_limit_number_of_iterations:
            flag_stop_iterating = True;
            break;
        global_batch_step += 1


        ### At Very First Time Step Only: ###
        if global_batch_step == 1:
            batch_size, number_of_channels, frame_height, frame_width = data_from_dataloader.shape
            crop_size_width = min(video_crop_size, frame_width);
            crop_size_height = min(video_crop_size, frame_height);
            start_index_height = random_integers(0,frame_height - crop_size_height);
            start_index_width = random_integers(0,frame_width - crop_size_width)
            if flag_save_videos:
                #ALL DETAILS:
                video_writer_original_ALL_DETAILS = cv2.VideoWriter(created_video_filename_original_ALL_DETAILS, fourcc, fps, (frame_width*1, frame_height))
                video_writer_noisy_ALL_DETAILS = cv2.VideoWriter(created_video_filename_noisy_ALL_DETAILS, fourcc, fps, (frame_width * 1, frame_height))
                video_writer_clean_ALL_DETAILS = cv2.VideoWriter(created_video_filename_clean_ALL_DETAILS, fourcc, fps, (frame_width * 1, frame_height))
                video_writer_Concat_ALL_DETAILS = cv2.VideoWriter(created_video_filename_Concat_ALL_DETAILS, fourcc, fps, (frame_width * 3, frame_height))
                #HIGHPASS:
                video_writer_original_HIGHPASS = cv2.VideoWriter(created_video_filename_original_HIGHPASS, fourcc, fps, (frame_width * 1, frame_height))
                video_writer_noisy_HIGHPASS = cv2.VideoWriter(created_video_filename_noisy_HIGHPASS, fourcc, fps, (frame_width * 1, frame_height))
                video_writer_clean_HIGHPASS = cv2.VideoWriter(created_video_filename_clean_HIGHPASS, fourcc, fps, (frame_width * 1, frame_height))
                video_writer_Concat_HIGHPASS = cv2.VideoWriter(created_video_filename_Concat_HIGHPASS, fourcc, fps, (frame_width * 3, frame_height))


        ###################################################################################################################################################################################################
        ########################################  Train Generator: #############################################
        torch.cuda.empty_cache()
        number_of_time_steps_from_batch_start = 0;
        averaged_image = 0;
        print('New Batch')


        ### Things to do at start: ###
        if number_of_time_steps_from_batch_start == 0:  # at first initialize with input_
            if flag_initialize_hidden_with_zero_or_input == 'zero':
                reset_hidden_states_flags_list = [1] * batch_size;
            elif flag_initialize_hidden_with_zero_or_input == 'input':
                reset_hidden_states_flags_list = [2] * batch_size;
        elif number_of_time_steps_from_batch_start > 0:
            reset_hidden_states_flags_list = [3] * batch_size;


        ### Zero Generator Grad & Reset Hidden States: ###
        torch.cuda.empty_cache()
        total_generator_loss = 0


        ############## Go Through Time Steps: #########################
        tic()
        for current_time_step in arange(0, number_of_time_steps):
            number_of_time_steps_from_batch_start += 1
            number_of_time_steps_so_far += 1; #global

            ### Time-Shift & Put Noise in input tensor: ###
            # (1). Get Original Tensors:
            if len(data_from_dataloader.shape)==5:
                #Time-Series
                current_clean_tensor = data_from_dataloader[:, current_time_step, :, :, :].type(torch.FloatTensor)
                current_noisy_tensor = data_from_dataloader[:, current_time_step, :, :, :].type(torch.FloatTensor)
                current_clean_tensor = current_clean_tensor / 255;
                current_noisy_tensor = current_noisy_tensor / 255;
            else:
                #Still Image:
                current_clean_tensor = data_from_dataloader.type(torch.FloatTensor)
                current_noisy_tensor = data_from_dataloader.type(torch.FloatTensor)



            # ### TODO: make sure this is appropriate!!!! ###
            # ### Change Clean Tensors Values Range To Avoid Over & Under Flows When Adding Noise / Deforming Using Turbulence: ###
            # current_clean_tensor = current_clean_tensor * 0.8
            # current_clean_tensor = current_clean_tensor + 0.1


            ### Get original data_from_dataloader to contrast with "actual" averaging: ###
            #(*). Note: i don't have the time or will to do so-to-speak the "inverse transform/warp" and to average so i simply take a certain crop, add noise and average it out and statistically it should be the same
            #           (this obviously assumes constant image deformations and not video....).
            #(*). Note: i need to understand whether Albumentations or IMGAUG can easily return the parameters
            if current_time_step == 0:
                current_batch_dataset = DataSet_Numpy_Deformations(numpy_array=np.transpose(current_clean_tensor.cpu().numpy(), [0,2,3,1]),
                                                                   transform=imgaug_transforms,
                                                                   crop_size=final_crop_size,
                                                                   Cn2=Cn2_value,
                                                                   flag_base_transform=True,
                                                                   flag_turbulence_transform=False,
                                                                   flag_output_YUV = False,
                                                                   flag_output_HSV = False,
                                                                   flag_output_channel_average = False)
                current_batch_dataloader = torch.utils.data.DataLoader(current_batch_dataset, batch_size=batch_size, shuffle=False, pin_memory=True);
                current_batch_dataset.images_torch = current_batch_dataset.images_torch.to(Generator_device)
                current_batch_dataset.images_torch_cropped = current_batch_dataset.images_torch_cropped.to(Generator_device)


            # ### Shift Tensors: ###
            current_clean_tensor = current_batch_dataloader.__iter__().__next__()
            current_noisy_tensor = current_clean_tensor

            ### Send to GPU Device: ###
            current_clean_tensor = current_clean_tensor.to(Generator_device);
            current_noisy_tensor = current_noisy_tensor.to(Generator_device);

            ### Add Noise: ###
            dshape = current_noisy_tensor.shape
            dshape = np.array(dshape)
            ### TODO: make sure whether to use this and whether to use flags or not etc.... ###
            # current_noisy_tensor = current_noisy_tensor.clamp(0,1)
            # current_clean_tensor = current_clean_tensor.clamp(0,1)
            current_noisy_tensor = noise_RGB_LinearShotNoise_Torch(current_noisy_tensor, gain=noise_gain, flag_output_uint8_or_float='float')



            ############################################################################################################################################################################
            ### Pass input through network: ###
            current_noisy_tensor = Variable(current_noisy_tensor, requires_grad=False)
            output_tensor = Generator_Network(current_noisy_tensor, reset_hidden_states_flags_list)
            reset_hidden_states_flags_list = [3] * batch_size;  #In any case after whatever initial initialization signal -> do nothing and continue accumulating graph


            ### Pass each input through SID: ###


            ### TNR: ###
            #############################################################################################################################################################################


            ### Keep Track Of Running Average: ###
            if current_time_step == 0 and number_of_time_steps_from_batch_start == 1:  # very start of batch
                print('initialize average image with noisy tensor')
                averaged_image = noise_RGB_LinearShotNoise_Torch(current_batch_dataset.images_torch_cropped, gain=noise_gain, flag_output_uint8_or_float='float');
            else:
                total_time_steps_this_batch = number_of_time_steps_from_batch_start
                alpha = total_time_steps_this_batch / (total_time_steps_this_batch + 1)
                current_noisy_tensor_for_averaging = noise_RGB_LinearShotNoise_Torch(current_batch_dataset.images_torch_cropped, gain=noise_gain, flag_output_uint8_or_float='float');
                averaged_image = (averaged_image * alpha + current_noisy_tensor_for_averaging * (1 - alpha));


            ### Get Ground-Truth & Generated Images (contingent on whether i used residual learning or not): ###
            if flag_residual_learning:
                if flag_learn_clean_or_running_average == 0:
                    real_images = current_clean_tensor - gaussian_blur_layer(current_noisy_tensor)
                else:
                    real_images = averaged_image - gaussian_blur_layer(current_noisy_tensor)
                fake_images = output_tensor[:, 0:3, :, :];
            else:
                if flag_learn_clean_or_running_average == 0:
                    real_images = current_clean_tensor;
                else:
                    real_images = averaged_image;
                fake_images = output_tensor[:, 0:3, :, :];



            ### Save Losses and Metrics: ###
            #(1). L1:
            average_to_clean_L1 = L1_Loss_Function(averaged_image, current_batch_dataset.images_torch_cropped).item();  # between average image and Clean
            average_to_target_L1 = L1_Loss_Function(averaged_image, real_images).item() #between average and target
            fake_to_clean_L1 = L1_Loss_Function(fake_images, current_clean_tensor).item();  # between clean image and Generated
            fake_to_average_L1 = L1_Loss_Function(fake_images, averaged_image).item();  # between average image and Generated
            fake_to_target_L1 = L1_Loss_Function(fake_images, real_images).item() #between Fake and Target
            #(2). PSNR:
            average_to_clean_PSNR = PSNR_Loss_Function(averaged_image, current_batch_dataset.images_torch_cropped).item();  # between average image and Clean
            average_to_target_PSNR = PSNR_Loss_Function(averaged_image, real_images).item()  # between average and target
            fake_to_clean_PSNR = PSNR_Loss_Function(fake_images, current_clean_tensor).item();  # between clean image and Generated
            fake_to_average_PSNR = PSNR_Loss_Function(fake_images, averaged_image).item();  # between average image and Generated
            fake_to_target_PSNR = PSNR_Loss_Function(fake_images, real_images).item()  # between Fake and Target
            #(3). SSIM:
            average_to_clean_SSIM = SSIM_Loss_Function(averaged_image, current_batch_dataset.images_torch_cropped).item();  # between average image and Clean
            average_to_target_SSIM = SSIM_Loss_Function(averaged_image, real_images).item()  # between average and target
            fake_to_clean_SSIM = SSIM_Loss_Function(fake_images, current_clean_tensor).item();  # between clean image and Generated
            fake_to_average_SSIM = SSIM_Loss_Function(fake_images, averaged_image).item();  # between average image and Generated
            fake_to_target_SSIM = SSIM_Loss_Function(fake_images, real_images).item()  # between Fake and Target

            ### Append to Metrics' Lists: ###
            fake_to_average_L1_list.append(fake_to_average_L1)
            fake_to_clean_L1_list.append(fake_to_clean_L1)
            average_to_clean_L1_list.append(average_to_clean_L1)
            fake_to_clean_PSNR_list.append(fake_to_clean_PSNR)
            fake_to_clean_SSIM_list.append(fake_to_clean_SSIM)
            average_to_clean_SSIM_list.append(average_to_clean_SSIM)
            average_to_clean_PSNR_list.append(average_to_clean_PSNR)

            # print('Fake-Clean: ' + str(generated_to_clean) + ', Fake-Average: ' + str(generated_to_average) + ', Average-Clean: ' + str(average_to_clean) + ', Fake-Target: ' + str(generated_to_target))
            print('Fake-Clean: ' + str(fake_to_clean_L1) +
                  ', Fake-Average: ' + str(fake_to_average_L1) +
                  ', Average-Clean: ' + str(average_to_clean_L1) +
                  ', Fake-Clean PSNR: ' + str(fake_to_clean_PSNR) +
                  ', Fake-Clean SSIM: ' + str(fake_to_clean_SSIM)
                  )



            ### Show Results: ###
            if flag_show_generated_images_while_Generator_training:
                if flag_residual_learning:
                    figure(2)
                    if flag_show_image_or_highpass_or_difference == 0:  # Image
                        final_real_images = real_images + gaussian_blur_layer(current_noisy_tensor)
                        final_fake_images = fake_images + gaussian_blur_layer(current_noisy_tensor)
                    if flag_show_image_or_highpass_or_difference == 1:  # HighPass Details
                        final_real_images = real_images
                        final_fake_images = fake_images
                    L1_loss_now = nn.L1Loss()(final_real_images, final_fake_images).item()
                    subplot(1, 2, 1)
                    imshow(final_real_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                    title('real image')
                    subplot(1, 2, 2)
                    imshow(final_fake_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                    title('fake image, L1: ' + str(L1_loss_now))
                    pause(0.1);
                    matplotlib.pyplot.show(block=False)
                else:
                    if flag_show_image_or_highpass_or_difference == 1:  # HighPass Details
                        final_real_images = real_images - gaussian_blur_layer(current_noisy_tensor)
                        final_fake_images = fake_images - gaussian_blur_layer(current_noisy_tensor)

                    if flag_show_image_or_highpass_or_difference == 0:  # Image
                        final_real_images = real_images;
                        final_fake_images = fake_images;
                    figure(1)
                    subplot(1, 2, 1)
                    imshow(final_real_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                    title('real image')
                    subplot(1, 2, 2)
                    imshow(final_fake_images.cpu().data.numpy()[0, :, :, :].squeeze().transpose([1, 2, 0]).astype(float))
                    title('fake image')
                    pause(0.1);
                    matplotlib.pyplot.show(block=False)


            ### Loop to show both: ###
            # (1). all details
            # (2). highpass:
            for flag_show_image_or_highpass_or_difference in arange(3):

                ### Save Results: ###
                if flag_residual_learning:
                    if flag_show_image_or_highpass_or_difference == 0 or flag_show_image_or_highpass_or_difference == 2:  # Image
                        final_real_images = real_images + gaussian_blur_layer(current_noisy_tensor)
                        final_fake_images = fake_images + gaussian_blur_layer(current_noisy_tensor)
                        final_noisy_images = current_noisy_tensor
                    if flag_show_image_or_highpass_or_difference == 1:  # HighPass Details
                        final_real_images = real_images + gaussian_blur_layer(current_noisy_tensor) - gaussian_blur_layer(current_clean_tensor)
                        final_fake_images = fake_images + gaussian_blur_layer(current_noisy_tensor) - gaussian_blur_layer(output_tensor[:,0:3,:,:])
                        final_noisy_images = current_noisy_tensor - gaussian_blur_layer(current_noisy_tensor)
                else:
                    if flag_show_image_or_highpass_or_difference == 1:  # HighPass Details
                        final_real_images = real_images - gaussian_blur_layer(real_images)
                        final_fake_images = fake_images - gaussian_blur_layer(fake_images)
                        final_noisy_images = current_noisy_tensor - gaussian_blur_layer(current_noisy_tensor)
                    if flag_show_image_or_highpass_or_difference == 0 or flag_show_image_or_highpass_or_difference == 2:  # Image
                        final_real_images = real_images;
                        final_fake_images = fake_images;
                        final_noisy_images = current_noisy_tensor


                # Get original frame numpy:
                frame_original = np.transpose(final_real_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
                frame_original = np.uint8(frame_original.round())
                # Get noisy frame numpy:
                frame_noisy = np.transpose(final_noisy_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
                frame_noisy = np.uint8(frame_noisy.round())
                # Get clean frame numpy:
                frame_clean = np.transpose(final_fake_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
                frame_clean = np.uint8(frame_clean.round())
                # Get Concat frame numpy:
                frame_concat = np.concatenate((frame_original, frame_noisy, frame_clean), axis=1)
                # Get Difference frame numpy:
                difference_DC_to_void_underflow = 0.3;
                frame_difference_original_noisy = frame_original-frame_noisy + difference_DC_to_void_underflow
                frame_difference_original_clean = frame_original-frame_clean + difference_DC_to_void_underflow
                frame_difference_noisy_clean = frame_noisy-frame_clean + difference_DC_to_void_underflow
                frame_difference_concat = np.concatenate((frame_difference_original_noisy, frame_difference_original_clean, frame_difference_noisy_clean), axis=1)


                noisy_min = fake_images.min();
                noisy_max = fake_images.max();

                ### Write Frame To Video: ###
                if flag_save_videos:
                    if flag_show_image_or_highpass_or_difference==0:
                        pause(0.01);
                        video_writer_Concat_ALL_DETAILS.write(frame_concat)
                        pause(0.01);
                        video_writer_original_ALL_DETAILS.write(frame_original);
                        pause(0.01);
                        video_writer_noisy_ALL_DETAILS.write(frame_noisy);
                        pause(0.01);
                        video_writer_clean_ALL_DETAILS.write(frame_clean);
                    if flag_show_image_or_highpass_or_difference==1:
                        pause(0.01);
                        video_writer_Concat_HIGHPASS.write(frame_concat)
                        pause(0.01);
                        video_writer_original_HIGHPASS.write(frame_original);
                        pause(0.01);
                        video_writer_noisy_HIGHPASS.write(frame_noisy);
                        pause(0.01);
                        video_writer_clean_HIGHPASS.write(frame_clean);


                ### Write Seperate Frames To Folders: ###
                if flag_save_video_or_images == 'images' or flag_save_video_or_images == 'both':
                    if flag_show_image_or_highpass_or_difference == 0:
                        save_image_numpy(original_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_original, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(noisy_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(clean_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_clean, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(Concat_video_images_folder_ALL_DETAILS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_concat, flag_convert_bgr2rgb=False, flag_scale=False)
                    if flag_show_image_or_highpass_or_difference == 1:
                        save_image_numpy(original_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_original, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(noisy_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(clean_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_clean, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(Concat_video_images_folder_HIGHPASS, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_concat, flag_convert_bgr2rgb=False, flag_scale=False)
                    if flag_show_image_or_highpass_or_difference == 2:
                        save_image_numpy(original_clean_video_images_folder_DIFFERENCE, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_difference_original_clean, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(noisy_clean_video_images_folder_DIFFERENCE, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_difference_noisy_clean, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(original_noisy_video_images_folder_DIFFERENCE, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_difference_original_noisy, flag_convert_bgr2rgb=False, flag_scale=False)
                        save_image_numpy(concat_video_images_folder_DIFFERENCE, filename=str(number_of_time_steps_from_batch_start).rjust(10, '0') + '.png', numpy_array=frame_difference_concat, flag_convert_bgr2rgb=False, flag_scale=False)





        ## Updates Lists By Noise Gains: ###
        fake_to_average_L1_list_by_noise.append(fake_to_average_L1_list)
        fake_to_clean_L1_list_by_noise.append(fake_to_clean_L1_list)
        average_to_clean_L1_list_by_noise.append(average_to_clean_L1_list)
        fake_to_clean_PSNR_list_by_noise.append(fake_to_clean_PSNR_list)
        average_to_clean_PSNR_list_by_noise.append(average_to_clean_PSNR_list)
        fake_to_clean_SSIM_list_by_noise.append(fake_to_clean_SSIM_list)
        average_to_clean_SSIM_list_by_noise.append(average_to_clean_SSIM_list)

        ### END Of Generator Loss Accumulation! ###





### Plot Metrics Over Time: ###
linestyles = [(0,(3,1,1,1,1,1)), (0,(5,5)), (0,())]
colors = ['b','g']
L1_legends = []
PSNR_legends = []
SSIM_legends = []
for noise_gain_index, noise_gain in enumerate(noise_gains):
    current_noise_gain_fake_to_clean_L1_list = fake_to_clean_L1_list_by_noise[noise_gain_index]
    current_noise_gain_average_to_clean_L1_list = average_to_clean_L1_list_by_noise[noise_gain_index]
    current_noise_gain_fake_to_clean_PSNR_list = fake_to_clean_PSNR_list_by_noise[noise_gain_index]
    current_noise_gain_average_to_clean_PSNR_list = average_to_clean_PSNR_list_by_noise[noise_gain_index]
    current_noise_gain_fake_to_clean_SSIM_list = fake_to_clean_SSIM_list_by_noise[noise_gain_index]
    current_noise_gain_average_to_clean_SSIM_list = average_to_clean_SSIM_list_by_noise[noise_gain_index]

    figure(2)
    plot(current_noise_gain_fake_to_clean_L1_list, color=colors[0], linestyle=linestyles[noise_gain_index])
    plot(current_noise_gain_average_to_clean_L1_list, color=colors[1], linestyle=linestyles[noise_gain_index])
    title('L1 Loss Over Time. ' + 'Net:' + decimal_notation(current_noise_gain_fake_to_clean_L1_list[0], 4) + '->' + decimal_notation(current_noise_gain_fake_to_clean_L1_list[-1], 4) + ', Avg:' + decimal_notation(current_noise_gain_average_to_clean_L1_list[0], 4) + '->' + decimal_notation(current_noise_gain_average_to_clean_L1_list[-1], 4))
    L1_legends.append('Network-Clean L1, Noise: ' + str(noise_gain))
    L1_legends.append('Average-Clean L1, Noise: ' + str(noise_gain))

    figure(3)
    plot(current_noise_gain_fake_to_clean_PSNR_list, color=colors[0], linestyle=linestyles[noise_gain_index])
    plot(current_noise_gain_average_to_clean_PSNR_list, color=colors[1], linestyle=linestyles[noise_gain_index])
    title('PSNR Over Time. ' + 'Net:' + decimal_notation(current_noise_gain_fake_to_clean_PSNR_list[0], 2) + '->' + decimal_notation(current_noise_gain_fake_to_clean_PSNR_list[-1], 2) + ', Avg:' + decimal_notation(current_noise_gain_average_to_clean_PSNR_list[0], 2) + '->' + decimal_notation(current_noise_gain_average_to_clean_PSNR_list[-1], 2))
    PSNR_legends.append('Network-Clean PSNR, Noise: ' + str(noise_gain))
    PSNR_legends.append('Average-Clean PSNR, Noise: ' + str(noise_gain))

    figure(4)
    plot(current_noise_gain_fake_to_clean_SSIM_list, color=colors[0], linestyle=linestyles[noise_gain_index])
    plot(current_noise_gain_average_to_clean_SSIM_list, color=colors[1], linestyle=linestyles[noise_gain_index])
    title('SSIM Over Time. ' + 'Net:' + decimal_notation(current_noise_gain_fake_to_clean_SSIM_list[0], 4) + '->' + decimal_notation(current_noise_gain_fake_to_clean_SSIM_list[-1], 4) + ', Avg:' + decimal_notation(current_noise_gain_average_to_clean_SSIM_list[0], 4) + '->' + decimal_notation(current_noise_gain_average_to_clean_SSIM_list[-1], 4))
    SSIM_legends.append('Network-Clean SSIM, Noise: ' + str(noise_gain))
    SSIM_legends.append('Average-Clean SSIM, Noise: ' + str(noise_gain))

figure(2)
legend(L1_legends)
savefig(os.path.join(save_figures_folder,'L1.png'))
figure(3)
legend(PSNR_legends)
savefig(os.path.join(save_figures_folder,'PSNR.png'))
figure(4)
legend(SSIM_legends)
savefig(os.path.join(save_figures_folder,'SSIM.png'))





if flag_save_videos:
    cv2.destroyAllWindows()
    video_writer_original_ALL_DETAILS.release()
    video_writer_noisy_ALL_DETAILS.release()
    video_writer_clean_ALL_DETAILS.release()
    video_writer_Concat_ALL_DETAILS.release()
    video_writer_original_HIGHPASS.release()
    video_writer_noisy_HIGHPASS.release()
    video_writer_clean_HIGHPASS.release()
    video_writer_Concat_HIGHPASS.release()





















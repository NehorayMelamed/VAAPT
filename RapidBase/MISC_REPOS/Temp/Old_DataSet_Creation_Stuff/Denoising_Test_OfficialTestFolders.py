
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






def noise_RGB_LinearShotNoise_Torch(original_images_input, gain, flag_output_uint8_or_float='float', flag_clip=False):

    original_images = torch.zeros(original_images_input.shape).to(Generator_device)
    original_images = original_images.type(torch.float32)
    original_images.copy_(original_images_input)

    #Input is either uint8 between [0,255]  or float between [0,1]
    if original_images.dtype == torch.float32 or original_images.dtype == torch.float64:
        original_images *= 255;
        original_images = original_images.type(torch.float32)

    if gain != 0:
        default_gain = 20;
        electrons_per_pixel = original_images * default_gain / gain;
        std_shot_noise = torch.sqrt(electrons_per_pixel);
        noise_image = torch.randn(*original_images.shape).to(original_images.device)*std_shot_noise;
        noisy_image = electrons_per_pixel + noise_image
        noisy_image = noisy_image * gain / default_gain;
    else:
        noisy_image = original_images;

    if flag_output_uint8_or_float==torch.uint8 or flag_output_uint8_or_float=='uint8':
        noisy_image = torch.clamp(noisy_image, 0,255)
        noisy_image = noisy_image.type(torch.uint8)
    else:
        noisy_image = noisy_image/255;
        if flag_clip:
            noisy_image = torch.clamp(noisy_image, 0,1)

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
default_device = device1
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
number_of_time_steps = 25; #number of time steps before doing a .backward()

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
load_Generator_filename = 'UNETV12_V3_alpha_estimator_NoiseGain30_TEST1_Step65900_GenLR_4d00em05.pt';

### Save filenames and folders: ###
save_models_folder = 'C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Official Testing'
save_Generator_filename = load_Generator_filename.split('_TEST1')[0]


### LOAD Network From Checkpoint If Wanted: ###
Generator_checkpoint_step = int(load_Generator_filename.split('Step')[1].split('_')[0])   #Assumes only one Step substring in the Generator load filename
Generator_Network = torch.load(os.path.join(models_folder,load_Generator_filename.split('_TEST1')[0], load_Generator_filename))
netF = get_netF(FeatureExtractor_name=netF_name, number_of_conv_layers_to_return=number_of_conv_layers)
Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device);
Generator_Network = Generator_to_eval_mode(Generator_Network)
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





























####################################################################   Test By Getting Official Test Images + focus on saving resulting images: #####################################################################################
#######################################################################################################################################
#(*). Get all image folders which contain image sequences to test (including all super images and noises and deformation parameters etc'):
super_folder_to_get_all_sub_folders_from = 'F:\OFFICIAL TEST IMAGES\Still Images'
image_folders_to_test = []
for dirpath, _, fnames in sorted(os.walk(super_folder_to_get_all_sub_folders_from)):
    ### get only "node" folders containing image sequences - my identifier is that the hierarchy is that above this folders (which take on the name of the image itself) there is a folder specifying noise gain: ###
    if 'NoiseGain'==dirpath.split('\\')[-2].split('_')[0] and 'NoiseGain' not in dirpath.split('\\')[-1]:
        image_folders_to_test.append(dirpath)

for i in image_folders_to_test:
    print(i)


### Loop over all image folders containing image sequences and for each folder use the network to test: ###
max_total_number_of_batches = 100
for images_folder in image_folders_to_test:
    ### Get Noise Gain: ###
    noise_gain = float32(images_folder.split('\\')[-2].split('_')[1])
    clean_image_name = images_folder.split('\\')[-1]
    clean_images_folder = os.path.join(os.path.join(os.path.split(os.path.split(images_folder)[0])[0] , 'NoiseGain_0') , clean_image_name)

    #(*). Initializations:
    image_index = 1;
    flag_show_results = True;
    flag_save_video_or_images = 'images' #'images' / 'videos' / 'both'

    if flag_save_video_or_images == 'images' or flag_save_video_or_images == 'both':
        #ALL DETAILS:
        network_subfolder_in_images_folder = os.path.join(images_folder, save_Generator_filename)
        sub_folder_to_save_current_results_videos_ALL_DETAILS = os.path.join(network_subfolder_in_images_folder, 'ALL_DETAILS')
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
        sub_folder_to_save_current_results_videos_HIGHPASS = os.path.join(network_subfolder_in_images_folder, 'HIGHPASS')
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
        sub_folder_to_save_current_results_videos_DIFFERENCE = os.path.join(network_subfolder_in_images_folder, 'DIFFERENCE')
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
    number_of_time_steps_so_far = 0
    current_step = 0;
    Generator_Network = Generator_to_GPU(Generator_Network, netF, Generator_device);
    Generator_Network = Generator_to_eval_mode(Generator_Network)


    ####################################################################   Test By Getting Official Test Images + focus on saving resulting images: #####################################################################################
    ### Initialize Lists to keep track of Metrics: ###
    fake_to_average_L1_list = []
    fake_to_clean_L1_list = []
    average_to_clean_L1_list = []
    fake_to_clean_PSNR_list = []
    average_to_clean_PSNR_list = []
    fake_to_clean_SSIM_list = []
    average_to_clean_SSIM_list = []


    ### Create DataSet and DataLoader Objects from current folder: ###
    crop_size = 1000
    max_number_of_images = 1000
    def toNp(pic):
        nppic = np.array(pic)
        return nppic
    train_transform_PIL = transforms.Compose([transforms.RandomCrop(crop_size), transforms.Lambda(toNp), transforms.ToTensor(), ])
    train_dataset = ImageFolderRecursive_MaxNumberOfImages(images_folder, transform=train_transform_PIL, loader='PIL', max_number_of_images=max_number_of_images, min_crop_size=crop_size, extention='png')
    clean_dataset = ImageFolderRecursive_MaxNumberOfImages(clean_images_folder, transform=train_transform_PIL, loader='PIL', max_number_of_images=max_number_of_images, min_crop_size=crop_size, extention='png')
    batch_size = len(train_dataset)


    ###################################################################################################################################################################################################
    ########################################  Train Generator: #############################################
    torch.cuda.empty_cache()
    number_of_time_steps_from_batch_start = 0;
    averaged_image = 0;
    print('New Batch')

    ### Things to do at start (RNN Flags): ###
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
    number_of_time_steps = len(train_dataset)  #make number of time steps equal the number of images in current subfolder
    print('dataset size = ' + str(number_of_time_steps))
    for current_time_step in arange(0, number_of_time_steps):
        number_of_time_steps_from_batch_start += 1
        number_of_time_steps_so_far += 1; #global

        ### Get Original Tensors: ###
        current_clean_tensor = torch.zeros_like(train_dataset[current_time_step].unsqueeze(0).type(torch.FloatTensor))
        current_clean_tensor.copy_(clean_dataset[current_time_step].unsqueeze(0).type(torch.FloatTensor))

        current_noisy_tensor = torch.zeros_like(train_dataset[current_time_step].unsqueeze(0).type(torch.FloatTensor))
        current_noisy_tensor.copy_(train_dataset[current_time_step].unsqueeze(0).type(torch.FloatTensor))

        current_clean_tensor = current_clean_tensor.to(Generator_device);
        current_noisy_tensor = current_noisy_tensor.to(Generator_device);


        ############################################################################################################################################################################
        ### Pass input through network: ###
        current_noisy_tensor = Variable(current_noisy_tensor, requires_grad=False)
        output_tensor = Generator_Network(current_noisy_tensor, reset_hidden_states_flags_list)
        reset_hidden_states_flags_list = [0] * batch_size;  #In any case after whatever initial initialization signal -> do nothing and continue accumulating graph


        ### Pass each input through SID: ###


        ### TNR: ###
        #############################################################################################################################################################################


        ### Keep Track Of Running Average: ###
        if current_time_step == 0 and number_of_time_steps_from_batch_start == 1:  # very start of batch
            print('initialize average image with noisy tensor')
            averaged_image = torch.zeros_like(current_clean_tensor)
            noisy_original_image_tensor = noise_RGB_LinearShotNoise_Torch(current_clean_tensor, gain=noise_gain, flag_output_uint8_or_float='float')
            averaged_image.copy_(current_clean_tensor);
        else:
            total_time_steps_this_batch = number_of_time_steps_from_batch_start
            alpha = total_time_steps_this_batch / (total_time_steps_this_batch + 1)
            noisy_original_image_tensor = noise_RGB_LinearShotNoise_Torch(current_clean_tensor, gain=noise_gain, flag_output_uint8_or_float='float')
            averaged_image = (averaged_image * alpha + noisy_original_image_tensor * (1 - alpha));


        ### Get Ground-Truth & Generated Images (contingent on whether i used residual learning or not): ###
        real_images = torch.zeros_like(current_clean_tensor)
        fake_images = torch.zeros_like(output_tensor[:,0:3,:,:])
        real_images.copy_(current_clean_tensor)
        fake_images.copy_(output_tensor[:,0:3,:,:])
        if flag_residual_learning:
            if flag_learn_clean_or_running_average == 0:
                real_images -= gaussian_blur_layer(current_noisy_tensor)
            else:
                real_images = averaged_image - gaussian_blur_layer(current_noisy_tensor)
        else:
            if flag_learn_clean_or_running_average == 0:
                1;
            else:
                real_images = averaged_image;



        ### Save Losses and Metrics: ###
        #(1). L1:
        average_to_clean_L1 = L1_Loss_Function(averaged_image, current_clean_tensor).item();  # between average image and Clean
        average_to_target_L1 = L1_Loss_Function(averaged_image, real_images).item() #between average and target
        fake_to_clean_L1 = L1_Loss_Function(fake_images, current_clean_tensor).item();  # between clean image and Generated
        fake_to_average_L1 = L1_Loss_Function(fake_images, averaged_image).item();  # between average image and Generated
        fake_to_target_L1 = L1_Loss_Function(fake_images, real_images).item() #between Fake and Target
        #(2). PSNR:
        average_to_clean_PSNR = PSNR_Loss_Function(averaged_image, current_clean_tensor).item();  # between average image and Clean
        average_to_target_PSNR = PSNR_Loss_Function(averaged_image, real_images).item()  # between average and target
        fake_to_clean_PSNR = PSNR_Loss_Function(fake_images, current_clean_tensor).item();  # between clean image and Generated
        fake_to_average_PSNR = PSNR_Loss_Function(fake_images, averaged_image).item();  # between average image and Generated
        fake_to_target_PSNR = PSNR_Loss_Function(fake_images, real_images).item()  # between Fake and Target
        #(3). SSIM:
        average_to_clean_SSIM = SSIM_Loss_Function(averaged_image, current_clean_tensor).item();  # between average image and Clean
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
        for flag_show_image_or_highpass_or_difference in arange(1):

            ### Save Results: ###
            final_real_images = torch.zeros_like(real_images)
            final_fake_images = torch.zeros_like(fake_images)
            final_noisy_images = torch.zeros_like(current_noisy_tensor)

            final_real_images.copy_(real_images)
            final_fake_images.copy_(fake_images)
            final_noisy_images.copy_(current_noisy_tensor)

            if flag_residual_learning:
                if flag_show_image_or_highpass_or_difference == 0 or flag_show_image_or_highpass_or_difference == 2:  # Image
                    final_real_images += gaussian_blur_layer(current_noisy_tensor)
                    final_fake_images += gaussian_blur_layer(current_noisy_tensor)
                    final_noisy_images = current_noisy_tensor
                if flag_show_image_or_highpass_or_difference == 1:  # HighPass Details
                    final_real_images += gaussian_blur_layer(current_noisy_tensor) - gaussian_blur_layer(current_clean_tensor)
                    final_fake_images += gaussian_blur_layer(current_noisy_tensor) - gaussian_blur_layer(output_tensor[:,0:3,:,:])
                    final_noisy_images += -1*gaussian_blur_layer(current_noisy_tensor)
            else:
                if flag_show_image_or_highpass_or_difference == 1:  # HighPass Details
                    final_real_images += -1*gaussian_blur_layer(real_images)
                    final_fake_images -= gaussian_blur_layer(fake_images)
                    final_noisy_images -= gaussian_blur_layer(current_noisy_tensor)
                if flag_show_image_or_highpass_or_difference == 0 or flag_show_image_or_highpass_or_difference == 2:  # Image
                    1;


            # Get original frame numpy:
            frame_original = np.transpose(final_real_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
            # Get noisy frame numpy:
            frame_noisy = np.transpose(final_noisy_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
            # Get clean frame numpy:
            frame_clean = np.transpose(final_fake_images[0,:,:,:].squeeze().clamp(0, 1).detach().cpu().numpy().squeeze(), [1, 2, 0]) * 255.0
            # Get Concat frame numpy:
            frame_concat = np.concatenate((frame_original, frame_clean), axis=1)
            # Get Difference frame numpy:
            difference_DC_to_void_underflow = 100;
            frame_difference_original_noisy = frame_original-frame_noisy + difference_DC_to_void_underflow
            frame_difference_original_clean = frame_original-frame_clean + difference_DC_to_void_underflow
            frame_difference_noisy_clean = frame_noisy-frame_clean + difference_DC_to_void_underflow
            frame_difference_concat = np.concatenate((frame_difference_original_noisy, frame_difference_noisy_clean), axis=1)

            ### Roudn and make uint8: ###
            frame_original = np.uint8(frame_original.round())
            frame_noisy = np.uint8(frame_noisy.round())
            frame_clean = np.uint8(frame_clean.round())
            frame_concat = np.uint8(frame_concat.round())
            frame_difference_concat = np.uint8(frame_difference_concat.round())


            ### Write Seperate Frames To Folders: ###
            if flag_save_video_or_images == 'images' or flag_save_video_or_images == 'both':
                if flag_show_image_or_highpass_or_difference == 0:
                    save_image_numpy(original_video_images_folder_ALL_DETAILS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_original, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(noisy_video_images_folder_ALL_DETAILS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_noisy, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(clean_video_images_folder_ALL_DETAILS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_clean, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(Concat_video_images_folder_ALL_DETAILS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_concat, flag_convert_bgr2rgb=True, flag_scale=False)
                if flag_show_image_or_highpass_or_difference == 1:
                    save_image_numpy(original_video_images_folder_HIGHPASS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_original, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(noisy_video_images_folder_HIGHPASS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_noisy, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(clean_video_images_folder_HIGHPASS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_clean, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(Concat_video_images_folder_HIGHPASS, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_concat, flag_convert_bgr2rgb=True, flag_scale=False)
                if flag_show_image_or_highpass_or_difference == 2:
                    save_image_numpy(original_clean_video_images_folder_DIFFERENCE, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_difference_original_clean, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(noisy_clean_video_images_folder_DIFFERENCE, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_difference_noisy_clean, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(original_noisy_video_images_folder_DIFFERENCE, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_difference_original_noisy, flag_convert_bgr2rgb=True, flag_scale=False)
                    save_image_numpy(concat_video_images_folder_DIFFERENCE, filename=str(current_time_step).rjust(10, '0') + '.png', numpy_array=frame_difference_concat, flag_convert_bgr2rgb=True, flag_scale=False)
    ### END Of Generator Loss Accumulation! ###





### Plot Metrics Over Time: ###
figure(2)
plot(fake_to_clean_L1_list)
plot(average_to_clean_L1_list)
legend(['Fake-Clean L1', 'Average-Clean L1'])

figure(3)
plot(fake_to_clean_PSNR_list)
plot(average_to_clean_PSNR_list)
legend(['Fake-Clean PSNR', 'Average-Clean PSNR'])

figure(4)
plot(fake_to_clean_SSIM_list)
plot(average_to_clean_SSIM_list)
legend(['Fake-Clean SSIM', 'Average-Clean SSIM'])


###################################################################################################################################################################################################################################################################################################################################
















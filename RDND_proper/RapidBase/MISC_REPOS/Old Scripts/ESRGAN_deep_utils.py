#Imports:
#(1). Auxiliary:
from __future__ import print_function
import functools
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
import graphviz
import pydot_ng
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

#Graph Visualization:
import network_graph_visualization
from network_graph_visualization import make_dot

#ESRGAN:
from ESRGAN_utils import *
# from ESRGAN_deep_utils import *
# from ESRGAN_basic_Blocks_and_Layers import *
# from ESRGAN_Models import *
# from ESRGAN_Losses import *
# from ESRGAN_Optimizers import *
# from ESRGAN_Visualizers import *
# from ESRGAN_dataset import *
# from ESRGAN_OPT import *





########################################################################################################################################################################################################################################################################################
####################
# Auxiliaries:
####################
def cv2_to_torch(input_image):
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2);
    input_image = input_image[...,[2,1,0]]; #BGR->RGB (cv2 returns BGR by default)
    input_image = np.transpose(input_image, (2,0,1)); #[H,W,C]->[C,H,W]
    input_image = torch.from_numpy(input_image).float(); #to float32
    input_image = input_image.unsqueeze(0); #[C,H,W] -> [1,C,H,W]
    input_image = input_image.to(device); #to cpu or gpu (should this be global or explicit?)
    return input_image

def numpy_to_torch(input_image):
    #Assumes "Natural" RGB form to there's no BGR->RGB conversion, can also accept BW image!:
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2);
    input_image = np.transpose(input_image, (2, 0, 1));  # [H,W,C]->[C,H,W]
    input_image = torch.from_numpy(input_image).float();  # to float32
    input_image = input_image.unsqueeze(0);  # [C,H,W] -> [1,C,H,W]
    input_image = input_image.to(device);  # to cpu or gpu (should this be global or explicit?)
    return input_image

def torch_to_numpy(input_tensor):
    input_tensor = input_tensor.data.numpy();
    input_tensor = np.transpose(input_tensor, [0,2,3,1]);
    input_tensor = input_tensor.squeeze();
    return input_tensor;


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def get_network_output_size_given_input_size(network, input_size):
    f = network.forward(autograd.Variable(torch.Tensor(1,*input_size)));
    return int(np.prod(f.size()[1:]))

def torch_flatten(torch_variable):
    torch_variable = torch_variable.view(torch_variable.size(0), -1)
    return torch_variable

# Get needed padding to keep image size the same after a convolution layer with a certain kernel size and dilation (TODO: what about strid?1!?!)
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding
########################################################################################################################################################################################################################################################################################











########################################################################################################################################################################################################################################################################################
####################
# Save & Load Models/Checkpoint:
####################
#(1). State Dictionary;
def save_network_state_dict(network, save_full_filename):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    #transfer parameters to cpu so that we will be able to save them to disk:
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_full_filename)


def load_state_dict_to_network_from_path(network, load_path,  flag_strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module;
    network.load_state_dict(torch.load(load_path), strict=flag_strict); #this changes


def load_state_dict_to_network_robust(current_network, pretrained_state_dictionary):
    current_network_state_dictionary = current_network.state_dict();
    #(1). filter out unnecessary keys:
    pretrained_state_dictionary_filtered  = {k:v for k,v in pretrained_state_dictionary.items() if k in current_network_state_dictionary.keys()}
    #(2). overwrite entries in existing state dictionary:
    current_network_state_dictionary.update(pretrained_state_dictionary_filtered);
    #(3). load the new state dictionary to current_model:
    current_network.load_state_dict(current_network_state_dictionary);



#(2). Network:
def save_network_simple(network, save_filename):
    if isinstance(network, nn.DataParallel):
        network = network.module;
    torch.save(network, save_filename);

def load_network_simple(load_path):
    return torch.load(load_path);

def load_network_robust(current_network, load_path):
    loaded_file = torch.load(load_path);
    if type(loaded_file) == collections.OrderedDict:
        load_state_dict_to_network_robust(current_network, loaded_file);
        return current_network
    else:
        return loaded_file


# #Example Of Use:
# folder_name = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS';
# complete_model_save_file_name = 'complete_model.pt';
# state_dictionary_save_file_name = 'state_dictionary.pth';
#
# complete_model_save_full_file_name = os.path.join(folder_name,complete_model_save_file_name);
# state_dictionary_save_full_file_name = os.path.join(folder_name,state_dictionary_save_file_name);
#
# current_model = torchvision.models.resnet18(True);
# current_model_state_dictionary = current_model.state_dict()
#
# torch.save(current_model,complete_model_save_full_file_name)
# torch.save(current_model_state_dictionary, state_dictionary_save_full_file_name)
#
# loaded_model_from_complete_model = load_network_robust(None, complete_model_save_full_file_name)
# loaded_model_from_state_dictionary = load_network_robust(current_model,state_dictionary_save_full_file_name)
# loaded_model_from_complete_model == current_model
# loaded_model_from_state_dictionary == current_model
#
# bla = torch.load(complete_model_save_full_file_name);
# bla == current_model


#(3. General/Custom Checkpoint:
def save_checkpoint_specific(network, optimizer=None, OPT=None, iteration_number=None, flag_is_best_model=False, full_filename='checkpoint.pth'):
    checkpoint_dictionary = {'model': network, 'state_dict': network.state_dict(), 'optimizer':optimizer, 'OPT':OPT, 'iteration': iteration_number}
    torch.save(checkpoint_dictionary, full_filename)

    if flag_is_best_model:
        shutil.copyfile(filename, 'model_best.pth.tar') #shutil.copyfile


def save_checkpoint_general(network, full_filename = 'checkpoint.pth', flag_save_network_state_dict_or_both='both', flag_is_best_model=False, **kwargs):
    if flag_save_network_state_dict_or_both == 'network':
        checkpoint_dictionary = {'model': network}
    elif flag_save_network_state_dict_or_both == 'state_dict':
        checkpoint_dictionary = {'state_dict': network.state_dict()}
    elif flag_save_network_state_dict_or_both == 'both':
        checkpoint_dictionary = {'model': network, 'state_dict': network.state_dict}

    for key in kwargs.keys():
        checkpoint_dictionary[key] = kwargs[key];

    torch.save(checkpoint_dictionary, full_filename);

    if flag_is_best_model:
        shutil.copyfile(full_filename, 'model_best.pth.tar');



def load_checkpoint(checkpoint_filename):
    if os.path.isfile(checkpoint_filename):
        print(" --> loading checkpoint '{}'".format(checkpoint_filename));
        checkpoint = torch.load(checkpoint_filename);
    return checkpoint



# #Use Case of the above save/load functions:
# #Parameters:
# blabla = torchvision.models.vgg19(pretrained=True)
# bla = torchvision.models.vgg11();
# base_path = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS'
# model_name = 'bla_model.pt'; #what's the difference between .pt & .pth  ????
# full_filename = os.path.join(base_path,model_name)
#
# ### Within Script: ###
# #Save & Load Entire Model within script:
# torch.save(bla, full_filename)
# bla2 = torch.load(full_filename)
#
# #Save & Load State Dictionary within script:
# bla_state_dictionary = bla.state_dict()
# for key, param in bla_state_dictionary.items():
#     bla_state_dictionary[key] = param.cpu();
# torch.save(bla_state_dictionary, full_model_name)
# bla2 = bla.load_state_dict(full_filename, strict=True) #State-Dict
#
# ### Using Function Wrappers: ###
# #Save & Load using checkpoint:
# save_checkpoint(bla, filename=full_filename);
# network_model, optimizer, checkpoint_epoch = load_checkpoint(full_filename)
# #Save & Load using network state dictionaries:
# save_network_state_dict(network, full_filename);
# bla.load_state_dict(torch.load(full_filename), strict=True)
# bla = load_state_dict_to_network(full_filename, bla) #Doesn't Work
# #Save & Load using entire network save&load:
# save_network_simple(bla,full_filename)
# bla2 = load_network_simple(full_filename)


# model = torch.load('./model_resnet50.pth.tar')
# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
#     transforms.Scale(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize
# ])
#
# img = Image.open(IMG_URL)
# img_tensor = preprocess(img)
# img_tensor.unsqueeze_(0)
# output = model(Variable(img_tensor))
########################################################################################################################################################################################################################################################################################






###################################################################################################
#### Revisit BatchNorm After Model Learning: ###
import torch
import torch.nn as nn


## A module dedicated to computing the true population statistics
## after the training is done following the original Batch norm paper

# Example of usage:

# Note: you might want to traverse the dataset a couple of times to get
# a better estimate of the population statistics
# Make sure your trainloader has shuffle=True and drop_last=True

# net.apply(adjust_bn_layers_to_compute_population_stats)
# for i in range(10):
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             _ = net(inputs.cuda())
# net.apply(restore_original_settings_of_bn_layers)


# Why this works --
# https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm1d
# if you set momentum property in batchnorm layer, it will
# compute cumulutive average or just simple average of observed
# values

def adjust_bn_layers_to_compute_population_stats_old(module):
    if isinstance(module, nn.BatchNorm2d):
        # Removing the stats computed using exponential running average
        # and resetting count
        module.reset_running_stats()

        # Doing this so that we can restore it later
        module._old_momentum = module.momentum

        # Setting the momentum to none makes the statistics computation go to cumulative moving average mode....i don't know if that's the right thing to do
        module.momentum = None

        #Note: now the batchnorm2d acts the following way: x_new = (1-momentum)*x_estimated_static + momentum*x_observed_value
        #the suggestion most people have is to set the momentum to something higher than the original 0.1, so i'm setting it to 0.5, which supposedly makes the network prefer curent value....so i don't really understand why that's the answer....
        #it would make sense if you just want to make the situation better when evaluating because it would be closer the .train() mode but if you want to collect statistics?
        # module.momentum = 0.5;

        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats

        module.training = True
        module.track_running_stats = True




def adjust_bn_layers_to_compute_population_stats(module):
    if isinstance(module, nn.BatchNorm2d):
        # Removing the stats computed using exponential running average
        # and resetting count
        module.reset_running_stats()

        # Doing this so that we can restore it later
        module._old_momentum = module.momentum

        #Note: now the batchnorm2d acts the following way: x_new = (1-momentum)*x_estimated_static + momentum*x_observed_value
        #the suggestion most people have is to set the momentum to something higher than the original 0.1, so i'm setting it to 0.5, which supposedly makes the network prefer curent value....so i don't really understand why that's the answer....
        #it would make sense if you just want to make the situation better when evaluating because it would be closer the .train() mode but if you want to collect statistics?
        module.momentum = 0.5;

        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats

        module.training = True # net.eval() makes this False, net.train() makes this True
        module.track_running_stats = True #net.eval() and net.train() doesn't change this


def restore_original_settings_of_bn_layers(module):
    if isinstance(module, nn.BatchNorm2d):
        # Restoring old settings
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats


def set_bn_layers_momentum(value=0.5):
    #TODO: understand how i can use this and pass in variables...as it is now the vanilla form of calling it doesn't work: network.apply(set_bn_layers_momentum(0.5)) - doesn't work!
    def set_bn_layers_with_specific_value(module):
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = value;
    return set_bn_layers_with_specific_value;
##########################################################################################################################################






########################################################################################################################################################################################################################################################################################
####################
# Get Network Layers:
####################
# Get Network Layers:
def get_network_model_up_to_block_indices(network, block_indices):
    1;

def get_network_model_up_to_layer_name(network, layer_name):
    1;

def get_network_layer_using_block_indices(network, block_indices):
    1;

def get_network_layer_using_layer_name(network, layer_name):
    1;

def get_network_model_up_to_layer_index(network, layer_index):
    #TODO: change this to accomodate block_indices and .features attributes (if hasattr(network, features) -> blablabla), if block_indices!=None --> get_network_layer_using_block_hierarcy(model,block_indices)
    # return nn.Sequential(*list(network.children())[layer_index]);
    return nn.Sequential(*get_network_layers_list_flat(network)[:layer_index]) #if layer_index is larger than number of layers no error pops up - we simply return the entire model

def get_network_model_up_to_layer_index_from_end(network, layer_index_from_end):
    return nn.Sequential(*list(network.children())[:-layer_index_from_end]);

def get_network_layer(network, layer_index):
    return get_network_layers_list_flat(network)[layer_index];

def flatten_list(input_list):
    # return [item for sublist in input_list for item in sublist]
    # return list(itertools.chain.from_iterable(input_list))
    total_list = [];
    def get_merged_list(current_list):
        if type(current_list) == list:
            for element in current_list:
                if type(element) == list:
                    get_merged_list(element);
                else:
                    total_list.append(element)

    get_merged_list(input_list)
    return total_list;

def get_network_layers_list(network):
    return list(network.children())

def get_network_layers_list_flat(network,verbose=0):
    # return flatten_list(list(network.children()))
    all_layers = []
    if list(network.children()) == []:
        all_layers = [network];

    def get_layers(network):
        for layer in network.children():
            if hasattr(layer, 'children'):  # if sequential layer, apply recursively to layers in sequential layer
                get_layers(layer)
            else:
                if verbose: print('no children')
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)
                if verbose: print(type(layer));

    get_layers(network)
    if verbose:
        for layer in all_layers:
            print(layer);
        len(all_layers)
    return all_layers;


# #TODO: still doesn't work
# def get_network_layers_list_flat_test(network,verbose=0):
#     # return flatten_list(list(network.children()))
#     all_layers = []
#     def get_layers(network):
#             try:  # if sequential layer, apply recursively to layers in sequential layer
#                 get_layers(network.children())
#             except:  # if leaf node, add it to list
#                 all_layers.append(network)
#     get_layers(network)
#     return all_layers;


def get_number_of_conv_layers(network):
    counter = 0;
    for current_module in get_network_layers_list_flat(network):
        if isinstance(current_module, nn.Conv2d):
            counter += 1;
    return counter;

def get_network_named_layers_list_flat(network, verbose=0):
    all_layers = []
    all_layers_names = [] #names hierarchies
    def get_layers(network, pre_string=''):
        for layer in network.named_children():
            if hasattr(layer[1], 'named_children'):  # if sequential layer, apply recursively to layers in sequential layer
                get_layers(layer[1], pre_string + '/' + layer[0])
            else:
                if verbose: print('no children')
            if list(layer[1].named_children()) == []:  # if leaf node, add it to list
                all_layers.append(layer[1])
                all_layers_names.append(pre_string + '/' + layer[0])
                if verbose: print(type(layer));

    get_layers(network)
    if verbose:
        for layer in all_layers:
            print(layer);
        len(all_layers)
    for i,layer_name in enumerate(all_layers_names):
        all_layers_names[i] = layer_name[1:]
    return all_layers, all_layers_names;



def get_network_variable_size_through_the_layers(network, input_variable):
    # Assuming the flow through the model is entirely known through the model.children() property.
    # Also probably assuming the network is built as convolutional/features part and classifier head.
    x = input_variable
    for i, layer in enumerate(get_network_layers_list_flat(pretrained_model)):
        print('index ' + str(i) + ',input to layer shape: ' + str(x.shape) + ', ' + str(layer))
        if 'Linear' in str(layer):
            x = x.view(x.size(0), -1)
        x = layer(x);
    print('index ' + str(i) + ',final shape: ' + str(x.shape))
########################################################################################################################################################################################################################################################################################






########################################################################################################################################################################################################################################################################################
####################
# Network Summary, Description, Graph:
####################
# Network String Description:
def get_network_description(network):
    if isinstance(network, nn.DataParallel):
        network = network.module
    network_string_description = str(network)
    number_of_parameters = sum(map(lambda x: x.numel(), network.parameters()))
    return network_string_description, number_of_parameters

def print_network_description(network):
    network_string_description, number_of_parameters = get_network_description(network)
    print(network_string_description);
    print('\n')
    print('Number of parameters in G: {:,d}'.format(number_of_parameters))



#one needs to add the dot.exe executable to environment variables. to do that you need to locate the graphviz folder and add it:
import os
os.environ["PATH"] += os.pathsep + 'C:/Users\dkarl\AppData\Local\conda\conda\envs\dudy_test\Library/bin/graphviz/'
#TODO: try other implementations of make_dot which can make things prettier and easier to read
# Print Network Graph to PDF file which is automatically opened:
def print_network_graph_to_pdf(network, input_size, filename='Network_Graph', directory=path_get_current_working_directory()):
    inputs = torch.randn(1,*input_size);
    outputs = network(Variable(inputs));
    network_graph = make_dot(outputs, params=dict(network.named_parameters()))
    network_graph.view(filename,directory);


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
# input_size = (1,28,28)
# network = Net()
#
# input_size = (3,224,224)
# network = torchvision.models.vgg19_bn()

#Print Network Summary (KERAS style):
def print_network_summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
# print_network_summary(torchvision.models.resnet18(True))
########################################################################################################################################################################################################################################################################################










########################################################################################################################################################################################################################################################################################
####################
# Gradients:
####################
def enable_gradients(network):
    for p in network.parameters():
        p.requires_grad = True
def unfreeze_gradients(network):
    for p in network.parameters():
        p.requires_grad = True
    return network
def disable_gradients(network):
    for p in network.parameters():
        p.requires_grad = False;
def freeze_gradients(network):
    for p in network.parameters():
        p.requires_grad = False;
    return network

def freeze_gradients_up_to_layer_index(network, layer_index):
    counter = 0;
    for child in network.children():
        if counter < layer_index:
            for param in child.parameters():
                param.requires_grad = False;
        counter += 1;


def plot_grad_flow(network):
    #Use this function AFTER doing a .backward() which updates the .grad property of the layers
    average_abs_gradients_list = []
    layers_list = []
    for parameter_name, parameter in network.named_parameters():
        if(parameter.requires_grad) and ("bias" not in parameter_name): #we usually don't really care about the bias term
            layers_list.append(parameter_name)
            average_abs_gradients_list.append(parameter.grad.abs().mean())
    plt.plot(average_abs_gradients_list, alpha=0.3, color="b")
    plt.hlines(0, 0, len(average_abs_gradients_list)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(average_abs_gradients_list), 1), layers_list, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(average_abs_gradients_list))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
# #Use Example:
# my_loss.backward();
# plot_grad_flow(network)


# #A way of getting middle layer output from the forward hook (without the need for a second, seperate forward pass):
# network = torchvision.models.vgg19();
# inputs = torch.randn(1,3,244,244);
# layer_index = 5;
# global_outputs_hooked = [];
# def hook(module, input, output):
#     global_outputs_hooked.append(output);
# get_network_layer(network,layer_index).register_forward_hook(hook);
# output1 = network(inputs);
# output2 = network(inputs);
# print(outputs);
####################################################################################################################################################################################################################################################################################################################################################################################################################################








#########################################################################################################################################
##############            CUDA Auxiliaries:           ##################
#########################################################################################################################################
def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__,
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
												   type(obj.data).__name__,
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "",
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass
	print("Total size:", total_size)
########################################################################################################################################









#########################################################################################################################################
##############            WEIGHT INITIALIZATIONS:           ##################
#########################################################################################################################################
########################################
# Weight Initializations:
########################################
def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))






#########################################################################################################################################
##############            NETWORK VISUALIZAITONS:           ##################
#########################################################################################################################################
#(1). Simple Layer Output Visualizations:
def plot_layer_outputs(network, input_variable, layer_index, filter_indices=None, crop_percent=0.8, flag_colorbar=1,
                       delta_subplots_spacing=0.1):
    #TODO: Switch to hook+.forward mode!!!!
    network_model_up_to_chosen_layer = get_network_model_up_to_layer_index(network, layer_index);
    output = network_model_up_to_chosen_layer(input_variable);
    output_numpy = output.numpy(); #Be Careful - can't the .detach() cause problems?
    output_numpy = output_numpy[0];  # Assuming i only input 1 input variable and this squeezes the batch_size dimension which would be exactly 1
    #The torch output is of form: [N,C,H,W] so i don't need to transpose and i can simply send it to plot_multiple_images as is, and it gets [C,H,W] and so it plots each channel as grayscale
    if filter_indices == None:
        filter_indices = arange(length(output_numpy));
    filter_indices = list(filter_indices)
    super_title = 'Layer ' + str(layer_index) + ' Outputs, Layer Name: ' + str(network_model_up_to_chosen_layer[-1]);
    titles_string_list = str(list(filter_indices))[1:-1].split(' ')
    plot_multiple_images(output_numpy, filter_indices, crop_percent, delta_subplots_spacing, flag_colorbar, super_title, titles_string_list)



# #Examples Of Use:
# image_directory = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
# image_name = 'cat_dog.png'
# full_image_path = os.path.join(image_directory,image_name)
# input_image = read_image_cv2(full_image_path);
# input_image = preprocess_image(input_image, resize_im=True)
# input_image.shape
#
# pretrained_model = torchvision.models.vgg16(True);
# layer_index = 7;
# pretrained_model_up_to_chosen_layer  = get_network_model_up_to_layer_index(pretrained_model, layer_index)
# pretrained_model_chosen_layer_output = pretrained_model_up_to_chosen_layer(input_image);
# pretrained_model_chosen_layer_output.shape
# bla = pretrained_model_chosen_layer_output.detach().numpy()
# bla = bla[0]
# bla.shape
# plot_images_mat(bla,list(arange(10)))
# plot_layer_outputs(pretrained_model, input_image, layer_index=7, filter_indices=arange(10), delta_subplots_spacing=0.05)




def plot_network_conv_filters(network, number_of_conv_layers):
    1;








#########################################################################################################################################
##############            NETWORK ARCHITECTURE AUGMENTATIONS:           ##################
#########################################################################################################################################
#TODO LIST:
#(1). Add deformable convolutions instead of current convolutions WHERE WANTED
#(2). Change from BatchNorm to SynchronizedBatchNorm
#(3). Change from BatchNorm to InstanceNorm / GroupNorm
#(4). Change dilation rate
#(5). Change from full 3x3 (for example) convolution to seperable convolution
#(6). Insert Dropout2d Layers where wanted
#(7). Make the "evil twin" network of a CNN -> a DCNN (Deconvolutional Neural Network)



































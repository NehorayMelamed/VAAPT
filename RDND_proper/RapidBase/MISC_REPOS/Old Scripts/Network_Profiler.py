#Imports:
#(1). Auxiliary:
from __future__ import print_function
import PIL
import argparse
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
import ESRGAN_dataset
import ESRGAN_Visualizers
import ESRGAN_Optimizers
import ESRGAN_Losses
import ESRGAN_deep_utils
import ESRGAN_utils
import ESRGAN_Models
import ESRGAN_basic_Blocks_and_Layers
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
import argparse
import torch
import torch.nn as nn
import numpy as np
import tensorboardX
from tensorboardX import SummaryWriter

import network_graph_visualization
from network_graph_visualization import *

SIMPLE_ACTIVATIONS = [nn.ReLU,nn.ReLU6,nn.LeakyReLU,nn.PReLU,nn.RReLU,nn.modules.activation.Sigmoid,nn.modules.activation.ELU,nn.modules.activation.SELU,nn.modules.activation.Tanh,nn.modules.activation.Hardtanh]
SOFTMAX_ACTIVATIONS = [nn.Softmax,nn.Softmax2d,nn.Softmin,nn.Softplus,nn.Softsign]
LOG_SOFTMAX_ACTIVATIONS = [nn.LogSoftmax]
LOG_ACTIVATIONS = [nn.LogSigmoid]
CONV_LAYERS = [nn.Conv2d,nn.Conv1d,nn.Conv3d]
CONV_TRANSPOSE_LAYERS = [nn.ConvTranspose1d,nn.ConvTranspose2d,nn.ConvTranspose3d]
MAX_POOLING_LAYERS = [nn.MaxPool2d,nn.MaxPool1d,nn.MaxPool3d]
AVG_POOLING_LAYERS = [nn.AvgPool1d,nn.AvgPool2d,nn.AvgPool3d]
ADAPTIVE_AVG_POOLING_LAYERS = [nn.AdaptiveAvgPool1d,nn.AdaptiveAvgPool2d,nn.AdaptiveAvgPool3d]
ADAPTIVE_MAX_POOLING_LAYERS = [nn.AdaptiveMaxPool1d,nn.AdaptiveMaxPool2d,nn.AdaptiveMaxPool3d]
BATCH_NORM_LAYERS = [nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d]
INSTANCE_NORM_LAYERS = [nn.InstanceNorm1d,nn.InstanceNorm2d,nn.InstanceNorm3d]
GROUP_NORM_LAYERS = [nn.GroupNorm]
LSTM_LAYERS = [nn.LSTM,nn.LSTMCell]
GRU_LAYERS = [nn.GRU,nn.GRUCell]
LINEAR_DENSE_LAYERS = [nn.Linear]
UPSAMPLING_LAYERS = [nn.Bilinear,nn.Upsample,nn.UpsamplingBilinear2d,nn.UpsamplingNearest2d]
EMBEDDING_LAYERS = [nn.Embedding]
DROPOUT_LAYERS = [nn.Dropout,nn.Dropout2d,nn.Dropout3d,nn.AlphaDropout,nn.FeatureAlphaDropout]
# CONV_LSTM_LAYERS = [ConvLSTMCell]



####################################################################################################################################################################################
# layer = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),stride=1,padding=0,dilation=1,groups=1,bias=True)
# layer.register_forward_hook(count_conv2d)
# input_tensor = torch.Tensor(randn(1,3,32,32))
# output_tensor = layer(input_tensor)




#####################################################################################################################################################################################
### Attribute PlayGround: ###
def set_attribute(module, attribute_name, attribute_value):
    setattr(module,attribute_name, attribute_value)
    return module;

def add_attribute(module, attribute_name, attribute_value):
    if hasattr(module,attribute_name) == False:
        setattr(module,attribute_name,attribute_value)
    else:
        #Get old attribute value:
        previous_value = getattr(module,attribute_name)
        #Do Calculation:
        new_value = previous_value + attribute_value
        #Set new attribute value:
        setattr(module, attribute_name, new_value)
    return module;

def multiply_attribute(module,attribute_name, attribute_value):
    if hasattr(module,attribute_name) == False:
        setattr(module,attribute_name,attribute_value)
    else:
        #Get old attribute value:
        previous_value = getattr(module,attribute_name)
        #Do Calculation:
        new_value = previous_value * attribute_value
        #Set new attribute value:
        setattr(module, attribute_name, new_value)
    return module;

def max_attribute(module,attribute_name,attribute_value):
    if hasattr(module,attribute_name) == False:
        setattr(module,attribute_name,attribute_value)
    else:
        #Get old attribute value:
        previous_value = getattr(module,attribute_name)
        #Do Calculation:
        new_value = max(previous_value,attribute_value)
        #Set new attribute value:
        setattr(module, attribute_name, new_value)
    return module;
######################################################################################################################



######################################################################################################################
############### Playing Around: ##################
height = 28;
width = 28;
batch_size = 1;
input_tensor = torch.Tensor(randn(batch_size,3,height,width))

#Single Conv:
input_size = (3,48,48)
input_tensor = torch.Tensor(randn(1,*input_size))
conv1 = nn.Conv2d(3,64,kernel_size=3, stride=2, padding=1, bias=False);
preprocess_network(conv1)
output_tensor = conv1(input_tensor)

#Multiple Convs:
conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
conv2 = nn.Conv2d(64,128, kernel_size=5, stride=1, padding=5//2, bias=False);
network = nn.Sequential(conv1,conv2)
preprocess_network(network)
input_size = (3,48,48)
input_tensor = torch.Tensor(randn(1,*input_size))
output_tensor = network(input_tensor)
postprocess_network(network)
#"Regular" Keras style network summary:
print_network_summary(network, input_size, batch_size=1, device="cpu")

#VGG:
network = torchvision.models.resnet18()
# network = torchvision.models.vgg19();
preprocess_network(network)
input_size = (3,224,224);
input_tensor = torch.Tensor(randn(1,*input_size))
output_tensor = network(input_tensor)
postprocess_network(network)


# #ResNet:
# input_size = (3,224,224)
# network = torchvision.models.resnet18()
# print_network_graph_to_pdf(network, input_size)

#One-Shot Encoder:
network = EncoderCell()
network = preprocess_network(network)
input_size = (6,224,224)
input_tensor = torch.Tensor(randn(1,*input_size))
output_tensor = network(input_tensor)
postprocess_network(network)


#ConvLSTMCell:
encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
               Variable(torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True))
my_layer = ConvLSTMCell(
                        64,
                        256,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        hidden_kernel_size=1,
                        bias=False)
my_layer.register_forward_hook(count_conv_lstm_2d);
initial_conv_output = conv1(input_tensor);
output1, output2 = my_layer(initial_conv_output,encoder_h_1)

bla = nn.Sequential(conv1,my_layer)
bli = torchvision.models.resnet18()
bla_list_flat = get_network_layers_list_flat(bla)
bla_list = list(bla.children())
bli_list = list(bli.children());
bli_list_flat = get_network_layers_list_flat(bli)
bli_list_4_list = list(bli_list[4].children())
######################################################################################################################

def get_object_parents(object):
    import inspect
    return inspect.getmro(object.__class__)

def pad_string(input_string, string_length=10):
    return input_string.ljust(string_length)

def create_append_list(output_tensor, list_attribute_name, input_tensor, new_value):
    # if hasattr(output_tensor,list_attribute_name)==False:
    #     input_list = []
    # else:
    #     input_list = getattr(output_tensor,list_attribute_name)

    input_list = []
    if hasattr(input_tensor,list_attribute_name)==True:
        input_list += getattr(input_tensor,list_attribute_name)

    if type(new_value) == list:
        output_tensor = set_attribute(output_tensor,list_attribute_name,input_list + new_value)
    else:
        output_tensor = set_attribute(output_tensor,list_attribute_name,input_list + [new_value])
    
    return output_tensor


def preprocess_network(network):
    #Get flat network layers list:
    layers_flat_list = get_network_layers_list_flat(network);
    children_list = list(network.children())

    #Hook individual Basic "leaf" layers with the appropriate hooks:
    for module in layers_flat_list:
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(count_conv2d)
        elif isinstance(module, nn.BatchNorm2d):
            module.register_forward_hook(count_bn2d)
        elif isinstance(module, nn.ReLU):
            module.register_forward_hook(count_relu)
        elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            module.register_forward_hook(count_maxpool)
        elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            module.register_forward_hook(count_avgpool)
        elif isinstance(module, nn.Linear):
            module.register_forward_hook(count_linear)
        elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            print("Not implemented for ", module)

    # #Loop over children (which can be containers/wrappers/custom-modules etc') and assign/perform what's needed to make possible general mapping of network:
    # for module in children_list:
    #     if isinstance(module, nn.Sequential):
    #         get_and_assign_atttributes_by_looping_over_sequential_layers(module); #TODO: impelement
    #     if isinstance(module, ConvLSTMCell):
    #         module.register_forward_hook(count_conv_lstm_2d)
    #     #TODO: continue to implement

    return network;



def postprocess_network_recursive(network):
    total_area = '?'
    recursion_depth = 0

    def print_recursive(network, recursion_depth):
        # Initialize Sum parameters:
        total_number_of_parameters = 0
        total_number_of_operations = 0;
        total_number_of_line_buffers = 0

        # Loop over network children and get needed stats:
        children_list = list(network.children());
        print_string = ''
        for i, module in enumerate(children_list):
            # If this is not a "Basic" module but a Sequential or Module then use the function on it:
            if nn.Sequential in get_object_parents(module):
                total_number_of_parameters, total_number_of_operations, total_number_of_line_buffers, print_string = print_recursive(module, recursion_depth + 1)

            # Get current module stats:
            current_number_of_parameters = module.number_of_parameters
            current_number_of_ops = module.number_of_ops
            current_number_of_line_buffers = module.number_of_line_buffers
            current_line_buffer_length = module.line_buffer_length
            current_line_buffer_depth = module.line_buffer_depth

            # Get current moduel string:
            tab_string = '  '
            print_string += tab_string * recursion_depth
            print_string += '({0:>4}). {1:>10} ||   Parameters: {2:>15,}   ||   Ops: {3:>15,}   ||   LB#: {4:>5,}   ||   LB_Length: {5:>5,}   ||   LB_Depth: {6:>5,}'. \format(i,
                                       str(module).split('(')[0],
                                       current_number_of_parameters,
                                       current_number_of_ops,
                                       current_number_of_line_buffers,
                                       current_line_buffer_length,
                                       current_line_buffer_depth) + '\n'

            # Sum Statistics:
            total_number_of_parameters += current_number_of_parameters
            total_number_of_operations += current_number_of_ops
            total_number_of_line_buffers += current_number_of_line_buffers


        return total_number_of_parameters, total_number_of_operations, total_number_of_line_buffers, print_string

    # Print final string:
    total_number_of_parameters, total_number_of_operations, total_number_of_line_buffers, print_string = print_recursive(networ k,0)
    print(print_string)

    # Print Sum Statistics:
    sum_print_string = '({0:>4}). {1:>10} ||   Parameters: {2:>15,}   ||   Ops: {3:>15,}   ||   LB#: {4:>5,} '. \
                           format('Sum',
                                  'Sum:',
                                  total_number_of_parameters,
                                  total_number_of_operations,
                                  total_number_of_line_buffers) + '\n'
    print(sum_print_string)



def postprocess_network(network):


    total_area = '?'
    recursion_depth = 0

    # Initialize Sum parameters:
    total_number_of_parameters = 0
    total_number_of_operations = 0;
    total_number_of_line_buffers = 0

    #Loop over network children and get needed stats:
    # children_list = list(network.children());
    children_list = get_network_layers_list_flat(network)
    print_string = ''
    for i,module in enumerate(children_list):
        if type(module) not in DROPOUT_LAYERS:
            #Get current module stats:
            current_number_of_parameters = module.number_of_parameters
            current_number_of_ops = module.number_of_ops
            current_number_of_line_buffers = module.number_of_line_buffers
            current_line_buffer_length = module.line_buffer_length
            current_line_buffer_depth = module.line_buffer_depth

            #Get current moduel string:
            tab_string = '  '
            print_string += tab_string * recursion_depth
            print_string += '({0:>4}). {1:>15} ||   Parameters: {2:>15,}   ||   Ops: {3:>15,}   ||   LB#: {4:>5,}   ||   LB_Length: {5:>5,}   ||   LB_Depth: {6:>5,}'.\
                                format(i,
                                       str(module).split('(')[0],
                                       current_number_of_parameters,
                                       current_number_of_ops,
                                       current_number_of_line_buffers,
                                       current_line_buffer_length,
                                       current_line_buffer_depth) + '\n'

            #Sum Statistics:
            total_number_of_parameters += current_number_of_parameters
            total_number_of_operations += current_number_of_ops
            total_number_of_line_buffers += current_number_of_line_buffers



    #Print final string:
    print(print_string)

    #Print Sum Statistics:
    sum_print_string = '({0:>4}). {1:>10} ||   Parameters: {2:>15,}   ||   Ops: {3:>15,}   ||   LB#: {4:>5,} '.\
                            format('Sum',
                                   'Sum:',
                                   total_number_of_parameters,
                                   total_number_of_operations,
                                   total_number_of_line_buffers) + '\n'
    print(sum_print_string)

# postprocess_network(network)

def count_conv_lstm_2d(module, input_tensors, output_tensors):
    # Get input tensor (returns [N,C,H,W]):
    # input_tensors = input_tensors[0]
    # print(input_tensors);
    # print(module)
    # print(output_tensors)
    module = set_attribute(module, 'number_of_ops', module.conv_ih.number_of_ops + module.conv_hh.number_of_ops) #Convolutions
    module = add_attribute(module, 'number_of_ops', module.conv_ih.last_output.cpu().numpy().size) #Activations
    module = add_attribute(module, 'number_of_ops', input_tensor[2].cpu().numpy().size*3) #(forgetgate * cx) + (ingate * cellgate) - two multiplications plus one addition
    module = add_attribute(module, 'number_of_ops', input_tensor[2].cpu().numpy().size) #F.tanh(cy)
    module = add_attribute(module, 'number_of_ops', input_tensor[2].cpu().numpy().size) #outgate * F.tanh(cy) - multiplication


#TODO: add some default input_tensor-s ....to be able to still use this way of thinking with hooks but also not have to define an input_tensor each time
def count_conv2d(module, input_tensor, output_tensor): #TODO: add more convolution layers....
    # Get input tensor (returns [N,C,H,W]):
    input_tensor = input_tensor[0]

    # Count number of Distinct input and output channels (if there are groups the the effective number of input and output channels is lower because the groups don't "Talk to each other":
    effective_number_of_input_channels = module.in_channels // module.groups
    effective_number_of_output_channels = module.out_channels // module.groups #currently, to make things easier we don't directly use this but simply use output_tensor...whose number of channels is effective_number_of_output_channels
                                                                               #this is the same reason we don't explicitely use stride here... because all the information we need is compounded into output_tensor.shape
    kernel_height, kernel_width = module.kernel_size
    batch_size = input_tensor.size()[0] #

    ### Number of Parameters: ###
    number_of_parameters = np.prod(module.cpu().weight.data.numpy().shape)

    ### Number of ops: ###
    # Ops per output pixel
    kernel_mul = kernel_height * kernel_width * effective_number_of_input_channels
    kernel_add = kernel_height * kernel_width * effective_number_of_input_channels - 1
    bias_ops = 1 if module.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops
    # Total number of ops
    num_out_elements = output_tensor.numel() # output_tensor.numel() = batch_size X number_of_output_channels X Height X Width
    number_of_ops = num_out_elements * ops
    number_of_ops = int(number_of_ops);

    ### Needed Support: ###
    needed_support_height = kernel_height * module.dilation[0] #TODO: make sure the index 0 is height and index 1 is width
    needed_support_width = kernel_width * module.dilation[1]

    ### Needed Line Buffers: ###
    number_of_line_buffers = kernel_height - 1;
    line_buffer_length = input_tensor.shape[3]; #input tensor width
    line_buffer_depth = effective_number_of_input_channels;

    ### Calculate Memory Needed: ###

    ### Accumulate Attributes Values: ###
    module = add_attribute(module, 'number_of_ops', number_of_ops)
    module = set_attribute(module, 'number_of_parameters', number_of_parameters)
    module = max_attribute(module, 'number_of_line_buffers', number_of_line_buffers)
    module = max_attribute(module, 'line_buffer_length', line_buffer_length)
    module = max_attribute(module, 'line_buffer_depth', line_buffer_depth);
    module = set_attribute(module, 'last_output', output_tensor);
    module = set_attribute(module, 'last_input', input_tensor)

    ### Make Conversions To Chip Size: ###


    ### Assign to tensor Attributes List: ###
    #(1). Count number of layers basically
    if hasattr(input_tensor,'counter')==False:
        input_tensor = set_attribute(input_tensor, 'counter', 1)
    output_tensor = set_attribute(output_tensor, 'counter', getattr(input_tensor,'counter')+1)
    #(2). Assign List To Tensor:
    output_tensor = create_append_list(output_tensor, 'number_of_ops', input_tensor, number_of_ops)
    output_tensor = create_append_list(output_tensor, 'number_of_parameters', input_tensor, number_of_parameters)
    output_tensor = create_append_list(output_tensor, 'number_of_line_buffers', input_tensor, number_of_line_buffers)
    output_tensor = create_append_list(output_tensor, 'line_buffer_length', input_tensor, line_buffer_length)
    output_tensor = create_append_list(output_tensor, 'line_buffer_depth', input_tensor, line_buffer_depth)
    
    
    
    # Auxiliary Printing:
    # print('count_conv2d: ' + str(number_of_ops))



def count_bn2d(module, input_tensor, output_tensor):
    # Get input tensor (returns [N,C,H,W]):
    input_tensor = input_tensor[0]

    # Count number of ops (each input element has to be normalized by (x-mean)/std)
    input_tensor_number_of_elements = input_tensor.numel()
    total_sub = input_tensor_number_of_elements
    total_div = input_tensor_number_of_elements
    number_of_ops = total_sub + total_div
    number_of_ops = int(number_of_ops)

    #More parameters:
    number_of_parameters = 0
    number_of_line_buffers = 1
    line_buffer_length = 0
    line_buffer_depth = 0

    module = add_attribute(module, 'number_of_ops', number_of_ops)
    module = set_attribute(module, 'number_of_parameters', number_of_parameters)
    module = max_attribute(module, 'number_of_line_buffers', number_of_line_buffers)
    module = max_attribute(module, 'line_buffer_length', line_buffer_length)
    module = max_attribute(module, 'line_buffer_depth', line_buffer_depth);
    module = set_attribute(module, 'last_output', output_tensor);
    module = set_attribute(module, 'last_input', input_tensor)

    if hasattr(input_tensor,'counter')==False:
        input_tensor = set_attribute(input_tensor, 'counter', 1)
    output_tensor = set_attribute(output_tensor, 'counter', getattr(input_tensor,'counter')+1)

    output_tensor = create_append_list(output_tensor, 'number_of_ops', input_tensor, number_of_ops)
    output_tensor = create_append_list(output_tensor, 'number_of_parameters', input_tensor, number_of_parameters)
    output_tensor = create_append_list(output_tensor, 'number_of_line_buffers', input_tensor, number_of_line_buffers)
    output_tensor = create_append_list(output_tensor, 'line_buffer_length', input_tensor, line_buffer_length)
    output_tensor = create_append_list(output_tensor, 'line_buffer_depth', input_tensor, line_buffer_depth)


def count_relu(module, input_tensor, output_tensor): #TODO: change to count_activation and generalize
    # Get input tensor (returns [N,C,H,W]):
    input_tensor = input_tensor[0]

    # Count number of ops (each input element is activated):
    input_tensor_number_of_elements = input_tensor.numel()
    number_of_ops = int(input_tensor_number_of_elements)

    # Other of attributes:
    if len(input_tensor.shape) == 4:
        #[N,C,H,W]
        number_of_line_buffers = 1;
        line_buffer_length = input_tensor.shape[3];
        line_buffer_depth = 1
    else:
        #[N,C]
        number_of_line_buffers = 0
        line_buffer_length = input_tensor.shape[1]
        line_buffer_depth = 1

    ### Accumulate Attributes Values: ###
    module = add_attribute(module, 'number_of_ops', number_of_ops)
    module = set_attribute(module, 'number_of_parameters', 0)
    module = max_attribute(module, 'number_of_line_buffers', number_of_line_buffers)
    module = max_attribute(module, 'line_buffer_length', line_buffer_length)
    module = max_attribute(module, 'line_buffer_depth', line_buffer_depth);
    module = set_attribute(module, 'last_output', output_tensor);
    module = set_attribute(module, 'last_input', input_tensor)

    if hasattr(input_tensor,'counter')==False:
        input_tensor = set_attribute(input_tensor, 'counter', 1)
    output_tensor = set_attribute(output_tensor, 'counter', getattr(input_tensor,'counter')+1)

    output_tensor = create_append_list(output_tensor, 'number_of_ops', input_tensor, number_of_ops)
    output_tensor = create_append_list(output_tensor, 'number_of_parameters', input_tensor, 0)
    output_tensor = create_append_list(output_tensor, 'number_of_line_buffers', input_tensor, number_of_line_buffers)
    output_tensor = create_append_list(output_tensor, 'line_buffer_length', input_tensor, line_buffer_length)
    output_tensor = create_append_list(output_tensor, 'line_buffer_depth', input_tensor, line_buffer_depth)


def count_softmax(module, input_tensor, output_tensor): #TODO: add more softmax layers...
    # Get input tensor (returns [N,C,H,W]):
    input_tensor = input_tensor[0]

    # Get mini-batch needed sizes:
    batch_size, number_of_input_features_elements = input_tensor.size()

    # Count number of ops (each input element is exponentiated and divided by the sum of all the rest: exp(xi)/sum(exp(xj)) )
    total_exp = number_of_input_features_elements
    total_add = number_of_input_features_elements - 1
    total_div = number_of_input_features_elements
    number_of_ops = batch_size * (total_exp + total_add + total_div)
    number_of_ops = int(number_of_ops)

    if hasattr(input_tensor,'counter')==False:
        input_tensor = set_attribute(input_tensor, 'counter', 1)
    output_tensor = set_attribute(output_tensor, 'counter', getattr(input_tensor,'counter')+1)

    output_tensor = create_append_list(output_tensor, 'number_of_ops', input_tensor, number_of_ops)
    output_tensor = create_append_list(output_tensor, 'number_of_parameters', input_tensor, 0)
    output_tensor = create_append_list(output_tensor, 'number_of_line_buffers', input_tensor, number_of_line_buffers)
    output_tensor = create_append_list(output_tensor, 'line_buffer_length', input_tensor, line_buffer_length)
    output_tensor = create_append_list(output_tensor, 'line_buffer_depth', input_tensor, line_buffer_depth)


def count_maxpool(module, input_tensor, output_tensor): #TODO: why is there no distinction between mini-batch number of elements and single example number of elements!!!!
    input_tensor = input_tensor[0]

    # Get number of operations for each max pooling final element:
    kernel_ops = torch.prod(torch.Tensor([module.kernel_size])) - 1

    # Count number of ops (each max pool calculation is repeated output_tensor_number_of_feature_elements times):
    num_elements = output_tensor.numel()
    number_of_ops = kernel_ops * num_elements
    number_of_ops = int(number_of_ops);

    # Other of attributes:
    number_of_line_buffers = module.kernel_size;
    line_buffer_length = input_tensor.shape[3];
    line_buffer_depth = 1

    ### Accumulate Attributes Values: ###
    module = add_attribute(module, 'number_of_ops', number_of_ops)
    module = set_attribute(module, 'number_of_parameters', 0)
    module = max_attribute(module, 'number_of_line_buffers', number_of_line_buffers)
    module = max_attribute(module, 'line_buffer_length', line_buffer_length)
    module = max_attribute(module, 'line_buffer_depth', line_buffer_depth);
    module = set_attribute(module, 'last_output', output_tensor);
    module = set_attribute(module, 'last_input', input_tensor)

    if hasattr(input_tensor,'counter')==False:
        input_tensor = set_attribute(input_tensor, 'counter', 1)
    output_tensor = set_attribute(output_tensor, 'counter', getattr(input_tensor,'counter')+1)

    output_tensor = create_append_list(output_tensor, 'number_of_ops', input_tensor, number_of_ops)
    output_tensor = create_append_list(output_tensor, 'number_of_parameters', input_tensor, 0)
    output_tensor = create_append_list(output_tensor, 'number_of_line_buffers', input_tensor, number_of_line_buffers)
    output_tensor = create_append_list(output_tensor, 'line_buffer_length', input_tensor, line_buffer_length)
    output_tensor = create_append_list(output_tensor, 'line_buffer_depth', input_tensor, line_buffer_depth)


def count_avgpool(module, input_tensor, output_tensor):
    input_tensor = input_tensor[0]

    # Get number of operations for each avg pooling final element:
    total_add = torch.prod(torch.Tensor([module.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div

    # Count number of ops (each avg pool calculation is repeated output_tensor_number_of_feature_elements times):
    num_elements = output_tensor.numel()
    number_of_ops = kernel_ops * num_elements
    number_of_ops = int(number_of_ops)

    # Other of attributes:
    number_of_line_buffers = input_tensor.shape[2];
    line_buffer_length = input_tensor.shape[3];
    line_buffer_depth = 1

    ### Accumulate Attributes Values: ###
    module = add_attribute(module, 'number_of_ops', number_of_ops)
    module = set_attribute(module, 'number_of_parameters', 0)
    module = max_attribute(module, 'number_of_line_buffers', number_of_line_buffers)
    module = max_attribute(module, 'line_buffer_length', line_buffer_length)
    module = max_attribute(module, 'line_buffer_depth', line_buffer_depth);
    module = set_attribute(module, 'last_output', output_tensor);
    module = set_attribute(module, 'last_input', input_tensor)

    if hasattr(input_tensor,'counter')==False:
        input_tensor = set_attribute(input_tensor, 'counter', 1)
    output_tensor = set_attribute(output_tensor, 'counter', getattr(input_tensor,'counter')+1)


def count_linear(module, input_tensor, output_tensor):
    input_tensor = input_tensor[0]

    # per output element
    total_mul = module.in_features
    total_add = module.in_features - 1
    num_elements = output_tensor.numel()
    number_of_ops = (total_mul + total_add) * num_elements
    number_of_ops = int(number_of_ops)

    # Other of attributes:
    number_of_line_buffers = 1;
    line_buffer_length = input_tensor.shape[1];
    line_buffer_depth = 1

    ### Accumulate Attributes Values: ###
    module = add_attribute(module, 'number_of_ops', number_of_ops)
    module = set_attribute(module, 'number_of_parameters', 0)
    module = max_attribute(module, 'number_of_line_buffers', number_of_line_buffers)
    module = max_attribute(module, 'line_buffer_length', line_buffer_length)
    module = max_attribute(module, 'line_buffer_depth', line_buffer_depth);
    module = set_attribute(module, 'last_output', output_tensor);
    module = set_attribute(module, 'last_input', input_tensor)

    if hasattr(input_tensor,'counter')==False:
        input_tensor = set_attribute(input_tensor, 'counter', 1)
    output_tensor = set_attribute(output_tensor, 'counter', getattr(input_tensor,'counter')+1)

    output_tensor = create_append_list(output_tensor, 'number_of_ops', input_tensor, number_of_ops)
    output_tensor = create_append_list(output_tensor, 'number_of_parameters', input_tensor, 0)
    output_tensor = create_append_list(output_tensor, 'number_of_line_buffers', input_tensor, number_of_line_buffers)
    output_tensor = create_append_list(output_tensor, 'line_buffer_length', input_tensor, line_buffer_length)
    output_tensor = create_append_list(output_tensor, 'line_buffer_depth', input_tensor, line_buffer_depth)







































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










###############################################################################################################################################################################################################################################################
### ENET: ###
import torch.nn as nn
import torch
from torch.autograd import Variable


class InitialBlock(nn.Module):
    """The initial block is composed of two branches:
    1. a main branch which performs a regular convolution with stride 2;
    2. an extension branch which performs max-pooling.
    Doing both operations in parallel and concatenating their results
    allows for efficient downsampling and expansion. 
    The main branch outputs 13 feature maps while the extension branch outputs 3, for a
    total of 16 feature maps after concatenation.
    Keyword arguments:
    - number_of_input_channels (int): the number of input channels.
    - number_of_output_channels (int): the number output channels.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 number_of_input_channels,
                 number_of_output_channels,
                 kernel_size=3,
                 padding=0,
                 bias=False,
                 relu=True): # True=ReLU , False=PReLU
        super().__init__()

        # Choose ReLU or PReLU:
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        ### Main branch: ###
        # As stated above the number of output channels for this branch is the total minus 3, since the remaining channels come from the extension branch
        self.main_branch = nn.Conv2d(
                                    number_of_input_channels,
                                    number_of_output_channels - 3,
                                    kernel_size=kernel_size,
                                    stride=2,
                                    padding=padding,
                                    bias=bias)

        ### Extension branch: ###
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)

        #### Initialize batch normalization to be used after concatenation: ###
        self.batch_norm = nn.BatchNorm2d(number_of_output_channels)

        ### PReLU layer to apply after concatenating the branches ###
        self.out_prelu = activation

    def forward(self, x):
        ### Pass input through main (convolution) branch and extention (MaxPool) branch: ###
        main_branch = self.main_branch(x)
        extention_branch = self.ext_branch(x)

        ### Concatenate branches: ###
        out = torch.cat((main_branch, extention_branch), 1)

        ### Apply batch normalization: ###
        out = self.batch_norm(out)

        ### Apply Activation and return: ###
        return self.out_prelu(out)
##########################################################################################################################################################################



############################################################################################################################################################################
class RegularBottleneck(nn.Module):
    """Regular bottlenecks are the main building block of ENet.
    Main branch:
    1. Shortcut connection.
    Extension branch:
    1. 1x1 convolution which decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. regular, dilated or asymmetric convolution;
    3. 1x1 convolution which increases the number of channels back to
    ``channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - channels (int): the number of input and output channels.
    - internal_ratio (int, optional): a scale factor applied to
    ``channels`` used to compute the number of
    channels after the projection. eg. given ``channels`` equal to 128 and
    internal_ratio equal to 2 the number of channels after the projection
    is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension
    branch. Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 number_of_input_output_channels,
                 input_to_intermediate_channels_factor=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, number_of_input_output_channels]
        if input_to_intermediate_channels_factor <= 1 or input_to_intermediate_channels_factor > number_of_input_output_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(number_of_input_output_channels, input_to_intermediate_channels_factor))

        ### Get intermediate/projection layer number of channels: ###
        intermediate_number_of_channels = number_of_input_output_channels // input_to_intermediate_channels_factor

        ### Choose Activation: ###
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        ### Main branch - shortcut connection ###

        ### Extension branch: ###
        # 1x1 convolution, followed by a regular, dilated or asymmetric convolution, followed by another 1x1 convolution, and, finally, a regularizer (spatial dropout). Number of channels is constant.
        #(*).  1x1 projection convolution
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                    number_of_input_output_channels,
                    intermediate_number_of_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias),
            nn.BatchNorm2d(intermediate_number_of_channels),
            activation)

        #(*).  If the convolution is Asymmetric/Seperable we split the main convolution in two. Eg. for a 5x5 asymmetric convolution we have two convolution: the first is 5x1 and the second is 1x5.
        if asymmetric: #asymmetric=seperable convolution
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(intermediate_number_of_channels,
                          intermediate_number_of_channels,
                          kernel_size=(kernel_size, 1),
                          stride=1,
                          padding=(padding, 0),
                          dilation=dilation,
                          bias=bias),
                nn.BatchNorm2d(intermediate_number_of_channels),
                activation,
                nn.Conv2d(intermediate_number_of_channels,
                          intermediate_number_of_channels,
                          kernel_size=(1, kernel_size),
                          stride=1,
                          padding=(0, padding),
                          dilation=dilation,
                          bias=bias),
                nn.BatchNorm2d(intermediate_number_of_channels),
                activation)
        else: #symmetric=full kernel convolution
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(intermediate_number_of_channels,
                          intermediate_number_of_channels,
                          kernel_size=kernel_size,
                          stride=1,
                          padding=padding,
                          dilation=dilation,
                          bias=bias), nn.BatchNorm2d(intermediate_number_of_channels), activation)

        #(*).  1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(intermediate_number_of_channels,
                      number_of_input_output_channels,
                      kernel_size=1,
                      stride=1,
                      bias=bias),
            nn.BatchNorm2d(number_of_input_output_channels),
            activation)

        #(*). DropOut:
        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)
##################################################################################################################################




##################################################################################################################################
class DownsamplingBottleneck(nn.Module):
    """Downsampling bottlenecks further downsample the feature map size.
    Main branch:
    1. max pooling with stride 2; indices are saved to be used for
    unpooling later.
    Extension branch:
    1. 2x2 convolution with stride 2 that decreases the number of channels
    by ``internal_ratio``, also called a projection;
    2. regular convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``number_of_output_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - number_of_input_channels (int): the number of input channels.
    - number_of_output_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``channels``
    used to compute the number of channels after the projection. eg. given
    ``channels`` equal to 128 and internal_ratio equal to 2 the number of
    channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in
    the convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the
    input. Default: 0.
    - dilation (int, optional): spacing between kernel elements for the
    convolution described in item 2 of the extension branch. Default: 1.
    - asymmetric (bool, optional): flags if the convolution described in
    item 2 of the extension branch is asymmetric or not. Default: False.
    - return_indices (bool, optional):  if ``True``, will return the max
    indices along with the outputs. Useful when unpooling later.
    - dropout_prob (float, optional): probability of an element to be
    zeroed. Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if
    ``True``. Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 number_of_input_channels,
                 number_of_output_channels,
                 intermediate_number_of_channels=4,
                 kernel_size=3,
                 padding=0,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if intermediate_number_of_channels <= 1 or intermediate_number_of_channels > number_of_input_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(number_of_input_channels, intermediate_number_of_channels))

        ### Get Intermediate number of channels: ###
        internal_channels = number_of_input_channels // intermediate_number_of_channels

        ### Choose Activation: ###
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        ### Main branch - max pooling followed by feature map (channels) padding: ###
        self.main_max1 = nn.MaxPool2d(
            kernel_size,
            stride=2,
            padding=padding,
            return_indices=return_indices)

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        #(*). 2x2 projection convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                number_of_input_channels,
                internal_channels,
                kernel_size=2,
                stride=2, #Stride 2 which lowers spatial size
                bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation)
        #(*). Convolution (why not use Asymmetric/Dilated/Whatever convolution her?):
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation)
        #(*). 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                number_of_output_channels,
                kernel_size=1,
                stride=1,
                bias=bias),
            nn.BatchNorm2d(number_of_output_channels),
            activation)
        #(*). DropOut:
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        #(*). PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = Variable(torch.zeros(n, ch_ext - ch_main, h, w)) #TODO: weird way of doing padding.... i need to have a layer like "pad_to_input_tensor_shape_layer" and "flexible_pad_layer" to be able to pad to what is needed according to the following kernel size and be able to do circular padding etc'

        # Before concatenating, check if main is on the CPU or GPU and convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out), max_indices
##################################################################################################################################



##################################################################################################################################
class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.
    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``intermediate_number_of_channels``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.
    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``intermediate_number_of_channels``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``number_of_output_channels``, also called an expansion;
    4. dropout as a regularizer.
    Keyword arguments:
    - number_of_input_channels (int): the number of input channels.
    - number_of_output_channels (int): the number of output channels.
    - intermediate_number_of_channels (int, optional): a scale factor applied to ``number_of_input_channels``
     used to compute the number of channels after the projection. eg. given
     ``number_of_input_channels`` equal to 128 and ``intermediate_number_of_channels`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in the
    convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the input.
    Default: 0.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.
    """

    def __init__(self,
                 number_of_input_channels,
                 number_of_output_channels,
                 intermediate_number_of_channels=4,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if intermediate_number_of_channels <= 1 or intermediate_number_of_channels > number_of_input_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(number_of_input_channels, intermediate_number_of_channels))

        ### Get intermediate number of channels: ###
        internal_channels = number_of_input_channels // intermediate_number_of_channels

        ### Get Activation: ###
        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        ### Main branch - max pooling followed by feature map (channels) padding: ###
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(number_of_input_channels, number_of_output_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(number_of_output_channels))

        ### Remember that the stride is the same as the kernel_size, just like the max pooling layers: ###
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2) #Todo: understand the tradeoff between accuracy performance and resource peformance between this maxunpool2d and shuffle layer

        ### Extension branch: ###
        # 1x1 convolution, followed by a regular, dilated or asymmetric convolution, followed by another 1x1 convolution. Number of channels is doubled.
        #(*).  1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
                    nn.Conv2d(
                        number_of_input_channels, internal_channels, kernel_size=1, bias=bias),
                    nn.BatchNorm2d(internal_channels), activation)
        #(*).  Transposed convolution
        self.ext_conv2 = nn.Sequential(
                    nn.ConvTranspose2d(
                        internal_channels,
                        internal_channels,
                        kernel_size=kernel_size,
                        stride=2, #TODO: so spatial size increases by 2 right?
                        padding=padding,
                        output_padding=1,
                        bias=bias),
                    nn.BatchNorm2d(internal_channels),
                    activation)
        #(*). 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
                    nn.Conv2d(internal_channels, number_of_output_channels, kernel_size=1, bias=bias),
                    nn.BatchNorm2d(number_of_output_channels),
                    activation)
        #(*). DropOut:
        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        #(*). PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x, max_indices):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)
##################################################################################################################################



##################################################################################################################################
class ENet(nn.Module):
    """Generate the ENet model.
    Keyword arguments:
    - num_classes (int): the number of classes to segment.
    - encoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the encoder blocks/layers; otherwise, PReLU
    is used. Default: False.
    - decoder_relu (bool, optional): When ``True`` ReLU is used as the
    activation function in the decoder blocks/layers; otherwise, PReLU
    is used. Default: True.
    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
                                                    number_of_input_channels = 16,
                                                    number_of_output_channels = 64,
                                                    padding=1,
                                                    return_indices=True,
                                                    dropout_prob=0.01,
                                                    relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(number_of_input_output_channels=64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(number_of_input_output_channels=64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(number_of_input_output_channels=64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(number_of_input_output_channels=64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
                                                    number_of_input_channels=64,
                                                    number_of_output_channels=128,
                                                    padding=1,
                                                    return_indices=True,
                                                    dropout_prob=0.1,
                                                    relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=3, padding=1, dilation=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=3, padding=2, dilation=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=5, padding=2, dilation=1, dropout_prob=0.1, relu=encoder_relu, asymmetric=True)
        self.dilated2_4 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=3, padding=4, dilation=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=3, padding=1, dilation=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=3, padding=8, dilation=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=5, padding=2, dilation=1, dropout_prob=0.1, relu=encoder_relu, asymmetric=True)
        self.dilated2_8 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=3, padding=16, dilation=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(number_of_input_output_channels=128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(number_of_input_output_channels=128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(number_of_input_output_channels=128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(number_of_input_output_channels=128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(number_of_input_output_channels=128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(number_of_input_output_channels=128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(number_of_input_output_channels=128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(number_of_input_output_channels=128, number_of_output_channels=64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(number_of_input_output_channels=64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(number_of_input_output_channels=64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(number_of_input_output_channels=64, number_of_output_channels=16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(number_of_input_output_channels=16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
                                                in_channels=16,
                                                out_channels=num_classes,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1,
                                                bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)

        return x

### END OF ENET ####
#############################################################################################################################################################################################################################








#############################################################################################################################################################################################################################
class Conv_BatchNorm_PReLU(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, number_of_input_channels, number_of_output_channels, kernel_size, stride=1):
        '''
        :param number_of_input_channels: number of input channels
        :param number_of_output_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(number_of_input_channels, number_of_output_channels, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(number_of_output_channels, eps=1e-03)
        self.act = nn.PReLU(number_of_output_channels)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
##################################################################################################################################



##################################################################################################################################
class BatchNorm_PReLU(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, number_of_output_channels):
        '''
        :param number_of_output_channels: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(number_of_output_channels, eps=1e-03)
        self.act = nn.PReLU(number_of_output_channels)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output
##################################################################################################################################



##################################################################################################################################
class Conv_BatchNorm(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, number_of_input_channels, number_of_output_channels, kernel_size, stride=1):
        '''
        :param number_of_input_channels: number of input channels
        :param number_of_output_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(number_of_input_channels, number_of_output_channels, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(number_of_output_channels, eps=1e-03)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output
##################################################################################################################################



##################################################################################################################################
class Conv(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, number_of_input_channels, number_of_output_channels, kernel_size, stride=1):
        '''
        :param number_of_input_channels: number of input channels
        :param number_of_output_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(number_of_input_channels, number_of_output_channels, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
##################################################################################################################################



##################################################################################################################################
class Conv_Dilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, number_of_input_channels, number_of_output_channels, kernel_size, stride=1, dilation=1):
        '''
        :param number_of_input_channels: number of input channels
        :param number_of_output_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param dilation: optional dilation rate
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2d(number_of_input_channels, number_of_output_channels, (kernel_size, kernel_size), stride=stride, padding=(padding, padding), bias=False, dilation=dilation)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
##################################################################################################################################



##################################################################################################################################
class DownSampler_MultipleDilations_Block(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels):
        super().__init__()

        ### Get number input=output channels for each dilation (and take care of cases when global number_of_input_channels isn't a multiple of 5: ###
        n = int(number_of_output_channels / 5)
        n1 = number_of_output_channels - 4 * n

        ### Get Dilated Convolutions: ###
        self.c1 = Conv(number_of_input_channels, n, kernel_size=3, stride=2) #DOWN-SAMPLING HERE
        self.d1 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n1, kernel_size=3, stride=1, dilation=1)
        self.d2 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=2)
        self.d4 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=4)
        self.d8 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=8)
        self.d16 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=16)
        self.bn = nn.BatchNorm2d(number_of_output_channels, eps=1e-3)
        self.act = nn.PReLU(number_of_output_channels)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output
##################################################################################################################################



##################################################################################################################################
class MultipleDilations_Residual_Block(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, number_of_input_channels, number_of_output_channels, flag_add_residual=True):
        '''
        :param number_of_input_channels: number of input channels
        :param number_of_output_channels: number of output channels
        :param flag_add_residual: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()

        ### Get number input=output channels for each dilation (and take care of cases when global number_of_input_channels isn't a multiple of 5: ###
        n = int(number_of_output_channels / 5)
        n1 = number_of_output_channels - 4 * n

        ### Get Dilated Convolutions: ###
        self.c1 = Conv(number_of_input_channels, n, kernel_size=1, stride=1)
        self.d1 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n1, kernel_size=3, stride=1, dilation=1)  # dilation rate of 2^0
        self.d2 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=2)  # dilation rate of 2^1
        self.d4 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=4)  # dilation rate of 2^2
        self.d8 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=8)  # dilation rate of 2^3
        self.d16 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n, kernel_size=3, stride=1, dilation=16)  # dilation rate of 2^4
        self.bn = BatchNorm_PReLU(number_of_output_channels)
        self.flag_add_residual = flag_add_residual

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce number of channels by ~5 using 1X1 convolution:
        output1 = self.c1(input)

        # split and transform:
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding:
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge:
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version:
        if self.flag_add_residual:
            combine = input + combine

        # BatchNorm at the end:
        output = self.bn(combine)
        return output
##################################################################################################################################



##################################################################################################################################
##################################################################################################################################
class MultipleDilations_Residual_Block_Generalized(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, number_of_input_channels, number_of_output_channels, flag_add_residual=True, number_of_dilations=5):
        '''
        :param number_of_input_channels: number of input channels
        :param number_of_output_channels: number of output channels
        :param flag_add_residual: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        self.number_of_dilations = number_of_dilations;
        ### Get number input=output channels for each dilation (and take care of cases when global number_of_input_channels isn't a multiple of 5: ###
        n = int(number_of_output_channels / number_of_dilations)
        n1 = number_of_output_channels - (number_of_dilations-1) * n

        ### Get Dilated Convolutions: ###
        self.c1 = Conv(number_of_input_channels, n, kernel_size=1, stride=1)
        self.d1 = Conv_Dilated(number_of_input_channels=n, number_of_output_channels=n1, kernel_size=3, stride=1, dilation=1)  # dilation rate of 2^0
        for current_dilation_power in arange(1,number_of_dilations):
            setattr(self, 'd'+str(2**current_dilation_power), Conv_Dilated(number_of_input_channels=n,number_of_output_channels=n,kernel_size=3,stride=1,dilation=2**current_dilation_power))
        self.bn = BatchNorm_PReLU(number_of_output_channels)
        self.flag_add_residual = flag_add_residual

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce number of channels by ~5 using 1X1 convolution:
        output1 = self.c1(input)

        # split and transform:
        dilations_outputs_list = []
        dilations_outputs_list.append(self.d1(output1));
        for current_dilation_power in arange(1,self.number_of_dilations):
            dilations_outputs_list.append( getattr(self, 'd'+str(2**current_dilation_power) )(output1) )

        # heirarchical fusion for de-gridding:
        for current_dilation_power in arange(2,self.number_of_dilations):
            dilations_outputs_list[current_dilation_power] = dilations_outputs_list[current_dilation_power] + dilations_outputs_list[current_dilation_power-1]

        # merge:
        combined = dilations_outputs_list[0];
        for current_dilation_power in arange(1,self.number_of_dilations):
            combined = torch.cat( [ combined, dilations_outputs_list[current_dilation_power] ], 1 )

        # if residual version:
        if self.flag_add_residual:
            combined = input + combined

        print(combined.shape)

        # BatchNorm at the end:
        output = self.bn(combined)
        return output


# number_of_input_channels = 15
# number_of_output_channels = 15;
# number_of_dilations = 5;
# module = MultipleDilations_Residual_Block_Generalized(number_of_input_channels,number_of_output_channels,flag_add_residual=False, number_of_dilations=number_of_dilations);
# input_tensor = torch.Tensor(randn(1,number_of_input_channels,100,100))
# output_tensor = module(input_tensor)
# print_network_graph_to_pdf(module,(number_of_input_channels,100,100))
# print_network_summary(module,(15,100,100),-1,'cpu')
##################################################################################################################################
##################################################################################################################################



##################################################################################################################################
class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    #TODO: well, obviously the class DOESN'T do what it says in the description... it simply gets an input variable which is the number of times we downsample (using AvgPool2d) by 2 (the equivallent of number_of_levels):
    #TODO: should be renamed to something like "pyramid_avgpool_downsampler" or "avgpool_concatenations"
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input
##################################################################################################################################



##################################################################################################################################
class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''

    def __init__(self,
                 number_of_classes=20,
                 number_of_input_channels=3,
                 number_of_intermediate_channels=16,
                 number_of_residual_channels1=64,
                 number_of_residual_channels2=128,
                 number_of_final_channels=256,
                 number_of_residual_layers_1=5,
                 number_of_residual_layers_2=3):
        '''
        :param number_of_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param number_of_residual_layers_1: depth multiplier
        :param number_of_residual_layers_2: depth multiplier
        '''
        super().__init__()
        #TODO: get certain hyper-parameters (numbers of channels, number of dilations etc') as input variables instead of being pre-determined:
        self.conv_batchnorm_prelu_1 = Conv_BatchNorm_PReLU(number_of_input_channels=number_of_input_channels, number_of_output_channels=number_of_intermediate_channels, kernel_size=3, stride=2)
        self.avgpool_concatenations1 = InputProjectionA(1) #Downsample once
        self.avgpool_concatenations2 = InputProjectionA(2) #Downsample twice

        self.batchnorm_prelu_1 = BatchNorm_PReLU(number_of_input_channels + number_of_intermediate_channels)

        self.downsample_multiple_dilations_1 = DownSampler_MultipleDilations_Block(number_of_input_channels + number_of_intermediate_channels, number_of_residual_channels1)
        self.multiple_dilations_residual_block_list_1 = nn.ModuleList()
        for i in range(0, number_of_residual_layers_1):
            self.multiple_dilations_residual_block_list_1.append(MultipleDilations_Residual_Block(number_of_residual_channels1, number_of_residual_channels1))
        self.batchnorm_prelu_2 = BatchNorm_PReLU(number_of_residual_channels2 + number_of_input_channels)

        self.downsample_multiple_dilations_2 = DownSampler_MultipleDilations_Block(number_of_residual_channels2 + number_of_input_channels, number_of_residual_channels2)
        self.multiple_dilations_residual_block_list_2 = nn.ModuleList()
        for i in range(0, number_of_residual_layers_2):
            self.multiple_dilations_residual_block_list_2.append(MultipleDilations_Residual_Block(number_of_residual_channels2, number_of_residual_channels2))
        self.batchnorm_prelu_3 = BatchNorm_PReLU(number_of_final_channels)

        self.classifier = Conv(number_of_final_channels, number_of_classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        #Downsample twice:
        output0 = self.conv_batchnorm_prelu_1(input) #Conv with Stride=2 so downsampling
        inp1 = self.avgpool_concatenations1(input) #Simple AvgPool2d
        output0_cat = self.batchnorm_prelu_1(torch.cat([output0, inp1], 1)) #Concatenate conv-downsampled and avgpool2d outputs
        initial_downsamples_output = self.downsample_multiple_dilations_1(output0_cat)  # down-sample further

        #Have multiple sequential layers of multiple-parallel-dilations:
        for i, layer in enumerate(self.multiple_dilations_residual_block_list_1):
            if i == 0:
                residual_layers_1_output = layer(initial_downsamples_output)
            else:
                residual_layers_1_output = layer(residual_layers_1_output)

        #Gather output of sequential residual layers, Concatenate it with the output of avgpool_concatenations2 which does avgpool2d twice...this would not be possible at final architecture:
        inp2 = self.avgpool_concatenations2(input)
        output1_cat = self.batchnorm_prelu_2(torch.cat([residual_layers_1_output, initial_downsamples_output, inp2], 1))

        #Downsample again and again insert into a sequential list of residual layers:
        output2_0 = self.downsample_multiple_dilations_2(output1_cat)  # down-sampled
        for i, layer in enumerate(self.multiple_dilations_residual_block_list_2):
            if i == 0:
                residual_layers_2_output = layer(output2_0)
            else:
                residual_layers_2_output = layer(residual_layers_2_output)
        output2_cat = self.batchnorm_prelu_3(torch.cat([output2_0, residual_layers_2_output], 1))

        ### Final Classifier Conv Layer: ###
        classifier = self.classifier(output2_cat)

        return classifier
##################################################################################################################################



##################################################################################################################################
class ESPNet(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self,
                 number_of_classes=20,
                 number_of_input_channels = 3,
                 number_of_intermediate_channels = 16,
                 number_of_residual_channels1 = 64,
                 number_of_residual_channels2 = 128,
                 number_of_final_channels = 256,
                 number_of_residual_layers_1=2,
                 number_of_residual_layers_2=3,
                 encoderFile=None):
        '''
        :param number_of_classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param number_of_residual_layers_1: depth multiplier
        :param number_of_residual_layers_2: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()

        #Get Encoder:
        #TODO: it's interesting to note the the encoder itself isn't used directly in the forward function....
        self.encoder = ESPNet_Encoder(number_of_classes=number_of_classes,
                                      number_of_input_channels=number_of_input_channels,
                                      number_of_intermediate_channels=number_of_intermediate_channels,
                                      number_of_residual_channels1=number_of_residual_channels1,
                                      number_of_residual_channels2=number_of_residual_channels2,
                                      number_of_final_channels=number_of_final_channels,
                                      number_of_residual_layers_1=number_of_residual_layers_1,
                                      number_of_residual_layers_2=number_of_residual_layers_2)

        #Load Pretrained encoder if wanted:
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')

        #Get encoder modules into a modules list:
        self.encoder_modules = []
        for i, m in enumerate(self.encoder.children()):
            self.encoder_modules.append(m)

        ### light-weight Decoder: ###
        self.level3_C = Conv(number_of_residual_channels2 + number_of_input_channels, number_of_classes, 1, 1)
        self.br = nn.BatchNorm2d(number_of_classes, eps=1e-03)
        self.conv = Conv_BatchNorm_PReLU(19 + number_of_classes, number_of_classes, 3, 1) #TODO: understand where this number "19" comes from
        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(number_of_classes, number_of_classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BatchNorm_PReLU(2 * number_of_classes),
                                           MultipleDilations_Residual_Block(2 * number_of_classes, number_of_classes, flag_add_residual=False))
        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(number_of_classes, number_of_classes, 2, stride=2, padding=0, output_padding=0, bias=False),
                                   BatchNorm_PReLU(number_of_classes))
        self.classifier = nn.ConvTranspose2d(number_of_classes, number_of_classes, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        #TODO: this is very specific to the predetermined encoder hyperparameters used by the authors...i need to adapt it to my general function
        output0 = self.encoder_modules[0](input)
        inp1 = self.encoder_modules[1](input)
        inp2 = self.encoder_modules[2](input)

        output0_cat = self.encoder_modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.encoder_modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.encoder_modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.encoder_modules[6](torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.encoder_modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.encoder_modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.encoder_modules[9](torch.cat([output2_0, output2], 1))  # concatenate for feature map width expansion

        output2_c = self.up_l3(self.br(self.encoder_modules[10](output2_cat)))  # RUM

        output1_C = self.level3_C(output1_cat)  # project to C-dimensional space
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1)))  # RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.classifier(concat_features)
        return classifier
##################################################################################################################################































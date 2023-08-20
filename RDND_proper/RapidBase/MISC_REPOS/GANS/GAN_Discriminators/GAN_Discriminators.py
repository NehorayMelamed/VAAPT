#Imports:
#(1). Auxiliary:
from __future__ import print_function
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
import psutil
#counter = collections.Counter()
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.ar_model import AR
import argparse
import pydot
import functools
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
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import vgg19
from torchvision.utils import make_grid
import torchvision
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
import torch.nn.init as init
#(6). My network graph visualization function:
import network_graph_visualization
from network_graph_visualization import *

#TensorFlow:
import tensorflow as tf
import tensorboard as TB
import tensorboardX as TBX

#Numpy:
from numpy.fft import *
from numpy.linalg import *
from numpy import *
from numpy.random import *
from numpy import power as power
from numpy import exp as exp


#ESRGAN:
import ESRGAN_dataset
import ESRGAN_Visualizers
import ESRGAN_Optimizers
import ESRGAN_Losses
import ESRGAN_deep_utils
import ESRGAN_utils
# import ESRGAN_Models
from ESRGAN_utils import *
from ESRGAN_deep_utils import *
# from ESRGAN_Models import *
from ESRGAN_Losses import *
# from ESRGAN_Optimizers import *
# from ESRGAN_Visualizers import *
# from ESRGAN_dataset import *
# from ESRGAN_OPT import *
import ESRGAN_basic_Blocks_and_Layers
from ESRGAN_basic_Blocks_and_Layers import *

from easydict import EasyDict as edict
from scipy.io import loadmat
# Our libs
# from Semantic_Segmentation.SemanticSegmentation_CVAIL.dataset import TestDataset
# from Semantic_Segmentation.SemanticSegmentation_CVAIL.models import ModelBuilder, SegmentationModule
# from Semantic_Segmentation.SemanticSegmentation_CVAIL.utils import colorEncode
# from Semantic_Segmentation.SemanticSegmentation_CVAIL.lib.nn import user_scattered_collate, async_copy_to
# from Semantic_Segmentation.SemanticSegmentation_CVAIL.lib.utils import as_numpy, mark_volatile
# import Semantic_Segmentation.SemanticSegmentation_CVAIL.lib.utils.data as torchdata
import cv2








# Discriminator
def get_Discriminator_Architecture(OPT, OPT_discriminator):
    #TODO: add emphasis on fully convolutional discriminators (patch-GANs)... wouldn't those be the ultimate in interpretability???
    gpu_ids = OPT.gpu_ids
    which_model = OPT_discriminator.model_type #TODO: change from model_type to model_name

    if which_model == 'discriminator_vgg_128':
        netD = Discriminator_VGG_128(number_of_input_channels=OPT_discriminator.number_of_input_channels,
                                     vgg_doubling_base_number_of_filters=OPT_discriminator.vgg_doubling_base_number_of_filters,
                                     normalization_type=OPT_discriminator.normalization_type,
                                     activation_type=OPT_discriminator.activation_type,
                                     mode=OPT_discriminator.convolution_block_mode)
    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = ACD_VGG_BN_96()

    elif which_model == 'discriminator_vgg_96':
        netD = Discriminator_VGG_96(number_of_input_channels=OPT_discriminator.number_of_input_channels,
                                    vgg_doubling_base_number_of_filters=OPT_discriminator.vgg_doubling_base_number_of_filters,
                                    normalization_type=OPT_discriminator.normalization_type,
                                    activation_type=OPT_discriminator.activation_type,
                                    mode=OPT_discriminator.convolution_block_mode)
    elif which_model == 'discriminator_vgg_192':
        netD = Discriminator_VGG_192(number_of_input_channels=OPT_discriminator.number_of_input_channels,
                                     vgg_doubling_base_number_of_filters=OPT_discriminator.vgg_doubling_base_number_of_filters,
                                     normalization_type=OPT_discriminator.normalization_type,
                                     activation_type=OPT_discriminator.activation_type,
                                     mode=OPT_discriminator.convolution_block_mode)
    elif which_model == 'discriminator_vgg_128_SN':
        netD = Discriminator_VGG_128_SN()  # TODO: look into it
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD










########################################
# Discriminators:
########################################
# (*). First VGG style discriminators for different sized inputs (TODO: get rid of the unecessary ones!!!!):
# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, number_of_input_channels, vgg_doubling_base_number_of_filters, normalization_type='batch_normalization',
                 activation_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = Conv_Block(number_of_input_channels, vgg_doubling_base_number_of_filters, kernel_size=3,
                           normalization_type=None, activation_type=activation_type, mode=mode)
        conv1 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters, kernel_size=4,
                           stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 64, 64
        conv2 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters * 2, kernel_size=3,
                           stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv3 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 2,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 32, 128
        conv4 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv5 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 16, 256
        conv6 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv7 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 8, 512
        conv8 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv9 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 4, 512
        self.features = Pile_Modules_On_Top_Of_Each_Other(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
                                                          conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x





class Discriminator_VGG_128_modified(nn.Module):
    def __init__(self, number_of_input_channels, vgg_doubling_base_number_of_filters, normalization_type='batch_normalization',
                 activation_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128_modified, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = Conv_Block(number_of_input_channels, vgg_doubling_base_number_of_filters, kernel_size=3,
                           normalization_type=None, activation_type=activation_type, mode=mode)
        conv1 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters, kernel_size=4,
                           stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 64, 64
        conv2 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters * 2, kernel_size=3,
                           stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv3 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 2,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 32, 128
        conv4 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv5 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 16, 256
        conv6 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv7 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 8, 512
        conv8 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv9 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 4, 512
        self.features = Pile_Modules_On_Top_Of_Each_Other(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
                                                          conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x,(4,4))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




# VGG style Discriminator with input size 128*128, Spectral Normalization (no BN!!!):
class Discriminator_VGG_128_SN(nn.Module):
    # Basic Building Block: Conv2d->Spectral_Norm->LeakyRelU
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        #TODO: should i add an AdaptiveAvgPooling or be contrained to a certain size (in this case 128)? - again...seems a semantic segmentation network would be much more appropriate....
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self, number_of_input_channels, vgg_doubling_base_number_of_filters, normalization_type='batch',
                 activation_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = Conv_Block(number_of_input_channels, vgg_doubling_base_number_of_filters, kernel_size=3,
                           normalization_type=None, activation_type=activation_type, mode=mode)
        conv1 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters, kernel_size=4,
                           stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 48, 64
        conv2 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters * 2, kernel_size=3,
                           stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv3 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 2,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 24, 128
        conv4 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv5 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 12, 256
        conv6 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv7 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 6, 512
        conv8 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv9 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 3, 512
        self.features = Pile_Modules_On_Top_Of_Each_Other(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,
                                                          conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, number_of_input_channels, vgg_doubling_base_number_of_filters, normalization_type='batch',
                 activation_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = Conv_Block(number_of_input_channels, vgg_doubling_base_number_of_filters, kernel_size=3,
                           normalization_type=None, activation_type=activation_type, mode=mode)
        conv1 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters, kernel_size=4,
                           stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 96, 64
        conv2 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters * 2, kernel_size=3,
                           stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv3 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 2,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 48, 128
        conv4 = Conv_Block(vgg_doubling_base_number_of_filters * 2, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv5 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 4,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 24, 256
        conv6 = Conv_Block(vgg_doubling_base_number_of_filters * 4, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv7 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 12, 512
        conv8 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=3, stride=1, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        conv9 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                           kernel_size=4, stride=2, normalization_type=normalization_type,
                           activation_type=activation_type, mode=mode)
        # 6, 512
        conv10 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                            kernel_size=3, stride=1, normalization_type=normalization_type,
                            activation_type=activation_type, mode=mode)
        conv11 = Conv_Block(vgg_doubling_base_number_of_filters * 8, vgg_doubling_base_number_of_filters * 8,
                            kernel_size=4, stride=2, normalization_type=normalization_type,
                            activation_type=activation_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10,
                                     conv11)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Auxiliary Classifier Discriminator (ACD):
class ACD_VGG_BN_96(nn.Module):
    def __init__(self):
        super(ACD_VGG_BN_96, self).__init__()

        # VGG Style CNN:
        self.feature = nn.Sequential(
            # TODO: ask, even though i can make up excuses, why is it that at the first layer i don't see a BN layer almost ever?
            # TODO: ask if i'm correct to observe that there is a trend from regular convolution + max pool to a strided convolution to lower image size? why is that?
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),

            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.1, True),
        )

        # GAN Discriminator Validity Head (a fully connected layer which at the end outputs a single number)
        # TODO: understand how to make this fully convolutional (maybe simple use the patch GAN approach?!?!# seems very flexible)
        self.gan = nn.Sequential(
            nn.Linear(512 * 6 * 6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 1)
        )

        # Auxiliary Discriminator Classification Head (a fully connected netwrok which outputs 8 number for 8 classes?!@$?!$!#?$ where is the softmax if that's correct!$!%#@%):
        self.cls = nn.Sequential(
            nn.Linear(512 * 6 * 6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 8)
        )

    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls]  # output regular validity score and the 8 numbers classification output



########################################
# Semantic(?) Segmentation Architectures:
########################################
class OutdoorSceneSeg(nn.Module):
    def __init__(self):
        super(OutdoorSceneSeg, self).__init__()
        # conv1
        blocks = []
        conv1_1 = Conv_Block(3, 64, 3, 2, 1, 1, False, 'zero', 'batch')  # /2
        conv1_2 = Conv_Block(64, 64, 3, 1, 1, 1, False, 'zero', 'batch')
        conv1_3 = Conv_Block(64, 128, 3, 1, 1, 1, False, 'zero', 'batch')
        max_pool = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)  # /2
        blocks = [conv1_1, conv1_2, conv1_3, max_pool]
        # conv2, 3 blocks

        # USE ResNet131 Block (with the 1X1 Convolution Bottle-Necks) Defined Above!:
        blocks.append(Res131(128, 64, 256))
        for i in range(2):
            blocks.append(Res131(256, 64, 256))
        # conv3, 4 blocks
        blocks.append(Res131(256, 128, 512, 1, 2))  # /2
        for i in range(3):
            blocks.append(Res131(512, 128, 512))
        # conv4, 23 blocks
        blocks.append(Res131(512, 256, 1024, 2))
        for i in range(22):
            blocks.append(Res131(1024, 256, 1024, 2))

        # conv5
        blocks.append(Res131(1024, 512, 2048, 4))
        blocks.append(Res131(2048, 512, 2048, 4))
        blocks.append(Res131(2048, 512, 2048, 4))
        blocks.append(Conv_Block(2048, 512, 3, 1, 1, 1, False, 'zero', 'batch'))
        blocks.append(nn.Dropout(0.1))

        # conv6
        blocks.append(nn.Conv2d(512, 8, 1, 1))

        self.feature = Pile_Modules_On_Top_Of_Each_Other(*blocks)

        # deconv
        self.deconv = nn.ConvTranspose2d(8, 8, 16, 8, 4, 0, 8, False, 1)

        # softmax
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.feature(x)
        x = self.deconv(x)
        x = self.softmax(x)
        return x









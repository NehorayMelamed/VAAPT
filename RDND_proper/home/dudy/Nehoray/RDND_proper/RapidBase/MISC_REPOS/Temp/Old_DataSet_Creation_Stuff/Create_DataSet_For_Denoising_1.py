
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


import os, time, scipy.io
import numpy as np
import rawpy
import glob
# from logger import Logger
import matplotlib.pyplot as plt
import cv2 as cv
from colour_demosaicing import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Menon2007
import colour
import pickle




def load_image(resize=None, indx = 1):
    #database_path = 'E:/learning_to_see_in_dark_Rony/dataset/test/'

    database_path = 'D:/Work/SeeInDark/datasets/test/'
    all_files = os.listdir(database_path)
    current_image = all_files[indx]


    img = cv.cvtColor(cv.imread(database_path + current_image), cv.COLOR_BGR2RGB)

    if resize is not None:
        h, w, _ = img.shape

        dx = resize if w > h else int(round(w * resize / h/2)*2)
        dy = resize if w < h else int(round(h * resize / w/2)*2)

        img = cv.resize(img, (dx, dy), interpolation=cv.INTER_AREA)

    return (img / np.iinfo(np.uint8).max).astype(np.float32)





def to_linear(img):
    """
    Apply reverse sRGB curve on input image as defined at
    https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation.
    """
    return np.where(img >= 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)


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






###### Go Over Wanted Images And Create Pickle Files: ######
#(1). Parameters:
memory_images = 100
num_of_images = 125000
lastepoch = 0
image_index = lastepoch*memory_images
allfolders = glob.glob('./result/*0')
#(2). Get All image filenames:



for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch, 4001):
    im_ind = 0
    if os.path.isdir("result/%04d" % epoch):
        continue


    if 1:
        gt_rgb_images = [None] * memory_images
        clean_bayer_images = [None] * memory_images
        input_images_std = [None] * memory_images

        for im_num in range(0,memory_images):
            ### Load image and use mosaicing to get the bayer image: ###
            img = load_image(resize=512, indx=image_index)
            image_index = image_index + 1
            bayer = to_linear(mosaicing_CFA_Bayer(img, 'GRBG'))  #to_linear = de-gamma
            h, w = bayer.shape


            # ### Mosaicing is pretty quick!: ###
            # tic()
            # input_rgb = randn(100,100,3);
            # for i in arange(100):
            #     output = to_linear(mosaicing_CFA_Bayer(input_rgb, 'GRBG'));
            # toc()
            #
            # ### Demosaicing is really slow!: ###
            # tic()
            # input_bayer = randn(400,400);
            # for i in arange(100):
            #     output = demosaicing_CFA_Bayer_Menon2007(input_bayer, 'GRBG')
            # toc()

            ### Get random gamma and gain (detector gain): ###
            gamma_rand = np.random.random(1)*1.5+1
            gain_rand = np.random.random(1)*0.8+0.2
            bayer = (bayer**gamma_rand)*gain_rand

            ### Get random R,B,G gains: ### color augmentation
            g_rand = np.random.random(1)*2+0.5
            gain_array = [g_rand, (np.random.random(1) * 2 + 0.5), g_rand, (np.random.random(1) * 2 + 0.5)]
            bayer = apply_gain(bayer, gain_array)

            ### Accumulate clean-bayer and "clean" (demoasiced again(!)) RGB images: ###
            clean_bayer_images[im_num] = bayer
            gt_rgb_images[im_num] = np.clip(demosaicing_CFA_Bayer_Menon2007(bayer, 'GRBG'), 0, 1)
            print("loading image %d " % im_num)


    ### Loop over current images and apply something(?): ###
    for ind in range(0,memory_images):

        ### Get current bayer and RGB (demoasiced from bayer) images: ###
        img_gt = gt_rgb_images[ind]
        bayer = clean_bayer_images[ind]
        h, w = bayer.shape

        ### Get more (White-Balance???) small random gains for the gr,gb and large gains for the R,B bayer channels: ###
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
        std_image = compute_noise_stddev(bayer_before_wb, analog_gain, use_mean=False)  ### Difference is use_mean=False  ???

        ### Apply Gain to the Noise itself: ###
        std_image = apply_gain(std_image, gain_array)

        ### Apply Multiplicative random noise: ###
        rand_noise = np.random.randn(h, w)
        noise_image = std_image * rand_noise #actually generate the noise

        ### Add noise to the bayer image: ###
        bayer = bayer + noise_image
        bayer = np.maximum(bayer, 0)
        bayer = np.minimum(bayer, 1)

        ### Again get bayer after INVERSE GAIN: ###
        bayer_before_wb = np.zeros((h, w))
        bayer_before_wb = apply_gain(bayer, inv_gain_array)
        std_image_estimated = compute_noise_stddev(bayer_before_wb, analog_gain, use_mean=True)  ### Difference is use_mean=True  ???
        std_image_estimated_after_wb = apply_gain(std_image_estimated, gain_array)

        ### What?: ###
        packed_bayer = pack_raw(bayer)
        packed_std = pack_raw(std_image_estimated_after_wb)
        bayer_std_packed = np.concatenate((packed_bayer,packed_std),axis=2)
        img_noise_extended = np.expand_dims(bayer_std_packed, axis=0) #->[1, H,W,C]
        img_gt_extended = np.expand_dims(img_gt, axis=0)


        ### Assign clean RGB and Noise: ###
        gt_rgb_images[ind] = np.float16(img_gt_extended)
        input_images_std[ind] = np.float16(img_noise_extended)



    if not os.path.isdir(gt_folder):
        os.makedirs(gt_folder)

    afile = open(gt_folder + 'gt_%d.pkl' %(epoch), 'wb')
    pickle.dump(gt_rgb_images, afile)
    afile.close()

    if not os.path.isdir(input_std_folder):
        os.makedirs(input_std_folder)

    afile = open(input_std_folder + 'input_std_%d.pkl' %(epoch), 'wb')
    pickle.dump(input_images_std, afile)
    afile.close()

    st = time.time()
    file2 = open(input_std_folder + 'input_std_%d.pkl' %(epoch), 'rb')
    new_d = pickle.load(file2)
    file2.close()
    print("Time=%.3f" % (time.time() - st))







































































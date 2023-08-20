#Imports:
#(1). Auxiliary:
import easydict
from easydict import EasyDict
import PIL
from PIL import Image
import argparse
import os
import numpy
import numpy as np
import math
import glob
from glob import glob
import random
import importlib
import collections
from collections import OrderedDict
import sys
from datetime import datetime
import cv2
#from skimage.measure import compare_ssim
from shutil import get_terminal_size
import pickle
# import lmdb
import ctypes  # An included library with Python install.


### TODO: change this back to pyplot!!!!!
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# plt.switch_backend('agg')
plt.switch_backend('TkAgg')
# plt.switch_backend('WXAgg')



from operator import contains
from functools import partial
from itertools import filterfalse
from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from sys import stdout
# import kornia
#
# import einops
import time
# import shapely
# from shapely.geometry import Point  #TODO: understand what's going on!!?!!!?
# import networkx
import re
from csv import reader
import tarfile
import copy
from copy import deepcopy
#import pydot
#import psutil
import shutil
from inspect import signature
# import numpngw
# import graphviz
# import pydot_ng
length = len #use length instead of len.... make sure it doesn't cause problems
from tqdm import tqdm
from easydict import EasyDict
from easydict import EasyDict as edict
from scipy.io import loadmat
import scipy.signal as signal
import numpy as np
import io
# import skimage

# Message Box:
def message_box(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
# import pymsgbox
# from pymsgbox import alert as alert_box
# from pymsgbox import confirm as confirm_box
# from pymsgbox import prompt as prompt_box

#(2). TorchVision (add FastAi stuff which are much faster and more intuitive as far as augmentations)
import torchvision
import torchvision.transforms as transforms
#(3). Torch Utils:
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
#(4). Torch NN:
import torch
import torch.fft
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
#(5). More Torch Stuff:
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import torch.utils.data as data
from torch.optim import Adam
import torch.cuda as cuda
#(7). TensorBoard:
# import tensorflow as tf
# import tensorboard as TB
# import tensorboardX as TBX
from torchvision import transforms
# from torchvision_x.transforms import functional as torchvisionX_F
# from tensorboardX import SummaryWriter

import re
import functools


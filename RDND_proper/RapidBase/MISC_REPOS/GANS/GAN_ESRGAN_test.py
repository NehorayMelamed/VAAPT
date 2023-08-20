#Imports:
#(1). Auxiliary:
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
import ctypes  # An included library with Python install.
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
#(3). Torch Utils:
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
#(4). Torch NN:
import torch.nn as nn
import torch.nn.functional as F
import torch


from collections import OrderedDict
import torch
import torch.nn as nn
import sys
import os
import math
from datetime import datetime
import numpy as np
# import cv2
from skimage.measure import compare_ssim
from torchvision.utils import make_grid
from shutil import get_terminal_size
import math
import pickle
import random
import numpy as np
import lmdb
import torch
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import torch.utils.data as data
import data.util as util

####################
# Utils
####################
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)



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


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def psnr(img1, img2):
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


def upscale_number_modified(input_number, upscale_factor, final_number_base_multiple, upper_clip_threshold):
    # Multiply input number by upscale_factor:
    final_number = int(input_number * upscale_factor)
    # Round it downwards if necesasry to make it a multiple of scale:
    final_number = (final_number // final_number_base_multiple) * final_number_base_multiple
    return upper_clip_threshold if final_number < upper_clip_threshold else final_number


# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_path, 'print_log.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Logger(object):
    def __init__(self, opt):
        self.exp_name = opt['name'] #experiment name
        self.use_tb_logger = opt['use_tb_logger'] #tb = tensorboard logger
        self.opt = opt['logger']
        self.log_dir = opt['path']['log']
        # loss log file
        self.loss_log_path = os.path.join(self.log_dir, 'loss_log.txt')
        with open(self.loss_log_path, 'a') as log_file:
            log_file.write('=============== Time: ' + get_timestamp() + ' =============\n')
            log_file.write('================ Training Losses ================\n')
        # val results log file
        self.val_log_path = os.path.join(self.log_dir, 'val_log.txt')
        with open(self.val_log_path, 'a') as log_file:
            log_file.write('================ Time: ' + get_timestamp() + ' ===============\n')
            log_file.write('================ Validation Results ================\n')
        if self.use_tb_logger and 'debug' not in self.exp_name:
            from tensorboard_logger import Logger as TensorboardLogger
            self.tb_logger = TensorboardLogger('../tb_logger/' + self.exp_name)

    def print_format_results(self, mode, rlt): #rlt????
        epoch = rlt.pop('epoch')
        iters = rlt.pop('iters')
        time = rlt.pop('time')
        model = rlt.pop('model')
        if 'lr' in rlt:
            lr = rlt.pop('lr')
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}, lr:{:.1e}> '.format(
                epoch, iters, time, lr)
        else:
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}> '.format(epoch, iters, time)

        ###########################
        #General Messages:        #
        for label, value in rlt.items():
            if mode == 'train':
                message += '{:s}: {:.2e} '.format(label, value)
            elif mode == 'val':
                message += '{:s}: {:.4e} '.format(label, value)
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                self.tb_logger.log_value(label, value, iters)

        # print in console
        print(message)
        # write in log file
        if mode == 'train':
            with open(self.loss_log_path, 'a') as log_file:
                log_file.write(message + '\n')
        elif mode == 'val':
            with open(self.val_log_path, 'a') as log_file:
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


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']





####################
# Files & IO
####################
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    #Look recursively at all current and children path's in given path and get all image files
    for dirpath, _, fnames in sorted(os.walk(path)): #os.walk!!!!
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path) #assert images is not an empty list
    return images


def _get_paths_from_lmdb(dataroot):
    #TODO: Understand again why this .lmdb suffix is a better file format and can be done with it!!
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    keys_cache_file = os.path.join(dataroot, '_keys_cache.p')

    #if lmdb keys file exists use it, and if not create it:
    if os.path.isfile(keys_cache_file):
        print('read lmdb keys from cache: {}'.format(keys_cache_file))
        keys = pickle.load(open(keys_cache_file, "rb"))
    else:
        with env.begin(write=False) as txn:
            print('creating lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, 'wb'))

    #Get paths from lmdb file:
    paths = sorted([key for key in keys if not key.endswith('.meta')])
    return env, paths


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'images':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def _read_lmdb_img(env, path):
    with env.begin(write=False) as txn:
        buf = txn.get(path.encode('ascii'))
        buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii') #encode and then decode? is this really efficient?
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = [int(s) for s in buf_meta.split(',')]
    img = img_flat.reshape(H, W, C)
    return img


def read_image(env, path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img






####################
# image processing
# process on numpy image
####################
#TODO: should probably use FastAI's API !
def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def convert_image_color_space(number_of_input_channels, target_type, image_list):
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


def crop_image_modified(input_image, crop_factor):
    #Crop from image image_size/crop_factor.
    #I think that the cropping is from the top left to the bottom right- check!

    # img_in: Numpy, HWC or HW
    input_image_numpy = np.copy(input_image)

    #If input image is a simple Matrix (1 channel):
    if input_image_numpy.ndim == 2:
        image_height, image_width = input_image_numpy.shape
        image_height_crop, image_width_crop = image_height % crop_factor, image_width % crop_factor
        input_image_numpy = input_image_numpy[:image_height - image_height_crop, :image_width - image_width_crop]
    #If input image is a multi channel image (>1 channels):
    elif input_image_numpy.ndim == 3:
        image_height, image_width, image_number_of_channels = input_image_numpy.shape
        image_height_crop, image_width_crop = image_height % crop_factor, image_width % crop_factor
        input_image_numpy = input_image_numpy[:image_height - image_height_crop, :image_width - image_width_crop, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img





####################
# Functions
####################
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







####################
# Create lmdb
####################
def create_lmdb_dataset():
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # configurations
    img_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800/*'  # glob matching pattern
    lmdb_save_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800.lmdb'  # must end with .lmdb

    img_list = sorted(glob.glob(img_folder))
    dataset = []
    data_size = 0

    print('Read images...')
    pbar = ProgressBar(len(img_list))
    for i, v in enumerate(img_list):
        pbar.update('Read {}'.format(v))
        img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        dataset.append(img)
        data_size += img.nbytes
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

    pbar = ProgressBar(len(img_list))
    with env.begin(write=True) as txn:  # txn is a Transaction object
        for i, v in enumerate(img_list):
            pbar.update('Write {}'.format(v))
            base_name = os.path.splitext(os.path.basename(v))[0]
            key = base_name.encode('ascii')
            data = dataset[i]
            if dataset[i].ndim == 2:
                H, W = dataset[i].shape
                C = 1
            else:
                H, W, C = dataset[i].shape
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




####################
# Spectral Normalization:
####################
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                # are the first left and right singular vectors.
                # This power iteration produces approximations of `u` and `v`.
                v = normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)

        u = normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)

        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} &= \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) &= \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    if dim is None:
        if isinstance(
                module,
            (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))






####################
# Extract Enlarged patches:
####################
def create_enlarged_patches_images_and_save_them():
    crt_path = os.path.dirname(os.path.realpath(__file__))

    # configurations
    h_start, h_len = 170, 64
    w_start, w_len = 232, 100
    enlarge_ratio = 3
    line_width = 2
    color = 'yellow'

    folder = os.path.join(crt_path, './ori/*')
    save_patch_folder = os.path.join(crt_path, './patch')
    save_rect_folder = os.path.join(crt_path, './rect')

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
    img_list = glob.glob(folder)
    images = []

    # make temp folder
    if not os.path.exists(save_patch_folder):
        os.makedirs(save_patch_folder)
        print('mkdir [{}] ...'.format(save_patch_folder))
    if not os.path.exists(save_rect_folder):
        os.makedirs(save_rect_folder)
        print('mkdir [{}] ...'.format(save_rect_folder))

    for i, path in enumerate(img_list):
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
        img_rect = cv2.rectangle(img, (w_start, h_start), (w_start + w_len, h_start + h_len),
            color, line_width)
        cv2.imwrite(os.path.join(save_rect_folder, base_name + '_rect.png'), img_rect)




def extract_subimgs_single():
    """A multi-thread tool to crop sub imags."""
    input_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800'
    save_folder = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub'
    n_thread = 20
    crop_sz = 480
    step = 240
    thres_sz = 48
    compression_level = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
                         args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')

    def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
        img_name = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))

        h_space = np.arange(0, h - crop_sz + 1, step)
        if h - (h_space[-1] + crop_sz) > thres_sz:
            h_space = np.append(h_space, h - crop_sz)
        w_space = np.arange(0, w - crop_sz + 1, step)
        if w - (w_space[-1] + crop_sz) > thres_sz:
            w_space = np.append(w_space, w - crop_sz)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                crop_img = np.ascontiguousarray(crop_img)
                # var = np.var(crop_img / 255)
                # if var > 0.008:
                #     print(img_name, index_str, var)
                cv2.imwrite(
                    os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(index))),
                    crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        return 'Processing {:s} ...'.format(img_name)






####################
# Loss Stuff:
####################
#(1). Define GAN Validity loss: [vanilla | lsgan | wgan-gp]
class GAN_validity_loss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        #Choose wanted gan Validity Loss:
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target is boolean:
                # if target=1 then we want the discriminator output to be 1 and want to minimize loss so we use -1*input.mean()
                # if target=0 then again wanting to minimize loss we will use a loss of +1*input.mean()
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))



    def get_target_label(self, input, flag_is_target_real):

        #If gan type is wgan-gp then we simply return the boolean itself
        if self.gan_type == 'wgan-gp':
            return flag_is_target_real

        #If gan type is NOT wgan-gp then we return a matrix the size of the discriminator output filled
        #with the correct label according to fag_is_target_real:
        if flag_is_target_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label_correct_size = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label_correct_size)
        return loss


#(2). Gradient Penalty Loss:
#TODO: look at my previous definitions of the gradient penalty loss!
class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()

        #TODO: what is register_buffer() property of the nn.Module Class???
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)


    #Get Gradient Outputs:
    #TODO: i've never seen grad_outputs.resize_().fill(1.0). why would we want to do that and in what situation:
    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs



    def forward(self, interp, interp_crit):
        # interp = a random alpha is assigned and interp is a linear combination of alpha*fake+(1-alpha)*real.
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit,
                                          inputs=interp,
                                          grad_outputs=grad_outputs,
                                          create_graph=True,
                                          retain_graph=True,
                                          only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1)**2).mean()
        return loss













####################
# Basic Network Blocks
####################
def Activation_Function(activation_type, negative_activation_slope=0.2, prelu_number_of_parameters=1, inplace=True):
    # helper selecting activation
    # negative_activation_slope: for leakyrelu and init of prelu
    # prelu_number_of_parameters: for p_relu num_parameters
    activation_type = activation_type.lower()
    if activation_type == 'relu':
        layer = nn.ReLU(inplace)
    elif activation_type == 'leakyrelu':
        layer = nn.LeakyReLU(negative_activation_slope, inplace)
    elif activation_type == 'prelu':
        layer = nn.PReLU(num_parameters=prelu_number_of_parameters, init=negative_activation_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return layer


def Normalization_Function(normalization_type, number_of_input_channels):
    # helper selecting normalization layer
    normalization_type = normalization_type.lower()
    if normalization_type == 'batch':
        layer = nn.BatchNorm2d(number_of_input_channels, affine=True)
    elif normalization_type == 'instance':
        layer = nn.InstanceNorm2d(number_of_input_channels, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % normalization_type)
    return layer


def Padding_Layer(padding_type, padding_size):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    padding_type = padding_type.lower()
    if padding_size == 0:
        return None
    if padding_type == 'reflect':
        layer = nn.ReflectionPad2d(padding_size)
    elif padding_type == 'replicate':
        layer = nn.ReplicationPad2d(padding_size)
    else:
        raise NotImplementedError('padding layer [%s] is not implemented' % padding_type)
    return layer


# Get needed padding to keep image size the same after a convolution layer with a certain kernel size and dilation (TODO: what about strid?1!?!)
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


# Concatenation Block:
# (*). A nn.Module subclass which is initialized with a Module, and when called upon it outputs a concatenation of that input with the Module output.
#     ... so it's basically a wrapper for Modules.
# (*). Notice - this basically means that module input and output sizes must be the same which is fine here because we're doing a pix2pix.
class Input_Output_Concat_Block(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    # Feed the module an input and output the module output concatenated with the input itself.
    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        tmpstr = 'Identity .. \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# Sum Block:
# (*). Again a very comfortable module wrapper. this time instead of outputing a concatenation (along the channels dim) of the input and output it sums them.
class Input_Output_Sum_Block(nn.Module):
    # Initialize this with a module
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


# Concatenate a list of input modules on top of each other to make on long module!:
def Pile_Modules_On_Top_Of_Each_Other(*args):
    # Flatten Sequential. It unwraps nn.Sequential.

    # If i only got one module (assuming it's a module) just return it:
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.

    # Take all the modules i got in the args input and simply concatenate them / pile them on top of each other and return a single "long" module.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# Basic Conv Block (every project has to have it's own....):
def Conv_Block(number_of_input_channels,
               number_of_output_channels,
               kernel_size,
               stride=1,
               dilation=1,
               groups=1,  # ?????!$@#%%$
               bias=True,
               padding_type='zero',
               # don't do padding .... this means the output size will be different from input size right?
               normalization_type=None,
               activation_type='relu',
               mode='CNA'):
    """
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    """

    # Check that input mode is one that's implemented
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [%s]' % mode

    # Pre-Padding:
    # (1). Get the padding size needed to pad the input with in order for the convolution operation output to be the same size as the origianl input:
    padding = get_valid_padding(kernel_size, dilation)
    # (2). Actually do the padding if wanted:
    padding_layer = pad(padding_type, padding) if (padding_type and padding_type != 'zero') else None
    padding = padding if padding_type == 'zero' else 0

    # Convolution:
    convolution_layer = nn.Conv2d(number_of_input_channels, number_of_output_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, \
                                  dilation=dilation, bias=bias, groups=groups)
    activation_layer = Activation_Function(activation_type) if activation_type else None

    # If the conv block mode is CNA Pile the layers on top of each other and output the pile:
    if 'CNA' in mode:
        normalization_layer = Normalization_Function(normalization_type,
                                                     number_of_output_channels) if normalization_type else None
        return Pile_Modules_On_Top_Of_Each_Other(padding_layer, convolution_layer, normalization_layer,
                                                 activation_layer)
    # If the conv block mode is NAC
    elif mode == 'NAC':
        if normalization_type is None and activation_type is not None:
            activation_layer = Activation_Function(activation_type, inplace=False)
            # Important! ???????????????????????????????????????????????????????????
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
            normalization_layer = Normalization_Function(normalization_type,
                                                         number_of_input_channels) if normalization_type else None
        return Pile_Modules_On_Top_Of_Each_Other(normalization_layer, activation_layer, padding_layer,
                                                 convolution_layer)


####################
# Useful blocks
####################
#(*). Regular ResNet Block (simple 3X3 conv):
class ResNet_Block(nn.Module):
    #################
    #TODO: have a list of all the relevant ResNet blocks, including the bottle necks and Residual Inception (as well as inception itself) blocks in one place!
    #################

    """
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    """

    #Initialization/Declaration Function for the "Regular" ResNet Block:
    def __init__(self,
                 number_of_input_channels,
                 number_of_middle_channels,
                 number_of_output_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 groups=1, \
                 bias=True,
                 padding_type='zero',
                 normalization_type=None,
                 activation_type='relu',
                 mode='CNA',
                 residual_branch_scaling=1):
        super(ResNet_Block, self).__init__()

        # First->Middle - a Convolutional Block
        conv0 = Conv_Block(number_of_input_channels, number_of_middle_channels, kernel_size, stride, dilation, groups,
                           bias, padding_type, normalization_type, activation_type, mode)

        # Middle->End
        if mode == 'CNA':
            # if Convolution-Normalization-Activation then continue the branch with Conv-Norm-Act-Conv-Norm
            activation_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            # if Conv-Norm-Act-Conv then continue the branch with Conv-Norm-Act-Conv-Conv ?!?!!?
            activation_type = None
            normalization_type = None
        conv1 = Conv_Block(number_of_middle_channels, number_of_output_channels, kernel_size, stride, dilation, groups,
                           bias, padding_type, normalization_type, activation_type, mode)

        # Maybe Implement again - a projector layer to keep number of channels consistent:
        # if number_of_input_channels != number_of_output_channels:
        #     self.project = conv_block(number_of_input_channels, number_of_output_channels, 1, stride, dilation, 1, bias, padding_type, \
        #         None, None)
        #     print('Need a projecter in ResNetBlock.')
        # else:
        #     self.project = lambda x:x

        # Pile the two Convolutions on top of each other to create the residual branch:
        self.residual_branch = Pile_Modules_On_Top_Of_Each_Other(conv0, conv1)
        self.residual_branch_scaling = residual_branch_scaling

    def forward(self, x):
        # against my initial intuition - the residual branch is callled the residual and not the main because it's assumed the mainly the input is passed with some residual
        residual_branch = self.residual_branch(x).mul(self.residual_branch_scaling)
        return x + residual_branch


#(*). Bottle-Necked Residual Block (1-3-1):
# TODO: add more scales / look for inception layers for pytorch!
class Res131(nn.Module):
    # Basic Residual Bottle Necked Layer (1conv-3conv-1conv);
    def __init__(self, number_of_input_channels, number_of_middle_channels, number_of_output_channels, dilation=1,
                 stride=1):
        super(Res131, self).__init__()

        # Convolution Branch:
        conv0 = Conv_Block(number_of_input_channels, number_of_middle_channels, 1, 1, 1, 1, False, 'zero', 'batch')
        conv1 = Conv_Block(number_of_middle_channels, number_of_middle_channels, 3, stride, dilation, 1, False, 'zero',
                           'batch')
        conv2 = Conv_Block(number_of_middle_channels, number_of_output_channels, 1, 1, 1, 1, False, 'zero', 'batch',
                           None)  # No ReLU
        self.res = Pile_Modules_On_Top_Of_Each_Other(conv0, conv1, conv2)
        if number_of_input_channels == number_of_output_channels:
            self.has_proj = False
        else:
            self.has_proj = True
            self.proj = Conv_Block(number_of_input_channels, number_of_output_channels, 1, stride, 1, 1, False, 'zero',
                                   'batch', None)
            #  No ReLU

    def forward(self, x):
        res = self.res(x)
        if self.has_proj:
            x = self.proj(x)
        return nn.functional.relu(x + res, inplace=True)



# Basic RDB block!:
class RDB(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    # TODO: number of convolutions (as well as their structure) is a Hyper-Parameter!$#$%%
    def __init__(self, number_of_input_channels, kernel_size=3, number_of_output_channels_for_each_conv_block=32,
                 stride=1, bias=True, padding_type='zero', \
                 normalization_type=None, activation_type='leakyrelu', mode='CNA'):
        # Note: number_of_output_channels_for_each_conv_block is not "really" that. it's the number of output channels for each of the individiaul Conv_Blocks.
        #      At the end the final output of the RDB is the size of the initial size (da...it's a residual block without a projector).

        super(RDB, self).__init__()
        # number_of_output_channels: growth channel, i.e. intermediate channels
        self.conv1 = Conv_Block(number_of_input_channels + 0 * number_of_output_channels_for_each_conv_block,
                                number_of_output_channels_for_each_conv_block, kernel_size, stride, bias=bias,
                                padding_type=padding_type, \
                                normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        self.conv2 = Conv_Block(number_of_input_channels + 1 * number_of_output_channels_for_each_conv_block,
                                number_of_output_channels_for_each_conv_block, kernel_size, stride, bias=bias,
                                padding_type=padding_type, \
                                normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        self.conv3 = Conv_Block(number_of_input_channels + 2 * number_of_output_channels_for_each_conv_block,
                                number_of_output_channels_for_each_conv_block, kernel_size, stride, bias=bias,
                                padding_type=padding_type, \
                                normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        self.conv4 = Conv_Block(number_of_input_channels + 3 * number_of_output_channels_for_each_conv_block,
                                number_of_output_channels_for_each_conv_block, kernel_size, stride, bias=bias,
                                padding_type=padding_type, \
                                normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = activation_type

        # At the final layer - RETURN TO INITIALE INPUT NUMBER OF CHANNELS!!!!
        self.conv5 = Conv_Block(number_of_input_channels + 4 * number_of_output_channels_for_each_conv_block,
                                number_of_input_channels, 3, stride, bias=bias, padding_type=padding_type, \
                                normalization_type=normalization_type, activation_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


# RRDB block!- an imrovement to the RDB block!:
class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, number_of_input_channels, kernel_size=3, number_of_output_channels_for_each_conv_block=32,
                 stride=1, bias=True, padding_type='zero', \
                 normalization_type=None, activation_type='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = RDB(number_of_input_channels, kernel_size, number_of_output_channels_for_each_conv_block, stride,
                        bias, padding_type, \
                        normalization_type, activation_type, mode)
        self.RDB2 = RDB(number_of_input_channels, kernel_size, number_of_output_channels_for_each_conv_block, stride,
                        bias, padding_type, \
                        normalization_type, activation_type, mode)
        self.RDB3 = RDB(number_of_input_channels, kernel_size, number_of_output_channels_for_each_conv_block, stride,
                        bias, padding_type, \
                        normalization_type, activation_type, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        # Again!- Notice that this is a residual block. final output size is the same as the input size. number_of_output_channels is only used internally.
        return out.mul(0.2) + x  # TODO: isn't the 0.2 a hyper parameter!??


####################
# Upsampler
####################
def Pixel_Shuffle_Block(number_of_input_channels, number_of_output_channels, upscale_factor=2, kernel_size=3, stride=1,
                        bias=True,
                        padding_type='zero', normalization_type=None, activation_type='relu'):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """

    # Instead of an explicit Upsample/ConvTranspose layer i do a trick with nn.PixelShuffle and hope for the best in terms of effectively implementing super pixel conv:
    conv_layer = Conv_Block(number_of_input_channels, number_of_output_channels * (upscale_factor ** 2), kernel_size,
                            stride, bias=bias,
                            padding_type=padding_type, normalization_type=None, activation_type=None)
    pixel_shuffle = nn.PixelShuffle(
        upscale_factor)  # Notice - now the output is of size (height*upsample_factor, width*upsample_factor)

    # After the Implicit Upsampling By Convolution and nn.PixelShuffle i use Normalization and Activation:
    normalization_layer = Normalization_Function(normalization_type,
                                                 number_of_output_channels) if normalization_type else None
    activation_layer = Activation_Function(activation_type) if activation_type else None
    return Pile_Modules_On_Top_Of_Each_Other(conv_layer, pixel_shuffle, normalization_layer, activation_layer)


# nn.Upsample -> Conv_Block Layer.
# TODO: understand the difference between this and the above pixel_shuffle_block
def Up_Conv_Block(number_of_input_channels, number_of_output_channels, upscale_factor=2, kernel_size=3, stride=1,
                  bias=True,
                  padding_type='zero', normalization_type=None, activation_type='relu', mode='nearest'):
    # Up conv
    # described in https://distill.pub/2016/deconv-checkerboard/
    upsample_layer = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv_layer = Conv_Block(number_of_input_channels, number_of_output_channels, kernel_size, stride, bias=bias,
                            padding_type=padding_type, normalization_type=normalization_type,
                            activation_type=activation_type)
    return Pile_Modules_On_Top_Of_Each_Other(upsample_layer, conv_layer)







########################################
# Perceptual Feature Extractors:
########################################
class VGGFeatureExtractor(nn.Module):
    # Assume input range is [0, 1]
    def __init__(self,
                 feature_layer=34, #the last Conv2d output (after which comes relu, maxpool and classifier)
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()

        #Choose the model wanted for feature extraction between the two vggs (vgg19 or vgg19 with BN):
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm

        #Normalize Channels to mean zero if wanted:
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)


        #Get the wanted feature layer from the vgg19 (or any other chosen model) model:
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        # Disable Gradient descent to feature extractor model as we don't necessarily want to change it:
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm

        # Normalize Channels to mean zero if wanted:
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        # Get the wanted feature layer from the ResNet101 model:
        self.features = nn.Sequential(*list(model.children())[:8]) #the model is divided into major concatenated blocks or children. get 1 through 8
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output



#A feature model that was perhapse trained by the authors.
#TODO: what is MINCNet???
class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out


# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval() #.eval() ??????
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output











########################################
# Generators:
########################################
#(1). RRDB Net:
class RRDB_Net(nn.Module):
    # Residual in Residual Dense Block Network (as described in the ESRGAN paper):
    def __init__(self,
                 number_of_input_channels,
                 number_of_output_channels,
                 number_of_input_channels_in_each_RRDB_block,
                 number_of_RRDB_blocks,
                 number_of_intermediate_channels_in_each_RRDB_block=32,
                 upscale_factor=4,
                 normalization_type=None,
                 activation_type='leakyrelu', \
                 mode='CNA',
                 res_scale=1,
                 upsample_mode='upconv'):

        # Activate Inherited Class initialization function (the nn.Module class):
        super(RRDB_Net, self).__init__()

        # Each Upscale layer we upscale by a factor of 2. so let's calculate the amount of times we need to do that in order to get upscale_factor:
        number_of_upscales = int(math.log(upscale_factor, 2))
        if upscale_factor == 3:
            number_of_upscales = 1

        #############
        # RRDB Stack:#
        # (1). Initial Conv Block As Initial Features Layer for RRDB Stack:
        initial_features_conv_block = Conv_Block(number_of_input_channels, number_of_input_channels_in_each_RRDB_block,
                                                 kernel_size=3, normalization_type=None, activation_type=None)
        # (2). RRDB Stack:
        RRDB_blocks = [RRDB(number_of_input_channels_in_each_RRDB_block,
                            kernel_size=3,
                            number_of_output_channels_for_each_conv_block=number_of_intermediate_channels_in_each_RRDB_block,
                            stride=1,
                            bias=True,
                            padding_type='zero',
                            normalization_type=normalization_type,
                            activation_type=activation_type,
                            mode='CNA')
                       for _ in range(
                number_of_RRDB_blocks)]  # output number of channels = number_of_input_channels_in_each_RRDB_block
        # (3). Final Conv Block
        LR_conv = Conv_Block(number_of_input_channels_in_each_RRDB_block, number_of_input_channels_in_each_RRDB_block,
                             kernel_size=3, normalization_type=normalization_type, activation_type=None, mode=mode)
        RRDB_stack = Pile_Modules_On_Top_Of_Each_Other(*RRDB_blocks, LR_conv);
        #############

        # Upsample (This is Super-Resolution After All):
        # (*). Choose Upsampling Method:
        if upsample_mode == 'upconv':
            upsample_block = Up_Conv_Block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = Pixel_Shuffle_Block()
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale_factor == 3:
            upsampler = upsample_block(number_of_input_channels_in_each_RRDB_block,
                                       number_of_input_channels_in_each_RRDB_block, 3, activation_type=activation_type)
        else:
            upsampler = [
                upsample_block(number_of_input_channels_in_each_RRDB_block, number_of_input_channels_in_each_RRDB_block,
                               activation_type=activation_type) for _ in range(number_of_upscales)]

        # Conv UpSampled Input:
        HR_conv0 = Conv_Block(number_of_input_channels_in_each_RRDB_block, number_of_input_channels_in_each_RRDB_block,
                              kernel_size=3, normalization_type=None, activation_type=activation_type)
        HR_conv1 = Conv_Block(number_of_input_channels_in_each_RRDB_block, number_of_output_channels, kernel_size=3,
                              normalization_type=None, activation_type=None)

        # Put Everything Together into a single module
        self.model = Pile_Modules_On_Top_Of_Each_Other(initial_features_conv_block,
                                                       Input_Output_Sum_Block(RRDB_stack),
                                                       *upsampler,
                                                       HR_conv0,
                                                       HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


#(2). Super Resolution ResNet Style:
class SRResNet(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, number_of_input_channels_in_each_ResNet_block, number_of_ResNet_blocks, upscale=4, normalization_type='batch', activation_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        #Get number of doubling upscales:
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        #Initiail Conv Block:
        fea_conv = Conv_Block(number_of_input_channels, number_of_input_channels_in_each_ResNet_block, kernel_size=3, normalization_type=None, activation_type=None)

        #ResNet Blocks Stack:
        resnet_blocks = [ResNet_Block(number_of_input_channels_in_each_ResNet_block,
                                      number_of_input_channels_in_each_ResNet_block,
                                      number_of_input_channels_in_each_ResNet_block,
                                      normalization_type=normalization_type,
                                      activation_type=activation_type,
                                      mode=mode,
                                      res_scale=res_scale)
                                            for _ in range(number_of_ResNet_blocks)]

        #Conv layer After ResNet Blocks Stack:
        LR_conv = Conv_Block(number_of_input_channels_in_each_ResNet_block, number_of_input_channels_in_each_ResNet_block, kernel_size=3, normalization_type=normalization_type, activation_type=None, mode=mode)

        #Choose Upsample Block to use (regular upconvolution or pixel shuffle approach):
        if upsample_mode == 'upconv':
            upsample_block = upsample_block
        elif upsample_mode == 'pixelshuffle':
            upsample_block = Pixel_Shuffle_Block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(number_of_input_channels_in_each_ResNet_block, number_of_input_channels_in_each_ResNet_block, 3, activation_type=activation_type)
        else:
            upsampler = [upsample_block(number_of_input_channels_in_each_ResNet_block, number_of_input_channels_in_each_ResNet_block, activation_type=activation_type) for _ in range(n_upscale)]


        #Final Two Conv Blocks After Upscaling to Fine Tune Results and Get Final HR Images:
        HR_conv0 = Conv_Block(number_of_input_channels_in_each_ResNet_block, number_of_input_channels_in_each_ResNet_block, kernel_size=3, normalization_type=None, activation_type=activation_type)
        HR_conv1 = Conv_Block(number_of_input_channels_in_each_ResNet_block, number_of_output_channels, kernel_size=3, normalization_type=None, activation_type=None)

        self.model = Pile_Modules_On_Top_Of_Each_Other(fea_conv, Input_Output_Sum_Block(Pile_Modules_On_Top_Of_Each_Other(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


#(3). SFT_Net
#(*). SFT linear transform (scale+shift) layer:
class SFTLayer(nn.Module): #Per Pixel Transformation Network
    def __init__(self):
        super(SFTLayer, self).__init__()
        #Per-Pixel Scale Inference Layer:
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 64, 1)
        #Per-Pixel Shift Inference Layer:
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 64, 1)

    #Take an input which has two parts: image features and segmentaion map prior. then use to try and inference a rectifying linear transform for each pixel (transformer network)
    def forward(self, x):
        # x[0]: fea (image convolutions) ; x[1]: cond (segmentation map prior):

        #Use Segmentation Prior Map to inference scale and shift:
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))

        #Use the inferenced scale and shift and transform the input image convolutions:
        return x[0] * (scale + 1) + shift


#(*). Residual SFT Block to connect residually the input with it's transformed self:
class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond

        #Spatially Transform and Conv input:
        x_scaled = self.sft0(x)
        x_scaled = F.relu(self.conv0(x_scaled), inplace=True)

        #Again! use the SFT layer with the originally used segmentation prior x[1]:
        x_scaled = self.sft1((x_scaled, x[1]))
        x_scaled = self.conv1(x_scaled)

        #Residually Connect the inferenced spatially transformed input and return that PLUS the original prior/conditional map:
        return (x[0] + x_scaled, x[1])  # return a tuple containing features and conditions



class SFT_Net(nn.Module):
    def __init__(self):
        super(SFT_Net, self).__init__()

        #Initial Convolutional Layer:
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)


        #Stack many Residual SFT blocks on top of each other (notice the original prior map is the one that's directly passed between the blocks):
        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT())

        #Add another SFT layer
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)


        #After the SFT stack use Pixel-Shuffle blocks to effectively Upsample the input:
        self.HR_branch = nn.Sequential(
            #TODO: understand the tradoffs between the different upsample options: number of parameters, speed, quality (filter overlap) etc'!!!$
            #Many Channels to Upsample Block:
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),

            #Many Channels to Upsample Block:
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),

            #Finally - simple convolutions to output the final HR image:
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )


        #CNN for the conditional/prior map (just like their paper says):
        self.CondNet = nn.Sequential(
            nn.Conv2d(8, 128, 4, 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        fea = self.conv0(x[0])
        res = self.sft_branch((fea, cond))
        fea = fea + res
        out = self.HR_branch(fea)
        return out











########################################
# Discriminators:
########################################
#(*). First VGG style discriminators for different sized inputs (TODO: get rid of the unecessary ones!!!!):
# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, number_of_input_channels, vgg_doubling_base_number_of_filters, normalization_type='batch', activation_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = Conv_Block(number_of_input_channels, vgg_doubling_base_number_of_filters, kernel_size=3, normalization_type=None, activation_type=activation_type, mode=mode)
        conv1 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 64, 64
        conv2 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters*2, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv3 = Conv_Block(vgg_doubling_base_number_of_filters*2, vgg_doubling_base_number_of_filters*2, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 32, 128
        conv4 = Conv_Block(vgg_doubling_base_number_of_filters*2, vgg_doubling_base_number_of_filters*4, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv5 = Conv_Block(vgg_doubling_base_number_of_filters*4, vgg_doubling_base_number_of_filters*4, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 16, 256
        conv6 = Conv_Block(vgg_doubling_base_number_of_filters*4, vgg_doubling_base_number_of_filters*8, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv7 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 8, 512
        conv8 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv9 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 4, 512
        self.features = Pile_Modules_On_Top_Of_Each_Other(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization (no BN!!!):
class Discriminator_VGG_128_SN(nn.Module):
    #Basic Building Block: Conv2d->Spectral_Norm->LeakyRelU
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
    def __init__(self, number_of_input_channels, vgg_doubling_base_number_of_filters, normalization_type='batch', activation_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = Conv_Block(number_of_input_channels, vgg_doubling_base_number_of_filters, kernel_size=3, normalization_type=None, activation_type=activation_type, mode=mode)
        conv1 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 48, 64
        conv2 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters*2, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv3 = Conv_Block(vgg_doubling_base_number_of_filters*2, vgg_doubling_base_number_of_filters*2, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 24, 128
        conv4 = Conv_Block(vgg_doubling_base_number_of_filters*2, vgg_doubling_base_number_of_filters*4, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv5 = Conv_Block(vgg_doubling_base_number_of_filters*4, vgg_doubling_base_number_of_filters*4, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 12, 256
        conv6 = Conv_Block(vgg_doubling_base_number_of_filters*4, vgg_doubling_base_number_of_filters*8, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv7 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 6, 512
        conv8 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv9 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 3, 512
        self.features = Pile_Modules_On_Top_Of_Each_Other(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, number_of_input_channels, vgg_doubling_base_number_of_filters, normalization_type='batch', activation_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = Conv_Block(number_of_input_channels, vgg_doubling_base_number_of_filters, kernel_size=3, normalization_type=None, activation_type=activation_type, mode=mode)
        conv1 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 96, 64
        conv2 = Conv_Block(vgg_doubling_base_number_of_filters, vgg_doubling_base_number_of_filters*2, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv3 = Conv_Block(vgg_doubling_base_number_of_filters*2, vgg_doubling_base_number_of_filters*2, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 48, 128
        conv4 = Conv_Block(vgg_doubling_base_number_of_filters*2, vgg_doubling_base_number_of_filters*4, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv5 = Conv_Block(vgg_doubling_base_number_of_filters*4, vgg_doubling_base_number_of_filters*4, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 24, 256
        conv6 = Conv_Block(vgg_doubling_base_number_of_filters*4, vgg_doubling_base_number_of_filters*8, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv7 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 12, 512
        conv8 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv9 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 6, 512
        conv10 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=3, stride=1, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        conv11 = Conv_Block(vgg_doubling_base_number_of_filters*8, vgg_doubling_base_number_of_filters*8, kernel_size=4, stride=2, normalization_type=normalization_type, activation_type=activation_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11)

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

        #VGG Style CNN:
        self.feature = nn.Sequential(
            #TODO: ask, even though i can make up excuses, why is it that at the first layer i don't see a BN layer almost ever?
            #TODO: ask if i'm correct to observe that there is a trend from regular convolution + max pool to a strided convolution to lower image size? why is that?
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


        #GAN Descriminator Validity Head (a fully connected layer which at the end outputs a single number)
        #TODO: understand how to make this fully convolutional (maybe simple use the patch GAN approach?!?!# seems very flexible)
        self.gan = nn.Sequential(
            nn.Linear(512*6*6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 1)
        )


        #Auxiliary Descriminator Classification Head (a fully connected netwrok which outputs 8 number for 8 classes?!@$?!$!#?$ where is the softmax if that's correct!$!%#@%):
        self.cls = nn.Sequential(
            nn.Linear(512*6*6, 100),
            nn.LeakyReLU(0.1, True),
            nn.Linear(100, 8)
        )

    
    def forward(self, x):
        fea = self.feature(x)
        fea = fea.view(fea.size(0), -1)
        gan = self.gan(fea)
        cls = self.cls(fea)
        return [gan, cls] #output regular validity score and the 8 numbers classification output







########################################
# Semantic(?) Segmentation Architectures:
########################################
class OutdoorSceneSeg(nn.Module):
    def __init__(self):
        super(OutdoorSceneSeg, self).__init__()
        # conv1
        blocks = []
        conv1_1 = Conv_Block(3, 64, 3, 2, 1, 1, False, 'zero', 'batch')  #  /2
        conv1_2 = Conv_Block(64, 64, 3, 1, 1, 1, False, 'zero', 'batch')
        conv1_3 = Conv_Block(64, 128, 3, 1, 1, 1, False, 'zero', 'batch')
        max_pool = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)  #  /2
        blocks = [conv1_1, conv1_2, conv1_3, max_pool]
        # conv2, 3 blocks

        #USE ResNet131 Block (with the 1X1 Convolution Bottle-Necks) Defined Above!:
        blocks.append(Res131(128, 64, 256))
        for i in range(2):
            blocks.append(Res131(256, 64, 256))
        # conv3, 4 blocks
        blocks.append(Res131(256, 128, 512, 1, 2))  #  /2
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









########################################
# Weight Initializations:
########################################
def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


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










################################################################################################################################################################
# Get Generator & Descriminator by using an OPT dictionary (i should get rid of that probably or rewrite it):
################################################################################################################################################################

####################
#### Generator  ####
def define_G(OPT):

    #Get Meta Options:
    gpu_ids = OPT['gpu_ids']
    opt_net = OPT['network_G']
    which_model = opt_net['which_model_G']

    #Choose Generator Model/Network to use:
    #(1). ResNet:
    if which_model == 'sr_resnet':  # SRResNet
        netG = SRResNet(number_of_input_channels=opt_net['number_of_input_channels'],
                        number_of_output_channels=opt_net['number_of_output_channels'],
                        number_of_input_channels_in_each_ResNet_blocks=opt_net['nf'],
                        number_of_ResNet_blocks=opt_net['nb'],
                        upscale=opt_net['scale'],
                        normalization_type=opt_net['normalization_type'],
                        activation_type='relu',
                        mode=opt_net['mode'],
                        upsample_mode='pixelshuffle')
    #(2). SFT (Spatial Feature Transformer):
    elif which_model == 'sft_arch':  # SFT-GAN
        netG = SFT_Net()
    #(3). RRDB (New Proposed Net Without BN):
    elif which_model == 'RRDB_net':  # RRDB
        netG = RRDB_Net(number_of_input_channels=opt_net['number_of_input_channels'],
                        number_of_output_channels=opt_net['number_of_output_channels'],
                        number_of_input_channels_in_each_RRDB_block=opt_net['nf'],
                        number_of_RRDB_blocks=opt_net['nb'],
                        number_of_intermediate_channels_in_each_RRDB_block=opt_net['gc'],
                        upscale_factor=opt_net['scale'],
                        normalization_type=opt_net['normalization_type'],
                        activation_type='leakyrelu',
                        mode=opt_net['mode'],
                        upsample_mode='upconv') #TODO: add a HyperParameter called residual_branch_scaling to RRDB & RRDB_Net:
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))


    #If we're in training mode then perform weights initialization according to desired way:
    if OPT['is_train']:
        init_weights(netG, init_type='kaiming', scale=0.1)


    #If there are several gpus (with gpu_ids) then use them and Distribute the model:
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG



########################
#### Discriminator  ####
def define_D(OPT):

    #Get Meta Options:
    gpu_ids = OPT['gpu_ids']
    opt_net = OPT['network_D']
    which_model = opt_net['which_model_D']


    # Choose Discriminator Model/Network to use:
    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(number_of_input_channels=opt_net['number_of_input_channels'],
                                          vgg_doubling_base_number_of_filters=opt_net['nf'],
                                          normalization_type=opt_net['normalization_type'],
                                          mode=opt_net['mode'],
                                          activation_type=opt_net['activation_type'])

    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(number_of_input_channels=opt_net['number_of_input_channels'],
                                         vgg_doubling_base_number_of_filters=opt_net['nf'],
                                         normalization_type=opt_net['normalization_type'],
                                         mode=opt_net['mode'],
                                         activation_type=opt_net['activation_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(number_of_input_channels=opt_net['number_of_input_channels'],
                                          vgg_doubling_base_number_of_filters=opt_net['nf'],
                                          normalization_type=opt_net['normalization_type'],
                                          mode=opt_net['mode'],
                                          activation_type=opt_net['activation_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    #Initialize The Weights (TODO: add a test time possibility because i would like to probe the discriminator):

    #If there are several gpus (with gpu_ids) then use them and Distribute the model:
    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


############################
#### Feature Extractor: ####
def define_F(OPT, use_bn=False):

    #Get Meta Options:
    gpu_ids = OPT['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')


    #Choose Wanted Layer according to feature layer extracted
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34


    #Get the Feature Extractor:
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                                    use_bn=use_bn,
                                    use_input_norm=True,
                                    device=device)
    # netF = ResNet101FeatureExtractor(use_input_norm=True, device=device)


    #If there are several gpus (with gpu_ids) then use them and Distribute the model:
    if gpu_ids:
        netF = nn.DataParallel(netF)


    #Return Feature_Extractor.eval() because we're not training it:
    #TODO: what does Net.eval() return!?$@#%^?#$$?
    netF.eval()
    return netF














################################################################################################################################################################
# Model Wrappers:
################################################################################################################################################################
#(1). Only SR itself / Only the Generator Itself!:
class SRModel:

    def __init__(self, OPT):

        #High Level Parameters:
        self.OPT = OPT
        self.save_dir = OPT['path']['models']  # save models
        self.device = torch.device('cuda' if OPT['gpu_ids'] is not None else 'cpu')
        self.flag_is_training = OPT['is_train']
        self.schedulers = []
        self.optimizers = [] #TODO: why a list of optimizers!?$?@#
        train_OPT = OPT['train'] #the "train" branch of the OPT dictionary

        #Get Generator By Passing into define_G the dictionary OPT:
        self.netG = networks.define_G(OPT).to(self.device)

        #Load Existing Generator Network:
        self.load_existing_generator_network()


        #If we're in training model:
        if self.flag_is_training:

            #Tell the Network to be in training mode:
            self.netG.train()

            #Define the Loss and Loss-Weights:
            loss_type = train_OPT['pixel_criterion']
            if loss_type == 'l1':
                self.direct_pixel_loss_function = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.direct_pixel_loss_function = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.lambda_direct_pixel_loss = train_OPT['pixel_weight']

            #Optimizers:
            #(1). Weight Decay:
            generator_weight_decay = train_OPT['weight_decay_G'] if train_OPT['weight_decay_G'] else 0
            #(2). Look at the generator model parameters and declare those parameters which we do not optimize:
            optimizer_parameters = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optimizer_parameters.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            #(3). Actually get the generator optimizer:
            self.optimizer_G = torch.optim.Adam(optimizer_parameters, lr=train_OPT['lr_G'], weight_decay=generator_weight_decay)
            self.optimizers.append(self.optimizer_G)


            #Learning Rate Schedualers:
            if train_OPT['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, train_OPT['lr_steps'], train_OPT['lr_gamma'])) #TODO: perhapse change to FastAI schedualer
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')


            #Initialize log dictionary for learning tracking:
            self.log_dict = OrderedDict()


        #Declare Model Initialization and Print Network Description:
        print('---------- Model initialized ------------------')
        self.print_network()
        print('----------------    -------------------------------')



    #################################
    ##### FEED-DATA FUNCTION: #######
    #################################
    def feed_data(self, data, flag_need_HR=True):
        #Feed data into internal parameters var_L (what kind of stupid name is that?!?#@)
        self.real_image_batch_LR = data['LR'].to(self.device)  # LR
        if flag_need_HR:
            self.real_image_batch_HR = data['HR'].to(self.device)  # HR


    #########################################
    ##### Back-Propagate / Train!!!!: #######
    #########################################
    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        current_pixel_loss = self.lambda_direct_pixel_loss * self.direct_pixel_loss_function(self.fake_H, self.real_H)
        current_pixel_loss.backward()
        self.optimizer_G.step()
        #Update Log Dictionary with Current Loss:
        self.log_dict['l_pix'] = current_pixel_loss.item()
    # Auxiliary Function to get current log dictionary:
    def get_current_log(self):
        return self.log_dict


    ###################################################################################################
    ##### Test Generator: #######
    ###################################################################################################
    #(*). Spot-Test: By Taking current real LR image batch and generating HR batch
    def test(self):
        self.netG.eval() #Evaluation/Test Mode!
        with torch.no_grad(): #TODO: understand in what instances do we use torch.no_grad()
            self.fake_image_batch_HR = self.netG(self.real_image_batch_LR)
        self.netG.train() #Get Back To Training Mode!


    #(*). Throughly Test:
    def test_x8(self):

        #Change Network State to Evaluation/Test Mode and Disable Gradient Descent:
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False


        #Auxiliary Transform Function:
        #TODO: check if the double conversion is necessary... there must be a better streamlined preprocessing API:
        def _transform(tensor_variable, transform_operation):
            # if self.precision != 'single': v = v.float()
            tensor_to_numpy = tensor_variable.data.cpu().numpy()
            if transform_operation == 'vertical_flip':
                transformed_numpy = tensor_to_numpy[:, :, :, ::-1].copy()
            elif transform_operation == 'horizontal_flip':
                transformed_numpy = tensor_to_numpy[:, :, ::-1, :].copy()
            elif transform_operation == 'transponse':
                transformed_numpy = tensor_to_numpy.transpose((0, 1, 3, 2)).copy()
            transformed_numpy_to_tensor = torch.Tensor(transformed_numpy).to(self.device)
            # if self.precision == 'half': ret = ret.half()
            return transformed_numpy_to_tensor


        #Turn Current LR image batch into a list:
        current_image_batch_LR_list = [self.real_image_batch_LR]
        #Take current LR image batch list and add to that list the transformed images for each image in the list:
        for tf in 'vertical_flip', 'horizontal_flip', 'transponse':
            current_image_batch_LR_list.extend([_transform(t, tf) for t in current_image_batch_LR_list])
        #Generate the fake HR images:
        fake_image_batch_HR_list = [self.netG(aug) for aug in current_image_batch_LR_list]
        #(?_)
        #TODO: why the hell the weird transform contingencies!?$@?$%?
        for i in range(len(fake_image_batch_HR_list)):
            if i > 3:
                fake_image_batch_HR_list[i] = _transform(fake_image_batch_HR_list[i], 'transponse')
            if i % 4 > 1:
                fake_image_batch_HR_list[i] = _transform(fake_image_batch_HR_list[i], 'horizontal_flip')
            if (i % 4) % 2 == 1:
                fake_image_batch_HR_list[i] = _transform(fake_image_batch_HR_list[i], 'vertical_flip')

        output_cat = torch.cat(fake_image_batch_HR_list, dim=0)
        self.fake_image_batch_HR = output_cat.mean(dim=0, keepdim=True) #What? why take the mean over zero index (batch index?)?
        #Reinstate gradient descent possibility:
        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train() #Return network mode to Train!


    #Get current results (real LR, real HR, fake HR):
    def get_current_visuals(self, flag_need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.real_image_batch_LR.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_image_batch_HR.detach()[0].float().cpu()
        if flag_need_HR:
            out_dict['HR'] = self.real_image_batch_HR.detach()[0].float().cpu()
        return out_dict


    ########################################
    ##### PRINT NETWORK DESCRIPTION: #######
    ########################################
    #General Function To Get Network Description using the __str__ descriptor and the network parameters:
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
    #Print Network Description To Console:
    def print_network(self):
        generator_string_description, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            current_generator_string_description = '-------------- Generator --------------\n' + generator_string_description + '\n'
            generator_description_path = os.path.join(self.save_dir, '../', 'network.txt') #TODO: change this to save everything i need in one folder without too many subfolder schemes!
            with open(generator_description_path, 'w') as f:
                f.write(current_generator_string_description)



    ########################################
    ##### LOAD PRE-EXISTING NETWORK: #######
    ########################################
    #General Function To Load Network Defined in load_path into passed in "network" object:
    def load_network(self, load_path, network, strict=True):
        #TODO: what going on here?
        if isinstance(network, nn.DataParallel):
            network = network.module
        #Load the Pre-Existing Network:
        #TODO: strict?!$?!$
        network.load_state_dict(torch.load(load_path), strict=strict)
    #Load Wanted Pre-Trained Generator Into self.netG using the auxiliary function load_network:
    def load_existing_generator_network(self):
        load_path_G = self.OPT['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)


    #####################################
    ###### SAVE CURRENT NETWORK: ########
    #####################################
    #General Function To Save Network passed into the function in the save_dir directory:
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
    #Save the Currently Used netG Generator function into the initialized save_dir directory:
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)




###############################################################################################################################################################
# (2). SR-GAN (Generator + Descriminator):
class SRGAN_Model:

    def __init__(self, OPT):

        # High Level Parameters:
        self.OPT = OPT
        self.save_dir = OPT['path']['models']  # save models
        self.device = torch.device('cuda' if OPT['gpu_ids'] is not None else 'cpu')
        self.flag_is_training = OPT['is_train']
        self.schedulers = []
        self.optimizers = []  # TODO: why a list of optimizers!?$?@#
        train_OPT = OPT['train']  # the "train" branch of the OPT dictionary

        # Get Generator & Descriminators By Passing into define_G & define_D the dictionary OPT:
        self.netG = networks.define_G(OPT).to(self.device)
        # Tell the Networks to be in training mode:
        if self.flag_is_training:
            self.netD = define_D(OPT).to(self.device);
            self.netF = define_F(OPT, use_bn=Flse).to(self.device)
            self.netG.train()
            self.netD.train();
        #Load Pre-Existing Networks:
        load_generator_and_descriminator_networks();



        #If we're in training mode then get all the losses and optimizers we need for training:
        if self.flag_is_training:
            ###################
            # LOSSES: #########
            ###################
            #Generator Direct Pixel Loss
            direct_pixel_loss_type = train_OPT['pixel_criterion']
            self.lambda_direct_pixel_loss = train_OPT['pixel_weight'];
            if direct_pixel_loss_type == 'l1':
                self.direct_pixel_loss_function = nn.L1Loss().to(self.device)
            elif direct_pixel_loss_type == 'l2':
                self.direct_pixel_loss_function = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(direct_pixel_loss_type))

            #Generator Feature (Perceptual) Loss:
            feature_space_loss_type = train_OPT['feature_criterion'];
            self.lambda_feature_space_loss = train_OPT['feature_weight'];
            if feature_space_loss_type == 'l1':
                self.feature_space_loss_function = nn.L1Loss().to(self.device)
            elif feature_space_loss_type == 'l2':
                self.feature_space_loss_function = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(feature_space_loss_type))


            #GAN/Descriminator Validity Loss:
            self.lambda_discriminator_validity_loss = train_OPT['gan_weight'];
            self.GAN_validity_loss_function = GAN_validity_loss(train_OPT['gan_type'],1.0,0.0).to(self.device) #Gan_validity_loss is an above defined function which return a pytorch loss function to our choosing


            #D/G update ratio (to try and avoid mode collapse):
            self.D_update_ratio = train_OPT['D_update_ratio'] if train_OPT['D_update_ratio'] else 1
            self.D_init_iters = train_OPT['D_init_iters'] if train_OPT['D_init_iters'] else 0


            #Gradient Penalty Loss:
            if train_OPT['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.lambda_gradient_penalty_loss = train_OPT['gp_weight'];
                self.gradient_penalty_loss_function = GradientPenaltyLoss(device=self.device).to(self.device);



            #######################
            # Optimizers: #########
            #######################
            #(*). Generator Optimizer:
            generator_weight_decay = train_OPT['weight_decay_G'] if train_OPT['weight_decay_G'] else 0
            #Look at the generator model parameters and declare those parameters which we do not optimize:
            optimizer_parameters = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optimizer_parameters.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            # (3). Actually get the generator optimizer:
            self.optimizer_G = torch.optim.Adam(optimizer_parameters, lr=train_OPT['lr_G'], weight_decay=generator_weight_decay)
            self.optimizers.append(self.optimizer_G)

            #(*). Descriminator Optimizer:
            descriminator_weight_decay = train_OPT['weight_decay_D'] if train_OPT['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)


            # Learning Rate Schedualers:
            if train_OPT['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, train_OPT['lr_steps'], train_OPT['lr_gamma']))  # TODO: perhapse change to FastAI schedualer
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            # Initialize log dictionary for learning tracking:
            self.log_dict = OrderedDict()

        # Declare Model Initialization and Print Network Description:
        print('---------- Model initialized ------------------')
        self.print_network()
        print('----------------    -------------------------------')




    #################################
    ##### FEED-DATA FUNCTION: #######
    #################################
    def feed_data(self, data, flag_need_HR=True):
        self.real_image_batch_LR = data['LR'].to(self.device)  # LR
        if flag_need_HR:
            self.real_image_batch_HR = data['HR'].to(self.device)  # HR
            input_reference = data['ref'] if 'ref' in data else data['HR']
            self.real_image_batch_reference = input_reference.to(self.device);



    #########################################
    ##### Back-Propagate / Train!!!!: #######
    #########################################
    def optimize_parameters(self, step):
        ######################
        #Optimize Generator:##
        #(1). Turn Off Descriminator Parameters Update:
        for p in self.netD.parameters():
            p.requires_grad = False
        #(2). Zero Generator gradient information and Generate fake HR image batch:
        self.optimizer_G.zero_grad()
        self.fake_image_batch_HR = self.netG(self.real_image_batch_LR)
        #Get Generator Loss (once for every D_update_ratio Descriminator optimization steps):
        total_generator_loss = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            #Direct Pixel Loss:
            if self.direct_pixel_loss_function:
                current_generator_direct_pixel_loss = self.lambda_direct_pixel_loss * self.direct_pixel_loss_function(self.fake_image_batch_HR, self.real_image_batch_HR)
                total_generator_loss += current_generator_direct_pixel_loss
            #Feature Space Loss:
            if self.feature_space_loss_function:
                real_image_batch_HR_features_extracted = self.netF(self.real_image_batch_HR).detach()
                fake_image_batch_HR_features_extracated = self.netF(self.fake_image_batch_HR)
                current_generator_feature_space_loss = self.lambda_feature_space_loss * self.feature_space_loss_function(fake_image_batch_HR_features_extracated, real_image_batch_HR_features_extracted)
                total_generator_loss += current_generator_feature_space_loss
            #Validity Loss:
            D_output_real_image_batch_validity = self.netD(self.real_image_batch_reference)
            D_output_fake_image_batch_validity = self.netD(self.fake_image_batch_HR).detach()
            current_generator_validity_loss = self.lambda_discriminator_validity_loss * \
                      (self.GAN_validity_loss_function(D_output_real_image_batch_validity - torch.mean(D_output_fake_image_batch_validity), False) +
                       self.GAN_validity_loss_function(D_output_fake_image_batch_validity - torch.mean(D_output_real_image_batch_validity), True)) / 2
            total_generator_loss += current_generator_validity_loss
            #Back-Propagate:
            total_generator_loss.backward()
            self.optimizer_G.step()
            #Log Results:
            if self.direct_pixel_loss_function:
                self.log_dict['generator_direct_pixel_loss'] = current_generator_direct_pixel_loss.item()
            if self.feature_space_loss_function:
                self.log_dict['generator_feature_space_loss'] = current_generator_feature_space_loss.item()
            self.log_dict['generator_GAN_validity_loss'] = current_generator_validity_loss.item()

        #########################
        #Optimize Descriminator:#
        #(1). Turn Back On Descriminator Parameters Update:
        for p in self.netD.parameters():
            p.requires_grad = True
        #(2). Zero Descriminator gradient information and Generate fake HR image batch:
        self.optimizer_D.zero_grad()
        total_descriminator_loss = 0;
        #Validity Loss:
        D_output_real_image_batch_validity = self.netD(self.real_image_batch_reference)
        D_output_fake_image_batch_validity = self.netD(self.fake_image_batch_HR).detach()
        current_descriminator_real_images_validity_loss = self.GAN_validity_loss_function(D_output_real_image_batch_validity - torch.mean(D_output_fake_image_batch_validity), False);
        current_descriminator_fake_images_validity_loss = self.GAN_validity_loss_function(D_output_fake_image_batch_validity - torch.mean(D_output_real_image_batch_validity), True);
        current_descriminator_validity_loss = self.lambda_discriminator_validity_loss * (current_descriminator_fake_images_validity_loss + current_descriminator_real_images_validity_loss)/2;
        total_descriminator_loss += current_descriminator_validity_loss
        #Gradient Penalty Loss:
        if self.OPT['train']['gan_type'] == 'wgan-gp':
            batch_size = self.real_image_batch_reference.size(0);
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interpolated_image = self.random_pt * self.fake_image_batch_HR.detach() + (1-self.random_pt) * self.real_image_batch_reference;
            interpolated_image.requires_grad = True;
            D_output_reference_image_batch_validity,_ = self.netD(interpolated_image);
            current_descriminator_gradient_penalty_loss = self.gradient_penalty_loss_function(interpolated_image, D_output_reference_image_batch_validity)
            total_descriminator_loss += current_descriminator_gradient_penalty_loss;
        #Back-Propagate:
        total_descriminator_loss.backward();
        self.optimizer_D.step();
        #Log Result:
        self.log_dict['descriminator_real_images_validity_loss'] = current_descriminator_real_images_validity_loss.item()
        self.log_dict['descriminator_fake_images_validity_loss'] = current_descriminator_fake_images_validity_loss.item()
        if self.OPT['train']['gan_type'] == 'wgan-gp':
            self.log_dict['descriminator_gradient_penalty_loss'] = current_descriminator_gradient_penalty_loss.item()
        self.log_dict['descriminator_output_real_images'] = torch.mean(D_output_real_image_batch_validity.detach())
        self.log_dict['descriminator_output_fake_images'] = torch.mean(D_output_fake_image_batch_validity.detach())



    ###################################################################################################
    ##### Test Generator: #######
    ###################################################################################################
    # (*). Spot-Test: By Taking current real LR image batch and generating HR batch
    def test(self):
        self.netG.eval()  # Evaluation/Test Mode!
        with torch.no_grad():  # TODO: understand in what instances do we use torch.no_grad()
            self.fake_image_batch_HR = self.netG(self.real_image_batch_LR)
        self.netG.train()  # Get Back To Training Mode!

    # (*). Throughly Test:
    def test_x8(self):

        # Change Network State to Evaluation/Test Mode and Disable Gradient Descent:
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False

        # Auxiliary Transform Function:
        # TODO: check if the double conversion is necessary... there must be a better streamlined preprocessing API:
        def _transform(tensor_variable, transform_operation):
            # if self.precision != 'single': v = v.float()
            tensor_to_numpy = tensor_variable.data.cpu().numpy()
            if transform_operation == 'vertical_flip':
                transformed_numpy = tensor_to_numpy[:, :, :, ::-1].copy()
            elif transform_operation == 'horizontal_flip':
                transformed_numpy = tensor_to_numpy[:, :, ::-1, :].copy()
            elif transform_operation == 'transponse':
                transformed_numpy = tensor_to_numpy.transpose((0, 1, 3, 2)).copy()
            transformed_numpy_to_tensor = torch.Tensor(transformed_numpy).to(self.device)
            # if self.precision == 'half': ret = ret.half()
            return transformed_numpy_to_tensor

        # Turn Current LR image batch into a list:
        current_image_batch_LR_list = [self.real_image_batch_LR]
        # Take current LR image batch list and add to that list the transformed images for each image in the list:
        for tf in 'vertical_flip', 'horizontal_flip', 'transponse':
            current_image_batch_LR_list.extend([_transform(t, tf) for t in current_image_batch_LR_list])
        # Generate the fake HR images:
        fake_image_batch_HR_list = [self.netG(aug) for aug in current_image_batch_LR_list]
        # (?_)
        # TODO: why the hell the weird transform contingencies!?$@?$%?
        for i in range(len(fake_image_batch_HR_list)):
            if i > 3:
                fake_image_batch_HR_list[i] = _transform(fake_image_batch_HR_list[i], 'transponse')
            if i % 4 > 1:
                fake_image_batch_HR_list[i] = _transform(fake_image_batch_HR_list[i], 'horizontal_flip')
            if (i % 4) % 2 == 1:
                fake_image_batch_HR_list[i] = _transform(fake_image_batch_HR_list[i], 'vertical_flip')

        output_cat = torch.cat(fake_image_batch_HR_list, dim=0)
        self.fake_image_batch_HR = output_cat.mean(dim=0, keepdim=True)  # What? why take the mean over zero index (batch index?)?
        # Reinstate gradient descent possibility:
        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train()  # Return network mode to Train!

    # Get current results (real LR, real HR, fake HR):
    def get_current_visuals(self, flag_need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.real_image_batch_LR.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_image_batch_HR.detach()[0].float().cpu()
        if flag_need_HR:
            out_dict['HR'] = self.real_image_batch_HR.detach()[0].float().cpu()
        return out_dict


    ########################################
    ##### PRINT NETWORK DESCRIPTION: #######
    ########################################
    # General Function To Get Network Description using the __str__ descriptor and the network parameters:
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    # Print Network Description To Console:
    def print_network(self):
        generator_string_description, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            current_generator_string_description = '-------------- Generator --------------\n' + generator_string_description + '\n'
            generator_description_path = os.path.join(self.save_dir, '../', 'network.txt')  # TODO: change this to save everything i need in one folder without too many subfolder schemes!
            with open(generator_description_path, 'w') as f:
                f.write(current_generator_string_description)


    ########################################
    ##### LOAD PRE-EXISTING NETWORK: #######
    ########################################
    # General Function To Load Network Defined in load_path into passed in "network" object:
    def load_network(self, load_path, network, strict=True):
        # TODO: what going on here?
        if isinstance(network, nn.DataParallel):
            network = network.module
        # Load the Pre-Existing Network:
        # TODO: strict?!$?!$
        network.load_state_dict(torch.load(load_path), strict=strict)
    # Load Wanted Pre-Trained Generator & Descriminator:
    def load_generator_and_descriminator_networks(self):
        load_path_G = self.OPT['path']['pretrain_model_G']
        if load_path_G is not None:
            print('loading model for G [{:s}}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        load_path_D = self.OPT['path']['pretrain_model_D']
        if self.OPT['is_train'] and load_path_D is not None:
            print('loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD)

    #####################################
    ###### SAVE CURRENT NETWORK: ########
    #####################################
    # General Function To Save Network passed into the function in the save_dir directory:
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # Save the Currently Used netG Generator function into the initialized save_dir directory:
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # Auxiliary Function to get current log dictionary:
    def get_current_log(self):
        return self.log_dict
################################################################################################################################################################







################################################################################################################################################################
# Data-Loading: (TODO: understand if this is any good in terms of speed!)
################################################################################################################################################################
class Creat_DataSet(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, OPT):
        super(LRHRDataset, self).__init__()
        #Options:
        self.OPT = OPT
        #Paths to LR and HR image file names:
        self.paths_LR = None
        self.paths_HR = None
        #Environment for lmdb
        self.LR_env = None
        self.HR_env = None

        #Get Image File-Names List:
        #(1). From txt file get only HR images file names witout LR images (only generate LR images on the fly):
        if OPT['subset_file'] is not None and OPT['phase'] == 'train':
            with open(OPT['subset_file']) as txt_file:
                self.paths_HR = sorted([os.path.join(OPT['dataroot_HR'], line.rstrip('\n')) for line in txt_file])
            if OPT['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        #(2). Read both HR & LR images file names (from lmdb or file name list):
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = get_image_paths(OPT['data_type'], OPT['dataroot_HR'])
            self.LR_env, self.paths_LR = get_image_paths(OPT['data_type'], OPT['dataroot_LR'])

        #In any case we need the HR image path as this is suppervised learning after all- so Check the HR images paths list exists:
        assert self.paths_HR, 'Error: HR path is empty.'

        #Check LR & HR paths are equal size - If i use pre-existing LR and HR images (not generating LR images) then make sure they're of equal length (for each LR hae a HR):
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        #See Below:
        self.random_scale_list = [1]


    #OverWrite of the __getiten__ functionality:
    def __getitem__(self, index):

        #Initialize Parameters:
        HR_path, LR_path = None, None
        image_rescale_factor = self.OPT['scale']
        HR_size = self.OPT['HR_size']

        ##################
        #Get HR image:  ##
        ##################
        HR_path = self.paths_HR[index]
        current_image_HR = read_image(self.HR_env, HR_path)
        #modcrop in the validation / test phase
        if self.OPT['phase'] != 'train':
            current_image_HR = crop_image_modified(current_image_HR, image_rescale_factor)
        #change color space if necessary
        if self.OPT['color']:
            current_image_HR = convert_image_color_space(current_image_HR.shape[2], self.OPT['color'], [current_image_HR])[0]

        #################
        #Get LR image:  #
        #################
        #TODO!!!: Fix all of this messy legacy code with a simple HR/LR pair grabbing API including the possibility to initially train on smaller size images as is implied below!
        #(1). By directly sampling LR image file if it exists and that's how we want to do it:
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            current_image_LR = read_image(self.LR_env, LR_path)
        #(2). By Down Sampling the HR image on-the-fly:
        else:
            #(*). First Resize HR image and basically just make sure it is divisible by image_rescale_factor:
            if self.OPT['phase'] == 'train':
                #Choose upscale image resize factor:
                #(NOTE!: usually random_scale will be 1 as by default self.random_scale_list is a list of 1's.
                #        a reason you would want a different number is to simply train the network on a different sized HR input image to see what happens!!!)
                random_original_HR_image_upscale_factor = random.choice(self.random_scale_list) #Choose one random element from self.random_scale_list list
                H_s, W_s, _ = current_image_HR.shape

                #Upscale/Resize HR Image to make it divisible by the HR_size/LR_size=image_rescale_factor:
                H_s = upscale_number_modified(input_number=H_s, upscale_factor=random_original_HR_image_upscale_factor, final_number_base_multiple=image_rescale_factor, upper_clip_threshold=HR_size)
                W_s = upscale_number_modified(input_number=W_s, upscale_factor=random_original_HR_image_upscale_factor, final_number_base_multiple=image_rescale_factor, upper_clip_threshold=HR_size)
                current_image_HR = cv2.resize(np.copy(current_image_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR) #why the np.copy(current_image_HR)?
                #Force to 3 channels if our current_image_HR is BW:
                if current_image_HR.ndim == 2:
                    current_image_HR = cv2.cvtColor(current_image_HR, cv2.COLOR_GRAY2BGR)

            #(*). Get LR from HR - Use matlab's imresize to get LR image by resizing HR image downwards by image_rescale_factor to create the LR image on the fly:
            image_height, image_width, _ = current_image_HR.shape
            current_image_LR = imresize_np(current_image_HR, 1 / image_rescale_factor, True)
            if current_image_LR.ndim == 2:
                current_image_LR = np.expand_dims(current_image_LR, axis=2)


        #Again- Get LR image on-the-fly from HR image:
        #(NOTE!: then why the hell was the above done then?!... why not simply directly use this?!?!):
        if self.OPT['phase'] == 'train':
            # if the image size is too small
            H, W, _ = current_image_HR.shape
            if H < HR_size or W < HR_size:
                current_image_HR = cv2.resize(np.copy(current_image_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                current_image_LR = imresize_np(current_image_HR, 1 / image_rescale_factor, True)
                if current_image_LR.ndim == 2:
                    current_image_LR = np.expand_dims(current_image_LR, axis=2)

            H, W, C = current_image_LR.shape
            LR_size = HR_size // image_rescale_factor



        ###################################
        # Pre-Processing / Augmentation:  #
        ###################################
        if self.OPT['phase'] == 'train':
            #Random Crop:
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            current_image_LR = current_image_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * image_rescale_factor), int(rnd_w * image_rescale_factor)
            current_image_HR = current_image_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            #Augmentation - flip, rotate (TODO: switch to FastAI API)
            current_image_LR, current_image_HR = augment([current_image_LR, current_image_HR], self.OPT['use_flip'], self.OPT['use_rot'])


        ####################################
        # Change color space if necessary  #
        ####################################
        #TODO: why only change LR image color space and not both!!@?$?!#
        if self.OPT['color']:
            current_image_LR = convert_image_color_space(C, self.OPT['color'], [current_image_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if current_image_HR.shape[2] == 3:
            current_image_HR = current_image_HR[:, :, [2, 1, 0]]
            current_image_LR = current_image_LR[:, :, [2, 1, 0]]
        current_image_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(current_image_HR, (2, 0, 1)))).float()
        current_image_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(current_image_LR, (2, 0, 1)))).float()


        #Return Output Dictionary with: (LR_image,HR_image,LR_path,HR_path):
        if LR_path is None:
            LR_path = HR_path
        return {'LR': current_image_LR, 'HR': current_image_HR, 'LR_path': LR_path, 'HR_path': HR_path}


    def __len__(self):
        return len(self.paths_HR)





















######################################################################################################################
#####    TRAINING PARAMETERS!:   #####################################################################################

######################################################
###  Basic Script DataSet and Paths Parameters:  #####
######################################################
#(1). Over Arching Parameters:
model_name = 'debug_002_RRDB_ESRGAN_x4_DIV2K'
flag_use_TensorBoard = True;
mode_type = 'srragan'; #TODO: where does this go?
HR_to_LR_upscale_factor = 4;
gpu_ids = [0]; #TODO: what goes into here?
#(2). DataSet & DataLoader:
#   Train DataSet:
train_dataset_name = 'DIV2K'; #i shouldn't care about the name unless this is used as input to another function which actually gets the files
train_dataset_mode = 'LRHR';
flag_get_LR_on_the_fly_or_from_folder = 1; #1=on-the-fly, 2=from folder
train_dataset_HR_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb';
train_dataset_LR_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/sub_bicLRx4.lmdb';
train_dataset_subset_file = None; #what's this?
train_dataset_flag_shuffle = True;
train_dataset_number_of_cpu_threads_in_BG = 8;
HR_image_size = 128;
#   Train Augmentation:
augmentation_flag_flip = True;
augmentation_flag_rotate = True;
#   Test DataSet:
test_dataset_name = 'val_set14_part';
test_dataset_mode = 'LRHR';
test_dataset_HR_path = 'mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14';
test_dataset_LR_path = 'mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14_bicLRx4';
#(3). Current/Pre-Existing Models Saving/Loading path:
model_saving_path = '/home/xtwang/Projects/BasicSR';
pretrained_G_model_path = '../experiments/pretrained_models/RRDB_PSNR_x4.pth';
pretrained_D_model_path = '../experiments/pretrained_models/RRDB_PSNR_x4.pth';
#(4). Logger:
log_path = 'blablabla_log.txt'


################################
### Networks Parameters:   #####
################################
#(1). Generator Network:
generator_model_type = 'RRDB_net' #or SR_ResNet
generator_normalization_type = None;
generator_convolution_block_mode = 'CNA'; #conv-BN-Act
generator_number_of_input_channels = 3;
generator_number_of_output_channels = 3;
generator_group = 1;
generator_normalization_type = None;
generator_activation_type = 'leakyrelu';
generator_upsample_model = 'pixel_shuffle'; #pixel_shuffle or upconv
# RRDB_Net:
generator_number_of_input_channels_in_each_RRDB_block = 64;
generator_number_of_intermediate_channels_in_each_RRDB_block = 32;
generator_number_of_RRDB_blocks = 32;
generator_residual_branch_scaling = 0.2;
# ResNet:
generator_number_of_input_channels_in_each_ResNet_block = 64;
generator_number_of_ResNet_blocks = 32;
generator_residual_branch_scaling = 0.2;

#(2). Descriminator Network:
descriminator_model_type = 'descriminator_vgg_128'; #depends on image size
descriminator_normalization_type = 'batch_normalization'
descriminator_activation_type = 'leakyrelu'; #TODO: make leaky relu negative slope a hyper parameter
descriminator_convolution_block_mode = 'CNA';
descriminator_number_of_input_channels = 3;
descriminator_vgg_doubling_base_number_of_filters = 64; #remember vgg is basically doubling number of channels whilst dividing image size

#(3). Feature Extractor Network:
feature_extractor_model_type = 'vgg'; #'vgg', 'ResNet'
feature_extractor_feature_layer_to_extract = 34;
feature_extractor_flag_use_BN = False;
feature_extractor_flag_use_input_normalization = True; #value normalization not like BN



################################
### Training Parameters:   #####
################################
#(1). Generator Optimizer:
generator_optimizer_type = 'Adam'; #try sgd with cycles!
generator_learning_rate = 1e-4;
generator_optimizer_weight_decay = 0;
generator_optimizer_beta1 = 0.9;
#(2). Descriminator Optimizer:
descriminator_optimizer_type = 'Adam';
descriminator_learning_rate = 1e-4;
descriminator_optimizer_weight_decay = 0;
descriminator_optimizer_beta1 = 0.9;
#(3). Learning Rate Schedualers:
learning_rate_schedualer_scheme = 'MultiStepLR'; #try out cycle!
learning_rate_steps = [50000,50000*2,50000*4,50000*6];
learning_rate_gamma = 0.5;
#(4). Training Program:
batch_size = 16;
number_of_training_iterations = 5e5;
D_to_G_update_ratio = 5;
D_initial_number_of_training_iterations = 0;
#(5). Validation/Test:
validation_frequency = 5e3;
#(6). Logger:
logger_print_frequency = 200;
model_save_checkpoint_frequency = 5e3



################################
### Loss Parameters:   #####
################################
#(1). Loss Types:
direct_pixel_loss_type = 'L1'; #'L1' / 'L2'
feature_space_loss_type = 'L1'; #'L1' / 'L2'
GAN_validity_loss_type = 'vanilla'; #'vanilly' | lsgan | wgan-gp
#(2). Loss Weights:
lambda_direct_pixel_loss = 1e-2;
lambda_feature_space_loss = 1;
lambda_GAN_validity_loss = 5e-3;
lambda_gradient_penalty_loss = 10;




##############################################################################################################################
#####    INITIALIZE AUXILIARY NEEDED STUFF!:   ###############################################################################
#(1). Create model saving path if needed:
mkdir_and_rename(model_saving_path);
#(2). Print Existing Log in Log-Path:
sys.stdout = PrintLogger(log_path)



##############################################################################################################################
#####    Create Train & Test DataLoader!:   ##################################################################################
#### Train DataSet: ####
#TODO: is there a possibility to load to RAM the jpeg data and convert it to float32 tensors as part of the preprocessing thus allowing us to load a lot more data to RAM?
train_dataset_images_to_RAM_or_from_disk = 2; #(1). all images to RAM (if my computer has large RAM), (2). from disk (for large datasets),
train_dataset_image_from_image_folder_or_from_lmdb = 1; #(1). from image folder, (2). from lmdb
flag_get_LR_on_the_fly_or_from_folder = 1; #(1). on-the-fly, (2). from LR dedicated folder



#(1). Get dataset file names source type: (TODO: have a possibility to simply load images into RAM!)
if train_dataset_HR_path.split('.')[-1] == 'lmdb':
    train_dataset_file_names_data_source_type = 'lmdb';
else:
    train_dataset_file_names_data_source_type = 'images';
#(2). Get dataset HR file names source itself:
HR_env, HR_paths = get_image_paths(data_type=train_dataset_file_names_data_source_type,dataroot=train_dataset_HR_path);
#(3). Get dataset LR file names source itself:
if flag_get_LR_on_the_fly_or_from_folder == 2: #From Folder (in this case lmdb)
    LR_env, LR_paths = get_image_paths(data_type=train_dataset_file_names_data_source_type,dataroot=train_dataset_LR_path);





train_dataset_name = 'DIV2K'; #i shouldn't care about the name unless this is used as input to another function which actually gets the files
train_dataset_mode = 'LRHR';
train_dataset_HR_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb';
train_dataset_LR_path = '/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/sub_bicLRx4.lmdb';
train_dataset_subset_file = None; #what's this?
train_dataset_flag_shuffle = True;
train_dataset_number_of_cpu_threads_in_BG = 8;











"datasets": {
    "train": {
      "name": "DIV2K"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub.lmdb"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/DIV2K800/DIV2K800_sub_bicLRx4.lmdb"
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 16
      , "HR_size": 128
      , "use_flip": true
      , "use_rot": true
    }
    , "val": {
      "name": "val_set14_part"
      , "mode": "LRHR"
      , "dataroot_HR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14"
      , "dataroot_LR": "/mnt/SSD/xtwang/BasicSR_datasets/val_set14_part/Set14_bicLRx4"
    }

# create train and val dataloader
for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train':
        train_set = create_dataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
        print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        total_iters = int(opt['train']['niter'])
        total_epoches = int(math.ceil(total_iters / train_size))
        print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
        train_loader = create_dataloader(train_set, dataset_opt)
    elif phase == 'val':
        val_dataset_opt = dataset_opt
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt)
        print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
    else:
        raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
assert train_loader is not None

# Create model
model = create_model(opt)
# create logger
logger = Logger(opt)

current_step = 0
start_time = time.time()
print('---------- Start training -------------')
for epoch in range(total_epoches):
    for i, train_data in enumerate(train_loader):
        current_step += 1
        if current_step > total_iters:
            break

        # training
        model.feed_data(train_data)
        model.optimize_parameters(current_step)

        time_elapsed = time.time() - start_time
        start_time = time.time()

        # log
        if current_step % opt['logger']['print_freq'] == 0:
            logs = model.get_current_log()
            print_rlt = OrderedDict()
            print_rlt['model'] = opt['model']
            print_rlt['epoch'] = epoch
            print_rlt['iters'] = current_step
            print_rlt['time'] = time_elapsed
            for k, v in logs.items():
                print_rlt[k] = v
            print_rlt['lr'] = model.get_current_learning_rate()
            logger.print_format_results('train', print_rlt)

        # save models
        if current_step % opt['logger']['save_checkpoint_freq'] == 0:
            print('Saving the model at the end of iter {:d}.'.format(current_step))
            model.save(current_step)

        # validation
        if current_step % opt['train']['val_freq'] == 0:
            print('---------- validation -------------')
            start_time = time.time()

            avg_psnr = 0.0
            idx = 0
            for val_data in val_loader:
                idx += 1
                img_name = os.path.splitext(os.path.basename(val_data['LR_path'][0]))[0]
                img_dir = os.path.join(opt['path']['val_images'], img_name)
                util.mkdir(img_dir)

                model.feed_data(val_data)
                model.test()

                visuals = model.get_current_visuals()
                sr_img = util.tensor2img(visuals['SR'])  # uint8
                gt_img = util.tensor2img(visuals['HR'])  # uint8

                # Save SR images for reference
                save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(\
                    img_name, current_step))
                util.save_img(sr_img, save_img_path)

                # calculate PSNR
                crop_size = opt['scale']
                cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                avg_psnr += util.psnr(cropped_sr_img, cropped_gt_img)

            avg_psnr = avg_psnr / idx
            time_elapsed = time.time() - start_time
            # Save to log
            print_rlt = OrderedDict()
            print_rlt['model'] = opt['model']
            print_rlt['epoch'] = epoch
            print_rlt['iters'] = current_step
            print_rlt['time'] = time_elapsed
            print_rlt['psnr'] = avg_psnr
            logger.print_format_results('val', print_rlt)
            print('-----------------------------------')

        # update learning rate
        model.update_learning_rate()

print('Saving the final model.')
model.save('latest')
print('End of training.')



































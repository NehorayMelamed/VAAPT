


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia
import einops
import math

from typing import Union

from typing import Tuple, List, Union
from torch import Tensor
from functools import reduce
import os
from PIL import Image
import cv2

def image_loading_post_processing(image, flag_convert_to_rgb=0, flag_normalize_to_float=0):
    if flag_convert_to_rgb == 1 and len(image.shape)==3 and image.shape[-1]==3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    if flag_normalize_to_float == 1:
        image = image.astype(np.float32) / 255 #normalize to range [0,1] instead of [0,255]
    if image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        image = np.expand_dims(image, axis=2)
    if image.shape[2] > 4:
        # In the weird case where the image has more then 3 channels only take the first 3 channels to make it conform with 3-channels image format:
        image = image[:,:,:3]
    return image


def ImageLoaderCV(path):
    image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(image.shape) == 3:  # opencv opens images as  BGR so we need to convert it to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.atleast_3d(image)  # MAKE SURE TO ALWAYS RETURN [H,W,C] FOR CONSISTENCY
    return np.float32(image)


def read_image_cv2(path):
    return ImageLoaderCV(path)

def read_image_general(path, flag_convert_to_rgb=1, flag_normalize_to_float=0, io_dict=None):
    if '.raw' in path:
        ### TODO: default parameters untill i insert io_dict from main script's train_dict: ###
        # 8 bit, unsigned, 2048 columns, 934 rows, little endian byte order:
        W = 2048
        H = 934

        # W = 640
        # H = 480

        scene_infile = open(path, 'rb')
        scene_image_array = np.fromfile(scene_infile, dtype=np.uint8, count=W * H)
        # scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
        image = Image.frombuffer("I", [W, H],
                                       scene_image_array.astype('I'),
                                       'raw', 'I', 0, 1)
        image = np.array(image)
        image = image_loading_post_processing(image, flag_convert_to_rgb, flag_normalize_to_float)
        return image
    else:
        return read_image_cv2(path)


def get_random_start_stop_indices_for_crop(crop_size, max_number):
    start_index = np.random.randint(0, max(0, max_number-crop_size))
    stop_index = start_index + min(crop_size,max_number)
    return start_index, stop_index

IMG_EXTENSIONS = ['.jpg',
                  '.JPG',
                  '.tif',
                  '.tiff',
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
                '.tif','.tiff',
                  '.jpeg',
                  '.JPEG',
                  '.ppm',
                  '.PPM',
                  '.bmp',
                  '.BMP']
IMG_EXTENSIONS_PNG = ['.png','.PNG']

def is_image_file(filename, img_extentions=IMG_EXTENSIONS_NO_PNG):
    return any(list((filename.endswith(extension) for extension in img_extentions)))


def read_images_from_filenames_list(image_filenames, flag_return_numpy_or_list='numpy', crop_size=np.inf, max_number_of_images=10, allowed_extentions=IMG_EXTENSIONS, flag_how_to_concat='C', crop_style='random', flag_return_torch=False, transform=None, flag_to_BW=False, flag_random_first_frame=False, first_frame_index=-1, flag_to_RGB=False, flag_return_position_indices=False, start_H=-1, start_W=-1):
    ### crop_style = 'random', 'random_deterministic', 'predetermined'
    number_of_images = min(len(image_filenames), max_number_of_images)
    image_concat_array = []

    if type(crop_size) is not list and type(crop_size) is not tuple:
        crop_size = [crop_size, crop_size]

    ### Decide whether to start from start or randomize initial frame returned: ###
    if flag_random_first_frame:
        start_index, stop_index = get_random_start_stop_indices_for_crop(number_of_images, len(image_filenames)) #TODO: notice everything's okey, i hopefully corrected from (number_of_images, number_of_images)
    else:
        start_index = 0
        stop_index = start_index + number_of_images

    if first_frame_index != -1:
        start_index = first_frame_index
        stop_index = start_index + number_of_images

    current_step = 0
    flag_already_set_crop_position = False
    for current_index in np.arange(start_index, stop_index, 1):
        image_filename = image_filenames[current_index]
        if is_image_file(image_filename, allowed_extentions):
            ### Read Image: ###
            current_image = read_image_general(image_filename)

            ### If Crop Size Is Legit continue on: ###
            if (current_image.shape[0] >= crop_size[0] and current_image.shape[1] >= crop_size[1]) or (crop_size == np.inf or crop_size == [np.inf, np.inf]):
                ### Get current crop: ###
                if crop_style == 'random_consistent' and flag_already_set_crop_position==False:
                    #TODO: need to be consistent, and instead of crop_x and crop_y i need to transfer to crop_H and crop_W
                    start_H, stop_H = get_random_start_stop_indices_for_crop(current_image.shape[-2], min(crop_size[0], current_image.shape[-2]))
                    start_W, stop_W = get_random_start_stop_indices_for_crop(current_image.shape[-1], min(crop_size[1], current_image.shape[-1]))
                    flag_already_set_crop_position = True #only set crop position once
                elif crop_style == 'predetermined' and start_H!=-1 and start_W!=-1:
                    stop_H = start_H + min(crop_size[0], current_image.shape[-2])
                    stop_W = start_W + min(crop_size[1], current_image.shape[-1])
                # current_crop = crop_numpy_batch(current_image, crop_size=crop_size, crop_style=crop_style, start_H=start_H, start_W=start_W)
                current_crop = crop_tensor(current_image, crop_size_tuple_or_scalar=crop_size, crop_style=crop_style, start_H=start_H, start_W=start_W)

                ### To BW/RGB If Wanted: ###
                if flag_to_BW:
                    current_crop = RGB2BW(current_crop)
                if flag_to_RGB:
                    current_crop = BW2RGB(current_crop)

                ### Transform If Exists: ###
                if transform is not None:
                    current_crop = transform(current_crop)

                ### Permute Dimensions To Torch Convention If Wanted: ###
                H, W, C = current_crop.shape
                if flag_return_torch:
                    current_crop = np.transpose(current_crop, [2,0,1])

                if flag_return_numpy_or_list == 'list':
                    image_concat_array.append(current_crop)

                else: #'numpy'
                    ### Initialize image concat array: ###
                    if current_step == 0:
                        if flag_how_to_concat == 'T':
                            image_concat_array = np.expand_dims(current_crop, 0)
                        elif flag_how_to_concat == 'C':
                            if flag_return_torch:
                                #TODO: in case there are mixed type- add flag for type
                                image_concat_array = np.zeros((C * number_of_images, H, W)).astype(current_crop.dtype)
                                image_concat_array[0:C,:,:] = current_crop
                            else:
                                image_concat_array = np.zeros((H, W, C * number_of_images)).astype(current_crop.dtype)
                                image_concat_array[:,:,0:C] = current_crop

                    ### Assign Current Image To Concatenated Image: ###
                    else:
                        if flag_how_to_concat == 'T':
                            image_concat_array = np.concatenate((image_concat_array, np.expand_dims(current_crop, 0)), axis=0)
                        elif flag_how_to_concat == 'C':
                            if flag_return_torch:
                                image_concat_array[current_step * C:(current_step + 1) * C, :, :] = current_crop
                            else:
                                image_concat_array[:, :, current_step * C:(current_step + 1) * C] = current_crop

                ### If we're at max_number_of_images --> break: ###
                if current_step == max_number_of_images:
                    break

            ### Uptick Current Step: ###
            current_step += 1

    if flag_return_torch:
        if image_concat_array.dtype == np.uint8:
            if flag_return_position_indices:
                return torch.Tensor(image_concat_array).type(torch.uint8), start_index, start_H, start_W
            else:
                return torch.Tensor(image_concat_array).type(torch.uint8)
        elif image_concat_array.dtype == np.uint16:
            if flag_return_position_indices:
                return torch.Tensor(image_concat_array).type(torch.uint16), start_index, start_H, start_W
            else:
                return torch.Tensor(image_concat_array).type(torch.uint16)
        else:
            if flag_return_position_indices:
                return torch.Tensor(image_concat_array), start_index, start_H, start_W
            else:
                return torch.Tensor(image_concat_array)
    else:
        if flag_return_position_indices:
            return image_concat_array, start_index, start_H, start_W
        else:
            return image_concat_array




def torch_image_flatten(input_tensor):
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if shape_len == 1:
        return input_tensor
    elif shape_len == 2:
        return input_tensor.reshape((H*W))
    elif shape_len == 3:
        return input_tensor.reshape((C, H*W))
    elif shape_len == 4:
        return input_tensor.reshape((T, C, H*W))
    elif shape_len == 5:
        return input_tensor.reshape((B, T, C, H*W))

def get_Sigma(I):
    dx = torch.diff(I, dim=-1)
    dx_flattened = torch_image_flatten(dx)
    s = 1.4826 * (dx_flattened - dx_flattened.median(-1)[0].unsqueeze(-1)).abs().median(-1)[0]  #TODO: why not do it on a column-wise basis instead of a single number?
    return s

def D1boxfilter(I_col, r):
    #(*). Notice!!!!: the secret to Matlab->Pytorch indexing is: in matlab the first possible index is 1 as opposed to 0, moreover,
    # in matlab when indexing an array start_index:stop_index it gets transferred to start_index-1:stop_index in pytorch!!!!
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(I_col)
    col_size = H #input shape is ...,H,1
    imDst = torch.zeros_like(I_col)
    imCum = torch.cumsum(I_col, -2)
    imDst[..., 0:r+1, :] = imCum[..., r:2*r+1, :]  #for the first r terms in the beginning simply use the cumulative sum of the first r "valid" (with at least r terms) sum
    imDst[..., r+1:H-r, :] = imCum[..., 2*r+1:H, :] - imCum[..., 0:H-2*r-1, :]  #moving sum (box filter) of 2r+1 terms
    imDst[..., H-r:H, :] = torch.cat([imCum[...,H-1:H, :]]*r, dim=-2) - imCum[..., H-2*r-1:H-r-1, :]  #
    return imDst

def verticalBoxfilter(I, r):
    H, W = I.shape[-2:]
    H_image = torch.ones((H,1))   # torch_image_flatten(I) - TODO: in the end make this torch way
    Nv = D1boxfilter(H_image, r)
    mI = D1boxfilter(I, r) / Nv
    return mI

def weightedLocalLinearFilter(x, y, w, r, eps):
    ww = verticalBoxfilter(w, r)
    wx = verticalBoxfilter(w * x, r) / ww
    wy = verticalBoxfilter(w * y, r) / ww
    wxy = verticalBoxfilter(w * x * y, r) / ww
    wxx = verticalBoxfilter(w * x * x, r) / ww
    a = (wxy - wx*wy + eps) / (wxx - wx*wx + eps)
    b = wy - wx * a
    mean_a = verticalBoxfilter(a, r)
    mean_b = verticalBoxfilter(b, r)
    u = (y - mean_b) / mean_a
    return u, a, b

### Vertical Weighted Local Ridge Regression Stage: ###
def edgeIndicator(L, xi):
    r = 5  #TODO: perhapse make this a variable?!!?!?!?
    mean_1 = (verticalBoxfilter(L.transpose(-1,-2), r)).transpose(-1,-2)
    mean_2 = (verticalBoxfilter((L*L).transpose(-1,-2), r)).transpose(-1,-2)
    Var = mean_2 - mean_1*mean_1
    m = Var.mean([-1,-2], True)
    Dire = torch.exp(-Var/(xi*m)) + 1e-10
    return Dire, m

def linearInverseOperator(a, b, c, f):
    ### Using Guass Elimination To Inverse a 3-point laplacian matrix: ###
    DL = a.shape[-1]

    ### Calculate the forward computation: ###
    c[:, 0] = c[:, 0] / b[:, 0]
    f[:, 0] = f[:, 0] / b[:, 0]

    # (*). this is recursive, so i think that this cannot be sped up... i need to think about it.....
    for k in np.arange(1, DL):
        c[:, k] = c[:, k] / (b[:, k] - c[:, k - 1] * a[:, k])
        f[:, k] = (f[:, k] - f[:, k - 1] * a[:, k]) / (b[:, k] - c[:, k - 1] * a[:, k])

    ### Calculate the backward computation: ###
    u = torch.zeros_like(f)
    u[:, DL - 1] = f[:, DL - 1]
    for k in np.arange(DL - 2, -1, -1):
        u[:, k] = f[:, k] - c[:, k] * u[:, k + 1]

    return u


def linearInverseOperator_parallel(a, b, c, f):
    ### Using Guass Elimination To Inverse a 3-point laplacian matrix: ###
    DL = a.shape[-1]

    ### Calculate the forward computation: ###
    c[..., 0] = c[..., 0] / b[..., 0]
    f[..., 0] = f[..., 0] / b[..., 0]

    # (*). this is recursive, so i think that this cannot be sped up... i need to think about it.....
    for k in np.arange(1, DL):
        c[..., k] = c[..., k] / (b[..., k] - c[..., k - 1] * a[..., k])
        f[..., k] = (f[..., k] - f[..., k - 1] * a[..., k]) / (b[..., k] - c[..., k - 1] * a[..., k])

    ### Calculate the backward computation: ###
    u = torch.zeros_like(f)
    u[..., DL - 1] = f[..., DL - 1]
    for k in np.arange(DL - 2, -1, -1):
        u[..., k] = f[..., k] - c[..., k] * u[..., k + 1]

    return u

def torch_unsqueeze_like(input_tensor, tensor_to_mimick, direction_to_unsqueeze=0):
    #direction_to_unsqueeze = (0:beginning, -1=end)
    (B1,T1,C1,H1,W1), shape_len1, shape_vec1 = get_full_shape_torch(input_tensor)
    (B2,T2,C2,H2,W2), shape_len2, shape_vec2 = get_full_shape_torch(tensor_to_mimick)
    for i in np.arange(shape_len2-shape_len1):
        input_tensor = input_tensor.unsqueeze(direction_to_unsqueeze)
    return input_tensor

def pad_with_zeros_frame_torch(input_tensor, dim=-1, padding_size=(1,1)):
    ### padding_size[0] padding at the beginning of the dim, padding[1] padding at the end of the dim
    # input_tensor = torch.ones((4,3,64,100))
    # dim = -3
    # padding_size = (1,2)

    dims_string = 'BTCHW'
    dims_string_final = dims_string[0:dim] + dims_string[dim+1:]
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    input_dims = dims_string[-shape_len:]
    input_dims = input_dims[dim]
    frame_to_concat = torch.zeros_like(input_tensor)
    frame_to_concat = torch.index_select(frame_to_concat, dim, torch.LongTensor([0]))
    if padding_size[0] > 0:
        frame_to_concat_1 = torch.cat([frame_to_concat]*padding_size[0], dim)
        output_tensor = torch.cat((frame_to_concat_1, input_tensor), dim)
    else:
        output_tensor = input_tensor
    if padding_size[1] > 0:
        frame_to_concat_2 = torch.cat([frame_to_concat]*padding_size[1], dim)
        output_tensor = torch.cat((output_tensor, frame_to_concat_2), dim)

    return output_tensor


### Horizontal edge preserving smoothing stage: ###
def fGHS(g, w, lambda_weight, sigma_n):
    H, W = g.shape[-2:]
    # g = torch_image_flatten(    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # H[-1,0] = 0
    # H[-1,1] = 0g)
    # w = torch_image_flatten(w)
    dg = torch.diff(g, n=1, dim=-1)
    dg = torch.exp(- dg ** 2 / torch_unsqueeze_like(sigma_n, dg, -1) ** 2)
    dg_01 = pad_with_zeros_frame_torch(dg, dim=-1, padding_size=(0,1))
    dg_10 = pad_with_zeros_frame_torch(dg, dim=-1, padding_size=(1,0))
    dga = -lambda_weight * dg_10
    dgc = -lambda_weight * dg_01
    dg = w + lambda_weight * (dg_10 + dg_01)
    u = linearInverseOperator_parallel(dga, dg, dgc, w*g)
    return u




def torch_get_5D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'B':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BH':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BC':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        elif input_dims == 'BT':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BHW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BCH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'BTC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    # (4).
    if len(input_tensor.shape) == 4:
        if input_dims is None:
            input_dims = 'TCHW'

        if input_dims == 'TCHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'BTCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'BCHW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'BTHW':
            return input_tensor.unsqueeze(2)

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor

# def torch_get_4D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor.unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 3:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 4:
#         return input_tensor
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[:,0]

def torch_image_flatten(input_tensor):
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if shape_len == 1:
        return input_tensor
    elif shape_len == 2:
        return input_tensor.reshape((H*W))
    elif shape_len == 3:
        return input_tensor.reshape((C, H*W))
    elif shape_len == 4:
        return input_tensor.reshape((T, C, H*W))
    elif shape_len == 5:
        return input_tensor.reshape((B, T, C, H*W))

def torch_stack_dims(input_tensor, dims_to_stack=None):
    # create a function which you input, for instance, 'BT' and it stacks them up, or something like that, maybe with indices
    1

def torch_transpose_HW(input_tensor):
    1


def torch_get_ND(input_tensor, input_dims=None, number_of_dims=None):
    if number_of_dims == 2:
        return torch_get_2D(input_tensor, input_dims)
    elif number_of_dims == 3:
        return torch_get_3D(input_tensor, input_dims)
    elif number_of_dims == 4:
        return torch_get_4D(input_tensor, input_dims)
    elif number_of_dims == 5:
        return torch_get_5D(input_tensor, input_dims)

def torch_unsqueeze_like(input_tensor, tensor_to_mimick, direction_to_unsqueeze=0):
    #direction_to_unsqueeze = (0:beginning, -1=end)
    (B1,T1,C1,H1,W1), shape_len1, shape_vec1 = get_full_shape_torch(input_tensor)
    (B2,T2,C2,H2,W2), shape_len2, shape_vec2 = get_full_shape_torch(tensor_to_mimick)
    for i in np.arange(shape_len2-shape_len1):
        input_tensor = input_tensor.unsqueeze(direction_to_unsqueeze)
    return input_tensor

def torch_equalize_dimensionality(input_tensor, tensor_to_mimick, direction_to_unsqueeze=0):
    return torch_unsqueeze_like(input_tensor, tensor_to_mimick, direction_to_unsqueeze)

def torch_get_4D(input_tensor, input_dims=None, flag_stack_BT=False):
    #[T,C,H,W]
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(0).unsqueeze(2)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(0).unsqueeze(-1)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1)

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor

    #(5).
    if len(input_tensor.shape) == 5:
        if flag_stack_BT:
            B,T,C,H,W = input_tensor.shape
            return input_tensor.view(B*T,C,H,W)
        else:
            return input_tensor[0]


# def torch_get_3D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 3:
#         return input_tensor
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0]

def torch_get_3D(input_tensor, input_dims=None):
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        return input_tensor

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0]

    #(5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0,0]  #TODO: maybe stack on top of each other?


# def torch_get_2D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor
#     elif len(input_tensor.shape) == 3:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0,0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0,0]


def torch_get_2D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        return input_tensor

    # (3).
    if len(input_tensor.shape) == 3:
        return input_tensor[0]

    # (4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0,0]

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0, 0, 0]  # TODO: maybe stack on top of each other?


def torch_at_least_2D(input_tensor):
    return torch_get_2D(input_tensor)
def torch_at_least_3D(input_tensor):
    return torch_get_3D(input_tensor)
def torch_at_least_4D(input_tensor):
    return torch_get_4D(input_tensor)
def torch_at_least_5D(input_tensor):
    return torch_get_5D(input_tensor)


def perform_Column_Correction_on_tensor(input_tensor, lambda_weight=50, number_of_iterations=3):
    # folder_path = '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/12.4.2022 - natznatz experiments/3_night_500fps_20000frames_640x320/Results/seq0'
    # filename = 'temporal_average.png'
    # input_tensor = read_image_torch(os.path.join(folder_path, filename))
    # input_tensor = RGB2BW(input_tensor)[0,0]

    ### Scale image into (0,1) Range: ###
    normalization_factor = input_tensor.max() #TODO: change to max_per_image
    input_tensor_original = input_tensor / normalization_factor

    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)

    ### Loop Over Iterations: ###
    output_tensor = torch.clone(input_tensor_original)
    for iteration_index in np.arange(number_of_iterations):
        print(iteration_index)
        Dire, _ = edgeIndicator(output_tensor, 0.3)
        sigma_n = get_Sigma(output_tensor)

        ones_vec = torch.ones(1,W).to(input_tensor.device)
        ones_vec = torch_get_ND(ones_vec, input_dims='HW', number_of_dims=shape_len)

        output_tensor = fGHS(output_tensor, ones_vec, lambda_weight, 2*sigma_n)

        output_tensor, a, b = weightedLocalLinearFilter(output_tensor, input_tensor_original, Dire, np.int(np.floor(H/6)), 0.01)
    print('done column correction')

    ### Scale output back: ###
    output_tensor = output_tensor * normalization_factor
    input_tensor_original = input_tensor_original * normalization_factor
    residual_tensor = (input_tensor_original - output_tensor)

    # imshow_torch(input_tensor_original)
    # imshow_torch(output_tensor)
    # imshow_torch(residual_tensor)

    return output_tensor, residual_tensor

def perform_RCC_on_tensor(input_tensor, lambda_weight=50, number_of_iterations=3):
    input_tensor, residual_tensor_column = perform_Column_Correction_on_tensor(input_tensor, lambda_weight=lambda_weight, number_of_iterations=number_of_iterations)
    input_tensor, residual_tensor_rows = perform_Column_Correction_on_tensor(input_tensor.transpose(-1, -2), lambda_weight=lambda_weight, number_of_iterations=number_of_iterations)
    return input_tensor.transpose(-1,-2), residual_tensor_rows.transpose(-1,-2) + residual_tensor_column
# perform_RCC_on_tensor(None)


class Gaussian_Blur_Layer(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussian_Blur_Layer, self).__init__()
        if type(kernel_size) is not list and type(kernel_size) is not tuple:
            kernel_size = [kernel_size] * dim
        if type(sigma) is not list and type(sigma) is not tuple:
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        self.padding = []
        for i in kernel_size:
            self.padding.append(i//2)
            self.padding.append(i//2)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(F.pad(input,self.padding), weight=self.weight, groups=self.groups)



def fftshift_torch(X, first_spatial_dim=-2):
    ### FFT-Shift for all dims from dim=first_spatial_dim onwards: ###
    # batch*channel*...*2
    # real, imag = X.chunk(chunks=2, dim=-1)
    # real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    shape_len = len(X.shape)
    if first_spatial_dim < 0:
        first_spatial_dim = shape_len - abs(first_spatial_dim)

    for dim in range(first_spatial_dim, len(X.shape)):
        X = roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim] / 2)))

    # real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    # X = torch.cat((real,imag),dim=-1)
    return X

def torch_fftshift(X, first_spatial_dim=-2):
    ### FFT-Shift for all dims from dim=first_spatial_dim onwards: ###
    # batch*channel*...*2
    # real, imag = X.chunk(chunks=2, dim=-1)
    # real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    if first_spatial_dim < 0:
        first_spatial_dim = len(X.shape) - np.abs(first_spatial_dim)

    for dim in range(first_spatial_dim, len(X.shape)):
        X = roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim] / 2)))

    # real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    # X = torch.cat((real,imag),dim=-1)
    return X

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift_torch_specific_dim(X, dim):
    return roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim]/2)))


start_cuda = torch.cuda.Event(enable_timing=True, blocking=True)
finish_cuda = torch.cuda.Event(enable_timing=True, blocking=True)

def gtic():
    torch.cuda.synchronize()
    start_cuda.record()

def gtoc(pre_string='', verbose=True):
    ### Print: ###
    finish_cuda.record()
    torch.cuda.synchronize()
    total_time = start_cuda.elapsed_time(finish_cuda)
    if pre_string != '':
        print(pre_string + ": %f msec." % total_time)
    else:
        print("Elapsed time: %f msec." % total_time)


def crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H,W = images.shape
        C = 1
    elif len(images.shape) == 3:
        C,H,W = images.shape
    elif len(images.shape) == 4:
        T, C, H, W = images.shape  # No Batch Dimension
    else:
        B, T, C, H, W = images.shape  # "Full" with Batch Dimension

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) == list or type(crop_size_tuple_or_scalar) == tuple:
        cropH = crop_size_tuple_or_scalar[0]
        cropW = crop_size_tuple_or_scalar[1]
    else:
        cropH = crop_size_tuple_or_scalar
        cropW = crop_size_tuple_or_scalar

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:  #center
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        end_x = W
        end_y = H
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###
    if len(images.shape) == 2:
        return pad_torch_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_torch_batch(images[:, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 4:
        return pad_torch_batch(images[:, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    else:
        return pad_torch_batch(images[:, :, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)


def crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H, W = images.shape
    elif len(images.shape) == 3:
        H, W, C = images.shape #BW batch
    else:
        T, H, W, C = images.shape #RGB/NOT-BW batch

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) == list or type(crop_size_tuple_or_scalar) == tuple:
        cropW = crop_size_tuple_or_scalar[1]
        cropH = crop_size_tuple_or_scalar[0]
    else:
        cropW = crop_size_tuple_or_scalar
        cropH = crop_size_tuple_or_scalar
    cropW = min(cropW, W)
    cropH = min(cropH, H)

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        end_x = W
        end_y = H
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###    imshow_torch(tensor1_crop)
    if len(images.shape) == 2:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)
    else:
        return pad_numpy_batch(images[:, start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)


def crop_tensor(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### crop_style = 'center', 'random', 'predetermined'
    if type(images) == torch.Tensor:
        return crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W)
    else:
        return crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W)



def pad_numpy_batch(input_arr: np.array, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(a, pad_size):  # create a rigid shape for padding in dims HWC
            pad_start = np.floor(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            return (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (0, 0)

        ### Make sure pad size is accurate for pad_shape function (if it's larger then input array shape -> clamp it to array size): ###
        pad_size_H, pad_size_W = pad_size
        H,W = input_arr.shape[-3:-1]
        pad_size_H = min(pad_size_H, H)
        pad_size_W = min(pad_size_W, W)
        pad_size = (pad_size_H, pad_size_W)

        if len(input_arr.shape) == 4:
            return np.array([np.pad(a, pad_shape(a, pad_size), 'constant', constant_values=0) for a in input_arr])
        else:
            return np.pad(input_arr, pad_shape(input_arr, pad_size), 'constant', constant_values=0)
    else:
        return None


def pad_torch_batch(input_tensor: Tensor, pad_size: Tuple[int, int], pad_style='center', pad_mode='constant'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(t, pad_size):  # create a rigid shape for padding in dims CHW
            pad_start = np.floor(np.subtract(pad_size, t.shape[-2:]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, t.shape[-2:]) / 2).astype(int)
            return pad_start[1], pad_end[1], pad_start[0], pad_end[0]

        return torch.nn.functional.pad(input_tensor, pad_shape(input_tensor, pad_size), mode=pad_mode)
    else:
        return None



class Warp_Object(nn.Module):
    # Initialize this with a module
    def __init__(self):
        super(Warp_Object, self).__init__()
        self.X = None
        self.Y = None

    def bilinear_interpolation_torch1(self, input_tensor, X, Y):
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

        x0 = torch.floor(X).type(dtype_long)
        x1 = x0 + 1

        y0 = torch.floor(Y).type(dtype_long)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, input_tensor.shape[1] - 1)
        x1 = torch.clamp(x1, 0, input_tensor.shape[1] - 1)
        y0 = torch.clamp(y0, 0, input_tensor.shape[0] - 1)
        y1 = torch.clamp(y1, 0, input_tensor.shape[0] - 1)

        Ia = input_tensor[y0, x0][0]
        Ib = input_tensor[y1, x0][0]
        Ic = input_tensor[y0, x1][0]
        Id = input_tensor[y1, x1][0]

        wa = (x1.type(dtype) - X) * (y1.type(dtype) - Y)
        wb = (x1.type(dtype) - X) * (Y - y0.type(dtype))
        wc = (X - x0.type(dtype)) * (y1.type(dtype) - Y)
        wd = (X - x0.type(dtype)) * (Y - y0.type(dtype))

        return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)

    def bilinear_interpolation_torch2(self, input_images, x_offset, y_offset, wrap_mode='border', tensor_type='torch.cuda.FloatTensor'):
        #TODO: this is a bilinear version for shifts only in the X axis, understand changes needed to be done to make this work for shifts in both axis

        num_batch, num_channels, height, width = input_images.size()

        # Handle both texture border types
        edge_size = 0
        if wrap_mode == 'border':
            edge_size = 1
            # Pad last and second-to-last dimensions by 1 from both sides
            input_images = torch.nn.functional.pad(input_images, (1, 1, 1, 1))
        elif wrap_mode == 'edge':
            edge_size = 0
        else:
            return None

        # Put channels to slowest dimension and flatten batch with respect to others
        input_images = input_images.permute(1, 0, 2, 3).contiguous()
        im_flat = input_images.view(num_channels, -1)

        # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated meshgrid function)
        #TODO: preallocate this, try and see if preallocating the concat tensor will help
        x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type).to('cuda')
        y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type).to('cuda')
        # Take padding into account
        x = x + edge_size
        y = y + edge_size
        # Flatten and repeat for each image in the batch
        x = x.view(-1).repeat(1, num_batch)
        y = y.view(-1).repeat(1, num_batch)

        # Now we want to sample pixels with indicies shifted by disparity in X direction
        # For that we convert disparity from % to pixels and add to X indicies
        x = x + x_offset.contiguous().view(-1) * width
        y = y + y_offset.contiguous().view(-1) * height
        # Make sure we don't go outside of image
        x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
        # Round disparity to sample from integer-valued pixel grid
        y0 = torch.floor(y)
        y1 = y0 + 1
        # In X direction round both down and up to apply linear interpolation between them later
        x0 = torch.floor(x)
        x1 = x0 + 1
        # After rounding up we might go outside the image boundaries again
        x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

        #TODO: understand where to put y1!!!!, and whether we need to flatten the image
        # Calculate indices to draw from flattened version of image batch
        dim2 = (width + 2 * edge_size)
        dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
        # Set offsets for each image in the batch
        base = dim1 * torch.arange(num_batch).type(tensor_type).to('cuda')
        base = base.view(-1, 1).repeat(1, height * width).view(-1)
        # One pixel shift in Y  direction equals dim2 shift in flattened array
        base_y0 = base + y0 * dim2
        # Add two versions of shifts in X direction separately
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        # Sample pixels from images
        pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
        pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

        # Apply linear interpolation to account for fractional offsets
        weight_l = x1 - x
        weight_r = x - x0
        output = weight_l * pix_l + weight_r * pix_r

        # Reshape back into image batch and permute back to (N,C,H,W) shape
        output = output.view(num_channels, num_batch, height, width).permute(1, 0, 2, 3)

        return output

    # Elementwise sum the output of a submodule to its input
    def forward(self, input_image, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear'):
        # delta_x = map of x deltas from meshgrid, shape=[B,H,W] or [B,C,H,W].... same for delta_Y
        B, C, H, W = input_image.shape
        BXC = B * C

        # ### ReOrder delta_x, delta_y: ###
        # #TODO: this expects delta_x,delta_y to be image sized tensors. but sometimes i just wanna pass in a single number per image
        # #(1). Dim=3 <-> [B,H,W], I Interpret As: Same Flow On All Channels:
        # if len(delta_x.shape) == 3:  # [B,H,W] - > [B,H,W,1]
        #     delta_x = delta_x.unsqueeze(-1)
        #     delta_y = delta_x.unsqueeze(-1)
        #     flag_same_on_all_channels = True
        # #(2). Dim=4 <-> [B,C,H,W], Different Flow For Each Channel:
        # elif (len(delta_x.shape) == 4 and delta_x.shape[1]==C):  # [B,C,H,W] - > [BXC,H,W,1] (because pytorch's function only warps all channels of a tensor the same way so in order to warp each channel seperately we need to transfer channels to batch dim)
        #     delta_x = delta_x.view(B * C, H, W).unsqueeze(-1)
        #     delta_y = delta_y.view(B * C, H, W).unsqueeze(-1)
        #     flag_same_on_all_channels = False
        # #(3). Dim=4 but C=1 <-> [B,1,H,W], Same Flow On All Channels:
        # elif len(delta_x.shape) == 4 and delta_x.shape[1]==1:
        #     delta_x = delta_x.permute([0,2,3,1]) #[B,1,H,W] -> [B,H,W,1]
        #     delta_y = delta_y.permute([0,2,3,1])
        #     flag_same_on_all_channels = True
        # #(4). Dim=4 but C=1 <-> [B,H,W,1], Same Flow On All Channels:
        # elif len(delta_x.shape) == 4 and delta_x.shape[3] == 1:
        #     flag_same_on_all_channels = True
        flag_same_on_all_channels = True

        ### Create "baseline" meshgrid (as the function ultimately accepts a full map of locations and not just delta's): ###
        #(*). ultimately X.shape=[BXC, H, W, 1]/[B,H,W,1]... so check if the input shape has changed and only then create a new meshgrid:
        flag_input_changed_from_last_time = (self.X is None) or (self.X.shape[0]!=BXC and flag_same_on_all_channels==False) or (self.X.shape[0]!=B and flag_same_on_all_channels==True) or (self.X.shape[1]!=H) or (self.X.shape[2]!=W)
        if flag_input_changed_from_last_time:
            print('new meshgrid')
            [X, Y] = np.meshgrid(np.arange(W), np.arange(H))  #X.shape=[H,W]
            if flag_same_on_all_channels:
                X = torch.Tensor([X] * B).unsqueeze(-1) #X.shape=[B,H,W,1]
                Y = torch.Tensor([Y] * B).unsqueeze(-1)
            else:
                X = torch.Tensor([X] * BXC).unsqueeze(-1) #X.shape=[BXC,H,W,1]
                Y = torch.Tensor([Y] * BXC).unsqueeze(-1)
            X = X.to(input_image.device)
            Y = Y.to(input_image.device)
            self.X = X
            self.Y = Y
            self.bilinear_grid = torch.zeros_like(torch.cat([X,Y], -1))
            self.new_X = torch.zeros_like(X)
            self.new_Y = torch.zeros_like(Y)


        # [X, Y] = np.meshgrid(np.arange(W), np.arange(H))
        # X = torch.Tensor([X] * BXC).unsqueeze(-1)
        # Y = torch.Tensor([Y] * BXC).unsqueeze(-1)
        # X = X.to(input_image.device)
        # Y = Y.to(input_image.device)


        ### Add Difference (delta) Maps to Meshgrid: ###
        ### Previous Try: ###
        # X += delta_x
        # Y += delta_y
        # X = (X - W / 2) / (W / 2 - 1)
        # Y = (Y - H / 2) / (H / 2 - 1)
        # ### Previous Use: ###
        # new_X = ((self.X + delta_x) - W / 2) / (W / 2 - 1)
        # new_Y = ((self.Y + delta_y) - H / 2) / (H / 2 - 1)
        ### New Use: ###
        #TODO: maybe also preallocate this?
        self.new_X = 2 * ((self.X + delta_x)) / max(W-1,1) - 1
        self.new_Y = 2 * ((self.Y + delta_y)) / max(H-1,1) - 1
        # ### No Internal Tensors: ###
        # new_X = 2 * ((X + delta_x)) / max(W - 1, 1) - 1
        # new_Y = 2 * ((Y + delta_y)) / max(H - 1, 1) - 1
        self.bilinear_grid[:,:,:,0:1] = self.new_X
        self.bilinear_grid[:,:,:,1:2] = self.new_Y

        if flag_same_on_all_channels:
            #input_image.shape=[B,C,H,W] , bilinear_grid.shape=[B,H,W,2]
            input_image_to_bilinear = input_image
            warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, self.bilinear_grid, mode=flag_bicubic_or_bilinear)

            # ### Bilinear Interpolation Without Concatenating X,Y: ###
            # warped_image2 = self.bilinear_interpolate_torch(input_image_to_bilinear, new_X.squeeze().unsqueeze(1), new_Y.squeeze().unsqueeze(1))

            return warped_image
        else:
            #input_image.shape=[BXC,1,H,W] , bilinear_grid.shape=[BXC,H,W,2]
            input_image_to_bilinear = input_image.reshape(-1, int(H), int(W)).unsqueeze(1) #[B,C,H,W]->[B*C,1,H,W]
            warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, self.bilinear_grid, mode=flag_bicubic_or_bilinear)
            warped_image = warped_image.view(B,C,H,W)
            return warped_image


class Super_CC_Layer(nn.Module):
    def __init__(self, flag_gaussian_filter=False,
                 flag_fftshift_before_median=False,
                 flag_median_per_image=True,
                 flag_mean_instead_of_median=False,
                 flag_zero_out_zero_component_each_CC=False,
                 flag_stretch_tensors=False,
                 flag_W_matrix_method=0,
                 flag_shift_CC_to_zero=False,
                 flag_round_shift_before_shifting_CC=False,
                 flag_CC_shift_method='bicubic',
                 max_shift=3,
                 max_shift_for_fit=15,
                 W_matrix_circle_radius=21,
                 warp_method='bilinear'):
        super().__init__()
        self.X = None
        self.Y = None

        ### Spatially Filter Frames (for less high frequency noise in the CC): ###
        if flag_gaussian_filter:
            # TODO: make this efficient, this is extremely slow and stupid
            self.gaussian_blur_layer = Gaussian_Blur_Layer(channels=1, kernel_size=23, sigma=5, dim=2).cuda()

        ### Initialize Gaussian Peak Finding Layer: ###
        self.find_gaussian_peak_layer = Find_Subpixl_Peak_Gaussian_Fit(flag_order='C',
                                                                       flag_W_matrix_method=flag_W_matrix_method,
                                                                       flag_shift_CC_to_zero=flag_shift_CC_to_zero,
                                                                       flag_round_shift_before_shifting_CC=flag_round_shift_before_shifting_CC,
                                                                       flag_CC_shift_method=flag_CC_shift_method,
                                                                       max_shift=max_shift,
                                                                       W_matrix_circle_radius=W_matrix_circle_radius)

        ### Initialize internal attributes: ###
        self.flag_gaussian_filter = flag_gaussian_filter
        self.flag_fftshift_before_median = flag_fftshift_before_median
        self.flag_median_per_image = flag_median_per_image
        self.flag_mean_instead_of_median = flag_mean_instead_of_median
        self.flag_zero_out_zero_component_each_CC = flag_zero_out_zero_component_each_CC
        self.flag_stretch_tensors = flag_stretch_tensors
        self.max_shift = max_shift
        self.q1 = None
        self.q2 = None
        self.gaussian_filter_fft = None
        self.warp_method = warp_method
        self.max_shift_for_fit = max_shift_for_fit
        self.warp_object_layer = Warp_Object()

    def Substract_Column_and_Row_Median(self, cc_CC):
        ### Lower Row Median From CC: ###
        # TODO: lower memory access by calculating row-wise median and col-wise median in one go
        if self.flag_fftshift_before_median:
            cc_CC = torch_fftshift(cc_CC)
        if self.flag_median_per_image:
            if self.flag_mean_instead_of_median:
                cc_CC_row_median = torch.mean(cc_CC, dim=-1, keepdim=True)
            else:
                cc_CC_row_median, cc_CC_row_median_indices = torch.median(cc_CC, dim=-1, keepdim=True)
        else:
            cc_CC_mean = cc_CC.mean(0, True)
            if self.flag_mean_instead_of_median:
                cc_CC_row_median = torch.mean(cc_CC_mean, dim=-1, keepdim=True)
            else:
                cc_CC_row_median, cc_CC_row_median_indices = torch.median(cc_CC_mean, dim=-1, keepdim=True)
        H, W = cc_CC.shape[-2:]

        ### Repeat and substract row median: ###
        cc_CC_row_median = cc_CC_row_median.repeat(1, 1, 1, W)  # TODO: can be saved with CUDA
        cc_CC = cc_CC - cc_CC_row_median
        if self.flag_fftshift_before_median:
            cc_CC = torch_fftshift(cc_CC)

        ### Lower Column Median From CC: ###
        if self.flag_fftshift_before_median:
            cc_CC = torch_fftshift(cc_CC)
        if self.flag_median_per_image:
            if self.flag_mean_instead_of_median:
                cc_CC_col_median = torch.mean(cc_CC, dim=-2, keepdim=True)
            else:
                cc_CC_col_median, cc_CC_col_median_indices = torch.median(cc_CC, dim=-2, keepdim=True)
        else:
            if self.flag_mean_instead_of_median:
                cc_CC_col_median = torch.mean(cc_CC_mean, dim=-2, keepdim=True)
            else:
                cc_CC_col_median, cc_CC_col_median_indices = torch.median(cc_CC_mean, dim=-2, keepdim=True)
        H, W = cc_CC.shape[-2:]
        ### Repeat and substract col median: ###
        cc_CC_col_median = cc_CC_col_median.repeat(1, 1, H, 1)  # TODO: can be saved with CUDA
        cc_CC = cc_CC - cc_CC_col_median
        if self.flag_fftshift_before_median:
            cc_CC = torch_fftshift(cc_CC)

        return cc_CC

    def Hist_Stretch_Tensors(self, input_tensor):
        # TODO: switch to CUDA if needed
        # TODO: make a parameter to update discretely or continuously q1 and q2
        if self.flag_stretch_tensors:
            if self.q1 is None:
                ### Stretch histogram for better viewing: ###
                input_tensor_stretched, (q1, q2) = scale_array_stretch_hist(input_tensor, flag_return_quantile=True, quantiles=(0.05, 0.95))
                self.q1 = q1
                self.q2 = q2
            else:
                input_tensor_stretched = scale_array_from_range(input_tensor.clip(self.q1, self.q2),
                                                                min_max_values_to_clip=(self.q1, self.q2),
                                                                min_max_values_to_scale_to=(0, 1))

            reference_tensor_stretched = input_tensor_stretched[0:1]
        else:
            input_tensor_stretched = input_tensor
            reference_tensor_stretched = input_tensor_stretched[0:1]

        return input_tensor_stretched, reference_tensor_stretched

    def forward(self, input_tensor):
        ### Stretch input_tensor and reference_tensor: ###
        gtic()
        input_tensor_stretched, reference_tensor_stretched = self.Hist_Stretch_Tensors(input_tensor)
        input_tensor_stretched_original = input_tensor_stretched * 1.0
        gtoc('stretch histogram')

        ### Get input tensor FFT: ###
        #TODO: YURI, transfer to half precision
        gtic()
        input_tensor_stretched_fft = torch.fft.fftn(input_tensor_stretched, dim=(-2, -1))
        reference_tensor_stretched_fft = input_tensor_stretched_fft[0:1]
        gtoc('Perform Initial FFT')

        ### Spatially Filter Frames (for less high frequency noise in the CC): ###
        if self.flag_gaussian_filter:
            # TODO: make this efficient, this is extremely slow and stupid
            # TODO: we can fuse between this and the cross correlation calculation below because the CC calculation involved FFT, and we can gaussian filter at FFT space by multiplication!!!!!
            # ### Real Space Gaussian Filtering: ###
            # input_tensor_stretched = self.gaussian_blur_layer.forward(input_tensor_stretched)
            # reference_tensor_stretched = self.gaussian_blur_layer.forward(reference_tensor_stretched)

            ### FFT Filtering: ###
            # TODO: there seems to be a shift between the fft filtering and the gaussian blur layer filtering method. understand where this comes from, maybe because of the padding or something? but in general it looks the same
            if self.gaussian_filter_fft is None:
                self.gaussian_filter_fft = torch.fft.fftn(self.gaussian_blur_layer.weight, s=input_tensor_stretched.shape[-2:], dim=(-2, -1))\

            ### Filter in FFT Space by multiplying by gaussian filter fft and IFFT: ###
            #TODO: YURI, implement this together with the above function in half precision
            gtic()
            input_tensor_stretched_fft = input_tensor_stretched_fft * self.gaussian_filter_fft
            reference_tensor_stretched_fft = reference_tensor_stretched_fft * self.gaussian_filter_fft
            gtoc('multiply FFT by gaussian filter kernel FFT')

            # ### IFFT just to take a look and the image and make sure everything is fine: ###
            # # TODO: probably to be deleted later
            # gtic()
            # input_tensor_stretched = torch.fft.ifftn(input_tensor_stretched_fft, dim=(-2, -1)).real
            # reference_tensor_stretched = torch.fft.ifftn(input_tensor_stretched_fft, dim=(-2, -1)).real
            # gtoc('IFFT after gaussian kernel fft multiplication')

            # ### Debugging: ###
            # imshow_torch(crop_tensor(input_tensor_stretched_filtered[10],1000))
            # imshow_torch(crop_tensor(input_tensor_stretched[10], 1000))

        # ### Perform Regualr Cross Correlation: ###
        # gtic()
        # cc_CC = circular_cross_correlation_classic(input_tensor_stretched, reference_tensor_stretched, normalize=False, fftshift=False)
        # cc_CC = cc_CC.unsqueeze(1)
        # H, W = cc_CC.shape[-2:]
        # midpoints = (H // 2, W // 2)
        # cc_CC = cc_CC.squeeze(1)
        # gtoc('circular cross correlation')

        ### Perform Cross Correlation Using The FFT Above: ###
        #TODO: YURI, transfer to half precision
        gtic()
        cc_CC = torch.fft.ifftn(input_tensor_stretched_fft * reference_tensor_stretched_fft.conj(), dim=[-1, -2]).real
        gtoc('circular cross correlation using above calculated FFT')
        # torch.save(cc_CC, r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\for_yuri/CC_tensor.pt')
        
        # ### "zero-out" center element with mean: ###
        #TODO: maybe try a different strategy later
        gtic()
        cc_CC[:, :, 0, 0] = cc_CC.mean([-2, -1])
        gtoc('Zero out zero shift component in Cross Correlation')

        ### Center CC Using FFTSHIFT: ###
        # TODO: right now i'm doing FFFTSHIFT but this is NOT!! necessary at the end probably because it seems that in order to subtract row/column median we don't need that, and we can always play with pixels.
        # TODO: we can do one of several things to get small amount of pixels:
        # (1). perform NCC with a predefined shift size and play with only the relevant indices.
        # (2). perform CCC and simply crop the relevant middle after fftshift.
        # (3). perform CCC and get the relevant pixels as in NCC
        gtic()
        cc_CC = torch_fftshift(cc_CC)
        gtoc('FFTSHIFT Cross Correlation Itself')

        # ### Stretch each image of the CCC to better show difference (understand if needed at all): ###
        # # TODO: probably won't be in the final algorithm, do not transfer to CUDA
        # if self.flag_zero_out_zero_component_each_CC:
        #     for i in np.arange(cc_CC.shape[0]):
        #         # # # (1). Replace center element to get more reflective range:
        #         cc_CC[i, :, 0, 0] = cc_CC[i, :, :, :].median()
        #         # (2). Scale each CC Image to between [0,1] to make the range consistent
        #         cc_CC[i] = scale_array_to_range(cc_CC[i])

        ### Substract Column and Row Median: ###
        #TODO: YURI, you can try and accelerate this with one memory access, along the way, please get the max element over all image (for every image)
        gtic()
        cc_CC = self.Substract_Column_and_Row_Median(cc_CC)
        gtoc('Substract Column and Row Median')
        # imshow_torch_video(torch_fftshift(cc_CC)[1:], FPS=25)

        ### Crop Center: ###
        # TODO: apparently we need the full CC row/col median in order to make it robust, which means we will need to fuse it with mean and row/col median. maybe we can cut time by finding the minimum ROI for this but still....
        # TODO: this means we will probably need a running median calculation?....interesting
        gtic()
        cc_CC = crop_tensor(cc_CC, (self.max_shift * 2 + 1, self.max_shift * 2 + 1))
        gtoc('Crop Cross Correlation')

        ### DEBUGGING: ###
        # cc_CC = torch_fftshift(crop_tensor(torch_fftshift(cc_CC), 100))
        # video_torch_array_to_video(scale_array_to_range(torch_fftshift(cc_CC), [0, 255]).type(torch.uint8), r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2/CC6.avi', FPS=25, flag_compressed=True)

        # imshow_torch_video(torch_fftshift(cc_CC), FPS=25)
        # cc_CC = crop_tensor(torch_fftshift(cc_CC), 100)
        # video_torch_array_to_video(scale_array_to_range(cc_CC, [0, 255]).type(torch.uint8), r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2/CC6.avi', FPS=25, flag_compressed=False)

        ### Get Shifts From New Cross-Correlation: ###
        # TODO: shift to CUDA, this is extremely slow
        # (1). Parabola Fit:
        # gtic()
        # shifts_H, shifts_W = shifts_from_circular_cc(cc_CC.unsqueeze(1), midpoints)
        # gtoc('parabola fit')
        # # (2). Gaussian Fit:
        # gtic()
        # subpixel_W, subpixel_H = find_subpixel_peak_gaussian_fit(cc_CC.squeeze(1),
        #                                                          W_matrix_flattened=None,
        #                                                          M_matrix_flattened=None,
        #                                                          flag_order='F',
        #                                                          X=None,
        #                                                          Y=None,
        #                                                          W_matrix=None,
        #                                                          flag_W_matrix_method=0,
        #                                                          flag_shift_CC_to_zero=False,
        #                                                          flag_round_shift_before_shifting_CC=False)
        # gtoc('gaussian peak')
        # (3). Gaussian Fit Torch_Layer:
        subpixel_W, subpixel_H = self.find_gaussian_peak_layer.forward(cc_CC)

        ### Shift Matrices: ###
        # TODO: save an FFT by using the one calculated from the CC, or use warp_method='bicubic'
        # TODO: use Warp_Object instead to avoid building grid every time, this is what is taking so long! enhance Warp_Object with FFT possibility (use yoav's object!!!)
        # TODO: accelerate Warp_Object to NOT perform torch.cat() and simply use bilinear grid interpolation!!!!
        gtic()
        # input_tensor_stretched_warped = shift_matrix_subpixel(input_tensor_stretched.unsqueeze(0), -subpixel_H, -subpixel_W, matrix_FFT=None, warp_method=self.warp_method).squeeze(0)
        # input_tensor_stretched_original_warped = shift_matrix_subpixel(input_tensor_stretched_original.unsqueeze(0), -subpixel_H, -subpixel_W, matrix_FFT=None, warp_method=self.warp_method).squeeze(0)
        # input_tensor_stretched_warped = Warp_Object.forward(input_tensor_stretched.unsqueeze(0), -subpixel_W, -subpixel_H, self.warp_method)
        input_tensor_stretched_original_warped = self.warp_object_layer.forward(input_tensor_stretched_original,
                                                                                subpixel_W.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                                                                subpixel_H.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), self.warp_method)
        # imshow_torch(input_tensor_stretched_original_warped.mean(0))
        gtoc('align image after discovering shifts')

        # ### Average Frames Running Window: ###
        # number_of_frames_to_average = 11
        # conv_layer = convn_layer_torch()
        # input_tensor_stretched_warped_running_mean = conv_layer.forward(input_tensor_stretched_warped, kernel=torch.ones(number_of_frames_to_average) / number_of_frames_to_average, dim=0)

        ### Average All Frames: ###
        # input_tensor_stretched_warped_mean = input_tensor_stretched_warped.mean(0, True)
        input_tensor_stretched_original_warped_mean = input_tensor_stretched_original_warped.mean(0, True)

        # ### Perform RCC on tensor: ###
        # # TODO: insert gaussian denoising scheme in
        # # TODO: make this run on GPU and later on switch to CUDA if extermely needed
        # gtic()
        # # input_tensor_stretched_warped_mean_RCC_corrected, _ = perform_RCC_on_tensor(input_tensor_stretched_warped_mean.cpu(), number_of_iterations=1)
        # input_tensor_stretched_warped_mean_RCC_corrected, _ = perform_RCC_on_tensor(input_tensor_stretched_original_warped_mean.cpu(), number_of_iterations=1)
        # gtoc('RCC on average image')
        # imshow_torch(input_tensor_stretched_warped_mean_RCC_corrected)
        # imshow_torch_histogram_stretch(input_tensor_stretched_warped_mean_RCC_corrected)
        # imshow_torch(input_tensor_stretched_warped_mean)
        # imshow_torch(input_tensor_stretched_original_warped_mean)

        # ### Show Images: ###
        # imshow_torch_video(torch.cat([input_tensor_stretched, input_tensor_stretched_warped_running_mean], -1), FPS=25)
        # imshow_torch_video(torch.cat([input_tensor_stretched, input_tensor_stretched_warped_running_mean], -1)[11:-11], FPS=25)
        # input_tensor_stretched_warped_mean_repeated = input_tensor_stretched_warped.mean(0,True).repeat(input_tensor_stretched.shape[0],1,1,1)
        # input_tensor_stretched_warped_mean_RCC_corrected, _ = perform_RCC_on_tensor(input_tensor_stretched_warped.mean(0,True).cpu(), number_of_iterations=1)
        # input_tensor_stretched_warped_mean_RCC_corrected = input_tensor_stretched_warped_mean_RCC_corrected.cuda()
        # input_tensor_stretched_warped_mean_RCC_corrected_repeated = input_tensor_stretched_warped_mean_RCC_corrected.repeat(input_tensor_stretched.shape[0],1,1,1)
        # imshow_torch(torch.cat([input_tensor_stretched_warped_mean_RCC_corrected, input_tensor_stretched_warped.mean(0,True)],-1))
        # concat_tensor = torch.cat([input_tensor_stretched, input_tensor_stretched_warped_mean_repeated], -1)
        # imshow_torch_video(concat_tensor, FPS=25)
        # noisy_tensor = RGB2BW(video_video_to_images_torch(r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2/very_noisy_compressed.mp4', 100))
        # noisy_tensor = noisy_tensor[:,:,:,260:860]
        # noisy_tensor = torch.nn.Upsample(size=(input_tensor_stretched.shape[-2:]))(noisy_tensor)
        # new_concat_tensor = torch.cat([scale_array_to_range(noisy_tensor.cuda()), scale_array_to_range(input_tensor_stretched_warped_mean_RCC_corrected_repeated)],-1)
        # imshow_torch_video(noisy_tensor, FPS=25)
        # imshow_torch_video(new_concat_tensor, FPS=25)
        # video_torch_array_to_video(new_concat_tensor, r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2/very_noisy_cleaned.avi', FPS=25, flag_compressed=True)
        # video_torch_array_to_video(input_tensor_stretched, r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2/very_noise.avi', FPS=25, flag_compressed=False)
        # imshow_torch(torch.cat([input_tensor_stretched[0], input_tensor_stretched_warped.mean(0)], -1))

        return subpixel_H, subpixel_W, input_tensor_stretched_original_warped_mean


def get_max_correct_form_torch(input_tensor, dim_to_leave):
    ### Assuming input_tensor.shape = [B,C,H,W] or 4D Tensor in any case: ###

    if dim_to_leave==0: #get max value for each batch example (max among all values of that batch example including all channels)
        values, indices = input_tensor.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        values = values.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #Return to pytorch format: [B,1,1,1]
    if dim_to_leave==1:  # get max value for each channel (max among all values of including batch dimension)
        values, indices = input_tensor.max(0)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        values = values.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]
    if dim_to_leave==(0,1) or dim_to_leave==[0,1]:  # get max value of each feature map
        if len(values.shape) > 2:
            values, indices = values.max(2)
        if len(values.shape) > 1:
            values, indices = values.max(2)
        values = values.unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]

    return values



class Find_Subpixl_Peak_Gaussian_Fit(nn.Module):
    def __init__(self, flag_order='F', flag_W_matrix_method=0, flag_shift_CC_to_zero=False, flag_round_shift_before_shifting_CC=False, flag_CC_shift_method='bicubic', max_shift=3, max_shift_for_fit=15, W_matrix_circle_radius=21):
        super().__init__()
        self.X = None
        self.Y = None
        self.X_flattened = None
        self.Y_flattened = None
        self.x_vec = None
        self.M_matrix_flattened = None
        self.W_matrix_flattened = None
        self.W_matrix = None
        self.flag_order = flag_order
        self.flag_W_matrix_method = flag_W_matrix_method
        self.flag_shift_CC_to_zero = flag_shift_CC_to_zero
        self.flag_round_shift_before_shifting_CC = flag_round_shift_before_shifting_CC
        self.flag_CC_shift_method = flag_CC_shift_method  #'discrete', 'FFT', 'bilinear', 'bicubic'
        self.max_shift = max_shift
        self.max_shift_for_fit = max_shift_for_fit
        self.total_max_location_over_t_tensor = None
        self.W_matrix_circle_radius = W_matrix_circle_radius

        ### Initialize M_matrix: ###
        self.Initialize_M_matrix()

    def Initialize_M_matrix(self):
        ### If X or Y Are Needed -> Build Them: ###
        if self.X is None or self.Y is None or self.W_matrix_flattened is None or self.M_matrix_flattened is None:
            ### Get weights for gaussian fit: ###
            x_vec = torch.arange(-self.max_shift_for_fit, self.max_shift_for_fit + 1)
            self.Y, self.X = torch.meshgrid(x_vec, x_vec)
            self.X = self.X.cuda()
            self.Y = self.Y.cuda()

        ### Build M_matrix if wanted: ###
        if self.M_matrix_flattened is None:
            if self.flag_order == 'F':
                # (1). 'F' order
                self.X_flattened = torch_flatten_image(self.X, order='F')
                self.Y_flattened = torch_flatten_image(self.Y, order='F')
                ones_vec = torch.ones_like(self.X_flattened)
                self.M_matrix_flattened = torch.cat([self.X_flattened ** 2, self.Y_flattened ** 2, self.X_flattened, self.Y_flattened, ones_vec], -1)
                self.M_matrix_flattened = torch.permute(self.M_matrix_flattened, [1, 0])
            elif self.flag_order == 'C':
                # (2). 'C' order:
                self.X_flattened = self.X.flatten().unsqueeze(-1)
                self.Y_flattened = self.Y.flatten().unsqueeze(-1)
                ones_vec = torch.ones_like(self.X_flattened)
                self.M_matrix_flattened = torch.cat([self.X_flattened ** 2, self.Y_flattened ** 2, self.X_flattened, self.Y_flattened, ones_vec], -1)
                self.M_matrix_flattened = torch.permute(self.M_matrix_flattened, [1, 0])


    def get_max_location_over_CC(self, C_input, flag_get_subpixel_max_location_using_parabola_fit=True):
        #TODO: for some reason it seams
        if self.total_max_location_over_t_tensor is None:
            subpixel_max_over_t, self.total_max_location_over_t_tensor = get_max_sub_pixel_over_t(C_input, flag_get_subpixel_max_location_using_parabola_fit=flag_get_subpixel_max_location_using_parabola_fit)  # TODO: CUDA ACCELERATE and also allow to only search for discrete max!!!
            self.total_max_location_over_t_tensor += 0.5
        self.max_location_over_t_H = self.total_max_location_over_t_tensor[:, 0]
        self.max_location_over_t_W = self.total_max_location_over_t_tensor[:, 1]
        self.max_location_over_t_W = self.max_location_over_t_W.unsqueeze(-1).unsqueeze(-1)
        self.max_location_over_t_H = self.max_location_over_t_H.unsqueeze(-1).unsqueeze(-1)
        self.max_location_over_t_W_discrete = self.max_location_over_t_W.round()
        self.max_location_over_t_H_discrete = self.max_location_over_t_H.round()

    def shift_CC_towards_zero_position(self, C_input):
        if self.flag_shift_CC_to_zero:
            ### Decide whether to perform discrete shift or subpixel shift: ###
            if self.flag_round_shift_before_shifting_CC:
                shifts_H = self.max_location_over_t_H_discrete
                shifts_W = self.max_location_over_t_W_discrete
            else:
                shifts_H = self.max_location_over_t_H
                shifts_W = self.max_location_over_t_W
            shifts_H = shifts_H.squeeze()
            shifts_W = shifts_W.squeeze()

            ### Shift: ###
            if self.flag_CC_shift_method == 'discrete':
                C_input = shift_matrix_integer_pixels(C_input.unsqueeze(1), -shifts_H, -shifts_W).squeeze(1)
            else:
                #TODO: switch this to a class to avoid reinitializing X,Y !!!!!!
                C_input = shift_matrix_subpixel(C_input.unsqueeze(1), -shifts_H, -shifts_W, matrix_FFT=None, warp_method=self.flag_CC_shift_method).squeeze(0)

        return C_input

    def Initialize_W_marix(self, C_input):
        ### Create W_matrix Flattened On The Spot: ###  #TODO: make all of this more efficient with the minimal amount of initializations needed
        #TODO: enable interpolating the C_input to zero position or moving it discretely towards max
        # (1). Centered Pre-Set W_Matrix for all cross-correlations (centered gaussian profile with set sigma for all):

        ### Get max location over time if needed for the Weights matrix later + Shift matrix to zero location if wanted: ###
        if self.flag_W_matrix_method in [2, 3] or self.flag_shift_CC_to_zero:  # (*). if this is the case --> probably i can make the W_matrix be about constant and save this function running every run!!!!
            ### Get Max Location Over Time: ###
            self.get_max_location_over_CC(C_input, flag_get_subpixel_max_location_using_parabola_fit=False)

            ### Shift CC To Zero: ###
            if self.flag_shift_CC_to_zero:
                C_input = self.shift_CC_towards_zero_position(C_input)
                shifts_W = self.max_location_over_t_W - self.max_location_over_t_W_discrete
                shifts_H = self.max_location_over_t_H - self.max_location_over_t_H_discrete
            else:
                shifts_W = self.max_location_over_t_W
                shifts_H = self.max_location_over_t_H

        ### Choose Weight Matrix Distribution: ###
        # (*). Predefined!!!!
        if self.W_matrix is None and self.flag_W_matrix_method in [0, 1]:
            # (0). Predefined Gaussian Weights
            if self.flag_W_matrix_method == 0:
                gaussian_sigma = 2
                self.W_matrix = torch.exp(-(self.X ** 2 + self.Y ** 2) / gaussian_sigma ** 2).unsqueeze(0)
            # (1). Predefined Circle:
            elif self.flag_W_matrix_method == 1:
                # circle_radius = 30  #TODO: maybe make this a parameter
                self.W_matrix = (self.X**2 + self.Y**2) <= self.W_matrix_circle_radius**2
                self.W_matrix = self.W_matrix.float()
            # (2). Shifted Pre-Set W_Matrix, a gaussian profile with set sigma centered around the max position at each cross correlation:

        # (*) INPUT DEPENDENT!!!!
        else:

            if self.flag_W_matrix_method == 2:
                gaussian_sigma = 2
                self.W_matrix = torch.exp(-((self.X - shifts_W) ** 2 + (self.Y - shifts_H) ** 2) / gaussian_sigma ** 2)
            # (3). A Circle/Gussian-Binary Cut-Off A Certain Distance From The Peak:
            elif self.flag_W_matrix_method == 3:
                #TODO: enable also simple predefined circle
                gaussian_value_threshold = 0.4
                gaussian_sigma = 2
                self.W_matrix = torch.exp(-((self.X - shifts_W) ** 2 + (self.Y - shifts_H) ** 2) / gaussian_sigma ** 2)
                self.W_matrix = (self.W_matrix > gaussian_value_threshold).float()
            # (4). A W_Matrix derived from the values of the cross correlation matrix itself:
            # (*). Seems to not do the job!!!!
            elif self.flag_W_matrix_method == 4:
                self.W_matrix = torch.ones_like(C_input)
                for i in np.arange(self.C_input.shape[0]):
                    self.W_matrix[i:i + 1] = scale_array_to_range(self.C_input[i:i + 1])
                self.W_matrix = self.W_matrix ** 0.33

        ### Scale CC to be between [0,1]: ###
        #TODO: when we will transfer to fft normalized cross correlation this will come out of the box probably
        max_per_frame = get_max_correct_form_torch(C_input, 0)  #TODO: unify in CUDA with row/col median substraction
        C_input = C_input / max_per_frame

        ### Crop CC (now that i've centered it) To Max Shift For Fit: ###
        C_input = crop_tensor(C_input, self.max_shift_for_fit*2+1)

        ### Correct W_matrix To Be Zero Where Cross Correlation Is Low (zero component): ###
        #TODO: unify in CUDA with row/col median substraction
        CC_zero_logical_mask = C_input > 1e-1
        self.W_matrix = self.W_matrix * CC_zero_logical_mask

        ### Flatten W_matrix: ###
        if self.W_matrix_flattened is None:
            if self.flag_order == 'F':
                self.W_matrix_flattened = torch_flatten_image(self.W_matrix, order='F')
            else:
                self.W_matrix_flattened = self.W_matrix.flatten(-2, -1)

        return C_input

    def forward(self, C_input):
        ### Clip values smaller or equal to zero before extracting log: ###
        C_input[C_input <= 0] = 2.22e-16  # matlab's eps
        if len(C_input.shape) == 4:
            C_input = C_input.squeeze(1)  # [T,C,H,W] -> [T,H,W]
        T, H, W = C_input.shape

        ### Initialize M_matrix: ###
        self.Initialize_M_matrix()

        ### Initialize W_matrix: ###
        gtic()
        C_input = self.Initialize_W_marix(C_input)
        gtoc('Find discrete Max in CC and initialize W_matrix')

        ### Crop Tensors: ###

        ### Flatten C_input: ###
        # TODO: there must be a better way then using F ordering and doing the squeeze-unsqueeze stuff!!!
        # TODO: this is still a very small amount of data...maybe i'm giving too much attention to the 'F' ordering vs. "efficient" 'C' ordering? maybe this is stupid and i should focus on other stuff?
        gtic()
        if self.flag_order == 'F':
            C_input_flattened = torch_flatten_image(torch.log(C_input+1e-6), order='F').squeeze(-1)
        elif self.flag_order == 'C':
            ### Use conventional flattening: ###
            # C_input_flattened = torch.flatten(torch.log(C_input))
            C_input_flattened = torch_flatten_image(torch.log(C_input+1e-2), order='C', flag_make_Nx1=False)
        gtoc('flatten CC and extract log')

        ### Perform Least-Squares Fit (basically paraboloid fit on the log of the input is almost as a least squares fit of a gaussian): ###
        # [W_matrix_flattened] = [1, H*W]
        # [M_matrix_flattened] = [F, H*w]
        # [M_matrix_flattened].unsqueeze(0) = [1,F,H*W]
        # [C_input_flattened] = [T, H*W]
        # TODO: CUDA accelerate all here
        gtic()
        A_matrix = self.W_matrix_flattened * self.M_matrix_flattened.unsqueeze(0)  # [A_matrix] = [1,F,H*W]
        # B_matrix = self.W_matrix_flattened * C_input_flattened.unsqueeze(1)  # [B_matrix] = [T,1,H*W]
        B_matrix = self.W_matrix_flattened * C_input_flattened  # [B_matrix] = [T,1,H*W]
        gtoc('Multiply A=W*M & B=W*CC')

        gtic()
        A_matrix = torch.permute(A_matrix, [0, 2, 1])  # [A_matrix] = [1,F,H*W] -> [1,H*W,F]
        B_matrix = torch.permute(B_matrix, [0, 2, 1])  # [B_matrix] = [T,1,H*W] -> [T,H*W,1] #(*). the A and B matrices basically agree with matlab except for where there are exceptionally low numbers for some reason. maaybe because matlab is single percision!!?!?!?
        gtoc('Permute A and B')

        ### New Trial Least-Squares: ###
        #TODO: YURI, understand if there's anything we can do. right now it's about 15[msec] which is a lot but not a huge bottleneck
        # CUDA aceelerate the below lstqs if possible
        # [A_matrix] = [1, H*W, F]
        # [B_matirx] = [T, H*W, 1]
        # A_matrix = A_matrix.contiguous()
        # B_matrix = B_matrix.contiguous()
        gtic()
        T = C_input.shape[0]
        # A_matrix_new = A_matrix
        # B_matrix_new = B_matrix
        abcde = torch.linalg.lstsq(A_matrix, B_matrix).solution.squeeze(-1)  # TODO: 'C' ordering isn't possible to insert into torch.linalg.lstsq because it expects something else anyway!!!
        subpixel_X = -abcde[:, 2] / (2 * abcde[:, 0])
        subpixel_Y = -abcde[:, 3] / (2 * abcde[:, 1])
        a1 = abcde[:, 0]
        b1 = abcde[:, 1]
        c1 = abcde[:, 2]
        d1 = abcde[:, 3]
        e1 = abcde[:, 4]
        gtoc('Least squares')

        C_peak = e1 - c1 ** 2 / (4 * a1) - d1 ** 2 / (4 * b1)
        C_peak = torch.exp(C_peak)
        bla = 1
        gtic()
        T = C_input.shape[0]
        # A_matrix_new = A_matrix
        # B_l_X
        final_shifts_X = self.max_location_over_t_W.squeeze() + subpixel_X
        final_shifts_Y = self.max_location_over_t_H.squeeze() + subpixel_Y

        # ### Plot Output Paraboloid and Gaussian: ###
        # X = self.X.unsqueeze(0)
        # Y = self.Y.unsqueeze(0)
        # a1 = a1.unsqueeze(-1).unsqueeze(-1)
        # b1 = b1.unsqueeze(-1).unsqueeze(-1)
        # c1 = c1.unsqueeze(-1).unsqueeze(-1)
        # d1 = d1.unsqueeze(-1).unsqueeze(-1)
        # e1 = e1.unsqueeze(-1).unsqueeze(-1)
        # paraboloid_output = a1*X**2 + b1*Y**2 + c1*X + d1*Y + e1
        # gaussian_output = torch.exp(paraboloid_output)
        # gaussian_fit_minus_log_CC = (gaussian_output.unsqueeze(1) - C_input)
        # ### 2D Plots: ###
        # imshow_torch((paraboloid_output[1]), title_str='paraboloid output')
        # imshow_torch((gaussian_output[1]), title_str='gaussian output')
        # imshow_torch(C_input[1], title_str='CC itself')
        # # imshow_torch(torch.log(C_input+1e-2)[1], title_str='CC after extraacting log')
        # imshow_torch(gaussian_fit_minus_log_CC, title_str='CC minus Gaussian Fit')
        #
        #
        # imshow_torch(self.W_matrix, title_str='W matrix')
        # ### 3D Meshes/Surface Plots: ### #TODO: add plot_surface_torch to RapidBase!!@#^
        # C_input_numpy = C_input.cpu().numpy()
        # C_input_log_numpy = torch.log(C_input).cpu().numpy()
        # W_matrix_numpy = W_matrix.cpu().numpy()
        # paraboloid_output_numpy = paraboloid_output.cpu().numpy()
        # gaussian_output_numpy = gaussian_output.cpu().numpy()
        # X_numpy = X[0].cpu().numpy()
        # Y_numpy = Y[0].cpu().numpy()
        # #(*). Plot CC input with Gaussian Fit:
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(X_numpy, Y_numpy, C_input_numpy[0], rstride=1, cstride=1, cmap='viridis', edgecolor='none');
        # ax.plot_surface(X_numpy, Y_numpy, gaussian_output_numpy[0], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        # plt.title('CC input and Gaussian')
        # #(*). Plot CC and W_matrix:
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(X_numpy, Y_numpy, C_input_numpy[0], rstride=1, cstride=1, cmap='viridis', edgecolor='none');
        # ax.plot_surface(X_numpy, Y_numpy, W_matrix_numpy, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        # plt.title('CC input and W_matrix')

        ### Plot: ###
        # imshow_torch(paraboloid_output[1])
        # imshow_torch_video(paraboloid_output.unsqueeze(1), FPS=25)
        # imshow_torch_video(gaussian_output.unsqueeze(1)[0:10], FPS=2)
        # imshow_torch_video(paraboloid_output.unsqueeze(1)[0:10], FPS=2)

        return final_shifts_X, final_shifts_Y



def torch_flatten_image(input_tensor, flag_make_Nx1=True, order='C'):
    #TODO: this assumes i want to flatten everything but the batch dimensions...which seems legit
    # torch_variable = torch_variable.view(torch_variable.size(0), -1)
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if order == 'F':
        input_tensor = einops.rearrange(input_tensor, '... h w -> ... w h')
        input_tensor = input_tensor.flatten(-2,-1)
    elif order == 'C':
        input_tensor = input_tensor.flatten(-2,-1)

    if flag_make_Nx1:
        input_tensor = input_tensor.unsqueeze(-1)

    return input_tensor

def get_full_shape_torch(input_tensor):
    if len(input_tensor.shape) == 1:
        W = input_tensor.shape
        H = 1
        C = 1
        T = 1
        B = 1
        shape_len = 1
        shape_vec = (W)
    elif len(input_tensor.shape) == 2:
        H, W = input_tensor.shape
        C = 1
        T = 1
        B = 1
        shape_len = 2
        shape_vec = (H,W)
    elif len(input_tensor.shape) == 3:
        C, H, W = input_tensor.shape
        T = 1
        B = 1
        shape_len = 3
        shape_vec = (C,H,W)
    elif len(input_tensor.shape) == 4:
        T, C, H, W = input_tensor.shape
        B = 1
        shape_len = 4
        shape_vec = (T,C,H,W)
    elif len(input_tensor.shape) == 5:
        B, T, C, H, W = input_tensor.shape
        shape_len = 5
        shape_vec = (B,T,C,H,W)
    shape_vec = np.array(shape_vec)
    return (B,T,C,H,W), shape_len, shape_vec



class InvalidMethodError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


def pick_method(functions, method, *args, **kwargs): # can return anything
    # pass functions in dictionary, and args in args and kwargs. Saves many lines of code
    # prune args of Null arguments
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("+++++++++++++++++++++++++")
    """
    args = [s_arg for s_arg in args if s_arg is not None]  # s_arg = single_arg
    kwargs = {arg_key: kwargs[arg_key] for arg_key in kwargs if kwargs[arg_key] is not None}
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("\n\n\n\n\n********************************************\n\n\n\n\n")
    """

    if method in functions.keys():
        return functions[method](*args, **kwargs)
    else:
        raise InvalidMethodError("Given method not valid")


def not_implemented():
    raise NotImplementedError


def shift_matrix_subpixel(matrix: Union[torch.Tensor, np.array, tuple, list],
                          shift_H: Union[torch.Tensor, list, tuple, float, int],
                          shift_W: Union[torch.Tensor, list, tuple, float, int],
                          matrix_FFT=None, warp_method='bilinear') -> torch.Tensor:
    """Performs subpixel shift on given matrix. Consider pixels over the side of frame to have undefined behavior.

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be shifted
    :param shift_H: singleton or length T dimension vector of vertical shift/s. Shifts downwards
    :param shift_W: singleton or length T dimension vector of horizontal shift/s. Shifts rightwards
    :param matrix_FFT: For use only when using 'fft' warp method. In case the fft of the matrix has already been calculated, it can be passed to the function to improve performance. FFT must be over dimensions -2, -1
    :param warp_method: method used to warp the matrix when shifting. Default: 'bilinear'. Options: 'bilinear', 'bicubic', 'nearest' and 'fft'
    :return: matrix shifted according to given shifts
    """
    # possible_methods/pick methods allows for checking for improper methods as well as a more concise selection code
    matrix, shift_H, shift_W, matrix_FFT, warp_method = format_subpixel_shift_params(matrix, shift_H, shift_W, matrix_FFT, warp_method)
    dimensions_memory = MatrixOrigami(matrix)
    expanded_matrix = dimensions_memory.expand_matrix()
    possible_methods = {'fft': _shift_matrix_subpixel_fft, 'bilinear': _shift_matrix_subpixel_interpolated,
                        'bicubic': _shift_matrix_subpixel_interpolated, 'nearest': _shift_matrix_subpixel_interpolated}
    shifted_matrix = pick_method(possible_methods, warp_method, expanded_matrix, shift_H, shift_W, matrix_FFT, warp_method=warp_method)
    return dimensions_memory.squeeze_to_original_dims(shifted_matrix)



def _shift_matrix_subpixel_interpolated(matrix: torch.Tensor, shift_H: Tensor, shift_W: Tensor, warp_method='bilinear') -> torch.Tensor:
    B, T, C, H, W = matrix.shape
    ##Question: rounded to int??
    N = B*T
    #TODO: dudy: don't!!! reinstantiate the grid all over every time!!! use a layer!!!!!
    affine_matrices = batch_affine_matrices((H,W), N, shifts=(extend_vector_length_n(shift_H, N), extend_vector_length_n(shift_W, N)))
    output_grid = torch.nn.functional.affine_grid(affine_matrices,
                                                  torch.Size((N, C, H, W))).to(matrix.device)
    matrix = matrix.reshape((B * T, C, H, W))  # grid sample can only handle 4D. #TODO: dudy: use view instead of reshape
    output_tensor = torch.nn.functional.grid_sample(matrix, output_grid, mode=warp_method)
    return output_tensor.reshape((B, T, C, H, W))

def calculate_meshgrids(input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
    ndims = len(input_tensor.shape)
    H = input_tensor.shape[-2]
    W = input_tensor.shape[-1]
    # Get tilt phases k-space:
    y = torch.arange(-math.floor(H / 2), math.ceil(H / 2), 1)
    x = torch.arange(-math.floor(W / 2), math.ceil(W / 2), 1)
    delta_f1 = 1 / H
    delta_f2 = 1 / W
    f_y = y * delta_f1
    f_x = x * delta_f2
    # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_y = torch.fft.fftshift(f_y)
    f_x = torch.fft.fftshift(f_x)

    # Build k-space meshgrid:
    [kx, ky] = torch.meshgrid(f_y, f_x, indexing='ij')
    # Frequency vec to tensor:
    for i in range(ndims - 2):
        kx = kx.unsqueeze(0)
        ky = ky.unsqueeze(0)
    if kx.device != input_tensor.device:
        kx = kx.to(input_tensor.device)
        ky = ky.to(input_tensor.device)
    return kx, ky


def _shift_matrix_subpixel_fft(matrix: torch.Tensor, shift_H: Tensor, shift_W: Tensor, matrix_fft: Tensor = None, warp_method: str = 'fft') -> torch.Tensor:
    """
    :param matrix: 5D matrix
    :param shift_H: either singleton or of length T
    :param shift_W: either singleton or of length T
    :param matrix_fft: fft of matrix, if precalculated
    :return: subpixel shifted matrix
    """
    ky, kx = calculate_meshgrids(matrix)
    #shift_W, shift_H = expand_shifts(matrix, shift_H, shift_W)
    ### Displace input image: ###
    displacement_matrix = torch.exp(-(1j * 2 * torch.pi * ky * shift_H + 1j * 2 * torch.pi * kx * shift_W)).to(matrix.device)
    fft_image = faster_fft(matrix, matrix_fft, dim=[-1, -2])
    fft_image_displaced = fft_image * displacement_matrix
    original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()
    return original_image_displaced





def raise_if(condition: bool, error: Exception = RuntimeError, message: str = "RuntimeError") -> None:
    if condition:
        raise error(message)


def raise_if_not(condition: bool, error: Exception = RuntimeError, message: str = "RuntimeError") -> None:
    if not condition:
        raise error(message)


def raise_if_not_close(a: Union[int, float], b: Union[float, int], error: Exception = RuntimeError,
                       message: str = "RuntimeError", closeness_distance: float = 1e-7) -> None:
    if closeness_distance < abs(a-b):
        raise error(message)



def format_shift(matrix: Tensor, shift: torch.Tensor) -> Tensor:
    """
    :param matrix: matrix shift will be applied to
    :param shift: either singleton or length N 1D vector. Exception will be raised if these do not hold
    :return: singleton vector or 1D vector of length time domain of the input matrix
    """
    if true_dimensionality(shift) > 0: # has more than one element. Must be same length as time dimension
        raise_if_not(true_dimensionality(shift) == 1, message="Shifts must be 1D vector or int") # shouldn't be 2D
        raise_if_not(true_dimensionality(matrix) >= 4, message="Shifts has larger dimensionality than input matrix") # matrix has time dimension
        raise_if_not(matrix.shape[-4] == shift.shape[0], message="Shifts not same length as time dimension") # has to have same length as time
        for i in range(3):
            shift = shift.unsqueeze(-1)
        return shift
    else:
        raise_if(len(shift) == 0)
        return shift



def type_check_parameters(arguments: List[Tuple]) -> None:
    """
    :param arguments: List of tuples of params and what their types can be
    """
    for num_arg, arg in enumerate(arguments):
        if type(arg[1]) in [list, tuple]:
            raise_if_not(type(arg[0]) in arg[1], message=f"Type {type(arg[0])} not valid for argument {num_arg}. Valid types: {arg[1]}")
        else:
            raise_if_not(type(arg[0]) == arg[1], message=f"Type {type(arg[0])} not valid for argument {num_arg}. Valid type: {arg[1]}")


def validate_warp_method(method: str, valid_methods=['bilinear', 'bicubic', 'nearest', 'fft']) -> None:
    raise_if_not(method in valid_methods, message="Invalid method")


def is_integer_argument(argument: Union[int, float, Tensor, list, Tuple]):
    if type(argument) == int:
        return True
    elif type(argument) in [list, tuple, Tensor]:
        return is_integer_iterable(argument)
    else:
        return False


def is_integer_builtin_iterable(iterable: Union[list, Tuple]) -> bool:
    for element in iterable:
        if type(element) != int:
            return False
    return True


def is_integer_iterable(iterable: Union[Tensor, list, Tuple]) -> bool:
    if type(iterable) == tuple or type(iterable) == tuple:
        return is_integer_builtin_iterable(iterable)
    elif type(iterable) == Tensor:
        return len(torch.nonzero(iterable-iterable.to(torch.int64))) == 0
    else:
        raise TypeError("Unsupported type for is_integer_iterable")

def format_subpixel_shift_params(matrix: Union[torch.Tensor, np.array, tuple, list], shift_H: Union[torch.Tensor, list, Tuple, float, int],
                                 shift_W: Union[torch.Tensor, list, Tuple, float, int], matrix_FFT=None, warp_method='bilinear')\
                                    -> Tuple[Tensor, Tensor, Tensor, Tensor, str]:
    type_check_parameters([(matrix, (torch.Tensor, np.ndarray, tuple, list)), (shift_H, (Tensor, np.ndarray, list, tuple, int, float)),
                           (shift_W, (Tensor, list, tuple, int, float, np.ndarray)), (warp_method, str), (matrix_FFT, (type(None), Tensor))])
    validate_warp_method(warp_method)
    matrix = construct_tensor(matrix)
    if matrix_FFT is not None:
        raise_if_not(warp_method=='fft', message="FFT should only be passed when using the FFT warp method")
        raise_if_not(equal_size_tensors(matrix_FFT, matrix))
    shift_H = construct_tensor(shift_H).to(matrix.device)
    shift_W = construct_tensor(shift_W).to(matrix.device)
    shift_H = format_shift(matrix, shift_H)
    shift_W = format_shift(matrix, shift_W)
    if shift_H.shape[0] > 1 or shift_W.shape[0] > 1:
        time_dimension_length = dimension_N(matrix, 4)
        shift_H = extend_tensor_length_N(shift_H, time_dimension_length)
        shift_W = extend_tensor_length_N(shift_W, time_dimension_length)
    return matrix, shift_H, shift_W, matrix_FFT, warp_method



def format_shift_integer_pixels_parameters(matrix: Union[Tensor, list, tuple, np.array],
                                           shift_H: Union[torch.Tensor, list, tuple, int],
                                           shift_W: Union[torch.Tensor, list, tuple, int]) -> Tuple[Tensor, Tensor, Tensor]:
    type_check_parameters([(matrix, (Tensor, list, tuple, np.ndarray)), (shift_H, (np.ndarray, Tensor, list, tuple, int)), (shift_W, (np.ndarray, Tensor, list, tuple, int))])
    matrix = construct_tensor(matrix)
    shift_H = construct_tensor(shift_H, matrix.device)
    shift_W = construct_tensor(shift_W, matrix.device)
    shift_H = format_shift(matrix, shift_H).to(torch.int32)
    shift_W = format_shift(matrix, shift_W).to(torch.int32)
    if shift_H.shape[0] > 1 or shift_W.shape[0] > 1:
        time_dimension_length = dimension_N(matrix, 4)
        shift_H = extend_tensor_length_N(shift_H, time_dimension_length)
        shift_W = extend_tensor_length_N(shift_W, time_dimension_length)
    print(shift_H)
    print(type(shift_H[0]))
    return matrix, shift_H, shift_W



def shift_n_pixels_uniform(matrix: torch.Tensor, shift_H: Tensor, shift_W: Tensor) -> Tensor:
    # shift matrix N pixels uniformly
    return torch.roll(matrix, [int(shift_H), int(shift_W)], dims=[-2, -1])


def shift_n_pixels(matrix: torch.Tensor, shift_H: Tensor, shift_W: Tensor) -> Tensor:
    """
    :param matrix: input matrix
    :param shift_H: 1D vector of length T(time dimension)
    :param shift_W: 1D vector of length T(time dimension)
    :return: shifted matrix
    """
    B,T,C,H,W = matrix.shape
    ret = []
    for i in range(B):
        current_batch = []
        for j in range(T):
            current_3D_matrix = matrix[i,j]
            current_batch.append(torch.roll(current_3D_matrix, [int(shift_H[j]), int(shift_W[j])], dims=[-2,-1]))
        ret.append(torch.stack(current_batch))
    return torch.stack(ret)


def shift_matrix_integer_pixels(matrix: Union[torch.Tensor, np.array, tuple, list],
                                shift_H: Union[torch.Tensor, np.array, list, tuple, int],
                                shift_W: Union[torch.Tensor, np.array, list, tuple, int]) -> torch.Tensor:
    """Shifts image integer pixels. Consider pixels over the side of frame to have undefined behavior
    Pixels will usually roll over, but this is not guaranteed

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be shifted
    :param shift_H: singleton or length T dimension vector of vertical shift/s. Shifts downwards
    :param shift_W: singleton or length T dimension vector of horizontal shift/s. Shifts rightwards
    :return: matrix shifted according to given shifts
    """
    matrix, shift_H, shift_W = format_shift_integer_pixels_parameters(matrix, shift_H, shift_W)
    dimensions_memory = MatrixOrigami(matrix)
    expanded_matrix = dimensions_memory.expand_matrix(5)
    if shift_W.shape[0] != 1 or shift_H.shape[0] != 1:
        shifted_matrix = shift_n_pixels(expanded_matrix, shift_H, shift_W)
    else:
        shifted_matrix = shift_n_pixels_uniform(expanded_matrix, shift_H, shift_W)
    return dimensions_memory.squeeze_to_original_dims(shifted_matrix)



def transformation_matrix_2D(center: Tuple[int, int], angle: float = 0, scale: float = 1, shifts: Tuple[float, float] = (0, 0)) -> Tensor:
    alpha = scale * math.cos(angle)
    beta = scale * math.sin(angle)
    #Issue
    affine_matrix = Tensor([[alpha, beta, (1-alpha)*center[1] - beta * center[0]],
                   [-beta, alpha, beta*center[1]+(1-alpha)*center[0]]])
    affine_matrix[1, 2] += float(shifts[0])  # shift_y
    affine_matrix[0, 2] += float(shifts[1])  # shift_x
    transformation_matrix = torch.zeros((3, 3))
    transformation_matrix[2, 2] = 1
    transformation_matrix[0:2, :] = affine_matrix
    return transformation_matrix


def param2theta(transformation: Tensor, H: int, W: int):
    param = torch.linalg.inv(transformation)
    theta = torch.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * H / W
    theta[0, 2] = param[0, 2] * 2 / W + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * W / H
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / H + theta[1, 0] + theta[1, 1] - 1
    return theta


def affine_transformation_matrix(dims: Tuple[int, int], angle: float = 0, scale: float = 1, shifts: Tuple[float, float] = (0, 0)) -> Tensor:
    H, W = dims
    invertable_transformation = transformation_matrix_2D((H/2, W/2), angle, scale, shifts)  #TODO: dudy changed from H//2,W//2
    transformation = param2theta(invertable_transformation, H, W)
    return transformation


def identity_transforms(N: int, angles: Tensor, scales: Tensor , shifts: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
    # returns identity angles, scale, shifts when they are not already defined
    if angles is None:
        angles = torch.zeros(N)
    if scales is None:
        scales = torch.ones(N)
    if shifts is None:
        shifts = (Tensor([0 for _ in range(N)]), Tensor([0 for _ in range(N)]))
    return angles, scales, shifts


def batch_affine_matrices(dims: Tuple[int, int], N: int = 1, angles: Tensor = None, scales: Tensor = None, shifts: Tuple[Tensor, Tensor] = None) -> Tensor:
    angles, scales, shifts = identity_transforms(N, angles, scales, shifts)
    affine_matrices = torch.zeros((N, 2, 3))
    for i in range(N):
        affine_matrices[i] = affine_transformation_matrix(dims, angles[i], scales[i], (shifts[0][i], shifts[1][i]))
    return affine_matrices





class MatrixOrigami:
    # expands matrix to 5 dimensions, then folds it back to original size at the end
    def __init__(self, matrix: Tensor):
        self.matrix = matrix
        self.original_dims = matrix.shape

    def expand_matrix(self, num_dims: int = 5) -> Tensor:
        # makes matrix requisite number of dimensions
        return fix_dimensionality(matrix=self.matrix, num_dims=num_dims)

    def expand_other_matrix(self, matrix: Tensor, num_dims = 5) -> Tensor:
        return fix_dimensionality(matrix, num_dims=num_dims)

    def squeeze_to_original_dims(self, matrix: Tensor) -> Tensor:
        # TODO Elisheva added this loop instead of squeeze()
        for dim in matrix.shape:
            if dim != 1:
                break
            matrix = matrix.squeeze(0)
        for i in range(len(self.original_dims) - len(matrix.shape)):
             matrix = matrix.unsqueeze(0)
        return matrix

# TODO Elisheva added this
class NoMemoryMatrixOrigami:
    # expands matrix to 5 dimensions, then folds it back to original size at the end
    def __init__(self, matrix: Tensor):
        self.original_dims = matrix.shape

    def expand_matrix(self, matrix: Tensor, num_dims: int = 5) -> Tensor:
        # makes matrix requisite number of dimensions
        return fix_dimensionality(matrix=matrix, num_dims=num_dims)

    def squeeze_to_original_dims(self, matrix: Tensor) -> Tensor:
        for dim in matrix.shape:
            if dim != 1:
                break
            matrix = matrix.squeeze(0)
        for i in range(len(self.original_dims) - len(matrix.shape)):
            matrix = matrix.unsqueeze(0)
        return matrix


def fix_dimensionality(matrix: Tensor, num_dims: int) -> Tensor:
    # makes matrix requisite number of dimensions
    current_dims = len(matrix.shape)
    if current_dims <= num_dims:
        for i in range(num_dims - len(matrix.shape)):
            matrix = matrix.unsqueeze(0)
        return matrix
    else:
        raise RuntimeError("Tried to expand a Tensor to a size smaller than itself")


def roll_n(matrix: Tensor, axis: int, n: int) -> Tensor:
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(matrix.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(matrix.dim()))
    front = matrix[f_idx]
    back = matrix[b_idx]
    return torch.cat([back, front], axis)


def faster_fft(matrix: Tensor, matrix_fft: Tensor, dim=[-2,-1]) -> Tensor:
    if matrix_fft is None:
        return torch.fft.fftn(matrix, dim=dim)
    else:
        return matrix_fft


def pad_matrix_to_size(matrix: Tensor, sizes: Tuple[int, int], resqueeze: bool = True) -> Tensor:
    # resqueeze is whether to bring back to less than 5 dims
    # adds padding to create Tensor of size sizes
    raise_if_not(len(sizes) == 2, message="Sizes must be of size H,W")
    B, T, C, old_H, old_W = matrix.shape
    H, W = sizes
    h_diff = H - old_H
    w_diff = W - old_W
    padded_matrix = torch.zeros((B,T,C,H,W))
    dimension_memory = MatrixOrigami(padded_matrix)
    expanded_padded_matrix = dimension_memory.expand_matrix()
    expanded_padded_matrix[:,:,:,h_diff//2:-h_diff//2, w_diff//2:-w_diff//2] = matrix
    if resqueeze:
        return dimension_memory.squeeze_to_original_dims(expanded_padded_matrix)
    else:
        return expanded_padded_matrix


def torch_int_types() -> List[torch.dtype]:
    return [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]


def true_dimensionality(matrix: Tensor) -> int:
    first_dim = len(matrix.shape)
    for num_dim, dim in enumerate(matrix.shape):
        if dim != 1:
            first_dim = num_dim
            break
    return len(matrix.shape)-first_dim


def dimension_N(matrix: Tensor, dimension: int) -> int:
    # returns size of dimension -dimension, or 1 if it doesn't exist
    if len(matrix.shape) >= dimension:
        return matrix.shape[-dimension]
    else:
        return 1


def extend_tensor_length_N(vector: Tensor, N: int) -> Tensor:
    """
    :param vector: 1D vector of size 1, or size N
    :param N: length the vector is, or will be extended to
    :return: vector of size N containing the element/s of input vector
    """
    if len(vector) == N:
        return vector
    else:
        return torch.ones(N) * vector


def compare_unequal_dimensionality_tensors(greater_dim_vector: Tensor, lesser_dim_vector: Tensor) -> bool:
    # returns if the tensors are the same size, consdiering dimensions of size 1 to be irrelevant
    # can also be used to compare equal sized vectors
    for i in range(-1, -len(greater_dim_vector.shape)-1, -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                return False
        except IndexError:
            if greater_dim_vector.shape[i] != 1:
                return False
    return True


# TODO Elisheva added
def compare_unequal_outer_dimensions(greater_dim_vector: Tensor, lesser_dim_vector: Tensor) -> bool:
    # A general test for cases where given tensors of different B, T dimensions
    # return true if lesser_dim_vector is a legitimate reference matrix for greater_dim_vector
    # Phase 1: CHW dimensions. must both exist and equal
    for i in range(-1, max(-len(greater_dim_vector.shape) - 1, -4), -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                return False
        except IndexError:
            return False
    # Phase 2: BT dimensions. False if lesser_dim_vector is not "private case" of greater_dim_vector.
    for i in range(-4, -len(greater_dim_vector.shape) - 1, -1):
        try:
            if greater_dim_vector.shape[i] != lesser_dim_vector.shape[i]:
                if lesser_dim_vector.shape[i] > 1:
                    return False
        except IndexError:
            return True
    return True


def equal_size_tensors(a: Tensor, b: Tensor) -> bool:
    # returns if tensors are of equal size
    if true_dimensionality(a) != true_dimensionality(b):
        return False
    elif len(a.shape) > len(b.shape):
        return compare_unequal_dimensionality_tensors(a, b)
    else: # need to know which tensor is greater dimensionality before comparing unequal dimensions
        return compare_unequal_dimensionality_tensors(b, a)


def extend_vector_length_n(vector: Tensor, N: int) -> Tensor:
    vector_length = len(vector)
    if vector_length == 1:
        return torch.ones(N) * vector
    elif int(N) == vector_length:
        return vector
    elif int(N) % vector_length == 0:
        return vector.repeat(int(N) // vector_length)
    else:
        raise RuntimeError("Cannot extend Tensor not factor length of N")


def non_interpolated_center_crop(matrix: Tensor, target_H: int, target_W: int) -> Tensor:
    B,T,C,H,W = matrix.shape
    excess_rows = H - target_H
    excess_columns = W - target_W
    start_y = int(excess_rows/2)
    start_x = int(excess_columns/2)
    stop_y = start_y+target_H
    stop_x = start_x+target_W
    return matrix[:, :, :, start_y:stop_y, start_x:stop_x]


def center_logical_mask(matrix: Tensor, zero_edges: int) -> Tensor:
    if len(matrix.shape) == 2:
        valid_mask = torch.zeros_like(matrix)
        H, W = matrix.shape
        valid_mask[zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    elif len(matrix.shape) == 3:
        valid_mask = torch.zeros_like(matrix)
        C, H, W = matrix.shape
        valid_mask[:, zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    elif len(matrix.shape) == 4:
        valid_mask = torch.zeros_like(matrix)
        T, C, H, W = matrix.shape
        valid_mask[:, :, zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    elif len(matrix.shape) == 5:
        valid_mask = torch.zeros_like(matrix)
        B, T, C, H, W = matrix.shape
        valid_mask[:, :, :, zero_edges:H - zero_edges, zero_edges:W - zero_edges] = 1
    else:
        raise RuntimeError("Matrix is not 2-5D")
    return valid_mask


# TODO Elisheva changed to allow parallelism and also named with torch for now
def unravel_index(index: Union[int, Tensor, np.array], dimensions: Union[List[int], Tuple[int]]) -> Tuple:
    # turns raw index into dimensioned index
    dimensioned_index = []
    dimensions = list(dimensions)
    dimensions.append(1)  # for determining last dimension
    for i in range(len(dimensions)-1):
        remaining_space = reduce(lambda x, y: x*y, dimensions[i+1:])
        dimensioned_index.append(index//remaining_space)
        index -= dimensioned_index[-1]*remaining_space
    return tuple(dimensioned_index)


# TODO Elisheva changed to allow parallelism and also named with torch for now
def unravel_index_torch(index: Union[int, Tensor, np.array], dimensions: Union[List[int], Tuple[int]]) -> Tuple:
    # turns raw index into dimensioned index
    remaining_index = index.clone()
    dimensioned_index = []
    dimensions = list(dimensions)
    dimensions.append(1)  # for determining last dimension
    for i in range(len(dimensions)-1):
        remaining_space = reduce(lambda x, y: x*y, dimensions[i+1:])
        dimensioned_index.append(remaining_index//remaining_space)
        remaining_index -= dimensioned_index[-1]*remaining_space
    return tuple(dimensioned_index)


# TODO Elisheva added
def ravel_multi_index(multi_index: tuple,
                      dimensions: Union[List[int], Tuple[int]]) -> Union[Tensor, np.array, int]:
    # turns dimensioned index into raw index
    flattened_index = multi_index[0] - multi_index[0]  # to keep the element type yet get zero
    for i in range(len(dimensions) - 1):
        dim_block_size = reduce(lambda x, y: x*y, dimensions[i+1:])
        flattened_index = flattened_index + multi_index[i] * dim_block_size
    return flattened_index


def convert_iterable_tensor(elements: Union[list, tuple]) -> Tensor:
    # recursively converts a list of any form into Tensor.
    if type(elements[0]) == Tensor:
        return torch.stack(elements)
    elif type(elements[0]) in [float, int, np.float, np.float64, np.float32, np.int, np.uint8, np.int16]:
        return Tensor(elements)
    else:
        return torch.stack([convert_iterable_tensor(iterable) for iterable in elements])


# def construct_tensor(elements: Union[Tensor, list, tuple, float, int, np.array], device: Union[str, torch.device] = "cpu"):
#     if type(elements) in [float, int]:
#         return Tensor([elements], device=device)
#     elif type(elements) == Tensor and str(elements.device) == str(device):
#         if len(elements.shape) == 0:  # is tensor of form tensor(float), as opposed to tensor([float]). Problem since former type is not iterable
#             return Tensor([float(elements)])
#         return elements
#     elif type(elements) in [list, tuple]:
#         return convert_iterable_tensor(elements).to(device)
#     else:
#         return Tensor(elements, device=device)

# TODO Elisheva changed to new version for construct tensor, notice and check carefully
def construct_tensor(elements: Union[Tensor, list, tuple, float, int, np.array], device: Union[str, torch.device] = "cpu"):
    if type(elements) in [float, int]:
        return torch.tensor([elements], device=device)
    # we have a device issue here - if tensor, leave the original device or change?
    elif type(elements) == Tensor:
        if len(elements.shape) == 0:  # is tensor of form torch.tensor(float), as opposed to torch.tensor([float]). Problem since former type is not iterable
            return torch.tensor([float(elements)]).to(elements.device)
        return elements.to(elements.device)
    elif type(elements) in [list, tuple]:
        return convert_iterable_tensor(elements).to(device)
    else:
        return torch.tensor(elements, device=device)


def stepped_arange(start: Union[int, float], stop: Union[int, float], steps: int) -> Tensor:
    """
    :param start: lower bound on arange (inclusive)
    :param stop: upper bound on arange (non-inclusive)
    :param steps: number of elements to be in the arange
    :return: arange from start to stopp in step steps
    """
    if stop-start%steps == 0:
        return torch.arange(start, stop, (stop-start)//steps)
    else:
        return torch.arange(start, stop, (stop-start)/steps)



def RGB2BW(input_image):
    if len(input_image.shape) == 2:
        return input_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 3:
            grayscale_image = 0.299 * input_image[0:1, :, :] + 0.587 * input_image[1:2, :, :] + 0.114 * input_image[2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1] + 0.587 * input_image[:, :, 1:2] + 0.114 * input_image[:, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, 0:1] + 0.587 * input_image[:, :, :, 1:2] + 0.114 * input_image[:, :, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, :, 0:1] + 0.587 * input_image[:, :, :, :, 1:2] + 0.114 * input_image[:, :, :, :, 2:3]
        else:
            grayscale_image = input_image

    return grayscale_image

def RGB2BW_interleave_C(input_image):  #Only for tensors for now, and if you know(!) they're RGB
    #(*). Unless given another input of C or T, this function cannot know whether it was really given a RGB image, SO IT ASSUMES IT!!! YOU NEED TO MAKE SURE!!!!!
    if len(input_image.shape)==3:
        TxC,H,W = input_image.shape
        new_C = TxC//3
        input_image = input_image.reshape(3, new_C, H, W)
        grayscale_image = 0.299 * input_image[0:1, :, :, :] + 0.587 * input_image[1:2, :, :, :] + 0.114 * input_image[2:3, :, :, :]

    elif len(input_image.shape)==4:
        B, TxC, H, W = input_image.shape
        new_C = TxC // 3
        input_image = input_image.reshape(B, 3, new_C, H, W)
        grayscale_image = 0.299 * input_image[:, 0:1, :, :, :] + 0.587 * input_image[:, 1:2, :, :, :] + 0.114 * input_image[:, 2:3, :, :, :]

    return grayscale_image

def RGB2BW_interleave_T(input_image):  #Only for tensors for now, and if you know(!) they're RGB
    if len(input_image.shape)==4:    #[T,C,H,W]
        T,C,H,W = input_image.shape
        if C == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        else:
            grayscale_image = input_image

    elif len(input_image.shape)==5:  #[B,T,C,H,W]
        B, T, C, H, W = input_image.shape
        if C == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        else:
            grayscale_image = input_image

    return grayscale_image


def BW2RGB_interleave_C(input_image):
    ### For Torch Tensors!: ###
    if len(input_image.shape) == 3:
        RGB_image = torch.repeat_interleave(input_image, 3, 0)

    elif len(input_image.shape) == 4:
        RGB_image = torch.repeat_interleave(input_image, 3, 1)

    elif len(input_image.shape) == 5:
        RGB_image = torch.cat([input_image]*3, 2)

    return RGB_image

def BW2RGB_interleave_T(input_image):
    ### For Torch Tensors!: ###
    if len(input_image.shape) == 4:  #[T,C,H,W]
        RGB_image = torch.repeat_interleave(input_image, 3, 1)

    elif len(input_image.shape) == 5:  #[B,T,C,H,W]
        RGB_image = torch.repeat_interleave(input_image, 3, 2)

    return RGB_image


def BW2RGB_MultipleFrames(input_image, flag_how_to_concat='C'):
    if flag_how_to_concat == 'C':
        RGB_image = BW2RGB_interleave_C(input_image)
    else:
        RGB_image = BW2RGB_interleave_T(input_image)
    return RGB_image

def RGB2BW_MultipleFrames(input_image, flag_how_to_concat='C'):
    if flag_how_to_concat == 'C':
        grayscale_image = RGB2BW_interleave_C(input_image)
    else:
        grayscale_image = RGB2BW_interleave_T(input_image)
    return grayscale_image

def BW2RGB(input_image):
    ### For Both Torch Tensors and Numpy Arrays!: ###
    # Actually... we're not restricted to RGB....
    if len(input_image.shape) == 2:
        if type(input_image) == torch.Tensor:
            RGB_image = input_image.unsqueeze(0)
            RGB_image = torch.cat([RGB_image,RGB_image,RGB_image], 0)
        elif type(input_image) == np.ndarray:
            RGB_image = np.atleast_3d(input_image)
            RGB_image = np.concatenate([RGB_image, RGB_image, RGB_image], -1)
        return RGB_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 0)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 1)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 2)
        else:
            RGB_image = input_image

    return RGB_image




def multiplicable(a: Tensor, b: Tensor) -> bool:
    for i in range(min(len(a.shape), len(b.shape))+1):
        if a.shape[-i] != b.shape[-i] and not (a.shape[-i] == 1 or b.shape[-i]==1):
            return False
    return True


def multiplicable_tensors(tensors: List[Tensor]) -> bool:
    # returns whether all tensors in list can multiply each other
    for i in range(len(tensors)):
        for j in range(len(tensors)):
            if i == j:
                continue  # same tensor
            else:
                if not multiplicable(tensors[i], tensors[j]):
                    return False
    return True

import itertools
from functools import partial

def get_max_sub_pixel_over_t(matrix: Tensor, flag_get_subpixel_max_location_using_parabola_fit=True):
    # expected THW tensor
    # Phase I: find discrete maximum
    T, H, W = matrix.shape
    flat_matrix = torch.flatten(matrix, start_dim=1)
    flat_max_indices = torch.argmax(flat_matrix, dim=1)
    max_indices_tuple = unravel_index_torch(flat_max_indices, (H, W))
    max_indices_tuple = [l.tolist() for l in max_indices_tuple]
    max_indices_list = list(zip(*max_indices_tuple))
    timed_max_indices = list(enumerate(max_indices_list))
    timed_max_indices = [(t, h, w) for (t, (h, w)) in timed_max_indices]

    # Phase II: do parabola fit
    def index_to_parabola_fit_vals(t, h, w, values_tensor):
        # Assumed shapes: values tensor of shape HxW
        # indices tensor of shape 2, 3
        H, W = values_tensor.shape[1:]
        if h == 0 or h == H - 1:
            H_vals = [np.nan, values_tensor[t, h, w].item(), np.nan]
        else:
            H_vals = [values_tensor[t, h - 1, w].item(), values_tensor[t, h, w].item(), values_tensor[t, h + 1, w].item()]
        if w == 0 or w == W - 1:
            W_vals = [np.nan, values_tensor[t, h, w].item(), np.nan]
        else:
            W_vals = [values_tensor[t, h, w - 1].item(), values_tensor[t, h, w].item(), values_tensor[t, h, w + 1].item()]

        return [H_vals, W_vals]

    def fit_polynomial(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[2] + y[0] - 2 * y[1]) / 2
        b = -(y[0] + 2 * a * x[1] - y[1] - a)
        c = y[1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    def max_delta_over_parabola_fit(y1, y2, y3):
        if y1 is np.nan:
            delta_shift_x = 0
        else:
            x_vec = [-1, 0, 1]
            c_x, b_x, a_x = fit_polynomial(x_vec, [y1, y2, y3])
            delta_shift_x = -b_x / (2 * a_x)

        return delta_shift_x

    ### Get Discrete Max Location Values: ###
    discrete_max_location_over_t_tensor = torch.tensor(max_indices_list).float().to(matrix.device)
    discrete_max_location_over_t_tensor[:, 0] -= H / 2
    discrete_max_location_over_t_tensor[:, 1] -= W / 2

    ### Get Discrete Max Values Themselves: ###
    map_values_for_parabola_fit = partial(index_to_parabola_fit_vals, values_tensor=matrix)
    values_for_parabola_fit = list(itertools.starmap(map_values_for_parabola_fit, timed_max_indices))
    discrete_max_values = [v[0][1] for v in values_for_parabola_fit]

    ### Perform parabola fit over values: ###
    if flag_get_subpixel_max_location_using_parabola_fit:
        subpixel_deltas_over_t = [[max_delta_over_parabola_fit(*vals_h), max_delta_over_parabola_fit(*vals_w)] for vals_h, vals_w in values_for_parabola_fit]
        subpixel_deltas_over_t_tensor = torch.tensor(subpixel_deltas_over_t)

        ### Combine discrete max location with subpixel max correction: ###
        total_max_location_over_t_tensor = discrete_max_location_over_t_tensor + subpixel_deltas_over_t_tensor
        subpixel_max_over_t = Tensor([d_max + shift_h + shift_w for d_max, (shift_h, shift_w) in
                                     zip(discrete_max_values, subpixel_deltas_over_t)])
    else:
        subpixel_max_over_t = discrete_max_values
        total_max_location_over_t_tensor = discrete_max_location_over_t_tensor


    return subpixel_max_over_t, total_max_location_over_t_tensor


def unravel_index_torch(index: Union[int, Tensor, np.array], dimensions: Union[List[int], Tuple[int]]) -> Tuple:
    # turns raw index into dimensioned index
    remaining_index = index.clone()
    dimensioned_index = []
    dimensions = list(dimensions)
    dimensions.append(1)  # for determining last dimension
    for i in range(len(dimensions)-1):
        remaining_space = reduce(lambda x, y: x*y, dimensions[i+1:])
        dimensioned_index.append(remaining_index//remaining_space)
        remaining_index -= dimensioned_index[-1]*remaining_space
    return tuple(dimensioned_index)


def normalize_array(input_tensor, min_max_values=(0,1)):
    input_tensor_normalized = (input_tensor - input_tensor.min()) / (input_tensor.max()-input_tensor.min()) * (min_max_values[1]-min_max_values[0]) + min_max_values[0]
    return input_tensor_normalized

def scale_array_to_range(input_tensor, min_max_values_to_scale_to=(0,1)):
    input_tensor_normalized = (input_tensor - input_tensor.min()) / (input_tensor.max()-input_tensor.min() + 1e-16) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    return input_tensor_normalized

def scale_array_each_image_to_range(input_tensor, min_max_values_to_scale_to=(0,1)):
    input_tensor_normalized = (input_tensor - input_tensor.min(-1, keepdim=True)[0]) /\
                              (input_tensor.max(-1, keepdim=True)[0]-input_tensor.min(-1, keepdim=True)[0] + 1e-16) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    return input_tensor_normalized

def scale_array_from_range(input_tensor, min_max_values_to_clip=(0,1), min_max_values_to_scale_to=(0,1)):
    if type(input_tensor) == torch.Tensor:
        input_tensor_normalized = (input_tensor.clamp(min_max_values_to_clip[0],min_max_values_to_clip[1]) - min_max_values_to_clip[0]) / (min_max_values_to_clip[1]-min_max_values_to_clip[0]) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    else:
        input_tensor_normalized = (input_tensor.clip(min_max_values_to_clip[0],min_max_values_to_clip[1]) - min_max_values_to_clip[0]) / (min_max_values_to_clip[1]-min_max_values_to_clip[0]) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    # output_tensor = (input_tensor - q1) / (q2-q1) * (1-0) + 0 -> input_tensor = output_tensor*(q2-q1) + q1
    return input_tensor_normalized


# output_tensor = kornia.enhance.sharpness(input_tensor, factor=0.5)
# output_tensor = kornia.enhance.equalize(input_tensor)
# output_tensor = kornia.enhance.equalize_clahe(input_tensor, clip_limit=40.0, grid_size=(8, 8), slow_and_differentiable=False)
# output_tensor = kornia.enhance.equalize3d(input_tensor) #Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format

def scale_array_clahe(input_tensor, flag_stretch_histogram_first=True, quantiles=(0.01, 0.99), flag_return_quantile=False):

    #TODO: make this accept pytorch or numpy
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if type(input_tensor) == torch.Tensor:
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = input_tensor.quantile(quantiles[0])
            q2 = input_tensor.quantile(quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = input_tensor[0,0].quantile(quantiles[0])
            q2 = input_tensor[0,0].quantile(quantiles[1])

        ### Pre-stretch/Pre-scale before CLAHE: ###
        if flag_stretch_histogram_first:
            scaled_array = scale_array_from_range(input_tensor.clip(q1, q2),
                                                  min_max_values_to_clip=(q1, q2),
                                                  min_max_values_to_scale_to=(0, 1))
        else:
            scaled_array = scale_array_to_range(input_tensor)

        ### Perform CLAHE: ###
        scaled_array = kornia.enhance.equalize_clahe(scaled_array, clip_limit=40.0, grid_size=(8, 8), slow_and_differentiable=False)

        if flag_return_quantile:
            return scaled_array, (q1.cpu().item(), q2.cpu().item())
        else:
            return scaled_array

    else:
        #Numpy array
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = np.quantile(input_tensor, quantiles[0])
            q2 = np.quantile(input_tensor, quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = np.quantile(input_tensor[0,0], quantiles[0])
            q2 = np.quantile(input_tensor[0,0], quantiles[1])

        ### Pre-stretch/Pre-scale before CLAHE: ###
        if flag_stretch_histogram_first:
            scaled_array = scale_array_from_range(input_tensor.clip(q1, q2),
                                                  min_max_values_to_clip=(q1, q2),
                                                  min_max_values_to_scale_to=(0, 1))
        else:
            scaled_array = scale_array_to_range(input_tensor)


        ### Perform CLAHE by switching to pytorch and back: ###
        scaled_array = torch.tensor(scaled_array)
        scaled_array = torch_get_4D(scaled_array)
        scaled_array = kornia.enhance.equalize_clahe(scaled_array, clip_limit=40.0, grid_size=(8, 8), slow_and_differentiable=False)
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            scaled_array = scaled_array[0,0].cpu().numpy()
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            scaled_array = scaled_array[0].cpu().numpy()
        elif shape_len == 4:
            #[T,C,H,W]
            scaled_array = scaled_array.cpu().numpy()

        if flag_return_quantile:
            return scaled_array, (q1,q2)
        else:
            return scaled_array


def scale_array_stretch_hist(input_tensor, quantiles=(0.01,0.99), min_max_values_to_scale_to=(0,1), flag_return_quantile=False):
    #TODO: make this accept pytorch or numpy
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if type(input_tensor) == torch.Tensor:
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = input_tensor.quantile(quantiles[0])
            q2 = input_tensor.quantile(quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = input_tensor[0,0].quantile(quantiles[0])
            q2 = input_tensor[0,0].quantile(quantiles[1])

        scaled_array = scale_array_from_range(input_tensor.clip(q1, q2),
                                              min_max_values_to_clip=(q1, q2),
                                              min_max_values_to_scale_to=min_max_values_to_scale_to)

        if flag_return_quantile:
            return scaled_array, (q1.cpu().item(), q2.cpu().item())
        else:
            return scaled_array

    else:
        #Numpy array
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = np.quantile(input_tensor, quantiles[0])
            q2 = np.quantile(input_tensor, quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = np.quantile(input_tensor[0,0], quantiles[0])
            q2 = np.quantile(input_tensor[0,0], quantiles[1])

        scaled_array = scale_array_from_range(input_tensor.clip(q1,q2),
                                              min_max_values_to_clip=(q1,q2),
                                              min_max_values_to_scale_to=min_max_values_to_scale_to)
        if flag_return_quantile:
            return scaled_array, (q1,q2)
        else:
            return scaled_array



def path_get_all_filenames_from_folder(path, flag_recursive=False, flag_full_filename=True):
    filenames_list = []
    # Look recursively at all current and children path's in given path and get all binary files
    for dirpath, _, fnames in sorted(os.walk(path)):  # os.walk!!!!
        for fname in sorted(fnames):
            if flag_full_filename:
                file_full_filename = os.path.join(dirpath, fname)
            else:
                file_full_filename = fname
            filenames_list.append(file_full_filename)
        if flag_recursive == False:
            break;
    return filenames_list;






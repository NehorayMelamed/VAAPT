import math

import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple

from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not, raise_if
from RapidBase.Anvil._transforms.matrix_transformations import batch_affine_matrices
from RapidBase.Anvil._internal_utils.torch_utils import faster_fft, true_dimensionality, \
    dimension_N, extend_tensor_length_N, equal_size_tensors, extend_vector_length_n, construct_tensor
from RapidBase.Anvil._internal_utils.parameter_checking import type_check_parameters, validate_warp_method


from RapidBase.Utils.IO.tic_toc import *
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


def format_blur_shift_params(matrix: torch.Tensor, shift_H: int, shift_W: int, N: int, matrix_FFT=None,
                           warp_method='bilinear'):
    matrix, shift_H, shift_W, matrix_FFT, warp_method = format_subpixel_shift_params(matrix, shift_H, shift_W, matrix_FFT, warp_method)
    type_check_parameters([(N, int)])
    raise_if_not(len(shift_H) == 1, message="Shift H must be singleton for blurring")
    raise_if_not(len(shift_W) == 1, message="Shift W must be singleton for blurring")
    raise_if(N <= 0, message="N must be greater than 0")
    return matrix, Tensor(shift_H), Tensor(shift_W), N, matrix_FFT, warp_method

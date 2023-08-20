from typing import Tuple, Union

import numpy as np
import torch, math
from torch import Tensor

from RapidBase.Anvil._internal_utils.parameter_checking import type_check_parameters
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not, raise_if
from RapidBase.Anvil._internal_utils.torch_utils import true_dimensionality, extend_tensor_length_N, construct_tensor
from RapidBase.Anvil._transforms.matrix_transformations import batch_affine_matrices


def size_of_rotated_matrix(matrix: torch.Tensor) -> Tuple[int, int, int, int, int]:
    # returns what the size of the matrix, if rotated would be
    B, T, C, H, W = matrix.shape
    return B, T, C, 2 * H, 2 * W


def encompassing_frame(matrix: Tensor) -> Tensor:
    # frame large enough to hold up to 45 degree rotation
    # input tensor is 5 dimensional tensor
    B,T,C,H,W = matrix.shape
    expanded_frame = torch.zeros(size_of_rotated_matrix(matrix), device=matrix.device)
    expanded_frame[:, :, :, H // 2:H // 2 + H, W // 2:W // 2 + W] = matrix
    return expanded_frame


def structure_thetas(matrix: Tensor, B: int, thetas):
    upscaled_thetas = torch.zeros(B, len(thetas), 1, 1, 1, device=matrix.device)
    upscaled_thetas[:, :, 0, 0, 0] = torch.Tensor(thetas)
    return upscaled_thetas


def fftshifted_formatted_vectors(matrix: Tensor, H: int, W: int) -> Tuple[Tensor, Tensor]:
    # returns FFT-shifted ranges from -H/2->H/2, -W/2->W/2
    Nx_vec = torch.arange(-math.floor(W / 2), math.ceil(W / 2), device=matrix.device)
    Ny_vec = torch.arange(-math.floor(H / 2), math.ceil(H / 2), device=matrix.device)
    Nx_vec = torch.fft.fftshift(Nx_vec, 0)
    Ny_vec = torch.fft.fftshift(Ny_vec, 0)
    Nx_vec = Nx_vec.unsqueeze(0)
    Ny_vec = Ny_vec.unsqueeze(1)
    Nx_vec = Nx_vec.unsqueeze(0).unsqueeze(0)
    Ny_vec = Ny_vec.unsqueeze(0).unsqueeze(0)
    return Nx_vec, Ny_vec
    
    
def rotate_matrix_fft(matrix: Tensor, thetas: Tensor):
    # len(thetas) = T
    B, T, C, H, W = matrix.shape
    thetas = structure_thetas(matrix, B, thetas)
    expanded_frames = encompassing_frame(matrix)

    ### Choose Rotation Angle and assign variables needed for later: % % %
    B, T, C, H, W = expanded_frames.shape
    a = torch.tan(thetas / 2)
    b = -torch.sin(thetas)
    Nx_vec, Ny_vec = fftshifted_formatted_vectors(matrix, H, W)


    ### Prepare Matrices to avoid looping: ###
    Nx_vec_mat = torch.repeat_interleave(Nx_vec, H, -2)
    Ny_vec_mat = torch.repeat_interleave(Ny_vec, W, -1)
    #ISSUE: BTCHW
    # max shape means select single dimension that isn't singleton
    column_mat = torch.zeros((B, T, C, max(Ny_vec.shape), max(Nx_vec.shape)), device=matrix.device)
    row_mat = torch.zeros((B, T, C, max(Ny_vec.shape), max(Nx_vec.shape)), device=matrix.device)
    for k in range(H):
        column_mat[:, :, :, k, :] = k - math.floor(H / 2)
    for k in range(W):
        row_mat[:, :, :, :, k] = k - math.floor(W / 2)
    Ny_vec_mat_final = Ny_vec_mat * row_mat * (-2 * 1j * math.pi) / H
    Nx_vec_mat_final = Nx_vec_mat * column_mat * (-2 * 1j * math.pi) / W

    ### Use Parallel Computing instead of looping: ###
    Ix_parallel = (
        torch.fft.ifftn(torch.fft.fftn(expanded_frames, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    Iy_parallel = (
        torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(Ny_vec_mat_final * b * -1).conj(),
                        dim=[-2])).real
    input_mat_rotated = (
        torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    return input_mat_rotated


def rotate_matrix_interpolated(matrix: Tensor, thetas: Tensor, warp_method='bilinear') -> Tensor:
    matrix = encompassing_frame(matrix)
    B, T, C, H, W = matrix.shape
    N = B*T
    affine_matrices = batch_affine_matrices((H,W), N, angles=extend_tensor_length_N(thetas, N))
    output_grid = torch.nn.functional.affine_grid(affine_matrices,
                                                  torch.Size((N, C, H, W)))
    matrix = matrix.reshape((B * T, C, H, W))  # grid sample can only handle 4D
    output_tensor = torch.nn.functional.grid_sample(matrix, output_grid, mode=warp_method)
    return output_tensor.reshape((B, T, C, H, W))


def check_thetas(matrix: Tensor, thetas: torch.Tensor) -> Tensor:
    """
    :param matrix: matrix shift will be applied to
    :param thetas: either singleton or length N 1D vector. Exception will be raised if these do not hold
    :return: singleton vector or 1D vector of length time domain of the input matrix
    """
    if true_dimensionality(thetas) > 0: # has more than one element. Must be same length as time dimension
        raise_if_not(true_dimensionality(thetas) == 1, message="Thetas must be 1D vector or int") # shouldn't be 2D
        raise_if_not(true_dimensionality(matrix) >= 4, message="Thetas has larger dimensionality than input matrix") # matrix has time dimension
        raise_if_not(matrix.shape[-4] == thetas.shape[0], message="Thetas not same length as time dimension") # has to have same length as time
        return thetas
    else:
        raise_if(len(thetas) == 0)
        return thetas


def format_rotation_parameters(matrix: Union[Tensor, np.array, tuple, list],
                               thetas: Union[torch.Tensor, list, tuple, float, int],
                               warp_method='bilinear') -> Tuple[Tensor, Tensor, str]:
    type_check_parameters([(matrix, (Tensor, np.array, tuple, list)), (thetas, (Tensor, list, tuple, int, float)), (warp_method, str)])
    thetas = construct_tensor(thetas)
    thetas = check_thetas(matrix, thetas)
    return matrix, thetas, warp_method


def format_blur_rotation_parameters(matrix: Union[torch.Tensor, np.array, tuple, list], thetas: torch.Tensor, N: int, warp_method='bilinear') -> Tuple[Tensor, Tensor, int, str]:
    matrix, thetas, warp_method = format_rotation_parameters(matrix, thetas, warp_method)
    type_check_parameters([(N, int)])
    raise_if_not(N > 0, message="N must be greater than 0")
    return matrix, thetas, N, warp_method

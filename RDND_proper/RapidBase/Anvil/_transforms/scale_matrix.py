from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from RapidBase.Anvil._internal_utils.parameter_checking import type_check_parameters, validate_warp_method
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not, raise_if
from RapidBase.Anvil._internal_utils.torch_utils import true_dimensionality, extend_vector_length_n, construct_tensor, \
    pad_matrix_to_size
from RapidBase.Anvil._transforms.matrix_transformations import batch_affine_matrices


def expand_matrix_for_scaling(matrix: Tensor, max_scaling: float) -> Tensor:
    B, T, C, H, W = matrix.shape
    if max_scaling < 1:
        max_scaling = 1
    matrix = pad_matrix_to_size(matrix, (int(H*max_scaling)+10, int(W*max_scaling)+10))
    return matrix


def _scale_matrix(matrix: torch.Tensor, scales: Tensor, warp_method='bilinear') -> Tensor:
    matrix = expand_matrix_for_scaling(matrix, float(max(scales)))
    B, T, C, H, W = matrix.shape
    ##Question: rounded to int??
    N = B * T
    scales = extend_vector_length_n(scales, N)
    affine_matrices = batch_affine_matrices((H, W), N, scales=scales)
    output_grid = torch.nn.functional.affine_grid(affine_matrices,
                                                  torch.Size((N, C, H, W)))
    matrix = matrix.reshape((B * T, C, H, W))  # grid sample can only handle 4D
    output_tensor = torch.nn.functional.grid_sample(matrix, output_grid, mode=warp_method)
    return output_tensor.reshape((B, T, C, H, W))


def check_scales(matrix: Tensor, scales: torch.Tensor):
    """
    :param matrix: matrix shift will be applied to
    :param scales: either singleton or length N 1D vector. Exception will be raised if these do not hold
    :return: singleton vector or 1D vector of length time domain of the input matrix
    """
    if true_dimensionality(scales) > 0: # has more than one element. Must be same length as time dimension
        raise_if_not(true_dimensionality(scales) == 1, message="Thetas must be 1D vector or int") # shouldn't be 2D
        raise_if_not(true_dimensionality(matrix) >= 4, message="Thetas has larger dimensionality than input matrix") # matrix has time dimension
        raise_if_not(matrix.shape[-4] == scales.shape[0], message="Thetas not same length as time dimension") # has to have same length as time
        raise_if_not((scales > torch.zeros_like(scales)).all(), message="Scaling must be positive")
    else:
        raise_if(len(scales) == 0, message="No scales provided")
        raise_if(float(scales[0]) <= 0, message="Scaling must be positive")


##### type int for scales
def format_scale_parameters(matrix: Union[torch.Tensor, np.array, tuple, list],
                            scales: Union[torch.Tensor, list, tuple, float, int],
                            warp_method='bilinear') -> Tuple[Tensor, Tensor, str]:
    type_check_parameters([(matrix, torch.Tensor, np.array, tuple, list), (scales, (torch.Tensor, list, tuple, float, int)), (warp_method, str)])
    validate_warp_method(warp_method, valid_methods=['bilinear', 'bicubic', 'nearest'])
    scales = construct_tensor(scales)
    check_scales(matrix, scales)
    return matrix, scales, warp_method

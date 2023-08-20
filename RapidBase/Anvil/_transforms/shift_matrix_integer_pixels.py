from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from RapidBase.Anvil._internal_utils.parameter_checking import type_check_parameters
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not, raise_if
from RapidBase.Anvil._internal_utils.torch_utils import true_dimensionality, dimension_N, \
    extend_tensor_length_N, construct_tensor


def format_shift(matrix: Tensor, shift: torch.Tensor) -> Tensor:
    """
    :param matrix: matrix shift will be applied to
    :param shift: either singleton or length N 1D vector. Exception will be raised if these do not hold
    :return: singleton vector or 1D vector of length time domain of the input matrix
    """
    if true_dimensionality(shift) > 0: # has more than one element. Shifts must be same length as time dimension, matrix must have 4+ dimensions
        raise_if_not(len(torch.nonzero(shift.to(torch.int64) - shift)) == 0, message="Shifts are not integer")
        raise_if_not(true_dimensionality(shift) == 1, message="Shifts must be 1D vector or int") # shouldn't be 2D
        raise_if_not(true_dimensionality(matrix) >= 4, message="Shifts has larger dimensionality than input matrix") # matrix has time dimension
        raise_if_not(matrix.shape[-4] == shift.shape[0], message="Shifts not same length as time dimension") # has to have same length as time
        return shift
    else:
        raise_if(len(shift) == 0)
        return shift


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



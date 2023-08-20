from typing import Union, Tuple

import torch
from torch import Tensor

from _transforms.matrix_transformations import transformation_matrix_2D
from _transforms.rotate_matrix import format_rotation_parameters, rotate_matrix_fft
from _transforms.shift_matrix_subpixel import format_subpixel_shift_params, _shift_matrix_subpixel_fft


def fft_rotate_shift(matrix: torch.Tensor, shifts_y: Tensor, shifts_x: Tensor, thetas: torch.Tensor,
                        matrix_fft: torch.Tensor=None):
    rotated_matrix = rotate_matrix_fft(matrix, thetas)
    shifted_matrix = _shift_matrix_subpixel_fft(rotated_matrix, shifts_y, shifts_x, matrix_fft)
    return shifted_matrix


def uniform_rotate_shift(matrix: Tensor, shifts_y: Tensor, shifts_x: Tensor, thetas: Tensor, warp_method: str) -> Tensor:
    H, W = matrix.shape[-2:]
    transformation = transformation_matrix_2D((H//2, W//2), shifts=(float(shifts_y), float(shifts_x)), angle=float(thetas))




def format_rotate_shift_params(matrix: torch.Tensor, shifts_y: Tensor, shifts_x: Tensor, thetas: torch.Tensor,
                        matrix_fft: torch.Tensor=None, warp_method='bilinear') -> Tensor:
    if warp_method == 'fft':
        return fft_rotate_shift(matrix, shifts_y, shifts_x, thetas, matrix_fft)
    else:
        if shifts_y.shape[-1] == 1 and shifts_x.shape[-1] == 1 and thetas.shape[-1] == 1:
            return uniform_rotate_shift(matrix, shifts_y, shifts_x, thetas, warp_method)


def format_rotate_shift_params(matrix: torch.Tensor, shifts_y: Union[torch.Tensor, list, Tuple, float],
                        shifts_x: Union[torch.Tensor, list, Tuple, float], thetas: torch.Tensor,
                        matrix_fft: torch.Tensor=None, warp_method='bilinear') -> Tuple[Tensor, Tensor, Tensor, Tensor, str, Union[Tensor, None]]:
    matrix, thetas, warp_method = format_rotation_parameters(matrix, thetas, warp_method)
    matrix, shifts_y, shifts_x, matrix_fft, warp_method = format_subpixel_shift_params(matrix, shifts_y, shifts_x, matrix_fft, warp_method)
    return matrix, shifts_y, shifts_x, thetas, matrix_fft, warp_method

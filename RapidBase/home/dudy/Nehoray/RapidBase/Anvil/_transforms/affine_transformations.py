from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor

from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not
from RapidBase.Anvil._internal_utils.torch_utils import extend_vector_length_n
from RapidBase.Anvil._transforms.rotate_matrix import format_rotation_parameters, encompassing_frame
from RapidBase.Anvil._transforms.shift_matrix_subpixel import format_subpixel_shift_params
from RapidBase.Anvil._transforms.scale_matrix import expand_matrix_for_scaling, format_scale_parameters
from RapidBase.Anvil._transforms.matrix_transformations import batch_affine_matrices


def affine_transform_interpolated(matrix: torch.Tensor, shifts_y: Tensor, shifts_x: Tensor, thetas: torch.Tensor, scales: Tensor,
                        warp_method='bilinear', expand=True) -> Tensor:
    # expand should only be used internally for now, haven't tested it enough yet
    if expand:
        matrix = encompassing_frame(matrix)  # rotation encompassing frame
        max_scale = float(max(scales))
        if max_scale > 1:
            matrix = expand_matrix_for_scaling(matrix, max_scale)
    B, T, C, H, W = matrix.shape
    N = B*T
    thetas = extend_vector_length_n(thetas, N)
    shifts_y = extend_vector_length_n(shifts_y, N)
    shifts_x = extend_vector_length_n(shifts_x, N)
    scales = extend_vector_length_n(scales, N)
    affine_matrices = batch_affine_matrices((matrix.shape[-2], matrix.shape[-1]), N, scales=scales, shifts=(shifts_y, shifts_x), angles=thetas)
    # output_grid = torch.nn.functional.affine_grid(affine_matrices,
    #                                               torch.Size((N, C, H, W))).to(matrix.device)
    matrix = matrix.view((B * T, C, H, W))  # grid sample can only handle 4D
    output_tensor = torch.nn.functional.grid_sample(matrix,
                                                    torch.nn.functional.affine_grid(affine_matrices, torch.Size((N, C, H, W))).to(matrix.device),
                                                    mode=warp_method)
    return output_tensor.view((B, T, C, H, W))


def default_affine_parameters(shifts_y: Union[torch.Tensor, list, Tuple, float] = None,
                        shifts_x: Union[torch.Tensor, list, Tuple, float] = None, thetas: torch.Tensor = None,
                              scales: Union[Tensor, list, Tuple, float] = None):
    # returns default parameters if shifts are undefined
    if shifts_y is None:
        shifts_y = 0
    if shifts_x is None:
        shifts_x = 0
    if thetas is None:
        thetas = 0
    if scales is None:
        scales = 1
    return shifts_y, shifts_x, thetas, scales


def format_affine_parameters(matrix: Union[torch.Tensor, np.array, tuple, list],
                        shifts_y: Union[Tensor, list, tuple, float, int] = None,
                        shifts_x: Union[Tensor, list, tuple, float, int] = None,
                        thetas: Union[Tensor, list, tuple, float, int] = None,
                        scales: Union[Tensor, list, tuple, float, int] = None,
                        warp_method='bilinear') -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str]:
    raise_if_not(warp_method in ['bilinear', 'bicubic', 'nearest'], message=f"{warp_method} is invalid. Must be bilinear, bicubic, or nearest")
    shifts_y, shifts_x, thetas, scales = default_affine_parameters(shifts_y, shifts_x, thetas, scales)
    matrix, shifts_y, shifts_x, _, warp_method = format_subpixel_shift_params(matrix, shifts_y, shifts_x, None, warp_method)
    matrix, thetas, warp_method = format_rotation_parameters(matrix, thetas, warp_method)
    matrix, scales, warp_method = format_scale_parameters(matrix, scales, warp_method)
    return matrix, shifts_y, shifts_x, thetas, scales, warp_method

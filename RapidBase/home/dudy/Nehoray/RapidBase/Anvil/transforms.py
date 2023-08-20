from typing import Union, Tuple
import torch
import numpy as np
from torch import Tensor

from RapidBase.Anvil._internal_utils.torch_utils import MatrixOrigami
from RapidBase.Anvil._transforms.scale_matrix import _scale_matrix, format_scale_parameters
from RapidBase.Anvil._transforms.shift_matrix_integer_pixels import format_shift_integer_pixels_parameters, shift_n_pixels, shift_n_pixels_uniform
from RapidBase.Anvil._transforms.shift_matrix_subpixel import _shift_matrix_subpixel_fft, \
    _shift_matrix_subpixel_interpolated, format_subpixel_shift_params, format_blur_shift_params
from RapidBase.Anvil._internal_utils.variadic import pick_method
from RapidBase.Anvil._transforms.rotate_matrix import rotate_matrix_fft, rotate_matrix_interpolated, \
    format_rotation_parameters, format_blur_rotation_parameters, size_of_rotated_matrix
from RapidBase.Anvil._transforms.affine_transformations import format_affine_parameters, affine_transform_interpolated


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


def blur_shift_matrix(matrix: Union[torch.Tensor, np.array, tuple, list],
                      shift_H: Union[int, float, torch.Tensor, list, tuple],
                      shift_W: Union[int, float, torch.Tensor, list, tuple], N: int, matrix_FFT=None,
                      warp_method='bilinear') -> torch.Tensor:
    """Blurs matrix over given shifts and iterations
    Note that the pixels beyond the maximum shift should be considered undefined behavior

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be blurred.
    :param shift_H: singleton maximum vertical shift for the shift
    :param shift_W: singleton maximum horizontal shift for the shift
    :param N: number of iterations to blur over
    :param matrix_FFT: For use only when using 'fft' warp method. In case the fft of the matrix has already been calculated, it can be passed to the function to improve performance. FFT must be over dimensions -2, -1
    :param warp_method: method used to warp the matrix when shifting. Default: 'bilinear'. Options: 'bilinear', 'bicubic', 'nearest' and 'fft'
    :return: blurred matrix
    """
    matrix, shift_H, shift_W, N, matrix_FFT, warp_method = format_blur_shift_params(matrix, shift_H, shift_W, N, matrix_FFT, warp_method)
    blurred_image = torch.zeros_like(matrix).to(matrix.device)
    dimensions_memory = MatrixOrigami(blurred_image)
    blurred_image = dimensions_memory.expand_matrix()
    matrix = dimensions_memory.expand_other_matrix(matrix)
    shifty_iteration = shift_H / N
    shiftx_iteration = shift_W / N
    possible_methods = {'fft': _shift_matrix_subpixel_fft, 'bilinear': _shift_matrix_subpixel_interpolated,
                        'bicubic': _shift_matrix_subpixel_interpolated, 'nearest': _shift_matrix_subpixel_interpolated}
    for i in range(N):
        # skips directly to pick method to avoid interface inefficiencies like parameter checking
        blurred_image += pick_method(possible_methods, warp_method, matrix, shifty_iteration * i, shiftx_iteration * i,
                                               matrix_FFT) / N
    return dimensions_memory.squeeze_to_original_dims(blurred_image)


def rotate_matrix(matrix: Union[torch.Tensor, np.array, tuple, list], thetas: Union[torch.Tensor, list, tuple, float, int],
                  warp_method='bilinear') -> torch.Tensor:
    """Rotate matrix counterclockwise according to given rotation. Returned matrix can be expanded to hold full matrix

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be rotated
    :param thetas: singleton or length T tensor of thetas in radians. Rotation goes counterclockwise
    :param warp_method: method used to warp the matrix when rotating. Default: 'bilinear'. Options: 'bilinear', 'bicubic', 'nearest' and 'fft'
    :return: matrix rotated counterclockwise thetas radians

    """
    matrix, thetas, warp_method = format_rotation_parameters(matrix, thetas, warp_method)
    dimensions_memory = MatrixOrigami(matrix)
    extended_matrix = dimensions_memory.expand_matrix()
    possible_methods = {'fft': rotate_matrix_fft, 'bilinear': rotate_matrix_interpolated,
                        'bicubic': rotate_matrix_interpolated, 'nearest': rotate_matrix_interpolated}
    rotated_matrix = pick_method(possible_methods, warp_method, extended_matrix, thetas)
    return dimensions_memory.squeeze_to_original_dims(rotated_matrix)


def blur_rotate_matrix(matrix: Union[torch.Tensor, np.array, tuple, list], thetas: Union[torch.Tensor, list, Tuple, float], N: int,
                       warp_method='bilinear') -> torch.Tensor:
    """Blurs matrix over given rotation and iterations. Returned matrix can be expanded to hold full matrix

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be blurred over rotation
    :param thetas: singleton maximum theta to rotate the matrix over. Rotation goes counterclockwise
    :param N: number of iterations to average over for the blur
    :param warp_method: method used to warp the matrix when rotating. Default: 'bilinear'. Options: 'bilinear', 'bicubic', 'nearest' and 'fft'
    :return: blurred matrix over counterclockwise rotation

    """
    # could use improvement of not parameter checking each time maybe?
    matrix, thetas, N, warp_method = format_blur_rotation_parameters(matrix, thetas, N, warp_method)
    dimensions_memory = MatrixOrigami(matrix)
    extended_matrix = dimensions_memory.expand_matrix()
    possible_methods = {'fft': rotate_matrix_fft, 'bilinear': rotate_matrix_interpolated,
                        'bicubic': rotate_matrix_interpolated, 'nearest': rotate_matrix_interpolated}
    blurred_image = torch.zeros(size_of_rotated_matrix(extended_matrix))
    for i in range(N+1):
        blurred_image += pick_method(possible_methods, warp_method, extended_matrix, (thetas * i)/N)/(N+1)
    return dimensions_memory.squeeze_to_original_dims(blurred_image)


def scale_matrix(matrix: Union[torch.Tensor, np.array, tuple, list], scales: Union[torch.Tensor, list, Tuple, float], warp_method='bilinear'):
    """Scales matrix, both to make it larger or smaller. Returned Matrix will be larger than input matrix. It will be zero padded slightly

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be scaled
    :param scales: singleton or length T scale/s to scale the matrix by
    :param warp_method: method used to warp the matrix when scaling. Default: 'bilinear'. Options: 'bilinear', 'bicubic', and 'nearest'
    :return: scaled matrix

    """
    matrix, scales, warp_method = format_scale_parameters(matrix, scales, warp_method)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    matrix = _scale_matrix(matrix, scales, warp_method)
    return dimensions_memory.squeeze_to_original_dims(matrix)


def affine_warp_matrix(matrix: Union[torch.Tensor, np.array, tuple, list],
                        shifts_y: Union[Tensor, list, tuple, float, int] = None,
                        shifts_x: Union[Tensor, list, tuple, float, int] = None,
                        thetas: Union[Tensor, list, tuple, float, int] = None,
                        scales: Union[Tensor, list, tuple, float, int] = None,
                        warp_method='bilinear')  -> torch.Tensor:
    """Performs affine warp on given matrix (ie. shifts, scaling, and rotation). For now, fft is not supported as a warp method
    If a given warp (ex. shift_H) is not specified, it will be set to the identity value.
    Warning: Can and will automatically expand tensor to fit rotation and scaling.

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be transformed
    :param shifts_y: singleton or length T vertical upwards shift/s of the matrix
    :param shifts_x: singleton or length T horizontal rightwards shift/s of the matrix
    :param thetas: singleton or length T horizontal counterclockwise rotation/s of the matrix
    :param scales: singleton or length T horizontal counterclockwise scaling/s of the matrix
    :param warp_method: method used to warp the matrix. Default: 'bilinear'. Options: 'bilinear', 'bicubic', and 'nearest'
    :return: warped matrix
    """
    matrix, shifts_y, shifts_x, thetas, scales, warp_method = format_affine_parameters(matrix, shifts_y, shifts_x, thetas, scales, warp_method)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    matrix = affine_transform_interpolated(matrix, shifts_y, shifts_x, thetas, scales, warp_method)
    return dimensions_memory.squeeze_to_original_dims(matrix)


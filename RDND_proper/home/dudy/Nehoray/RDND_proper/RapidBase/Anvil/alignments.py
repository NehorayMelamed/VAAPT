from typing import Tuple, Union
import torch
import numpy as np
from torch import Tensor

from RapidBase.Anvil._alignments.cross_correlation_alignments import circular_cross_correlation_classic, \
    format_parameters_classic_circular_cc, format_parameters_normalized_cc, normalized_cc, \
    classic_circular_cc_shifts_calc, normalized_cc_shifts_calc, weighted_circular_cc_calc, format_weighted_ccc_params, format_align_to_reference_frame_circular_cc_params, format_align_to_center_frame_circular_cc_params, align_circular_cc
from RapidBase.Anvil._alignments.minSAD_alignments import format_min_SAD_params, interpolated_minSAD_shifts
from RapidBase.Anvil._internal_utils.torch_utils import MatrixOrigami, NoMemoryMatrixOrigami


###FI checking that the ffts and matrix and reference tensor all come from the same device


def circular_cross_correlation(matrix: Union[torch.Tensor, np.array, tuple, list],
                               reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                               matrix_fft: torch.Tensor = None,
                               reference_fft: torch.Tensor = None,
                               normalize_over_matrix: bool = False,
                               fftshift: bool = False) -> torch.Tensor:
    """Computes circular cross correlation of matrix and reference matrix. ifft(fft(matrix)*conjugate(fft(reference_matrix))).

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against reference matrix
    :param reference_matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against
    :param matrix_fft: possible to pass matrix fft if already computed, to improve performance. FFT should be over dims [-2, -1]
    :param reference_fft: possible to pass reference fft if already computed, to improve performance. FFT should be over dims [-2, -1]
    :param normalize_over_matrix: normalize cross correlation values to be between [0,1]
    :param fftshift: flag whether the cc-ed matrix should also be fftshifted
    :return: circular cross correlation of matrix and reference matrix
    """
    matrix, reference_matrix, matrix_fft, reference_fft, normalize, fftshift = format_parameters_classic_circular_cc(matrix, reference_matrix, matrix_fft, reference_fft, normalize_over_matrix, fftshift)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    reference_matrix = dimensions_memory.expand_other_matrix(reference_matrix)
    cc = circular_cross_correlation_classic(matrix, reference_matrix, matrix_fft, reference_fft, normalize, fftshift)
    return cc


def normalized_cross_correlation(matrix: Union[torch.Tensor, np.array, tuple, list], reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                                 correlation_size: int, fftshift: bool = False) -> torch.Tensor:
    """Computes normalized cross correlation of matrix and reference matrix manually

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against reference matrix
    :param reference_matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against
    :param correlation_size: correlation size of the windows
    :param fftshift: whether the cc-ed matrix should also be fftshifted
    :return: normalized cross correlation of matrix and reference matrix
    """
    matrix, reference_matrix, correlation_size, fftshift = format_parameters_normalized_cc(matrix,
                                                                                           reference_matrix, correlation_size, fftshift)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    reference_matrix = dimensions_memory.expand_other_matrix(reference_matrix)
    return normalized_cc(matrix, reference_matrix, correlation_size)


def circular_cc_shifts(matrix: Union[torch.Tensor, np.array, tuple, list], reference_matrix: Union[torch.Tensor, np.array, list, tuple],
                       matrix_fft: Union[torch.Tensor, np.array, tuple, list] = None, reference_matrix_fft: Union[torch.Tensor, np.array, list, tuple] = None,
                       normalize_over_matrix: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes subpixel shifts between given matrix and reference matrix using circular cross correlation

    :param matrix:  2-5D matrix of form (B,T,C,H,W) to be correlated against reference matrix
    :param reference_matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against
    :param matrix_fft: possible to pass matrix fft if already computed, to improve performance. FFT should be over dims [-2, -1]
    :param reference_matrix_fft: possible to pass reference fft if already computed, to improve performance. FFT should be over dims [-2, -1]
    :param normalize_over_matrix: whether to normalize the matrix values when calculating the cross correlation
    :return: tuple of three tensors. The first tensor is vertical shifts, the second tensor is horizontal shifts, and the third tensor is the circular cross correlation
    """
    matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, _ = format_parameters_classic_circular_cc(
                                                                    matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, False)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    reference_matrix = dimensions_memory.expand_other_matrix(reference_matrix)
    shifts_y, shifts_x, cc = classic_circular_cc_shifts_calc(matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, False)
    return shifts_y, shifts_x, cc


def normalized_cc_shifts(matrix: Union[torch.Tensor, np.array, tuple, list], reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                         correlation_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes subpixel shifts between given matrix and reference matrix using normalized cross correlation

    :param matrix:  2-5D matrix of form (B,T,C,H,W) to be correlated against reference matrix
    :param reference_matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against
    :param correlation_size: correlation size of the windows
    :return: tuple of three tensors. The first tensor is vertical shifts, the second tensor is horizontal shifts, and the third tensor is the normalized cross correlation
    """
    matrix, reference_matrix, correlation_size, _ = format_parameters_normalized_cc(matrix, reference_matrix, correlation_size)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    reference_matrix = dimensions_memory.expand_other_matrix(reference_matrix)
    shifts_y, shifts_x, cc = normalized_cc_shifts_calc(matrix, reference_matrix, correlation_size)
    return shifts_y, shifts_x, cc


def min_SAD_detect_shifts(matrix: Union[torch.Tensor, np.array, tuple, list],
                          reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                          shifty_vec: Union[Tensor, tuple, list] = None,
                          shiftx_vec: Union[Tensor, tuple, list] = None,
                          rotation_vec: Union[Tensor, tuple, list] = None,
                          scale_vec: Union[Tensor, tuple, list] = None,
                          warp_method: str='bilinear') -> Tuple[float, float, float, float, Tensor]:
    """Computes affine shifts/rotations/scaling from matrix to reference matrix using minimum sum of differences.
    Correct warps should not be placed at edges of search spaces for warps. Pad if needed.

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against reference matrix
    :param reference_matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against matrix
    :param shiftx_vec: search space for horizontal shifts. Contains all values that will be checked for horizontal shifts
    :param shifty_vec: search space for vertical shifts. Contains all values that will be checked for vertical shifts
    :param rotation_vec: search space for rotation. Contains all values that will be checked for rotation
    :param scale_vec: search space for scaling. Contains all values that will be checked for scaling
    :param warp_method: warp_method: method used to warp the matrix when calculating sum of differences. Default: 'bilinear'. Options: 'bilinear', 'bicubic', and 'nearest'
    :return: tuple of 5 elements. First 4 are vertical shift, horizontal shift, rotational shift, and scaling that minimize the sum of difference with the reference matrix. Final element is minimum sum of differences matrix itself
    """
    matrix, reference_matrix, shifty_vec, shiftx_vec, rotation_vec, scale_vec, warp_method = format_min_SAD_params(matrix,
                                                                                                                   reference_matrix,
                                                                                                                   shifty_vec,
                                                                                                                   shiftx_vec,
                                                                                                                   rotation_vec,
                                                                                                                   scale_vec,
                                                                                                                   warp_method)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    reference_matrix = dimensions_memory.expand_other_matrix(reference_matrix)
    shift_y, shift_x, rotational_shift, scale_shift, min_sad_matrix = interpolated_minSAD_shifts(matrix, reference_matrix,
                                                                                 shifty_vec, shiftx_vec, rotation_vec, scale_vec, warp_method)

    return shift_y, shift_x, rotational_shift, scale_shift, min_sad_matrix


def weighted_circular_cross_correlation(matrix: Union[torch.Tensor, np.array, tuple, list],
                                        reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                                        weights: Union[torch.Tensor, np.array, tuple, list],
                                        matrix_fft: Tensor = None, reference_matrix_fft: Tensor = None) -> Tensor:
    """Untested!!! Use with caution! Computes weighted cross correlation

    :param matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against reference matrix
    :param reference_matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against matrix
    :param weights: weights to be used for cross correlation
    :param matrix_fft: possible to pass matrix fft if already computed, to improve performance. FFT should be over dims [-2, -1]
    :param reference_matrix_fft: possible to pass reference fft if already computed, to improve performance. FFT should be over dims [-2, -1]
    :return: weighted cross correlation of matrix and reference matrix
    """
    matrix, reference_matrix, weights = format_weighted_ccc_params(matrix, reference_matrix, weights, matrix_fft, reference_matrix_fft)
    dimensions_memory = MatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix()
    reference_matrix = dimensions_memory.expand_other_matrix(reference_matrix)
    weighted_cc = weighted_circular_cc_calc(matrix, reference_matrix, weights, matrix_fft, reference_matrix_fft)
    return weighted_cc


# TODO Elisheva added this
def align_to_reference_frame_circular_cc(matrix: Union[torch.Tensor, np.array, tuple, list],
                                         reference_matrix: Union[torch.Tensor, np.array, list, tuple],
                                         matrix_fft: Union[torch.Tensor, np.array, tuple, list] = None,
                                         reference_matrix_fft: Union[torch.Tensor, np.array, list, tuple] = None,
                                         normalize_over_matrix: bool = False,
                                         warp_method: str = 'bilinear',
                                         crop_warped_matrix: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Validate parameters
    matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, warp_method, crop_warped_matrix, _ = \
        format_align_to_reference_frame_circular_cc_params(matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix,
                                                                     warp_method, crop_warped_matrix, False)
    # Memorize dimensions
    dimensions_memory = NoMemoryMatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix(matrix)
    reference_matrix = dimensions_memory.expand_matrix(reference_matrix)
    if matrix_fft is not None:
        matrix_fft = dimensions_memory.expand_matrix(matrix_fft)
    if reference_matrix_fft is not None:
        reference_matrix_fft = dimensions_memory.expand_matrix(reference_matrix_fft)
    # Warp matrix
    warped_matrix, shifts_h, shifts_w, cc = align_circular_cc(
        matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, crop_warped_matrix, warp_method, False)
    # Return to original dimensionality
    warped_matrix = dimensions_memory.squeeze_to_original_dims(warped_matrix)
    cc = dimensions_memory.squeeze_to_original_dims(cc)

    return warped_matrix, shifts_h, shifts_w, cc


def align_to_center_frame_circular_cc(matrix: Union[torch.Tensor, np.array, tuple, list],
                                      matrix_fft: Union[torch.Tensor, np.array, tuple, list] = None,
                                      normalize_over_matrix: bool = False,
                                      warp_method: str = 'bilinear',
                                      crop_warped_matrix: bool=False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Validate parameters
    matrix, matrix_fft, normalize_over_matrix, warp_method, crop_warped_matrix = format_align_to_center_frame_circular_cc_params(
        matrix, matrix_fft, normalize_over_matrix, warp_method, crop_warped_matrix)
    # Memorize dimensions
    dimensions_memory = NoMemoryMatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix(matrix)
    # Calculate center frame
    B, T, C, H, W = matrix.shape
    reference_matrix = matrix[:, T//2: T//2 + 1]
    if matrix_fft is not None:
        matrix_fft = dimensions_memory.expand_matrix(matrix_fft)
        reference_matrix_fft = matrix_fft[:, T//2: T//2 + 1]
    else:
        reference_matrix_fft = None
    # Warp matrix
    warped_matrix, shifts_h, shifts_w, cc = align_circular_cc(
        matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, crop_warped_matrix, warp_method, False)
    # Return to original dimensionality
    warped_matrix = dimensions_memory.squeeze_to_original_dims(warped_matrix)
    cc = dimensions_memory.squeeze_to_original_dims(cc)

    return warped_matrix, shifts_h, shifts_w, cc


def align_to_reference_frame_weighted_circular_cc(matrix: Union[torch.Tensor, np.array, tuple, list],
                                                  reference_matrix: Union[torch.Tensor, np.array, list, tuple],
                                                  weights_tensor: Union[torch.Tensor, np.array, list, tuple],
                                                  matrix_fft: Union[torch.Tensor, np.array, tuple, list] = None,
                                                  reference_matrix_fft: Union[torch.Tensor, np.array, list, tuple] = None,
                                                  normalize_over_matrix: bool = False,
                                                  warp_method: str = 'bilinear',
                                                  crop_warped_matrix: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Validate parameters
    #TODO: make one for weighs_cross_correlation
    matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, warp_method, crop_warped_matrix, _ = \
        format_align_to_reference_frame_circular_cc_params(matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix,
                                                                     warp_method, crop_warped_matrix, False)

    # Memorize dimensions
    dimensions_memory = NoMemoryMatrixOrigami(matrix)
    matrix = dimensions_memory.expand_matrix(matrix)
    reference_matrix = dimensions_memory.expand_matrix(reference_matrix)
    if matrix_fft is not None:
        matrix_fft = dimensions_memory.expand_matrix(matrix_fft)
    if reference_matrix_fft is not None:
        reference_matrix_fft = dimensions_memory.expand_matrix(reference_matrix_fft)
    # Warp matrix
    warped_matrix, shifts_h, shifts_w, cc = align_circular_cc(
        matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, crop_warped_matrix, warp_method, False)
    # Return to original dimensionality
    warped_matrix = dimensions_memory.squeeze_to_original_dims(warped_matrix)
    cc = dimensions_memory.squeeze_to_original_dims(cc)

    return warped_matrix, shifts_h, shifts_w, cc




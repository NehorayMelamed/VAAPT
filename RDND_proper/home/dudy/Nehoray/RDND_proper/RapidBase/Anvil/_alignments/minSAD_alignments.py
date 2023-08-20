from typing import Union, Tuple

import math
import numpy as np
import torch
from torch import Tensor

from RapidBase.Anvil._alignments.cross_correlation_alignments import fit_polynomial
from RapidBase.Anvil._internal_utils.parameter_checking import type_check_parameters, validate_warp_method
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if, raise_if_not
from RapidBase.Anvil._internal_utils.torch_utils import unravel_index, true_dimensionality, \
    construct_tensor, multiplicable_tensors, pad_matrix_to_size, ravel_multi_index
from RapidBase.Anvil._transforms.affine_transformations import affine_transform_interpolated


def interpolated_minSAD_defaults(shifty_vec: torch.Tensor = None, shiftx_vec: torch.Tensor = None,
                                 rotation_vec: torch.Tensor = None,
                                 scale_vec: torch.Tensor = None, device='cpu') -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    ###Q what should the defaults be
    if shifty_vec is None:
        shifty_vec = Tensor([0])
    if shiftx_vec is None:
        shiftx_vec = Tensor([0])
    if rotation_vec is None:
        rotation_vec = Tensor([0])
    if scale_vec is None:
        scale_vec = Tensor([1])
    return shifty_vec, shiftx_vec, rotation_vec, scale_vec


def max_displacement(matrix_dimensions: Tuple[int, int, int, int, int], shift_vec: torch.Tensor,
                     rotation_vec: torch.Tensor,
                     scale_vec: torch.Tensor) -> int:
    # returns max displacement for one of H or W
    # backup padding is the added safety zone
    B, T, C, H, W = matrix_dimensions
    diagonal = math.sqrt(H ** 2 + W ** 2)
    max_translation = float(torch.max(torch.abs(shift_vec)))
    max_rotation_angle = float(torch.max(torch.abs(rotation_vec)))
    max_displacement = (max_translation + diagonal * math.tan(max_rotation_angle)) / max(scale_vec)
    return math.ceil(max_displacement)


def displacement(matrix_dimensions: Tuple[int, int, int, int, int], shift_x: torch.Tensor, rotation_vec: torch.Tensor,
                 scale_vec: torch.Tensor) -> int:
    raise NotImplemented


# TODO Elisheva changed this to also match parallel compute for B or T > 1
def sub_pixel_shifts(shifts: Tensor, central_shift_index: int, SAD_values: Tensor) -> float:
    # extract sub pixel shift from 3 SAD values
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, SAD_values)
    delta_shift_x = -b_x / (2 * a_x)
    return shifts[central_shift_index] + delta_shift_x * (shifts[2] - shifts[1])


# TODO Elisheva wrote this, hopefully a final polished version
def parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec: Tensor,
                                      shifts_min_indices_nd: Tensor,
                                      shift_dimension: int,
                                      SAD_values_flattened: Tensor,
                                      SAD_dimensions: tuple) -> Tensor:
    """Calculates parabola fit for a given n dimensional tensor of shift indices over a specified dimension.

    :param possible_shifts_1d_vec: all possible discrete shifts for the chosen dimension. Assumption: shifts are
        organized in increasing order with fixed intervals.
    :param shifts_min_indices_nd: n dimensional array of shift indices for minimal SAD values
    :param shift_dimension: dimension to fit parabola over
    :param SAD_values_flattened: all returned SAD values, **already flattened**. Expected dimensions
        are (mult(len(shift_vecs)), B*T)
    :param SAD_dimensions: dimensions of the unflattened SAD values; practically the shift vectors' lengths
    :return: exact shift over the chosen dimension calculated with parabola fit
    """
    # if only one possible shift value, no need to fit
    if possible_shifts_1d_vec.shape[0] == 1:
        return possible_shifts_1d_vec[shifts_min_indices_nd[shift_dimension]]

    # Support method for readability: add a fixed value to a row in a tensor (row -> single lower dimension unit)
    def add_val_to_row(tsr: Tensor, row_num: int, val: Union[int, float]) -> Tensor:
        return torch.stack([(tsr[i] + val if i == row_num else tsr[i]) for i in range(tsr.shape[0])])

    # Create interval for parabola fit over the chosen dimension: minimal value indices and the two neighboring values
    # over the chosen dimensionshift
    shift_index_minus1 = add_val_to_row(shifts_min_indices_nd, row_num=shift_dimension, val=-1)
    shift_index_plus1 = add_val_to_row(shifts_min_indices_nd, row_num=shift_dimension, val=1)

    # get flattened indices of the interval
    shift_index_minus1_flattened = ravel_multi_index(tuple(shift_index_minus1), SAD_dimensions)
    shift_index_flattened = ravel_multi_index(tuple(shifts_min_indices_nd), SAD_dimensions)
    shift_index_plus1_flattened = ravel_multi_index(tuple(shift_index_plus1), SAD_dimensions)

    # Extract the min interval values and construct the tensor for parabola fit
    BxT = SAD_values_flattened.shape[-1]
    shift_matrix_to_fix = torch.Tensor(
        [[SAD_values_flattened[shift_index_minus1_flattened[i], i] for i in range(BxT)],
         [SAD_values_flattened[shift_index_flattened[i], i] for i in range(BxT)],
         [SAD_values_flattened[shift_index_plus1_flattened[i], i] for i in range(BxT)]])
    shift_matrix_to_fix = torch.transpose(shift_matrix_to_fix, 1, 0)

    # Do parabola fit
    x_vec = [-1, 0, 1]
    c_x, b_x, a_x = fit_polynomial(x_vec, shift_matrix_to_fix)
    delta_shift_x = -b_x / (2 * a_x)
    total_shift = possible_shifts_1d_vec[shifts_min_indices_nd[shift_dimension]] + delta_shift_x * (
            possible_shifts_1d_vec[2] - possible_shifts_1d_vec[1])

    return total_shift


# TODO this method doesn't work for 0D vectors. Elisheva changed (added initial condition)
def non_edge_shift_index(min_index: int, warp_vec: Tensor) -> int:
    """
    :param min_index: min index of given warp vector
    :param warp_vec: warp vector from which the min index is selected
    :return: min index that is not on the border
    """
    if len(warp_vec) <= 2:
        return min_index
    min_index = max(1, min_index)
    min_index = min(warp_vec.shape[-1] - 2, min_index)
    return min_index


def extract_warp(warp_vec: Tensor, global_minimum_index,
                 SAD_matrix_flattened: Tensor,
                 y_index: int,
                 x_index: int,
                 rotation_index: int,
                 scale_index: int) -> float:
    if len(warp_vec) == 1:
        return float(warp_vec[0])
    max_indices = Tensor([global_minimum_index - 1, global_minimum_index, global_minimum_index + 1]).to(torch.long)
    if y_index is None:
        warp = sub_pixel_shifts(warp_vec, global_minimum_index,
                                SAD_matrix_flattened[max_indices, x_index, rotation_index, scale_index])
    elif x_index is None:
        warp = sub_pixel_shifts(warp_vec, global_minimum_index,
                                SAD_matrix_flattened[y_index, max_indices, rotation_index, scale_index])
    elif rotation_index is None:
        warp = sub_pixel_shifts(warp_vec, global_minimum_index,
                                SAD_matrix_flattened[y_index, x_index, max_indices, scale_index])
    else:
        warp = sub_pixel_shifts(warp_vec, global_minimum_index,
                                SAD_matrix_flattened[y_index, x_index, rotation_index, max_indices])
    return warp


def interpolated_minSAD_shifts(matrix: torch.Tensor, reference_matrix: torch.Tensor, shifty_vec: torch.Tensor = None,
                               shiftx_vec: torch.Tensor = None,
                               rotation_vec: torch.Tensor = None, scale_vec: torch.Tensor = None,
                               warp_method: str = 'bilinear') \
        -> Tuple[float, float, float, float, Tensor]:
    print(shifty_vec)
    print(rotation_vec)
    shifty_vec, shiftx_vec, rotation_vec, scale_vec = interpolated_minSAD_defaults(shifty_vec, shiftx_vec, rotation_vec,
                                                                                   scale_vec, matrix.device)
    B, T, C, H, W = matrix.shape
    SAD_dimensions: Tuple[int] = (max(shifty_vec.shape), max(shiftx_vec.shape), max(rotation_vec.shape), max(scale_vec.shape))
    ### Build All Possible Grids: ##
    # TODO: accelerate this!!!! parallelize
    # TODO: instead of saving the entire images to gpu memory - at each iteration calculate the SAD!!!!
    counter = 0
    encompassing_length = int(math.sqrt(H ** 2 + W ** 2) * max(float(torch.max(scale_vec)), 1) + 10)  # 10 is extra
    reference_frame = pad_matrix_to_size(reference_matrix, (encompassing_length, encompassing_length))
    matrix_frame = pad_matrix_to_size(matrix, (encompassing_length, encompassing_length))
    SAD_list = []
    for shift_y_counter, current_shift_y in enumerate(shifty_vec):
        for shift_x_counter, current_shift_x in enumerate(shiftx_vec):
            for rotation_counter, current_rotation_angle in enumerate(rotation_vec):
                for scale_counter, current_scale in enumerate(scale_vec):
                    # TODO: only crop what you need, not everything
                    warped_matrix = affine_transform_interpolated(matrix_frame, construct_tensor(current_shift_y),
                                                                  construct_tensor(current_shift_x),
                                                                  construct_tensor(current_rotation_angle),
                                                                  construct_tensor(current_scale),
                                                                  warp_method=warp_method, expand=False)

                    current_SAD = (
                                (warped_matrix - reference_frame) * (warped_matrix > 0) * (reference_frame > 0)).abs()
                    current_SAD_mean = current_SAD.mean(-1, True).mean(-2, True).mean(-3, True)
                    ### Add SAD to list: ###
                    SAD_list.append(current_SAD_mean)
                    counter = counter + 1

    ### Get Min SAD: ###
    SAD_matrix = torch.cat(SAD_list, 0)
    print(torch.min(SAD_matrix))
    min_index = int(torch.argmin(SAD_matrix))
    min_indices = unravel_index(min_index, SAD_dimensions)
    print(min_index)
    print(min_indices)
    shift_y_index = non_edge_shift_index(int(min_indices[0]), shifty_vec)
    shift_x_index = non_edge_shift_index(int(min_indices[1]), shiftx_vec)
    rotation_index = non_edge_shift_index(int(min_indices[2]), rotation_vec)
    scale_index = non_edge_shift_index(int(min_indices[3]), scale_vec)

    SAD_matrix_2_reshaped = torch.reshape(SAD_matrix.squeeze(), SAD_dimensions)

    # (2). Get Indices For Parabola Fitting:   #TODO: make sure this is correct!!!!!! use the cross correlation parabola fit corrections!!!!

    if len(shifty_vec) == 1:
        shift_y_sub_pixel = float(shifty_vec)
    else:
        shift_y_indices = Tensor([shift_y_index - 1, shift_y_index, shift_y_index + 1]).to(torch.long)
        shift_y_sub_pixel = sub_pixel_shifts(shifty_vec, shift_y_index, SAD_matrix_2_reshaped[
            shift_y_indices, shift_x_index, rotation_index, scale_index])

    if len(shiftx_vec) == 1:
        shift_x_sub_pixel = float(shiftx_vec[0])
    else:
        shift_x_indices = Tensor([shift_x_index - 1, shift_x_index, shift_x_index + 1]).to(torch.long)
        shift_x_sub_pixel = sub_pixel_shifts(shiftx_vec, shift_x_index, SAD_matrix_2_reshaped[
            shift_y_index, shift_x_indices, rotation_index, scale_index])

    if len(rotation_vec) == 1:
        rotation_sub_pixel = float(rotation_vec)
    else:
        rotation_indices = Tensor([rotation_index - 1, rotation_index, rotation_index + 1]).to(torch.long)
        rotation_sub_pixel = sub_pixel_shifts(rotation_vec, rotation_index, SAD_matrix_2_reshaped[
            shift_x_index, shift_y_index, rotation_indices, scale_index])

    if len(scale_vec) == 1:
        scale_sub_pixel = float(scale_vec[0])
    else:
        scale_indices = Tensor([scale_index - 1, scale_index, scale_index + 1]).to(torch.long)
        scale_sub_pixel = sub_pixel_shifts(scale_vec, scale_index, SAD_matrix_2_reshaped[
            shift_x_index, shift_y_index, rotation_index, scale_indices])
    return float(shift_y_sub_pixel), float(shift_x_sub_pixel), float(rotation_sub_pixel), float(
        scale_sub_pixel), SAD_matrix


def check_shift_vec(vec: Tensor) -> Tensor:
    if vec.numel() > 1:
        vec = vec.squeeze()  # TODO: problem because if i receive a 1D tensor of 1 element it makes it a 0 element tensor
    raise_if_not(len(vec.shape) == 1)
    raise_if(len(vec.shape) == 2)  # impossible to do anything with this
    return vec


# TODO Elisheva added device
def format_min_SAD_params(matrix: Union[Tensor, tuple, list, np.array],
                          reference_matrix: Union[Tensor, tuple, list, np.array],
                          shifty_vec: Union[Tensor, tuple, list] = None, shiftx_vec: Union[Tensor, tuple, list] = None,
                          rotation_vec: Union[Tensor, tuple, list] = None, scale_vec: Union[Tensor, tuple, list] = None,
                          warp_method: str = 'bilinear', device: Union[str, torch.device] = 'cpu'
                          ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str]:
    type_check_parameters([(matrix, (Tensor, tuple, list, np.ndarray)),
                           (reference_matrix, (Tensor, tuple, list, np.ndarray)),
                           (shifty_vec, (Tensor, tuple, list, type(None))),
                           (shiftx_vec, (Tensor, tuple, list, type(None))),
                           (rotation_vec, (Tensor, tuple, list, type(None))),
                           (scale_vec, (Tensor, tuple, list, type(None))), (warp_method, str)])
    matrix = construct_tensor(matrix, device)
    reference_matrix = construct_tensor(reference_matrix, device)
    if shifty_vec is not None:
        shifty_vec = check_shift_vec(construct_tensor(shifty_vec))
    if shiftx_vec is not None:
        shiftx_vec = check_shift_vec(construct_tensor(shiftx_vec))
    if rotation_vec is not None:
        rotation_vec = check_shift_vec(construct_tensor(rotation_vec))
    if scale_vec is not None:
        scale_vec = check_shift_vec(construct_tensor(scale_vec))

    # TODO Elisheva added this
    # if reference matrix dimension is smaller than matrix dimension (for example frame vs video), unsqueeze
    while len(reference_matrix.shape) < len(matrix.shape):
        reference_matrix = reference_matrix.unsqueeze(0)

    raise_if(true_dimensionality(matrix) == 0, message="Matrix is empty")
    raise_if(true_dimensionality(reference_matrix) == 0, message="Reference Matrix is empty")
    raise_if_not(multiplicable_tensors([matrix, reference_matrix]),
                 message="Matrix and Reference Matrix have different sizes")
    validate_warp_method(warp_method, valid_methods=['bilinear', 'bicubic', 'nearest'])
    return matrix, reference_matrix, shifty_vec, shiftx_vec, rotation_vec, scale_vec, warp_method


# TODO Elisheva added this
def format_minSAD_align_parameters(matrix: Union[torch.Tensor, np.array, tuple, list],
                                   reference_matrix: Union[torch.Tensor, np.array, tuple, list, type(None)],
                                   shift_h_vec: Union[Tensor, tuple, list] = None,
                                   shift_w_vec: Union[Tensor, tuple, list] = None,
                                   rotation_vec: Union[Tensor, tuple, list] = None,
                                   scale_vec: Union[Tensor, tuple, list] = None,
                                   warp_method: str = 'bilinear',
                                   align_to_center_frame: bool = False,
                                   warp_matrix: bool = True,
                                   return_shifts: bool = False,
                                   device: Union[str, torch.device] = 'cpu'
                                   ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str, bool, bool, bool]:
    # Validate the parameters that don't exist in regular minSAD calculation
    type_check_parameters([(align_to_center_frame, bool), (warp_matrix, bool), (return_shifts, bool)])

    # Validate reference with align_to_center_frame and take care of the case reference_matrix = None
    if reference_matrix is None:
        raise_if_not(align_to_center_frame,
                     message="reference matrix can't be of type None if reference is not center frame")
        # if no reference matrix needed, create a fake reference to pass the validation:
        reference_matrix = matrix
    else:
        raise_if(align_to_center_frame,
                 message="Reference matrix should not be sent if required to reference the center frame")

    # Validate all the tests of regular minSAD shifts calculation
    matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec, warp_method = \
        format_min_SAD_params(matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec,
                              warp_method, device)

    return matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec, warp_method, \
           align_to_center_frame, warp_matrix, return_shifts

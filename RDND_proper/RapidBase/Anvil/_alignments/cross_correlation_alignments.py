import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple, Union

from RapidBase.Anvil._alignments.canny_edge_detection import canny_edge_detection
from RapidBase.Anvil._internal_utils.parameter_checking import type_check_parameters, validate_warp_method
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if, raise_if_not
from RapidBase.Anvil._internal_utils.torch_utils import faster_fft, true_dimensionality, \
    compare_unequal_dimensionality_tensors, construct_tensor, RGB2BW, BW2RGB, center_logical_mask, \
    multiplicable_tensors, non_interpolated_center_crop, compare_unequal_outer_dimensions

# TODO parabola fit is not matched for parallel computation - hard coded for one dimension. to reverse my changes,
#  make every y[..., a] to y[a]
from RapidBase.Anvil import shift_matrix_subpixel


def fit_polynomial(x: Union[torch.Tensor, list], y: Union[torch.Tensor, list]) -> List[float]:
    # solve for 2nd degree polynomial deterministically using three points seperated by distance of 1
    a = (y[..., 2] + y[..., 0] - 2 * y[..., 1]) / 2
    b = -(y[..., 0] + 2 * a * x[1] - y[..., 1] - a)
    c = y[..., 1] - b * x[1] - a * x[1] ** 2
    return [c, b, a]


def normalize_cc_matrix(cc_matrix: Tensor, matrix: Tensor, reference_matrix: Tensor) -> Tensor:
    H, W = cc_matrix.shape[-2:]
    A_sum = reference_matrix.sum(dim=[-1, -2])
    A_sum2 = (reference_matrix ** 2).sum(dim=[-1, -2])
    sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
    sigmaB = (matrix).std(dim=[-1, -2]) * (H * W - 1) ** (1 / 2)
    B_mean = (matrix).mean(dim=[-1, -2])
    normalized_cc = (cc_matrix - (A_sum * B_mean).unsqueeze(-1).unsqueeze(-1)) / (sigmaA * sigmaB).unsqueeze(
        -1).unsqueeze(-1)
    return normalized_cc


def circular_cross_correlation_classic(matrix: Tensor, reference_matrix: Tensor, matrix_fft: Union[Tensor, None] = None,
                                       reference_matrix_fft: Union[Tensor, None] = None,
                                       normalize: bool = False, fftshift: bool = False) -> Tensor:
    #TODO: dudy
    # matrix_fft = faster_fft(matrix, matrix_fft)
    # reference_tensor_fft = faster_fft(reference_matrix, reference_matrix_fft)
    # cc = torch.fft.ifftn(matrix_fft * reference_tensor_fft.conj(), dim=[-1, -2]).real
    a=1+1
    cc = torch.fft.ifftn(faster_fft(matrix, matrix_fft) * faster_fft(reference_matrix, reference_matrix_fft).conj(), dim=[-1, -2]).real
    if normalize:
        cc = normalize_cc_matrix(cc, matrix, reference_matrix)
    if fftshift:
        cc = torch.fft.fftshift(cc, dim=[-2, -1])
    return cc


def normalized_cc(tensor1: Tensor, tensor2: Tensor, correlation_size: int):
    if correlation_size % 2 == 0:
        correlation_size += 1
    B, T, C, H, W = tensor2.shape
    Correlations = torch.zeros((B, T, C, correlation_size, correlation_size)).to(tensor1.device)
    trim = int(np.floor(correlation_size / 2))
    RimU = trim
    RimD = trim
    RimL = trim
    RimR = trim

    ### A,B Length: ###
    BLy = H - RimU - RimD
    BLx = W - RimL - RimR
    ALy = BLy
    ALx = BLx

    ### Displacement: ###
    B_upper_left = [RimU, RimL]  # the location of the upper-left corner of the Broi matrix
    DispUD = np.arange(-RimD, RimU + 1)
    DispLR = np.arange(-RimL, RimR + 1)

    ### B-ROI: ###
    Broi = tensor2[:, :, :, RimU:H - RimD, RimL:W - RimR]
    Broibar = Broi.mean(3, True).mean(4, True)
    Broiup = (Broi - Broibar)
    Broidown = ((Broi - Broibar) ** 2).sum(3, True).sum(4, True)

    ### Get Cross-Correlation: ###
    for iin in np.arange(len(DispUD)):
        for jjn in np.arange(len(DispLR)):
            shift_y = DispUD[iin]
            shift_x = DispLR[jjn]
            A_upper_left = [B_upper_left[0] + shift_y, B_upper_left[1] + shift_x]
            Atmp = tensor1[:, :, :, A_upper_left[0]:A_upper_left[0] + ALy, A_upper_left[1]:A_upper_left[1] + ALx]
            Abar = Atmp.mean(3, True).mean(4, True)
            Aup = (Atmp - Abar)
            Adown = ((Atmp - Abar) ** 2).sum(3, True).sum(4, True)
            # from icecream import ic
            # ic.configureOutput(includeContext=True)
            # ic(Aup.shape)
            # ic(Broiup.shape)
            Correlations[:, :, :, iin, jjn] = (Broiup * Aup).sum(3, False).sum(3, False) / torch.sqrt(
                Broidown * Adown).squeeze(3).squeeze(3)
    return Correlations


def format_parameters_normalized_cc(matrix: Union[torch.Tensor, np.array, tuple, list],
                                    reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                                    correlation_size: int, fftshift: bool = False):
    type_check_parameters(
        [(matrix, (Tensor, np.array, tuple, list)), (reference_matrix, (Tensor, np.array, tuple, list)),
         (correlation_size, int), (fftshift, bool)])
    matrix = construct_tensor(matrix)
    reference_matrix = construct_tensor(reference_matrix)
    raise_if(true_dimensionality(matrix) == 0, message="Matrix is empty")
    raise_if(true_dimensionality(reference_matrix) == 0, message="Reference Matrix is empty")
    if len(matrix.shape) > 2:
        raise_if_not(matrix.shape[-3] != 1 or matrix.shape[-3] != 3, message="Matrix must be grayscale or RGB")
    if len(reference_matrix.shape) > 2:
        raise_if_not(matrix.shape[-3] != 1 or matrix.shape[-3] != 3, message="Matrix must be grayscale or RGB")
    gray_matrix = RGB2BW(matrix)
    gray_reference_matrix = RGB2BW(reference_matrix)
    if correlation_size % 2 == 0:
        correlation_size += 1
    return gray_matrix, gray_reference_matrix, correlation_size, fftshift


def format_parameters_classic_circular_cc(matrix: Union[Tensor, np.array, tuple, list],
                                          reference_matrix: Union[Tensor, np.array, tuple, list],
                                          matrix_fft: torch.Tensor = None, reference_fft: torch.Tensor = None,
                                          normalize: bool = False,
                                          fftshift: bool = False) -> Tuple[
    Tensor, Tensor, Union[Tensor, None], Union[Tensor, None], bool, bool]:
    type_check_parameters(
        [(matrix, (Tensor, np.array, tuple, list)), (reference_matrix, (Tensor, np.array, tuple, list)),
         (matrix_fft, (Tensor, type(None))), (reference_fft, (Tensor, type(None))), (fftshift, bool),
         (normalize, bool)])
    matrix = construct_tensor(matrix)
    reference_matrix = construct_tensor(reference_matrix)
    raise_if(true_dimensionality(matrix) == 0, message="Matrix is empty")
    raise_if(true_dimensionality(reference_matrix) == 0, message="Reference Matrix is empty")
    raise_if_not(compare_unequal_outer_dimensions(matrix, reference_matrix),
                 message="Matrix and Reference matrix are not same size")
    if matrix_fft is not None:  # is not None
        raise_if(compare_unequal_dimensionality_tensors(matrix, matrix_fft),
                 message="Matrix and Matrix FFT are same size")
    if reference_fft is not None:
        raise_if(compare_unequal_dimensionality_tensors(reference_matrix, reference_fft),
                 message="Reference matrix and Reference matrix FFT are not same size")
    if len(matrix.shape) > 2:
        raise_if_not(matrix.shape[-3] != 1 or matrix.shape[-3] != 3, message="Matrix must be grayscale or RGB")
    if len(reference_matrix.shape) > 2:
        raise_if_not(matrix.shape[-3] != 1 or matrix.shape[-3] != 3, message="Matrix must be grayscale or RGB")
    gray_matrix = RGB2BW(matrix)
    gray_reference_matrix = RGB2BW(reference_matrix)
    return gray_matrix, gray_reference_matrix, matrix_fft, reference_fft, normalize, fftshift


def shifts_from_circular_cc(batch_cc: Tensor, midpoints: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
    # dudi wrote this function, I had nothing to do with it
    # Elisheva added the B dim to enable parallelism :)
    B, T, _, H, W = batch_cc.shape  # _ is C, but C must be 1
    output_CC_flattened_indices = torch.argmax(batch_cc.contiguous().view(B, T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    # (1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    # (2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > W] -= W  # TODO commit this change to Anvil
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * W
    output_CC_flattened_indices_i0 = i1 + i0 * W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W

    ### Get Proper Values For Fit: ###
    output_CC = batch_cc.contiguous().view(B, T, H * W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, -1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, -1, output_CC_flattened_indices_i1_plus1)

    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat(
        [output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat(
        [output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(x_vec, fitting_points_x.squeeze())
    [c_y, b_y, a_y] = fit_polynomial(y_vec, fitting_points_y.squeeze())
    delta_shiftx = -b_x / (2 * a_x)
    delta_shifty = -b_y / (2 * a_y)
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    return shifty, shiftx


# TODO Elisheva changed to match parallel compute. max_index is a list of indices. not so pretty but well...
#  actually could write it more elegantly but wanted to move fast
def detect_edge_cases(max_index: Union[int, torch.Tensor], correlation_size: int) \
        -> Union[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if max_index is int:
        if max_index == correlation_size - 1:
            return max_index - 1, max_index, max_index - 1
        elif max_index == 0:
            return max_index + 1, max_index, max_index + 1
        else:
            return max_index - 1, max_index, max_index + 1
    else:  # max_index tensor of indices
        idx1, idx2, idx3 = torch.zeros_like(max_index), torch.zeros_like(max_index), torch.zeros_like(max_index)
        for i, max_idx in enumerate(max_index):
            if max_idx == correlation_size - 1:
                idx1[i], idx2[i], idx3[i] = max_idx - 1, max_idx, max_idx - 1
            elif max_idx == 0:
                idx1[i], idx2[i], idx3[i] = max_idx + 1, max_idx, max_idx + 1
            else:
                idx1[i], idx2[i], idx3[i] = max_idx - 1, max_idx, max_idx + 1
        return idx1, idx2, idx3



def shifts_from_normalized_cc(output_CC: Tensor, correlation_size: int) -> Tuple[float, float]:
    T, _, H, W = output_CC.shape  ## maybe matrix
    ### Get Correct Max Indices For Fit: ###
    # TODO: add what to do when max index is at the edges
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, correlation_size * correlation_size),
                                               dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // correlation_size  # (*). discrete row (H dimension) max indices
    i1 = output_CC_flattened_indices - i0 * correlation_size  # (*). discrete col (W dimension) max indices

    i0_minus1, i0_original, i0_plus1 = detect_edge_cases(i0, correlation_size)
    i1_minus1, i1_original, i1_plus1 = detect_edge_cases(i1, correlation_size)
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * correlation_size
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * correlation_size
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * correlation_size
    output_CC_flattened_indices_i0 = i1 + i0 * correlation_size
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * correlation_size
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * correlation_size
    ### Get Proper Values For Fit: ###
    output_CC = output_CC.contiguous().view(-1, correlation_size * correlation_size)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat(
        [output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat(
        [output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #

    [c_x, b_x, a_x] = fit_polynomial(x_vec, fitting_points_x.squeeze())
    [c_y, b_y, a_y] = fit_polynomial(y_vec, fitting_points_y.squeeze())
    # find the sub-pixel max value and location using the parabola coefficients: #
    # TODO: parabola max isn't correct!!!!!!
    delta_shiftx = -b_x / (2 * a_x)
    delta_shifty = -b_y / (2 * a_y)
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # Substract Center Cross Correlation: ###
    shiftx = shiftx - correlation_size // 2
    shifty = shifty - correlation_size // 2
    return (shifty, shiftx)


# TODO Elisheva changed and made it parallel
def classic_circular_cc_shifts_calc(matrix: Tensor, reference_matrix: Tensor, matrix_fft: Tensor,
                                    reference_tensor_fft: Tensor,
                                    normalize: bool = False, fftshift: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    ## returns tuple of vertical shifts, horizontal shifts, and cross correlation
    cc = circular_cross_correlation_classic(matrix, reference_matrix, matrix_fft, reference_tensor_fft, normalize,
                                            fftshift)
    B, T, C, H, W = matrix.shape
    midpoints = (H // 2, W // 2)
    shifty, shiftx = shifts_from_circular_cc(cc, midpoints)
    return shifty, shiftx, cc


def classic_weighted_circular_cc_shifts_calc(matrix: Tensor,
                                             reference_matrix: Tensor,
                                             weights_tensor: Tensor,
                                             matrix_fft: Tensor,
                                             reference_tensor_fft: Tensor,
                                             normalize: bool = False,
                                             fftshift: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    ## returns tuple of vertical shifts, horizontal shifts, and cross correlation
    cc = weighted_circular_cc_calc(matrix,
                                   reference_matrix,
                                   weights_tensor,
                                   matrix_fft,
                                   reference_tensor_fft)
    B, T, C, H, W = matrix.shape
    midpoints = (H // 2, W // 2)
    shifty, shiftx = shifts_from_circular_cc(cc, midpoints)
    return shifty, shiftx, cc


def normalized_cc_shifts_calc(matrix: Union[torch.Tensor, np.array, tuple, list],
                              reference_matrix: Union[torch.Tensor, np.array, list, tuple],
                              correlation_size: int) -> Tuple[Tensor, Tensor, Tensor]:
    B, T, C, H, W = matrix.shape
    midpoints = (H // 2, W // 2)
    shifts_x = []
    shifts_y = []
    cc = normalized_cc(matrix, reference_matrix, correlation_size)  # Elisheva changed
    for i in range(B):
        # TODO elisheva added this change, formalize into Anvil
        batch_cc = cc[i].squeeze(0)
        shifty, shiftx = shifts_from_normalized_cc(batch_cc, correlation_size)
        shifts_x.append(shiftx)
        shifts_y.append(shifty)
    return construct_tensor(shifts_y), construct_tensor(shifts_x), batch_cc  # torch.stack(normalized_cross_correlations)


def default_weights_weighted_cc(matrix: Tensor) -> Tensor:
    # matrix is b,t,c,h,w
    canny_edge_detection_layer = canny_edge_detection(10)
    matrix_mean = matrix.mean(0)  # Assuming [T,C,H,W]
    _, _, _, _, _, weights = canny_edge_detection_layer.forward(BW2RGB(matrix_mean.unsqueeze(0)), 3)
    weights = center_logical_mask(weights, zero_edges=3) * weights  # zeros 3 pixels from edge
    return weights


def format_weighted_ccc_params(matrix: Union[torch.Tensor, np.array, tuple, list],
                               reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                               weights: Union[torch.Tensor, np.array, tuple, list],
                               matrix_fft: Tensor, reference_matrix_fft: Tensor):
    type_check_parameters([(weights, (torch.Tensor, np.array, tuple, list))])
    weights = construct_tensor(weights)
    matrix, reference_matrix, matrix_fft, reference_matrix_fft, _, _ = \
        format_parameters_classic_circular_cc(matrix, reference_matrix, matrix_fft, reference_matrix_fft, False, False)
    raise_if_not(multiplicable_tensors([weights, reference_matrix, matrix]),
                 message="Matrices are not of equivalent sizes")
    return matrix, reference_matrix, matrix_fft, reference_matrix_fft, weights


def weighted_circular_cc_calc(matrix: Tensor, reference_matrix: Tensor, weights: Tensor,
                              matrix_fft: Tensor, reference_matrix_fft: Tensor) -> Tensor:
    weights = weights / weights.sum()
    weighted_reference_mean = (weights * reference_matrix).sum([-1, -2],
                                                               True)  # since weights_tensor is normalized this is a weighted mean
    mean_subtracted_reference = reference_matrix - weighted_reference_mean
    subtracted_weighted_reference = mean_subtracted_reference * weights
    subtracted_weighted_reference_sum = subtracted_weighted_reference.sum(dims=[-2, -1])
    weighted_cross_correlation = circular_cross_correlation_classic(matrix, reference_matrix, matrix_fft,
                                                                    reference_matrix_fft, fftshift=True)
    processed_reference_cc = circular_cross_correlation_classic(matrix, subtracted_weighted_reference, matrix_fft, None,
                                                                fftshift=True)
    W_cov_XY = processed_reference_cc - weighted_cross_correlation * subtracted_weighted_reference_sum
    W_cov_XX = circular_cross_correlation_classic(matrix ** 2, weights, fftshift=True) - weighted_cross_correlation ** 2
    W_cov_YY = (mean_subtracted_reference ** 2 + weights).sum([-1, -2], True)
    Denom = torch.sqrt((W_cov_XX * W_cov_YY).clip(0)) + 1e-6  # to make sure it's non zero
    cross_correlation = W_cov_XY / Denom
    cross_correlation = torch.fft.fftshift(cross_correlation, dim=[-1, -2])
    return cross_correlation


# TODO Elisheva added
def align_circular_cc(matrix: Tensor, reference_matrix: Tensor, matrix_fft: Tensor,
                      reference_tensor_fft: Tensor,
                      normalize: bool=False,
                      crop_warped_matrix: bool=False,
                      warp_method: str='bilinear',
                      fftshift: bool=False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # calculate shifts
    shifts_h, shifts_w, cc = classic_circular_cc_shifts_calc(matrix, reference_matrix, matrix_fft, reference_tensor_fft, normalize, fftshift)
    # warp matrix
    warped_matrix = shift_matrix_subpixel(matrix, -shifts_h, -shifts_w, matrix_FFT=None, warp_method=warp_method)

    if crop_warped_matrix:
        B, T, C, H, W = warped_matrix.shape
        max_shift_h = shifts_h.max()
        max_shift_w = shifts_w.max()
        new_h = H - (max_shift_h.abs().int().item() * 2 + 5)  # TODO make sure no need to also add .cpu().numpy() instead of .item()
        new_w = W - (max_shift_w.abs().int().item() * 2 + 5)  # TODO also make safety margins not rigid (5 is arbitrary)
        warped_matrix = non_interpolated_center_crop(warped_matrix, new_h, new_w)

    return warped_matrix, shifts_h, shifts_w, cc


def align_weighted_circular_cc(matrix: Tensor,
                               reference_matrix: Tensor,
                               matrix_fft: Tensor,
                               reference_tensor_fft: Tensor,
                               weights_tensor: Tensor,
                               normalize: bool=False,
                               crop_warped_matrix: bool=False,
                               warp_method: str='bilinear',
                               fftshift: bool=False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # calculate shifts
    shifts_h, shifts_w, cc = classic_circular_cc_shifts_calc(matrix, reference_matrix, matrix_fft, reference_tensor_fft, normalize, fftshift)
    # warp matrix
    warped_matrix = shift_matrix_subpixel(matrix, -shifts_h, -shifts_w, matrix_FFT=None, warp_method=warp_method)

    if crop_warped_matrix:
        B, T, C, H, W = warped_matrix.shape
        max_shift_h = shifts_h.max()
        max_shift_w = shifts_w.max()
        new_h = H - (max_shift_h.abs().int().item() * 2 + 5)  # TODO make sure no need to also add .cpu().numpy() instead of .item()
        new_w = W - (max_shift_w.abs().int().item() * 2 + 5)  # TODO also make safety margins not rigid (5 is arbitrary)
        warped_matrix = non_interpolated_center_crop(warped_matrix, new_h, new_w)

    return warped_matrix, shifts_h, shifts_w, cc


def format_align_to_reference_frame_circular_cc_params(matrix: Union[Tensor, np.array, tuple, list],
                                                       reference_matrix: Union[Tensor, np.array, tuple, list],
                                                       matrix_fft: torch.Tensor=None,
                                                       reference_fft: torch.Tensor=None,
                                                       normalize: bool=False,
                                                       warp_method: str='bilinear',
                                                       crop_warped_matrix: bool=False,
                                                       fftshift: bool=False) -> Tuple[
    Tensor, Tensor, Union[Tensor, None], Union[Tensor, None], bool, str, bool, bool]:
    matrix, reference_matrix, matrix_fft, reference_fft, normalize, fftshift = format_parameters_classic_circular_cc(matrix, reference_matrix, matrix_fft,
                                                                                                                     reference_fft, normalize, fftshift)
    type_check_parameters([(warp_method, str), (crop_warped_matrix, bool)])
    validate_warp_method(warp_method, valid_methods=['bilinear', 'bicubic', 'nearest', 'fft'])

    return matrix, reference_matrix, matrix_fft, reference_fft, normalize, warp_method, crop_warped_matrix, fftshift


# TODO Elisheva added
def format_align_to_center_frame_circular_cc_params(matrix: Union[Tensor, np.array, tuple, list],
                                                    matrix_fft: Union[torch.Tensor, np.array, tuple, list] = None,
                                                    normalize: bool = False,
                                                    warp_method: str = 'bilinear',
                                                    crop_warped_matrix: bool = False) -> Tuple[Tensor, Tensor, bool, str, bool]:
    type_check_parameters([(matrix, (Tensor, tuple, list, np.ndarray)), (matrix_fft, (Tensor, type(None))),
                           (normalize, bool), (warp_method, str), (crop_warped_matrix, bool)])
    matrix = construct_tensor(matrix)
    if matrix_fft is not None:  # is not None
        raise_if(compare_unequal_dimensionality_tensors(matrix, matrix_fft),
                 message="Matrix and Matrix FFT are not the same size")

    return matrix, matrix_fft, normalize, warp_method, crop_warped_matrix
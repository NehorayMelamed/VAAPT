import os.path
from math import floor, ceil
from typing import Tuple, Union

import kornia
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch import Tensor

from RapidBase.Utils.Classical_DSP.Noise_Estimation import estimate_noise_in_image_EMVA


def center_crop(mat: Tensor, crop_size: Tuple[int, int]) -> Tensor:
    H = mat.shape[-2]
    W = mat.shape[-1]
    h_diff = max(mat.shape[-2] - crop_size[0], 0)//2
    w_diff = max(mat.shape[-1] - crop_size[1], 0)//2
    if len(mat.shape) == 2:
        return mat[h_diff:H-h_diff, w_diff:W-w_diff]
    elif len(mat.shape) == 3:
        return mat[:, h_diff:H-h_diff, w_diff:W-w_diff]
    elif len(mat.shape) == 4:
        return mat[:, :, h_diff:H-h_diff, w_diff:W-w_diff]
    elif len(mat.shape) == 5:
        return mat[:, :, :, h_diff:H-h_diff, w_diff:W-w_diff]
    else:
        raise RuntimeError("Too many dims, or not enough dims")


def find_outliers(aligned_tensor: Tensor,
                  bg_estimation: Tensor,
                  noise_mult_factor: int = 20,
                  outliers_threshold: int = 20,
                  estimate_outliers_threshold: bool = False,
                  apply_canny: bool = True,
                  canny_low_threshold: float = 0.05 * 255,
                  canny_high_threshold: float = 0.2 * 255,
                  dilate_canny: bool = True,
                  dilation_kernel: Tensor = torch.ones(3, 3)):
    # aligned tensor and bg estimation assumed to be of equal sizes
    while len(bg_estimation.shape) < 4:
        bg_estimation = bg_estimation.unsqueeze(0)
    # calculate outliers without canny edge detection
    diff_tensor = (aligned_tensor - bg_estimation).abs()

    # find outliers threshold
    if estimate_outliers_threshold:
        noise_map = estimate_noise_in_image_EMVA(aligned_tensor)
        outliers_threshold = noise_map * noise_mult_factor
        del noise_map
    del aligned_tensor

    outliers_tensor = (diff_tensor > outliers_threshold).float()

    # release memory
    del diff_tensor
    torch.cuda.empty_cache()

    # If needed, calculate canny
    if apply_canny:
        # TODO split to avoid cuda out of memory, undo if enough memory is available
        _, bg_canny_threshold = kornia.filters.canny(bg_estimation, canny_low_threshold, canny_high_threshold)
        if dilate_canny:
            bg_canny_threshold = kornia.morphology.dilation(bg_canny_threshold, dilation_kernel.to(bg_canny_threshold.device))
        outliers_tensor_after_canny = (outliers_tensor - bg_canny_threshold).clamp(0)

        return outliers_tensor_after_canny

    return outliers_tensor


def read_bin_images_batch(filenames, height, width):
    images = []
    for img_file in filenames:
        with open(img_file, 'rb') as current_img:
            current_arr = np.fromfile(current_img, dtype=np.uint8).reshape(height, width)
            images.append(current_arr)
    video_arr = np.stack(images)
    video_tensor = torch.from_numpy(video_arr)
    return video_tensor


def pad_numpy_batch(input_arr: np.array, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(a, pad_size):  # create a rigid shape for padding in dims HWC
            pad_start = np.floor(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            return (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (0, 0)

        return np.array([np.pad(a, pad_shape(a, pad_size), 'constant', constant_values=0) for a in input_arr])
    else:
        return None


def pad_torch_batch(input_tensor: Tensor, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(t, pad_size):  # create a rigid shape for padding in dims CHW
            pad_start = np.floor(np.subtract(pad_size, t.shape[-2:]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, t.shape[-2:]) / 2).astype(int)
            return pad_start[1], pad_end[1], pad_start[0], pad_end[0]

        return torch.nn.functional.pad(input_tensor, pad_shape(input_tensor, pad_size), mode='constant')
    else:
        return None


def read_bin_video_to_numpy(bin_path, H, W, C, dtype=np.float32):
    with open(bin_path, 'rb') as f:
        numpy_data = np.fromfile(f, dtype=dtype)
    video_arr = np.reshape(numpy_data, (-1, H, W, C))
    return video_arr


def crop_size_after_homography(h_matrix: Tensor, H: int, W: int, add_just_in_case: int = 5):
    """
    Returns crop sizes for a transformed matrix after applying homography to it
    Args:
        h_matrix: homography matrix to apply (enables stacking)
        H: matrix's H
        W: matrix's W
        add_just_in_case: additive value to the crp in each margin, just in case :)

    Returns: crop_h_start, crop_h_end, crop_w_start, crop_w_end

    """

    # Generate 4 point locations close to the 4 corners of the image (homogeneous coordinates)
    o_o = torch.tensor([0, 0, 1]).float().to(h_matrix.device)
    o_w = torch.Tensor([W, 0, 1]).float().to(h_matrix.device)
    h_o = torch.Tensor([0, H, 1]).float().to(h_matrix.device)
    h_w = torch.Tensor([W, H, 1]).float().to(h_matrix.device)

    # find shifted points according to the homography matrix (homogeneous coordinates)
    shifted_o_o = h_matrix @ o_o
    shifted_o_w = h_matrix @ o_w
    shifted_h_o = h_matrix @ h_o
    shifted_h_w = h_matrix @ h_w

    # transfer original points to non-homogeneous coordinates
    o_o = o_o[..., :2]
    o_w = o_w[..., :2]
    h_o = h_o[..., :2]
    h_w = h_w[..., :2]

    # transfer to non homogeneous coordinates
    shifted_o_o = shifted_o_o[..., :2] / shifted_o_o[..., 2:]
    shifted_o_w = shifted_o_w[..., :2] / shifted_o_w[..., 2:]
    shifted_h_o = shifted_h_o[..., :2] / shifted_h_o[..., 2:]
    shifted_h_w = shifted_h_w[..., :2] / shifted_h_w[..., 2:]

    # crops of each side of the image according to the expected shifts
    h_start_1 = (o_o[..., 1] - shifted_o_o[..., 1]).clamp(0)
    h_start_2 = (o_w[..., 1] - shifted_o_w[..., 1]).clamp(0)
    h_start = torch.ceil(torch.max(h_start_1, h_start_2) + add_just_in_case).int()

    h_end_1 = H - (shifted_h_o[..., 1] - h_o[..., 1]).clamp(0)
    h_end_2 = H - (shifted_h_w[..., 1] - h_w[..., 1]).clamp(0)
    h_end = torch.ceil(torch.min(h_end_1, h_end_2) - add_just_in_case).int()

    w_start_1 = (o_o[..., 0] - shifted_o_o[..., 0]).clamp(0)
    w_start_2 = (h_o[..., 0] - shifted_h_o[..., 0]).clamp(0)
    w_start = torch.ceil(torch.max(w_start_1, w_start_2) + add_just_in_case).int()

    w_end_1 = W - (shifted_o_w[..., 0] - o_w[..., 0]).clamp(0)
    w_end_2 = W - (shifted_h_w[..., 0] - h_w[..., 0]).clamp(0)
    w_end = torch.ceil(torch.min(w_end_1, w_end_2) - add_just_in_case).int()

    return h_start, h_end, w_start, w_end


def estimate_difference_between_homography_matrices(h_matrix_1: Tensor,
                                                    h_matrix_2: Tensor,
                                                    H: int = 1000,
                                                    W: int = 1000):
    """
    Estimates the difference between two homography matrices by checking the difference in the movement in the image's
    corners.
    This function takes H, W as well to enable determining the desired accuracy (max difference allowed will be the
    algorithm's accuracy).
    Also enables default values for H, W to allow a relative measure that doesn't have a meaningful absolute meaning.

    Args:
        h_matrix_1: homography matrix 1
        h_matrix_2: homography matrix 2
        H: target image's height
        W: target image's width

    Returns: maximal shift between pixels after applying each of the homography matrices.
    """
    # validate devices
    device = h_matrix_1.device
    if h_matrix_2.device != device:
        h_matrix_2.device = device

    # Generate 4 point locations close to the 4 corners of the image (homogeneous coordinates)
    o_o = torch.tensor([0, 0, 1]).float().to(device)
    o_w = torch.Tensor([W, 0, 1]).float().to(device)
    h_o = torch.Tensor([0, H, 1]).float().to(device)
    h_w = torch.Tensor([W, H, 1]).float().to(device)

    # find shifted points according to the homography matrices (homogeneous coordinates)
    shifted_o_o_1 = h_matrix_1 @ o_o
    shifted_o_w_1 = h_matrix_1 @ o_w
    shifted_h_o_1 = h_matrix_1 @ h_o
    shifted_h_w_1 = h_matrix_1 @ h_w

    shifted_o_o_2 = h_matrix_2 @ o_o
    shifted_o_w_2 = h_matrix_2 @ o_w
    shifted_h_o_2 = h_matrix_2 @ h_o
    shifted_h_w_2 = h_matrix_2 @ h_w

    # transfer original points to non-homogeneous coordinates
    o_o = o_o[..., :2]
    o_w = o_w[..., :2]
    h_o = h_o[..., :2]
    h_w = h_w[..., :2]

    # transfer shifted points to non-homogeneous coordinates
    shifted_o_o_1 = shifted_o_o_1[..., :2] / shifted_o_o_1[..., 2:]
    shifted_o_w_1 = shifted_o_w_1[..., :2] / shifted_o_w_1[..., 2:]
    shifted_h_o_1 = shifted_h_o_1[..., :2] / shifted_h_o_1[..., 2:]
    shifted_h_w_1 = shifted_h_w_1[..., :2] / shifted_h_w_1[..., 2:]

    shifted_o_o_2 = shifted_o_o_2[..., :2] / shifted_o_o_2[..., 2:]
    shifted_o_w_2 = shifted_o_w_2[..., :2] / shifted_o_w_2[..., 2:]
    shifted_h_o_2 = shifted_h_o_2[..., :2] / shifted_h_o_2[..., 2:]
    shifted_h_w_2 = shifted_h_w_2[..., :2] / shifted_h_w_2[..., 2:]

    # find maximal differences
    difference_o_o = (shifted_o_o_1 - shifted_o_o_2).abs()
    difference_o_w = (shifted_o_w_1 - shifted_o_w_2).abs()
    difference_h_o = (shifted_h_o_1 - shifted_h_o_2).abs()
    difference_h_w = (shifted_h_w_1 - shifted_h_w_2).abs()

    # create measurement
    max_difference = max(*difference_o_o, *difference_o_w, *difference_h_o, *difference_h_w)

    return max_difference

def interpolate_by_factor_1d(data_points: Union[Tensor, np.array], factor: int, kind='cubic'):
    """
    Interpolates data points to a new size determined by factor
    Note that new size is (old_size - 1) * factor + 1
    Example: old_size = 4, factor = 3
        data points:     *     *     *     *
        new data points: * - - * - - * - - *
        new_size = (old_size - 1) * factor + 1 = (4 - 1) * 3 + 1 = 10
    Args:
        data_points:
        factor:
        kind:

    Returns:

    """
    if isinstance(data_points, Tensor):
        device = data_points.device
        data_points = data_points.cpu()

    # interpolate points to function
    y = data_points
    x = torch.linspace(0, y.shape[-1] - 1, y.shape[-1])
    f = interp1d(x, y, kind=kind, axis=-1)

    # generate new data points
    new_x = torch.linspace(0, y.shape[-1] - 1, (y.shape[-1] - 1) * factor + 1)
    new_y = f(new_x)

    new_data_points = new_y
    if isinstance(data_points, Tensor):
        new_data_points = Tensor(new_data_points).to(device)

    return new_data_points


def interpolate_h_matrices_to_factor(h_matrices, factor, interpolation_kind='cubic'):
    T = h_matrices.shape[0]
    # reshape to enable interpolation:
    h_matrices_flattened = torch.stack([h_matrices[:, int(i / 3), i % 3] for i in range(8)])

    # interpolate
    h_matrices_flattened_interpolated = interpolate_by_factor_1d(h_matrices_flattened, factor, interpolation_kind)

    # reshape to matrices
    new_size = h_matrices_flattened_interpolated.shape[-1]
    h_matrices_interpolated = torch.ones((new_size, 3, 3))
    for i in range(8):
        h_matrices_interpolated[:, int(i / 3), i % 3] = h_matrices_flattened_interpolated[i]

    return h_matrices_interpolated

    # TODO temp
    # x = torch.arange(0, h_matrices.shape[0])
    # y = h_matrices_flattened[0].cpu()
    # new_x = torch.linspace(0, h_matrices.shape[0] - 1, new_size).cpu()
    # new_y = h_matrices_flattened_interpolated[0].cpu()
    # plot_torch(x, y)
    # plot_torch(new_x, new_y)
    # plt.plot(new_x.cpu().numpy(), new_y.cpu().numpy(), 'go', x.cpu().numpy(), y.cpu().numpy(), 'ro')

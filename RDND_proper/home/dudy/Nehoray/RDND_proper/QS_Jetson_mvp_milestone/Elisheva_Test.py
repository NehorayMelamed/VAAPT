# ------- (1): Create sterilized video to align -------
import torch
from torch import Tensor
from typing import Union

from RapidBase.import_all import *
from RapidBase.Anvil._alignments.minSAD_alignments import interpolated_minSAD_defaults, format_min_SAD_params, non_edge_shift_index, \
    sub_pixel_shifts
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not
from RapidBase.Anvil._internal_utils.torch_utils import MatrixOrigami, construct_tensor
from RapidBase.Anvil._transforms.rotate_matrix import structure_thetas, fftshifted_formatted_vectors
from RapidBase.Anvil.alignments import circular_cc_shifts, normalized_cc_shifts, min_SAD_detect_shifts
from RapidBase.Anvil import shift_matrix_subpixel, rotate_matrix


# base_image_path = r'/home/elisheva/Projects/IMOD general/data/DIV2K/DIV2K_train_HR/0016.png'
# save_path_folder = r'/home/elisheva/Projects/IMOD general/data/temp palantir trials'
#
# video_len = 5
# shifts_h = [1, 2, 3, 4, 5]
# shifts_w = [5, 4, 3, 2, 1]
# edge_crop_size = 10
#
# image_arr = np.asarray(Image.open(base_image_path))
# image_tensor = torch.from_numpy(image_arr).permute([2, 0, 1]).unsqueeze(0)
# video_base_tensor = torch.cat([image_tensor] * video_len, dim=0)
#
# shifted_video = shift_matrix_subpixel(matrix=video_base_tensor,
#                                       shift_H=shifts_h,
#                                       shift_W=shifts_w,
#                                       warp_method='fft')
#
# cropped_shifted_video = shifted_video[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]
# cropped_image = image_tensor[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]
# cropped_base_video = video_base_tensor[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]
#
# shifted_video_np = np.asarray(cropped_shifted_video.permute([0, 2, 3, 1]))
#
# # Save clip to folder
# shifted_video_path = os.path.join(save_path_folder, 'shifted base')
# for fidx, frame in enumerate(shifted_video_np):
#     frame_path = os.path.join(shifted_video_path, f'{string_rjust(fidx, 2)}.jpg')
#     cv2.imwrite(filename=frame_path, img=frame)


# ------- (2) test different alignment methods -------
# (2.1) circular cross correlation
def test_circular_cc():
    """
    Works really not too bad! example results:
        gt shifts h: [1, 2, 3, 4, 5]
        rt shifts h: tensor([[0.9880, 1.9880, 2.9882, 3.9881, 4.9879]])
        gt shifts w: [5, 4, 3, 2, 1]
        rt shifts w: tensor([[4.9742, 3.9744, 2.9747, 1.9748, 0.9749]])
    :return:
    """
    shifts_y, shifts_x, cc = circular_cc_shifts(matrix=cropped_shifted_video,
                                                reference_matrix=cropped_base_video)

    print(f'gt shifts h: {shifts_h}\nrt shifts h: {shifts_y}\ngt shifts w: {shifts_w}\nrt shifts w: {shifts_x}')


# TODO note - reference matrix should be a frame when matrix could be a video. Shouldn't have to match on T dimension.
#  also there's an unsolved issue with B dimension - should it be greater than 1 in any case?

# (2.2) normalized cross correlation
def test_normalized_cc():
    """
    Test results - super exact but slow as hell:
        gt shifts h: [1, 2, 3, 4, 5]
        rt shifts h: tensor([[0.9996, 1.9996, 2.9996, 3.9996, 4.9996]])
        gt shifts w: [5, 4, 3, 2, 1]
        rt shifts w: tensor([[4.9998, 3.9998, 2.9998, 1.9997, 0.9997]])
    :return:
    """
    estimated_max_overall_movement = int(2 * max(shifts_h + shifts_w) + 2)
    shifts_y, shifts_x, cc = normalized_cc_shifts(matrix=cropped_shifted_video,
                                                  reference_matrix=cropped_base_video,
                                                  correlation_size=estimated_max_overall_movement)

    print(f'gt shifts h: {shifts_h}\nrt shifts h: {shifts_y}\ngt shifts w: {shifts_w}\nrt shifts w: {shifts_x}')


# (2.3) minSAD
def tensor_center_crop(img: torch.Tensor, crop_height: int, crop_width: int) -> torch.Tensor:
    height, width = img.shape[-2:]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[..., y1:y2, x1:x2]
    return img


def rotate_without_expand_fft(matrix: Tensor, thetas: Tensor):
    B, T, C, H, W = matrix.shape
    thetas = structure_thetas(matrix, B, thetas)

    ### Choose Rotation Angle and assign variables needed for later: % % %
    a = torch.tan(thetas / 2)
    b = -torch.sin(thetas)
    Nx_vec, Ny_vec = fftshifted_formatted_vectors(matrix, H, W)

    ### Prepare Matrices to avoid looping: ###
    Nx_vec_mat = torch.repeat_interleave(Nx_vec, H, -2)
    Ny_vec_mat = torch.repeat_interleave(Ny_vec, W, -1)
    # ISSUE: BTCHW
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
        torch.fft.ifftn(torch.fft.fftn(matrix, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    Iy_parallel = (
        torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(Ny_vec_mat_final * b * -1).conj(),
                        dim=[-2])).real
    input_mat_rotated = (
        torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    return input_mat_rotated


def plot_tensor_img(frame):
    data = floor(np.array(frame.permute(1, 2, 0))).astype(int)
    from matplotlib import pyplot as plt
    plt.imshow(data, interpolation='nearest')
    plt.show()


class minSAD_transforms_elisheva(nn.Module):
    def __init__(self, B, T, H, W, device='cpu'):
        super().__init__()
        self.B = B
        self.T = T
        self.H = H
        self.W = W
        self.device = device

        # Calculate meshgrid to save future computation time
        self._generate_meshgrid()

    def _generate_meshgrid(self):
        # Create meshgrid:
        meshgrid_H, meshgrid_W = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        # Turn meshgrid to be tensors:
        meshgrid_W = meshgrid_W.type(torch.FloatTensor).to(self.device)
        meshgrid_H = meshgrid_H.type(torch.FloatTensor).to(self.device)

        # Normalize meshgrid
        meshgrid_W_max = meshgrid_W[-1, -1]
        meshgrid_W_min = meshgrid_W[0, 0]
        meshgrid_H_max = meshgrid_H[-1, -1]
        meshgrid_H_min = meshgrid_H[0, 0]
        meshgrid_W_centered = meshgrid_W - (meshgrid_W_max - meshgrid_W_min) / 2
        meshgrid_H_centered = meshgrid_H - (meshgrid_H_max - meshgrid_H_min) / 2
        meshgrid_W = meshgrid_W_centered
        meshgrid_H = meshgrid_H_centered

        # stack T*B times
        meshgrid_W = torch.stack([meshgrid_W] * (self.B * self.T), 0)
        meshgrid_H = torch.stack([meshgrid_H] * (self.B * self.T), 0)

        self.meshgrid_W = meshgrid_W
        self.meshgrid_H = meshgrid_H

    def _format_min_SAD_params(self,
                               matrix: Union[Tensor, tuple, list, np.array],
                               reference_matrix: Union[Tensor, tuple, list, np.array],
                               shifty_vec: Union[Tensor, tuple, list] = None,
                               shiftx_vec: Union[Tensor, tuple, list] = None,
                               rotation_vec: Union[Tensor, tuple, list] = None,
                               scale_vec: Union[Tensor, tuple, list] = None,
                               warp_method: str = 'bilinear') -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str]:
        # Using Anvil's wonderful validation first
        matrix, reference_matrix, shifty_vec, shiftx_vec, rotation_vec, scale_vec, warp_method = format_min_SAD_params(
            matrix,
            reference_matrix,
            shifty_vec,
            shiftx_vec,
            rotation_vec,
            scale_vec,
            warp_method)
        raise_if_not(matrix.shape[-2] == self.H and matrix.shape[-1] == self.W,
                     message=f"bad matrix dimensionality - expected H, W to be {self.H}, {self.W}")
        return matrix, reference_matrix, shifty_vec, shiftx_vec, rotation_vec, scale_vec, warp_method

    def _affine_transform_interpolated(self, matrix: Tensor, shift_h: Union[int, float], shift_w: Union[int, float],
                                       rotation_angle: Union[int, float], scale: Union[int, float],
                                       warp_method: str = 'bilinear') -> Tensor:
        B, T, C, H, W = matrix.shape
        # phase 1: construct the transform based on the precalculated meshgrid
        # expand the meshgrid from 1 to B*T
        # affine_W = torch.stack([self.meshgrid_W] * (B * T), 0)
        # affine_H = torch.stack([self.meshgrid_H] * (B * T), 0)
        affine_W = self.meshgrid_W
        affine_H = self.meshgrid_H

        # add shifts:
        shift_h_tensor = Tensor([shift_h]).unsqueeze(1).unsqueeze(2).to(matrix.device)
        shift_w_tensor = Tensor([shift_w]).unsqueeze(1).unsqueeze(2).to(matrix.device)
        affine_W = affine_W + shift_w_tensor * 1
        affine_H = affine_H + shift_h_tensor * 1

        # add scale:
        scale_tensor = Tensor([scale]).unsqueeze(1).unsqueeze(2).to(matrix.device)
        affine_W *= 1 / scale_tensor
        affine_H *= 1 / scale_tensor

        # add rotation
        rotation_tensor = Tensor([rotation_angle]).unsqueeze(1).unsqueeze(2).to(matrix.device)
        affine_W_rotated = torch.cos(rotation_tensor) * affine_W + torch.sin(rotation_tensor) * affine_H
        affine_H_rotated = -torch.sin(rotation_tensor) * affine_W + torch.cos(rotation_tensor) * affine_H
        affine_W = affine_W_rotated
        affine_H = affine_H_rotated

        # Normalize meshgrid
        affine_W = affine_W / ((self.W - 1) / 2)
        affine_H = affine_H / ((self.H - 1) / 2)

        # warp matrix
        affine_grid = torch.stack((affine_W, affine_H), 3)
        reshaped_matrix = matrix.reshape((B * T, C, H, W))  # grid sample can only handle 4D
        output_tensor = torch.nn.functional.grid_sample(reshaped_matrix, affine_grid, mode=warp_method)

        return output_tensor.reshape((B, T, C, H, W))
    # plot_tensor_img(output_tensor[0])

    def _interpolated_minSAD_shifts(self,
                                    matrix: torch.Tensor,
                                    reference_matrix: torch.Tensor,
                                    shift_h_vec: torch.Tensor = None,
                                    shift_w_vec: torch.Tensor = None,
                                    rotation_vec: torch.Tensor = None,
                                    scale_vec: torch.Tensor = None,
                                    warp_method: str = 'bilinear') -> Tuple[float, float, float, float, Tensor]:
        shift_h_vec, shift_w_vec, rotation_vec, scale_vec = interpolated_minSAD_defaults(shift_h_vec, shift_w_vec,
                                                                                         rotation_vec, scale_vec,
                                                                                         matrix.device)
        B, T, C, H, W = matrix.shape
        rotation_vec_rads = rotation_vec * torch.pi / 180

        # crop size to ensure no invalid subtracted values. Do validate the math here
        crop_h_size = int(H - (2 * (max(shift_h_vec.abs()) + (0.5 * H * tan(max(rotation_vec_rads)).abs())) * max(scale_vec) + 10))  # 10 just extra
        crop_w_size = int(W - (2 * (max(shift_w_vec.abs()) + (0.5 * W * tan(max(rotation_vec_rads)).abs())) * max(scale_vec) + 10))
        # crop_h_size = crop_h_size.item()
        # crop_w_size = crop_w_size.item()

        # make a table for all possible transforms for every frame in B, T
        SAD_dimensions = (max(shift_h_vec.shape), max(shift_w_vec.shape), max(rotation_vec_rads.shape), max(scale_vec.shape))
        SAD_list = []

        # calculate min SAD
        for shift_h_idx, current_shift_h in enumerate(shift_h_vec):
            for shift_w_idx, current_shift_w in enumerate(shift_w_vec):
                for rotation_idx, current_rotation_angle in enumerate(rotation_vec_rads):
                    for scale_idx, current_scale in enumerate(scale_vec):
                        # warp matrix
                        warped_matrix = self._affine_transform_interpolated(matrix, current_shift_h, current_shift_w,
                                                                            current_rotation_angle, current_scale,
                                                                            warp_method=warp_method)
                        # center crop to ensure no edge error occurs
                        cropped_warped_matrix = tensor_center_crop(warped_matrix, crop_h_size, crop_w_size)
                        cropped_reference_matrix = tensor_center_crop(reference_matrix, crop_h_size, crop_w_size)

                        # calculate min SAD
                        current_SAD = (cropped_warped_matrix - cropped_reference_matrix).abs()
                        current_SAD_mean = current_SAD.mean(-1, True).mean(-2, True).mean(-3, True)
                        current_SAD_mean = current_SAD_mean.squeeze(-1).squeeze(-1).squeeze(-1).reshape((B * T))  #TODO: reshape actually shifts memory around, use .view

                        # add to the SAD mapping table
                        SAD_list.append(current_SAD_mean)

        # find minimum for every frame
        SAD_matrix = torch.stack(SAD_list, 0)
        min_SAD = torch.argmin(SAD_matrix, dim=0)
        SAD_matrix_2_reshaped = torch.reshape(SAD_matrix.squeeze(), (B*T, *SAD_dimensions))
        min_transforms_indicies = unravel_index(min_SAD, SAD_dimensions)
        shift_h_indices = Tensor([non_edge_shift_index(i, warp_vec=shift_h_vec) for i in min_transforms_indicies[0]]).to(torch.long)
        shift_w_indices = Tensor([non_edge_shift_index(i, warp_vec=shift_w_vec) for i in min_transforms_indicies[1]]).to(torch.long)
        rotation_indices = Tensor([non_edge_shift_index(i, warp_vec=rotation_vec) for i in min_transforms_indicies[2]]).to(torch.long)
        scale_indices = Tensor([non_edge_shift_index(i, warp_vec=scale_vec) for i in min_transforms_indicies[3]]).to(torch.long)
        # shift_h_min = shift_h_vec[min_transforms_indicies[0]]
        # shift_w_min = shift_w_vec[min_transforms_indicies[1]]
        # rotation_min = rotation_vec[min_transforms_indicies[2]]
        # scale_min = scale_vec[min_transforms_indicies[3]]

        # shift_h_min = shift_h_min.reshape((B, T))
        # shift_w_min = shift_w_min.reshape((B, T))
        # rotation_min = rotation_min.reshape((B, T))
        # scale_min = scale_min.reshape((B, T))

        # Do parabola fit to find exact shifts - TODO polish this
        if len(shift_h_vec) == 1:
            shift_h_sub_pixel = float(shift_h_vec)
        else:
            shift_h_indices_range = torch.stack([shift_h_indices - 1, shift_h_indices, shift_h_indices + 1], dim=0).to(torch.long)
            shift_h_sub_pixel = sub_pixel_shifts(shift_h_vec, shift_h_indices, SAD_matrix_2_reshaped[
                shift_h_indices_range, shift_w_indices, rotation_indices, scale_indices])

        if len(shift_w_vec) == 1:
            shift_w_sub_pixel = float(shift_w_vec[0])
        else:
            shift_w_indices_range = torch.stack([shift_w_indices - 1, shift_w_indices, shift_w_indices + 1], dim=0).to(torch.long)
            shift_w_sub_pixel = sub_pixel_shifts(shift_w_vec, shift_w_indices, SAD_matrix_2_reshaped[
                shift_h_indices, shift_w_indices_range, rotation_indices, scale_indices])

        if len(rotation_vec) == 1:
            rotation_sub_pixel = float(rotation_vec)
        else:
            rotation_indices_range = torch.stack([rotation_indices - 1, rotation_indices, rotation_indices + 1], dim=0).to(torch.long)
            rotation_sub_pixel = sub_pixel_shifts(rotation_vec, rotation_indices, SAD_matrix_2_reshaped[
                shift_w_indices, shift_h_indices, rotation_indices_range, scale_indices])

        if len(scale_vec) == 1:
            scale_sub_pixel = float(scale_vec[0])
        else:
            scale_indices_range = torch.stack([scale_indices - 1, scale_indices, scale_indices + 1], dim=0).to(torch.long)
            scale_sub_pixel = sub_pixel_shifts(scale_vec, scale_indices, SAD_matrix_2_reshaped[
                shift_w_indices, shift_h_indices, rotation_indices, scale_indices_range])

        shift_h_sub_pixel = shift_h_sub_pixel.reshape((B, T), dtype=float)
        shift_w_sub_pixel = shift_w_sub_pixel.reshape((B, T), dtype=float)
        rotation_sub_pixel = rotation_sub_pixel.reshape((B, T), dtype=float)
        scale_sub_pixel = scale_sub_pixel.reshape((B, T), dtype=float)

        return shift_h_sub_pixel, shift_w_sub_pixel, rotation_sub_pixel, scale_sub_pixel, SAD_matrix


        # TODO add parabola fit
        # return shift_h_min, shift_w_min, rotation_min, scale_min, SAD_tensor

    def forward(self,
                matrix: Union[torch.Tensor, np.array, tuple, list],
                reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                shifty_vec: Union[Tensor, tuple, list] = None,
                shiftx_vec: Union[Tensor, tuple, list] = None,
                rotation_vec: Union[Tensor, tuple, list] = None,
                scale_vec: Union[Tensor, tuple, list] = None,
                warp_method: str = 'bilinear') -> Tuple[float, float, float, float, Tensor]:
        # validate and format parameters
        matrix, reference_matrix, shifty_vec, shiftx_vec, rotation_vec, scale_vec, warp_method = \
            self._format_min_SAD_params(matrix, reference_matrix, shifty_vec, shiftx_vec, rotation_vec, scale_vec,
                                        warp_method)
        dimensions_memory = MatrixOrigami(matrix)
        matrix = dimensions_memory.expand_matrix()
        reference_matrix = dimensions_memory.expand_other_matrix(reference_matrix)

        # calculate minSAD
        shift_y, shift_x, rotational_shift, scale_shift, min_sad_matrix = \
            self._interpolated_minSAD_shifts(matrix, reference_matrix, shifty_vec, shiftx_vec, rotation_vec, scale_vec,
                                             warp_method)

        return shift_y, shift_x, rotational_shift, scale_shift, min_sad_matrix


def test_minSAD():
    video_len = 5
    shifts_h = [0.3, 0.7, 0.9, 1.2, 0.9]
    shifts_w = [0.2, -0.1, -0.3, -0.7, -1]
    rotations = [0.4, 0.8, 1.3, 1, 1.4]
    edge_crop_size = 10

    image_arr = np.array(Image.open(base_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).permute([2, 0, 1]).unsqueeze(0)
    image_tensor_downsampled = fast_binning_2D_overap_flexible_AvgPool2d(image_tensor, (2, 2), (0, 0))
    video_base_tensor = torch.cat([image_tensor_downsampled] * video_len, dim=0)

    shifted_video = shift_matrix_subpixel(matrix=video_base_tensor,
                                          shift_H=shifts_h,
                                          shift_W=shifts_w,
                                          warp_method='fft').clamp(0, 255)
    rotated_shifted_video = rotate_without_expand_fft(matrix=shifted_video.unsqueeze(0),
                                                      thetas=(Tensor(rotations) * np.pi / 180)).clamp(0, 255).squeeze(0)

    cropped_shifted_video = rotated_shifted_video[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]
    cropped_base_video = video_base_tensor[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]

    T, C, H, W = cropped_shifted_video.shape
    minSAD = minSAD_transforms_elisheva(1, T, H, W)
    shifts_y, shifts_x, rotational_shifts, scale_shifts, min_sad_matrix = minSAD(
        matrix=cropped_shifted_video,
        reference_matrix=cropped_base_video,
        shifty_vec=[-1, 0, 1],
        shiftx_vec=[-1, 0, 1],
        rotation_vec=[-2, -1, 0, 1, 2],
        scale_vec=None,
        warp_method='bilinear')

    print(f'gt shifts h: {shifts_h}\nrt shifts h: {shifts_y}')
    print(f'gt shifts w: {shifts_w}\nrt shifts w: {shifts_x}')
    print(f'gt rotations: {rotations}\nrt rotations: {rotational_shifts}')

# import cProfile
# cProfile.run('test_minSAD()')
# test_minSAD()

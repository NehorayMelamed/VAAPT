import torch

from torch import Tensor
from typing import Union, TypeVar, Optional

from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not, raise_if
from RapidBase.import_all import *
from RapidBase.Anvil._alignments.minSAD_alignments import interpolated_minSAD_defaults, format_min_SAD_params, \
    parabola_fit_1d_over_n_dimensions, format_minSAD_align_parameters
from RapidBase.Anvil._internal_utils.torch_utils import unravel_index_torch, NoMemoryMatrixOrigami, construct_tensor, \
    true_dimensionality, multiplicable_tensors
import kornia as K
import warnings


class minSAD_transforms_layer(nn.Module):
    def __init__(self, B=None, T=None, H=None, W=None,
                 shift_h_vec=torch.tensor([0.]),
                 shift_w_vec=torch.tensor([0.]),
                 rotation_vec=torch.tensor([0.]),
                 scale_vec=torch.tensor([1.]),
                 device='cpu'):
        super().__init__()
        self.B = B
        self.T = T
        self.H = H
        self.W = W
        self.shift_h_vec = shift_h_vec
        self.shift_w_vec = shift_w_vec
        self.rotation_vec = rotation_vec
        self.scale_vec = scale_vec
        self.device = device

        # Calculate meshgrid to save future computation time
        if all([self.B, self.T, self.H, self.W]):
            self._generate_meshgrid()

        # TODO if ever needed to add precalculated affine grids, do it here

    def _generate_meshgrid(self):
        """Generates a meshgrid based on the data sizes B, T, H, W. The generated meshgrid is a property of the minSAD object

        :return:
        """
        # Create meshgrid:
        meshgrid_H, meshgrid_W = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        # Turn meshgrid to be tensors:
        meshgrid_W = meshgrid_W.type(torch.FloatTensor).to(self.device)
        meshgrid_H = meshgrid_H.type(torch.FloatTensor).to(self.device)

        # Normalize meshgrid
        # meshgrid_W_max = meshgrid_W[-1, -1]
        # meshgrid_W_mcdin = meshgrid_W[0, 0]
        # meshgrid_H_max = meshgrid_H[-1, -1]
        # meshgrid_H_min = meshgrid_H[0, 0]
        # meshgrid_W_centered = meshgrid_W - (meshgrid_W_max - meshgrid_W_min) / 2
        # meshgrid_H_centered = meshgrid_H - (meshgrid_H_max - meshgrid_H_min) / 2
        # meshgrid_W = meshgrid_W_centered
        # meshgrid_H = meshgrid_H_centered

        # stack T*B times
        meshgrid_W = torch.stack([meshgrid_W] * (self.B * self.T), 0)
        meshgrid_H = torch.stack([meshgrid_H] * (self.B * self.T), 0)

        self.meshgrid_W = meshgrid_W
        self.meshgrid_H = meshgrid_H

    def _generate_affine_grid(self, shift_h: Union[int, float],
                              shift_w: Union[int, float],
                              rotation_angle: Union[int, float],
                              scale: Union[int, float]) -> Tensor:
        """Generates a single affine grid suited to the data size

        :param current_shift_h:
        :param current_shift_w:
        :param current_rotation_angle:
        :param current_scale:
        :return:
        """
        if (self.meshgrid_H is None) or (self.meshgrid_W is None):
            self._generate_meshgrid()

        affine_W = self.meshgrid_W
        affine_H = self.meshgrid_H

        # add shifts:
        shift_h_tensor = Tensor([shift_h]).unsqueeze(1).unsqueeze(2).to(self.device)
        shift_w_tensor = Tensor([shift_w]).unsqueeze(1).unsqueeze(2).to(self.device)
        affine_W = affine_W + shift_w_tensor * 1
        affine_H = affine_H + shift_h_tensor * 1

        # add scale:
        scale_tensor = Tensor([scale]).unsqueeze(1).unsqueeze(2).to(self.device)
        affine_W *= 1 / scale_tensor
        affine_H *= 1 / scale_tensor

        # add rotation
        # 1. get the center of the grid:
        X0_max = affine_W[0, -1, -1]
        X0_min = affine_W[0, 0, 0]
        Y0_max = affine_H[0, -1, -1]
        Y0_min = affine_H[0, 0, 0]
        # 2. rotate around the center of the grid:
        rotation_tensor = Tensor([rotation_angle]).unsqueeze(1).unsqueeze(2).to(self.device)
        affine_W_rotated = torch.cos(rotation_tensor) * (affine_W - (X0_max - X0_min) / 2) - torch.sin(
            rotation_tensor) * (affine_H - (Y0_max - Y0_min) / 2)
        affine_H_rotated = torch.sin(rotation_tensor) * (affine_W - (X0_max - X0_min) / 2) + torch.cos(
            rotation_tensor) * (affine_H - (Y0_max - Y0_min) / 2)
        affine_W = affine_W_rotated
        affine_H = affine_H_rotated

        # Normalize meshgrid
        affine_W = affine_W / ((self.W - 1) / 2)
        affine_H = affine_H / ((self.H - 1) / 2)

        affine_grid = torch.stack((affine_W, affine_H), 3)

        return affine_grid

    # TODO temp
    def _generate_affine_grid_alternative(self, shift_h: Union[int, float],
                                          shift_w: Union[int, float],
                                          rotation_angle: Union[int, float],
                                          scale: Union[int, float]) -> Tensor:
        translation_tensor = torch.tensor([[shift_w, shift_h]])
        center_tensor = torch.tensor([[self.meshgrid_W.shape[2] / 2, self.meshgrid_W.shape[1] / 2]])
        scale_tensor = torch.tensor([[scale, scale]])
        angle_tensor = torch.tensor([rotation_angle])

        affine_matrix = K.geometry.transform.get_affine_matrix2d(translation_tensor, center_tensor, scale_tensor,
                                                                 angle_tensor)
        # Just trying :)
        affine_matrix = affine_matrix[:, :2].squeeze(0).transpose(1, 0).to(self.device)
        meshgrid = torch.stack([self.meshgrid_W, self.meshgrid_H, torch.ones_like(self.meshgrid_H)], 3)

        affine_grid = torch.matmul(meshgrid, affine_matrix)

        return affine_grid

    def _warp_matrix(self, matrix, affine_grid, warp_method):
        """warp matrix with given affine grid

        :param matrix:
        :param affine_grid:
        :param warp_method:
        :return:
        """
        B, T, C, H, W = matrix.shape
        reshaped_matrix = matrix.reshape((B * T, C, H, W))  # grid sample can only handle 4D
        output_tensor = torch.nn.functional.grid_sample(reshaped_matrix, affine_grid, mode=warp_method)

        return output_tensor.reshape((B, T, C, H, W))

    def _generate_affine_transforms(self):
        """Generates affine grids (input sized) for all the wanted forms of minSAD
        The transforms are saved as object properties in the following format:
        self.transforms[shift_h_idx, shift_w_idx, rotation_idx, scale_idx]
        is the suitable affine grid when the indices refer the transform vectors.
        This function is very optional, not yet used!!!

        :return:
        """
        self.transforms = torch.zeros(len(self.shift_h_vec), len(self.shift_w_vec), len(self.rotation_vec),
                                      len(self.scale_vec), self.B * self.T, self.H, self.W, 2)
        a = len(self.shift_h_vec) * len(self.shift_w_vec) * len(self.rotation_vec) * len(
            self.scale_vec) * self.B * self.T * self.H * self.W * 2
        for shift_h_idx, current_shift_h in enumerate(self.shift_h_vec):
            for shift_w_idx, current_shift_w in enumerate(self.shift_w_vec):
                for rotation_idx, current_rotation_angle in enumerate(self.rotation_vec):
                    for scale_idx, current_scale in enumerate(self.scale_vec):
                        current_affine_grid = self._generate_affine_grid(current_shift_h, current_shift_w,
                                                                         current_rotation_angle, current_scale)
                        self.transforms[shift_h_idx, shift_w_idx, rotation_idx, scale_idx] = current_affine_grid

    def _interpolated_minSAD_shifts(self,
                                    matrix: torch.Tensor,
                                    reference_matrix: torch.Tensor,
                                    shift_h_vec: torch.Tensor = None,
                                    shift_w_vec: torch.Tensor = None,
                                    rotation_vec: torch.Tensor = None,
                                    scale_vec: torch.Tensor = None,
                                    warp_method: str = 'bilinear') -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        shift_h_vec, shift_w_vec, rotation_vec, scale_vec = interpolated_minSAD_defaults(shift_h_vec, shift_w_vec,
                                                                                         rotation_vec, scale_vec,
                                                                                         matrix.device)
        B, T, C, H, W = matrix.shape
        rotation_vec_rads = rotation_vec * torch.pi / 180

        # crop size to ensure no invalid subtracted values. Do validate the math here
        crop_h_size = int(H - (
                    2 * (max(shift_h_vec.abs()) + (0.5 * H * np.tan(max(abs(rotation_vec_rads))).abs())) * max(
                scale_vec) + 10))  # 10 just extra
        crop_w_size = int(W - (
                    2 * (max(shift_w_vec.abs()) + (0.5 * W * np.tan(max(abs(rotation_vec_rads))).abs())) * max(
                scale_vec) + 10))

        # make a table for all possible transforms for every frame in B, T
        SAD_dimensions = (
        max(shift_h_vec.shape), max(shift_w_vec.shape), max(rotation_vec_rads.shape), max(scale_vec.shape))
        SAD_list = []

        # calculate min SAD
        counter = 0
        for shift_h_idx, current_shift_h in enumerate(shift_h_vec):
            for shift_w_idx, current_shift_w in enumerate(shift_w_vec):
                for rotation_idx, current_rotation_angle in enumerate(rotation_vec_rads):
                    for scale_idx, current_scale in enumerate(scale_vec):
                        print(counter)
                        counter += 1
                        # get affine grid
                        # TODO temporarily banished
                        # affine_grid = self._generate_affine_grid(current_shift_h, current_shift_w,
                        #                                          current_rotation_angle, current_scale)
                        # torch.cuda.empty_cache()
                        # # warp matrix
                        # warped_matrix = self._warp_matrix(matrix, affine_grid, warp_method)
                        warped_matrix = affine_transform_interpolated(matrix,
                                                                      construct_tensor(-current_shift_h),
                                                                      construct_tensor(-current_shift_w),
                                                                      construct_tensor(-current_rotation_angle),
                                                                      construct_tensor(1 / current_scale),
                                                                      warp_method=warp_method,
                                                                      expand=False)

                        # center crop to ensure no edge error occurs
                        # cropped_warped_matrix = tensor_center_crop(warped_matrix, crop_h_size, crop_w_size)
                        # cropped_reference_matrix = tensor_center_crop(reference_matrix, crop_h_size, crop_w_size)
                        cropped_warped_matrix = crop_torch_batch(warped_matrix, (crop_h_size, crop_w_size))
                        cropped_reference_matrix = crop_torch_batch(reference_matrix, (crop_h_size, crop_w_size))

                        # #TODO: dudy, delete:
                        # cropped_warped_images_list.append(cropped_warped_matrix.cpu())
                        # cropped_reference_images_list.append(cropped_reference_matrix.cpu())

                        # calculate min SAD
                        current_SAD = (cropped_warped_matrix - cropped_reference_matrix).abs()
                        # SAD_matrix_list.append(current_SAD)
                        current_SAD_mean = current_SAD.mean(-1, True).mean(-2, True).mean(-3, True)
                        current_SAD_mean = current_SAD_mean.squeeze(-1).squeeze(-1).squeeze(-1).reshape((B * T))

                        # add to the SAD mapping table
                        SAD_list.append(current_SAD_mean)

        # find minimum for every frame
        # Create flattened SAD matrix and min indices
        SAD_matrix = torch.stack(SAD_list, 0)
        min_SAD_flattened_indices = torch.argmin(SAD_matrix, dim=0)

        # Create SAD min indices. avoid edges to enable parabola fit
        min_transforms_indices_tensor = torch.stack(unravel_index_torch(min_SAD_flattened_indices, SAD_dimensions)).to(
            torch.long)

        # Validate minSAD to include no edge values
        if len(shift_h_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(shift_h_vec) - 1).item() for x in min_transforms_indices_tensor[0]]),
                message="input contains edge values in the shift h range. do expand the range"
            )
        if len(shift_w_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(shift_w_vec) - 1).item() for x in min_transforms_indices_tensor[1]]),
                message="input contains edge values in the shift w range. do expand the range"
            )
        if len(rotation_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(rotation_vec) - 1).item() for x in min_transforms_indices_tensor[2]]),
                message="input contains edge values in the rotation range. do expand the range"
            )
        if len(scale_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(scale_vec) - 1).item() for x in min_transforms_indices_tensor[3]]),
                message="input contains edge values in the scale range. do expand the range"
            )

        # shift H sub pixel:
        shift_h_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=shift_h_vec,
                                                              shifts_min_indices_nd=min_transforms_indices_tensor,
                                                              shift_dimension=0,
                                                              SAD_values_flattened=SAD_matrix,
                                                              SAD_dimensions=SAD_dimensions)

        # shift W sub pixel:
        shift_w_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=shift_w_vec,
                                                              shifts_min_indices_nd=min_transforms_indices_tensor,
                                                              shift_dimension=1,
                                                              SAD_values_flattened=SAD_matrix,
                                                              SAD_dimensions=SAD_dimensions)

        # rotation sub pixel:
        rotation_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=rotation_vec,
                                                               shifts_min_indices_nd=min_transforms_indices_tensor,
                                                               shift_dimension=2,
                                                               SAD_values_flattened=SAD_matrix,
                                                               SAD_dimensions=SAD_dimensions)

        # scale sub pixel:
        scale_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=scale_vec,
                                                            shifts_min_indices_nd=min_transforms_indices_tensor,
                                                            shift_dimension=3,
                                                            SAD_values_flattened=SAD_matrix,
                                                            SAD_dimensions=SAD_dimensions)

        return shift_h_sub_pixel, shift_w_sub_pixel, rotation_sub_pixel, scale_sub_pixel, SAD_matrix

    def _interpolated_minSAD_shifts_weighted(self,
                                             matrix: torch.Tensor,
                                             reference_matrix: torch.Tensor,
                                             shift_h_vec: torch.Tensor = None,
                                             shift_w_vec: torch.Tensor = None,
                                             rotation_vec: torch.Tensor = None,
                                             scale_vec: torch.Tensor = None,
                                             quantile_factor: float = 0.95,
                                             warp_method: str = 'bilinear') -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        shift_h_vec, shift_w_vec, rotation_vec, scale_vec = interpolated_minSAD_defaults(shift_h_vec, shift_w_vec,
                                                                                         rotation_vec, scale_vec,
                                                                                         matrix.device)
        B, T, C, H, W = matrix.shape
        rotation_vec_rads = rotation_vec * torch.pi / 180

        # crop size to ensure no invalid subtracted values. Do validate the math here
        crop_h_size = int(H - (
                    2 * (max(shift_h_vec.abs()) + (0.5 * H * np.tan(max(abs(rotation_vec_rads))).abs())) * max(
                scale_vec) + 10))  # 10 just extra
        crop_w_size = int(W - (
                    2 * (max(shift_w_vec.abs()) + (0.5 * W * np.tan(max(abs(rotation_vec_rads))).abs())) * max(
                scale_vec) + 10))

        # make a table for all possible transforms for every frame in B, T
        SAD_dimensions = (
        max(shift_h_vec.shape), max(shift_w_vec.shape), max(rotation_vec_rads.shape), max(scale_vec.shape))
        SAD_list = []

        # calculate min SAD
        counter = 0
        for shift_h_idx, current_shift_h in enumerate(shift_h_vec):
            for shift_w_idx, current_shift_w in enumerate(shift_w_vec):
                for rotation_idx, current_rotation_angle in enumerate(rotation_vec_rads):
                    for scale_idx, current_scale in enumerate(scale_vec):
                        print(counter)
                        counter += 1
                        # get affine grid
                        # TODO temporarily banished
                        # affine_grid = self._generate_affine_grid(current_shift_h, current_shift_w,
                        #                                          current_rotation_angle, current_scale)
                        # torch.cuda.empty_cache()
                        # # warp matrix
                        # warped_matrix = self._warp_matrix(matrix, affine_grid, warp_method)
                        warped_matrix = affine_transform_interpolated(matrix,
                                                                      construct_tensor(-current_shift_h),
                                                                      construct_tensor(-current_shift_w),
                                                                      construct_tensor(-current_rotation_angle),
                                                                      construct_tensor(1 / current_scale),
                                                                      warp_method=warp_method,
                                                                      expand=False)

                        # center crop to ensure no edge error occurs
                        cropped_warped_matrix = crop_torch_batch(warped_matrix, (crop_h_size, crop_w_size))
                        cropped_reference_matrix = crop_torch_batch(reference_matrix, (crop_h_size, crop_w_size))

                        # calculate min SAD
                        #(1). Absolute Error:
                        current_SAD = (cropped_warped_matrix - cropped_reference_matrix).abs()
                        # #(2). Relative Error:
                        # current_SAD = ((cropped_warped_matrix - cropped_reference_matrix).abs() / cropped_reference_matrix)

                        # Calculate weights
                        logical_mask = current_SAD > current_SAD.quantile(0.99, dim=1)
                        current_SAD_filtered = current_SAD
                        current_SAD_filtered[logical_mask] = torch.nan
                        # SAD_matrix_list.append(current_SAD)
                        current_SAD_mean = current_SAD_filtered.nanmean(-1, True).nanmean(-2, True).nanmean(-3, True)
                        current_SAD_mean = current_SAD_mean.squeeze(-1).squeeze(-1).squeeze(-1).reshape((B * T))

                        # add to the SAD mapping table
                        SAD_list.append(current_SAD_mean)

        # find minimum for every frame
        # Create flattened SAD matrix and min indices
        SAD_matrix = torch.stack(SAD_list, 0)
        min_SAD_flattened_indices = torch.argmin(SAD_matrix, dim=0)

        # Create SAD min indices. avoid edges to enable parabola fit
        min_transforms_indices_tensor = torch.stack(unravel_index_torch(min_SAD_flattened_indices, SAD_dimensions)).to(
            torch.long)

        # Validate minSAD to include no edge values
        if len(shift_h_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(shift_h_vec) - 1).item() for x in min_transforms_indices_tensor[0]]),
                message="input contains edge values in the shift h range. do expand the range"
            )
        if len(shift_w_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(shift_w_vec) - 1).item() for x in min_transforms_indices_tensor[1]]),
                message="input contains edge values in the shift w range. do expand the range"
            )
        if len(rotation_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(rotation_vec) - 1).item() for x in min_transforms_indices_tensor[2]]),
                message="input contains edge values in the rotation range. do expand the range"
            )
        if len(scale_vec) > 1:
            raise_if(
                any([(x == 0 or x == len(scale_vec) - 1).item() for x in min_transforms_indices_tensor[3]]),
                message="input contains edge values in the scale range. do expand the range"
            )

        # shift H sub pixel:
        shift_h_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=shift_h_vec,
                                                              shifts_min_indices_nd=min_transforms_indices_tensor,
                                                              shift_dimension=0,
                                                              SAD_values_flattened=SAD_matrix,
                                                              SAD_dimensions=SAD_dimensions)

        # shift W sub pixel:
        shift_w_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=shift_w_vec,
                                                              shifts_min_indices_nd=min_transforms_indices_tensor,
                                                              shift_dimension=1,
                                                              SAD_values_flattened=SAD_matrix,
                                                              SAD_dimensions=SAD_dimensions)

        # rotation sub pixel:
        rotation_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=rotation_vec,
                                                               shifts_min_indices_nd=min_transforms_indices_tensor,
                                                               shift_dimension=2,
                                                               SAD_values_flattened=SAD_matrix,
                                                               SAD_dimensions=SAD_dimensions)

        # scale sub pixel:
        scale_sub_pixel = parabola_fit_1d_over_n_dimensions(possible_shifts_1d_vec=scale_vec,
                                                            shifts_min_indices_nd=min_transforms_indices_tensor,
                                                            shift_dimension=3,
                                                            SAD_values_flattened=SAD_matrix,
                                                            SAD_dimensions=SAD_dimensions)

        return shift_h_sub_pixel, shift_w_sub_pixel, rotation_sub_pixel, scale_sub_pixel, SAD_matrix

    def _format_min_SAD_params(self,
                               matrix: Union[Tensor, tuple, list, np.array],
                               reference_matrix: Union[Tensor, tuple, list, np.array],
                               shift_h_vec: Union[Tensor, tuple, list] = None,
                               shift_w_vec: Union[Tensor, tuple, list] = None,
                               rotation_vec: Union[Tensor, tuple, list] = None,
                               scale_vec: Union[Tensor, tuple, list] = None,
                               warp_method: str = 'bilinear') -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str]:

        # make sure matrix and reference are on the same device
        if torch.is_tensor(matrix) and torch.is_tensor(reference_matrix):
            if matrix.device != reference_matrix.device:
                reference_matrix = reference_matrix.to(matrix.device)

        # TODO think of a more elegant solution to the device problem
        # choose device according to the input data if is tensor
        if torch.is_tensor(matrix):
            device = matrix.device
        elif isinstance(matrix, list) and len(matrix) > 0 and torch.is_tensor(matrix[0]):
            device = matrix[0].device
        else:
            device = self.device

        # Using Anvil's wonderful validation
        matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec, warp_method = format_min_SAD_params(
            matrix,
            reference_matrix,
            shift_h_vec,
            shift_w_vec,
            rotation_vec,
            scale_vec,
            warp_method,
            device)

        return matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec, warp_method

    def _validate_internal_parameters(self, matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec):
        """Validate all the minSAD intrnal params to match the given input. if any difference exists, change the
        corresponding internal values

        :param matrix:
        :param shift_h_vec:
        :param shift_w_vec:
        :param rotation_vec:
        :param scale_vec:
        :return:
        """
        if self.device != matrix.device:
            self.device = matrix.device
            self.meshgrid_H = self.meshgrid_H.to(self.device)
            self.meshgrid_W = self.meshgrid_W.to(self.device)
            # TODO if affine grids are precalculated convert to device

        flag_change_affine_grids = False
        B, T, C, H, W = matrix.shape
        if B != self.B or T != self.T or H != self.H or W != self.W:
            self.B = B
            self.T = T
            self.H = H
            self.W = W
            warnings.warn("generating new meshgrid due to dimensionality change")
            self._generate_meshgrid()
            flag_change_affine_grids = True

        if not (torch.equal(self.shift_h_vec, shift_h_vec) and
                torch.equal(self.shift_w_vec, shift_h_vec) and
                torch.equal(self.rotation_vec, rotation_vec) and
                torch.equal(self.scale_vec, scale_vec)):
            self.shift_h_vec = shift_h_vec
            self.shift_w_vec = shift_w_vec
            self.rotation_vec = rotation_vec
            self.scale_vec = scale_vec
            flag_change_affine_grids = True

        # Clean Memory just in case
        torch.cuda.empty_cache()

        if flag_change_affine_grids:
            pass  # TODO if parallel option goes possible, do _generate_affine_grids

    def forward(self,
                matrix: Union[torch.Tensor, np.array, tuple, list],
                reference_matrix: Union[torch.Tensor, np.array, tuple, list],
                shift_h_vec: Union[Tensor, tuple, list] = None,
                shift_w_vec: Union[Tensor, tuple, list] = None,
                rotation_vec: Union[Tensor, tuple, list] = None,
                scale_vec: Union[Tensor, tuple, list] = None,
                warp_method: str = 'bilinear') -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # validate and format parameters
        matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec, warp_method = \
            self._format_min_SAD_params(matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec,
                                        warp_method)
        dimensions_memory = NoMemoryMatrixOrigami(matrix)
        matrix = dimensions_memory.expand_matrix(matrix)
        reference_matrix = dimensions_memory.expand_matrix(reference_matrix)
        self._validate_internal_parameters(matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec)

        # calculate minSAD
        # shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
        #     self._interpolated_minSAD_shifts(matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec,
        #                                      scale_vec,
        #                                      warp_method)
        shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
            self._interpolated_minSAD_shifts_weighted(matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec,
                                                      scale_vec, 0.97, warp_method)

        return shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix

    def align(self,
              matrix: Union[torch.Tensor, np.array, tuple, list],
              reference_matrix: Union[torch.Tensor, np.array, tuple, list, None],
              shift_h_vec: Union[Tensor, tuple, list] = None,
              shift_w_vec: Union[Tensor, tuple, list] = None,
              rotation_vec: Union[Tensor, tuple, list] = None,
              scale_vec: Union[Tensor, tuple, list] = None,
              warp_method: str = 'bilinear',
              align_to_center_frame: bool = False,
              warp_matrix: bool = True,
              return_shifts: bool = False) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
                                                    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor, None]:
        # Format parameters
        matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec, \
        warp_method, align_to_center_frame, warp_matrix, return_shifts = \
            format_minSAD_align_parameters(matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec,
                                           scale_vec, warp_method, align_to_center_frame, warp_matrix, return_shifts)
        # Expand matrix to 5D
        dimensions_memory = NoMemoryMatrixOrigami(matrix)
        matrix = dimensions_memory.expand_matrix(matrix)
        B, T, C, H, W = matrix.shape

        # Create reference matrix if needed, just expand otherwise:
        if align_to_center_frame:
            reference_matrix = matrix[:, T // 2: T // 2 + 1]
        else:
            reference_matrix = dimensions_memory.expand_matrix(reference_matrix)

        # find shifts
        shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
            self.forward(matrix, reference_matrix, shift_h_vec, shift_w_vec, rotation_vec, scale_vec, warp_method)

        shifts_h = shifts_h.to(matrix.device)
        shifts_w = shifts_w.to(matrix.device)
        rotational_shifts = rotational_shifts.to(matrix.device)
        scale_shifts = scale_shifts.to(matrix.device)
        min_sad_matrix = min_sad_matrix.to(matrix.device)

        # # TODO: dudy added
        min_sad_matrix = min_sad_matrix.cpu()
        # torch.cuda.empty_cache()

        # Warp matrix if needed
        warped_matrix = None
        if warp_matrix:
            # convert rotational shifts from degrees to radians
            rotational_shifts_rads = rotational_shifts * np.pi / 180
            # warp matrix
            warped_matrix = affine_warp_matrix(matrix, -shifts_h, -shifts_w, -rotational_shifts_rads, scale_shifts,
                                               warp_method)
            warped_matrix = dimensions_memory.squeeze_to_original_dims(warped_matrix)

        # Return what's requested
        if return_shifts:
            if warp_matrix:
                return warped_matrix, shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix
            else:
                return shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix
        if not return_shifts:
            if warp_matrix:
                return warped_matrix
        return None  # that's basically a tragedy


class CrossCorrelationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO initialize meshgrid and stuff
        # TODO also calculate matrix fft only once maybe? might it save time in any constellation?

    def classic_circular_cc_shifts_calc(self,
                                        matrix: Tensor,
                                        reference_matrix: Tensor,
                                        matrix_fft: Tensor,
                                        reference_tensor_fft: Tensor,
                                        normalize: bool = False,
                                        fftshift: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        # returns tuple of vertical shifts, horizontal shifts, and cross correlation
        cc = self.circular_cross_correlation_classic(matrix, reference_matrix, matrix_fft, reference_tensor_fft,
                                                     normalize, fftshift)
        B, T, C, H, W = matrix.shape
        midpoints = (H // 2, W // 2)
        shifth, shiftw = self.shifts_from_circular_cc(cc, midpoints)
        return shifth, shiftw, cc

    def forward(self,
                matrix: Union[torch.Tensor, np.array, tuple, list],
                reference_matrix: Union[torch.Tensor, np.array, list, tuple],
                matrix_fft: Union[torch.Tensor, np.array, tuple, list] = None,
                reference_matrix_fft: Union[torch.Tensor, np.array, list, tuple] = None,
                normalize_over_matrix: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """Computes subpixel shifts between given matrix and reference matrix using circular cross correlation

            :param matrix:  2-5D matrix of form (B,T,C,H,W) to be correlated against reference matrix
            :param reference_matrix: 2-5D matrix of form (B,T,C,H,W) to be correlated against
            :param matrix_fft: possible to pass matrix fft if already computed, to improve performance. FFT should be over dims [-2, -1]
            :param reference_matrix_fft: possible to pass reference fft if already computed, to improve performance. FFT should be over dims [-2, -1]
            :param normalize_over_matrix: whether to normalize the matrix values when calculating the cross correlation
            :return: tuple of three tensors. The first tensor is vertical shifts, the second tensor is horizontal shifts, and the third tensor is the circular cross correlation
            """
        matrix, reference_matrix, matrix_fft, reference_matrix_fft, normalize_over_matrix, _ = \
            format_parameters_classic_circular_cc(matrix, reference_matrix, matrix_fft, reference_matrix_fft,
                                                  normalize_over_matrix, False)
        dimensions_memory = NoMemoryMatrixOrigami(matrix)
        matrix = dimensions_memory.expand_matrix(matrix)
        reference_matrix = dimensions_memory.expand_matrix(reference_matrix)
        shifts_h, shifts_w, cc = self.classic_circular_cc_shifts_calc(matrix, reference_matrix, matrix_fft,
                                                                      reference_matrix_fft, normalize_over_matrix,
                                                                      False)
        return shifts_h, shifts_w, cc


class LinearPolarLayer(nn.Module):
    # TODO maybe later optionally send the matrix fft too and do the gaussian blur + hann window over the fft
    def __init__(self,
                 B: int = 1,
                 T: int = 1,
                 C: int = 1,
                 H: int = None,
                 W: int = None,
                 B_ref: int = None,
                 T_ref: int = None,
                 kernel_size: int = 25,
                 blur_low_sigma: Union[int, float] = 5 ** 2,
                 blur_high_sigma: Union[int, float] = 20 ** 2,
                 flag_use_hann_window: bool = True,
                 radius_div_factor: Union[int, float] = 32,
                 device: Union[torch.device, str] = None):
        super(LinearPolarLayer, self).__init__()
        # Initialize device
        if device is not None:
            self.device = device
        else:  # in case no device was sent, initialize all over the cpu
            self.device = None
            device = 'cpu'

        # Initialize object fields
        self.kernel_size = kernel_size
        self.blur_low_sigma = blur_low_sigma
        self.blur_high_sigma = blur_high_sigma
        self.flag_use_hann_window = flag_use_hann_window
        self.radius_div_factor = radius_div_factor
        self.gaussian_blur_layer_low_sigma = Gaussian_Blur_Layer(channels=C, kernel_size=kernel_size,
                                                                 sigma=blur_low_sigma, dim=2).to(device)
        self.gaussian_blur_layer_high_sigma = Gaussian_Blur_Layer(channels=C, kernel_size=kernel_size,
                                                                  sigma=blur_high_sigma, dim=2).to(device)
        # Initialize future use fields
        self.B = B
        self.T = T
        self.C = C
        self.H = H
        self.W = W
        if B_ref is None:
            self.B_ref = B
        else:
            self.B_ref = B_ref
        if T_ref is None:
            self.T_ref = T
        else:
            self.T_ref = T_ref
        self.center = None
        self.theta_meshgrid = None
        self.radius_meshgrid = None
        self.flow_grid_mat = None
        self.flow_grid_ref = None
        self.hann_window = None

        if self.H is not None and self.W is not None:
            self.center = (self.H / 2, self.W / 2)
            self._initialize_meshgrids(device)

    T = TypeVar('T')

    def internals_to(self: T, device: Optional[Union[int, torch.device]] = ...) -> T:
        if self.gaussian_blur_layer_low_sigma is not None:
            self.gaussian_blur_layer_low_sigma = self.gaussian_blur_layer_low_sigma.to(device)
        if self.gaussian_blur_layer_high_sigma is not None:
            self.gaussian_blur_layer_high_sigma = self.gaussian_blur_layer_high_sigma.to(device)
        if self.theta_meshgrid is not None:
            self.theta_meshgrid = self.theta_meshgrid.to(device)
        if self.radius_meshgrid is not None:
            self.radius_meshgrid = self.radius_meshgrid.to(device)
        if self.flow_grid_mat is not None:
            self.flow_grid_mat = self.flow_grid_mat.to(device)
        if self.flow_grid_ref is not None:
            self.flow_grid_ref = self.flow_grid_ref.to(device)
        if self.hann_window is not None:
            self.hann_window = self.hann_window.to(device)

        self.device = device

        return self

    def _initialize_meshgrids(self, device: Union[int, torch.device] = None):
        if device is None:
            device = self.device

        # Assumes H and W to be correctly initialized!
        number_of_angles = self.H
        number_of_radii = self.W

        # Initialize theta meshgrid (meshgrid along y)
        theta = torch.zeros((number_of_angles, number_of_radii)).to(device)
        theta.T[:] = torch.Tensor(torch.linspace(0, 2 * np.pi, number_of_angles + 1))[:-1].to(device)

        # Initialize radius meshgrid (meshgrid along x)
        # radius = (torch.arange(number_of_radii) // self.radius_div_factor).to(device)
        radius = (torch.arange(number_of_radii) / self.radius_div_factor).to(device)
        radius = radius.unsqueeze(0).repeat(theta.shape[0], 1)  # TODO: dudy

        # Get (x,y) meshgrid for subsequent interpolation from cartesian to polar coordinates
        x = radius * torch.cos(theta) + self.center[0]
        y = radius * torch.sin(theta) + self.center[1]
        x = x / max(x.shape[0], 1)
        y = y / max(x.shape[1], 1)
        x = (x - 0.5) * 2
        y = (y - 0.5) * 2

        # Change to shape BTHW like any traditional meshgrid
        x = x.unsqueeze(0).unsqueeze(-1)
        y = y.unsqueeze(0).unsqueeze(-1)

        self.flow_grid_mat = torch.cat((x, y), dim=-1).to(device)

        # Calculate flow grids' differ between reference flow grid and matrix flow grid if needed
        if self.B != self.B_ref or self.T != self.T_ref:
            self.flow_grid_ref = torch.cat([self.flow_grid_mat] * (self.B_ref * self.T_ref), dim=0).to(device)
            self.flow_grid_mat = torch.cat([self.flow_grid_mat] * (self.B * self.T), dim=0).to(device)
        else:
            self.flow_grid_mat = torch.cat([self.flow_grid_mat] * (self.B * self.T), dim=0).to(device)
            self.flow_grid_ref = self.flow_grid_mat

    def polar_transform(self, matrix: Tensor, grid_shape='mat'):
        """Return polar transformed image."""
        B, T, C, H, W = matrix.shape
        matrix = matrix.view(B * T, C, H, W)
        output = Tensor([])
        if grid_shape == 'mat':
            output = torch.nn.functional.grid_sample(matrix.to(self.device), self.flow_grid_mat,
                                                     mode='bicubic')
        elif grid_shape == 'ref':
            output = torch.nn.functional.grid_sample(matrix.to(self.device), self.flow_grid_ref,
                                                     mode='bicubic')
        return output.view(B, T, C, H, W)

    def polar_rotation_shifts_calc(self, matrix: Tensor, reference_matrix: Tensor) -> Tensor:
        B, T, C, H, W = matrix.shape
        Br, Tr, _, _, _ = reference_matrix.shape
        # Convert matrix to 4d for the preprocessing
        matrix = matrix.view(B * T, C, H, W)
        reference_matrix = reference_matrix.view(Br * Tr, C, H, W)

        # Apply band pass filter over the image (why is that necessary?)
        matrix = self.gaussian_blur_layer_low_sigma(matrix) - self.gaussian_blur_layer_high_sigma(matrix)
        reference_matrix = self.gaussian_blur_layer_low_sigma(reference_matrix) - self.gaussian_blur_layer_high_sigma(
            reference_matrix)

        # Apply Hann window - to avoid effects from the image edges
        if self.flag_use_hann_window:
            if self.hann_window is None:
                self.hann_window = torch_2D_hann_window(matrix.shape).to(self.device)
            matrix = matrix * self.hann_window
            reference_matrix = reference_matrix * self.hann_window

        # Bring matrix back to 5d
        matrix = matrix.view(B, T, C, H, W)
        reference_matrix = reference_matrix.view(Br, Tr, C, H, W)

        # Calculate tensors' shifted FFT
        matrix_fft_abs = torch.abs(torch_fftshift(torch_fft2(matrix))).to(self.device)
        reference_matrix_fft_abs = torch.abs(torch_fftshift(torch_fft2(reference_matrix))).to(self.device)

        shape = (matrix_fft_abs.shape[-2], matrix_fft_abs.shape[-1])

        # Calculate the Linear Polar transform
        matrix_fft_abs_polar = self.polar_transform(matrix_fft_abs, grid_shape='mat')
        reference_matrix_fft_abs_polar = self.polar_transform(reference_matrix_fft_abs, grid_shape='ref')

        # Get cross correlation & shifts
        matrix_fft_abs_polar = matrix_fft_abs_polar[..., :shape[0] // 2, :]  # only use half of FFT
        reference_matrix_fft_abs_polar = reference_matrix_fft_abs_polar[..., :shape[0] // 2, :]
        shifts_r, shifts_c, cc = circular_cc_shifts(matrix_fft_abs_polar, reference_matrix_fft_abs_polar)

        # Get rotation from returned value
        recovered_angles = - (360 / shape[0]) * shifts_r
        recovered_angles = recovered_angles.view(B, T)  # TODO make sure nailed the shape issues

        return recovered_angles

    def forward(self,
                matrix: Union[torch.Tensor, np.array, tuple, list],
                reference_matrix: Union[torch.Tensor, np.array, list, tuple]) -> Tensor:
        matrix, reference_matrix, device = self._format_parameters_polar_cc(matrix, reference_matrix)

        dimensions_memory = NoMemoryMatrixOrigami(matrix)
        matrix = dimensions_memory.expand_matrix(matrix)
        reference_matrix = dimensions_memory.expand_matrix(reference_matrix)

        self._validate_internal_parameters(matrix, reference_matrix)

        # find shifts
        recovered_angles = self.polar_rotation_shifts_calc(matrix, reference_matrix)

        # Take care of dimensionality
        original_dims = dimensions_memory.original_dims
        for i in range(len(original_dims) - 2):
            recovered_angles = recovered_angles.squeeze(0)

        return recovered_angles

    def align(self,
              matrix: Union[torch.Tensor, np.array, tuple, list],
              reference_matrix: Union[torch.Tensor, np.array, tuple, list, None],
              warp_method: str = 'bilinear',
              align_to_center_frame: bool = False,
              warp_matrix: bool = True,
              return_shifts: bool = False):
        # TODO finish when all the rest is working
        # matrix, reference_matrix, warp_method, align_to_center_frame, warp_matrix, return_shifts = self._format_parameters_polar_cc()
        # Expand matrix to 5D
        dimensions_memory = NoMemoryMatrixOrigami(matrix)
        matrix = dimensions_memory.expand_matrix(matrix)
        B, T, C, H, W = matrix.shape

        # Create reference matrix if needed, just expand otherwise:
        if align_to_center_frame:
            reference_matrix = matrix[:, T // 2: T // 2 + 1]
        else:
            reference_matrix = dimensions_memory.expand_matrix(reference_matrix)

        # Find shifts
        # TODO notice that dimensionality issues are not yet taken care of
        rotational_shifts = self.polar_rotation_shifts_calc(matrix, reference_matrix)

        # Warp matrix if needed
        warped_matrix = None
        if warp_matrix:
            # convert rotational shifts from degrees to radians
            rotational_shifts_rads = rotational_shifts * np.pi / 180
            # warp matrix
            warped_matrix = affine_warp_matrix(matrix, thetas=-rotational_shifts_rads, warp_method=warp_method)
            warped_matrix = dimensions_memory.squeeze_to_original_dims(warped_matrix)

        # Return what's requested
        if return_shifts:
            if warp_matrix:
                return warped_matrix, rotational_shifts
            else:
                return rotational_shifts
        if not return_shifts:
            if warp_matrix:
                return warped_matrix
        return None  # that's basically a tragedy

    def _validate_internal_parameters(self, matrix: Tensor, reference_matrix: Tensor):
        # validate internal params
        B, T, C, H, W = matrix.shape
        B_ref, T_ref, _, _, _ = reference_matrix.shape
        if self.H != H or self.W != W:
            self.H = H
            self.W = W
            self.B = B
            self.T = T
            self.center = (self.H / 2, self.W / 2)
            self._initialize_meshgrids()
        elif (self.B != B or self.T != T) or (self.B_ref != B_ref or self.T_ref != T_ref):
            if self.B != B or self.T != T:
                self.B = B
                self.T = T
                self.flow_grid_mat = torch.cat([self.flow_grid_mat[:1]] * (B*T), dim=0)
            if self.B_ref != B_ref or self.T_ref != T_ref:
                self.B_ref = B_ref
                self.T_ref = T_ref
                self.flow_grid_ref = torch.cat([self.flow_grid_ref[:1]] * (B_ref*T_ref), dim=0)
        if self.C != C:
            self.C = C
            self.gaussian_blur_layer_low_sigma = Gaussian_Blur_Layer(channels=self.C, kernel_size=self.kernel_size,
                                                                     sigma=self.blur_low_sigma, dim=2).to(self.device)
            self.gaussian_blur_layer_high_sigma = Gaussian_Blur_Layer(channels=self.C, kernel_size=self.kernel_size,
                                                                      sigma=self.blur_high_sigma, dim=2).to(self.device)

    def _format_parameters_polar_cc(self,
                                    matrix: Union[torch.Tensor, np.array, tuple, list],
                                    reference_matrix: Union[torch.Tensor, np.array, list, tuple],
                                    device: Union[torch.device, str] = None
                                    ) -> Tuple[Tensor, Tensor, Union[torch.device, str]]:
        # validate device issues first
        if device is None:
            if self.device is None:
                if torch.is_tensor(matrix):
                    self.device = matrix.device
                elif isinstance(matrix, list) and len(matrix) > 0 and torch.is_tensor(matrix[0]):
                    self.device = matrix[0].device
                else:
                    self.device = 'cpu'
                self.internals_to(device)
        else:
            self.device = device
            self.internals_to(device)

        matrix = construct_tensor(matrix, self.device).to(self.device)
        reference_matrix = construct_tensor(reference_matrix, self.device).to(self.device)

        # if reference matrix dimension is smaller than matrix dimension (for example frame vs video), unsqueeze
        while len(reference_matrix.shape) < len(matrix.shape):
            reference_matrix = reference_matrix.unsqueeze(0)
            raise_if(true_dimensionality(matrix) == 0, message="Matrix is empty")
            raise_if(true_dimensionality(reference_matrix) == 0, message="Reference Matrix is empty")
            raise_if_not(multiplicable_tensors([matrix, reference_matrix]),
                         message="Matrix and Reference Matrix have different sizes")

        return matrix, reference_matrix, device


class Gimbaless_Rotation_Layer_Torch(nn.Module):

    # Initialize this with a module
    def __init__(self):
        super(Gimbaless_Rotation_Layer_Torch, self).__init__()

        ### Prediction Grid Definition: ###
        self.prediction_block_size = int16(4 / 1)  # for one global prediction for everything simply use W or H
        self.overlap_size = 0  # 0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
        self.temporal_lowpass_number_of_steps = 9

        ### Ctypi Parameters: ###
        self.reg_step = .1
        self.filt_len = 11
        self.dif_ord = 2
        self.dif_len = 7
        self.CorW = 7
        self.dYExp = 0
        self.dXExp = 0
        self.reg_lst = np.arange(0, 0.5, 0.1)

        ### Define convn layers: ###
        self.torch_convn_layer = convn_layer_torch()

        ### create filter: ###
        self.params = EasyDict()
        self.params.dif_len = 9
        # spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((1,1,1,params.dif_len))  #Read this from file
        self.dict_from_matlab = scipy.io.loadmat(r'/home/dudy/Projects/temp/gimbaless_rotation_support/spatial_lowpass_9tap.mat')  # Read this from file
        self.spatial_lowpass_before_temporal_derivative_filter_x = self.dict_from_matlab['spatial_lowpass_before_temporal_derivative_filter'].flatten()
        self.spatial_lowpass_before_temporal_derivative_filter_x = torch_get_4D(torch.Tensor(self.spatial_lowpass_before_temporal_derivative_filter_x))
        self.spatial_lowpass_before_temporal_derivative_filter_y = self.spatial_lowpass_before_temporal_derivative_filter_x.permute([0, 1, 3, 2])

        ### Preallocate filters: ###
        # (1). temporal derivative:
        self.temporal_derivative_filter = torch.Tensor(np.array([-1, 1]))
        self.temporal_derivative_filter = torch.reshape(self.temporal_derivative_filter, (1, 2, 1, 1))
        # (2). spatial derivative:
        self.grad_x_kernel = torch_get_4D(torch.Tensor(np.array([-1, 0, 1]) / 2), 'W')
        self.grad_y_kernel = self.grad_x_kernel.permute([0, 1, 3, 2])
        # (3). temporal averaging:
        self.temporal_averaging_filter_before_spatial_gradient = torch.ones((1, self.temporal_lowpass_number_of_steps, 1, 1, 1)) / self.temporal_lowpass_number_of_steps

        self.spatial_lowpass_before_temporal_derivative_filter_fft_x = None
        self.X = None

    def forward(self, input_tensor, reference_tensor=None):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.spatial_lowpass_before_temporal_derivative_filter_fft_x is None:
            ### Get FFT Filters: ###
            self.spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
            self.spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])
            self.temporal_derivative_fft_filter = torch.fft.fftn(self.temporal_derivative_filter, s=T, dim=[1])
            self.grad_x_filter_fft = torch.fft.fftn(self.grad_x_kernel, s=W, dim=[-1])
            self.grad_y_filter_fft = torch.fft.fftn(self.grad_y_kernel, s=H, dim=[-2])
            self.temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(self.temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

        if self.X is None:
            derivative_filter_lowpass_equivalent_length = len(self.spatial_lowpass_before_temporal_derivative_filter_x)
            shift_filter_length = 11 #TODO: later on with the full version there will be an equivalent shift filter
            invalid_frame_size = np.round((derivative_filter_lowpass_equivalent_length + shift_filter_length)/4)
            x_start = -input_tensor.shape[-1]/2 + 0.5 + invalid_frame_size
            x_stop = input_tensor.shape[-1]/2 - 0.5 - invalid_frame_size
            y_start = -input_tensor.shape[-2] / 2 + 0.5 + invalid_frame_size
            y_stop = input_tensor.shape[-2] / 2 - 0.5 - invalid_frame_size
            x_vec = my_linspace_step2(x_start, x_stop, 1)
            y_vec = my_linspace_step2(y_start, y_stop, 1)
            [X,Y] = np.meshgrid(x_vec, y_vec)
            self.X = torch.tensor(X).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.Y = torch.tensor(Y).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.X_flattened = torch.flatten(self.X, -3, -1)
            self.Y_flattened = torch.flatten(self.Y, -3, -1)

        ######################################################################################################
        ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###
        # # (1). Only input tensor
        # px = self.torch_convn_layer.forward(input_tensor, self.grad_x_kernel.flatten(), -1)
        # py = self.torch_convn_layer.forward(input_tensor, self.grad_y_kernel.flatten(), -2)
        # (2). input_tensor + reference_tensor
        px = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_x_kernel.flatten(), -1)
        py = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_y_kernel.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        # TODO: perhapse don't(!!!!) center crop now, but later. i can also hard-code this into the fast_binning function start and stop indices
        start_index = np.int16(floor(self.params.dif_len / 2))
        px = px[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        py = py[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ######################################################################################################

        ######################################################################################################
        ### Calculate Temporal Derivative: ###
        ABtx = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
        ABty = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        ABtx = ABtx[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ABty = ABty[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        #####################################################################################################


        ######################################################################################################
        ### Find Shift Between Frames Matrix Notation: ###
        #(*). i assume that gimbaless rotation is use for the ENTIRE FRAME only (no local predictions), so global sums only
        B2, T2, C2, H2, W2 = px.shape
        px = torch.flatten(px, -3, -1)
        py = torch.flatten(py, -3, -1)
        ABtx = torch.flatten(ABtx, -3, -1)
        ABty = torch.flatten(ABty, -3, -1)

        momentum_operator_tensor = px * self.Y_flattened - py * self.X_flattened
        p = torch.cat([px.unsqueeze(-1), py.unsqueeze(-1), momentum_operator_tensor.unsqueeze(-1)], -1).squeeze(0)
        p_mat = torch.bmm(torch.transpose(p, -1, -2), p)
        d1 = torch.linalg.inv(p_mat)
        d2 = torch.cat([(ABtx*px).sum(-1, True), (ABty*py).sum(-1, True), (ABtx*(px*self.Y_flattened) - ABty*(py*self.X_flattened)).sum(-1, True)], -1).squeeze(0).unsqueeze(-1)
        d_vec = torch.matmul(d1, d2).squeeze(-1)
        delta_x = d_vec[:, 0]
        delta_y = d_vec[:, 1]
        delta_theta = d_vec[:, 2]

        return delta_x, delta_y, delta_theta
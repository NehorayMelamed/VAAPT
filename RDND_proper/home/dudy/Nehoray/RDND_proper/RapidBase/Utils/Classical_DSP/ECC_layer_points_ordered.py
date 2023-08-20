import cv2
import numpy as np
import scipy.ndimage
import torch.nn.functional
import sys
sys.path.append("/home/mafat/python_bindings_libs/Ritalin/ecc_v3")
sys.path.append("/home/mafat/python_bindings_libs/Ritalin/ecc_v2")
sys.path.append("/home/mafat/python_bindings_libs/Ritalin/ecc_bi_lin_interp")
# import ecc_reduction
import ecc_bilinear_interpolation
import calc_delta_p_v3
import calc_delta_p_v2

from RapidBase.import_all import *

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def ecc_calc_delta_p(gx_chosen_values, gy_chosen_values, Jx, Jy, Jxx_prime,
                     Jxy_prime, Jyx_prime, Jyy_prime, current_level_reference_tensor_zero_mean,
                     current_level_input_tensor_warped):

    G, Gt, Gw, C = ecc_reduction.ecc_reduction(gx_chosen_values, gy_chosen_values,
                                               Jx, Jy,
                                               Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
                                               current_level_reference_tensor_zero_mean,
                                               current_level_input_tensor_warped)

    i_C = torch.linalg.inv(C)

    num = (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1, -2))).unsqueeze(-1) ** 2 - torch.transpose(Gw, -1, -2) @ i_C @ Gw
    den = (current_level_input_tensor_warped * current_level_reference_tensor_zero_mean).sum([-1, 2]).unsqueeze(-1) - torch.transpose(Gt, -1, -2) @ i_C @ Gw
    lambda_correction = (num / den).unsqueeze(-1)

    # (2). compute error vector:
    imerror = lambda_correction * current_level_reference_tensor_zero_mean - current_level_input_tensor_warped

    Ge = (G * imerror.squeeze().unsqueeze(-1)).sum([-2])
    delta_p = torch.matmul(i_C, Ge.unsqueeze(-1))

    return delta_p


from QS_Jetson_mvp_milestone.functional_utils.Elisheva_utils import *


def correct_H_matrix_for_coming_level_torch(H_matrix_in, transform_string, high_flag):
    # %H_matrix=correct_H_matrix_for_coming_level_numpy(H_matrix_in, transform_string, HIGH_FLAG)
    # % This function modifies appropriately the WARP values in order to apply
    # % the warp in the next level. If HIGH_FLAG is equal to 1, the function
    # % makes the warp appropriate for the next level of higher resolution.
    # If HIGH_FLAG is equal to 0, the function makes the warp appropriate for the previous level of lower resolution.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               'tranlation','affine' and 'homography'.
    # % HIGH_FLAG:    The flag which defines the 'next' level. 1 means that the
    # %               the next level is a higher resolution level,
    # %               while 0 means that it is a lower resolution level.
    # % Output:
    # % H_matrix:         the next-level warp transform

    H_matrix = H_matrix_in
    if high_flag == 'higher_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] / 2

        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

        if transform_string == 'translation':
            H_matrix = H_matrix * 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

    elif high_flag == 'lower_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] * 2

        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2

        if transform_string == 'translation':
            H_matrix = H_matrix / 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2

    return H_matrix


def update_transform_params_torch(H_matrix_in, delta_p, transform_string):
    # % H_matrix_out=update_transform_params_numpy(H_matrix_in,DELTA_P,transform_string)
    # % This function updates the parameter values by adding the correction values
    # % of DELTA_P to the current warp values in H_matrix_in.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % DELTA_P:      the current correction parameter vector,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               {'translation','euclidean','affine','homography'}.
    # % Output:
    # % H_matrix:         the new (updated) warp transform

    if transform_string == 'homography':
        # T = H_matrix_in.shape[0]
        # delta_p = torch.cat([delta_p, torch.zeros((T, 1, 1)).to(H_matrix_in.device)], 1)  # TODO: understand what's this
        # H_matrix_out = H_matrix_in + torch_reshape_image(delta_p, (3, 3), order='F')
        # H_matrix_out[2, 2] = 1

        H_matrix_out = H_matrix_in
        H_matrix_out[:, 0, 0] += delta_p[:, 0, 0]
        H_matrix_out[:, 1, 0] += delta_p[:, 1, 0]
        H_matrix_out[:, 2, 0] += delta_p[:, 2, 0]
        H_matrix_out[:, 0, 1] += delta_p[:, 3, 0]
        H_matrix_out[:, 1, 1] += delta_p[:, 4, 0]
        H_matrix_out[:, 2, 1] += delta_p[:, 5, 0]
        H_matrix_out[:, 0, 2] += delta_p[:, 6, 0]
        H_matrix_out[:, 1, 2] += delta_p[:, 7, 0]
        H_matrix_out[:, 2, 2] = 1

    if transform_string == 'affine':
        H_matrix_out = torch.zeros((2, 3)).to(H_matrix_in.device)
        H_matrix_out[0:2, :] = H_matrix_in[0:2, :] + torch_reshape_image(delta_p, (2, 3), order='F')
        H_matrix_out = torch.cat([H_matrix_out, torch.zeros((1, 3))], 0)
        H_matrix_out[2, 2] = 1

    if transform_string == 'translation':
        H_matrix_out = H_matrix_in + delta_p

    if transform_string == 'euclidean':
        theta = sign(H_matrix_in[1, 0]) * torch.arccos(H_matrix_in[0, 0]) + delta_p[0]
        tx = H_matrix_in[0, 2] + delta_p[1]
        ty = H_matrix_in[1, 2] + delta_p[2]
        H_matrix_out = torch.eye(3).to(H_matrix_in.device)
        H_matrix_out[0, :] = torch.tensor([torch.cos(theta), -torch.sin(theta), tx])
        H_matrix_out[1, :] = torch.tensor([torch.sin(theta), torch.cos(theta), ty])

    return H_matrix_out

class ECC_Layer_Torch_Points_Batch(nn.Module):
    # Initialize this with a module
    def __init__(self, input_tensor, reference_tensor, number_of_iterations_per_level, number_of_levels=1,
                 transform_string='homography', number_of_pixels_to_use=20000, delta_p_init=None, precision=torch.half):
        super(ECC_Layer_Torch_Points_Batch, self).__init__()
        self.X = None
        self.Y = None
        self.device = reference_tensor.device
        self.precision = precision

        ### Initialize Parameters: ###
        transform_string = str.lower(transform_string)
        # C_reference, H_reference, W_reference = reference_tensor.shape
        T,C,H,W = input_tensor.shape

        ### Initialize New Images For Algorithm To Change: ###
        initTemplate = reference_tensor
        reference_tensor = RGB2BW(reference_tensor).type(torch.float32).squeeze()

        ### Initialize H_matrix matrix: ###
        H_matrix, number_of_parameters = self.initialize_H_matrix(delta_p_init=None, transform_string=transform_string, device=self.device)

        ### in case of pyramid implementation, the initial transformation must be appropriately modified: ###
        for level_index in np.arange(0, number_of_levels - 1):
            H_matrix = correct_H_matrix_for_coming_level_torch(H_matrix, transform_string, 'lower_resolution')

        ### Assign To Internal Attributes: ###
        self.number_of_levels = number_of_levels
        self.number_of_iterations_per_level = number_of_iterations_per_level
        self.transform_string = transform_string
        self.number_of_parameters = number_of_parameters
        self.H_matrix = H_matrix.unsqueeze(0).repeat(T,1,1)
        self.reference_tensor_output_list = None
        self.number_of_pixels_to_use = number_of_pixels_to_use
        total_number_of_pixels = H * W
        self.quantile_to_use = 1 - self.number_of_pixels_to_use / total_number_of_pixels

    def initialize_H_matrix(self, delta_p_init=None, transform_string='homography', device='cpu'):
        # (1). Translation:
        if transform_string == 'translation':
            number_of_parameters = 2  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.zeros((2, 1))
            else:
                H_matrix = delta_p_init
        # (2). Euclidean:
        elif transform_string == 'euclidean':
            number_of_parameters = 3  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
                H_matrix[-1, -1] = 0
            else:
                H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
        # (3). Affine:
        elif transform_string == 'affine':
            number_of_parameters = 6  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
                H_matrix[-1, -1] = 0
            else:
                H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
        # (4). Homography:
        elif transform_string == 'homography':
            number_of_parameters = 8  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
            else:
                H_matrix = delta_p_init

        ### Send To Device: ###
        H_matrix = H_matrix.to(self.device).type(torch.float32)

        return H_matrix, number_of_parameters

    def spatial_interpolation_points_torch(self, chosen_indices, input_image, H_matrix, interpolation_method, transform_string, x_vec, y_vec, X_mat, Y_mat, H, W, bilinear_grid=None, X_mat_chosen_values=None, Y_mat_chosen_values=None):
        #################################################################################################################################
        # (2). New Faster Method (currently assumes homography transform):
        ### Try Simple Transformation: ###
        if bilinear_grid is None:
            H_matrix_corrected = H_matrix.unsqueeze(-1).type(self.precision)
            X_mat_chosen_values_corrected = X_mat_chosen_values.unsqueeze(-1).unsqueeze(-1)
            Y_mat_chosen_values_corrected = Y_mat_chosen_values.unsqueeze(-1).unsqueeze(-1)
            denom = (H_matrix_corrected[:, 2:3, 0:1] * X_mat_chosen_values_corrected + H_matrix_corrected[:, 2:3, 1:2] * Y_mat_chosen_values_corrected + H_matrix_corrected[:, 2:3, 2:3])
            xx_new = 2 * (H_matrix_corrected[:, 0:1, 0:1] * X_mat_chosen_values_corrected + H_matrix_corrected[:, 0:1, 1:2] * Y_mat_chosen_values_corrected + H_matrix_corrected[:, 0:1, 2:3]) / denom / max(W - 1, 1) - 1
            yy_new = 2 * (H_matrix_corrected[:, 1:2, 0:1] * X_mat_chosen_values_corrected + H_matrix_corrected[:, 1:2, 1:2] * Y_mat_chosen_values_corrected + H_matrix_corrected[:, 1:2, 2:3]) / denom / max(H - 1, 1) - 1
            ### Subpixel Interpolation 2: ###
            bilinear_grid = torch.cat([xx_new, yy_new], dim=3)  #TODO: this seems to take time! i should probably move towards self.bilinear_grid...for some reason it didn't show speedup!!!!

        out = torch.nn.functional.grid_sample(input_image, bilinear_grid, align_corners=True, mode='bilinear')  #[out] = [1,1,N,1]
        # out = self.bilinear_interpolate_torch(input_image, xx_new, yy_new)
        #################################################################################################################################

        return out, bilinear_grid

    def spatial_interpolation_torch(self, input_image, H_matrix, interpolation_method, transform_string, x_vec, y_vec, X_mat, Y_mat, H, W, bilinear_grid=None):
        # %OUT = spatial_interpolation_numpy(IN, H_matrix, STR, transform_string, x_vec, y_vec)
        # % This function implements the 2D spatial interpolation of image IN
        # %(inverse warping). The coordinates defined by x_vec,y_vec are projected through
        # % H_matrix thus resulting in new subpixel coordinates. The intensity values in
        # % new pixel coordinates are computed via bilinear interpolation
        # % of image IN. For other valid interpolation methods look at the help
        # % of Matlab function INTERP2.
        # %
        # % Input variables:
        # % IN:           the input image which must be warped,
        # % H_matrix:         the H_matrix transform,
        # % STR:          the string corresponds to interpolation method: 'linear',
        # %               'cubic' etc (for details look at the help file of
        # %               Matlab function INTERP2),
        # % transform_string:    the type of adopted transform: {'translation','euclidean','affine','homography'}
        # % x_vec:           the x-coordinate values of horizontal side of ROI (i.e. [xmin:xmax]),
        # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
        # %
        # % Output:
        # % OUT:          The warped (interpolated) image

        #################################################################################################################################
        # (1). Older, Slower Method:
        # ### Correct H_matrix If Needed: ###
        # if transform_string == 'affine' or transform_string == 'euclidean':
        #     if H_matrix.shape[0] == 2:
        #         H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)
        # if transform_string == 'translation':
        #     H_matrix = torch.cat([torch.eye(2), H_matrix], -1)
        #     H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)
        #
        # ### create meshgrid and flattened coordinates array ([x,y,1] basis): ###
        # xy = torch.cat([torch.transpose(torch_flatten_image(X_mat, True, 'F'), -1, -2),
        #                 torch.transpose(torch_flatten_image(Y_mat, True, 'F'), -1, -2),
        #                 torch.ones((1, len(torch_flatten_image(Y_mat, True, 'F')))).to(input_image.device)], 0).to(input_image.device)
        #
        # ### 3X3 matrix transformation: ###
        # A = H_matrix
        # A[-1, -1] = 1
        #
        # ### new coordinates: ###
        # xy_prime = torch.matmul(A, xy)
        #
        # ### division due to homogenous coordinates: ###
        # if transform_string == 'homography':
        #     xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
        #     xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]
        #
        # ### Ignore third row: ###
        # xy_prime = xy_prime[0:2, :]
        #
        # ### Turn to float32 instead of float64: ###
        # xy_prime = xy_prime.type(torch.float32)
        #
        # ### Subpixel interpolation: ###
        # # out = cv2.remap(input_image, np.reshape(xy_prime[0,:]+1, (H,W)), np.reshape(xy_prime[1,:]+1, (H,W)), cv2.INTER_CUBIC)
        # final_X_grid = torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')
        # final_Y_grid = torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')
        # new_X = 2 * final_X_grid / max(W - 1, 1) - 1
        # new_Y = 2 * final_Y_grid / max(H - 1, 1) - 1
        # bilinear_grid = torch.cat([torch_get_4D(new_X, 'CH'), torch_get_4D(new_Y, 'CH')], dim=3)
        # out = torch.nn.functional.grid_sample(input_image.unsqueeze(0).unsqueeze(0), bilinear_grid, mode='bicubic').squeeze(0).squeeze(0)
        #################################################################################################################################

        #################################################################################################################################
        # (2). New Faster Method (currently assumes homography transform):
        ### Try Simple Transformation: ###
        if bilinear_grid is None:
            H_matrix_corrected = H_matrix.unsqueeze(-1)
            denom = (H_matrix_corrected[:, 2:3, 0:1] * X_mat + H_matrix_corrected[:, 2:3, 1:2] * Y_mat + H_matrix_corrected[:, 2:3, 2:3])
            xx_new = 2 * (H_matrix_corrected[:, 0:1, 0:1] * X_mat + H_matrix_corrected[:, 0:1, 1:2] * Y_mat + H_matrix_corrected[:, 0:1, 2:3]) / denom / max(W - 1, 1) - 1
            yy_new = 2 * (H_matrix_corrected[:, 1:2, 0:1] * X_mat + H_matrix_corrected[:, 1:2, 1:2] * Y_mat + H_matrix_corrected[:, 1:2, 2:3]) / denom / max(H - 1, 1) - 1
            ### Subpixel Interpolation 2: ###
            bilinear_grid = torch.cat([xx_new, yy_new], dim=3)

        out = torch.nn.functional.grid_sample(input_image, bilinear_grid, mode='bilinear')
        #################################################################################################################################

        return out, bilinear_grid

    def images_and_gradients_to_delta_p_yuri(self, H_matrix,
                                             current_level_reference_tensor_zero_mean,
                                             current_level_input_tensor_warped,
                                             Jx_chosen_values, Jy_chosen_values,
                                             gx_chosen_values, gy_chosen_values):
        ### [H_matrix] = [T,3,3]
        ### [Jx_chosen_values] = [T,N]
        ### [Jy_chosen_values] = [T,N]
        ### [gx_chosen_values] = [T,N]
        ### [gy_chosen_values] = [T,N]
        ### [current_level_reference_tensor_zero_mean] = [T,N]
        ### [current_level_input_tensor_warped] = [T,N]

        ### Example Values: ###  #TODO: delete!!!
        T = 25
        N = 25000
        H_matrix = torch.randn((T,3,3)).cuda()
        gx_chosen_values = torch.randn((T,N)).cuda()
        gy_chosen_values = torch.randn((T,N)).cuda()
        Jx_chosen_values = torch.randn((T,N)).cuda()
        Jy_chosen_values = torch.randn((T,N)).cuda()
        current_level_reference_tensor_zero_mean = torch.randn((T,N)).cuda()
        current_level_input_tensor_warped = torch.randn((T,N)).cuda()

        ### Correct dimensions for pytorch arithmatic: ###
        Jx_chosen_values = Jx_chosen_values.unsqueeze(1).unsqueeze(1)  #-> [T,N,1,1]
        Jy_chosen_values = Jy_chosen_values.unsqueeze(1).unsqueeze(1)  #-> [T,N,1,1]
        H_matrix_corrected = H_matrix.unsqueeze(-1)  #-> [T,3,3,1]

        ### Calculate den once: ###
        den = (H_matrix_corrected[:, 2:3, 0:1] * Jx_chosen_values +
               H_matrix_corrected[:, 2:3, 1:2] * Jy_chosen_values +
               H_matrix_corrected[:, 2:3, 2:3])
        denom_inverse = 1 / den

        ### H Transform xy_prime values: ###
        xy_prime_reshaped_X = (H_matrix_corrected[:, 0:1, 0:1] * Jx_chosen_values +
                               H_matrix_corrected[:, 0:1, 1:2] * Jy_chosen_values +
                               H_matrix_corrected[:, 0:1, 2:3]) * denom_inverse
        xy_prime_reshaped_Y = (H_matrix_corrected[:, 1:2, 0:1] * Jx_chosen_values +
                               H_matrix_corrected[:, 1:2, 1:2] * Jy_chosen_values +
                               H_matrix_corrected[:, 1:2, 2:3]) * denom_inverse

        ### Correct Jx,Jy values: ###
        Jx_chosen_values = Jx_chosen_values * denom_inverse  # element-wise
        Jy_chosen_values = Jy_chosen_values * denom_inverse  # element-wise

        #### Get final Jxx,Jxy,Jyy,Jyx values: ####
        Jxx_prime = Jx_chosen_values * xy_prime_reshaped_X  # element-wise.
        Jyx_prime = Jy_chosen_values * xy_prime_reshaped_X
        Jxy_prime = Jx_chosen_values * xy_prime_reshaped_Y  # element-wise
        Jyy_prime = Jy_chosen_values * xy_prime_reshaped_Y

        # ### Get final jacobian of the H_matrix with respect to the different parameters: ###
        # J_list = [Jx_chosen_values, Jy_chosen_values, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime]

        ### Yuri calculations: ###
        current_level_reference_tensor_zero_mean = current_level_reference_tensor_zero_mean.unsqueeze(1).unsqueeze(1)  #->[T,1,1,N]
        current_level_input_tensor_warped = current_level_input_tensor_warped.unsqueeze(1).unsqueeze(1)  #->[T,1,1,N]
        gx_chosen_values = gx_chosen_values.unsqueeze(1).unsqueeze(1)  #->[T,1,1,N]
        gy_chosen_values = gy_chosen_values.unsqueeze(1).unsqueeze(1)  #->[T,1,1,N]
        delta_p = ecc_calc_delta_p(gx_chosen_values, gy_chosen_values,
                                   Jx_chosen_values, Jy_chosen_values,
                                   Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
                                   current_level_reference_tensor_zero_mean,
                                   current_level_input_tensor_warped)
        return delta_p

    def get_jacobian_for_warp_transform_points_torch_yuri(self, H_matrix, Jx_chosen_values, Jy_chosen_values):
        ### [H_matrix] = [T,3,3]
        ### [Jx_chosen_values] = [T,N]
        ### [Jy_chosen_values] = [T,N]

        # ### Example Values: ###  #TODO: delete!!!
        # T = 25
        # N = 25000
        # H_matrix = torch.randn((T,3,3)).cuda()
        # Jx_chosen_vales = torch.randn((T,N)).cuda()
        # Jy_chosen_vales = torch.randn((T,N)).cuda()

        ### Correct dimensions for pytorch arithmatic: ###
        Jx_chosen_values = Jx_chosen_values.unsqueeze(1).unsqueeze(1)
        Jy_chosen_values = Jy_chosen_values.unsqueeze(1).unsqueeze(1)
        H_matrix_corrected = H_matrix.unsqueeze(-1).type(self.precision)

        ### Calculate den once: ###
        den = (H_matrix_corrected[:, 2:3, 0:1] * Jx_chosen_values +
               H_matrix_corrected[:, 2:3, 1:2] * Jy_chosen_values +
               H_matrix_corrected[:, 2:3, 2:3])
        denom_inverse = 1 / den

        ### H Transform xy_prime values: ###
        xy_prime_reshaped_X = (H_matrix_corrected[:, 0:1, 0:1] * Jx_chosen_values +
                               H_matrix_corrected[:, 0:1, 1:2] * Jy_chosen_values +
                               H_matrix_corrected[:, 0:1, 2:3]) * denom_inverse
        xy_prime_reshaped_Y = (H_matrix_corrected[:, 1:2, 0:1] * Jx_chosen_values +
                               H_matrix_corrected[:, 1:2, 1:2] * Jy_chosen_values +
                               H_matrix_corrected[:, 1:2, 2:3]) * denom_inverse

        ### Correct Jx,Jy values: ###
        Jx_chosen_values = Jx_chosen_values * denom_inverse  # element-wise
        Jy_chosen_values = Jy_chosen_values * denom_inverse  # element-wise

        #### Get final Jxx,Jxy,Jyy,Jyx values: ####
        Jxx_prime = Jx_chosen_values * xy_prime_reshaped_X  # element-wise.
        Jyx_prime = Jy_chosen_values * xy_prime_reshaped_X
        Jxy_prime = Jx_chosen_values * xy_prime_reshaped_Y  # element-wise
        Jyy_prime = Jy_chosen_values * xy_prime_reshaped_Y

        ### Get final jacobian of the H_matrix with respect to the different parameters: ###
        J_list = [Jx_chosen_values, Jy_chosen_values, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime]

        return J_list

    def get_jacobian_for_warp_transform_points_torch(self, x_vec, y_vec, H_matrix, Jx, Jy, J0, J1, transform_string, H, W,
                                                     Jx_chosen_values, Jy_chosen_values, J0_chosen_values, J1_chosen_values):
        if str.lower(transform_string) == 'homography':
            ### New, Better Way: ###
            #TODO: i also think the both H*Jx and H*X_mat can be calculated at the same time when going to CUDA!!!!
            Jx_chosen_values = Jx_chosen_values.unsqueeze(1).unsqueeze(1)
            Jy_chosen_values = Jy_chosen_values.unsqueeze(1).unsqueeze(1)
            H_matrix_corrected = H_matrix.unsqueeze(-1).type(self.precision)

            den = (H_matrix_corrected[:, 2:3, 0:1] * Jx_chosen_values +
                   H_matrix_corrected[:, 2:3, 1:2] * Jy_chosen_values +
                   H_matrix_corrected[:, 2:3, 2:3])  #TODO: this is used three times here!!!!! calculate once and use it!!!!!
            denom_inverse = 1/den
            xy_prime_reshaped_X = (H_matrix_corrected[:, 0:1, 0:1] * Jx_chosen_values +
                                   H_matrix_corrected[:, 0:1, 1:2] * Jy_chosen_values +
                                   H_matrix_corrected[:, 0:1, 2:3]) * denom_inverse
            xy_prime_reshaped_Y = (H_matrix_corrected[:, 1:2, 0:1] * Jx_chosen_values +
                                   H_matrix_corrected[:, 1:2, 1:2] * Jy_chosen_values +
                                   H_matrix_corrected[:, 1:2, 2:3]) * denom_inverse

            #TODO: make sure! am i not CHANGING the values of Jx_chosen_values OUTSIDE the loop as well?!?!!? this is important also for the other versions!!!!!
            Jx_chosen_values = Jx_chosen_values * denom_inverse  # element-wise
            Jy_chosen_values = Jy_chosen_values * denom_inverse  # element-wise

            #### V2: ####
            #TODO: here i'm getting huge values which cause overflow in single precision!!!
            #TODO: single precision max value is about 65000, here we're off by about 3 orders of magnitude!!!! and later on by much more!!!!
            Jxx_prime = Jx_chosen_values * xy_prime_reshaped_X  # element-wise.
            Jyx_prime = Jy_chosen_values * xy_prime_reshaped_X
            Jxy_prime = Jx_chosen_values * xy_prime_reshaped_Y  # element-wise
            Jyy_prime = Jy_chosen_values * xy_prime_reshaped_Y

            ### Get final jacobian of the H_matrix with respect to the different parameters: ###
            # J_up = torch.cat([Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0], -1)
            # J_down = torch.cat([J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1], -1)
            # J = torch.cat([J_up, J_down], 0)
            J_list = [Jx_chosen_values, Jy_chosen_values, J0_chosen_values, J1_chosen_values, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime]

        elif str.lower(transform_string) == 'affine':
            # TODO: can be calculated in advance!!!
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            J_up = torch.cat([Jx, J0, Jy, J0, J1, J0], -1)
            J_down = torch.cat([J0, Jx, J0, Jy, J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        elif str.lower(transform_string) == 'translation':
            # TODO: can be calculated in advance!
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            J_up = torch.cat([J1, J0], -1)
            J_down = torch.cat([J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        elif str.lower(transform_string) == 'euclidean':
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            mycos = H_matrix[1, 1]
            mysin = H_matrix[2, 1]

            Jx_prime = -mysin * Jx - mycos * Jy
            Jy_prime = mycos * Jx - mysin * Jy

            J_up = torch.cat([Jx_prime, J1, J0], -1)
            J_down = torch.cat([Jy_prime, J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        return J_list

    def image_jacobian_points_torch(self, gx_chosen_values, gy_chosen_values, J_list, number_of_parameters):
        ### Unroll All Variables V2 (using the fact the J0,J1 are simply zeros and ones and disregarding them): ###
        gx_chosen_values = gx_chosen_values.squeeze(-1).unsqueeze(1)
        gy_chosen_values = gy_chosen_values.squeeze(-1).unsqueeze(1)
        [Jx, Jy, J0, J1, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime] = J_list
        G0 = gx_chosen_values * Jx
        G1 = gy_chosen_values * Jx
        G2 = -gx_chosen_values * Jxx_prime - gy_chosen_values * Jxy_prime
        G3 = gx_chosen_values * Jy
        G4 = gy_chosen_values * Jy
        G5 = -gx_chosen_values * Jyx_prime - gy_chosen_values * Jyy_prime
        G6 = gx_chosen_values
        G7 = gy_chosen_values

        # TODO: understand if making a list here takes time?!??
        G_list = [G0,
                  G1,
                  G2,
                  G3,
                  G4,
                  G5,
                  G6,
                  G7]

        ### PreCalculate C=(Gt*G): ###
        # TODO: would be smart to combine everything here together in the same memory run
        # TODO: make this batch-operations
        T,C,H,W = gx_chosen_values.shape
        C = torch.zeros((T, 8, 8)).to(gx_chosen_values.device)
        C[:, 0, 0] = (G0 * G0).sum([-1,-2,-3])
        C[:, 0, 1] = (G0 * G1).sum([-1,-2,-3])
        C[:, 0, 2] = (G0 * G2).sum([-1,-2,-3])
        C[:, 0, 3] = (G0 * G3).sum([-1,-2,-3])
        C[:, 0, 4] = (G0 * G4).sum([-1,-2,-3])
        C[:, 0, 5] = (G0 * G5).sum([-1,-2,-3])
        C[:, 0, 6] = (G0 * G6).sum([-1,-2,-3])
        C[:, 0, 7] = (G0 * G7).sum([-1,-2,-3])
        #
        C[:, 1, 0] = C[:, 0, 1]
        C[:, 1, 1] = (G1 * G1).sum([-1,-2,-3])
        C[:, 1, 2] = (G1 * G2).sum([-1,-2,-3])
        C[:, 1, 3] = (G1 * G3).sum([-1,-2,-3])
        C[:, 1, 4] = (G1 * G4).sum([-1,-2,-3])
        C[:, 1, 5] = (G1 * G5).sum([-1,-2,-3])
        C[:, 1, 6] = (G1 * G6).sum([-1,-2,-3])
        C[:, 1, 7] = (G1 * G7).sum([-1,-2,-3])
        #
        C[:, 2, 0] = C[:, 0, 2]
        C[:, 2, 1] = C[:, 1, 2]
        C[:, 2, 2] = (G2 * G2).sum([-1,-2,-3])
        C[:, 2, 3] = (G2 * G3).sum([-1,-2,-3])
        C[:, 2, 4] = (G2 * G4).sum([-1,-2,-3])
        C[:, 2, 5] = (G2 * G5).sum([-1,-2,-3])
        C[:, 2, 6] = (G2 * G6).sum([-1,-2,-3])
        C[:, 2, 7] = (G2 * G7).sum([-1,-2,-3])
        #
        C[:, 3, 0] = C[:, 0, 3]
        C[:, 3, 1] = C[:, 1, 3]
        C[:, 3, 2] = C[:, 2, 3]
        C[:, 3, 3] = (G3 * G3).sum([-1,-2,-3])
        C[:, 3, 4] = (G3 * G4).sum([-1,-2,-3])
        C[:, 3, 5] = (G3 * G5).sum([-1,-2,-3])
        C[:, 3, 6] = (G3 * G6).sum([-1,-2,-3])
        C[:, 3, 7] = (G3 * G7).sum([-1,-2,-3])
        #
        C[:, 4, 0] = C[:, 0, 4]
        C[:, 4, 1] = C[:, 1, 4]
        C[:, 4, 2] = C[:, 2, 4]
        C[:, 4, 3] = C[:, 3, 4]
        C[:, 4, 4] = (G4 * G4).sum([-1,-2,-3])
        C[:, 4, 5] = (G4 * G5).sum([-1,-2,-3])
        C[:, 4, 6] = (G4 * G6).sum([-1,-2,-3])
        C[:, 4, 7] = (G4 * G7).sum([-1,-2,-3])
        #
        C[:, 5, 0] = C[:, 0, 5]
        C[:, 5, 1] = C[:, 1, 5]
        C[:, 5, 2] = C[:, 2, 5]
        C[:, 5, 3] = C[:, 3, 5]
        C[:, 5, 4] = C[:, 4, 5]
        C[:, 5, 5] = (G5 * G5).sum([-1,-2,-3])
        C[:, 5, 6] = (G5 * G6).sum([-1,-2,-3])
        C[:, 5, 7] = (G5 * G7).sum([-1,-2,-3])
        #
        C[:, 6, 0] = C[:, 0, 6]
        C[:, 6, 1] = C[:, 1, 6]
        C[:, 6, 2] = C[:, 2, 6]
        C[:, 6, 3] = C[:, 3, 6]
        C[:, 6, 4] = C[:, 4, 6]
        C[:, 6, 5] = C[:, 5, 6]
        C[:, 6, 6] = (G6 * G6).sum([-1,-2,-3])
        C[:, 6, 7] = (G6 * G7).sum([-1,-2,-3])
        #
        C[:, 7, 0] = C[:, 0, 7]
        C[:, 7, 1] = C[:, 1, 7]
        C[:, 7, 2] = C[:, 2, 7]
        C[:, 7, 3] = C[:, 3, 7]
        C[:, 7, 4] = C[:, 4, 7]
        C[:, 7, 5] = C[:, 5, 7]
        C[:, 7, 6] = C[:, 6, 7]
        C[:, 7, 7] = (G7 * G7).sum([-1,-2,-3])
        return G_list, C

    def initialize_things_for_first_run(self, input_tensor, reference_tensor):
        H, W = reference_tensor.shape[-2:]
        T, C, H, W = input_tensor.shape

        ### Initialize Image Pyramids: ###
        self.reference_tensor_output_list = [0] * self.number_of_levels
        self.H_list = [0] * self.number_of_levels
        self.W_list = [0] * self.number_of_levels
        self.x_vec_list = [0] * self.number_of_levels
        self.y_vec_list = [0] * self.number_of_levels
        self.X_mat_list = [0] * self.number_of_levels
        self.Y_mat_list = [0] * self.number_of_levels
        self.Jx_list = [0] * self.number_of_levels
        self.Jy_list = [0] * self.number_of_levels
        self.J0_list = [0] * self.number_of_levels
        self.J1_list = [0] * self.number_of_levels
        self.Jx_chosen_values_list = [0] * self.number_of_levels
        self.Jy_chosen_values_list = [0] * self.number_of_levels
        self.J1_chosen_values_list = [0] * self.number_of_levels
        self.J0_chosen_values_list = [0] * self.number_of_levels
        self.X_mat_chosen_values_list = [0] * self.number_of_levels
        self.Y_mat_chosen_values_list = [0] * self.number_of_levels
        self.reference_tensor_chosen_values_list = [0] * self.number_of_levels
        self.Gt = torch.zeros((T, 8, 1)).to(input_tensor.device)
        self.Gw = torch.zeros((T, 8, 1)).to(input_tensor.device)
        self.Ge = torch.zeros((T, 8, 1)).to(input_tensor.device)

        ### Get Image Pyramid: ###
        # (1). First Level (Highest Resolution):
        self.input_tensor_warped = torch.zeros_like(input_tensor)
        self.reference_tensor_output_list[0] = reference_tensor
        self.H_list[0] = H
        self.W_list[0] = W
        self.x_vec_list[0] = torch.arange(0, W).to(reference_tensor.device)
        self.y_vec_list[0] = torch.arange(0, H).to(reference_tensor.device)
        [yy, xx] = torch.meshgrid(self.y_vec_list[0], self.x_vec_list[0])
        self.X_mat_list[0] = xx.unsqueeze(-1).unsqueeze(0).type(self.precision)  #TODO: make sure this is correct
        self.Y_mat_list[0] = yy.unsqueeze(-1).unsqueeze(0).type(self.precision)
        x_vec_length = len(self.x_vec_list[0])
        y_vec_length = len(self.y_vec_list[0])
        self.x_vec_unsqueezed = self.x_vec_list[0].unsqueeze(0)
        self.y_vec_unsqueezed = self.y_vec_list[0].unsqueeze(-1)
        self.Jx_list[0] = torch.repeat_interleave(self.x_vec_unsqueezed, y_vec_length, 0).unsqueeze(0).unsqueeze(0).type(self.precision)
        self.Jy_list[0] = torch.repeat_interleave(self.y_vec_unsqueezed, x_vec_length, 1).unsqueeze(0).unsqueeze(0).type(self.precision)
        self.J0_list[0] = 0 * self.Jx_list[0]  # could also use zeros_like  #TODO: obviously this, like others, can and should be created beforehand!!!
        self.J1_list[0] = self.J0_list[0] + 1  # could also use ones_like
        ### Get Points Of High Gradients On Reference: ###
        [vy_reference, vx_reference] = torch.gradient(reference_tensor, dim=[-2, -1])
        v_total_reference = torch.sqrt(vx_reference ** 2 + vy_reference ** 2)
        v_total_reference_mean = v_total_reference.float().quantile(self.quantile_to_use)
        # reference_tensor_gradient_above_mean_logical_mask = v_total_reference > 9
        self.reference_tensor_gradient_above_mean_logical_mask = v_total_reference > v_total_reference_mean
        self.reference_tensor_chosen_values_list[0] = reference_tensor[self.reference_tensor_gradient_above_mean_logical_mask]
        self.number_of_pixels_to_use_final = self.reference_tensor_chosen_values_list[0].shape[0]
        self.bilinear_grid = torch.zeros((T,self.number_of_pixels_to_use_final,1,2)).to(input_tensor.device)
        X_mat_chosen_values = self.X_mat_list[0].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)  # TODO: using logical mask returns all values into a 1D array, but i need [T,N] tensor
        Y_mat_chosen_values = self.Y_mat_list[0].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        Jx_chosen_values = self.Jx_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        Jy_chosen_values = self.Jy_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        J1_chosen_values = self.J1_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        J0_chosen_values = self.J0_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        ### Repeat Values For Parallel Processing: ###   #TODO: understand if there's a better way then this!!! i guess using cuda but still
        self.Jx_chosen_values_list[0] = Jx_chosen_values.repeat(T, 1)
        self.Jy_chosen_values_list[0] = Jy_chosen_values.repeat(T, 1)
        self.J1_chosen_values_list[0] = J1_chosen_values.repeat(T, 1)
        self.J0_chosen_values_list[0] = J0_chosen_values.repeat(T, 1)
        self.X_mat_chosen_values_list[0] = X_mat_chosen_values.repeat(T, 1)
        self.Y_mat_chosen_values_list[0] = Y_mat_chosen_values.repeat(T, 1)
        # (2). Subsequence Levels (Lower Resolutions):
        for level_index in np.arange(1, self.number_of_levels):
            ### Interpolate: ###
            self.reference_tensor_output_list[level_index] = torch.nn.functional.interpolate(self.reference_tensor_output_list[level_index - 1], scale_factor=0.5)

            ### Get Meshgrids & Vecs: ###
            H_current = self.H_list[level_index]
            W_current = self.W_list[level_index]
            x_vec = torch.arange(0, W_current).to(reference_tensor.device)
            y_vec = torch.arange(0, H_current).to(reference_tensor.device)
            self.x_vec_list[level_index] = x_vec
            self.y_vec_list[level_index] = y_vec
            [yy, xx] = torch.meshgrid(self.y_vec_list[level_index], self.x_vec_list[level_index])
            self.X_mat_list[level_index] = xx.unsqueeze(-1).unsqueeze(0).type(self.precision)
            self.Y_mat_list[level_index] = yy.unsqueeze(-1).unsqueeze(0).type(self.precision)

            ### Get Jacobian Auxiliary Tensors: ###
            x_vec_length = len(self.x_vec_list[level_index])
            y_vec_length = len(self.y_vec_list[level_index])
            x_vec_unsqueezed = self.x_vec_list[level_index].unsqueeze(0)
            y_vec_unsqueezed = self.y_vec_list[level_index].unsqueeze(-1)
            self.Jx_list[level_index] = torch.repeat_interleave(x_vec_unsqueezed, y_vec_length, 0).unsqueeze(0).unsqueeze(0).type(self.precision)
            self.Jy_list[level_index] = torch.repeat_interleave(y_vec_unsqueezed, x_vec_length, 1).unsqueeze(0).unsqueeze(0).type(self.precision)
            self.J0_list[level_index] = 0 * self.Jx_list[level_index]  # could also use zeros_like  #TODO: obviously this, like others, can and should be created beforehand!!!
            self.J1_list[level_index] = self.J0_list[level_index] + 1  # could also use ones_like

            ### Get Points Of High Gradients On Reference: ###
            [vy_reference, vx_reference] = torch.gradient(self.reference_tensor_output_list[level_index], dim=[-2, -1])
            v_total_reference = torch.sqrt(vx_reference ** 2 + vy_reference ** 2)
            v_total_reference_mean = v_total_reference.quantile(0.98)
            # reference_tensor_gradient_above_mean_logical_mask = v_total_reference > 9
            self.reference_tensor_gradient_above_mean_logical_mask = v_total_reference > v_total_reference_mean
            self.reference_tensor_chosen_values_list[level_index] = self.reference_tensor_output_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask]
            X_mat_chosen_values = self.X_mat_list[level_index].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)  # TODO: using logical mask returns all values into a 1D array, but i need [T,N] tensor
            Y_mat_chosen_values = self.Y_mat_list[level_index].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
            Jx_chosen_values = self.Jx_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
            Jy_chosen_values = self.Jy_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
            J1_chosen_values = self.J1_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
            J0_chosen_values = self.J0_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
            ### Repeat Values For Parallel Processing: ###   #TODO: understand if there's a better way then this!!! i guess using cuda but still
            self.Jx_chosen_values_list[level_index] = Jx_chosen_values.repeat(T, 1)
            self.Jy_chosen_values_list[level_index] = Jy_chosen_values.repeat(T, 1)
            self.J1_chosen_values_list[level_index] = J1_chosen_values.repeat(T, 1)
            self.J0_chosen_values_list[level_index] = J0_chosen_values.repeat(T, 1)
            self.X_mat_chosen_values_list[level_index] = X_mat_chosen_values.repeat(T, 1)
            self.Y_mat_chosen_values_list[level_index] = Y_mat_chosen_values.repeat(T, 1)

    def get_new_reference_tensor_gradient_logical_mask(self, reference_tensor):
        ### Get Points Of High Gradients On Reference: ###
        [vy_reference, vx_reference] = torch.gradient(reference_tensor, dim=[-2, -1])
        v_total_reference = torch.sqrt(vx_reference ** 2 + vy_reference ** 2)
        v_total_reference_mean = v_total_reference.quantile(self.quantile_to_use)
        # reference_tensor_gradient_above_mean_logical_mask = v_total_reference > 9
        self.reference_tensor_gradient_above_mean_logical_mask = v_total_reference > v_total_reference_mean
        self.reference_tensor_chosen_values_list[0] = reference_tensor[self.reference_tensor_gradient_above_mean_logical_mask]
        X_mat_chosen_values = self.X_mat_list[0].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)  # TODO: using logical mask returns all values into a 1D array, but i need [T,N] tensor
        Y_mat_chosen_values = self.Y_mat_list[0].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        Jx_chosen_values = self.Jx_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        Jy_chosen_values = self.Jy_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        J1_chosen_values = self.J1_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        J0_chosen_values = self.J0_list[0][self.reference_tensor_gradient_above_mean_logical_mask].type(self.precision)
        ### Repeat Values For Parallel Processing: ###   #TODO: understand if there's a better way then this!!! i guess using cuda but still
        self.Jx_chosen_values_list[0] = Jx_chosen_values.repeat(T, 1)
        self.Jy_chosen_values_list[0] = Jy_chosen_values.repeat(T, 1)
        self.J1_chosen_values_list[0] = J1_chosen_values.repeat(T, 1)
        self.J0_chosen_values_list[0] = J0_chosen_values.repeat(T, 1)
        self.X_mat_chosen_values_list[0] = X_mat_chosen_values.repeat(T, 1)
        self.Y_mat_chosen_values_list[0] = Y_mat_chosen_values.repeat(T, 1)
        # (2). Subsequence Levels (Lower Resolutions):
        for level_index in np.arange(1, self.number_of_levels):
            ### Interpolate: ###
            self.reference_tensor_output_list[level_index] = torch.nn.functional.interpolate(self.reference_tensor_output_list[level_index - 1], scale_factor=0.5)

            ### Get Points Of High Gradients On Reference: ###
            [vy_reference, vx_reference] = torch.gradient(self.reference_tensor_output_list[level_index], dim=[-2, -1])
            v_total_reference = torch.sqrt(vx_reference ** 2 + vy_reference ** 2)
            v_total_reference_mean = v_total_reference.quantile(0.98)
            # reference_tensor_gradient_above_mean_logical_mask = v_total_reference > 9
            self.reference_tensor_gradient_above_mean_logical_mask = v_total_reference > v_total_reference_mean
            self.reference_tensor_chosen_values_list[level_index] = self.reference_tensor_output_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask]
            X_mat_chosen_values = self.X_mat_list[level_index].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask]  # TODO: using logical mask returns all values into a 1D array, but i need [T,N] tensor
            Y_mat_chosen_values = self.Y_mat_list[level_index].squeeze(-1).unsqueeze(1)[self.reference_tensor_gradient_above_mean_logical_mask]
            Jx_chosen_values = self.Jx_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask]
            Jy_chosen_values = self.Jy_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask]
            J1_chosen_values = self.J1_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask]
            J0_chosen_values = self.J0_list[level_index][self.reference_tensor_gradient_above_mean_logical_mask]
            ### Repeat Values For Parallel Processing: ###   #TODO: understand if there's a better way then this!!! i guess using cuda but still
            self.Jx_chosen_values_list[level_index] = Jx_chosen_values.repeat(T, 1)
            self.Jy_chosen_values_list[level_index] = Jy_chosen_values.repeat(T, 1)
            self.J1_chosen_values_list[level_index] = J1_chosen_values.repeat(T, 1)
            self.J0_chosen_values_list[level_index] = J0_chosen_values.repeat(T, 1)
            self.X_mat_chosen_values_list[level_index] = X_mat_chosen_values.repeat(T, 1)
            self.Y_mat_chosen_values_list[level_index] = Y_mat_chosen_values.repeat(T, 1)

    def interpolations_dudy(self, current_level_input_tensor, reference_tensor_chosen_values, vx, vy, H_matrix, H, W,
                            X_mat_chosen_values, Y_mat_chosen_values):
        ### Interpolations: ###
        current_level_input_tensor_warped, bilinear_grid = self.spatial_interpolation_points_torch(None,
                                                                                                   current_level_input_tensor,
                                                                                                   H_matrix, 'linear',
                                                                                                   None, None, None,
                                                                                                   None, None, H, W,
                                                                                                   None,
                                                                                                   X_mat_chosen_values,
                                                                                                   Y_mat_chosen_values)  # inverse(backward) warping
        current_level_input_tensor_warped = current_level_input_tensor_warped.squeeze(-1).unsqueeze(1)
        # gtoc('initial warp of input tensor itself + building the warp bilinear grid')

        ###### DON'T DO ANYTHING: #######
        current_level_reference_tensor_zero_mean = reference_tensor_chosen_values

        ### Gradient Image interpolation (warped gradients): ###
        # gtic()
        vx_warped, bilinear_grid = self.spatial_interpolation_points_torch(None, vx, H_matrix, 'linear',
                                                                           None, None, None, None, None,
                                                                           H, W,
                                                                           bilinear_grid,
                                                                           X_mat_chosen_values,
                                                                           Y_mat_chosen_values)
        vy_warped, bilinear_grid = self.spatial_interpolation_points_torch(None, vy, H_matrix, 'linear',
                                                                           None, None, None, None, None,
                                                                           H, W,
                                                                           bilinear_grid,
                                                                           X_mat_chosen_values,
                                                                           Y_mat_chosen_values)
        # gtoc('warp vx and vy')
        return current_level_input_tensor_warped, current_level_reference_tensor_zero_mean, vx_warped, vy_warped

    def interpolations_yuri(self, current_level_input_tensor, reference_tensor_chosen_values, vx, vy,
                            H_matrix, H, W,
                            X_mat_chosen_values, Y_mat_chosen_values):
        current_level_input_tensor_warped, vx_warped, vy_warped = ecc_bilinear_interpolation.ecc_bilinear_interpolation(
            current_level_input_tensor.squeeze(), vx, vy,
            H_matrix,
            X_mat_chosen_values,
            Y_mat_chosen_values)

        ###### DON'T DO ANYTHING: #######
        current_level_reference_tensor_zero_mean = reference_tensor_chosen_values

        return current_level_input_tensor_warped, current_level_reference_tensor_zero_mean, vx_warped, vy_warped

    def forward_iterative_dudy(self, input_tensor, reference_tensor, max_shift_threshold=2e-3, flag_print=False,
                          delta_p_init=None, number_of_images_per_batch=10, flag_calculate_gradient_in_advance=False):
        ### Calculate Gradients: ###
        ### Get Image Pyramid For input_tensor: ###
        # (1). First Level (Highest Resolution):
        self.number_of_images_per_batch = number_of_images_per_batch
        input_tensor_output_list = [0] * self.number_of_levels
        input_tensor_vx_output_list = [0] * self.number_of_levels
        input_tensor_vy_output_list = [0] * self.number_of_levels
        flag_calculate_gradient_every_forward = ~flag_calculate_gradient_in_advance
        if flag_calculate_gradient_in_advance:
            input_tensor_output_list[0] = input_tensor
            [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2, -1])  # TODO: i actually don't really really need to do this as i only need the gradients at and near the sampled points, no?
            input_tensor_vx_output_list[0] = vx
            input_tensor_vy_output_list[0] = vy
            # (2). Subsequence Levels (Lower Resolutions):
            for level_index in np.arange(1, self.number_of_levels):
                ### Shape: ###
                H, W = input_tensor_output_list[level_index - 1].shape

                ### Interpolate: ###
                input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

                ### Get Gradients: ###
                # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
                [vy, vx] = torch.gradient(input_tensor_output_list[level_index], dim=[0, 1])  # TODO: only this needs to stay in the forward loop
                input_tensor_vx_output_list[level_index] = vx
                input_tensor_vy_output_list[level_index] = vy

        ### Get The Negative Of The CC Step H_matrix: ###
        if delta_p_init is None:
            delta_p_init = torch.eye(3).to(input_tensor.device).unsqueeze(0).repeat(input_tensor.shape[0],1,1)
        delta_p_init_negative = delta_p_init + 0
        delta_p_init_negative[:, 0, -1] *= -1
        delta_p_init_negative[:, 1, -1] *= -1

        ### Loop Over The Sub-Sequences: ###
        number_of_batches = input_tensor.shape[0] / number_of_images_per_batch
        output_images_list = []
        H_matrix_previous_batch = torch.eye(3).cuda()
        for batch_index in np.arange(number_of_batches):
            # gtic()
            ### Get Indices: ###
            start_index = int(batch_index * number_of_images_per_batch)
            stop_index = int(start_index + number_of_images_per_batch)
            current_input_tensor = input_tensor[start_index:stop_index]

            ### Get H_matrix From Previous Registration Steps: ###
            if delta_p_init is not None:
                CC_delta_p_init = delta_p_init[start_index:stop_index]
            else:
                CC_delta_p_init = torch.eye(3).repeat(number_of_images_per_batch, 1, 1).cuda()

            ### Perform ECC Step On Current Batch: ###
            if batch_index == 0:
                H_matrix_init = CC_delta_p_init
            else:
                H_matrix_init = delta_p_init_negative[start_index-1:start_index] @ H_matrix_previous_batch @ CC_delta_p_init
            # gtic()
            H_matrix_output, current_input_tensor_warped = self.forward_dudy(current_input_tensor,
                                                                     reference_tensor,
                                                                     max_shift_threshold=max_shift_threshold,
                                                                     flag_print=False,
                                                                     delta_p_init=H_matrix_init,
                                                                     input_tensor_output_list=input_tensor_output_list,
                                                                     input_tensor_vx_output_list=input_tensor_vx_output_list,
                                                                     input_tensor_vy_output_list=input_tensor_vy_output_list,
                                                                     flag_calculate_gradient_every_forward=flag_calculate_gradient_every_forward)
            H_matrix_previous_batch = H_matrix_output[-1:]
            # gtoc()
            # output_images_list.append(current_input_tensor_warped.cpu())
            input_tensor[start_index:stop_index] = current_input_tensor_warped
            # gtoc('iteration took: ')
        # output_images_list = torch.cat(output_images_list, 0)
        return input_tensor

    def forward_iterative_yuri(self, input_tensor, reference_tensor, max_shift_threshold=2e-3, flag_print=False,
                          delta_p_init=None, number_of_images_per_batch=10, flag_calculate_gradient_in_advance=False):
        ### Calculate Gradients: ###
        ### Get Image Pyramid For input_tensor: ###
        # (1). First Level (Highest Resolution):
        self.number_of_images_per_batch = number_of_images_per_batch
        input_tensor_output_list = [0] * self.number_of_levels
        input_tensor_vx_output_list = [0] * self.number_of_levels
        input_tensor_vy_output_list = [0] * self.number_of_levels
        flag_calculate_gradient_every_forward = ~flag_calculate_gradient_in_advance
        if flag_calculate_gradient_in_advance:
            input_tensor_output_list[0] = input_tensor
            [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2, -1])  # TODO: i actually don't really really need to do this as i only need the gradients at and near the sampled points, no?
            input_tensor_vx_output_list[0] = vx
            input_tensor_vy_output_list[0] = vy
            # (2). Subsequence Levels (Lower Resolutions):
            for level_index in np.arange(1, self.number_of_levels):
                ### Shape: ###
                H, W = input_tensor_output_list[level_index - 1].shape

                ### Interpolate: ###
                input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

                ### Get Gradients: ###
                # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
                [vy, vx] = torch.gradient(input_tensor_output_list[level_index], dim=[0, 1])  # TODO: only this needs to stay in the forward loop
                input_tensor_vx_output_list[level_index] = vx
                input_tensor_vy_output_list[level_index] = vy

        ### Get The Negative Of The CC Step H_matrix: ###
        if delta_p_init is None:
            delta_p_init = torch.eye(3).to(input_tensor.device).unsqueeze(0).repeat(input_tensor.shape[0],1,1)
        delta_p_init_negative = delta_p_init + 0
        delta_p_init_negative[:, 0, -1] *= -1
        delta_p_init_negative[:, 1, -1] *= -1

        ### Loop Over The Sub-Sequences: ###
        number_of_batches = input_tensor.shape[0] / number_of_images_per_batch
        output_images_list = []
        H_matrix_previous_batch = torch.eye(3).cuda()
        for batch_index in np.arange(number_of_batches):
            # gtic()
            ### Get Indices: ###
            start_index = int(batch_index * number_of_images_per_batch)
            stop_index = int(start_index + number_of_images_per_batch)
            current_input_tensor = input_tensor[start_index:stop_index]

            ### Get H_matrix From Previous Registration Steps: ###
            if delta_p_init is not None:
                CC_delta_p_init = delta_p_init[start_index:stop_index]
            else:
                CC_delta_p_init = torch.eye(3).repeat(number_of_images_per_batch, 1, 1).cuda()

            ### Perform ECC Step On Current Batch: ###
            if batch_index == 0:
                H_matrix_init = CC_delta_p_init
            else:
                H_matrix_init = delta_p_init_negative[start_index-1:start_index] @ H_matrix_previous_batch @ CC_delta_p_init
            # gtic()
            H_matrix_output, current_input_tensor_warped = self.forward_yuri(current_input_tensor,
                                                                     reference_tensor,
                                                                     max_shift_threshold=max_shift_threshold,
                                                                     flag_print=False,
                                                                     delta_p_init=H_matrix_init,
                                                                     input_tensor_output_list=input_tensor_output_list,
                                                                     input_tensor_vx_output_list=input_tensor_vx_output_list,
                                                                     input_tensor_vy_output_list=input_tensor_vy_output_list,
                                                                     flag_calculate_gradient_every_forward=flag_calculate_gradient_every_forward)
            H_matrix_previous_batch = H_matrix_output[-1:]
            # gtoc()
            # output_images_list.append(current_input_tensor_warped.cpu())
            input_tensor[start_index:stop_index] = current_input_tensor_warped
            # gtoc('iteration took: ')
        # output_images_list = torch.cat(output_images_list, 0)
        return input_tensor


    def forward_iterative_yuri_v2(self, input_tensor, reference_tensor, max_shift_threshold=2e-3, flag_print=False,
                          delta_p_init=None, number_of_images_per_batch=10, flag_calculate_gradient_in_advance=False):
        ### Calculate Gradients: ###
        ### Get Image Pyramid For input_tensor: ###
        # (1). First Level (Highest Resolution):
        self.number_of_images_per_batch = number_of_images_per_batch
        input_tensor_output_list = [0] * self.number_of_levels
        input_tensor_vx_output_list = [0] * self.number_of_levels
        input_tensor_vy_output_list = [0] * self.number_of_levels
        flag_calculate_gradient_every_forward = ~flag_calculate_gradient_in_advance
        if flag_calculate_gradient_in_advance:
            input_tensor_output_list[0] = input_tensor
            [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2, -1])  # TODO: i actually don't really really need to do this as i only need the gradients at and near the sampled points, no?
            input_tensor_vx_output_list[0] = vx
            input_tensor_vy_output_list[0] = vy
            # (2). Subsequence Levels (Lower Resolutions):
            for level_index in np.arange(1, self.number_of_levels):
                ### Shape: ###
                H, W = input_tensor_output_list[level_index - 1].shape

                ### Interpolate: ###
                input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

                ### Get Gradients: ###
                # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
                [vy, vx] = torch.gradient(input_tensor_output_list[level_index], dim=[0, 1])  # TODO: only this needs to stay in the forward loop
                input_tensor_vx_output_list[level_index] = vx
                input_tensor_vy_output_list[level_index] = vy

        ### Get The Negative Of The CC Step H_matrix: ###
        if delta_p_init is None:
            delta_p_init = torch.eye(3).to(input_tensor.device).unsqueeze(0).repeat(input_tensor.shape[0],1,1)
        delta_p_init_negative = delta_p_init + 0
        delta_p_init_negative[:, 0, -1] *= -1
        delta_p_init_negative[:, 1, -1] *= -1

        ### Loop Over The Sub-Sequences: ###
        number_of_batches = input_tensor.shape[0] / number_of_images_per_batch
        output_images_list = []
        H_matrix_previous_batch = torch.eye(3).cuda()
        for batch_index in np.arange(number_of_batches):
            # gtic()
            ### Get Indices: ###
            start_index = int(batch_index * number_of_images_per_batch)
            stop_index = int(start_index + number_of_images_per_batch)
            current_input_tensor = input_tensor[start_index:stop_index]

            ### Get H_matrix From Previous Registration Steps: ###
            if delta_p_init is not None:
                CC_delta_p_init = delta_p_init[start_index:stop_index]
            else:
                CC_delta_p_init = torch.eye(3).repeat(number_of_images_per_batch, 1, 1).cuda()

            ### Perform ECC Step On Current Batch: ###
            if batch_index == 0:
                H_matrix_init = CC_delta_p_init
            else:
                H_matrix_init = delta_p_init_negative[start_index-1:start_index] @ H_matrix_previous_batch @ CC_delta_p_init
            # gtic()
            H_matrix_output, current_input_tensor_warped = self.forward_yuri_v2(current_input_tensor,
                                                                     reference_tensor,
                                                                     max_shift_threshold=max_shift_threshold,
                                                                     flag_print=False,
                                                                     delta_p_init=H_matrix_init,
                                                                     input_tensor_output_list=input_tensor_output_list,
                                                                     input_tensor_vx_output_list=input_tensor_vx_output_list,
                                                                     input_tensor_vy_output_list=input_tensor_vy_output_list,
                                                                     flag_calculate_gradient_every_forward=flag_calculate_gradient_every_forward)
            H_matrix_previous_batch = H_matrix_output[-1:]
            # gtoc()
            # output_images_list.append(current_input_tensor_warped.cpu())
            input_tensor[start_index:stop_index] = current_input_tensor_warped
            # gtoc('iteration took: ')
        # output_images_list = torch.cat(output_images_list, 0)
        return input_tensor

    def forward_yuri_v2(self, input_tensor, reference_tensor, max_shift_threshold=2e-3, flag_print=False,
                delta_p_init=None,
                sub_sequence_index=0,
                input_tensor_output_list=None,
                input_tensor_vx_output_list=None,
                input_tensor_vy_output_list=None,
                flag_calculate_gradient_every_forward=True):
        ### Initialize Things For Subsequence Runs: ###
        if self.reference_tensor_output_list is None:
            self.initialize_things_for_first_run(input_tensor, reference_tensor)

        ### Assign previous H matrix for later stoppage condition: ###
        H_matrix_previous = None

        ### Calculate Gradients Vx & Vy: ###
        if flag_calculate_gradient_every_forward:
            input_tensor_output_list = [0] * self.number_of_levels
            input_tensor_vx_output_list = [0] * self.number_of_levels
            input_tensor_vy_output_list = [0] * self.number_of_levels
            input_tensor_output_list[0] = input_tensor
            [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2,
                                                                        -1])  # TODO: i actually don't really really need to do this as i only need the gradients at and near the sampled points, no?
            input_tensor_vx_output_list[0] = vx
            input_tensor_vy_output_list[0] = vy
            # (2). Subsequence Levels (Lower Resolutions):
            for level_index in np.arange(1, self.number_of_levels):
                ### Shape: ###
                H, W = input_tensor_output_list[level_index - 1].shape

                ### Interpolate: ###
                input_tensor_output_list[level_index] = torch.nn.functional.interpolate(
                    input_tensor_output_list[level_index - 1],
                    scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

                ### Get Gradients: ###
                # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
                [vy, vx] = torch.gradient(input_tensor_output_list[level_index],
                                          dim=[0, 1])  # TODO: only this needs to stay in the forward loop
                input_tensor_vx_output_list[level_index] = vx
                input_tensor_vy_output_list[level_index] = vy

        ### Run ECC algorithm for each level of the pyramid: ###
        for level_index in np.arange(self.number_of_levels, 0,
                                     -1):  # start with lowest resolution (highest level of the pyramid)
            ### Get Current Level input_tensor and reference_tensor: ###
            start_index = sub_sequence_index * self.number_of_images_per_batch
            stop_index = start_index + self.number_of_images_per_batch
            current_level_input_tensor = input_tensor_output_list[level_index - 1][start_index:stop_index]
            current_level_reference_tensor = self.reference_tensor_output_list[level_index - 1]
            if len(current_level_reference_tensor.shape) == 4:
                T, C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 3:
                C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 2:
                H, W = current_level_input_tensor.shape

            ### Get input_tensor gradients: ###
            vx = input_tensor_vx_output_list[level_index - 1][start_index:stop_index]
            vy = input_tensor_vy_output_list[level_index - 1][start_index:stop_index]

            ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
            x_vec = self.x_vec_list[level_index - 1]
            y_vec = self.y_vec_list[level_index - 1]
            X_mat = self.X_mat_list[level_index - 1]
            Y_mat = self.Y_mat_list[level_index - 1]
            current_level_reference_tensor = current_level_reference_tensor.type(torch.float32)

            ### Get Current Level Jacobian Auxiliary Tensors: ###
            # TODO: i wonder if all these assignments take anything up...i think NOT!
            Jx = self.Jx_list[level_index - 1]
            Jy = self.Jy_list[level_index - 1]
            J0 = self.J0_list[level_index - 1]
            J1 = self.J1_list[level_index - 1]

            ### Get Current Level Tensors Chosen Values On Grid (according to gradient high points): ###
            Jx_chosen_values = self.Jx_chosen_values_list[level_index - 1]
            Jy_chosen_values = self.Jy_chosen_values_list[level_index - 1]
            J1_chosen_values = self.J1_chosen_values_list[level_index - 1]
            J0_chosen_values = self.J0_chosen_values_list[level_index - 1]
            X_mat_chosen_values = self.X_mat_chosen_values_list[level_index - 1]
            Y_mat_chosen_values = self.Y_mat_chosen_values_list[level_index - 1]
            reference_tensor_chosen_values = self.reference_tensor_chosen_values_list[level_index - 1]

            ### Initialize H matrix: ###
            if delta_p_init is None:
                H_matrix = self.H_matrix
            else:
                H_matrix = delta_p_init

            ### ECC, Forward Additive Algorithm: ###
            for iteration_index in np.arange(self.number_of_iterations_per_level):
                if flag_print:
                    print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))

                # ### Interpolations Yuri: ###
                # current_level_input_tensor_warped, \
                #     current_level_reference_tensor_zero_mean, \
                #     vx_warped, \
                #     vy_warped = self.interpolations_yuri(current_level_input_tensor, reference_tensor_chosen_values, vx,
                #                                          vy, H_matrix, H, W, X_mat_chosen_values, Y_mat_chosen_values)

                ### Interpolations Dudy: ###
                current_level_input_tensor_warped, \
                    current_level_reference_tensor_zero_mean, \
                    vx_warped, \
                    vy_warped = self.interpolations_dudy(current_level_input_tensor, reference_tensor_chosen_values,
                                                         vx, vy, H_matrix, H, W, X_mat_chosen_values, Y_mat_chosen_values)

                ####################################################################################################################################
                ### Compute the jacobian of warp transform_string: ###
                J_list = self.get_jacobian_for_warp_transform_points_torch(x_vec + 1, y_vec + 1, H_matrix, Jx, Jy, J0,
                                                                           J1, self.transform_string, H, W,
                                                                           Jx_chosen_values.squeeze(),
                                                                           Jy_chosen_values.squeeze(),
                                                                           J0_chosen_values,
                                                                           J1_chosen_values)
                Jx_chosen_values, Jy_chosen_values, \
                    J0_chosen_values, J1_chosen_values, \
                    Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime = J_list


                ### Make tensors shape compatible with yuri: ###
                vx_warped = vx_warped.squeeze().unsqueeze(1).unsqueeze(1).contiguous()
                vy_warped = vy_warped.squeeze().unsqueeze(1).unsqueeze(1).contiguous()
                current_level_reference_tensor_zero_mean = current_level_reference_tensor_zero_mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                current_level_input_tensor_warped = current_level_input_tensor_warped

                ### Yuri: ###
                delta_p = calc_delta_p_v2.ecc_calc_delta_p(vx_warped, vy_warped,
                                                           Jx_chosen_values, Jy_chosen_values,
                                                           Jxx_prime, Jxy_prime,
                                                           Jyx_prime, Jyy_prime,
                                                           current_level_reference_tensor_zero_mean,
                                                           current_level_input_tensor_warped)

                # delta_p_norm = torch.norm(delta_p, dim=1)
                # gtoc('bla2')
                ####################################################################################################################################

                ### Update Parameters: ###
                H_matrix_previous = H_matrix * 1.0
                H_matrix = update_transform_params_torch(H_matrix, delta_p, self.transform_string)

                ### Break the loop if reached max number of iterations per level: ###
                max_difference = estimate_difference_between_homography_matrices(H_matrix_previous, H_matrix, H=H, W=W)
                # print('iteration: ' + str(iteration_index) + ',  max elisheva difference: ' + str(max_difference.max().item()))
                flag_end_iterations = (iteration_index == self.number_of_iterations_per_level) or (
                            max_difference < max_shift_threshold).all()
                if flag_end_iterations:  # the algorithm is executed (number_of_iterations_per_level-1) times
                    # print('stopped due to condition')
                    # print(iteration_index)
                    break

                # print(H_matrix)
                # print(delta_p)

                print('iteration number : ' + str(iteration_index))

            ### END OF INTERNAL ITERATIONS (PER LEVEL) LOOP
        ### END OF PYRAMID number_of_levels LOOP:
        # print('number of iterations this batch: ' + str(iteration_index))

        ### Get final H_matrix: ###
        final_warp = H_matrix

        ### return the final warped image using the whole support area (including margins): ###
        # TODO: fix this to allow for B,C,H,W ; B,T,C,H,W; C,H,W; H,W
        nx2 = self.x_vec_list[0]
        ny2 = self.y_vec_list[0]
        H, W = input_tensor.shape[-2:]
        C_input = 1
        self.warped_image, bilinear_grid = self.spatial_interpolation_torch(input_tensor, final_warp, 'bicubic',
                                                                            self.transform_string, nx2, ny2,
                                                                            self.X_mat_list[0], self.Y_mat_list[0], H,
                                                                            W)
        # for ii in torch.arange(C_input):
        #     warpedImage[ii, :, :] = spatial_interpolation_torch(input_tensor[ii, :, :], final_warp, 'linear', self.transform_string, nx2, ny2, H, W)
        H_matrix = final_warp
        return H_matrix, self.warped_image


    def forward_dudy(self, input_tensor, reference_tensor, max_shift_threshold=2e-3, flag_print=False,
                delta_p_init=None,
                sub_sequence_index=0,
                input_tensor_output_list=None,
                input_tensor_vx_output_list=None,
                input_tensor_vy_output_list=None,
                flag_calculate_gradient_every_forward=True):
        ### Initialize Things For Subsequence Runs: ###
        if self.reference_tensor_output_list is None:
            self.initialize_things_for_first_run(input_tensor, reference_tensor)

        ### Assign previous H matrix for later stoppage condition: ###
        H_matrix_previous = None

        ### Calculate Gradients Vx & Vy: ###
        if flag_calculate_gradient_every_forward:
            input_tensor_output_list = [0] * self.number_of_levels
            input_tensor_vx_output_list = [0] * self.number_of_levels
            input_tensor_vy_output_list = [0] * self.number_of_levels
            input_tensor_output_list[0] = input_tensor
            [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2, -1])  # TODO: i actually don't really really need to do this as i only need the gradients at and near the sampled points, no?
            input_tensor_vx_output_list[0] = vx
            input_tensor_vy_output_list[0] = vy
            # (2). Subsequence Levels (Lower Resolutions):
            for level_index in np.arange(1, self.number_of_levels):
                ### Shape: ###
                H, W = input_tensor_output_list[level_index - 1].shape

                ### Interpolate: ###
                input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

                ### Get Gradients: ###
                # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
                [vy, vx] = torch.gradient(input_tensor_output_list[level_index], dim=[0, 1])  # TODO: only this needs to stay in the forward loop
                input_tensor_vx_output_list[level_index] = vx
                input_tensor_vy_output_list[level_index] = vy

        ### Run ECC algorithm for each level of the pyramid: ###
        for level_index in np.arange(self.number_of_levels, 0, -1):  # start with lowest resolution (highest level of the pyramid)
            ### Get Current Level input_tensor and reference_tensor: ###
            start_index = sub_sequence_index * self.number_of_images_per_batch
            stop_index = start_index + self.number_of_images_per_batch
            current_level_input_tensor = input_tensor_output_list[level_index - 1][start_index:stop_index]
            current_level_reference_tensor = self.reference_tensor_output_list[level_index - 1]
            if len(current_level_reference_tensor.shape) == 4:
                T, C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 3:
                C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 2:
                H, W = current_level_input_tensor.shape

            ### Get input_tensor gradients: ###
            vx = input_tensor_vx_output_list[level_index - 1][start_index:stop_index]
            vy = input_tensor_vy_output_list[level_index - 1][start_index:stop_index]

            ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
            x_vec = self.x_vec_list[level_index - 1]
            y_vec = self.y_vec_list[level_index - 1]
            X_mat = self.X_mat_list[level_index - 1]
            Y_mat = self.Y_mat_list[level_index - 1]
            current_level_reference_tensor = current_level_reference_tensor.type(torch.float32)

            ### Get Current Level Jacobian Auxiliary Tensors: ###
            # TODO: i wonder if all these assignments take anything up...i think NOT!
            Jx = self.Jx_list[level_index - 1]
            Jy = self.Jy_list[level_index - 1]
            J0 = self.J0_list[level_index - 1]
            J1 = self.J1_list[level_index - 1]

            ### Get Current Level Tensors Chosen Values On Grid (according to gradient high points): ###
            Jx_chosen_values = self.Jx_chosen_values_list[level_index - 1]
            Jy_chosen_values = self.Jy_chosen_values_list[level_index - 1]
            J1_chosen_values = self.J1_chosen_values_list[level_index - 1]
            J0_chosen_values = self.J0_chosen_values_list[level_index - 1]
            X_mat_chosen_values = self.X_mat_chosen_values_list[level_index - 1]
            Y_mat_chosen_values = self.Y_mat_chosen_values_list[level_index - 1]
            reference_tensor_chosen_values = self.reference_tensor_chosen_values_list[level_index - 1]

            ### Initialize H matrix: ###
            if delta_p_init is None:
                H_matrix = self.H_matrix
            else:
                H_matrix = delta_p_init

            ### ECC, Forward Additive Algorithm: ###
            for iteration_index in np.arange(self.number_of_iterations_per_level):
                if flag_print:
                    print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))

                ### Interpolations Dudy: ###
                current_level_input_tensor_warped, \
                    current_level_reference_tensor_zero_mean,\
                    vx_warped,\
                    vy_warped = self.interpolations_dudy(current_level_input_tensor,
                                                         reference_tensor_chosen_values, vx, vy,
                                                         H_matrix, H, W, X_mat_chosen_values, Y_mat_chosen_values)

                ####################################################################################################################################
                ### Compute the jacobian of warp transform_string: ###
                J_list = self.get_jacobian_for_warp_transform_points_torch(x_vec + 1, y_vec + 1, H_matrix,
                                                                           Jx, Jy, J0, J1,
                                                                           self.transform_string, H, W,
                                                                           Jx_chosen_values.squeeze(),
                                                                           Jy_chosen_values.squeeze(),
                                                                           J0_chosen_values,
                                                                           J1_chosen_values)
                Jx_chosen_values, Jy_chosen_values,\
                    J0_chosen_values, J1_chosen_values,\
                    Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime = J_list

                ### Compute the jacobian of warped image wrt parameters (matrix G in the paper): ###
                G_list, C = self.image_jacobian_points_torch(vx_warped, vy_warped, J_list, self.number_of_parameters)
                G0, G1, G2, G3, G4, G5, G6, G7 = G_list

                ### Coompute Hessian and its inverse: ###
                i_C = torch.linalg.inv(C)

                ### Compute projections of images into G: ###
                # (*). Calculate Gt:
                # self.Gt = 0 * self.Gt
                self.Gt[:, 0] = (G0 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                self.Gt[:, 1] = (G1 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                self.Gt[:, 2] = (G2 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                self.Gt[:, 3] = (G3 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                self.Gt[:, 4] = (G4 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                self.Gt[:, 5] = (G5 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                self.Gt[:, 6] = (G6 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                self.Gt[:, 7] = (G7 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                # (*). alculate Gw:
                self.Gw[:, 0] = (G0 * current_level_input_tensor_warped).sum([-1,-2])
                self.Gw[:, 1] = (G1 * current_level_input_tensor_warped).sum([-1,-2])
                self.Gw[:, 2] = (G2 * current_level_input_tensor_warped).sum([-1,-2])
                self.Gw[:, 3] = (G3 * current_level_input_tensor_warped).sum([-1,-2])
                self.Gw[:, 4] = (G4 * current_level_input_tensor_warped).sum([-1,-2])
                self.Gw[:, 5] = (G5 * current_level_input_tensor_warped).sum([-1,-2])
                self.Gw[:, 6] = (G6 * current_level_input_tensor_warped).sum([-1,-2])
                self.Gw[:, 7] = (G7 * current_level_input_tensor_warped).sum([-1,-2])

                ### ECC Closed Form Solution: ###
                # (1). compute lambda parameter:
                # TODO: maybe the norm of the warped tensor remains approximately the same and i can skip this stage????
                num = (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1,-2))).unsqueeze(-1) ** 2 - torch.transpose(self.Gw, -1, -2) @ i_C @ self.Gw
                den = (current_level_input_tensor_warped * current_level_reference_tensor_zero_mean).sum([-1, 2]).unsqueeze(-1) - torch.transpose(self.Gt, -1, -2) @ i_C @ self.Gw
                lambda_correction = (num / den).unsqueeze(-1)

                # (2). compute error vector:
                imerror = lambda_correction * current_level_reference_tensor_zero_mean - current_level_input_tensor_warped
                # (3). compute the projection of error vector into Jacobian G:
                self.Ge[:, 0] = (G0 * imerror).sum([-1,-2])
                self.Ge[:, 1] = (G1 * imerror).sum([-1,-2])
                self.Ge[:, 2] = (G2 * imerror).sum([-1,-2])
                self.Ge[:, 3] = (G3 * imerror).sum([-1,-2])
                self.Ge[:, 4] = (G4 * imerror).sum([-1,-2])
                self.Ge[:, 5] = (G5 * imerror).sum([-1,-2])
                self.Ge[:, 6] = (G6 * imerror).sum([-1,-2])
                self.Ge[:, 7] = (G7 * imerror).sum([-1,-2])
                # (4). compute the optimum parameter correction vector:
                delta_p = torch.matmul(i_C, self.Ge)
                # gtoc('delta_p dudy')

                # delta_p_norm = torch.norm(delta_p, dim=1)
                # gtoc('bla2')
                ####################################################################################################################################


                ### Update Parameters: ###
                H_matrix_previous = H_matrix * 1.0
                H_matrix = update_transform_params_torch(H_matrix, delta_p, self.transform_string)

                ### Break the loop if reached max number of iterations per level: ###
                max_difference = estimate_difference_between_homography_matrices(H_matrix_previous, H_matrix, H=H, W=W)
                # print('iteration: ' + str(iteration_index) + ',  max elisheva difference: ' + str(max_difference.max().item()))
                flag_end_iterations = (iteration_index == self.number_of_iterations_per_level) or (max_difference < max_shift_threshold).all()
                if flag_end_iterations:  # the algorithm is executed (number_of_iterations_per_level-1) times
                    # print('stopped due to condition')
                    # print(iteration_index)
                    break

                # print(H_matrix)
                # print(delta_p)

            print('iteration number : ' + str(iteration_index))

            ### END OF INTERNAL ITERATIONS (PER LEVEL) LOOP
        ### END OF PYRAMID number_of_levels LOOP:
        # print('number of iterations this batch: ' + str(iteration_index))

        ### Get final H_matrix: ###
        final_warp = H_matrix

        ### return the final warped image using the whole support area (including margins): ###
        # TODO: fix this to allow for B,C,H,W ; B,T,C,H,W; C,H,W; H,W
        nx2 = self.x_vec_list[0]
        ny2 = self.y_vec_list[0]
        H, W = input_tensor.shape[-2:]
        C_input = 1
        self.warped_image, bilinear_grid = self.spatial_interpolation_torch(input_tensor, final_warp, 'bicubic', self.transform_string, nx2, ny2, self.X_mat_list[0], self.Y_mat_list[0], H, W)
        # for ii in torch.arange(C_input):
        #     warpedImage[ii, :, :] = spatial_interpolation_torch(input_tensor[ii, :, :], final_warp, 'linear', self.transform_string, nx2, ny2, H, W)
        H_matrix = final_warp
        return H_matrix, self.warped_image

    def forward_yuri_v3(self, input_tensor, reference_tensor, max_shift_threshold=2e-3, flag_print=False,
                delta_p_init=None,
                sub_sequence_index=0,
                input_tensor_output_list=None,
                input_tensor_vx_output_list=None,
                input_tensor_vy_output_list=None,
                flag_calculate_gradient_every_forward=True):
        ### Initialize Things For Subsequence Runs: ###
        if self.reference_tensor_output_list is None:
            self.initialize_things_for_first_run(input_tensor, reference_tensor)

        ### Assign previous H matrix for later stoppage condition: ###
        H_matrix_previous = None

        ### Calculate Gradients Vx & Vy: ###
        if flag_calculate_gradient_every_forward:
            input_tensor_output_list = [0] * self.number_of_levels
            input_tensor_vx_output_list = [0] * self.number_of_levels
            input_tensor_vy_output_list = [0] * self.number_of_levels
            input_tensor_output_list[0] = input_tensor
            [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2, -1])  # TODO: i actually don't really really need to do this as i only need the gradients at and near the sampled points, no?
            input_tensor_vx_output_list[0] = vx
            input_tensor_vy_output_list[0] = vy
            # (2). Subsequence Levels (Lower Resolutions):
            for level_index in np.arange(1, self.number_of_levels):
                ### Shape: ###
                H, W = input_tensor_output_list[level_index - 1].shape

                ### Interpolate: ###
                input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

                ### Get Gradients: ###
                # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
                [vy, vx] = torch.gradient(input_tensor_output_list[level_index], dim=[0, 1])  # TODO: only this needs to stay in the forward loop
                input_tensor_vx_output_list[level_index] = vx
                input_tensor_vy_output_list[level_index] = vy

        ### Run ECC algorithm for each level of the pyramid: ###
        for level_index in np.arange(self.number_of_levels, 0, -1):  # start with lowest resolution (highest level of the pyramid)
            ### Get Current Level input_tensor and reference_tensor: ###
            start_index = sub_sequence_index * self.number_of_images_per_batch
            stop_index = start_index + self.number_of_images_per_batch
            current_level_input_tensor = input_tensor_output_list[level_index - 1][start_index:stop_index]
            current_level_reference_tensor = self.reference_tensor_output_list[level_index - 1]
            if len(current_level_reference_tensor.shape) == 4:
                T, C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 3:
                C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 2:
                H, W = current_level_input_tensor.shape

            ### Get input_tensor gradients: ###
            vx = input_tensor_vx_output_list[level_index - 1][start_index:stop_index]
            vy = input_tensor_vy_output_list[level_index - 1][start_index:stop_index]

            ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
            x_vec = self.x_vec_list[level_index - 1]
            y_vec = self.y_vec_list[level_index - 1]
            X_mat = self.X_mat_list[level_index - 1]
            Y_mat = self.Y_mat_list[level_index - 1]
            current_level_reference_tensor = current_level_reference_tensor.type(torch.float32)

            ### Get Current Level Jacobian Auxiliary Tensors: ###
            # TODO: i wonder if all these assignments take anything up...i think NOT!
            Jx = self.Jx_list[level_index - 1]
            Jy = self.Jy_list[level_index - 1]
            J0 = self.J0_list[level_index - 1]
            J1 = self.J1_list[level_index - 1]

            ### Get Current Level Tensors Chosen Values On Grid (according to gradient high points): ###
            Jx_chosen_values = self.Jx_chosen_values_list[level_index - 1]
            Jy_chosen_values = self.Jy_chosen_values_list[level_index - 1]
            J1_chosen_values = self.J1_chosen_values_list[level_index - 1]
            J0_chosen_values = self.J0_chosen_values_list[level_index - 1]
            X_mat_chosen_values = self.X_mat_chosen_values_list[level_index - 1]
            Y_mat_chosen_values = self.Y_mat_chosen_values_list[level_index - 1]
            reference_tensor_chosen_values = self.reference_tensor_chosen_values_list[level_index - 1]

            ### Initialize H matrix: ###
            if delta_p_init is None:
                H_matrix = self.H_matrix
            else:
                H_matrix = delta_p_init

            ### ECC, Forward Additive Algorithm: ###
            for iteration_index in np.arange(self.number_of_iterations_per_level):
                if flag_print:
                    print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))

                # # ### Interpolation Yuri: ###
                # gtic()
                # current_level_input_tensor_warped_yuri, vx_warped_yuri, vy_warped_yuri = ecc_bilinear_interpolation.ecc_bilinear_interpolation_no_grad(current_level_input_tensor.squeeze(),
                #                                                                   H_matrix,
                #                                                                   X_mat_chosen_values,
                #                                                                   Y_mat_chosen_values)
                # gtoc('interpolations yuri')

                ### Interpolation Yuri: ###
                current_level_input_tensor_warped, current_level_reference_tensor_zero_mean, vx_warped, vy_warped = \
                    self.interpolations_yuri(current_level_input_tensor, reference_tensor_chosen_values,
                                             vx, vy,
                                             H_matrix, H, W,
                                             X_mat_chosen_values, Y_mat_chosen_values)

                # ### Interpolation Dudy: ###
                # current_level_input_tensor_warped, bilinear_grid = self.spatial_interpolation_points_torch(None,
                #                                                                                            current_level_input_tensor,
                #                                                                                            H_matrix,
                #                                                                                            'linear',
                #                                                                                            self.transform_string,
                #                                                                                            x_vec, y_vec,
                #                                                                                            X_mat, Y_mat,
                #                                                                                            H, W, None,
                #                                                                                            X_mat_chosen_values,
                #                                                                                            Y_mat_chosen_values)  # inverse(backward) warping
                # current_level_input_tensor_warped = current_level_input_tensor_warped.squeeze(-1).unsqueeze(1)
                # vx_warped, bilinear_grid = self.spatial_interpolation_points_torch(None, vx, H_matrix, 'linear',
                #                                                                    self.transform_string, x_vec, y_vec,
                #                                                                    X_mat, Y_mat, H, W,
                #                                                                    bilinear_grid, X_mat_chosen_values,
                #                                                                    Y_mat_chosen_values)
                # vy_warped, bilinear_grid = self.spatial_interpolation_points_torch(None, vy, H_matrix, 'linear',
                #                                                                    self.transform_string, x_vec, y_vec,
                #                                                                    X_mat, Y_mat, H, W,
                #                                                                    bilinear_grid, X_mat_chosen_values,
                #                                                                    Y_mat_chosen_values)
                # # plot_torch((current_level_input_tensor_warped.squeeze() - current_level_input_tensor_warped_yuri)[0])
                # plot_torch((vx_warped.squeeze() - vx_warped_yuri)[0])

                # delta_p = calc_delta_p.ecc_calc_delta_p(vx_warped, vy_warped,
                #                                         Jx, Jy,
                #                                         Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime,
                #                                         current_level_reference_tensor_zero_mean,
                #                                         current_level_input_tensor_warped)

                # gtic()
                current_level_reference_tensor_zero_mean = reference_tensor_chosen_values
                delta_p = calc_delta_p_v3.ecc_calc_delta_p(H_matrix,
                     current_level_reference_tensor_zero_mean,
                     current_level_input_tensor_warped.squeeze().contiguous(),
                     Jx_chosen_values, Jy_chosen_values,
                     vx_warped.squeeze(), vy_warped.squeeze())
                # gtoc('delta_p yuri')

                ### Update Parameters: ###
                # gtic()
                H_matrix_previous = H_matrix * 1.0
                H_matrix = update_transform_params_torch(H_matrix, delta_p, self.transform_string)
                # gtoc('update H')

                ### Break the loop if reached max number of iterations per level: ###
                # gtic()
                max_difference = estimate_difference_between_homography_matrices(H_matrix_previous, H_matrix, H=H, W=W)
                # print('iteration: ' + str(iteration_index) + ',  max elisheva difference: ' + str(max_difference.max().item()))
                flag_end_iterations = (iteration_index == self.number_of_iterations_per_level) or (max_difference < max_shift_threshold).all()
                if flag_end_iterations:  # the algorithm is executed (number_of_iterations_per_level-1) times
                    # print('stopped due to condition')
                    # print(iteration_index)
                    break
                # gtoc('difference between H')
                # print(H_matrix)
                # print(delta_p)

            print('iteration number: ' + str(iteration_index))

            ### END OF INTERNAL ITERATIONS (PER LEVEL) LOOP
        ### END OF PYRAMID number_of_levels LOOP:
        # print('number of iterations this batch: ' + str(iteration_index))

        ### Get final H_matrix: ###
        final_warp = H_matrix

        ### return the final warped image using the whole support area (including margins): ###
        # TODO: fix this to allow for B,C,H,W ; B,T,C,H,W; C,H,W; H,W
        nx2 = self.x_vec_list[0]
        ny2 = self.y_vec_list[0]
        H, W = input_tensor.shape[-2:]
        C_input = 1
        self.warped_image, bilinear_grid = self.spatial_interpolation_torch(input_tensor, final_warp, 'bicubic', self.transform_string, nx2, ny2, self.X_mat_list[0], self.Y_mat_list[0], H, W)
        # for ii in torch.arange(C_input):
        #     warpedImage[ii, :, :] = spatial_interpolation_torch(input_tensor[ii, :, :], final_warp, 'linear', self.transform_string, nx2, ny2, H, W)
        H_matrix = final_warp
        return H_matrix, self.warped_image



def MM_with_transpose_einsum(input_tensor_1, input_tensor_2):
    # [input_tensor_1] = [H,W]
    # [input_tensor_2] = [H,W]
    # output_tensor = torch.transpose(input_tensor, -1, -2) * input_tensor_2
    return torch.einsum('k i, k j -> i j', input_tensor_1, input_tensor_2)


def BMM_with_transpose_einsum(input_tensor_1, input_tensor_2):
    # [input_tensor_1] = [B,H,W]
    # [input_tensor_2] = [B.H,W]
    # output_tensor = torch.transpose(input_tensor, -1, -2) * input_tensor_2
    return torch.einsum('b k i, b k j -> b i j', input_tensor_1, input_tensor_2)





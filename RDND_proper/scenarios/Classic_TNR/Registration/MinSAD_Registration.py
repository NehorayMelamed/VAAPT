from RapidBase.import_all import *
from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *

class minSAD_Bilinaer_Affine_Registration_Layer(nn.Module):
    def __init__(self, *args):
        super(minSAD_Bilinaer_Affine_Registration_Layer, self).__init__()

        ### Parameters Space: ###
        self.shifts_vec = [-1, 0, 1]
        self.rotation_angle_vec = [-1, 0, 1]
        self.scale_factor_vec = [1 - 1e-6, 1, 1 + 1e-6]

        ### Initialize Warp Objects: ###
        self.affine_layer_torch = Warp_Tensors_Affine_Layer()

        ### Pre-Allocate Flow Grid: ###
        self.L_shifts = len(self.shifts_vec)
        self.L_rotation_angle = len(self.rotation_angle_vec)
        self.L_scale = len(self.scale_factor_vec)
        self.number_of_possible_warps = self.L_shifts * self.L_shifts * self.L_rotation_angle * self.L_scale

        self.B = None
        self.C = None
        self.H = None
        self.W = None

    def forward(self, input_image_1, input_image_2):
        ############################################################################################################################################################

        # shifts_vec = my_linspace(-3, 3, 5)
        # rotation_angle_vec = [-4, -2, 0, 2, 4]
        # scale_factor_vec = [0.95, 1, 1.05]
        final_crop_size = (800, 800)

        if self.B is None:
            ### Initialize: ###
            self.B, self.C, self.H, self.W = input_image_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec),len(self.scale_factor_vec))).to(input_image_2.device)
            SAD_matrix_numpy = self.SAD_matrix.cpu().numpy()


        crop_H_start = (self.H - final_crop_size[0]) // 2
        crop_H_final = crop_H_start + final_crop_size[0]
        crop_W_start = (self.W - final_crop_size[1]) // 2
        crop_W_final = crop_W_start + final_crop_size[1]

        ### Build All Possible Grids: ## #TODO: accelerate this!!!!
        counter = 0
        for shift_x_counter, current_shift_x in enumerate(self.shifts_vec):
            for shift_y_counter, current_shift_y in enumerate(self.shifts_vec):
                for rotation_counter, current_rotation_angle in enumerate(self.rotation_angle_vec):
                    for scale_counter, current_scale_factor in enumerate(self.scale_factor_vec):
                        ### Warp Previous Image Many Ways To Find Best Fit: ###
                        current_shift_x = np.float32(current_shift_x)
                        current_shift_y = np.float32(current_shift_y)
                        current_scale_factor = np.float32(current_scale_factor)
                        current_rotation_angle = np.float32(current_rotation_angle)
                        input_image_2_warped, current_flow_grid = self.affine_layer_torch.forward(input_image_2,
                                                                                             current_shift_x,
                                                                                             current_shift_y,
                                                                                             current_scale_factor,
                                                                                             current_rotation_angle,
                                                                                             return_flow_grid=True,
                                                                                             flag_interpolation_mode='bilinear')
                        self.Flow_Grids[counter] = current_flow_grid
                        counter = counter + 1

        ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
        possible_warps_tensor = input_image_1.repeat((self.number_of_possible_warps, 1, 1, 1))

        ### Get All Possible Warps Of Previous Image: ###
        possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, self.Flow_Grids, mode='bilinear')
        # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

        ### Get Min SAD: ###
        # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
        SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_2)[:, :, :, :].mean(-1, True).mean(-2, True)
        min_index_2 = torch.argmin(SAD_matrix_2)
        min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
        min_indices = np.unravel_index(min_index_2, SAD_matrix_numpy.shape)
        shift_x_inference = self.shifts_vec[min_indices[0]]
        shift_y_inference = self.shifts_vec[min_indices[1]]
        rotation_angle_inference = self.rotation_angle_vec[min_indices[2]]
        scale_factor_inference = self.scale_factor_vec[min_indices[3]]

        ### Correct Sub-Pixel: ###
        # (1). Get Min Indices:
        SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (self.L_shifts, self.L_shifts, self.L_rotation_angle, self.L_scale))
        x_vec = [-1, 0, 1]
        y_vec = [-1, 0, 1]
        shift_x_index = min_indices[0]
        shift_y_index = min_indices[1]
        rotation_index = min_indices[2]
        scale_index = min_indices[3]
        # (2). Get Indices For Parabola Fitting:   #TODO: make sure this is correct!!!!!! use the cross correlation parabola fit corrections!!!!
        shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
        shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
        rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
        scale_indices = np.array([scale_index - 1, scale_index, scale_index + 1])
        shift_x_indices[shift_x_indices < 0] += self.L_shifts
        shift_x_indices[shift_x_indices >= self.L_shifts] -= self.L_shifts
        shift_y_indices[shift_y_indices < 0] += self.L_shifts
        shift_y_indices[shift_y_indices >= self.L_shifts] -= self.L_shifts
        rotation_indices[rotation_indices < 0] += self.L_rotation_angle
        rotation_indices[rotation_indices >= self.L_rotation_angle] -= self.L_rotation_angle
        scale_indices[scale_indices < 0] += self.L_scale
        scale_indices[scale_indices >= self.L_scale] -= self.L_scale

        #### Fot Shift X Parabola: ###
        fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index, scale_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
        delta_shift_x = -b_x / (2 * a_x)
        shift_x_sub_pixel = self.shifts_vec[shift_x_index] + delta_shift_x * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Shit Y Parabola: ###
        fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index, scale_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
        delta_shift_y = -b_x / (2 * a_x)
        shift_y_sub_pixel = self.shifts_vec[shift_y_index] + delta_shift_y * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Rotation Parabola: ###
        fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices, scale_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
        delta_rotation = -b_x / (2 * a_x)
        rotation_sub_pixel = self.rotation_angle_vec[rotation_index] + delta_rotation * (self.rotation_angle_vec[2] - self.rotation_angle_vec[1])

        ### Fit Scale Parabola: ###
        fitting_points_scale = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_index, scale_indices]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_scale)
        delta_scale = -b_x / (2 * a_x)
        scale_sub_pixel = self.scale_factor_vec[scale_index] + delta_scale * (self.scale_factor_vec[2] - self.scale_factor_vec[1])

        return shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, scale_sub_pixel


class minSAD_FFT_Affine_Registration_Layer(nn.Module):
    def __init__(self, *args):
        super(minSAD_FFT_Affine_Registration_Layer, self).__init__()

        ### Parameters Space: ###
        # self.shifts_vec = [-1, 0, 1]
        # self.rotation_angle_vec = [-1, 0, 1]
        self.shifts_vec = np.arange(-2,3)
        self.rotation_angle_vec = np.arange(-10,11)

        ### Initialize Warp Objects: ###
        # self.affine_layer_torch = Warp_Tensors_Affine_Layer()
        self.affine_layer_torch = FFT_Translation_Rotation_Layer()

        ### Pre-Allocate Flow Grid: ###
        self.L_shifts = len(self.shifts_vec)
        self.L_rotation_angle = len(self.rotation_angle_vec)
        self.number_of_possible_warps = self.L_shifts * self.L_shifts * self.L_rotation_angle

        self.B = None
        self.C = None
        self.H = None
        self.W = None

    def forward(self, input_image_1, input_image_2):
        ############################################################################################################################################################

        # shifts_vec = my_linspace(-3, 3, 5)
        # rotation_angle_vec = [-4, -2, 0, 2, 4]
        # scale_factor_vec = [0.95, 1, 1.05]
        final_crop_size = (800, 800)

        if self.B is None:
            ### Initialize: ###
            self.B, self.C, self.H, self.W = input_image_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec))).to(input_image_2.device)
            SAD_matrix_numpy = self.SAD_matrix.cpu().numpy()


        crop_H_start = (self.H - final_crop_size[0]) // 2
        crop_H_final = crop_H_start + final_crop_size[0]
        crop_W_start = (self.W - final_crop_size[1]) // 2
        crop_W_final = crop_W_start + final_crop_size[1]

        ### Build All Possible Grids: ## #TODO: accelerate this!!!!
        counter = 0
        image2_warped_list = []
        for shift_x_counter, current_shift_x in enumerate(self.shifts_vec):
            for shift_y_counter, current_shift_y in enumerate(self.shifts_vec):
                for rotation_counter, current_rotation_angle in enumerate(self.rotation_angle_vec):
                    ### Warp Previous Image Many Ways To Find Best Fit: ###
                    current_shift_x = np.float32(current_shift_x)
                    current_shift_y = np.float32(current_shift_y)
                    current_rotation_angle = np.float32(current_rotation_angle)

                    current_shift_x_tensor = torch.Tensor([current_shift_x])
                    current_shift_y_tensor = torch.Tensor([current_shift_y])
                    current_rotation_angle_tensor = torch.Tensor([current_rotation_angle]) * pi/180

                    ### Perform Affine Transform In FFT Space: ###
                    input_image_2_warped = self.affine_layer_torch.forward(input_image_2,
                                                                             current_shift_x_tensor,
                                                                             current_shift_y_tensor,
                                                                             current_rotation_angle_tensor)
                    image2_warped_list.append(input_image_2_warped)
                    counter = counter + 1

        ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
        input_image_1_stacked = input_image_1.repeat((self.number_of_possible_warps, 1, 1, 1))
        image2_different_warps_tensor = torch.cat(image2_warped_list, 0)
        input_image_1_stacked = crop_torch_batch(input_image_1_stacked, (int(self.W*0.9), int(self.H*0.9))).cuda()
        image2_different_warps_tensor = crop_torch_batch(image2_different_warps_tensor, (int(self.W*0.9), int(self.H*0.9))).cuda()

        ### Get Min SAD: ###
        # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
        SAD_matrix_2 = torch.abs(input_image_1_stacked - image2_different_warps_tensor)[:, :, :, :].mean(-1, True).mean(-2, True)
        min_index_2 = torch.argmin(SAD_matrix_2)
        min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
        min_indices = np.unravel_index(min_index_2, (len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec)))
        shift_x_inference = self.shifts_vec[min_indices[0]]
        shift_y_inference = self.shifts_vec[min_indices[1]]
        rotation_angle_inference = self.rotation_angle_vec[min_indices[2]]

        ### Correct Sub-Pixel: ###
        # (1). Get Min Indices:
        SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (self.L_shifts, self.L_shifts, self.L_rotation_angle))
        x_vec = [-1, 0, 1]
        y_vec = [-1, 0, 1]
        shift_x_index = min_indices[0]
        shift_y_index = min_indices[1]
        rotation_index = min_indices[2]
        # (2). Get Indices For Parabola Fitting:   #TODO: make sure this is correct!!!!!! use the cross correlation parabola fit corrections!!!!
        shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
        shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
        rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
        shift_x_indices[shift_x_indices < 0] += self.L_shifts
        shift_x_indices[shift_x_indices >= self.L_shifts] -= self.L_shifts
        shift_y_indices[shift_y_indices < 0] += self.L_shifts
        shift_y_indices[shift_y_indices >= self.L_shifts] -= self.L_shifts
        rotation_indices[rotation_indices < 0] += self.L_rotation_angle
        rotation_indices[rotation_indices >= self.L_rotation_angle] -= self.L_rotation_angle

        #### Fot Shift X Parabola: ###
        fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
        delta_shift_x = -b_x / (2 * a_x)
        shift_x_sub_pixel = self.shifts_vec[shift_x_index] + delta_shift_x * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Shit Y Parabola: ###
        fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
        delta_shift_y = -b_x / (2 * a_x)
        shift_y_sub_pixel = self.shifts_vec[shift_y_index] + delta_shift_y * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Rotation Parabola: ###
        fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
        delta_rotation = -b_x / (2 * a_x)
        rotation_sub_pixel = self.rotation_angle_vec[rotation_index] + delta_rotation * (self.rotation_angle_vec[2] - self.rotation_angle_vec[1])
        rotation_sub_pixel = rotation_sub_pixel * pi/180

        ### Transform Second Image To Align With First: ###
        input_image_2_aligned = self.affine_layer_torch.forward(input_image_2,
                                                               -shift_x_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               -shift_y_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               -rotation_sub_pixel.unsqueeze(0).unsqueeze(0))
        input_image_2_aligned = crop_torch_batch(input_image_2_aligned, (self.W, self.H)).cuda()

        return shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, input_image_2_aligned


def get_scale_rotation_translation_minSAD_affine(input_image_1, input_image_2):
    ############################################################################################################################################################
    ### Sizes: ###
    H_initial = 500  # initial crop size
    W_initial = 500  # initial crop size
    H_final = 500
    W_final = 500
    initial_index_H_final = np.int((H_initial - H_final) / 2)
    initial_index_W_final = np.int((W_initial - W_final) / 2)
    final_index_H_final = initial_index_H_final + H_final
    final_index_W_final = initial_index_W_final + W_final

    ### Get Image: ###
    previous_image = read_image_default_torch()
    previous_image = RGB2BW(previous_image)
    previous_image = previous_image[:,:,0:H_initial,0:W_initial].cuda()
    # previous_image = crop_torch_batch(previous_image, (H_initial, W_initial)).cuda()

    ### GT parameters: ###
    GT_shift_x = np.float32(0.2)
    GT_shift_y = np.float32(-0.4)
    GT_rotation_angle = np.float32(0.1)
    GT_scale = np.float32(1.00)

    ### Initialize Warp Objects: ###
    affine_layer_torch = Warp_Tensors_Affine_Layer()

    ### Warp Image According To Above Parameters: ###
    current_image = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bilinear')
    # current_image_2 = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bicubic')

    ### Final Crop: ###
    # current_image = crop_torch_batch(current_image, (H_final, W_final))
    # previous_image = crop_torch_batch(previous_image, (H_final, W_final))
    # current_image = current_image[:, :, 0:H_final, 0:W_final]
    # previous_image = previous_image[:, :, 0:H_final, 0:W_final]
    input_image_1 = previous_image
    input_image_2 = current_image
    ############################################################################################################################################################

    ############################################################################################################################################################
    ### Parameters Space: ###
    shifts_vec = [-1, 0, 1]
    rotation_angle_vec = [-1, 0, 1]
    scale_factor_vec = [1-1e-6, 1, 1+1e-6]
    # shifts_vec = my_linspace(-3, 3, 5)
    # rotation_angle_vec = [-4, -2, 0, 2, 4]
    # scale_factor_vec = [0.95, 1, 1.05]
    final_crop_size = (800,800)

    ### Initialize Warp Objects: ###
    affine_layer_torch = Warp_Tensors_Affine_Layer()

    ### Pre-Allocate Flow Grid: ###
    L_shifts = len(shifts_vec)
    L_rotation_angle = len(rotation_angle_vec)
    L_scale = len(scale_factor_vec)
    number_of_possible_warps = L_shifts * L_shifts * L_rotation_angle * L_scale
    B, C, H, W = input_image_1.shape
    Flow_Grids = torch.zeros(number_of_possible_warps, H, W, 2).cuda()

    crop_H_start = (H-final_crop_size[0]) // 2
    crop_H_final = crop_H_start + final_crop_size[0]
    crop_W_start = (W - final_crop_size[1]) // 2
    crop_W_final = crop_W_start + final_crop_size[1]

    ### Loop over all Possible Parameters: ###
    SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec), len(scale_factor_vec))).to(input_image_2.device)
    SAD_matrix_numpy = SAD_matrix.cpu().numpy()

    ### Build All Possible Grids: ## #TODO: accelerate this!!!!
    counter = 0
    for shift_x_counter, current_shift_x in enumerate(shifts_vec):
        for shift_y_counter, current_shift_y in enumerate(shifts_vec):
            for rotation_counter, current_rotation_angle in enumerate(rotation_angle_vec):
                for scale_counter, current_scale_factor in enumerate(scale_factor_vec):
                    ### Warp Previous Image Many Ways To Find Best Fit: ###
                    current_shift_x = np.float32(current_shift_x)
                    current_shift_y = np.float32(current_shift_y)
                    current_scale_factor = np.float32(current_scale_factor)
                    current_rotation_angle = np.float32(current_rotation_angle)
                    input_image_2_warped, current_flow_grid = affine_layer_torch.forward(input_image_2,
                                                                                          current_shift_x,
                                                                                          current_shift_y,
                                                                                          current_scale_factor,
                                                                                          current_rotation_angle,
                                                                                          return_flow_grid=True,
                                                                                          flag_interpolation_mode='bilinear')
                    Flow_Grids[counter] = current_flow_grid
                    counter = counter + 1


    ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
    possible_warps_tensor = input_image_1.repeat((number_of_possible_warps, 1, 1, 1))

    ### Get All Possible Warps Of Previous Image: ###
    possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, Flow_Grids, mode='bilinear')
    # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

    ### Get Min SAD: ###
    # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
    SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_2)[:,:, :, :].mean(-1, True).mean(-2, True)
    min_index_2 = torch.argmin(SAD_matrix_2)
    min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
    min_indices = np.unravel_index(min_index_2, SAD_matrix_numpy.shape)
    shift_x_inference = shifts_vec[min_indices[0]]
    shift_y_inference = shifts_vec[min_indices[1]]
    rotation_angle_inference = rotation_angle_vec[min_indices[2]]
    scale_factor_inference = scale_factor_vec[min_indices[3]]

    ### Correct Sub-Pixel: ###
    #(1). Get Min Indices:
    SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (L_shifts, L_shifts, L_rotation_angle, L_scale))
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    shift_x_index = min_indices[0]
    shift_y_index = min_indices[1]
    rotation_index = min_indices[2]
    scale_index = min_indices[3]
    #(2). Get Indices For Parabola Fitting:
    shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
    shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
    rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
    scale_indices = np.array([scale_index - 1, scale_index, scale_index + 1])
    shift_x_indices[shift_x_indices < 0] += L_shifts
    shift_x_indices[shift_x_indices >= L_shifts] -= L_shifts
    shift_y_indices[shift_y_indices < 0] += L_shifts
    shift_y_indices[shift_y_indices >= L_shifts] -= L_shifts
    rotation_indices[rotation_indices < 0] += L_rotation_angle
    rotation_indices[rotation_indices >= L_rotation_angle] -= L_rotation_angle
    scale_indices[scale_indices < 0] += L_scale
    scale_indices[scale_indices >= L_scale] -= L_scale

    #### Fot Shift X Parabola: ###
    fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
    delta_shift_x = -b_x / (2 * a_x)
    shift_x_sub_pixel = shifts_vec[shift_x_index] + delta_shift_x * (shifts_vec[2] - shifts_vec[1])

    ### Fit Shit Y Parabola: ###
    fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
    delta_shift_y = -b_x / (2 * a_x)
    shift_y_sub_pixel = shifts_vec[shift_y_index] + delta_shift_y * (shifts_vec[2] - shifts_vec[1])

    ### Fit Rotation Parabola: ###
    fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
    delta_rotation = -b_x / (2 * a_x)
    rotation_sub_pixel = rotation_angle_vec[rotation_index] + delta_rotation * (rotation_angle_vec[2] - rotation_angle_vec[1])

    ### Fit Scale Parabola: ###
    fitting_points_scale = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_index, scale_indices]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_scale)
    delta_scale = -b_x / (2 * a_x)
    scale_sub_pixel = scale_factor_vec[scale_index] + delta_scale * (scale_factor_vec[2] - scale_factor_vec[1])

    print('recovered shift x: ' + str(shift_x_sub_pixel))
    print('recovered shift y: ' + str(shift_y_sub_pixel))
    print('recovered rotation: ' + str(rotation_sub_pixel))
    print('recovered scale: ' + str(scale_sub_pixel))


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

### Sizes: ###
H_initial = 1000 #initial crop size
W_initial = 1000 #initial crop size
H_final = 500
W_final = 500
initial_index_H_final = np.int((H_initial-H_final)/2)
initial_index_W_final = np.int((W_initial-W_final)/2)
final_index_H_final = initial_index_H_final + H_final
final_index_W_final = initial_index_W_final + W_final

### Get Image: ###
previous_image = read_image_default_torch()
previous_image = RGB2BW(previous_image)
previous_image = previous_image[:,:,0:H_initial,0:W_initial].cuda()
# previous_image = crop_torch_batch(previous_image, (H_initial, W_initial)).cuda()

### GT parameters: ###
GT_shift_x = np.float32(1.3)
GT_shift_y = np.float32(-1.5)
GT_rotation_angle = np.float32(2)
GT_scale = np.float32(1.04)

### Parameters Space: ###
shifts_vec = my_linspace(-3,3,5)
rotation_angle_vec = [-4,-2,0,2,4]  #[degrees]
scale_factor_vec = [0.95,1,1.05]
number_of_affine_parameters = 4  #delta_x, delta_y, rotation_angle, scale

### Initialize Warp Objects: ###
shift_layer_torch = Shift_Layer_Torch()
affine_layer_torch = Warp_Tensors_Affine_Layer()

### Warp Image According To Above Parameters: ###
current_image = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle)

### Add Noise: ###
noise_SNR = np.inf
noise_sigma = 1/sqrt(noise_SNR)
current_image += torch.randn_like(current_image) * noise_sigma
previous_image += torch.randn_like(previous_image) * noise_sigma

### Final Crop Center Image: ###
current_image_crop = current_image[:,:,initial_index_H_final:final_index_H_final, initial_index_W_final:final_index_W_final]

### Pre-Allocate Flow Grid: ###
L_shifts = len(shifts_vec)
L_rotation_angle = len(rotation_angle_vec)
L_scale = len(scale_factor_vec)
number_of_possible_warps = L_shifts * L_shifts * L_rotation_angle * L_scale
Flow_Grids = torch.zeros(number_of_possible_warps, H_initial, W_initial, 2).cuda()

### Loop over all Possible Parameters: ###
SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec), len(scale_factor_vec))).to(previous_image.device)
SAD_matrix_numpy = SAD_matrix.cpu().numpy()

counter = 0
#TODO: i should look how to speed things up and be able to build the possible flow grids on the spot to avoid using these loops and in the grand
#TODO: scheme of things -> avoid scale-rotation-shifting of the image before feeding it to this stage
#TODO: the above should be easy i think....simply build an object called "build_affine_grids" or something which accepts a starting point for (scale,rotation,translation)
#TODO: and builds all the possible grids in parallel. should be easy
for shift_x_counter, current_shift_x in enumerate(shifts_vec):
    for shift_y_counter, current_shift_y in enumerate(shifts_vec):
        for rotation_counter, current_rotation_angle in enumerate(rotation_angle_vec):
            for scale_counter, current_scale_factor in enumerate(scale_factor_vec):
                ### Warp Previous Image Many Ways To Find Best Fit: ###
                current_shift_x = np.float32(current_shift_x)
                current_shift_y = np.float32(current_shift_y)
                current_scale_factor = np.float32(current_scale_factor)
                current_rotation_angle = np.float32(current_rotation_angle)
                previous_image_warped, current_flow_grid = affine_layer_torch.forward(previous_image, current_shift_x, current_shift_y,
                                                                              current_scale_factor, current_rotation_angle, return_flow_grid=True)
                Flow_Grids[counter] = current_flow_grid

                ### Final Crop Center Image (we do another crop to avoid non valid regions): ###
                # previous_image_warped_cropped = previous_image_warped[:, :, initial_index_H_final:final_index_H_final, initial_index_W_final:final_index_W_final]
                #
                # current_SAD = torch.abs(current_image_crop - previous_image_warped_cropped).mean()
                # SAD_matrix[shift_x_counter, shift_y_counter, rotation_counter, scale_counter] = current_SAD
                counter = counter + 1

# ### Search For Min in SAD_matrix: ###
# min_index = torch.argmin(SAD_matrix,None)
# min_index = np.atleast_1d(min_index.cpu().numpy())[0]
# L_shifts_x,L_shifts_y,L_rotations,L_scale = SAD_matrix.shape
# min_indices = np.unravel_index(min_index, SAD_matrix_numpy.shape)
# shift_x_inference = shifts_vec[min_indices[0]]
# shift_y_inference = shifts_vec[min_indices[1]]
# rotation_angle_inference = rotation_angle_vec[min_indices[2]]
# scale_factor_inference = scale_factor_vec[min_indices[3]]

############################################################################################################################################################

### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
possible_warps_tensor = previous_image.repeat((number_of_possible_warps,1,1,1))

### Get All Possible Warps Of Previous Image: ###
possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, Flow_Grids, mode='bilinear')

### Get Min SAD: ###
SAD_matrix_2 = torch.abs(possible_warps_tensor - current_image).mean(-1,True).mean(-2,True)
min_index_2 = torch.argmin(SAD_matrix_2)
min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
min_indices = np.unravel_index(min_index_2, SAD_matrix_numpy.shape)
shift_x_inference = shifts_vec[min_indices[0]]
shift_y_inference = shifts_vec[min_indices[1]]
rotation_angle_inference = rotation_angle_vec[min_indices[2]]
scale_factor_inference = scale_factor_vec[min_indices[3]]

### Correct Sub-Pixel: ###
SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (L_shifts,L_shifts,L_rotation_angle,L_scale))
x_vec = [-1, 0, 1]
y_vec =[-1, 0, 1]
shift_x_index = min_indices[0]
shift_y_index = min_indices[1]
rotation_index = min_indices[2]
scale_index = min_indices[3]

shift_x_indices = np.array([shift_x_index-1, shift_x_index, shift_x_index+1])
shift_y_indices = np.array([shift_y_index-1, shift_y_index, shift_y_index+1])
rotation_indices = np.array([rotation_index-1, rotation_index, rotation_index+1])
scale_indices = np.array([scale_index-1, scale_index, scale_index+1])

shift_x_indices[shift_x_indices < 0] += L_shifts
shift_x_indices[shift_x_indices >= L_shifts] -= L_shifts
shift_y_indices[shift_y_indices < 0] += L_shifts
shift_y_indices[shift_y_indices >= L_shifts] -= L_shifts
rotation_indices[rotation_indices < 0] += L_rotation_angle
rotation_indices[rotation_indices >= L_rotation_angle] -= L_rotation_angle
scale_indices[scale_indices < 0] += L_scale
scale_indices[scale_indices >= L_scale] -= L_scale


fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index, scale_index]
y_vec = [-1,0,1]
[c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
delta_shift_x = -b_x / (2 * a_x)
# shift_x_sub_pixel = shift_x_index + delta_shift_x
shift_x_sub_pixel = shifts_vec[shift_x_index] + delta_shift_x * (shifts_vec[2]-shifts_vec[1])

fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index, scale_index]
y_vec = [-1,0,1]
[c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
delta_shift_y = -b_x / (2 * a_x)
# shift_x_sub_pixel = shift_y_index + delta_shift_y
shift_y_sub_pixel = shifts_vec[shift_y_index] + delta_shift_y * (shifts_vec[2]-shifts_vec[1])

fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices, scale_index]
y_vec = [-1,0,1]
[c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
delta_rotation = -b_x / (2 * a_x)
# rotation_sub_pixel = rotation_index + delta_rotation
rotation_sub_pixel = rotation_angle_vec[rotation_index] + delta_rotation * (rotation_angle_vec[2]-rotation_angle_vec[1])

fitting_points_scale = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_index, scale_indices]
y_vec = [-1,0,1]
[c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_scale)
delta_scale = -b_x / (2 * a_x)
# scale_sub_pixel = scale_index + delta_scale
scale_sub_pixel = scale_factor_vec[scale_index] + delta_scale * (scale_factor_vec[2]-scale_factor_vec[1])

print('recovered discrete shift x: ' + str(shifts_vec[shift_x_index]))
print('recovered discrete shift y: '+ str(shifts_vec[shift_y_index]))
print('recovered discrete rotation: ' + str(rotation_angle_vec[rotation_index]))
print('recovered discrete  scale: ' + str(scale_factor_vec[scale_index]))

print('recovered shift x: ' + str(shift_x_sub_pixel))
print('recovered shift y: '+ str(shift_y_sub_pixel))
print('recovered rotation: ' + str(rotation_sub_pixel))
print('recovered scale: ' + str(scale_sub_pixel))
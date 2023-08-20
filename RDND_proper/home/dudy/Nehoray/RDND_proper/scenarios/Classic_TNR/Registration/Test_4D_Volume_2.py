from RapidBase.import_all import *
from torch.utils.data import DataLoader
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.Transforms_Grids import *

########################################################################################################################
### Sizes: ###
H_initial = 200 #initial crop size
W_initial = 200 #initial crop size
H_final = 100
W_final = 100
input_resolution = (1200, 1200)
full_resolution = (1024, 1024)
initial_index_H_final = np.int((H_initial-H_final)/2)
initial_index_W_final = np.int((W_initial-W_final)/2)
final_index_H_final = initial_index_H_final + H_final
final_index_W_final = initial_index_W_final + W_final

### Get Image: ###
previous_image = read_image_default_torch()
previous_image = RGB2BW(previous_image)
previous_image = crop_numpy_batch(previous_image, input_resolution).cuda()

### GT parameters: ###
GT_shift_x = np.float32(1.3)
GT_shift_y = np.float32(-1.5)
GT_rotation_angle = np.float32(2)
GT_scale = np.float32(1.04)

### Parameters Space: ###
pyramid_levels = [16,8,4,2,1]
number_of_pyramid_levels = len(pyramid_levels)
shifts_dict = {'16': my_linspace(-10,10,21),
              '8': my_linspace(-1,2,3),
              '4': my_linspace(-1,2,3),
              '2': my_linspace(-1,2,3),
              '1': my_linspace(-1,2,3)}
rotation_angle_dict = {'16': my_linspace(-1,1,5),
              '8': my_linspace(-1,2,3),
              '4': my_linspace(-1,2,3),
              '2': my_linspace(-1,2,3),
              '1': my_linspace(-1,2,3)}
scale_factor_dict = {'16': my_linspace(0.95/1, 1.05/1,5),
              '8': my_linspace(0.95/1, 1.05/1,3),
              '4': my_linspace(0.95/1, 1.05/1,3),
              '2': my_linspace(0.95/1, 1.05/1,3),
              '1': my_linspace(0.95/1, 1.05/1,3)}
non_valid_outer_frame_dict = {'16': 10,
              '8': 10,
              '4': 10,
              '2': 10,
              '1': 10}

### Initialize Warp Objects: ###
shift_layer_torch = Shift_Layer_Torch()
affine_layer_torch = Warp_Tensors_Affine_Layer()

### Warp Image According To Above Parameters: ###
current_image = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle)

### Add Noise: ###
noise_SNR = 20
noise_sigma = 1/sqrt(noise_SNR)
current_image += torch.randn_like(current_image) * noise_sigma
previous_image += torch.randn_like(previous_image) * noise_sigma

### Final Crop Center Image: ###
current_image_crop = crop_torch_batch(current_image, full_resolution).cuda()
previous_image_crop = crop_torch_batch(previous_image, full_resolution).cuda()
########################################################################################################################


########################################################################################################################
### Loop Over The Different Pyramid Scales & Build Flow_Grids For Later Use: ###
Flow_Grids_list = []
warp_objects_list = []
identity_flow_grid_list = []
for pyramid_level_index in np.arange(number_of_pyramid_levels):
    ### Current Vecs & Sizes: ###
    current_scale_factor = np.int(2**pyramid_level_index)
    shifts_vec = shifts_dict[str(current_scale_factor)]
    rotation_angle_vec = rotation_angle_dict[str(current_scale_factor)]
    scale_factor_vec = scale_factor_dict[str(current_scale_factor)]
    H_current = np.int(full_resolution[0]/current_scale_factor)
    W_current = np.int(full_resolution[1]/current_scale_factor)

    ### Pre-Allocate Flow Grid & Stuff: ###
    L_shifts = len(shifts_vec)
    L_rotation_angle = len(rotation_angle_vec)
    L_scale = len(scale_factor_vec)
    number_of_possible_warps = L_shifts * L_shifts * L_rotation_angle * L_scale
    Flow_Grids = torch.zeros(number_of_possible_warps, H_current, W_current, 2).cuda()
    current_dummy_tensor = torch.zeros(1, 1, H_current, W_current).cuda()
    affine_layer_torch = Warp_Tensors_Affine_Layer()
    warp_layer_torch = Warp_Object()
    identity_flow_grid_list.append(identity_grid((H_current,W_current)).unsqueeze(0).cuda())

    ### Loop Over Pre-Defined Affine Parameters To Create Pre-Defined Flow_Grids: ###
    counter = 0
    for shift_x_counter, current_shift_x in enumerate(shifts_vec):
        for shift_y_counter, current_shift_y in enumerate(shifts_vec):
            for rotation_counter, current_rotation_angle in enumerate(rotation_angle_vec):
                for scale_counter, current_scale in enumerate(scale_factor_vec):
                    ### Warp Previous Image Many Ways To Find Best Fit: ###
                    current_shift_x = np.float32(current_shift_x)
                    current_shift_y = np.float32(current_shift_y)
                    current_scale = np.float32(current_scale)
                    current_rotation_angle = np.float32(current_rotation_angle)
                    current_dummy_tensor, current_flow_grid = affine_layer_torch.forward(current_dummy_tensor,
                                                                                          current_shift_x,
                                                                                          current_shift_y,
                                                                                          current_scale,
                                                                                          current_rotation_angle,
                                                                                          return_flow_grid=True)
                    Flow_Grids[counter, :, :, :] = current_flow_grid
                    if counter == 0:
                        _ = warp_layer_torch.forward(current_dummy_tensor, torch.zeros_like(current_dummy_tensor).cuda(), torch.zeros_like(current_dummy_tensor).cuda())
                        warp_objects_list.append(warp_layer_torch)
                    counter += 1

    ### Add Current Pyramid Scale Flow Grids Into Flow_Grids_list: ###
    Flow_Grids_list.append(Flow_Grids)
########################################################################################################################


########################################################################################################################
### Create Image Pyramids: ###
current_image_pyramid_list = []
previous_image_pyramid_list = []
for pyramid_level_index in np.arange(number_of_pyramid_levels):
    current_image_pyramid_list.append(nn.AvgPool2d(np.int(2**pyramid_level_index))(current_image_crop))
    previous_image_pyramid_list.append(nn.AvgPool2d(np.int(2**pyramid_level_index))(previous_image_crop))

### Loop Over Scales & Increasingly Refine Estimation: ###
B0,C0,H0,W0 = previous_image_pyramid_list[-1].shape
diff_flow_grid = torch.zeros((1,H0,W0,2)).cuda()
total_shift_x = 0
total_shift_y = 0
total_angle = 0
total_scale = 1
for pyramid_level_index in np.flip(np.arange(number_of_pyramid_levels)):
    ### Current Vecs & Sizes: ###
    current_scale_factor = np.int(2 ** pyramid_level_index)
    shifts_vec = shifts_dict[str(current_scale_factor)]
    rotation_angle_vec = rotation_angle_dict[str(current_scale_factor)]
    scale_factor_vec = scale_factor_dict[str(current_scale_factor)]
    L_shifts = len(shifts_vec)
    L_rotation_angle = len(rotation_angle_vec)
    L_scale = len(scale_factor_vec)
    current_shape = (L_shifts, L_shifts, L_rotation_angle, L_scale)
    number_of_possible_warps = L_shifts * L_shifts * L_rotation_angle * L_scale
    current_non_valid_border = non_valid_outer_frame_dict[str(current_scale_factor)]
    identity_flow_grid = identity_flow_grid_list[pyramid_level_index]
    H_current = np.int(full_resolution[0] / current_scale_factor)
    W_current = np.int(full_resolution[1] / current_scale_factor)

    ### Get Current Pyramid Level Objects: ###
    possible_flow_grids = Flow_Grids_list[pyramid_level_index]
    current_warp_object = warp_objects_list[pyramid_level_index]
    current_image_level = current_image_pyramid_list[pyramid_level_index]
    previous_image_level = previous_image_pyramid_list[pyramid_level_index]

    ### Stack Current Image Pyramid Level: ###
    previous_image_level_possible_warps = previous_image_level.repeat((number_of_possible_warps, 1, 1, 1))

    ### Get All Possible Warps: ###
    possible_flow_grids += diff_flow_grid
    possible_warps_tensor = torch.nn.functional.grid_sample(previous_image_level_possible_warps, possible_flow_grids, mode='bilinear')

    ### Get SAD: ###
    current_SAD_matrix = torch.abs(possible_warps_tensor - current_image_level)[:, :,
                         current_non_valid_border:-current_non_valid_border,
                         current_non_valid_border:-current_non_valid_border].mean(-1).mean(-1)[:,0]
    current_SAD_matrix = torch.reshape(current_SAD_matrix, current_shape)

    ### Get Min: ###
    min_index = torch.argmin(current_SAD_matrix)
    min_index = np.atleast_1d(min_index.cpu().numpy())[0]
    min_indices = np.unravel_index(min_index, current_shape)
    shift_x_inference = shifts_vec[min_indices[0]]
    shift_y_inference = shifts_vec[min_indices[1]]
    rotation_angle_inference = rotation_angle_vec[min_indices[2]]
    scale_factor_inference = scale_factor_vec[min_indices[3]]

    ### Get Current Grid Sample: ###
    previous_flow_grid = possible_flow_grids[min_index:min_index+1]
    diff_flow_grid = previous_flow_grid - identity_flow_grid  #TODO:
    diff_flow_grid = diff_flow_grid * 2
    diff_flow_grid = F.interpolate(diff_flow_grid.permute([0,3,1,2]), scale_factor=2).permute([0,2,3,1])

    ### Keep Track Of Total Transform: ###
    total_shift_x += shift_x_inference
    total_shift_y += shift_y_inference
    total_angle += rotation_angle_inference
    total_scale *= scale_factor_inference
    if pyramid_level_index > 0:
        total_shift_x *= 2
        total_shift_y *= 2
        total_angle *= 2
        total_scale = 1 + 2*(total_scale-1)

    # ### Get Corrected Image: ###
    # previous_image_level_corrected = possible_warps_tensor[min_index:min_index+1]
    # imshow_torch_multiple((current_image_level, previous_image_level_corrected))
########################################################################################################################

1


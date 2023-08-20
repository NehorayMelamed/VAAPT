# ------- (1): Create sterilized video to align -------
import os
from math import floor

import torch
from PIL.Image import Image
from albumentations import get_center_crop_coords
import kornia as K

from RapidBase.Anvil.Dudy import Align_ECC, affine_parameters_from_homography_matrix
from RapidBase.Anvil._transforms.affine_transformations import affine_transform_interpolated
from RapidBase.Anvil.alignments_layers import minSAD_transforms_layer, LinearPolarLayer, Gimbaless_Rotation_Layer_Torch
from RapidBase.import_all import *
from RapidBase.Anvil._internal_utils.torch_utils import MatrixOrigami, construct_tensor, unravel_index_torch
from RapidBase.Anvil._transforms.rotate_matrix import structure_thetas, fftshifted_formatted_vectors
from RapidBase.Anvil.alignments import circular_cc_shifts, normalized_cc_shifts


# from RapidBase.Utils.Add_Noise import add_noise_to_images_full
# from RapidBase.Utils.Array_Tensor_Manipulation import fast_binning_3D_overlap_flexible_AvgPool2d, crop_torch_batch, \
#     RGB2BW
# from RapidBase.Utils.Convolution_Utils import read_image_torch
# from RapidBase.Utils.Imshow_and_Plots import imshow_torch
# from RapidBase.Utils.MISCELENEOUS import string_rjust
# from RapidBase.Utils.Pytorch_Numpy_Utils import auto_canny_edge_detection
# from RapidBase.Utils.Unified_Registration_Utils import Warp_Tensors_Affine_Layer
# from RapidBase.Utils.linspace_arange import my_linspace
# from KAIR_light.utils.utils_image import IMG_EXTENSIONS



# ### TODO: put this shit in a function or something for fuck sake :) : ###
# base_image_path = r'/home/dudy/Projects/data/cat.jpeg'
# save_path_folder = r'/home/dudy/Projects/temp_palantir_trials'
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


def get_default_noise_dict():
    # Initialize Dict:
    IO_dict = EasyDict()

    # Misc:
    IO_dict.max_number_of_noise_images = np.inf
    IO_dict.flag_recursive = True

    # Paths:
    # (1). External Readout Noise Path:
    IO_dict.readout_noise_external_image_path = ''
    IO_dict.readout_noise_search_pattern = '*'
    # (2). External Non-Uniformity Pattern Path:
    IO_dict.NU_external_image_offset_path = ''
    IO_dict.NU_external_image_offset_search_pattern = '*'
    IO_dict.NU_external_image_gain_path = ''
    IO_dict.NU_external_image_gain_search_pattern = '*'
    IO_dict.allowed_extentions = IMG_EXTENSIONS

    # External readout noise parameters
    IO_dict.flag_noise_images_to_RAM = False
    # IO_dict.noise_images_path = None
    IO_dict.max_number_of_noise_images = np.inf


    # Flags: Each flag requires definition to a set of correlated parameters
    IO_dict.flag_add_per_pixel_readout_noise = False
    IO_dict.flag_add_shot_noise = False
    IO_dict.flag_add_dark_current_noise = False
    IO_dict.flag_add_per_pixel_NU = False
    IO_dict.flag_add_row_NU = False
    IO_dict.flag_add_col_NU = False
    IO_dict.flag_add_blob_NU = False
    IO_dict.flag_add_external_image_NU = False
    IO_dict.flag_add_row_readout_noise = False
    IO_dict.flag_add_col_readout_noise = False
    IO_dict.flag_add_external_readout_noise = False
    IO_dict.flag_add_dead_pixels_noise = False
    IO_dict.flag_add_white_pixels_noise = False
    IO_dict.flag_quantize_images = False

    # Parameters:
    # Shot noise parameters
    #TODO: make all the following numbers a range!!!: PPP, dark_current_background, blob_NU_blob_size, per_pixel_readout_noise_sigma,
    # row_readout_noise_sigma, col_readout_noise_sigma, dead_pixels_fraction, white_pixels_fraction
    IO_dict.QE = 1  # Quantum Efficiency
    IO_dict.Ge = 14.53  # Electrons per Gray level
    # Dark current noise parameters
    IO_dict.dark_current_background = 10
    # Pixel non uniformity parameters
    IO_dict.PRNU_sigmas_polynomial_params = [1, 0.05, 0.00001]  # Pixel Readout Non Uniformity. polynomial coefficients
    # Row non uniformity params
    IO_dict.RRNU_sigmas_polynomial_params = [1, 0.05, 0.00001]  # Row Readout Non Uniformity. polynomial coefficients
    IO_dict.flag_row_NU_random = True
    IO_dict.row_NU_PSD = None
    IO_dict.flag_same_row_NU_on_all_channels = True
    IO_dict.flag_same_row_NU_on_all_batch = True
    # Column non uniformity parameters
    IO_dict.CRNU_sigmas_polynomial_params = [1, 0.05, 0.00001]  # Col Readout Non Uniformity. polynomial coefficients
    IO_dict.flag_col_NU_random = True
    IO_dict.col_NU_PSD = None
    IO_dict.flag_same_col_NU_on_all_channels = True
    IO_dict.flag_same_col_NU_on_all_batch = True
    # Blob non uniformity parameters
    IO_dict.BRNU_sigmas_polynomial_params = [0, 0.2, 0]  # Blob Readout Non Uniformity. polynomial coefficients
    IO_dict.blob_NU_blob_size = 10
    IO_dict.flag_same_blob_NU_on_all_channels = True
    IO_dict.flag_same_blob_NU_on_all_batch = True
    # External image non uniformity parameters
    IO_dict.NU_external_image_offset = None
    IO_dict.NU_external_image_gain = None
    # Per pixel readout noise (AWGN) parameters
    IO_dict.per_pixel_readout_noise_sigma = 10
    # Row readout noise parameters
    IO_dict.flag_row_readout_noise_random = True
    IO_dict.row_readout_noise_PSD = None
    IO_dict.row_readout_noise_sigma = 10
    IO_dict.flag_same_row_readout_noise_on_all_channels = False
    # Column readout noise parameters
    IO_dict.flag_col_readout_noise_random = True
    IO_dict.col_readout_noise_PSD = None
    IO_dict.col_readout_noise_sigma = 10
    IO_dict.flag_same_col_readout_noise_on_all_channels = False
    # Dead pixels parameters
    IO_dict.dead_pixels_fraction = 0.001
    IO_dict.flag_same_dead_pixels_on_all_channels = True
    IO_dict.flag_same_dead_pixels_on_all_batch = True
    IO_dict.dead_pixels_value = 0 * 255
    # White pixels parameters
    IO_dict.white_pixels_fraction = 0.001
    IO_dict.flag_same_white_pixels_on_all_channels = True
    IO_dict.flag_same_white_pixels_on_all_batch = True
    IO_dict.white_pixels_value = 1 * 255

    return IO_dict


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
    shifts_y, shifts_x, cc = circular_cc_shifts(matrix=cropped_shifted_video[:, :1],
                                                reference_matrix=cropped_base_video[:1, :1])

    print(f'gt shifts h: {shifts_h}\nrt shifts h: {shifts_y}\ngt shifts w: {shifts_w}\nrt shifts w: {shifts_x}')


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

    # Choose Rotation Angle and assign variables needed for later:
    a = torch.tan(thetas / 2)
    b = -torch.sin(thetas)
    Nx_vec, Ny_vec = fftshifted_formatted_vectors(matrix, H, W)

    # Prepare Matrices to avoid looping:
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

    # Use Parallel Computing instead of looping:
    Ix_parallel = (
        torch.fft.ifftn(torch.fft.fftn(matrix, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    Iy_parallel = (
        torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(Ny_vec_mat_final * b * -1).conj(),
                        dim=[-2])).real
    input_mat_rotated = (
        torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    return input_mat_rotated


def plot_tensor_img(frame):
    frame = frame.cpu()
    data = np.array(torch.floor((frame.permute(1, 2, 0)))).astype(int)
    from matplotlib import pyplot as plt
    plt.imshow(data, interpolation='nearest')
    plt.show()


def test_minSAD():
    video_len = 4
    shifts_h = [-0.1, 0.1, 0.3, 0]
    shifts_w = [-0.1, 0.1, 0.3, 0]
    rotations = [1.2, 0.85, 0.93, 0.9]
    edge_crop_size = 10

    image_arr = np.array(Image.open(base_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).permute([2, 0, 1]).unsqueeze(0)
    image_tensor_downsampled = fast_binning_3D_overlap_flexible_AvgPool2d(image_tensor, (2, 2), (0, 0))
    video_base_tensor = torch.cat([image_tensor_downsampled] * video_len, dim=0)

    # shifted_video = shift_matrix_subpixel(matrix=video_base_tensor,
    #                                       shift_H=shifts_h,
    #                                       shift_W=shifts_w,
    #                                       warp_method='fft')
    # (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(shifted_video)
    # rotated_shifted_video = rotate_matrix(matrix=video_base_tensor, thetas=(Tensor(rotations) * np.pi / 180), warp_method='bilinear')
    # rotated_shifted_video = tensor_center_crop(rotated_shifted_video, crop_height=H, crop_width=W).clamp(0, 255)
    rotations_rads = [r * np.pi / 180 for r in rotations]

    rotated_shifted_video = affine_transform_interpolated(video_base_tensor.unsqueeze(0).float(),
                                                  construct_tensor(shifts_h),
                                                  construct_tensor(shifts_w),
                                                  construct_tensor(rotations_rads),
                                                  construct_tensor(1),
                                                  warp_method='bilinear',
                                                  expand=False)
    rotated_shifted_video = rotated_shifted_video.squeeze(1)
    # imshow_torch((rotated_shifted_video - video_base_tensor).abs().mean(-3,True)/255)
    # imshow_torch(shifted_video/255); imshow_torch(rotated_shifted_video/255)

    cropped_shifted_video = rotated_shifted_video[:, :, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]
    cropped_base_video = video_base_tensor[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]

    B, T, C, H, W = cropped_shifted_video.shape
    minSAD = minSAD_transforms_layer(1, T, H, W, device='cpu')
    shifts_y, shifts_x, rotational_shifts, scale_shifts, min_sad_matrix = minSAD.forward(
        matrix=cropped_shifted_video.float(),
        reference_matrix=cropped_base_video[0].float(),
        shift_h_vec=[-1, -0.5, 0, 0.5, 1],
        shift_w_vec=[-1, -0.5, 0, 0.5, 1],
        rotation_vec=[0.5, 0.75, 1, 1.25, 1.5],
        scale_vec=None,
        warp_method='bilinear')


    print(f'gt shifts h: {shifts_h}\nrt shifts h: {shifts_y}')
    print(f'gt shifts w: {shifts_w}\nrt shifts w: {shifts_x}')
    print(f'gt rotations: {rotations}\nrt rotations: {rotational_shifts}')


def _generate_affine_grid_alternative(B: int, T: int, H: int, W: int, shift_h: Union[int, float],
                          shift_w: Union[int, float],
                          rotation_angle: Union[int, float],
                          scale: Union[int, float],
                          device) -> Tensor:
    meshgrid_H, meshgrid_W = torch.meshgrid(torch.arange(H), torch.arange(W))
    meshgrid_W = meshgrid_W.type(torch.FloatTensor).to(device)
    meshgrid_H = meshgrid_H.type(torch.FloatTensor).to(device)
    meshgrid_W = torch.stack([meshgrid_W] * (B * T), 0)
    meshgrid_H = torch.stack([meshgrid_H] * (B * T), 0)

    translation_tensor = torch.tensor([[shift_w, shift_h]])
    center_tensor = torch.tensor([[meshgrid_W.shape[1] / 2, meshgrid_W.shape[0] / 2]])
    scale_tensor = torch.tensor([[scale, scale]])
    angle_tensor = torch.tensor([rotation_angle])

    affine_matrix = K.geometry.transform.get_affine_matrix2d(translation_tensor, center_tensor, scale_tensor, angle_tensor)
    # Just trying :)
    affine_matrix = affine_matrix[:, :2].squeeze(0).transpose(1, 0).to(device)
    meshgrid = torch.stack([meshgrid_W, meshgrid_H, torch.ones_like(meshgrid_H)], 3)

    affine_grid = torch.matmul(meshgrid, affine_matrix)

    return affine_grid


def test_affine():
    image_arr = np.array(Image.open(base_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).permute([2, 0, 1]).float().unsqueeze(0).unsqueeze(0)
    shift_w = Tensor([10])
    shift_h = Tensor([10])
    rotation = Tensor([2 * np.pi / 180])
    scale = Tensor([1])

    B, T, C, H, W = image_tensor.shape
    affine_grid = _generate_affine_grid_alternative(B, T, H, W, shift_h.item(), shift_w.item(), rotation.item(), scale.item(), device='cpu')

    augmented_tensor, grid = affine_transform_interpolated(image_tensor, shift_h, shift_w, rotation, scale, warp_method='bilinear', expand=False)
    augmented_tensor = augmented_tensor.squeeze(0)


    warped_tensor = torch.nn.functional.grid_sample(augmented_tensor, affine_grid, mode='bilinear')

    plot_tensor_img(image_tensor[0, 0] - warped_tensor[0])

def test_canny():
    edgy_image_path = '/home/elisheva/Projects/IMOD general/data/coco/val2017/000000007784.jpg'
    image_arr = np.array(Image.open(edgy_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).permute([2, 0, 1]).float()
    image_tensor = image_tensor.unsqueeze(0).cuda()
    image_tensor = crop_torch_batch(image_tensor, (500,500))
    canny_layer = auto_canny_edge_detection()

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold, binary_threshold = \
        canny_layer.forward(image_tensor, binary_threshold=0.95)
    magnitude, edges = K.filters.canny(image_tensor/255); imshow_torch(edges)

    imshow_torch(magnitude)
    imshow_torch(thresholded)
    imshow_torch(binary_threshold)


def test_linearPolar():
    video_len = 4
    shifts_h = [-0.1, 0.1, 0.3, 0]
    shifts_w = [-0.1, 0.1, 0.3, 0]
    rotations = [1.2, 0.85, 0.93, 0.9]
    edge_crop_size = 10

    image_arr = np.array(Image.open(base_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).permute([2, 0, 1]).unsqueeze(0)
    video_base_tensor = torch.cat([image_tensor] * video_len, dim=0)

    rotations_rads = [r * np.pi / 180 for r in rotations]

    rotated_shifted_video = affine_transform_interpolated(video_base_tensor.unsqueeze(0).float(),
                                                          construct_tensor(shifts_h),
                                                          construct_tensor(shifts_w),
                                                          construct_tensor(rotations_rads),
                                                          construct_tensor(1),
                                                          warp_method='bilinear',
                                                          expand=False)
    rotated_shifted_video = rotated_shifted_video.squeeze(1)
    cropped_shifted_video = rotated_shifted_video[:, :, :, edge_crop_size: -edge_crop_size,
                            edge_crop_size: -edge_crop_size]
    cropped_base_video = video_base_tensor[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]

    LinearPolar = LinearPolarLayer()
    rotational_shifts = LinearPolar.forward(cropped_shifted_video, cropped_base_video)


def test_FFTLogPolar_registration():
    ### Paramters: ###
    warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
    shift_x = np.array([0])
    shift_y = np.array([0])
    scale_factor_warp = np.array([1.00])
    rotation_angles = my_linspace(-10, 11, 21)

    inferred_rotation_list_of_lists = []
    images_folder = r'/home/dudy/Projects/data/DIV2K_valid_HR'
    images_list = [os.path.join(images_folder, img_path) for img_path in os.listdir(images_folder)[:10]]

    # T, C, H, W = input_tensor_1.shape  # 10 just extra
    # crop_h_size = int(H - ((0.5 * H * abs(np.tan(max(abs(rotation_angles))))) + 10))
    # crop_w_size = int(W - ((0.5 * W * abs(np.tan(max(abs(rotation_angles))))) + 10))
    crop_h_size = 700
    crop_w_size = 700

    fft_logpolar_registration_layer = LinearPolarLayer(B=1,
                                                       T=1,
                                                       C=1,
                                                       H=None,
                                                       W=None,
                                                       kernel_size=25,
                                                       blur_low_sigma=5 ** 2,
                                                       blur_high_sigma=20 ** 2,
                                                       flag_use_hann_window=True,
                                                       radius_div_factor=32,
                                                       device='cpu')

    for image in images_list:
        # Prepare Data
        image = os.path.join(images_folder, image)
        input_tensor_1 = read_image_torch(image, flag_convert_to_rgb=0)
        input_tensor_1 = RGB2BW(input_tensor_1)
        a = logpolar(input_tensor_1[0, 0])
        # noise tensor for testing
        noise_dict = get_default_noise_dict()
        # noise_dict.flag_add_per_pixel_readout_noise = True
        # noise_dict.flag_add_shot_noise = True
        # noise_dict.flag_add_row_readout_noise = True
        # noise_dict.flag_add_column_readout_noise = True
        # noise_dict.flag_quantize_images = True
        input_tensor_1, _, _ = add_noise_to_images_full(input_tensor_1, noise_dict,
                                                        flag_check_c_or_t=False)
        input_tensor_1 = (input_tensor_1 / 255)#.cuda()
        input_tensor_1_numpy = input_tensor_1.cpu().numpy()[0, 0]

        warped_tensors_list_numpy = []
        warped_tensors_list_torch = []
        inferred_rotation_list = []

        for i in np.arange(len(rotation_angles)):
            print(i)
            ### Warp Tensor: ###
            rotation_angle = np.array([rotation_angles[i]])
            rotation_rads = rotation_angle * np.pi / 180
            # add a random small shift
            # shift_y = torch.randn(1)
            # shift_x = torch.randn(1)
            input_tensor_2 = affine_transform_interpolated(input_tensor_1.unsqueeze(0).float(),
                                                           construct_tensor(shift_y),
                                                           construct_tensor(shift_x),
                                                           construct_tensor(rotation_rads),
                                                           construct_tensor([1]),
                                                           warp_method='bilinear',
                                                           expand=False).squeeze(0)

            # input_tensor_2 = warp_tensors_affine_layer.forward(input_tensor_1, shift_x, shift_y, scale_factor_warp, rotation_angle)
            input_tensor_2_cropped = crop_torch_batch(input_tensor_2, (crop_h_size, crop_w_size))
            input_tensor_1_cropped = crop_torch_batch(input_tensor_1, (crop_h_size, crop_w_size))
            # input_tensor_2_cropped = crop_torch_batch(input_tensor_2, (700, 700))
            # input_tensor_1_cropped = crop_torch_batch(input_tensor_1, (700, 700))
            input_tensor_2_numpy = input_tensor_2_cropped.cpu().numpy()[0, 0]
            warped_tensors_list_numpy.append(input_tensor_2_numpy)
            warped_tensors_list_torch.append(input_tensor_2_cropped)
            recovered_angle = fft_logpolar_registration_layer.forward(input_tensor_1_cropped,
                                                                      input_tensor_2_cropped)
            inferred_rotation_list.append(recovered_angle.item())
        inferred_rotation_list_of_lists.append(inferred_rotation_list)
    rotation_numpy = np.array(inferred_rotation_list_of_lists)
    estimated_bias = np.mean(rotation_numpy, axis=0)

    ### Present Outputs: ###
    figure()
    plot(rotation_angles, rotation_angles)
    plot(rotation_angles, -estimated_bias)
    plot(my_linspace(rotation_angles[0], 11, 21), rotation_angles - estimated_bias)
    plt.xlabel('Rotation Angle')
    plt.legend(['GT Rotation', 'torch layer'])
    plt.title('Translation plot')

    np.save(r'/home/dudy/Projects/temp_palantir_trials/estimated_bias_bicubic', rotation_angles - estimated_bias)


def test_sift_feature_based():
    palantir_image_path = r'/home/dudy/Projects/data/pngsh/pngs/2/01700.png'
    video_len = 4
    shifts_h = [0, 0, 0, 0]
    shifts_w = [0, 0, 0, 0]
    rotations = [1.2, 0.85, 0.93, 0.9]
    edge_crop_size = 10

    image_arr = np.array(Image.open(palantir_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to('cuda:0')
    image_tensor = RGB2BW(image_tensor)
    # image_tensor_downsampled = fast_binning_3D_overlap_flexible_AvgPool2d(image_tensor, (2, 2), (0, 0))
    # video_base_tensor = torch.cat([image_tensor_downsampled] * video_len, dim=0)
    video_base_tensor = torch.cat([image_tensor] * video_len, dim=0)

    rotations_rads = [r * np.pi / 180 for r in rotations]

    rotated_shifted_video = affine_transform_interpolated(video_base_tensor.unsqueeze(0).float(),
                                                          construct_tensor(shifts_h),
                                                          construct_tensor(shifts_w),
                                                          construct_tensor(rotations_rads),
                                                          construct_tensor(1),
                                                          warp_method='bilinear',
                                                          expand=False)
    rotated_shifted_video = rotated_shifted_video.squeeze(1)
    # imshow_torch((rotated_shifted_video - video_base_tensor).abs().mean(-3,True)/255)
    # imshow_torch(shifted_video/255); imshow_torch(rotated_shifted_video/255)

    cropped_shifted_video = rotated_shifted_video[:, :, :, edge_crop_size: -edge_crop_size,
                            edge_crop_size: -edge_crop_size]
    cropped_base_video = video_base_tensor[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]
    B, T, C, H, W = cropped_shifted_video.shape
    aligned_tensor_FeatureAlign, H_matrix_list, rotation_list_degrees, rotation_list_rads = Align_FeatureBased_SIFT(
        cropped_shifted_video[0, :, :1].float(), cropped_base_video.float())
    rotation_list = [rotation_list_rads]

    # print(f'gt shifts h: {shifts_h}\nrt shifts h: {shifts_y}')
    # print(f'gt shifts w: {shifts_w}\nrt shifts w: {shifts_x}')
    print(f'gt rotations: {rotations}\nrt rotations: {rotation_list_rads}')
    print(f'gt rotations: {rotations}\nrt rotations: {rotation_list_degrees}')
    video_base_tensor = video_base_tensor[..., 10: -10, 10: -10]
    imshow_torch_video(video_base_tensor - aligned_tensor_FeatureAlign, FPS=1)


def test_ecc():
    palantir_image_path = r'/home/dudy/Projects/data/pngsh/pngs/2/01700.png'
    video_len = 4
    shifts_h = [0, 0, 0, 0]
    shifts_w = [0, 0, 0, 0]
    rotations = [1.2, 0.85, 0.93, 0.9]
    edge_crop_size = 10

    image_arr = np.array(Image.open(palantir_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0)
    video_base_tensor = torch.cat([image_tensor] * video_len, dim=0)

    rotations_rads = [r * np.pi / 180 for r in rotations]

    rotated_shifted_video = affine_transform_interpolated(video_base_tensor.unsqueeze(0).float(),
                                                          construct_tensor(shifts_h),
                                                          construct_tensor(shifts_w),
                                                          construct_tensor(rotations_rads),
                                                          construct_tensor(1),
                                                          warp_method='bilinear',
                                                          expand=False)
    rotated_shifted_video = rotated_shifted_video.squeeze(1)
    # imshow_torch((rotated_shifted_video - video_base_tensor).abs().mean(-3,True)/255)
    # imshow_torch(shifted_video/255); imshow_torch(rotated_shifted_video/255)

    cropped_shifted_video = rotated_shifted_video[:, :, :, edge_crop_size: -edge_crop_size,
                            edge_crop_size: -edge_crop_size]
    cropped_base_video = video_base_tensor[:, :, edge_crop_size: -edge_crop_size, edge_crop_size: -edge_crop_size]
    B, T, C, H, W = cropped_shifted_video.shape
    aligned_tensor_FeatureAlign, H_matrix_list, rotation_rad_list, rotation_degree_list, shift_H_list, shift_W_list = \
        Align_ECC(cropped_shifted_video[0].float(),
                  cropped_base_video.float())
    # print(f'gt shifts h: {shifts_h}\nrt shifts h: {shifts_y}')
    # print(f'gt shifts w: {shifts_w}\nrt shifts w: {shifts_x}')
    # print(f'gt rotations: {rotations}\nrt rotations: {rotation_rad_list}')
    print(f'gt rotations: {rotations}\nrt rotations: {rotation_degree_list}')
    video_base_tensor = video_base_tensor[..., 10: -10, 10: -10]
    aligned_tensor_FeatureAlign = RGB2BW(aligned_tensor_FeatureAlign.squeeze(1))
    imshow_torch_video(video_base_tensor - aligned_tensor_FeatureAlign.squeeze(1), FPS=1)


def test_homography_collection():
    ### Shift Parameters: ###
    palantir_image_path = r'/home/dudy/Projects/data/pngsh/pngs/2/01700.png'
    video_len = 4
    shifts_h = [1, 2, 3, 4]
    shifts_w = [1, 2, 3, 4]
    rotations = [0, 0, 0, 0]
    rotations_rads = [r * np.pi / 180 for r in rotations]
    edge_crop_size = 10

    ### Load Base Image: ###
    image_arr = np.array(Image.open(palantir_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(f'cuda:{0}')
    original_image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(f'cuda:{0}')
    video_base_tensor = torch.cat([image_tensor] * video_len, dim=0)
    T, C, H, W = video_base_tensor.shape

    ### Initialize cumsum shifts and affine matrix: ###
    affine_matrix = torch.stack([torch.eye(3, 3)] * T)
    total_shift_h = 0
    total_shift_w = 0
    total_rotation = 0

    ### Warp Base Image: ###
    rotated_shifted_video = affine_transform_interpolated(video_base_tensor.unsqueeze(0).float(),
                                                          construct_tensor(shifts_h),
                                                          construct_tensor(shifts_w),
                                                          construct_tensor(rotations_rads),
                                                          construct_tensor(1),
                                                          warp_method='bicubic',
                                                          expand=False).squeeze(0)
    # ### Immediately warp back to check for "purest" process possible: ###
    # rotated_shifted_video_warped_back = affine_transform_interpolated(rotated_shifted_video.unsqueeze(0),
    #                                                       -construct_tensor(shifts_h),
    #                                                       -construct_tensor(shifts_w),
    #                                                       -construct_tensor(rotations_rads),
    #                                                       construct_tensor(1),
    #                                                       warp_method='bicubic',
    #                                                       expand=False).squeeze(0)
    # imshow_torch_video(video_base_tensor - rotated_shifted_video_warped_back)


    # I. cross correlation
    # aligned_tensor_CC_ECC, H_matrix, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = Align_ECC(rotated_shifted_video, image_tensor)
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = align_to_reference_frame_circular_cc(matrix=rotated_shifted_video,
                                                                                          reference_matrix=image_tensor,
                                                                                          matrix_fft=None,
                                                                                          normalize_over_matrix=False,
                                                                                          warp_method='bicubic',
                                                                                          crop_warped_matrix=False)


    ### Add Cross-Correlation Results To Cumsum shifts: ###
    total_shift_h += shifts_h_CC
    total_shift_w += shifts_w_CC
    ### Update Affine Matrix: ###
    affine_matrix_cc = torch.stack([torch.eye(3, 3)] * T)
    affine_matrix_cc[:, 0, 2] = shifts_h_CC  #TODO: understand whether it's correct to put a minus sign
    affine_matrix_cc[:, 1, 2] = shifts_w_CC
    affine_matrix = affine_matrix_cc @ affine_matrix

    ### Crop Aligned And Image Tensors To Same Size: ###
    max_shift_h = shifts_h_CC.abs().max()
    max_shift_w = shifts_w_CC.abs().max()
    new_H = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)
    #(*). crop:
    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, [new_H, new_W])
    image_tensor = crop_torch_batch(image_tensor, [new_H, new_W])

    # II. enhanced cross correlation
    aligned_tensor_CC_ECC, H_matrix, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = Align_ECC(aligned_tensor_CC, image_tensor)
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{0}')

    ### TODO: understand whether we need to crop after ECC (maybe it happens automatically or something?)

    ### update cumsum shifts from ECC calculation: ###
    total_shift_h += shifts_w_CC_ECC
    total_shift_w += shifts_h_CC_ECC
    total_rotation += shifts_rotation_CC_ECC
    ### update affine_matrix using H_matrix from ECC function: ###
    affine_matrix = H_matrix.to(torch.float32).cpu() @ affine_matrix

    # III. cross correlation
    aligned_tensor_CC_ECC_CC, shifts_h_CC_ECC_CC, shifts_w_CC_ECC_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=aligned_tensor_CC_ECC,
                                             reference_matrix=image_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)

    ### update cumsum shifts from 2nd cross-correlation: ###
    total_shift_h += shifts_w_CC_ECC_CC
    total_shift_w += shifts_h_CC_ECC_CC
    ### update affine_matrix from 2nd cross-correlation: ###
    affine_matrix_cc_ecc_cc = torch.stack([torch.eye(3, 3)] * T)
    affine_matrix_cc_ecc_cc[:, 0, 2] = shifts_w_CC_ECC_CC
    affine_matrix_cc_ecc_cc[:, 1, 2] = shifts_h_CC_ECC_CC
    affine_matrix = affine_matrix_cc_ecc_cc @ affine_matrix

    ### Crop again after 2nd cross-correlation: ###
    max_shift_H_CC_ECC_CC = shifts_h_CC_ECC_CC.abs().max()
    max_shift_W_CC_ECC_CC = shifts_w_CC_ECC_CC.abs().max()
    new_H_CC_ECC_CC = new_H - (max_shift_H_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC_CC = new_W - (max_shift_W_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    aligned_tensor_CC_ECC_CC = crop_torch_batch(aligned_tensor_CC_ECC_CC, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    ### Manually Warp To Check Consistency: ###
    # affine_matrix = affine_matrix[:, :2]  #turn H_matrix into affine matrix
    shift_H, shift_W, theta_total_rads, R_total, theta_total_degrees = affine_parameters_from_homography_matrix(affine_matrix, flag_assume_small_angle=True)

    # #(*). Using kornia:
    # aligned_tensor_3 = kornia.geometry.warp_affine(rotated_shifted_video.cuda(), affine_matrix.cuda(),
    #                                                (new_H_CC_ECC_CC, new_W_CC_ECC_CC),
    #                                                mode='bicubic')
    # aligned_tensor_3 = crop_torch_batch(aligned_tensor_3, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    #(*). Using our warp implementation and using cumsum shifts:
    # aligned_tensor_2 = affine_transform_interpolated(rotated_shifted_video.unsqueeze(0).float(),
    #                                                  -construct_tensor(shifts_h),
    #                                                  -construct_tensor(shifts_w),
    #                                                  -construct_tensor(rotations_rads),
    #                                                  construct_tensor(1),
    #                                                  warp_method='bicubic',
    #                                                  expand=False).squeeze(0)
    aligned_tensor_2 = affine_transform_interpolated(rotated_shifted_video.unsqueeze(0).float(),
                                                     -construct_tensor(total_shift_h),
                                                     -construct_tensor(total_shift_w),
                                                     -construct_tensor(total_rotation),
                                                     construct_tensor(1),
                                                     warp_method='bicubic',
                                                     expand=False).squeeze(0)
    #(*). Using our warp implementation and using affine_matrix:




    ### Crop: ###
    aligned_tensor_2 = crop_torch_batch(aligned_tensor_2, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    original_image_tensor_final_cropped = crop_torch_batch(original_image_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    imshow_torch_video(aligned_tensor_CC_ECC_CC - aligned_tensor_2)
    imshow_torch_video(original_image_tensor_final_cropped - aligned_tensor_2)  #still works

    cropped_base = crop_torch_batch(video_base_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    imshow_torch_video(cropped_base - aligned_tensor_2)
    imshow_torch_video(cropped_base - aligned_tensor_CC_ECC_CC)


def test_gimbaless_rotation():
    palantir_image_path = r'/home/dudy/Projects/data/pngsh/pngs/2/01700.png'
    video_len = 4
    shifts_h = [0.1, 0.05, 0.3, -0.07]
    shifts_w = [0.2, -0.15, 0.02, 0.4]
    rotations = [0.1, 0.2, 0.3, 0.4]
    rotations_rads = [r * np.pi / 180 for r in rotations]
    edge_crop_size = 10

    ### Load Base Image: ###
    image_arr = np.array(Image.open(palantir_image_path), dtype=int)
    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(f'cuda:{0}')
    original_image_tensor = torch.from_numpy(image_arr).unsqueeze(0).unsqueeze(0).to(f'cuda:{0}')
    video_base_tensor = torch.cat([image_tensor] * video_len, dim=0)
    T, C, H, W = video_base_tensor.shape

    ### Warp Base Image: ###
    rotated_shifted_video = affine_transform_interpolated(video_base_tensor.unsqueeze(0).float(),
                                                          construct_tensor(shifts_h),
                                                          construct_tensor(shifts_w),
                                                          construct_tensor(rotations_rads),
                                                          construct_tensor(1),
                                                          warp_method='bicubic',
                                                          expand=False)

    gimbaless_rotation_layer = Gimbaless_Rotation_Layer_Torch()

    delta_x, delta_y, delta_theta = gimbaless_rotation_layer.forward(rotated_shifted_video, original_image_tensor)

    print(f'gt shift x: {shifts_w}\nrt shift x: {-delta_x}')
    print(f'gt shift y: {shifts_h}\nrt shift y: {-delta_y}')
    print(f'gt rotations: {rotations_rads}\nrt rotations: {-delta_theta}')


def get_max_sub_pixel_over_t(matrix: Tensor):
    # expected THW tensor
    # Phase I: find discrete maximum
    T, H, W = matrix.shape
    flat_matrix = torch.flatten(matrix, start_dim=1)
    flat_max_indices = torch.argmax(flat_matrix, dim=1)
    max_indices_tuple = unravel_index_torch(flat_max_indices, (H, W))
    max_indices_tuple = [l.tolist() for l in max_indices_tuple]
    max_indices_list = list(zip(*max_indices_tuple))
    timed_max_indices = list(enumerate(max_indices_list))
    timed_max_indices = [(t, h, w) for (t, (h, w)) in timed_max_indices]

    # Phase II: do parabola fit
    def index_to_parabola_fit_vals(t, h, w, values_tensor):
        # Assumed shapes: values tensor of shape HxW
        # indices tensor of shape 2, 3
        H, W = values_tensor.shape[1:]
        if h == 0 or h == H - 1:
            H_vals = [nan, values_tensor[t, h, w].item(), nan]
        else:
            H_vals = [values_tensor[t, h - 1, w].item(), values_tensor[t, h, w].item(), values_tensor[t, h + 1, w].item()]
        if w == 0 or w == W - 1:
            W_vals = [nan, values_tensor[t, h, w].item(), nan]
        else:
            W_vals = [values_tensor[t, h, w - 1].item(), values_tensor[t, h, w].item(), values_tensor[t, h, w + 1].item()]

        return [H_vals, W_vals]

    def max_delta_over_parabola_fit(y1, y2, y3):
        if y1 is nan:
            delta_shift_x = 0
        else:
            x_vec = [-1, 0, 1]
            c_x, b_x, a_x = fit_polynomial(x_vec, [y1, y2, y3])
            delta_shift_x = -b_x / (2 * a_x)

        return delta_shift_x

    map_values_for_parabola_fit = partial(index_to_parabola_fit_vals, values_tensor=matrix)
    values_for_parabola_fit = list(itertools.starmap(map_values_for_parabola_fit, timed_max_indices))
    discrete_max_values = [v[0][1] for v in values_for_parabola_fit]

    # subpixel_deltas_over_t = [(max_delta_over_parabola_fit(*vals_h), max_delta_over_parabola_fit(*vals_w)) for vals_h, vals_w in values_for_parabola_fit]
    subpixel_deltas_over_t = [[max_delta_over_parabola_fit(*vals_h), max_delta_over_parabola_fit(*vals_w)] for vals_h, vals_w in values_for_parabola_fit]
    subpixel_max_over_t = Tensor([d_max + shift_h + shift_w for d_max, (shift_h, shift_w) in
                                 zip(discrete_max_values, subpixel_deltas_over_t)])

    return subpixel_max_over_t, subpixel_deltas_over_t


# matrix = Tensor([[[2, 1, 3],
#                   [4, 15, 6],
#                   [7, 8, 9]],
#                  [[2, 1, 3],
#                   [4, 5, 6],
#                   [10, 13, 5]],
#                  [[11, 18, 9],
#                   [10, 11, 12],
#                   [12, 14, 1]],
#                  [[11, 18, 9],
#                   [10, 11, 12],
#                   [0, 3, 6]]])
# print(get_max_sub_pixel_over_t(matrix))

# test_canny()
# import cProfile
# cProfile.run('test_minSAD()')
# test_minSAD()
# test_circular_cc()
# test_affine()
# test_FFTLogPolar_registration()
# test_linearPolar()
# test_sift_feature_based()
# test_ecc()
# test_homography_collection()
# test_gimbaless_rotation()

# plot_tensor_img(output_tensor[0])
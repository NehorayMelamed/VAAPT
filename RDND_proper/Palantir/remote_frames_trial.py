import numpy as np

from Palantir.Palantir_utils import read_bin_images_batch, find_outliers
from Dudy import Align_ECC, affine_parameters_from_homography_matrix
from RapidBase.Utils.Classical_DSP.ECC import ECC_torch
from alignments import align_to_reference_frame_circular_cc
from RapidBase.import_all import *

gpu_idx = 0


def align_two_frames_CC(input_tensor: Tensor, reference_tensor: Tensor):
    _, C, H, W = input_tensor.shape
    aligned_tensor, shifts_h, shifts_w, _ = \
        align_to_reference_frame_circular_cc(matrix=input_tensor,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)
    # crop tensors to the same size
    max_shift_h = shifts_h.abs().max()
    max_shift_w = shifts_w.abs().max()
    new_H = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor = crop_torch_batch(aligned_tensor, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    return input_tensor, reference_tensor, aligned_tensor, shifts_h, shifts_w


def align_two_frames_ECC(input_tensor: Tensor,
                         reference_tensor: Tensor,
                         warp_mode: str = 'homography',
                         warp_matrix: Tensor = torch.eye(3, 3),
                         iterations: int = 25):
    _, C, H, W = input_tensor.shape
    H_matrix, aligned_tensor, ecc, delta_p = ECC_torch(
        input_tensor.squeeze(0), reference_tensor.squeeze(0), transform_string=warp_mode, delta_p_init=warp_matrix,
        number_of_levels=1, number_of_iterations_per_level=iterations)
    aligned_tensor = aligned_tensor.float().to(f'cuda:{gpu_idx}')

    # get shifts
    # TODO this function is really not accurate mathematically. Requires some thinking later on
    shifts_h, shifts_w, shifts_rotation, _, _ = affine_parameters_from_homography_matrix(H_matrix.unsqueeze(0))

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    max_rotation_shift_H = (W / 2) * torch.tan(shifts_rotation).abs().max()
    max_rotation_shift_W = (H / 2) * torch.tan(shifts_rotation).abs().max()
    max_shift_H = shifts_h.abs().max() + max_rotation_shift_H
    max_shift_W = shifts_w.abs().max() + max_rotation_shift_W

    new_H = H - (max_shift_H.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_W.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor = crop_torch_batch(aligned_tensor, [new_H, new_W]).unsqueeze(0)
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    return input_tensor, reference_tensor, aligned_tensor, ecc, delta_p, shifts_h, shifts_w, shifts_rotation


def align_two_frames(input_tensor: Tensor, reference_tensor: Tensor):
    # align two frames using different methods
    C, H, W = input_tensor.shape
    total_shift_h = Tensor([0]).to(f'cuda:{gpu_idx}')
    total_shift_w = Tensor([0]).to(f'cuda:{gpu_idx}')
    total_shift_rotation = Tensor([0]).to(f'cuda:{gpu_idx}')
    input_tensor = input_tensor.unsqueeze(0).to(f'cuda:{gpu_idx}')
    reference_tensor = reference_tensor.unsqueeze(0).to(f'cuda:{gpu_idx}')

    # I: CC - always helps
    input_tensor, reference_tensor, aligned_tensor_CC, shifts_h_CC, shifts_w_CC = \
        align_two_frames_CC(input_tensor, reference_tensor)

    # sum up shifts
    total_shift_h += shifts_h_CC
    total_shift_w += shifts_w_CC

    # show results
    imshow_torch(reference_tensor)
    imshow_torch(aligned_tensor_CC)
    imshow_torch(aligned_tensor_CC - reference_tensor)

    # II. ECC - homography is essential
    input_tensor, reference_tensor, aligned_tensor_CC_ECC, _, _, shifts_h_CC_ECC, shifts_w_CC_ECC, shifts_rotation_CC_ECC =\
        align_two_frames_ECC(aligned_tensor_CC, reference_tensor)

    # sum up shifts
    total_shift_h += shifts_h_CC_ECC
    total_shift_w += shifts_w_CC_ECC
    total_shift_rotation += shifts_rotation_CC_ECC

    new_H, new_W = reference_tensor.shape[-2:]
    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, (new_H, new_W))

    # show results
    imshow_torch(reference_tensor)
    imshow_torch(aligned_tensor_CC)
    imshow_torch(aligned_tensor_CC_ECC)
    imshow_torch(aligned_tensor_CC_ECC - reference_tensor)

    # III. CC again - may help?
    input_tensor, reference_tensor, aligned_tensor_CC_ECC_CC, shifts_h_CC_ECC_CC, shifts_w_CC_ECC_CC = \
        align_two_frames_CC(input_tensor, reference_tensor)

    # sum up shifts
    total_shift_h += shifts_h_CC_ECC_CC
    total_shift_w += shifts_w_CC_ECC_CC

    aligned_tensor_final = aligned_tensor_CC_ECC_CC

    # analize results
    diff = (reference_tensor - aligned_tensor_final)
    diff = scale_array_stretch_hist(diff, (0.1, 0.99))
    imshow_torch(diff > diff.quantile(0.99))


def optimize_iterations_number(input_tensor: Tensor, reference_tensor: Tensor, max_iterations: int = 60):
    """
    Optimize iterations number required for very remote frames (assumed ~1500 frames apart)
    align frames, nd output number of outliers, rho (difference measurement in the algorithm) and difference between the
    homography matrices

    Args:
        input_tensor:
        reference_tensor:

    Returns:

    """
    C, H, W = input_tensor.shape
    input_tensor = input_tensor.unsqueeze(0).to(f'cuda:{gpu_idx}')
    reference_tensor = reference_tensor.unsqueeze(0).to(f'cuda:{gpu_idx}')

    # First align the frames initially with cross correlation
    input_tensor, reference_tensor, aligned_tensor_CC, _, _ = \
        align_two_frames_CC(input_tensor, reference_tensor)

    # ECC - show perfomance over different iteration numbers
    num_outliers_list = []
    ecc_sum_list = []
    delta_p_norm_list = []
    for i in range(1, max_iterations + 1):
        # do ECC
        temp_input_tensor, temp_reference_tensor, aligned_tensor_CC_ECC, ecc, delta_p, _, _, _ = \
            align_two_frames_ECC(aligned_tensor_CC, reference_tensor, iterations=i)

        # do CC to get final result
        temp_input_tensor, temp_reference_tensor, aligned_tensor_CC_ECC_CC, _, _ = \
            align_two_frames_CC(aligned_tensor_CC_ECC, temp_reference_tensor)

        # generate measurements
        num_outliers = find_outliers(aligned_tensor_CC_ECC_CC, temp_reference_tensor, noise_mult_factor=25).sum()
        ecc_sum = ecc.abs().sum()
        delta_p_norm = torch.linalg.norm(delta_p)

        # print results for progress sense :)
        print(f"{i} iterations:\nnum_outliers: {num_outliers}, ecc_sum: {ecc_sum}, delta_p_norm: {delta_p_norm}")

        # add to list
        num_outliers_list += [num_outliers]
        ecc_sum_list += [ecc_sum]
        delta_p_norm_list += [delta_p_norm]


    # show results
    num_samples = min(max_iterations, len(num_outliers_list))
    plt.plot(np.linspace(1, num_samples, num_samples), array([val.cpu() for val in num_outliers_list]))
    rho_array = array([val.cpu() for val in ecc_sum_list])
    rho_array_diff = np.diff(rho_array)
    plt.plot(np.linspace(1, len(rho_array), len(rho_array)), rho_array)
    plt.plot(np.linspace(1, len(rho_array_diff), len(rho_array_diff)), rho_array_diff)
    plt.plot(np.linspace(1, len(delta_p_norm_list), len(delta_p_norm_list)), array([val.cpu() for val in delta_p_norm_list]))  #delta_p_norm_thershold ~1e-3

if __name__ == "__main__":
    experiment_idx = 1
    original_frames_folder = f'/home/dudy/Projects/data/Palantir/p{experiment_idx}'
    frame_1_name = '00100.Bin'
    frame_2_name = '02000.Bin'
    original_height = 540
    original_width = 8192
    filenames = [os.path.join(original_frames_folder, frame_1_name), os.path.join(original_frames_folder, frame_2_name)]
    frames = read_bin_images_batch(filenames, original_height, original_width).float()
    frame_1 = frames[0].unsqueeze(0)
    frame_2 = frames[1].unsqueeze(0)
    optimize_iterations_number(frame_1, frame_2)

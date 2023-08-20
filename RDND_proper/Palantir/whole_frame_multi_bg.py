import os
from math import ceil

import cv2
import kornia
import numpy as np
import torch
from easydict import EasyDict
from numpy import array
from torch import Tensor

from RapidBase.import_all import *
from Palantir.Palantir_utils import pad_numpy_batch, read_bin_images_batch, \
    pad_torch_batch, find_outliers
from alignments import circular_cc_shifts
from alignments_layers import minSAD_transforms_layer
from transforms import affine_warp_matrix, shift_matrix_subpixel
# from RapidBase.Utils.Array_Tensor_Manipulation import crop_torch_batch, torch_to_numpy, crop_numpy_batch, \
#     numpy_array_to_video_ready, scale_array_stretch_hist
# from RapidBase.Utils.MISCELENEOUS import string_rjust
# from RapidBase.Utils.Path_and_Reading_utils import read_image_filenames_from_folder
# from RapidBase.Utils.Imshow_and_Plots import imshow_torch_video, imshow_torch
from Palantir.palantir_optional_stabilizations import align_palantir_CC_ECC_CC


# ----- Phase I: Initialize Paths and Global Variables -----
def initialize_experiment(original_frames_folder, results_path, experiment_name, sub_experiment_idx,
                          original_height: int = 540, original_width: int = 8192):
    experiment_dict = EasyDict()
    experiment_dict.original_frames_folder = original_frames_folder
    experiment_dict.results_path = results_path
    experiment_dict.experiment_name = experiment_name
    experiment_dict.experiment_idx = sub_experiment_idx

    experiment_dict.experiment_path = os.path.join(results_path, experiment_name)
    experiment_dict.save_path = os.path.join(experiment_dict.experiment_path, string_rjust(int(sub_experiment_idx), 3))
    experiment_dict.BG_estimations_folder = os.path.join(experiment_dict.save_path, 'bg_estimations')
    experiment_dict.BG_estimation_names = []  # to be updated in the process

    # result bin file path
    experiment_dict.bin_name_aligned = os.path.join(experiment_dict.save_path, 'aligned_video.Bin')
    experiment_dict.bin_name_outliers = os.path.join(experiment_dict.save_path, 'outliers_video.Bin')

    # result avi's and graphs paths
    experiment_dict.videos_path = os.path.join(experiment_dict.save_path, 'videos')
    experiment_dict.shift_logs_path = os.path.join(experiment_dict.save_path, 'shift_logs')
    experiment_dict.video_name_original = os.path.join(experiment_dict.videos_path, 'original_video.avi')
    experiment_dict.video_name_aligned = os.path.join(experiment_dict.videos_path, 'aligned_video.avi')
    experiment_dict.video_name_outliers = os.path.join(experiment_dict.videos_path, 'outliers_video.avi')
    experiment_dict.video_name_diff = os.path.join(experiment_dict.videos_path, 'diff_video.avi')
    experiment_dict.log_name_shifts_h = os.path.join(experiment_dict.shift_logs_path, 'shifts_h.npy')
    experiment_dict.log_name_shifts_w = os.path.join(experiment_dict.shift_logs_path, 'shifts_w.npy')
    experiment_dict.log_name_rotations = os.path.join(experiment_dict.shift_logs_path, 'rotations.npy')

    # generate paths
    if not os.path.exists(experiment_dict.experiment_path):
        os.mkdir(experiment_dict.experiment_path)
    if not os.path.exists(experiment_dict.save_path):
        os.mkdir(experiment_dict.save_path)
    if not os.path.exists(experiment_dict.videos_path):
        os.mkdir(experiment_dict.videos_path)
    if not os.path.exists(experiment_dict.BG_estimations_folder):
        os.mkdir(experiment_dict.BG_estimations_folder)
    if not os.path.exists(experiment_dict.shift_logs_path):
        os.mkdir(experiment_dict.shift_logs_path)

    # clean former results
    if os.path.exists(experiment_dict.bin_name_aligned):
        os.remove(experiment_dict.bin_name_aligned)
    if os.path.exists(experiment_dict.bin_name_outliers):
        os.remove(experiment_dict.bin_name_outliers)
    if os.path.exists(experiment_dict.log_name_shifts_h):
        os.remove(experiment_dict.log_name_shifts_h)
    if os.path.exists(experiment_dict.log_name_shifts_w):
        os.remove(experiment_dict.log_name_shifts_w)
    if os.path.exists(experiment_dict.log_name_rotations):
        os.remove(experiment_dict.log_name_rotations)
    for bg_name in os.listdir(experiment_dict.BG_estimations_folder):
        os.remove(os.path.join(experiment_dict.BG_estimations_folder, bg_name))

    # crop sizes
    experiment_dict.original_height = original_height
    experiment_dict.original_width = original_width
    experiment_dict.alignment_crop_h, experiment_dict.alignment_crop_w = (original_height - 10, original_width - 10)
    experiment_dict.final_crop_h, experiment_dict.final_crop_w = (original_height - 50, original_width - 50)

    # result avis' objects
    experiment_dict.save_video_fps = 25
    stupid_format = cv2.VideoWriter_fourcc(*'MP42')
    experiment_dict.video_object_original = cv2.VideoWriter(experiment_dict.video_name_original,
                                                            stupid_format, float(experiment_dict.save_video_fps),
                                                            (experiment_dict.final_crop_w, experiment_dict.final_crop_h))
    experiment_dict.video_object_aligned = cv2.VideoWriter(experiment_dict.video_name_aligned, stupid_format,
                                                           float(experiment_dict.save_video_fps),
                                                           (experiment_dict.final_crop_w, experiment_dict.final_crop_h))
    experiment_dict.video_object_outliers = cv2.VideoWriter(experiment_dict.video_name_outliers, stupid_format,
                                                            float(experiment_dict.save_video_fps),
                                                            (experiment_dict.final_crop_w, experiment_dict.final_crop_h))
    experiment_dict.video_object_diff = cv2.VideoWriter(experiment_dict.video_name_diff, stupid_format,
                                                        float(experiment_dict.save_video_fps),
                                                        (experiment_dict.final_crop_w, experiment_dict.final_crop_h))

    # image filenames
    experiment_dict.image_filenames = read_image_filenames_from_folder(path=original_frames_folder,
                                                                       number_of_images=np.inf,
                                                                       allowed_extentions=['.png', '.Bin'],
                                                                       flag_recursive=True,
                                                                       string_pattern_to_search='*')

    # background estimation parameters
    experiment_dict.number_of_images_for_bg_estimation = 50
    experiment_dict.image_index_to_start_from_bg_estimation = 29  # TODO check for wierd movements and set
    experiment_dict.flag_calculate_separate_shifts_for_bg = True

    # plotting parameters
    experiment_dict.show_video_fps = 10

    # general parameters
    experiment_dict.gpu_idx = 0
    experiment_dict.number_of_frames_per_batch = 80
    experiment_dict.number_of_frames_per_sub_batch = 20
    experiment_dict.number_of_images_total = len(experiment_dict.image_filenames)

    # flags
    experiment_dict.flag_plot = False
    experiment_dict.flag_estimate_bg = True
    experiment_dict.flag_video_from_bin_files = True

    return experiment_dict


# ----- Phase II: Estimate Background -----
def estimate_bg(experiment_dict: EasyDict,
                base_tensor: Tensor,
                crop_size_h: int,
                crop_size_w: int,
                bg_range: (int, int),
                flag_calculate_shifts: bool = True,
                total_shift_h: Tensor = None,
                total_shift_w: Tensor = None,
                total_rotation: Tensor = None):
    if flag_calculate_shifts:
        initial_crop_h = 500
        initial_crop_w = 500

        base_tensor = base_tensor.to(f'cuda:{experiment_dict.gpu_idx}').float()
        input_tensor = crop_torch_batch(base_tensor, (initial_crop_h + 20, initial_crop_w + 20),
                                        crop_style='center')
        T, C, H, W = input_tensor.shape
        total_shift_h = 0
        total_shift_w = 0
        total_rotation = 0

        # I: cross correlation
        # stabilize the tensor with cc
        shift_h, shift_w, _ = circular_cc_shifts(input_tensor, input_tensor[T // 2: T // 2 + 1])
        del _
        torch.cuda.empty_cache()

        total_shift_h += shift_h
        total_shift_w += shift_w

        # warp the tensor
        max_shift_h = shift_h.max()
        max_shift_w = shift_w.max()
        torch.cuda.empty_cache()
        aligned_tensor_1 = shift_matrix_subpixel(input_tensor[:T // 2], -shift_h[:T // 2], -shift_w[:T // 2],
                                                 matrix_FFT=None, warp_method='fft')
        aligned_tensor_2 = shift_matrix_subpixel(input_tensor[T // 2:], -shift_h[T // 2:], -shift_w[T // 2:],
                                                 matrix_FFT=None, warp_method='fft')
        aligned_tensor = torch.cat([aligned_tensor_1, aligned_tensor_2]).to(f'cuda:{experiment_dict.gpu_idx}')

        # crop the tensors
        aligned_tensor = crop_torch_batch(aligned_tensor, (initial_crop_h + 10, initial_crop_w + 10))
        input_tensor = crop_torch_batch(input_tensor, (initial_crop_h + 10, initial_crop_w + 10))

        # free up GPU memory
        del input_tensor
        del aligned_tensor_1
        del aligned_tensor_2
        del shift_h
        del shift_w
        torch.cuda.empty_cache()

        # II: minSAD
        # stabilize the tensor with minSAD
        T, C, H, W = aligned_tensor.shape
        minSAD = minSAD_transforms_layer(1, T - 1, H, W)
        shifts_h, shifts_w, rotational_shifts, scale_shifts, _ = minSAD(
            matrix=aligned_tensor.float(),
            reference_matrix=aligned_tensor[T // 2: T // 2 + 1].float(),
            shift_h_vec=[-2, -1, 0, 1, 2],
            shift_w_vec=[-2, -1, 0, 1, 2],
            rotation_vec=[-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4],
            scale_vec=[1],
            warp_method='bicubic')

        total_shift_h += shifts_h.to(total_shift_h.device)
        total_shift_w += shifts_w.to(total_shift_w.device)
        total_rotation += (rotational_shifts * torch.pi / 180).to(total_shift_h.device)

        del _
        del shifts_h
        del shifts_w
        del rotational_shifts
        del scale_shifts
        torch.cuda.empty_cache()

    # warp entire video
    T = base_tensor.shape[0]
    T1, T2, T3, T4, T5 = T // 5, 2 * T // 5, 3 * T // 5, 4 * T // 5, T

    aligned_whole_tensor_1 = crop_torch_batch(
        affine_warp_matrix(base_tensor[:T1].unsqueeze(0).to(f'cuda:{experiment_dict.gpu_idx}'),
                           -total_shift_h[:T1], -total_shift_w[:T1],
                           -total_rotation[:T1], warp_method='bicubic'), (crop_size_h, crop_size_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_2 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T1: T2].unsqueeze(0).to(f'cuda:{experiment_dict.gpu_idx}'),
                           -total_shift_h[T1: T2], -total_shift_w[T1: T2],
                           -total_rotation[T1: T2], warp_method='bicubic'), (crop_size_h, crop_size_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_3 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T2: T3].unsqueeze(0).to(f'cuda:{experiment_dict.gpu_idx}'),
                           -total_shift_h[T2: T3], -total_shift_w[T2: T3],
                           -total_rotation[T2: T3], warp_method='bicubic'), (crop_size_h, crop_size_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_4 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T3: T4].unsqueeze(0).to(f'cuda:{experiment_dict.gpu_idx}'),
                           -total_shift_h[T3: T4], -total_shift_w[T3: T4],
                           -total_rotation[T3: T4], warp_method='bicubic'), (crop_size_h, crop_size_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_5 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T4:].unsqueeze(0).to(f'cuda:{experiment_dict.gpu_idx}'),
                           -total_shift_h[T4:], -total_shift_w[T4:],
                           -total_rotation[T4:], warp_method='bicubic'), (crop_size_h, crop_size_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor = torch.cat([aligned_whole_tensor_1, aligned_whole_tensor_2, aligned_whole_tensor_3,
                                      aligned_whole_tensor_4, aligned_whole_tensor_5])

    # calculate median
    bg_estimation = aligned_whole_tensor.median(0)[0]

    # save result
    bg_name = os.path.join(experiment_dict.BG_estimations_folder, f'bg_{bg_range[0]}_{bg_range[1]}.pt')
    experiment_dict.BG_estimation_names.append(bg_name)
    torch.save(bg_estimation, bg_name)


def stabilize_batch(input_tensor, input_range, bg_range, experiment_dict):
    # --- I. Get BG Estimation ---
    current_bg_name = os.path.join(experiment_dict.BG_estimations_folder, f'bg_{bg_range[0]}_{bg_range[1]}.pt')
    if not os.path.exists(current_bg_name):
        print("Estimating new BG")
        estimate_bg(experiment_dict=experiment_dict,
                    base_tensor=input_tensor[experiment_dict.image_index_to_start_from_bg_estimation:
                                             experiment_dict.number_of_images_for_bg_estimation],
                    crop_size_h=experiment_dict.alignment_crop_h,
                    crop_size_w=experiment_dict.alignment_crop_w,
                    bg_range=bg_range,
                    flag_calculate_shifts=True)
    bg_tensor = torch.load(current_bg_name).unsqueeze(0)

    # --- II. Crop BG and Input to the Same Size ---
    alignment_h, alignment_w = experiment_dict.alignment_crop_h, experiment_dict.alignment_crop_w
    input_tensor = crop_torch_batch(input_tensor, (alignment_h, alignment_w))
    bg_tensor = crop_torch_batch(bg_tensor, (alignment_h, alignment_w))

    # --- III. Stabilize Tensor ---
    print("Stabilizing tensor")
    # divide to multiple sub batches (limited by the available memory)
    aligned_tensor = []
    total_shift_h = []
    total_shift_w = []
    total_rotation = []
    sub_batch_size = experiment_dict.number_of_frames_per_sub_batch
    batch_size = experiment_dict.number_of_frames_per_batch
    num_sub_batches = int(ceil(batch_size / sub_batch_size))
    for sub_batch_index in range(0, num_sub_batches):
        sub_batch_start = sub_batch_index * sub_batch_size
        sub_input_tensor = input_tensor[sub_batch_start: sub_batch_start + sub_batch_size]
        _, _, sub_aligned_tensor, shift_h, shift_w, rotation = \
            align_palantir_CC_ECC_CC(sub_input_tensor, bg_tensor)
        # sum up results to total result
        del _
        sub_aligned_tensor = pad_torch_batch(sub_aligned_tensor, (alignment_h, alignment_w))
        aligned_tensor += [sub_aligned_tensor]
        total_shift_h += shift_h
        total_shift_w += shift_w
        total_rotation += rotation
    aligned_tensor = torch.cat(aligned_tensor)
    total_shift_h = Tensor(total_shift_h)
    total_shift_w = Tensor(total_shift_w)
    total_rotation = Tensor(total_rotation)

    # --- IV. Make BG and Input of same size ---
    final_h, final_w = experiment_dict.final_crop_h, experiment_dict.final_crop_w
    aligned_tensor = crop_torch_batch(aligned_tensor, (final_h, final_w))
    aligned_tensor = pad_torch_batch(aligned_tensor, (final_h, final_w))
    bg_tensor = crop_torch_batch(bg_tensor, (final_h, final_w))

    # --- V. Find Outliers ---
    print("Finding outliers")
    T = input_tensor.shape[0]
    outliers_tensor_1 = find_outliers(aligned_tensor[: T // 2], bg_tensor.cuda(), noise_mult_factor=20).cpu()
    outliers_tensor_2 = find_outliers(aligned_tensor[T // 2:], bg_tensor.cuda(), noise_mult_factor=20).cpu()
    outliers_tensor = torch.cat([outliers_tensor_1, outliers_tensor_2])

    # --- VI. Calculate Diff tensor ---
    print("Calculating difference tensor")
    diff_tensor = (aligned_tensor.cpu() - bg_tensor).abs()
    diff_tensor = scale_array_stretch_hist(diff_tensor, (0.1, 0.99))

    # --- VII. Save Results ---
    print("Saving results")
    # transfer aligned tensor to cpu for memory management:
    tensor_for_bg_estimation = input_tensor
    input_tensor = input_tensor.cpu()
    bg_tensor = bg_tensor.cpu()
    aligned_tensor = aligned_tensor.cpu()
    outliers_tensor = outliers_tensor.cpu()
    torch.cuda.empty_cache()

    # add shifts for graphs
    with open(experiment_dict.log_name_shifts_h, 'ab') as hf:
        np.save(hf, array(total_shift_h.cpu()))
    with open(experiment_dict.log_name_shifts_w, 'ab') as wf:
        np.save(wf, array(total_shift_w.cpu()))
    with open(experiment_dict.log_name_rotations, 'ab') as rf:
        np.save(rf, array(total_rotation.cpu()))

    # prepare frames for video - make numpy arrays of exact size (final_crop_h, final_crop_w) and of 3 channels
    aligned_tensor = torch_to_numpy(aligned_tensor)
    aligned_tensor = crop_numpy_batch(aligned_tensor, (final_h, final_w), crop_style='center')
    aligned_tensor = pad_numpy_batch(aligned_tensor, (final_h, final_w), pad_style='center')

    # write aligned video to .bin file
    with open(experiment_dict.bin_name_aligned, 'ab') as f:
        f.write(aligned_tensor[..., :1].tobytes())

    # turn aligned tensor to be video-ready for save/dispaly:
    aligned_tensor = numpy_array_to_video_ready(aligned_tensor)

    # write video to avi file
    T = input_tensor.shape[0]
    for frame_index in np.arange(T):
        experiment_dict.video_object_aligned.write(aligned_tensor[frame_index])

    # delete tensor for memory saving
    del aligned_tensor

    # prepare and write the rest of the tensors to videos:
    # current batch:
    input_tensor = torch_to_numpy(input_tensor)
    input_tensor = crop_numpy_batch(input_tensor, (final_h, final_w), crop_style='center')
    input_tensor = pad_numpy_batch(input_tensor, (final_h, final_w), pad_style='center')
    input_tensor = numpy_array_to_video_ready(input_tensor)
    # write video to avi file
    for frame_index in np.arange(T):
        experiment_dict.video_object_original.write(input_tensor[frame_index])
    # delete tensor for memory saving
    del input_tensor

    # outlier tensor
    outliers_tensor = torch_to_numpy(outliers_tensor)
    outliers_tensor = crop_numpy_batch(outliers_tensor, (final_h, final_w), crop_style='center')
    outliers_tensor = pad_numpy_batch(outliers_tensor, (final_h, final_w), pad_style='center')
    outliers_tensor = numpy_array_to_video_ready(outliers_tensor)
    # write video to avi file
    for frame_index in np.arange(T):
        experiment_dict.video_object_outliers.write(outliers_tensor[frame_index])
    # delete tensor for memory saving
    del outliers_tensor

    # outlier tensor
    diff_tensor = torch_to_numpy(diff_tensor)
    diff_tensor = crop_numpy_batch(diff_tensor, (final_h, final_w), crop_style='center')
    diff_tensor = pad_numpy_batch(diff_tensor, (final_h, final_w), pad_style='center')
    diff_tensor = numpy_array_to_video_ready(diff_tensor)
    # write video to avi file
    for frame_index in np.arange(T):
        experiment_dict.video_object_diff.write(diff_tensor[frame_index])
    # delete tensor for memory saving
    del diff_tensor


    # --- VI. Estimate New BG ---
    # To avoid decrease in frame size over time, it is needed to reverse the crop size
    print("Estimating new BG")
    estimate_bg(experiment_dict=experiment_dict,
                base_tensor=tensor_for_bg_estimation[-experiment_dict.number_of_images_for_bg_estimation:],
                crop_size_h=experiment_dict.alignment_crop_h,
                crop_size_w=experiment_dict.alignment_crop_w,
                bg_range=input_range,
                flag_calculate_shifts=experiment_dict.flag_calculate_separate_shifts_for_bg,
                total_shift_h=total_shift_h[-experiment_dict.number_of_images_for_bg_estimation:],
                total_shift_w=total_shift_w[-experiment_dict.number_of_images_for_bg_estimation:],
                total_rotation=total_rotation[-experiment_dict.number_of_images_for_bg_estimation:])

    # --- VII. Return Results ---
    # return input_tensor, bg_tensor, aligned_tensor, outliers_tensor, diff_tensor


if __name__ == '__main__':
    for experiment_idx in range(1, 20):
        original_frames_folder = f'/home/dudy/Projects/data/Palantir/p{experiment_idx}'
        results_path = '/home/dudy/Projects/temp/temp_palantir_results'
        experiment_name = f'whole_video_multi_bg_estimation_ecc_homography'
        original_height = 540
        original_width = 8192
        experiment_dict = initialize_experiment(original_frames_folder=original_frames_folder,
                                                results_path=results_path,
                                                experiment_name=experiment_name,
                                                sub_experiment_idx=experiment_idx,
                                                original_height=540,
                                                original_width=8192)

        batch_size = experiment_dict.number_of_frames_per_batch
        num_frames = experiment_dict.number_of_images_total
        num_batches = int(ceil(num_frames / batch_size))
        input_range = (0, min(num_frames, batch_size))
        for batch_index in range(num_batches):
            print(f'BATCH: {batch_index}')
            start_video = batch_index * batch_size
            bg_range = input_range
            input_range = (start_video, min(num_frames, start_video + batch_size))
            # read batch of images
            video_tensor = read_bin_images_batch(
                filenames=experiment_dict.image_filenames[input_range[0]: input_range[1]],
                height=original_height, width=original_width).float()
            video_tensor = video_tensor.unsqueeze(1)
            stabilize_batch(video_tensor, input_range=input_range, bg_range=bg_range, experiment_dict=experiment_dict)
            # _, _, _, _, _ = stabilize_batch(video_tensor, input_range=input_range, bg_range=bg_range, experiment_dict=experiment_dict)
            # del _
            torch.cuda.empty_cache()

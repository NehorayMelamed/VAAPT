
import torch
from torch import Tensor

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import crop_torch_batch
from alignments import circular_cc_shifts
from alignments_layers import minSAD_transforms_layer
from transforms import shift_matrix_subpixel, affine_warp_matrix


def calculate_bg(base_tensor: Tensor, params):
    initial_crop_h = 500
    initial_crop_w = 500
    gpu_idx = params.local.gpu_idx
    base_tensor = base_tensor.to(f'cuda:{gpu_idx}').float()
    base_tensor = base_tensor.reshape(base_tensor.shape[0], 1, *base_tensor.shape[1:])
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

    torch.cuda.empty_cache()
    aligned_tensor_1 = shift_matrix_subpixel(input_tensor[:T // 2], -shift_h[:T // 2], -shift_w[:T // 2],
                                             matrix_FFT=None, warp_method='fft')
    aligned_tensor_2 = shift_matrix_subpixel(input_tensor[T // 2:], -shift_h[T // 2:], -shift_w[T // 2:],
                                             matrix_FFT=None, warp_method='fft')
    aligned_tensor = torch.cat([aligned_tensor_1, aligned_tensor_2]).to(f'cuda:{gpu_idx}')

    # crop the tensors
    aligned_tensor = crop_torch_batch(aligned_tensor, (initial_crop_h + 10, initial_crop_w + 10))

    # free up GPU memory
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
        shift_w_vec=[-3, -2, -1, 0, 1, 2, 3],
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
        affine_warp_matrix(base_tensor[:T1].unsqueeze(0).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[:T1], -total_shift_w[:T1],
                           -total_rotation[:T1], warp_method='bicubic'), (params.algo.alignment_crop_h, params.algo.alignment_crop_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_2 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T1: T2].unsqueeze(0).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T1: T2], -total_shift_w[T1: T2],
                           -total_rotation[T1: T2], warp_method='bicubic'), (params.algo.alignment_crop_h, params.algo.alignment_crop_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_3 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T2: T3].unsqueeze(0).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T2: T3], -total_shift_w[T2: T3],
                           -total_rotation[T2: T3], warp_method='bicubic'), (params.algo.alignment_crop_h, params.algo.alignment_crop_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_4 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T3: T4].unsqueeze(0).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T3: T4], -total_shift_w[T3: T4],
                           -total_rotation[T3: T4], warp_method='bicubic'), (params.algo.alignment_crop_h, params.algo.alignment_crop_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_5 = crop_torch_batch(
        affine_warp_matrix(base_tensor[T4:].unsqueeze(0).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T4:], -total_shift_w[T4:],
                           -total_rotation[T4:], warp_method='bicubic'), (params.algo.alignment_crop_h, params.algo.alignment_crop_w)).squeeze(0).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor = torch.cat([aligned_whole_tensor_1, aligned_whole_tensor_2, aligned_whole_tensor_3,
                                      aligned_whole_tensor_4, aligned_whole_tensor_5])
    # calculate median
    bg_estimation = aligned_whole_tensor.median(0)[0]
    return bg_estimation
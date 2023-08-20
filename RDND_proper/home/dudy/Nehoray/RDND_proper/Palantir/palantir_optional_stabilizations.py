import kornia.geometry
import torch
from torch import Tensor

from Dudy import Align_ECC, get_homography_from_optical_flow, affine_parameters_from_homography_matrix
from Palantir.Palantir_utils import crop_size_after_homography, center_crop
from RapidBase.Utils.Classical_DSP.ECC import ECC_torch, ECC_torch_batch, ECC_torch_batch_time_stride
from RapidBase.Utils.IO.tic_toc import tic, toc
from alignments import align_to_reference_frame_circular_cc
from alignments_layers import minSAD_transforms_layer, Gimbaless_Rotation_Layer_Torch
from transforms import shift_matrix_subpixel, rotate_matrix, affine_warp_matrix
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import crop_torch_batch, crop_tensor
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch_video, imshow_torch

show_video_fps = 10
gpu_idx = 0


def align_palantir_CC_ECC_CC_Gimbaless(input_tensor, reference_tensor, flag_plot=False):
    """Aligns given tensor to reference with the following methods:
       Circular Cross Correlation (CC)
       Enhanced Correlation Coefficient (ECC)
       Circular Cross Correlation (CC)
       Gimbaless translation
    TODO optional: return homography parameters for warping later over entire video while stabilization only performed
     over a fixed cropped section
    TODO I believe there is a delicate rotation yet, check it out with minSAD or so

    :param input_tensor:
    :param reference_tensor:
    :param flag_plot:
    :return:
    """
    T, C, H, W = input_tensor.shape
    total_shift_h = 0
    total_shift_w = 0
    total_shift_rotation = 0

    # I: Cross Correlation
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                          reference_matrix=reference_tensor,
                                                                                          matrix_fft=None,
                                                                                          normalize_over_matrix=False,
                                                                                          warp_method='fft',
                                                                                          crop_warped_matrix=False)

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC
    total_shift_w = total_shift_w + shifts_w_CC

    # crop tensors to the same size
    max_shift_h = shifts_h_CC.abs().max()
    max_shift_w = shifts_w_CC.abs().max()
    new_H = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC - aligned_tensor_CC[T//2: T//2+1], FPS=show_video_fps)

    # free up GPU memory
    input_tensor = input_tensor.cpu()

    # II: Enhanced Correlation Coefficient
    aligned_tensor_CC_ECC, _, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = Align_ECC(aligned_tensor_CC, reference_tensor)
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC_ECC
    total_shift_w = total_shift_w + shifts_w_CC_ECC
    total_shift_rotation = total_shift_rotation + shifts_rotation_CC_ECC

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T//2: T//2+1], FPS=show_video_fps)

    # III: Cross Correlation
    aligned_tensor_CC_ECC_CC, shifts_h_CC_ECC_CC, shifts_w_CC_ECC_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=aligned_tensor_CC_ECC,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='fft',
                                             crop_warped_matrix=False)

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC_ECC_CC
    total_shift_w = total_shift_w + shifts_w_CC_ECC_CC

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    max_shift_H_CC_ECC_CC = shifts_h_CC_ECC_CC.abs().max()
    max_shift_W_CC_ECC_CC = shifts_w_CC_ECC_CC.abs().max()
    new_H_CC_ECC_CC = new_H - (max_shift_H_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC_CC = new_W - (max_shift_W_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC_ECC_CC = crop_torch_batch(aligned_tensor_CC_ECC_CC, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    # Free up GPU memory:
    del aligned_tensor_CC_ECC

    # IV: Gimbaless translation
    # create meshgrid (better move it out later)
    y = torch.arange(new_H_CC_ECC_CC).to(aligned_tensor_CC_ECC_CC.device)
    x = torch.arange(new_W_CC_ECC_CC).to(aligned_tensor_CC_ECC_CC.device)
    y_meshgrid, x_meshgrid = torch.meshgrid(y, x)
    y_meshgrid = torch.stack([y_meshgrid] * T, dim=0).unsqueeze(0)
    x_meshgrid = torch.stack([x_meshgrid] * T, dim=0).unsqueeze(0)

    # get homography matirx
    shifts_h_CC_ECC_CC_Gimbaless, shift_w_CC_ECC_CC_Gimbaless, shifts_rotation_CC_ECC_CC_Gimbaless,\
    scale, rotation_degrees = get_homography_from_optical_flow(aligned_tensor_CC_ECC_CC,
                                                               reference_tensor.unsqueeze(0),
                                                               gimbaless_block_size=32,
                                                               gimbaless_overlap=31,
                                                               input_meshgrid=(y_meshgrid, x_meshgrid))

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC_ECC_CC_Gimbaless
    total_shift_w = total_shift_w + shift_w_CC_ECC_CC_Gimbaless
    total_shift_rotation = total_shift_rotation + shifts_rotation_CC_ECC_CC_Gimbaless

    # free up gpu memory
    del y_meshgrid
    del x_meshgrid

    # warp matrix
    aligned_tensor_CC_ECC_CC_Gimbaless = shift_matrix_subpixel(aligned_tensor_CC_ECC_CC,
                                                               shift_H=shifts_h_CC_ECC_CC_Gimbaless,
                                                               shift_W=shift_w_CC_ECC_CC_Gimbaless,
                                                               warp_method='fft')
    aligned_tensor_CC_ECC_CC_Gimbaless = rotate_matrix(aligned_tensor_CC_ECC_CC_Gimbaless,
                                                       thetas=shifts_rotation_CC_ECC_CC_Gimbaless,
                                                       warp_method='fft')

    aligned_tenser_final = aligned_tensor_CC_ECC_CC_Gimbaless
    return aligned_tenser_final, total_shift_h, total_shift_w, total_shift_rotation


def align_palantir_CC_ECC_CC_minSAD(input_tensor, reference_tensor, flag_plot=False):
    """Aligns given tensor to reference with the following methods:
       Circular Cross Correlation (CC)
       Enhanced Correlation Coefficient (ECC)
       Circular Cross Correlation (CC)
       Minimum Sum of Difference (minSAD)
    TODO optional: return homography parameters for warping later over entire video while stabilization only performed
     over a fixed cropped section

    :param input_tensor:
    :param reference_tensor:
    :param flag_plot:
    :return:
    """
    T, C, H, W = input_tensor.shape
    total_shift_h = 0
    total_shift_w = 0
    total_shift_rotation = 0
    affine_matrix = torch.stack([torch.eye(3, 3)] * T)

    # I: Cross Correlation
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                          reference_matrix=reference_tensor,
                                                                                          matrix_fft=None,
                                                                                          normalize_over_matrix=False,
                                                                                          warp_method='bicubic',
                                                                                          crop_warped_matrix=False)

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC
    total_shift_w = total_shift_w + shifts_w_CC
    affine_matrix_cc = torch.stack([torch.eye(3, 3)] * T)
    affine_matrix_cc[:, 0, 2] = -shifts_w_CC
    affine_matrix_cc[:, 1, 2] = -shifts_h_CC
    affine_matrix = affine_matrix_cc @ affine_matrix
    # affine_matrix = affine_matrix @ affine_matrix_cc

    # crop tensors to the same size
    max_shift_h = shifts_h_CC.abs().max()
    max_shift_w = shifts_w_CC.abs().max()
    new_H = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC - aligned_tensor_CC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    input_tensor = input_tensor.cpu()
    shifts_h_CC = shifts_h_CC.cpu()
    shifts_w_CC = shifts_w_CC.cpu()

    # II: Enhanced Correlation Coefficient
    aligned_tensor_CC_ECC, H_matrix, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = Align_ECC(aligned_tensor_CC,
                                                                                                      reference_tensor)
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC_ECC
    total_shift_w = total_shift_w + shifts_w_CC_ECC
    total_shift_rotation = total_shift_rotation + shifts_rotation_CC_ECC
    affine_matrix = H_matrix.to(torch.float32).cpu() @ affine_matrix
    # affine_matrix = affine_matrix @ H_matrix.to(torch.float32).cpu()

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # III: Cross Correlation
    aligned_tensor_CC_ECC_CC, shifts_h_CC_ECC_CC, shifts_w_CC_ECC_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=aligned_tensor_CC_ECC,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC_ECC_CC
    total_shift_w = total_shift_w + shifts_w_CC_ECC_CC
    affine_matrix_cc_ecc_cc = torch.stack([torch.eye(3, 3)] * T)
    affine_matrix_cc_ecc_cc[:, 0, 2] = -shifts_w_CC_ECC_CC
    affine_matrix_cc_ecc_cc[:, 1, 2] = -shifts_h_CC_ECC_CC
    affine_matrix = affine_matrix_cc_ecc_cc @ affine_matrix
    # affine_matrix = affine_matrix @ affine_matrix_cc_ecc_cc

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    max_shift_H_CC_ECC_CC = shifts_h_CC_ECC_CC.abs().max()
    max_shift_W_CC_ECC_CC = shifts_w_CC_ECC_CC.abs().max()
    new_H_CC_ECC_CC = new_H - (max_shift_H_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC_CC = new_W - (max_shift_W_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC_ECC_CC = crop_torch_batch(aligned_tensor_CC_ECC_CC, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    # Free up GPU memory:
    del aligned_tensor_CC_ECC
    shifts_h_CC_ECC_CC = shifts_h_CC_ECC_CC.cpu()
    shifts_w_CC_ECC_CC = shifts_w_CC_ECC_CC.cpu()

    # IV: minSAD for delicate rotations
    minSAD = minSAD_transforms_layer(1, T, H, W)
    aligned_tensor_CC_ECC_CC_minSAD, shifts_h_CC_ECC_CC_minSAD, shifts_w_CC_ECC_CC_minSAD, \
    shifts_rotation_CC_ECC_CC_minSAD, _, _ = minSAD.align(matrix=aligned_tensor_CC_ECC_CC,
                                                          reference_matrix=reference_tensor,
                                                          shift_h_vec=[-0.05, -0.025, 0, 0.025, 0.05],
                                                          shift_w_vec=[-0.05, -0.025, 0, 0.025, 0.05],
                                                          rotation_vec=[-0.015, -0.01, -0.005, 0, 0.005, 0.01, 0.015],
                                                          scale_vec=[1],
                                                          warp_method='bicubic',
                                                          return_shifts=True)

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC_ECC_CC_minSAD
    total_shift_w = total_shift_w + shifts_w_CC_ECC_CC_minSAD
    total_shift_rotation = total_shift_rotation + shifts_rotation_CC_ECC_CC_minSAD
    affine_matrix_cc_ecc_cc_minsad = torch.stack([torch.eye(3, 3)] * T)
    affine_matrix_cc_ecc_cc_minsad[:, 0, 2] = -shifts_w_CC_ECC_CC_minSAD
    affine_matrix_cc_ecc_cc_minsad[:, 1, 2] = -shifts_h_CC_ECC_CC_minSAD
    sin_theta = torch.sin(shifts_rotation_CC_ECC_CC_minSAD)
    cos_theta = torch.cos(shifts_rotation_CC_ECC_CC_minSAD)
    affine_matrix_cc_ecc_cc_minsad[:, 0, 0] = cos_theta
    affine_matrix_cc_ecc_cc_minsad[:, 0, 1] = -sin_theta
    affine_matrix_cc_ecc_cc_minsad[:, 1, 0] = sin_theta
    affine_matrix_cc_ecc_cc_minsad[:, 1, 1] = cos_theta
    affine_matrix = affine_matrix_cc_ecc_cc_minsad @ affine_matrix
    # affine_matrix = affine_matrix @ affine_matrix_cc_ecc_cc_minsad

    affine_matrix = affine_matrix[:, :2]

    # crop tensors
    aligned_tensor_CC_ECC_CC_minSAD = crop_torch_batch(aligned_tensor_CC_ECC_CC_minSAD, (new_H_CC_ECC_CC, new_W_CC_ECC_CC))

    aligned_tenser_final = aligned_tensor_CC_ECC_CC_minSAD
    return aligned_tenser_final, affine_matrix, total_shift_h, total_shift_w, total_shift_rotation


def align_palantir_CC_ECC_CC(input_tensor: Tensor,
                             reference_tensor: Tensor,
                             ECC_warp_mode: str = 'homography',
                             ECC_warp_matrix: Tensor = torch.eye(3, 3),
                             ECC_iterations: int = 100,
                             flag_plot=False):
    """Aligns given tensor to reference with the following methods:
       Circular Cross Correlation (CC)
       Enhanced Correlation Coefficient (ECC)
       Circular Cross Correlation (CC)

    :param input_tensor:
    :param reference_tensor:
    :param flag_plot:
    :return:
    """
    T, C, H, W = input_tensor.shape
    gpu_usage_log = [torch.cuda.memory_allocated(0)/1024/1024/1024]
    total_shift_h = torch.zeros(T).to(f'cuda:{gpu_idx}')
    total_shift_w = torch.zeros(T).to(f'cuda:{gpu_idx}')
    total_shift_rotation = torch.zeros(T).to(f'cuda:{gpu_idx}')

    input_tensor = input_tensor.to(f'cuda:{gpu_idx}')
    reference_tensor = reference_tensor.to(f'cuda:{gpu_idx}')

    # I: Cross Correlation
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                          reference_matrix=reference_tensor,
                                                                                          matrix_fft=None,
                                                                                          normalize_over_matrix=False,
                                                                                          warp_method='bicubic',
                                                                                          crop_warped_matrix=False)
    gpu_usage_log += [torch.cuda.memory_allocated(0)/1024/1024/1024]

    # crop tensors to the same size
    max_shift_h = shifts_h_CC.abs().max()
    max_shift_w = shifts_w_CC.abs().max()
    new_H_CC = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, [new_H_CC, new_W_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC, new_W_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC, new_W_CC])

    # sum up shifts
    total_shift_h += shifts_h_CC
    total_shift_w += shifts_w_CC

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC - aligned_tensor_CC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    input_tensor = input_tensor.cpu()
    del shifts_h_CC
    del shifts_w_CC
    del _
    torch.cuda.empty_cache()

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # II: Enhanced Correlation Coefficient
    aligned_tensor_CC_ECC, H_matrix, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = \
        Align_ECC(aligned_tensor_CC, reference_tensor, warp_mode=ECC_warp_mode, warp_matrix=ECC_warp_matrix,
                  iterations=ECC_iterations)
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    max_rotation_shift_H_CC_ECC = (W / 2) * torch.tan(shifts_rotation_CC_ECC).abs().max()
    max_rotation_shift_W_CC_ECC = (H / 2) * torch.tan(shifts_rotation_CC_ECC).abs().max()
    max_shift_H_CC_ECC = max((shifts_h_CC_ECC.min() + max_rotation_shift_H_CC_ECC).abs(),
                             shifts_h_CC_ECC.max() + max_rotation_shift_H_CC_ECC)
    max_shift_W_CC_ECC = max((shifts_w_CC_ECC.min() + max_rotation_shift_W_CC_ECC).abs(),
                             shifts_w_CC_ECC.max() + max_rotation_shift_W_CC_ECC)

    new_H_CC_ECC = new_H_CC - (max_shift_H_CC_ECC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC = new_W_CC - (max_shift_W_CC_ECC.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC_ECC = crop_torch_batch(aligned_tensor_CC_ECC, [new_H_CC_ECC, new_W_CC_ECC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC_ECC, new_W_CC_ECC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC_ECC, new_W_CC_ECC])

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # sum up shifts
    total_shift_h += shifts_h_CC_ECC
    total_shift_w += shifts_w_CC_ECC
    total_shift_rotation += shifts_rotation_CC_ECC

    # free up GPU memory
    del shifts_rotation_CC_ECC
    del shifts_h_CC_ECC
    del shifts_w_CC_ECC
    del _
    torch.cuda.empty_cache()

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # III: Cross Correlation
    aligned_tensor_CC_ECC_CC, shifts_h_CC_ECC_CC, shifts_w_CC_ECC_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=aligned_tensor_CC_ECC,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    max_shift_H_CC_ECC_CC = shifts_h_CC_ECC_CC.abs().max()
    max_shift_W_CC_ECC_CC = shifts_w_CC_ECC_CC.abs().max()
    new_H_CC_ECC_CC = new_H_CC_ECC - (max_shift_H_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC_CC = new_W_CC_ECC - (max_shift_W_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC_ECC_CC = crop_torch_batch(aligned_tensor_CC_ECC_CC, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    # sum up shifts
    total_shift_h += shifts_h_CC_ECC_CC
    total_shift_w += shifts_w_CC_ECC_CC

    # Free up GPU memory:
    del aligned_tensor_CC_ECC
    del shifts_h_CC_ECC_CC
    del shifts_w_CC_ECC_CC
    del _
    torch.cuda.empty_cache()

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    aligned_tenser_final = aligned_tensor_CC_ECC_CC
    return input_tensor, reference_tensor, aligned_tenser_final, H_matrix, total_shift_h, total_shift_w, total_shift_rotation


def align_palantir_CC_ECC_CC_pytorch(input_tensor: Tensor,
                                     reference_tensor: Tensor,
                                     ECC_warp_mode: str = 'homography',
                                     ECC_warp_matrix: Tensor = torch.eye(3, 3),
                                     ECC_iterations: int = 50,
                                     flag_plot=False):
    """Aligns given tensor to reference with the following methods:
           Circular Cross Correlation (CC)
           Enhanced Correlation Coefficient (ECC)
           Circular Cross Correlation (CC)

        :param input_tensor:
        :param reference_tensor:
        :param flag_plot:
        :return:
        """
    T, C, H, W = input_tensor.shape

    input_tensor = input_tensor.to(f'cuda:{gpu_idx}')
    reference_tensor = reference_tensor.to(f'cuda:{gpu_idx}')

    # I: Cross Correlation
    tic()
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                          reference_matrix=reference_tensor,
                                                                                          matrix_fft=None,
                                                                                          normalize_over_matrix=False,
                                                                                          warp_method='bicubic',
                                                                                          crop_warped_matrix=False)
    toc(f"First CC")
    # crop tensors to the same size
    tic()
    max_shift_h = shifts_h_CC.abs().max()
    max_shift_w = shifts_w_CC.abs().max()
    new_H_CC = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, [new_H_CC, new_W_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC, new_W_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC, new_W_CC])

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC - aligned_tensor_CC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    del shifts_h_CC
    del shifts_w_CC
    del _
    torch.cuda.empty_cache()
    toc("First CC Cleaning up")
    # II. Enhanced Cross Correlation
    centered = True
    if centered:
        tic()
        center_cropped_tensor = center_crop(aligned_tensor_CC, (300, 300))
        center_cropped_reference = center_crop(reference_tensor, (300, 300))
        toc("Center crop for ECC centered")
    tic()
    H_matrix_ECC, aligned_tensor_CC_ECC, num_iterations_list, final_delta_shift_list =\
        ECC_torch_batch(center_cropped_tensor, center_cropped_reference,
                        number_of_levels=1,
                        number_of_iterations_per_level=ECC_iterations,
                        transform_string=ECC_warp_mode,
                        delta_p_init=ECC_warp_matrix,
                        flag_update_delta_p_init=True)
    toc("ECC")

    tic()
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')
    # crop tensors to the same size
    # find maximal necessary crop sizes for the batch
    h_start_vec, h_end_vec, w_start_vec, w_end_vec = crop_size_after_homography(H_matrix_ECC, new_H_CC, new_W_CC)
    h_start = int(max(h_start_vec))
    h_end = int(min(h_end_vec))
    w_start = int(max(w_start_vec))
    w_end = int(min(w_end_vec))

    new_H_CC_ECC = h_end - h_start
    new_W_CC_ECC = w_end - w_start

    aligned_tensor_CC_ECC = crop_tensor(images=aligned_tensor_CC_ECC,
                                        crop_size_tuple_or_scalar=(new_H_CC_ECC, new_W_CC_ECC),
                                        crop_style='predetrmined',
                                        start_H=h_start,
                                        start_W=w_start)
    input_tensor = crop_tensor(images=input_tensor,
                               crop_size_tuple_or_scalar=(new_H_CC_ECC, new_W_CC_ECC),
                               crop_style='predetrmined',
                               start_H=h_start,
                               start_W=w_start)
    reference_tensor = crop_tensor(images=reference_tensor,
                                   crop_size_tuple_or_scalar=(new_H_CC_ECC, new_W_CC_ECC),
                                   crop_style='predetrmined',
                                   start_H=h_start,
                                   start_W=w_start)

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T // 2: T // 2 + 1], FPS=show_video_fps)


    # free up GPU memory
    del h_start_vec, h_end_vec, w_start_vec, w_end_vec
    torch.cuda.empty_cache()
    toc("ECC cleanup")
    # III: Cross Correlation
    tic()
    aligned_tensor_CC_ECC_CC, shifts_h_CC_ECC_CC, shifts_w_CC_ECC_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=aligned_tensor_CC_ECC,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)
    toc("Second CC")

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    tic()
    max_shift_H_CC_ECC_CC = shifts_h_CC_ECC_CC.abs().max()
    max_shift_W_CC_ECC_CC = shifts_w_CC_ECC_CC.abs().max()
    new_H_CC_ECC_CC = new_H_CC_ECC - (max_shift_H_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC_CC = new_W_CC_ECC - (max_shift_W_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC_ECC_CC = crop_torch_batch(aligned_tensor_CC_ECC_CC, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    # Free up GPU memory:
    del aligned_tensor_CC_ECC
    del shifts_h_CC_ECC_CC
    del shifts_w_CC_ECC_CC
    del _
    torch.cuda.empty_cache()

    aligned_tenser_final = aligned_tensor_CC_ECC_CC
    toc("Second CC cleanup")
    return input_tensor, reference_tensor, aligned_tenser_final, H_matrix_ECC, \
           num_iterations_list, final_delta_shift_list


def align_palantir_CC_ECC_CC_pytorch_time_stride(input_tensor: Tensor,
                                                 reference_tensor: Tensor,
                                                 ECC_warp_mode: str = 'homography',
                                                 ECC_warp_matrix: Tensor = torch.eye(3, 3),
                                                 ECC_iterations: int = 50,
                                                 ECC_time_stride: int = 2,
                                                 flag_plot=False):
    """Aligns given tensor to reference with the following methods:
           Circular Cross Correlation (CC)
           Enhanced Correlation Coefficient (ECC)
           Circular Cross Correlation (CC)

        :param input_tensor:
        :param reference_tensor:
        :param ECC_warp_mode:
        :param ECC_warp_matrix:
        :param ECC_iterations:
        :param ECC_time_stride:
        :param flag_plot:

        :return:
        """
    T, C, H, W = input_tensor.shape

    input_tensor = input_tensor.to(f'cuda:{gpu_idx}')
    reference_tensor = reference_tensor.to(f'cuda:{gpu_idx}')

    # I: Cross Correlation
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                          reference_matrix=reference_tensor,
                                                                                          matrix_fft=None,
                                                                                          normalize_over_matrix=False,
                                                                                          warp_method='bicubic',
                                                                                          crop_warped_matrix=False)

    # crop tensors to the same size
    max_shift_h = shifts_h_CC.abs().max()
    max_shift_w = shifts_w_CC.abs().max()
    new_H_CC = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, [new_H_CC, new_W_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC, new_W_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC, new_W_CC])

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC - aligned_tensor_CC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    del shifts_h_CC
    del shifts_w_CC
    del _
    torch.cuda.empty_cache()

    # II. Enhanced Cross Correlation
    H_matrix_ECC, aligned_tensor_CC_ECC, num_iterations_list, final_delta_shift_list =\
        ECC_torch_batch_time_stride(aligned_tensor_CC, reference_tensor,
                                    number_of_levels=1,
                                    number_of_iterations_per_level=ECC_iterations,
                                    transform_string=ECC_warp_mode,
                                    delta_p_init=ECC_warp_matrix,
                                    time_stride=ECC_time_stride,
                                    flag_update_delta_p_init=True)

    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')

    # crop tensors to the same size
    # find maximal necessary crop sizes for the batch
    h_start_vec, h_end_vec, w_start_vec, w_end_vec = crop_size_after_homography(H_matrix_ECC, new_H_CC, new_W_CC)
    h_start = int(max(h_start_vec))
    h_end = int(min(h_end_vec))
    w_start = int(max(w_start_vec))
    w_end = int(min(w_end_vec))

    new_H_CC_ECC = h_end - h_start
    new_W_CC_ECC = w_end - w_start

    aligned_tensor_CC_ECC = crop_tensor(images=aligned_tensor_CC_ECC,
                                        crop_size_tuple_or_scalar=(new_H_CC_ECC, new_W_CC_ECC),
                                        crop_style='predetrmined',
                                        start_H=h_start,
                                        start_W=w_start)
    input_tensor = crop_tensor(images=input_tensor,
                               crop_size_tuple_or_scalar=(new_H_CC_ECC, new_W_CC_ECC),
                               crop_style='predetrmined',
                               start_H=h_start,
                               start_W=w_start)
    reference_tensor = crop_tensor(images=reference_tensor,
                                   crop_size_tuple_or_scalar=(new_H_CC_ECC, new_W_CC_ECC),
                                   crop_style='predetrmined',
                                   start_H=h_start,
                                   start_W=w_start)

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T // 2: T // 2 + 1], FPS=show_video_fps)


    # free up GPU memory
    del h_start_vec, h_end_vec, w_start_vec, w_end_vec
    torch.cuda.empty_cache()

    # III: Cross Correlation
    aligned_tensor_CC_ECC_CC, shifts_h_CC_ECC_CC, shifts_w_CC_ECC_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=aligned_tensor_CC_ECC,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    max_shift_H_CC_ECC_CC = shifts_h_CC_ECC_CC.abs().max()
    max_shift_W_CC_ECC_CC = shifts_w_CC_ECC_CC.abs().max()
    new_H_CC_ECC_CC = new_H_CC_ECC - (max_shift_H_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC_CC = new_W_CC_ECC - (max_shift_W_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC_ECC_CC = crop_torch_batch(aligned_tensor_CC_ECC_CC, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC_ECC_CC, new_W_CC_ECC_CC])

    # Free up GPU memory:
    del aligned_tensor_CC_ECC
    del shifts_h_CC_ECC_CC
    del shifts_w_CC_ECC_CC
    del _
    torch.cuda.empty_cache()

    aligned_tenser_final = aligned_tensor_CC_ECC_CC
    return input_tensor, reference_tensor, aligned_tenser_final, H_matrix_ECC, \
           num_iterations_list, final_delta_shift_list


def align_palantir_CC_Gimbaless_CC(input_tensor, reference_tensor, flag_plot=False):
    """Aligns given tensor to reference with the following methods:
           Circular Cross Correlation (CC)
           Gimbaless Rotation
           Circular Cross Correlation (CC)

    :param input_tensor:
    :param reference_tensor:
    :param flag_plot:
    :return: aligned input tensor to reference tensor
    """
    T, C, H, W = input_tensor.shape
    total_shift_h = 0
    total_shift_w = 0
    total_shift_rotation = 0

    # I: Cross Correlation
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                          reference_matrix=reference_tensor,
                                                                                          matrix_fft=None,
                                                                                          normalize_over_matrix=False,
                                                                                          warp_method='bicubic',
                                                                                          crop_warped_matrix=False)
    # crop tensors to the same size
    max_shift_h = shifts_h_CC.abs().max()
    max_shift_w = shifts_w_CC.abs().max()
    new_H = H - (max_shift_h.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_w.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC = crop_torch_batch(aligned_tensor_CC, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    # sum up shifts
    total_shift_h += shifts_h_CC
    total_shift_w += shifts_w_CC

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC - aligned_tensor_CC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    input_tensor = input_tensor.cpu()
    shifts_h_CC = shifts_h_CC.cpu()
    shifts_w_CC = shifts_w_CC.cpu()
    del _

    # II. Gimbalss Rotation:
    gimbaless_rotation_layer = Gimbaless_Rotation_Layer_Torch()
    _, _, shifts_rotations_CC_Gimbaless =\
        gimbaless_rotation_layer.forward(aligned_tensor_CC.unsqueeze(0), reference_tensor.unsqueeze(0))
    shifts_rotations_CC_Gimbaless = shifts_rotations_CC_Gimbaless.squeeze(0)

    # align tensor
    aligned_tensor_CC_Gimbaless = rotate_matrix(aligned_tensor_CC, shifts_rotations_CC_Gimbaless, warp_method='fft')

    # crop tensors to the same size:
    aligned_tensor_CC_Gimbaless = crop_torch_batch(aligned_tensor_CC_Gimbaless, (new_H - 3, new_W - 3))
    reference_tensor = crop_torch_batch(reference_tensor, (new_H - 3, new_W - 3))

    total_shift_rotation += shifts_rotations_CC_Gimbaless

    # free up GPU memory
    del shifts_rotations_CC_Gimbaless
    del aligned_tensor_CC
    del _

    # III: Cross Correlation
    aligned_tensor_CC_Gimbaless_CC, shifts_h_CC_Gimbaless_CC, shifts_w_CC_Gimbaless_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=aligned_tensor_CC_Gimbaless,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)

    # crop tensors to the same size
    # Crop both original and aligned to be the same size and correct for size disparities
    max_shift_h_CC_Gimbaless_CC = shifts_h_CC_Gimbaless_CC.abs().max()
    max_shift_w_CC_Gimbaless_CC = shifts_w_CC_Gimbaless_CC.abs().max()
    new_H_CC_Gimbaless_CC = new_H - (max_shift_h_CC_Gimbaless_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_Gimbaless_CC = new_W - (max_shift_w_CC_Gimbaless_CC.abs().ceil().int().cpu().numpy() * 2 + 5)

    aligned_tensor_CC_Gimbaless_CC = crop_torch_batch(aligned_tensor_CC_Gimbaless_CC, [new_H_CC_Gimbaless_CC, new_W_CC_Gimbaless_CC])
    input_tensor = crop_torch_batch(input_tensor, [new_H_CC_Gimbaless_CC, new_W_CC_Gimbaless_CC])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H_CC_Gimbaless_CC, new_W_CC_Gimbaless_CC])

    # sum up shifts
    total_shift_h += shifts_h_CC_Gimbaless_CC
    total_shift_w += shifts_w_CC_Gimbaless_CC

    # Free up GPU memory:
    del aligned_tensor_CC_Gimbaless
    del shifts_h_CC_Gimbaless_CC
    del shifts_w_CC_Gimbaless_CC
    del _

    aligned_tenser_final = aligned_tensor_CC_Gimbaless_CC
    return input_tensor, reference_tensor, aligned_tenser_final, total_shift_h, total_shift_w, total_shift_rotation
import kornia.geometry
import torch

from RapidBase.Anvil.Dudy import Align_ECC, get_homography_from_optical_flow
from RapidBase.Anvil.alignments import align_to_reference_frame_circular_cc
from RapidBase.Anvil.alignments_layers import minSAD_transforms_layer, Gimbaless_Rotation_Layer_Torch
from RapidBase.Anvil import shift_matrix_subpixel, rotate_matrix, affine_warp_matrix
from RapidBase.Utils.Array_Tensor_Manipulation import crop_torch_batch
from RapidBase.Utils.Imshow_and_Plots import imshow_torch_video

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


def align_palantir_CC_ECC_CC(input_tensor, reference_tensor, flag_plot=False):
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
    total_shift_h = 0
    total_shift_w = 0
    total_shift_rotation = 0

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
    del shifts_h_CC
    del shifts_w_CC
    del _
    torch.cuda.empty_cache()

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # II: Enhanced Correlation Coefficient
    aligned_tensor_CC_ECC, H_matrix, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = Align_ECC(aligned_tensor_CC,
                                                                                                      reference_tensor)
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # sum up shifts
    total_shift_h += shifts_h_CC_ECC
    total_shift_w += shifts_w_CC_ECC
    total_shift_rotation += shifts_rotation_CC_ECC

    # free up GPU memory
    del H_matrix
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
    new_H_CC_ECC_CC = new_H - (max_shift_H_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W_CC_ECC_CC = new_W - (max_shift_W_CC_ECC_CC.abs().ceil().int().cpu().numpy() * 2 + 5)

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
    return input_tensor, reference_tensor, aligned_tenser_final, total_shift_h, total_shift_w, total_shift_rotation


def get_affine_params_palantir_CC_ECC_CC(input_tensor, reference_tensor, flag_plot=False):
    """Aligns given tensor to reference with the following methods:
       Circular Cross Correlation (CC)
       Enhanced Correlation Coefficient (ECC)
       Circular Cross Correlation (CC)
    TODO optional: return homography parameters for warping later over entire video while stabilization only performed
     over a fixed cropped section

    :param input_tensor:
    :param reference_tensor:
    :param flag_plot:
    :return: affine parameters to align input tensor to reference tensor
    """
    T, C, H, W = input_tensor.shape
    total_shift_h = 0
    total_shift_w = 0
    total_shift_rotation = 0
    gpu_usage_log = [torch.cuda.memory_allocated(0)/1024/1024/1024]

    # I: Cross Correlation
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=input_tensor,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)
    gpu_usage_log += [torch.cuda.memory_allocated(0)/1024/1024/1024]

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

    # temp try: see summing up errors
    mid_warped_tensor = affine_warp_matrix(input_tensor, -total_shift_h, -total_shift_w, warp_method='bicubic')
    mid_warped_tensor = crop_torch_batch(mid_warped_tensor, (new_H, new_W))
    # test:
    imshow_torch_video(mid_warped_tensor - aligned_tensor_CC)
    # del
    del mid_warped_tensor

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC - aligned_tensor_CC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    input_tensor = input_tensor.cpu()
    shifts_h_CC = shifts_h_CC.cpu()
    shifts_w_CC = shifts_w_CC.cpu()
    del _

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # II: Enhanced Correlation Coefficient
    aligned_tensor_CC_ECC, H_matrix, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = Align_ECC(aligned_tensor_CC,
                                                                                                      reference_tensor)
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # add shifts to shifts sum
    total_shift_h = total_shift_h + shifts_h_CC_ECC
    total_shift_w = total_shift_w + shifts_w_CC_ECC
    total_shift_rotation = total_shift_rotation + shifts_rotation_CC_ECC

    # temp try: see summing up errors
    mid_warped_tensor = affine_warp_matrix(input_tensor, -total_shift_h, -total_shift_w, -total_shift_rotation,
                                           warp_method='bicubic')
    mid_warped_tensor = crop_torch_batch(mid_warped_tensor, (new_H, new_W))
    mid_warped_tensor = kornia.geometry.warp_affine(aligned_tensor_CC.to(torch.float64), H_matrix[:, :2],
                                                    mode='bicubic', dsize=(new_H, new_W)).float()
    # test:
    imshow_torch_video(mid_warped_tensor - aligned_tensor_CC_ECC)
    imshow_torch_video(mid_warped_tensor - mid_warped_tensor[T//2])
    imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T//2])
    # del
    del mid_warped_tensor

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    H_matrix = H_matrix.cpu()
    shifts_rotation_CC_ECC = shifts_rotation_CC_ECC.cpu()
    shifts_h_CC_ECC = shifts_h_CC_ECC.cpu()
    shifts_w_CC_ECC = shifts_w_CC_ECC.cpu()
    del _

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
    shifts_h_CC_ECC_CC = shifts_h_CC_ECC_CC.cpu()
    shifts_w_CC_ECC_CC = shifts_w_CC_ECC_CC.cpu()
    del _

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # experimental: try one warp over entire shift
    summed_warped_tensor = affine_warp_matrix(input_tensor, total_shift_h, total_shift_w,
                                              total_shift_rotation, warp_method='bicubic')

    aligned_tenser_final = aligned_tensor_CC_ECC_CC
    return aligned_tenser_final, total_shift_h, total_shift_w, total_shift_rotation


def get_affine_matrix_palantir_CC_ECC_CC(input_tensor, reference_tensor, flag_plot=False):
    """Aligns given tensor to reference with the following methods:
       Circular Cross Correlation (CC)
       Enhanced Correlation Coefficient (ECC)
       Circular Cross Correlation (CC)

    :param input_tensor:
    :param reference_tensor:
    :param flag_plot:
    :return: affine parameters to align input tensor to reference tensor
    """
    T, C, H, W = input_tensor.shape
    affine_matrix = torch.stack([torch.eye(3, 3)] * T)
    gpu_usage_log = [torch.cuda.memory_allocated(0)/1024/1024/1024]

    # I: Cross Correlation
    aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = \
        align_to_reference_frame_circular_cc(matrix=input_tensor,
                                             reference_matrix=reference_tensor,
                                             matrix_fft=None,
                                             normalize_over_matrix=False,
                                             warp_method='bicubic',
                                             crop_warped_matrix=False)
    gpu_usage_log += [torch.cuda.memory_allocated(0)/1024/1024/1024]

    # add shifts to shifts sum
    affine_matrix_cc = torch.stack([torch.eye(3, 3)] * T)
    affine_matrix_cc[:, 0, 2] = shifts_w_CC
    affine_matrix_cc[:, 1, 2] = shifts_h_CC
    affine_matrix = affine_matrix_cc @ affine_matrix

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
    del _

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # II: Enhanced Correlation Coefficient
    aligned_tensor_CC_ECC, H_matrix, shifts_rotation_CC_ECC, _, shifts_h_CC_ECC, shifts_w_CC_ECC = Align_ECC(aligned_tensor_CC,
                                                                                                      reference_tensor)
    aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:{gpu_idx}')

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    # add shifts to shifts sum
    affine_matrix = H_matrix.to(torch.float32).cpu() @ affine_matrix

    # show results
    if flag_plot:
        imshow_torch_video(aligned_tensor_CC_ECC, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_CC_ECC - aligned_tensor_CC_ECC[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    H_matrix = H_matrix.cpu()
    shifts_rotation_CC_ECC = shifts_rotation_CC_ECC.cpu()
    shifts_h_CC_ECC = shifts_h_CC_ECC.cpu()
    shifts_w_CC_ECC = shifts_w_CC_ECC.cpu()
    del _

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

    # add shifts to shifts sum
    affine_matrix_cc_ecc_cc = torch.stack([torch.eye(3, 3)] * T)
    affine_matrix_cc_ecc_cc[:, 0, 2] = shifts_w_CC_ECC_CC
    affine_matrix_cc_ecc_cc[:, 1, 2] = shifts_h_CC_ECC_CC
    affine_matrix = affine_matrix_cc_ecc_cc @ affine_matrix

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
    del _

    gpu_usage_log += [torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024]

    affine_matrix = affine_matrix[:, :2]
    aligned_tenser_final = aligned_tensor_CC_ECC_CC

    return aligned_tenser_final, affine_matrix


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
import kornia.geometry

from Palantir.Palantir_utils import pad_numpy_batch, read_bin_video_to_numpy, read_bin_images_batch
from RapidBase.Anvil.Dudy import *
from RapidBase.Anvil import affine_warp_matrix
from Palantir.palantir_optional_stabilizations import align_palantir_CC_ECC_CC, \
    get_affine_matrix_palantir_CC_ECC_CC, get_affine_params_palantir_CC_ECC_CC


# # ----- Phase I: Define and Initialize Global Parameters -----
# # paths
# experiment_idx = 1
# original_frames_folder = f'/home/dudy/Projects/data/Palantir/p{experiment_idx}'
# # original_frames_folder = f'/home/dudy/Projects/data/Palantir/Individual_images/{experiment_idx}'
# # original_frames_folder = '/home/dudy/Projects/data/pngsh/pngs/2'
# # final_aligned_frames_folder = '/home/dudy/Projects/data/pngsh/final_aligned_pngs/2'
# # original_frames_folder = '/home/dudy/Projects/data/Palantir/3/pngs'
# # final_aligned_frames_folder = '/home/dudy/Projects/data/Palantir/3/final_aligned_pngs'
# # original_frames_folder = '/home/elisheva/temp/pngsh/pngs'
# # final_aligned_frames_folder = '/home/elisheva/temp/pngsh/final_aligned_pngs'
#
# results_path = '/home/dudy/Projects/temp/temp_palantir_results'
# # results_path = '/home/elisheva/temp/palantir_results'
# experiment_name = f'new_new_videos_{experiment_idx}'
#
# # experiment_name = f'new_videos_{experiment_idx}_Gimbaless'
# save_path = os.path.join(results_path, experiment_name)
#
# BG_image_folder = os.path.join(results_path, 'bg_estimates')
# BG_file_name = f'palantir_BG_new_new_{experiment_idx}.pt'
#
# # result bin file path
# bin_name_aligned = os.path.join(save_path, 'aligned_video.Bin')
#
# # result avi's and graphs paths
# videos_path = os.path.join(save_path, 'videos')
# shift_logs_path = os.path.join(save_path, 'shift_logs')
# video_name_original = os.path.join(videos_path, 'original_video.avi')
# video_name_aligned = os.path.join(videos_path, 'aligned_video.avi')
# video_name_outliers = os.path.join(videos_path, 'outliers_video.avi')
# log_name_shifts_h = os.path.join(shift_logs_path, 'shifts_h.npy')
# log_name_shifts_w = os.path.join(shift_logs_path, 'shifts_w.npy')
# log_name_rotations = os.path.join(shift_logs_path, 'rotations.npy')
#
#
# # generate paths
# if not os.path.exists(BG_image_folder):
#     os.mkdir(BG_image_folder)
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# if not os.path.exists(videos_path):
#     os.mkdir(videos_path)
# if not os.path.exists(shift_logs_path):
#     os.mkdir(shift_logs_path)
#
# # crop sizes
# initial_crop_h, initial_crop_w = (500, 2200)
# final_crop_h, final_crop_w = (480, 2000)
# # initial_crop_h, initial_crop_w = (410, 8100)
# # final_crop_h, final_crop_w = (390, 8000)
#
# # result avis' objects
# video_object_original = cv2.VideoWriter(video_name_original, 0, 10, (final_crop_w, final_crop_h))
# video_object_aligned = cv2.VideoWriter(video_name_aligned, 0, 10, (final_crop_w, final_crop_h))
# video_object_outliers = cv2.VideoWriter(video_name_outliers, 0, 10, (final_crop_w, final_crop_h))
#
# # image filenames
# image_filenames = read_image_filenames_from_folder(path=original_frames_folder,
#                                                    number_of_images=np.inf,
#                                                    allowed_extentions=['.png', '.Bin'],
#                                                    flag_recursive=True,
#                                                    string_pattern_to_search='*')
#
# # background estimation parameters
# number_of_images_for_bg_estimation = 100
# image_index_to_start_from_bg_estimation = 100  # TODO check for wierd movements and set
#
# # plotting parameters
# show_video_fps = 10
#
# # general parameters
# gpu_idx = 0
# # number_of_frames_per_video = 600
# number_of_frames_per_batch = 20
# number_of_images_total = len(image_filenames)
# original_height = 540
# original_width = 8192
#
# # flags
# flag_plot = False
# flag_estimate_bg = False
# flag_video_from_bin_files = True


# ----- Phase II: Estimate Background -----
def estimate_bg(number_of_images=100, image_index_to_start_from=0):
    total_shift_h = 0
    total_shift_w = 0
    total_rotation = 0

    initial_crop_h = 500
    initial_crop_w = 500

    # --- Get video to extract background from ---
    if flag_video_from_bin_files:
        image_concat_array = read_bin_images_batch(image_filenames[image_index_to_start_from:
                                                                   image_index_to_start_from + number_of_images],
                                                   height=original_height, width=original_width)
    else:
        image_concat_array = read_images_from_filenames_list(image_filenames,
                                                             flag_return_numpy_or_list='numpy',
                                                             crop_size=np.inf,
                                                             max_number_of_images=number_of_images,
                                                             flag_how_to_concat='C',
                                                             crop_style='center',
                                                             flag_return_torch=True,
                                                             transform=None,
                                                             first_frame_index=image_index_to_start_from)
    torch.cuda.empty_cache()
    input_tensor = image_concat_array.unsqueeze(1).to(f'cuda:{gpu_idx}').float()
    input_tensor = crop_torch_batch(input_tensor, (initial_crop_h + 20, initial_crop_w + 20),
                                    crop_style='predetermined', start_H=2, start_W=2480)
    T, C, H, W = input_tensor.shape

    # --- Stabilize video ---
    # I: cross correlation
    # stabilize the tensor with cc
    shift_h, shift_w, _ = circular_cc_shifts(input_tensor, input_tensor[T // 2: T // 2 + 1])
    del _
    # del image_concat_array

    total_shift_h += shift_h
    total_shift_w += shift_w

    # warp the tensor
    max_shift_h = shift_h.max()
    max_shift_w = shift_w.max()
    torch.cuda.empty_cache()
    aligned_tensor_1 = shift_matrix_subpixel(input_tensor[:T//2], -shift_h[:T//2], -shift_w[:T//2], matrix_FFT=None, warp_method='fft')
    aligned_tensor_2 = shift_matrix_subpixel(input_tensor[T//2:], -shift_h[T//2:], -shift_w[T//2:], matrix_FFT=None, warp_method='fft')
    aligned_tensor = torch.cat([aligned_tensor_1, aligned_tensor_2]).to(f'cuda:{gpu_idx}')

    # crop the tensors
    aligned_tensor = crop_torch_batch(aligned_tensor, (initial_crop_h + 10, initial_crop_w + 10))

    input_tensor = crop_torch_batch(input_tensor, (initial_crop_h + 10, initial_crop_w + 10))

    # plot the tensors
    if flag_plot:
        concat_tensor = torch.cat([input_tensor, aligned_tensor], dim=-1)
        imshow_torch_video(input_tensor, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor, FPS=show_video_fps)
        imshow_torch_video(concat_tensor, FPS=show_video_fps)
        imshow_torch_video(input_tensor - input_tensor[T // 2: T // 2 + 1], FPS=show_video_fps)
        imshow_torch_video(aligned_tensor - aligned_tensor[T // 2: T // 2 + 1], FPS=show_video_fps)

    # free up GPU memory
    del input_tensor
    del aligned_tensor_1
    del aligned_tensor_2
    del shift_h
    del shift_w
    if flag_plot:
        del concat_tensor

    # II: minSAD
    # stabilize the tensor with minSAD
    T, C, H, W = aligned_tensor.shape
    minSAD = minSAD_transforms_layer(1, T - 1, H, W)
    shifts_y, shifts_x, rotational_shifts, scale_shifts, _ = minSAD(
        matrix=aligned_tensor.float(),
        reference_matrix=aligned_tensor[T // 2: T // 2 + 1].float(),
        shift_h_vec=[-2, -1, 0, 1, 2],
        shift_w_vec=[-2, -1, 0, 1, 2],
        rotation_vec=[-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4],
        scale_vec=[1],
        warp_method='bicubic')

    total_shift_h += shifts_y.to(total_shift_h.device)
    total_shift_w += shifts_x.to(total_shift_h.device)
    total_rotation += (rotational_shifts * torch.pi / 180).to(total_shift_h.device)

    del _
    del shifts_y
    del shifts_x
    del rotational_shifts

    # warp the tensor
    # split into two parts to avoid cuda out of memory issues
    # warped_tensor_1 = affine_warp_matrix(aligned_tensor[:T // 2],
    #                                      -shifts_y[:T // 2],
    #                                      -shifts_x[:T // 2],
    #                                      -rotational_shifts[:T // 2] * np.pi / 180,
    #                                      scale_shifts[:T // 2],
    #                                      warp_method='bicubic')
    # warped_tensor_2 = affine_warp_matrix(aligned_tensor[T // 2:],
    #                                      -shifts_y[T // 2:],
    #                                      -shifts_x[T // 2:],
    #                                      -rotational_shifts[T // 2:] * np.pi / 180,
    #                                      scale_shifts[T // 2:],
    #                                      warp_method='bicubic')
    # warped_tensor = torch.cat([warped_tensor_1, warped_tensor_2], dim=0)
    #
    # # free up GPU memory
    # del warped_tensor_1
    # del warped_tensor_2
    #
    # # crop the tensors
    # aligned_tensor = crop_torch_batch(aligned_tensor, (initial_crop_h, initial_crop_w))
    # warped_tensor = crop_torch_batch(warped_tensor, (initial_crop_h, initial_crop_w))
    # warped_tensor = warped_tensor.to(f'cuda:{gpu_idx}')
    # aligned_tensor = aligned_tensor.to(f'cuda:{gpu_idx}')

    # plot the tensors
    # if flag_plot:
    #     concat_tensor = torch.cat([aligned_tensor, warped_tensor], dim=-1)
    #     plot_torch(rotational_shifts)
    #     plt.show()
    #     imshow_torch_video(warped_tensor, FPS=show_video_fps)
    #     imshow_torch_video(concat_tensor, FPS=show_video_fps)
    #     imshow_torch_video(warped_tensor - warped_tensor[T // 2: T // 2 + 1], FPS=show_video_fps)

    # --- Calculate background ---
    # experimental - warp entire video
    H, W = image_concat_array.shape[-2:]
    torch.cuda.empty_cache()
    T1, T2, T3, T4, T5 = T//5, 2*T//5, 3*T//5, 4*T//5, T

    aligned_whole_tensor_1 = crop_torch_batch(
        affine_warp_matrix(image_concat_array[:T1].unsqueeze(1).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[:T1], -total_shift_w[:T1],
                           -total_rotation[:T1], warp_method='bicubic'), (H, W)).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_2 = crop_torch_batch(
        affine_warp_matrix(image_concat_array[T1: T2].unsqueeze(1).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T1: T2], -total_shift_w[T1: T2],
                           -total_rotation[T1: T2], warp_method='bicubic'), (H, W)).cpu()
    torch.cuda.empty_cache()
    # imshow_torch_video(aligned_whole_tensor_2)

    aligned_whole_tensor_3 = crop_torch_batch(
        affine_warp_matrix(image_concat_array[T2: T3].unsqueeze(1).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T2: T3], -total_shift_w[T2: T3],
                           -total_rotation[T2: T3], warp_method='bicubic'), (H, W)).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_4 = crop_torch_batch(
        affine_warp_matrix(image_concat_array[T3: T4].unsqueeze(1).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T3: T4], -total_shift_w[T3: T4],
                           -total_rotation[T3: T4], warp_method='bicubic'), (H, W)).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor_5 = crop_torch_batch(
        affine_warp_matrix(image_concat_array[T4:].unsqueeze(1).to(f'cuda:{gpu_idx}'),
                           -total_shift_h[T4:], -total_shift_w[T4:],
                           -total_rotation[T4:], warp_method='bicubic'), (H, W)).cpu()
    torch.cuda.empty_cache()

    aligned_whole_tensor = torch.cat([aligned_whole_tensor_1, aligned_whole_tensor_2, aligned_whole_tensor_3,
                                      aligned_whole_tensor_4, aligned_whole_tensor_5])


    # calculate median
    input_tensor_BG = aligned_whole_tensor.median(0)[0]
    # save result
    torch.save(input_tensor_BG, os.path.join(BG_image_folder, BG_file_name))

    # free up GPU memory
    # del aligned_tensor
    # del warped_tensor
    # if flag_plot:
    #     del concat_tensor


# ----- Phase III: Stabilize Video -----
def stabilize_tensor(input_video_tensor: Tensor,
                     reference_tensor: Tensor,
                     frames_per_batch: int = 20,
                     flag_plot: bool = False,
                     warp_mode: str = 'align'):
    """Stabilizes a given video tensor to reference tensor.
    Stabilization pipeline is:
        Circular Cross Correlation (CC)
        Enhanced Correlation Coefficient (ECC)
        Circular Cross Correlation (CC)
        Gimbaless translation
    Notice: the function converts the tensors to cuda

    :param input_video_tensor: video to stabilize, expected shape TCHW
    :param reference_tensor: reference frame, expected shape 1CHW
    :param frames_per_batch: batch size for parallel computation
    :return: stabilized tensor
    """
    # Prepare tensors
    #(*). crop input tensor to reference_tensor shape
    _, C_ref, H_ref, W_ref = reference_tensor.shape
    input_video_tensor = crop_torch_batch(input_video_tensor, (H_ref, W_ref))
    T, C, H, W = input_video_tensor.shape
    reference_tensor = reference_tensor.to(f'cuda:{gpu_idx}')

    # Stabilize each batch
    number_of_batches = int(ceil(T / frames_per_batch))
    aligned_tensor_final = torch.zeros(1, C, H, W).to(f'cuda:{gpu_idx}')
    for batch_index in np.arange(number_of_batches):
        print(f'BATCH {batch_index}')
        # get batch tensor
        start_index = batch_index * frames_per_batch
        input_tensor = input_video_tensor[start_index: start_index + frames_per_batch]
        input_tensor = input_tensor.to(f'cuda:{gpu_idx}')

        # stabilize tensor
        # option I: align tensor directly
        if warp_mode == 'align':
            aligned_tensor = align_palantir_CC_ECC_CC(input_tensor, reference_tensor)

        # option II: get affine parameters
        elif warp_mode == 'shifts':
            gt_aligned_tensor, shift_h, shift_w, shift_rotation = get_affine_params_palantir_CC_ECC_CC(input_tensor, reference_tensor)
            gt_h, gt_w = gt_aligned_tensor.shape[-2:]
            aligned_tensor = affine_warp_matrix(input_tensor, -shift_h, -shift_w, -shift_rotation, warp_method='bicubic')
            aligned_tensor = crop_torch_batch(aligned_tensor, (gt_h, gt_w))
            # test:
            imshow_torch_video(gt_aligned_tensor - aligned_tensor)
            imshow_torch_video(aligned_tensor)
            imshow_torch_video(aligned_tensor - aligned_tensor[frames_per_batch//2])

        # option III: get affine matrix
        elif warp_mode == 'affine_matrix':
            gt_aligned_tensor, affine_matrix = get_affine_matrix_palantir_CC_ECC_CC(input_tensor, reference_tensor)
            gt_h, gt_w = gt_aligned_tensor.shape[-2:]
            aligned_tensor = kornia.geometry.warp_affine(input_tensor, affine_matrix.cuda(), (initial_crop_h, initial_crop_w), mode='bicubic')
            aligned_tensor = crop_torch_batch(aligned_tensor, (gt_h, gt_w))
            # test:
            imshow_torch_video(gt_aligned_tensor - aligned_tensor)
            imshow_torch_video(aligned_tensor - aligned_tensor[T // 2])

        # crop tensors to the same size
        input_tensor = crop_torch_batch(input_tensor, (initial_crop_h, initial_crop_w))
        aligned_tensor = crop_torch_batch(aligned_tensor, (initial_crop_h, initial_crop_w))
        H_current, W_current = aligned_tensor.shape[-2:]
        aligned_tensor_final = crop_torch_batch(aligned_tensor_final, (H_current, W_current))

        # add to result tensor
        aligned_tensor_final = torch.cat([aligned_tensor_final, aligned_tensor], dim=0)
    # remove first empty frame
    aligned_tensor_final = aligned_tensor_final[1:]

    # crop input video
    H_aligned, W_aligned = aligned_tensor_final.shape[-2:]
    input_video_tensor = crop_torch_batch(input_video_tensor, (H_aligned, W_aligned))

    # Show results
    if flag_plot:
        concat_tensor = torch.cat([input_video_tensor, aligned_tensor_final.cpu()], dim=-1)
        imshow_torch_video(input_video_tensor, FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_final, FPS=show_video_fps, frame_stride=5)
        imshow_torch_video(concat_tensor, FPS=show_video_fps)
        imshow_torch_video(input_video_tensor - input_video_tensor[number_of_frames_per_batch//2:
                                                                   number_of_frames_per_batch//2+1], FPS=show_video_fps)
        imshow_torch_video(aligned_tensor_final - aligned_tensor_final[number_of_frames_per_batch//2:
                                                                       number_of_frames_per_batch//2+1], FPS=show_video_fps // 2)

    return input_video_tensor, aligned_tensor_final


# ----- Phase IV: Extract Outliers -----
def find_outliers(aligned_tensor: Tensor,
                  bg_estimation: Tensor,
                  outliers_threshold: int = 20,
                  estimate_outliers_threshold: bool = False,
                  apply_canny: bool = True,
                  canny_low_threshold: float = 0.05 * 255,
                  canny_high_threshold: float = 0.2 * 255,
                  dilate_canny: bool = True,
                  dilation_kernel: Tensor = torch.ones(3, 3)):
    # aligned tensor and bg estimation assumed to be of equal sizes
    # calculate outliers without canny edge detection
    diff_tensor = (aligned_tensor - bg_estimation).abs()

    # find outliers threshold
    if estimate_outliers_threshold:
        noise_std = torch.std(estimate_noise_in_image_EMVA(aligned_tensor))
        outliers_threshold = noise_std * 3

    outliers_tensor = (diff_tensor > outliers_threshold).float()

    # If needed, calculate canny
    if apply_canny:
        _, bg_canny_threshold = kornia.filters.canny(bg_estimation, canny_low_threshold, canny_high_threshold)
        if dilate_canny:
            bg_canny_threshold = kornia.morphology.dilation(bg_canny_threshold, dilation_kernel.to(bg_canny_threshold.device))
        outliers_tensor_after_canny = (outliers_tensor - bg_canny_threshold).clamp(0)

        return outliers_tensor_after_canny

    return outliers_tensor


if __name__ == '__main__':
    for i in range(1, 21):
        i=17
        print(f"----------------- Video {i}, Gimbaless -----------------")
        # ----- Phase I: Define and Initialize Global Parameters -----
        # paths
        experiment_idx = 1
        original_frames_folder = f'/home/dudy/Projects/data/Palantir/p{experiment_idx}'

        results_path = '/home/dudy/Projects/temp/temp_palantir_results'
        experiment_name = f'Gimbaless_new_new_videos/video_{experiment_idx}_temp'

        save_path = os.path.join(results_path, experiment_name)

        BG_image_folder = os.path.join(results_path, 'bg_estimates')
        BG_file_name = f'palantir_BG_new_{experiment_idx}_temp.pt'

        # result bin file path
        bin_name_aligned = os.path.join(save_path, 'aligned_video.Bin')
        bin_name_outliers = os.path.join(save_path, 'outliers_video.Bin')

        # result avi's and graphs paths
        videos_path = os.path.join(save_path, 'videos')
        shift_logs_path = os.path.join(save_path, 'shift_logs')
        video_name_original = os.path.join(videos_path, 'original_video.avi')
        video_name_aligned = os.path.join(videos_path, 'aligned_video.avi')
        video_name_outliers = os.path.join(videos_path, 'outliers_video.avi')
        log_name_shifts_h = os.path.join(shift_logs_path, 'shifts_h.npy')
        log_name_shifts_w = os.path.join(shift_logs_path, 'shifts_w.npy')
        log_name_rotations = os.path.join(shift_logs_path, 'rotations.npy')

        # generate paths
        if not os.path.exists(BG_image_folder):
            os.mkdir(BG_image_folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(videos_path):
            os.mkdir(videos_path)
        if not os.path.exists(shift_logs_path):
            os.mkdir(shift_logs_path)

        # crop sizes
        # initial_crop_h, initial_crop_w = (500, 2200)
        # final_crop_h, final_crop_w = (480, 2000)
        original_height = 540
        original_width = 8192
        alignment_crop_h, alignment_crop_w = (540, 8192)
        # alignment_crop_h, alignment_crop_w = (500, 1000)
        final_crop_h, final_crop_w = (500, 8150)

        # result avis' objects
        video_object_original = cv2.VideoWriter(video_name_original, 0, 10, (original_width, original_height))
        video_object_aligned = cv2.VideoWriter(video_name_aligned, 0, 10, (original_width, original_height))
        video_object_outliers = cv2.VideoWriter(video_name_outliers, 0, 10, (original_width, original_height))

        # image filenames
        image_filenames = read_image_filenames_from_folder(path=original_frames_folder,
                                                           number_of_images=np.inf,
                                                           allowed_extentions=['.png', '.Bin'],
                                                           flag_recursive=True,
                                                           string_pattern_to_search='*')

        # background estimation parameters
        number_of_images_for_bg_estimation = 20
        image_index_to_start_from_bg_estimation = 120  # TODO check for wierd movements and set

        # plotting parameters
        show_video_fps = 10

        # general parameters
        gpu_idx = 0
        # number_of_frames_per_video = 600
        number_of_frames_per_batch = 20
        number_of_images_total = len(image_filenames)

        # flags
        flag_plot = False
        flag_estimate_bg = True
        flag_video_from_bin_files = True

        # --- Do Background Estimation ---
        if flag_estimate_bg:
            estimate_bg(number_of_images_for_bg_estimation, image_index_to_start_from_bg_estimation)

        # --- Stabilize movie from file ---
        # Get background tensor
        bg_tensor = torch.load(os.path.join(BG_image_folder, BG_file_name)).unsqueeze(0)
        _, C, H, W = bg_tensor.shape

        # Generate stabilized video
        # video_bin_array = np.array(np.zeros((1, final_crop_h, final_crop_w, C)))
        frame_index_global = 0
        shift_h_over_t = Tensor([])
        shift_w_over_t = Tensor([])
        rotation_over_t = Tensor([])
        number_of_videos = int(ceil(number_of_images_total / number_of_frames_per_batch))
        with open(bin_name_aligned, 'wb') as f:
            # for video_index in range(number_of_videos):
            for video_index in range(7, 13):
                print(f'BATCH: {video_index}')
                # read batch of images
                start_video = video_index * number_of_frames_per_batch
                video_tensor = read_bin_images_batch(
                    filenames=image_filenames[start_video: start_video + number_of_frames_per_batch],
                    height=540, width=8192).float()
                video_tensor = video_tensor.unsqueeze(1)

                # pre-process data
                _, C_ref, H_ref, W_ref = bg_tensor.shape
                # video_tensor = crop_torch_batch(video_tensor, (H_ref, W_ref),
                #                                 crop_style='predetermined', start_H=12, start_W=2500)
                input_tensor = crop_torch_batch(video_tensor, (alignment_crop_h, alignment_crop_w))
                reference_tensor = crop_torch_batch(bg_tensor, (alignment_crop_h, alignment_crop_w))
                T, C, H, W = video_tensor.shape
                reference_tensor = reference_tensor.to(f'cuda:{gpu_idx}')
                input_tensor = input_tensor.to(f'cuda:{gpu_idx}')

                # empty cuda cache:
                torch.cuda.empty_cache()

                ### stabilize batch: ###
                # inp, ref, alg, shift_h, shift_w, rotation = \
                #     align_palantir_CC_Gimbaless_CC(input_tensor, reference_tensor)
                input_tensor, reference_tensor, aligned_tensor, shift_h, shift_w, rotation = \
                    align_palantir_CC_ECC_CC(input_tensor, reference_tensor)

                # imshow_torch_video(alg-crop_torch_batch(alg2, (alg.shape[-2], alg.shape[-1])))
                # imshow_torch_video(alg2-ref2)
                # imshow_torch_video(alg-ref)

                # delete data for memory disposal
                del input_tensor
                # del reference_tensor
                torch.cuda.empty_cache()

                # process results
                # video_tensor = crop_torch_batch(video_tensor, (initial_crop_h, initial_crop_w))
                # aligned_tensor = crop_torch_batch(aligned_tensor, (initial_crop_h, initial_crop_w))
                # video_tensor = video_tensor.cpu()
                # T, C, H, W = aligned_tensor.shape

                # TODO validate
                # max_shift_rotation = rotation.abs().max()
                # max_shift_h = shift_h.abs().max()
                # # max_shift_h = max(shift_h.abs().max(), torch.tan(max_shift_rotation) * original_width / 2)
                # max_shift_w = shift_w.abs().max()
                # # crop_h, crop_w = ((alignment_crop_h - 2 * max_shift_h).to(torch.int), (alignment_crop_w - 2 * max_shift_w).to(torch.int))
                # crop_h, crop_w = ((original_height - 2 * max_shift_h).to(torch.int), (original_width - 2 * max_shift_w).to(torch.int))
                #
                # aligned_tensor = shift_matrix_subpixel(video_tensor, -shift_h, -shift_w)
                # aligned_tensor = crop_torch_batch(aligned_tensor, (original_height, original_width))
                #
                # aligned_tensor = crop_torch_batch(aligned_tensor, (crop_h, crop_w))
                # reference_tensor = crop_torch_batch(bg_tensor, (crop_h, crop_w))
                # # reference_tensor, _, _, _ = align_to_reference_frame_circular_cc(reference_tensor, aligned_tensor.median())
                #
                # aligned_tensor = rotate_matrix(aligned_tensor, rotation, warp_method='fft')
                # aligned_tensor = crop_torch_batch(aligned_tensor, (crop_h - 70, crop_w - 20))
                #
                # # correct a wierd misalignment between the bg and the aligned tensor
                # reference_tensor = crop_torch_batch(bg_tensor, (crop_h - 70, crop_w - 20))
                # aligned_bg, shifts_h_bg, shifts_w_bg, _ = align_to_reference_frame_circular_cc(
                #     matrix=reference_tensor,
                #     reference_matrix=aligned_tensor[0],
                #     matrix_fft=None,
                #     normalize_over_matrix=False,
                #     warp_method='bicubic',
                #     crop_warped_matrix=False)
                # # aligned_bg = shift_matrix_subpixel(reference_tensor, -shifts_h_bg, -shifts_w_bg)
                # aligned_bg = shift_matrix_integer_pixels(reference_tensor, -torch.round(shifts_h_bg), -torch.round(shifts_w_bg))
                #
                # aligned_tensor = crop_torch_batch(aligned_tensor, (crop_h - 100, crop_w - 40))
                # reference_tensor = crop_torch_batch(aligned_bg, (crop_h - 100, crop_w - 40))
                # # reference_tensor, _, _, _ = align_to_reference_frame_circular_cc(reference_tensor,
                # #                                                                  aligned_tensor.median())
                #
                # imshow_torch_video(aligned_tensor - reference_tensor)


                # aligned_tensor = crop_torch_batch(affine_warp_matrix(video_tensor, -shift_h, -shift_w, -rotation,
                #                                   warp_method='bicubic'), (original_height, original_width))
                # aligned_tensor = crop_torch_batch(affine_warp_matrix(video_tensor, -shift_h, -shift_w, -rotation,
                #                                                      warp_method='bicubic'), (crop_h, crop_w))
                # reference_tensor = crop_torch_batch(bg_tensor, (crop_h, crop_w))

                # empty cuda cache for memory management:
                torch.cuda.empty_cache()

                # find outliers: #TODO: add dudy's script functionality of adaptive threshold from EMVA1288
                outliers_tensor = find_outliers(aligned_tensor, reference_tensor)

                # transfer aligned tensor to cpu for memory management:
                aligned_tensor = aligned_tensor.cpu()
                torch.cuda.empty_cache()

                # add shifts for graphs
                shift_h_over_t = torch.cat([shift_h_over_t, shift_h.cpu()])
                shift_w_over_t = torch.cat([shift_w_over_t, shift_w.cpu()])
                rotation_over_t = torch.cat([rotation_over_t, rotation.cpu()])

                # show results
                if flag_plot:
                    concat_tensor = torch.cat([video_tensor, aligned_tensor], dim=-1)
                    imshow_torch_video(concat_tensor, FPS=show_video_fps)
                    imshow_torch_video(aligned_tensor, FPS=show_video_fps)
                    imshow_torch_video(aligned_tensor - aligned_tensor[T // 2: T // 2 + 1], FPS=show_video_fps)

                # prepare frames for video - make numpy arrays of exact size (final_crop_h, final_crop_w) and of 3 channels
                aligned_tensor = torch_to_numpy(aligned_tensor)
                aligned_tensor = crop_numpy_batch(aligned_tensor, (final_crop_h, final_crop_w), crop_style='center')
                aligned_tensor = pad_numpy_batch(aligned_tensor, (final_crop_h, final_crop_w), pad_style='center')

                # write aligned video to .bin file
                f.write(aligned_tensor[..., :1].tobytes())

                # turn aligned tensor to be video-ready for save/dispaly:
                aligned_tensor = numpy_array_to_video_ready(aligned_tensor)

                # crop tensors to required final size:
                #(1). current batch:
                video_tensor = torch_to_numpy(video_tensor)
                video_tensor = crop_numpy_batch(video_tensor, (final_crop_h, final_crop_w), crop_style='center')
                video_tensor = pad_numpy_batch(video_tensor, (final_crop_h, final_crop_w), pad_style='center')
                video_tensor = numpy_array_to_video_ready(video_tensor)
                #(2). outlier tensor
                outliers_tensor = torch_to_numpy(outliers_tensor)
                outliers_tensor = crop_numpy_batch(outliers_tensor, (final_crop_h, final_crop_w), crop_style='center')
                outliers_tensor = pad_numpy_batch(outliers_tensor, (final_crop_h, final_crop_w), pad_style='center')
                outliers_tensor = numpy_array_to_video_ready(outliers_tensor)

                # write video to avi file
                for frame_index in np.arange(T):
                    video_object_original.write(video_tensor[frame_index])
                    video_object_aligned.write(aligned_tensor[frame_index])
                    video_object_outliers.write(outliers_tensor[frame_index])

        # save shift results to disk:
        np.save(log_name_shifts_h, np.array(shift_h_over_t))
        np.save(log_name_shifts_w, np.array(shift_w_over_t))
        np.save(log_name_rotations, np.array(rotation_over_t))
        # imshow_torch_video(numpy_to_torch(video[: 700]), frame_stride=5, FPS=10)

        ### Load Entire Stabilized Batch: ###
        stabilized_entire_video = read_bin_video_to_numpy(
            '/home/dudy/Projects/temp/temp_palantir_results/Gimbaless_new_new_videos/video_1_temp/aligned_video.Bin',
            H=500, W=8150, C=1)
        stabilized_entire_video = numpy_to_torch(stabilized_entire_video)

        reference_tensor = torch_to_numpy(reference_tensor)
        reference_tensor = crop_numpy_batch(reference_tensor, (final_crop_h, final_crop_w), crop_style='center')
        reference_tensor = pad_numpy_batch(reference_tensor, (final_crop_h, final_crop_w), pad_style='center')
        reference_tensor = numpy_to_torch(reference_tensor)

        ### Present results for debugging: ###
        #(1). show BG:
        imshow_torch(reference_tensor)
        #(2). center frame:
        T,C,H,W = stabilized_entire_video.shape
        imshow_torch(stabilized_entire_video[T//2])
        #(3). stabilized - BG:
        imshow_torch_video(stabilized_entire_video - reference_tensor, FPS=10, frame_stride=1)
        imshow_torch(stabilized_entire_video - reference_tensor)
        #(4). stabilized - center_stabilized
        imshow_torch_video(stabilized_entire_video - stabilized_entire_video[T//2:T//2+1], FPS=10, frame_stride=1)
        imshow_torch(stabilized_entire_video - stabilized_entire_video[T//2:T//2+1])
        #(5). outliers from both options:
        outliers_tensor_from_BG = find_outliers(stabilized_entire_video, reference_tensor)
        outliers_tensor_from_center = find_outliers(stabilized_entire_video, stabilized_entire_video[T//2:T//2+1])
        imshow_torch_video(outliers_tensor_from_BG, FPS=10, frame_stride=5)
        imshow_torch_video(outliers_tensor_from_center, FPS=10, frame_stride=5)

        video_torch_array_to_video(outliers_tensor_from_BG, video_name_outliers, FPS=25)
        save_image_torch('/home/dudy/Projects/temp/temp_palantir_results/Gimbaless_new_new_videos/video_1_temp',
                         'bg_estimation.jpg',
                         reference_tensor)

# code archive
#     for i in range(1, 21):
#         print(f"----------------- Video {i}, ECC -----------------")
#         # ----- Phase I: Define and Initialize Global Parameters -----
#         # paths
#         experiment_idx = 1
#         original_frames_folder = f'/home/dudy/Projects/data/Palantir/p{experiment_idx}'
#         # original_frames_folder = f'/home/dudy/Projects/data/Palantir/Individual_images/{experiment_idx}'
#         # original_frames_folder = '/home/dudy/Projects/data/pngsh/pngs/2'
#         # final_aligned_frames_folder = '/home/dudy/Projects/data/pngsh/final_aligned_pngs/2'
#         # original_frames_folder = '/home/dudy/Projects/data/Palantir/3/pngs'
#         # final_aligned_frames_folder = '/home/dudy/Projects/data/Palantir/3/final_aligned_pngs'
#         # original_frames_folder = '/home/elisheva/temp/pngsh/pngs'
#         # final_aligned_frames_folder = '/home/elisheva/temp/pngsh/final_aligned_pngs'
#
#         results_path = '/home/dudy/Projects/temp/temp_palantir_results'
#         # results_path = '/home/elisheva/temp/palantir_results'
#         experiment_name = f'ECC_new_new_videos/new_videos_{experiment_idx}'
#
#         # experiment_name = f'new_videos_{experiment_idx}_Gimbaless'
#         save_path = os.path.join(results_path, experiment_name)
#
#         BG_image_folder = os.path.join(results_path, 'bg_estimates')
#         BG_file_name = f'palantir_BG_new_{experiment_idx}.pt'
#
#         # result bin file path
#         bin_name_aligned = os.path.join(save_path, 'aligned_video.Bin')
#
#         # result avi's and graphs paths
#         videos_path = os.path.join(save_path, 'videos')
#         shift_logs_path = os.path.join(save_path, 'shift_logs')
#         video_name_original = os.path.join(videos_path, 'original_video.avi')
#         video_name_aligned = os.path.join(videos_path, 'aligned_video.avi')
#         video_name_outliers = os.path.join(videos_path, 'outliers_video.avi')
#         log_name_shifts_h = os.path.join(shift_logs_path, 'shifts_h.npy')
#         log_name_shifts_w = os.path.join(shift_logs_path, 'shifts_w.npy')
#         log_name_rotations = os.path.join(shift_logs_path, 'rotations.npy')
#
#         # generate paths
#         if not os.path.exists(BG_image_folder):
#             os.mkdir(BG_image_folder)
#         if not os.path.exists(save_path):
#             os.mkdir(save_path)
#         if not os.path.exists(videos_path):
#             os.mkdir(videos_path)
#         if not os.path.exists(shift_logs_path):
#             os.mkdir(shift_logs_path)
#
#         # crop sizes
#         initial_crop_h, initial_crop_w = (500, 2200)
#         final_crop_h, final_crop_w = (480, 2000)
#         # initial_crop_h, initial_crop_w = (410, 8100)
#         # final_crop_h, final_crop_w = (390, 8000)
#
#         # result avis' objects
#         video_object_original = cv2.VideoWriter(video_name_original, 0, 10, (final_crop_w, final_crop_h))
#         video_object_aligned = cv2.VideoWriter(video_name_aligned, 0, 10, (final_crop_w, final_crop_h))
#         video_object_outliers = cv2.VideoWriter(video_name_outliers, 0, 10, (final_crop_w, final_crop_h))
#
#         # image filenames
#         image_filenames = read_image_filenames_from_folder(path=original_frames_folder,
#                                                            number_of_images=np.inf,
#                                                            allowed_extentions=['.png', '.Bin'],
#                                                            flag_recursive=True,
#                                                            string_pattern_to_search='*')
#
#         # background estimation parameters
#         number_of_images_for_bg_estimation = 100
#         image_index_to_start_from_bg_estimation = 150  # TODO check for wierd movements and set
#
#         # plotting parameters
#         show_video_fps = 10
#
#         # general parameters
#         gpu_idx = 0
#         # number_of_frames_per_video = 600
#         number_of_frames_per_batch = 20
#         number_of_images_total = len(image_filenames)
#         original_height = 540
#         original_width = 8192
#
#         # flags
#         flag_plot = False
#         flag_estimate_bg = True
#         flag_video_from_bin_files = True
#
#         # --- Do Background Estimation ---
#         if flag_estimate_bg:
#             estimate_bg(number_of_images_for_bg_estimation, image_index_to_start_from_bg_estimation)
#
#         # --- Stabilize movie from file ---
#         # Get background tensor
#         bg_tensor = torch.load(os.path.join(BG_image_folder, BG_file_name)).unsqueeze(0).to(f'cuda:{gpu_idx}')
#         _, C, H, W = bg_tensor.shape
#
#         # Generate stabilized video
#         # video_bin_array = np.array(np.zeros((1, final_crop_h, final_crop_w, C)))
#         frame_index_global = 0
#         shift_h_over_t = Tensor([])
#         shift_w_over_t = Tensor([])
#         rotation_over_t = Tensor([])
#         number_of_videos = int(ceil(number_of_images_total / number_of_frames_per_batch))
#         with open(bin_name_aligned, 'wb') as f:
#             for video_index in range(number_of_videos):
#                 print(f'BATCH: {video_index}')
#                 # read batch of images
#                 start_video = video_index * number_of_frames_per_batch
#                 video_tensor = read_bin_images_batch(filenames=image_filenames[start_video: start_video + number_of_frames_per_batch],
#                                                      height=540, width=8192).float()
#                 video_tensor = video_tensor.unsqueeze(1)
#
#                 # pre-process data
#                 _, C_ref, H_ref, W_ref = bg_tensor.shape
#                 video_tensor = crop_torch_batch(video_tensor, (H_ref, W_ref),
#                                                       crop_style='predetermined', start_H=12, start_W=2500)
#                 T, C, H, W = video_tensor.shape
#                 reference_tensor = bg_tensor.to(f'cuda:{gpu_idx}')
#                 video_tensor = video_tensor.to(f'cuda:{gpu_idx}')
#
#                 # stabilize batch
#                 video_tensor, reference_tensor, aligned_tensor, shift_h, shift_w, rotation =\
#                     align_palantir_CC_ECC_CC(video_tensor, bg_tensor)
#
#                 # process results
#                 video_tensor = crop_torch_batch(video_tensor, (initial_crop_h, initial_crop_w))
#                 aligned_tensor = crop_torch_batch(aligned_tensor, (initial_crop_h, initial_crop_w))
#                 video_tensor = video_tensor.cpu()
#                 T, C, H, W = aligned_tensor.shape
#                 torch.cuda.empty_cache()
#
#                 # find outliers
#                 outliers_tensor = find_outliers(aligned_tensor, reference_tensor)
#
#                 aligned_tensor = aligned_tensor.cpu()
#
#                 # add shifts for graphs
#                 shift_h_over_t = torch.cat([shift_h_over_t, shift_h.cpu()])
#                 shift_w_over_t = torch.cat([shift_w_over_t, shift_w.cpu()])
#                 rotation_over_t = torch.cat([rotation_over_t, rotation.cpu()])
#
#                 # show results
#                 if flag_plot:
#                     concat_tensor = torch.cat([video_tensor, aligned_tensor], dim=-1)
#                     imshow_torch_video(concat_tensor, FPS=show_video_fps)
#                     imshow_torch_video(aligned_tensor, FPS=show_video_fps)
#                     imshow_torch_video(aligned_tensor - aligned_tensor[T // 2: T // 2 + 1], FPS=show_video_fps)
#
#                 # prepare frames for video - make numpy arrays of exact size (final_crop_h, final_crop_w) and of 3 channels
#                 aligned_tensor = torch_to_numpy(aligned_tensor)
#                 aligned_tensor = crop_numpy_batch(aligned_tensor, (final_crop_h, final_crop_w), crop_style='center')
#                 aligned_tensor = pad_numpy_batch(aligned_tensor, (final_crop_h, final_crop_w), pad_style='center')
#
#                 # write aligned video to bin file
#                 f.write(aligned_tensor[..., :1].tobytes())
#
#                 aligned_tensor = numpy_array_to_video_ready(aligned_tensor)
#
#                 video_tensor = torch_to_numpy(video_tensor)
#                 video_tensor = crop_numpy_batch(video_tensor, (final_crop_h, final_crop_w), crop_style='center')
#                 video_tensor = pad_numpy_batch(video_tensor, (final_crop_h, final_crop_w), pad_style='center')
#                 video_tensor = numpy_array_to_video_ready(video_tensor)
#
#                 outliers_tensor = torch_to_numpy(outliers_tensor)
#                 outliers_tensor = crop_numpy_batch(outliers_tensor, (final_crop_h, final_crop_w), crop_style='center')
#                 outliers_tensor = pad_numpy_batch(outliers_tensor, (final_crop_h, final_crop_w), pad_style='center')
#                 outliers_tensor = numpy_array_to_video_ready(outliers_tensor)
#
#                 # write video to avi file
#                 for frame_index in np.arange(T):
#                     video_object_original.write(video_tensor[frame_index])
#                     video_object_aligned.write(aligned_tensor[frame_index])
#                     video_object_outliers.write(outliers_tensor[frame_index])
#
#         # save video array
#         # with open(bin_name_aligned, 'wb') as f:
#         #     f.write(video_bin_array[1:].tobytes())
#         # video = read_bin_video_to_numpy(bin_name_aligned, final_crop_h, final_crop_w, 1)
#         #
#         np.save(log_name_shifts_h, np.array(shift_h_over_t))
#         np.save(log_name_shifts_w, np.array(shift_w_over_t))
#         np.save(log_name_rotations, np.array(rotation_over_t))
#
#         # imshow_torch_video(numpy_to_torch(video[: 700]), frame_stride=5, FPS=10)
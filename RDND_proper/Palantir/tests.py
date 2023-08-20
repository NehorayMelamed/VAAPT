"""
test time stride:
generate graphs for both full and interpolated cases - parameter wise.
Also generate videos aligned with the different time strides.
"""
from Palantir.Palantir_utils import crop_size_after_homography, read_bin_images_batch
from RapidBase.Utils.Classical_DSP.ECC import ECC_torch_batch_time_stride
from RapidBase.import_all import *
from alignments import align_to_reference_frame_circular_cc


def align_and_save_with_time_stride(input_tensor, reference_tensor, time_stride, parent_dir, final_h=490, final_w=8142):
    save_dir = os.path.join(parent_dir, f"{time_stride:0>3}")
    h_matrices_save_name = os.path.join(save_dir, 'h_matrices.npy')
    num_iterations_save_name = os.path.join(save_dir, 'num_iterations.npy')
    delta_shifts_save_name = os.path.join(save_dir, 'delta_shifts.npy')

    video_format = cv2.VideoWriter_fourcc(*'MP42')
    save_fps = 25
    input_frames_video_name = os.path.join(save_dir, 'InputFrames.avi')
    input_frames_video_writer = cv2.VideoWriter(input_frames_video_name, video_format, save_fps, (final_h, final_w))
    aligned_frames_video_name = os.path.join(save_dir, 'AlignedFrames.avi')
    aligned_frames_video_writer = cv2.VideoWriter(aligned_frames_video_name, video_format, save_fps, (final_h, final_w))


    T, C, H, W = input_tensor.shape
    batch_size = 20
    last_h_matrix = torch.eye(3, 3)

    for batch_index in range(0, int(ceil(T / batch_size))):
        current_tensor = input_tensor[batch_index * batch_size: batch_index * batch_size + batch_size]
        current_reference_tensor = reference_tensor

        # first do CC
        aligned_tensor_CC, shifts_h_CC, shifts_w_CC, _ = \
            align_to_reference_frame_circular_cc(matrix=current_tensor,
                                                 reference_matrix=current_reference_tensor,
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
        current_reference_tensor = crop_torch_batch(current_reference_tensor, [new_H_CC, new_W_CC])

        # free up GPU memory
        del shifts_h_CC
        del shifts_w_CC
        del _
        torch.cuda.empty_cache()

        # Then test ECC
        # save the results in a way that enables recovery - save the h matrices in batches?
        # II. Enhanced Cross Correlation
        H_matrix_ECC, aligned_tensor_CC_ECC, num_iterations_list, final_delta_shift_list = \
            ECC_torch_batch_time_stride(aligned_tensor_CC, current_reference_tensor,
                                        number_of_levels=1,
                                        number_of_iterations_per_level=50,
                                        transform_string='homography',
                                        delta_p_init=last_h_matrix,
                                        flag_update_delta_p_init=True,
                                        time_stride=time_stride)

        aligned_tensor_CC_ECC = aligned_tensor_CC_ECC.float().to(f'cuda:0')
        last_h_matrix = H_matrix_ECC[-1]

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
        current_reference_tensor = crop_tensor(images=current_reference_tensor,
                                               crop_size_tuple_or_scalar=(new_H_CC_ECC, new_W_CC_ECC),
                                               crop_style='predetrmined',
                                               start_H=h_start,
                                               start_W=w_start)

        # make of final size
        aligned_tensor_CC_ECC = crop_tensor(images=aligned_tensor_CC_ECC,
                                            crop_size_tuple_or_scalar=(final_h, final_w))

        # save results
        # save videos:
        for i in range(batch_size):
            input_frame = BW2RGB(input_tensor[i])
            input_frames_video_writer.write(input_frame)
            aligned_frame = BW2RGB(aligned_tensor_CC_ECC[i])
            aligned_frames_video_writer.write(aligned_frame)

        # save homography matrices, iterations and shifts:
        f_h_matrices = open(h_matrices_save_name, 'a')
        np.save(f_h_matrices, H_matrix_ECC)
        f_h_matrices.close()

        f_num_iterations = open(num_iterations_save_name, 'a')
        np.save(f_num_iterations, array(num_iterations_list))
        f_num_iterations.close()

        f_delta_shifts = open(delta_shifts_save_name, 'a')
        np.save(f_delta_shifts, array(final_delta_shift_list))
        f_delta_shifts.close()
    input_frames_video_writer.clo


def compare_time_strides(stride_1, stride_2, source_dir):
    source_dir_1 = os.path.join(source_dir, f"{stride_1:0>3}")
    source_dir_2 = os.path.join(source_dir, f"{stride_2:0>3}")


def main():
    # read frames
    data_dir = '/home/dudy/Projects/data/Palantir/p1'
    filenames = [os.path.join(data_dir, fn) for fn in sorted(os.listdir(data_dir))[:500]]
    original_height = 540
    original_width = 8192
    input_tensor = read_bin_images_batch(filenames, original_height, original_width).unsqueeze(1).float().cuda()

    # read bg
    bg_path = '/home/dudy/Projects/temp/temp_palantir_results/whole_video_multi_bg_offline/p1/backgrounds/bg_1.pt'
    reference_tensor = torch.load(bg_path).cuda()

    # crop frames to bg size
    Cbg, Hbg, Wbg = reference_tensor.shape
    input_tensor = crop_tensor(input_tensor, (Hbg, Wbg))

    # define experiment's params
    parent_dir = '/home/dudy/Projects/temp/temp_palantir_results/time_strides'
    time_stride = 1
    align_and_save_with_time_stride(input_tensor, reference_tensor, time_stride, parent_dir, final_h=490, final_w=8142)


if __name__ == '__main__':
    main()

import shutil

from RapidBase.Anvil._internal_utils.torch_utils import faster_fft
from RapidBase.Anvil._transforms.shift_matrix_subpixel import calculate_meshgrids
from RapidBase.Anvil.alignments import align_to_reference_frame_circular_cc
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch_video
from RapidBase.Utils.Path_and_Reading_utils import read_image_torch, save_image_torch
from RapidBase.import_all import *


def _shift_matrix_subpixel_fft_batch_with_channels(matrix: torch.Tensor, shift_H, shift_W, matrix_fft=None,
                                                   warp_method: str = 'fft') -> torch.Tensor:
    """
    matrix of t, c, h, w, shifts of size t

    :param matrix: 5D matrix
    :param shift_H: either singleton or of length T
    :param shift_W: either singleton or of length T
    :param matrix_fft: fft of matrix, if precalculated
    :return: subpixel shifted matrix
    """
    t, c, h, w = matrix.shape
    if isinstance(shift_W, float):
        shift_W = torch.tensor(shift_W).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(t, 1, 1).to(matrix.device)
    if isinstance(shift_H, float):
        shift_H = torch.tensor(shift_H).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(t, 1, 1).to(matrix.device)
    if shift_W.dim() == 1:
        shift_W = shift_W.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(matrix.device)
    if shift_H.dim() == 1:
        shift_H = shift_H.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(matrix.device)

    # print(shift_W.shape)
    # print(shift_H.shape)

    ky, kx = calculate_meshgrids(matrix)
    # shift_W, shift_H = expand_shifts(matrix, shift_H, shift_W)
    ### Displace input image: ###
    displacement_matrix = torch.exp(-(
                1j * 2 * torch.pi * ky.repeat(t, c, 1, 1).to(matrix.device) * shift_H + 1j * 2 * torch.pi * kx.repeat(t,
                                                                                                                      c,
                                                                                                                      1,
                                                                                                                      1).to(
            matrix.device) * shift_W)).to(matrix.device)
    fft_image = faster_fft(matrix, matrix_fft, dim=[-1, -2])
    fft_image_displaced = fft_image * displacement_matrix
    original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()

    # imshow_torch(original_image_displaced[0])
    # imshow_torch(original_image_displaced[1])
    # imshow_torch(original_image_displaced[2])
    # imshow_torch(original_image_displaced[3])
    # imshow_torch(original_image_displaced[4])

    return original_image_displaced


def extract_frames_from_video(video_path, output_dir):
    """
    Extract frames from a video and save them to a specified directory.

    Args:
    - video_path (str): Path to the video file.
    - output_dir (str): Directory to save the extracted frames.

    Returns:
    - None
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_filename = os.path.join(output_dir, f"frame_{count:04d}.png")
        cv2.imwrite(frame_filename, image)
        success, image = vidcap.read()
        count += 1


import os


def prepare_input_source(input_path):
    """
    Prepare the input source for processing.

    Args:
    - input_path (str): Path to the video file or folder.

    Returns:
    - str: Path to the directory containing the frames.
    """

    # If it's a video, extract frames to a temp directory
    if is_video_file(input_path):
        temp_dir = "temp_frames_dir"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        extract_frames_from_video(input_path, temp_dir)
        return temp_dir

    # If it's a directory, return the input path as is
    else:
        return input_path


def get_avg_reference_given_shifts(video, shifts):
    sx, sy = shifts
    warped = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy.to(video.device), -sx.to(video.device))
    return warped.mean(0)


def clean_1_frame(estimator_video, averaged_video, index, max_window=None):
    """
    warp all video towrds 1 frame and average the frame.
    Args:
        estimator_video:
        averaged_video:
        index:

    Returns:

    """
    if max_window is None:
        _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video.unsqueeze(0),
                                                            estimator_video[index:index + 1].unsqueeze(0).repeat(1,
                                                                                                                 estimator_video.shape[
                                                                                                                     0],
                                                                                                                 1, 1,
                                                                                                                 1))
        avg_ref = get_avg_reference_given_shifts(averaged_video, [sx, sy])
        return avg_ref, sx, sy
    if index + max_window < estimator_video.shape[0]:
        _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video[index:index + max_window].unsqueeze(0),
                                                            estimator_video[index:index + 1].unsqueeze(0).repeat(1,
                                                                                                                 max_window,
                                                                                                                 1, 1,
                                                                                                                 1))
        avg_ref = get_avg_reference_given_shifts(averaged_video[index:index + max_window], [sx, sy])
        return avg_ref, sx, sy
    else:
        _, sy, sx, _ = align_to_reference_frame_circular_cc(estimator_video[index - max_window:index].unsqueeze(0),
                                                            estimator_video[index:index + 1].unsqueeze(0).repeat(1,
                                                                                                                 max_window,
                                                                                                                 1, 1,
                                                                                                                 1))
        avg_ref = get_avg_reference_given_shifts(averaged_video[index - max_window:index], [sx, sy])
        return avg_ref, sx, sy


def shift_2_center(video):
    """
    warp all video towrds center frame
    """
    middle_frame = (video.shape[0] // 2 - 1)
    _, sy, sx, _ = align_to_reference_frame_circular_cc(video.unsqueeze(0),
                                                        video[middle_frame:middle_frame + 1].unsqueeze(0).repeat(1,
                                                                                                                 video.shape[
                                                                                                                     0],
                                                                                                                 1, 1,
                                                                                                                 1))
    video = _shift_matrix_subpixel_fft_batch_with_channels(video, -sy, -sx)
    return video


def stabilize_ccc_video_no_avg(folder_path, max_frames=None, video_path_to_save="output_video_ecc_no_avg.mp4"):
    """

    Args:
        folder_path: path to video frames(png)
        max_window: window size to average upon, larger will be slower
        max_frames: stabilize first max_frames frames
        loop: iteration count of averaging
    Returns:

    """
    #### generate data
    # img = read_image_torch(default_image_filename_to_load1) # get single channel of demo image
    file_names = sorted(glob(os.path.join(folder_path, "*.png")))
    video = torch.cat([read_image_torch(name) for name in file_names])[:max_frames]
    averaged_video = shift_2_center(video)
    # save_video_torch(video/255, "/raid/yoav/temp_garbage/ecc/original_no_avg")
    # save_video_torch(averaged_video/255, "/raid/yoav/temp_garbage/ecc/averaged_no_avg")
    video_torch_array_to_video(averaged_video / 255, video_name=video_path_to_save)
    return averaged_video


def stabilize_ccc_video(folder_path, max_window=5, max_frames=None, loop=1, video_path_to_save="output_video_ecc_avg.mp4"):
    """

    Args:
        folder_path: path to video frames(png)
        max_window: window size to average upon, larger will be slower
        max_frames: stabilize first max_frames frames
        loop: iteration count of averaging

    Returns:

    """
    file_names = sorted(glob(os.path.join(folder_path, "*.png")))
    video = torch.cat([read_image_torch(name) for name in file_names])[:max_frames]

    averaged_video = video.clone()
    for i in range(loop):
        previous_avg_video = averaged_video.clone()
        for j in range(averaged_video.shape[0]):
            print(i, j)
            averaged_video[j], _, _ = clean_1_frame(previous_avg_video, video, j, max_window)

    averaged_video = shift_2_center(averaged_video)
    video_torch_array_to_video(averaged_video / 255, video_name=video_path_to_save)
    return averaged_video


def stabilize_ccc_image(folder_path, image_output_name="ecc.png",output_dir_path="output", max_frames=None, loop=1):
    """

    Args:
        folder_path: path to video frames(png)
        max_window: window size to average upon, larger will be slower
        max_frames: stabilize first max_frames frames
        loop: iteration count of averaging

    Returns:

    """
    #### generate data
    # img = read_image_torch(default_image_filename_to_load1) # get single channel of demo image
    file_names = sorted(glob(os.path.join(folder_path, "*.png")))
    video = torch.cat([read_image_torch(name) for name in file_names])[:max_frames]

    averaged_video = video.clone()
    for i in range(loop):
        # imshow_torch(video[i]/255)
        ###### estimate shits on avg video
        # for each index average all tensor
        previous_avg_video = averaged_video.clone()
        for j in range(averaged_video.shape[0]):
            print(i, j)
            averaged_video[j], _, _ = clean_1_frame(previous_avg_video, video, j)

    averaged_video = shift_2_center(video)
    save_image_torch(folder_path=output_dir_path, filename=image_output_name, torch_tensor=averaged_video[averaged_video.shape[0] // 2],
                     flag_convert_bgr2rgb=False, flag_scale_by_255=False, flag_save_figure=True)
    return averaged_video.mean(0)

#
# if __name__ == '__main__':
#     input_folder_path = "/home/nehoray/PycharmProjects/Shaback/output/video_example_from_security_camera/crops/2/4"
#     input_video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/scene_0_resized_short.mp4"
#
#     output_dir_path = "/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/yoav_stabilization/output"
#     output_image_name = "ecc.png"
#     output_path_to_save_video = "output_video_ecc.mp4"
#
#     # Prepare the input source and resize the frames
#     frame_dir = prepare_input_source(input_video_path)
#     # Now, proceed with the ECC functions using the frame_dir as the folder path
#
#     # a = stabilize_ccc_image(folder_path=frame_dir, max_frames=20, loop=1,output_dir_path=output_dir_path, image_output_name=output_image_name)
#     vid = stabilize_ccc_video_no_avg(folder_path=frame_dir, video_path_to_save=output_path_to_save_video)
#



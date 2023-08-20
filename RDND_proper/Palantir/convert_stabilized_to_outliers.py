import numpy as np
import torch, kornia, argparse
from torch import Tensor


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    show_video_intro = "RAW Video Viewer\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=show_video_intro)
    parser.add_argument("-v", "--video_file", help="RAW video file", required=True)
    parser.add_argument("-o", "--output_file", help="name of output file", required=True)
    parser.add_argument("-H", "--height", help="height of video frames", required=True, type=int)
    parser.add_argument("-W", "--width", help="width of video frames", required=True, type=int)
    parser.add_argument("-n", "--nbytes", help="num bytes", type=int, default=1)
    parser.add_argument("-t", "--threshold", help="outliers threshold", type=int, default=20)
    return parser


def find_outliers(aligned_tensor: Tensor,
                  bg_estimation: Tensor,
                  outliers_threshold: int = 20,
                  apply_canny: bool = True,
                  canny_low_threshold: float = 0.05 * 255,
                  canny_high_threshold: float = 0.2 * 255,
                  dilate_canny: bool = True,
                  dilation_kernel: Tensor = torch.ones(3, 3)):
    # aligned tensor and bg estimation assumed to be of equal sizes
    while len(bg_estimation.shape) < 4:
        bg_estimation = bg_estimation.unsqueeze(0)
    # calculate outliers without canny edge detection
    diff_tensor = (aligned_tensor - bg_estimation).abs()

    # find outliers threshold

    outliers_tensor = (diff_tensor > outliers_threshold).float()

    # release memory

    # If needed, calculate canny
    if apply_canny:
        # TODO split to avoid cuda out of memory, undo if enough memory is available
        _, bg_canny_threshold = kornia.filters.canny(bg_estimation, canny_low_threshold, canny_high_threshold)
        if dilate_canny:
            bg_canny_threshold = kornia.morphology.dilation(bg_canny_threshold, dilation_kernel.to(bg_canny_threshold.device))
        outliers_tensor_after_canny = (outliers_tensor - bg_canny_threshold).clamp(0)

        return outliers_tensor_after_canny
    return outliers_tensor


def main():
    parser = create_parser()
    args = parser.parse_args()
    H, W = args.height, args.width
    output_file = open(args.output_file, "wb+")
    video =  open(args.video_file, "rb")
    initial_t = 100
    last_t = initial_t
    frame_num = 0
    bg_num = 0
    frames_per_bg = 1000
    bg = None
    while last_t == initial_t:
        frames = np.fromfile(video, count=H*W*last_t, dtype=np.uint8)
        frames = frames.reshape(frames.shape[0]//(H*W), H, W)
        if (frame_num >= (frames_per_bg * bg_num)):
            bg_num+=1
            bg = np.median(frames, axis=0)
        outliers = find_outliers(torch.from_numpy(frames), torch.from_numpy(bg), args.threshold)
        output_file.write(outliers.numpy().astype(np.uint8).tobytes())
        last_t = frames.shape[0]
        frame_num+=last_t

    output_file.close()
    video.close()


if __name__ == '__main__':
    main()

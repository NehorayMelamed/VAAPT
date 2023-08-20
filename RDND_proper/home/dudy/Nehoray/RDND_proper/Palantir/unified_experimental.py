import os
from typing import Dict

import torch
from torch import Tensor

from Palantir.Palantir_utils import read_bin_images_batch
from Palantir.ResultsSaver import ResultsSaver
from Palantir.Results import Results
from Palantir.calculate_bg import calculate_bg
from Palantir.palantir_optional_stabilizations import align_palantir_CC_ECC_CC_pytorch
from Palantir.params import parameters

from icecream import ic
ic.configureOutput(includeContext=True)


class Orthodontist:
    def __init__(self, experiment_dir, params):
        self.experiment_dir = experiment_dir
        self.current_bg = None
        self.params = params
        self.current_results = ResultsSaver(params, self.experiment_dir)
        self.last_ecc_h_matrix = torch.eye(3, 3)
        img_files = list(sorted(os.listdir(self.experiment_dir)))
        # self.experimental_frames is a list of all the filenames of frames in the experimental directory in sorted order
        self.experimental_frames = [os.path.join(self.experiment_dir, img) for img in img_files]

    def align_frames(self, frames: Tensor) -> Results:
        frames = frames[:, self.params.algo.alignment_crop_h_diff//2:-self.params.algo.alignment_crop_h_diff//2,
                 self.params.algo.alignment_crop_w_diff//2:-self.params.algo.alignment_crop_w_diff//2]
        frames = frames.reshape(frames.shape[0], 1, *frames.shape[1:])
        _, _, sub_aligned_tensor, H_matrix, num_iterations_list_ecc, delta_p_list_ecc = \
            align_palantir_CC_ECC_CC_pytorch(input_tensor=frames,
                                             reference_tensor=self.current_bg,
                                             ECC_warp_mode='homography',
                                             ECC_warp_matrix=self.last_ecc_h_matrix,
                                             ECC_iterations=100)
        # sum up results to total result
        del _
        ret = Results(self.current_bg, sub_aligned_tensor, frames, H_matrix, self.params)
        return ret

    def align_all(self):
        H, W = self.params.algo.Height, self.params.algo.Width
        experimental_frames = self.experimental_frames
        bg_estimation_start_idx = self.params.algo.image_index_to_start_from_bg_estimation
        frames_per_bg_estimate = self.params.algo.number_of_images_for_bg_estimation
        frames_per_batch = self.params.local.frames_per_batch
        current_frame = 0
        batch_end_frame = frames_per_batch
        total_frames = len(experimental_frames)  # total frames in experiment
        while batch_end_frame <= total_frames:
            # runs in a loop until the batch_end_frame >= total frames in experiment
            # Then, if there are any frames left it aligns those remaining frames (there will be less than a batch left)
            if (current_frame % self.params.algo.number_of_frames_per_bg) < frames_per_batch:
                self.current_results.save_background(self.current_bg)
                frames_for_bg = read_bin_images_batch(
                filenames=experimental_frames[current_frame + bg_estimation_start_idx:
                                              current_frame + bg_estimation_start_idx + frames_per_bg_estimate],
                height=H, width=W).float()
                # frames_for_bg = frames_for_bg.reshape(frames_for_bg.shape[0], 1, H, W)
                self.current_bg = calculate_bg(frames_for_bg, self.params).to(f"cuda:{self.params.local.gpu_idx}")
                self.last_ecc_h_matrix = torch.eye(3,3)
            video_tensor = read_bin_images_batch(
                filenames=experimental_frames[current_frame: batch_end_frame],
                height=self.params.algo.Height, width=self.params.algo.Width).float()
            results = self.align_frames(video_tensor)
            self.current_results.save_results(results)
            current_frame = batch_end_frame
            batch_end_frame = current_frame + frames_per_batch
        if current_frame < total_frames: ## for the final subbatch, if exists. Will exist if total_frames modulo frames per batch is not 0
            video_tensor = read_bin_images_batch(
                filenames=experimental_frames[current_frame:],
                height=self.params.algo.Height, width=self.params.algo.Width).float()
            results = self.align_frames(video_tensor)
            self.last_ecc_h_matrix = results.H_matrix[-1]
            self.current_results.save_results(results)

        self.current_results.close()


def main():
    params = parameters()
    for experiment_dir in params.local.experiment_directories:
        orthodonist = Orthodontist(experiment_dir, params)
        orthodonist.align_all()
        del orthodonist


if __name__ == '__main__':
    main()

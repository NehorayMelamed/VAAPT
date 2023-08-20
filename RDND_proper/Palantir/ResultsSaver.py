import os, torch, cv2
from typing import List

import numpy as np
from torch import Tensor
from icecream import ic

from Palantir.Results import Results
from Palantir.ReferenceInt import ReferenceInt
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import BW2RGB

ic.configureOutput(includeContext=True)


class ResultsSaver:
    # class holds all the logic for saving results
    # all methods intended for public interface have comment marking them as public
    def __init__(self, params, experimental_dir):
        # public method
        self.params = params
        self.is_closed = False
        self.batches_per_save = self.params.local.batches_per_saving
        self.current_batch = 0

        self.results_dir = None
        self.bg_dir = None
        self.homography_matrix_dir = None
        self.aligned_frames_dir = None
        self.input_frames_dir = None
        self.outliers_frames_dir = None
        self.diff_tensor_dir = None

        # the following video_writers/binary files will only be instantiated if they are to be used
        self.input_frames_video_writer = None
        self.aligned_frames_video_writer = None
        self.aligned_frames_binary_file = None
        self.outliers_binary_file = None
        self.outliers_video_writer = None
        self.diff_video_writer = None

        self.num_orig_png_saved = ReferenceInt(0)
        self.num_aligned_png_saved = ReferenceInt(0)
        self.num_outliers_png_saved = ReferenceInt(0)
        self.num_diff_tensors_saved = ReferenceInt(0)

        self.num_bg_saved = 0
        self.num_h_matrices_saved = 0

        self.make_directory_structure(experimental_dir)
        self.make_video_files()
        self.results: List[Results] = []

    def make_directory_structure(self, exp):
        # makes needed directories
        def p(option, directory_path):
            if option and not os.path.exists(directory_path):
                os.mkdir(directory_path)

        exp_name = exp[max(exp.rfind(os.path.sep, 0, len(exp)-1)+1, 0):]
        if exp_name[-1] == os.path.sep:
            exp_name = exp_name[:-1]
        self.results_dir = os.path.join(self.params.local.results_directory, exp_name)
        p(True, self.results_dir)

        self.bg_dir = os.path.join(self.results_dir, "backgrounds")
        p(self.params.results.save_old_bgs, self.bg_dir)

        self.homography_matrix_dir = os.path.join(self.results_dir, "h_matrices")
        p(self.params.results.save_h_matrices, self.homography_matrix_dir,)

        self.aligned_frames_dir = os.path.join(self.results_dir, "aligned_frames")
        p(self.params.results.save_aligned_frames_as_png, self.aligned_frames_dir)

        self.input_frames_dir = os.path.join(self.results_dir, "input_frames")
        p(self.params.results.save_input_frames_as_png, self.input_frames_dir)

        self.outliers_frames_dir = os.path.join(self.results_dir, "outlier_frames")
        p(self.params.results.save_outliers_as_png, self.outliers_frames_dir)

        self.diff_tensor_dir = os.path.join(self.results_dir, "diff_tensors")
        p(self.params.results.save_diff_tensor_as_avi, self.diff_tensor_dir)

    def make_video_files(self):
        video_format = cv2.VideoWriter_fourcc(*'MP42')
        if self.params.results.save_input_frames_as_avi:
            self.input_frames_video_writer = cv2.VideoWriter(os.path.join(self.results_dir, "InputFrames.avi"),
                                                            video_format, float(self.params.results.save_video_fps),
                                                            (self.params.algo.final_crop_w, self.params.algo.final_crop_h))
        if self.params.results.save_aligned_frames_as_avi:
            self.aligned_frames_video_writer = cv2.VideoWriter(os.path.join(self.results_dir, "AlignedFrames.avi"),
                                                            video_format, float(self.params.results.save_video_fps),
                                                            (self.params.algo.final_crop_w, self.params.algo.final_crop_h))
        if self.params.results.save_aligned_frames_as_Bin:
            self.aligned_frames_binary_file = open(os.path.join(self.results_dir,
                                                    f"AlignedVideo_{self.params.algo.final_crop_h}_{self.params.algo.final_crop_w}.Bin"), 'wb+')

        if self.params.results.save_outliers_as_Bin:
            self.outliers_binary_file = open(os.path.join(self.results_dir,
                                                    f"Outliers_{self.params.algo.final_crop_h}_{self.params.algo.final_crop_w}.Bin"), 'wb+')
        if self.params.results.save_outliers_as_avi:
            self.outliers_video_writer = cv2.VideoWriter(os.path.join(self.results_dir, "Outliers.avi"),
                                                            video_format, float(self.params.results.save_video_fps),
                                                            (self.params.algo.final_crop_w, self.params.algo.final_crop_h))
        if self.params.results.save_diff_tensor_as_avi:
            self.diff_video_writer = cv2.VideoWriter(os.path.join(self.results_dir, "Diff.avi"),
                                                            video_format, float(self.params.results.save_video_fps),
                                                            (self.params.algo.final_crop_w, self.params.algo.final_crop_h))

    def write_array_to_video(self, video_writer: cv2.VideoWriter, frames: Tensor):
        for i in range(frames.shape[0]):
            fr = BW2RGB(frames[i])
            video_writer.write(fr)

    def save_array_as_pngs(self, base_dir: str, frames: Tensor, counter: ReferenceInt, num_zeros=6):
        for i in range(frames.shape[0]):
            cv2.imwrite(os.path.join(base_dir, f"{str(counter).rjust(num_zeros, '0')}.png"), frames[i])
            counter+=1

    def save_h_matrix(self, result: Results):
        torch.save(result.H_matrix,
                   os.path.join(self.homography_matrix_dir, f"{str(self.num_h_matrices_saved).rjust(4, '0')}.pt"))
        self.num_h_matrices_saved += 1

    def convert_to_uint8_np_grayscale_video(self, video: Tensor):
        video = video.numpy()[:, 0, :, :]
        video = ((video/video.max())*255).clip(0, 255)
        if type(video) == torch.Tensor:
            return video.numpy().astype(np.uint8)
        else:
            return video.astype(np.uint8)

    def save_cached_results(self):
        print("writing results")
        for result in self.results:
            # not all of these need to be always made...
            original_frames = self.convert_to_uint8_np_grayscale_video(result.original_frames)
            aligned_video = self.convert_to_uint8_np_grayscale_video(result.aligned_tensor)
            if self.params.results.use_outliers:
                outliers = self.convert_to_uint8_np_grayscale_video(result.outliers)
            if self.params.results.use_diff_tensor:
                diff_tensor = self.convert_to_uint8_np_grayscale_video(result.diff_tensor)
            if self.params.results.save_input_frames_as_avi:
                self.write_array_to_video(self.input_frames_video_writer, original_frames)
            if self.params.results.save_input_frames_as_png:
                self.save_array_as_pngs(self.input_frames_dir, original_frames, self.num_orig_png_saved)
            if self.params.results.save_aligned_frames_as_Bin:
                self.aligned_frames_binary_file.write(aligned_video.tobytes())
            if self.params.results.save_aligned_frames_as_avi:
                self.write_array_to_video(self.aligned_frames_video_writer, aligned_video)
            if self.params.results.save_aligned_frames_as_png:
                self.save_array_as_pngs(self.aligned_frames_dir, aligned_video, self.num_aligned_png_saved)
            if self.params.results.save_outliers_as_Bin:
                self.outliers_binary_file.write(outliers.tobytes())
            if self.params.results.save_outliers_as_avi:
                self.write_array_to_video(self.outliers_video_writer, outliers)
            if self.params.results.save_outliers_as_png:
                self.save_array_as_pngs(self.outliers_frames_dir, outliers, self.num_outliers_png_saved)
            if self.params.results.save_diff_tensor_as_avi:
                self.write_array_to_video(self.diff_video_writer, diff_tensor)
            if self.params.results.save_dif_tensor_as_png:
                self.save_array_as_pngs(self.diff_tensor_dir, diff_tensor, self.num_diff_tensors_saved)
            if self.params.results.save_h_matrices:
                self.save_h_matrix(result)

    def save_results(self, results: Results):
        # public method
        # saves results. Note that it doesn't actually save results right away, it stores them and waits for the storage
        # to be sufficiently large before writing all stored results to filesystem
        self.results.append(results)
        self.current_batch+=1
        if self.current_batch == self.batches_per_save:
            self.save_cached_results()
            self.results = []
            self.current_batch = 0

    def save_background(self, current_bg: Tensor):
        # public method
        # saves background
        if self.params.results.save_old_bgs:
            torch.save(current_bg, os.path.join(self.bg_dir, f"bg_{self.num_bg_saved}.pt"))
            self.num_bg_saved+=1

    def close(self):
        # public method
        # ensures that all results have been written to filesystem. Is useful because class holds results and does not
        # write them immediately, so something could get lost if we didn't close when done,
        # and also the files and video writers must be closed
        self.save_cached_results()
        for video_writer in [self.aligned_frames_video_writer, self.aligned_frames_video_writer, self.diff_video_writer]:
            if video_writer is not None:
                video_writer.release()
        if self.aligned_frames_binary_file:
            self.aligned_frames_binary_file.close()
        self.is_closed = True

    def __del__(self):
        # custom destructor ensures the results saver is closed
        if not self.is_closed:
            self.close()

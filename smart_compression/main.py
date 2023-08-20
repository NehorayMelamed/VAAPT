import os
import shutil
import time
from glob import glob
from pathlib import Path

import cv2
import torch

from RapidBase.Utils.Path_and_Reading_utils import save_image_torch


class VideoProcessor:
    def __init__(self, base_directory_path_input_for_output: str):
        print("set_global_variables")
        self.base_directory_path = base_directory_path_input_for_output
        self.VIDEOS_SAVE_DIR_PATH = f"{self.base_directory_path}/videos"
        self.MERGE_VIDEOS_EXP_PATH = f"{self.base_directory_path}/merge_videos_exp"
        self.TEMP1_DIRECTORY_PATH = f"{self.base_directory_path}/temp1"
        self.TEMP2_DIRECTORY_PATH = f"{self.base_directory_path}/temp2"

        if not os.path.exists(self.base_directory_path):
            os.mkdir(self.base_directory_path)
        if not os.path.exists(self.VIDEOS_SAVE_DIR_PATH):
            os.mkdir(self.VIDEOS_SAVE_DIR_PATH)
        if not os.path.exists(self.MERGE_VIDEOS_EXP_PATH):
            os.mkdir(self.MERGE_VIDEOS_EXP_PATH)
        if not os.path.exists(self.TEMP1_DIRECTORY_PATH):
            os.mkdir(self.TEMP1_DIRECTORY_PATH)
        if not os.path.exists(self.TEMP2_DIRECTORY_PATH):
            os.mkdir(self.TEMP2_DIRECTORY_PATH)

    def extract_frames_from_mp4(self, video_path, save_path):
        os.makedirs(save_path, exist_ok=True)
        command = f"ffmpeg -i  {video_path} -y {os.path.join(save_path, 'out_%05d.png')}"
        os.system(command)


    def encode_h265_with_given_crf(self, images_dir, preset='medium', crf=23,
                                   string="", fps=100):
        file_save_dir = self.VIDEOS_SAVE_DIR_PATH
        flags = '-y -psnr'
        # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
        file_save_path = f"{file_save_dir}/h265_{crf}_{string}_{fps}.mp4"


        old_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "crf={} preset={}" {}'.format(
            fps, images_dir, flags, crf, preset, file_save_path)
        new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v hevc -crf {} -preset {} {}'.format(
            fps, images_dir, flags, crf, preset, file_save_path)

        start = time.time()
        os.system(new_command)
        end = time.time()
        print(f"encoding time was : {end - start}")

        print(f"file size with crf={crf} is {os.path.getsize(file_save_path)}")
        return file_save_path


    def encode_h264_with_given_crf(self, images_dir, preset='medium', crf=23,
                                   string="", fps=100, ext='.png'):
        file_save_dir = self.VIDEOS_SAVE_DIR_PATH
        flags = '-y'
        # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
        file_save_path = f"{file_save_dir}/h264_{crf}_{string}_{fps}.mp4"

        new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*{}" {} -c:v h264 -crf {} -preset {} {}'.format(
            fps, images_dir, ext, flags, crf, preset, file_save_path)

        start = time.time()
        os.system(new_command)
        end = time.time()
        print(f"encoding time was : {end - start}")

        print(f"file size with crf={crf} is {os.path.getsize(file_save_path)}")
        return file_save_path


    def encode_h265_with_given_qp(self, images_dir, preset='medium', qp=0,
                                  string="", fps=100):
        file_save_dir = self.VIDEOS_SAVE_DIR_PATH
        flags = '-y -psnr'
        # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
        file_save_path = f"{file_save_dir}/h265_{qp}_{string}_{fps}.mp4"


        old_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx265 -x265-params "qp={} preset={}" {}'.format(
            fps, images_dir, flags, qp, preset, file_save_path)
        new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v hevc -qp {} -preset {} {}'.format(
            fps, images_dir, flags, qp, preset, file_save_path)

        start = time.time()
        os.system(new_command)
        end = time.time()
        print(f"encoding time was : {end - start}")

        print(f"file size with QP={qp} is {os.path.getsize(file_save_path)}")


    def encode_h264_with_given_qp(self, images_dir, preset='medium', qp=0,
                                  string="", fps=100):
        file_save_dir = self.VIDEOS_SAVE_DIR_PATH
        flags = '-y'
        # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
        file_save_path = f"{file_save_dir}/h264_{qp}_{string}_{fps}.mp4"

        old_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v libx264 -x264-params "qp={} preset={}" {}'.format(
            fps, images_dir, flags, qp, preset, file_save_path)
        new_command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v h264 -qp {} -preset {} {}'.format(
            fps, images_dir, flags, qp, preset, file_save_path)

        start = time.time()
        os.system(new_command)
        end = time.time()
        print(f"encoding time was : {end - start}")

        print(f"file size with QP={qp} is {os.path.getsize(file_save_path)}")
        return file_save_path


    def encode_h264_skip_mode(self, images_dir, preset='medium', crf=0,
                              string="", fps=100):
        file_save_dir = self.VIDEOS_SAVE_DIR_PATH
        # todo: figure out how to control skip mode in encoder
        flags = '-y'
        # file_save_path = os.path.join(images_dir, f"h265_{crf}.mp4")
        # static_thresh = 20000  # from 0 to max int
        for static_thresh in [0, 100, 5000, 25000]:
            file_save_path = f"{file_save_dir}/h264_{crf}_{string}_{fps}_{static_thresh}.mp4"

            skip_mode_encoding = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} -c:v h264 -crf {} -preset {} -static-thresh {} {}'.format(
                fps, images_dir, flags, crf, preset, static_thresh, file_save_path)

            start = time.time()
            os.system(skip_mode_encoding)
            end = time.time()
            print(f"encoding time was : {end - start}")

            print(f"file size with CRF={crf} is {os.path.getsize(file_save_path)}")
        return file_save_path


    def merge_frames_according_to_mask(self, frames_dir_1, frames_dir_2, mask_dir,
                                       ):
        save_path = self.MERGE_VIDEOS_EXP_PATH
        """
        merge 2 videos, where mask=1 take video 1 and when mask=0 take dir 2
        Args:
            frames_dir_1: video 1
            frames_dir_2: video 2
            mask_dir: masks

        Returns:
        """
        # output_frames = []
        frames_1 = sorted(glob(os.path.join(frames_dir_1, '*.png')))
        frames_2 = sorted(glob(os.path.join(frames_dir_2, '*.png')))
        masks = sorted(glob(os.path.join(mask_dir, '*.png')))
        for i, (frame1, frame2, mask) in enumerate(zip(frames_1, frames_2, masks)):
            frame1 = torch.tensor(cv2.cvtColor(cv2.imread(frame1), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            frame2 = torch.tensor(cv2.cvtColor(cv2.imread(frame2), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            mask = torch.tensor(cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

            frame_merged = torch.zeros_like(frame1)
            frame_merged[mask == 0] = frame2[mask == 0]
            frame_merged[mask == 255] = frame1[mask == 255]
            # output_frames.append(frame_merged)
            save_image_torch(save_path, f'merged_frame_{str(i).zfill(4)}.png', frame_merged)



    def double_compression_method(self, images_dir, preset='medium',
                                  fps=25, encoder='h264', metric='crf', gammas=None, mask_dir=None, extra_str=""):
        temp_save_dir = self.VIDEOS_SAVE_DIR_PATH
        final_save_dir = self.VIDEOS_SAVE_DIR_PATH
        """
        A method we will call double compression.
        We basically want a way to take a video and control the QP per macro block ourselves.
        This can be done the following way:
        * compress once in the highest quality possible - gamma 1 = 0
        * compress second time in the lowest quality that you are willing to tolerate(the unsegmented/not-important part) - gamma 2
        * merge both videos based on segmentation mask
        * compress result based on the lowest quality that you are willing to tolerate(the segmented/important part) - gamma 3

        Args:
            images_dir:
            preset:
            file_save_dir:
            fps:
            encoder:
            metric:
            gammas: [gamma1. gamma2. gamma3]

        Returns:

        """

        if str.upper(encoder) == 'H264':
            if str.lower(metric) == 'crf':
                encoding_function = self.encode_h264_with_given_crf
            elif str.lower(metric) == 'qp':
                encoding_function = self.encode_h264_with_given_qp
            else:
                raise RuntimeError('only valid metrics are qp and crf')
        elif str.upper(encoder) == 'H265':
            if str.lower(metric) == 'crf':
                encoding_function = self.encode_h265_with_given_crf
            elif str.lower(metric) == 'qp':
                encoding_function = self.encode_h265_with_given_qp
            else:
                raise RuntimeError('only valid metrics are qp and crf')
        else:
            raise RuntimeError('encoder not supported, H264 or H265 only')

        gamma_1 = gammas[0] if gammas else 0  # might be called gamma
        gamma_2 = gammas[1] if gammas else 51  # might be called beta
        gamma_3 = gammas[2] if gammas else 25  # might be called alpha
        # encode with different crf before merging
        file_path_high_quality = encoding_function(images_dir, preset, gamma_1, string=metric, fps=fps)
        file_path_low_quality = encoding_function(images_dir, preset, gamma_2, string=metric, fps=fps)
        # extract frames in order to merge
        temp_save_folder1 = self.TEMP1_DIRECTORY_PATH
        if os.path.exists(temp_save_folder1):
            shutil.rmtree(temp_save_folder1)
        os.makedirs(temp_save_folder1, exist_ok=True)
        temp_save_folder2 = self.TEMP2_DIRECTORY_PATH
        if os.path.exists(temp_save_folder2):
            shutil.rmtree(temp_save_folder2)
        os.makedirs(temp_save_folder2, exist_ok=True)
        self.extract_frames_from_mp4(file_path_high_quality, temp_save_folder1)
        self.extract_frames_from_mp4(file_path_low_quality, temp_save_folder2)

        # extract segmentation mask
        if mask_dir is None: # create a mask dir and save to it
            raise RuntimeError("Please implement me")
            mask_dir = "/raid/yoav/DVC/small_garden/temp_masking"
            if os.path.exists(mask_dir):
                shutil.rmtree(mask_dir)
            os.makedirs(mask_dir, exist_ok=True)
            save_semantic_segmentation_masks(images_dir, mask_dir)
            save_yolo_semantic_segmentation_masks(images_dir, mask_dir, interesting_class_names=['person', 'car'])
            save_outlier_masks(images_dir, mask_dir)

        save_path_merged = self.MERGE_VIDEOS_EXP_PATH
        if os.path.exists(save_path_merged):
            shutil.rmtree(save_path_merged)
        os.makedirs(save_path_merged, exist_ok=True)
        # merge frames
        self.merge_frames_according_to_mask(temp_save_folder1, temp_save_folder2, mask_dir=mask_dir)
        # encode merged frames

        self.merged_video_path = encoding_function(save_path_merged, 'medium', gamma_3,
                                              string=f'{extra_str}_merged_video_{gamma_1}_{gamma_2}_{gamma_3}', fps=fps)

        _ = encoding_function(images_dir, 'medium', gamma_3,
                              string=f'{extra_str}_original_video_{gamma_3}', fps=fps)
        # return merged_video_path
        return True




# def main_interface(mask_dir_path)

# if __name__ == '__main__':
#     images_dir_path = "/home/nehoray/PycharmProjects/Shaback/yolo_tracking/examples/runs/track/exp/frames"
#     mask_dir_path = "/home/nehoray/PycharmProjects/Shaback/yolo_tracking/examples/runs/track/exp/masks"
#     output_base_dir_path = "/home/nehoray/PycharmProjects/Shaback/smart_compression/output"
#     vp = VideoProcessor(base_directory_path_input_for_output=output_base_dir_path)
#     vp.double_compression_method(images_dir=images_dir_path, mask_dir=mask_dir_path)
#

import numpy as np
import os
import glob
from application.pipeline.AlgorithmParams import AlgorithmParams


class FrameGenerator:
    def __init__(self, params: AlgorithmParams):
        # self.binary_video = open(params.video_path, 'rb')
        self.num_frames_stack = params.num_frames  # number of frames per stack
        self.height = params.height
        self.width = params.width
        # self.stack_size = self.height*self.width*self.num_frames_stack
        # self.video_size = os.path.getsize(params.video_path)
        # print(f"VIDEO SIZE IS {self.video_size}")
        self.cur_byte = 0

        self.num_images_per_stack = 500
        # self.image_dir = "/media/efcom/storage/QS_data/full_frame"
        self.image_dir = "/media/efcom/storage/QS_data/roi"
        self.total_num_images = len(glob.glob(f"{self.image_dir}/*.raw"))
        self.current_image = 0
        self.dtype = params.dtype
        self.dtype_size = self.size_of(params.dtype)

    def size_of(self, dtype):
        if dtype == np.uint8:
            return 1
        elif dtype == np.uint16:
            return 2
        elif dtype == np.uint32:
            return 4
        else:
            return 8

    def get_next_stack(self) -> np.array:  # -> pushes np.array of form 30 X Height X Width
        if self.cur_byte < self.video_size - self.stack_size:
            raw_data = np.fromfile(self.binary_video, dtype=self.dtype, count=self.stack_size)
            current_stack = raw_data.reshape(self.num_frames_stack, self.height, self.width)
            self.cur_byte += self.dtype_size*self.stack_size
            return current_stack
        else:
            return None

    def get_next_image_stack(self) -> np.array: # -> pushed np.array of form 30 X Height X Width of raw images from disk
        frame_stack = []
        current_image = 0
        print("Getting next image stack!")
        # Get 30 frames OR finish when the directory is finished
        while current_image < self.num_images_per_stack:
            # print(f"Reading raw data from {self.image_dir}/full_frame_{current_image}.raw")
            raw_data = np.fromfile(os.path.join(self.image_dir, f"full_frame_{current_image}.raw"), dtype=self.dtype)
            reshaped_image = raw_data.reshape((self.height, self.width))
            frame_stack.append(reshaped_image)
            current_image += 1
            self.current_image += 1

        result = np.array(frame_stack)
        return result

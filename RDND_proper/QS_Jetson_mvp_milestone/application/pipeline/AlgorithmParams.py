import numpy as np
from pathlib import Path
import vpi, os
from datetime import datetime

from .file_utils import extract_video_name, ensure_dirs


class AlgorithmParams:
    def __init__(self, video_path: str, height: int, width: int, find_harris: bool, dtype: type , vpi_dtype, b_learn_rate: float,
                 b_threshold: float, b_shadow, b_n_sigma: float, b_use_custom: bool, h_sensitivity: float,  h_strength: float,  h_min_nms_distance: float,
                 r_max_iterations: int, num_frames: int, save_all: bool, r_min_samples: int, r_residual_threshold: float,
                 pnts_in_trj_limit: float, max_trajectory_velocity: float, max_t_sus_disappear: int ):
        if video_path:
            self.video_name = extract_video_name(video_path)
        else:
            now = datetime.now()
            self.video_name = f"{now.day}_{now.month}_{now.year}_{now.hour}_{now.minute}_{now.second}"
        print(f"VIDEO NAME IS {self.video_name}\n")
        self.video_path = video_path
        self.height = height
        self.width = width
        self.find_harris = find_harris
        self.dtype = dtype
        self.vpi_dtype = vpi_dtype
        self.b_learn_rate = b_learn_rate
        self.b_threshold = b_threshold
        self.b_shadow = b_shadow
        self.b_n_sigma = b_n_sigma
        self.b_use_custom = b_use_custom
        self.h_sensitivity = h_sensitivity
        self.h_strength = h_strength
        self.h_min_nms_distance = h_min_nms_distance
        self.r_min_samples = r_min_samples
        self.r_max_iterations = r_max_iterations
        self.r_residual_threshold = r_residual_threshold
        self.num_frames = num_frames  # number of frames per ransac
        self.save_all = save_all
        self.pnts_in_trj_limit = pnts_in_trj_limit
        self.max_trajectory_velocity = max_trajectory_velocity
        self.max_t_sus_disappear = max_t_sus_disappear
        self.exposure_time = -1

        s = os.path.sep
        great_grandparent_directory = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent)
        self.data_dir = f"{great_grandparent_directory}{s}data{s}"
        self.results_dir = f"{self.data_dir}results{s}"
        self.harris_dir = f"{self.results_dir}HARRIS{s}"
        self.BGS_dir = f"{self.results_dir}BGS{s}"
        self.ransac_dir = f"{self.results_dir}RANSAC{s}"
        self.ransac_inliers_dir = f"{self.ransac_dir}INLIERS{s}"
        self.ransac_regression_dir = f"{self.ransac_dir}REGRESSIONS{s}"
        self.ransac_outliers_dir = f"{self.ransac_dir}OUTLIERS{s}"
        self.original_dir = f"{self.results_dir}ORIGINAL{s}"

        ensure_dirs([self.data_dir, self.results_dir, self.harris_dir, self.BGS_dir, self.ransac_dir, self.ransac_inliers_dir,
                     self.ransac_regression_dir, self.original_dir, self.ransac_outliers_dir])

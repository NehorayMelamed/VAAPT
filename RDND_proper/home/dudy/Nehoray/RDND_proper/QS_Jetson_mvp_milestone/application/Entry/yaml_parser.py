import os, yaml, vpi, sys
from pathlib import Path
from typing import Tuple
import numpy as np

from application.pipeline.AlgorithmParams import AlgorithmParams


def get_config_file_path() -> str:
    grandparent_directory = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    return os.path.join(grandparent_directory, "config.yml")


def get_video_path(cfg: dict, rt: bool):
    video_path = None
    if not rt:
        video_path = cfg["filesystem"]["path"]
        # assert os.path.exists(video_path), "GIVEN VIDEO PATH DOES NOT EXIST"
    return video_path


def get_height_width(cfg: dict, rt: bool) -> Tuple[int, int]:
    if rt:
        return (-1, -1)
    else:
        return int(cfg['filesystem']['height']), int(cfg['filesystem']['width'])


def extract_dtype(cfg: dict):
    num_bytes = cfg['algorithm']['num_bytes']
    if num_bytes == 1:
        return np.uint8, vpi.U8
    elif num_bytes == 2:
        return np.uint16, vpi.U16
    elif num_bytes == 4:
        return np.uint32, vpi.F32
    elif num_bytes == 8:
        return np.uint64, vpi.F64
    else:
        print("UNKNOWN DTYPE ARGUMENT, NUM_BYTES MUST BE 1,2,4, or 8")
        sys.exit(1)


def create_params_from_yaml(cfg: dict, rt: bool) -> AlgorithmParams:
    video_path = get_video_path(cfg, rt)
    height, width = get_height_width(cfg, rt)
    find_harris = bool(cfg["algorithm"]["find_harris"])
    dtype, vpi_dtype = extract_dtype(cfg)
    save_all = bool(cfg["algorithm"]["save_all"])
    b_learn_rate = float(cfg["algorithm"]["bgs_learn_rate"])
    b_threshold = float(cfg["algorithm"]["bgs_threshold"])
    b_shadow = int(cfg["algorithm"]["bgs_shadow"])
    b_use_custom = bool(cfg["algorithm"]["use_custom_bgs"])
    b_n_sigma = float(cfg["algorithm"]["use_custom_bgs"])
    h_sensitivity = float(cfg["algorithm"]["harris_sensitivity"])
    h_strength = float(cfg["algorithm"]["harris_strength"])
    h_min_nms_distance = float(cfg["algorithm"]["harris_min_nms_distance"])
    r_residual_threshold = float(cfg["algorithm"]["ransac_residual_threshold"])
    if r_residual_threshold + 1 < 1e-2: # ie is basically 0
        r_residual_threshold = None  # use default
    r_min_samples = int(cfg["algorithm"]["ransac_min_samples"])
    r_max_iterations = int(cfg["algorithm"]["ransac_min_samples"])
    pnts_in_trj_limit = float(cfg["algorithm"]["pnts_in_trj_limit"])
    max_trajectory_velocity = float(cfg["algorithm"]["max_trajectory_velocity"])
    max_t_sus_disappear = int(cfg["algorithm"]["max_t_sus_disappear"])
    num_frames = int(cfg["algorithm"]["num_frames"])
    return AlgorithmParams(video_path=video_path, height=height, width=width, find_harris=find_harris, dtype=dtype,
                           vpi_dtype=vpi_dtype, b_learn_rate=b_learn_rate, b_threshold=b_threshold, b_shadow=b_shadow, b_n_sigma=b_n_sigma, b_use_custom=b_use_custom,
                           h_sensitivity=h_sensitivity, h_strength=h_strength, h_min_nms_distance=h_min_nms_distance,
                           r_residual_threshold=r_residual_threshold, r_max_iterations=r_max_iterations, r_min_samples=r_min_samples,
                           num_frames=num_frames, save_all=save_all, pnts_in_trj_limit=pnts_in_trj_limit, max_trajectory_velocity=max_trajectory_velocity,
                           max_t_sus_disappear=max_t_sus_disappear)


def parse_config() -> Tuple[bool, bool, AlgorithmParams]:
    config_file_path = get_config_file_path()
    assert os.path.exists(config_file_path), "CONFIG FILE NAME/LOCATION MUST NOT BE CHANGED"
    with open(config_file_path, 'r') as config_file:
        cfg = yaml.load(config_file)
    print(f"\nCONFIG PARAMETERS: {cfg}\n")
    rt = cfg["use_rt"]
    kaya_fast = bool(cfg["kaya"]["use_fast"])
    params = create_params_from_yaml(cfg, rt)
    return rt, kaya_fast, params

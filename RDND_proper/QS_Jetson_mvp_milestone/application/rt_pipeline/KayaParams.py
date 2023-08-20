import os, yaml
from pathlib import Path


class BaseKayaParams:
    def __init__(self, exposure_time: float, fps: float, batches_per_exposure: int,
                    max_gray_level: int, max_saturated_pixels: int, saturation_range: float, max_exposure_time: float):
        self.exposure_time = exposure_time
        self.fps = fps
        self.pixel_format = 0x101
        self.batches_per_exposure = batches_per_exposure
        self.max_gray_level = max_gray_level
        self.max_saturated_pixels = max_saturated_pixels
        self.saturation_range = saturation_range
        self.max_exposure_time = max_exposure_time


class SlowModeKayaParams(BaseKayaParams):
    def __init__(self, exposure_time: float, fps: float, batches_per_exposure: int,
                    max_gray_level: int, max_saturated_pixels: int, saturation_range: int, max_exposure_time: float,
                 height: int, width: int):

        super().__init__(exposure_time, fps, batches_per_exposure,
                    max_gray_level, max_saturated_pixels, saturation_range, max_exposure_time)
        self.height = height
        self.width = width


class FastModeKayaParams(BaseKayaParams):
    def __init__(self, exposure_time: float, fps: float, batches_per_exposure: int,
                    max_gray_level: int, max_saturated_pixels: int, saturation_range: float, max_exposure_time: float,
                 multi_roi_height: int, multi_roi_width: int, offset_x: int, offset_y: int):
        super().__init__(exposure_time, fps, batches_per_exposure,
                    max_gray_level, max_saturated_pixels, saturation_range, max_exposure_time)
        self.height = multi_roi_height
        self.width = multi_roi_width
        self.offset_x = offset_x
        self.offset_y = offset_y


def get_config_file_path() -> str:
    grandparent_directory = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)
    return os.path.join(grandparent_directory, "config.yml")


def load_config() -> dict:
    config_file_path = get_config_file_path()
    assert os.path.exists(config_file_path), "CONFIG FILE NAME/LOCATION MUST NOT BE CHANGED"
    with open(config_file_path, 'r') as config_file:
        cfg = yaml.load(config_file)
    return cfg


def get_base_params(cfg: dict, mode: str) -> list:
    exposure_time = float(cfg["kaya"][mode]["exposure_time"])
    fps = float(cfg["kaya"][mode]["fps"])
    batches_per_exposure = int(cfg["kaya"][mode]["batches_per_exposure"])
    max_gray_level = int(cfg["kaya"][mode]["max_gray_level"])
    max_saturated_pixels = int(cfg["kaya"][mode]["max_saturated_pixels"])
    saturation_range = int(cfg["kaya"][mode]["saturation_range"])
    max_exposure_time = float(cfg["kaya"][mode]["max_exposure"])
    return [exposure_time, fps, batches_per_exposure, max_gray_level, max_saturated_pixels, saturation_range, max_exposure_time]


def build_slow_mode_params() -> SlowModeKayaParams:
    cfg = load_config()
    base_params = get_base_params(cfg, "slow_mode")  # list of needed params
    height = cfg["kaya"]["slow_mode"]["height"]
    width = cfg["kaya"]["slow_mode"]["width"]
    return SlowModeKayaParams(*base_params, height, width)  # extract list


def build_fast_mode_params() -> FastModeKayaParams:
    cfg = load_config()
    base_params = get_base_params(cfg, "fast_mode")
    multi_roi_height = cfg["kaya"]["fast_mode"]["roi_height"]
    multi_roi_width = cfg["kaya"]["fast_mode"]["roi_width"]
    offset_x = cfg["kaya"]["fast_mode"]["offset_x"]
    offset_y = cfg["kaya"]["fast_mode"]["offset_y"]
    return FastModeKayaParams(*base_params, multi_roi_height, multi_roi_width, offset_x, offset_y)

import yaml, os

from Model import Model
from KayaParams import KayaParams


def get_config_file_path() -> str:
    grandparent_directory = str(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(grandparent_directory, "config.yml")


def get_kaya_params(cfg: dict, mode: str) -> KayaParams:
    height = int(cfg[mode]["height"])
    width = int(cfg[mode]["width"])
    fps = float(cfg[mode]["fps"])
    exposure_time = float(cfg[mode]["exposure_time"])
    num_frames = int(cfg[mode]["num_frames"])
    return KayaParams(height=height, width=width, fps=fps, exposure_time=exposure_time, num_frames=num_frames)


def parse_config(config_path: str, model_queue, model_cond, forward_callback) -> Model:
    with open(config_path, 'r') as config_file:
        cfg = yaml.load(config_file)
    print(cfg)
    slow_params = get_kaya_params(cfg, 'slow_mode')
    fast_params = get_kaya_params(cfg, 'fast_mode')
    data_dir = cfg["file_saving_path"]
    assert os.path.exists(data_dir), "GIVEN DATA DIRECTORY DOES NOT EXIST"
    return Model(slow_params=slow_params, fast_params=fast_params, file_saving_path=data_dir,
                 mp_queue=model_queue, mp_cond=model_cond, grabber_callback=forward_callback)


def get_model(model_queue, model_lock, forward_callback):
    config_file_path = get_config_file_path()
    assert os.path.exists(config_file_path), "CONFIG FILE NAME/LOCATION MUST NOT BE CHANGED"
    return parse_config(config_file_path, model_queue, model_lock, forward_callback)

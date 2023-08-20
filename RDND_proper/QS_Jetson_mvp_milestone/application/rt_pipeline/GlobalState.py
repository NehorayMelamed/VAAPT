from multiprocessing import Queue

from application.pipeline.MainPipeline import MainPipeline
from application.pipeline.AlgorithmParams import AlgorithmParams
from application.rt_pipeline.FrameQueue import FrameQueue
from application.rt_pipeline.KayaParams import build_slow_mode_params, build_fast_mode_params, FastModeKayaParams, SlowModeKayaParams


class GlobalState:
    def __init__(self, params: AlgorithmParams, queue: Queue, processing_busy_lock, processing_busy_flag, is_fast):
        if is_fast:
            self.kaya_params: FastModeKayaParams = build_fast_mode_params()
        else:
            self.kaya_params: SlowModeKayaParams = build_slow_mode_params()
        self.is_fast = is_fast
        self.frame_queue = FrameQueue(q_size=params.num_frames)
        self.current_batch_idx = 0
        self.current_frame = 0
        self.params = params
        self.params.height = self.kaya_params.height
        self.params.width = self.kaya_params.width
        self.pipeline = MainPipeline(params)
        self.current_exposure_time = self.kaya_params.exposure_time
        self.mp_q = queue  # kaya frames will be pushed to this queue
        self.processing_busy_lock = processing_busy_lock
        self.processing_busy_flag = processing_busy_flag
        self.image_stream = None

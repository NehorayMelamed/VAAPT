from multiprocessing import Process, Queue
import time

from application.fs_pipeline.FrameGenerator import FrameGenerator
from application.pipeline.AlgorithmParams import AlgorithmParams


class GeneratorProcess(Process):
    def __init__(self, params: AlgorithmParams, mp_q: Queue):
        super().__init__()
        self.params = params
        self.frame_generator = FrameGenerator(params)
        self.frame_queue = mp_q
        self.time_elapse = self.params.num_frames / 30

    def sync_fps(self, processing_time: float):
        if processing_time < self.time_elapse:
            time.sleep(self.time_elapse - processing_time)

    def run(self):
        while True:
            start = time.time()

            # current_stack = self.frame_generator.get_next_stack()
            current_stack = self.frame_generator.get_next_image_stack()
            self.frame_queue.put(current_stack)
            if current_stack is None:
                return
            end = time.time()
            self.sync_fps(end-start)

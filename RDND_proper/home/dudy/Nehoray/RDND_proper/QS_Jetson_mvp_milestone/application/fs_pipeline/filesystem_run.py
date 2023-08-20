from multiprocessing import Queue

from application.pipeline.AlgorithmParams import AlgorithmParams
from application.fs_pipeline.GeneratorProcess import GeneratorProcess
from application.pipeline.MainPipeline import MainPipeline


def run_from_filesystem(params: AlgorithmParams):
    queue = Queue()
    frame_process = GeneratorProcess(params, queue)
    frame_process.start()
    pipeline = MainPipeline(params)
    while True:
        original_stack = queue.get()
        if original_stack is None:
            return
        else:
            pipeline.run_pipeline(original_stack, params)

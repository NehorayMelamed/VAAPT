from multiprocessing import Queue, Value, Lock
from application.rt_pipeline.KYFGLib import *

from application.rt_pipeline.GlobalState import GlobalState
from yaml_parser import parse_config
from application.rt_pipeline.kaya_pipeline import initialize_kaya, push_frames_queue
from application.fs_pipeline.filesystem_run import run_from_filesystem
from application.rt_pipeline.kaya_pipeline import reset_exposure_time, set_exposure_time


def forward_callback(buffer_handle: STREAM_HANDLE, userContext):  # forwards KAYA callback, and adds parameters
    global state
    state.processing_busy_lock.acquire()
    busy = state.processing_busy_flag.value
    state.processing_busy_lock.release()

    if not busy:
        push_frames_queue(buffer_handle, state)


if __name__ == '__main__':
    INTER_BATCH_DELAY_SECONDS = 10
    rt, kaya_fast, params = parse_config()
    if not rt:
        run_from_filesystem(params)
    else:  # live, real time
        frame_queue = Queue()
        processing_busy_lock = Lock()
        processing_busy_flag = Value('i', False)
        state = GlobalState(params, frame_queue, processing_busy_lock, processing_busy_flag, kaya_fast)
        image_stream = initialize_kaya(forward_callback, state, 0) # also sets GlobalState image stream, important state owns image stream before callback runs
        #set_exposure_time(state.kaya_params.exposure_time, image_stream, False)
        while True:
            s = time.perf_counter()

            frame_stack = frame_queue.get()
            # I am busy processing stuff
            state.processing_busy_lock.acquire()
            state.processing_busy_flag.value = True
            state.processing_busy_lock.release()

            print("\n\n\n\n\n\nGOT STACK!!!!\n\n\n\n\n")
            state.pipeline.run_pipeline(frame_stack, state.params)
            e = time.perf_counter()
            print(f"TIME TO RUN PIPELINE {1000*(e-s)} ms")

            time.sleep(INTER_BATCH_DELAY_SECONDS)
            state.processing_busy_lock.acquire()
            state.processing_busy_flag.value = False
            state.processing_busy_lock.release()

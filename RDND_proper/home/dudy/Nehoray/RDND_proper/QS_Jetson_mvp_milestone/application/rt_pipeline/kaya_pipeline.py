from application.rt_pipeline.KYFGLib import *
import numpy as np
from ctypes import string_at

from application.rt_pipeline.frame_grabber import initialize_libs, print_device_data, open_grabber, ImageStream, connect_to_grabber, \
    start_grabber
from application.rt_pipeline.autoexposure_calc import calc_new_exposure_time
from application.rt_pipeline.GlobalState import GlobalState
from application.rt_pipeline.frame_grabber import set_exposure_time


def set_slow_mode(state: GlobalState):
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "ExposureTime", state.kaya_params.exposure_time)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "AcquisitionFrameRate", state.kaya_params.fps)


def set_fast_mode(state: GlobalState):
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiMode", 1)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiIndex", 0)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiHorizontalEnableNumber", 1)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiVerticalEnableNumber", 1)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiWidth", state.kaya_params.width)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiHeight", state.kaya_params.height)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiOffsetX", state.kaya_params.offset_x)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiOffsetY",  state.kaya_params.offset_y)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "ExposureTime", state.kaya_params.exposure_time)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "AcquisitionFrameRate", state.kaya_params.fps)


def initialize_kaya(callback_function, state: GlobalState, num_frames: int):
    KYFG_Init()
    initialize_libs()
    print_device_data()
    connection: FGHANDLE = open_grabber(callback_function, 0)
    #image_stream: ImageStream = connect_to_grabber(connection)
    #time.sleep(3)
    #initialize_libs()
    #connection: FGHANDLE = open_grabber(callback_function, 0)
    image_stream: ImageStream = connect_to_grabber(connection)
    state.image_stream = image_stream
    if state.is_fast:
        set_fast_mode(state)
    else:
        set_slow_mode(state)
    start_grabber(image_stream, num_frames)
    return image_stream


def extract_array(byte_array: bytes, height: int, width: int) -> np.array:
    intermediate = np.fromstring(byte_array, dtype=np.uint8) # fromstring copies the data
    reshaped = np.resize(intermediate, (height, width))
    return reshaped.copy()


def extract_buffer(buffer_handle: STREAM_HANDLE, size: int) -> bytes:
    (status, buffIndex) = KYFG_StreamGetFrameIndex(buffer_handle)
    (buffData,) = KYFG_StreamGetPtr(buffer_handle, buffIndex)
    data: bytes = string_at(buffData, size)
    return data


def reset_exposure_time(last_frame: np.array, state: GlobalState):
    new_exposure_time = calc_new_exposure_time(last_frame, state.current_exposure_time, state.kaya_params.max_gray_level,
                                                state.kaya_params.max_saturated_pixels,
                                               state.kaya_params.saturation_range, state.kaya_params.max_exposure_time)

    if abs(new_exposure_time / state.current_exposure_time - 1) > 1e-2:  # if the difference is signfigant
        state.pipeline.close()
        set_exposure_time(new_exposure_time, state.image_stream)
        state.current_exposure_time = new_exposure_time
        state.params.exposure_time = new_exposure_time
        state.pipeline.reopen()
        print("COMPLETED SETTING EXPOSURE TIME")


def push_frames_queue(buffer_handle: STREAM_HANDLE, state: GlobalState):
    data = extract_buffer(buffer_handle, state.kaya_params.height*state.kaya_params.width)
    frame = extract_array(data, state.kaya_params.height, state.kaya_params.width)
    state.frame_queue.add_frame(frame)
    if (state.current_batch_idx % state.kaya_params.batches_per_exposure == 0 and state.current_frame == 0) or \
            (state.current_batch_idx == 0 and state.current_frame <= 5):
        print("\n\n\n\nRESETTING EXPOSURE TIME\n\n\n")
        reset_exposure_time(frame, state)
    state.current_frame += 1
    print(state.current_frame)
    if state.frame_queue.is_full():
        state.mp_q.put(state.frame_queue.get_queue())
        state.frame_queue.empty_queue()
        state.current_frame = 0
        state.current_batch_idx+=1


"""
def run_simple_pipeline(buffer_handle, state: GlobalState):
        (status, buffIndex) = KYFG_StreamGetFrameIndex(buffer_handle)
        (buffData,) = KYFG_StreamGetPtr(buffer_handle, buffIndex)
        data: bytes = string_at(buffData, state.params.HEIGHT * state.params.WIDTH)
        frame = extract_array(data, state.params.HEIGHT, state.params.WIDTH)
        cv2.imsave(f"/home/efcom/Desktop/savedframes{state.current_batch_idx}.png", frame)
        state.current_batch_idx+=1
"""
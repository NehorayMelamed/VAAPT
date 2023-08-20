from KYFGLib import *
import numpy as np
from ctypes import string_at

from frame_grabber import initialize_libs, print_device_data, open_grabber, start_grabber, stop_grabber, connect_to_grabber, \
    ImageStream


def set_slow_mode(state):
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiMode", 0)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "Height", state.slow_params.height)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "Width", state.slow_params.width)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "ExposureTime", state.slow_params.exposure_time)
    KYFG_SetCameraValue(state.image_stream.camera_handles[0], "AcquisitionFrameRate", state.slow_params.fps)


def set_fast_mode(state):
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


def initialize_kaya(callback_function, state, num_frames: int):
    KYFG_Init()
    initialize_libs()
    print_device_data()
    connection: FGHANDLE = open_grabber(callback_function, 0)
    image_stream: ImageStream = connect_to_grabber(connection)
    state.image_stream = image_stream
    set_slow_mode(state)
    start_grabber(image_stream, num_frames)
    return image_stream


def extract_array(byte_array: bytes, height: int, width: int) -> np.array:
    intermediate = np.fromstring(byte_array, dtype=np.uint8)  # fromstring copies the data
    reshaped = np.resize(intermediate, (height, width))
    return reshaped.copy()


def extract_buffer(buffer_handle: STREAM_HANDLE, size: int) -> bytes:
    (status, buffIndex) = KYFG_StreamGetFrameIndex(buffer_handle)
    (buffData,) = KYFG_StreamGetPtr(buffer_handle, buffIndex)
    data: bytes = string_at(buffData, size)
    return data


def push_frames_queue(buffer_handle: STREAM_HANDLE, state):
    if state.fast_mode:
        data = extract_buffer(buffer_handle, state.fast_params.height * state.fast_params.width)
        frame = extract_array(data, state.fast_params.height, state.fast_params.width)
    else:
        data = extract_buffer(buffer_handle, state.slow_params.height * state.slow_params.width)
        frame = extract_array(data, state.slow_params.height, state.slow_params.width)
    state.mp_cond.acquire()
    state.mp_q.put(frame)
    state.mp_cond.notify()
    state.mp_cond.release()

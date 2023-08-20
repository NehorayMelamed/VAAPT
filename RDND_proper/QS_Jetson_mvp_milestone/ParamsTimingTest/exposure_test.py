import time
import KYFGLib
from KYFGLib import *
import numpy as np

from frame_grabber import initialize_libs, print_device_data, open_grabber, connect_to_grabber, start_grabber, ImageStream, set_exposure_time


class GlobalState:
    def __init__(self, size=(1024, 2048), image_stream = None):
        self.num_frame = 0
        self.size = size
        self.image_stream = image_stream


def extract_buffer(buffer_handle: STREAM_HANDLE, size: int) -> bytes:
    (status, buffIndex) = KYFG_StreamGetFrameIndex(buffer_handle)
    (buffData,) = KYFG_StreamGetPtr(buffer_handle, buffIndex)
    data: bytes = string_at(buffData, size)
    return data


def write_frame(data, state: GlobalState):
    with open(f"resulting_frames/{state.num_frame}.raw", 'wb+') as frame_file:
        frame_file.write(data)


def save_frames_callback(buffer_handle, null):
    global state
    print(f"CALLBACK FOR FRAME {state.num_frame}")
    data = extract_buffer(buffer_handle, state.size[0]*state.size[1])
    write_frame(data, state)
    state.num_frame+=1
    if state.num_frame == 3:
        print(f"TIME IS {time.time()}")
        KYFG_SetCameraValue(state.image_stream.camera_handles[0], "MultiRoiMode", 1)
        set_exposure_time(30000.0, state.image_stream)
    if state.num_frame == 4:
        print(f"TIME IS {time.time()}")

    if state.num_frame == 10:
        print(f"time is {time.time()}")
    if state.num_frame == 11:
        print(f"time is {time.time()}")


def initialize_kaya(callback_function, num_frames: int = 0):
    """
    initialize_libs()
    print_device_data()
    connection: FGHANDLE = open_grabber(callback_function, 0)
    image_stream: ImageStream = connect_to_grabber(connection)
    #set_slow_mode(image_stream)
    time.sleep(3)
    """
    initialize_libs()
    print_device_data()

    connection: FGHANDLE = open_grabber(callback_function, 0)
    image_stream: ImageStream = connect_to_grabber(connection)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "AcquisitionFrameRate", 10.0)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "ExposureTime", 20000.0)


    time.sleep(1)
    size = [-1, -1]
    size[0] = KYFG_GetCameraValue(image_stream.camera_handles[0], "Height")[1]
    size[1] = KYFG_GetCameraValue(image_stream.camera_handles[0], "Width")[1]
    print(f"SIZE IS {size}")
    start_grabber(image_stream, num_frames)
    return image_stream, size
    # input("Press Enter to exit: ")
    #stop_grabber(image_stream)


if __name__ == '__main__':
    image_stream, size = initialize_kaya(save_frames_callback, 12)
    state = GlobalState(tuple(size), image_stream)
    while True:
        time.sleep(1)

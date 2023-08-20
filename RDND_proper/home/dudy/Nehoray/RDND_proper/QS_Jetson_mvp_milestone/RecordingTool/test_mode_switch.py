from multiprocessing import Queue, Condition, Lock

import cv2

from datetime import datetime
from frame_grabber import *
from KYFGLib import *
import numpy as np
import traceback
import yaml


class CameraManager:
    def __init__(self, slow_params, fast_params, mp_queue, mp_cond, camera_connection, image_stream):
        self.slow_params = slow_params
        self.fast_params = fast_params

        self.is_fast_mode = False
        self.camera_connection = camera_connection
        self.image_stream = image_stream

        self.mp_q = mp_queue  # multiprocessing queue for frames
        self.mp_cond = mp_cond
        self.last_grabbing_time = 0.0
        self.current_frame = 0
        self.all_frames = []

    def switch_to_fast_mode(self, x, y):
        # Get correct X,Y coordinates for ROI
        self.set_offset_x(x)
        self.set_offset_y(y)
        print(f"Switched to ROI at: ({self.fast_params.offset_x, self.fast_params.offset_y}")

        stop_grabber(self.image_stream)
        # TODO: Should this be before or after setting the parameters?
        # self.camera_connection = open_grabber(camera_callback, 0)
        # self.image_stream = connect_to_grabber(self.camera_connection)
        set_fast_mode(self.image_stream, self.fast_params)
        self.is_fast_mode = True

    def switch_to_slow_mode(self):
        stop_grabber(self.image_stream)
        # TODO: Should this be before or after setting the parameters?
        # self.camera_connection = open_grabber(camera_callback, 0)
        # self.image_stream = connect_to_grabber(self.camera_connection)
        set_slow_mode(self.image_stream, self.slow_params)
        self.is_fast_mode = False

    def set_offset_x(self, candidate_x):
        x_base = 128
        self.fast_params.offset_x = x_base * round(candidate_x / x_base)

    def set_offset_y(self, candidate_y):
        y_base = 4
        self.fast_params.offset_y = y_base * round(candidate_y / y_base)


class FastModeParams:
    def __init__(self, width, height, exposure_time, fps, offset_x, offset_y, num_frames):
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.fps = fps
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.num_frames = num_frames


class SlowModeParams:
    def __init__(self, width, height, exposure_time, fps, num_frames):
        self.width = width
        self.height = height
        self.exposure_time = exposure_time
        self.fps = fps
        self.num_frames = num_frames


def extract_array(byte_array: bytes, height: int, width: int) -> np.array:
    intermediate = np.fromstring(byte_array, dtype=np.uint8)  # fromstring copies the data
    reshaped = np.resize(intermediate, (height, width))
    return reshaped.copy()


def extract_buffer(buffer_handle: STREAM_HANDLE, size: int) -> bytes:
    (status, buffIndex) = KYFG_StreamGetFrameIndex(buffer_handle)
    (buffData,) = KYFG_StreamGetPtr(buffer_handle, buffIndex)
    data: bytes = string_at(buffData, size)
    return data


# TODO: We should maybe change this to manage a running list and push with a subprocess/thread to avoid waiting
def push_frames_queue(buffer_handle):
    global camera_manager
    if camera_manager.is_fast_mode:
        # print("Extracting with fast mode")
        data = extract_buffer(buffer_handle, camera_manager.fast_params.height * camera_manager.fast_params.width)
        frame = extract_array(data, camera_manager.fast_params.height, camera_manager.fast_params.width)
    else:
        # print("Extracting with slow mode")
        data = extract_buffer(buffer_handle, camera_manager.slow_params.height * camera_manager.slow_params.width)
        frame = extract_array(data, camera_manager.slow_params.height, camera_manager.slow_params.width)

    if camera_manager.current_frame < 10000:
        camera_manager.all_frames.append(frame)
        camera_manager.current_frame += 1

    if camera_manager.current_frame == 5000:
        print("Callback fired a signal!")
        camera_manager.mp_cond.acquire()
        # camera_manager.mp_q.put(frame)
        camera_manager.mp_cond.notify()
        camera_manager.mp_cond.release()


def camera_callback(buffer_handle, userContext):  # forwards KAYA callback, and adds parameters
    global camera_manager
    current_grabbing_time = datetime.utcnow().timestamp()
    print(f"Total time to grab frame = {current_grabbing_time - camera_manager.last_grabbing_time } seconds")
    camera_manager.last_grabbing_time = datetime.utcnow().timestamp()
    try:
        push_frames_queue(buffer_handle)
    except KeyboardInterrupt:
        print("Forward callback caught a keyboard interrupt!")
        # stop_grabber(camera_manager.image_stream)
    except Exception as e:
        # stop_grabber(camera_manager.image_stream)
        print(f"Exception!\n{e}\n{traceback.format_exc()}")


# ================================== MANAGE CONFIG ===============================================
def load_config(config_file_path) -> dict:
    assert os.path.exists(config_file_path), "CONFIG FILE NAME/LOCATION MUST NOT BE CHANGED"
    with open(config_file_path, 'r') as config_file:
        cfg = yaml.load(config_file)
    return cfg


def get_params(cfg: dict, mode: str) -> list:
    exposure_time = float(cfg[mode]["exposure_time"])
    fps = float(cfg[mode]["fps"])
    height = int(cfg[mode]["height"])
    width = int(cfg[mode]["width"])
    num_frames = int(cfg[mode]["num_frames"])
    return [width, height, exposure_time, fps, num_frames]


# ================================ INIT KAYA STUFF =================================================
def initialize_kaya(callback_function, grabber_idx):
    KYFG_Init()
    initialize_libs()
    print_device_data()
    connection: FGHANDLE = open_grabber(callback_function, grabber_idx)
    image_stream = connect_to_grabber(connection)
    return connection, image_stream


def set_slow_mode(image_stream, slow_mode_params):
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiMode", 0)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "Height", slow_mode_params.height)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "Width", slow_mode_params.width)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "ExposureTime", slow_mode_params.exposure_time)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "AcquisitionFrameRate", slow_mode_params.fps)


def set_fast_mode(image_stream, fast_mode_params):
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiMode", 1)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiIndex", 0)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiHorizontalEnableNumber", 1)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiVerticalEnableNumber", 1)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiWidth", fast_mode_params.width)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiHeight", fast_mode_params.height)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiOffsetX", fast_mode_params.offset_x)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "MultiRoiOffsetY",  fast_mode_params.offset_y)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "ExposureTime", fast_mode_params.exposure_time)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "AcquisitionFrameRate", fast_mode_params.fps)

# =============================== FINISH INIT KAYA =======================================


# =============================== GUI STUFF =============================================
def start_slow_mode_window():
    window_name = "Full Frame Viewer"
    cv2.destroyAllWindows()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)


def start_fast_mode_window():
    window_name = "ROI Viewer"
    cv2.destroyAllWindows()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)


def on_mouse_click(event, x, y, flags, param):
    # Left click
    global camera_manager
    if event == cv2.EVENT_LBUTTONDOWN and not camera_manager.is_fast_mode:
        print("left click! Changing to fast mode!!!")
        # Extract X,Y for ROI
        x = int(x * camera_manager.slow_params.width / 1280)
        y = int(y * camera_manager.slow_params.height / 720)

        # Notify main we got an image!
        click_message = {"click_x": x, "click_y": y}
        camera_manager.mp_cond.acquire()
        camera_manager.mp_q.put(click_message)
        camera_manager.mp_cond.notify()
        camera_manager.mp_cond.release()


def show_image(frame):
    global camera_manager
    shape = (1280, 720) if not camera_manager.is_fast_mode else (camera_manager.fast_params.width, camera_manager.fast_params.height)
    window_name = "Full Frame Viewer" if not camera_manager.is_fast_mode else "ROI Viewer"
    scaled_frame = cv2.resize(frame, shape)
    cv2.imshow(window_name, scaled_frame)
    cv2.waitKey(1)


# ================================== MAIN RECORDING TOOL FUNCTION ======================================
def record_grabber():
    global camera_manager
    program_start_time_str = str(int(datetime.utcnow().timestamp()))
    print("initializing")
    # INITIALIZE CAMERA MANAGER WITH EVERYTHING NEEDED FOR IT
    config_file_path = str(os.path.dirname(os.path.abspath(__file__))) + "/config.yml"
    current_recording_session = 0
    current_saved_frame = 0
    save_frame_directory = "/media/efcom/Crocodile/for_maor/11_AM"

    cfg = load_config(config_file_path)
    # save_frame_directory = cfg["file_saving_path"]
    os.makedirs(save_frame_directory, exist_ok=True)
    save_frame_directory = os.path.join(save_frame_directory, program_start_time_str)
    os.makedirs(save_frame_directory, exist_ok=False)
    frame_queue = Queue()
    frame_queue_cond = Condition(Lock())

    # SET SLOW AND FAST MODE PARAMS
    sm_width, sm_height, sm_exposure_time, sm_fps, sm_num_frames = get_params(cfg, "slow_mode")
    slow_mode_params = SlowModeParams(sm_width, sm_height, sm_exposure_time, sm_fps, sm_num_frames)

    # width, height, exposure_time, fps, num_frames = get_params(cfg, "fast_mode")
    width, height, exposure_time, fps, num_frames = 8192, 512, 1950.0, 250.0, 500
    fast_mode_params = FastModeParams(width, height, exposure_time, fps, offset_x=0, offset_y=0, num_frames=num_frames)

    connection, image_stream = initialize_kaya(camera_callback, grabber_idx=0)
    # TRANSFER ALL PARAMS AND KAYA STUFF INTO CAMERA MANAGER FOR CAMERA MODE SWITCH FUNCTIONALITY
    camera_manager = CameraManager(slow_mode_params, fast_mode_params, frame_queue, frame_queue_cond,
                                   connection, image_stream)

    # RUN KAYA IN SLOW MODE
    # set_slow_mode(camera_manager.image_stream, camera_manager.slow_params)
    # start_slow_mode_window()

    # ====================RECORDING DATA FROM REAL TOOL FAST MODE OVERRIDE ==================
    camera_manager.set_offset_x(0)
    camera_manager.set_offset_y(2800)
    set_fast_mode(camera_manager.image_stream, camera_manager.fast_params)
    camera_manager.is_fast_mode = True
    # ====================================== ================================================

    start_grabber(camera_manager.image_stream, num_frames=0)

    last_frame_time = time.time()

    # while True:
    #     # Get a frame if 1 is available, or wait for the next available frame.
    #     print("######################## Waiting for frames! ########################## ")
    #     camera_manager.mp_cond.acquire()
    #     while camera_manager.mp_q.empty():
    #         camera_manager.mp_cond.wait()
    #     next_frame = camera_manager.mp_q.get()
    #     q_size = camera_manager.mp_q.qsize()
    #     camera_manager.mp_cond.release()
    #     print("######################## Got frames! ########################## ")
    #     print(f"Total time to get frame = {time.time() - last_frame_time} and qsize = {q_size}")
    #     last_frame_time = time.time()

    print("############ MAIN WAITING FOR SIGNAL!")
    camera_manager.mp_cond.acquire()
    camera_manager.mp_cond.wait()
    camera_manager.mp_cond.release()
    stop_grabber(camera_manager.image_stream)
    print("############ MAIN GOT SIGNAL!")

    for i, image in enumerate(camera_manager.all_frames[:10000]):
        current_saved_frame = i
        # ======================== RECORDING DATA FROM REAL TIME TOOL =================================
        current_image_name = os.path.join(save_frame_directory, f"full_frame_{current_saved_frame}.raw")
        print(f"Saving image {current_image_name}")
        with open(current_image_name, 'wb') as raw_image:
            raw_image.write(image.tobytes())

    exit(0)
    # =============================================================================================
    while True:
        # Brute force way to change the mode from main via GUI click...
        if type(next_frame) == dict:
            click_x = next_frame["click_x"]
            click_y = next_frame["click_y"]
            start_fast_mode_window()
            camera_manager.switch_to_fast_mode(click_x, click_y)
            # Flushing the queue
            camera_manager.mp_cond.acquire()
            while not camera_manager.mp_q.empty():
                camera_manager.mp_q.get()
            camera_manager.mp_cond.release()
            # Restart the grabber
            start_grabber(camera_manager.image_stream, 0)

        # We got an image, show it and save it if it is in fast mode
        else:
            show_image(next_frame)
            if camera_manager.is_fast_mode:
                print("I will now save the next frame as raw bytes!")
                current_directory = os.path.join(save_frame_directory, str(current_recording_session))
                # Create the recording directory if it does not exist
                if not os.path.exists(current_directory):
                    os.makedirs(current_directory)

                # Write current image as raw bytes
                current_image_name = os.path.join(save_frame_directory, str(current_recording_session),
                                                  f"frame_{current_saved_frame}.raw")
                with open(current_image_name, 'wb') as raw_image:
                    raw_image.write(next_frame.tobytes())

                # Update recording session index and current recorded frame index
                current_saved_frame += 1
                if current_saved_frame >= camera_manager.fast_params.num_frames:
                    # Reset frame counter, advance to next session and change back to slow mode
                    current_recording_session += 1
                    current_saved_frame = 0
                    start_slow_mode_window()
                    camera_manager.switch_to_slow_mode()
                    # Flushing the queue
                    camera_manager.mp_cond.acquire()
                    while not camera_manager.mp_q.empty():
                        camera_manager.mp_q.get()
                    camera_manager.mp_cond.release()
                    # Restart the grabber
                    start_grabber(camera_manager.image_stream, 0)


if __name__ == "__main__":
    # GLOBALS
    camera_manager = None
    record_grabber()



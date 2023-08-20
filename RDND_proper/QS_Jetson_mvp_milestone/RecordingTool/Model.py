from multiprocessing import Queue
from KYFGLib import *
from KayaParams import KayaParams
from frame_grabber import start_grabber, stop_grabber, open_grabber, connect_to_grabber, ImageStream
import time


class Model:
    # contains all parameters for the application and is passed along with the GUI
    def __init__(self, slow_params: KayaParams, fast_params: KayaParams, file_saving_path: str,
                 mp_queue, mp_cond, grabber_callback):
        self.fast_mode = False
        self.test_flag = True
        self.slow_params = slow_params
        self.fast_params = fast_params
        self.file_saving_path = file_saving_path
        self.grabber_callback = grabber_callback
        self.gui = None
        self.image_stream = None  # camera stream
        self.mp_q = mp_queue  # multiprocessing queue for frames
        self.mp_cond = mp_cond

    def get_image_from_queue(self):
        self.mp_cond.acquire()
        while self.mp_q.empty():
            self.mp_cond.wait()
        image = self.mp_q.get()
        self.mp_cond.release()
        return image

    def switch_to_fast_mode(self, x, y):
        stop_grabber(self.image_stream)
        self.__set_fast_mode(x, y)
        start_grabber(self.image_stream, 0)
        print("started grabber")

    def switch_to_slow_mode(self):
        stop_grabber(self.image_stream)
        self.__set_slow_mode()
        start_grabber(self.image_stream, 0)

    def __set_fast_mode(self, x, y):
        final_x, final_y = self.__round_roi_values(x, y)
        self.fast_mode = True
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiMode", 1)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiIndex", 0)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiHorizontalEnableNumber", 1)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiVerticalEnableNumber", 1)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiWidth", self.fast_params.width)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiHeight", self.fast_params.height)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiOffsetX", final_x)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiOffsetY", final_y)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "ExposureTime", self.fast_params.exposure_time)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "AcquisitionFrameRate", self.fast_params.fps)

    def __set_slow_mode(self):
        self.fast_mode = False
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "MultiRoiMode", 0)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "Height", self.slow_params.height)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "Width", self.slow_params.width)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "ExposureTime", self.slow_params.exposure_time)
        KYFG_SetCameraValue(self.image_stream.camera_handles[0], "AcquisitionFrameRate", self.slow_params.fps)

    def __round_roi_values(self, x, y):
        x_base = 128
        y_base = 4
        return x_base * round(x / x_base), y_base * round(y / y_base)




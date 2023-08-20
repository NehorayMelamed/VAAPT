import cv2, time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from multiprocessing import Queue, Value, Lock ,Condition, Process
from datetime import datetime


class GUI:
    def __init__(self, model):
        self.model = model
        self.window_name_slow = "Full Frame Viewer"
        self.window_name_fast = "ROI Viewer"

    def start_slow_mode_window(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(self.window_name_slow)
        cv2.setMouseCallback(self.window_name_slow, self.on_mouse_click)

    def start_fast_mode_window(self):
        cv2.destroyAllWindows()
        cv2.namedWindow(self.window_name_fast)
        cv2.setMouseCallback(self.window_name_fast, self.on_mouse_click)

    def on_mouse_click(self, event, x, y, flags, param):
        # Left click
        if event == cv2.EVENT_LBUTTONDOWN and not self.model.fast_mode:
            self.model.test_flag = False
            time.sleep(1)
            print("left click! Changing to fast mode!!!")
            self.model.mp_cond.acquire()
            while not self.model.mp_q.empty():
                self.model.mp_q.get()
            self.model.mp_cond.release()
            x = int(x*self.model.slow_params.width/1280)
            y = int(y*self.model.slow_params.height/720)
            self.start_fast_mode_window()
            self.model.switch_to_fast_mode(x, y)
            print("letting grabber start, sleeping 5 seconds")
            time.sleep(5)
            print("continuing")
            self.model.test_flag = True


    def show_image(self, frame: np.array):
        shape = (1280, 720) if not self.model.fast_mode else (self.model.fast_params.width, self.model.fast_params.height)
        window_name = self.window_name_slow if not self.model.fast_mode else self.window_name_fast
        scaled_frame = cv2.resize(frame, shape)
        cv2.imshow(window_name, scaled_frame)
        cv2.waitKey(1)


import os
import sys
import time

import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from Grounded_Segment_Anything.loop_over_dir_of_images import process_images_in_dir
from cut_video_by_gui import VideoEditor
sys.path.append("/home/nehoray/PycharmProjects/Shaback/yolo_tracking/examples")
sys.path.append("Grounded_Segment_Anything")



from examples.my_traker_2 import main_manger_create_crops_car_via_yolo

BASE_DIRECTORY_FOR_VIDEO_PATH_UPLOADING = "data/videos"

CROP_DIRECTORY ="crop"
SEGMENTATION_DIRECTORY = "segmentation"


Video_Stabilization = "Video_Stabilization"
Video_Enhancer = "Video_Enhancer"
Video_Processing = "Video_Processing"
Video_Compression = "Video_Compression"

class SegmentationWindow:
    def __init__(self, master, logger, video_path, output_directory):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.video_path = video_path
        self.output_directory = output_directory
        # Text label for detection method selection
        self.detection_label = tk.Label(self.window, text="Choose detection method:")
        self.detection_label.pack()

        # Detection options dropdown
        self.detection_options = ["Optical Flow", "Segmentation"]
        self.detection_combobox = ttk.Combobox(self.window, values=self.detection_options)
        self.detection_combobox.pack()

        # Text label for frame stride input
        self.stride_label = tk.Label(self.window, text="Enter frame stride:")
        self.stride_label.pack()

        # Frame stride entry box
        self.stride_entry = tk.Entry(self.window)
        self.stride_entry.pack()

        # Apply button
        self.apply_button = tk.Button(self.window, text="Apply", command=self.apply)
        self.apply_button.pack()

        self.logger = logger

    def build_data_manager(self, detection_option, stride_value):
        self.logger.log(f"Selected detection option: {detection_option}, Stride value: {stride_value}")
        self.logger.log("Performing data building")
        if detection_option == self.detection_options[0]:
            self.optical_flow_manager(stride_value)
        elif detection_option == self.detection_options[1]:
            self.segmentation_manager(stride_value)

    def segmentation_manager(self, stride_value):
        crop_directory = f"{self.output_directory}/{CROP_DIRECTORY}"
        ### create crop image
        print(crop_directory)
        main_manger_create_crops_car_via_yolo(video_path=self.video_path, save_dir_path=crop_directory, logger_object=self.logger)
        self.logger.log(f"Crop data were built into = {crop_directory}")
        self.window.destroy()

        ### perform segmentation

        # usage
        grounded_segment_anything_args = {
            'desired_size': (512, 512),
            'sam_checkpoint': "Grounded_Segment_Anything/sam_vit_h_4b8939.pth",
            'grounded_checkpoint': "Grounded_Segment_Anything/groundingdino_swint_ogc.pth",
            'config_file': "Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            'sam_hq_checkpoint': None,
            'use_sam_hq': False,
            'box_threshold': 0.3,
            'text_threshold': 0.25,
            'device': "cpu"
        }

        segmentation_output_directory = f"{self.output_directory}/{SEGMENTATION_DIRECTORY}"
        self.logger.info(f"About to perform segmentation on the images in directory - {crop_directory}")
        self.logger.info(f"Data will be saved into - {crop_directory}")
        process_images_in_dir(crop_directory, "car", segmentation_output_directory, grounded_segment_anything_args)
        self.logger.info(f"Segmentation process successful, data was saved into - {segmentation_output_directory}")

    def optical_flow_manager(self, stride_value):
        pass

    def apply(self):
        # Here you can retrieve and use the selected detection option and stride value
        detection_option = self.detection_combobox.get()
        stride_value = int(self.stride_entry.get())
        self.window.after(0, self.build_data_manager, detection_option, stride_value)
        # self.window.destroy()


class AppDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing App")
        self.video_path = None
        self.root.geometry("1000x600")  # Adjust window size here
        self.output_dir = "output"  # Output directory name

        # Browse video button
        self.browse_video_button = tk.Button(root, text="Browse video", command=self.browse_video)
        self.browse_video_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Browse directory button
        self.browse_directory_button = tk.Button(root, text="Browse directory", command=self.browse_directory)
        self.browse_directory_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Video Stabilization section
        self.title_stabilization = tk.Label(root, text=Video_Stabilization)
        self.title_stabilization.grid(row=0, column=1, padx=10, pady=10)
        self.stabilization_methods = ["ECC_(Dudy)", "ECC_(Yoav)"]
        self.stabilization_combobox = ttk.Combobox(root, values=self.stabilization_methods)
        self.stabilization_combobox.grid(row=1, column=1, padx=10, pady=10)
        self.apply_stabilization_button = tk.Button(root, text="Apply", command=self.pre_display_stabilized_video)
        self.apply_stabilization_button.grid(row=2, column=1, padx=10, pady=10)

        # Video Enhancer section
        self.title_enhancer = tk.Label(root, text=Video_Enhancer)
        self.title_enhancer.grid(row=0, column=2, padx=10, pady=10)
        self.enhancement_methods = ["ECC_base_segmentation_mask", "De_noise", "De_blur", "De_jpeg", "De_focuse" "Optical_flow_avaraging_(image)", "Optical_flow_avaraging_(video)"]
        self.enhancement_combobox = ttk.Combobox(root, values=self.enhancement_methods)
        self.enhancement_combobox.grid(row=1, column=2, padx=10, pady=10)
        self.apply_enhancement_button = tk.Button(root, text="Apply", command=self.pre_display_enhanced_video)
        self.apply_enhancement_button.grid(row=2, column=2, padx=10, pady=10)

        # Video Processing section
        self.title_processing = tk.Label(root, text=Video_Processing)
        self.title_processing.grid(row=0, column=3, padx=10, pady=10)
        self.processing_methods = ["Blur_kernel", "segmentation", "detection", "optical_flow", "frame_interpolation" "question_and_answer", "Blur_base_OF"]
        self.processing_combobox = ttk.Combobox(root, values=self.processing_methods)
        self.processing_combobox.grid(row=1, column=3, padx=10, pady=10)
        self.apply_processing_button = tk.Button(root, text="Apply", command=self.process_video)
        self.apply_processing_button.grid(row=2, column=3, padx=10, pady=10)

        # Video Compression section
        self.title_compression = tk.Label(root, text=Video_Compression)
        self.title_compression.grid(row=0, column=4, padx=10, pady=10)
        self.compression_methods = ["h264"]
        self.compression_combobox = ttk.Combobox(root, values=self.compression_methods)
        self.compression_combobox.grid(row=1, column=4, padx=10, pady=10)
        self.apply_compression_button = tk.Button(root, text="Apply", command=self.compress_video)
        self.apply_compression_button.grid(row=2, column=4, padx=10, pady=10)

        # Log box
        self.log_box = tk.Text(root)
        self.log_box.grid(row=3, column=0, columnspan=5, padx=10, pady=10)

    def info(self, message):
        self.log(f"Info | {message}")

    def log(self, message):
        self.log_box.insert(tk.END, message + '\n')  # append the message to the log box

    def browse_directory(self):
        default_directory = BASE_DIRECTORY_FOR_VIDEO_PATH_UPLOADING  # Set your default path here
        self.directory_path = filedialog.askdirectory(initialdir=default_directory, title='Select directory')
        self.log(f"Directory: {self.directory_path}")


    def browse_video(self):
        default_directory = BASE_DIRECTORY_FOR_VIDEO_PATH_UPLOADING  # Set your default path here
        self.video_path = filedialog.askopenfilename(initialdir=default_directory,
                                                     filetypes=[("Video files", "*.mp4 *.avi")],
                                                     title='Select video file')
        self.log(f"Video file: {self.video_path}")

        # Get the base name and extension of the input video
        video_base_name = os.path.basename(self.video_path)
        video_base_name_without_ext = os.path.splitext(video_base_name)[0]

        # Define the output directory and path
        output_directory = self.output_dir  # Use the selected output directory
        output_path = os.path.join(output_directory, video_base_name_without_ext + "_processed.mp4")

        # Instantiate VideoEditor after video file selection
        self.video_editor = VideoEditor(self.video_path, output_path, UiManager=self)
        self.video_editor.process_video()  # or wherever you need to call process_video



    def save_video(self, video_capture, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_size = (
        int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        self.log("Video processing finished. Saved to " + output_path)

    def stabilize_video(self, video_capture, output_path):
        # You should insert your implementation for the stabilization here

        self.save_video(video_capture, output_path)

    def enhance_video(self, video_capture, output_path):
        # You should insert your implementation for the enhancement here
        self.save_video(video_capture, output_path)

    def process_video(self):
        processing_method = self.processing_combobox.get()
        self.log(f"Applying video processing method: {processing_method}")
        # Add your implementation here

    def compress_video(self):
        compression_method = self.compression_combobox.get()
        self.log(f"Applying video compression method: {compression_method}")
        # Add your implementation here

    def pre_display_stabilized_video(self):
        self.log("Stabilizing video... This might take a while.")
        time.sleep(0.2)
        self.display_stabilized_video()

    def display_stabilized_video(self):
        if self.video_path is None:
            self.log("Please select a video file first")
            return

        stabilization_method = self.stabilization_combobox.get()
        output_dir = os.path.join(self.output_dir, os.path.basename(self.video_path).split('.')[
            0] + '__stabilization_' + stabilization_method)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'video.mp4')

        video_capture = cv2.VideoCapture(self.video_path)
        self.stabilize_video(video_capture, output_path)
        video_capture.release()

        # Instantiate VideoEditor for stabilized video
        self.video_editor = VideoEditor(self.video_path, output_path, UiManager=self)
        self.video_editor.process_video()  # or wherever you need to call process_video

    def pre_display_enhanced_video(self):
        if not self.enhancement_combobox.get():
            self.log("Please select an enhancement method from the dropdown list.")
            return

        self.log("Enhancing video... This might take a while.")
        time.sleep(0.2)

        if self.enhancement_combobox.get() == "ECC_base_segmentation_mask":
            video_path = self.video_path
            output_directory = os.path.join(self.output_dir, os.path.basename(self.video_path).split('.')[0])
            SegmentationWindow(self.root, self, video_path, output_directory)

        self.display_enhanced_video()

    def display_enhanced_video(self):
        if self.video_path is None:
            self.log("Please select a video file first")
            return
        enhancement_method = self.enhancement_combobox.get()
        output_dir = os.path.join(self.output_dir, os.path.basename(self.video_path).split('.')[
            0] + '__enhancement_' + enhancement_method)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'video.mp4')

        video_capture = cv2.VideoCapture(self.video_path)
        self.enhance_video(video_capture, output_path)
        video_capture.release()


root = tk.Tk()
demo = AppDemo(root)
root.mainloop()

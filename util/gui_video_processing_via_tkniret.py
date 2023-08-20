import os
import time

import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from cut_video_by_gui import VideoEditor

class AppDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing App")
        self.directory_path = None
        self.video_path = None
        self.root.geometry("800x400")  # Adjust window size here

        # Browse base directory button
        self.browse_directory_button = tk.Button(root, text="Browse base directory", command=self.browse_directory)
        self.browse_directory_button.pack()

        # Browse video button
        self.browse_video_button = tk.Button(root, text="Browse video", command=self.browse_video)
        self.browse_video_button.pack()

        # Video stabilization drop-down list
        self.stabilization_methods = ["ECC", "Cross_Correlation", "ECC_entire_image"]
        self.stabilization_combobox = ttk.Combobox(root, values=self.stabilization_methods)
        self.stabilization_combobox.pack()

        # Display stabilized video button
        self.display_stabilized_button = tk.Button(root, text="Display Stabilized Video", command=self.pre_display_stabilized_video)
        self.display_stabilized_button.pack()

        # Video enhancement drop-down list
        self.enhancement_methods = ["ECC_base_optical_flow", "ECC_base_segmentation_mask", "Denoise", "Deblur", "De-jpeg", "Dfocuse"]
        self.enhancement_combobox = ttk.Combobox(root, values=self.enhancement_methods)
        self.enhancement_combobox.pack()

        # Display enhanced video button
        self.display_enhanced_button = tk.Button(root, text="Display Enhanced Video", command=self.pre_display_enhanced_video)
        self.display_enhanced_button.pack()

        # Log box
        self.log_box = tk.Text(root)
        self.log_box.pack()

    def log(self, message):
        self.log_box.insert(tk.END, message + '\n')  # append the message to the log box

    def browse_directory(self):
        self.directory_path = filedialog.askdirectory(title='Select base directory')
        self.log(f"Base directory: {self.directory_path}")

    def browse_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")], title='Select video file')
        self.log(f"Video file: {self.video_path}")

        # Get the base name and extension of the input video
        video_base_name = os.path.basename(self.video_path)
        video_base_name_without_ext = os.path.splitext(video_base_name)[0]

        # Define the output directory and path
        output_directory = self.directory_path  # Use the selected output directory
        output_path = os.path.join(output_directory, video_base_name_without_ext + "_processed.mp4")

        # Instantiate VideoEditor after video file selection
        self.video_editor = VideoEditor(self.video_path, output_path, UiManager=self)
        self.video_editor.process_video()  # or wherever you need to call process_video

    def save_video(self, video_capture, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
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

    def pre_display_stabilized_video(self):
        self.log("Stabilizing video... This might take a while.")
        time.sleep(0.2)
        self.display_stabilized_video()

    def display_stabilized_video(self):
        if self.video_path is None:
            self.log("Please select a video file first")
            return

        stabilization_method = self.stabilization_combobox.get()
        output_path = os.path.join(self.directory_path, os.path.basename(self.video_path).split('.')[0] + '__stabilization_' + stabilization_method + '.mp4')

        video_capture = cv2.VideoCapture(self.video_path)
        self.stabilize_video(video_capture, output_path)
        video_capture.release()

        # Instantiate VideoEditor for stabilized video
        self.video_editor = VideoEditor(self.video_path, output_path,UiManager=self)
        self.video_editor.process_video()  # or wherever you need to call process_video


    def pre_display_enhanced_video(self):
        self.log("Enhancing video... This might take a while.")
        time.sleep(0.2)
        self.display_enhanced_video()

    def display_enhanced_video(self):
        if self.video_path is None:
            self.log("Please select a video file first")
            return
        enhancement_method = self.enhancement_combobox.get()
        output_path = os.path.join(self.directory_path, os.path.basename(self.video_path).split('.')[0] + '__enhancement_' + enhancement_method + '.mp4')

        video_capture = cv2.VideoCapture(self.video_path)
        self.enhance_video(video_capture, output_path)
        video_capture.release()




root = tk.Tk()
demo = AppDemo(root)
root.mainloop()

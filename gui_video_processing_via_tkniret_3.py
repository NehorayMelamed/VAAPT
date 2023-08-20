import multiprocessing
import os
import sys
import threading
import time
import webbrowser

import cv2
from tkinter import ttk

import torch

import PARAMETER
from remove_object.inpaint_main_modify_using_model import remove_object_from_video_base_model
from remove_object.remove_with_roi import main_remove_by_ROI
from yolov8_tracking.Web.Gui_SHABACK.app import run_flask_web_page



ERROR_INVALID_SELECTION = 3

### Denoise and tracking optical flow yoav
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/pips_main")

## frame interpolation
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/EMA")
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}/EMA/ckpt")

sys.path.append(f"{PARAMETER.BASE_PROJECT}/yolo_tracking/examples")  # ToDO
sys.path.append(f"{PARAMETER.RDND_BASE_PATH}")
sys.path.append(
    f"{PARAMETER.RDND_BASE_PATH}/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation/utils_NUBKE")
sys.path.append(
    f"{PARAMETER.RDND_BASE_PATH}/models/NonUniformBlurKernelEstimation/NonUniformBlurKernelEstimation")
sys.path.append("Grounded_Segment_Anything")

### deblur
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/to_neo")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Omer/RVRT_deblur_inference.py")


### quesations and answering
sys.path.append(f"{PARAMETER.BASE_PROJECT}/question_AND_answering_new")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/question_AND_answering_new/haggingfacedir")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/question_AND_answering_new/Blip_implementation")






#### Yoav stabilization
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/yoav_stabilization")



from Yoav_denoise_new.yoav_stabilization.main import stabilize_ccc_video_no_avg, prepare_input_source, \
    stabilize_ccc_video, stabilize_ccc_image

from Yoav_denoise_new.pips_main.chain_demo import chain_demo_main_interface
from smart_compression.main import VideoProcessor
import question_AND_answering_new.haggingfacedir.main as hagging_face
import question_AND_answering_new.Blip_implementation.main as blip
from question_AND_answering_new.data.people.questions import PersonQuestion
from question_AND_answering_new.main_manager import question_and_answering_loop_directory, \
    question_and_answering_single_image



from Omer.to_neo.test_deblur import main_deblur
from Yoav_denoise_new.denois.main import main_denoise
from cut_video_by_gui import VideoEditor

# import yolo_tracking.examples.my_traker_2 as yolo_detector
from windows_selections import *
from Grounded_Segment_Anything.loop_over_dir_of_images import process_images_in_dir
from PARAMETER import *
from RDND_proper.EMA.demo_2x import generate_intermediate_frame
import Yoav_denoise_new.yoav_blur_kernel.Deblur_Pipeline_1_NUBKE_DWD as deblur_interface

BASE_DIRECTORY_DATA_UPLOADING = "data"
BASE_DIRECTORY_FOR_VIDEO_PATH_UPLOADING = os.path.join(BASE_DIRECTORY_DATA_UPLOADING, "videos")
BASE_DIRECTORY_FOR_IMAGES_PATH_UPLOADING = os.path.join(BASE_DIRECTORY_DATA_UPLOADING, "images")
CROP_DIRECTORY = "crop"
SEGMENTATION_DIRECTORY = "segmentation"
OPTICAL_FLOW_TRACKING = "optical_flow_tracking"
Video_Stabilization = "Video_Stabilization"
Video_Enhancer = "Video_Enhancer"
Video_Processing = "Video_Processing"
Video_Compression = "Video_Compression"
OUTPUT_DIRECTORY = "output"
detection = "detection"
GROUNDED_SEGMENTATION = "GROUNDED_SEGMENTATION"
STABILIZATION_YOAV = "STABILIZATION_YOAV"
REMOVE_OBJECT_CLASSIC = "REMOVE_OBJECT_CLASSIC"
REMOVE_OBJECT_MODEL = "REMOVE_OBJECT_MODEL"
BLUR_KERNEL = "BLUR_KERNEL"
ROOT_OUTPUT_DIR_PATH = os.getcwd()
FULL_PATH_OUTPUT = os.path.join(ROOT_OUTPUT_DIR_PATH, OUTPUT_DIRECTORY)
web_server_is_working = False

use_cuda_global = 1 # True

def exit_on(err):
    exit(err)

if os.path.exists(FULL_PATH_OUTPUT) is False:
    os.mkdir(FULL_PATH_OUTPUT)


def to_use_cuda():
    return True if use_cuda_global == 1 else False


def last_three_directories(path):
    # This will split the path into its individual components
    parts = path.split(os.sep)
    # This will filter out any empty components (e.g., leading or trailing slashes)
    parts = [part for part in parts if part]
    # This will return the last three components
    return parts[-3:]


def build_directory_to_service(name_of_service, directory_name):
    if os.path.exists(FULL_PATH_OUTPUT) is False:
        os.mkdir(FULL_PATH_OUTPUT)
    path_to_output_directory_service = os.path.join(FULL_PATH_OUTPUT, name_of_service)
    if os.path.exists(path_to_output_directory_service) is False:
        os.mkdir(path_to_output_directory_service)
    path_to_output_directory_service_with_directory_name = os.path.join(path_to_output_directory_service, directory_name)
    if os.path.exists(path_to_output_directory_service_with_directory_name) is False:
        os.mkdir(path_to_output_directory_service_with_directory_name)

    return path_to_output_directory_service_with_directory_name


torch.cuda.empty_cache()




class RemoveObjectWindow:
    """Remove Object Window"""
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Remove Object from Video")
        self.logger = logger

        # Description
        self.description_label = tk.Label(self.window, text="Here we provide functionality to remove some objects during video. There are two options:\n1. Using a model\n2. Using a classic ROI")
        self.description_label.pack(pady=10)

        # Button to browse for video file
        self.browse_button = tk.Button(self.window, text="Upload Video", command=self.upload_video)
        self.browse_button.pack(pady=10)

        # Dropdown for selecting removal method
        self.removal_methods = ["Classic ROI", "Model-based Removal"]
        self.method_var = tk.StringVar(self.window)
        self.method_var.set(self.removal_methods[0])  # default value
        self.method_dropdown = ttk.Combobox(self.window, textvariable=self.method_var, values=self.removal_methods)
        self.method_dropdown.pack(pady=10)

        # Button to perform object removal
        self.remove_button = tk.Button(self.window, text="Perform Removal", command=self.perform_removal)
        self.remove_button.pack(pady=10)

        # Variable to store the video path
        self.video_path = ""

    def upload_video(self):
        file_types = [("MP4 Video files", "*.mp4"), ("AVI Video files", "*.avi"), ("All files", "*.*")]
        path = filedialog.askopenfilename(initialdir=BASE_DIRECTORY_DATA_UPLOADING, title="Select video file", filetypes=file_types)
        if path:
            self.video_path = path

    def perform_removal(self):
        if self.video_path:
            selected_method = self.method_var.get()
            self.window.destroy()
            directory_name = os.path.basename(self.video_path).split('.')[0]
            # Depending on the selected method, you'd integrate the logic for object removal here.
            if selected_method == "Classic ROI":
                output_path = build_directory_to_service(REMOVE_OBJECT_CLASSIC, directory_name)
                main_remove_by_ROI(self.video_path, output_path)
            elif selected_method == "Model-based Removal":
                output_path = build_directory_to_service(REMOVE_OBJECT_MODEL, directory_name)
                use_cuda = to_use_cuda()
                remove_object_from_video_base_model(self.video_path, output_path, use_cuda=use_cuda)
            else:
                self.logger.info("Nothing to do, invalid option selection")
                return

            self.logger.info(f"Object removal process finished using method: {selected_method}")
            self.logger.info(f"Data processed from {self.video_path}")
        else:
            self.logger.info("Got an issue while trying to upload the video")


class TrackerOpticalFlowWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Optical Flow Tracking")
        self.logger = logger
        self.directorypath_var = tk.StringVar()

        # Description label
        self.description_label = tk.Label(self.window,
                                          text="Optical Flow Tracker: Select a directory and perform tracking.")
        self.description_label.pack()

        # Button to Browse directory
        self.browse_button = tk.Button(self.window, text="Browse Directory", command=self.browse_directory)
        self.browse_button.pack()

        # Perform button
        self.perform_button = tk.Button(self.window, text="Perform", command=self.perform_tracking)
        self.perform_button.pack()

    def browse_directory(self):
        directorypath = filedialog.askdirectory(initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        self.directorypath_var.set(directorypath)

    def perform_tracking(self):
        directory_path_to_images = self.directorypath_var.get()
        # Insert your code for the TrackerOpticalFlow operation here using directory_path.
        print(f"Performing optical flow tracking on videos in: {directory_path_to_images}")

        if directory_path_to_images:
            directory_name = os.path.basename(directory_path_to_images).split('.')[0]
            path_to_directory_output = build_directory_to_service(name_of_service=OPTICAL_FLOW_TRACKING, directory_name=directory_name)
            if chain_demo_main_interface(base_directory_of_images=directory_path_to_images, base_output=path_to_directory_output) is True:

                self.logger.log("Segmentation process finished with success")
                self.logger.log(f"Data saved into {path_to_directory_output}")

        else:
            self.logger.log("Got an issue while trying to upload directory path")


        self.window.destroy()


class StabilizationWindowAveraging:
    """Yoav stabilization Average"""
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Stabilization")
        self.logger = logger

        # Description
        self.description_label = tk.Label(self.window, text="Please select a stabilization averaging method and upload a directory of images or a video file.")
        self.description_label.pack()

        # Dropdown for selecting stabilization method
        self.stabilization_methods = ["stabilize to video", "stabilize to video with averaging", "stabilize to image"]
        self.method_var = tk.StringVar(self.window)
        self.method_var.set(self.stabilization_methods[0])  # default value
        self.method_dropdown = ttk.Combobox(self.window, textvariable=self.method_var, values=self.stabilization_methods)
        self.method_dropdown.pack()

        # Button to browse for file or directory
        self.browse_button = tk.Button(self.window, text="Upload", command=self.upload_file_or_directory)
        self.browse_button.pack()

        # Button to perform stabilization
        self.stabilize_button = tk.Button(self.window, text="Perform Stabilization", command=self.perform_stabilization)
        self.stabilize_button.pack()

        # Variable to store the file or directory path
        self.path = ""
        self.is_directory = False

    def upload_file_or_directory(self):
        selected_method = self.method_var.get()

        if selected_method == "stabilize image sequence":
            path = filedialog.askdirectory(title="Select directory of images")
            if path:
                self.path = path
                self.is_directory = True
        else:
            file_types = [("MP4 Video files", "*.mp4"), ("AVI Video files", "*.avi"), ("All files", "*.*")]
            path = filedialog.askopenfilename(initialdir=BASE_DIRECTORY_DATA_UPLOADING,
                                              title="Select video file", filetypes=file_types)
            if path:
                self.path = path
                self.is_directory = False

    def perform_stabilization(self):
        if self.path:
            selected_method = self.method_var.get()
            self.window.destroy()
            directory_name = os.path.basename(self.path).split('.')[0]
            full_path_to_output = build_directory_to_service(STABILIZATION_YOAV, directory_name)
            input_source = prepare_input_source(self.path)
            if selected_method == "stabilize to video":
                full_path_to_output_specific = os.path.join(full_path_to_output, "video_stabilize.mp4")
                stabilize_ccc_video_no_avg(folder_path=input_source, video_path_to_save=full_path_to_output_specific)
            elif selected_method == "stabilize to video with averaging":
                full_path_to_output_specific = os.path.join(full_path_to_output, "video_stabilize_averaging.mp4")
                stabilize_ccc_video(folder_path=input_source, video_path_to_save=full_path_to_output_specific)
            elif selected_method == "stabilize to image":
                stabilize_ccc_image(folder_path=input_source, output_dir_path=full_path_to_output)
            else:
                self.logger.info("Nothing to do, invalid option selection")
                return

            self.logger.log(f"Stabilization process finished using method: {selected_method}")
            self.logger.log(f"Data processed from {self.path}")

        else:
            self.logger.log("Got an issue while trying to upload the file or directory")


# Remember, you'll need to add the stabilization logic in the `perform_stabilization` method based on your requirements.


class DetectionWindow:
    def __init__(self, master):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Object Detection")
        self.selections = None
        self.filepath_var = tk.StringVar()
        self.follow_one_object_var = tk.BooleanVar()
        self.display_live_var = tk.IntVar()  # For RadioButton
        self.detected_classes_var = tk.StringVar()
        self.license_plate_var = tk.StringVar()

        # Button to Load file
        self.load_button = tk.Button(self.window, text="Load", command=self.load_file)
        self.load_button.pack()

        # Checkboxes
        self.follow_one_object_checkbox = tk.Checkbutton(self.window, text="Follow only one object",
                                                         variable=self.follow_one_object_var)
        self.follow_one_object_checkbox.pack()

        # Title for classes
        self.classes_label = tk.Label(self.window, text="Object Classes:")
        self.classes_label.pack()

        # Multiple selection Listbox for classes
        self.listbox = tk.Listbox(self.window, selectmode=tk.MULTIPLE)
        options = [0, 1, 2, 3, 4, 5]
        for item in options:
            self.listbox.insert(tk.END, item)
        self.listbox.pack()

        # Radio buttons for real-time display
        self.show_real_time = tk.Radiobutton(self.window, text="Show Real Time", variable=self.display_live_var,
                                             value=1)
        self.show_real_time.pack()
        self.do_not_show_real_time = tk.Radiobutton(self.window, text="Do Not Show Real Time",
                                                    variable=self.display_live_var, value=0)
        self.do_not_show_real_time.pack()

        # Apply button
        self.apply_button = tk.Button(self.window, text="Apply", command=self.apply)
        self.apply_button.pack()

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")],
                                              initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        self.filepath_var.set(filepath)

    def get_selections(self):
        return {
            'video_path': self.filepath_var.get(),
            'save_specific_object': self.follow_one_object_var.get(),
            'show': self.display_live_var.get(),
            'classes': [self.listbox.get(idx) for idx in self.listbox.curselection()],
            'license_plate': self.license_plate_var.get()
        }

    def detection(self):
        selections = self.selections
        video_path = selections["video_path"]
        classes = selections["classes"]
        show = selections["show"]
        save_specific_object = selections["save_specific_object"]
        output_directory_name = os.path.basename(video_path).split('.')[0]
        detection_path = os.path.join(FULL_PATH_OUTPUT, "DETECTION")
        print(selections)
        print(1)
        yolo_detector.main_interface(video_path=video_path, classes=classes, show=show,
                                     save_dir_path=detection_path,
                                     save_specific_object=save_specific_object,
                                     output_directory_name=output_directory_name)
        self.window.destroy()

    def apply(self):
        self.selections = self.get_selections()
        self.detection()


class SegmentationWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Segmentation")
        self.logger = logger
        # Description
        self.description_label = tk.Label(self.window, text="Please type in the Entry box a text \n"
                                                            "for the objects u wish to segment, for example\n"
                                                            "car. person\n"
                                                            "please not to type \".\" and space between")
        self.description_label.pack()

        # Entry box for free text input
        self.text_entry = tk.Entry(self.window)
        self.text_entry.pack()

        # Boolean radio button selection
        self.segment_anything_var = tk.BooleanVar(value=True)  # Default selection is "Segment Anything"
        self.segment_anything_radio = tk.Radiobutton(self.window, text="Segment Anything",
                                                     variable=self.segment_anything_var, value=True)
        self.segment_anything_radio.pack()

        self.segment_biggest_mask_var = tk.BooleanVar(value=False)
        self.segment_biggest_mask_radio = tk.Radiobutton(self.window, text="Segment Biggest Mask",
                                                         variable=self.segment_anything_var, value=False)
        self.segment_biggest_mask_radio.pack()

        # Button to browse for directory path
        self.browse_button = tk.Button(self.window, text="Browse Directory", command=self.browse_directory)
        self.browse_button.pack()

        # Button to perform segmentation
        self.segment_button = tk.Button(self.window, text="Perform Segmentation", command=self.perform_segmentation)
        self.segment_button.pack()

        # Variable to store the directory path
        self.directory_path = ""

    def browse_directory(self):
        directory_path = filedialog.askdirectory(initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        if directory_path:
            self.directory_path = directory_path

    def perform_segmentation(self):
        if self.directory_path:
            text_input = self.text_entry.get()
            grounded_segment_anything_args = {
                'desired_size': (512, 512),
                'sam_checkpoint': path_sam_vit_h_4b8939,
                'grounded_checkpoint': path_groundingdino_swint_ogc_PTH,
                'config_file': path_GroundingDINO_SwinT_OGC_PY,
                'sam_hq_checkpoint': None,
                'use_sam_hq': False,
                'box_threshold': 0.3,
                'text_threshold': 0.25,
                'device': "cuda" if to_use_cuda() is True else "cpu"
            }
            directory_name = os.path.basename(self.directory_path).split('.')[0]
            name_of_service = GROUNDED_SEGMENTATION
            directory_as_directory_path_output = build_directory_to_service(name_of_service, directory_name)
            # Get the selected segmentation method
            segmentation_method = not self.segment_anything_var.get()
            self.window.destroy()
            if process_images_in_dir(input_dir=self.directory_path, text_prompt=text_input,
                                     grounded_segment_anything_args=grounded_segment_anything_args,
                                     base_output_dir=directory_as_directory_path_output,
                                     use_biggest_mask=segmentation_method, logger=self.logger) is True:
                self.logger.log("Segmentation process finished with success")
                self.logger.log(f"Data saved into {directory_as_directory_path_output}")

        else:
            self.logger.log("Got an issue while trying to upload directory path")


class BlurKernelWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Blur Kernel")
        self.logger = logger
        self.filepath_var = tk.StringVar()

        # Description
        self.description_label = tk.Label(self.window, text="Description: Please upload a picture to get "
                                                            "its BLUR KERNEL and other data")
        self.description_label.pack()

        # Button to Load file
        self.load_button = tk.Button(self.window, text="Load Image", command=self.load_file)
        self.load_button.pack()

    def get_blur_kernel(self, image_file_path):
        file_name = os.path.basename(image_file_path).split('.')[0]
        full_path_directory = build_directory_to_service(BLUR_KERNEL, file_name)
        ans = deblur_interface.main_interface(image_path=image_file_path, output_dir_for_global=full_path_directory)
        if ans is True:
            self.logger.log("Blur kernel extraction pass with success")
            self.logger.log(f"Data saved into - {full_path_directory}")
        self.window.destroy()

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")],
                                              initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        self.filepath_var.set(filepath)
        self.get_blur_kernel(filepath)


class FrameInterpolationWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Frame Interpolation")
        self.logger = logger
        self.filepath1_var = tk.StringVar()
        self.filepath2_var = tk.StringVar()

        # Description
        self.description_label = tk.Label(self.window,
                                          text="Description: Load two images and apply frame interpolation to generate an intermediate image.")
        self.description_label.pack()

        # Buttons to Load Images
        self.load_img1_button = tk.Button(self.window, text="Load Image 1", command=self.load_image1)
        self.load_img1_button.pack()

        self.load_img2_button = tk.Button(self.window, text="Load Image 2", command=self.load_image2)
        self.load_img2_button.pack()

        self.apply_button = tk.Button(self.window, text="Apply Frame Interpolation", command=self.apply_interpolation)
        self.apply_button.pack()

    def load_image1(self):
        filepath1 = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        self.filepath1_var.set(filepath1)
        self.logger.log(f"Image 1 loaded from {filepath1}")

    def load_image2(self):
        filepath2 = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        self.filepath2_var.set(filepath2)
        self.logger.log(f"Image 2 loaded from {filepath2}")

    def preform_frame_intermediate(self):
        self.logger.log("About to perform frame interpolation")
        image1_name = os.path.basename(self.filepath1_var.get()).split('.')[0]
        image2_name = os.path.basename(self.filepath2_var.get()).split('.')[0]

        frame_inter_output_dir_Path = os.path.join(FULL_PATH_OUTPUT, "frame_interpolation")
        if os.path.exists(frame_inter_output_dir_Path) is False:
            os.mkdir(frame_inter_output_dir_Path)

        directory_output_name = f"{image1_name}_and_{image2_name}"
        directory_output_path = os.path.join(frame_inter_output_dir_Path, directory_output_name)
        if os.path.exists(directory_output_path) is False:
            os.mkdir(directory_output_path)
        self.window.destroy()

        if generate_intermediate_frame(img_path1=self.filepath1_var.get(), img_path2=self.filepath2_var.get(),
                                       output_dir=directory_output_path) is True:
            self.logger.log(f"Frame interpolation applied and result saved in {directory_output_path}")
        self.window.destroy()

    def apply_interpolation(self):
        self.preform_frame_intermediate()


class DenoiseWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Denoise")
        self.logger = logger

        # Description
        self.description_label = tk.Label(self.window,
                                          text="This is a denoise section for removing noise from the video")
        self.description_label.pack()

        # Button to browse for video file
        self.browse_video_button = tk.Button(self.window, text="Browse Video", command=self.browse_video)
        self.browse_video_button.pack()

        # Radio button for selecting ROI or entire video
        self.use_roi_var = tk.BooleanVar(value=False)  # Default selection is not using ROI
        self.roi_radio = tk.Radiobutton(self.window, text="Use ROI", variable=self.use_roi_var, value=True)
        self.roi_radio.pack()

        self.entire_video_radio = tk.Radiobutton(self.window, text="Use Entire Video", variable=self.use_roi_var,
                                                 value=False)
        self.entire_video_radio.pack()

        # Labels and entry boxes for window_size_temporal and stride parameters
        self.window_size_label = tk.Label(self.window, text="window_size_temporal (Default: 4):")
        self.window_size_label.pack()
        self.window_size_temporal_entry = tk.Entry(self.window)
        self.window_size_temporal_entry.insert(tk.END, "4")  # Default value for window_size_temporal
        self.window_size_temporal_entry.pack()

        self.stride_label = tk.Label(self.window, text="stride (Default: 1):")
        self.stride_label.pack()
        self.stride_entry = tk.Entry(self.window)
        self.stride_entry.insert(tk.END, "1")  # Default value for stride
        self.stride_entry.pack()

        # Button to perform denoising
        self.apply_button = tk.Button(self.window, text="Apply", command=self.perform_denoising)
        self.apply_button.pack()

        # Variable to store video path
        self.video_path = ""

    def browse_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")], initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        if video_path:
            self.video_path = video_path

    def perform_denoising(self):
        if self.video_path:
            use_roi = self.use_roi_var.get()

            # Get window_size_temporal and stride values
            window_size_temporal = int(self.window_size_temporal_entry.get())
            stride = int(self.stride_entry.get())

            base_video_name = os.path.basename(self.video_path).split('.')[0]
            denoise_output_directory = os.path.join(FULL_PATH_OUTPUT, "DENOISE")
            if os.path.exists(denoise_output_directory) is False:
                os.mkdir(denoise_output_directory)
            full_output_video_path = os.path.join(denoise_output_directory, f"denoise_{base_video_name}.mp4")
            full_input_for_output_video_path = os.path.join(denoise_output_directory,
                                                            f"input_noise_{base_video_name}.mp4")

            if use_roi is True:
                loop_over = "all_video"
            else:
                loop_over = ""

            if main_denoise(window_size_temporal=window_size_temporal, stride=stride, video_path=self.video_path,
                            base_directory_of_cropped_images=None,
                            final_crop_size=(256, 256),
                            loop_over=loop_over, use_roi=use_roi, frame_index_to_start=0, frame_index_to_end=30,
                            noise_save_video_file_name=full_input_for_output_video_path,
                            denoise_save_video_file_name=full_output_video_path, show_video=False) is True:
                self.logger.log(f"Denoise process finished successfully")
                self.logger.log(f"Data saved into {full_input_for_output_video_path}")
            else:
                self.logger.log(f"Denoise process got an error")

        else:
            self.logger.log("Please select a video file.")
        self.window.destroy()


class DeblurWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Deblur")
        self.logger = logger

        # Description
        self.description_label = tk.Label(self.window, text="This is a deblur section for removing blur from the video")
        self.description_label.pack()

        # Button to browse for video file
        self.browse_video_button = tk.Button(self.window, text="Browse Video", command=self.browse_video)
        self.browse_video_button.pack()

        # Checkbox for selecting ROI
        self.use_roi_var = tk.BooleanVar(value=False)  # Default selection is not using ROI
        self.use_roi_checkbox = tk.Checkbutton(self.window, text="Use ROI", variable=self.use_roi_var)
        self.use_roi_checkbox.pack()

        # Button to apply deblur
        self.apply_button = tk.Button(self.window, text="Apply", command=self.apply_deblur)
        self.apply_button.pack()

        # Variable to store video path
        self.video_path = ""

    def browse_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")],
                                                initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        if video_path:
            self.video_path = video_path

    def apply_deblur(self):
        if self.video_path:
            use_roi = self.use_roi_var.get()
            # Perform deblur on the video
            # You can add your deblur code here
            base_video_name = os.path.basename(self.video_path).split('.')[0]
            deblur_output_directory = os.path.join(FULL_PATH_OUTPUT, "DEBLUR")
            if os.path.exists(deblur_output_directory) is False:
                os.mkdir(deblur_output_directory)
            deblur_output_directory_with_video_name = os.path.join(deblur_output_directory, base_video_name)
            if os.path.exists(deblur_output_directory_with_video_name) is False:
                os.mkdir(deblur_output_directory_with_video_name)
            full_output_video_path = os.path.join(deblur_output_directory_with_video_name,
                                                  f"deblur_{base_video_name}.mp4")
            full_input_for_output_video_path = os.path.join(deblur_output_directory_with_video_name,
                                                            f"input_blur_{base_video_name}.mp4")

            if main_deblur(self.video_path, use_roi=use_roi, start_read_frame=0,
                           end_read_frame=50, save_videos=True,
                           blur_video_mp4=full_output_video_path,
                           deblur_video_mp4=full_input_for_output_video_path):
                self.logger.log("Deblur applied successfully!")
                self.logger.log(f"Data saved into directory {deblur_output_directory_with_video_name}")

        else:
            self.logger.log("Please select a video file.")


class QuestionAnsweringWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Question Answering")
        self.logger = logger

        # Initial explanation paragraph
        self.intro_label = tk.Label(self.window, text="Welcome to the Question Answering section.(For Person!!!) Here, you can "
                                                      "upload images or directories of images. After uploading, "
                                                      "you can click the 'Apply' button to perform question "
                                                      "answering on the selected images.")
        self.intro_label.pack()

        # Button to browse for directory of images
        self.browse_directory_button = tk.Button(self.window, text="Upload Directory of Images", command=self.browse_directory)
        self.browse_directory_button.pack()

        # Button to browse for single image
        self.browse_image_button = tk.Button(self.window, text="Upload Image", command=self.browse_image)
        self.browse_image_button.pack()

        # Button to apply question answering
        self.apply_button = tk.Button(self.window, text="Apply", command=self.apply_question_answering)
        self.apply_button.pack()

        # Variables to store image paths
        self.directory_path = ""
        self.image_path = ""

    def browse_directory(self):
        directory_path = filedialog.askdirectory(initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        if directory_path:
            self.directory_path = directory_path

    def browse_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")],
                                                initialdir=BASE_DIRECTORY_DATA_UPLOADING
                                                )
        if image_path:
            self.image_path = image_path

    def apply_question_answering(self):
        #ToDo support to split person and vehicle questations by radio button

        ###H ERE WE ARE SUING CUDA INSIDE

        if self.directory_path or self.image_path:
            a_and_q_output_directory = os.path.join(FULL_PATH_OUTPUT, "ANSWER_AND_QUESTIONS")
            if os.path.exists(a_and_q_output_directory) is False:
                os.mkdir(a_and_q_output_directory)

            # Models
            hugging_face_model = hagging_face.get_model_and_stuff()
            blip_model, vis_processors, txt_processors = blip.get_model_and_stuff(blip.Action.QandA)

            question_categories = {
                "Outfit": PersonQuestion.Outfit,
                "Visibility": PersonQuestion.Visibility,
                "Actions": PersonQuestion.Actions
            }

            if self.directory_path:
                # For looping on directory of images
                images_directory = self.directory_path
                results_directory = a_and_q_output_directory
                if question_and_answering_loop_directory(images_directory, results_directory, question_categories,
                                                         hugging_face_model, blip_model, vis_processors, txt_processors) is True:
                    self.logger.log("Question answering applied successfully!")
                    self.logger.log(f"Data saved into {a_and_q_output_directory}")
                else:
                    self.logger.log("Got an error on performing questions and answering")


            elif self.image_path:
                single_image_path = self.image_path
                results_directory = a_and_q_output_directory
                if question_and_answering_single_image(single_image_path, results_directory, question_categories,
                                                    hugging_face_model, blip_model, vis_processors, txt_processors) is True:
                    self.logger.log("Question answering applied successfully!")
                    self.logger.log(f"Data saved into {a_and_q_output_directory}")
                else:
                    self.logger.log("Got an error on performing questions and answering")
        else:
            self.logger.log("Please select an image or a directory of images.")
        self.window.destroy()


class VideoSmartCompressionWindow:
    def __init__(self, master, logger):
        self.master = master
        self.window = tk.Toplevel(self.master)
        self.window.title("Video Smart Compression")
        self.logger = logger

        # Initial explanation paragraph
        self.intro_label = tk.Label(self.window, text="Welcome to the Video Smart Compression section. Here, you can "
                                                      "upload directories for frame images and mask images. After uploading, "
                                                      "you can click the 'Apply' button to perform smart compression on the selected images.")
        self.intro_label.pack()

        # Button to browse for directory of frame images
        self.browse_frames_directory_button = tk.Button(self.window, text="Upload Frames Directory",
                                                        command=self.browse_frames_directory)
        self.browse_frames_directory_button.pack()

        # Button to browse for directory of mask images
        self.browse_masks_directory_button = tk.Button(self.window, text="Upload Masks Directory",
                                                       command=self.browse_masks_directory)
        self.browse_masks_directory_button.pack()

        # Button to apply smart compression
        self.apply_button = tk.Button(self.window, text="Apply", command=self.apply_smart_compression)
        self.apply_button.pack()

        # Variables to store directory paths
        self.frames_directory_path = ""
        self.masks_directory_path = ""

    def browse_frames_directory(self):
        frames_directory_path = filedialog.askdirectory(initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        if frames_directory_path:
            self.frames_directory_path = frames_directory_path

    def browse_masks_directory(self):
        masks_directory_path = filedialog.askdirectory(initialdir=BASE_DIRECTORY_DATA_UPLOADING)
        if masks_directory_path:
            self.masks_directory_path = masks_directory_path

    def apply_smart_compression(self):
        if self.frames_directory_path and self.masks_directory_path:
            # Perform smart compression
            # You can add your smart compression code here
            output_base_dir_path = os.path.join(FULL_PATH_OUTPUT,"SMART_COMPRESSION")
            if os.path.exists(output_base_dir_path) is False:
                os.mkdir(output_base_dir_path)
            # get the parent directory
            parent_dir = os.path.dirname(self.frames_directory_path)
            # now get the parent's directory name
            parent_dir_name = os.path.basename(parent_dir)
            base_directory_output_name = parent_dir_name
            full_output_directory_path = os.path.join(output_base_dir_path, base_directory_output_name)
            if os.path.exists(full_output_directory_path) is False:
                os.mkdir(full_output_directory_path)
            vp = VideoProcessor(base_directory_path_input_for_output=full_output_directory_path)
            if vp.double_compression_method(images_dir=self.frames_directory_path, mask_dir=self.masks_directory_path) is True:
                self.logger.log("Smart Compression applied successfully!")
                self.logger.log(f"Data saved into {full_output_directory_path}")
            else:
                self.logger.log("Got an issue with the smart compression process")
        else:
            self.logger.log("Please select both frame and mask directories.")
        self.window.destroy()



class AppDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing App")
        self.root.geometry("1000x800")  # Adjust window size here

        # Add a button for video selection
        self.select_video_button = tk.Button(root, text="Video editor", command=self.select_video)
        self.select_video_button.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Use CUDA section
        self.cuda_var = tk.IntVar()
        self.cuda_var.set(1)  # Default to "Yes"

        self.cuda_label = tk.Label(root, text="Use Cuda")
        self.cuda_label.grid(row=0, column=1, padx=10, pady=10)  # Assuming column=1 is the center column
        self.cuda_yes_radiobutton = tk.Radiobutton(root, text="Yes", variable=self.cuda_var, value=1,
                                                   command=self.use_cuda)
        self.cuda_yes_radiobutton.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        self.cuda_no_radiobutton = tk.Radiobutton(root, text="No", variable=self.cuda_var, value=0,
                                                  command=self.use_cuda)
        self.cuda_no_radiobutton.grid(row=0, column=1, padx=10, pady=5, sticky='e')



        # Video Stabilization section
        self.title_stabilization = tk.Label(root, text="Video Stabilization")
        self.title_stabilization.grid(row=1, column=0, padx=10, pady=10)
        self.stabilization_methods = ["ECC_(Classic-Dud)", "Stabilization(OF-AVR)"]
        self.stabilization_combobox = ttk.Combobox(root, values=self.stabilization_methods)
        self.stabilization_combobox.grid(row=2, column=0, padx=10, pady=10)
        self.apply_stabilization_button = tk.Button(root, text="Apply", command=self.stabilize_video)
        self.apply_stabilization_button.grid(row=3, column=0, padx=10, pady=10)

        # Video Enhancer section
        self.title_enhancer = tk.Label(root, text="Video Enhancer")
        self.title_enhancer.grid(row=1, column=1, padx=10, pady=10)
        self.enhancement_methods = ["ECC_base_segmentation_mask", "De_noise", "De_blur", "De_jpeg", "Super regulation"]
        self.enhancement_combobox = ttk.Combobox(root, values=self.enhancement_methods)
        self.enhancement_combobox.grid(row=2, column=1, padx=10, pady=10)
        self.apply_enhancement_button = tk.Button(root, text="Apply", command=self.enhance_video)
        self.apply_enhancement_button.grid(row=3, column=1, padx=10, pady=10)

        # Video Processing section
        self.title_processing = tk.Label(root, text="Video Processing")
        self.title_processing.grid(row=1, column=2, padx=10, pady=10)
        self.processing_methods = ["Blur_kernel", "segmentation", "detection", "optical_flow",
                                   "frame_interpolation", "question_and_answer", "Blur_base_OF", "remove_object", "segmentation_optical_flow"]
        self.processing_combobox = ttk.Combobox(root, values=self.processing_methods)
        self.processing_combobox.grid(row=2, column=2, padx=10, pady=10)
        self.apply_processing_button = tk.Button(root, text="Apply", command=self.process_video)
        self.apply_processing_button.grid(row=3, column=2, padx=10, pady=10)

        # Video Compression section
        self.title_compression = tk.Label(root, text="Video Compression")
        self.title_compression.grid(row=1, column=3, padx=10, pady=10)
        self.compression_methods = ["h264"]
        self.compression_combobox = ttk.Combobox(root, values=self.compression_methods)
        self.compression_combobox.grid(row=2, column=3, padx=10, pady=10)
        self.apply_compression_button = tk.Button(root, text="Apply", command=self.compress_video)
        self.apply_compression_button.grid(row=3, column=3, padx=10, pady=10)

        # Add the desired text
        self.info_label = tk.Label(root,
                                   text="|||||||||||||||||||||||||||| Vehicle & License Plate System browse into: |||||||||||||||||||||||||||||||||||||||")
        self.info_label.grid(row=5, column=0, columnspan=4, pady=10)  # Adjust the grid positioning as needed

        # Making the link clickable
        self.link_label = tk.Label(root, text="http://127.0.0.1:5000", fg="blue", cursor="hand2")
        self.link_label.grid(row=6, column=0, columnspan=3, pady=10)  # Adjust the grid positioning as needed
        self.link_label.bind("<Button-1>", self.open_link)

        # Log box
        self.log_box = tk.Text(root, width=100, height=30)  # Set your preferred width and height
        self.log_box.grid(row=4, column=0, columnspan=4, padx=50, pady=10)

    def info(self, message):
        self.log(f"Info | {message}")

    def log(self, message):
        self.log_box.insert(tk.END, message + '\n')  # append the message to the log boxFVF

    def start_web_server(self):
        # Use the imported function to start the Flask server
        self.log("Starting the web server...")
        run_flask_web_page()
        # multiprocessing.set_start_method('spawn')  # or 'forkserver'
        # p = multiprocessing.Process(target=run_flask_web_page)  # Use the imported function here
        # p.start()


    def open_link(self, event):
        # Start a new thread to open the browser tab
        threading.Thread(target=self.open_browser_tab).start()

        # Start the Flask server
        self.start_web_server()



    def open_browser_tab(self):
        # Wait for 2 seconds (or however long you think the server might need to start up)
        time.sleep(2)

        # Open the browser tab
        webbrowser.open_new_tab("http://127.0.0.1:5000/")



    def select_video(self):
        # Open a file dialog for video selection
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")],
                                                initialdir=BASE_DIRECTORY_DATA_UPLOADING)

        # Create a new instance of VideoEditor with the selected video path

        output_directory_full_path = os.path.join(FULL_PATH_OUTPUT, "VIDEO_EDITOR")
        if os.path.exists(output_directory_full_path) is False:
            os.mkdir(output_directory_full_path)

        video_base_name = os.path.basename(video_path).split('.')[0]

        editor = VideoEditor(video_path,
                             output_video_path=os.path.join(output_directory_full_path, f"{video_base_name}.mp4"),
                             UiManager=self)

        # Log the selected video path
        self.log(f"Selected video: {video_path}")

        # Process the selected video using the VideoEditor instance
        editor.process_video()

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

    def stabilize_video(self):
        selection = self.stabilization_combobox.get()
        if not selection:
            self.log("Please select an stabilization method from the dropdown list.")
            exit_on(ERROR_INVALID_SELECTION)
        self.log("Stabilising video... This might take a while.")
        time.sleep(0.2)
        if selection == "ECC_(Dudy)":
            raise NotImplementedError
        elif selection == "Stabilization(OF-AVR)":
            StabilizationWindowAveraging(self.root, self)

    def enhance_video(self):
        selection = self.enhancement_combobox.get()
        if not selection:
            self.log("Please select an enhancement method from the dropdown list.")
            return
        self.log("Enhancing video... This might take a while.")
        time.sleep(0.2)
        if selection == "ECC_base_segmentation_mask":
            self.ECC_base_segmentation_mask()

        elif selection == "De_noise":
            DenoiseWindow(self.root, self)

        elif selection == "De_blur":
            DeblurWindow(self.root, self)

        elif selection == "De_jpeg":
            self.De_jpeg()

        elif selection == "Super_regulation":
            self.Super_regulation()



    def process_video(self):
        selection = self.processing_combobox.get()
        if not selection:
            self.log("Please select a processing method from the dropdown list.")
            return
        self.log("Processing video... This might take a while.")
        time.sleep(0.2)
        if selection == "Blur_kernel":
            BlurKernelWindow(self.root, self)
        elif selection == "segmentation":
            SegmentationWindow(self.root, self)
        elif selection == "detection":
            # DetectionWindow(self.root)
            self.log("Please use the App below and brows to the wep page application\n for detection and other stuff")
        elif selection == "optical_flow":
            self.optical_flow()

        elif selection == "frame_interpolation":
            FrameInterpolationWindow(self.root, self)

        elif selection == "question_and_answer":
            QuestionAnsweringWindow(self.root,self)

        elif selection == "Blur_base_OF":
            self.Blur_base_OF()

        elif selection == "remove_object":
            RemoveObjectWindow(self.root, self)

        elif selection == "segmentation_optical_flow":
            TrackerOpticalFlowWindow(self.root, self)

    def compress_video(self):
        selection = self.compression_combobox.get()
        if not selection:
            self.log("Please select an compression method from the dropdown list.")
            return
        self.log("compressing video... This might take a while.")
        time.sleep(0.2)
        if selection == "h264":
            VideoSmartCompressionWindow(self.root,self)


    def use_cuda(self):
        global  use_cuda_global
        use_cuda_global = self.cuda_var.get()
        print(use_cuda_global)


root = tk.Tk()
demo = AppDemo(root)
root.mainloop()



# base_output = "/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/output1"
# base_directory_of_images = "/home/nehoray/PycharmProjects/Shaback/output/video_example_from_security_camera/crops/2/7"
# chain_demo_main_interface(base_directory_of_images=base_directory_of_images, base_output=base_output)




import fnmatch
import json
import os
import time
from enum import Enum
from typing import List

# import moviepy
# import moviepy.video
# import moviepy.video.io
# import moviepy.video.io.ImageSequenceClip

import cv2
import matplotlib.pyplot as plt
from PIL import Image

from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch, imshow_torch_video
from RapidBase.Utils.Path_and_Reading_utils import video_torch_array_to_video
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_to_numpy, numpy_to_torch

os.environ["OMP_NUM_THREADS"]           = "1"
os.environ["OPENBLAS_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]           = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]    = "1"
os.environ["NUMEXPR_NUM_THREADS"]       = "1"

import argparse
import glob
import pathlib
import re
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# from RapidBase.Utils.Path_and_Reading_utils import video_numpy_array_to_video
from plate_recognizer_company.backend_api import get_objects_detection_api_call_data_of
from util.get_color_image import most_common_color, get_color_name, get_closest_name


import imghdr
import mimetypes

import logging
import sys
sys.path.append("/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking")
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, \
    process_mask_native
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker
import sys
import platform
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
START_FRAME_INDEX_TO_READ = 0



class PlateRecognizerOptionalCarColor(Enum):
    black   = "black"
    blue    = "blue"
    brown   = "brown"
    green   = "green"
    red     = "red"
    silver  = "silver"
    white   = "white"
    yellow  = "yellow"
    unknown = "unknown"


class PlateRecognizerOptionalCarOrientation(Enum):
    Front = "Front"
    Rear = "Rear"
    Unknown = "Unknown"


class PlateRecognizerOptionalCarType(Enum):
    Big_Truck = "Big Truck"
    Bus = "Bus"
    Motorcycle = "Motorcycle"
    Pickup_Truck = "Pickup Truck"
    Sedan = "Sedan"
    SUV = "SUV"
    Van = "Van"
    Unknown = "Unknown"




class OurOptionalLicensePlateColor(Enum):
    white = "2",
    yellow = "3",
    green = "1",
    unknown = "-1",



class UserInputOptionalTerms:
    def __init__(self,
        input_user_vehicle_color: List[PlateRecognizerOptionalCarColor] = None,
        input_user_vehicle_make_model_make: List[str] = None,
        input_user_vehicle_make_model_model: List[str] = None,
        input_user_vehicle_orientation: List[PlateRecognizerOptionalCarOrientation] = None,
        input_user_vehicle_type: List[PlateRecognizerOptionalCarType] = None,
        input_user_license_plate_regular_expression: str = None,
        input_user_license_plate_color: List[OurOptionalLicensePlateColor] = None,
    ):
        self.input_user_vehicle_color: PlateRecognizerOptionalCarColor = input_user_vehicle_color
        self.input_user_vehicle_make_model_make: str = input_user_vehicle_make_model_make
        self.input_user_vehicle_make_model_model: str = input_user_vehicle_make_model_model
        self.input_user_vehicle_orientation: PlateRecognizerOptionalCarOrientation = input_user_vehicle_orientation
        self.input_user_vehicle_type: PlateRecognizerOptionalCarType = input_user_vehicle_type
        self.input_user_license_plate_regular_expression: str  = input_user_license_plate_regular_expression
        self.input_user_license_plate_color: OurOptionalLicensePlateColor = input_user_license_plate_color




if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class VehicleCounter:
    def __init__(self):
        self.counter_truck = 0
        self.counter_car = 0
        self.counter_bus = 0



def is_video_file(file_path):
    # Check if the file extension is a video format
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']
    if any(file_path.lower().endswith(ext) for ext in video_extensions):
        return True

    # Check if the file type is a video format using mimetypes and imghdr modules
    file_type, _ = mimetypes.guess_type(file_path)
    if file_type is not None and file_type.startswith('video/'):
        return True
    elif imghdr.what(file_path) == 'mpg':
        return True

    return False


def chw_2_hwc(input_array):
    return input_array.transpose([1, 2, 0])




def numpy_unsqueeze(input_tensor, dim=-1):
    return np.expand_dims(input_tensor, dim)


def upsample_image(input_array, size):
    upsample_layer = torch.nn.Upsample(size, mode="bilinear")
    if type(input_array) == np.ndarray:
        output_tensor = upsample_layer(numpy_to_torch(input_array).unsqueeze(dim=0))
    else:
        output_tensor = upsample_layer(input_array.unsqueeze(dim=0))
    return output_tensor.squeeze()


def update_window_cv(image):
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Close window on pressing 'q'
        cv2.destroyAllWindows()


def update_window(fig, ax, image):
    ax.clear()
    ax.axis('off')
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.pause(0.00001)
    fig.canvas.draw()



def path_get_files_recursively(main_path, string_to_find='', flag_full_path=True):
    files = []
    for r,d,f in os.walk(main_path):
        for file in f:
            if string_to_find in file:
                if flag_full_path:
                    files.append(os.path.join(r, file))
                else:
                    files.append(file)
    return files


def video_from_tensor(list_of_files: list, output_video: str, file_type: str, fps: int = 25,
                      size: "list[int]" = [500, 640]):
    print(output_video)
    """Takes in a folder of npy/pt images and outputs a video

    Args:
        base_folder (str): The path to the folder that holds all the images
        output_video (str): Path to save the video to
        file_type (str): The file kind (npy, pt)
        fps (int): FPS that the video will be saved at. Defaults to 25
        size (tuple, optional): The size of the video. Defaults to (500, 640).
    """

    if not file_type.startswith("."):
        file_type = f".{file_type}"

    files_list = list_of_files
    # files_list = path_get_files_recursively(list_of_, file_type)
    files_list = sorted(files_list, key=lambda x: int(os.path.split(x)[1].split(".")[0]))
    for frame in np.arange(len(files_list)):
        print(frame)
        current_filename = files_list[frame]

        if file_type == ".npy" and os.path.split(current_filename)[-1].split(".")[-1] == file_type[1:]:
            video = np.load(current_filename)

        elif file_type == ".pt" and os.path.split(current_filename)[-1].split(".")[-1] == file_type[1:]:
            video = torch.load(current_filename)
            video = torch_to_numpy(video)

        elif ".jpg":
            video = plt.imread(current_filename)

        if len(video.shape) == 2:
            video = numpy_unsqueeze(video, dim=-1)

        if frame == 0:
            video_min = video.min()
            video_max = video.max()
            # size = [video.shape[0], video.shape[1]]
            fourcc = cv2.VideoWriter_fourcc(*'MP42')
            video_writer_object = cv2.VideoWriter(output_video, fourcc, fps, (size[1], size[0]))

        ### Scale array: ###
        video_new = (video - video_min) / (video_max - video_min) * 255
        video_new = video_new.clip(0, 255)
        current_frame = video_new
        current_frame_numpy = np.asarray((torch_to_numpy(upsample_image(current_frame, size))), dtype=np.uint8)
        current_frame_numpy = current_frame_numpy.clip(0, 255)
        # current_frame_numpy = BW2RGB(current_frame_numpy.astype(np.uint8))
        # plt.imshow(current_frame_numpy)
        # plt.show()
        # print(current_frame_numpy)
        video_writer_object.write(current_frame_numpy)

    video_writer_object.release()


def init_window(window_title):
    fig, ax = plt.subplots()
    fig.suptitle(window_title)
    ax.axis('off')
    return fig, ax




def create_video_from_list_of_images_path(images, output_video_name, output_directory_path, fps=10):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory_path, exist_ok=True)

    # Determine the video codec and frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = cv2.imread(images[0]).shape[1::-1]

    # Create the output video writer
    output_video_path = os.path.join(output_directory_path, output_video_name)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Write the frames to the output video
    for i, image_path in enumerate(images):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f'Error: Could not read image {image_path}')
            continue
        video_writer.write(frame)
        print(f'Processed image {i+1}/{len(images)}')

    # Release the video writer and return the output video path
    video_writer.release()
    return output_video_path



def get_center_x_y(x, y, w, h):
    x1 = int(w / 2)

    y1 = int(h / 2)

    cx = x + x1

    cy = y + y1

    return (cx, cy)




def string_match_pattern(input_string, input_pattern):
    return fnmatch.fnmatch(input_string, input_pattern)


def path_get_files_from_folder(path, number_of_files=np.inf, flag_recursive=False, string_pattern_to_search='*',
                               flag_full_path=True):
    count = 0
    image_filenames_list = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        if count >= number_of_files and number_of_files != np.inf:
            break
        for fname in sorted(fnames):
            if count >= number_of_files and number_of_files != np.inf:
                break

            # if string_match_pattern(fname, string_pattern_to_search):
            if string_match_pattern(os.path.join(dirpath, fname), string_pattern_to_search):
                if flag_full_path:
                    img_path = os.path.join(dirpath, fname)
                else:
                    img_path = fname

                image_filenames_list.append(img_path)
                count += 1

        if flag_recursive == False:
            break
    return image_filenames_list


def upsample_image(input_array, size):
    upsample_layer = torch.nn.Upsample(size, mode="bilinear")
    if type(input_array) == np.ndarray:
        output_tensor = upsample_layer(numpy_to_torch(input_array).unsqueeze(dim=0))
    else:
        output_tensor = upsample_layer(input_array.unsqueeze(dim=0))
    return output_tensor.squeeze()


def video_from_tensor(base_folder: str, output_video: str, file_type: str, fps: int = 25,
                      size: "list[int]" = [500, 640]):
    """Takes in a folder of npy/pt images and outputs a video

    Args:
        base_folder (str): The path to the folder that holds all the images
        output_video (str): Path to save the video to
        file_type (str): The file kind (npy, pt)
        fps (int): FPS that the video will be saved at. Defaults to 25
        size (tuple, optional): The size of the video. Defaults to (500, 640).
    """

    if not file_type.startswith("."):
        file_type = f".{file_type}"

    files_list = path_get_files_recursively(base_folder, file_type)
    files_list = sorted(files_list, key=lambda x: int(os.path.split(x)[1].split(".")[0]))
    for frame in np.arange(len(files_list)):
        print(frame)
        current_filename = files_list[frame]

        if file_type == ".npy" and os.path.split(current_filename)[-1].split(".")[-1] == file_type[1:]:
            video = np.load(current_filename)

        elif file_type == ".pt" and os.path.split(current_filename)[-1].split(".")[-1] == file_type[1:]:
            video = torch.load(current_filename)
            video = torch_to_numpy(video)

        elif ".jpg":
            video = plt.imread(current_filename)

        if len(video.shape) == 2:
            video = numpy_unsqueeze(video, dim=-1)

        if frame == 0:
            video_min = video.min()
            video_max = video.max()
            # size = [video.shape[0], video.shape[1]]
            # fourcc = cv2.VideoWriter_fourcc(*'MP42')
            video_writer_object = cv2.VideoWriter(output_video, 0, fps, (size[1], size[0]))

        ### Scale array: ###
        video_new = (video - video_min) / (video_max - video_min) * 255
        video_new = video_new.clip(0, 255)
        current_frame = video_new
        current_frame_numpy = np.asarray((torch_to_numpy(upsample_image(current_frame, size))), dtype=np.uint8)
        current_frame_numpy = current_frame_numpy.clip(0, 255)
        # current_frame_numpy = BW2RGB(current_frame_numpy.astype(np.uint8))
        # plt.imshow(current_frame_numpy)
        # plt.show()
        video_writer_object.write(cv2.cvtColor(current_frame_numpy, cv2.COLOR_BGR2RGB))

    video_writer_object.release()


def correct_over_under_flow_in_BB(BB, HW_tuple):
    H, W = HW_tuple
    x1, y1, x2, y2 = BB
    x1 = min(x1, W)
    x1 = max(x1, 0)
    y1 = min(y1, H)
    y1 = max(y1, 0)
    x2 = min(x2, W)
    x2 = max(x2, 0)
    y2 = min(y2, H)
    y2 = max(y2, 0)
    return np.array([x1, y1, x2, y2]).astype(np.int16)


@torch.no_grad()
def run(
        save_only_relevant_user_input_data=False,
        user_input_object_terms: UserInputOptionalTerms = UserInputOptionalTerms(),
        source='0',
        yolo_weights=WEIGHTS / "yolov8x-seg.pt", # "best_3.pt", #"best_1.pt",    #'yolov8s-seg.pt', #"best_1.pt", # model.pt path(s),  # /home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/weights/yolov8s-seg.pt
        # yolo_weights=WEIGHTS / "yolov8s-seg.pt",
        # yolo_weights=WEIGHTS/"last_best_1.pt",
        # yolo_weights= pathlib.Path("/home/dudy/Downloads/last.pt"), # 'yolov8s-seg.pt',  # model.pt path(s),  # /home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/weights/yolov8s-seg.pt
        # yolo_weights=WEIGHTS / "best_3.pt",

        # reid_weights=WEIGHTS / 'mobilenetv2_x1_4_msmt17.engine',  # model.pt path,
        # reid_weights=WEIGHTS / 'osnet_x0_25_market1501.pt',  # model.pt path,
        reid_weights=WEIGHTS / 'resnet50_msmt17.onnx',  # model.pt path,
        tracking_method='bytetrack',
        tracking_config=ROOT / 'trackers' / "bytetrack" / 'configs' / ("bytetrack" + '.yaml'),
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.50,  # confidence threshold
        iou_thres=0.50,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        flag_augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project_path_to_save_results=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        deblur_frames_read_start=0,
        deblur_frames_read_end=None

):

        source = str(source)

        ### source loader - check which source this is (video file, url or webcam) and act accordingly ###
        # save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        ### Directories to save results to: ###
        if not isinstance(yolo_weights, list):  # single yolo model
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(yolo_weights[0]).stem
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + reid_weights.stem
        #########
        # for save in one direcory
        # exp_name = "____all_input_____"
        save_dir = increment_path(Path(project_path_to_save_results) / exp_name, exist_ok=exist_ok)  # increment run

        #########
        # for one directory evrithing
        save_dir = Path(str(save_dir)[:-1])

        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        ### Load model (detection, segmentation): ###
        device = select_device(device)
        flag_perform_segmentation = '-seg' in str(yolo_weights)
        model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_imgsz(imgsz, stride=stride)  # check image size

        ### Initialize Dataloader: ###
        bs = 1
        if webcam:
            show_vid = check_imshow(warn=True)
            dataset = LoadStreams(
                source,
                imgsz=imgsz,
                stride=stride,
                auto=pt,
                transforms=getattr(model.model, 'transforms', None),
                vid_stride=vid_stride
            )
            bs = len(dataset)
        else:
            dataset = LoadImages(
                source,
                imgsz=imgsz,
                stride=stride,
                auto=pt,
                transforms=getattr(model.model, 'transforms', None),
                vid_stride=vid_stride
            )
        vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

        ### TRACKER - Create as many strong-sort instances as there are video sources: ###
        tracker_list = []
        for i in range(bs):
            tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        tracker_outputs = [None] * bs

        ### Loop over all frames from dataset object: ###
        # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        curr_frames, prev_frames = [None] * bs, [None] * bs
        vehicles_count = 0
        list_of_counted_vehicles = []
        # draw_line_widget = DrawLineWidget()
        list_of_polygons_traffic_coordinates = []
        list_of_counted_vehicles_by_id = []
        total_vehicles = 0
        total_trucks = 0
        image_result_counters = np.zeros([100, 100, 3], dtype=np.uint8)
        image_result_counters.fill(255)  # or img[:] = 255

        list_of_polygons_counters = []

        # frame_stride = 130
        ### Loop over frames: ###
        for frame_idx, batch in enumerate(dataset):
            # if frame_idx % vid_stride !=0:
            #     print(f"{frame_idx} - continue")
            #     continue
            # print(frame_idx)
            ### Get current image path, image after scaling, original image etc': ###
            path, current_frame_after_downsample, current_frame_original_size, video_capture_object, print_string = batch

            # ### Frame striding (in case of high FPS, or i just want to skip images because we don't need prediction for every single frame): ###
            # if frame_idx > -1:
            #     for _ in np.arange(frame_stride):
            #         flag_frame_available, frame = video_capture_object.read()
            #

            ### Visualize - downsample frame for yolo etc': ###
            # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            with dt[0]:  # TODO: understand what the different context decorators do here
                current_frame_after_downsample = torch.from_numpy(current_frame_after_downsample).to(device)
                current_frame_after_downsample = current_frame_after_downsample.half() if half else current_frame_after_downsample.float()  # uint8 to fp16/32
                current_frame_after_downsample /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(current_frame_after_downsample.shape) == 3:
                    current_frame_after_downsample = current_frame_after_downsample[None]  # expand for batch dim

            ### If this is first frame - flag it: ###
            if frame_idx == 0:
                if is_video_file(source):
                    first_frame_flag, first_frame = video_capture_object.read()
                if webcam is True:
                    #ToDo support it
                    raise  NotImplementedError
                else:
                    ##For image processing
                    try:
                        first_frame = cv2.imread(source)
                    except Exception:
                        first_frame_flag = False
                    first_frame_flag =True

            ### YOLO Inference: ###
            with dt[1]:
                # TODO: understand what is the model output
                preds = model(current_frame_after_downsample, augment=flag_augment, visualize=visualize)

            ### Apply NMS To BB: ###
            with dt[2]:
                if flag_perform_segmentation:
                    masks = []  # TODO: rename p
                    prediction_outputs = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms,
                                                             max_det=max_det, nm=32)
                    proto = preds[1][-1]  # TODO: what is proto?
                else:
                    prediction_outputs = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms,
                                                             max_det=max_det)

            ### Loop over detections: ###  #TODO: probably rename i to detection_object_counter or something
            for detection_object_index, det in enumerate(prediction_outputs):  # detections per image
                # print(prediction_outputs)
                seen += 1
                ### Take care of paths according to source: ###
                if webcam:  # bs >= 1
                    path, current_frame_original, _ = path[detection_object_index], current_frame_original_size[
                        detection_object_index].copy(), dataset.count
                    path = Path(path)  # to Path
                    print_string += f'{detection_object_index}: '
                    txt_file_name = path.name
                    save_path = str(save_dir / path.name)  # im.jpg, vid.mp4, ...
                else:
                    if frame_idx == 59:
                        bla = 1
                    ### Get ?????: ###
                    path, current_frame_original, _ = path, current_frame_original_size.copy(), getattr(dataset,
                                                                                                        'frame', 0)
                    path = Path(path)  # to Path

                    ### video file: ### #TODO: understand what this saves
                    if source.endswith(VID_FORMATS):
                        txt_file_name = path.stem
                        save_path = str(save_dir / path.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        txt_file_name = path.parent.name  # get folder name containing current img
                        save_path = str(save_dir / path.parent.name)  # im.jpg, vid.mp4, ...

                ### Get original image: ###
                curr_frames[detection_object_index] = current_frame_original

                ### TODO: what is being saved? ###
                txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                print_string += '%gx%g ' % current_frame_after_downsample.shape[2:]  # print string
                current_frame_original_copy = current_frame_original.copy() if save_crop else current_frame_original  # for save_crop

                ### Initialize Annotator object responsible for drawing detection BB and segmentation masks on image: ###
                annotator = Annotator(current_frame_original, line_width=line_thickness, example=str(names))

                ### Compensate for camera motion - (TODO: probably uses ECC, understand how long this takes): ###
                if hasattr(tracker_list[detection_object_index], 'tracker') and hasattr(
                        tracker_list[detection_object_index].tracker, 'camera_update'):
                    if prev_frames[detection_object_index] is not None and curr_frames[
                        detection_object_index] is not None:  # camera motion compensation
                        tracker_list[detection_object_index].tracker.camera_update(prev_frames[detection_object_index],
                                                                                   curr_frames[detection_object_index])

                ### If we have a detection - do something: ###
                # seems like det consists of 38 number:
                # (BB[0],BB[1],BB[2],BB[3],Conf(?),Class, 32 numbers) representing numbers to be matrix multiplied by proto (segmentation mask) to be inputed into a sigmoid and output final mask
                if det is not None and len(det):
                    if flag_perform_segmentation:
                        shape = current_frame_original.shape
                        ### scale bbox first the crop masks: ###
                        if retina_masks:
                            det[:, 0:4] = scale_boxes(current_frame_after_downsample.shape[2:], det[:, 0:4],
                                                      shape).round()  # rescale boxes to current_frame_original size
                            masks.append(process_mask_native(proto[detection_object_index], det[:, 6:], det[:, 0:4],
                                                             current_frame_original.shape[:2]))  # HWC
                        else:
                            # [segmentation_mask_on_downsampled_image] = [N_BB, H, W]
                            segmentation_mask_on_downsampled_image = process_mask(proto[detection_object_index],
                                                                                  det[:, 6:], det[:, :4],
                                                                                  current_frame_after_downsample.shape[
                                                                                  2:], upsample=True)
                            masks.append(segmentation_mask_on_downsampled_image)  # HWC
                            det[:, :4] = scale_boxes(current_frame_after_downsample.shape[2:], det[:, :4],
                                                     shape).round()  # rescale boxes to current_frame_original size
                    else:
                        det[:, :4] = scale_boxes(current_frame_after_downsample.shape[2:], det[:, :4],
                                                 current_frame_original.shape).round()  # rescale boxes to current_frame_original size

                    ### Understand which class current BB is and add it to statistics: ###
                    for output_class in det[:, 5].unique():
                        number_of_detection_per_this_class = (det[:, 5] == output_class).sum()  # detections per class
                        print_string += f"{number_of_detection_per_this_class} {names[int(output_class)]}{'s' * (number_of_detection_per_this_class > 1)}, "  # add to string  #TODO: understand if 's' should be changed to 'pring_string'

                    ### TRACK bounding boxes using tracker: ###
                    with dt[3]:
                        tracker_outputs[detection_object_index] = tracker_list[detection_object_index].update(det.cpu(),
                                                                                                              current_frame_original)

                    ### Draw boxes for visualization: ###
                    if len(tracker_outputs[detection_object_index]) > 0:
                        ### Plot segmentation on image using annotator object: ###
                        if flag_perform_segmentation:
                            ### Segmentation masks: ###
                            annotator.masks(
                                masks[detection_object_index],
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=torch.as_tensor(current_frame_original, dtype=torch.float16).to(device).permute(
                                    2, 0, 1).flip(0).contiguous() /
                                       255 if retina_masks else current_frame_after_downsample[detection_object_index]
                            )

                            # ### Save segmentation mask: ###
                            # torch.save(masks[detection_object_index], f'{base_frames_saved_directory}/tensor_{detection_object_index}.pt')

                            #### save segmentation per object id and index ####

                        ### Loop over detection tracker_outputs: ###
                        for tracker_BB_index, (output) in enumerate(tracker_outputs[detection_object_index]):
                            ### Assign detector model tracker_outputs to variable names: ###
                            if frame_idx == 59:
                                bla = 1
                            bbox = output[0:4]
                            id = output[4]
                            cls = output[5]
                            conf = output[6]

                            ### Correct for overflow or underflow in shape: ###
                            bbox = correct_over_under_flow_in_BB(bbox, current_frame_original.shape[0:2])

                            ### Check if current Tracker Bounding Box is inside user-defined polygon for counting: ###
                            # centerx, centery = (numpy.average(bbox[:2]), numpy.average(bbox[2:]))
                            centerx, centery = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                            cv2.circle(current_frame_original, (int(centerx), int(centery)), 8, (0, 0, 255), -1)
                            center_point_as_point_obj = Point(centerx, centery)
                            if id not in list_of_counted_vehicles_by_id:
                                for pol_index, polygon in enumerate(list_of_polygons_traffic_coordinates):
                                    if Polygon(list(polygon)).contains(center_point_as_point_obj):
                                        if cls == 2:
                                            list_of_polygons_counters[pol_index].counter_car += 1
                                            list_of_counted_vehicles_by_id.append(id)
                                        elif cls == 7:
                                            list_of_polygons_counters[pol_index].counter_truck += 1
                                            list_of_counted_vehicles_by_id.append(id)
                                        elif cls == 5:
                                            list_of_polygons_counters[pol_index].counter_bus += 1
                                            list_of_counted_vehicles_by_id.append(id)
                            # print("\n\n\n\n\n")
                            # for idx, pol in enumerate(list_of_polygons_counters):
                            #     print(f"Polygon index --- {idx} ---")
                            #     print(f"    Total Cars:  ", pol.counter_car)
                            #     print(f"    Total Trucks:", pol.counter_truck)
                            #     print(f"    Total Bus:   ", pol.counter_bus)
                            #     print("\n")

                            # update_counting_vehicles()

                            ### Export results to txt file for MOT (multi-object-tracking) format: ###
                            if save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1,
                                                                   detection_object_index))



                            ### If there's anything to save or to show - do it: ###
                            if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                                ### Assign detector outputs to variable names: ###
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else
                                                                  (
                                                                      f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))

                                ### Append to the label text te license plate
                                try:
                                    current_file_name_per_id_for_label = save_dir / 'crops' / names[c] / f'{id}' / f'most_score_information.json'
                                    with open(current_file_name_per_id_for_label) as f1:
                                        current_data_for_label = json.load(f1)
                                        license_plate_only = current_data_for_label["PlateRecognizer"]["plate"]["props"]["plate"]["value"]
                                        label = f'{label} LP: {str(license_plate_only)}'
                                except Exception as e:
                                    print(f"license_plate_only | Got an exepetion - \n\n - {e} \n\ ")

                                color = colors(c, True)

                                ### Actually draw the bounding box: ###

                                ### Cheking the conditation and desice if to disply or not

                                try:

                                    ################################################
                                    ################################################
                                    #######     Checking user terms      ##########
                                    ################################################
                                    ################################################


                                    ## Define the most score data per current trached id object
                                    current_file_name_per_id_for_checking_terms = save_dir / 'crops' / \
                                                                                                 names[
                                                                                                     c] / f'{id}' / f'most_score_information.json'
                                    with open(current_file_name_per_id_for_checking_terms) as f1:
                                        current_data = json.load(f1)
                                        current_plate_recognizer_data = current_data["PlateRecognizer"]

                                        flag_draw_object = True
                                        #### User input terms checking  #####

                                        ### Vehicle ###
                                        #### color
                                        if user_input_object_terms.input_user_vehicle_color is not None and flag_draw_object is True:
                                            if current_plate_recognizer_data["vehicle"]["props"]["color"]["value"] not in user_input_object_terms.input_user_vehicle_color and current_plate_recognizer_data["vehicle"]["props"]["color"]["value"] != "":
                                                flag_draw_object = False

                                        ### make model - make
                                        if user_input_object_terms.input_user_vehicle_make_model_make is not None and flag_draw_object is True:
                                            if current_plate_recognizer_data["vehicle"]["props"]["make_model"]["make"] not in user_input_object_terms.input_user_vehicle_make_model_make and current_plate_recognizer_data["vehicle"]["props"]["make_model"]["make"] != "":
                                                flag_draw_object = False

                                        ### make model - model
                                        if user_input_object_terms.input_user_vehicle_make_model_model is not None and flag_draw_object is True:
                                            if current_plate_recognizer_data["vehicle"]["props"]["make_model"]["model"] not in user_input_object_terms.input_user_vehicle_make_model_model and current_plate_recognizer_data["vehicle"]["props"]["make_model"]["model"] != "":
                                                flag_draw_object = False

                                        ### orientation
                                        if user_input_object_terms.input_user_vehicle_orientation is not None and flag_draw_object is True:
                                            if current_plate_recognizer_data["vehicle"]["props"]["orientation"]["value"] not in user_input_object_terms.input_user_vehicle_orientation and user_input_object_terms.input_user_vehicle_orientation != "":
                                                flag_draw_object = False

                                        ### type
                                        if user_input_object_terms.input_user_vehicle_type is not None and flag_draw_object is True:
                                            if current_plate_recognizer_data["vehicle"]["type"] not in user_input_object_terms.input_user_vehicle_type and current_plate_recognizer_data["vehicle"]["type"] != "":
                                                flag_draw_object = False


                                        #### License plate ####
                                        ### regular expression for the license plate
                                        # if user_input_object_terms.input_user_license_plate_regular_expression is not None and flag_draw_object is True:
                                        #     if current_plate_recognizer_data["plate"]["props"]["value"] not in user_input_object_terms.input_user_license_plate_regular_expression:
                                        #         flag_draw_object = False

                                        ### License plate color
                                        if  user_input_object_terms.input_user_license_plate_color is not None and flag_draw_object is True:
                                            #ToDo
                                            pass


                                        ### Actually draw bounding box
                                        if flag_draw_object is True:
                                            annotator.box_label(bbox, label, color=color)

                                        flag_draw_object = True
                                        current_frame_original = annotator.result()

                                except Exception as e:
                                    print(f"Failed to check terms for drawing - {e}")
                                    flag_draw_object = True

                                ### Save tracking outputs to list: ###
                                if save_trajectories and tracking_method == 'strongsort':
                                    q = output[7]  # TODO: what is q????
                                    tracker_list[detection_object_index].trajectory(current_frame_original, q,
                                                                                    color=color)

                                ### Save bounding box crop if wanted: ###
                                if save_crop:
                                    # TODO: understand what's being saved here and how for shabak
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    bbox = np.array(bbox, dtype=np.int16)
                                    crop = current_frame_original_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                    resized_crop = cv2.resize(crop, (224, 224))
                                    h, w, _ = resized_crop.shape
                                    new_bb = [0, 0, w, h]
                                    # save_one_box(np.array(new_bb, dtype=np.int16), resized_crop,
                                    # file=save_dir / 'crops' / txt_file_name / "dudy" / "for_dudy" / f'{path.stem}.jpg', BGR=True)

                                    ### Save Crop from original sized image with Bounding Box (not the downsampled image): ###
                                    save_one_box(np.array(bbox, dtype=np.int16), current_frame_original_copy, gain=1,
                                                 pad=0,
                                                 file=save_dir / 'crops' / txt_file_name / names[
                                                     c] / f'{id}' / f'{frame_idx}.png', BGR=True)


                                    if "-seg" in str(yolo_weights.name):
                                        ### Save Segmentation Mask from original sized image resolution: ###
                                        H, W = current_frame_original_copy.shape[0:2]
                                        current_id_segmentation_mask = segmentation_mask_on_downsampled_image[
                                                                       tracker_BB_index:tracker_BB_index + 1].unsqueeze(
                                            0)  # [1,1,H,W]
                                        current_id_segmentation_mask_upsampled = torch.nn.Upsample(size=(H, W),
                                                                                                   mode='nearest')(
                                            current_id_segmentation_mask)
                                        current_id_segmentation_mask_upsampled_crop = current_id_segmentation_mask_upsampled[
                                                                                      :, :, bbox[1]:bbox[3],
                                                                                      bbox[0]:bbox[2]]

                                    ### Get most common color on the object detected ####
                                    # crop_image_for_color_detection = current_frame_original_copy[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                                    # crop_segmentation_mask_for_color_detection = torch_to_numpy(current_id_segmentation_mask_upsampled_crop)[0]
                                    # most_common_color_rgb = most_common_color_via_segmentation_mask(crop_segmentation_mask_for_color_detection,crop_image_for_color_detection)
                                    # print("most common color is", most_common_color_rgb)
                                    # closest_color_name = get_closest_name(most_common_color_rgb)
                                    # print("closest color name ",closest_color_name)

                                    ###### Get total info from plate recognizer #######
                                    ## save boundin box ###
                                    crop_image_as_bgr = cv2.cvtColor(current_frame_original_copy, cv2.COLOR_RGB2BGR)
                                    img = Image.fromarray(crop_image_as_bgr[bbox[1]:bbox[3], bbox[0]:bbox[2], :])
                                    # Save the image to disk
                                    crop_image_file_name = "CROP_IMAGE_FOR.jpg"
                                    img.save(crop_image_file_name)

                                    # ### Test saving all frame
                                    # img = Image.fromarray(
                                    #     crop_image_as_bgr)
                                    # # Save the image to disk
                                    # crop_image_file_name = "total_frame.jpg"
                                    # img.save(crop_image_file_name)

                                    ### Get object detection PR information ###
                                    plate_recognizer_current_object_id_output_data_js = get_objects_detection_api_call_data_of(
                                        crop_image_file_name)
                                    print(plate_recognizer_current_object_id_output_data_js)

                                    #### Write car information per frame as a json format

                                    current_object_id_most_score_information_vehicle_file_name = save_dir / 'crops' / txt_file_name / \
                                                                                                 names[
                                                                                                     c] / f'{id}' / f'most_score_information.json'
                                    json_information_template_file = {
                                        "PlateRecognizer": {
                                            "plate": {
                                                "type": "Plate",
                                                "score": 0.0,

                                                "props": {
                                                    "plate":
                                                        {
                                                            "value": "",
                                                            "score": 0.0
                                                        },
                                                    "region":
                                                        {
                                                            "value": "",
                                                            "score": 0.0
                                                        }

                                                }
                                            },
                                            "vehicle": {
                                                "type": "",
                                                "score": 0.0,

                                                "props": {
                                                    "make_model":
                                                        {
                                                            "make": "",
                                                            "model": "",
                                                            "score": 0.0
                                                        }
                                                    ,
                                                    "orientation":
                                                        {
                                                            "value": "",
                                                            "score": 0.0
                                                        }
                                                    ,
                                                    "color":
                                                        {
                                                            "value": "",
                                                            "score": 0.0
                                                        },
                                                }
                                            }
                                        },
                                        "Our": {
                                            "vehicle": {
                                                "color": "",
                                                "type": " "
                                            }
                                        }
                                    }
                                    most_score_information_data_result_TEMPLATE = json_information_template_file.copy()
                                    json_information_file_template_per_frame = json_information_template_file.copy()

                                    if not os.path.exists(current_object_id_most_score_information_vehicle_file_name):
                                        #### Initilize the json output file and some other things ###
                                        with open(current_object_id_most_score_information_vehicle_file_name, "w") as f:
                                            most_score_information_data_result_TEMPLATE_object_data = json.dumps(
                                                most_score_information_data_result_TEMPLATE)
                                            f.write(most_score_information_data_result_TEMPLATE_object_data)

                                    ### saving result as json file for each object and each frame
                                    json_information_file_to_save_name_per_frame = save_dir / 'crops' / txt_file_name / \
                                                                                   names[
                                                                                       c] / f'{id}' / f'{frame_idx}.json'
                                    data_to_write = json_information_file_template_per_frame.copy()
                                    try:
                                        #### Write plate recognizer info
                                        data_to_write["PlateRecognizer"] = \
                                        plate_recognizer_current_object_id_output_data_js[0]

                                        #### write our info
                                        data_to_write["Our"]["vehicle"]["type"] = names[c]
                                        data_to_write["Our"]["vehicle"]["color"] = "Unknow"  # ToDo fic color detection

                                        ### write data into file
                                        with open(json_information_file_to_save_name_per_frame, "w") as f:
                                            json_object = json.dumps(data_to_write)
                                            f.write(json_object)
                                    except:
                                        pass

                                    ###### Write the most score vehicle information json file ############

                                    with open(current_object_id_most_score_information_vehicle_file_name) as f:
                                        most_score_information_data_result = json.load(f)
                                        most_score_information_data_result_plate_recognizer = most_score_information_data_result.copy()
                                        most_score_information_data_result_plate_recognizer = \
                                        most_score_information_data_result_plate_recognizer["PlateRecognizer"]
                                        #### Plate recognizer company
                                        plate_recognizer_current_frame_js_information = data_to_write[
                                            "PlateRecognizer"].copy()
                                        ######## license plate information
                                        ## Detection plate
                                        try:
                                            if float(plate_recognizer_current_frame_js_information["plate"]["score"]) > \
                                                    most_score_information_data_result_plate_recognizer["plate"][
                                                        "score"]:
                                                most_score_information_data_result_plate_recognizer["plate"][
                                                    "score"] = float(
                                                    plate_recognizer_current_frame_js_information["plate"]["score"])
                                        except:
                                            pass
                                        try:
                                            if float(plate_recognizer_current_frame_js_information["plate"]["props"][
                                                         "region"][0]["score"]) > \
                                                    most_score_information_data_result_plate_recognizer["plate"][
                                                        "props"]["region"]["score"]:
                                                most_score_information_data_result_plate_recognizer["plate"]["props"][
                                                    "region"]["score"] = \
                                                plate_recognizer_current_frame_js_information["plate"]["props"][
                                                    "region"][0]["score"]
                                                most_score_information_data_result_plate_recognizer["plate"]["props"][
                                                    "region"]["value"] = \
                                                plate_recognizer_current_frame_js_information["plate"]["props"][
                                                    "region"][0]["value"]
                                        except:
                                            pass

                                        try:
                                            ### Reading plate and save the most readable license plate
                                            if float(plate_recognizer_current_frame_js_information["plate"]["props"][
                                                         "plate"][0]["score"]) >= \
                                                    most_score_information_data_result_plate_recognizer["plate"][
                                                        "props"]["plate"]["score"]:
                                                # license_plate_color = ""
                                                try:
                                                    box = plate_recognizer_current_frame_js_information["plate"]["box"]
                                                    # current_frame_original_copy
                                                    xmin = box["xmin"]
                                                    ymin = box["ymin"]
                                                    xmax = box["xmax"]
                                                    ymax = box["ymax"]
                                                    current_id_crop_image = current_frame_original_copy[bbox[1]:bbox[3],
                                                                            bbox[0]:bbox[2]]
                                                    license_plate_numpy_only = current_id_crop_image[ymin:ymax,
                                                                               xmin:xmax, ...]

                                                    img = Image.fromarray(cv2.cvtColor(license_plate_numpy_only, cv2.COLOR_BGR2RGB))
                                                    # Save the image to disk
                                                    crop_image_file_name_path = save_dir / 'crops' / txt_file_name / \
                                                                                names[c] / f'{id}' / "license_plate.jpg"
                                                    img.save(crop_image_file_name_path)

                                                    license_plate_most_common_color = most_common_color(
                                                        license_plate_numpy_only)

                                                except Exception:
                                                    pass

                                                most_score_information_data_result_plate_recognizer["plate"]["props"][
                                                    "plate"]["score"] = \
                                                plate_recognizer_current_frame_js_information["plate"]["props"][
                                                    "plate"][0]["score"]
                                                most_score_information_data_result_plate_recognizer["plate"]["props"][
                                                    "plate"]["value"] = \
                                                plate_recognizer_current_frame_js_information["plate"]["props"][
                                                    "plate"][0]["value"]

                                            ########## Vehicle information
                                            ### recognize vehicle type
                                        except:
                                            pass
                                        try:
                                            if float(
                                                    plate_recognizer_current_frame_js_information["vehicle"]["score"]) > \
                                                    most_score_information_data_result_plate_recognizer["vehicle"][
                                                        "score"]:
                                                most_score_information_data_result_plate_recognizer["vehicle"][
                                                    "score"] = float(
                                                    plate_recognizer_current_frame_js_information["vehicle"]["score"])
                                                most_score_information_data_result_plate_recognizer["vehicle"]["type"] = \
                                                plate_recognizer_current_frame_js_information["vehicle"]["type"]
                                            ### vehicle make model
                                        except:
                                            pass
                                        try:
                                            if float(plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                         "make_model"][0]["score"]) > \
                                                    most_score_information_data_result_plate_recognizer["vehicle"][
                                                        "props"]["make_model"]["score"]:
                                                most_score_information_data_result_plate_recognizer["vehicle"]["props"][
                                                    "make_model"]["score"] = float(
                                                    plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                        "make_model"][0]["score"])
                                                most_score_information_data_result_plate_recognizer["vehicle"]["props"][
                                                    "make_model"]["make"] = \
                                                plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                    "make_model"][0]["make"]
                                                most_score_information_data_result_plate_recognizer["vehicle"]["props"][
                                                    "make_model"]["model"] = \
                                                plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                    "make_model"][0]["model"]

                                        except:
                                            pass
                                        try:
                                            ### vehicle orientation
                                            if float(plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                         "orientation"][0]["score"]) > \
                                                    most_score_information_data_result_plate_recognizer["vehicle"][
                                                        "props"]["orientation"]["score"]:
                                                most_score_information_data_result_plate_recognizer["vehicle"]["props"][
                                                    "orientation"]["score"] = \
                                                plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                    "orientation"][0]["score"]
                                                most_score_information_data_result_plate_recognizer["vehicle"]["props"][
                                                    "orientation"]["value"] = \
                                                plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                    "orientation"][0]["value"]
                                        except:
                                            pass
                                        try:
                                            ### Vehicle color
                                            if float(plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                         "color"][0]["score"]) > \
                                                    most_score_information_data_result_plate_recognizer["vehicle"][
                                                        "props"]["color"]["score"]:
                                                most_score_information_data_result_plate_recognizer["vehicle"]["props"][
                                                    "color"]["score"] = \
                                                plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                    "color"][0]["score"]
                                                most_score_information_data_result_plate_recognizer["vehicle"]["props"][
                                                    "color"]["value"] = \
                                                plate_recognizer_current_frame_js_information["vehicle"]["props"][
                                                    "color"][0]["value"]
                                        except:
                                            pass

                                    #### Write and update most score information result #####
                                    most_score_information_data_result[
                                        "PlateRecognizer"] = most_score_information_data_result_plate_recognizer

                                    with open(current_object_id_most_score_information_vehicle_file_name, "w") as f:
                                        most_score_information_data_result = json.dumps(
                                            most_score_information_data_result)

                                        f.write(most_score_information_data_result)

                                    ###### Save segmentaion as pt file tensor #####
                                    string_to_save = save_dir / 'crops' / txt_file_name / names[
                                        c] / f'{id}' / f'{frame_idx}.pt'
                                    torch.save(current_id_segmentation_mask_upsampled_crop.cpu(), string_to_save)

                else:  # NO DETECTIONS
                    pass
                    # tracker_list[detection_object_index].tracker.pred_n_update_all_trucks()

                ### Stream results to CV2 window: ###
                current_frame_original = annotator.result()
                if show_vid:
                    if platform.system() == 'Linux' and path not in windows:
                        windows.append(path)
                        # Convert the OpenCV BGR image to an RGB image


                        # fig, ax = init_window(str(path))
                        # Display the frame using matplotlib

                        cv2.namedWindow(str(path),
                                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        # cv2.resizeWindow(str(path), current_frame_original.shape[1], current_frame_original.shape[0])

                    ### Plot line via mouse on the first frame ###
                    if frame_idx == 0:
                        if first_frame_flag:

                            # draw_line_widget = DrawPolygon(original_image=first_frame, image_name_window=path)
                            print(
                                "Please plot line by clicking two pint via the mouse,\n please type Enter to save the coordinates")
                            # while True:
                            #     if first_frame_flag:
                            #         cv2.imshow(str(path), draw_line_widget.show_image())
                            #         key = cv2.waitKey(1)
                            #         # save results and continue by enter
                            #         if key == 13:
                            #             list_of_polygons_traffic_coordinates = draw_line_widget.get_list_of_polygons()
                            #             list_of_polygons_counters = [VehicleCounter() for _ in range(len(list_of_polygons_traffic_coordinates))]
                            #             break
                            #     else:
                            #         print("Failed to read first frame for plot lines, exiting....")
                            #         exit(1)
                        else:
                            print("Can not read first frame for plot lines, exiting")
                            exit(1)
                    else:
                        #   Plot the polygons after the first frame for each frame
                        for polygon in list_of_polygons_traffic_coordinates:
                            index_polygon = 0
                            while index_polygon < len(polygon) - 1:
                                cv2.line(current_frame_original, polygon[index_polygon], polygon[index_polygon + 1],
                                         (36, 255, 12), 2)
                                index_polygon += 1

                        #### Plot total counting
                        # total_counting_text = f"Total Vehicles: {total_vehicles}    |    Total trucks:{total_trucks}"
                        #
                        # x_plot_text, y_plot_text, w_plot_text, h_plot_text = 0, 0, 400, 250
                        # # Create background rectangle with color
                        #
                        # # for idx in list_of_polygons_counters:
                        # cv2.rectangle(image_result_counters, (x_plot_text, x_plot_text), (x_plot_text + w_plot_text * 10, y_plot_text + h_plot_text), (0, 0, 0), -1)
                        # cv2.putText(current_frame_original, text=total_counting_text, org=(x_plot_text + int(w_plot_text / 10), y_plot_text + int(h_plot_text / 1.5)),
                        #             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255, 0, 0), thickness=5)

                        # cv2.imwrite(f"{base_frames_saved_directory}/frame_ind_{frame_idx}.jpg", current_frame_original)

                        # update_window(fig, ax, current_frame_original)
                        # update_window_cv(current_frame_original)
                        # imshow_torch_video(numpy_to_torch(current_frame_original))
                        cv2.imshow(str(path), current_frame_original)
                        # imshow_torch_video(current_frame_original)
                        # plt.imshow(current_frame_original)
                        #
                        # plt.pause(2)


                        # if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                        #     exit()

                ### Save results (image with detections): ###
                if save_vid:
                    if vid_path[detection_object_index] != save_path:  # new video
                        vid_path[detection_object_index] = save_path
                        if isinstance(vid_writer[detection_object_index], cv2.VideoWriter):
                            vid_writer[detection_object_index].release()  # release previous video writer
                        if video_capture_object:  # video
                            fps = video_capture_object.get(cv2.CAP_PROP_FPS)
                            w = int(video_capture_object.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(video_capture_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, current_frame_original.shape[1], current_frame_original.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[detection_object_index] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                                             fps, (w, h))
                    vid_writer[detection_object_index].write(current_frame_original)

                prev_frames[detection_object_index] = curr_frames[detection_object_index]

            ### Print total time (preprocessing + inference + NMS + tracking): ###
            LOGGER.info(
                f"{print_string}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

            with open("log_file.log", "w+") as f:
                f.write(
                    f"{print_string}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

        ### Print results: ###
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_vid:
            print_string = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{print_string}")
        if update:
            print("Not supported continue....")
            # strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning) #ToDo





def main():
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    imgsz = [640, 640]
    tracking_method = 'bytetrack'  # help='strongsort, ocsort, bytetrack'
    tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # EILON-COMBINED
    # classes = [0] # person only
    # classes = [1, 2, 3, 5, 7]  #YOLO-REGULAR (....)
    # classes = [1, 2, 3, 5, 7, 0] #YOLO-REGULAR (....)with people
    source = "/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/scene_0.mp4"
    run(source=source, save_crop=True, classes=classes, imgsz=imgsz, tracking_method=tracking_method,
        tracking_config=tracking_config, show_vid=False, save_vid=False, save_txt=True,save_conf=True, save_trajectories=True
        )


def interface_run(source,
                  input_reid_algorithm,
                  input_yolo_weights,
                  input_show_realtime,
                  input_tracker_method,
                  input_cars_type_list,
                  input_cars_orientation_list,
                  input_cars_color_list,
                  input_vehicle_make,
                  input_vehicle_model,
                  input_plate_color,
                  input_plate_regex,
                  input_save_txt,
                  input_save_conf,
                  input_save_crop,
                  input_save_trajectories,
                  input_save_vid,
                  input_clean_data_set,
                  input_create_crop_videos,
                  export_results,
                  input_vehicle_confidence):

    user_input_object_terms = UserInputOptionalTerms(input_user_vehicle_color=input_cars_color_list,
                                                    input_user_vehicle_make_model_make=input_vehicle_make,
                                                    input_user_vehicle_make_model_model=input_vehicle_model,
                                                    input_user_vehicle_type=input_cars_type_list,
                                                    input_user_vehicle_orientation=input_cars_orientation_list
                                                    )
    run(
        user_input_object_terms=user_input_object_terms,
        source=source,
        yolo_weights=WEIGHTS / input_yolo_weights,
        reid_weights=WEIGHTS / input_reid_algorithm,
        tracking_method=input_tracker_method,
        tracking_config=ROOT / 'trackers' / input_tracker_method / 'configs' / (input_tracker_method + '.yaml'),
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=input_vehicle_confidence,  # confidence threshold
        iou_thres=0.50,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=input_show_realtime,  # show results
        save_txt=input_save_txt,  # save results to *.txt
        save_conf=input_save_conf,  # save confidences in --save-txt labels
        save_crop=input_save_crop,  # save cropped prediction boxes
        save_trajectories=input_save_trajectories,  # save trajectories for each track
        save_vid=input_save_vid,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos #ToDo understand it
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ,  # filter by class: --class 0, or --class 0 2 3 #ToDo if using our
        project_path_to_save_results=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=2000,  # video frame-rate stride
        retina_masks=False,
    )

    print("!!!!!! Post-processing...  !!!!!!")
    time.sleep(1)
    base_crop_directory_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/video_example_from_security_camera.mp4"
    post_processing(only_best_result=input_clean_data_set, create_crop_movie=input_create_crop_videos,base_crop_directory_path=base_crop_directory_path)


# so when i want to update the table, what should i do ?
def post_processing(only_best_result=False, only_user_input_terms=False, create_crop_movie=False, base_crop_directory_path=None):

    if base_crop_directory_path is not None:
        if create_crop_movie is True:
            # Loop over all directories in the base directory
            for dir_name in os.listdir(base_crop_directory_path):
                if "ex" not in dir_name:
                    continue
                dir_path = os.path.join(base_crop_directory_path, dir_name)
                # Check if the item is a directory
                if os.path.isdir(dir_path):
                    # Loop over all directories in the current directory
                    for sub_dir_name in os.listdir(dir_path):
                        sub_dir_path = os.path.join(dir_path, sub_dir_name)
                        # Check if the sub-item is a directory
                        if os.path.isdir(sub_dir_path):
                            # Loop over all directories in the current sub-directory
                            for sub_sub_dir_name in os.listdir(sub_dir_path):
                                sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir_name)
                                # Check if the sub-sub-item is a directory
                                if os.path.isdir(sub_sub_dir_path):
                                    # Loop over all directories in the current sub-sub-directory
                                    for sub_sub_sub_dir_name in os.listdir(sub_sub_dir_path):
                                        sub_sub_sub_dir_path = os.path.join(sub_sub_dir_path, sub_sub_sub_dir_name)
                                        # Check if the sub-sub-sub-item is a directory
                                        if os.path.isdir(sub_sub_sub_dir_path):
                                            # Print the directory path
                                            list_of_jpg_files = []
                                            for file in os.listdir(sub_sub_sub_dir_path):
                                                if file != "license_plate.jpg" and file.endswith(".jpg"):
                                                    list_of_jpg_files.append(os.path.join(sub_sub_sub_dir_path, file))
                                            # print(list_of_jpg_files)
                                            video_from_tensor(list_of_files=list_of_jpg_files, output_video=f"{sub_sub_sub_dir_path}/TYPE_{dir_name}_ID_{sub_sub_sub_dir_name}.mp4", size=[256,256], file_type=".jpg")
                                            create_video_from_list_of_images_path(list_of_jpg_files, output_video_name=f"TYPE_{dir_name}_ID_{sub_sub_sub_dir_name}.mp4", output_directory_path=sub_sub_sub_dir_path)
                                            ## For each object detection, build a video

    if only_best_result is True:
        if base_crop_directory_path is not None:
            # Loop over all directories in the base directory
            for dir_name in os.listdir(base_crop_directory_path):
                if "ex" not in dir_name:
                    continue
                dir_path = os.path.join(base_crop_directory_path, dir_name)
                # Check if the item is a directory
                if os.path.isdir(dir_path):
                    # Loop over all directories in the current directory
                    for sub_dir_name in os.listdir(dir_path):
                        sub_dir_path = os.path.join(dir_path, sub_dir_name)
                        # Check if the sub-item is a directory
                        if os.path.isdir(sub_dir_path):
                            # Loop over all directories in the current sub-directory
                            for sub_sub_dir_name in os.listdir(sub_dir_path):
                                sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir_name)
                                # Check if the sub-sub-item is a directory
                                if os.path.isdir(sub_sub_dir_path):
                                    # Loop over all directories in the current sub-sub-directory
                                    for sub_sub_sub_dir_name in os.listdir(sub_sub_dir_path):
                                        sub_sub_sub_dir_path = os.path.join(sub_sub_dir_path, sub_sub_sub_dir_name)
                                        # Check if the sub-sub-sub-item is a directory
                                        if os.path.isdir(sub_sub_sub_dir_path):
                                            # Print the directory path
                                            list_of_jpg_files = []
                                            for file in os.listdir(sub_sub_sub_dir_path):
                                                if file != "license_plate.jpg" and file != "most_score_information.json" and "mp4" not in file:
                                                    # Use os.remove() to delete the file
                                                    os.remove(os.path.join(sub_sub_sub_dir_path, file))

                                            ### For each object detection, build a video

    if only_user_input_terms is True:
        print("only_user_input_terms")



main()
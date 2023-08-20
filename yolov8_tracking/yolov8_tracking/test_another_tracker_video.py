import argparse
import glob
import pathlib

import cv2
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy

from SHABACK_POC_NEW.util.draw_line_multy_using_mouse import DrawPolygon
from SHABACK_POC_NEW.util.draw_line_using_mouse_cv2 import DrawLineWidget

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, \
    process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker


class VehicleCounter:
    def __init__(self):
        self.counter_truck = 0
        self.counter_car = 0
        self.counter_bus = 0

def chw_2_hwc(input_array):
    return input_array.transpose([1,2,0])

def get_center_x_y(x, y, w, h):

    x1 = int(w / 2)

    y1 = int(h / 2)

    cx = x + x1

    cy = y + y1

    return (cx, cy)


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
        source='0',
        yolo_weights=WEIGHTS / "yolov8l-seg.pt" , # "best_3.pt", #"best_1.pt",    #'yolov8s-seg.pt', #"best_1.pt", # model.pt path(s),  # /home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/weights/yolov8s-seg.pt
        # yolo_weights= pathlib.Path("/home/dudy/Downloads/last.pt"), # 'yolov8s-seg.pt',  # model.pt path(s),  # /home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/weights/yolov8s-seg.pt
        # yolo_weights=WEIGHTS / "best_3.pt",

        reid_weights=WEIGHTS / 'osnet_x0_25_market1501.pt',  # model.pt path,
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
    exp_name = "____all_input_____"
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
    outputs = [None] * bs

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

    frame_stride = 100
    ### Loop over frames: ###
    for frame_idx, batch in enumerate(dataset):
        print(frame_idx)
        ### Get current image path, image after scaling, original image etc': ###
        path, current_frame_after_downsample, current_frame_original_size, video_capture_object, print_string = batch

        # ### Frame striding (in case of high FPS, or i just want to skip images because we don't need prediction for every single frame): ###
        # if frame_idx > -1:
        #     for _ in np.arange(frame_stride):
        #         flag_frame_available, frame = video_capture_object.read()


        ### Visualize: ###
        # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False

        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:  #TODO: understand what the different context decorators do here
            current_frame_after_downsample = torch.from_numpy(current_frame_after_downsample).to(device)
            current_frame_after_downsample = current_frame_after_downsample.half() if half else current_frame_after_downsample.float()  # uint8 to fp16/32
            current_frame_after_downsample /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(current_frame_after_downsample.shape) == 3:
                current_frame_after_downsample = current_frame_after_downsample[None]  # expand for batch dim

        if frame_idx == 0:
            first_frame_flag, first_frame = video_capture_object.read()

        ### Inference: ###
        with dt[1]:
            #TODO: understand what is the model output
            preds = model(current_frame_after_downsample, augment=flag_augment, visualize=visualize)

        ### Apply NMS: ###
        with dt[2]:
            if flag_perform_segmentation:
                masks = []  #TODO: rename p
                prediction_outputs = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]  #TODO: what is proto?
            else:
                prediction_outputs = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        ### Loop over detections detections: ###  #TODO: probably rename i to detection_object_counter or something
        for i, det in enumerate(prediction_outputs):  # detections per image
            # print(prediction_outputs)
            seen += 1
            ### Take care of paths according to source: ###
            if webcam:  # bs >= 1
                path, current_frame_original, _ = path[i], current_frame_original_size[i].copy(), dataset.count
                path = Path(path)  # to Path
                print_string += f'{i}: '
                txt_file_name = path.name
                save_path = str(save_dir / path.name)  # im.jpg, vid.mp4, ...
            else:
                if frame_idx == 59:
                    bla = 1
                ### Get ?????: ###
                path, current_frame_original, _ = path, current_frame_original_size.copy(), getattr(dataset, 'frame', 0)
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
            curr_frames[i] = current_frame_original

            ### TODO: what is being saved? ###
            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            print_string += '%gx%g ' % current_frame_after_downsample.shape[2:]  # print string
            current_frame_original_copy = current_frame_original.copy() if save_crop else current_frame_original  # for save_crop

            ### TODO: is this responsible for detection drawing or for segmentation: ###
            annotator = Annotator(current_frame_original, line_width=line_thickness, example=str(names))

            ### Compensate for camera motion: ###
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            ### If we have a detection - do something: ### #TODO: understand what?!?!?
            # seems like det consists of: BB[0],BB[1],BB[2],BB[3],Conf(?),Class, 32 numbers representing numbers to be matrix multiplied by proto (segmentation mask) to be inputed into a sigmoid and output final mask
            if det is not None and len(det):
                if flag_perform_segmentation:
                    shape = current_frame_original.shape
                    ### scale bbox first the crop masks: ###
                    if retina_masks:
                        det[:, :4] = scale_boxes(current_frame_after_downsample.shape[2:], det[:, :4], shape).round()  # rescale boxes to current_frame_original size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], current_frame_original.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], current_frame_after_downsample.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(current_frame_after_downsample.shape[2:], det[:, :4], shape).round()  # rescale boxes to current_frame_original size
                else:
                    det[:, :4] = scale_boxes(current_frame_after_downsample.shape[2:], det[:, :4], current_frame_original.shape).round()  # rescale boxes to current_frame_original size

                ### Print results: ###
                for output_class in det[:, 5].unique():
                    number_of_detection_per_this_class = (det[:, 5] == output_class).sum()  # detections per class
                    print_string += f"{number_of_detection_per_this_class} {names[int(output_class)]}{'s' * (number_of_detection_per_this_class > 1)}, "  # add to string  #TODO: understand if 's' should be changed to 'pring_string'

                ### TRACK bounding boxes using tracker: ###
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), current_frame_original)

                ### Draw boxes for visualization: ###
                if len(outputs[i]) > 0:
                    ### Plot segmentation on image using annotator object: ###
                    if flag_perform_segmentation:
                        ### Segmentation masks: ###
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(current_frame_original, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                                   255 if retina_masks else current_frame_after_downsample[i]
                        )

                    ### Loop over detection outputs: ###
                    for j, (output) in enumerate(outputs[i]):
                        ### Assign detector model outputs to variable names: ###
                        if frame_idx == 59:
                            bla = 1
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        ### Correct for overflow or underflow in shape: ###
                        bbox = correct_over_under_flow_in_BB(bbox, current_frame_original.shape[0:2])

                        # centerx, centery = (numpy.average(bbox[:2]), numpy.average(bbox[2:]))
                        centerx,centery = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
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

                        ### Loop over the different polygons defined by the user and check whether this car id has been there, and if not than update it: ###
                        #TODO: make this!!!!

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
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        ### If there's anything to save or to show - do it: ###
                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            ### Assign detector outputs to variable names: ###
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                  (
                                                                      f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)

                            ### Actually draw the bounding box: ###
                            annotator.box_label(bbox, label, color=color)

                            ### Save tracking outputs to list: ###
                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]  #TODO: what is q????
                                tracker_list[i].trajectory(current_frame_original, q, color=color)

                            ### Save bounding box crop if wanted: ###
                            if save_crop:
                                #TODO: understand what's being saved here and how for shabak
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                bbox = np.array(bbox, dtype=np.int16)
                                crop = current_frame_original_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                resized_crop = cv2.resize(crop, (224, 224))
                                h, w, _ = resized_crop.shape
                                new_bb = [0, 0, w, h]
                                save_one_box(np.array(new_bb, dtype=np.int16), resized_crop,
                                             file=save_dir / 'crops' / txt_file_name / "dudy" / "for_dudy" / f'{path.stem}.jpg', BGR=True)

                                # save_one_box(np.array(bbox, dtype=np.int16), current_frame_original_copy,
                                #              file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{path.stem}.jpg', BGR=True)



            else: #NO DETECTIONS
                pass
                # tracker_list[i].tracker.pred_n_update_all_trucks()

            ### Stream results to CV2 window: ###
            current_frame_original = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and path not in windows:
                    windows.append(path)
                    cv2.namedWindow(str(path), cv2.WINDOW_NORMAL)  # allow window resize (Linux)
                    cv2.resizeWindow(str(path), current_frame_original.shape[1], current_frame_original.shape[0])


                ### Plot line via mouse on the first frame ###
                if frame_idx == 0:
                    if first_frame_flag:
                        draw_line_widget = DrawPolygon(original_image=first_frame, image_name_window=path)
                        print("Please plot line by clicking two pint via the mouse,\n please type Enter to save the coordinates")
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
                        while index_polygon < len(polygon) -1:
                           cv2.line(current_frame_original, polygon[index_polygon], polygon[index_polygon+1], (36, 255, 12), 2)
                           index_polygon += 1
                    total_counting_text = f"Total Vehicles: {total_vehicles}    |    Total trucks:{total_trucks}"

                    x_plot_text, y_plot_text, w_plot_text, h_plot_text = 0, 0, 400, 250
                    # Create background rectangle with color

                    # for idx in list_of_polygons_counters:
                    cv2.rectangle(image_result_counters, (x_plot_text, x_plot_text), (x_plot_text + w_plot_text * 10, y_plot_text + h_plot_text), (0, 0, 0), -1)
                    cv2.putText(current_frame_original, text=total_counting_text, org=(x_plot_text + int(w_plot_text / 10), y_plot_text + int(h_plot_text / 1.5)),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255, 0, 0), thickness=5)

                    cv2.imshow(str(path), current_frame_original)



                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            ### Save results (image with detections): ###
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if video_capture_object:  # video
                        fps = video_capture_object.get(cv2.CAP_PROP_FPS)
                        w = int(video_capture_object.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(video_capture_object.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, current_frame_original.shape[1], current_frame_original.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(current_frame_original)

            prev_frames[i] = curr_frames[i]

        ### Print total time (preprocessing + inference + NMS + tracking): ###
        LOGGER.info(
            f"{print_string}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    ### Print results: ###
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print_string = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{print_string}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning) #ToDo


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt',
                        help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--flag_augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project_path_to_save_results', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main():
    # base_directory_videos = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel"
    # list_of_target_videos = glob.glob(f"{base_directory_videos}/*")
    # for video_path in list_of_target_videos:
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/video_example_from_security_camera.mp4"
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/video_from_security_cmera_dark_mode.mp4"
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/ch01_00000000016000000.mp4"
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/ch01_00000000059000100.mp4"
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/ch01_00000000093000000.mp4"
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/ch01_00000000161000200.mp4"
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/MEHLAF_NAHAL_RAVID_2/ch01_00000000011000000.mp4"
    source= "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/ch01_00000000161000200.mp4"
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    # run(**vars(opt))
    imgsz = [640, 640]
    # imgsz *= 2 if len(imgsz) == 1 else 1
    # source = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/ch01_00000000027000000.mp4"
    # source = video_path
    tracking_method = 'bytetrack' #help='strongsort, ocsort, bytetrack'
    tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
    # classes = [0, 1, 2] #EILON-COMBINED (car, big-car, motorcycle)
    classes = [0,1,2,3,4,5,6,7] #YOLO-REGULAR (....)
    try:
        run(source=source, save_crop=True, classes=classes, imgsz=imgsz, tracking_method=tracking_method, tracking_config=tracking_config)
    except Exception as e:
        pass


if __name__ == "__main__":
    # opt = parse_opt()
    main()

import argparse
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import sys
# sys.path.append("/home/mafat/Desktop/eylon/RDND_proper")
from RapidBase.import_all import *
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from draw_polygons import *
from collections import defaultdict
matplotlib.use('TkAgg')
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
# sys.path.append("/home/mafat/Desktop/eylon/ultralytics")
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker
from pathlib import Path
import csv
# from post_process import get_statistics

class VehicleCounter:
    def __init__(self, model):
        self.class_dict = model.model.names
        self.counter_dict = defaultdict(int)
        self.polygon_lane = 0
        self.is_start_polygon = False
        self.id_list_passed_in_polygon = []


def is_video_file(file_path):
    # Check if the file extension is a video format
    video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv']
    if any(file_path.lower().endswith(ext) for ext in video_extensions):
        return True


def make_output_folder(path):
    if not os.path.exists(path):
        Path.mkdir(Path(path), parents=True)


def list_to_csv_column(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in data:
            writer.writerow([item])


def find_common_elements(id_list_1, id_list_2):
    set2 = set(id_list_2)
    common_elements = [element for element in id_list_1 if element in set2]
    return common_elements


def put_text_on_image(image, text, font_tpye = 0,font_size = 1, color=(0,0,0),line_thickness =1,  origin= (0,25)):
    image = cv2.putText(image, text, origin, font_tpye, font_size, color, line_thickness, cv2.LINE_AA)
    return image
@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='botsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
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
        video_path="/home/mafat/Desktop/eylon/outputs/test_config/output.mp"
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    list_of_counted_vehicles_by_id = []
    print(video_path)
    list_of_polygons_traffic_coordinates = []
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = os.path.split(yolo_weights)[-1].split(".")[0]
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size
    # print(opt.source)
    # base_dir = "/media/mafat/My_Passport/txt_regev/{}".format(os.path.split(source)[1].split('.')[0])
    base_dir = f"/home/dudy/txt_for_regev/{os.path.split(source)[1].split('.')[0]}"
    make_output_folder("{}/txt_files".format(base_dir))
    saved_frame_list = ""
    # Dataloader
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
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, imgsz[0], imgsz[1]))

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    for frame_idx, batch in enumerate(dataset):
        path, im, original_image, vid_cap, s = batch
        visualize = increment_path(base_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            if frame_idx == 0:
                if is_video_file(source):
                    first_frame_flag, first_frame = vid_cap.read()
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

        if frame_idx % 30 == 0:
            saved_frame_path = saved_frame_list[frame_idx]
            predictions = np.loadtxt(saved_frame_path)
            predictions = predictions[:, 2:6]
            predictions[:, 2] += predictions[:, 0]
            predictions[:, 3] += predictions[:, 1]

        else:
        # Inference
            with dt[1]:
                preds = model(im, augment=augment, visualize=visualize)

            # Apply NMS
            with dt[2]:
                if is_seg:
                    masks = []
                    predictions = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                    proto = preds[1][-1]
                else:
                    predictions = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for i, det in enumerate(predictions):  # detections per image
            seen += 1
            if webcam:  # bs >= 1
                path, im0, _ = path[i], original_image[i].copy(), dataset.count
                path = Path(path)  # to Path
                s += f'{i}: '
                txt_file_name = path.name
                save_path = str(base_dir / path.name)  # im.jpg, vid.mp4, ...
            else:
                path, im0, _ = path, original_image.copy(), getattr(dataset, 'frame', 0)
                path = Path(path)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = path.stem
                    save_path = str(base_dir)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = path.parent.name  # get folder name containing current img
                    save_path = str(base_dir)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = f"{base_dir}/txt_files/tracks/{txt_file_name}"
            make_output_folder(txt_path)
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                # draw boxes for visualization
                if len(outputs[i]) > 0:

                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )
                    # tic()
                    for j, (output) in enumerate(outputs[i]):

                        bbox = np.asarray(output[0:4], dtype=np.int16)
                        bbox = [0 if num < 0 else num for num in bbox]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        id = int(output[4])
                        cls = output[5]
                        conf = output[6]
                        numpy_image_rgb = original_image
                        numpy_image_rgb_roi = put_text_on_image(numpy_image_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]], str(int(cls)))
                        centerx, centery = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                        cv2.circle(im0, (int(centerx), int(centery)), 8, (0, 0, 255), -1)
                        center_point_as_point_obj = Point(centerx, centery)
                        if not cls == 9:
                            # if id not in list_of_counted_vehicles_by_id:
                            list_of_counted_vehicles_by_id.append(id)
                            for pol_index, polygon in enumerate(list_of_polygons_traffic_coordinates):
                                if Polygon(list(polygon)).contains(center_point_as_point_obj) and id not in list_of_polygons_counters[pol_index].id_list_passed_in_polygon:
                                    # flag_id_new_passed[pol_index] = True
                                    current_class = model.model.names[cls]
                                    list_of_polygons_counters[pol_index].counter_dict[current_class] += 1
                                    list_of_polygons_counters[pol_index].id_list_passed_in_polygon.append(id)
                                    print(list_of_polygons_counters[pol_index].counter_dict.items())
                            if id in list_of_counted_vehicles_by_id:
                                make_output_folder(f"{base_dir}/{id}")
                                plt.imsave(f"{base_dir}/{id}/{frame_idx}.jpg",
                                           cv2.cvtColor(numpy_image_rgb_roi, cv2.COLOR_BGR2RGB))
                                with open(f"{base_dir}/txt_files/{id}_data.txt", 'a') as f:
                                    f.write((('%g ' * 6 + '\n') % (frame_idx, centerx, centery, cls, bbox_w, bbox_h)))
                        # make_output_folder(f"{base_dir}/{id}")
                        # plt.imsave(f"{base_dir}/{id}/{frame_idx}.jpg",  cv2.cvtColor(numpy_image_rgb_roi, cv2.COLOR_BGR2RGB))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(f"{txt_path}/{frame_idx}.txt", 'a') as f:
                                f.write(('%g ' * 7 + '\n') % (frame_idx + 1, cls, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, id))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)

                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=base_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{path.stem}.jpg', BGR=True)

                    # toc("iterate over polygons")

            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()

            # Stream results
            current_frame_original = annotator.result()


            if show_vid:
                if platform.system() == 'Linux' and path not in windows:
                    windows.append(path)
                    # fig = plt.figure(str(path))
                    cv2.namedWindow(str(path),
                                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(path), current_frame_original.shape[1], current_frame_original.shape[0])
                    make_output_folder(f"{base_dir}/annotated_frames")
                ### Plot line via mouse on the first frame ###
                if frame_idx == 0:
                    if first_frame_flag:
                        plt.imsave(f"{base_dir}/first_frame.jpg", cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                        plt.imsave(f"{base_dir}/annotated_frames/{frame_idx}.jpg",
                                   cv2.cvtColor(current_frame_original, cv2.COLOR_BGR2RGB))

                        draw_line_widget = DrawPolygon(original_image=original_image, image_name_window=path)
                        print(
                            "Please plot line by clicking two pint via the mouse,\n please click right mouse button to "
                            "save polygon,\nand Enter to finish selecting bounding boxes coordinates")
                        while True:
                            if first_frame_flag:
                                # cv2.imshow(str(path), draw_line_widget.show_image())
                                # key = cv2.waitKey(1)
                                # save results and continue by enter
                                # list_of_polygons_traffic_coordinates = np.load("/home/mafat/Desktop/eylon/outputs/polygon_list.npy").tolist()
                                # list_of_polygons_numbers_coordinates = np.load("/home/mafat/Desktop/eylon/outputs/polygon_list_numbers.npy").tolist()
                                # flag_id_new_passed = []
                                # for pol_index in range(len(list_of_polygons_numbers_coordinates)):
                                #     flag_id_new_passed.append(False)
                                # if key == 13:
                                if True:
                                #     list_of_polygons_traffic_coordinates = draw_line_widget.get_list_of_polygons()
                                    list_of_polygons_numbers_coordinates = draw_line_widget.get_list_of_polygons_numbers()
                                    if not list_of_polygons_traffic_coordinates:
                                        h, w, c = current_frame_original.shape
                                        list_of_polygons_traffic_coordinates = [[(0, 0), (w, 0), (w, h), (0, h)]]
                                    else:
                                        print(list_of_polygons_traffic_coordinates)
                                    list_of_polygons_counters = [VehicleCounter(model) for _ in range(len(list_of_polygons_traffic_coordinates))]
                                    # for pol_index, polygon in enumerate(list_of_polygons_counters):
                                    #     list_of_polygons_counters[pol_index].polygon_lane = list_of_polygons_numbers_coordinates
                                    break
                            else:
                                print("Failed to read first frame for plot lines, exiting....")
                                exit(1)
                    else:
                        print("Can not read first frame for plot lines, exiting")
                        exit(1)
                else:
                    #   Plot the polygons after the first frame for each frame
                    if frame_idx % 25 == 0:
                        plt.imsave(f"{base_dir}/annotated_frames/{frame_idx}.jpg",  cv2.cvtColor(current_frame_original, cv2.COLOR_BGR2RGB))
                    for pol_number, polygon in enumerate(list_of_polygons_traffic_coordinates):
                        index_polygon = 0
                        # if flag_id_new_passed[pol_number]:
                        #     flag_id_new_passed[pol_number] = False
                        #     while index_polygon < len(polygon) - 1:
                        #         cv2.line(current_frame_original, polygon[index_polygon], polygon[index_polygon + 1],
                        #                  (255, 255, 102), 2)
                        #         index_polygon += 1
                        # else:
                        while index_polygon < len(polygon) - 1:
                            cv2.line(current_frame_original, polygon[index_polygon], polygon[index_polygon + 1],
                                        (36, 255, 12), 2)
                            index_polygon += 1

                    #### Plot total counting
                    # total_counting_text = f"Total Vehicles: {total_vehicles}    |    Total trucks:{total_trucks}"
                    #
                    x_plot_text, y_plot_text, w_plot_text, h_plot_text = 0, 0, 100, 100
                    # # Create background rectangle with color
                    #
                    # # for idx in list_of_polygons_counters:
                    # cv2.rectangle(image_result_counters, (x_plot_text, x_plot_text), (x_plot_text + w_plot_text * 10, y_plot_text + h_plot_text), (0, 0, 0), -1)
                    cv2.putText(current_frame_original, text=str(frame_idx), org=(x_plot_text + int(w_plot_text / 10), y_plot_text + int(h_plot_text / 1.5)),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(0, 0, 0), thickness=5)

                    # cv2.imwrite(f"{base_frames_saved_directory}/frame_ind_{frame_idx}.jpg", current_frame_original)

                    cv2.imshow(str(path), current_frame_original)
                    # imshow_torch_video(current_frame_original)
                    # plt.imshow(current_frame_original)
                    #
                    # plt.pause(2)

                    if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                        exit()
            if show_vid:
                if platform.system() == 'Linux' and path not in windows:
                    windows.append(path)
                    cv2.namedWindow(str(path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(path), im0.shape[1], im0.shape[0])
                cv2.imshow(str(path), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, current_frame_original.shape[1], current_frame_original.shape[0]
                    # save_path = str(Path(f"{save_path}/annotated_video").with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    save_path = str(Path(video_path))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    print(video_path)
                    # make_output_folder(video_path)
                vid_writer[i].write(current_frame_original)

            prev_frames[i] = curr_frames[i]

        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

    # lane_list_and_id_in_them = []
    # ### testing ###
    # compared_pairs = set()
    # for pol_index_1, polygon_1 in enumerate(list_of_polygons_counters):
    #     for pol_index_2, polygon_2 in enumerate(list_of_polygons_counters):
    #         if not pol_index_1 == pol_index_2 and (pol_index_2, pol_index_1) not in compared_pairs and list_of_polygons_counters[pol_index_1].polygon_lane[pol_index_1] == list_of_polygons_counters[pol_index_2].polygon_lane[pol_index_2]:
    #             compared_pairs.add((pol_index_1, pol_index_2))
    #             common_id_in_lane = find_common_elements(list_of_polygons_counters[pol_index_1].id_list_passed_in_polygon, list_of_polygons_counters[pol_index_2].id_list_passed_in_polygon)
    #             lane_list_and_id_in_them.append([list_of_polygons_counters[pol_index_1].polygon_lane[pol_index_1] ,common_id_in_lane])

    # for lane_number in range(len(lane_list_and_id_in_them)):
    #     id_list = np.array(lane_list_and_id_in_them[lane_number][1])
    #     lane_output_folder = os.path.join(base_dir, "lanes")
    #     make_output_folder(lane_output_folder)
    #     np.savetxt(f"{lane_output_folder}/lane_{lane_list_and_id_in_them[lane_number][0]}.txt", id_list)

    # get_statistics(base_dir)


    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_vid:
    #     s = f"\n{len(list((base_dir / 'tracks').glob('*.txt')))} tracks saved to {base_dir / 'tracks'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', base_dir)}{s}")
    # for pol_index, polygon in enumerate(list_of_polygons_traffic_coordinates):
    #     with open("/home/simteam-j/Documents/stam.txt", 'a') as f:
    #         f.write(f"polygon_number: {pol_index}, ({polygon})\n{list(list_of_polygons_counters[pol_index].counter_dict.items())}")
    # if update:
    #     strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8s-seg.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',default=False,  help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
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
    parser.add_argument('--video-path', type=str, default="/home/mafat/Desktop/eylon/outputs/test_config/output.mp4", help='save path video')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    opt.yolo_weights = "trained_yolo.pt"
    opt.source = r"/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/output1.mp4"
    video_name = opt.source.split(".")[0]
    opt.video_path = f"{video_name}/annotated_vid.mp4"
    opt.save_txt = True
    # opt.classes = [0,1,2,3,4,5,6,7,8,10]
    opt.save_vid = True
    main(opt)


# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch
import argparse
import numpy as np
import cv2
from types import SimpleNamespace

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import select_device

tr = TestRequirements()
tr.check_packages(('ultralytics',))  # install

from ultralytics.yolo.engine.model import YOLO, TASK_MAP

from ultralytics.yolo.utils import SETTINGS, colorstr, ops, is_git_dir, IterableSimpleNamespace
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.utils.plotting import save_one_box

from multi_yolo_backend import MultiYolo
from utils import write_MOT_results

from boxmot.utils import EXAMPLES



# Mouse callback function for OpenCV window
points = []  # Store click points
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = \
        ROOT / \
        'boxmot' / \
        predictor.args.tracking_method / \
        'configs' / \
        (predictor.args.tracking_method + '.yaml')
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.device,
            predictor.args.half
        )
        predictor.trackers.append(tracker)



@torch.no_grad()
def run(args):
    model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)

    # extract task predictor
    predictor = model.predictor

    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)
    predictor.args.device = select_device(args['device'])
    LOGGER.info(args)

    # setup source and model
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=False)
    predictor.setup_source(predictor.args.source)

    predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
    predictor.save_dir = increment_path(Path(predictor.args.project) / predictor.args.name,
                                        exist_ok=predictor.args.exist_ok)

    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True,
                                                                                                 exist_ok=True)
    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(
            imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (
    ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
    predictor.add_callback('on_predict_start', on_predict_start)
    predictor.run_callbacks('on_predict_start')
    model = MultiYolo(
        model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
        device=predictor.device,
        args=predictor.args
    )
    for frame_idx, batch in enumerate(predictor.dataset):
        predictor.run_callbacks('on_predict_batch_start')
        predictor.batch = batch
        path, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, exist_ok=True,
                                   mkdir=True) if predictor.args.visualize and (
            not predictor.dataset.source_type.tensor) else False

        n = len(im0s)
        predictor.results = [None] * n

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model(im, im0s)

        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks('on_predict_postprocess_end')

        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):

            if predictor.dataset.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)

            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                # get tracker predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach().numpy(), im0)
            predictor.results[i].speed = {
                'preprocess': predictor.profilers[0].dt * 1E3 / n,
                'inference': predictor.profilers[1].dt * 1E3 / n,
                'postprocess': predictor.profilers[2].dt * 1E3 / n,
                'tracking': predictor.profilers[3].dt * 1E3 / n
            }

            # filter boxes masks and pose results by tracking results
            model.filter_results(i, predictor)
            # overwrite bbox results with tracker predictions
            model.overwrite_results(i, im0.shape[:2], predictor)

            # write inference results to a file or directory
            if predictor.args.verbose or predictor.args.save or predictor.args.save_txt or predictor.args.show or predict.args.save_id_crops:

                s += predictor.write_results(i, predictor.results, (p, im, im0))
                predictor.txt_path = Path(predictor.txt_path)

                # write MOT specific results
                if predictor.args.source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                else:
                    # append folder name containing current img
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name

                if predictor.tracker_outputs[i].size != 0 and predictor.args.save_mot:
                    write_MOT_results(
                        predictor.MOT_txt_path,
                        predictor.results[i],
                        frame_idx,
                        i,
                    )

                if predictor.args.save_id_crops:
                    for d in predictor.results[i].boxes:
                        save_one_box(
                            d.xyxy,
                            im0.copy(),
                            file=predictor.save_dir / 'crops' / str(int(d.cls.cpu().numpy().item())) / str(
                                int(d.id.cpu().numpy().item())) / f'{p.stem}.jpg',
                            BGR=True
                        )
                        #### Save entire frame
                        save_entire_frame_directory_path = Path(predictor.save_dir / "entire_frame" / "base")

                        #### Save entire frame masks
                        save_entire_frame_masks_directory_path = Path(predictor.save_dir / "entire_frame" / "masks")

            # display an image in a window using OpenCV imshow()
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p.parent)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

        predictor.run_callbacks('on_predict_batch_end')

        # print time (inference-only)
        if predictor.args.verbose:
            LOGGER.info(
                f'{s}YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms')

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape '
            f'{(1, 3, *predictor.args.imgsz)}' % t)
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
        s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks('on_predict_end')


def main():
    # /home/nehoray/PycharmProjects/Shaback/data/videos/video_example_from_security_camera.mp4
    video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/video_example_from_security_camera.mp4"
    args = {
        'yolo_model': 'yolov8n.pt',
        'reid_model': 'mobilenetv2_x1_4_dukemtmcreid.pt',
        'tracking_method': 'deepocsort',
        'source': video_path,
        'imgsz': [640],
        'conf': 0.5,
        'iou': 0.7,
        'device': '',
        'show': True,
        'save': True,
        'classes': [0],  # Just an example, you might want to adjust this
        'project': 'runs/track',
        'name': 'exp',
        'exist_ok': True,
        'half': False,
        'vid_stride': 1,
        'hide_label': False,
        'hide_conf': False,
        'save_txt': True,
        'save_id_crops': False,
        'save_mot': True,
        "save_frame" : True,
        "save_mask": True
    }

    run(args)

if __name__ == "__main__":
    main()







import cv2, time, sys, os, time
import numpy as np
import argparse, cv2


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    show_video_intro = "RAW Video Viewer\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=show_video_intro)
    parser.add_argument("-v", "--video_directory_prefix", help="png prefix", required=True)
    parser.add_argument("-f", "--fps", help="fps of video", type=int, default=30)
    parser.add_argument("-s", "--single_frame", help="is a single frame", action='store_true')
    return parser


def exit_on(message: str, status: int = 1):
    # print message, and exit
    print("ERROR: " + message)
    sys.exit(status)


class FrameViewer:
    def __init__(self, title: str ="Video Viewer"):
        self.title = title

    def show_frame(self, frame):
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)


def check_input(arguments: argparse.Namespace) -> None:
    if not os.path.exists(f"{arguments.video_directory_prefix}0.png"):
        exit_on("GIVEN VIDEO FILE DOES NOT EXIST")
    if  arguments.fps < 0:
        exit_on("Negative FPS value")

if __name__ == '__main__' :
    parser = create_parser()
    args = parser.parse_args()
    check_input(args)
    if args.fps > 1000:
        wait=0
    else:
        wait = 1/args.fps
    frame_viewer = FrameViewer()
    current_frame = 0
    next_frame = f"{args.video_file}_{current_frame}.png"
    while os.path.exists(next_frame):
        frame_viewer.show_frame(cv2.imread(next_frame))
        if args.single_frame:
            while True:
                time.sleep(10)
        current_frame+=1
        time.sleep(wait)

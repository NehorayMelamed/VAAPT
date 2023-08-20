import cv2, time, sys, os, time
import numpy as np
import argparse


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    show_video_intro = "RAW Video Viewer\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=show_video_intro)
    parser.add_argument("-v", "--video_prefix", help="RAW video file", required=True)
    parser.add_argument("-H", "--height", help="height of video frames", required=True)
    parser.add_argument("-W", "--width", help="width of video frames", required=True)
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


def load_frame(filename, height=340, width=1024):
    with open(filename, 'rb') as vf:
        file_data = np.fromfile(vf, dtype=np.uint8, count=height*width)
        frames_array = file_data.reshape(height, width)
    return frames_array


def check_input(arguments: argparse.Namespace) -> None:
    if not os.path.exists(f"{arguments.video_prefix}0.raw"):
        exit_on("GIVEN FRAMES DO NOT EXIST")
    if arguments.fps < 0:
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
    next_path = f"{args.video_prefix}0.raw"
    while os.path.exists(next_path):
        frame_viewer.show_frame(load_frame(next_path, int(args.height), int(args.width)))
        if args.single_frame:
            while True:
                time.sleep(10)
        time.sleep(wait)
        current_frame+=1
        next_path = f"{args.video_prefix}{current_frame}.raw"
    print("VIDEO OVER")

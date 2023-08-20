import cv2, time, sys, os, time
import numpy as np
import argparse


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    show_video_intro = "RAW Video Viewer\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=show_video_intro)
    parser.add_argument("-v", "--video_file", help="RAW video file", required=True)
    parser.add_argument("-H", "--height", help="height of video frames", required=True, type=int)
    parser.add_argument("-W", "--width", help="width of video frames", required=True, type=int)
    parser.add_argument("-n", "--nbytes", help="num bytes", type=int, default=1)
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


def get_frames(filename, height=340, width=1024, num_bytes=1, get_single_frame=False):
    if num_bytes == 1:
        dt = np.uint8
    else:
        dt = np.uint16
    with open(filename, 'rb') as vf:
        if not get_single_frame:
            file_data = np.fromfile(vf, dtype=dt)
            max_frames = int(len(file_data) / height / width)
            frames_array = file_data.reshape(max_frames, height, width)
        else:
            file_data = np.fromfile(vf, dtype=np.uint8, count=height*width)
            frames_array = file_data.reshape(1, height, width)
    print(frames_array.mean())
    return frames_array


def check_input(arguments: argparse.Namespace) -> None:
    if not os.path.exists(arguments.video_file):
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
    test_frames = get_frames(args.video_file, args.height, args.width, num_bytes=args.nbytes, get_single_frame=bool(args.single_frame))
    for i in range(test_frames.shape[0]):
        frame_viewer.show_frame(test_frames[i])
        if args.single_frame:
            while True:
                time.sleep(10)
        time.sleep(wait)

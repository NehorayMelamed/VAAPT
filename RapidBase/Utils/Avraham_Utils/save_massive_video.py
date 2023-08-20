import cv2, time, sys, os, time
import numpy as np
import argparse


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    show_video_intro = "RAW Video Viewer\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=show_video_intro)
    parser.add_argument("-p", "--prefix", help="prefix for written files", required=True, type=str)
    parser.add_argument("-v", "--video_file", help="RAW video file", required=True)
    parser.add_argument("-H", "--height", help="height of video frames", required=True, type=int)
    parser.add_argument("-W", "--width", help="width of video frames", required=True, type=int)
    parser.add_argument("-n", "--nbytes", help="num bytes", type=int, default=1)
    return parser


def exit_on(message: str, status: int = 1):
    # print message, and exit
    print("ERROR: " + message)
    sys.exit(status)


class FrameWriter:
    def __init__(self, name: str):
        self.name = name
        self.num_frame = 0

    def save_frame(self, frame):
        cv2.imwrite(f"{self.name}{self.num_frame}.png", frame)
        self.num_frame+=1


class FrameReader:
    def __init__(self, video_path: str, num_bytes: int, height: int, width: int):
        self.video_file = open(video_path)
        self.frame_size = height * width
        self.height = height
        self.width = width
        self.num_bytes = num_bytes
        if num_bytes == 1:
            self.dtype = np.uint8
        elif num_bytes == 2:
            self.dtype = np.uint16
        elif num_bytes == 4:
            self.dtype = np.uint32
        elif num_bytes == 8:
            self.dtype = np.uint64

    def get_frame(self):
        fr = np.fromfile(self.video_file, dtype=self.dtype, count=self.frame_size*self.num_bytes)
        if len(fr) != self.frame_size:
            return None
        else:
            return np.reshape(fr, (self.height, self.width))


def check_input(arguments: argparse.Namespace) -> None:
    if not os.path.exists(arguments.video_file):
        exit_on("GIVEN VIDEO FILE DOES NOT EXIST")
    if arguments.nbytes not in [1,2,4,8]:
        exit_on("Number of bytes must be 1,2,4 or 8")


if __name__ == '__main__' :
    parser = create_parser()
    args = parser.parse_args()
    check_input(args)

    frame_writer = FrameWriter(args.prefix)
    frame_reader = FrameReader(args.video_file, args.nbytes, args.height, args.width)
    frame = frame_reader.get_frame()
    while frame is not None:
        frame_writer.save_frame(frame)
        frame = frame_reader.get_frame()

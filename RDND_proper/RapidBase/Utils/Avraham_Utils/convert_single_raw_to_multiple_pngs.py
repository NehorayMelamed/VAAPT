import cv2, time, sys, os, time
import numpy as np
import argparse


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    converter_intro = "RAW Video converted\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=converter_intro)
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


if __name__ == '__main__' :
    parser = create_parser()
    args = parser.parse_args()
    check_input(args)
    frame_viewer = FrameWriter(args.prefix)
    test_frames = get_frames(args.video_file, args.height, args.width, num_bytes=args.nbytes)
    for i in range(test_frames.shape[0]):
        frame_viewer.save_frame(test_frames[i])

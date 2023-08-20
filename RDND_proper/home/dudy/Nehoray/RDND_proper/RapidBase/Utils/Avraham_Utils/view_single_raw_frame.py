import cv2, time, sys, os, time
import numpy as np
import argparse


def create_parser() -> argparse.ArgumentParser:
    # :return: creates parser with all command line arguments arguments
    show_video_intro = "RAW Video Viewer\n Authors: Avraham Kahan"
    parser = argparse.ArgumentParser(description=show_video_intro)
    parser.add_argument("-p", "--picture_file", help="RAW file", required=True)
    parser.add_argument("-H", "--height", help="height of video frames", required=True)
    parser.add_argument("-W", "--width", help="width of video frames", required=True)
    return parser


def exit_on(message: str, status: int = 1):
    # print message, and exit
    print("ERROR: " + message)
    sys.exit(status)


def load_frame(filename, height=340, width=1024):
    with open(filename, 'rb') as vf:
        file_data = np.fromfile(vf, dtype=np.uint8, count=height*width)
        frames_array = file_data.reshape(height, width)
    return frames_array


def check_input(arguments: argparse.Namespace) -> None:
    if not os.path.exists(f"{arguments.picture_file}"):
        exit_on("GIVEN FRAME DOES NOT EXIST")



if __name__ == '__main__' :
    parser = create_parser()
    args = parser.parse_args()
    check_input(args)
    frame = load_frame(args.picture_file, int(args.height), int(args.width))
    cv2.imshow("ALLIGATOR", frame)
    cv2.waitKey(1)
    while True:
        time.sleep(10)

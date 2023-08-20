import sys, cv2, time, os

from typing import Tuple


def exit_on(message, err_code = -1):
    print(message)
    sys.exit(err_code)


def analyze_input(argv) -> Tuple[str, int]:
    argc = len(argv)
    if argc != 2 and argc != 3:
        exit_on("Usage: python3 show_bgs_movie.py [frames_prefix] [Optional[fps]]")
    if argc == 2:
        fps = 30
    else:
        fps = int(argv[2])
    if fps < 0:
        exit_on("fps must be positive")
    if not os.path.exists(f"{argv[1]}1_0.png"):
        exit_on("Frames do not exist")
    return argv[1], fps


if __name__ == '__main__':
    video_prefix, fps = analyze_input(sys.argv)
    current_batch = 0
    current_frame = 1
    wait = 1/fps
    while True:
        image_frame = cv2.imread(f"{video_prefix}{current_batch}_{current_frame}.png")
        cv2.imshow("BGS Viewer", image_frame)
        cv2.waitKey(1)
        time.sleep(wait)
        current_frame+=1
        if not os.path.exists(f"{video_prefix}{current_batch}_{current_frame}.png"):
            current_frame = 0
            current_batch+=1
            if not os.path.exists(f"{video_prefix}{current_batch}_{current_frame}.png"):
                exit_on("End of Video!", 0)

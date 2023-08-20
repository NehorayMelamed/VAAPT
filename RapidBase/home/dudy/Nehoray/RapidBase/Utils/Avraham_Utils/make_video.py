import cv2, sys, time, os

from typing import Tuple
"""
image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
"""



def exit_on(message, err_code = -1):
    print(message)
    sys.exit(err_code)


def analyze_input(argv) -> Tuple[str, int]:
    argc = len(argv)
    if argc != 2 and argc != 3:
        exit_on("Usage: python3 make_video.py [frames_prefix]")
    if argc == 2:
        fps = 30
    else:
        fps = int(argv[2])
    if fps < 0:
        exit_on("fps must be positive")
    if not os.path.exists(f"{argv[1]}0_0.png"):
        exit_on("Frames do not exist")
    return argv[1], fps


if __name__ == '__main__':
    video_prefix, fps = analyze_input(sys.argv)
    current_batch = 0
    current_frame = 0
    width = 1024
    height = 340
    wait = 1/fps
    video = cv2.VideoWriter("COMBINED_VIDEO.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (width,height))
#    video = cv2.VideoWriter("COMBINED_VIDEO.avi", 0, 1, (width, height))
    while True:
        image_frame = cv2.imread(f"{video_prefix}{current_batch}_{current_frame}.png")
        video.write(image_frame)
        cv2.waitKey(1)
        time.sleep(wait)
        current_frame+=1
        if not os.path.exists(f"{video_prefix}{current_batch}_{current_frame}.png"):
            current_frame = 0
            current_batch+=1
            if not os.path.exists(f"{video_prefix}{current_batch}_{current_frame}.png"):
                cv2.destroyAllWindows()
                video.release()
                exit_on("End of Video!", 0)


import cv2
import os

# path to the video file
video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/scene_0_resized_short.mp4"


base_name = os.path.basename(video_path)

# Split the base name into name and extension
file_name, file_extension = os.path.splitext(base_name)

# specify the directory to save frames
save_dir = f'/home/nehoray/PycharmProjects/Shaback/data/images/{file_name}'

# make sure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# desired image size after resizing (width, height)
# desired_size = (640, 360)

frame_count = 0
while True:
    # read a frame
    ret, frame = cap.read()

    # if the frame is properly read
    if ret:
        # resize the frame
        # resized_frame = cv2.resize(frame, desired_size)

        # create a file path for saving
        save_path = os.path.join(save_dir, f'{frame_count}.png')

        cv2.imwrite(save_path, frame)

        # save the resized frame as a jpeg image
        # cv2.imwrite(save_path, resized_frame)

        # increment frame count
        frame_count += 1
    else:
        # if the frame is not properly read, stop the loop
        break

# release the VideoCapture object
cap.release()

import cv2

# Input and output video paths
input_video_path = '/home/nehoray/PycharmProjects/Shaback/data/videos/street_from_drone__5__.mp4'
output_video_path = '/home/nehoray/PycharmProjects/Shaback/data/videos/street_from_drone__5___NEW_FPS_5.mp4'
import cv2


# Desired FPS
desired_fps = 5

# Open the input video
input_video = cv2.VideoCapture(input_video_path)

# Get the original FPS from the input video
original_fps = input_video.get(cv2.CAP_PROP_FPS)

# Frame skip factor (how many frames to skip)
skip_factor = int(original_fps / desired_fps)

# Get the codec info and frame size from the input video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Change codec to MJPG
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the output video
output_video = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (frame_width, frame_height))

frame_count = 0
while True:
    ret, frame = input_video.read()

    if not ret:
        break

    # Write every nth frame to the output video
    if frame_count % skip_factor == 0:
        output_video.write(frame)

    frame_count += 1

# Release the video objects
input_video.release()
output_video.release()

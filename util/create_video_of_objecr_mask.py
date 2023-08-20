import cv2
import numpy as np

# Set your video resolution
width, height = 1920, 1080

# Create a VideoWriter object
out = cv2.VideoWriter('yoav_masks_object.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width, height))

# Read the .txt file and process line by line
with open('/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/exp18/tracks/CLIP.txt', 'r') as file:
    lines = file.readlines()
    frame = None
    last_frame_idx = -1

    for line in lines:
        data = line.split()
        frame_idx, id, bbox_left, bbox_top, bbox_w, bbox_h = map(float, data[:6])

        if frame_idx != last_frame_idx:
            # If this is a new frame, write the previous frame to the video
            if frame is not None:
                out.write(frame)

            # Create a new frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            last_frame_idx = frame_idx

        # Draw the white bounding box
        pt1 = (int(bbox_left), int(bbox_top))
        pt2 = (int(bbox_left + bbox_w), int(bbox_top + bbox_h))
        cv2.rectangle(frame, pt1, pt2, (255, 255, 255), -1)

    # Write the last frame
    if frame is not None:
        out.write(frame)

out.release()

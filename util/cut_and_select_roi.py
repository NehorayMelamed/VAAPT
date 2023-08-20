import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

def nothing(x):
    pass

def toggle_play(x):
    global is_playing
    is_playing = not is_playing

def save_video(x):
    global cap
    start_frame = cv2.getTrackbarPos('Start', 'Video')
    end_frame = cv2.getTrackbarPos('End', 'Video')
    cap.release()
    cv2.destroyAllWindows()

    clip = VideoFileClip(input_video_path)
    start_sec = start_frame / fps
    end_sec = end_frame / fps
    subclip = clip.subclip(start_sec, end_sec)
    subclip.write_videofile(output_video_path)

    cap = cv2.VideoCapture(input_video_path)

def select_roi(x):
    global roi_selected, x_start, y_start, x_end, y_end
    roi_selected = True
    _, frame = cap.read()
    r = cv2.selectROI(frame)
    x_start, y_start, w, h = map(int, r)
    x_end = x_start + w
    y_end = y_start + h

input_video_path = '/home/nehoray/PycharmProjects/Shaback/data/videos/toem.mp4'
output_video_path = 'output.mp4'

# Load your video
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Get some video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow('Video')

# Create trackbars for selecting the cut and navigation
cv2.createTrackbar('Start', 'Video', 0, frame_count-1, nothing)
cv2.createTrackbar('End', 'Video', 0, frame_count-1, nothing)
cv2.createTrackbar('Navigate', 'Video', 0, frame_count-1, nothing)

# Button to toggle playback
cv2.createButton('Play/Pause', toggle_play)

# Button for full frame
cv2.createButton('Select ROI', select_roi)

# Button to save the video
cv2.createButton('Save', save_video)

last_moved = 'Navigate'
is_playing = False
roi_selected = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

while True:
    start_frame = cv2.getTrackbarPos('Start', 'Video')
    end_frame = cv2.getTrackbarPos('End', 'Video')
    nav_frame = cv2.getTrackbarPos('Navigate', 'Video')

    if not is_playing:
        if last_moved == 'Start':
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        elif last_moved == 'End':
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
        else:  # last_moved == 'Navigate'
            cap.set(cv2.CAP_PROP_POS_FRAMES, nav_frame)
    else:
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame+1 >= end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            nav_frame = start_frame
        else:
            nav_frame = current_frame + 1

    ret, frame = cap.read()
    if not ret:
        break

    # If ROI is selected, apply it to frame
    if roi_selected:
        frame = frame[y_start:y_end, x_start:x_end]

    # Display frame
    cv2.imshow('Video', frame)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF  # Delay based on original video's frame rate

    # If any trackbar was moved, update last_moved
    if cv2.getTrackbarPos('Start', 'Video') != start_frame:
        last_moved = 'Start'
    elif cv2.getTrackbarPos('End', 'Video') != end_frame:
        last_moved = 'End'
    elif cv2.getTrackbarPos('Navigate', 'Video') != nav_frame:
        last_moved = 'Navigate'
        nav_frame = cv2.getTrackbarPos('Navigate', 'Video')

    cv2.setTrackbarPos('Navigate', 'Video', nav_frame)

cap.release()
cv2.destroyAllWindows()

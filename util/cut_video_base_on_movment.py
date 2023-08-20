import cv2
import numpy as np

def cut_video_on_movement(video_path, threshold_movement_percent, frame_before, frame_after, seconds_before, seconds_after, use_seconds=True, output_video_name='output_video.mp4', max_duration_seconds=None):
    # Read input video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))

    # Convert seconds to frames if use_seconds is set to True
    if use_seconds:
        frame_before = int(seconds_before * fps)
        frame_after = int(seconds_after * fps)

    # Calculate the maximum number of frames to process if max_duration_seconds is set
    if max_duration_seconds is not None:
        max_frames = int(max_duration_seconds * fps)
    else:
        max_frames = None

    # Initialize variables
    ret, prev_frame = cap.read()
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.GaussianBlur(gray_prev_frame, (21, 21), 0)

    buffer = []
    movement_detected = False
    frame_count = 0

    # Calculate threshold_movement based on the percentage of total pixels
    threshold_movement = (width * height) * (threshold_movement_percent / 100)

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break
        print(f"In progress ({frame_count} / {fps * max_duration_seconds})")
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        frame_diff = cv2.absdiff(gray_prev_frame, gray_frame)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate movement based on the area of bounding rectangles of the detected contours
        movement = sum((cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3]) for cnt in contours)

        if movement > threshold_movement:
            movement_detected = True

        if movement_detected:
            buffer.append(frame)

            if len(buffer) > frame_after:
                for buffered_frame in buffer[:frame_before]:
                    out.write(buffered_frame)
                buffer = buffer[frame_before:]
                movement_detected = False

        gray_prev_frame = gray_frame
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = "/media/dudy/FB31-3461/Downtest_ch0001_00010001837000000.mp4"
    threshold_movement_percent = 10
    output_video_name = "people_in_kichen.mp4"
    frame_before = 10
    frame_after = 10
    seconds_before = 2
    seconds_after = 2
    use_seconds = True
    max_duration_seconds = 50
    cut_video_on_movement(video_path, threshold_movement_percent, frame_before, frame_after, seconds_before, seconds_after, use_seconds, output_video_name=output_video_name, max_duration_seconds=max_duration_seconds)

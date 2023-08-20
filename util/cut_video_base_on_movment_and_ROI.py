import cv2
import numpy as np

# Mouse callback f
#


drawing = False
roi_selected = False
roi = None



def resize_image(image, max_size=800):
    height, width = image.shape[:2]

    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width, new_height = int(scale * width), int(scale * height)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    return resized_image


#
# Mouse callback function to draw ROI
def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, roi_selected, roi, x_start, y_start, x_end, y_end

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi = [(ix, iy), (x, y)]
            x_start, y_start = ix, iy
            x_end, y_end = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = [(ix, iy), (x, y)]
        roi_selected = True
        x_start, y_start = ix, iy
        x_end, y_end = x, y

def select_roi(video_path):
    global drawing, roi_selected, roi
    roi_selected = False

    # Read input video
    cap = cv2.VideoCapture(video_path)

    # Get first frame of video
    ret, frame = cap.read()
    cap.release()


    # Resize frame for selecting ROI
    frame = resize_image(frame)

    # Display first frame for selecting ROI
    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', draw_roi)

    while True:
        img = frame.copy()
        if roi_selected:
            cv2.rectangle(img, roi[0], roi[1], (0, 255, 0), 2)
        cv2.imshow('Select ROI', img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:  # Press 'Esc' or 'Enter' to exit
            break

    cv2.destroyAllWindows()

    return roi



def cut_video_on_movement(video_path, threshold_movement_percent, frame_before, frame_after, seconds_before, seconds_after, use_seconds=True, output_video_name='output_video.mp4', max_duration_seconds=None):
    # Read input video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    # Calculate max_frames based on max_duration_seconds
    max_frames = None
    if max_duration_seconds is not None:
        max_frames = max_duration_seconds * fps

    # Initialize the output video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))


    print(f"Video width: {width}, height: {height}, fps: {fps}")

    # Initialize variables
    ret, prev_frame = cap.read()

    if not ret:
        print("Error: Could not read the video file.")
        cap.release()
        return

    print(f"First frame shape: {prev_frame.shape}")

    if roi_selected:
        prev_frame = prev_frame[y_start:y_end, x_start:x_end]

    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.GaussianBlur(gray_prev_frame, (21, 21), 0)

    buffer = []
    movement_detected = False
    frame_count = 0

    # Calculate threshold_movement based on the percentage of the total ROI area
    roi_area = width_roi * height_roi
    threshold_movement = roi_area * (threshold_movement_percent / 100)

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and frame_count >= max_frames):
            break

        if roi_selected:
            roi_frame = frame[y_start:y_end, x_start:x_end]
        else:
            roi_frame = frame
        print(f"In progress - {frame_count}/{max_duration_seconds * fps}")
        gray_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray_roi_frame = cv2.GaussianBlur(gray_roi_frame, (21, 21), 0)

        frame_diff = cv2.absdiff(gray_prev_frame, gray_roi_frame)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Adjust the contour detection parameters
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        movement = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            movement += w * h
            print(f"Contour area (w * h): {w * h}")
        print(f"Total movement: {movement}, Threshold movement: {threshold_movement}")

        # Calculate movement based on the area of bounding rectangles of the detected contours
        # Calculate movement based on the actual contour area
        movement = sum(cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3] for cnt in contours)

        print(f"Contour areas: {[cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3] for cnt in contours]}")
        print(f"Total movement: {movement}, Threshold movement: {threshold_movement}")

        if movement > threshold_movement:
            movement_detected = True
            print("Movement detected")

        if movement_detected and len(buffer) > 0:
            buffer.append(frame)

            if len(buffer) > frame_after:
                for buffered_frame in buffer[:frame_before]:
                    out.write(buffered_frame)
                buffer = buffer[frame_before:]

            movement_detected = False  # Moved out of the if statement

            if roi_selected:
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)


        gray_prev_frame = gray_roi_frame
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    video_path = "/media/dudy/FB31-3461/Downtest_ch0001_00010001837000000.mp4"
    threshold_movement_percent = 5
    output_video_name = "people_in_kichen.avi"
    frame_before = 1
    frame_after = 10
    seconds_before = 2
    seconds_after = 2

    use_seconds = True
    max_duration_seconds = 100

    roi = select_roi(video_path)

    if roi:
        # Get ROI coordinates
        x_start, y_start = roi[0]
        x_end, y_end = roi[1]
        width_roi = x_end - x_start
        height_roi = y_end - y_start


    cut_video_on_movement(video_path, threshold_movement_percent, frame_before, frame_after, seconds_before, seconds_after, use_seconds, output_video_name=output_video_name, max_duration_seconds=max_duration_seconds)



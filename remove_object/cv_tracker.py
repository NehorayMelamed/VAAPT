import cv2

def track_object(video_path, start_frame, end_frame):
    # Initialize the OpenCV tracker
    tracker = cv2.TrackerKCF_create()

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Initialize list for storing bounding box coordinates
    bbox_coordinates = []

    frame_count = 0
    resize_dimensions = (2200, 1700)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # If the current frame is before the start frame, continue to next iteration
        if frame_count < start_frame:
            frame_count += 1
            continue

        # If the current frame is the start frame, let the user select the ROI from a resized frame
        if frame_count == start_frame:
            original_dimensions = frame.shape[1], frame.shape[0] # width, height
            display_frame = cv2.resize(frame, resize_dimensions) # resize for display
            bbox = cv2.selectROI(display_frame, False)

            # Scale the bounding box coordinates back to the original frame resolution
            bbox_scaled = (
                int(bbox[0] * original_dimensions[0] / resize_dimensions[0]),
                int(bbox[1] * original_dimensions[1] / resize_dimensions[1]),
                int(bbox[2] * original_dimensions[0] / resize_dimensions[0]),
                int(bbox[3] * original_dimensions[1] / resize_dimensions[1])
            )
            ok = tracker.init(frame, bbox_scaled)

        # Update tracker and save bounding box coordinates
        if frame_count >= start_frame and frame_count <= end_frame:
            ok, bbox = tracker.update(frame)
            if ok:
                bbox_coordinates.append(bbox)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else:
                print('Tracking failure detected at frame: ', frame_count)
                break

            # Display result with a constant size
            display_frame = cv2.resize(frame, resize_dimensions) # resize for display
            cv2.imshow("Tracking", display_frame)

        # Stop tracking after end_frame
        if frame_count > end_frame:
            break

        frame_count += 1

        # Exit if ESC key is pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Return the list of bounding box coordinates
    return bbox_coordinates



bbox_list = track_object("/data/videos/dji_shapira__2__.mp4", 0, 150)
print(bbox_list)

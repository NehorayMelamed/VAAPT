import cv2


def resize_video(input_video_path, output_video_path, desired_size=(700, 600)):
    # Open the video file
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print(f"Could not open video: {input_video_path}")
        return

    # Get the original video's frames per second (fps)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, desired_size)

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            # Resize the frame
            resized_frame = cv2.resize(frame, desired_size, interpolation=cv2.INTER_AREA)

            # Write the frame to the new video file
            out.write(resized_frame)
        else:
            break

    # Release the VideoCapture and VideoWriter objects and close the video files
    video.release()
    out.release()

    print(f"Resized video is saved to: {output_video_path}")






resize_video("/home/nehoray/PycharmProjects/Shaback/data/videos/scene_0.mp4", "/home/nehoray/PycharmProjects/Shaback/data/videos/scene_0_resized.mp4", desired_size=(400, 300))

# import os
# import cv2
# import numpy as np
# from cut_video_by_gui import VideoEditor
#
# # Define the input video file
# input_video = "/home/nehoray/PycharmProjects/Shaback/remove_object/scene_0.mp4"
#
# # Get the base name
# base_name_video_path = os.path.basename(input_video)
# # If you want to remove the extension as well
# base_name_video_path = os.path.splitext(base_name_video_path)[0]
#
# # Define the output video file
# output_video = f"remove_with_roi__exe_3{base_name_video_path}.mp4"
#
# ### Display video for getting the start and end frame range ###
# editor = VideoEditor(input_video, 'dummy_output.mp4')
# frame_start, frame_end = editor.process_video()
# print(f"Start frame: {frame_start}, End frame: {frame_end}")
#
# # Ask user to select the frame for ROI
# roi_frame = int(input("Enter the frame number from which you want to select ROI: "))
#
# # Create a VideoCapture object
# cap = cv2.VideoCapture(input_video)
#
# # Get video info
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # Create a VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
#
# frame_count = 0
# roi_img = None
# roi_selected = False
#
# # Select ROI function
# def select_roi(frame):
#     roi = cv2.selectROI("ROI Selector - Press ENTER to validate", frame, False, False)
#     cv2.destroyWindow("ROI Selector - Press ENTER to validate")
#     return roi
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     if frame_count == roi_frame and not roi_selected:
#         # Select and extract the ROI
#         roi = select_roi(frame)
#         roi_img = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])].copy()
#         roi_selected = True
#
#     if frame_start <= frame_count <= frame_end and roi_img is not None:
#         # Paste the ROI into the current frame
#         frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = roi_img
#
#     # Write the frame to the output video file
#     out.write(frame)
#
#     frame_count += 1
#
# cap.release()
# out.release()


import os
import cv2
from cut_video_by_gui import VideoEditor


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, frame_width, frame_height


def select_roi(frame):
    # Resize the frame for displaying
    display_scale = 0.8  # You can adjust this value according to your preference
    display_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

    # Select ROI on the resized frame
    roi = cv2.selectROI("ROI Selector - Press ENTER to validate", display_frame, False, False)
    cv2.destroyWindow("ROI Selector - Press ENTER to validate")

    # Scale the ROI coordinates back to the original size
    roi = (int(roi[0] / display_scale),
           int(roi[1] / display_scale),
           int(roi[2] / display_scale),
           int(roi[3] / display_scale))

    return roi


def process_video_for_roi(video_path, output_video):
    cap = cv2.VideoCapture(video_path)
    fps, frame_width, frame_height = get_video_info(video_path)

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    editor = VideoEditor(video_path, 'dummy_output.mp4')
    frame_start, frame_end = editor.process_video()
    print(f"Start frame: {frame_start}, End frame: {frame_end}")

    roi_frame = int(input("Enter the frame number from which you want to select ROI: "))

    frame_count = 0
    roi_img = None
    roi_selected = False

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count == roi_frame and not roi_selected:
            roi = select_roi(frame)
            roi_img = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])].copy()
            roi_selected = True

        if frame_start <= frame_count <= frame_end and roi_img is not None:
            frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = roi_img

        out.write(frame)

        frame_count += 1

    cap.release()
    out.release()


def main_remove_by_ROI(video_path, base_output_path):
    base_name_video_path = os.path.basename(video_path)
    base_name_video_path = os.path.splitext(base_name_video_path)[0]
    output_video = os.path.join(base_output_path, f"remove_with_roi_{base_name_video_path}.mp4")
    process_video_for_roi(video_path, output_video)


# if __name__ == '__main__':
#     # Example usage
#     input_video = "/home/nehoray/PycharmProjects/Shaback/remove_object/scene_0.mp4"
#     output_path = "/home/nehoray/PycharmProjects/Shaback/remove_object/output"
#     main_remove_by_ROI(input_video,output_path)



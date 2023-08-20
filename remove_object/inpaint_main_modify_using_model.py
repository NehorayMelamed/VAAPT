import os
import cv2
import shutil
import sys
import time
import subprocess
import torch

import PARAMETER

sys.path.append(f"{PARAMETER.BASE_PROJECT}/Yoav_denoise_new/pips_main")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/Inpaint_Anything")
from Inpaint_Anything.remove_anything_modify import inpaint_images
from cut_video_by_gui import VideoEditor
from util.save_images_from_video import extract_frames_from_video



inpaint_dir_path = f"{PARAMETER.BASE_PROJECT}/Inpaint_Anything"


def extract_frames(video_path: str, frame_start, frame_end, frames_dir_path):
    os.makedirs(frames_dir_path, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(total_frames):
        success, image = vidcap.read()
        if i > frame_end:
            break
        if frame_start <= i:
            cv2.imwrite(os.path.join(frames_dir_path, f'{i}.png'), image)


def track_object(video_path, start_frame, end_frame, resize_dimensions=(800, 600)):
    import cv2

    # Initialize the OpenCV tracker
    tracker = cv2.TrackerMIL.create()

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Initialize list for storing bounding box coordinates
    bbox_coordinates = []

    frame_count = 0

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
            original_dimensions = frame.shape[1], frame.shape[0]  # width, height
            display_frame = cv2.resize(frame, resize_dimensions)  # resize for display
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
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                print('Tracking failure detected at frame: ', frame_count)
                break

            # Display result with a constant size
            display_frame = cv2.resize(frame, resize_dimensions)  # resize for display
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


def get_centers(bbox_list):
    centers = []
    for bbox in bbox_list:
        x, y, w, h = bbox
        center = (x + w / 2, y + h / 2)
        centers.append(center)
    return centers



def choose_inpainting_version(directory):
    """
    Display the inpainted images for the user to choose the desired version.
    The user can switch between versions using the arrow keys.
    Pressing the 'Enter' key confirms the selection.
    """
    inpaint_versions = [
        'inpainted_with_mask_0.png',
        'inpainted_with_mask_1.png',
        'inpainted_with_mask_2.png'
    ]

    idx = 0  # Start with the first version
    while True:
        # Read the image
        img_path = os.path.join(directory, inpaint_versions[idx])
        image = cv2.imread(img_path)

        # Resize the image for better visibility
        display_size = (600, 400)
        image_resized = cv2.resize(image, display_size)

        # Add textual instructions on the image
        font = cv2.FONT_ITALIC
        color = (0, 255, 0)
        thickness = 1
        cv2.putText(image_resized, f"Version: {idx}", (10, 30), font, 0.8, color, thickness, cv2.LINE_AA)
        cv2.putText(image_resized, "Use left and right arrows to switch versions.", (10, 60), font, 0.6, color,
                    thickness, cv2.LINE_AA)
        cv2.putText(image_resized, "Press 'Enter' to select the version.", (10, 90), font, 0.6, color, thickness,
                    cv2.LINE_AA)

        # Display the image
        cv2.imshow("Choose Inpainting Version", image_resized)

        # Wait for user input
        key = cv2.waitKey(0)  # Wait indefinitely for a key press

        if key == 13:  # 'Enter' key
            cv2.destroyAllWindows()
            return inpaint_versions[idx]  # Return the chosen version
        elif key == 81:  # Left arrow key
            idx = (idx - 1) % 3  # Cycle through versions
        elif key == 83:  # Right arrow key
            idx = (idx + 1) % 3  # Cycle through versions




def remove_object_from_video_base_model(video_path: str, BASE_OUTPUT_PATH, use_cuda=False):
    use_cuda = "cuda" if use_cuda is True else "cpu"

    start_time = time.time()

    base_name_video_path = os.path.splitext(os.path.basename(video_path))[0]
    output_directory_path_for_current_video = os.path.join(BASE_OUTPUT_PATH, "output", base_name_video_path)

    # Create necessary directories
    frames_images_full_path = os.path.join(output_directory_path_for_current_video, "frames")
    output_optical_flow = os.path.join(output_directory_path_for_current_video, "output_optical_flow")
    directory_of_desired_frames_to_remove_object = os.path.join(output_directory_path_for_current_video,
                                                                "temp_directory_for_desired_frame_to_remove_object")
    final_result = os.path.join(output_directory_path_for_current_video, "final_result")

    # Clean or create directories
    for directory in [frames_images_full_path, output_optical_flow, directory_of_desired_frames_to_remove_object,
                      final_result]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

    # Get frame range for editing
    editor = VideoEditor(video_path, 'dummy_output.mp4')
    start_frame, end_frame = editor.process_video()
    print(f"Start frame: {start_frame}, End frame: {end_frame}")

    # Track object and get bounding box points
    list_of_bbox_points = track_object(video_path, start_frame, end_frame)
    center_points = get_centers(list_of_bbox_points)

    # Extract frames
    extract_frames(video_path, start_frame, end_frame, frames_dir_path=directory_of_desired_frames_to_remove_object)

    # Inpaint images
    inpaint_frames(directory_of_desired_frames_to_remove_object, center_points, final_result, video_path,
                   start_frame, end_frame, use_cuda=use_cuda)

    # Create final video
    create_final_video(video_path, final_result, output_directory_path_for_current_video)

    end_time = time.time()
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Program executed in: {int(hours)}h {int(minutes)}m {int(seconds)}s")


def inpaint_frames(directory_of_desired_frames_to_remove_object, center_points, final_result, video_path, start_frame,
                   end_frame, use_cuda="cpu"):
    # Configuration for the inpainting
    sam_model_type = "vit_h"
    sam_ckpt = os.path.join(inpaint_dir_path, "pretrained_models", "sam_vit_h_4b8939.pth")
    lama_config = os.path.join(inpaint_dir_path, "lama", "configs", "prediction", "default.yaml")
    lama_ckpt = os.path.join(inpaint_dir_path, "pretrained_models", "big-lama")
    dilate_kernel_size = 15

    points_as_expected = [[coord for coord in point] for point in center_points]

    while inpaint_images(
            directory_of_desired_frames_to_remove_object,
            points_as_expected,
            1,
            dilate_kernel_size,
            final_result,
            sam_model_type,
            sam_ckpt,
            lama_config,
            lama_ckpt,
            device=use_cuda
    ) == -1:
        # If inpainting fails, re-track object and get new center points
        list_of_bbox_points = track_object(video_path, start_frame=start_frame, end_frame=end_frame)
        center_points = get_centers(list_of_bbox_points)
        points_as_expected = [[coord for coord in point] for point in center_points]


def create_final_video(video_path, final_result, output_directory_path_for_current_video):
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None

    # Get all processed frame directories in the final_result directory
    processed_frame_dirs = [f.path for f in os.scandir(final_result) if f.is_dir()]
    processed_frames = {}
    for i, dir in enumerate(processed_frame_dirs):
        if i == 0:
            selected_version = choose_inpainting_version(dir).split('_')[-1].split('.')[
                0]  # Extracts the number from the filename
        frame_number = int(os.path.basename(dir))
        img_path = os.path.join(dir, f'inpainted_with_mask_{selected_version}.png')
        processed_frames[frame_number] = cv2.imread(img_path)

    for i in range(num_frames):
        ret, frame = video.read()
        if ret:
            output_frame = processed_frames.get(i, frame)
            if out is None:
                height, width, layers = output_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                output_path_to_video = os.path.join(output_directory_path_for_current_video, "output_video.avi")
                out = cv2.VideoWriter(output_path_to_video, fourcc, frame_rate, (width, height))
            out.write(output_frame)

    video.release()
    if out:
        out.release()
    shutil.copy2(video_path, os.path.join(output_directory_path_for_current_video, f"input_video{os.path.splitext(video_path)[1]}"))
    print(f"Data saved into {output_directory_path_for_current_video}")


# if __name__ == '__main__':
#     BASE_OUTPUT_PATH = "/home/nehoray/PycharmProjects/Shaback/remove_object"
#     remove_object_from_video_base_model("/home/nehoray/PycharmProjects/Shaback/data/videos/scene_0.mp4", BASE_OUTPUT_PATH)

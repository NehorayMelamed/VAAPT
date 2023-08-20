import os.path
import os
import shutil
import subprocess
import sys
import time
from typing import List
import cv2
import torch

start_time = time.time()  # save start time

sys.path.append("/home/nehoray/PycharmProjects/Shaback/Yoav_denoise_new/pips_main")
sys.path.append("/home/nehoray/PycharmProjects/Shaback/Inpaint_Anything")
from Inpaint_Anything.remove_anything_modify import inpaint_images
from cut_video_by_gui import VideoEditor
from util.save_images_from_video import extract_frames_from_video


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
    tracker = cv2.TrackerKCF_create()

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


# video_path = "/home/nehoray/PycharmProjects/Shaback/Inpaint_Anything/data/videos/car_road_5.mov"
video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/dji_shapira__2__.mp4"
# video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/road_from_stabilize_drone.mp4"
# video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/street_from_drone__5__.mp4"
video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/dji_shapira__2___new_FPS.mp4"
video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/street_from_drone__4___NEW_FPS_5.mp4"

video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/street_from_drone__5___NEW_FPS_5.mp4"
video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/dji_shapira__2__.mp4"
video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/scene_0.mp4"

# Get the base name
base_name_video_path = os.path.basename(video_path)
# If you want to remove the extension as well
base_name_video_path = os.path.splitext(base_name_video_path)[0]

base_output_path = "/home/nehoray/PycharmProjects/Shaback/remove_object"
output_directory = os.path.join(base_output_path, "output")
if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)

output_directory_path_for_current_video = os.path.join(output_directory, base_name_video_path)

if os.path.exists(output_directory_path_for_current_video) is True:
    shutil.rmtree(output_directory_path_for_current_video)
os.mkdir(output_directory_path_for_current_video)

frames_images_full_path = os.path.join(output_directory_path_for_current_video, "frames")
output_optical_flow = os.path.join(output_directory_path_for_current_video, "output_optical_flow")
directory_of_desired_frames_to_remove_object = os.path.join(output_directory_path_for_current_video,
                                                            "temp_directory_for_desired_frame_to_remove_object")
final_result = os.path.join(output_directory_path_for_current_video, "final_result")

###Disply video for getting the start and end fram range ###
editor = VideoEditor(video_path, 'dummy_output.mp4')
start_frame, end_frame = editor.process_video()
print(f"Start frame: {start_frame}, End frame: {end_frame}")

### Get the bounding box of the object for the desired frames range ###
# ToDo allow the user to select small, big, huge displaying
list_of_bbox_points = track_object(video_path, start_frame=start_frame, end_frame=end_frame)

### Get the desired frames into the directory: ###
### Using simple cv2 tracker ###
if os.path.exists(directory_of_desired_frames_to_remove_object) is True:
    shutil.rmtree(directory_of_desired_frames_to_remove_object)
os.mkdir(directory_of_desired_frames_to_remove_object)

extract_frames(video_path, start_frame, end_frame, frames_dir_path=directory_of_desired_frames_to_remove_object)

### Get the desired frames into the directory: ###
#### Using optical flow #####
# directory_of_desired_frames_to_remove_object ="/home/nehoray/PycharmProjects/Shaback/output/video_example_from_security_camera/crops/2/7"
#### Get the points base on optical flow for all the frames ###
# center_point_of_bounding_box = main_interface_cotracker(directory_of_desired_frames_to_remove_object, output_optical_flow, get_only_points=True)
# print(center_point_of_bounding_box)


### Get center point of the bbox list coordinates ###
center_points = get_centers(list_of_bbox_points)
print(center_points)

## Perform the inpaint ###

if os.path.exists(final_result) is True:
    shutil.rmtree(final_result)
os.mkdir(final_result)

# Configuration for the inpainting
inpaint_dir_path = "/home/nehoray/PycharmProjects/Shaback/Inpaint_Anything"
sam_model_type = "vit_h"
sam_ckpt = os.path.join(inpaint_dir_path, "pretrained_models", "sam_vit_h_4b8939.pth")
lama_config = os.path.join(inpaint_dir_path, "lama", "configs", "prediction", "default.yaml")
lama_ckpt = os.path.join(inpaint_dir_path, "pretrained_models", "big-lama")
dilate_kernel_size = 15

points_as_expected = [[coord for coord in point] for point in center_points]


while inpaint_images(
    directory_of_desired_frames_to_remove_object,
    points_as_expected,  # replace with your list of points
    1,
    15,
    final_result,
    sam_model_type,
    sam_ckpt,
    lama_config,
    lama_ckpt,
    device="cpu"
) == -1:
    ### Get the bounding box of the object for the desired frames range ###
    # ToDo allow the user to select small, big, huge displaying
    list_of_bbox_points = track_object(video_path, start_frame=start_frame, end_frame=end_frame)

    ### Get center point of the bbox list coordinates ###
    center_points = get_centers(list_of_bbox_points)

    points_as_expected = [[coord for coord in point] for point in center_points]

# Perform the inpaint
##Using command line
# for index, center_point in enumerate(center_points):
#     torch.cuda.empty_cache()  # Free up GPU memory
#
#     # Define the input image path
#     input_image = os.path.join(directory_of_desired_frames_to_remove_object, f'{start_frame + index}.png')
#
#     # Define the output directory path for this image
#     image_output_dir = os.path.join(final_result, f'{start_frame + index}')
#     os.makedirs(image_output_dir, exist_ok=True)
#
#     # Prepare the command
#     script_code ="/home/nehoray/PycharmProjects/Shaback/Inpaint_Anything/remove_anything.py"
#     command = f"python {script_code} --input_img {input_image} --coords_type key_in --point_coords {center_point[0]} {center_point[1]} --point_labels 1 --dilate_kernel_size {dilate_kernel_size} --output_dir {image_output_dir} --sam_model_type {sam_model_type} --sam_ckpt {sam_ckpt} --lama_config {lama_config} --lama_ckpt {lama_ckpt}"
#
#     # Execute the command
#     subprocess.run(command, shell=True)
#     torch.cuda.empty_cache()  # Free up GPU memory


# Load the original video
video = cv2.VideoCapture(video_path)

# Get the frame rate of the video
frame_rate = video.get(cv2.CAP_PROP_FPS)

# Get the total number of frames in the video
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the output video (width and height will be set later)
out = None

# Get all processed frame directories in the final_result directory
processed_frame_dirs = [f.path for f in os.scandir(final_result) if f.is_dir()]

# Create a dictionary to store the processed frames and their corresponding frame numbers
processed_frames = {}

# Populate the dictionary with the processed frames
for dir in processed_frame_dirs:
    frame_number = int(os.path.basename(dir))  # Get the frame number from the directory name
    img_path = os.path.join(dir, 'inpainted_with_mask_2.png')  # Full path to the processed frame
    processed_frames[frame_number] = cv2.imread(img_path)  # Store the processed frame in the dictionary

# Process the frames from the original video
for i in range(num_frames):
    ret, frame = video.read()
    if ret:
        # Get the processed frame for the current frame number, if it exists
        processed_frame = processed_frames.get(i, None)

        # If the processed frame exists, use it; otherwise, use the original frame
        output_frame = processed_frame if processed_frame is not None else frame

        # If the VideoWriter is not defined, define it
        if out is None:
            height, width, layers = output_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_path_to_video = os.path.join(output_directory_path_for_current_video, "output_video.avi")
            out = cv2.VideoWriter(output_path_to_video, fourcc, frame_rate, (width, height))

        # Write the output frame to the video
        out.write(output_frame)

# Release the video objects
video.release()
if out is not None:
    out.release()


shutil.copy2(video_path, os.path.join(output_directory_path_for_current_video, f"input_video{os.path.splitext(video_path)}"))
end_time = time.time()  # save end time

execution_time = end_time - start_time

hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Program executed in: {int(hours)}h {int(minutes)}m {int(seconds)}s")

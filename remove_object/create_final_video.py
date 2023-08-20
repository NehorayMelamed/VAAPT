import cv2
import os


video_path = "/home/nehoray/PycharmProjects/Shaback/data/videos/scene_0.mp4"

# Load the original video
video = cv2.VideoCapture(video_path)

# Get the frame rate of the video
frame_rate = video.get(cv2.CAP_PROP_FPS)

# Get the total number of frames in the video
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the output video (width and height will be set later)
out = None

# Directory for final results
final_result = "/home/nehoray/PycharmProjects/Shaback/remove_object/output/scene_0/final_result"

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
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # Get the base name
            base_name = os.path.basename(video_path)
            # If you want to remove the extension as well
            file_name = os.path.splitext(base_name)[0]
            out = cv2.VideoWriter(f'{file_name}.mp4', fourcc, frame_rate, (width, height))

        # Write the output frame to the video
        out.write(output_frame)

# Release the video objects
video.release()
if out is not None:
    out.release()

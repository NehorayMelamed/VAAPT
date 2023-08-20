import os

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image


def extract_frames_from_video(video_path, output_dir, frame_stride=1, desired_size=None):
    """
    Function to extract frames from video, save them as PNG, and resize them.
    Args:
    - video_path (str): The path to the video file.
    - output_dir (str): The directory where the frames will be saved.
    - frame_stride (int): The interval at which frames will be saved.
    - desired_size (tuple): The desired size of the output images.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    frame_index = 0

    while True:
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the next frame from the video
        ret, frame = video.read()

        # If we couldn't read a frame, we're done
        if not ret:
            break

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PIL Image
        frame = Image.fromarray(frame)

        # Convert the PIL Image to a torch tensor
        frame = transforms.ToTensor()(frame)

        # Add a batch dimension
        frame = frame.unsqueeze(0)

        # Resize the frame
        if desired_size is not None:
            frame = torch.nn.functional.interpolate(frame, size=desired_size, mode='bilinear', align_corners=False)

        # Remove the batch dimension and convert the tensor back to a PIL Image
        frame = transforms.ToPILImage()(frame.squeeze(0))

        # Save the frame as a PNG file
        frame.save(os.path.join(output_dir, f'{frame_index}.png'))

        frame_index += frame_stride

    # Release the video file
    video.release()

# # Usage
# extract_frames_from_video("/home/nehoray/PycharmProjects/Shaback/Inpaint-Anything/data/videos/drone_follows_a_sports_car_from_left_.mp4",
#                           '/home/nehoray/PycharmProjects/Shaback/Inpaint-Anything/data/images/drone_follows_a_sports_car_from_left_', frame_stride=1, desired_size=(720,1280))

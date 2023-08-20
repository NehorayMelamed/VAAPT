import os

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image


def resize_images(source_dir, target_dir, desired_size):
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Transformation to tensor
    to_tensor = transforms.ToTensor()

    # Transformation to PIL Image
    to_pil = transforms.ToPILImage()

    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the full file path
            file_path = os.path.join(source_dir, filename)

            # Read the image and convert to tensor
            img = Image.open(file_path)
            img_tensor = to_tensor(img)

            # Add a dimension for batch
            img_tensor = img_tensor.unsqueeze(0)

            # Resize the image
            resized_tensor = torch.nn.functional.interpolate(img_tensor, size=desired_size, mode='bicubic', align_corners=False)

            # Convert tensor to PIL Image
            resized_img = to_pil(resized_tensor.squeeze(0))

            # Construct the path to save the resized image
            output_path = os.path.join(target_dir, filename)

            # Save the resized image
            resized_img.save(output_path)


def resize_video_frames(video_path, target_dir, desired_size):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Transformation to tensor
    to_tensor = transforms.ToTensor()

    # Transformation to PIL Image
    to_pil = transforms.ToPILImage()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR frame from OpenCV to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame_rgb)
        img_tensor = to_tensor(img)

        # Add a dimension for batch
        img_tensor = img_tensor.unsqueeze(0)

        # Resize the image
        resized_tensor = torch.nn.functional.interpolate(img_tensor, size=desired_size, mode='bicubic',
                                                         align_corners=False)

        # Convert tensor to PIL Image
        resized_img = to_pil(resized_tensor.squeeze(0))

        # Construct the path to save the resized frame
        output_path = os.path.join(target_dir, f"frame_{frame_number:04d}.png")

        # Save the resized frame
        resized_img.save(output_path)

        frame_number += 1

    # Release the video object
    cap.release()

    return target_dir



# desired_width = 256
# desired_height = 256
# # Use the function
# resize_images('/home/dudy/Nehoray/Yoav_denoise/pips_main/input_data/png_new/taxi_crop/1', '/home/dudy/Nehoray/Yoav_denoise/pips_main/input_data/png_new/taxi_crop/1', (desired_width, desired_height))



# desired_width = 256
# desired_height = 256
#
# # Use the function
# output_dir = resize_video_frames('/path/to/video/file.mp4', '/path/to/output/directory',
#                                  (desired_width, desired_height))
# print(f"Resized frames saved in: {output_dir}")

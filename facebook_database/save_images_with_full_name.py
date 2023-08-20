import pandas as pd
import requests
import os
import cv2
import numpy as np

# Load the CSV data
data = pd.read_csv('ramalla_people.csv')

# Ensure directory exists
os.makedirs('images', exist_ok=True)

# Function to download and save image
def download_and_save_image(url, save_path):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response:
                file.write(chunk)

# For each row in the data
for index, row in data.iterrows():
    # Get the profile picture URL
    url = row['profilePicture']

    # Construct the filename
    file_name = f"{row['firstName']}_{row['lastName']}.png"
    file_path = os.path.join('images', file_name)

    # Download and save the image
    download_and_save_image(url, file_path)

    # Open the image using OpenCV
    img = cv2.imread(file_path)

    # Resize the image to 250x250
    img = cv2.resize(img, (250, 250))

    # Choose a font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Scale and color of the font
    fontScale = 0.5
    fontColor = (255,255,255)
    lineType = 2

    # Position of the text
    bottomLeftCornerOfText = (10,30)

    # Add text to the image
    cv2.putText(img, f"{row['firstName']} {row['lastName']}",
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

    # Convert image to sRGB before saving if it's in another color space
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Save the image
    cv2.imwrite(file_path, img)

import cv2
from PIL import Image
import numpy as np

def closest_color(image_path):
    # Load image and convert to RGB format
    img = Image.open(image_path).convert('RGB')

    # Convert image to a numpy array
    img_array = np.array(img)
    # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Get the most common color in the image
    colors, counts = np.unique(img_array.reshape(-1,3), axis=0, return_counts=True)
    most_common_color = colors[np.argmax(counts)]

    # Define the colors we want to compare to
    colors_to_compare = {'green': (0, 255, 0), 'yellow': (255, 255, 0), 'white': (255, 255, 255)}

    # Calculate the distance between the most common color and each of the colors we want to compare to
    distances = []
    for color_name, color in colors_to_compare.items():
        distance = np.sqrt(np.sum((most_common_color - color)**2))
        distances.append((color_name, distance))

    # Return the name of the closest color
    closest_color = min(distances, key=lambda x: x[1])[0]

    cv2.imshow("image", img_array)
    cv2.waitKey(0)
    return closest_color

# Check if the most common color in the image 'example.jpg' is within 5% of the color red
path = "/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex1/crops/car/3/license_plate.jpg"
yellow = closest_color(path)

print(yellow)
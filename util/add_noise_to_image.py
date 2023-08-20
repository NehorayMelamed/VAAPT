import cv2
import numpy as np


def add_noise(input_image_path, mean, std_dev, output_image_path):
    # Read the image
    image = cv2.imread(input_image_path)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    # Save the noisy image
    cv2.imwrite(output_image_path, noisy_image)




add_noise('/home/dudy/Nehoray/my_deblur/2022-CVPR-AirNet/test/demo/Downtest_ch0007_0001000028400000067_raw_image.jpg', 5, 2, 'noisy_image.jpg')

import cv2
import numpy as np

def compress_image_outside_mask(image_path, mask_path, output_path):
    # Load the original image
    original_img = cv2.imread(image_path)

    # Load the mask image
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the mask image
    _, mask_img = cv2.threshold(mask_img, 128, 255, cv2.THRESH_BINARY)

    # Invert the mask so that the background is white and foreground is black
    mask_img_inv = cv2.bitwise_not(mask_img)

    # Perform the bit-wise AND operation on the original image and the inverted mask
    # This will make all pixels that correspond to the white part in the mask black in the output image
    masked_img = cv2.bitwise_and(original_img, original_img, mask=mask_img_inv)

    # Save the output image
    cv2.imwrite(output_path, masked_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Test the function
compress_image_outside_mask('input.jpg', 'mask.png', 'output.png')

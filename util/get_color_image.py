from __future__ import print_function
import binascii
import struct

import webcolors

# from RapidBase.import_all import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import torch
from webcolors import *

#
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]
#
def get_closest_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return closest_name
#
# # requested_colour = (119, 172, 152)
# # actual_name, closest_name = get_colour_name(requested_colour)
#
#
#
def get_color_name(rgb_color):
    # Find the closest named color to the input RGB color
    closest_color = rgb_to_name(rgb_color)

    # Return the name of the closest named color
    return closest_color
#
def most_common_color(image):
    # Reshape the array to a 2D array with 3 columns (one for each color channel)
    image_flat = image.reshape(-1, 3)

    # Use numpy.bincount() to count the frequency of each color in the array
    color_counts = np.bincount(image_flat[:, 0]*256**2 + image_flat[:, 1]*256 + image_flat[:, 2])

    # Find the index of the most common color
    most_common_color_index = color_counts.argmax()

    # Convert the color index to a tuple representing the RGB values of the color
    most_common_color = (most_common_color_index // 256**2, (most_common_color_index // 256) % 256, most_common_color_index % 256)

    # Return the most common color
    return most_common_color


# def most_common_color(tensor_array):
#     # Convert the tensor to integer values
#     tensor_array = tensor_array.type(torch.int64)
#
#     # Flatten the tensor to a 1D array
#     flattened_tensor = tensor_array.flatten()
#
#     # Find the bin count of each value in the tensor
#     bin_counts = torch.bincount(flattened_tensor)
#
#     # Find the most common value in the tensor
#     most_common_value = torch.argmax(bin_counts)
#
#     return most_common_value.item()
#
#
# def get_list_most_color_value_in_np_array(input_numpy_array):
#     NUM_CLUSTERS = 5
#
#     ar = input_numpy_array
#     shape = ar.shape
#     ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
#
#     print('finding clusters')
#     codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
#     print('cluster centres:\n', codes)
#
#     vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
#     counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences
#
#     index_max = scipy.argmax(counts)  # find most frequent
#     peak = codes[index_max]
#     colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
#     # print('most frequent is %s (#%s)' % (peak, colour))
#     return peak
#



# ###### Load the image as a PIL Image object #####
# image = Image.open("/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex/crops/car/1/0.jpg")
#
# # Convert the PIL Image to a PyTorch tensor
# tensor_array = torch.tensor(image)
#
# # Call the most_common_color function
# most_common_value = most_common_color(tensor_array)
#
# # Print the result
# print(f"The most common color in the image is {most_common_value}")


##### Load the tensor from the .pt file ####
# tensor = torch.load('/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex/crops/car/1/1.pt', map_location=torch.device('cpu'))
# most_common_valee = (most_common_color(tensor))
# print(most_common_valee)
# numpy_image_tensor = torch_to_numpy(tensor)
# plt.imshow(tensor)
# plt.show()
# print()
#
import numpy as np
from collections import Counter


# def most_common_color(segmentation_array, image_array, segmented_value):
#     # Create mask for segmented area
#     mask = (segmentation_array == segmented_value)
#
#     # Apply mask to image array to extract pixels
#     pixels = image_array[mask]
#
#     # Find most common color in extracted pixels
#     color_counts = Counter(map(tuple, pixels))
#     most_common_color = color_counts.most_common(1)[0][0]
#
#     return most_common_color
#

def most_common_color_via_segmentation_mask(seg, img):
    # Flatten segmentation array
    seg_flat = seg.reshape(-1)
    # Reshape image array to 2D
    img_2d = img.reshape((-1, 3))
    # Select pixels based on segmentation
    selected_pixels = img_2d[seg_flat > 0]
    # Count occurrence of each color
    counts = np.bincount(selected_pixels[:, 0]*65536 + selected_pixels[:, 1]*256 + selected_pixels[:, 2])
    # Find most common color
    most_common_color_int = np.argmax(counts)
    # Convert integer color to RGB tuple
    r = (most_common_color_int // 65536) % 256
    g = (most_common_color_int // 256) % 256
    b = most_common_color_int % 256
    most_common_color_rgb = (r, g, b)
    return most_common_color_rgb




# from PIL import Image
# crop_image_path  = "/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex/crops/car/1/1.jpg"
# segm_file_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/ex/crops/car/1/1.pt"
#
# crop_image = Image.open(crop_image_path)
# crop_image_numpy = np.array(crop_image)
#
# segm_image = torch.load(segm_file_path)
# segm_image = segm_image[0]
#
#
# segm_image_numpy = torch_to_numpy(segm_image)
#

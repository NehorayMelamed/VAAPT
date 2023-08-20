

# from RapidBase.import_all import *
# import csaps
import kornia
import torch
import numpy as np
import cv2
import torchvision.ops

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import get_full_shape_torch, RGB2BW
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch
from RapidBase.Utils.IO.Path_and_Reading_utils import read_image_default, read_image_default_torch, read_outliers_default
import skimage
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from shapely.geometry import Point  #TODO: understand what's going on!!?!!!?
import networkx as nx

from numpy import zeros, ones, asarray
from numpy.linalg import norm
from math import pi
from scipy.ndimage.filters import gaussian_laplace, minimum_filter
from tifffile import imread
from numpy import loadtxt, delete
from pickle import load
import matplotlib
from numpy import ones, triu, seterr
from numpy import ones, nonzero, transpose
from numpy import empty, asarray
from itertools import repeat
from scipy import ndimage, misc
import matplotlib.pyplot as plt

import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial

from RapidBase.Utils.Classical_DSP.Convolution_Utils import filter2D_torch, convn_layer_Filter2D_torch, torch_get_3D
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import logical_mask_to_indices_torch
from RapidBase.Utils.IO.Path_and_Reading_utils import video_torch_array_to_video
from RapidBase.Utils.MISCELENEOUS import create_empty_list_of_lists
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_get_ND, unravel_index, unravel_index_torch
import torchvision

#####################################################################################################################################################################
### Connected Components Analysis: ###
def get_connected_components_logical_mask(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0,
                                          flag_return_logical_mask_or_label_values='label_values', flag_return_number_of_components=False,
                                          flag_return_COM_and_Covariance=False, flag_return_BB=False):
    # flag_return_logical_mask_or_label_values = 'label_values', 'logical_mask', 'none'

    ### Perform Connected Components Using Kornia: ###
    labels_out_tensor = kornia.contrib.connected_components(input_tensor, num_iterations=number_of_iterations)

    ### Go Over The Different Labels And Get What You Need: ###
    if flag_return_logical_mask_or_label_values == 'label_values':
        final_CC_logical_mask = torch.ones_like(labels_out_tensor) * label_zero_value  # f there's only one component -> leave it
    elif flag_return_logical_mask_or_label_values == 'logical_mask':
        final_CC_logical_mask = torch.zeros_like(labels_out_tensor)

    ### Loop Over The Different Labels And Get What You Need: ###
    unique_labels = labels_out_tensor.unique()
    number_of_components_list = []
    flag_label_allowed_list = []
    torch_covariance_template = torch.ones((2, 2)).to(input_tensor.device)
    COM_HW_list = []
    Covariance_HW_list = []
    BB_list = []
    for label_index in np.arange(len(unique_labels)):
        ### Get Current Labels Logical Mask: ###
        label_value = unique_labels[label_index]
        current_label_logical_mask = (labels_out_tensor == label_value)

        ### If Current Label is Not BG --> Get Stats: ###
        if label_value > 0:
            ### Get Number Of Pixels/Components Of Current Label: ###
            number_of_components = current_label_logical_mask.float().sum().item()  # TODO: this sum is over all possible images, and i want per image!!!
            number_of_components_list.append((number_of_components))

            ### Check If Number Of Components Satisfies Condition: ###
            flag_current_label_allowed = (number_of_components >= number_of_components_tuple[0] and number_of_components < number_of_components_tuple[1])
            flag_label_allowed_list.append(flag_current_label_allowed)
            if flag_current_label_allowed:
                ### Populate Output Tensor: ###
                if flag_return_logical_mask_or_label_values == 'label_values':
                    final_CC_logical_mask[current_label_logical_mask] = label_value
                elif flag_return_logical_mask_or_label_values == 'logical_mask':
                    final_CC_logical_mask[current_label_logical_mask] = 1

                ### Get COM and Variance: ###
                if flag_return_COM_and_Covariance:
                    #TODO: there's no need to go this way! i can make all of this in parallel / batch-wise using x_vec/y_vec/X_mat/Y_mat,
                    # which will delete ths need for using list comprehension. OH WAIT!!! THERE ARE DIFFERENT COMPONENTS AND LABELS PER FRAME!!!!!
                    # ..i can't really get all this in batch operations. maybe perform batch operations on whatever i can and perform list comprehension on the rest
                    ### Get Indices From Logical Mask: ###
                    current_sub_blob_indices = logical_mask_to_indices_torch(current_label_logical_mask, flag_return_tensor_or_list_of_tuples='tensor')

                    ### Find Sub-Blob Center: ###
                    current_sub_blob_W_indices = current_sub_blob_indices[:, -1].float()
                    current_sub_blob_H_indices = current_sub_blob_indices[:, -2].float()
                    current_sub_blob_center_W = current_sub_blob_W_indices.mean().item()
                    current_sub_blob_center_H = current_sub_blob_H_indices.mean().item()
                    current_sub_blob_center_HW = (current_sub_blob_center_H, current_sub_blob_center_W)
                    current_sub_blob_variance_WW = ((current_sub_blob_W_indices - current_sub_blob_center_W) * (current_sub_blob_W_indices - current_sub_blob_center_W)).mean().item()
                    current_sub_blob_variance_HH = ((current_sub_blob_H_indices - current_sub_blob_center_H) * (current_sub_blob_H_indices - current_sub_blob_center_H)).mean().item()
                    current_sub_blob_variance_HW = ((current_sub_blob_H_indices - current_sub_blob_center_H) * (current_sub_blob_W_indices - current_sub_blob_center_W)).mean().item()
                    current_sub_blob_covariance = 1 * torch_covariance_template
                    current_sub_blob_covariance[0, 0] = current_sub_blob_variance_HH
                    current_sub_blob_covariance[1, 1] = current_sub_blob_variance_WW
                    current_sub_blob_covariance[0, 1] = current_sub_blob_variance_HW
                    current_sub_blob_covariance[1, 0] = current_sub_blob_variance_HW
                    current_sub_blob_center_W_int = int(current_sub_blob_center_W)
                    current_sub_blob_center_H_int = int(current_sub_blob_center_H)

                    ### Assign Output Lists: ###
                    COM_HW_list.append(current_sub_blob_center_HW)
                    Covariance_HW_list.append(current_sub_blob_covariance)

                ### Get Bounding Box Around Blobs: ###
                if flag_return_BB:
                    ### Get Bounding-Box Encircling All The Current Blob: ###
                    #TODO: again, make this more efficient
                    current_sub_blob_indices = logical_mask_to_indices_torch(current_label_logical_mask, flag_return_tensor_or_list_of_tuples='tensor')
                    current_sub_blob_indices_H_min = current_sub_blob_indices[:, 2].min().unsqueeze(0).item()
                    current_sub_blob_indices_H_max = current_sub_blob_indices[:, 2].max().unsqueeze(0).item()
                    current_sub_blob_indices_W_min = current_sub_blob_indices[:, 3].min().unsqueeze(0).item()
                    current_sub_blob_indices_W_max = current_sub_blob_indices[:, 3].max().unsqueeze(0).item()

                    ### Assign Current Blob Bounding-Box: ###
                    current_sub_blob_BB_tensor_XYXY = (current_sub_blob_indices_W_min,
                                                       current_sub_blob_indices_H_min,
                                                       current_sub_blob_indices_W_max,
                                                       current_sub_blob_indices_H_max)
                    BB_list.append(current_sub_blob_BB_tensor_XYXY)

    ### Return Stuff: ###
    outputs_list = []
    if flag_return_logical_mask_or_label_values != 'none' and flag_return_logical_mask_or_label_values != False:
        outputs_list.append(final_CC_logical_mask)
    if flag_return_number_of_components:
        outputs_list.append(number_of_components_list)
    if flag_return_COM_and_Covariance:
        outputs_list.append(COM_HW_list)
        outputs_list.append(Covariance_HW_list)
    if flag_return_BB:
        outputs_list.append(BB_list)
    # final_CC_logical_mask, number_of_components_list, COM_HW_list, Covariance_HW_list, BB_list
    return outputs_list

def get_connected_components_logical_mask_batch(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0,
                                                flag_return_logical_mask_or_label_values='label_values',
                                                flag_return_number_of_components=False,
                                                flag_return_COM_and_Covariance=False,
                                                flag_return_BB=False):
    #TODO: make this more parallel then list comprehension. understand how pytorch know to take several outputs and performs torch.cat(outputs.unsqueeze(0)) on them in the dataloader
    T, C, H, W = input_tensor.shape
    outputs_list = [get_connected_components_logical_mask(input_tensor[i:i + 1],
                                                                   number_of_iterations=number_of_iterations,
                                                                   number_of_components_tuple=number_of_components_tuple,
                                                                   label_zero_value=label_zero_value,
                                                                   flag_return_logical_mask_or_label_values=flag_return_logical_mask_or_label_values,
                                                                   flag_return_number_of_components=flag_return_number_of_components,
                                                                   flag_return_COM_and_Covariance=flag_return_COM_and_Covariance,
                                                                   flag_return_BB=flag_return_BB)
                             for i in np.arange(T)]
    # final_CC_logical_mask = torch.cat(final_CC_logical_mask, 0)
    return outputs_list


def mask_according_to_connected_components_stats(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0):
    final_CC_logical_mask = get_connected_components_logical_mask(input_tensor,
                                                                  number_of_iterations=number_of_iterations,
                                                                  number_of_components=number_of_components_tuple)
    ### Multiple Final Outliers By CC Logical Mask: ###
    input_tensor = input_tensor * final_CC_logical_mask
    return input_tensor


def mask_according_to_connected_components_stats_batch(input_tensor, number_of_iterations=10, number_of_components_tuple=(5,5), label_zero_value=0):
    logical_mask = get_connected_components_logical_mask_batch(input_tensor,
                                                               number_of_iterations=number_of_iterations,
                                                               number_of_components=number_of_components_tuple)
    input_tensor = input_tensor * logical_mask
    return input_tensor
#####################################################################################################################################################################


#####################################################################################################################################################################
### Blob Detection Scipy: ###

def opencv_blob_detection(input_image):
    ### Read Image: ###
    # input_image = cv2.imread(r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/blob.png')

    ### Set up the detector with default parameters: ###
    detector = cv2.SimpleBlobDetector_create()

    ### Detect Blobs: ###
    keypoints = detector.detect(input_image)

    ### Draw detected blobs as red circles (cv2.draw_matches_flags_draw_rich_keypoints ensure the size of the circle corresponds to the size of the blob): ###
    im_with_keypoints = cv2.drawKeypoints(np.ones_like(input_image)*input_image, keypoints, np.array([0]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # ### Show Keypoints: ###
    # cv2.imshow('Keypoints', im_with_keypoints)
    # cv2.waitKey(0)

    return im_with_keypoints, keypoints


def opencv_blob_detection_torch(input_image_torch):
    ### Prepare numpy image for function: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_image_torch)
    if shape_len == 2:
        input_image_numpy = input_image_torch.cpu().numpy()
    elif shape_len == 3:
        input_image_numpy = input_image_torch.permute([1, 2, 0]).cpu().numpy()
    elif shape_len == 4:
        input_image_numpy = input_image_torch[0].permute([1, 2, 0]).cpu().numpy()

    ### Get blobs: ###
    input_image_with_keypoints, keypoints = opencv_blob_detection(input_image_numpy)
    return input_image_with_keypoints, keypoints


def opencv_blob_detection_with_parameters(input_image,
                                          min_threshold=10, max_threshold=200,
                                          flag_filter_by_area=False, min_area=1500,
                                          filter_by_circularity=False, min_circularity=0.1,
                                          flag_filter_by_convexity=False, min_convexity=0.87,
                                          flag_filter_by_interia=False, min_intertia_ratio=0.01):

    ############################################################################
    #### Better control of parameters: ####
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = min_threshold
    params.maxThreshold = max_threshold

    # Filter by Area.
    params.filterByArea = flag_filter_by_area
    params.minArea = min_area

    # Filter by Circularity
    params.filterByCircularity = filter_by_circularity
    params.minCircularity = min_circularity

    # Filter by Convexity
    params.filterByConvexity = flag_filter_by_convexity
    params.minConvexity = min_convexity

    # Filter by Inertia
    params.filterByInertia = flag_filter_by_interia
    params.minInertiaRatio = min_intertia_ratio

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    ### Detect Blobs: ###
    keypoints = detector.detect(input_image)

    ### Draw detected blobs as red circles (cv2.draw_matches_flags_draw_rich_keypoints ensure the size of the circle corresponds to the size of the blob): ###
    im_with_keypoints = cv2.drawKeypoints(np.ones_like(input_image)*input_image, keypoints, np.array([0]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # ### Show Keypoints: ###
    # cv2.imshow('Keypoints', im_with_keypoints)
    # cv2.waitKey(0)

    return im_with_keypoints, keypoints


def opencv_blob_detection_with_parameters_torch(input_image_torch,
                                                min_threshold=10, max_threshold=200,
                                                flag_filter_by_area=False, min_area=1500,
                                                filter_by_circularity=False, min_circularity=0.1,
                                                flag_filter_by_convexity=False, min_convexity=0.87,
                                                flag_filter_by_interia=False, min_intertia_ratio=0.01):
    ### Prepare numpy image for function: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_image_torch)
    if shape_len == 2:
        input_image_numpy = input_image_torch.cpu().numpy()
    elif shape_len == 3:
        input_image_numpy = input_image_torch.permute([1, 2, 0]).cpu().numpy()
    elif shape_len == 4:
        input_image_numpy = input_image_torch[0].permute([1, 2, 0]).cpu().numpy()

    ### Get blobs: ###
    input_image_with_keypoints, keypoints = opencv_blob_detection_with_parameters(input_image_numpy,
                                                                                  min_threshold=10, max_threshold=200,
                                                                                  flag_filter_by_area=False, min_area=1500,
                                                                                  filter_by_circularity=False, min_circularity=0.1,
                                                                                  flag_filter_by_convexity=False, min_convexity=0.87,
                                                                                  flag_filter_by_interia=False, min_intertia_ratio=0.01)
    return input_image_with_keypoints, keypoints


def blob_detection_scipy_LaplacianOfGaussian(input_image, min_sigma=5, max_sigma=30, number_of_sigmas=20, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    ### Get blobs using Laplacian of Gaussian (LoG): ###
    blobs_array = skimage.feature.blob_log(RGB2BW(input_image), min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=number_of_sigmas, threshold=threshold, overlap=overlap)
    # Compute radii in the 3rd column.
    blobs_array[:, 2] = blobs_array[:, 2] * np.sqrt(2)

    # [blobs_log] = [N,3]. the three columns are: (center_H, center_W, radii)

    ### Plot On Image: ###
    if flag_return_image_with_circles:
        input_image_with_blobs = (np.ones_like(input_image) * input_image)
        for blob in blobs_array:
            y, x, dont_know, r = blob
            cv2.circle(input_image_with_blobs, (int(x),int(y)), int(r), (125, 125, 125), 3)

    # plt.imshow(input_image_with_blobs)

    blobs_list = list(blobs_array)

    if flag_return_image_with_circles:
        return input_image_with_blobs, blobs_list
    else:
        return blobs_list

def blob_detection_scipy_LaplacianOfGaussian_torch(input_image_torch, min_sigma=10, max_sigma=30, number_of_sigmas=10, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    ### Prepare numpy image for function: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_image_torch)
    if shape_len == 2:
        input_image_numpy = input_image_torch.cpu().numpy()
    elif shape_len == 3:
        input_image_numpy = input_image_torch.permute([1, 2, 0]).cpu().numpy()
    elif shape_len == 4:
        input_image_numpy = input_image_torch[0].permute([1, 2, 0]).cpu().numpy()

    return blob_detection_scipy_LaplacianOfGaussian(input_image_numpy, min_sigma=min_sigma, max_sigma=max_sigma, number_of_sigmas=number_of_sigmas, threshold=threshold, overlap=overlap, flag_return_image_with_circles=flag_return_image_with_circles)

def blob_detection_scipy_LaplacianOfGaussian_torch_batch(input_image_torch, min_sigma=10, max_sigma=30, number_of_sigmas=10, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    final_answer = [blob_detection_scipy_LaplacianOfGaussian_torch(input_image_torch[i:i+1],
                                                           min_sigma=min_sigma,
                                                           max_sigma=max_sigma,
                                                           number_of_sigmas=number_of_sigmas,
                                                           threshold=threshold,
                                                                   overlap=overlap,
                                                           flag_return_image_with_circles=flag_return_image_with_circles) for i in np.arange(input_image_torch.shape[0])]
    final_images = [final_answer[i][0] for i in np.arange(len(final_answer))]
    final_images = np.concatenate(final_images, -1)
    final_centers = [final_answer[i][1] for i in np.arange(len(final_answer))]
    return final_images, final_centers


def blob_detection_scipy_DifferenceOfGaussian(input_image, min_sigma=10, max_sigma=30, sigma_ratio=1.6, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    ### Get blobs using Laplacian of Gaussian (LoG): ###
    blobs_array = skimage.feature.blob_dog(RGB2BW(input_image), min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap)
    # Compute radii in the 3rd column.
    blobs_array[:, 2] = blobs_array[:, 2] * np.sqrt(2)

    # [blobs_log] = [N,3]. the three columns are: (center_H, center_W, radii)

    ### Plot On Image: ###
    if flag_return_image_with_circles:
        input_image_with_blobs = np.ones_like(input_image) * input_image
        for blob in blobs_array:
            y, x, dont_know, r = blob
            cv2.circle(input_image_with_blobs, (int(x), int(y)), int(r), (125, 125, 125), 3)

    blobs_list = list(blobs_array)
    if flag_return_image_with_circles:
        return input_image_with_blobs, blobs_list
    else:
        return blobs_list


def blob_detection_scipy_DifferenceOfGaussian_torch(input_image_torch, min_sigma=10, max_sigma=30, sigma_ratio=1.6, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    ### Prepare numpy image for function: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_image_torch)
    if shape_len == 2:
        input_image_numpy = input_image_torch.cpu().numpy()
    elif shape_len == 3:
        input_image_numpy = input_image_torch.permute([1, 2, 0]).cpu().numpy()
    elif shape_len == 4:
        input_image_numpy = input_image_torch[0].permute([1, 2, 0]).cpu().numpy()

    return blob_detection_scipy_DifferenceOfGaussian(input_image_numpy, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, threshold=threshold, overlap=overlap, flag_return_image_with_circles=flag_return_image_with_circles)


def blob_detection_scipy_DifferenceOfGaussian_torch_batch(input_image_torch, min_sigma=10, max_sigma=30, sigma_ratio=1.6, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    final_answer = [blob_detection_scipy_DifferenceOfGaussian_torch(input_image_torch[i:i+1],
                                                           min_sigma=min_sigma,
                                                           max_sigma=max_sigma,
                                                           sigma_ratio=sigma_ratio,
                                                           threshold=threshold,
                                                           overlap=overlap,
                                                           flag_return_image_with_circles=flag_return_image_with_circles) for i in np.arange(input_image_torch.shape[0])]
    final_images = [final_answer[i][0] for i in np.arange(len(final_answer))]
    final_images = np.concatenate(final_images, -1)
    final_centers = [final_answer[i][1] for i in np.arange(len(final_answer))]
    return final_images, final_centers


def blob_detection_scipy_DeterminantOfHessian(input_image, min_sigma=10, max_sigma=30, number_of_sigmas=10, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    ### Get blobs using Laplacian of Gaussian (LoG): ###
    blobs_array = skimage.feature.blob_doh(RGB2BW(input_image).squeeze(), min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=number_of_sigmas, threshold=threshold, overlap=overlap)
    # Compute radii in the 3rd column.
    # blobs_array[:, 2] = blobs_array[:, 2] * np.sqrt(2)

    # [blobs_log] = [N,3]. the three columns are: (center_H, center_W, radii)

    ### Plot On Image: ###
    if flag_return_image_with_circles:
        input_image_with_blobs = np.ones_like(input_image) * input_image
        for blob in blobs_array:
            y, x, r = blob
            cv2.circle(input_image_with_blobs, (int(x), int(y)), int(r), (125, 125, 125), 3)

    blobs_list = list(blobs_array)
    if flag_return_image_with_circles:
        return input_image_with_blobs, blobs_list
    else:
        return blobs_list

def blob_detection_scipy_DeterminantOfHessian_torch(input_image_torch, min_sigma=10, max_sigma=30, number_of_sigmas=10, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    ### Prepare numpy image for function: ###
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_image_torch)
    if shape_len == 2:
        input_image_numpy = input_image_torch.cpu().numpy()
    elif shape_len == 3:
        input_image_numpy = input_image_torch.permute([1, 2, 0]).cpu().numpy()
    elif shape_len == 4:
        input_image_numpy = input_image_torch[0].permute([1, 2, 0]).cpu().numpy()

    return blob_detection_scipy_DeterminantOfHessian(input_image_numpy, min_sigma=min_sigma, max_sigma=max_sigma, number_of_sigmas=number_of_sigmas, threshold=threshold, overlap=overlap, flag_return_image_with_circles=flag_return_image_with_circles)

def blob_detection_scipy_DeterminantOfHessian_torch_batch(input_image_torch, min_sigma=10, max_sigma=30, number_of_sigmas=10, threshold=0.1, overlap=0.2, flag_return_image_with_circles=True):
    final_answer = [blob_detection_scipy_DeterminantOfHessian_torch(input_image_torch[i:i+1],
                                                           min_sigma=min_sigma,
                                                           max_sigma=max_sigma,
                                                           number_of_sigmas=number_of_sigmas,
                                                           threshold=threshold,
                                                                    overlap=overlap,
                                                           flag_return_image_with_circles=flag_return_image_with_circles) for i in np.arange(input_image_torch.shape[0])]
    final_images = [final_answer[i][0] for i in np.arange(len(final_answer))]
    final_images = np.concatenate(final_images, -1)
    final_centers = [final_answer[i][1] for i in np.arange(len(final_answer))]
    return final_images, final_centers


def demo_blob_detection_scipy():
    ### Get demo data: ###
    image = skimage.data.hubble_deep_field()[0:500, 0:500]
    image_gray = rgb2gray(image)

    ### Get blobs using Laplacian of Gaussian (LoG): ###
    blobs_log = skimage.feature.blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)

    ### Get blobs using Difference of Gaussian (DoG): ###
    blobs_dog = skimage.feature.blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

    ### Get blobs using Determinant of Hessian (DoH): ###
    blobs_doh = skimage.feature.blob_doh(image_gray, max_sigma=30, threshold=.01)

    ### Plot Results: ###
    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()



def get_COM_and_MOI_tensor_torch(input_tensor):
    #(*). assumes 1 cluster, for more then 1 cluster use something like k-means
    B,C,H,W = input_tensor.shape

    ### Get mean and variance: ###
    input_tensor_sum = input_tensor.sum([-1, -2], True)
    input_tensor_normalized = input_tensor / (input_tensor_sum + 1e-6)
    H, W = input_tensor_normalized.shape[-2:]
    x_vec = torch.arange(W)
    y_vec = torch.arange(H)
    x_vec = x_vec - W // 2
    y_vec = y_vec - H // 2
    x_vec = x_vec.to(input_tensor_normalized.device)
    y_vec = y_vec.to(input_tensor_normalized.device)
    x_grid = x_vec.unsqueeze(0).repeat(H, 1)
    y_grid = y_vec.unsqueeze(1).repeat(1, W)

    ### Mean: ###
    cx = (input_tensor_normalized * x_grid).sum([-1, -2], True).squeeze(-1).squeeze(-1)
    cy = (input_tensor_normalized * y_grid).sum([-1, -2], True).squeeze(-1).squeeze(-1)
    cx_corrected = cx + W//2
    cy_corrected = cy + H//2

    ### MOI: ###
    x_grid_batch = x_grid.unsqueeze(0).repeat(len(cx), 1, 1)
    y_grid_batch = y_grid.unsqueeze(0).repeat(len(cy), 1, 1)
    x_grid_batch_modified = (x_grid_batch - cx.unsqueeze(-1).unsqueeze(-1))
    y_grid_batch_modified = (y_grid_batch - cy.unsqueeze(-1).unsqueeze(-1))
    cx2 = (input_tensor_normalized * x_grid_batch_modified ** 2).sum([-1, -2], True).squeeze()
    cy2 = (input_tensor_normalized * y_grid_batch_modified ** 2).sum([-1, -2], True).squeeze()
    cxy = (input_tensor_normalized * x_grid_batch_modified * y_grid_batch_modified).sum([-1, -2], True).squeeze()
    MOI_tensor = torch.zeros((2,2)).to(input_tensor.device)
    MOI_tensor[0,0] = cx2
    MOI_tensor[0,1] = cxy
    MOI_tensor[1,0] = cxy
    MOI_tensor[1,1] = cy2
    return cx_corrected, cy_corrected, cx2, cy2, cxy, MOI_tensor
#################################################################################################################################################################################################################################




#################################################################################################################################################################################################################################
### Blob Detection Numpy V1: ###


def scipy_minimum_filter_example():
    ### Scipy: ###
    fig = plt.figure()
    plt.gray()  # show the filtered result in grayscale
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ascent = misc.ascent()
    result = ndimage.minimum_filter(ascent, size=20)
    ax1.imshow(ascent)
    ax2.imshow(result)
    plt.show()

    ### Pytorch Version: ###
    ascent_torch = torch.tensor(ascent).unsqueeze(0).unsqueeze(0).float()
    maxpool_layer = torch.nn.MaxPool2d(20,1,9)
    result_torch = -1*maxpool_layer.forward(-ascent_torch)
    imshow_torch(result_torch)
    result_torch_numpy = result_torch[0,0].cpu().numpy()
    diff_results = result - result_torch_numpy

def localMinima(data, threshold):
    if threshold is not None:
        peaks = data < threshold
    else:
        peaks = ones(data.shape, dtype=data.dtype)

    peaks &= data == minimum_filter(data, size=(3,) * data.ndim)
    return transpose(nonzero(peaks))

def blobLOG(data, scales=range(1, 10, 1), threshold=-30):
    """Find blobs. Returns [[scale, x, y, ...], ...]"""
    data = asarray(data)
    scales = asarray(scales)

    log = empty((len(scales),) + data.shape, dtype=data.dtype)
    for slog, scale in zip(log, scales):
        slog[...] = scale ** 2 * gaussian_laplace(data, scale)

    peaks = localMinima(log, threshold=threshold)
    peaks[:, 0] = scales[peaks[:, 0]]
    return peaks

def sphereIntersection(r1, r2, d):
    # https://en.wikipedia.org/wiki/Spherical_cap#Application

    valid = (d < (r1 + r2)) & (d > 0)
    return (pi * (r1 + r2 - d) ** 2
            * (d ** 2 + 6 * r2 * r1
               + 2 * d * (r1 + r2)
               - 3 * (r1 - r2) ** 2)
            / (12 * d)) * valid

def circleIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    from numpy import arccos, sqrt

    return (r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            + r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            - sqrt((-d + r1 + r2) * (d + r1 - r2)
                   * (d - r1 + r2) * (d + r1 + r2)) / 2)



def findBlobs(img, scales=range(1, 10), threshold=30, max_overlap=0.05):

    old_errs = seterr(invalid='ignore')

    peaks = blobLOG(img, scales=scales, threshold=-threshold)
    radii = peaks[:, 0]
    positions = peaks[:, 1:]

    distances = norm(positions[:, None, :] - positions[None, :, :], axis=2)

    if positions.shape[1] == 2:
        intersections = circleIntersection(radii, radii.T, distances)
        volumes = pi * radii ** 2
    elif positions.shape[1] == 3:
        intersections = sphereIntersection(radii, radii.T, distances)
        volumes = 4/3 * pi * radii ** 3
    else:
        raise ValueError("Invalid dimensions for position ({}), need 2 or 3."
                         .format(positions.shape[1]))

    delete = ((intersections > (volumes * max_overlap))
              # Remove the smaller of the blobs
              & ((radii[:, None] < radii[None, :])
                 # Tie-break
                 | ((radii[:, None] == radii[None, :])
                    & triu(ones((len(peaks), len(peaks)), dtype='bool'))))
    ).any(axis=1)

    seterr(**old_errs)
    return peaks[~delete]

def peakEnclosed(peaks, shape, size=1):
    shape = asarray(shape)
    return ((size <= peaks).all(axis=-1) & (size < (shape - peaks)).all(axis=-1))


def plot(args):

    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image = imread(str(args.image)).T
    scale = asarray(args.scale) if args.scale else ones(image.ndim, dtype='int')

    if args.peaks.suffix == '.txt':
        peaks = loadtxt(str(args.peaks), ndmin=2)
    elif args.peaks.suffix == ".csv":
        peaks = loadtxt(str(args.peaks), ndmin=2, delimiter=',')
    elif args.peaks.suffix == ".pickle":
        with args.peaks.open("rb") as f:
            peaks = load(f)
    else:
        raise ValueError("Unrecognized file type: '{}', need '.pickle' or '.csv'"
                         .format(args.peaks.suffix))
    peaks = peaks / scale

    proj_axes = tuple(filterfalse(partial(contains, args.axes), range(image.ndim)))
    image = image.max(proj_axes)
    peaks = delete(peaks, proj_axes, axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image.T, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(*peaks.T, edgecolor='red', facecolor='none')
    if args.outfile is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(args.outfile))

def find(args):

    image = imread(str(args.image)).astype('float32')

    scale = asarray(args.scale) if args.scale else ones(image.ndim, dtype='int')
    blobs = findBlobs(image, range(*args.size), args.threshold)[:, 1:] # Remove scale
    blobs = blobs[peakEnclosed(blobs, shape=image.shape, size=args.edge)]
    blobs = blobs[:, ::-1] # Reverse to xyz order
    blobs = blobs * scale

    if args.format == "pickle":
        from pickle import dump, HIGHEST_PROTOCOL
        from functools import partial
        dump = partial(dump, protocol=HIGHEST_PROTOCOL)

        dump(blobs, stdout.buffer)
    else:
        import csv

        if args.format == 'txt':
            delimiter = ' '
        elif args.format == 'csv':
            delimiter = ','
        writer = csv.writer(stdout, delimiter=delimiter)
        for blob in blobs:
            writer.writerow(blob)

# For setuptools entry_points
def blob_detection_numpy_v1_testing(args=None):


    parser = ArgumentParser(description="Find peaks in an nD image")
    subparsers = parser.add_subparsers()

    find_parser = subparsers.add_parser("find")
    find_parser.add_argument("image", type=Path, help="The image to process")
    find_parser.add_argument("--size", type=int, nargs=2, default=(1, 1),
                             help="The range of sizes (in px) to search.")
    find_parser.add_argument("--threshold", type=float, default=5,
                             help="The minimum spot intensity")
    find_parser.add_argument("--format", choices={"csv", "txt", "pickle"}, default="csv",
                             help="The output format (for stdout)")
    find_parser.add_argument("--edge", type=int, default=0,
                             help="Minimum distance to edge allowed.")
    find_parser.set_defaults(func=find)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("image", type=Path, help="The image to process")
    plot_parser.add_argument("peaks", type=Path, help="The peaks to plot")
    plot_parser.add_argument("outfile", nargs='?', type=Path, default=None,
                             help="Where to save the plot (omit to display)")
    plot_parser.add_argument("--axes", type=int, nargs=2, default=(0, 1),
                             help="The axes to plot")
    plot_parser.set_defaults(func=plot)

    for p in (plot_parser, find_parser):
        p.add_argument("--scale", nargs="*", type=float,
                       help="The scale for the points along each axis.")

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)
#################################################################################################################################################################################################################################


#################################################################################################################################################################################################################################
### Second Algorithm: ###

def LoG(sigma, kernel_size=None):
    #window size
    if kernel_size is None:
        kernel_size = np.ceil(sigma*6)

    y,x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y) ) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4))
    return final_filter


def get_LOG_filter_torch(sigma, kernel_size=None, device='cpu', kernel_size_to_sigma_factor=2):
    ### Get Filter Kernel Size:###
    if kernel_size is None:
        kernel_size = np.ceil(sigma*kernel_size_to_sigma_factor)

    ### Build The Filter Using Dirac Multiplication Of 1D Tensors: ###
    y,x = np.ogrid[-kernel_size//2:kernel_size//2+1, -kernel_size//2:kernel_size//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (-(2*sigma**2) + (x*x + y*y) ) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4))

    ### Get Filter As A Torch Tensor: ###
    final_filter = torch.tensor(final_filter).to(device)
    return final_filter

def get_fan_filters_bank():
    #TODO: to make the filters try and really match things i can make the image [-1,1] instead of binary [0,1]...that we if the filter is big and wasteful it would score less
    #TODO: add directional blur filteers into the mix, perhapse by convolving the filters below, perhapse by using bilinear interpolation (affine rotation)
    kernel_sizes_list = [(5,5), (10,10), (20,20), (30,30), (10,50), (50,10), (10,100), (100,10), (20,50), (50,20)]
    filters_bank_list = []
    return filters_bank_list

def get_response_from_filters_bank(input_tensor, filters_bank_list):
    ### Loop Over The Filters And Get The Appropriate Filter Response: ###
    filter_responses_tensor = []
    for filter_index, current_filter in enumerate(filters_bank_list):
        ### Filter Image And Square It: ###
        output_tensor = filter2D_torch(input_tensor, current_filter.unsqueeze(0))

        ### Add To Images List: ###
        filter_responses_tensor.append(output_tensor)

    ### Concatenate All Images Into Single Tensor: ###
    filter_responses_tensor = torch.cat(filter_responses_tensor, 1)  # TX[B,1,H,W] -> [B,T,H,W]

    return filter_responses_tensor

def get_binning_filters_response(input_tensor):
    kernel_sizes_list = [(5, 5), (10, 10), (20, 20), (30, 30), (10, 50), (50, 10), (10, 100), (100, 10), (20, 50), (50, 20)]
    # sort binning sizes from smallest to largest and build things hierarchically
    1

def LoG_convolve(img, sigma_size_basis=1.414, number_of_sigmas=9, basic_sigma_factor=1):
    log_images = []
    for i in np.arange(0, number_of_sigmas):
        y = np.power(sigma_size_basis, i)
        sigma_1 = basic_sigma_factor * y
        filter_log = LoG(sigma_1)
        image = cv2.filter2D(img,-1,filter_log)
        if len(image.shape) == 2:
            image = np.pad(image,((1,1),(1,1)),'constant')
        elif len(image.shape) == 3:
            image = np.pad(image, ((1, 1), (1, 1),(0,0)), 'constant')
        image = np.square(image)  #simply square.... like **2
        log_images.append(image)
    log_image_np = np.array([i for i in log_images])  #[N_sigmas, H, W, 3]
    return log_image_np

def get_full_resolution_BB_from_low_resolution_BB_batch(BB_list_of_lists, binning_factor=16):
    return [get_full_resolution_BB_from_low_resolution_BB(BB_list_of_lists[frame_index][0], binning_factor=binning_factor) for frame_index in np.arange(len(BB_list_of_lists))]

def get_full_resolution_BB_from_low_resolution_BB(BB_list, binning_factor=16):
    BB_full_resolution_list = []
    for BB_index in np.arange(len(BB_list)):
        ### Get Current BB Coordinates: ###
        current_BB = BB_list[BB_index]
        current_sub_blob_indices_W_min, current_sub_blob_indices_H_min, \
        current_sub_blob_indices_W_max, current_sub_blob_indices_H_max = current_BB

        ### Get Full Resolution BB Coordinates: ###
        upscale_extra_room = 0.3
        full_resolution_sub_blob_indices_W_min = int((current_sub_blob_indices_W_min - upscale_extra_room) * binning_factor)
        full_resolution_sub_blob_indices_H_min = int((current_sub_blob_indices_H_min - upscale_extra_room) * binning_factor)
        full_resolution_sub_blob_indices_W_max = int((current_sub_blob_indices_W_max + upscale_extra_room) * binning_factor)
        full_resolution_sub_blob_indices_H_max = int((current_sub_blob_indices_H_max + upscale_extra_room) * binning_factor)

        current_sub_blob_BB_tensor_XYXY_full_resolution = (full_resolution_sub_blob_indices_W_min,
                                                           full_resolution_sub_blob_indices_H_min,
                                                           full_resolution_sub_blob_indices_W_max,
                                                           full_resolution_sub_blob_indices_H_max)
        BB_full_resolution_list.append(current_sub_blob_BB_tensor_XYXY_full_resolution)

    return BB_full_resolution_list

def get_reduced_outliers_tensor_from_BB_using_connected_components_within_BB(input_tensor, list_of_BB, CC_number_of_iterations=20, number_of_components_tuple=(0,np.inf)):
    ### Initialize Things For Coming Loops: ###
    number_of_BBs_per_frame = []
    number_of_pixels_per_frame_per_label_list = create_empty_list_of_lists(input_tensor.shape[0])
    sublob_center_per_frame_per_label_list = create_empty_list_of_lists(input_tensor.shape[0])
    sublob_covariance_per_frame_per_label_list = create_empty_list_of_lists(input_tensor.shape[0])
    input_tensor_reduced = torch.ones_like(input_tensor) * input_tensor
    ### Loop Over Each Frame, And For Each Frame Loop Over All Bounding-Boxes And Reduce All Connected-Components Sub-Blobs There To Single Outliers: ###
    for batch_index in np.arange(input_tensor.shape[0]):
        # print(batch_index)
        ### Get all bounding boxes in current frame: ###
        current_BBs = list_of_BB[batch_index]
        number_of_BBs_per_frame.append(len(current_BBs))

        ### Extract All Bounding Boxes As Sub-Images Into a List: ###
        current_frame = input_tensor[batch_index:batch_index + 1]
        current_frame_sub_frames_list = []
        current_frame_output = input_tensor_reduced[batch_index:batch_index + 1]
        torch_covariance_template = torch.ones((2, 2)).to(input_tensor.device)

        ### Loop Over All Bounding Boxes For This Frame, Find Sub-Blobs, And Reduce Them To Single Point: ###
        for frame_BB_index in np.arange(len(current_BBs)):
            ### Get Current Sub Frame Contained Inside Bounding Box: ###
            current_BB = current_BBs[frame_BB_index]
            start_W = current_BB[0]
            start_H = current_BB[1]
            stop_W = current_BB[2]
            stop_H = current_BB[3]
            current_frame_sub_frame = current_frame[:, :, start_H:stop_H, start_W:stop_W]
            current_frame_sub_frames_list.append(current_frame_sub_frame)

            ### Activate Connected Components Algorithm On Each Sub-Image: ###
            CC_outputs_list = get_connected_components_logical_mask(current_frame_sub_frame,
                                                                               number_of_iterations=CC_number_of_iterations,
                                                                               number_of_components_tuple=number_of_components_tuple,  # allowed sizes
                                                                               label_zero_value=0,
                                                                               flag_return_logical_mask_or_label_values='label_values',
                                                                               flag_return_number_of_components=False,
                                                                               flag_return_COM_and_Covariance=False,
                                                                               flag_return_BB=False)
            current_BB_CC_logical_mask = CC_outputs_list[0]

            ### Find Sub-Blobs / Different Labels In The Connected-Components Logical Mask: ###
            CC_logical_mask_unique_values = current_BB_CC_logical_mask.unique()  # starts with 0 which is the BG
            number_of_sub_blobs = CC_logical_mask_unique_values.numel()

            ### Initialize New Sub Frame: ###
            current_frame_sub_frame_new = torch.zeros_like(current_frame_sub_frame)

            ### Find Center For Each Sub-Blob: ###
            for CC_label_index in np.arange(1, number_of_sub_blobs):
                ### Get Current Sub-Blob / Label (from connected components analysis): ###
                current_label = CC_logical_mask_unique_values[CC_label_index]
                current_sub_blob_logical_mask = (current_BB_CC_logical_mask == current_label)
                current_label_number_of_pixels = current_sub_blob_logical_mask.sum().item()
                number_of_pixels_per_frame_per_label_list[batch_index].append(current_label_number_of_pixels)
                current_sub_blob_indices = logical_mask_to_indices_torch(current_sub_blob_logical_mask, flag_return_tensor_or_list_of_tuples='tensor')
                H_sub_blob, W_sub_blob = current_sub_blob_logical_mask.shape[-2:]

                ### Find Sub-Blob Center: ###
                current_sub_blob_W_indices = current_sub_blob_indices[:, -1].float()
                current_sub_blob_H_indices = current_sub_blob_indices[:, -2].float()
                current_sub_blob_center_W = float(min(current_sub_blob_W_indices.mean().item(), W_sub_blob-1))
                current_sub_blob_center_H = float(min(current_sub_blob_H_indices.mean().item(), H_sub_blob-1))
                current_sub_blob_variance_WW = ((current_sub_blob_W_indices - current_sub_blob_center_W) * (current_sub_blob_W_indices - current_sub_blob_center_W)).mean().item()
                current_sub_blob_variance_HH = ((current_sub_blob_H_indices - current_sub_blob_center_H) * (current_sub_blob_H_indices - current_sub_blob_center_H)).mean().item()
                current_sub_blob_variance_HW = ((current_sub_blob_H_indices - current_sub_blob_center_H) * (current_sub_blob_W_indices - current_sub_blob_center_W)).mean().item()
                current_sub_blob_covariance = 1 * torch_covariance_template
                current_sub_blob_covariance[0, 0] = current_sub_blob_variance_HH
                current_sub_blob_covariance[1, 1] = current_sub_blob_variance_WW
                current_sub_blob_covariance[0, 1] = current_sub_blob_variance_HW
                current_sub_blob_covariance[1, 0] = current_sub_blob_variance_HW
                current_sub_blob_center_W_int = int(current_sub_blob_center_W)
                current_sub_blob_center_H_int = int(current_sub_blob_center_H)

                ### Replace Every Sub-Blob With One "Equivalent" Outlier In The Center Of The Sub-Blob: ###
                if current_label_number_of_pixels >= number_of_components_tuple[0] and current_label_number_of_pixels <= number_of_components_tuple[1]:
                    # (*). Put 1 where the center of the blob is:
                    current_frame_sub_frame_new[:, :, int(np.ceil(current_sub_blob_center_H)), int(np.ceil(current_sub_blob_center_W))] = 1
                    current_frame_sub_frame_new[:, :, int(np.floor(current_sub_blob_center_H)), int(np.ceil(current_sub_blob_center_W))] = 1
                    current_frame_sub_frame_new[:, :, int(np.ceil(current_sub_blob_center_H)), int(np.floor(current_sub_blob_center_W))] = 1
                    current_frame_sub_frame_new[:, :, int(np.floor(current_sub_blob_center_H)), int(np.floor(current_sub_blob_center_W))] = 1
                else:
                    current_frame_sub_frame_new[current_sub_blob_logical_mask] = 1

                ### Register Sub-Blob Stats: ###
                sublob_center_per_frame_per_label_list[batch_index].append((current_sub_blob_center_H, current_sub_blob_center_W))
                sublob_covariance_per_frame_per_label_list[batch_index].append(current_sub_blob_covariance)

            ### Replace Large Frame Outlier Blob With One Outlier In The Center: ###
            current_frame_output[:, :, start_H:stop_H, start_W:stop_W] = current_frame_sub_frame_new

    return list_of_BB, input_tensor_reduced, \
           sublob_center_per_frame_per_label_list, sublob_covariance_per_frame_per_label_list, \
           number_of_BBs_per_frame, number_of_pixels_per_frame_per_label_list


import torch.nn as nn
def get_blobs_by_binning(outliers_tensor):
    ### Get Outliers Tensor After Average Pooling: ###
    # outliers_tensor_2x2 = average_layer_2x2.forward(outliers_tensor)
    # outliers_tensor_4x4 = average_layer_4x4.forward(outliers_tensor)
    # outliers_tensor_8x8 = average_layer_8x8.forward(outliers_tensor)
    # outliers_tensor_16x16 = average_layer_16x16.forward(outliers_tensor)
    # ### Return Full Resolution: ###
    # outliers_tensor_2x2_full = fast_binning_2D_AvgPool2d(outliers_tensor, (2, 2), (1, 1), flag_sum_instead_of_average=True)
    # outliers_tensor_4x4_full = fast_binning_2D_AvgPool2d(outliers_tensor, (4, 4), (3, 3), flag_sum_instead_of_average=True)
    # outliers_tensor_8x8_full = fast_binning_2D_AvgPool2d(outliers_tensor, (8, 8), (7, 7), flag_sum_instead_of_average=True)
    # outliers_tensor_16x16_full = fast_binning_2D_AvgPool2d(outliers_tensor, (16, 16), (15, 15), flag_sum_instead_of_average=True)
    # ### Return Binned Resolution: ###
    # outliers_tensor_2x2_binned = fast_binning_2D_AvgPool2d(outliers_tensor, (2, 2), (0, 0), flag_sum_instead_of_average=True)
    # outliers_tensor_4x4_binned = fast_binning_2D_AvgPool2d(outliers_tensor, (4, 4), (0, 0), flag_sum_instead_of_average=True)
    # outliers_tensor_8x8_binned = fast_binning_2D_AvgPool2d(outliers_tensor, (8, 8), (0, 0), flag_sum_instead_of_average=True)
    outliers_tensor_16x16_binned = fast_binning_2D_AvgPool2d(outliers_tensor, (16, 16), (0, 0), flag_sum_instead_of_average=True)

    ### Threshold Tensors: ###
    outliers_tensor_16x16_binned = (outliers_tensor_16x16_binned > 5).float()  #TODO: make this a parameter

    ### Get Connected Components On DownSampled Tensor: ###
    ### Activate Connected Components Algorithm On Each Sub-Image: ###
    CC_number_of_iterations = 15
    number_of_components_tuple = (0, np.inf)
    label_zero_value = 0
    outputs_list = \
        get_connected_components_logical_mask_batch(outliers_tensor_16x16_binned,
                                              number_of_iterations=CC_number_of_iterations,
                                              number_of_components_tuple=number_of_components_tuple,
                                              label_zero_value=label_zero_value,
                                              flag_return_logical_mask_or_label_values=False,
                                              flag_return_number_of_components=False,
                                              flag_return_COM_and_Covariance=False,
                                              flag_return_BB=True)
    BB_list = outputs_list


    # imshow_torch_video(outliers_tensor, FPS=5, frame_stride=2)
    # imshow_torch_video(outliers_tensor_16x16_binned, FPS=5, frame_stride=2)
    # imshow_torch(outliers_tensor[-1])
    # imshow_torch(outliers_tensor_16x16_binned[-1])

    ### Loop Over BB Found In Binned Image And Perform Connected Components In Full Resolution On Respective Bounding-Boxes: ###
    BB_full_resolution_list = get_full_resolution_BB_from_low_resolution_BB_batch(BB_list, binning_factor=16)

    ### Perform Connected Components Labeling On Bounding-Boxes Within Full Resolution Images: ###
    #TODO: add conditions and maybe simply erase big blobs
    #TODO: no need to actually get outliers_tensor_reduced!!!! i only need a list of outliers really!!! --> add a flag for this!!!!
    list_of_BB, outliers_tensor_reduced, \
    sublob_center_per_frame_per_label_list, sublob_covariance_per_frame_per_label_list, \
    number_of_BBs_per_frame, number_of_pixels_per_frame_per_label_list =\
        get_reduced_outliers_tensor_from_BB_using_connected_components_within_BB(outliers_tensor,
                                                                                 BB_full_resolution_list,
                                                                                 CC_number_of_iterations=20,
                                                                                 number_of_components_tuple=(0,np.inf))

    # imshow_torch_video(outliers_tensor_reduced, FPS=5, frame_stride=2)
    # imshow_torch_video(outliers_tensor, FPS=5)

    # ### Present Bounding-Boxes On Full-Resolution Frame: ###
    # outliers_tensor_with_BB_list = []
    # for frame_index in np.arange(outliers_tensor.shape[0]):
    #     current_outliers_tensor_numpy = torch_to_numpy(outliers_tensor[frame_index].squeeze())
    #     current_BB_list = BB_full_resolution_list[frame_index]
    #     current_full_resolution_with_BB = draw_bounding_boxes_with_labels_on_image_XYXY(BW2RGB(current_outliers_tensor_numpy) * 255,
    #                                                         BB_tuples_list=current_BB_list,
    #                                                         BB_labels_list=[''],
    #                                                         line_thickness=2,
    #                                                         color=(255, 0, 0),
    #                                                         flag_draw_on_same_image=False)
    #     outliers_tensor_with_BB_list.append(numpy_to_torch(current_full_resolution_with_BB).unsqueeze(0))
    # outliers_tensor_with_BB_list = torch.cat(outliers_tensor_with_BB_list)
    # # imshow_torch_video(outliers_tensor_with_BB_list, FPS=10)

    return outliers_tensor_reduced




class Blob_Detection_Layer(torch.nn.Module):
    def __init__(self,
                 number_of_components_tuple=(10,np.inf),
                 CC_number_of_iterations=25,
                 number_of_sigmas=5,
                 flag_sigma_grid='linear',
                 minimum_sigma=2,
                 maximum_sigma=5,
                 kernel_size_to_sigma_factor=1.3,
                 sigma_size_basis=1.414,
                 detection_threshold=0.1,
                 iou_threshold_modified=0.02,
                 maxpool_spatial_kernel_size=3,
                 bounding_box_frame_addition_for_robust_overlap=5):
        super(Blob_Detection_Layer, self).__init__()

        ### List of sigmas to check: ###
        self.sigmas_list = []
        if flag_sigma_grid == 'geometric':
            # (1). Geometric Grid:
            for sigma_power in np.arange(number_of_sigmas):
                self.sigmas_list.append(sigma_size_basis ** sigma_power)
        elif flag_sigma_grid == 'linear':
            # (2). Linear Grid:
            self.sigmas_list = np.linspace(minimum_sigma, maximum_sigma, number_of_sigmas)

        ### Convolve Images With The Different Sized LOG Filters (with the different sigmas): ###
        # TODO: extend filter_2D_torch to accept also strides!!!!!!! to be able to lower the effective resolution of the blob detection because there's really no need for this!!!  in any case i'm doing NMS!!!!
        # TODO: try and use simple fast_binning_2D and simple connected components or something to search for blobs!!!
        self.log_convolve_torch_layer = LoG_convolve_torch_layer(self.sigmas_list, kernel_size_to_sigma_factor=kernel_size_to_sigma_factor)
        self.detection_threshold = detection_threshold
        self.iou_threshold_modified = iou_threshold_modified
        self.maxpool_spatial_kernel_size = maxpool_spatial_kernel_size
        self.bounding_box_frame_addition_for_robust_overlap = bounding_box_frame_addition_for_robust_overlap
        self.number_of_components_tuple = number_of_components_tuple
        self.CC_number_of_iterations = CC_number_of_iterations

    def forward(self, input_tensor, flag_plot_BB_on_output_tensor=True):
        ### Convolve Images With The Different Sized LOG Filters (with the different sigmas): ###
        filters_response_tensor = self.log_convolve_torch_layer.forward(input_tensor)

        ### Pytorch Detect Blobs: ###
        co_ordinates = detect_blobs_torch(input_tensor,
                                          filters_response_tensor,
                                          sigmas_list=self.sigmas_list,
                                          maxpool_spatial_kernel_size=self.maxpool_spatial_kernel_size,
                                          detection_threshold=self.detection_threshold,
                                          iou_threshold_modified=self.iou_threshold_modified)  # [B,T,C,H,W,sigma]

        ### Connect Blobs Together To Create Large Bounding-Boxes: ###
        # TODO: try and use NMS pytorch functions to search for overlapping circles to create overlapping circles sets instead of transferring to numpy
        co_ordinates = co_ordinates.cpu().numpy()
        co_ordinates[:, -1] = co_ordinates[:, -1] + self.bounding_box_frame_addition_for_robust_overlap
        list_of_connected_blobs = get_minimum_set_of_overlapping_circles_torch_batch(co_ordinates, input_tensor_shape=input_tensor.shape)

        ### Loop Over Sets And Create Large Bounding Boxes: ###
        list_of_BB = get_bounding_boxes_from_list_of_connected_blobs(list_of_connected_blobs, co_ordinates)

        ### Draww Bounding Boxes On Image Using Kornia Utils: ###
        BB_tensor = list_of_BB_lists_to_torch_tensor(list_of_BB)  # returns a [BB_tensor] = [B, max_number_of_BB_in_a_single_frame, 4]
        if flag_plot_BB_on_output_tensor:
            output_tensor_with_BB = kornia.utils.draw_rectangle(input_tensor, BB_tensor, color=torch.tensor([1]), flag_draw_inplace=False)
        else:
            output_tensor_with_BB = None

        ### Initialize Things For Coming Loops: ###
        number_of_BBs_per_frame = []
        number_of_pixels_per_frame_per_label_list = create_empty_list_of_lists(input_tensor.shape[0])
        sublob_center_per_frame_per_label_list = create_empty_list_of_lists(input_tensor.shape[0])
        sublob_covariance_per_frame_per_label_list = create_empty_list_of_lists(input_tensor.shape[0])

        ### Initialize input_tensor_reduced: ###
        input_tensor_reduced = torch.ones_like(input_tensor) * input_tensor


        ### Loop Over Each Frame, And For Each Frame Loop Over All Bounding-Boxes And Reduce All Connected-Components Sub-Blobs There To Single Outliers: ###
        for batch_index in np.arange(input_tensor.shape[0]):
            # print(batch_index)
            ### Get all bounding boxes in current frame: ###
            current_BBs = list_of_BB[batch_index]
            number_of_BBs_per_frame.append(len(current_BBs))

            ### Extract All Bounding Boxes As Sub-Images Into a List: ###
            current_frame = input_tensor[batch_index:batch_index+1]
            current_frame_sub_frames_list = []
            current_frame_output = input_tensor_reduced[batch_index:batch_index+1]
            torch_covariance_template = torch.ones((2,2)).to(input_tensor.device)

            ### Loop Over All Bounding Boxes For This Frame, Find Sub-Blobs, And Reduce Them To Single Point: ###
            for frame_BB_index in np.arange(len(current_BBs)):
                ### Get Current Sub Frame Contained Inside Bounding Box: ###
                current_BB = current_BBs[frame_BB_index]
                start_W = current_BB[0]
                start_H = current_BB[1]
                stop_W = current_BB[2]
                stop_H = current_BB[3]
                current_frame_sub_frame = current_frame[:, :, start_H:stop_H, start_W:stop_W]
                current_frame_sub_frames_list.append(current_frame_sub_frame)

                ### Activate Connected Components Algorithm On Each Sub-Image: ###
                CCL_outputs_list = get_connected_components_logical_mask(current_frame_sub_frame,
                                                                                  number_of_iterations=self.CC_number_of_iterations,
                                                                                  number_of_components_tuple=(0, np.inf),  #allowed sizes
                                                                                  label_zero_value=0)
                current_BB_CC_logical_mask = CCL_outputs_list[0]

                ### Find Sub-Blobs / Different Labels In The Connected-Components Logical Mask: ###
                CC_logical_mask_unique_values = current_BB_CC_logical_mask.unique()  #starts with 0 which is the BG
                number_of_sub_blobs = CC_logical_mask_unique_values.numel()

                ### Initialize New Sub Frame: ###
                current_frame_sub_frame_new = torch.zeros_like(current_frame_sub_frame)

                ### Find Center For Each Sub-Blob: ###
                for CC_label_index in np.arange(1, number_of_sub_blobs):
                    ### Get Current Sub-Blob / Label (from connected components analysis): ###
                    current_label = CC_logical_mask_unique_values[CC_label_index]
                    current_sub_blob_logical_mask = (current_BB_CC_logical_mask == current_label)
                    current_label_number_of_pixels = current_sub_blob_logical_mask.sum().item()
                    number_of_pixels_per_frame_per_label_list[batch_index].append(current_label_number_of_pixels)
                    current_sub_blob_indices = logical_mask_to_indices_torch(current_sub_blob_logical_mask, flag_return_tensor_or_list_of_tuples='tensor')

                    ### Find Sub-Blob Center: ###
                    current_sub_blob_W_indices = current_sub_blob_indices[:, -1].float()
                    current_sub_blob_H_indices = current_sub_blob_indices[:, -2].float()
                    current_sub_blob_center_W = current_sub_blob_W_indices.mean().item()
                    current_sub_blob_center_H = current_sub_blob_H_indices.mean().item()
                    current_sub_blob_variance_WW = ((current_sub_blob_W_indices - current_sub_blob_center_W) * (current_sub_blob_W_indices - current_sub_blob_center_W)).mean().item()
                    current_sub_blob_variance_HH = ((current_sub_blob_H_indices - current_sub_blob_center_H) * (current_sub_blob_H_indices - current_sub_blob_center_H)).mean().item()
                    current_sub_blob_variance_HW = ((current_sub_blob_H_indices - current_sub_blob_center_H) * (current_sub_blob_W_indices - current_sub_blob_center_W)).mean().item()
                    current_sub_blob_covariance = 1 * torch_covariance_template
                    current_sub_blob_covariance[0,0] = current_sub_blob_variance_HH
                    current_sub_blob_covariance[1,1] = current_sub_blob_variance_WW
                    current_sub_blob_covariance[0,1] = current_sub_blob_variance_HW
                    current_sub_blob_covariance[1,0] = current_sub_blob_variance_HW
                    current_sub_blob_center_W_int = int(current_sub_blob_center_W)
                    current_sub_blob_center_H_int = int(current_sub_blob_center_H)

                    ### Replace Every Sub-Blob With One "Equivalent" Outlier In The Center Of The Sub-Blob: ###
                    if current_label_number_of_pixels >= self.number_of_components_tuple[0] and current_label_number_of_pixels <= self.number_of_components_tuple[1]:
                        #(*). Put 1 where the center of the blob is:
                        current_frame_sub_frame_new[:, :, int(np.ceil(current_sub_blob_center_H)), int(np.ceil(current_sub_blob_center_W))] = 1
                        current_frame_sub_frame_new[:, :, int(np.floor(current_sub_blob_center_H)), int(np.ceil(current_sub_blob_center_W))] = 1
                        current_frame_sub_frame_new[:, :, int(np.ceil(current_sub_blob_center_H)), int(np.floor(current_sub_blob_center_W))] = 1
                        current_frame_sub_frame_new[:, :, int(np.floor(current_sub_blob_center_H)), int(np.floor(current_sub_blob_center_W))] = 1
                    else:
                        current_frame_sub_frame_new[current_sub_blob_logical_mask] = 1

                    ### Register Sub-Blob Stats: ###
                    sublob_center_per_frame_per_label_list[batch_index].append((current_sub_blob_center_H, current_sub_blob_center_W))
                    sublob_covariance_per_frame_per_label_list[batch_index].append(current_sub_blob_covariance)

                ### Replace Large Frame Outlier Blob With One Outlier In The Center: ###
                current_frame_output[:, :, start_H:stop_H, start_W:stop_W] = current_frame_sub_frame_new

                # imshow_torch(current_frame_sub_frame)
                # imshow_torch(current_frame_sub_frame_new)
                # imshow_torch(current_sub_blob_logical_mask)
                # imshow_torch(current_frame)
                # imshow_torch(current_frame_output)
                # imshow_torch(input_tensor)
                # imshow_torch(output_tensor)

        # ### Show Results: ###
        # imshow_torch_video(input_tensor, FPS=10, frame_stride=1)
        # imshow_torch_video(output_tensor, FPS=10, frame_stride=1)
        # concat_tensor = torch.cat([input_tensor, output_tensor], -1)
        # imshow_torch_video(concat_tensor, FPS=10, frame_stride=1)
        # video_torch_array_to_video(concat_tensor, video_name=r'C:\Users\dudyk\Downloads/outlier_reduction.avi', FPS=10)
        return list_of_BB, BB_tensor, input_tensor_reduced, output_tensor_with_BB,\
               sublob_center_per_frame_per_label_list, sublob_covariance_per_frame_per_label_list,\
               number_of_BBs_per_frame, number_of_pixels_per_frame_per_label_list


class LoG_convolve_torch_layer(torch.nn.Module):
    def __init__(self, sigmas_list, device='cuda', kernel_size_to_sigma_factor=2):
        super(LoG_convolve_torch_layer, self).__init__()
        self.filter_list = []
        for sigma_index, current_sigma in enumerate(sigmas_list):
            ### Get Current Sigma LOG Filter: ###
            current_filter = get_LOG_filter_torch(current_sigma, None, device, kernel_size_to_sigma_factor=kernel_size_to_sigma_factor)
            self.filter_list.append(current_filter)

    def forward(self, input_tensor):
        log_images = []
        for filter_index, current_filter in enumerate(self.filter_list):
            ### Filter Image And Square It: ###
            output_tensor = filter2D_torch(input_tensor, current_filter.unsqueeze(0)) ** 2

            ### Add To Images List: ###
            log_images.append(output_tensor)

        ### Concatenate All Images Into Single Tensor: ###
        log_images = torch.cat(log_images, 1)  # TX[B,1,H,W] -> [B,T,H,W]

        return log_images

def LoG_convolve_torch(input_tensor, sigmas_list):
    ### Loop Over The Different Sigmas And Get The Appropriate Filter Response: ###
    log_images = []
    for sigma_index, current_sigma in enumerate(sigmas_list):
        ### Get Current Sigma LOG Filter: ###
        current_filter = get_LOG_filter_torch(current_sigma)

        ### Filter Image And Square It: ###
        output_tensor = filter2D_torch(input_tensor, current_filter.unsqueeze(0)) ** 2

        ### Add To Images List: ###
        log_images.append(output_tensor)

    ### Concatenate All Images Into Single Tensor: ###
    log_images = torch.cat(log_images, 1)  #TX[B,1,H,W] -> [B,T,H,W]

    return log_images


def blobs_intersect(blob1, blob2):
    ### Get Circle Radii: ###
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim
    d = sqrt(np.sum((blob1[:-1] - blob2[:-1]) ** 2))
    if d>r1+r2:
        return 0
    else:
        return 1
    
def blobs_area_overlap(blob1, blob2):
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)
    # print(n_dim)

    # radius of two blobs
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim

    d = sqrt(np.sum((blob1[:-1] - blob2[:-1]) ** 2))
    
    # no overlap between two blobs
    if d > r1 + r2:
        return 0
    # one blob is inside the other, the smaller blob must die
    elif d <= abs(r1 - r2):
        return 1
    else:
        # computing the area of overlap between blobs
        ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
        ratio1 = np.clip(ratio1, -1, 1)
        acos1 = math.acos(ratio1)

        ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
        ratio2 = np.clip(ratio2, -1, 1)
        acos2 = math.acos(ratio2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1

        area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d)))
        return area / (math.pi * (min(r1, r2) ** 2))


def blobs_area_overlap_torch(blob1, blob2):
    n_dim = len(blob1) - 1
    root_ndim = sqrt(n_dim)
    # print(n_dim)

    # radius of two blobs
    r1 = blob1[-1] * root_ndim
    r2 = blob2[-1] * root_ndim

    d = sqrt(np.sum((blob1[-3:-1] - blob2[-3:-1]) ** 2))

    # no overlap between two blobs
    if d > r1 + r2:
        return 0
    # one blob is inside the other, the smaller blob must die
    elif d <= abs(r1 - r2):
        return 1
    else:
        # computing the area of overlap between blobs
        ratio1 = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
        ratio1 = np.clip(ratio1, -1, 1)
        acos1 = math.acos(ratio1)

        ratio2 = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
        ratio2 = np.clip(ratio2, -1, 1)
        acos2 = math.acos(ratio2)

        a = -d + r2 + r1
        b = d - r2 + r1
        c = d + r2 - r1
        d = d + r2 + r1

        area = (r1 ** 2 * acos1 + r2 ** 2 * acos2 - 0.5 * sqrt(abs(a * b * c * d)))
        return area / (math.pi * (min(r1, r2) ** 2))


def redundancy(blobs_array, overlap):
    sigma = blobs_array[:, -1].max()
    distance = 2 * sigma * sqrt(blobs_array.shape[1] - 1)

    ### Build a tree of all coordinates: ###
    tree = spatial.cKDTree(blobs_array[:, :-1])

    ### Get all coordinates within a certain distance=distance from each other: ###
    #TODO: use Non-maximum suppression instead probably!!!
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if blobs_area_overlap(blob1, blob2) > overlap:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.array([b for b in blobs_array if b[-1] > 0])


def delete_smaller_overlapping_circles(circle_tuple_1, circle_tuple_2, overlap_threshold):
    if blobs_area_overlap(circle_tuple_1, circle_tuple_2) > overlap_threshold:
        if circle_tuple_1[-1] > circle_tuple_2[-1]:
            circle_tuple_2[-1] = 0
        else:
            circle_tuple_1[-1] = 0


def get_GCC(blobs_array):
    ### [blobs_array] = [N,3]
    n = len(blobs_array)
    circles = [Point(x[0], x[1]).buffer(x[2]) for x in blobs_array]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i, circle_i in enumerate(circles):
        for j, circle_j in enumerate(circles):
            if circle_i.intersects(circle_j):
                G.add_edge(i, j)

    return list(nx.connected_components(G))


def get_minimum_set_of_overlapping_circles_using_loops(blobs_array):
    points_not_assigned_to_chain_yet = []
    list_of_connected_sets = []
    circle_set_list = -1 * np.ones((blobs_array.shape[0])).astype(int)
    set_counter = 0
    for blob_index_1 in np.arange(blobs_array.shape[0]):
        ### Get Current Circle Tuple: ###
        blob_tuple_1 = blobs_array[blob_index_1]
        H1, W1, sigma_1 = blob_tuple_1
        flag_current_blob_assigned = False

        ### Loop Over All Sets: ###
        for set_index in np.arange(len(list_of_connected_sets)):
            current_set = list_of_connected_sets[set_index]
            ### Loop Over all elements in each set: ###
            for intra_set_index in np.arange(len(current_set)):
                current_set_element_index_in_original_array = current_set[intra_set_index]
                blob_tuple_2 = blobs_array[current_set_element_index_in_original_array]
                ### If the above blobs intersects with any of the current set members, add current blob to set and exit: ###
                if blobs_intersect(blob_tuple_1, blob_tuple_2):
                    current_set.append(blob_index_1)  # add current blob to set
                    circle_set_list[blob_index_1] = set_index
                    flag_current_blob_assigned = True

        ### If current blob not associated with any existing set, open a new set for it: ###
        if flag_current_blob_assigned == False:
            list_of_connected_sets.append([blob_index_1])  # add an element to the list, a new set containing blob_index as a member
            circle_set_list[blob_index_1] = set_counter  # update the appropriate list to keep track of which set does the current element belong to
            set_counter += 1

    return list_of_connected_sets

def blobs_zero_out_overlapping_circles(blob_tuple_1, blob_tuple_2, overlap_threshold):
    if blobs_area_overlap(blob_tuple_1, blob_tuple_2) > overlap_threshold:
        if blob_tuple_1[-1] > blob_tuple_2[-1]:
            blob_tuple_2[-1] = 0
        else:
            blob_tuple_1[-1] = 0

def NMS_for_blobs_torch(blobs_array, overlap_threshold):
    #blob_array is assumed to be a numpy array
    ### Get Rid Of All Circles With A Significant Overlap: ###

    ### If Array Is Empty Just Return It: ###
    if blobs_array.nbytes == 0:
        return blobs_array

    ### Get Max Distance Between Pairs: ###
    max_sigma = blobs_array[:, -1].max()
    distance = 2 * max_sigma * sqrt(blobs_array.shape[1] - 1)

    ### Build a tree of all coordinates: ###
    tree = spatial.cKDTree(blobs_array[:, -2:])
    ### Get all coordinates within a certain distance=distance from each other: ###
    pairs = np.array(list(tree.query_pairs(distance)))

    ### Get Rid Of All Circles With A Significant Overlap: ###
    if len(pairs) == 0:
        return blobs_array
    else:
        ### Perform NMS By Looping: ###
        for (i, j) in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if blobs_area_overlap_torch(blob1, blob2) > overlap_threshold:
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
        blobs_array = np.array([b for b in blobs_array if b[-1] > 0])


    # ### Perform NMS By List-Comprehenssion: ###
    # bla = [blobs_zero_out_overlapping_circles(blobs_array[i], blobs_array[j], overlap_threshold) for (i, j) in pairs]
    # blobs_array = np.array([b for b in blobs_array if b[-1] > 0])

    return blobs_array


def NMS_for_blobs_torch_batch(blobs_array, overlap_threshold, input_tensor_shape):
    ### Get Rid Of All Circles With A Significant Overlap: ###
    B,C,H,W = input_tensor_shape
    blobs_array = np.concatenate([NMS_for_blobs_torch(blobs_array[blobs_array[:,0]==i], overlap_threshold) for i in np.arange(B)])
    return blobs_array


def get_minimum_set_of_overlapping_circles(blobs_array):
    ### Get Minimum Sets Of Overlapping Circles: ###
    
    ### Get Connected Circles Using GCC (Graph Connected Components): ###
    list_of_connected_sets = get_GCC(blobs_array)

    # ### Perform Connected Circles Analysis By Looping: ###
    # list_of_connected_sets = get_minimum_set_of_overlapping_circles_using_loops(blobs_array)

    return list_of_connected_sets

from RapidBase.Utils.IO.tic_toc import *
def get_minimum_set_of_overlapping_circles_torch_batch(blobs_array, input_tensor_shape):
    ### Get Minimum Sets Of Overlapping Circles: ###
    B, C, H, W = input_tensor_shape

    ### Get Connected Circles Using GCC (Graph Connected Components): ###
    list_of_connected_sets = [get_GCC(blobs_array[blobs_array[:,0]==i][:, -3:]) for i in np.arange(B)]

    # ### Perform Connected Circles Analysis By Looping: ###
    # list_of_connected_sets = [get_minimum_set_of_overlapping_circles_using_loops(blobs_array[blobs_array[:,0]==i][:, -3:]) for i in np.arange(B)]

    return list_of_connected_sets

def detect_blobs(img, log_image_np, sigma_size_basis, basic_sigma_factor, detection_threshold=0.03, minimum_sigma=5):
    co_ordinates = []
    if len(img.shape) == 2:
        H,W = img.shape
    elif len(img.shape) == 3:
        H,W,C = img.shape

    ### Loop over all pixels and for each pixel Spatio-Temporal patch (T=9xH=3xW=3) find the patch-max: ###
    for i in range(1, H):
        for j in range(1, W):
            slice_img = log_image_np[:, i-1:i+2, j-1:j+2]
            result = np.amax(slice_img)  #max along a specific dimension (the dim=0 i suppose)
            #result_1 = np.amin(slice_img)

            if result >= detection_threshold:
                z,x,y = np.unravel_index(slice_img.argmax(), slice_img.shape)  #if the max is above a certain threshold get the index of the patch
                # print((sigma_size_basis**z)*basic_sigma_factor)
                if (sigma_size_basis**z)*basic_sigma_factor > minimum_sigma:
                    co_ordinates.append((i+x-1, j+y-1, (sigma_size_basis**z)*basic_sigma_factor))
                    # co_ordinates.append([i+x-1, j+y-1, (sigma_size_basis**z)*basic_sigma_factor])
    return co_ordinates

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_to_numpy, numpy_to_torch
def detect_blobs_torch(input_tensor, filters_response_tensor, sigmas_list, maxpool_spatial_kernel_size=3, detection_threshold=0.03, iou_threshold_modified=0.05):
    ### Initialize Stuff: ###
    co_ordinates = []
    (B1, T1, C1, H1, W1), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    (B2, T2, C2, H2, W2), shape_len, shape_vec = get_full_shape_torch(filters_response_tensor)

    ### Loop over all pixels and for each pixel Spatio-Temporal patch (T=9xH=3xW=3) find the patch-max: ###
    # (1). define 3d maxpool layer to find the max indices and values over each spatial patch over the entire sigmas list
    filters_response_tensor = filters_response_tensor.unsqueeze(-4)
    maxpool_kernel_size_final = (C2, maxpool_spatial_kernel_size, maxpool_spatial_kernel_size)
    maxpool_padding_size_final = (0, maxpool_spatial_kernel_size // 2, maxpool_spatial_kernel_size // 2)
    maxpool_layer = torch.nn.MaxPool3d(kernel_size=maxpool_kernel_size_final, stride=1, padding=maxpool_padding_size_final, return_indices=True)
    maxpool_values, maxpool_raveled_indices = maxpool_layer.forward(filters_response_tensor)
    ### I think maxpool_raveled_indices return indices between [0,H*W*D*C]....doesn't include the B dimension!!!!...that's why B_indices_vec is all zeros...correct this: ###
    maxpool_raveled_indices += ((H2*W2*C2)*torch.arange(0,T2)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(input_tensor.device)
    ### Get Unraveled Tuple Of Indices: ###
    maxpool_indices_tuple_of_tensors = unravel_index_torch(maxpool_raveled_indices, filters_response_tensor.shape)

    # (2). Threshold maxpool values and only get the indices and values above the certain threshold
    logical_mask = maxpool_values > detection_threshold
    maxpool_values_vec = maxpool_values[logical_mask].unsqueeze(-1)
    B_indices_vec = maxpool_indices_tuple_of_tensors[0][logical_mask].unsqueeze(-1)
    T_indices_vec = maxpool_indices_tuple_of_tensors[1][logical_mask].unsqueeze(-1)
    C_indices_vec = maxpool_indices_tuple_of_tensors[2][logical_mask].unsqueeze(-1)
    sigmas_tensor = torch.tensor(sigmas_list).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, H2, W2).to(input_tensor.device)
    C_indices_tensor = maxpool_indices_tuple_of_tensors[2]
    max_sigmas_tensor = torch.gather(sigmas_tensor, dim=0, index=C_indices_tensor)
    max_sigma_vec = max_sigmas_tensor[logical_mask].unsqueeze(-1)
    H_indices_vec = maxpool_indices_tuple_of_tensors[3][logical_mask].unsqueeze(-1)
    W_indices_vec = maxpool_indices_tuple_of_tensors[4][logical_mask].unsqueeze(-1)

    # (3). Final form of output:
    BB_indices_vec_H_start = H_indices_vec - max_sigma_vec
    BB_indices_vec_H_stop = H_indices_vec + max_sigma_vec
    BB_indices_vec_W_start = W_indices_vec - max_sigma_vec
    BB_indices_vec_W_stop = W_indices_vec + max_sigma_vec
    # final_indices_tensor = torch.cat([B_indices_vec, T_indices_vec, C_indices_vec, H_indices_vec, W_indices_vec, max_sigma_vec,
    #                                   BB_indices_vec_H_start, BB_indices_vec_H_stop, BB_indices_vec_W_start, BB_indices_vec_W_stop], -1)
    # final_indices_tensor = torch.cat([B_indices_vec, T_indices_vec, C_indices_vec, H_indices_vec, W_indices_vec, max_sigma_vec], -1)
    # final_indices_tensor = torch.unique(final_indices_tensor, dim=0)

    #(4). NMS:
    boxes_tensor_NMS = torch.cat([BB_indices_vec_H_start, BB_indices_vec_H_stop, BB_indices_vec_W_start, BB_indices_vec_W_stop], -1) #TODO: comment this out!!!!
    # boxes_tensor_NMS = torch.cat([BB_indices_vec_W_start, BB_indices_vec_H_start, BB_indices_vec_W_stop, BB_indices_vec_H_stop], -1) #TODO; uncomment this in!!!!
    # iou_threshold = 0.9  #discard all overlapping boxes with IOU>threshold, so threshold=0 means -> keep nothing; threshold=1 means -> keep everything
    # iou_threshold_modified = 0.05 #keep all overlapping boxes with IOU<threshold, so threshold=0 means -> keep everything; threshold=1 means -> keep nothing
    remaining_indices = torchvision.ops.batched_nms(boxes=boxes_tensor_NMS,
                                                    scores=maxpool_values_vec.squeeze(-1),
                                                    idxs=B_indices_vec.squeeze(-1),
                                                    iou_threshold=1-iou_threshold_modified)
    B_indices_vec_after_NMS = B_indices_vec[remaining_indices]
    T_indices_vec_after_NMS = T_indices_vec[remaining_indices]
    C_indices_vec_after_NMS = C_indices_vec[remaining_indices]
    H_indices_vec_after_NMS = H_indices_vec[remaining_indices]
    W_indices_vec_after_NMS = W_indices_vec[remaining_indices]
    BB_indices_vec_H_start_after_NMS = BB_indices_vec_H_start[remaining_indices]
    BB_indices_vec_H_stop_after_NMS = BB_indices_vec_H_stop[remaining_indices]
    BB_indices_vec_W_start_after_NMS = BB_indices_vec_W_start[remaining_indices]
    BB_indices_vec_W_stop_after_NMS = BB_indices_vec_W_stop[remaining_indices]
    max_sigma_vec_after_NMS = max_sigma_vec[remaining_indices]
    BB_XYXY_coordinates_tensor_after_NMS = boxes_tensor_NMS[remaining_indices]

    #(5). Return What's left after NMS:
    final_indices_tensor = torch.cat([B_indices_vec_after_NMS,
                                      T_indices_vec_after_NMS,
                                      C_indices_vec_after_NMS,
                                      H_indices_vec_after_NMS,
                                      W_indices_vec_after_NMS,
                                      max_sigma_vec_after_NMS], -1)
    final_indices_tensor_XYXY = torch.cat([B_indices_vec_after_NMS,
                                           T_indices_vec_after_NMS,
                                           BB_XYXY_coordinates_tensor_after_NMS], -1)

    # ## Show NMS Results: ###
    # boxes_tensor_NMS = torch.cat([BB_indices_vec_H_start_after_NMS, BB_indices_vec_H_stop_after_NMS, BB_indices_vec_W_start_after_NMS, BB_indices_vec_W_stop_after_NMS], -1)
    # new_maxpool_vec = maxpool_values_vec[remaining_indices]
    # remaining_indices_2 = torchvision.ops.batched_nms(boxes=boxes_tensor_NMS,
    #                                                 scores=new_maxpool_vec.squeeze(-1),
    #                                                 idxs=B_indices_vec_after_NMS.squeeze(-1),
    #                                                 iou_threshold=iou_threshold_modified)

    # ### View All BB Before NMS: ###
    # final_indices_tensor_XYXY_before_NMS = torch.cat([B_indices_vec,
    #                                                    T_indices_vec,
    #                                                    boxes_tensor_NMS], -1)
    # final_indices_tensor_B_ordered = torch.sort(final_indices_tensor_XYXY_before_NMS, dim=0)[0]
    # final_indices_tensor_B_ordered = torch.unique(final_indices_tensor_B_ordered, dim=0)
    # output_list = []
    # for frame_index in np.arange(input_tensor.shape[0]):
    #     current_frame_tensor = input_tensor[frame_index]
    #     current_frame_numpy = torch_to_numpy(current_frame_tensor).squeeze()
    #     current_frame_BB = final_indices_tensor_B_ordered[final_indices_tensor_B_ordered[:, 0] == frame_index]
    #     current_frame_BB_XYXY = current_frame_BB[:, 2:]
    #     current_frame_BB_XYXY_numpy = current_frame_BB_XYXY.cpu().numpy().astype(int)
    #     current_frame_BB_XYXY_list = current_frame_BB_XYXY_numpy.tolist()
    #     current_frame_tensor_with_BB = draw_bounding_boxes_with_labels_on_image_XYXY(BW2RGB(current_frame_numpy) * 255,
    #                                                                                  BB_tuples_list=current_frame_BB_XYXY_list,
    #                                                                                  BB_labels_list=[''],
    #                                                                                  line_thickness=2,
    #                                                                                  color=(255, 0, 0),
    #                                                                                  flag_draw_on_same_image=False)
    #     output_list.append(numpy_to_torch(current_frame_tensor_with_BB).unsqueeze(0))
    # output_tensor = torch.cat(output_list)
    # imshow_torch_video(output_tensor, FPS=10)
    # #
    #
    # ### Order/Sort According To The B dimension: ###
    # final_indices_tensor_B_ordered = torch.sort(final_indices_tensor_XYXY, dim=0)[0]
    # final_indices_tensor_B_ordered = torch.unique(final_indices_tensor_B_ordered, dim=0)
    # output_list = []
    # for frame_index in np.arange(input_tensor.shape[0]):
    #     current_frame_tensor = input_tensor[frame_index]
    #     current_frame_numpy = torch_to_numpy(current_frame_tensor).squeeze()
    #     current_frame_BB = final_indices_tensor_B_ordered[final_indices_tensor_B_ordered[:,0] == frame_index]
    #     current_frame_BB_XYXY = current_frame_BB[:, 2:]
    #     current_frame_BB_XYXY_numpy = current_frame_BB_XYXY.cpu().numpy().astype(int)
    #     current_frame_BB_XYXY_list = current_frame_BB_XYXY_numpy.tolist()
    #     current_frame_tensor_with_BB = draw_bounding_boxes_with_labels_on_image_XYXY(BW2RGB(current_frame_numpy)*255,
    #                                                   BB_tuples_list=current_frame_BB_XYXY_list,
    #                                                   BB_labels_list=[''],
    #                                                   line_thickness=2,
    #                                                   color=(255,0,0),
    #                                                   flag_draw_on_same_image=False)
    #     output_list.append(torch.tensor(current_frame_tensor_with_BB).unsqueeze(0).unsqueeze(0))
    # output_tensor = torch.cat(output_list)
    # imshow_torch_video(output_tensor, FPS=10)


    return final_indices_tensor



def get_BB_from_blobs_set(current_set_indices, co_ordinates, flag_BB_mode='XYXY'):
    co_ordinates_subset = co_ordinates[current_set_indices]
    co_ordinates_subset_minus_R = co_ordinates_subset - co_ordinates_subset[:, -1:]
    co_ordinates_subset_plus_R = co_ordinates_subset + co_ordinates_subset[:, -1:]
    min_H = max(0, int(co_ordinates_subset_minus_R[:, -3].min()))
    max_H = int(co_ordinates_subset_plus_R[:, -3].max())
    min_W = max(0, int(co_ordinates_subset_minus_R[:, -2].min()))
    max_W = int(co_ordinates_subset_plus_R[:, -2].max())
    BB_H = max_H - min_H
    BB_W = max_W - min_W
    if flag_BB_mode == 'XYXY':
        BB_XYXY = np.array([min_W, min_H, max_W, max_H])
        return BB_XYXY
    else:
        BB_XYHW = np.array([min_W, min_H, BB_W, BB_H])
        return BB_XYHW

def get_bounding_boxes_from_list_of_connected_blobs(list_of_connected_blobs, co_ordinates):
    list_of_BB = []
    for batch_index in np.arange(len(list_of_connected_blobs)):
        ### Get current Batch Sets of indices: ###
        current_set_indices = list(list_of_connected_blobs[batch_index])
        ### Turn sets into lists: ###
        current_set_indices = [list(subset) for subset in current_set_indices]
        ### Get current Batch coordinates: ###
        current_coordinates = co_ordinates[co_ordinates[:,0]==batch_index]
        list_of_BB.append([get_BB_from_blobs_set(current_set_indices[set_index], current_coordinates, 'XYXY') for set_index in np.arange(len(current_set_indices))])
    return list_of_BB


def list_of_BB_lists_to_torch_tensor(list_of_BB, output_device='cuda'):
    ### Get Max Length Element: ###
    max_element_length = 0
    number_of_elements = len(list_of_BB)
    for i in np.arange(number_of_elements):
        current_BB_list = list_of_BB[i]
        max_element_length = max(max_element_length, len(current_BB_list))

    ### Initialize Tensor: ###
    BB_tensor = torch.zeros((number_of_elements, max_element_length, 4)).to(output_device)

    ### Loop Over and Fill Tensor: ###
    for i in np.arange(number_of_elements):
        current_BB_list = list_of_BB[i]
        for j in np.arange(len(current_BB_list)):
            BB_tensor[i, j, :] = torch.tensor(current_BB_list[j]).to(output_device)
            # BB_tensor[i, j, :] = current_BB_list[j]

    return BB_tensor

def plot_GaussianMixture_results_on_tensor(input_tensor, input_tensor_features, GMM_model, number_of_components):
    #[input_tensor] = [C,H,W]
    y = GMM_model.predict(input_tensor_features)
    n = y.shape[0]
    N, D = input_tensor_features.shape

    ### Plot Predictions: ###
    logical_masks_list = []
    input_tensor = BW2RGB(input_tensor)  #TODO: instead of this, later on take the indices and get the appropriate bounding box coordinates
    colors_list = get_n_colors(number_of_components)
    for label_index in np.arange(number_of_components):
        current_logical_mask = (y==label_index)
        current_HW_indices = input_tensor_features[current_logical_mask]
        input_tensor[:, current_HW_indices[:,0].long(), current_HW_indices[:,1].long()] = torch.tensor(colors_list[label_index]).unsqueeze(-1).repeat(1, current_HW_indices.shape[0]).to(input_tensor.device)
    return input_tensor

def plot_GMM_Batch_results_on_tensor(input_tensor, input_tensor_features, y, number_of_components):
    #[input_tensor] = [C,H,W]
    # y = GMM_model.predict(input_tensor_features)
    n = y.shape[0]
    N, D = input_tensor_features.shape

    ### Plot Predictions: ###
    logical_masks_list = []
    input_tensor = BW2RGB(input_tensor)  #TODO: instead of this, later on take the indices and get the appropriate bounding box coordinates
    colors_list = get_n_colors(number_of_components)
    for label_index in np.arange(number_of_components):
        current_logical_mask = (y==label_index)
        current_HW_indices = input_tensor_features[current_logical_mask]
        input_tensor[:, current_HW_indices[:,0].long(), current_HW_indices[:,1].long()] = torch.tensor(colors_list[label_index]).unsqueeze(-1).repeat(1, current_HW_indices.shape[0]).to(input_tensor.device)
    return input_tensor

def plot_kmeans_fun_gpu_results_on_tensor(input_tensor, input_tensor_features, kmeans_label, number_of_components):
    # [input_tensor] = [C,H,W]
    n = kmeans_label.shape[0]
    N, D = input_tensor_features.shape

    ### Plot Predictions: ###
    logical_masks_list = []
    input_tensor = BW2RGB(input_tensor)  # TODO: instead of this, later on take the indices and get the appropriate bounding box coordinates
    colors_list = get_n_colors(number_of_components)
    for label_index in np.arange(number_of_components):
        current_logical_mask = (kmeans_label == label_index)
        current_HW_indices = input_tensor_features[current_logical_mask]
        input_tensor[:, current_HW_indices[:, 0].long(), current_HW_indices[:, 1].long()] = torch.tensor(colors_list[label_index]).unsqueeze(-1).repeat(1, current_HW_indices.shape[0]).to(input_tensor.device)
    return input_tensor

from RapidBase.Utils.IO.Imshow_and_Plots import draw_bounding_boxes_with_labels_on_image_XYXY, draw_bounding_boxes_with_labels_on_images_XYXY
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import BW2RGB
from RapidBase.Utils.ML_and_Metrics.Clustering import GaussianMixture, kmeans_fun_gpu, GMM, GMM_Batch
from RapidBase.Utils.MISCELENEOUS import get_n_colors


def get_GaussianMixture_model_predictions_torch(input_tensor,
                                                input_tensor_features=None,
                                                number_of_components=6,
                                                number_of_iterations=20,
                                                flag_plot_colors_on_tensor=True,
                                                flag_plot_BB_on_tensor=True,
                                                device='cpu'):
    ### Initialize GaussianMixture model and fit: ###
    output_tensor = None
    if input_tensor_features is None:
        input_tensor_features = (input_tensor == 1).nonzero()[:, -2:].float()

    if input_tensor_features.shape[0] >= 10:
        GMM_model = GaussianMixture(number_of_components, 2).to(device)
        GMM_model.fit(input_tensor_features, n_iter=number_of_iterations, delta=1e-3)
        # .. used to predict the data points as they where shifted
        labels_prediction = GMM_model.predict(input_tensor_features)

        ### Plot colors of the different labels on tensor: ###
        if flag_plot_colors_on_tensor:
            output_tensor = plot_GaussianMixture_results_on_tensor(input_tensor, input_tensor_features, GMM_model, number_of_components)
            # imshow_torch(output_tensor)

        ### Add Bounding Boxes To Predictions: ###
        labels_BB_list = []
        for label_index in np.arange(number_of_components):
            current_label_indices = input_tensor_features[labels_prediction == label_index]
            if current_label_indices.numel() > 0:
                min_H = current_label_indices[:, 0].min().item()
                max_H = current_label_indices[:, 0].max().item()
                min_W = current_label_indices[:, 1].min().item()
                max_W = current_label_indices[:, 1].max().item()
                BB_height = max_H - min_H
                BB_width = max_W - min_W
                labels_BB_list.append([min_W, min_H, max_W, max_H])  # [XYXY]
                # labels_BB_list.append([min_W, min_H, BB_width, BB_height])  #[XYWH]
            else:
                labels_BB_list.append([0, 0, 0, 0])

        ### Plot Bounding Boxesd On Image: ###
        BB_tensor = torch.tensor(labels_BB_list).to(input_tensor.device).unsqueeze(0)
        if flag_plot_BB_on_tensor:
            if output_tensor is None:
                output_tensor = kornia.utils.draw_rectangle(torch_get_ND(input_tensor, number_of_dims=4), BB_tensor, color=torch.tensor([1]))
            else:
                output_tensor = kornia.utils.draw_rectangle(torch_get_ND(output_tensor, number_of_dims=4), BB_tensor, color=torch.tensor([1]))
            # imshow_torch(output_tensor)

    else:
        labels_prediction = torch.zeros(input_tensor_features.shape[0]).to(input_tensor_features.device)
        labels_BB_list = [[0, 0, 0, 0]]
        BB_tensor = torch.tensor(labels_BB_list).unsqueeze(0)
        if flag_plot_BB_on_tensor or flag_plot_colors_on_tensor:
            output_tensor = BW2RGB(torch_get_ND(input_tensor, number_of_dims=4))

    return labels_prediction, output_tensor, labels_BB_list, BB_tensor


def get_GaussianMixture_model_predictions_torch_batch(input_tensor,
                                       input_tensor_features=None,
                                       number_of_components=6,
                                       number_of_iterations=100,
                                       batch_size=128,
                                       flag_plot_colors_on_tensor=True,
                                       flag_plot_BB_on_tensor=True,
                                       device='cpu'):
    outputs = [get_GaussianMixture_model_predictions_torch(input_tensor[i],
                                       input_tensor_features,
                                       number_of_components,
                                       number_of_iterations=number_of_iterations,
                                       flag_plot_colors_on_tensor=flag_plot_colors_on_tensor,
                                       flag_plot_BB_on_tensor=flag_plot_BB_on_tensor,
                                       device=device) for i in np.arange(input_tensor.shape[0])]

    labels_predictions_list = []
    B,C,H,W = input_tensor.shape
    output_tensor_final = torch.ones((0,3,H,W)).to(input_tensor.device)
    labels_BB_list_of_lists = []
    BB_tensor = []
    for i in np.arange(len(outputs)):
        labels_prediction, output_tensor, labels_BB_list, BB_tensor = outputs[i]
        # print(len(labels_BB_list))
        labels_predictions_list.append(labels_prediction)
        labels_BB_list_of_lists.append(labels_BB_list)
        if flag_plot_colors_on_tensor or flag_plot_BB_on_tensor:
            output_tensor_final = torch.cat([output_tensor_final, output_tensor], 0)

    ### Draw Bounding Boxes On Image Using Kornia Utils: ###
    BB_tensor = list_of_BB_lists_to_torch_tensor(labels_BB_list_of_lists)

    return labels_predictions_list, output_tensor_final, labels_BB_list_of_lists, BB_tensor

def get_GMM_Batch_model_predictions_torch(input_tensor,
                                          input_tensor_features=None,
                                          number_of_components=6,
                                          number_of_iterations=100,
                                          batch_size=100,
                                          flag_plot_colors_on_tensor=True,
                                          flag_plot_BB_on_tensor=True,
                                          device='cpu'):
    ### Initialize GaussianMixture model and fit: ###
    output_tensor = None
    if input_tensor_features is None:
        input_tensor_features = (input_tensor == 1).nonzero()[:, -2:].float()

    if input_tensor_features.shape[0] >= 10:
        GMM_model = GMM_Batch(K=number_of_components, type='full')
        _, labels_prediction = GMM_model.fit(input_tensor_features, batch_size=batch_size, max_iters=number_of_iterations)
        labels_prediction = torch.cat([labels_prediction, torch.tensor([0]).to(labels_prediction.device)], -1)  # TODO: i don't know why the prediction is on N_elements-1

        ### Plot colors on tensor: ###
        if flag_plot_colors_on_tensor:
            output_tensor = plot_GMM_Batch_results_on_tensor(input_tensor, input_tensor_features, labels_prediction, number_of_components)
            # imshow_torch(input_tensor)

        ### Add Bounding Boxes To Predictions: ###
        labels_BB_list = []
        for label_index in np.arange(number_of_components):
            current_label_indices = input_tensor_features[labels_prediction == label_index]
            if current_label_indices.numel() > 0:
                min_H = current_label_indices[:, 0].min().item()
                max_H = current_label_indices[:, 0].max().item()
                min_W = current_label_indices[:, 1].min().item()
                max_W = current_label_indices[:, 1].max().item()
                BB_height = max_H - min_H
                BB_width = max_W - min_W
                labels_BB_list.append([min_W, min_H, max_W, max_H])  # [XYXY]
                # labels_BB_list.append([min_W, min_H, BB_width, BB_height])  #[XYWH]
            else:
                labels_BB_list.append([0, 0, 0, 0])

        ### Plot Bounding Boxesd On Image: ###
        BB_tensor = torch.tensor(labels_BB_list).to(input_tensor.device).unsqueeze(0)
        if flag_plot_BB_on_tensor:
            if output_tensor is None:
                output_tensor = kornia.utils.draw_rectangle(torch_get_ND(input_tensor, number_of_dims=4), BB_tensor, color=torch.tensor([1]))
            else:
                output_tensor = kornia.utils.draw_rectangle(torch_get_ND(output_tensor, number_of_dims=4), BB_tensor, color=torch.tensor([1]))
            # imshow_torch(output_tensor)

    else:
        labels_prediction = torch.zeros(input_tensor_features.shape[0]).to(input_tensor_features.device)
        labels_BB_list = [[0, 0, 0, 0]]
        BB_tensor = torch.tensor(labels_BB_list).unsqueeze(0)
        if flag_plot_BB_on_tensor or flag_plot_colors_on_tensor:
            output_tensor = BW2RGB(torch_get_ND(input_tensor, number_of_dims=4))

    return labels_prediction, output_tensor, labels_BB_list, BB_tensor


def get_GMM_Batch_model_predictions_torch_batch(input_tensor,
                                       input_tensor_features=None,
                                       number_of_components=6,
                                       number_of_iterations=100,
                                       batch_size=128,
                                       flag_plot_colors_on_tensor=True,
                                       flag_plot_BB_on_tensor=True,
                                       device='cpu'):
    outputs = [get_GMM_Batch_model_predictions_torch(input_tensor[i],
                                       input_tensor_features,
                                       number_of_components,
                                       number_of_iterations=number_of_iterations,
                                       batch_size=batch_size,
                                       flag_plot_colors_on_tensor=flag_plot_colors_on_tensor,
                                       flag_plot_BB_on_tensor=flag_plot_BB_on_tensor,
                                       device=device) for i in np.arange(input_tensor.shape[0])]

    labels_predictions_list = []
    B,C,H,W = input_tensor.shape
    output_tensor_final = torch.ones((0,3,H,W)).to(input_tensor.device)
    labels_BB_list_of_lists = []
    BB_tensor = []
    for i in np.arange(len(outputs)):
        labels_prediction, output_tensor, labels_BB_list, BB_tensor = outputs[i]
        # print(len(labels_BB_list))
        labels_predictions_list.append(labels_prediction)
        labels_BB_list_of_lists.append(labels_BB_list)
        if flag_plot_colors_on_tensor or flag_plot_BB_on_tensor:
            output_tensor_final = torch.cat([output_tensor_final, output_tensor], 0)

    ### Draw Bounding Boxes On Image Using Kornia Utils: ###
    BB_tensor = list_of_BB_lists_to_torch_tensor(labels_BB_list_of_lists)

    return labels_predictions_list, output_tensor_final, labels_BB_list_of_lists, BB_tensor


def get_Kmeans_model_predictions_torch(input_tensor,
                                       input_tensor_features=None,
                                       number_of_components=6,
                                       number_of_iterations=100,
                                       batch_size=128,
                                       flag_plot_colors_on_tensor=True,
                                       flag_plot_BB_on_tensor=True,
                                       device='cpu'):
    output_tensor = None
    if input_tensor_features is None:
        input_tensor_features = (input_tensor == 1).nonzero()[:, -2:].float()

    if input_tensor_features.shape[0] >= 10:

        ### Initialize kmeans model and perform fitting: ###
        mean, labels_prediction = kmeans_fun_gpu(input_tensor_features, K=number_of_components, max_iter=number_of_iterations, batch_size=batch_size, tol=1e-40)

        ### Plot colors on the input tensor: ###
        if flag_plot_colors_on_tensor:
            output_tensor = plot_kmeans_fun_gpu_results_on_tensor(input_tensor, input_tensor_features, labels_prediction, number_of_components)
            # imshow_torch(output_tensor)

        ### Add Bounding Boxes To Predictions: ###
        labels_BB_list = []
        for label_index in np.arange(number_of_components):
            current_label_indices = input_tensor_features[labels_prediction == label_index]
            if current_label_indices.numel() > 0:
                min_H = current_label_indices[:, 0].min().item()
                max_H = current_label_indices[:, 0].max().item()
                min_W = current_label_indices[:, 1].min().item()
                max_W = current_label_indices[:, 1].max().item()
                BB_height = max_H - min_H
                BB_width = max_W - min_W
                labels_BB_list.append([min_W, min_H, max_W, max_H])  # [XYXY]
                # labels_BB_list.append([min_W, min_H, BB_width, BB_height])  #[XYWH]
            else:
                labels_BB_list.append([0, 0, 0, 0])

        ### Plot Bounding Boxesd On Image: ###
        BB_tensor = torch.tensor(labels_BB_list).to(input_tensor.device).unsqueeze(0)
        if flag_plot_BB_on_tensor:
            if output_tensor is None:
                output_tensor = kornia.utils.draw_rectangle(torch_get_ND(input_tensor, number_of_dims=4), BB_tensor, color=torch.tensor([1]))
            else:
                output_tensor = kornia.utils.draw_rectangle(torch_get_ND(output_tensor, number_of_dims=4), BB_tensor, color=torch.tensor([1]))
            # imshow_torch(output_tensor)

    else:
        labels_prediction = torch.zeros(input_tensor_features.shape[0]).to(input_tensor_features.device)
        labels_BB_list = [[0,0,0,0]]
        BB_tensor = torch.tensor(labels_BB_list).unsqueeze(0)
        if flag_plot_BB_on_tensor or flag_plot_colors_on_tensor:
            output_tensor = BW2RGB(torch_get_ND(input_tensor, number_of_dims=4))

    return labels_prediction, output_tensor, labels_BB_list, BB_tensor


def get_Kmeans_model_predictions_torch_batch(input_tensor,
                                       input_tensor_features=None,
                                       number_of_components=6,
                                       number_of_iterations=100,
                                       batch_size=128,
                                       flag_plot_colors_on_tensor=True,
                                       flag_plot_BB_on_tensor=True,
                                       device='cpu'):
    outputs = [get_Kmeans_model_predictions_torch(input_tensor[i],
                                       input_tensor_features,
                                       number_of_components,
                                       number_of_iterations=number_of_iterations,
                                       batch_size=batch_size,
                                       flag_plot_colors_on_tensor=flag_plot_colors_on_tensor,
                                       flag_plot_BB_on_tensor=flag_plot_BB_on_tensor,
                                       device=device) for i in np.arange(input_tensor.shape[0])]

    labels_predictions_list = []
    B,C,H,W = input_tensor.shape
    output_tensor_final = torch.ones((0,3,H,W)).to(input_tensor.device)
    labels_BB_list_of_lists = []
    BB_tensor = []
    for i in np.arange(len(outputs)):
        labels_prediction, output_tensor, labels_BB_list, BB_tensor = outputs[i]
        # print(len(labels_BB_list))
        labels_predictions_list.append(labels_prediction)
        labels_BB_list_of_lists.append(labels_BB_list)
        if flag_plot_colors_on_tensor or flag_plot_BB_on_tensor:
            output_tensor_final = torch.cat([output_tensor_final, output_tensor], 0)

    ### Draw Bounding Boxes On Image Using Kornia Utils: ###
    BB_tensor = list_of_BB_lists_to_torch_tensor(labels_BB_list_of_lists)

    return labels_predictions_list, output_tensor_final, labels_BB_list_of_lists, BB_tensor

from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import fast_binning_2D_AvgPool2d, fast_binning_2D_PixelBinning
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch_video, imshow_torch
from RapidBase.Utils.IO.Path_and_Reading_utils import video_torch_array_to_video, video_numpy_array_to_video, video_torch_array_to_images
def detect_blobs_testing():
    ### Read Image: ###
    # img = read_image_default()
    outliers = read_outliers_default()
    input_tensor_numpy = outliers[-1,0].cpu().numpy()
    input_tensor_numpy = RGB2BW(input_tensor_numpy)
    # input_tensor = torch.tensor(input_tensor_numpy).cuda().unsqueeze(0).unsqueeze(0)
    input_tensor = outliers
    k = 1.414
    sigma_size_basis = 1.414
    basic_sigma_factor = 1.0
    number_of_sigmas = 5
    minimum_sigma = 2
    maximum_sigma = 5
    detection_threshold = 0.1
    flag_sigma_grid = 'linear'  #'geometric', 'linear'
    # imshow(img)

    ### Get Rid Of Small without immediate neighbours: ###
    binning_size = 3
    binning_size_tuple = (binning_size, binning_size)
    overlap_size_tuple = (binning_size-1, binning_size-1)
    outliers_binned = fast_binning_2D_AvgPool2d(outliers, binning_size_tuple=binning_size_tuple, overlap_size_tuple=overlap_size_tuple)
    outliers[outliers_binned <= 0.25] = 0
    # imshow_torch(outliers[-1])

    ### Use Blob Detection Layer: ###
    blob_detection_layer = Blob_Detection_Layer(number_of_sigmas=5,
                                                flag_sigma_grid='linear',
                                                minimum_sigma=2,
                                                maximum_sigma=5,
                                                kernel_size_to_sigma_factor=1.3,
                                                sigma_size_basis=1.414,
                                                detection_threshold=0.1,
                                                iou_threshold_modified=0.02,
                                                maxpool_spatial_kernel_size=3,
                                                bounding_box_frame_addition_for_robust_overlap=5)
    BB_tensor, output_tensor = blob_detection_layer.forward(input_tensor, True)
    # imshow_torch(output_tensor[50])
    # imshow_torch_video(output_tensor, FPS=10)
    # video_torch_array_to_video(output_tensor, video_name=r'C:\Users\dudyk\Downloads/Blobs.avi', FPS=5)

    ### TODO: the problem of using Kmeans and GMM in batch form is because it accepts [N_features, N_dimensions].
    ### TODO: the problem is that every frame in the batch/sequence can have a different number of outliers...and so even if the algorithm can be parallelized it's a problem.
    ### TODO: one way around this is to get the frame with the most amount of outliers and fill/pad all the rest of the frames to the same thing. that would, of course, use more computation.
    ### TODO: another way is simply by list comprehension, but then you're not taking advantage of pytorch parallel computing.

    # ### Try GMM Using GaussianMixture Function: ###
    # frame_index = 90
    # input_tensor_features = (input_tensor[frame_index:frame_index+1] == 1).nonzero()[:,-2:]
    # number_of_components = 6
    # labels_prediction, output_tensor, labels_BB_list, BB_tensor = get_GaussianMixture_model_predictions_torch(input_tensor[frame_index],
    #                                                                                                           input_tensor_features,
    #                                                                                                           number_of_components,
    #                                                                                                           number_of_iterations=100,
    #                                                                                                           flag_plot_colors_on_tensor=True,
    #                                                                                                           flag_plot_BB_on_tensor=True,
    #                                                                                                           device=input_tensor.device)
    # # imshow_torch(output_tensor)
    # # imshow_torch(input_tensor[frame_index])
    # labels_predictions_list, output_tensor_final, labels_BB_list_of_lists, BB_tensor = \
    #     get_GaussianMixture_model_predictions_torch_batch(input_tensor,
    #                                                 input_tensor_features=None,
    #                                                 number_of_components=6,
    #                                                 number_of_iterations=100,
    #                                                 flag_plot_colors_on_tensor=True,
    #                                                 flag_plot_BB_on_tensor=True,
    #                                                 device=input_tensor.device)
    # imshow_torch(output_tensor_final[50])


    # ### Try GMM Using GMM_Batch Function: ###
    # frame_index = 90
    # input_tensor_features = (input_tensor[frame_index:frame_index + 1] == 1).nonzero()[:, -2:].float()
    # number_of_components = 7
    # labels_prediction, output_tensor, labels_BB_list, BB_tensor = get_GMM_Batch_model_predictions_torch(input_tensor[frame_index],
    #                                                                                                     input_tensor_features,
    #                                                                                                     number_of_components,
    #                                                                                                     number_of_iterations=100,
    #                                                                                                     batch_size=128,
    #                                                                                                     flag_plot_colors_on_tensor=True,
    #                                                                                                     flag_plot_BB_on_tensor=True,
    #                                                                                                     device=input_tensor.device)
    # # imshow_torch(output_tensor)
    # # imshow_torch(input_tensor[frame_index])
    # labels_predictions_list, output_tensor_final, labels_BB_list_of_lists, BB_tensor = \
    #     get_GMM_Batch_model_predictions_torch_batch(input_tensor,
    #                                              input_tensor_features=None,
    #                                              number_of_components=6,
    #                                              number_of_iterations=100,
    #                                              batch_size=128,
    #                                              flag_plot_colors_on_tensor=True,
    #                                              flag_plot_BB_on_tensor=True,
    #                                              device=input_tensor.device)
    # imshow_torch(output_tensor_final[-1])


    # ### Try KMeans: ###
    # frame_index = 90
    # input_tensor_features = (input_tensor[frame_index:frame_index + 1] == 1).nonzero()[:, -2:].float()
    # number_of_components = 6
    # labels_prediction, output_tensor, labels_BB_list, BB_tensor = get_Kmeans_model_predictions_torch(input_tensor[frame_index],
    #                                                                                                  input_tensor_features=None,
    #                                                                                                  number_of_components=6,
    #                                                                                                  number_of_iterations=100,
    #                                                                                                  batch_size=128,
    #                                                                                                  flag_plot_colors_on_tensor=True,
    #                                                                                                  flag_plot_BB_on_tensor=True,
    #                                                                                                  device=input_tensor.device)
    #
    # labels_predictions_list, output_tensor_final, labels_BB_list_of_lists, BB_tensor =\
    #     get_Kmeans_model_predictions_torch_batch(input_tensor,
    #                                              input_tensor_features=None,
    #                                              number_of_components=6,
    #                                              number_of_iterations=100,
    #                                              batch_size=128,
    #                                              flag_plot_colors_on_tensor=True,
    #                                              flag_plot_BB_on_tensor=True,
    #                                              device=input_tensor.device)
    # imshow_torch(output_tensor_final[-1])
    # # imshow_torch(input_tensor[frame_index])




# detect_blobs_testing()

#################################################################################################################################################################################################################################


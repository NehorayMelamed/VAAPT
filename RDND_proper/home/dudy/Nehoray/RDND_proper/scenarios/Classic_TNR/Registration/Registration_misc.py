from RapidBase.import_all import *

import numpy as np
import torch
import torch.nn.functional as F
# loading images
from PIL import Image
from os import listdir
# # import image_registration


# import the necessary packages
import numpy as np
# import imutils
import cv2
import torch.fft
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

### Functions for the old TNR modules and layers and from scripts trying to compare algorithms for gimbaless: ### #TODO: to be deleted probably
#TODO: add Maor's algorithms (gimbaless) to pytorch!!!!
#TODO: add finding homomorphic matrices from optical flow with pytorch/numpy
#TODO: make sure i have the OpenCV feature->homomorphic transform shit down!
### use ORB to detect features and match features: ###

# import imutils

### TODO: add wrapper to allow registration of several frames. for most algorithms it will mean iteratively registering image pairs, for MAOR it means else

### use ORB to detect features and match features: ###
def align_images_OpenCVFeatures(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x,y-coordinates) from the
    # top matches -- we'll use these coordinates to compute our homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H_matrix, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # H[-1,0] = 0
    # H[-1,1] = 0
    # print(H[-1,-1])
    # bla_scale = H[-1,-1]
    # H[0:2,0:2] = H[0:2,0:2]/bla_scale
    # H[-1,-1] = 1

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H_matrix, (w, h))
    # aligned = cv2.warpAffine(image, H[0:2,:], (w, h))

    # return the aligned image
    return aligned, H_matrix



def align_images_ECC(img1, img2):

    # Find the width and height of the color image
    height = img1.shape[0]
    width = img1.shape[1]

    # Allocate space for aligned image
    im_aligned = np.zeros((height, width, 1), dtype=np.uint8)

    # Define motion model
    warp_mode = cv2.MOTION_AFFINE  #TODO: what other types are there
    # warp_mode = cv2.MOTION_HOMOGRAPHY  #TODO: what other types are there

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-10)

    # Warp the blue and green channels to the red channel
    (cc, warp_matrix) = cv2.findTransformECC(np.mean(get_gradient(img1), -1),
                                             np.mean(get_gradient(img2), -1),
                                             warp_matrix,
                                             warp_mode,
                                             criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use Perspective warp when the transformation is a Homography
        im_aligned = cv2.warpPerspective(img2,
                                        warp_matrix,
                                        (width, height),
                                        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    else:
        # Use Affine warp when the transformation is not a Homography
        im_aligned = cv2.warpAffine(img2,
                                    warp_matrix,
                                    (width, height),
                                    flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

    return im_aligned, warp_matrix


def register_images(img1, img2,
                    registration_algorithm,
                    search_area=5,
                    inner_crop_size_to_use=1000,
                    downsample_factor=1, #binning / average_pooling
                    flag_do_initial_SAD=True, initial_SAD_search_area=5):
    #TODO: get rid of outer frame before doing SAD
    #TODO: enable batch processing
    if flag_do_initial_SAD:
        shifts_array_init, img2 = register_image_SAD(img1, img2, search_area=initial_SAD_search_area)
    else:
        shifts_array_init = [0,0]

    if registration_algorithm == 'MAOR':
        shifts_array, clean_frame_previous_warped = register_images_MAOR(img1, img2, inner_crop_size_to_use)
        flow_map = None
    elif registration_algorithm == 'CrossCorrelation':
        shifts_array, z_max, clean_frame_previous_warped = register_images_CrossCorrelation(img1, img2, search_area, inner_crop_size_to_use)
        flow_map = None
    elif registration_algorithm == 'NormalizedCrossCorrelation':
        shifts_array, z_max, clean_frame_previous_warped = register_images_NCC(img1, img2, search_area, inner_crop_size_to_use, downsample_factor)
        flow_map = None
    elif registration_algorithm == 'PhaseCorrelation':
        shifts_array, z_max, clean_frame_previous_warped = register_images_PhaseCorrelation(img1, img2, search_area, inner_crop_size_to_use)
        flow_map = None
    elif registration_algorithm == 'FeatureHomography':
        shifts_array, clean_frame_previous_warped = register_images_FeatureHomography(img1, img2)
        flow_map = None
    elif registration_algorithm == 'DeepFlow':
        shifts_array, clean_frame_previous_warped, flow_map = register_images_OpticalFlow(img1, img2, 'DeepFlow')  # DeepFlow, SimpleFlow, and TVL1

    shift_x = shifts_array[0] + shifts_array_init[0]
    shift_y = shifts_array[1] + shifts_array_init[1]
    shifts_array = (shift_x, shift_y)
    return shifts_array, clean_frame_previous_warped, flow_map


def register_images_batch(images_batch,
                    registration_algorithm,
                    search_area=5,
                    inner_crop_size_to_use=1000,
                    downsample_factor=1, #binning / average_pooling
                    flag_do_initial_SAD=True, initial_SAD_search_area=5, center_frame_index=None):
    ### Initial handling of indices according to input type: ###
    if type(images_batch) == list:
        if center_frame_index is None:
            center_frame_index = len(images_batch) // 2
        number_of_images = len(images_batch)
    if type(images_batch) == torch.Tensor:
        #if [T,C,H,W] i assume T is frames index, if [B,T,C,H,W]
        if len(images_batch.shape) == 4:
            frames_dim = 0
            number_of_images = images_batch.shape[0]
            if center_frame_index is None:
                center_frame_index = number_of_images//2
            center_frame = images_batch[center_frame_index:center_frame_index+1, :, :, :]
        elif len(images_batch.shape) == 5:
            frames_dim = 1
            number_of_images = images_batch.shape[1]
            if center_frame_index is None:
                center_frame_index = number_of_images//2
            center_frame = images_batch[:,center_frame_index:center_frame_index+1,:,:,:]

    ### Loop over images and register each of them in reference to reference_frame: ###
    #TODO: this is compatible with algorithms which work on two images at a time, some algorithms, like maor, act on all algorithms together.
    warped_frames_list = []
    shifts_array_list = []
    for frame_index in np.arange(number_of_images):
        ### Register Individual Images Pairs
        if frame_index != center_frame_index:
            ### Get current frame: ###
            if len(center_frame.shape) == 4:
                current_frame = images_batch[frame_index:frame_index+1,:,:,:]
            elif len(center_frame.shape) == 5:
                current_frame = images_batch[:,frame_index:frame_index+1,:,:,:]

            ### Register current frame to reference frame: ###
            shifts_array, current_frame_warped = register_images(center_frame.data, current_frame.data,
                                                                registration_algorithm,
                                                                search_area=search_area,
                                                                inner_crop_size_to_use=inner_crop_size_to_use,
                                                                downsample_factor=downsample_factor,  # binning / average_pooling
                                                                flag_do_initial_SAD=flag_do_initial_SAD, initial_SAD_search_area=initial_SAD_search_area)
        else:
            current_frame = center_frame
            current_frame_warped = center_frame
            shifts_array = (0,0)

        ### Append To Lists: ###
        warped_frames_list.append(current_frame_warped)
        shifts_array_list.append(shifts_array)

    return shifts_array_list, warped_frames_list

### Discrete min-SAD: ###
def calculate_SAD_torch(tensor1, tensor2, search_area=5):
    ### Trims: ###
    B, C, H, W = tensor2.shape
    SAD = torch.zeros((B, C, search_area, search_area)).to(tensor1.device)
    trim = int(floor(search_area / 2))
    RimU = trim
    RimD = trim
    RimL = trim
    RimR = trim

    ### A,B Length: ###
    BLy = H - RimU - RimD
    BLx = W - RimL - RimR
    ALy = BLy
    ALx = BLx

    ### Displacement: ###
    B_upper_left = [RimU, RimL]  # the location of the upper-left corner of the Broi matrix
    DispUD = arange(-RimD, RimU + 1)
    DispLR = arange(-RimL, RimR + 1)

    ### B-ROI: ###
    Broi = tensor2[:, :, RimU:H - RimD, RimL:W - RimR]
    Broibar = Broi.mean(2, True).mean(3, True)
    Broiup = (Broi - Broibar)
    Broidown = ((Broi - Broibar) ** 2).sum(2, True).sum(3, True)

    ### Get SAD: ###
    for iin in arange(len(DispUD)):
        for jjn in arange(len(DispLR)):
            shift_y = DispUD[iin]
            shift_x = DispLR[jjn]
            A_upper_left = [B_upper_left[0] + shift_y, B_upper_left[1] + shift_x]
            Atmp = tensor1[:, :, A_upper_left[0]:A_upper_left[0] + ALy, A_upper_left[1]:A_upper_left[1] + ALx]
            Abar = Atmp.mean(2, True).mean(3, True)
            Aup = (Atmp - Abar)
            Adown = ((Atmp - Abar) ** 2).sum(2, True).sum(3, True)
            # SAD[:, :, iin, jjn] = (Broiup - Aup).abs().sum(2, False).sum(2, False)
            SAD[:, :, iin, jjn] = (Broi - Atmp).abs().sum(2, False).sum(2, False)  #SAD, maybe this can be better refined but still


    return SAD

### Min SAD: ###
def register_image_SAD(img1, img2, search_area=5):
    SAD = calculate_SAD_torch(img1,img2,search_area)
    max_index = np.argmin(SAD)
    max_row = max_index // search_area
    max_col = max_index % search_area
    shift_x = max_col - (search_area//2)
    shift_y = max_row - (search_area//2)
    shifts_array = (shift_x,shift_y)
    img2_warped = shift_matrix_subpixel_torch(img2, shift_x, shift_y)
    return shifts_array, img2_warped

### Normalized Cross Correlation: ###
def register_images_NCC(input_image1, input_image2, search_area=5, inner_crop_size_to_use=None, downsample_factor=1):
    ### Get Only Valid Inner Crop Of The Images If Wanted: ###
    if inner_crop_size_to_use is None:
        img1 = input_image1
        img2 = input_image2
    else:
        img1 = crop_torch_batch(input_image1, inner_crop_size_to_use, 'center')
        img2 = crop_torch_batch(input_image2, inner_crop_size_to_use, 'center')

    ### DownSample if wanted: ###
    if downsample_factor > 1:
        downsample_layer = nn.AvgPool2d(downsample_factor)
        img1 = downsample_layer(img1)
        img2 = downsample_layer(img2)

    ### Calculate NCC: ###
    CC = torch_Cross_Correlation(img1, img2, search_area)
    shifts_array, z_max = return_shifts_using_parabola_fit_torch(CC)
    shift_x = shifts_array[0] * downsample_factor
    shift_y = shifts_array[1] * downsample_factor
    shift_x = -shift_x #(!!!). THIS is for consistency purposes. this function finds A shift next to B, i usually want B shift from A
    shift_y = -shift_y #(!!!). THIS is for consistency purposes. this function finds A shift next to B, i usually want B shift from A
    shifts_array = (shift_x, shift_y)

    ### If shift values or z-value doesn't make sense -> limit shift or disregard it: ###
    if abs(shift_x) > search_area:
        shift_x = 0
    if abs(shift_y) > search_area:
        shift_y = 0

    img2_warped = shift_matrix_subpixel_torch(input_image2, -shift_x, -shift_y)
    return shifts_array, z_max, img2_warped

### MAOR ALGORITTHM!: ###
def register_images_MAOR(img1, img2, inner_crop_size_to_use):
    # frame1 = np.expand_dims(img1, 0)
    # frame2 = np.expand_dims(img2, 0)
    # images_stack = np.concatenate((frame1, frame2), 0)

    frame1 = img1.squeeze(0) #receives [1,1,H,W] as input
    frame2 = img2.squeeze(0)
    images_stack = torch.cat((frame1, frame2),0)  #receives [2,H,W] as input
    shifts_vec, aligned_stack = align(images_stack, inner_crop_size_to_use)
    # aligned_stack = aligned_stack.numpy()
    warped_image = aligned_stack[-1,:,:]

    return shifts_vec, warped_image


### Using Cross Correlation: ###
# def register_images_CrossCorrelation(img1, img2):
#     if type(img1) == torch.Tensor:
#         img1 = img1.numpy().squeeze(0).squeeze(0)
#     if type(img2) == torch.Tensor:
#         img2 = img2.numpy().squeeze(0).squeeze(0)
#
#     ### Register Images: ###
#     deltay, deltax = image_registration.cross_correlation_shifts(img1, img2)
#
#     ### Shift Clean Image: ###
#     img2_warped = shift_matrix_subpixel(np.expand_dims(img2,-1), deltax, deltay).squeeze()
#
#     return (deltay, deltax), img2_warped.squeeze()


def register_images_CrossCorrelation(input_image1, input_image2, search_area, inner_crop_size_to_use=None):
    #TODO: turn to pytorch instead of numpy
    ### Get Only Valid Inner Crop Of The Images If Wanted: ###
    if inner_crop_size_to_use is None:
        img1 = input_image1
        img2 = input_image2
    else:
        img1 = crop_torch_batch(input_image1, inner_crop_size_to_use, 'center')
        img2 = crop_torch_batch(input_image2, inner_crop_size_to_use, 'center')
    if type(img1) == torch.Tensor:
        img1 = img1.numpy().squeeze(0).squeeze(0)
    if type(img2) == torch.Tensor:
        img2 = img2.numpy().squeeze(0).squeeze(0)

    ### Get Shape: ###
    H, W = img2.shape

    ### Get FFTs: ###
    f1 = cv2.dft(img1.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f2 = cv2.dft(img2.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f1_shf = np.fft.fftshift(f1)
    f2_shf = np.fft.fftshift(f2)

    ### Get Complex numbers outputs instead of 2 output channels: ###
    f1_shf_cplx = f1_shf[:, :, 0] + 1j * f1_shf[:, :, 1]
    f2_shf_cplx = f2_shf[:, :, 0] + 1j * f2_shf[:, :, 1]

    ### Get absolute values needed for calculations: ###
    f1_shf_abs = np.abs(f1_shf_cplx)
    f2_shf_abs = np.abs(f2_shf_cplx)
    total_abs = f1_shf_abs * f2_shf_abs
    P_real = (np.real(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.imag(f1_shf_cplx) * np.imag(f2_shf_cplx))
    P_imag = (np.imag(f1_shf_cplx) * np.real(f2_shf_cplx) -
              np.real(f1_shf_cplx) * np.imag(f2_shf_cplx))
    P_complex = P_real + 1j * P_imag

    ### Get inverse FFT of normalized multiplication of FFTs (phase only part of FFT cross correlation calculation): ###
    P_inverse = np.abs(np.fft.ifft2(P_complex))  # inverse FFT
    P_inverse = np.fft.fftshift(P_inverse)

    ### Get Normalized Cross Correlation (max possible value = 1): ###
    A_sum = img2.sum()
    A_sum2 = (img2**2).sum()
    sigmaA = (A_sum2 - A_sum**2 / (H*W)) ** (1/2)
    sigmaB = (img1).std() * (H*W-1)**(1/2)
    B_mean = (img1).mean()
    P_inverse = (P_inverse - A_sum*B_mean)/(sigmaA*sigmaB)

    # ### Get CC Max: ###
    # max_index = np.argmax(P_inverse)
    # max_row = max_index // W
    # max_col = max_index % W
    # max_value = P_inverse[max_row, max_col]
    # center_row = H // 2
    # center_col = W // 2

    ### Search CC Max Around Center: ###
    center_row = H // 2
    center_col = W // 2
    max_value = -np.inf
    for index_row in np.arange(center_row-search_area,center_row+search_area):
        for index_col in np.arange(center_col-search_area,center_col+search_area):
            current_value = P_inverse[index_row, index_col]
            if current_value > max_value:
                max_value = current_value
                max_row = index_row
                max_col = index_col

    ### Assign CC Values around max: ###
    CC = zeros((search_area, search_area))
    for index_col in np.arange(0, search_area):
        for index_row in np.arange(0, search_area):
            CC[index_row, index_col] = P_inverse[max_row - search_area // 2 + index_row,
                                                 max_col - search_area // 2 + index_col]

    ### Use parabola fit to interpolate sub-pixel shift: ###
    shifts_total, z_max_vec = return_shifts_using_parabola_fit_numpy(CC)
    shifts_total = [shifts_total[0], shifts_total[1]]
    shifts_total[1] -= (center_row - max_row)
    shifts_total[0] -= (center_col - max_col)
    shift_x = shifts_total[0]
    shift_y = shifts_total[1]

    ### If shift values or z-value doesn't make sense -> limit shift or disregard it: ###
    if abs(shift_x) > search_area:
        shift_x = 0
    if abs(shift_x) > search_area:
        shift_y = 0
    
    ### Shift Clean Image: ###
    img2_warped = shift_matrix_subpixel(np.expand_dims(input_image2.cpu().numpy().squeeze(0).squeeze(0), -1), -shift_x, -shift_y).squeeze()

    return (shift_x, shift_y), z_max_vec, img2_warped



### Using Phase Only Correlation: ###
def register_images_PhaseCorrelation(input_image1, input_image2, search_area, inner_crop_size_to_use=None):
    ### Get Only Valid Inner Crop Of The Images If Wanted: ###
    if inner_crop_size_to_use is None:
        img1 = input_image1
        img2 = input_image2
    else:
        img1 = crop_torch_batch(input_image1, inner_crop_size_to_use, 'center')
        img2 = crop_torch_batch(input_image2, inner_crop_size_to_use, 'center')

    if type(img1) == torch.Tensor:
        img1 = img1.numpy().squeeze(0).squeeze(0)
    if type(img2) == torch.Tensor:
        img2 = img2.numpy().squeeze(0).squeeze(0)

    ### Get Shape: ###
    H, W = img2.shape

    ### Get FFTs: ###
    f1 = cv2.dft(img1.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f2 = cv2.dft(img2.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f1_shf = np.fft.fftshift(f1)
    f2_shf = np.fft.fftshift(f2)

    ### Get Complex numbers outputs instead of 2 output channels: ###
    f1_shf_cplx = f1_shf[:, :, 0] + 1j * f1_shf[:, :, 1]
    f2_shf_cplx = f2_shf[:, :, 0] + 1j * f2_shf[:, :, 1]

    ### Get absolute values needed for calculations: ###
    f1_shf_abs = np.abs(f1_shf_cplx)
    f2_shf_abs = np.abs(f2_shf_cplx)
    total_abs = f1_shf_abs * f2_shf_abs
    P_real = (np.real(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.imag(f1_shf_cplx) * np.imag(f2_shf_cplx)) / (total_abs + 1e-3)
    P_imag = (np.imag(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.real(f1_shf_cplx) * np.imag(f2_shf_cplx)) / (total_abs + 1e-3)
    P_complex = P_real + 1j * P_imag

    ### Get inverse FFT of normalized multiplication of FFTs (phase only part of FFT cross correlation calculation): ###
    P_inverse = np.abs(np.fft.ifft2(P_complex))  # inverse FFT
    P_inverse = np.fft.fftshift(P_inverse)

    ### Get Normalized Cross Correlation (max possible value = 1): ###
    A_sum = img2.sum()
    A_sum2 = (img2 ** 2).sum()
    sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
    sigmaB = (img1).std() * (H * W - 1) ** (1 / 2)
    B_mean = (img1).mean()
    P_inverse = (P_inverse - A_sum * B_mean) / (sigmaA * sigmaB)

    # ### Get CC Max: ###
    # max_index = np.argmax(P_inverse)  #TODO: instead of using expensive argmax simply search small area around zero shift within search_area
    # max_row = max_index // W
    # max_col = max_index % W
    # max_value = P_inverse[max_row, max_col]
    # center_row = H // 2
    # center_col = W // 2

    ### Search CC Max Around Center: ###
    center_row = H // 2
    center_col = W // 2
    max_value = -np.inf
    for index_row in np.arange(center_row - search_area, center_row + search_area):
        for index_col in np.arange(center_col - search_area, center_col + search_area):
            current_value = P_inverse[index_row, index_col]
            if current_value > max_value:
                max_value = current_value
                max_row = index_row
                max_col = index_col

    ### Assign CC Values around max: ###
    CC = zeros((3, 3))
    for index_col in np.arange(0, 3):
        for index_row in np.arange(0, 3):
            CC[index_row, index_col] = P_inverse[max_row - search_area//2 + index_row, max_col - search_area//2 + index_col]

    ### Use parabola fit to interpolate sub-pixel shift: ###
    shifts_total, z_max_vec = return_shifts_using_parabola_fit_numpy(CC)
    shifts_total = [shifts_total[0], shifts_total[1]]
    shifts_total[0] -= (center_row - max_row)
    shifts_total[1] -= (center_col - max_col)
    shift_x = shifts_total[1]
    shift_y = shifts_total[0]

    # max_id = [0, 0]
    # max_val = 0
    # for idy in np.arange(search_area):
    #     for idx in np.arange(search_area):
    #         if P_inverse[idy, idx] > max_val:
    #             max_val = P_inverse[idy, idx]
    #             max_id = [idy, idx]
    # shift_x = search_area - max_id[0]
    # shift_y = search_area - max_id[1]
    # print(shift_x, shift_y)

    ### If shift values or z-value doesn't make sense -> limit shift or disregard it: ###
    if abs(shift_x) > search_area:
        shift_x = 0
    if abs(shift_x) > search_area:
        shift_y = 0
        
    ### Shift Clean Image: ###
    img2_warped = shift_matrix_subpixel_torch(input_image2, shift_x, shift_y)

    return (-shift_x, -shift_y), z_max_vec, img2_warped

### Using Feature Detection and Homography matrix: ###
def register_images_FeatureHomography(img1, img2):
    img1 = img1.squeeze(0).squeeze(0)
    img2 = img2.squeeze(0).squeeze(0)
    img1 = uint8((img1*255).clamp(0,255))
    img2 = uint8((img2*255).clamp(0,255))
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img2, homography, (width, height))
    transformed_img = transformed_img/255

    shift_x = homography[0,2]
    shift_y = homography[1,2]
    shifts_vec = (-shift_x,-shift_y)

    return shifts_vec, transformed_img


### Using Optical Flow: ###
def register_images_OpticalFlow(img_source, img_to_align, mc_alg='DeepFlow'):
    # Applies to img_to_align a transformation which converts it into img_source.
    # Args:
    # 	img_to_align: HxWxC image
    # 	img_source: HxWxC image
    # 	mc_alg: selects between DeepFlow, SimpleFlow, and TVL1. DeepFlow runs by default.
    # Returns:
    # 	HxWxC aligned image

    img_to_align_numpy = torch.zeros_like(img_to_align)
    img_source_numpy = torch.zeros_like(img_source)
    img_to_align_numpy[:] = img_to_align[:]
    img_source_numpy[:] = img_source[:]

    ### Tensor To Numpy: ###
    if type(img_to_align_numpy) == torch.Tensor:
        img_to_align_numpy = img_to_align_numpy.cpu().numpy().squeeze(0).squeeze(0)
    if type(img_source_numpy) == torch.Tensor:
        img_source_numpy = img_source_numpy.cpu().numpy().squeeze(0).squeeze(0)

    ### Some of the algorithms demand uint8, do it: ###
    # img_source_copy = (img_source_copy * 255).clip(0, 255)
    # img_to_align_copy = (img_to_align_copy * 255).clip(0, 255)
    # img_source_copy = img_source_copy.astype(np.uint8)
    # img_to_align_copy = img_to_align_copy.astype(np.uint8)

    img0 = img_to_align_numpy[:, :]
    img1 = img_source_numpy[:, :]
    out_img = None

    # Align frames according to selection in mc_alg
    flow = estimate_invflow(img0, img1, mc_alg)

    ### Get Global Shift: ###
    shift_x = flow[0].mean()
    shift_y = flow[1].mean()
    shifts_vec = (shift_x, shift_y)

    # rectifier
    out_img = warp_flow(img_to_align_numpy, flow)

    return shifts_vec, out_img, flow



def estimate_AffineParamters_FromOpticalFlow(frame1, frame2, optical_flow):
    1 #TODO: for shifts can simply use averages. for more complex homographies more elaborate fitting strategies are needed


def estimate_invflow(img0, img1, me_algo):
    # Estimates inverse optical flow by using the me_algo algorithm.
    # # # img0, img1 have to be uint8 grayscale
    # assert img0.dtype == 'uint8' and img1.dtype == 'uint8'

    # Create estimator object
    if me_algo == "DeepFlow":
        of_estim = cv2.optflow.createOptFlow_DeepFlow()
    elif me_algo == "SimpleFlow":
        of_estim = cv2.optflow.createOptFlow_SimpleFlow()
    elif me_algo == "TVL1":
        of_estim = cv2.DualTVL1OpticalFlow_create()
    else:
        raise Exception("Incorrect motion estimation algorithm")

    # Run flow estimation (inverse flow)
    flow = of_estim.calc(img1, img0, None)
    #	flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def warp_flow(img, flow):
    # Applies to img the transformation described by flow.
    assert len(flow.shape) == 3 and flow.shape[-1] == 2

    hf, wf = flow.shape[:2]
    # flow 		= -flow
    flow[:, :, 0] += np.arange(wf)
    flow[:, :, 1] += np.arange(hf)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res




########################################################################################################################
### Old Maor Stuff - TODO: to be deleted!!!!: ###
import cv2
def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad
########################################################################################################################



########################################################################################################################
y_filter = torch.Tensor([[1 / 2, 0, -1 / 2]])
x_filter = y_filter.transpose(0, 1)
y_spectral_filter = torch.Tensor([[0.0116850998497429921230139626686650444753468036651611328125,
                                   -0.0279730819380002923568717676516826031729578971862792968750,
                                   0.2239007887600356350166208585505955852568149566650390625000,
                                   0.5847743866564433234955799889576155692338943481445312500000,
                                   0.2239007887600356350166208585505955852568149566650390625000,
                                   -0.0279730819380002923568717676516826031729578971862792968750,
                                   0.0116850998497429921230139626686650444753468036651611328125]])
x_spectral_filter = y_spectral_filter.transpose(0, 1)

def complex_mul(A, B):
    """
    out = A*B for complex torch.tensors
    A - [a, b, 2] if vector a or b should be 1
    B - [b, c, 2]
    out - [a, c, 2]
    """
    A_real, A_imag = A[..., 0], A[..., 1]
    B_real, B_imag = B[..., 0], B[..., 1]
    return torch.stack([A_real @ B_real - A_imag @ B_imag,
                        A_real @ B_imag + A_imag @ B_real],
                       dim=-1)


def complex_vector_brodcast(A, B):
    """
    out = A*B for complex torch.tensors
    A - [1, b, 2]
    B - [a, b, 2]
    out - [a, b, 2]
    """
    A_real, A_imag = A[..., 0], A[..., 1]
    B_real, B_imag = B[..., 0], B[..., 1]
    return torch.stack([A_real * B_real - A_imag * B_imag,
                        A_real * B_imag + A_imag * B_real],
                       dim=-1)


def conv2_torch(img, filt):
    """
    filter image using pytorch
    img - input image [H, W]
    filter - Tensor [f_H, f_W]
    TODO: check the padding
    """
    H, W = img.shape
    f_H, f_W = filt.shape
    return F.conv2d(img.expand(1, 1, H, W),
                    filt.expand(1, 1, f_H, f_W),
                    padding=(int((f_H - 1) // 2), int((f_W - 1) // 2))).squeeze()


def dxdy(ref, moving, Dx=None, Dy=None, A=None):
    """
    ref - Tensor[H,W]
    moving - Tensor[H,W]
    Dx - ref x derivative Tensor[H,W]
    Dy - ref y derivative Tensor[H,W]
    """
    N = int(np.floor(len(x_filter) // 2))
    if Dx == None:
        Dx = conv2_torch(ref, x_filter)[N:-N, N:-N]
    if Dy == None:
        Dy = conv2_torch(ref, y_filter)[N:-N, N:-N]
    if A == None:
        A = torch.Tensor([[torch.sum(Dx * Dx), torch.sum(Dx * Dy)],
                          [torch.sum(Dy * Dx), torch.sum(Dy * Dy)]])

    diff_frame_dx = conv2_torch((moving - ref), x_spectral_filter)[N:-N, N:-N]
    diff_frame_dy = conv2_torch((moving - ref), y_spectral_filter)[N:-N, N:-N]
    b = torch.Tensor([[torch.sum(Dx * diff_frame_dx)],
                      [torch.sum(Dy * diff_frame_dy)]])
    return torch.solve(b, A)[0]  # return the result only


def shift_image(img, dx, dy):
    """
    img - Tensor[H,W]
    dx - float
    dy - float
    """
    N, M = img.size()
    # fft needs the last dim to be 2 (real,complex) TODO: faster implementation
    img_padded = torch.stack((img, torch.zeros(N, M)), dim=2)
    fft_img = torch.fft(img_padded, 2)
    tmp = np.exp(-1.j * 2 * np.pi * np.fft.fftfreq(N) * dx)  # TODO: remove np vector
    X = torch.from_numpy(tmp.view("(2,)float")).float()
    tmp = np.exp(-1.j * 2 * np.pi * np.fft.fftfreq(M) * dy)
    Y = torch.from_numpy(tmp.view("(2,)float")).float()
    # clac the shifted image
    tmp = complex_vector_brodcast(fft_img, X.unsqueeze(1))
    tmp = complex_vector_brodcast(Y.unsqueeze(0), tmp)
    return torch.ifft(tmp, 2).norm(dim=2)


def shift_image_torch(img, dx, dy):
    """
    img - Tensor[H,W]
    dx - float
    dy - float
    """
    N, M = img.size()
    # fft needs the last dim to be 2 (real,complex) TODO: faster implementation
    img_padded = torch.stack((img, torch.zeros(N, M)), dim=2)  #make into a pseudo complex number with the imaginary part being zero
    fft_img = torch.fft.fftn(img_padded, dim=2) #FFT the image
    tmp = np.exp(-1.j * 2 * np.pi * np.fft.fftfreq(N) * dx)  #Get complex exponential   # TODO: remove np vector
    X = torch.from_numpy(tmp.view("(2,)float")).float()   #Make tmp into a pseudo complex number with two channels (real,img)
    tmp = np.exp(-1.j * 2 * np.pi * np.fft.fftfreq(M) * dy) #same as the above for X
    Y = torch.from_numpy(tmp.view("(2,)float")).float() #save as the above for X
    # clac the shifted image
    tmp = complex_vector_brodcast(fft_img, X.unsqueeze(1))
    tmp = complex_vector_brodcast(Y.unsqueeze(0), tmp)
    # return torch.fft.ifft(tmp, dim=2).norm(dim=2)
    return torch.fft.ifftn(tmp, dim=2)[:,:,0].real  #TOOO: this doesn't work with the new pytorch


def align(stack, inner_crop_size_to_use):
    """
    stack - [batch, width, height]
    output - align stack
    """
    stack_warped = torch.zeros_like(stack)
    stack_warped[0] = stack[0]
    ref = crop_torch_batch(stack[0],inner_crop_size_to_use,'center')  # set the first frame as refernce
    # clac derivative and A matrix
    N = len(x_filter) // 2
    Dx = conv2_torch(ref, x_filter)
    Dx = conv2_torch(Dx, x_spectral_filter)[N:-N, N:-N]  # TODO: could be merged
    Dy = conv2_torch(ref, y_filter)
    Dy = conv2_torch(Dy, y_spectral_filter)[N:-N, N:-N]  # TODO: could be merged
    A = torch.Tensor([[torch.sum(Dx * Dx), torch.sum(Dx * Dy)],
                      [torch.sum(Dy * Dx), torch.sum(Dy * Dy)]])

    for i in range(1, len(stack)):
        # dx,dy = dxdy(ref,stack[i],Dx,Dy,A).numpy()
        # dx, dy = dxdy(ref, stack[i], None, None, None).numpy()
        dy, dx = dxdy(ref, crop_torch_batch(stack[i],inner_crop_size_to_use,'center'), None, None, None).numpy()
        # print('infered shifts: ' + str(dx) + ', ' + str(dy))
        # stack_warped[i] = shift_image(stack[i], -dx, -dy)
        stack_warped[i] = shift_image_torch(stack[i], -dx, -dy)

    shifts_vec = [dx,dy]
    return shifts_vec, stack_warped


def merge(stack):
    """
    stack - align stack of images [batch, height, width]
    output - single image [height ,width]
    """
    return torch.mean(stack, dim=0)

from PIL import Image
def load_images(path, N):
    """
    path - path to images directory example /tmp/exp1/
    N - number of images to load
    output - stack images as pytorch tensor
    """
    images_files = listdir(path)
    images_files.sort()
    if N > 0:
        images_files = images_files[:N]

    images = [path + img for img in images_files if img.endswith(".tif")]
    stack = np.asarray([np.array(Image.open(img)) for img in images],
                       dtype=np.float32)  # have to be float32 for conv2d input
    return torch.from_numpy(stack)
#############################################################################################################################################################




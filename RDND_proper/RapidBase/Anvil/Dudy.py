

from kornia.geometry import ImageRegistrator

from RapidBase.Anvil._internal_utils.torch_utils import construct_tensor
from RapidBase.Anvil._transforms.affine_transformations import affine_transform_interpolated
from RapidBase.Anvil.alignments import align_to_reference_frame_circular_cc
# from RapidBase.Anvil.alignments_layers import minSAD_transforms_layer
from RapidBase.import_all import *
import kornia
from kornia_moons.feature import *

def affine_parameters_from_homography_matrix(H_matrix_tensor, flag_assume_small_angle=True):
    #(1). Translation:
    shift_W = H_matrix_tensor[:, 0, -1]
    shift_H = H_matrix_tensor[:, 1, -1]
    #(2). Scale + Rotation:
    A = H_matrix_tensor[:, 0, 0]
    B = H_matrix_tensor[:, 0, 1]
    D = H_matrix_tensor[:, 1, 0]
    E = H_matrix_tensor[:, 1, 1]
    # A = R*cos(theta)
    # B = -R*sin(theta)
    # D = R*sin(theta)
    # E = R*cos(theta)
    R_1 = torch.sqrt(A**2+B**2)
    R_2 = torch.sqrt(D**2+E**2)
    R_total = (R_1+R_2)/2
    theta_1 = torch.pi/2 - torch.atan(-A/B)
    theta_2 = torch.atan(D/E)
    theta_total_rads = (theta_1 + theta_2)/2

    if flag_assume_small_angle:
        theta_total_rads[theta_total_rads > np.pi/4] = np.pi/2 - theta_total_rads[theta_total_rads > np.pi/4]

    theta_total_degrees = theta_total_rads * 180/np.pi

    return shift_H, shift_W, theta_total_rads, R_total, theta_total_degrees


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


def align_images_ECC(img1, img2, warp_mode='homography', warp_matrix=np.eye(3, 3, dtype=np.float32), iterations=25):

   # Find the width and height of the color image
   height = img1.shape[0]
   width = img1.shape[1]

   # Allocate space for aligned image
   im_aligned = np.zeros((height, width, 1), dtype=np.uint8)

   # Define motion model
   warp_mode = warp_mode.lower()
   if warp_mode == 'translation':
       warp_mode = cv2.MOTION_TRANSLATION
   elif warp_mode == 'euclidean':
       warp_mode = cv2.MOTION_EUCLIDEAN
   elif warp_mode == 'affine':
       warp_mode = cv2.MOTION_AFFINE
   elif warp_mode == 'homography':
       warp_mode = cv2.MOTION_HOMOGRAPHY
   else:
       raise ValueError("Invalid warp mode")

   # Set the warp matrix to identity.
   if warp_mode == cv2.MOTION_HOMOGRAPHY:
       if warp_matrix.shape != (3, 3):
           raise ValueError("Invalid warp matrix: Expected shape(3, 3)")
   else:
       if warp_matrix == np.eye(3, 3, dtype=np.float32):
           warp_matrix = np.eye(2, 3, dtype=np.float32)
       elif warp_matrix.shape != (2, 3):
           raise ValueError("Invalid warp matrix: Expected shape(2, 3)")

   # Set the stopping criteria for the algorithm.
   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-10)

   # Warp the blue and green channels to the red channel
   (cc, warp_matrix) = cv2.findTransformECC(np.mean(get_gradient(img1), -1),
                                            np.mean(get_gradient(img2), -1),
                                            warp_matrix.astype(float32),
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
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

   return im_aligned, warp_matrix



def FeaturesAlign_getFeatures_1(input_tensor_1, input_tensor_2, number_of_features=500, snn_matcher_threshold=0.8, input_mask=None):
    with torch.no_grad():
        #TODO: add possibility of input_mask to avoid/focus on specific regions which might actually be of use to me more then others
        #TODO: add possibility of Non-Maxima suppression of something to avoid simply using very close feature bunddles instead of features from all over the image
        ### Feature Detector + Descriptor + LAFs: ###
        # (1).
        SIFT_detector_and_descriptor = kornia.feature.SIFTFeature(num_features=number_of_features, upright=False, rootsift=True, device=input_tensor_1.device)
        detected_laf_1, response_function_for_laf_1, local_descriptors_1 = SIFT_detector_and_descriptor(input_tensor_1, mask=input_mask)
        detected_laf_2, response_function_for_laf_2, local_descriptors_2 = SIFT_detector_and_descriptor(input_tensor_2, mask=input_mask)
        # (2).
        # GFTT_Hardnet_detector_and_descriptor = kornia.feature.GFTTAffNetHardNet(num_features=number_of_features, upright=False, device=input_tensor.device)
        # detected_laf, response_function_for_laf, local_descriptors = GFTT_Hardnet_detector_and_descriptor.forward(input_tensor, mask=None)
        # #(3).
        # KeyNet_AffNet_detector_and_descriptor = kornia.feature.KeyNetAffNetHardNet(num_features=number_of_features, upright=False, device=input_tensor.device)
        # detected_laf, response_function_for_laf, local_descriptors = KeyNet_AffNet_detector_and_descriptor(input_tensor, mask=None)
        # #(4).
        # KeyNet_HardNet_detector_and_descriptor = kornia.feature.KeyNetHardNet(num_features=number_of_features, upright=False, device=input_tensor.device)
        # detected_laf, response_function_for_laf, local_descriptors = KeyNet_HardNet_detector_and_descriptor(input_tensor, mask=None)

        # ### Get Keypoint Centers Coordinates: ###
        # detected_keypoint_coordinates_XY_1 = detected_laf_1[:, :, :, 2:3].squeeze()
        # detected_keypoint_coordinates_XY_2 = detected_laf_2[:, :, :, 2:3].squeeze()

        ### Match Descriptors Using Kornia: ###
        descriptors_distance, matching_indices = kornia.feature.match_snn(local_descriptors_1.squeeze(0), local_descriptors_2.squeeze(0), th=snn_matcher_threshold, dm=None)
        # descriptors_distance, matching_indices = kornia.feature.match_nn(local_descriptors_1.squeeze(0), local_descriptors_2.squeeze(0), dm=None)

    # ### Get Default Homography Matrix For Now: ###
    # H_matrix_default = torch.zeros(3, 3)
    # H_matrix_default[0, 0] = 1
    # H_matrix_default[1, 1] = 1
    # H_matrix_default[2, 2] = 1
    # H_matrix_default_numpy = H_matrix_default.cpu().numpy()
    # # (*). Draw matches on images:
    # visualize_LAF(input_tensor_1, detected_laf_1, 0, 'y', figsize=(8, 6))
    # draw_LAF_matches(detected_laf_1.cpu(),
    #                  detected_laf_2.cpu(),
    #                  matching_indices.cpu(),
    #                  BW2RGB(input_tensor_1.cpu().clamp(0,255)[0, 0].numpy()).astype(np.uint8),
    #                  BW2RGB(input_tensor_2.cpu().clamp(0,255)[0, 0].numpy()).astype(np.uint8),
    #                  draw_dict={"inlier_color": (0.2, 1, 0.2),
    #                             "tentative_color": (0.8, 0.8, 0),
    #                             "feature_color": None,
    #                             "vertical": True},
    #                  H=H_matrix_default_numpy)

    ### Get Matching-Pairs Indices And Centers: ###
    matching_indices_1 = matching_indices[:, 0]
    matching_indices_2 = matching_indices[:, 1]
    detected_keypoints_centers_1 = detected_laf_1[:, matching_indices_1, :, 2]
    detected_keypoints_centers_2 = detected_laf_2[:, matching_indices_2, :, 2]
    detected_laf_1 = detected_laf_1[:, matching_indices_1]
    detected_laf_2 = detected_laf_2[:, matching_indices_1]
    local_descriptors_1 = local_descriptors_1[:, matching_indices_1]
    local_descriptors_2 = local_descriptors_2[:, matching_indices_1]
    response_function_for_laf_1 = response_function_for_laf_1[:, matching_indices_1]
    response_function_for_laf_2 = response_function_for_laf_2[:, matching_indices_1]

    return detected_laf_1, detected_laf_2, \
           descriptors_distance, \
           detected_keypoints_centers_1, detected_keypoints_centers_2, \
           local_descriptors_1, local_descriptors_2, \
           response_function_for_laf_1, response_function_for_laf_2


def FeaturesAlign_getFeatures_AndHomography_1(input_tensor_1, input_tensor_2, number_of_features=500, snn_matcher_threshold=0.8, input_mask=None, flag_return_warp_error=False):
    with torch.no_grad():
        #TODO: add possibility of input_mask to avoid/focus on specific regions which might actually be of use to me more then others
        #TODO: add possibility of Non-Maxima suppression of something to avoid simply using very close feature bunddles instead of features from all over the image
        ### Feature Detector + Descriptor + LAFs: ###
        # (1).
        SIFT_detector_and_descriptor = kornia.feature.SIFTFeature(num_features=number_of_features, upright=False, rootsift=True, device=input_tensor_1.device)
        detected_laf_1, response_function_for_laf_1, local_descriptors_1 = SIFT_detector_and_descriptor(input_tensor_1, mask=input_mask)
        detected_laf_2, response_function_for_laf_2, local_descriptors_2 = SIFT_detector_and_descriptor(input_tensor_2, mask=input_mask)
        # (2).
        # GFTT_Hardnet_detector_and_descriptor = kornia.feature.GFTTAffNetHardNet(num_features=number_of_features, upright=False, device=input_tensor.device)
        # detected_laf, response_function_for_laf, local_descriptors = GFTT_Hardnet_detector_and_descriptor.forward(input_tensor, mask=None)
        # #(3).
        # KeyNet_AffNet_detector_and_descriptor = kornia.feature.KeyNetAffNetHardNet(num_features=number_of_features, upright=False, device=input_tensor.device)
        # detected_laf, response_function_for_laf, local_descriptors = KeyNet_AffNet_detector_and_descriptor(input_tensor, mask=None)
        # #(4).
        # KeyNet_HardNet_detector_and_descriptor = kornia.feature.KeyNetHardNet(num_features=number_of_features, upright=False, device=input_tensor.device)
        # detected_laf, response_function_for_laf, local_descriptors = KeyNet_HardNet_detector_and_descriptor(input_tensor, mask=None)

        # ### Get Keypoint Centers Coordinates: ###
        # detected_keypoint_coordinates_XY_1 = detected_laf_1[:, :, :, 2:3].squeeze()
        # detected_keypoint_coordinates_XY_2 = detected_laf_2[:, :, :, 2:3].squeeze()

        ### Match Descriptors Using Kornia: ###
        #TODO: add possibility of using batches in match_snn
        descriptors_distance, matching_indices = kornia.feature.match_snn(local_descriptors_1[0], local_descriptors_2[0], th=snn_matcher_threshold, dm=None)
        # descriptors_distance, matching_indices = kornia.feature.match_nn(local_descriptors_1[0], local_descriptors_2[0], dm=None)
        # descriptors_distance, matching_indices = kornia.feature.match_nn(local_descriptors_1.squeeze(0), local_descriptors_2.squeeze(0), dm=None)

    # ### Get Default Homography Matrix For Now: ###
    # H_matrix_default = torch.zeros(3, 3)
    # H_matrix_default[0, 0] = 1
    # H_matrix_default[1, 1] = 1
    # H_matrix_default[2, 2] = 1
    # H_matrix_default_numpy = H_matrix_default.cpu().numpy()
    # # (*). Draw matches on images:
    # visualize_LAF(input_tensor_1, detected_laf_1, 0, 'y', figsize=(8, 6))
    # draw_LAF_matches(detected_laf_1.cpu(),
    #                  detected_laf_2.cpu(),
    #                  matching_indices.cpu(),
    #                  BW2RGB(input_tensor_1.cpu().clamp(0,255)[0, 0].numpy()).astype(np.uint8),
    #                  BW2RGB(input_tensor_2.cpu().clamp(0,255)[0, 0].numpy()).astype(np.uint8),
    #                  draw_dict={"inlier_color": (0.2, 1, 0.2),
    #                             "tentative_color": (0.8, 0.8, 0),
    #                             "feature_color": None,
    #                             "vertical": True},
    #                  H=H_matrix_default_numpy)

    ### Get Matching-Pairs Indices And Centers: ###
    matching_indices_1 = matching_indices[:, 0]
    matching_indices_2 = matching_indices[:, 1]
    detected_keypoints_centers_1 = detected_laf_1[:, matching_indices_1, :, 2]
    detected_keypoints_centers_2 = detected_laf_2[:, matching_indices_2, :, 2]
    detected_laf_1 = detected_laf_1[:, matching_indices_1]
    detected_laf_2 = detected_laf_2[:, matching_indices_1]
    local_descriptors_1 = local_descriptors_1[:, matching_indices_1]
    local_descriptors_2 = local_descriptors_2[:, matching_indices_1]
    response_function_for_laf_1 = response_function_for_laf_1[:, matching_indices_1]
    response_function_for_laf_2 = response_function_for_laf_2[:, matching_indices_1]

    # (*). Find Homography Using Kornia Lease-Squares:
    # (*). Lease Squares DLT:
    H_matrix = kornia.geometry.homography.find_homography_dlt(detected_keypoints_centers_1,
                                                              detected_keypoints_centers_2,
                                                              weights=input_mask)

    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)

    # (*). Return transfer error in image 2 for correspondences given the homography matrix:
    if flag_return_warp_error:
        L1_error_per_point_pair = kornia.geometry.homography.oneway_transfer_error(detected_keypoints_centers_1,
                                                                                   detected_keypoints_centers_2,
                                                                                   H_matrix,
                                                                                   squared=True,  # TODO: doesn't this mean L2?!?!?
                                                                                   eps=1e-08)
    else:
        L1_error_per_point_pair = None

    return rotation_degrees, detected_laf_1, detected_laf_2, \
           descriptors_distance, \
           detected_keypoints_centers_1, detected_keypoints_centers_2, \
           local_descriptors_1, local_descriptors_2, \
           response_function_for_laf_1, response_function_for_laf_2


def FeaturesAlign_getFeatures_AndHomography_1_Batch_SNN(input_tensor_1, input_tensor_2, number_of_features=500, snn_matcher_threshold=0.8, input_mask=None, flag_return_warp_error=False):
    with torch.no_grad():
        #TODO: add possibility of input_mask to avoid/focus on specific regions which might actually be of use to me more then others
        #TODO: add possibility of Non-Maxima suppression of something to avoid simply using very close feature bunddles instead of features from all over the image
        ### Feature Detector + Descriptor + LAFs: ###
        # # (1).
        SIFT_detector_and_descriptor = kornia.feature.SIFTFeature(num_features=number_of_features, upright=False, rootsift=True, device=input_tensor_1.device)
        detected_laf_1, response_function_for_laf_1, local_descriptors_1 = SIFT_detector_and_descriptor(input_tensor_1, mask=input_mask)
        detected_laf_2, response_function_for_laf_2, local_descriptors_2 = SIFT_detector_and_descriptor(input_tensor_2, mask=input_mask)
        # (2).
        # GFTTAffNetHardNet_detector_and_descriptor = kornia.feature.GFTTAffNetHardNet(num_features=number_of_features, upright=False, device=input_tensor_1.device)
        # detected_laf_1, response_function_for_laf_1, local_descriptors_1 = GFTTAffNetHardNet_detector_and_descriptor(input_tensor_1, mask=input_mask)
        # detected_laf_2, response_function_for_laf_2, local_descriptors_2 = GFTTAffNetHardNet_detector_and_descriptor(input_tensor_2, mask=input_mask)
        # # (3).  #TODO: this only supports single image input!
        # KeyNetAffNetHardNet_detector_and_descriptor = kornia.feature.KeyNetAffNetHardNet(num_features=number_of_features, upright=False, device=input_tensor_1.device)
        # detected_laf_1, response_function_for_laf_1, local_descriptors_1 = KeyNetAffNetHardNet_detector_and_descriptor(input_tensor_1[0:1], mask=input_mask)
        # detected_laf_2, response_function_for_laf_2, local_descriptors_2 = KeyNetAffNetHardNet_detector_and_descriptor(input_tensor_2[0:1], mask=input_mask)
        # # #(4). #TODO: this only supports single image input!
        # KeyNetHardNet_detector_and_descriptor = kornia.feature.KeyNetHardNet(num_features=number_of_features, upright=False, device=input_tensor_1.device)
        # detected_laf_1, response_function_for_laf_1, local_descriptors_1 = KeyNetHardNet_detector_and_descriptor(input_tensor_1, mask=input_mask)
        # detected_laf_2, response_function_for_laf_2, local_descriptors_2 = KeyNetHardNet_detector_and_descriptor(input_tensor_2, mask=input_mask)

        # ### Get Keypoint Centers Coordinates: ###
        # detected_keypoint_coordinates_XY_1 = detected_laf_1[:, :, :, 2:3].squeeze()
        # detected_keypoint_coordinates_XY_2 = detected_laf_2[:, :, :, 2:3].squeeze()

        ### Match Descriptors Using Kornia: ###
        #TODO: add possibility of using batches in match_snn
        # descriptors_distance, matching_indices = kornia.feature.match_nn_batch(local_descriptors_1, local_descriptors_2, dm=None)
        descriptors_distance, matching_indices = kornia.feature.match_snn_batch(local_descriptors_1, local_descriptors_2, th=snn_matcher_threshold, dm=None)
        # descriptors_distance, matching_indices = kornia.feature.match_snn(local_descriptors_1[0], local_descriptors_2[0], th=snn_matcher_threshold, dm=None)
        # descriptors_distance, matching_indices = kornia.feature.match_nn(local_descriptors_1.squeeze(0), local_descriptors_2.squeeze(0), dm=None)

    # ### Get Default Homography Matrix For Now: ###
    # H_matrix_default = torch.zeros(3, 3)
    # H_matrix_default[0, 0] = 1
    # H_matrix_default[1, 1] = 1
    # H_matrix_default[2, 2] = 1
    # H_matrix_default_numpy = H_matrix_default.cpu().numpy()
    # # (*). Draw matches on images:
    # visualize_LAF(input_tensor_1, detected_laf_1, 0, 'y', figsize=(8, 6))
    # draw_LAF_matches(detected_laf_1.cpu(),
    #                  detected_laf_2.cpu(),
    #                  matching_indices.cpu(),
    #                  BW2RGB(input_tensor_1.cpu().clamp(0,255)[0, 0].numpy()).astype(np.uint8),
    #                  BW2RGB(input_tensor_2.cpu().clamp(0,255)[0, 0].numpy()).astype(np.uint8),
    #                  draw_dict={"inlier_color": (0.2, 1, 0.2),
    #                             "tentative_color": (0.8, 0.8, 0),
    #                             "feature_color": None,
    #                             "vertical": True},
    #                  H=H_matrix_default_numpy)

    ### Get Matching-Pairs Indices And Centers: ###
    matching_indices_1 = matching_indices[..., 0]
    matching_indices_2 = matching_indices[..., 1]

    ### Repeat the reference tensor as many times as needed (1 or B) to allow for proper handling of indices below: ###
    [B1,N1,_,_] = detected_laf_1.shape
    [B2,N2,_,_] = detected_laf_2.shape
    repeat_ratio = (B1//B2)
    detected_laf_2 = detected_laf_2.repeat(repeat_ratio,1,1,1)
    response_function_for_laf_2 = response_function_for_laf_2.repeat(repeat_ratio,1)
    local_descriptors_2 = local_descriptors_2.repeat(repeat_ratio,1,1)

    detected_laf_1 = torch.gather(detected_laf_1, dim=1, index=matching_indices_1.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,3))
    detected_laf_2 = torch.gather(detected_laf_2, dim=1, index=matching_indices_2.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2,3))
    detected_keypoints_centers_1 = detected_laf_1[..., 2]
    detected_keypoints_centers_2 = detected_laf_2[..., 2]

    local_descriptors_1 = torch.gather(local_descriptors_1, dim=1, index=matching_indices_1.unsqueeze(-1))
    local_descriptors_2 = torch.gather(local_descriptors_2, dim=1, index=matching_indices_2.unsqueeze(-1))

    response_function_for_laf_1 = torch.gather(response_function_for_laf_1, dim=1, index=matching_indices_1)
    response_function_for_laf_2 = torch.gather(response_function_for_laf_2, dim=1, index=matching_indices_2)

    # (*). Find Homography Using Kornia Lease-Squares:
    # (*). Lease Squares DLT:
    H_matrix = kornia.geometry.homography.find_homography_dlt(detected_keypoints_centers_1,
                                                              detected_keypoints_centers_2,
                                                              weights=input_mask)

    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)

    # (*). Return transfer error in image 2 for correspondences given the homography matrix:
    if flag_return_warp_error:
        L1_error_per_point_pair = kornia.geometry.homography.oneway_transfer_error(detected_keypoints_centers_1,
                                                                                   detected_keypoints_centers_2,
                                                                                   H_matrix,
                                                                                   squared=True,  # TODO: doesn't this mean L2?!?!?
                                                                                   eps=1e-08)
        L1_error_per_point_pair = L1_error_per_point_pair.mean(-1)
    else:
        L1_error_per_point_pair = None

    return rotation_degrees, detected_laf_1, detected_laf_2, \
           descriptors_distance, \
           detected_keypoints_centers_1, detected_keypoints_centers_2, \
           local_descriptors_1, local_descriptors_2, \
           response_function_for_laf_1, response_function_for_laf_2


def FeaturesAlign_getFeatures_2(input_tensor_1, input_tensor_2, number_of_features=500, snn_matcher_threshold=0.8, input_mask=None):
    ### Feature Detector + Descriptor + LAFs + Matcher: ###

    ### Feature Detector + Descriptor: ###
    feature_detector_and_descriptor_module = kornia.feature.GFTTAffNetHardNet(number_of_features).to(input_tensor_1.device)
    # feature_detector_and_descriptor_module = kornia.feature.SIFTFeature(num_features=500, upright=False, rootsift=True, device=input_tensor.device)

    ### Matcher: ###
    descriptor_matcher_module = kornia.feature.DescriptorMatcher('snn', snn_matcher_threshold).to(input_tensor_1.device)

    ### Combined Module: ###
    gftt_hardnet_detector_plus_matcher = kornia.feature.LocalFeatureMatcher(local_feature=feature_detector_and_descriptor_module,
                                                                            matcher=descriptor_matcher_module)

    ### Define input_dict for module: ###
    # (*). input tensors should be normalized to [0,1]!!!!!!!!!!!!!!!!!!!!!!1
    input_dict_to_matcher = {'image0': input_tensor_1,
                             'image1': input_tensor_2}
    with torch.no_grad():
        output_dict = gftt_hardnet_detector_plus_matcher.forward(input_dict_to_matcher)
        detected_keypoints_centers_1 = output_dict['keypoints0']
        detected_keypoints_centers_2 = output_dict['keypoints1']
        detected_laf_1 = output_dict['lafs0']
        detected_laf_2 = output_dict['lafs1']
        matching_confidence = output_dict['confidence']
        batch_indexes_KorniaWrapper = output_dict['batch_indexes']

        # ### Put condition on confidence if wanted: ###
        # matching_indices_logical_mask = confidence_KorniaWrapper > 0.3
        # lafs_1_KorniaWrapper_valid = lafs_1_KorniaWrapper[:, matching_indices_logical_mask, :, :]
        # lafs_2_KorniaWrapper_valid = lafs_2_KorniaWrapper[:, matching_indices_logical_mask, :, :]
        # arange_indices_tensor = torch.arange(0, lafs_1_KorniaWrapper.shape[1])
        # valid_indices = arange_indices_tensor[matching_indices_logical_mask].long()
        #
        # ### Get Keypoints Centers From LAFs: ###
        # detected_keypoints_centers_1 = lafs_1_KorniaWrapper[:, valid_indices, :, 2]
        # detected_keypoints_centers_2 = lafs_2_KorniaWrapper[:, valid_indices, :, 2]
        # confidence_KorniaWrapper = confidence_KorniaWrapper[valid_indices]

    H_matrix_default = torch.zeros(3, 3)
    H_matrix_default[0, 0] = 1
    H_matrix_default[1, 1] = 1
    H_matrix_default[2, 2] = 1
    H_matrix_default_numpy = H_matrix_default.cpu().numpy()

    # (*). Draw matches on images:
    draw_LAF_matches_from_result_dict(output_dict,
                     BW2RGB(input_tensor_1.cpu()[0, 0].numpy()),
                     BW2RGB(input_tensor_2.cpu()[0, 0].numpy()),
                     draw_dict={"inlier_color": (0.2, 1, 0.2),
                                "tentative_color": (0.8, 0.8, 0),
                                "feature_color": None,
                                "vertical": True},
                     H=H_matrix_default_numpy)

    return detected_laf_1, detected_laf_2, \
           matching_confidence, \
           detected_keypoints_centers_1, detected_keypoints_centers_2, \


def get_Homography_matrix_from_keypoints_DLT(detected_keypoints_centers_1, detected_keypoints_centers_2, mask_weights=None, flag_return_warp_error=False):
    # (*). Find Homography Using Kornia Lease-Squares:
    # (*). Lease Squares DLT:
    H_matrix = kornia.geometry.homography.find_homography_dlt(detected_keypoints_centers_1,
                                                              detected_keypoints_centers_2,
                                                              weights=mask_weights)

    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)

    # (*). Return transfer error in image 2 for correspondences given the homography matrix:
    if flag_return_warp_error:
        L1_error_per_point_pair = kornia.geometry.homography.oneway_transfer_error(detected_keypoints_centers_1,
                                                                                   detected_keypoints_centers_2,
                                                                                   H_matrix,
                                                                                   squared=True,  # TODO: doesn't this mean L2?!?!?
                                                                                   eps=1e-08)
    else:
        L1_error_per_point_pair = None

    return shift_H, shift_W, rotation_rads, scale, rotation_degrees, H_matrix, L1_error_per_point_pair


def get_Homography_matrix_from_keypoints_ReweightedIterativeDLT(detected_keypoints_centers_1, detected_keypoints_centers_2, mask_weights=None, soft_inl_th=4, n_iter=5, flag_return_warp_error=False):
    # (*). Iterative ReWeighted DLT:
    H_matrix = kornia.geometry.homography.find_homography_dlt_iterated(detected_keypoints_centers_1,
                                                                       detected_keypoints_centers_2,
                                                                       weights=mask_weights,
                                                                       soft_inl_th=soft_inl_th,
                                                                       n_iter=n_iter)
    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)

    # (*). Return transfer error in image 2 for correspondences given the homography matrix:
    if flag_return_warp_error:
        L1_error_per_point_pair = kornia.geometry.homography.oneway_transfer_error(detected_keypoints_centers_1,
                                                                                   detected_keypoints_centers_2,
                                                                                   H_matrix,
                                                                                   squared=True,  # TODO: doesn't this mean L2?!?!?
                                                                                   eps=1e-08)
    else:
        L1_error_per_point_pair = None

    return shift_H, shift_W, rotation_rads, scale, rotation_degrees, H_matrix, L1_error_per_point_pair


def get_Homography_matrix_from_keypoints_RANSAC(detected_keypoints_centers_1, detected_keypoints_centers_2,
                                                inl_th=2, batch_size=2048, max_iter=10, confidence=0.99, max_lo_iters=5, flag_return_warp_error=False):
    ### Find Homography Using Kornia RANSAC: ###
    # TODO: add RANSAC for affine model instead of only homography
    RANSAC_module = kornia.geometry.ransac.RANSAC(model_type='homography',
                                                  inl_th=inl_th,
                                                  batch_size=batch_size,
                                                  max_iter=max_iter,
                                                  confidence=confidence,
                                                  max_lo_iters=max_lo_iters)
    H_matrix, inlinear_mask = RANSAC_module.forward(detected_keypoints_centers_1,
                                                    detected_keypoints_centers_2)

    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)

    # (*). Return transfer error in image 2 for correspondences given the homography matrix:
    if flag_return_warp_error:
        L1_error_per_point_pair = kornia.geometry.homography.oneway_transfer_error(detected_keypoints_centers_1,
                                                                                   detected_keypoints_centers_2,
                                                                                   H_matrix,
                                                                                   squared=True,  # TODO: doesn't this mean L2?!?!?
                                                                                   eps=1e-08)
    else:
        L1_error_per_point_pair = None

    return shift_H, shift_W, rotation_rads, scale, rotation_degrees, H_matrix, L1_error_per_point_pair

def Align_Kornia(input_tensor, reference_tensor):
    T2, C2, H2, W2 = reference_tensor.shape
    T1, C1, H1, W1 = input_tensor.shape
    registrator = ImageRegistrator('similarity')

    aligned_frames_list_1 = []
    aligned_frames_list_2 = []
    H_matrix_list = []
    aligned_frames_list_2 = []
    for i in np.arange(T1):
        print(i)
        current_tensor = input_tensor[i:i+1]
        homo = registrator.register(current_tensor, reference_tensor)
        # homo = registrator.register(aligned_tensor, aligned_tensor[T//2:T//2+1].repeat(T,1,1,1))
        warped_aligned_1 = kornia.geometry.transform.warp_perspective(current_tensor, homo.to(current_tensor.device), (H1,W1), mode='bilinear', padding_mode='zeros', align_corners=True)
        warped_aligned_2 = kornia.geometry.transform.warp_perspective(reference_tensor, homo.to(current_tensor.device), (H1,W1), mode='bilinear', padding_mode='zeros', align_corners=True)
        aligned_frames_list_1.append(warped_aligned_1.cpu())
        aligned_frames_list_2.append(warped_aligned_2.cpu())
        H_matrix_list.append(homo)
    aligned_frames_Kornia_1 = torch.cat(aligned_frames_list_1, 0).squeeze(0)
    aligned_frames_Kornia_2 = torch.cat(aligned_frames_list_2, 0).squeeze(0)

    return aligned_frames_Kornia_1, aligned_frames_Kornia_2, H_matrix_list


def Align_FeatureBased(input_tensor, reference_tensor):
    aligned_frames_list = []
    H_matrix_list = []
    T2, C2, H2, W2 = reference_tensor.shape
    T1, C1, H1, W1 = input_tensor.shape

    reference_frame_RGB = BW2RGB(reference_tensor.cpu()[0, 0].clamp(0, 255).type(torch.uint8).numpy())
    for i in np.arange(T1):
        print(i)
        current_frame_RGB = BW2RGB(input_tensor[i:i + 1].cpu()[0, 0].clamp(0, 255).type(torch.uint8).numpy())
        current_frame_aligned, H_matrix = align_images_OpenCVFeatures(current_frame_RGB, reference_frame_RGB, maxFeatures=1000, keepPercent=0.5, debug=False)
        current_frame_aligned = current_frame_aligned.astype(np.float)
        current_frame_aligned = RGB2BW(current_frame_aligned).squeeze(-1)
        aligned_frames_list.append(torch.tensor(current_frame_aligned).unsqueeze(0).unsqueeze(0))
        H_matrix_list.append(H_matrix)

    aligned_tensor_FeatureAlign = torch.cat(aligned_frames_list, 0).squeeze(0)
    return aligned_tensor_FeatureAlign, H_matrix_list


def Align_ECC(input_tensor, reference_tensor, warp_mode='homography', warp_matrix=torch.eye(3, 3).float(), iterations=25):
   aligned_frames_list = []
   H_matrix_list = []
   T2 ,C2 ,H2 ,W2 = reference_tensor.shape
   T1 ,C1 ,H1 ,W1 = input_tensor.shape
   shift_H_list = []
   shift_W_list = []
   rotation_rad_list = []
   rotation_degree_list = []
   reference_frame_RGB = BW2RGB(reference_tensor.cpu()[0 ,0].clamp(0 ,255).type(torch.uint8).numpy())
   for i in np.arange(T1):
       print('Frame: ' + str(i))
       current_frame_RGB = BW2RGB(input_tensor[i:i + 1].cpu()[0, 0].clamp(0, 255).type(torch.uint8).numpy())
       if isinstance(warp_matrix, Tensor):
           warp_matrix = warp_matrix.cpu()
       warp_matrix = np.array(warp_matrix)
       current_frame_aligned, H_matrix = \
           align_images_ECC(reference_frame_RGB, current_frame_RGB, warp_mode, warp_matrix, iterations)
       warp_matrix = H_matrix
       current_frame_aligned = RGB2BW(current_frame_aligned.astype(np.float))
       H, W, C = current_frame_aligned.shape
       aligned_frames_list.append(torch.tensor(current_frame_aligned).view(C, H, W).unsqueeze(0).unsqueeze(0))
       H_matrix_list.append(H_matrix)

       shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
           affine_parameters_from_homography_matrix(torch.tensor(H_matrix).unsqueeze(0))
       shift_H_list.append(shift_H)
       shift_W_list.append(shift_W)
       rotation_rad_list.append(rotation_rads)
       rotation_degree_list.append(rotation_rads * 180/np.pi)

   rotation_rad = torch.cat(rotation_rad_list).to(input_tensor.device)
   rotation_degree = torch.cat(rotation_degree_list).to(input_tensor.device)
   shift_H = torch.cat(shift_H_list).to(input_tensor.device)
   shift_W = torch.cat(shift_W_list).to(input_tensor.device)
   H_matrix = torch.from_numpy(np.array(H_matrix_list)).to(input_tensor.device)
   aligned_tensor_FeatureAlign = torch.cat(aligned_frames_list, 1).squeeze(0).to(input_tensor.device)

   return aligned_tensor_FeatureAlign, H_matrix, rotation_rad, rotation_degree, shift_H, shift_W


def Align_Palantir_CC_Then_Classic(input_tensor, reference_tensor, flag_plot=False):
    ### Cross Correlation initial correction: ###
    # (1). compare to reference (center) frame:
    T,C,H,W = input_tensor.shape

    # shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(input_tensor, input_tensor[:,T//2:T//2+1], None, None)
    aligned_tensor, shifts_h, shifts_w, CC_matrix = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                         reference_matrix=reference_tensor,
                                                                                         matrix_fft=None,
                                                                                         normalize_over_matrix=False,
                                                                                         warp_method='bilinear',
                                                                                         crop_warped_matrix=False)
    max_shift_H = shifts_h.abs().max()
    max_shift_W = shifts_w.abs().max()
    new_H = H - (max_shift_H.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_W.abs().ceil().int().cpu().numpy() * 2 + 5)

    ### Shift input tensor to make aligned tensor: ###
    input_tensor = input_tensor.cpu()
    torch.cuda.empty_cache()

    ### Crop both original and aligned to be the same size and correct for size disparities (TODO: will be corrected to be seemless): ###
    aligned_tensor = crop_torch_batch(aligned_tensor, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    ### Present results after cross correlation: ###
    if flag_plot:
        concat_tensor = torch.cat([input_tensor.cpu(), aligned_tensor.cpu()], dim=-1)
        imshow_torch_video(input_tensor, FPS=50)
        imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(concat_tensor, FPS=50)
        imshow_torch_video(input_tensor - input_tensor[T // 2:T // 2 + 1], FPS=50)
        imshow_torch_video(aligned_tensor - aligned_tensor[T // 2:T // 2 + 1], FPS=50)
        del concat_tensor

    ### Free up GPU memory: ###
    input_tensor = input_tensor.to('cpu')
    aligned_tensor = aligned_tensor.to('cpu')
    del CC_matrix
    torch.cuda.empty_cache()

    ####################################################################################
    ### Second Stage Alignment Using "Classical" Methods: ###
    # ### Try Kornia: ###
    # aligned_frames_Kornia_1, aligned_frames_Kornia_2, H_matrix_list = Align_Kornia(aligned_tensor, reference_tensor)
    # aligned_frames_final = aligned_frames_Kornia_1
    # if flag_plot:
    #     imshow_torch_video(aligned_frames_Kornia_2, FPS=50, frame_stride=5)
    #     imshow_torch_video(aligned_frames_Kornia_1 - aligned_frames_Kornia_1[T // 2:T // 2 + 1], FPS=50, frame_stride=5)

    ### Try OpenCV Feature Based: ###
    aligned_tensor_FeatureAlign, H_matrix_list = Align_FeatureBased(aligned_tensor, reference_tensor)
    aligned_tensor_final = aligned_tensor_FeatureAlign
    if flag_plot:
        imshow_torch_video(aligned_tensor_FeatureAlign, FPS=50, frame_stride=5)
        imshow_torch_video(aligned_tensor_FeatureAlign - aligned_tensor_FeatureAlign[T // 2:T // 2 + 1], FPS=50, frame_stride=5)

    # ### Try OpenCV ECC: ###
    # aligned_frames_ECC_tensor, H_matrix_list = Align_ECC(aligned_tensor, reference_tensor)
    # if flag_plot:
    #     imshow_torch_video(aligned_frames_ECC_tensor, FPS=50, frame_stride=5)
    #     imshow_torch_video(aligned_frames_ECC_tensor - aligned_frames_ECC_tensor[T // 2:T // 2 + 1], FPS=50, frame_stride=5)
    # aligned_tensor_final = aligned_frames_ECC_tensor
    ####################################################################################

    return input_tensor, aligned_tensor, aligned_tensor_final


def Align_Palantir_CC_Then_Classic_Then_CC(input_tensor, reference_tensor, flag_plot=False):
    ### Cross Correlation initial correction: ###
    # (1). compare to reference (center) frame:
    T,C,H,W = input_tensor.shape

    # shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(input_tensor, input_tensor[:,T//2:T//2+1], None, None)
    aligned_tensor, shifts_h, shifts_w, CC_matrix = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                         reference_matrix=reference_tensor,
                                                                                         matrix_fft=None,
                                                                                         normalize_over_matrix=False,
                                                                                         warp_method='fft',
                                                                                         crop_warped_matrix=False)
    max_shift_H = shifts_h.abs().max()
    max_shift_W = shifts_w.abs().max()
    new_H = H - (max_shift_H.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_W.abs().ceil().int().cpu().numpy() * 2 + 5)

    ### Shift input tensor to make aligned tensor: ###
    input_tensor = input_tensor.cpu()
    torch.cuda.empty_cache()

    ### Crop both original and aligned to be the same size and correct for size disparities (TODO: will be corrected to be seemless): ###
    aligned_tensor = crop_torch_batch(aligned_tensor, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    ### Present results after cross correlation: ###
    if flag_plot:
        concat_tensor = torch.cat([input_tensor.cpu(), aligned_tensor.cpu()], dim=-1)
        imshow_torch_video(input_tensor, FPS=50)
        imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(concat_tensor, FPS=50)
        imshow_torch_video(input_tensor - input_tensor[T // 2:T // 2 + 1], FPS=50)
        imshow_torch_video(aligned_tensor - aligned_tensor[T // 2:T // 2 + 1], FPS=50)
        del concat_tensor

    ### Free up GPU memory: ###
    input_tensor = input_tensor.to('cpu')
    aligned_tensor = aligned_tensor.to('cpu')
    del CC_matrix
    torch.cuda.empty_cache()

    ####################################################################################
    ### Second Stage Alignment Using "Classical" Methods: ###
    # ### Try Kornia: ###
    # aligned_frames_Kornia_1, aligned_frames_Kornia_2, H_matrix_list = Align_Kornia(aligned_tensor, reference_tensor)
    # aligned_frames_final = aligned_frames_Kornia_1
    # if flag_plot:
    #     imshow_torch_video(aligned_frames_Kornia_2, FPS=50, frame_stride=5)
    #     imshow_torch_video(aligned_frames_Kornia_1 - aligned_frames_Kornia_1[T // 2:T // 2 + 1], FPS=50, frame_stride=5)

    ### Try OpenCV Feature Based: ###
    aligned_tensor_FeatureAlign, H_matrix_list = Align_FeatureBased(aligned_tensor, reference_tensor)
    aligned_frames_classic_tensor = aligned_tensor_FeatureAlign
    if flag_plot:
        imshow_torch_video(aligned_tensor_FeatureAlign, FPS=50, frame_stride=5)
        imshow_torch_video(aligned_tensor_FeatureAlign - aligned_tensor_FeatureAlign[T // 2:T // 2 + 1], FPS=50, frame_stride=5)

    # ### Try OpenCV ECC: ###
    # aligned_frames_ECC_tensor, H_matrix_list = Align_ECC(aligned_tensor, reference_tensor)
    # H_matrix_tensor = torch.tensor(np.concatenate(numpy_unsqueeze(H_matrix_list,0), 0))
    # aligned_frames_classic_tensor = aligned_frames_ECC_tensor
    # ### Motion Estimation From H_matrix: ###
    # #(1). Translation:
    # shifts_x_ECC = H_matrix_tensor[:, 0, -1]
    # shifts_y_ECC = H_matrix_tensor[:, 1, -1]
    # #(2). Scale + Rotation:
    # A = H_matrix_tensor[:, 0, 0]
    # B = H_matrix_tensor[:, 0, 1]
    # D = H_matrix_tensor[:, 1, 0]
    # E = H_matrix_tensor[:, 1, 1]
    # # A = R*cos(theta)
    # # B = -R*sin(theta)
    # # D = R*sin(theta)
    # # E = R*cos(theta)
    # R_1 = torch.sqrt(A**2+B**2)
    # R_2 = torch.sqrt(D**2+E**2)
    # R_total = (R_1+R_2)/2
    # theta_1 = torch.pi/2 - torch.atan(-A/B)
    # theta_2 = torch.atan(D/E)
    # theta_total = (theta_1 + theta_2)/2

    #TODO: use the H_matrix to get the translation and rotations to be able to perform precise fft alignment
    if flag_plot:
        imshow_torch_video(aligned_frames_classic_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(aligned_frames_classic_tensor - aligned_frames_classic_tensor[T // 2:T // 2 + 1], FPS=50, frame_stride=5)
    ####################################################################################


    ####################################################################################
    ### Cross Correlation Again: ###
    aligned_frames_classic_tensor = aligned_frames_classic_tensor.to(reference_tensor.device)
    H3,W3 = aligned_frames_classic_tensor.shape[-2:]
    before_final_CC_ROI = (H3-10,W3-10)
    aligned_tensor_after_classic_and_CC, shifts_h, shifts_w, CC_matrix = align_to_reference_frame_circular_cc(matrix=crop_tensor(aligned_frames_classic_tensor, before_final_CC_ROI),
                                                                                         reference_matrix=crop_tensor(reference_tensor, before_final_CC_ROI),
                                                                                         matrix_fft=None,
                                                                                         normalize_over_matrix=False,
                                                                                         warp_method='fft',
                                                                                         crop_warped_matrix=False)
    ####################################################################################

    # aligned_tensor_final = aligned_frames_ECC_tensor
    aligned_tensor_final = aligned_tensor_after_classic_and_CC
    return input_tensor, aligned_tensor, aligned_tensor_final



def Align_Palantir_CC_Then_Classic_Then_minSAD(input_tensor, reference_tensor, flag_plot=False):
    ### Cross Correlation initial correction: ###
    # (1). compare to reference (center) frame:
    T,C,H,W = input_tensor.shape

    # shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(input_tensor, input_tensor[:,T//2:T//2+1], None, None)
    aligned_tensor, shifts_h, shifts_w, CC_matrix = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                         reference_matrix=reference_tensor,
                                                                                         matrix_fft=None,
                                                                                         normalize_over_matrix=False,
                                                                                         warp_method='fft',
                                                                                         crop_warped_matrix=False)
    max_shift_H = shifts_h.abs().max()
    max_shift_W = shifts_w.abs().max()
    new_H = H - (max_shift_H.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_W.abs().ceil().int().cpu().numpy() * 2 + 5)

    ### Shift input tensor to make aligned tensor: ###
    input_tensor = input_tensor.cpu()
    torch.cuda.empty_cache()

    ### Crop both original and aligned to be the same size and correct for size disparities (TODO: will be corrected to be seemless): ###
    aligned_tensor = crop_torch_batch(aligned_tensor, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])
    reference_tensor = crop_torch_batch(reference_tensor, [new_H, new_W])

    ### Present results after cross correlation: ###
    if flag_plot:
        concat_tensor = torch.cat([input_tensor.cpu(), aligned_tensor.cpu()], dim=-1)
        imshow_torch_video(input_tensor, FPS=50)
        imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(concat_tensor, FPS=50)
        imshow_torch_video(input_tensor - input_tensor[T // 2:T // 2 + 1], FPS=50)
        imshow_torch_video(aligned_tensor - aligned_tensor[T // 2:T // 2 + 1], FPS=50)
        del concat_tensor

    ### Free up GPU memory: ###
    input_tensor = input_tensor.to('cpu')
    aligned_tensor = aligned_tensor.to('cpu')
    del CC_matrix
    torch.cuda.empty_cache()

    ####################################################################################
    ### Second Stage Alignment Using "Classical" Methods: ###
    # ### Try Kornia: ###
    # aligned_frames_Kornia_1, aligned_frames_Kornia_2, H_matrix_list = Align_Kornia(aligned_tensor, reference_tensor)
    # aligned_frames_final = aligned_frames_Kornia_1
    # if flag_plot:
    #     imshow_torch_video(aligned_frames_Kornia_2, FPS=50, frame_stride=5)
    #     imshow_torch_video(aligned_frames_Kornia_1 - aligned_frames_Kornia_1[T // 2:T // 2 + 1], FPS=50, frame_stride=5)

    ### Try OpenCV Feature Based: ###
    aligned_tensor_FeatureAlign, H_matrix_list = Align_FeatureBased(aligned_tensor, reference_tensor)
    aligned_frames_classic_tensor = aligned_tensor_FeatureAlign
    if flag_plot:
        imshow_torch_video(aligned_tensor_FeatureAlign, FPS=50, frame_stride=5)
        imshow_torch_video(aligned_tensor_FeatureAlign - aligned_tensor_FeatureAlign[T // 2:T // 2 + 1], FPS=50, frame_stride=5)

    # ### Try OpenCV ECC: ###
    # aligned_frames_ECC_tensor, H_matrix_list = Align_ECC(aligned_tensor, reference_tensor)
    # H_matrix_tensor = torch.tensor(np.concatenate(numpy_unsqueeze(H_matrix_list,0), 0))
    # aligned_frames_classic_tensor = aligned_frames_ECC_tensor
    # ### Motion Estimation From H_matrix: ###
    # #(1). Translation:
    # shifts_x_ECC = H_matrix_tensor[:, 0, -1]
    # shifts_y_ECC = H_matrix_tensor[:, 1, -1]
    # #(2). Scale + Rotation:
    # A = H_matrix_tensor[:, 0, 0]
    # B = H_matrix_tensor[:, 0, 1]
    # D = H_matrix_tensor[:, 1, 0]
    # E = H_matrix_tensor[:, 1, 1]
    # # A = R*cos(theta)
    # # B = -R*sin(theta)
    # # D = R*sin(theta)
    # # E = R*cos(theta)
    # R_1 = torch.sqrt(A**2+B**2)
    # R_2 = torch.sqrt(D**2+E**2)
    # R_total = (R_1+R_2)/2
    # theta_1 = torch.pi/2 - torch.atan(-A/B)
    # theta_2 = torch.atan(D/E)
    # theta_total = (theta_1 + theta_2)/2

    #TODO: use the H_matrix to get the translation and rotations to be able to perform precise fft alignment
    if flag_plot:
        imshow_torch_video(aligned_frames_classic_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(aligned_frames_classic_tensor - aligned_frames_classic_tensor[T // 2:T // 2 + 1], FPS=50, frame_stride=5)
    ####################################################################################


    ####################################################################################
    ### Cross Correlation Again: ###
    aligned_frames_classic_tensor = aligned_frames_classic_tensor.to(reference_tensor.device).float()
    # (2). min-SAD Reference Comparison:
    binning_size = 4
    aligned_tensor_binned = fast_binning_2D_overap_flexible_AvgPool2d(aligned_frames_classic_tensor, (binning_size, binning_size), (0, 0))
    reference_tensor_binned = fast_binning_2D_overap_flexible_AvgPool2d(reference_tensor, (binning_size, binning_size), (0, 0))
    T, C, H, W = aligned_tensor_binned.shape
    minSAD = minSAD_transforms_layer(1, T, H, W)
    warped_tensor, shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
        minSAD.align(matrix=aligned_tensor_binned,
                     reference_matrix=reference_tensor_binned,
                     shift_h_vec=[-1, 0, 1],
                     shift_w_vec=[-1, 0, 1],
                     rotation_vec=[-0.4, -0.2, 0, 0.2, 0.4],
                     scale_vec=[1],
                     warp_method='bicubic',
                     return_shifts=True)
    H, W = aligned_tensor_binned.shape[-2:]
    warped_tensor = crop_torch_batch(warped_tensor, (H, W)).cpu()
    aligned_tensor_binned = aligned_tensor_binned.cpu()

    if flag_plot:
        plot_torch(shifts_h);
        plt.show()
        plot_torch(shifts_w);
        plt.show()
        plot_torch(rotational_shifts);
        plt.show()
        sad_torch_list_2 = (warped_tensor - aligned_tensor_binned[T // 2:T // 2 + 1]).abs().mean([-1, -2]).squeeze()
        sad_torch_list_1 = (aligned_tensor_binned - aligned_tensor_binned[T // 2:T // 2 + 1]).abs().mean([-1, -2]).squeeze()
        string_list = []
        for i in np.arange(len(sad_torch_list_1)):
            string_list.append(decimal_notation(sad_torch_list_1[i].item(), 2) + ' , ' + decimal_notation(sad_torch_list_2[i].item(), 2))
        concat_tensor = torch.cat([aligned_tensor_binned, warped_tensor], -1)
        concat_aligned_tensor = torch.cat([aligned_tensor_binned, aligned_tensor_binned], -1)
        imshow_torch_video(concat_tensor, FPS=25, frame_stride=5, video_title_list=string_list)
        imshow_torch_video(concat_tensor - concat_aligned_tensor[T // 2:T // 2 + 1], FPS=25, frame_stride=2, video_title_list=string_list)

    ### Warp original resolution instead of binned resolution: ###
    shifts_h_full_resolution = shifts_h * binning_size
    shifts_w_full_resolution = shifts_w * binning_size
    rotational_shifts_full_resolution = rotational_shifts * binning_size
    scale_shifts_full_resolution = scale_shifts * 1
    warped_tensor_full_resolution = affine_transform_interpolated(aligned_frames_classic_tensor.unsqueeze(0),
                                                                  construct_tensor(-shifts_h_full_resolution),
                                                                  construct_tensor(-shifts_w_full_resolution),
                                                                  construct_tensor(-rotational_shifts_full_resolution * np.pi / 180),
                                                                  construct_tensor(1 / scale_shifts_full_resolution),
                                                                  warp_method='bilinear',
                                                                  expand=False)
    warped_tensor_full_resolution = warped_tensor_full_resolution.squeeze(0)
    H, W = aligned_tensor.shape[-2:]
    warped_tensor_full_resolution = crop_torch_batch(warped_tensor_full_resolution, (H, W))
    aligned_tensor_after_CC_classic_minSAD = warped_tensor_full_resolution
    ####################################################################################

    # aligned_tensor_final = aligned_frames_ECC_tensor
    aligned_tensor_final = aligned_tensor_after_CC_classic_minSAD
    return input_tensor, aligned_tensor, aligned_tensor_final



def Align_Palantir_CC_Then_minSADReference(input_tensor, reference_tensor, flag_plot=False):
    ### Cross Correlation initial correction: ###
    # (1). compare to reference (center) frame:
    T, C, H, W = input_tensor.shape

    # shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(input_tensor, input_tensor[:,T//2:T//2+1], None, None)
    aligned_tensor, shifts_h, shifts_w, CC_matrix = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                         reference_matrix=reference_tensor,
                                                                                         matrix_fft=None,
                                                                                         normalize_over_matrix=False,
                                                                                         warp_method='bilinear',
                                                                                         crop_warped_matrix=False)
    max_shift_H = shifts_h.abs().max()
    max_shift_W = shifts_w.abs().max()
    new_H = H - (max_shift_H.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_W.abs().ceil().int().cpu().numpy() * 2 + 5)

    ### Shift input tensor to make aligned tensor: ###
    input_tensor = input_tensor.cpu()
    torch.cuda.empty_cache()

    ### Crop both original and aligned to be the same size and correct for size disparities (TODO: will be corrected to be seemless): ###
    aligned_tensor = crop_torch_batch(aligned_tensor, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])

    ### Present results after cross correlation: ###
    if flag_plot:
        concat_tensor = torch.cat([input_tensor.cpu(), aligned_tensor.cpu()], dim=-1)
        imshow_torch_video(input_tensor, FPS=50)
        imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(concat_tensor, FPS=50)
        imshow_torch_video(input_tensor - input_tensor[T // 2:T // 2 + 1], FPS=50)
        imshow_torch_video(aligned_tensor - aligned_tensor[T // 2:T // 2 + 1], FPS=50)
        del concat_tensor

    ### Free up GPU memory: ###
    input_tensor = input_tensor.to('cpu')
    del CC_matrix
    torch.cuda.empty_cache()

    # (2). min-SAD Reference Comparison:
    binning_size = 4
    aligned_tensor_binned = fast_binning_2D_overap_flexible_AvgPool2d(aligned_tensor, (binning_size, binning_size), (0, 0))
    T, C, H, W = aligned_tensor.shape
    minSAD = minSAD_transforms_layer(1, T, H, W)
    warped_tensor, shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
        minSAD.align(matrix=aligned_tensor_binned,
                     reference_matrix=aligned_tensor_binned[T // 2:T // 2 + 1],
                     shift_h_vec=[-2, -1, 0, 1, 2],
                     shift_w_vec=[-2, -1, 0, 1, 2],
                     rotation_vec=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                     scale_vec=[1],
                     warp_method='bicubic',
                     return_shifts=True)
    H, W = aligned_tensor_binned.shape[-2:]
    warped_tensor = crop_torch_batch(warped_tensor, (H, W)).cpu()
    aligned_tensor_binned = aligned_tensor_binned.cpu()

    if flag_plot:
        plot_torch(shifts_h);
        plt.show()
        plot_torch(shifts_w);
        plt.show()
        plot_torch(rotational_shifts);
        plt.show()
        sad_torch_list_2 = (warped_tensor - aligned_tensor_binned[T // 2:T // 2 + 1]).abs().mean([-1, -2]).squeeze()
        sad_torch_list_1 = (aligned_tensor_binned - aligned_tensor_binned[T // 2:T // 2 + 1]).abs().mean([-1, -2]).squeeze()
        string_list = []
        for i in np.arange(len(sad_torch_list_1)):
            string_list.append(decimal_notation(sad_torch_list_1[i].item(), 2) + ' , ' + decimal_notation(sad_torch_list_2[i].item(), 2))
        concat_tensor = torch.cat([aligned_tensor_binned, warped_tensor], -1)
        concat_aligned_tensor = torch.cat([aligned_tensor_binned, aligned_tensor_binned], -1)
        imshow_torch_video(concat_tensor, FPS=25, frame_stride=5, video_title_list=string_list)
        imshow_torch_video(concat_tensor - concat_aligned_tensor[T // 2:T // 2 + 1], FPS=25, frame_stride=2, video_title_list=string_list)

    ### Warp original resolution instead of binned resolution: ###
    shifts_h_full_resolution = shifts_h * binning_size
    shifts_w_full_resolution = shifts_w * binning_size
    rotational_shifts_full_resolution = rotational_shifts * binning_size
    scale_shifts_full_resolution = scale_shifts * binning_size
    warped_tensor_full_resolution = affine_transform_interpolated(aligned_tensor.unsqueeze(0),
                                                                  construct_tensor(-shifts_h_full_resolution),
                                                                  construct_tensor(-shifts_w_full_resolution),
                                                                  construct_tensor(-rotational_shifts_full_resolution * np.pi / 180),
                                                                  construct_tensor(1 / scale_shifts_full_resolution),
                                                                  warp_method='bilinear',
                                                                  expand=False)
    warped_tensor_full_resolution = warped_tensor_full_resolution.squeeze(0)
    H, W = aligned_tensor.shape[-2:]
    warped_tensor_full_resolution = crop_torch_batch(warped_tensor_full_resolution, (H, W))




def Align_Palantir_CC_Then_minSADPairWise(input_tensor, reference_tensor, flag_plot=False):
    ### Cross Correlation initial correction: ###
    # (1). compare to reference (center) frame:
    T, C, H, W = input_tensor.shape

    # shift_H, shift_W, CC_tensor = classic_circular_cc_shifts_calc(input_tensor, input_tensor[:,T//2:T//2+1], None, None)
    aligned_tensor, shifts_h, shifts_w, CC_matrix = align_to_reference_frame_circular_cc(matrix=input_tensor,
                                                                                         reference_matrix=reference_tensor,
                                                                                         matrix_fft=None,
                                                                                         normalize_over_matrix=False,
                                                                                         warp_method='bilinear',
                                                                                         crop_warped_matrix=False)
    max_shift_H = shifts_h.abs().max()
    max_shift_W = shifts_w.abs().max()
    new_H = H - (max_shift_H.abs().ceil().int().cpu().numpy() * 2 + 5)
    new_W = W - (max_shift_W.abs().ceil().int().cpu().numpy() * 2 + 5)

    ### Shift input tensor to make aligned tensor: ###
    input_tensor = input_tensor.cpu()
    torch.cuda.empty_cache()

    ### Crop both original and aligned to be the same size and correct for size disparities (TODO: will be corrected to be seemless): ###
    aligned_tensor = crop_torch_batch(aligned_tensor, [new_H, new_W])
    input_tensor = crop_torch_batch(input_tensor, [new_H, new_W])

    ### Present results after cross correlation: ###
    if flag_plot:
        concat_tensor = torch.cat([input_tensor.cpu(), aligned_tensor.cpu()], dim=-1)
        imshow_torch_video(input_tensor, FPS=50)
        imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(concat_tensor, FPS=50)
        imshow_torch_video(input_tensor - input_tensor[T // 2:T // 2 + 1], FPS=50)
        imshow_torch_video(aligned_tensor - aligned_tensor[T // 2:T // 2 + 1], FPS=50)
        del concat_tensor

    ### Free up GPU memory: ###
    input_tensor = input_tensor.to('cpu')
    del CC_matrix
    torch.cuda.empty_cache()

    ### Try Pair-Wise Alignments: ###
    binning_size = 4
    aligned_tensor_binned = fast_binning_2D_overap_flexible_AvgPool2d(aligned_tensor, (binning_size, binning_size), (0, 0))
    T, C, H, W = aligned_tensor.shape
    minSAD = minSAD_transforms_layer(1, T - 1, H, W)
    warped_tensor, shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
        minSAD.align(matrix=aligned_tensor_binned[1:],
                     reference_matrix=aligned_tensor_binned[0:-1],
                     shift_h_vec=[-1, 0, 1],
                     shift_w_vec=[-1, 0, 1],
                     rotation_vec=[-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
                     scale_vec=[1],
                     warp_method='bicubic',
                     return_shifts=True)
    warped_tensor_bias, shifts_h_bias, shifts_w_bias, rotational_shifts_bias, scale_shifts_bias, min_sad_matrix_bias = \
        minSAD.align(matrix=aligned_tensor_binned[1:],
                     reference_matrix=aligned_tensor_binned[1:],
                     shift_h_vec=[-1, 0, 1],
                     shift_w_vec=[-1, 0, 1],
                     rotation_vec=[-0.1, 0, 0.1],
                     scale_vec=[1],
                     warp_method='bicubic',
                     return_shifts=True)
    shifts_h = shifts_h + shifts_h_bias
    shifts_w = shifts_w + shifts_w_bias
    rotational_shifts = rotational_shifts + rotational_shifts_bias
    shifts_h = shifts_h.cumsum(-1)
    shifts_w = shifts_w.cumsum(-1)
    rotational_shifts = rotational_shifts.cumsum(-1)
    shifts_h_full_resolution = shifts_h * binning_size
    shifts_w_full_resolution = shifts_w * binning_size
    rotational_shifts_full_resolution = rotational_shifts * binning_size
    scale_shifts_full_resolution = (1 - scale_shifts) * binning_size + 1

    warped_tensor_binned_resolution = affine_transform_interpolated(aligned_tensor_binned[1:].unsqueeze(0),
                                                                    construct_tensor(shifts_h),
                                                                    construct_tensor(shifts_w),
                                                                    construct_tensor(rotational_shifts * np.pi / 180),
                                                                    construct_tensor(1 / scale_shifts),
                                                                    warp_method='bilinear',
                                                                    expand=False).squeeze(0)

    warped_tensor_full_resolution = affine_transform_interpolated(aligned_tensor[1:].unsqueeze(0),
                                                                  construct_tensor(shifts_h_full_resolution),
                                                                  construct_tensor(shifts_w_full_resolution),
                                                                  construct_tensor(rotational_shifts_full_resolution * np.pi / 180),
                                                                  construct_tensor(1 / scale_shifts_full_resolution),
                                                                  warp_method='bilinear',
                                                                  expand=False).squeeze(0)

    H, W = aligned_tensor_binned.shape[-2:]
    warped_tensor_binned_resolution = crop_torch_batch(warped_tensor_binned_resolution, [H, W])
    aligned_tensor_binned = crop_torch_batch(aligned_tensor_binned[1:], [H, W])
    warped_tensor_binned_resolution = warped_tensor_binned_resolution.cuda()
    concat_tensor_min_sad = torch.cat([aligned_tensor_binned, warped_tensor_binned_resolution], dim=-1)

    ### Plot: ###
    if flag_plot:
        plot_torch(rotational_shifts);
        plt.show()
        plot_torch(rotational_shifts - rotational_shifts[len(rotational_shifts) // 2]);
        plt.show()
        plot_torch(shifts_h);
        plt.show()
        plot_torch(shifts_w);
        plt.show()
        imshow_torch_video(warped_tensor, FPS=50)
        imshow_torch_video(aligned_tensor, FPS=50, frame_stride=5)
        imshow_torch_video(concat_tensor_min_sad, FPS=50)
        imshow_torch_video(BW2RGB(concat_tensor_min_sad - concat_tensor_min_sad[0]).abs()[1:] / 15, FPS=50)
        imshow_torch_video(warped_tensor - warped_tensor[T // 2:T // 2 + 1], FPS=50, frame_stride=5)


from torch.autograd import Variable as V
from torch import Tensor as T
import numpy as np
import torch.nn.functional as F

def VT(x): return V(T(x), requires_grad=False)

def three_conv(dx, dy, dz, dt, fac=1):
    # Factor for adding that single minus on the dt conv
    conv = torch.nn.Conv3d(1, 4, 2)
    conv.weight = torch.nn.Parameter(T(np.concatenate([dx, dy, dz, fac * dt], axis=0)))
    conv.bias = torch.nn.Parameter(T(np.array([0, 0, 0, 0])))
    return conv


def img_derivatives(img1, img2):
    ones = np.ones((2, 2, 2))
    dx = (0.25 * ones * np.array([-1, 1]))[None, None, ...]
    dy = (0.25 * ones * np.array([-1, 1])[:, None])[None, None, ...]
    dz = 0.25 * np.stack([-np.ones((2, 2)), np.ones((2, 2))])[None, None, ...]
    dt = ones[None, None, ...]

    conv1 = three_conv(dx, dy, dz, dt)
    conv2 = three_conv(dx, dy, dz, dt, fac=-1)
    res = 0.5 * (conv1(VT(img1[None, ...])) + conv2(VT(img2[None, ...])))[0]
    # Returns a 4,50,50,50 for the 4 derivatives including time
    return F.pad(res, (1, 0, 1, 0, 1, 0))


def LK_optical_flow(dimg, r=2):
    d = dimg.shape[-1]
    x = np.ones((1, 1, 2, 2, 2))
    calc = (dimg[None, 0:3, ...] * dimg[:, None, ...])
    conv_next = torch.nn.Conv3d(3, 3, 2)
    conv_next.weight = torch.nn.Parameter(T(x))
    conv_next.bias = torch.nn.Parameter(T(np.array([0])))

    sum_conv = torch.cat([conv_next(i[:, None, ...]) for i in torch.unbind(calc, 1)], 1)
    dim = sum_conv.shape[-1]

    a = sum_conv
    b = a.permute(2, 3, 4, 0, 1)
    c = b[..., :-1, :].contiguous().view(-1, 3, 3)
    d = b[..., -1, :].contiguous().view(-1, 3, 1)

    inv = torch.stack([mat.inverse() for mat in torch.unbind(c, 0)])
    out = torch.bmm(inv, d)
    out = out.transpose(0, 1).contiguous().view(3, dim, dim, dim)
    return out




class Gimbaless_Layer_Torch(nn.Module):

    # Initialize this with a module
    def __init__(self, block_size=32, overlap_size=16):
        super(Gimbaless_Layer_Torch, self).__init__()

        ### Prediction Grid Definition: ###
        self.prediction_block_size = int16(block_size / 1)  # for one global prediction for everything simply use W or H
        self.overlap_size = overlap_size  # 0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
        self.temporal_lowpass_number_of_steps = 9

        ### Ctypi Parameters: ###
        self.reg_step = .1
        self.filt_len = 11
        self.dif_ord = 2
        self.dif_len = 7
        self.CorW = 7
        self.dYExp = 0
        self.dXExp = 0
        self.reg_lst = np.arange(0, 0.5, 0.1)

        ### Define convn layers: ###
        self.torch_convn_layer = convn_layer_torch()

        ### create filter: ###
        self.params = EasyDict()
        self.params.dif_len = 9
        # spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((1,1,1,params.dif_len))  #Read this from file
        self.dict_from_matlab = scipy.io.loadmat(
            r'C:/Users/dudyk/Desktop\dudy_karl/spatial_lowpass_9tap.mat')  # Read this from file
        self.spatial_lowpass_before_temporal_derivative_filter_x = self.dict_from_matlab[
            'spatial_lowpass_before_temporal_derivative_filter'].flatten()
        self.spatial_lowpass_before_temporal_derivative_filter_x = torch_get_4D(
            torch.Tensor(self.spatial_lowpass_before_temporal_derivative_filter_x))
        self.spatial_lowpass_before_temporal_derivative_filter_y = self.spatial_lowpass_before_temporal_derivative_filter_x.permute(
            [0, 1, 3, 2])

        ### Preallocate filters: ###
        # (1). temporal derivative:
        self.temporal_derivative_filter = torch.Tensor(np.array([-1, 1]))
        self.temporal_derivative_filter = torch.reshape(self.temporal_derivative_filter, (1, 2, 1, 1))
        # (2). spatial derivative:
        self.grad_x_kernel = torch_get_4D(torch.Tensor(np.array([-1, 0, 1]) / 2), 'W')
        self.grad_y_kernel = self.grad_x_kernel.permute([0, 1, 3, 2])
        # (3). temporal averaging:
        self.temporal_averaging_filter_before_spatial_gradient = torch.ones(
            (1, self.temporal_lowpass_number_of_steps, 1, 1, 1)) / self.temporal_lowpass_number_of_steps

        self.spatial_lowpass_before_temporal_derivative_filter_fft_x = None

    def forward(self, input_tensor, reference_tensor=None, flag_pairwise_or_reference='reference'):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.spatial_lowpass_before_temporal_derivative_filter_fft_x is None:
            ### Get FFT Filters: ###
            self.spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(
                self.spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
            self.spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(
                self.spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])
            self.temporal_derivative_fft_filter = torch.fft.fftn(self.temporal_derivative_filter, s=T, dim=[1])
            self.grad_x_filter_fft = torch.fft.fftn(self.grad_x_kernel, s=W, dim=[-1])
            self.grad_y_filter_fft = torch.fft.fftn(self.grad_y_kernel, s=H, dim=[-2])
            self.temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(
                self.temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

        ######################################################################################################
        ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###\
        # (1).
        px = self.torch_convn_layer.forward(input_tensor, self.grad_x_kernel.flatten(), -1)
        py = self.torch_convn_layer.forward(input_tensor, self.grad_y_kernel.flatten(), -2)
        if flag_pairwise_or_reference == 'pairwise':
            # #(*). Pair-Wise Shifts:
            px = px[:, 0:T - 1]  # T-1 displacements to output
            py = py[:, 0:T - 1]
        else:
            # (*). Center Frame Reference Shifts:
            px = px
            py = py

        # ### Cut-Out Invalid Parts (Center-Crop): ###
        # # TODO: perhapse don't(!!!!) center crop now, but later. i can also hard-code this into the fast_binning function start and stop indices
        # start_index = np.int16(floor(self.params.dif_len / 2))
        # px = px[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        # py = py[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ######################################################################################################

        ######################################################################################################
        ### Calculate Temporal Derivative: ###
        ABtx = self.torch_convn_layer.forward(input_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
        ABty = self.torch_convn_layer.forward(input_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)
        if flag_pairwise_or_reference == 'pairwise':
            # (1). Pair-Wise Shifts:
            ABtx = ABtx[:, 1:T] - ABtx[:, 0:T - 1]
            ABty = ABty[:, 1:T] - ABty[:, 0:T - 1]
        elif flag_pairwise_or_reference == 'center':
            # (2). Center Tensor Reference Shifts:
            ABtx = ABtx - ABtx[:, T // 2:T // 2 + 1] #TODO: replace this with possible external reference tensor, but make sure it undergoes same filtering!!!!
            ABty = ABty - ABty[:, T // 2:T // 2 + 1]
        elif flag_pairwise_or_reference == 'reference':
            ABtx_reference = self.torch_convn_layer.forward(reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
            ABty_reference = self.torch_convn_layer.forward(reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)
            ABtx = ABtx - ABtx_reference
            ABty = ABty - ABty_reference

        # ### Cut-Out Invalid Parts (Center-Crop): ###
        # ABtx = ABtx[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        # ABty = ABty[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        #####################################################################################################

        ######################################################################################################
        ### Find Shift Between Frames: ###
        # (*) inv([A,B;C,D]) = 1/(AD-BC)*[D,-B;-C,A]
        ### Get elements of the first matrix in the equation we're solving: ###
        # Local Prediction:
        B2, T2, C2, H2, W2 = px.shape
        prediction_block_size = min(self.prediction_block_size, H2)
        prediction_block_size_tuple = to_tuple_of_certain_size(self.prediction_block_size, 2)
        overlap_block_size_tuple = to_tuple_of_certain_size(self.overlap_size, 2)
        pxpy = fast_binning_2D_AvgPool2d(px * py, prediction_block_size_tuple, overlap_block_size_tuple)
        px2 = fast_binning_2D_AvgPool2d(px ** 2, prediction_block_size_tuple, overlap_block_size_tuple)
        py2 = fast_binning_2D_AvgPool2d(py ** 2, prediction_block_size_tuple, overlap_block_size_tuple)

        ### Invert the above matrix explicitly in a hard coded fashion until i find
        ### a more elegant way to invert matrices in parallel: ###
        # P = [px2,pxpy; py2,pxpy] -> matrix we are explicitely inverting
        # inv_P = 1./(A.*D-B.*C) .* [D,-B;-C,A];
        common_factor = 1. / (px2 * py2 - pxpy * pxpy)
        inv_P_xx = common_factor * py2
        inv_P_xy = -common_factor * pxpy
        inv_P_yx = -common_factor * pxpy
        inv_P_yy = common_factor * px2

        ### Solve the needed equation explicitly: ###
        # d = inv_P * [squeeze(sum(ABtx.*px, [1,2])); squeeze(sum(ABty.*py,[1,2]))];
        # Local Prediction:
        A = (fast_binning_2D_AvgPool2d((ABtx * px), prediction_block_size_tuple, overlap_block_size_tuple))
        B = (fast_binning_2D_AvgPool2d((ABty * py), prediction_block_size_tuple, overlap_block_size_tuple))
        delta_x = inv_P_xx * A + inv_P_xy * B
        delta_y = inv_P_yx * A + inv_P_yy * B

        return (delta_x, delta_y)


def test_get_affine_flow_field_and_get_homography():
    # (*). Get Original And Warped Images:
    BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
    input_tensor = torch.load(os.path.join(BG_image_folder, 'palantir_BG.pt'))
    input_tensor = read_image_default_torch()*255
    shift_H = 10.5
    shift_W = 11.7
    rotation_degrees = 0.7
    input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0),
                                                        construct_tensor(shift_H),
                                                        construct_tensor(shift_W),
                                                        construct_tensor(rotation_degrees * np.pi / 180),
                                                        construct_tensor(1),
                                                        warp_method='bicubic',
                                                        expand=False).squeeze(0)
    flag_plot = False

    ### Center Crop and Clamp: ###
    input_tensor = crop_torch_batch(input_tensor, (1000,1000))
    input_tensor_warped = crop_torch_batch(input_tensor_warped, (1000,1000))
    input_tensor = input_tensor.clamp(0,255)
    input_tensor_warped = input_tensor_warped.clamp(0,255)

    ### Turn results to numpy: ###
    input_tensor_numpy = input_tensor.clamp(0, 255).cpu()[0, 0].numpy()
    input_tensor_warped_numpy = input_tensor_warped.clamp(0, 255).cpu()[0, 0].numpy()
    input_tensor_numpy = (input_tensor_numpy).astype(np.uint8)
    input_tensor_warped_numpy = (input_tensor_warped_numpy).astype(np.uint8)

    ### Save Results: ###
    folder_path = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data'
    save_image_torch(folder_path, 'Image_1.png', input_tensor)
    save_image_torch(folder_path, 'Image_2.png', input_tensor_warped)

    ### Get Flow Field And Pairs: ###
    H, W = input_tensor.shape[-2:]
    x_vec = torch.arange(W)
    y_vec = torch.arange(H)
    [Y, X] = torch.meshgrid(y_vec, x_vec)
    X_centered = X - W / 2
    Y_centered = Y - H / 2
    rotation_degress_tensor = torch.tensor([rotation_degrees]) * np.pi / 180
    R_meshgrid = (X_centered ** 2 + Y_centered ** 2) ** (1 / 2)
    theta_meshgrid = torch.atan(-Y_centered / X_centered)
    theta_meshgrid += (X_centered < 0).float() * np.pi
    theta_meshgrid += (X_centered >= 0).float() * (Y_centered > 0).float() * np.pi * 2
    theta_meshgrid[H // 2, W // 2] = 0
    x_shift_sign = 2 * ((Y_centered < 0).float() - 0.5)
    y_shift_sign = 2 * ((X_centered > 0).float() - 0.5)
    flow_field_X = R_meshgrid * torch.cos(theta_meshgrid + rotation_degress_tensor) - R_meshgrid * torch.cos(theta_meshgrid)
    flow_field_Y = R_meshgrid * torch.sin(theta_meshgrid + rotation_degress_tensor) - R_meshgrid * torch.sin(theta_meshgrid)
    flow_field_magnitude = torch.sqrt(flow_field_X ** 2 + flow_field_Y ** 2)
    X_new = torch.cos(rotation_degress_tensor) * X_centered - torch.sin(rotation_degress_tensor) * Y_centered
    Y_new = torch.sin(rotation_degress_tensor) * X_centered + torch.cos(rotation_degress_tensor) * Y_centered
    X_diff = X_new - X_centered
    Y_diff = Y_new - Y_centered

    ### Get Random Indices From The Flow-Field To Get Keypoint-Pairs: ###
    indices = torch.randperm(H * W)[0:1000]
    indices = torch.sort(indices).values
    keypoints_1_X = torch.flatten(X_centered)[indices]
    keypoints_1_Y = torch.flatten(Y_centered)[indices]
    keypoints_centers_1 = torch.cat([keypoints_1_X.unsqueeze(-1), keypoints_1_Y.unsqueeze(-1)], -1)
    keypoints_2_X = torch.flatten(X_new)[indices]
    keypoints_2_Y = torch.flatten(Y_new)[indices]
    keypoints_centers_2 = torch.cat([keypoints_2_X.unsqueeze(-1), keypoints_2_Y.unsqueeze(-1)], -1)
    H_matrix = kornia.geometry.homography.find_homography_dlt(keypoints_centers_1.unsqueeze(0).float(),
                                                              keypoints_centers_2.unsqueeze(0).float(),
                                                              weights=None)
    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)


def test_use_optical_flow_to_get_homography():
    # (*). Get Original And Warped Images:
    BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
    input_tensor = torch.load(os.path.join(BG_image_folder, 'palantir_BG.pt'))
    shift_H = 0
    shift_W = 0
    rotation_degrees = 0.7
    input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0),
                                                        construct_tensor(shift_H),
                                                        construct_tensor(shift_W),
                                                        construct_tensor(rotation_degrees * np.pi / 180),
                                                        construct_tensor(1),
                                                        warp_method='bicubic',
                                                        expand=False).squeeze(0)
    input_tensor_numpy = input_tensor.clamp(0, 255).cpu()[0, 0].numpy()
    input_tensor_warped_numpy = input_tensor_warped.clamp(0, 255).cpu()[0, 0].numpy()
    input_tensor_numpy = (input_tensor_numpy).astype(np.uint8)
    input_tensor_warped_numpy = (input_tensor_warped_numpy).astype(np.uint8)
    flag_plot = False

    # ### Get TVNet Instance: ###
    # # from RDND_proper.models.pytorch_tvnet.tvnet import *
    # from RDND_proper.models.tvnet_pytorch.model.net.tvnet import *
    # tvnet_dict = EasyDict()
    # tvnet_dict.zfactor = 0.5
    # tvnet_dict.max_nscale = 1
    # tvnet_dict.data_size = input_tensor.shape
    # tvnet_dict.tau = 0.25
    # tvnet_dict.lbda = 0.15
    # tvnet_dict.theta = 0.3
    # tvnet_dict.n_warps = 1
    # tvnet_dict.n_iters = 30
    # tvnet_layer = TVNet(tvnet_dict)
    # with torch.no_grad():
    #     u1, u2, rho = tvnet_layer.forward(input_tensor.cpu()/1, input_tensor_warped.cpu()/1)
    #     flow_magnitude = torch.sqrt(u1**2 + u2**2)
    #     imshow_torch(flow_magnitude)

    ### Gimbaless As Optical-Flow: ###
    input_tensor_gimbaless = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor, (4, 4), (0, 0)).unsqueeze(0)
    input_tensor_warped_gimbaless = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_warped, (4, 4), (0, 0)).unsqueeze(0)
    input_tensor_gimbaless = crop_tensor(input_tensor_gimbaless, (90, 250))
    input_tensor_warped_gimbaless = crop_tensor(input_tensor_warped_gimbaless, (90, 250))
    gimbaless_layer = Gimbaless_Layer_Torch(32, 31)
    delta_x, delta_y = gimbaless_layer.forward(input_tensor_gimbaless.cpu(), input_tensor_warped_gimbaless.cpu())
    flow_magnitude = torch.sqrt(delta_x ** 2 + delta_y ** 2)
    input_tensor_gimbaless = crop_tensor(input_tensor_gimbaless, (flow_magnitude.shape[-2], flow_magnitude.shape[-1]))
    # imshow_torch(flow_magnitude[0])

    ### Get Weights Using Canny Edge Detection: ###
    canny_magnitude, canny_threshold = kornia.filters.canny(input_tensor_gimbaless.squeeze(0) / 255, 0.1, 0.12)
    ### Get Random Indices From The Flow-Field To Get Keypoint-Pairs: ###
    H, W = input_tensor_gimbaless.shape[-2:]
    x_vec = torch.arange(W)
    y_vec = torch.arange(H)
    [Y, X] = torch.meshgrid(y_vec, x_vec)
    X_centered = X - W / 2
    Y_centered = Y - H / 2
    indices = torch.randperm(H * W)[0:10000]
    indices = torch.sort(indices).values
    X_new = X_centered + delta_x
    Y_new = Y_centered + delta_y
    keypoints_1_X = torch.flatten(X_centered)[indices]
    keypoints_1_Y = torch.flatten(Y_centered)[indices]
    keypoints_centers_1 = torch.cat([keypoints_1_X.unsqueeze(-1), keypoints_1_Y.unsqueeze(-1)], -1)
    keypoints_2_X = torch.flatten(X_new)[indices]
    keypoints_2_Y = torch.flatten(Y_new)[indices]
    keypoints_centers_2 = torch.cat([keypoints_2_X.unsqueeze(-1), keypoints_2_Y.unsqueeze(-1)], -1)
    weights = torch.flatten(canny_magnitude)[indices]
    H_matrix = kornia.geometry.homography.find_homography_dlt(keypoints_centers_1.unsqueeze(0).float(),
                                                              keypoints_centers_2.unsqueeze(0).float(),
                                                              weights=weights.unsqueeze(0).cpu())
    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)


def get_homography_from_optical_flow(input_tensor_1, input_tensor_2, gimbaless_block_size=32, gimbaless_overlap=31, input_meshgrid=None):
    ### Gimbaless As Optical-Flow: ###

    gimbaless_layer = Gimbaless_Layer_Torch(gimbaless_block_size, gimbaless_overlap)
    delta_x, delta_y = gimbaless_layer.forward(input_tensor_1, input_tensor_2)
    # flow_magnitude = torch.sqrt(delta_x ** 2 + delta_y ** 2)
    # input_tensor_gimbaless = crop_tensor(input_tensor_gimbaless, (flow_magnitude.shape[-2], flow_magnitude.shape[-1]))

    ### Equalize delta_x and delta_y for now: ###


    ### Get Random Indices From The Flow-Field To Get Keypoint-Pairs: ###
    #TODO: needs to adjust to batch processing, when i flatten i don't take into account batch processing, so i need to flatten only on the spatial domain
    H, W = input_tensor_2.shape[-2:]
    # x_vec = torch.arange(W)
    # y_vec = torch.arange(H)
    # [Y, X] = torch.meshgrid(y_vec, x_vec)
    Y,X = input_meshgrid
    X_centered = X - W / 2
    Y_centered = Y - H / 2
    indices = torch.randperm(H * W)[0:10000]
    indices = torch.sort(indices).values
    X_new = X_centered + delta_x
    Y_new = Y_centered + delta_y
    keypoints_1_X = torch.flatten(X_centered)[indices]
    keypoints_1_Y = torch.flatten(Y_centered)[indices]
    keypoints_centers_1 = torch.cat([keypoints_1_X.unsqueeze(-1), keypoints_1_Y.unsqueeze(-1)], -1)
    keypoints_2_X = torch.flatten(X_new)[indices]
    keypoints_2_Y = torch.flatten(Y_new)[indices]
    keypoints_centers_2 = torch.cat([keypoints_2_X.unsqueeze(-1), keypoints_2_Y.unsqueeze(-1)], -1)
    H_matrix = kornia.geometry.homography.find_homography_dlt(keypoints_centers_1.unsqueeze(0).float(),
                                                              keypoints_centers_2.unsqueeze(0).float(),
                                                              weights=None)
    shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
        affine_parameters_from_homography_matrix(H_matrix)

    return shift_H, shift_W, rotation_rads, scale, rotation_degrees



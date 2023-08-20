import cv2.xfeatures2d
import torch

from RapidBase.import_all import *
import kornia

from RapidBase.Anvil.alignments_layers import *
from RapidBase.Anvil.alignments import classic_circular_cc_shifts_calc, align_to_center_frame_circular_cc
import kornia

from RapidBase.Anvil.import_all import *
from kornia_moons.feature import *

#################################
# ### Check speeds for stuff: ###
#################################

### Keypoint detection and Homographic matrix fit: ###
# detected_keypints_1 = torch.randn(500,512*100,2).cuda()
# detected_keypints_2 = torch.randn(500,512*100,2).cuda()
# NUM_WARMUP_ITERS = 2
# for _ in range(NUM_WARMUP_ITERS):
#     #################################################################
#     for i in np.arange(1):
#         H_matrix = kornia.geometry.homography.find_homography_dlt(detected_keypints_1,
#                                                                   detected_keypints_2,
#                                                                   weights=None)
#     #################################################################
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True, blocking=True)
# finish = torch.cuda.Event(enable_timing=True, blocking=True)
# start.record()
# #################################################################
# for i in np.arange(1):
#     H_matrix = kornia.geometry.homography.find_homography_dlt(detected_keypints_1,
#                                                               detected_keypints_2,
#                                                               weights=None)
# #################################################################
# torch.cuda.synchronize()
# finish.record()
# total_time = start.elapsed_time(finish)
# print(f"Total time to execute = {total_time} ms")



# ### Check speeds for stuff: ###
# BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
# input_tensor = torch.load(os.path.join(BG_image_folder, 'palantir_BG.pt'))
# shift_H = [0]
# shift_W = [0]
# rotation_degrees = [0.3]
# # rotation_degrees = [0.3]*5
# scale = [1]
# B = 10
# total_B = 500
# T = total_B//B
# input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0).repeat(len(shift_H),1,1,1,1),
#                                                       construct_tensor(shift_H),
#                                                       construct_tensor(shift_W),
#                                                       construct_tensor(rotation_degrees) * np.pi / 180,
#                                                       construct_tensor(scale),
#                                                       warp_method='bicubic',
#                                                       expand=False).squeeze(1)
# input_tensor_numpy = input_tensor.clamp(0,255).cpu()[0,0].numpy()
# input_tensor_warped_numpy = input_tensor_warped.clamp(0,255).cpu()[0,0].numpy()
# input_tensor_numpy = (input_tensor_numpy).astype(np.uint8)
# input_tensor_warped_numpy = (input_tensor_warped_numpy).astype(np.uint8)
# flag_plot = False
#
# input_tensor_full = input_tensor.repeat(1, 1, 1, 1)
# input_tensor_warped_full = input_tensor_warped.repeat(B, 1, 1, 1)
# input_tensor_warped_full = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_warped_full, (2, 2), (0, 0))
# input_tensor_full = fast_binning_2D_overap_flexible_AvgPool2d(input_tensor_full, (2, 2), (0, 0))
#
# NUM_WARMUP_ITERS = 2
# for _ in range(NUM_WARMUP_ITERS):
#     #################################################################
#     ### Feature Align: ###
#     for i in np.arange(T):
#         print(i)
#         rotation_degrees, detected_laf_1, detected_laf_2, \
#         descriptors_distance, \
#         detected_keypoints_centers_1, detected_keypoints_centers_2, \
#         local_descriptors_1, local_descriptors_2, \
#         response_function_for_laf_1, response_function_for_laf_2 = \
#             FeaturesAlign_getFeatures_AndHomography_1_Batch_SNN(input_tensor_warped_full, input_tensor_full, number_of_features=1500, snn_matcher_threshold=0.8, input_mask=None)
#     #################################################################
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True, blocking=True)
# finish = torch.cuda.Event(enable_timing=True, blocking=True)
# start.record()
# #################################################################
# ### Feature Align: ###
# for i in np.arange(T):
#     print(i)
#     rotation_degrees, detected_laf_1, detected_laf_2, \
#     descriptors_distance, \
#     detected_keypoints_centers_1, detected_keypoints_centers_2, \
#     local_descriptors_1, local_descriptors_2, \
#     response_function_for_laf_1, response_function_for_laf_2 = \
#         FeaturesAlign_getFeatures_AndHomography_1_Batch_SNN(input_tensor_warped_full, input_tensor_full, number_of_features=1500, snn_matcher_threshold=0.8, input_mask=None)
# #################################################################
# torch.cuda.synchronize()
# finish.record()
# total_time = start.elapsed_time(finish)
# print(f"Total time to execute = {total_time} ms")




# ### Check speeds for stuff: ###
# from RapidBase.Anvil.import_all import *
# B = 100
# total_B = 500
# T = total_B//B
# input_tensor_1 = read_image_default_torch().cuda()
# shift_H = [0]
# shift_W = [0]
# rotation_degrees = [0.05]
# scale = [1]
# input_tensor_2 = affine_transform_interpolated(input_tensor_1.unsqueeze(0).repeat(len(shift_H),1,1,1,1),
#                                                       construct_tensor(shift_H),
#                                                       construct_tensor(shift_W),
#                                                       construct_tensor(rotation_degrees) * np.pi / 180,
#                                                       construct_tensor(scale),
#                                                       warp_method='bicubic',
#                                                       expand=False).squeeze(1)
# binning_size = 4
# T1, C1, H1, W1 = input_tensor_1.shape
# minSAD = minSAD_transforms_layer(1, T1, H1, W1)
# input_tensor_1 = crop_tensor(input_tensor_1, (512,512))
# input_tensor_2 = crop_tensor(input_tensor_2, (512,512))
# input_tensor_1 = RGB2BW(input_tensor_1)
# input_tensor_2 = RGB2BW(input_tensor_2)
# input_tensor_1 = input_tensor_1.repeat(B,1,1,1)
#
# NUM_WARMUP_ITERS = 2
# for _ in range(NUM_WARMUP_ITERS):
#     #################################################################
#     for i in np.arange(T):
#         warped_tensor, shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
#             minSAD.align(matrix=input_tensor_1,
#                          reference_matrix=input_tensor_2,
#                          shift_h_vec=[0],
#                          shift_w_vec=[0],
#                          rotation_vec=[-0.2, 0, 0.2],
#                          scale_vec=[1],
#                          warp_method='bilinear',
#                          warp_matrix=False,
#                          return_shifts=True)
#     #################################################################
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True, blocking=True)
# finish = torch.cuda.Event(enable_timing=True, blocking=True)
# start.record()
# #################################################################
# for i in np.arange(T):
#     warped_tensor, shifts_h, shifts_w, rotational_shifts, scale_shifts, min_sad_matrix = \
#         minSAD.align(matrix=input_tensor_1,
#                      reference_matrix=input_tensor_2,
#                      shift_h_vec=[0],
#                      shift_w_vec=[0],
#                      rotation_vec=[-0.2, 0, 0.2],
#                      scale_vec=[1],
#                      warp_method='bilinear',
#                      warp_matrix=False,
#                      return_shifts=True)
# #################################################################
# torch.cuda.synchronize()
# finish.record()
# total_time = start.elapsed_time(finish)
# print(f"Total time to execute = {total_time} ms")





# #(*). Get Original And Warped Images:
# BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
# input_tensor = torch.load(os.path.join(BG_image_folder, 'palantir_BG.pt'))
# shift_H = [0]
# shift_W = [0]
# rotation_degrees = [0.2]
# # rotation_degrees = [0.3]*5
# scale = [1]
# B = 100
# total_B = 500
# T = total_B//B
# input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0).repeat(len(shift_H),1,1,1,1),
#                                                       construct_tensor(shift_H),
#                                                       construct_tensor(shift_W),
#                                                       construct_tensor(rotation_degrees) * np.pi / 180,
#                                                       construct_tensor(scale),
#                                                       warp_method='bicubic',
#                                                       expand=False).squeeze(1)
# input_tensor_numpy = input_tensor.clamp(0,255).cpu()[0,0].numpy()
# input_tensor_warped_numpy = input_tensor_warped.clamp(0,255).cpu()[0,0].numpy()
# input_tensor_numpy = (input_tensor_numpy).astype(np.uint8)
# input_tensor_warped_numpy = (input_tensor_warped_numpy).astype(np.uint8)
# flag_plot = False
#
# ### Gimbaless to homographic matrix: ###
# binning_size = 4
# binning_size_tuple = (binning_size, binning_size)
# overlap_tuple = (binning_size-1, binning_size-1)
# input_tensor_gimbaless = fast_binning_2D_AvgPool2d(input_tensor, binning_size_tuple, overlap_tuple)
# input_tensor_warped_gimbaless = fast_binning_2D_AvgPool2d(input_tensor_warped, binning_size_tuple, overlap_tuple)
#
# H, W = input_tensor_gimbaless.shape[-2:]
# x_vec = torch.arange(W)
# y_vec = torch.arange(H)
# [Y, X] = torch.meshgrid(y_vec, x_vec)
# Y = Y.cuda()
# X = X.cuda()
# input_meshgrid = (Y,X)
#
# input_tensor_gimbaless = input_tensor_gimbaless.repeat(B,1,1,1)
# input_tensor_warped_gimbaless = input_tensor_warped_gimbaless.repeat(B,1,1,1)
#
# NUM_WARMUP_ITERS = 2
# for _ in range(NUM_WARMUP_ITERS):
#     #################################################################
#     for i in np.arange(T):
#         shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
#             get_homography_from_optical_flow(input_tensor_gimbaless.unsqueeze(1),
#                                              input_tensor_warped_gimbaless.unsqueeze(1),
#                                              gimbaless_block_size=32,
#                                              gimbaless_overlap=31,
#                                              input_meshgrid=input_meshgrid)
#     #################################################################
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True, blocking=True)
# finish = torch.cuda.Event(enable_timing=True, blocking=True)
# start.record()
# #################################################################
# for i in np.arange(T):
#     shift_H, shift_W, rotation_rads, scale, rotation_degrees = \
#         get_homography_from_optical_flow(input_tensor_gimbaless.unsqueeze(0),
#                                          input_tensor_warped_gimbaless.unsqueeze(0),
#                                          gimbaless_block_size=32,
#                                          gimbaless_overlap=31,
#                                          input_meshgrid=input_meshgrid)
# #################################################################
# torch.cuda.synchronize()
# finish.record()
# total_time = start.elapsed_time(finish)
# print(f"Total time to execute = {total_time} ms")


##############################################################################################################################################################################
### Test Kornia/Pytorch Feature Detectors: ###
#####################
### Kornia Moons: ###
#####################
# Match keypoints using kornia, find homography using OpenCV, and draw matches using kornia-moons:
#(*). Get Original And Warped Images:
BG_image_folder = '/home/mafat/PycharmProjects/TNR_Results'
input_tensor = torch.load(os.path.join(BG_image_folder, 'palantir_BG.pt'))
shift_H = [0]
shift_W = [0]
rotation_degrees = [0.2]
# rotation_degrees = [0.3]*5
scale = [1]
B = 100
total_B = 500
T = total_B//B
input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(0).repeat(len(shift_H),1,1,1,1),
                                                      construct_tensor(shift_H),
                                                      construct_tensor(shift_W),
                                                      construct_tensor(rotation_degrees) * np.pi / 180,
                                                      construct_tensor(scale),
                                                      warp_method='bicubic',
                                                      expand=False).squeeze(1)
input_tensor_numpy = input_tensor.clamp(0,255).cpu()[0,0].numpy()
input_tensor_warped_numpy = input_tensor_warped.clamp(0,255).cpu()[0,0].numpy()
input_tensor_numpy = (input_tensor_numpy).astype(np.uint8)
input_tensor_warped_numpy = (input_tensor_warped_numpy).astype(np.uint8)
flag_plot = False






# ### Feature Align: ###
# rotation_degrees, detected_laf_1, detected_laf_2, \
# descriptors_distance, \
# detected_keypoints_centers_1, detected_keypoints_centers_2, \
# local_descriptors_1, local_descriptors_2, \
# response_function_for_laf_1, response_function_for_laf_2 = \
#     FeaturesAlign_getFeatures_AndHomography_1(input_tensor_warped[0:1], input_tensor, number_of_features=500, snn_matcher_threshold=0.8, input_mask=None)

#################################################################
### Feature Align: ###
input_tensor_full = input_tensor.repeat(1,1,1,1)
input_tensor_warped_full = input_tensor_warped.repeat(5,1,1,1)
input_tensor_warped_full = fast_binning_2D_AvgPool2d(input_tensor_warped_full, (2,2), (0,0))
input_tensor_full = fast_binning_2D_AvgPool2d(input_tensor_full, (2,2), (0,0))
rotation_degrees, detected_laf_1, detected_laf_2, \
descriptors_distance, \
detected_keypoints_centers_1, detected_keypoints_centers_2, \
local_descriptors_1, local_descriptors_2, \
response_function_for_laf_1, response_function_for_laf_2 = \
    FeaturesAlign_getFeatures_AndHomography_1_Batch_SNN(input_tensor_warped_full, input_tensor_full, number_of_features=1500, snn_matcher_threshold=0.8, input_mask=None)
#################################################################



#################################################################



# #################################################################
# #(*). Detect keypoints and descriptions using OpenCV (in open-cv, keypoints are object):
# det = cv2.ORB_create(500)
# kps1_from_OpenCV, descs1_from_OpenCV = det.detectAndCompute(input_tensor_numpy, None)
# kps2_from_OpenCV, descs2_from_OpenCV = det.detectAndCompute(input_tensor_warped_numpy, None)
#
# #(*). Plot keypoints using OpenCV function:
# out_img = cv2.drawKeypoints(input_tensor_numpy, kps1_from_OpenCV, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# if flag_plot:
#     plt.imshow(out_img)
#
# #(*). convert keypoints to lafs:  #TODO: check if this can accept tensors as well!, i still don't see anything in kornia to give me lafs from keypoints/descriptors
# lafs1_from_OpenCV, r1 = laf_from_opencv_kpts(kps1_from_OpenCV, 1.0, with_resp=True)
# lafs2_from_OpenCV, r2 = laf_from_opencv_kpts(kps2_from_OpenCV, 1.0, with_resp=True)
# if flag_plot:
#     visualize_LAF(K.image_to_tensor(input_tensor_numpy, False), lafs1_from_OpenCV, 0, 'y', figsize=(8,6))
#     visualize_LAF(K.image_to_tensor(input_tensor_warped_numpy, False), lafs2_from_OpenCV, 0, 'y', figsize=(8,6))
#
# #(*). Match:
# match_dists_DescriptionOpenCV_MatchingKornia, \
# match_idxs_DescriptionOpenCV_MatchingKornia =\
#     K.feature.match_snn(torch.from_numpy(descs1_from_OpenCV).float(),
#                         torch.from_numpy(descs2_from_OpenCV).float(), 0.5)
# #################################################################
#
#
# #####################################################################################################
# #(*). Detect keypoints and descriptions and lafs using kornia --> Match Descriptors Using Kornia: ###
# #####################################################################################################
# #(1).
# SIFT_detector_and_descriptor = kornia.feature.SIFTFeature(num_features=500, upright=False, rootsift=True, device=input_tensor.device)
# detected_laf_1, response_function_for_laf_1, local_descriptors_1 = SIFT_detector_and_descriptor(input_tensor, mask=None)
# detected_laf_2, response_function_for_laf_2, local_descriptors_2 = SIFT_detector_and_descriptor(input_tensor_warped, mask=None)
# detected_keypoint_coordinates_XY_1 = detected_laf_1[:,:,:,2:3].squeeze()
# detected_keypoint_coordinates_XY_2 = detected_laf_2[:,:,:,2:3].squeeze()
# #(2).
# # GFTT_Hardnet_detector_and_descriptor = kornia.feature.GFTTAffNetHardNet(num_features=500, upright=False, device=input_tensor.device)
# # detected_laf, response_function_for_laf, local_descriptors = GFTT_Hardnet_detector_and_descriptor.forward(input_tensor, mask=None)
# # #(3).
# # KeyNet_AffNet_detector_and_descriptor = kornia.feature.KeyNetAffNetHardNet(num_features=500, upright=False, device=input_tensor.device)
# # detected_laf, response_function_for_laf, local_descriptors = KeyNet_AffNet_detector_and_descriptor(input_tensor, mask=None)
# # #(4).
# # KeyNet_HardNet_detector_and_descriptor = kornia.feature.KeyNetHardNet(num_features=500, upright=False, device=input_tensor.device)
# # detected_laf, response_function_for_laf, local_descriptors = KeyNet_HardNet_detector_and_descriptor(input_tensor, mask=None)
#
# ### Match Descriptors Using Kornia: ###
# descriptors_distance, matching_indices = kornia.feature.match_snn(local_descriptors_1.squeeze(0), local_descriptors_2.squeeze(0), th=0.8, dm=None)
# matching_indices_1 = matching_indices[:, 0]
# matching_indices_2 = matching_indices[:, 1]
# detected_keypoints_centers_1 = detected_laf_1[:, matching_indices_1, :, 2]
# detected_keypoints_centers_2 = detected_laf_2[:, matching_indices_2, :, 2]
# #################################################################
#
#
# #########################################################
# ### Feature Detector + Descriptor + Matcher Wrappers: ###
# #########################################################
# # Module, which finds correspondences between two images based on local features.
# feature_detector_and_descriptor_module = kornia.feature.GFTTAffNetHardNet(1000).to(input_tensor.device)
# # feature_detector_and_descriptor_module = kornia.feature.SIFTFeature(num_features=500, upright=False, rootsift=True, device=input_tensor.device)
# descriptor_matcher_module = kornia.feature.DescriptorMatcher('snn', 0.8).to(input_tensor.device)
# gftt_hardnet_detector_plus_matcher = kornia.feature.LocalFeatureMatcher(local_feature=feature_detector_and_descriptor_module,
#                                                                         matcher=descriptor_matcher_module)
# # input_dict_to_matcher = EasyDict()
# # input_dict_to_matcher.image0 = input_tensor
# # input_dict_to_matcher.image1 = input_tensor_warped
# input_dict_to_matcher = {'image0': input_tensor/255,
#                          'image1': input_tensor_warped/255}
# with torch.no_grad():
#     output_dict = gftt_hardnet_detector_plus_matcher.forward(input_dict_to_matcher)
#     keypoints_coords_1_KorniaWrapper = output_dict['keypoints0']
#     keypoints_coords_2_KorniaWrapper = output_dict['keypoints1']
#     lafs_1_KorniaWrapper = output_dict['lafs0']
#     lafs_2_KorniaWrapper = output_dict['lafs1']
#     confidence_KorniaWrapper = output_dict['confidence']
#     batch_indexes_KorniaWrapper = output_dict['batch_indexes']
#
#     ### Put condition on confidence if wanted: ###
#     matching_indices_logical_mask = confidence_KorniaWrapper > 0.3
#     lafs_1_KorniaWrapper_valid = lafs_1_KorniaWrapper[:, matching_indices_logical_mask, :, :]
#     lafs_2_KorniaWrapper_valid = lafs_2_KorniaWrapper[:, matching_indices_logical_mask, :, :]
#     arange_indices_tensor = torch.arange(0, lafs_1_KorniaWrapper.shape[1])
#     valid_indices = arange_indices_tensor[matching_indices_logical_mask].long()
#
#     ### Get Keypoints Centers From LAFs: ###
#     detected_keypoints_centers_1 = lafs_1_KorniaWrapper[:, valid_indices, :, 2]
#     detected_keypoints_centers_2 = lafs_2_KorniaWrapper[:, valid_indices, :, 2]
#     confidence_KorniaWrapper = confidence_KorniaWrapper[valid_indices]
# #########################################################
#
#
# #################################################################
# #(*). Find Homography Using OpenCV (RANSAC):
# detected_keypoints_centers_1_to_CV2_findHomography = detected_keypoints_centers_1.detach().cpu().numpy().reshape(-1,2)
# detected_keypoints_centers_2_to_CV2_findHomography = detected_keypoints_centers_2.detach().cpu().numpy().reshape(-1,2)
#
# H_OpenCV, mask = cv2.findHomography(detected_keypoints_centers_1_to_CV2_findHomography,
#                                     detected_keypoints_centers_2_to_CV2_findHomography,
#                                     ransacReprojThreshold=0.5)
# H_OpenCV_tensor = torch.tensor(H_OpenCV).unsqueeze(0)
# shift_H_OpenCV, shift_W_OpenCV, rotation_rads_OpenCV, scale_OpenCV, rotation_degrees_OpenCV = affine_parameters_from_homography_matrix(H_OpenCV_tensor)
#
#
# #(*). Find Homography Using Kornia Lease-Squares:
# #(*). Lease Squares DLT:
# H_matrix_dlt = kornia.geometry.homography.find_homography_dlt(detected_keypoints_centers_1,
#                                                               detected_keypoints_centers_2,
#                                                               weights=confidence_KorniaWrapper.unsqueeze(0)**2)
# shift_H_dlt, shift_W_dlt, rotation_rads_dlt, scale_dlt, rotation_degrees_dlt =\
#     affine_parameters_from_homography_matrix(H_matrix_dlt)
# #(*). Iterative ReWeighted DLT:
# H_matrix_iterative_dlt = kornia.geometry.homography.find_homography_dlt_iterated(detected_keypoints_centers_1,
#                                                                                  detected_keypoints_centers_2,
#                                                                                  weights=confidence_KorniaWrapper.unsqueeze(0),
#                                                                                  soft_inl_th=4.0,
#                                                                                  n_iter=5)
# shift_H_iterative_dlt, shift_W_iterative_dlt, rotation_rads_iterative_dlt, scale_iterative_dlt, rotation_degrees_iterative_dlt =\
#     affine_parameters_from_homography_matrix(H_matrix_iterative_dlt)
#
# #(*). Return transfer error in image 2 for correspondences given the homography matrix:
# L1_error_per_point_pair = kornia.geometry.homography.oneway_transfer_error(detected_keypoints_centers_1, detected_keypoints_centers_2, H_matrix_dlt, squared=True, eps=1e-08)
#
# #(*). Get rid of high error pairs:
# arange_indices = torch.arange(detected_keypoints_centers_1.shape[1])
# small_error_logical_mask = (L1_error_per_point_pair < 0.5)
# final_valid_indices = arange_indices[small_error_logical_mask.squeeze()]
# detected_keypoints_centers_1_final = detected_keypoints_centers_1[:,final_valid_indices]
# detected_keypoints_centers_2_final = detected_keypoints_centers_2[:,final_valid_indices]
# confidence_KorniaWrapper_final = confidence_KorniaWrapper[final_valid_indices]
# H_matrix_iterative_dlt_final = kornia.geometry.homography.find_homography_dlt(detected_keypoints_centers_1_final,
#                                                                                  detected_keypoints_centers_2_final,
#                                                                                  weights=confidence_KorniaWrapper_final.unsqueeze(0))
# shift_H_iterative_dlt_final, shift_W_iterative_dlt_final, rotation_rads_iterative_dlt_final, scale_iterative_dlt_final, rotation_degrees_iterative_dlt_final =\
#     affine_parameters_from_homography_matrix(H_matrix_iterative_dlt_final)
#
# ### Find Homography Using Kornia RANSAC: ###
# RANSAC_module = kornia.geometry.ransac.RANSAC(model_type='homography', inl_th=2.0, batch_size=2048, max_iter=10, confidence=0.99, max_lo_iters=5)
# H_matrix_RANSAC_on_keypoints, inlinear_mask = RANSAC_module.forward(detected_keypoints_centers_1_final.squeeze(0),
#                                                                     detected_keypoints_centers_2_final.squeeze(0))
#
#
#
#
# #(*). Draw matches on images:
# draw_LAF_matches(lafs_1_KorniaWrapper,
#                  lafs_2_KorniaWrapper,
#                  valid_indices[0:100].unsqueeze(0),
#                  input_tensor_numpy,
#                  input_tensor_warped_numpy,
#                  draw_dict={"inlier_color": (0.2, 1, 0.2),
#                                "tentative_color": (0.8, 0.8, 0),
#                                "feature_color": None,
#                               "vertical": True},
#                  H = H_OpenCV)
# ##############################################################################################################################################################################
#
#
# ##################
# ### DETECTORS: ###
# ##################
# #(1). Modules:
# detector_module = kornia.feature.BlobHessian(grads_mode='sobel')
# detector_module = kornia.feature.CornerGFTT(grads_mode='sobel')
# detector_module = kornia.feature.CornerHarris(k=0.04, grads_mode='sobel')
# detector_module = kornia.feature.BlobDoG()
# detector_module = kornia.feature.KeyNet(pretrained=True)
#
# #(2). Functionals:
# ### Compute the Shi-Tomasi cornerness function: ###
# gftt_response_tensor = kornia.feature.gftt_response(input_tensor, grads_mode='sobel', sigmas=None)
# ### Compute the Harris cornerness function: ###
# harris_response_tensor = kornia.feature.harris_response(input_tensor, k=0.04, grads_mode='sobel', sigmas=None)
# ### Compute the absolute of determinant of the Hessian matrix: ###
# hessian_response_tensor = kornia.feature.hessian_response(input_tensor, grads_mode='sobel', sigmas=None)
# ### Compute the Difference-of-Gaussian response: ###
# difference_of_gaussians_repsponse = kornia.feature.dog_response(input_tensor)
#
#
# ####################
# ### DESCRIPTORS: ###
# ####################
# ### Dense SIFT Descriptor: ###  (*). This is much more comfortable i think....don't know about the speed
# dense_SIFT_descriptor = kornia.feature.DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4, spatial_bin_size=4, rootsift=True, clipval=0.2, stride=1, padding=1)
# descriptors = dense_SIFT_descriptor(input_tensor)
#
# ### Regular SIFT Descriptor: ### (TODO; the input must be of shape [B,1,patch_size,patch_size]
# T,C,H,W = input_tensor.shape
# patch_size = (41,41)
# padding = kornia.contrib.compute_padding((H,W), patch_size)
# input_tensor_to_patches = kornia.contrib.extract_tensor_patches(input_tensor, window_size=patch_size, stride=patch_size, padding=padding)
# input_tensor_from_patches = kornia.contrib.combine_tensor_patches(input_tensor_to_patches, original_size=(H,W), window_size=patch_size, stride=patch_size, unpadding=padding)
# SIFT_descriptor = kornia.feature.SIFTDescriptor(patch_size=41, num_ang_bins=8, num_spatial_bins=4, rootsift=True, clipval=0.2)
# descriptors = SIFT_descriptor(input_tensor_to_patches.squeeze(0))
#
# ### Multiple Kernel Local Descriptors: ###
# mkd_descriptor = kornia.feature.MKDDescriptor(patch_size=32, kernel_type='concat', whitening='pcawt', training_set='liberty', output_dims=128)
# descriptors = mkd_descriptor(input_tensor)
# ### HardNet: ###
# hardnet_descriptor = kornia.feature.HardNet(pretrained=True)
# descriptors = hardnet_descriptor(input_tensor)
# ### HardNet8: ###
# hardnet_descriptor = kornia.feature.HardNet8(pretrained=True)
# descriptors = hardnet_descriptor(input_tensor)
# ### HyNet: ###
# hynet_descriptor = kornia.feature.HyNet(pretrained=False, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.3, eps_l2_norm=1e-10)[source]
# descriptors = hardnet_descriptor(input_tensor)
# ### TFeat: ###
# tfeat_descriptor = kornia.feature.TFeat(pretrained=False)
# descriptors = tfeat_descriptor(input_tensor)
# ### SOSNet: ###
# sos_descriptor = kornia.feature.SOSNet(pretrained=False)
# descriptors = sos_descriptor(input_tensor)
# ### LAF (Local Affine Features), the lafs i get from keypoints detector: ###
# laf_descriptor = kornia.feature.LAFDescriptor(patch_descriptor_module=None, patch_size=32, grayscale_descriptor=True)
# patch_descriptor = kornia.feature.DenseSIFTDescriptor(num_ang_bins=8, num_spatial_bins=4, spatial_bin_size=4, rootsift=True, clipval=0.2, stride=1, padding=1)
# kornia.feature.get_laf_descriptors(input_tensor, lafs, patch_descriptor, patch_size=32, grayscale_descriptor=True)
# descriptors = laf_descriptor.forward(input_tensor, lafs)
#
#
# ############################
# ### Descriptor Matching: ###
# ############################
# #(*). Function, which finds nearest neighbors in desc2 for each vector in desc1.
# #     If the distance matrix dm is not provided, torch.cdist() is used.
# descriptors_distance, matching_indices = kornia.feature.match_nn(desc1, desc2, dm=None)
# #(*). Function, which finds mutual nearest neighbors in desc2 for each vector in desc1.
# descriptors_distance, matching_indices = kornia.feature.match_mnn(desc1, desc2, dm=None)
# #(*). Function, which finds nearest neighbors in desc2 for each vector in desc1.
# # The method satisfies first to second nearest neighbor distance <= th
# descriptors_distance, matching_indices = kornia.feature.match_snn(desc1, desc2, th=0.8, dm=None)
# #(*). Function, which finds mutual nearest neighbors in desc2 for each vector in desc1.
# # the method satisfies first to second nearest neighbor distance <= th.
# descriptors_distance, matching_indices = kornia.feature.match_smnn(desc1, desc2, th=0.8, dm=None)
#
#
# ###############################################
# ### Feature Detector + Descriptor Wrappers: ###
# ###############################################
# # RETURNS
# # Detected local affine frames with shape [B,N,2,3]
# # Response function values for corresponding lafs with shape [B,N,1]
# # Local descriptors of shape [B,N,D], where D is descriptor size
# detector_module = kornia.feature.BlobDoG()
# detector_output = detector_module.forward(input_tensor)
#
# sift_descriptor = kornia.feature.SIFTDescriptor(patch_size=41, num_ang_bins=8, num_spatial_bins=4, rootsift=True, clipval=0.2)
#
# # TODO: the detector needs to output both (lafs, responses), i can wrap them all to get both keypoints, lafs and responses, i think
# local_features_detector_and_descriptor = kornia.feature.LocalFeature(detector=detector_module,
#                                                                      descriptor=sift_descriptor)
# detected_features, response_function, local_descriptors = local_features_detector_and_descriptor(input_tensor, mask=None)
#
# # Convenience module, which implements DoG detector + (Root)SIFT descriptor.
# # Still not as good as OpenCV/VLFeat because of https://github.com/kornia/kornia/pull/884, but we are working on it
# #(*). TODO: this is very comfortable!, need to understand why this is, supposedly, not as good
# SIFT_detector_and_descriptor = kornia.feature.SIFTFeature(num_features=500, upright=False, rootsift=True, device=input_tensor.device)
# detected_laf, response_function_for_laf, local_descriptors = SIFT_detector_and_descriptor(input_tensor, mask=None)
#
# # Convenience module, which implements GFTT detector + AffNet-HardNet descriptor
# GFTT_Hardnet_detector_and_descriptor = kornia.feature.GFTTAffNetHardNet(num_features=500, upright=False, device=input_tensor.device)
# detected_laf, response_function_for_laf, local_descriptors = GFTT_Hardnet_detector_and_descriptor.forward(input_tensor, mask=None)
#
# # Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor.
# KeyNet_AffNet_detector_and_descriptor = kornia.feature.KeyNetAffNetHardNet(num_features=500, upright=False, device=input_tensor.device)
# detected_laf, response_function_for_laf, local_descriptors = KeyNet_AffNet_detector_and_descriptor(input_tensor, mask=None)
#
# # Convenience module, which implements KeyNet detector + HardNet descriptor.
# KeyNet_HardNet_detector_and_descriptor = kornia.feature.KeyNetHardNet(num_features=500, upright=False, device=input_tensor.device)
# detected_laf, response_function_for_laf, local_descriptors = KeyNet_HardNet_detector_and_descriptor(input_tensor, mask=None)
#
#
#
# #########################################################
# ### Feature Detector + Descriptor + Matcher Wrappers: ###
# #########################################################
# # Module, which finds correspondences between two images based on local features.
# GFTT_Hardnet_detector_and_descriptor = kornia.feature.GFTTAffNetHardNet(num_features=8000, upright=False, device=input_tensor.device)
# gftt_hardnet_detector_plus_matcher = kornia.feature.LocalFeatureMatcher(local_feature=kornia.feature.GFTTAffNetHardNet(100).to(input_tensor.device),
#                                                          matcher=kornia.feature.DescriptorMatcher('snn', 0.99).to(input_tensor.device))
# # input_dict_to_matcher = EasyDict()
# # input_dict_to_matcher.image0 = input_tensor
# # input_dict_to_matcher.image1 = input_tensor_warped
# input_dict_to_matcher = {'image0': input_tensor/255,
#                          'image1': input_tensor_warped/255}
# with torch.no_grad():
#     output_dict = gftt_hardnet_detector_plus_matcher.forward(input_dict_to_matcher)
#     keypoints0 = output_dict['keypoints0']
#     keypoints1 = output_dict['keypoints1']
#     lafs0 = output_dict['lafs0']
#     lafs1 = output_dict['lafs1']
#     confidence = output_dict['confidence']
#     batch_indexes = output_dict['batch_indexes']
#
# H_matrix_output = kornia.geometry.homography.find_homography_dlt(keypoints1.unsqueeze(0), keypoints0.unsqueeze(0), weights=confidence.unsqueeze(0))
# affine_parameters_from_homography_matrix(H_matrix_output)
#
# # Module, which finds correspondences between two images.
# # This is based on the original code from paper “LoFTR: Detector-Free Local Feature Matching with Transformers”. See [SSW+21] for more details.
# LoFTR_correspondence = kornia.feature.LoFTR(pretrained='outdoor')
# input_dict_to_matcher = {'image0': input_tensor.cpu()/255,
#                          'image1': input_tensor_warped.cpu()/255}
# with torch.no_grad():
#     output_dict = LoFTR_correspondence(input_dict_to_matcher)
#     keypoints0 = output_dict['keypoints0']
#     keypoints1 = output_dict['keypoints1']
#     confidence = output_dict['confidence']
#     batch_indexes = output_dict['batch_indexes']
#
#
#
#
# #########################################################
# ### Local Affine Features: ###
# #########################################################
# #TODO: there are still many functions for LAFs, understand what's going on there!!!!
# ### Get Descriptions From LAFs: ###
# laf_descriptor_module = kornia.feature.LAFDescriptor(patch_descriptor_module=None, patch_size=32, grayscale_descriptor=True)
# bla = kornia.feature.get_laf_descriptors(input_tensor, lafs, patch_descriptor, patch_size=32, grayscale_descriptor=True)
# bla = laf_descriptor_module.forward(input_tensor, lafs)
#
# # Extract patches defined by LAFs from image tensor.
# # Patches are extracted from appropriate pyramid level.
# kornia.feature.extract_patches_from_pyramid(input_tensor, laf=torch.randn(1,00,2,3), PS=32, normalize_lafs_before_extraction=True)
#
# #########################################################
# ### Filter Response Normalization!!!!: ###
# #########################################################
# kornia.feature.FilterResponseNorm2d(num_features, eps=1e-06, is_bias=True, is_scale=True, is_eps_leanable=False)[source]
#
#
# #########################################################
# ### Image Patches: ###
# #########################################################
# #(1). Functions:
# kornia.contrib.compute_padding(original_size, window_size)
# kornia.contrib.extract_tensor_patches(input, window_size, stride=1, padding=0)
# kornia.contrib.combine_tensor_patches(patches, original_size, window_size, stride, unpadding=0)
# kornia.contrib.ExtractTensorPatches(window_size, stride=1, padding=0)
# kornia.contrib.CombineTensorPatches(original_size, window_size, unpadding=0)
# #(2). Example:
# T,C,H,W = input_tensor.shape
# patch_size = (3,3)
# padding = kornia.contrib.compute_padding((H,W), patch_size)
# out = kornia.contrib.extract_tensor_patches(input_tensor, window_size=patch_size, stride=(3, 3), padding=padding)
# output_tensor = kornia.contrib.combine_tensor_patches(out, original_size=(H,W), window_size=patch_size, stride=(3, 3), unpadding=padding)
#
#
# #########################################################
# ### Image Enhancement And Histogram: ###
# #########################################################
# output_tensor = kornia.enhance.sharpness(input_tensor, factor=0.5)
# output_tensor = kornia.enhance.equalize(input_tensor)
# output_tensor = kornia.enhance.equalize_clahe(input_tensor, clip_limit=40.0, grid_size=(8, 8), slow_and_differentiable=False)
# output_tensor = kornia.enhance.equalize3d(input_tensor) #Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format
#
# output_histogram = kornia.enhance.histogram(x, bins=100, bandwidth=1, epsilon=1e-10)[source]
#
# #########################################################
# ### Conversions: ###
# #########################################################
# kornia.geometry.conversions.rad2deg(tensor)
# kornia.geometry.conversions.deg2rad(tensor)
# kornia.geometry.conversions.pol2cart(rho, phi)
# kornia.geometry.conversions.cart2pol(x, y, eps=1e-08)
#
# kornia.geometry.conversions.convert_affinematrix_to_homography(A)
#
#
# #########################################################
# ### Homography Geometry And RANSAC: ###
# #########################################################
# kornia.geometry.epipolar.find_fundamental(points1, points2, weights)
# kornia.geometry.homography.find_homography_dlt(points1, points2, weights=None)
# kornia.geometry.homography.find_homography_dlt_iterated(points1, points2, weights, soft_inl_th=3.0, n_iter=5)
#
# #Return transfer error in image 2 for correspondences given the homography matrix:
# kornia.geometry.homography.oneway_transfer_error(pts1, pts2, H, squared=True, eps=1e-08)
#
# ### RANSAC: ###
# RANSAC_module = kornia.geometry.ransac.RANSAC(model_type='homography', inl_th=2.0, batch_size=2048, max_iter=10, confidence=0.99, max_lo_iters=5)
# estimated_model, inlinear_mask = RANSAC_module.forward(kp1, kp2)
#
#
#
# #########################################################
# ### Geometry/Warping: ###
# #########################################################
# ### Warp according to affine matrix: ###
# kornia.geometry.transform.warp_affine(src, Affine_matrix, dsize, mode='bilinear', padding_mode='zeros', align_corners=True, fill_value=torch.zeros(3))
#
# ### Warp according to flow: ###
# #   image (Tensor) – the tensor to remap with shape (B, C, H, W). Where C is the number of channels.
# #   map_x (Tensor) – the flow in the x-direction in pixel coordinates. The tensor must be in the shape of (B, H, W).
# #   map_y (Tensor) – the flow in the y-direction in pixel coordinates. The tensor must be in the shape of (B, H, W).
# #   mode (str, optional) – interpolation mode to calculate output values 'bilinear' | 'nearest'. Default: 'bilinear'
# #   padding_mode (str, optional) – padding mode for outside grid values 'zeros' | 'border' | 'reflection'. Default: 'zeros'
# #   align_corners (Optional[bool], optional) – mode for grid_generation. Default: None
# #   normalized_coordinates (bool, optional) – whether the input coordinates are normalized in the range of [-1, 1]. Default: False
# kornia.geometry.transform.remap(image, map_x, map_y, mode='bilinear', padding_mode='zeros', align_corners=None, normalized_coordinates=False)
#
# ### Resize (to specific size) and Rescale (according to scale factors): ###
# kornia.geometry.transform.resize(input, size, interpolation='bilinear', align_corners=None, side='short', antialias=False)
# kornia.geometry.transform.rescale(input, factor, interpolation='bilinear', align_corners=None, antialias=False)
#
#
# #############################
# ### Elastic Deformations: ###
# #############################
# #   PARAMETERS
# #   image (Tensor) – Input image to be transformed with shape (B,C,H,W).
# #   noise (Tensor) – Noise image used to spatially transform the input image. Same resolution as the input image with shape (B,2,H,W).
# #                    The coordinates order it is expected to be in x-y   (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!).
# #   kernel_size (Tuple[int, int], optional) – the size of the Gaussian kernel. Default: (63, 63)
# #   sigma (Tuple[float, float], optional) – The standard deviation of the Gaussian in the y and x directions, respectively. Larger sigma results in smaller pixel displacements. Default: (32.0, 32.0)
# #   alpha (Tuple[float, float], optional) – The scaling factor that controls the intensity of the deformation in the y and x directions, respectively. Default: (1.0, 1.0)
# #   align_corners (bool, optional) – Interpolation flag used by `grid_sample`. Default: False
# #   mode (str, optional) – Interpolation mode used by `grid_sample`. Either 'bilinear' or 'nearest'. Default: 'bilinear'
# #   padding_mode (str, optional) – The padding used by `grid_sample`. Either 'zeros', 'border' or 'refection'. Default: 'zeros'
# kornia.geometry.transform.elastic_transform2d(image,
#                                               noise,
#                                               kernel_size=(63, 63),
#                                               sigma=(32.0, 32.0),
#                                               alpha=(1.0, 1.0),
#                                               align_corners=False,
#                                               mode='bilinear',
#                                               padding_mode='zeros')
# ### Example: ###
# image = torch.rand(1, 3, 5, 5)
# noise = torch.rand(1, 2, 5, 5, requires_grad=True)
# image_hat = elastic_transform2d(image, noise, (3, 3))
# image_hat.mean().backward()
#
#
# #############################
# ### Pyramids: ###
# #############################
# kornia.geometry.transform.pyrdown(input, border_type='reflect', align_corners=False, factor=2.0)
# kornia.geometry.transform.pyrup(input, border_type='reflect', align_corners=False)
# kornia.geometry.transform.build_pyramid(input, max_level, border_type='reflect', align_corners=False)
#
# kornia.geometry.transform.PyrDown(border_type='reflect', align_corners=False, factor=2.0)
# kornia.geometry.transform.PyrUp(border_type='reflect', align_corners=False)
#
#
#
#

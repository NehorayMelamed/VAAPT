# import torch
#
# from RDND_proper.RapidBase.import_all import *
#
#
# # input_tensor = scipy.io.loadmat("/home/simteam-j/with_movement.mat")
# # input_tensor = numpy_to_torch(np.transpose(input_tensor["input_movie"], (3, 0, 1, 2)))
# ### Get rid of first frames which are not slow-motion: ###
# # input_tensor = torch.load("/home/mafat/Desktop/romeo_juliet/no_movement.pt")
# # input_tensor = torch.load("/home/mafat/Desktop/romeo_juliet/with_movement.pt")
#
# import torch
# # Nehoray
# input_tensor = torch.load("/home/simteam-j/Desktop/RDND_proper/with_movement_mat.pt")
# input_tensor_as_numpy = torch_to_numpy(input_tensor)
# ## Select roi
# ### Functions for selecting the ROI
# # Callback function for mouse events
#
# # def get_roi_from_user(frame):
# #     fig, ax = plt.subplots(1)
# #     ax.imshow(frame)
# #
# #     roi = plt.ginput(2)  # Get two points from the user
# #     plt.close()
# #
# #     x1, y1 = roi[0]
# #     x2, y2 = roi[1]
# #     roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
# #     return roi
# #
# # def crop_video_array(video_array, roi):
# #     # Assuming input video_array shape is (T, H, W, C)
# #     # and roi is a tuple of (x, y, w, h)
# #     x, y, w, h = map(int, roi)  # Convert ROI values to integers
# #     return video_array[:, y:y+h, x:x+w, :]
# #
# # def crop_video_interactive(video_array):
# #     # Get the first frame
# #     first_frame = video_array[0]
# #
# #     # Get the ROI from the user
# #     roi = get_roi_from_user(first_frame)
# #
# #     # Crop the video array using the selected ROI
# #     cropped_video_array = crop_video_array(video_array, roi)
# #
# #     return cropped_video_array
# #
# # roi_tensor_numpy = crop_video_interactive(input_tensor_as_numpy)
# # roi_tensor_torch = numpy_to_torch(roi_tensor_numpy)
# # roi_tensor_torch_mean =roi_tensor_torch.mean([-1,-2])
#
# ### Continue here to shibulet
#
#
# # imshow_torch_video(input_tensor, FPS=25, frame_stride=20)
# # imshow_torch_video(input_tensor[0:20], FPS=1, frame_stride=1)
#
#
# ### Get Background: ###
# #(1). use first frame as BG:
# # input_tensor_BG = input_tensor[0:1]
# #(2). use median frame as BG:
# input_tensor_BG = input_tensor.median(0, keepdim=True)[0]
# # imshow_torch(input_tensor_BG)
#
# ### Present BGS: ###
# # input_tensor_BGS = input_tensor - input_tensor_BG
# # imshow_torch_video(BW2RGB(input_tensor_BGS)/255, FPS=25, frame_stride=5)
# # imshow_torch_video(input_tensor_BGS/255, FPS=25, frame_stride=20)
# # print(1)
#
# ### Present Pair-Wise Differences: ###
# # input_tensor_pairwise_diff = input_tensor[1:] - input_tensor[0:-1]
# # input_tensor_pairwise_diff_total_intensity = input_tensor_pairwise_diff.mean([-1,-2])
# # plot_torch(input_tensor_pairwise_diff_total_intensity); plt.show()
# # imshow_torch_video(scale_array_to_range(input_tensor_pairwise_diff), FPS=25, frame_stride=1)
#
# ### Present total intensity of the entire image: ###
# input_tensor_total_intensity = input_tensor.mean([-1,-2]).squeeze()
# # plot_torch(input_tensor_total_intensity) ;plt.show()
#
# ### Present total intensity of the window only: ###
#
# ### Present Running mean average over the total intensity over window only: ###
# # input_tensor_window_only_total_intensity_running_mean = convn_torch(input_tensor_window_only_total_intensity, kernel=torch.ones(21)/21, dim=0)
# H_start = 0
# H_stop = 550
# W_start = 400
# W_stop = 750
# input_tensor_window_only = input_tensor[..., H_start:H_stop, W_start:W_stop]
# input_tensor_window_only_total_intensity = input_tensor_window_only.mean([-1,-2]).squeeze()
# plot_torch(input_tensor_window_only_total_intensity); plt.show()
# imshow_torch_video(input_tensor_window_only, FPS=25, frame_stride=5)
# # plot_torch(input_tensor_window_only_total_intensity); plt.show()
# # plot_torch(input_tensor_window_only_total_intensity_running_mean); plt.show()
#
# ### Perform FIR filtering on the signal (later on to be switched to FFT_OLA layer for real-time filtering): ###
# f1, f2 = 0.1, 0.2
# f_highpass = 0.8
# filter_number_of_samples = 121
# FIR_filter_numerator = signal.firwin(filter_number_of_samples, f_highpass, pass_zero=True)
# frequency_vec = get_frequency_vec_torch(N=filter_number_of_samples, FPS=200)
# FIR_filter_numerator_torch = torch.tensor(FIR_filter_numerator)
# FIR_filter_numerator_FFT = fftshift_torch_specific_dim(torch.fft.fftn(FIR_filter_numerator_torch, dim=-1), dim=0)
# # input_tensor_total_intensity_DC_substracted = input_tensor_total_intensity-input_tensor_total_intensity.mean(0, True)
# # plt.plot(FIR_filter_numerator); plt.show()
# # plot_torch(FIR_filter_numerator_torch); plt.show()
# # plot_torch(frequency_vec, FIR_filter_numerator_FFT.abs()); plt.show()
# # plot_torch(frequency_vec, 10*torch.log(FIR_filter_numerator_FFT.abs())); plt.show()
#
#
# ### Present FFT of the total intensity MINUS the mean (so no DC): ###
# input_tensor_total_intensity_DC_substracted = input_tensor_total_intensity-input_tensor_total_intensity.mean(0, True)
# # plot_torch(input_tensor_total_intensity_DC_substracted); plt.show()
# input_tensor_total_intensity_DC_substracted_numpy = input_tensor_total_intensity_DC_substracted.cpu().numpy()
# input_tensor_total_intensity_DC_substracted_numpy_filtered = scipy.signal.lfilter(FIR_filter_numerator, [1], input_tensor_total_intensity_DC_substracted_numpy)
# plt.plot(input_tensor_total_intensity_DC_substracted_numpy)
# plt.plot(input_tensor_total_intensity_DC_substracted_numpy_filtered)
# # plt.figure()
# plt.legend(['before filtering', 'after filtering'])
# plt.title('signal itself')
# plt.show()
#
# ### Plot FFT of filtered signal: ###
# input_tensor_total_intensity_DC_substracted_numpy_filtered_torch = torch.tensor(input_tensor_total_intensity_DC_substracted_numpy_filtered)
# input_tensor_total_intensity_DC_substracted_numpy_filtered_torch_fft = torch_fft_and_fftshift(input_tensor_total_intensity_DC_substracted_numpy_filtered_torch)
# plot_torch(input_tensor_total_intensity_DC_substracted_numpy_filtered_torch_fft.abs()); plt.show()
# plt.title('FFT of filtered with movment')
#
#
# ### Plot FFT of filtered signal: ###
# input_tensor_total_intensity_DC_substracted_numpy_filtered_torch = torch.tensor(input_tensor_total_intensity_DC_substracted_numpy_filtered)
# frequency_vec = get_frequency_vec_torch(N=len(input_tensor_total_intensity), FPS=200)
# input_tensor_total_intensity_DC_substracted_numpy_filtered_torch_fft = torch.fft.fftn(input_tensor_total_intensity_DC_substracted_numpy_filtered_torch, dim=-1)
# input_tensor_total_intensity_DC_substracted_numpy_filtered_torch_fft = fftshift_torch_specific_dim(input_tensor_total_intensity_DC_substracted_numpy_filtered_torch_fft, dim=0)
# plot_torch(frequency_vec, input_tensor_total_intensity_DC_substracted_numpy_filtered_torch_fft.abs()**2)
# plt.show()
#
#
# input_tensor_total_intensity_FFT = torch.fft.fftn(input_tensor_total_intensity_DC_substracted, dim=-1)
# input_tensor_total_intensity_FFT = fftshift_torch_specific_dim(input_tensor_total_intensity_FFT, dim=0)
# frequency_vec = get_frequency_vec_torch(N=len(input_tensor_total_intensity), FPS=200)
# plot_torch(frequency_vec, input_tensor_total_intensity_FFT.abs()**2)
# plt.title('FFT')
# plt.xlabel('frequency axis')
# plt.ylabel('A.U')
# plt.show()
#
# ### Show gradient intensities: ###
# input_tensor_gradient_H, input_tensor_gradient_W = torch.gradient(input_tensor, dim=[-1,-2])
# input_tensor_gradient_total_intensity = (input_tensor_gradient_W**2 + input_tensor_gradient_H**2)**0.5
# # imshow_torch_video(input_tensor_gradient_total_intensity/input_tensor_gradient_total_intensity.max(), FPS=25, frame_stride=20)
#
# ### Show histogram stretched video: ###
# input_tensor_hist_stretched = scale_array_stretch_hist(input_tensor)
# imshow_torch_video(BW2RGB(input_tensor_hist_stretched)/input_tensor_hist_stretched.max(), FPS=25, frame_stride=5)
#
# ### Perform semantic segmentation on BG: ###
# model = nn.Sequential(torchvision.models.segmentation.fcn_resnet50(pretrained=False), Lambda(lambda x: x['out']))
# input_tensor_BG_semantic_segmentation = model(BW2RGB(input_tensor_BG))
# input_tensor_BG_semantic_segmentation_final = input_tensor_BG_semantic_segmentation.max(1, True)[0]
# torchvision.utils.draw_segmentation_masks(BW2RGB(scale_array_to_range(input_tensor_BG.squeeze(0), (0,255)).type(torch.uint8)),
#                                           input_tensor_BG_semantic_segmentation.squeeze(0))
#
# plt.show()
# bla = 1
# imshow_torch_video(BW2RGB(input_tensor)/255, FPS=25, frame_stride=5)

from RDND_proper.RapidBase.import_all import *
from carebear import *
import subprocess
import sys

### initialize stuff ###
frame_height = 600
frame_width = 800
frame_data_size = frame_height * frame_width * 4
frame_num = 0
buffer_size = 25
allocate_fixed_array("shm_filename", frame_data_size)

### open subprocess for c++ code ###
cpp_executable = "/home/simteam-j/daheng_recording/pipelines/DahengAcquisition/build/imshow_daheng"
process = subprocess.Popen([cpp_executable], stdout=sys.stdout, stderr=sys.stderr)
flag = True
while process.poll() is None:

    # Reshape the array into the original frame dimensions
    # allocate_value("shm_filename")
    pack_value("shm_filename")
    # if flag:
    #     pack_fixed_array("shm_filename")
    #     flag = False
    # else:
    #     pack_fixed_array_inplace("shm_filename")
    free_value("shm_filename")
    value = unpack_fixed_array("shm_filename").numpy()
    # cv2.imshow('Python Video', frame)
    # if cv2.waitKey(30) >= 0:
    #     break

cv2.destroyAllWindows()

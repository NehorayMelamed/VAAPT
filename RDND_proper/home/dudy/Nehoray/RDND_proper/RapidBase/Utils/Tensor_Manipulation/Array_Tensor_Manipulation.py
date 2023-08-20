#Imports:
#(1). Auxiliary:
from __future__ import print_function
import cv2
import math
import random
import ctypes  # An included library with Python install.

import torchvision.transforms.functional
from matplotlib.pyplot import *
length = len #use length instead of len.... make sure it doesn't cause problems
from torchvision.utils import make_grid
import torch
import einops
from torch import Tensor
from functools import reduce
from typing import Tuple, List, Union
from RapidBase.Utils.MISCELENEOUS import to_list_of_certain_size, to_tuple_of_certain_size
from torch.nn.modules.utils import _pair, _quadruple
import torch.nn as nn
import scipy
# from RapidBase.Utils.Classical_DSP.Convolution_Utils import convn_torch
from scipy.signal import gaussian




#######################################################################################################################################
def get_full_shape_torch(input_tensor):
    if len(input_tensor.shape) == 1:
        W = input_tensor.shape
        H = 1
        C = 1
        T = 1
        B = 1
        shape_len = 1
        shape_vec = (W)
    elif len(input_tensor.shape) == 2:
        H, W = input_tensor.shape
        C = 1
        T = 1
        B = 1
        shape_len = 2
        shape_vec = (H,W)
    elif len(input_tensor.shape) == 3:
        C, H, W = input_tensor.shape
        T = 1
        B = 1
        shape_len = 3
        shape_vec = (C,H,W)
    elif len(input_tensor.shape) == 4:
        T, C, H, W = input_tensor.shape
        B = 1
        shape_len = 4
        shape_vec = (T,C,H,W)
    elif len(input_tensor.shape) == 5:
        B, T, C, H, W = input_tensor.shape
        shape_len = 5
        shape_vec = (B,T,C,H,W)
    shape_vec = np.array(shape_vec)
    return (B,T,C,H,W), shape_len, shape_vec
#######################################################################################################################################



#######################################################################################################################################
### Structure Tensor orientation & coherence - a way to find clear structure in an image and might help us when deciding upon semantic map: ###
def get_structure_tensor(inputIMG, w=10):
    # w = 10  # window size is WxW
    # C_Thr = 0.43  # threshold for coherency
    # LowThr = 35  # threshold1 for orientation, it ranges from 0 to 180
    # HighThr = 57  # threshold2 for orientation, it ranges from 0 to 180

    img = inputIMG.astype(np.float32)
    # GST components calculation (start)
    # J =  (J11 J12 J12 J22) - GST
    imgDiffX = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
    imgDiffY = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    imgDiffXY = cv2.multiply(imgDiffX, imgDiffY)

    imgDiffXX = cv2.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv2.multiply(imgDiffY, imgDiffY)
    J11 = cv2.boxFilter(imgDiffXX, cv2.CV_32F, (w, w))
    J22 = cv2.boxFilter(imgDiffYY, cv2.CV_32F, (w, w))
    J12 = cv2.boxFilter(imgDiffXY, cv2.CV_32F, (w, w))
    # GST components calculations (stop)
    # eigenvalue calculation (start)
    # lambda1 = J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2)
    # lambda2 = J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2)
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv2.multiply(tmp2, tmp2)
    tmp3 = cv2.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = tmp1 + tmp4  # biggest eigenvalue
    lambda2 = tmp1 - tmp4  # smallest eigenvalue
    # eigenvalue calculation (stop)
    # Coherency calculation (start)
    # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    # Coherency is anisotropy degree (consistency of local orientation)
    imgCoherencyOut = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)
    # Coherency calculation (stop)
    # orientation angle calculation (start)
    # tan(2*Alpha) = 2*J12/(J22 - J11)
    # Alpha = 0.5 atan2(2*J12/(J22 - J11))
    imgOrientationOut = cv2.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
    imgOrientationOut = 0.5 * imgOrientationOut
    # orientation angle calculation (stop)
    return imgCoherencyOut, imgOrientationOut


class Gaussian_Blur_Layer(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussian_Blur_Layer, self).__init__()
        if type(kernel_size) is not list and type(kernel_size) is not tuple:
            kernel_size = [kernel_size] * dim
        if type(sigma) is not list and type(sigma) is not tuple:
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        self.padding = []
        for i in kernel_size:
            self.padding.append(i // 2)
            self.padding.append(i // 2)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(F.pad(input, self.padding), weight=self.weight, groups=self.groups)


def Gaussian_Blur_Wrapper_Torch(gaussian_blur_layer, input_tensor, number_of_channels_per_frame, frames_dim=0, channels_dim=1):
    # TODO: this is extremely inefficient!!!! never have for loops (without parallelization) in script!!! take care of this!!!!
    # TODO: simply have the gaussian blur work on all channels together or something. in case of wanting several blurs within the batch we can do that
    ### indices pre processing: ###
    frames_dim_channels = input_tensor.shape[frames_dim]
    channels_dim_channels = input_tensor.shape[channels_dim]
    flag_frames_concatenated_along_channels = (frames_dim == channels_dim)
    if frames_dim == channels_dim:  # frames concatenated along channels dim
        number_of_frames = int(channels_dim_channels / number_of_channels_per_frame)
    else:
        number_of_frames = frames_dim_channels
    output_tensor = torch.zeros_like(input_tensor)

    if len(input_tensor.shape) == 4:  # [B,C,H,W]
        B, C, H, W = input_tensor.shape

        if flag_frames_concatenated_along_channels:  # [B,C*T,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[:, i * number_of_channels_per_frame * (i + 1) * number_of_channels_per_frame, :, :] = \
                    gaussian_blur_layer(input_tensor[:, i * number_of_channels_per_frame * (i + 1) * number_of_channels_per_frame, :, :])
        else:  # [T,C,H,W] or [B,C,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[i:i + 1, :, :, :] = gaussian_blur_layer(input_tensor[i:i + 1, :, :, :])

    elif len(input_tensor.shape) == 5:  # [B,T,C,H,W]
        B, T, C, H, W = input_tensor.shape

        for i in np.arange(number_of_frames):
            output_tensor[:, i, :, :, :] = gaussian_blur_layer(input_tensor[:, i, :, :, :])

    return output_tensor


#TODO: don't use this!!! use kornia instead!!!
### Canny Edge Detection - in case we decide the sobel edge detection isn't good enough: ###
class canny_edge_detection(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(canny_edge_detection, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1, filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, filter_size), padding=(0, filter_size // 2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size, 1), padding=(filter_size // 2, 0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0] // 2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([[0, 0, 0],
                             [0, 1, -1],
                             [0, 0, 0]])

        filter_45 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, -1]])

        filter_90 = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, -1, 0]])

        filter_135 = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [-1, 0, 0]])

        filter_180 = np.array([[0, 0, 0],
                               [-1, 1, 0],
                               [0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_270 = np.array([[0, -1, 0],
                               [0, 1, 0],
                               [0, 0, 0]])

        filter_315 = np.array([[0, 0, -1],
                               [0, 1, 0],
                               [0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

        self.gaussian_filter_horizontal = self.gaussian_filter_horizontal.to(device)
        self.gaussian_filter_vertical = self.gaussian_filter_vertical.to(device)
        self.sobel_filter_horizontal = self.sobel_filter_horizontal.to(device)
        self.sobel_filter_vertical = self.sobel_filter_vertical.to(device)
        self.directional_filter = self.directional_filter.to(device)

    def forward(self, img, threshold=None):
        # TODO: when all the functions will be consistent with anvil make sure to transfer this from BW2RGB
        img_r = img[:, 0:1]
        img_g = img[:, 1:2]
        img_b = img[:, 2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r, blurred_img_g, blurred_img_b], dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r ** 2 + grad_y_r ** 2)
        grad_mag += torch.sqrt(grad_x_g ** 2 + grad_y_g ** 2)
        grad_mag += torch.sqrt(grad_x_b ** 2 + grad_y_b ** 2)
        grad_orientation = (torch.atan2(grad_y_r + grad_y_g + grad_y_b, grad_x_r + grad_x_g + grad_x_b) * (180.0 / 3.14159))
        grad_orientation += 180.0
        grad_orientation = torch.round(grad_orientation / 45.0) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)]).to(img.device)
        # if self.use_cuda:
        #     pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1, height, width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1, height, width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max == 0] = 0.0

        # THRESHOLD
        if threshold is None:
            threshold = self.threshold

        thresholded = thin_edges.clone()
        thresholded[thin_edges < threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag < threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        pool = lambda x: F.max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        return einops.rearrange(
            pool(einops.rearrange(input, "n c w h -> n (w h) c")),
            "n (w h) c -> n c w h",
            n=n,
            w=w,
            h=h,
        )


def round_to_nearest_half(input_tensor):
    if np.isscalar(input_tensor):
        return round((input_tensor * 2)) / 2
    else:
        return (input_tensor * 2).round() / 2
########################################################################################################################################





#######################################################################################################################################
### Peak Detection: ###
def peak_detect_pytorch(input_vec, window_size=11, dim=0, flag_use_ratio_threshold=False, ratio_threshold=2.5, flag_plot=False):
    # This is a specific test case for debug, delete if not needed
    # window_size = 11
    # flag_plot = False
    # input_vec = torch.randn(100).cumsum(0)
    # input_vec = input_vec - torch.linspace(0, input_vec[-1].item(), input_vec.size(0))
    # input_vec = input_vec.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # input_vec = input_vec.repeat(1, 4, 2, 3)
    # dim = 0

    # Get Frame To Concat
    if dim < 0:
        dim = len(input_vec.shape) - np.abs(dim)
    to_concat = torch.zeros([1]).to(input_vec.device)
    for dim_index in np.arange(len(input_vec.shape) - 1):
        to_concat = to_concat.unsqueeze(0)
    for dim_index in np.arange(len(input_vec.shape)):
        if dim_index != dim:
            to_concat = torch.cat([to_concat] * input_vec.shape[dim_index], dim_index)

    #  Phase one: find peaks mask by comparing each element to its neighbors over the chosen dimension
    I1 = torch.arange(0, input_vec.shape[dim] - 2).to(input_vec.device)
    I2 = torch.arange(1, input_vec.shape[dim] - 1).to(input_vec.device)
    I3 = torch.arange(2, input_vec.shape[dim] - 0).to(input_vec.device)
    shape_len = len(input_vec.shape)
    shape_characters = 'BTCHW'
    current_shape_characters = shape_characters[-shape_len:]
    current_shape_characters = current_shape_characters[dim]
    I1 = torch_get_ND(I1, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    I2 = torch_get_ND(I2, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    I3 = torch_get_ND(I3, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    shape_vec = torch.tensor(input_vec.shape)
    shape_vec[dim] = 1
    shape_vec = tuple(shape_vec.cpu().numpy())
    I1 = I1.repeat(shape_vec)
    I2 = I2.repeat(shape_vec)
    I3 = I3.repeat(shape_vec)
    a1 = torch.gather(input_vec, dim, I1)  # input_vec[:, :, :-2]
    a2 = torch.gather(input_vec, dim, I2)  # input_vec[:, :, 1:-1]
    a3 = torch.gather(input_vec, dim, I3)  # input_vec[:, :, 2:]
    peak_mask = torch.cat([to_concat, (a1 < a2) & (a3 < a2), to_concat], dim=dim)

    # Phase two: find peaks that are also local maximas
    # This section performs generic rearrangement for the torch max_pool1d_with_indices function.
    # This function requires an up to 3d input and operates max pool over the last dimension only.
    # Therefore, dimensioned are swapped and unified for the operation and later restored
    dims_string = 'b t c h w'
    initial_dims_string = dims_string[-shape_len * 2 + 1:]
    dims_dict = dict(zip(initial_dims_string.split(' '), input_vec.shape))
    new_dims_string = initial_dims_string[0:2 * dim] + initial_dims_string[dim * 2 + 1:] + ' ' + initial_dims_string[2 * dim:2 * dim + 1]
    new_dims_string = str.replace(new_dims_string, ' ', '')

    if len(new_dims_string) > 3:
        first_dim_string = new_dims_string[:-2]
        new_first_dim_string = str.replace(first_dim_string, '', ' ')[1:-1]
        new_first_dim_string = f'({new_first_dim_string})'
        rest_dims_string = new_dims_string[-2:]
        new_rest_dim_string = str.replace(rest_dims_string, '', ' ')[1:-1]
        new_dims_string = f'{new_first_dim_string} {new_rest_dim_string}'
    else:
        new_dims_string = str.replace(new_dims_string, '', ' ')[1:-1]

    total_dims_string_1 = initial_dims_string + ' -> ' + new_dims_string
    total_dims_string_2 = new_dims_string + ' -> ' + initial_dims_string
    # rearrange and do max pool
    input_vec2 = einops.rearrange(input_vec, total_dims_string_1)
    input_vec2_maxpool_values, input_vec2_maxpool_indices = torch.nn.functional.max_pool1d_with_indices(input_vec2, window_size, 1, padding=window_size // 2)
    dims_to_separate_dict = {k: v for k, v in dims_dict.items() if k in first_dim_string}
    input_vec2_maxpool_indices = einops.rearrange(input_vec2_maxpool_indices, total_dims_string_2, **dims_to_separate_dict)

    # ### Get Window Mean: ###
    # ratio_threshold = 3
    # window_mean = convn_torch(input_vec2, torch.ones(window_size)/window_size, dim=-1)
    # max_value_enough_above_value_logical_mask = (input_vec2_maxpool_values/window_mean > ratio_threshold)
    # max_value_enough_above_value_logical_mask = einops.rearrange(max_value_enough_above_value_logical_mask, total_dims_string_2, **dims_to_separate_dict)

    ### Get Window Median: ###
    if flag_use_ratio_threshold:
        input_vec2_unfolded = input_vec2.unfold(dimension=-1, size=window_size, step=1)
        input_vec2_unfolded_median = input_vec2_unfolded.median(-1)[0]
        to_concat_median1 = torch.ones(window_size // 2).to(input_vec.device).unsqueeze(0).unsqueeze(0)
        to_concat_median2 = torch.ones(window_size // 2).to(input_vec.device).unsqueeze(0).unsqueeze(0)
        input_vec2_unfolded_median = torch.cat((to_concat_median1, input_vec2_unfolded_median, to_concat_median2), -1)
        max_value_enough_above_value_logical_mask = (input_vec2_maxpool_values / input_vec2_unfolded_median > ratio_threshold)
        max_value_enough_above_value_logical_mask = einops.rearrange(max_value_enough_above_value_logical_mask, total_dims_string_2, **dims_to_separate_dict)

    # keep only peaks, each contains the value of the local maximum in the surrounding window
    filtered_peaks_with_window_maxima_value = input_vec2_maxpool_indices * peak_mask

    ### Only Keep Those Peaks Which Are Sufficiently Above Window Mean: ###
    if flag_use_ratio_threshold:
        filtered_peaks_with_window_maxima_value = filtered_peaks_with_window_maxima_value * max_value_enough_above_value_logical_mask

    # make a tensor of indices over the wanted dimension, repeated in the other dimensions
    arange_in_correct_dim_tensor = torch.arange(0, input_vec.shape[dim])
    arange_in_correct_dim_tensor = torch_get_ND(arange_in_correct_dim_tensor, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    dims_to_repeat = input_vec.shape
    dims_to_repeat = torch.tensor(dims_to_repeat)
    dims_to_repeat[dim] = 1
    arange_in_correct_dim_tensor = arange_in_correct_dim_tensor.repeat(*dims_to_repeat).to(input_vec.device)
    # keep only the picks that are also maximal in their local window
    # Assumption: no picks in the 0 index (not possible due to pick definition)
    # to avoid true values at the 0 indices, change the 0 values in filtered_picks_with_window_maxima_value to -1
    filtered_peaks_with_window_maxima_value = torch.where((filtered_peaks_with_window_maxima_value == 0),
                                                          torch.ones_like(filtered_peaks_with_window_maxima_value) * (-1),
                                                          filtered_peaks_with_window_maxima_value)
    # get a mask of the picks that are also window maximas
    maxima_peaks = (filtered_peaks_with_window_maxima_value == arange_in_correct_dim_tensor)

    # Several plot options for different dimensionalities
    # Plot Results: this is a specific case of chw tensor, chosen dim w
    # i = 1
    # j = 1
    # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[i, j])[maxima_peaks[i, j]]
    # plt.figure()
    # plt.plot(input_vec[i, j].numpy())
    # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[i, j, maxima_peaks_to_plot].numpy(), '.')
    # plt.title('local maxima peaks')

    # Plot Results: this is a specific case of tchw tensor, chosen dim t
    # i = 1
    # j = 1
    # k = 1
    # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, i, j, k]])[:, i, j, k]
    # plt.figure()
    # plt.plot(input_vec[:, i, j, k].numpy())
    # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, i, j, k].numpy(), '.')
    # plt.title('local maxima peaks')

    # Plot Results: this is a specific case of btchw tensor, chosen dim t
    # i = 1
    # j = 1
    # k = 1
    # l = 1
    # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[i])[maxima_peaks[i, :, j, k, l]][:, j, k, l]
    # plt.figure()
    # plt.plot(input_vec[i, :, j, k, l].numpy())
    # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[i, maxima_peaks_to_plot, j, k, l].numpy(), '.')
    # plt.title('local maxima peaks')

    return maxima_peaks, arange_in_correct_dim_tensor
#######################################################################################################################################



#######################################################################################################################################
### Pytorch Morphology: ###
#TODO: don't use this!!!! use kornia instead!!!!
class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')


class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')
#######################################################################################################################################


#######################################################################################################################################
### Torch Conditions: ###
def first_nonzero_torch(x, axis=0):
    # TODO: make first_zero_torch()
    nonz = (x > 0)
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)

def torch_get_where_condition_holds(condition):
    return (condition).nonzero(as_tuple=False)  # reutrns a tensor of size (number_of_matches X tensor_number_of_dimensions)


def torch_get_where_condition_holds_as_list_of_tuples(condition):
    matches_list_per_dimension = (condition).nonzero(as_tuple=False)  # reutrns a tensor of size (number_of_matches X tensor_number_of_dimensions)
    matches_list_of_tuples = []
    number_of_dimensions = len(matches_list_per_dimension)
    for match_index in matches_list_per_dimension[0].shape[0]:
        current_list = []
        for dim_index in number_of_dimensions:
            current_list.append(matches_list_per_dimension[dim_index][match_index])
        current_list = tuple(current_list)
        matches_list_of_tuples.append(current_list)
#######################################################################################################################################

#################
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

################

#######################################################################################################################################
def logical_mask_to_indices_torch(input_logical_mask, flag_return_tensor_or_list_of_tuples='tensor'):
    # flag_return_tensor_or_list_of_tuples = 'tensor' / 'list_of_tuples'
    if flag_return_tensor_or_list_of_tuples == 'tensor':
        return input_logical_mask.nonzero(as_tuple=False)
    else:
        return input_logical_mask.nonzero(as_tuple=True)


def indices_to_logical_mask_torch(input_logical_mask=None, input_tensor_reference=None, indices=None, flag_one_based=False):
    #TODO: don't forget to enter correct indices dimensions
    ### If i didn't get logical mask --> create it: ###
    if input_logical_mask is None:
        if flag_one_based:
            input_logical_mask = torch.ones_like(input_tensor_reference).bool()
        else:
            input_logical_mask = torch.zeros_like(input_tensor_reference).bool()

    ### Which Value To Fill: ###
    if flag_one_based:
        value_to_fill = torch.tensor(0).bool()
    else:
        value_to_fill = torch.tensor(1).bool()

    ### Fill Values Where Indices Say: ###
    if type(indices) == tuple:
        input_logical_mask = torch.index_put_(input_logical_mask, indices.long(), value_to_fill)
    else:
        input_logical_mask = torch.index_put_(input_logical_mask, tuple(indices.t().long()), value_to_fill)

    return input_logical_mask
#######################################################################################################################################



#######################################################################################################################################
def upsample_image_fft_numpy(original_image, upsample_factor=1):
    ### Assuming image is H=W and is even size TODO: take care of general case later
    bla = read_image_default()[:,:,0]
    original_image_fft = np.fft.fft2(original_image)
    original_image_fft = np.fft.fftshift(original_image_fft)
    original_image_fft_padded = np.pad(original_image_fft, int(bla.shape[0]/2))
    original_image_fft_padded = np.fft.fftshift(original_image_fft_padded)
    original_image_upsampled = np.fft.ifft2(original_image_fft_padded)
    # imshow(bla)
    # figure()
    # imshow(np.abs(original_image_upsampled))
    return original_image_upsampled

def upsample_image_fft_torch(original_image, upsample_factor=2):
    # original_image = read_image_default_torch()[:,0:1,:,:]
    original_image_fft = torch.fft.fftn(original_image, dim=[-1,-2])
    original_image_fft = fftshift_torch(original_image_fft)
    padding_size = int(original_image.shape[-1]/2)
    original_image_fft_padded = torch.nn.functional.pad(original_image_fft, [padding_size,padding_size,padding_size,padding_size])
    original_image_fft_padded = fftshift_torch(original_image_fft_padded)
    original_image_upsampled = torch.fft.ifftn(original_image_fft_padded, dim=[-1,-2])
    # imshow_torch(original_image)
    # imshow_torch(original_image_upsampled.abs())
    return original_image_upsampled
#######################################################################################################################################


# from RapidBase.Utils.IO.Path_and_Reading_utils import read_image_stack_default_torch
# from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch
def Tstack_to_Cstack(input_tensor):
    if len(input_tensor.shape) == 4:
        T,C,H,W = input_tensor.shape
        input_tensor = torch.reshape(input_tensor, (C * T, H, W))
    elif len(input_tensor.shape) == 5:
        B,T,C,H,W = input_tensor.shape
        input_tensor = torch.reshape(input_tensor, (B, C * T, H, W))
    return input_tensor



def Cstack_to_Tstack(input_tensor, number_of_channels=1):
    if len(input_tensor.shape) == 4:
        B,CxT,H,W = input_tensor.shape
        C = number_of_channels
        T = CxT // C
        input_tensor = torch.reshape(input_tensor, (B,T,C,H,W))
    elif len(input_tensor.shape) == 3:
        CxT,H,W = input_tensor.shape
        C = number_of_channels
        T = CxT // C
        input_tensor = torch.reshape(input_tensor, (T,C,H,W))
    return input_tensor

def convert_dtype_torch(input_tensor, new_dtype):
    return torch.as_tensor(input_tensor, dtype=new_dtype)

#Useful for if you want to only take those elements where a condition is satisfied:
def indices_from_logical_torch(logical_array):
    logical_array = np.asarray(logical_array*1)
    return np.reshape(np.argwhere(logical_array==1),-1)

def find_numpy(condition_logical_array, mat_in=None):
    #TODO: have find_torch
    """" Like Matlab's Find:
         condition_logical_array is something like mat_in>1, which is a logical (bolean) array 
    """""
    #Use: good_elements = find(mat_in>5, mat_in)
    indices = np.asarray(condition_logical_array.nonzero())
    if mat_in==None:
        values = None
    else:
        values = mat_in(condition_logical_array)
    return indices, values



def torch_get_valid_center(input_tensor, non_valid_border_size):
    if len(input_tensor.shape) == 4:
        valid_mask = torch.zeros_like(input_tensor)
        B,C,H,W = input_tensor.shape
        valid_mask[:,:,non_valid_border_size:H-non_valid_border_size, non_valid_border_size:W-non_valid_border_size] = 1
    elif len(input_tensor.shape) == 5:
        valid_mask = torch.zeros_like(input_tensor)
        B, T, C, H, W = input_tensor.shape
        valid_mask[:, :, :, non_valid_border_size:H - non_valid_border_size, non_valid_border_size:W - non_valid_border_size] = 1
    return valid_mask


#########################################################################################################################################
### Squeeze / Unsqueeze: ###

def torch_get_5D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'B':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BW':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BH':
            return input_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BC':
            return input_tensor.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        elif input_dims == 'BT':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # (3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1).unsqueeze(0)
        elif input_dims == 'BHW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'BCH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'BTC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    # (4).
    if len(input_tensor.shape) == 4:
        if input_dims is None:
            input_dims = 'TCHW'

        if input_dims == 'TCHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'BTCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'BCHW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'BTHW':
            return input_tensor.unsqueeze(2)

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor

# def torch_get_4D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor.unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 3:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 4:
#         return input_tensor
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[:,0]

def torch_image_flatten(input_tensor):
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if shape_len == 1:
        return input_tensor
    elif shape_len == 2:
        return input_tensor.reshape((H*W))
    elif shape_len == 3:
        return input_tensor.reshape((C, H*W))
    elif shape_len == 4:
        return input_tensor.reshape((T, C, H*W))
    elif shape_len == 5:
        return input_tensor.reshape((B, T, C, H*W))

def torch_stack_dims(input_tensor, dims_to_stack=None):
    # create a function which you input, for instance, 'BT' and it stacks them up, or something like that, maybe with indices
    1

def torch_transpose_HW(input_tensor):
    1


def torch_get_ND(input_tensor, input_dims=None, number_of_dims=None):
    if number_of_dims == 2:
        return torch_get_2D(input_tensor, input_dims)
    elif number_of_dims == 3:
        return torch_get_3D(input_tensor, input_dims)
    elif number_of_dims == 4:
        return torch_get_4D(input_tensor, input_dims)
    elif number_of_dims == 5:
        return torch_get_5D(input_tensor, input_dims)

def torch_unsqueeze_like(input_tensor, tensor_to_mimick, direction_to_unsqueeze=0):
    #direction_to_unsqueeze = (0:beginning, -1=end)
    (B1,T1,C1,H1,W1), shape_len1, shape_vec1 = get_full_shape_torch(input_tensor)
    (B2,T2,C2,H2,W2), shape_len2, shape_vec2 = get_full_shape_torch(tensor_to_mimick)
    for i in np.arange(shape_len2-shape_len1):
        input_tensor = input_tensor.unsqueeze(direction_to_unsqueeze)
    return input_tensor

def torch_equalize_dimensionality(input_tensor, tensor_to_mimick, direction_to_unsqueeze=0):
    return torch_unsqueeze_like(input_tensor, tensor_to_mimick, direction_to_unsqueeze)

def torch_get_4D(input_tensor, input_dims=None, flag_stack_BT=False):
    #[T,C,H,W]
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'T':
            return input_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(0).unsqueeze(2)
        elif input_dims == 'TW':
            return input_tensor.unsqueeze(1).unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(0).unsqueeze(-1)
        elif input_dims == 'TH':
            return input_tensor.unsqueeze(1).unsqueeze(-1)
        elif input_dims == 'TC':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        if input_dims is None:
            input_dims = 'CHW'

        if input_dims == 'CHW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'TCH':
            return input_tensor.unsqueeze(-1)
        elif input_dims == 'THW':
            return input_tensor.unsqueeze(1)

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor

    #(5).
    if len(input_tensor.shape) == 5:
        if flag_stack_BT:
            B,T,C,H,W = input_tensor.shape
            return input_tensor.view(B*T,C,H,W)
        else:
            return input_tensor[0]


# def torch_get_3D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0).unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 3:
#         return input_tensor
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0]

def torch_get_3D(input_tensor, input_dims=None):
    #(1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1).unsqueeze(0)
        elif input_dims == 'C':
            return input_tensor.unsqueeze(-1).unsqueeze(-1)

    #(2).
    if len(input_tensor.shape) == 2:
        if input_dims is None:
            input_dims = 'HW'

        if input_dims == 'HW':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'CW':
            return input_tensor.unsqueeze(1)
        elif input_dims == 'CH':
            return input_tensor.unsqueeze(-1)

    #(3).
    if len(input_tensor.shape) == 3:
        return input_tensor

    #(4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0]

    #(5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0,0]  #TODO: maybe stack on top of each other?


# def torch_get_2D(input_tensor):
#     if len(input_tensor.shape) == 1:
#         return input_tensor.unsqueeze(0)
#     elif len(input_tensor.shape) == 2:
#         return input_tensor
#     elif len(input_tensor.shape) == 3:
#         return input_tensor[0]
#     elif len(input_tensor.shape) == 4:
#         return input_tensor[0,0]
#     elif len(input_tensor.shape) == 5:
#         return input_tensor[0,0,0]


def torch_get_2D(input_tensor, input_dims=None):
    # (1).
    if len(input_tensor.shape) == 1:
        if input_dims is None:
            input_dims = 'W'

        if input_dims == 'W':
            return input_tensor.unsqueeze(0)
        elif input_dims == 'H':
            return input_tensor.unsqueeze(-1)

    # (2).
    if len(input_tensor.shape) == 2:
        return input_tensor

    # (3).
    if len(input_tensor.shape) == 3:
        return input_tensor[0]

    # (4).
    if len(input_tensor.shape) == 4:
        return input_tensor[0,0]

    # (5).
    if len(input_tensor.shape) == 5:
        return input_tensor[0, 0, 0]  # TODO: maybe stack on top of each other?


def torch_at_least_2D(input_tensor):
    return torch_get_2D(input_tensor)
def torch_at_least_3D(input_tensor):
    return torch_get_3D(input_tensor)
def torch_at_least_4D(input_tensor):
    return torch_get_4D(input_tensor)
def torch_at_least_5D(input_tensor):
    return torch_get_5D(input_tensor)
#########################################################################################################################################



#########################################################################################################################################
### Scaling And Stretching: ###
def normalize_array(input_tensor, min_max_values=(0,1)):
    input_tensor_normalized = (input_tensor - input_tensor.min()) / (input_tensor.max()-input_tensor.min()) * (min_max_values[1]-min_max_values[0]) + min_max_values[0]
    return input_tensor_normalized

def scale_array_to_range(input_tensor, min_max_values_to_scale_to=(0,1)):
    input_tensor_normalized = (input_tensor - input_tensor.min()) / (input_tensor.max()-input_tensor.min() + 1e-16) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    return input_tensor_normalized

def scale_array_from_range(input_tensor, min_max_values_to_clip=(0,1), min_max_values_to_scale_to=(0,1)):
    if type(input_tensor) == torch.Tensor:
        input_tensor_normalized = (input_tensor.clamp(min_max_values_to_clip[0],min_max_values_to_clip[1]) - min_max_values_to_clip[0]) / (min_max_values_to_clip[1]-min_max_values_to_clip[0]) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    else:
        input_tensor_normalized = (input_tensor.clip(min_max_values_to_clip[0],min_max_values_to_clip[1]) - min_max_values_to_clip[0]) / (min_max_values_to_clip[1]-min_max_values_to_clip[0]) * (min_max_values_to_scale_to[1]-min_max_values_to_scale_to[0]) + min_max_values_to_scale_to[0]
    # output_tensor = (input_tensor - q1) / (q2-q1) * (1-0) + 0 -> input_tensor = output_tensor*(q2-q1) + q1
    return input_tensor_normalized


# output_tensor = kornia.enhance.sharpness(input_tensor, factor=0.5)
# output_tensor = kornia.enhance.equalize(input_tensor)
# output_tensor = kornia.enhance.equalize_clahe(input_tensor, clip_limit=40.0, grid_size=(8, 8), slow_and_differentiable=False)
# output_tensor = kornia.enhance.equalize3d(input_tensor) #Implements Equalize function for a sequence of images using PyTorch ops based on uint8 format

import kornia
def scale_array_clahe(input_tensor, flag_stretch_histogram_first=True, quantiles=(0.01, 0.99), flag_return_quantile=False):

    #TODO: make this accept pytorch or numpy
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if type(input_tensor) == torch.Tensor:
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = input_tensor.quantile(quantiles[0])
            q2 = input_tensor.quantile(quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = input_tensor[0,0].quantile(quantiles[0])
            q2 = input_tensor[0,0].quantile(quantiles[1])

        ### Pre-stretch/Pre-scale before CLAHE: ###
        if flag_stretch_histogram_first:
            scaled_array = scale_array_from_range(input_tensor.clip(q1, q2),
                                                  min_max_values_to_clip=(q1, q2),
                                                  min_max_values_to_scale_to=(0, 1))
        else:
            scaled_array = scale_array_to_range(input_tensor)

        ### Perform CLAHE: ###
        scaled_array = kornia.enhance.equalize_clahe(scaled_array, clip_limit=40.0, grid_size=(8, 8), slow_and_differentiable=False)

        if flag_return_quantile:
            return scaled_array, (q1.cpu().item(), q2.cpu().item())
        else:
            return scaled_array

    else:
        #Numpy array
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = np.quantile(input_tensor, quantiles[0])
            q2 = np.quantile(input_tensor, quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = np.quantile(input_tensor[0,0], quantiles[0])
            q2 = np.quantile(input_tensor[0,0], quantiles[1])

        ### Pre-stretch/Pre-scale before CLAHE: ###
        if flag_stretch_histogram_first:
            scaled_array = scale_array_from_range(input_tensor.clip(q1, q2),
                                                  min_max_values_to_clip=(q1, q2),
                                                  min_max_values_to_scale_to=(0, 1))
        else:
            scaled_array = scale_array_to_range(input_tensor)


        ### Perform CLAHE by switching to pytorch and back: ###
        scaled_array = torch.tensor(scaled_array)
        scaled_array = torch_get_4D(scaled_array)
        scaled_array = kornia.enhance.equalize_clahe(scaled_array, clip_limit=40.0, grid_size=(8, 8), slow_and_differentiable=False)
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            scaled_array = scaled_array[0,0].cpu().numpy()
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            scaled_array = scaled_array[0].cpu().numpy()
        elif shape_len == 4:
            #[T,C,H,W]
            scaled_array = scaled_array.cpu().numpy()

        if flag_return_quantile:
            return scaled_array, (q1,q2)
        else:
            return scaled_array


def scale_array_to_range_logical_mask(input_tensor, logical_mask, output_range=(0,255)):
    output_tensor_valid_values = input_tensor[logical_mask]
    min_value = output_tensor_valid_values.min()
    max_value = output_tensor_valid_values.max()
    input_tensor_stretched = scale_array_to_range(input_tensor.clip(min_value, max_value), min_max_values_to_scale_to=output_range)
    return input_tensor_stretched

def scale_array_stretch_hist(input_tensor, quantiles=(0.01,0.99), min_max_values_to_scale_to=(0,1), flag_return_quantile=False):
    #TODO: make this accept pytorch or numpy
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if type(input_tensor) == torch.Tensor:
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = input_tensor.quantile(quantiles[0])
            q2 = input_tensor.quantile(quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = input_tensor[0].quantile(quantiles[0])
            q2 = input_tensor[0].quantile(quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = input_tensor[0,0].quantile(quantiles[0])
            q2 = input_tensor[0,0].quantile(quantiles[1])

        scaled_array = scale_array_from_range(input_tensor.clip(q1, q2),
                                              min_max_values_to_clip=(q1, q2),
                                              min_max_values_to_scale_to=min_max_values_to_scale_to)

        if flag_return_quantile:
            return scaled_array, (q1.cpu().item(), q2.cpu().item())
        else:
            return scaled_array

    else:
        #Numpy array
        if shape_len == 2 or (shape_len==3 and C<=3):
            #[H,W] or [C,H,W]
            q1 = np.quantile(input_tensor, quantiles[0])
            q2 = np.quantile(input_tensor, quantiles[1])
        elif shape_len == 3:
            #[T,H,W] - for estimation use only the first frame to not take too much time
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 4:
            #[T,C,H,W]
            q1 = np.quantile(input_tensor[0], quantiles[0])
            q2 = np.quantile(input_tensor[0], quantiles[1])
        elif shape_len == 5:
            #[B,T,C,H,W]  #TODO: better way would be to scale each batch according to it's quantiles, but fuck it right now
            q1 = np.quantile(input_tensor[0,0], quantiles[0])
            q2 = np.quantile(input_tensor[0,0], quantiles[1])

        scaled_array = scale_array_from_range(input_tensor.clip(q1,q2),
                                              min_max_values_to_clip=(q1,q2),
                                              min_max_values_to_scale_to=min_max_values_to_scale_to)
        if flag_return_quantile:
            return scaled_array, (q1,q2)
        else:
            return scaled_array
#########################################################################################################################################


#########################################################################################################################################
####  Image ColorSpace Transforms: ####
#######################################
from RapidBase.Utils.IO.String_utils import *

def print_info(input_tensor, input_name='input tensor', number_of_digits=2):
    input_shape = input_tensor.shape
    data_type = input_tensor.dtype
    array_type = type(input_tensor)
    if type(input_tensor) == np.ndarray:
        array_type = 'np.array'
        min_value = np.min(input_tensor)
        max_value = np.max(input_tensor)
        mean_value = np.mean(input_tensor)
        median_value = np.median(input_tensor)
        std_value = np.std(input_tensor)
    elif type(input_tensor) == torch.Tensor:
        array_type = 'torch.Tensor'
        min_value = input_tensor.min()
        max_value = input_tensor.max()
        mean_value = input_tensor.mean()
        median_value = input_tensor.median()
        std_value = input_tensor.std()
    print_string = input_name + ', array type = ' + array_type + ' shape = ' + str(input_shape) + ', dtype = ' + str(data_type)
    print(print_string)
    print_string = input_name + \
                   ' (min,max) = (' + decimal_notation(min_value, number_of_digits) + ', ' + decimal_notation(max_value, number_of_digits) + ')' + \
                   ', mean = ' + decimal_notation(mean_value, number_of_digits) + \
                   ', median = ' + decimal_notation(median_value, number_of_digits) + ', std = ' + decimal_notation(std_value, number_of_digits)
    print(print_string)

def get_info_torch(input_tensor, input_name='input tensor', number_of_digits=2):
    return print_info(input_tensor, input_name=input_name, number_of_digits=number_of_digits)

def print_info_torch(input_tensor, input_name='input tensor', number_of_digits=2):
    return print_info(input_tensor, input_name=input_name, number_of_digits=number_of_digits)

def torch_reshape_image_end(input_tensor, H, W):
    ### convert from [B,C,H*W] -> [B,C,H,W]: ###
    ### input_tensor.shape[-1] = H * W
    (B1,T1,C1,H1,W1), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    new_shape_tuple = tuple(np.concatenate((shape_vec[0:-1], [H], [W]), -1))
    return input_tensor.reshape(new_shape_tuple)


def RGB2BW(input_image):
    if len(input_image.shape) == 2:
        return input_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 3:
            grayscale_image = 0.299 * input_image[0:1, :, :] + 0.587 * input_image[1:2, :, :] + 0.114 * input_image[2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1] + 0.587 * input_image[:, :, 1:2] + 0.114 * input_image[:, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, 0:1] + 0.587 * input_image[:, :, :, 1:2] + 0.114 * input_image[:, :, :, 2:3]
        else:
            grayscale_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 3:
            grayscale_image = 0.299 * input_image[:, :, :, :, 0:1] + 0.587 * input_image[:, :, :, :, 1:2] + 0.114 * input_image[:, :, :, :, 2:3]
        else:
            grayscale_image = input_image

    return grayscale_image

def RGB2BW_interleave_C(input_image):  #Only for tensors for now, and if you know(!) they're RGB
    #(*). Unless given another input of C or T, this function cannot know whether it was really given a RGB image, SO IT ASSUMES IT!!! YOU NEED TO MAKE SURE!!!!!
    if len(input_image.shape)==3:
        TxC,H,W = input_image.shape
        new_C = TxC//3
        input_image = input_image.reshape(3, new_C, H, W)
        grayscale_image = 0.299 * input_image[0:1, :, :, :] + 0.587 * input_image[1:2, :, :, :] + 0.114 * input_image[2:3, :, :, :]

    elif len(input_image.shape)==4:
        B, TxC, H, W = input_image.shape
        new_C = TxC // 3
        input_image = input_image.reshape(B, 3, new_C, H, W)
        grayscale_image = 0.299 * input_image[:, 0:1, :, :, :] + 0.587 * input_image[:, 1:2, :, :, :] + 0.114 * input_image[:, 2:3, :, :, :]

    return grayscale_image

def RGB2BW_interleave_T(input_image):  #Only for tensors for now, and if you know(!) they're RGB
    if len(input_image.shape)==4:    #[T,C,H,W]
        T,C,H,W = input_image.shape
        if C == 3:
            grayscale_image = 0.299 * input_image[:, 0:1, :, :] + 0.587 * input_image[:, 1:2, :, :] + 0.114 * input_image[:, 2:3, :, :]
        else:
            grayscale_image = input_image

    elif len(input_image.shape)==5:  #[B,T,C,H,W]
        B, T, C, H, W = input_image.shape
        if C == 3:
            grayscale_image = 0.299 * input_image[:, :, 0:1, :, :] + 0.587 * input_image[:, :, 1:2, :, :] + 0.114 * input_image[:, :, 2:3, :, :]
        else:
            grayscale_image = input_image

    return grayscale_image


def BW2RGB_interleave_C(input_image):
    ### For Torch Tensors!: ###
    if len(input_image.shape) == 3:
        RGB_image = torch.repeat_interleave(input_image, 3, 0)

    elif len(input_image.shape) == 4:
        RGB_image = torch.repeat_interleave(input_image, 3, 1)

    elif len(input_image.shape) == 5:
        RGB_image = torch.cat([input_image]*3, 2)

    return RGB_image

def BW2RGB_interleave_T(input_image):
    ### For Torch Tensors!: ###
    if len(input_image.shape) == 4:  #[T,C,H,W]
        RGB_image = torch.repeat_interleave(input_image, 3, 1)

    elif len(input_image.shape) == 5:  #[B,T,C,H,W]
        RGB_image = torch.repeat_interleave(input_image, 3, 2)

    return RGB_image


def BW2RGB_MultipleFrames(input_image, flag_how_to_concat='C'):
    if flag_how_to_concat == 'C':
        RGB_image = BW2RGB_interleave_C(input_image)
    else:
        RGB_image = BW2RGB_interleave_T(input_image)
    return RGB_image

def RGB2BW_MultipleFrames(input_image, flag_how_to_concat='C'):
    if flag_how_to_concat == 'C':
        grayscale_image = RGB2BW_interleave_C(input_image)
    else:
        grayscale_image = RGB2BW_interleave_T(input_image)
    return grayscale_image

def BW2RGB(input_image):
    ### For Both Torch Tensors and Numpy Arrays!: ###
    # Actually... we're not restricted to RGB....
    if len(input_image.shape) == 2:
        if type(input_image) == torch.Tensor:
            RGB_image = input_image.unsqueeze(0)
            RGB_image = torch.cat([RGB_image,RGB_image,RGB_image], 0)
        elif type(input_image) == np.ndarray:
            RGB_image = np.atleast_3d(input_image)
            RGB_image = np.concatenate([RGB_image, RGB_image, RGB_image], -1)
        return RGB_image

    if len(input_image.shape) == 3:
        if type(input_image) == torch.Tensor and input_image.shape[0] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 0)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 4:
        if type(input_image) == torch.Tensor and input_image.shape[1] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 1)
        elif type(input_image) == np.ndarray and input_image.shape[-1] == 1:
            RGB_image = np.concatenate([input_image, input_image, input_image], -1)
        else:
            RGB_image = input_image

    elif len(input_image.shape) == 5:
        if type(input_image) == torch.Tensor and input_image.shape[2] == 1:
            RGB_image = torch.cat([input_image, input_image, input_image], 2)
        else:
            RGB_image = input_image

    return RGB_image




def convert_image_list_color_space(image_list, number_of_input_channels, target_type):
    # conversion among BGR, gray and y
    if number_of_input_channels == 3 and target_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in image_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif number_of_input_channels == 3 and target_type == 'y':  # BGR to y?????
        y_list = [bgr2ycbcr(img, only_y=True) for img in image_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif number_of_input_channels == 1 and target_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in image_list]
    else:
        return image_list


def numpy_unsqueeze(input_tensor, dim=-1):
    return np.expand_dims(input_tensor, dim)

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)
#########################################################################################################################################









########################################################################################################################################################################################################################################################################################
####################
# Auxiliaries:
####################
def any_data_type_to_torch(current_input, device):
    if np.isscalar(current_input):
        return torch.Tensor([current_input]).to(device)
    elif type(current_input) == np.ndarray:
        return torch.Tensor(current_input).to(device)
    elif type(current_input) == list:
        return list_to_torch(current_input, device)
    elif type(current_input) == tuple:
        return tuple_to_torch(current_input, device)
    elif type(current_input) == torch.Tensor:
        return current_input

def any_data_type_to_numpy(current_input):
    if np.isscalar(current_input):
        return np.array([current_input])
    elif type(current_input) == np.ndarray:
        return current_input
    elif type(current_input) == list:
        return list_to_numpy(current_input)
    elif type(current_input) == tuple:
        return tuple_to_numpy(current_input)
    elif type(current_input) == torch.Tensor:
        return current_input.cpu().numpy()

def cv2_to_torch(input_image, device='cpu'):
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2)
    input_image = input_image[...,[2,1,0]] #BGR->RGB (cv2 returns BGR by default)
    input_image = np.transpose(input_image, (2,0,1)) #[H,W,C]->[C,H,W]
    input_image = torch.from_numpy(input_image).float() #to float32
    input_image = input_image.unsqueeze(0) #[C,H,W] -> [1,C,H,W]
    input_image = input_image.to(device) #to cpu or gpu (should this be global or explicit?)
    return input_image

def list_to_torch(input_list, device='cpu'):
    for index in np.arange(len(input_list)):
        input_list[index] = any_data_type_to_torch(input_list[index], device)
    return input_list

def list_to_numpy(input_list):
    for index in np.arange(len(input_list)):
       input_list[index] = any_data_type_to_numpy(input_list[index])
    return input_list

def make_tuple_int(input_tuple):
    input_tuple = (np.int(np.round(input_tuple[0])), np.int(np.round(input_tuple[1])))
    return input_tuple

def tuple_to_torch(input_tuple, device='cpu'):
    return tuple(any_data_type_to_torch(input_tuple[i], device) for i in np.arange(len(input_tuple)))

def tuple_to_numpy(input_tuple):
    return tuple(any_data_type_to_numpy(input_tuple[i]) for i in np.arange(len(input_tuple)))


def whole_dict_to_torch(input_dict, device='cpu'):
    for key, value in input_dict:
        input_dict[key] = any_data_type_to_torch(input_dict[key], device)
    return input_dict

def whole_torch_dict_to_numpy(input_dict):
    for key, value in input_dict:
        input_dict[key] = any_data_type_to_numpy(input_dict[key])
    return input_dict

def torch_list_to_device(input_list, device='cpu'):
    for index in np.arange(len(input_list)):
        if type(input_list[index]) == np.array:
            input_list[index] = input_list[index].to(device)
    return input_list


def numpy_to_torch(input_image, device='cpu', flag_unsqueeze=False):
    #Assumes "Natural" RGB form to there's no BGR->RGB conversion, can also accept BW image!:
    if input_image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        input_image = np.expand_dims(input_image, axis=2) #[H,W]->[H,W,1]
    if input_image.ndim == 3:
        input_image = np.transpose(input_image, (2, 0, 1))  # [H,W,C]->[C,H,W]
    elif input_image.ndim == 4:
        input_image = np.transpose(input_image, (0, 3, 1, 2)) #[T,H,W,C] -> [T,C,H,W]
    input_image = torch.from_numpy(input_image.astype(np.float)).float().to(device) # to float32

    if flag_unsqueeze:
        input_image = input_image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    return input_image

def torch_to_numpy(input_tensor):
    if type(input_tensor) == torch.Tensor:
        (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
        input_tensor = input_tensor.cpu().data.numpy()
        if shape_len == 2:
            #[H,W]
            return input_tensor
        elif shape_len == 3:
            #[C,H,W] -> [H,W,C]
            return np.transpose(input_tensor, [1,2,0])
        elif shape_len == 4:
            #[T,C,H,W] -> [T,H,W,C]
            return np.transpose(input_tensor, [0,2,3,1])
        elif shape_len == 5:
            #[B,T,C,H,W] -> [B,T,H,W,C]
            return np.transpose(input_tensor, [0,1,3,4,2])
    return input_tensor


def torch_flatten_image(input_tensor, flag_make_Nx1=True, order='C'):
    #TODO: this assumes i want to flatten everything but the batch dimensions...which seems legit
    # torch_variable = torch_variable.view(torch_variable.size(0), -1)
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if order == 'F':
        input_tensor = einops.rearrange(input_tensor, '... h w -> ... w h')
        input_tensor = input_tensor.flatten()
    elif order == 'C':
        input_tensor = input_tensor.flatten()

    if flag_make_Nx1:
        input_tensor = input_tensor.unsqueeze(-1)

    return input_tensor

def torch_reshape_flattened_image(input_tensor_flattened, final_shape_tuple, order='C'):
    H, W = final_shape_tuple
    if order == 'F':
        if len(input_tensor_flattened.shape) == 2:  #[N,1]
            input_tensor_flattened = einops.rearrange(input_tensor_flattened, '(h w) 1 -> w h', h=W, w=H)
        else:
            input_tensor_flattened = einops.rearrange(input_tensor_flattened, '(h w) -> w h', h=W, w=H)
    elif order == 'C':
        if len(input_tensor_flattened.shape) == 2:  #[N,1]
            input_tensor_flattened = einops.rearrange(input_tensor_flattened, '(h w) 1 -> h w', h=H, w=W)
        else:
            input_tensor_flattened = einops.rearrange(input_tensor_flattened, '(h w) -> h w', h=H, w=W)

    return input_tensor_flattened


def crop_size_after_homography(h_matrix: Tensor, H: int, W: int, add_just_in_case: int = 5):
    """
    Returns crop sizes for a transformed matrix after applying homography to it
    Args:
        h_matrix: homography matrix to apply (enables stacking)
        H: matrix's H
        W: matrix's W
        add_just_in_case: additive value to the crp in each margin, just in case :)

    Returns: crop_h_start, crop_h_end, crop_w_start, crop_w_end

    """

    # Generate 4 point locations close to the 4 corners of the image (homogeneous coordinates)
    o_o = torch.tensor([0, 0, 1]).float().to(h_matrix.device)
    o_w = torch.Tensor([W, 0, 1]).float().to(h_matrix.device)
    h_o = torch.Tensor([0, H, 1]).float().to(h_matrix.device)
    h_w = torch.Tensor([W, H, 1]).float().to(h_matrix.device)

    # find shifted points according to the homography matrix (homogeneous coordinates)
    shifted_o_o = h_matrix @ o_o
    shifted_o_w = h_matrix @ o_w
    shifted_h_o = h_matrix @ h_o
    shifted_h_w = h_matrix @ h_w

    # transfer original points to non-homogeneous coordinates
    o_o = o_o[..., :2]
    o_w = o_w[..., :2]
    h_o = h_o[..., :2]
    h_w = h_w[..., :2]

    # transfer to non homogeneous coordinates
    shifted_o_o = shifted_o_o[..., :2] / shifted_o_o[..., 2:]
    shifted_o_w = shifted_o_w[..., :2] / shifted_o_w[..., 2:]
    shifted_h_o = shifted_h_o[..., :2] / shifted_h_o[..., 2:]
    shifted_h_w = shifted_h_w[..., :2] / shifted_h_w[..., 2:]

    # crops of each side of the image according to the expected shifts
    h_start_1 = (o_o[..., 1] - shifted_o_o[..., 1]).clamp(0)
    h_start_2 = (o_w[..., 1] - shifted_o_w[..., 1]).clamp(0)
    h_start = torch.ceil(torch.max(h_start_1, h_start_2) + add_just_in_case).int()

    h_end_1 = H - (shifted_h_o[..., 1] - h_o[..., 1]).clamp(0)
    h_end_2 = H - (shifted_h_w[..., 1] - h_w[..., 1]).clamp(0)
    h_end = torch.ceil(torch.min(h_end_1, h_end_2) - add_just_in_case).int()

    w_start_1 = (o_o[..., 0] - shifted_o_o[..., 0]).clamp(0)
    w_start_2 = (h_o[..., 0] - shifted_h_o[..., 0]).clamp(0)
    w_start = torch.ceil(torch.max(w_start_1, w_start_2) + add_just_in_case).int()

    w_end_1 = W - (shifted_o_w[..., 0] - o_w[..., 0]).clamp(0)
    w_end_2 = W - (shifted_h_w[..., 0] - h_w[..., 0]).clamp(0)
    w_end = torch.ceil(torch.min(w_end_1, w_end_2) - add_just_in_case).int()

    return h_start, h_end, w_start, w_end

def crop_tensor_after_homography(input_tensor, H_matrix, add_just_in_case=5):
    H,W = input_tensor.shape[-2:]
    h_start, h_end, w_start, w_end = crop_size_after_homography(H_matrix, H, W, add_just_in_case)
    return input_tensor[...,h_start:h_end, w_start:w_end]

def torch_reshape_image(input_tensor, final_shape_tuple, order='C'):
    H, W = final_shape_tuple
    if order == 'F':
        input_tensor = input_tensor.t().contiguous()
        input_tensor = torch.reshape(input_tensor, (final_shape_tuple[1], final_shape_tuple[0])).t()

    elif order == 'C':
        input_tensor = torch.reshape(input_tensor, final_shape_tuple)

    return input_tensor

def numpy_flatten(input_array, flag_make_Nx1=True, order='C'):
    if flag_make_Nx1:
        return input_array.flatten(order).reshape(np.prod(input_array.shape), -1)
    else:
        return input_array.flatten(order)
################################################################################################################################################


################################################################################################################################################
### Crop Functions: ####

def crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H,W = images.shape
        C = 1
    elif len(images.shape) == 3:
        C,H,W = images.shape
    elif len(images.shape) == 4:
        T, C, H, W = images.shape  # No Batch Dimension
    else:
        B, T, C, H, W = images.shape  # "Full" with Batch Dimension

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) == list or type(crop_size_tuple_or_scalar) == tuple:
        cropH = crop_size_tuple_or_scalar[0]
        cropW = crop_size_tuple_or_scalar[1]
    else:
        cropH = crop_size_tuple_or_scalar
        cropW = crop_size_tuple_or_scalar

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:  #center
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        end_x = W
        end_y = H
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###
    if len(images.shape) == 2:
        return pad_torch_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_torch_batch(images[:, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 4:
        return pad_torch_batch(images[:, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    else:
        return pad_torch_batch(images[:, :, :, start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)


def crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### Initial: ###
    if len(images.shape) == 2:
        H, W = images.shape
    elif len(images.shape) == 3:
        H, W, C = images.shape #BW batch
    else:
        T, H, W, C = images.shape #RGB/NOT-BW batch

    ### Assign cropW and cropH: ###
    if type(crop_size_tuple_or_scalar) == list or type(crop_size_tuple_or_scalar) == tuple:
        cropW = crop_size_tuple_or_scalar[1]
        cropH = crop_size_tuple_or_scalar[0]
    else:
        cropW = crop_size_tuple_or_scalar
        cropH = crop_size_tuple_or_scalar
    cropW = min(cropW, W)
    cropH = min(cropH, H)

    ### Decide On Crop Size According To Input: ###
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropW = cropW
    else:
        cropW = min(cropW, W)
    if (cropW != np.inf or cropW != torch.inf) and flag_pad_if_needed:
        cropH = cropH
    else:
        cropH = min(cropH, W)

    ### Get Start-Stop Indices To Crop: ###
    if crop_style == 'random':
        if cropW < W:
            start_W = np.random.randint(0, W - cropW)
        else:
            start_W = 0
        if cropH < H:
            start_H = np.random.randint(0, H - cropH)
        else:
            start_H = 0
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    elif crop_style == 'predetermined':
        start_H = start_H
        start_W = start_W
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)
    else:
        mat_in_rows_excess = H - cropH
        mat_in_cols_excess = W - cropW
        start = 0
        end_x = W
        end_y = H
        start_W = int(start + mat_in_cols_excess / 2)
        start_H = int(start + mat_in_rows_excess / 2)
        stop_W = start_W + cropW
        stop_H = start_H + cropH
        stop_W = min(stop_W, W)
        stop_H = min(stop_H, H)

    ### Crop Images (consistently between images/frames): ###    imshow_torch(tensor1_crop)
    if len(images.shape) == 2:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W], crop_size_tuple_or_scalar)
    elif len(images.shape) == 3:
        return pad_numpy_batch(images[start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)
    else:
        return pad_numpy_batch(images[:, start_H:stop_H, start_W:stop_W, :], crop_size_tuple_or_scalar)


def crop_tensor(images, crop_size_tuple_or_scalar, crop_style='center', start_H=-1, start_W=-1, flag_pad_if_needed=True):
    ### crop_style = 'center', 'random', 'predetermined'
    if type(images) == torch.Tensor:
        return crop_torch_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W)
    else:
        return crop_numpy_batch(images, crop_size_tuple_or_scalar, crop_style, start_H, start_W)



def pad_numpy_batch(input_arr: np.array, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(a, pad_size):  # create a rigid shape for padding in dims HWC
            pad_start = np.floor(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, a.shape[-3: -1]) / 2).astype(int)
            return (pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (0, 0)

        return np.array([np.pad(a, pad_shape(a, pad_size), 'constant', constant_values=0) for a in input_arr])
    else:
        return None


def pad_torch_batch(input_tensor: Tensor, pad_size: Tuple[int, int], pad_style='center'):
    # TODO sometime support other pad styles.
    # expected BHWC array
    if pad_style == 'center':
        def pad_shape(t, pad_size):  # create a rigid shape for padding in dims CHW
            pad_start = np.floor(np.subtract(pad_size, t.shape[-2:]) / 2).astype(int)
            pad_end = np.ceil(np.subtract(pad_size, t.shape[-2:]) / 2).astype(int)
            return pad_start[1], pad_end[1], pad_start[0], pad_end[0]

        return torch.nn.functional.pad(input_tensor, pad_shape(input_tensor, pad_size), mode='constant')
    else:
        return None

start = -1
end = -1


def get_list_of_crops_numpy(input_image=None, flag_get_all_crops=False, number_of_random_crops=10, crop_size_x=None, crop_size_y=None):
    input_image = np.atleast_3d(input_image)
    frame_height, frame_width, channels = input_image.shape
    if crop_size_x is None or crop_size_x == np.inf:
        crop_size_x = frame_width
    if crop_size_y is None or crop_size_y == np.inf:
        crop_size_y = frame_height

    number_of_width_crops = max(frame_width // crop_size_x, 1)
    number_of_height_crops = max(frame_height // crop_size_y, 1)
    current_crop_size_height = min(crop_size_y, frame_height)
    current_crop_size_width = min(crop_size_x, frame_width)

    final_list = list()

    if flag_get_all_crops:
        for i in np.arange(number_of_width_crops):
            for j in np.arange(number_of_height_crops):
                start_x = i*crop_size_x
                start_y = j*crop_size_y
                stop_x = start_x + crop_size_x
                stop_y = start_y + crop_size_y
                final_list.append(input_image[start_y:stop_y, start_x:stop_x, :])
    else:
        for i in np.arange(number_of_random_crops):
            random_start_x = random.randint(0, frame_width-crop_size_x)
            random_start_y = random.randint(0, frame_height-crop_size_y)
            random_stop_x = random_start_x + crop_size_x
            random_stop_y = random_start_y + crop_size_y
            current_crop = input_image[random_start_y:random_stop_y, random_start_x:random_stop_x, :]
            final_list.append(current_crop)

    return final_list
################################################################################################################################################



################################################################################################################################################
def get_max_correct_form_torch(input_tensor, dim_to_leave):
    ### Assuming input_tensor.shape = [B,C,H,W] or 4D Tensor in any case: ###

    if dim_to_leave==0: #get max value for each batch example (max among all values of that batch example including all channels)
        values, indices = input_tensor.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        values = values.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #Return to pytorch format: [B,1,1,1]
    if dim_to_leave==1:  # get max value for each channel (max among all values of including batch dimension)
        values, indices = input_tensor.max(0)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        if len(values.shape) > 1:
            values, indices = values.max(1)
        values = values.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]
    if dim_to_leave==(0,1) or dim_to_leave==[0,1]:  # get max value of each feature map
        if len(values.shape) > 2:
            values, indices = values.max(2)
        if len(values.shape) > 1:
            values, indices = values.max(2)
        values = values.unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]

    return values



def get_min_correct_form_torch(input_tensor, dim_to_leave):
    ### Assuming input_tensor.shape = [B,C,H,W] or 4D Tensor in any case: ###

    if dim_to_leave==0: #get max value for each batch example (max among all values of that batch example including all channels)
        values, indices = input_tensor.min(1)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        values = values.unsqueeze(1).unsqueeze(2).unsqueeze(3)  #Return to pytorch format: [B,1,1,1]
    if dim_to_leave==1:  # get max value for each channel (max among all values of including batch dimension)
        values, indices = input_tensor.min(0)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        if len(values.shape) > 1:
            values, indices = values.min(1)
        values = values.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]
    if dim_to_leave==(0,1) or dim_to_leave==[0,1]:  # get max value of each feature map
        if len(values.shape) > 2:
            values, indices = values.min(2)
        if len(values.shape) > 1:
            values, indices = values.min(2)
        values = values.unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [1,C,1,1]

    return values


def max_of_two_torch(input1, input2):
    return torch.max(input1, input2)
def min_of_two_torch(input1, input2):
    return torch.min(input1, input2)
def closer_to_1_torch(input1, input2):
    residual1 = torch.abs(input1-1)
    residual2 = torch.abs(input2-1)
    if residual1 > residual2:
        return input2
    else:
        return input1

def vec_to_pytorch_format(input_vec, dim_to_put_scalar_in=0):
    if dim_to_put_scalar_in == 1: #channels
        output_tensor = torch.Tensor(input_vec).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    if dim_to_put_scalar_in == 0: #batches
        output_tensor = torch.Tensor(input_vec).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return output_tensor
################################################################################################################################################



################################################################################################################################################
### Robust and Masked Statistical Estimators: ###
def torch_median_absolute_devation(input_tensor, dim=0):
    return (input_tensor - input_tensor.median(dim, True)[0]).abs().median(dim, True)[0]


def torch_robust_std_1(input_tensor, dim=0):
    return 1.4826 * torch_median_absolute_devation(input_tensor, dim)


def torch_robust_std_2(input_tensor, dim=0):
    MAD = torch_median_absolute_devation(input_tensor, dim)
    logical_mask = (input_tensor - input_tensor.median(dim, True)[0]).abs() <= MAD*2
    return torch_masked_std(input_tensor, logical_mask, dim)


def torch_masked_median(x, mask, dim=0):
    """Compute the median of tensor x along dim, ignoring values where mask is False.
    x and mask need to be broadcastable.

    Args:
        x (Tensor): Tensor to compute median of.
        mask (BoolTensor): Same shape as x with True where x is valid and False
            where x should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.
        dim (int, optional): Dimension to take median of. Defaults to 0.

    Returns:
        Tensor: Same shape as x, except dimension dim reduced.
    """
    # uncomment this assert for safety but might impact performance
    # assert (
    #     mask.sum(dim=dim).ne(0).all()
    # ), "mask should not be all False in any column, causes zero division"
    x_nan = x.float().masked_fill(~mask, float("nan"))
    x_median, _ = x_nan.nanmedian(dim=dim)
    return x_median.unsqueeze(dim)


def torch_masked_mean(x, mask, dim=0):
    """Compute the median of tensor x along dim, ignoring values where mask is False.
    x and mask need to be broadcastable.

    Args:
        x (Tensor): Tensor to compute median of.
        mask (BoolTensor): Same shape as x with True where x is valid and False
            where x should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.
        dim (int, optional): Dimension to take median of. Defaults to 0.

    Returns:
        Tensor: Same shape as x, except dimension dim reduced.
    """
    # uncomment this assert for safety but might impact performance
    # assert (
    #     mask.sum(dim=dim).ne(0).all()
    # ), "mask should not be all False in any column, causes zero division"
    x_nan = x.float().masked_fill(~mask, float("nan"))
    x_median, _ = x_nan.nanmean(dim=dim)
    return x_median.unsqueeze(dim)


def torch_masked_sum(x, mask, dim=0):
    """Compute the median of tensor x along dim, ignoring values where mask is False.
    x and mask need to be broadcastable.

    Args:
        x (Tensor): Tensor to compute median of.
        mask (BoolTensor): Same shape as x with True where x is valid and False
            where x should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.
        dim (int, optional): Dimension to take median of. Defaults to 0.

    Returns:
        Tensor: Same shape as x, except dimension dim reduced.
    """
    # uncomment this assert for safety but might impact performance
    # assert (
    #     mask.sum(dim=dim).ne(0).all()
    # ), "mask should not be all False in any column, causes zero division"
    x_nan = x.float().masked_fill(~mask, float("nan"))
    x_median, _ = x_nan.nansum(dim=dim)
    return x_median.unsqueeze(dim)


def torch_masked_std(x, mask, dim=0):
    """Compute the median of tensor x along dim, ignoring values where mask is False.
    x and mask need to be broadcastable.

    Args:
        x (Tensor): Tensor to compute median of.
        mask (BoolTensor): Same shape as x with True where x is valid and False
            where x should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.
        dim (int, optional): Dimension to take median of. Defaults to 0.

    Returns:
        Tensor: Same shape as x, except dimension dim reduced.
    """
    # uncomment this assert for safety but might impact performance
    # assert (
    #     mask.sum(dim=dim).ne(0).all()
    # ), "mask should not be all False in any column, causes zero division"
    x_nan = x.float().masked_fill(~mask, float("nan"))

    # x_std, _ = x_nan.nanstd(dim=dim)  #TODO: upgrade torch version
    N_per_pixel = mask.sum(dim=dim)
    x_std = torch.nansum((x_nan - torch.nanmean(x_nan, dim=dim)).abs()**2, dim=dim) / torch.sqrt(N_per_pixel)

    return x_std.unsqueeze(dim)
################################################################################################################################################




################################################################################################################################################
import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Downsample_AntiAliasing(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample_AntiAliasing, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
################################################################################################################################################



################################################################################################################################################
### Padding Functions: ###
def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


# Get needed padding to keep image size the same after a convolution layer with a certain kernel size and dilation (TODO: what about strid?1!?!)
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def pad_with_ones_frame_torch(input_tensor, dim=-1, padding_size=(1,1)):
    ### padding_size[0] padding at the beginning of the dim, padding[1] padding at the end of the dim
    # input_tensor = torch.ones((4,3,64,100))
    # dim = -3
    # padding_size = (1,2)

    dims_string = 'BTCHW'
    dims_string_final = dims_string[0:dim] + dims_string[dim+1:]
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    input_dims = dims_string[-shape_len:]
    input_dims = input_dims[dim]
    frame_to_concat = torch.ones_like(input_tensor)
    frame_to_concat = torch.index_select(frame_to_concat, dim, torch.LongTensor([0]))
    if padding_size[0] > 0:
        frame_to_concat_1 = torch.cat([frame_to_concat]*padding_size[0], dim)
        output_tensor = torch.cat((frame_to_concat_1, input_tensor), dim)
    else:
        output_tensor = input_tensor
    if padding_size[1] > 0:
        frame_to_concat_2 = torch.cat([frame_to_concat]*padding_size[1], dim)
        output_tensor = torch.cat((output_tensor, frame_to_concat_2), dim)

    return output_tensor

def pad_with_zeros_frame_torch(input_tensor, dim=-1, padding_size=(1,1)):
    ### padding_size[0] padding at the beginning of the dim, padding[1] padding at the end of the dim
    # input_tensor = torch.ones((4,3,64,100))
    # dim = -3
    # padding_size = (1,2)

    dims_string = 'BTCHW'
    dims_string_final = dims_string[0:dim] + dims_string[dim+1:]
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    input_dims = dims_string[-shape_len:]
    input_dims = input_dims[dim]
    frame_to_concat = torch.zeros_like(input_tensor)
    frame_to_concat = torch.index_select(frame_to_concat, dim, torch.LongTensor([0]))
    if padding_size[0] > 0:
        frame_to_concat_1 = torch.cat([frame_to_concat]*padding_size[0], dim)
        output_tensor = torch.cat((frame_to_concat_1, input_tensor), dim)
    else:
        output_tensor = input_tensor
    if padding_size[1] > 0:
        frame_to_concat_2 = torch.cat([frame_to_concat]*padding_size[1], dim)
        output_tensor = torch.cat((output_tensor, frame_to_concat_2), dim)

    return output_tensor


def get_valid_padding(H, W, kernel_size, stride):
    # The total padding applied along the height and width is computed as:

    if (H % stride[0] == 0):
        pad_along_height = max(kernel_size[0] - stride[0], 0)
    else:
        pad_along_height = max(kernel_size[0] - (H % stride[0]), 0)
    if (W % stride[1] == 0):
        pad_along_width = max(kernel_size[1] - stride[1], 0)
    else:
        pad_along_width = max(kernel_size[1] - (W % stride[1]), 0)

    # Finally, the padding on the top, bottom, left and right are:
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding_vec = (pad_left, pad_right, pad_top, pad_bottom)
    return padding_vec

def get_valid_padding_on_tensor(input_tensor, kernel_size, stride):
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if (H % stride[0] == 0):
        pad_along_height = max(kernel_size[0] - stride[0], 0)
    else:
        pad_along_height = max(kernel_size[0] - (H % stride[0]), 0)
    if (W % stride[1] == 0):
        pad_along_width = max(kernel_size[1] - stride[1], 0)
    else:
        pad_along_width = max(kernel_size[1] - (W % stride[1]), 0)

    # Finally, the padding on the top, bottom, left and right are:
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding_vec = (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(input_tensor, padding_vec)
################################################################################################################################################



################################################################################################################################################
### Pooling Layers ###
class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

# input_tensor = torch.Tensor(randn(1,3,100,100))
# layer = MedianPool2d(kernel_size=2, stride=2, padding=0, same=True)
# output_tensor = layer(input_tensor)
# print(output_tensor.shape)


class StridedPool2d(nn.Module):
    def __init__(self, stride=1):
        super(StridedPool2d, self).__init__()
        self.stride = stride

    def forward(self, x):
        x_strided = x[:,:,0:-1:self.stride,0:-1:self.stride]
        return x_strided

# ### Use Example: ###
# input_tensor = torch.Tensor(randn(1,3,100,100))
# layer = StridedPool2d(stride=2)
# output_tensor = layer(input_tensor)
# print(output_tensor.shape)

class MinPool2d(nn.Module):
    def __init__(self, kernel_size=1, stride=1):
        super(MinPool2d, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.layer = nn.MaxPool2d(kernel_size, stride=stride)

    def forward(self, x):
        x_strided = -self.layer(-x)
        return x_strided

# ### Use Example: ###
# input_tensor = torch.Tensor(randn(1,3,100,100))
# layer = MinPool2d(2,2)
# output_tensor = layer(input_tensor)
# print(output_tensor.shape)


### Detail preserving pooling: ####
class pospowbias(nn.Module):
    def __init__(self):
        super(pospowbias, self).__init__()
        self.Lambda = nn.Parameter(torch.zeros(1))
        self.Alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.Alpha + x**torch.exp(self.Lambda)  #raising to the power of exp(Lambda) instead of simply to the power of Lambda because we need the power to be positive and putting it in an exp insures that


class Downsample1D_AntiAlias(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D_AntiAlias, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer
################################################################################################################################################


################################################################################################################################################
### Fast Binning Functions: ###
def fast_binning_2D_PixelBinning(input_tensor, binning_size, overlap_size):
    # input_tensor = torch.randn((100,512,512))
    # binning_size = 10
    # overlap_size = 0

    T,H,W = input_tensor.shape
    step_size = binning_size - overlap_size
    H_final = 1 + np.int16((H - binning_size) / step_size)
    W_final = 1 + np.int16((W - binning_size) / step_size)

    column_cumsum = torch.cat((torch.zeros((T, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, 2)), 2) #pad to the left, maybe there's a faster way
    T1,H1,W1 = column_cumsum.shape
    column_binning = column_cumsum[:, :, binning_size:W1:step_size] - column_cumsum[:, :, 0:W1 - binning_size:step_size]

    row_cumsum = torch.cat((torch.zeros(T,1,W_final).to(input_tensor.device), torch.cumsum(column_binning,1)), 1)
    T2,H2,W2 = row_cumsum.shape
    binned_matrix_final = row_cumsum[:, binning_size:H2:step_size,:] - row_cumsum[:, 0:H2-binning_size:step_size]

    return binned_matrix_final


def fast_binning_2D_AvgPool2d(input_tensor, binning_size_tuple, overlap_size_tuple, flag_sum_instead_of_average=False):
    #(*). since this uses nn.AvgPool2d it is easier and more natural for it to return a tensor THE SAME SIZE AS THE ORIGINAL if stride==1, unlike the above functions
    # input_tensor = torch.randn((100,512,512))
    # binning_size_tuple = (10,21)
    # stride_size_tuple = (1,1)
    # overlap_size_tuple = (0,0)

    ### Expecfting tuple: ###
    binning_size_H, binning_size_W = binning_size_tuple
    overlap_size_H, overlap_size_W = overlap_size_tuple
    stride_size_H = binning_size_H - overlap_size_H
    stride_size_W = binning_size_W - overlap_size_W
    stride_size_tuple = (stride_size_H, stride_size_W)
    # stride_size_H, stride_size_W = stride_size_tuple

    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    average_pooling_layer = nn.AvgPool2d(binning_size_tuple, stride_size_tuple)
    input_tensor_padded = get_valid_padding_on_tensor(input_tensor, binning_size_tuple, stride_size_tuple)
    if len(input_tensor_padded.shape) == 5:
        flag_origami = True
        B1,T1,C1,H1,W1 = input_tensor_padded.shape
        input_tensor_padded = input_tensor_padded.view(B1*T1,C1,H1,W1)
    else:
        flag_origami = False
    binned_matrix_final = average_pooling_layer(input_tensor_padded)
    if flag_origami:
        binned_matrix_final = binned_matrix_final.view(B,T,C,H,W)

    if flag_sum_instead_of_average:
        binned_matrix_final *= binning_size_tuple[0] * binning_size_tuple[1]
    return binned_matrix_final


def fast_binning_2D_MaxPool2d(input_tensor, binning_size_tuple, overlap_size_tuple):
    #(*). since this uses nn.AvgPool2d it is easier and more natural for it to return a tensor THE SAME SIZE AS THE ORIGINAL if stride==1, unlike the above functions
    # input_tensor = torch.randn((100,512,512))
    # binning_size_tuple = (10,21)
    # stride_size_tuple = (1,1)
    # overlap_size_tuple = (0,0)

    ### Expecfting tuple: ###
    binning_size_H, binning_size_W = binning_size_tuple
    overlap_size_H, overlap_size_W = overlap_size_tuple
    stride_size_H = binning_size_H - overlap_size_H
    stride_size_W = binning_size_W - overlap_size_W
    stride_size_tuple = (stride_size_H, stride_size_W)
    # stride_size_H, stride_size_W = stride_size_tuple

    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    average_pooling_layer = nn.MaxPool2d(binning_size_tuple, stride_size_tuple)
    input_tensor_padded = get_valid_padding_on_tensor(input_tensor, binning_size_tuple, stride_size_tuple)
    binned_matrix_final = average_pooling_layer(input_tensor_padded)

    return binned_matrix_final


def fast_binning_2D_MinPool2d(input_tensor, binning_size_tuple, overlap_size_tuple):
    #(*). since this uses nn.AvgPool2d it is easier and more natural for it to return a tensor THE SAME SIZE AS THE ORIGINAL if stride==1, unlike the above functions
    # input_tensor = torch.randn((100,512,512))
    # binning_size_tuple = (10,21)
    # stride_size_tuple = (1,1)
    # overlap_size_tuple = (0,0)

    binning_size_tuple = to_tuple_of_certain_size(binning_size_tuple, 2)
    overlap_size_tuple = to_tuple_of_certain_size(overlap_size_tuple, 2)

    ### Expecfting tuple: ###
    binning_size_H, binning_size_W = binning_size_tuple
    overlap_size_H, overlap_size_W = overlap_size_tuple
    stride_size_H = binning_size_H - overlap_size_H
    stride_size_W = binning_size_W - overlap_size_W
    stride_size_tuple = (stride_size_H, stride_size_W)
    # stride_size_H, stride_size_W = stride_size_tuple

    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    average_pooling_layer = nn.MaxPool2d(binning_size_tuple, stride_size_tuple)
    input_tensor_padded = get_valid_padding_on_tensor(input_tensor, binning_size_tuple, stride_size_tuple)
    binned_matrix_final = -1*average_pooling_layer(-1*input_tensor_padded)

    return binned_matrix_final


def fast_binning_2D_MedianPool2D(input_tensor, binning_size_tuple, overlap_size_tuple):
    # (*). since this uses nn.AvgPool2d it is easier and more natural for it to return a tensor THE SAME SIZE AS THE ORIGINAL if stride==1, unlike the above functions
    # input_tensor = torch.randn((100,512,512))
    # binning_size_tuple = (10,21)
    # stride_size_tuple = (1,1)
    # overlap_size_tuple = (0,0)

    binning_size_tuple = to_tuple_of_certain_size(binning_size_tuple, 2)
    overlap_size_tuple = to_tuple_of_certain_size(overlap_size_tuple, 2)

    ### Expecfting tuple: ###
    binning_size_H, binning_size_W = binning_size_tuple
    overlap_size_H, overlap_size_W = overlap_size_tuple
    stride_size_H = binning_size_H - overlap_size_H
    stride_size_W = binning_size_W - overlap_size_W
    stride_size_tuple = (stride_size_H, stride_size_W)
    # stride_size_H, stride_size_W = stride_size_tuple

    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    average_pooling_layer = MedianPool2d(binning_size_tuple, stride_size_tuple)
    input_tensor_padded = get_valid_padding_on_tensor(input_tensor, binning_size_tuple, stride_size_tuple)
    binned_matrix_final = -1 * average_pooling_layer(-1 * input_tensor_padded)

    return binned_matrix_final


def fast_binning_2D_PixelBinning(input_tensor, binning_size, overlap_size):
    #TODO: make possible inputs of [T,H,W], [T,C,H,W], [B,T,C,H,W]
    # input_tensor = torch.randn((100,512,512))
    # binning_size = 10
    # overlap_size = 0

    # T,H,W = input_tensor.shape
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    step_size = binning_size - overlap_size
    H_final = 1 + np.int16((H - binning_size) / step_size)
    W_final = 1 + np.int16((W - binning_size) / step_size)

    if shape_len == 3:
        column_cumsum = torch.cat((torch.zeros((T, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)), -1) #pad to the left, maybe there's a faster way
        T1,H1,W1 = column_cumsum.shape
        column_binning = column_cumsum[..., binning_size:W1:step_size] - column_cumsum[..., 0:W1 - binning_size:step_size]

        row_cumsum = torch.cat((torch.zeros(T,1,W_final).to(input_tensor.device), torch.cumsum(column_binning, -2)), -2)
        T2,H2,W2 = row_cumsum.shape
        binned_matrix_final = row_cumsum[..., binning_size:H2:step_size, :] - row_cumsum[..., 0:H2-binning_size:step_size, :]
    elif shape_len == 4:
        column_cumsum = torch.cat((torch.zeros((T, C, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)), -1)
        T1, C1, H1, W1 = column_cumsum.shape
        column_binning = column_cumsum[..., binning_size:W1:step_size] - column_cumsum[..., 0:W1 - binning_size:step_size]

        row_cumsum = torch.cat((torch.zeros(B, T, C, 1, W_final).to(input_tensor.device), torch.cumsum(column_binning, -2)), -2)
        T2, C2, H2, W2 = row_cumsum.shape
        binned_matrix_final = row_cumsum[..., binning_size:H2:step_size, :] - row_cumsum[..., 0:H2 - binning_size:step_size, :]
    elif shape_len == 5:
        column_cumsum = torch.cat((torch.zeros((B, T, C, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, -1)), -1)
        B1, T1, C1, H1, W1 = column_cumsum.shape
        column_binning = column_cumsum[..., binning_size:W1:step_size] - column_cumsum[..., 0:W1 - binning_size:step_size]

        row_cumsum = torch.cat((torch.zeros(B, T, C, 1, W_final).to(input_tensor.device), torch.cumsum(column_binning, -2)), -2)
        B2, T2, C2, H2, W2 = row_cumsum.shape
        binned_matrix_final = row_cumsum[..., binning_size:H2:step_size, :] - row_cumsum[..., 0:H2 - binning_size:step_size, :]

    return binned_matrix_final


def fast_binning_1D_overlap_flexible(input_tensor, binning_size, overlap_size, flag_return_average=True, dim=0):
    # input_tensor = torch.randn((100,512,512))
    # binning_size = (10,20)
    # overlap_size = (0,0)

    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    L = input_tensor.shape[dim]
    step_size = binning_size - overlap_size
    L_final = 1 + np.int16((L - binning_size) / step_size)

    ### Get Frame To Concat: ###
    to_concat = torch.zeros([1])
    for dim_index in np.arange(shape_len - 1):
        to_concat = to_concat.unsqueeze(0)
    for dim_index in np.arange(shape_len):
        if dim_index != dim:
            to_concat = torch.cat([to_concat] * input_tensor.shape[dim_index], dim_index)

    # column_cumsum = torch.cat((to_concat.to(input_tensor.device), torch.cumsum(input_tensor, dim_index)), dim)
    column_cumsum = torch.cat((to_concat.to(input_tensor.device), torch.cumsum(input_tensor, dim)), dim) #TODO: check whether this switch is correct!!!! compare to slow binning function!!!!
    # column_cumsum = torch.cat((torch.zeros((1)).to(input_tensor.device), torch.cumsum(input_tensor, 0)), 0) #pad to the left, maybe there's a faster way

    L1 = column_cumsum.shape[dim]
    index_arange = np.arange(column_cumsum.shape[dim])
    bla1 = torch.index_select(column_cumsum, dim, torch.LongTensor(index_arange[binning_size:L1:step_size]).to(column_cumsum.device))
    bla2 = torch.index_select(column_cumsum, dim, torch.LongTensor(index_arange[0:L1 - binning_size:step_size]).to(column_cumsum.device))
    binned_signal = bla1 - bla2
    # binned_signal = column_cumsum[binning_size:L1:step_size] - column_cumsum[0:L1 - binning_size:step_size]
    if flag_return_average:
        binned_signal = binned_signal/binning_size
    return binned_signal
#####################################################################################################################################################################



####################################current_frame_for_bin#################################################################################################################################
### Make Frames Ready For VideoWriter: ###
def numpy_array_to_video_ready(input_tensor):
    if len(input_tensor.shape) == 2:
        input_tensor = numpy_unsqueeze(input_tensor, -1)
    elif len(input_tensor.shape) == 3 and (input_tensor.shape[0]==1 or input_tensor.shape[0]==3):
        input_tensor = input_tensor.transpose([1,2,0])  #[C,H,W]/[1,H,W] -> [H,W,C]
    input_tensor = BW2RGB(input_tensor)

    threshold_at_which_we_say_input_needs_to_be_normalized = 2
    if input_tensor.max() < threshold_at_which_we_say_input_needs_to_be_normalized:
        scale = 255
    else:
        scale = 1
        
    input_tensor = (input_tensor*scale).clip(0,255).astype(np.uint8)
    return input_tensor

def torch_to_numpy_video_ready(input_tensor, flag_BGR2RGB=False):
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)
    if shape_len == 2:
        #[H,W]
        output_tensor = BW2RGB(input_tensor.unsqueeze(-1)).cpu().numpy()
    elif shape_len == 3:
        #[C,H,W]
        output_tensor = BW2RGB(torch_to_numpy(input_tensor))
    elif shape_len == 4:
        #[T,C,H,W]
        output_tensor = BW2RGB(torch_to_numpy(input_tensor))

    threshold_at_which_we_say_input_needs_to_be_normalized = 2
    if output_tensor[0].max() < threshold_at_which_we_say_input_needs_to_be_normalized:
        output_tensor = output_tensor * 255

    output_tensor = (output_tensor).clip(0,255)
    output_tensor = output_tensor.astype(np.uint8)

    return output_tensor
#####################################################################################################################################################################







#####################################################################################################################################################################
### Ravel/Unravel/Roll
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


# TODO Elisheva changed to allow parallelism and also named with torch for now
def unravel_index(index: Union[int, Tensor, np.array], dimensions: Union[List[int], Tuple[int]]) -> Tuple:
    # turns raw index into dimensioned index
    dimensioned_index = []
    dimensions = list(dimensions)
    dimensions.append(1)  # for determining last dimension
    for i in range(len(dimensions)-1):
        remaining_space = reduce(lambda x, y: x*y, dimensions[i+1:])
        dimensioned_index.append(index//remaining_space)
        index -= dimensioned_index[-1]*remaining_space
    return tuple(dimensioned_index)


# TODO Elisheva changed to allow parallelism and also named with torch for now
def unravel_index_torch(index: Union[int, Tensor, np.array], dimensions: Union[List[int], Tuple[int]]) -> Tuple:
    # turns raw index into dimensioned index
    remaining_index = index.clone()
    dimensioned_index = []
    dimensions = list(dimensions)
    dimensions.append(1)  # for determining last dimension
    for i in range(len(dimensions)-1):
        remaining_space = reduce(lambda x, y: x*y, dimensions[i+1:])
        dimensioned_index.append(remaining_index//remaining_space)
        remaining_index -= dimensioned_index[-1]*remaining_space
    return tuple(dimensioned_index)


# TODO Elisheva added
def ravel_multi_index(multi_index: tuple,
                      dimensions: Union[List[int], Tuple[int]]) -> Union[Tensor, np.array, int]:
    # turns dimensioned index into raw index
    flattened_index = multi_index[0] - multi_index[0]  # to keep the element type yet get zero
    for i in range(len(dimensions) - 1):
        dim_block_size = reduce(lambda x, y: x*y, dimensions[i+1:])
        flattened_index = flattened_index + multi_index[i] * dim_block_size
    return flattened_index



def tensor_stride_from_tensor_shape(tensor_shape):
    stride_tuple = list(np.cumprod(np.flip(tensor_shape)))
    stride_tuple.insert(0,1)
    stride_tuple.pop(-1)
    stride_tuple = np.flip(stride_tuple)
    stride_tuple = tuple(stride_tuple)
    return stride_tuple
    # print(bla)
    # print(x.stride())

def test_ravel_unravel():
    B = 4
    T = 5
    C = 3
    H = 128
    W = 152
    tensor_shape = (B,T,C,H,W)
    tensor_size = B*T*C*H*W

    # x = torch.arange(30).view(10, 3)
    x = torch.randn((B, T, C, H, W))
    x = torch.arange(0,tensor_size).view(B,T,C,H,W)

    index_to_choose = 250
    index_tuple = unravel_index(index_to_choose, tensor_shape)
    print(index_tuple)

    # for i in range(x.numel()):
    #     assert i == x[unravel_index(i, x.shape)]
#####################################################################################################################################################################


#####################################################################################################################################################################
### Old Scripts: ###
#TODO: don't use this, call kornia instead
def torch_histogram_equalize(image):
    def build_lut(histo, step):
        # Compute the cumulative sum, shifting by step // 2
        # and then normalization by step.
        lut = (torch.cumsum(histo, 0) + (step // 2)) // step
        # Shift lut, prepending with 0.
        lut = torch.cat([torch.zeros(1), lut[:-1]])
        # Clip the counts to be in range.  This is done in the C code for image.point.
        return torch.clamp(lut, 0, 255)

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, c, :, :]*255   #assumes input is between [0,1]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)  # .type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im)

        return result

    s1 = scale_channel(image, 0)
    return s1.unsqueeze(0).type(torch.float32)



### TODO: Legacy, use scale_array functions instead: ###
def to_range(input_array, low, high):
    new_range_delta = high-low
    old_range_delta = input_array.max() - input_array.min()
    new_min = low
    old_min = input_array.min()
    input_array = ((input_array-old_min)*new_range_delta/old_range_delta) + new_min
    return input_array

def stretch_tensor_values(input_tensor, low, high):
    new_range_delta = high - low
    old_range_delta = input_tensor.max() - input_tensor.min()
    new_min = low
    old_min = input_tensor.min()
    input_array = ((input_tensor - old_min) * new_range_delta / old_range_delta) + new_min
    return input_array

def stretch_tensor_quantiles(input_tensor, min_quantile, max_quantile):
    min_value = input_tensor.quantile(min_quantile)
    max_value = input_tensor.quantile(max_quantile)
    return stretch_tensor_values(input_tensor.clamp(min_value,max_value), 0, 1)


def convert_dtype_torch(input_tensor, new_dtype):
    return torch.as_tensor(input_tensor, dtype=new_dtype)
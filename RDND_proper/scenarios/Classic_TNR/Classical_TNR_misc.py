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



def get_image_partition_according_to_bins(input_image, value_bins_edges):
    input_image = get_random_number_in_range(0,255,(100,100)).astype(np.uint8)
    value_bins_edges = my_linspace(0,255,11)

    number_of_bins = len(value_bins_edges) - 1
    boolean_matrix = np.zeros((input_image.shape[0],input_image.shape[1],number_of_bins))
    for bin_counter in np.arange(number_of_bins):
        boolean_matrix[:,:,bin_counter] = (input_image > value_bins_edges[bin_counter]) * (input_image < value_bins_edges[bin_counter + 1])

    return boolean_matrix


########################################################################################################################
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
#############################################################################################################################################################v


# downsample_kernel_size = 8
# average_pooling_layer = nn.AvgPool2d(downsample_kernel_size)
# upsample_layer = nn.UpsamplingNearest2d(scale_factor=downsample_kernel_size)
# for i in np.arange(1,10):
#     bla = torch.randn((1,1,2048,2048))
#     bla_time_averaged = 1/sqrt(i)*torch.randn((1,1,2048,2048))
#     bla_space_averaged = upsample_layer(average_pooling_layer(bla))
#     blo1 = ((bla-bla_time_averaged)**2).mean().sqrt()
#     blo2 = ((bla-bla_space_averaged)**2).mean().sqrt()
#     print(blo1)
#     print(blo2)


def combine_frames(noisy_frame_current,
                   clean_frame_last_warped,
                   noise_map_current,
                   noise_map_last_warped,
                   pixel_counts_map_previous,
                   downsample_kernel_size = 8):

    # imshow_torch(noisy_frame_current.clamp(0,1)); imshow_torch(clean_frame_last_warped.clamp(0,1)

    average_pooling_layer = nn.AvgPool2d(downsample_kernel_size)
    upsample_layer = nn.UpsamplingNearest2d(scale_factor=downsample_kernel_size)

    # ### Get error surface: ###
    # noisy_frame_current = noisy_frame_current
    # error_surface = (noisy_frame_current - clean_frame_last_warped).abs()
    # error_surface_squared = error_surface ** 2
    # ### Average the errors out if wanted: ###
    # error_surface_squared_averaged_downsampled = average_pooling_layer(error_surface_squared)
    # std_from_clean_image_estimate_surface = torch.sqrt(upsample_layer(error_surface_squared_averaged_downsampled))

    ### Get error surface: ###
    clean_frame_last_warped_averaged_upsampled = upsample_layer(average_pooling_layer(clean_frame_last_warped))
    error_surface = (noisy_frame_current - clean_frame_last_warped_averaged_upsampled).abs()
    error_surface_squared = error_surface ** 2
    ### Average the errors out if wanted: ###
    error_surface_squared_averaged_downsampled = average_pooling_layer(error_surface_squared)
    std_from_clean_image_estimate_surface = torch.sqrt(upsample_layer(error_surface_squared_averaged_downsampled))
    ### Get error surface exponent: ###
    normalized_distance = torch.abs(std_from_clean_image_estimate_surface / noise_map_current)

    # ### Get Error Surface: ###
    # clean_frame_last_warped_averaged_upsampled = upsample_layer(average_pooling_layer(clean_frame_last_warped.unsqueeze(0).unsqueeze(0)))
    # noisy_frame_current_averaged_upsampled = upsample_layer(average_pooling_layer(noisy_frame_current))
    # error_surface = (noisy_frame_current_averaged_upsampled - clean_frame_last_warped_averaged_upsampled).abs()
    # normalized_distance = torch.abs(error_surface / noise_map_current)

    # ### 1 -> 0: ###
    # min_value_distance_activation = 0.01  # number of noise sigma's below which i i consider statistical fluctuations
    # max_value_distance_activation = 0.02  # max_difference = 1.4 for same image,  #number of noise sigma's above which i say this is a new mean value
    # normalized_distance_slope = (1-0) / (max_value_distance_activation - min_value_distance_activation)
    # reset_gate = torch.clip(1 + min_value_distance_activation - normalized_distance * normalized_distance_slope, 0, 1)

    ### 0 -> 1 -> 0: ###
    min_value_distance_activation = 1 - 1/downsample_kernel_size / 1
    max_value_distance_activation = 1 + 1/downsample_kernel_size / 1
    normalized_distance_slope = 10
    normalized_distance_above_max = (normalized_distance>max_value_distance_activation)
    normalized_distance_below_min = (normalized_distance<min_value_distance_activation)
    normalized_distance_in_between = 1 - normalized_distance_above_max.float() - normalized_distance_below_min.float()
    normalized_distance_in_between = normalized_distance_in_between.bool()
    reset_gate = normalized_distance.data
    reset_gate[normalized_distance_below_min] = torch.clamp((1-min_value_distance_activation*normalized_distance_slope) + normalized_distance_slope * normalized_distance[normalized_distance_below_min] , 0, 1)
    reset_gate[normalized_distance_above_max] = torch.clamp((1+max_value_distance_activation*normalized_distance_slope) - normalized_distance_slope * normalized_distance[normalized_distance_above_max] , 0, 1)
    reset_gate[normalized_distance_in_between] = 1
    # print(reset_gate.min())

    # normalized_distance = torch.Tensor(my_linspace(0,4,100))
    # reset_gate = torch.clip(y_max - normalized_distance_slope * torch.abs(normalized_distance - x_0), 0, 1)
    # plot_torch(normalized_distance, reset_gate)
    # plot_torch(normalized_distance)

    ### Get updated pixel count map according to error surface: ###
    pixel_counts_map_current = pixel_counts_map_previous * reset_gate + 1  #if error is low then reset_gate->1, if error is high: reset_gate->0
    # pixel_counts_map_current = pixel_counts_map_previous + 1  #if error is low then reset_gate->1, if error is high: reset_gate->0
    # pixel_counts_map_current = 1  #if error is low then reset_gate->1, if error is high: reset_gate->0

    ### Combine: ###
    clean_frame_current = noisy_frame_current * (1/pixel_counts_map_current) + \
                          clean_frame_last_warped * (1 - 1/pixel_counts_map_current)

    return clean_frame_current, pixel_counts_map_current, reset_gate



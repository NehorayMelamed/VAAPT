




import skimage
# import imutils
import skimage.measure as skimage_measure
import skimage.metrics as skimage_metrics
## import image_registration
from numpy import linalg as LA
import numpy as np
from easydict import EasyDict
import scipy
from RapidBase.Utils.Registration.Warping_Shifting import shift_image_integer_pixels

### Blur measurement: ###
def blur_measurement(original_image, blurred_image):
    [m1, n1] = original_image.shape
    [m2, n2] = blurred_image.shape
    original_image_expanded = np.expand_dims(original_image, -1)
    mat1 = shift_image_integer_pixels(original_image_expanded, -1, -1)
    mat2 = shift_image_integer_pixels(original_image_expanded, -1, 0)
    mat3 = shift_image_integer_pixels(original_image_expanded, -1, +1)
    mat4 = shift_image_integer_pixels(original_image_expanded, 0, -1)
    mat5 = shift_image_integer_pixels(original_image_expanded, 0, +1)
    mat6 = shift_image_integer_pixels(original_image_expanded, +1, -1)
    mat7 = shift_image_integer_pixels(original_image_expanded, +1, 0)
    mat8 = shift_image_integer_pixels(original_image_expanded, +1, +1)
    large_mat = np.concatenate([original_image_expanded, mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8], -1)
    large_mat = large_mat.max(-1)

    blurred_image_expanded = np.expand_dims(blurred_image, -1)
    mat1 = shift_image_integer_pixels(blurred_image_expanded, -1, -1)
    mat2 = shift_image_integer_pixels(blurred_image_expanded, -1, 0)
    mat3 = shift_image_integer_pixels(blurred_image_expanded, -1, +1)
    mat4 = shift_image_integer_pixels(blurred_image_expanded, 0, -1)
    mat5 = shift_image_integer_pixels(blurred_image_expanded, 0, +1)
    mat6 = shift_image_integer_pixels(blurred_image_expanded, +1, -1)
    mat7 = shift_image_integer_pixels(blurred_image_expanded, +1, 0)
    mat8 = shift_image_integer_pixels(blurred_image_expanded, +1, +1)
    large_mat2 = np.concatenate([blurred_image_expanded, mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8], -1)
    large_mat2 = large_mat2.max(-1)

    blur1_mean = large_mat.mean()
    blur2_mean = large_mat2.mean()

    blurperc = abs(blur1_mean - blur2_mean) / blur1_mean
    return blurperc

    # for i in np.arange(2,m1):
    #     for j in np.arange(2,n1):
    #         A1 = [abs(original_image[i,j]-original_image[i-1,j-1]),
    #               abs(original_image[i,j]-original_image[i-1,j]),
    #               abs(original_image[i,j]-original_image[i-1,j+1]),
    #               abs(original_image[i,j]-original_image[i,j-1]),
    #               abs(original_image[i,j]-original_image[i,j+1]),
    #               abs(original_image[i,j]-original_image[i+1,j-1]),
    #               abs(original_image[i,j]-original_image[i+1,j]),
    #               abs(original_image[i,j]-original_image[i+1,j+1])]
    #         maximum1 = max(A1)
    #         M1[i,j] = maximum1



def contrast_measure(original_image):
    H,W = original_image.shape
    original_image_filtered = scipy.ndimage.sobel(original_image)
    gij_xij = original_image * original_image_filtered
    h_avg = np.ones((3,3))
    original_image_averaged = scipy.ndimage.filters.convolve(original_image_filtered, h_avg)
    gij_xij_avg = scipy.ndimage.filters.convolve(gij_xij, h_avg)
    eij = gij_xij_avg / (original_image_averaged + 0.0001)
    cij = abs(original_image - eij) / abs(original_image + eij + 0.0001)
    cij_u = (np.uint8((cij*255).round())).sum()
    EB = cij_u / (H*W)
    return EB

def contrast_measure_delta(clean_image, blurred_image):
    delta = contrast_measure(clean_image) - contrast_measure(blurred_image)
    return delta


def eigen_focus(input_image):
    H,W,C = input_image.shape[-3:]
    image_normalized = input_image / np.sqrt(((input_image**2).mean()))
    I = image_normalized - image_normalized.mean()
    image_covariance = np.matmul(I,np.transpose(I,[1,0])) / (H*W-1)
    w, V = LA.eig(image_covariance)
    var = np.diag(V)
    var_sorted = np.sort(-var)
    eigen_focus = sum(var_sorted[0:5])
    eigen_focus = np.real(eigen_focus)
    return eigen_focus

from scipy.signal import convolve2d as conv2d
def laplacian_variance(input_image):
    laplacian_kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])
    H, W, C = input_image.shape[-3:]
    if C==3:
        input_image_to_laplacian = input_image.mean(-1)
    else:
        input_image_to_laplacian = input_image
    laplacianImage = conv2d(input_image_to_laplacian, laplacian_kernel)
    FM = laplacianImage.var()
    return FM

### Metrics: ###
### Denoise Metrics: ###
# PSNR, SSIM, MSSIM, MSE, SNR, Eigen-Values, Edge Metrics
def get_metrics_image_pair(noisy_image, clean_image):
    #Keys: SSIM, NMSE, MSE, SNR_linear, SNR_dB, PSNR, VOI, contrast_measure_delta, blur_measurement
    metrics_SSIM = skimage_metrics.structural_similarity(clean_image, noisy_image, win_size=9, multichannel=True)
    # metrics_NMSE = skimage_metrics.normalized_root_mse(clean_image, noisy_image)
    metrics_NMSE = abs(clean_image - noisy_image).std() / abs(noisy_image+1e-4).std()
    metrics_MSE = skimage_metrics.mean_squared_error(clean_image, noisy_image)
    metrics_SNR_linear = 1 / (metrics_NMSE ** 2 + 1e-4)
    metrics_SNR_dB = 10 * np.log10(metrics_SNR_linear)
    metrics_PSNR = skimage_metrics.peak_signal_noise_ratio(clean_image.clip(0, 1), noisy_image.clip(0, 1))
    # metrics_VOI = skimage_metrics.variation_of_information((clean_image*255).astype(np.uint8), (noisy_image*255).astype(np.uint8))
    # metrics_eigen = eigen_metric()
    ### Blur Metrics: ###
    metrics_contrast_measure_delta = contrast_measure_delta(clean_image, noisy_image)
    metrics_blur_measurement = blur_measurement(clean_image, noisy_image)
    # metrics_eigen_focus = eigen_focus(noisy_image)
    metrics_laplacian_variance = laplacian_variance(noisy_image)
    ### Gradient Metrics ###
    clean_image_gx, clean_image_gy = np.gradient(clean_image)
    noisy_image_gx, noisy_image_gy = np.gradient(noisy_image)
    clean_image_gradient = np.sqrt(clean_image_gx ** 2 + clean_image_gy ** 2) / 2  # TODO: fix
    noisy_image_gradient = np.sqrt(
        noisy_image_gx ** 2 + clean_image_gy ** 2) / 2  # TODO: the 1/2 is a fudge factor to make the other metrics not receive outside [0,1] range
    metrics_gradient_NMSE = abs(clean_image_gradient - noisy_image_gradient).std() / abs(clean_image_gradient).std()
    metrics_gradient_MSE = skimage_metrics.mean_squared_error(clean_image_gradient, noisy_image_gradient)
    metrics_gradient_SNR_linear = 1 / (metrics_gradient_NMSE + 0.0001)
    metrics_gradient_SNR_dB = 10 * np.log10(metrics_SNR_linear)
    metrics_gradient_PSNR = skimage_metrics.peak_signal_noise_ratio(clean_image_gradient.clip(0, 1),
                                                                    noisy_image_gradient.clip(0, 1))


    output_dict = EasyDict()
    output_dict.SSIM = metrics_SSIM
    output_dict.NMSE = metrics_NMSE
    output_dict.MSE = metrics_MSE
    output_dict.SNR_linear = metrics_SNR_linear
    output_dict.SNR_dB = metrics_SNR_dB
    output_dict.PSNR = metrics_PSNR
    # output_dict.VOI = metrics_VOI
    output_dict.contrast_measure_delta = metrics_contrast_measure_delta
    output_dict.blur_measurement = metrics_blur_measurement
    # output_dict.eigen_focus = metrics_eigen_focus
    output_dict.laplacian_variance = metrics_laplacian_variance
    output_dict.gradient_MSE = metrics_gradient_MSE
    output_dict.gradient_NMSE = metrics_gradient_NMSE
    output_dict.gradient_SNR_linear = metrics_gradient_SNR_linear
    output_dict.gradient_SNR_dB = metrics_gradient_SNR_dB
    output_dict.gradient_PSNR = metrics_gradient_PSNR

    return output_dict

from RapidBase.Utils.MISCELENEOUS import AverageMeter_Dict, KeepValuesHistory_Dict
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_to_numpy, RGB2BW
from RapidBase.Utils.IO.Path_and_Reading_utils import read_images_from_folder
def get_metrics_image_pair_torch(noisy_images, clean_images):
    #Keys: SSIM, NMSE, MSE, SNR_linear, SNR_dB, PSNR, VOI, contrast_measure_delta, blur_measurement
    average_meter_dict = AverageMeter_Dict()
    ### Loop over batch indices and get matrices: ###
    for b in np.arange(noisy_images.shape[0]):
        noisy_image = noisy_images[0]
        clean_image = clean_images[0]
        if len(noisy_image.shape)==4:
            noisy_image = noisy_image[0]
            clean_image = clean_image[0]
        noisy_image = torch_to_numpy(noisy_image)
        clean_image = torch_to_numpy(clean_image)

        if len(noisy_image.shape) == 3:
            noisy_image = RGB2BW(noisy_image).squeeze()
            clean_image = RGB2BW(clean_image).squeeze()

        metrics_SSIM = skimage_metrics.structural_similarity(clean_image, noisy_image, win_size=9, channel_axis=True)
        # metrics_NMSE = skimage_metrics.normalized_root_mse(clean_image, noisy_image)
        metrics_NMSE = abs(clean_image-noisy_image).std()/abs(clean_image+1e-4).std()
        metrics_MSE = skimage_metrics.mean_squared_error(clean_image, noisy_image)
        metrics_SNR_linear = 1/(metrics_NMSE**2 + 1e-4)
        metrics_SNR_dB = 10*np.log10(metrics_SNR_linear)
        metrics_PSNR = skimage_metrics.peak_signal_noise_ratio(clean_image.clip(0,1), noisy_image.clip(0,1))
        # metrics_VOI = skimage_metrics.variation_of_information((clean_image*255).astype(np.uint8), (noisy_image*255).astype(np.uint8))
        # metrics_eigen = eigen_metric()
        ### Blur Metrics: ###
        metrics_contrast_measure_delta = contrast_measure_delta(clean_image, noisy_image)
        metrics_blur_measurement = blur_measurement(clean_image, noisy_image)
        # metrics_eigen_focus = eigen_focus(noisy_image)
        metrics_laplacian_variance = laplacian_variance(noisy_image)
        ### Gradient Metrics ###
        clean_image_gx, clean_image_gy = np.gradient(clean_image)
        noisy_image_gx, noisy_image_gy = np.gradient(noisy_image)
        clean_image_gradient = np.sqrt(clean_image_gx ** 2 + clean_image_gy ** 2)/2 #TODO: fix
        noisy_image_gradient = np.sqrt(noisy_image_gx ** 2 + clean_image_gy ** 2)/2 #TODO: the 1/2 is a fudge factor to make the other metrics not receive outside [0,1] range
        metrics_gradient_NMSE = abs(clean_image_gradient-noisy_image_gradient).std()/abs(clean_image_gradient).std()
        metrics_gradient_MSE = skimage_metrics.mean_squared_error(clean_image_gradient, noisy_image_gradient)
        metrics_gradient_SNR_linear = 1 / (metrics_gradient_NMSE**2 + 0.0001)
        metrics_gradient_SNR_dB = 10 * np.log10(metrics_SNR_linear)
        metrics_gradient_PSNR = skimage_metrics.peak_signal_noise_ratio(clean_image_gradient.clip(0,1), noisy_image_gradient.clip(0,1))

        output_dict = EasyDict()
        output_dict.SSIM = metrics_SSIM
        output_dict.NMSE = metrics_NMSE
        output_dict.MSE = metrics_MSE
        output_dict.SNR_linear = metrics_SNR_linear
        output_dict.SNR_dB = metrics_SNR_dB
        output_dict.PSNR = metrics_PSNR
        # output_dict.VOI = metrics_VOI
        output_dict.contrast_measure_delta = metrics_contrast_measure_delta
        output_dict.blur_measurement = metrics_blur_measurement
        # output_dict.eigen_focus = metrics_eigen_focus
        output_dict.laplacian_variance = metrics_laplacian_variance
        output_dict.gradient_MSE = metrics_gradient_MSE
        output_dict.gradient_NMSE = metrics_gradient_NMSE
        output_dict.gradient_SNR_linear = metrics_gradient_SNR_linear
        output_dict.gradient_SNR_dB = metrics_gradient_SNR_dB
        output_dict.gradient_PSNR = metrics_gradient_PSNR

        average_meter_dict.update_dict(output_dict)

    return average_meter_dict.inner_dict


def get_metrics_video(clean_images_folder, noisy_images_folder, number_of_images=np.inf):
    noisy_images_list = read_images_from_folder(noisy_images_folder)
    clean_images_list = read_images_from_folder(clean_images_folder)
    output_dict_average = AverageMeter_Dict()
    output_dict_history = KeepValuesHistory_Dict()
    max_number_of_images = min(number_of_images, len(noisy_images_list))
    for i in np.arange(max_number_of_images):
        noisy_image = noisy_images_list[i]
        clean_image = clean_images_list[i]
        noisy_image = noisy_image[:, :, 0:1]

        noisy_image = noisy_image.squeeze()
        clean_image = clean_image.squeeze()

        noisy_image = noisy_image/255
        clean_image = clean_image/255

        output_dict = get_metrics_image_pair(noisy_image, clean_image)
        output_dict_average.update_dict(output_dict)
        output_dict_history.update_dict(output_dict)
        print('frame: ' + str(i))
    # plot(output_dict_history.PSNR)
    # title('PSNR over time')

    return output_dict_average, output_dict_history


def get_metrics_video_lists(clean_images_list, noisy_images_list, number_of_images=np.inf):
    output_dict_average = AverageMeter_Dict()
    output_dict_history = KeepValuesHistory_Dict()
    max_number_of_images = min(number_of_images, len(noisy_images_list))
    for i in np.arange(max_number_of_images):
        noisy_image = noisy_images_list[i]
        clean_image = clean_images_list[i]

        if noisy_image.shape[-1] != clean_image.shape[-1]:
            noisy_image = noisy_image[:, :, 0:1]

        noisy_image = noisy_image.squeeze()
        clean_image = clean_image.squeeze()

        noisy_image = noisy_image/255
        clean_image = clean_image/255

        output_dict = get_metrics_image_pair(noisy_image, clean_image)
        output_dict_average.update_dict(output_dict)
        output_dict_history.update_dict(output_dict)
        print('frame: ' + str(i))
    # plot(output_dict_history.PSNR)
    # title('PSNR over time')

    return output_dict_average, output_dict_history




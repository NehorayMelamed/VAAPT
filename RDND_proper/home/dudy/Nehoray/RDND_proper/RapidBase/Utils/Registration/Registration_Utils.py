
# from RapidBase.import_all import *

from RapidBase.Utils.Classical_DSP.FFT_utils import fftshift_torch
from RapidBase.Utils.Classical_DSP.Fitting_And_Optimization import return_shifts_using_parabola_fit_torch
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import crop_torch_batch
from RapidBase.Utils.IO.tic_toc import *
from RapidBase.Utils.Registration.Warping_Shifting import *
from RapidBase.Utils.Registration.Warp_Layers import *

from RapidBase.Utils.Classical_DSP.FFT_utils import torch_fft2, torch_fftshift, torch_ifft2, fftshift_torch
from RapidBase.Utils.Registration.Transforms_Grids import logpolar, logpolar_torch, logpolar_scipy_torch, highpass, highpass_torch
try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

from numpy.fft import fft2, ifft2, fftshift
from RapidBase.Utils.IO.Path_and_Reading_utils import read_image_default_torch
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import crop_numpy_batch
from RapidBase.Utils.Registration.Transforms_Grids import logpolar_grid
import torch.nn.functional as F

from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
from scipy.fftpack import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation as skimage_phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float
from RapidBase.Utils.Classical_DSP.Fitting_And_Optimization import *
from RapidBase.Utils.Registration.Transforms_Grids import *

from RapidBase.Utils.IO.Path_and_Reading_utils import path_get_all_filenames_from_folder
import torch
import numpy as np

from RapidBase.Utils.Registration.Warping_Shifting import Shift_Layer_Torch
from RapidBase.Utils.Registration.Warp_Layers import Warp_Object, Warp_Tensors_Affine_Layer
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import RGB2BW

# from RapidBase.import_all import *
from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
# from RapidBase.TrainingCore.datasets import *
# import RapidBase.TrainingCore.datasets
# import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
# from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *

############################################################################################################################################################
### Cross Correlation Stuff: ###
def get_Circular_Cross_Correlation_With_FineTune_torch_InputFFTToSaveCalculations(tensor1, tensor2, tensor1_fft, tensor2_fft, kx, ky, flag_initial_shift_method, flag_CC_finetunning):
    ### Get Cross Correlation From FFT: ###
    CC = torch.fft.ifftn(tensor1_fft * tensor2_fft.conj(), dim=[-2, -1]).abs()
    CC = fftshift_torch(CC)
    ### Get Sub-Pixel Shifts Using Parabola Fit: ###
    shifts_vec, z_vec = return_shifts_using_parabola_fit_torch(CC)
    shiftx1 = shifts_vec[0]
    shifty1 = shifts_vec[1]
    shiftx_int = int(shifts_vec[0])
    shifty_int = int(shifts_vec[1])
    shiftx_torch = torch.Tensor([float(shifts_vec[0])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    shifty_torch = torch.Tensor([float(shifts_vec[1])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    ### Shift Tensor2 For Maximum Valid Percent: ###
    if flag_initial_shift_method == 'integer_roll':
        tensor2_displaced = torch.roll(tensor2, [shiftx_int, shifty_int], dims=[-1, -2])
    elif flag_initial_shift_method == 'sub_pixel_fft':
        displacement_matrix = torch.exp(-(1j * 2 * np.pi * ky * shifty_torch + 1j * 2 * np.pi * kx * shiftx_torch))
        tensor2_displaced = torch.fft.ifftn(tensor2_fft * displacement_matrix, dim=[-1, -2]).real

    ### Use Manual Cross Correlation To Find Final Sub-Pixel Shift: ###
    if flag_CC_finetunning == 'manual_CC':
        ### Crop Images For Maximum Hafifa: ###
        B, C, H, W = tensor1.shape
        tensor1_crop = crop_torch_batch(tensor1, [H-shifty_int, W-shiftx_int], crop_style='center')
        tensor2_displaced_crop = crop_torch_batch(tensor2_displaced, [H-shifty_int, W-shiftx_int], crop_style='center')
        B2, C2, H2, W2 = tensor1_crop.shape
        ### Get Manual Cross Correlation: ###
        CC2 = get_Normalized_Cross_Correlation_torch(tensor1_crop, tensor2_displaced_crop, 5)
        shifts_vec2, z_vec = return_shifts_using_parabola_fit_torch(CC2)
        shiftx_int = int(shifts_vec2[0])
        shifty_int = int(shifts_vec2[1])
        # toc('calculate manual cross correlation after initial shift')  #TODO: this takes a huggggge chunk of time, needs to be optimized
        ### Final Sub-Pixel Shift To Final Location: ###
        shiftx2_tensor = torch.Tensor([float(shifts_vec2[0])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shifty2_tensor = torch.Tensor([float(shifts_vec2[1])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shiftx2_tensor += shiftx1
        shifty2_tensor += shifty1
        displacement_matrix2 = torch.exp(-(1j * 2 * np.pi * ky * shifty2_tensor + 1j * 2 * np.pi * kx * shiftx2_tensor))
        tensor2_fft_displaced = tensor2_fft * displacement_matrix2
        tensor2_displaced = torch.fft.ifftn(tensor2_fft_displaced, dim=[-1, -2]).real
        shifts_vec[0] += shifts_vec2[0]
        shifts_vec[1] += shifts_vec2[1]

    return shifts_vec, tensor2_displaced

def get_Circular_Cross_Correlation_With_FineTune_torch(tensor1, tensor2, flag_initial_shift_method, flag_CC_finetunning):
    # ### Flags: ###
    # flag_initial_shift_method = 'sub_pixel_fft'  # 'integer_roll', 'sub_pixel_fft'
    # flag_CC_finetunning = 'none' # 'none', 'manual_CC', 'FFT_CC'
    # ### Read Tensors: ###
    # tensor1 = RGB2BW(read_image_default_torch())
    # tensor2 = RGB2BW(read_image_default_torch())
    # tensor1 = upsample_image_torch(tensor1).real
    # tensor2 = upsample_image_torch(tensor2).real
    # tensor1 = upsample_image_torch(tensor1).real
    # tensor2 = upsample_image_torch(tensor2).real
    # # tensor1 = torch.randn((1,1,2048, 2048))
    # # tensor2 = torch.randn((1,1,2048, 2048))
    # tensor2 = shift_matrix_subpixel_torch(tensor2, 10.4, 20.4)
    # B, C, H, W = tensor1.shape
    # ### Initial Crop To Simulate Real Conditions With Non-Valid Pixels: ###
    # tensor1 = crop_torch_batch(tensor1, [W-50, H-50], crop_style='center')
    # tensor2 = crop_torch_batch(tensor2, [W-50, H-50], crop_style='center')
    # B, C, H, W = tensor1.shape


    ### Prepare Stuff For Sub-Pixel Shifts: ###
    B, C, H, W = tensor1.shape
    # Get tilt phases k-space:
    x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
    y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
    delta_f1 = 1 / W
    delta_f2 = 1 / H
    f_x = x * delta_f1
    f_y = y * delta_f2
    # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_x = np.fft.fftshift(f_x)
    f_y = np.fft.fftshift(f_y)
    # Build k-space meshgrid:
    [kx, ky] = np.meshgrid(f_x, f_y)
    # Frequency vec to tensor:
    kx = torch.Tensor(kx)
    ky = torch.Tensor(ky)
    kx = kx.unsqueeze(0).unsqueeze(0)
    ky = ky.unsqueeze(0).unsqueeze(0)

    ### FFT Original Image: ###
    tensor1_fft = torch.fft.fftn(tensor1, dim=[-2, -1])

    tic()

    ### FFT Image Coming Image: ###
    # tic()
    tensor2_fft = torch.fft.fftn(tensor2, dim=[-2, -1])
    # toc('tensor2 fft')

    ### Get Cross Correlation From FFT: ###
    # tic()
    CC = torch.fft.ifftn(tensor1_fft * tensor2_fft.conj(), dim=[-2, -1]).abs()
    CC = fftshift_torch(CC)
    # toc('ifft to get initial CC')

    # ### Get Normalized Cross Correlation (max possible value = 1): ###
    # A_sum = tensor2.sum()
    # A_sum2 = (tensor2 ** 2).sum()
    # sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
    # sigmaB = (tensor1).std() * (H * W - 1) ** (1 / 2)
    # B_mean = (tensor1).mean()
    # ### Normalize CrossCorrelation: ###
    # CC = (CC - A_sum * B_mean) / (sigmaA * sigmaB)

    ### Get Sub-Pixel Shifts Using Parabola Fit: ###
    # tic()
    shifts_vec, z_vec = return_shifts_using_parabola_fit_torch(CC)
    shiftx1 = shifts_vec[0]
    shifty1 = shifts_vec[1]
    shiftx_int = int(shifts_vec[0])
    shifty_int = int(shifts_vec[1])
    shiftx_torch = torch.Tensor([float(shifts_vec[0])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    shifty_torch = torch.Tensor([float(shifts_vec[1])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # toc('return shifts using parabola fit initial CC')

    if flag_initial_shift_method == 'integer_roll':
        # tic()
        ### Integer Pixel Shift Image To Found Shift: ###
        tensor2_displaced = torch.roll(tensor2, [shiftx_int, shifty_int], dims=[-1, -2])
        # toc('integer roll tensor2 according to initial shift found')
    elif flag_initial_shift_method == 'sub_pixel_fft':
        ### FFT-Shift Image To Found Shift: ###
        # tic()
        displacement_matrix = torch.exp(-(1j * 2 * np.pi * ky * shifty_torch + 1j * 2 * np.pi * kx * shiftx_torch))
        tensor2_displaced = torch.fft.ifftn(tensor2_fft * displacement_matrix, dim=[-1, -2]).real
        # toc('sub pixel shift using fft for tensor2 according to initial shift found')

    ### Crop Images For Maximum Hafifa: ###
    # tic()
    tensor1_crop = crop_torch_batch(tensor1, [W-2*abs(shiftx_int), H-2*abs(shifty_int)], crop_style='center')
    tensor2_displaced_crop = crop_torch_batch(tensor2_displaced, [W-2*abs(shiftx_int), H-2*abs(shifty_int)], crop_style='center')
    B2, C2, H2, W2 = tensor1_crop.shape
    # toc('crop tensors for maximum valid percent according to initial shift found')

    ### Use Manual Cross Correlation To Find Final Sub-Pixel Shift: ###
    if flag_CC_finetunning == 'manual_CC':
        # tic()
        CC2 = get_Normalized_Cross_Correlation_torch(tensor1_crop, tensor2_displaced_crop, 5)
        shifts_vec2, z_vec = return_shifts_using_parabola_fit_torch(CC2)
        shiftx_int = int(shifts_vec2[0])
        shifty_int = int(shifts_vec2[1])
        # toc('calculate manual cross correlation after initial shift')  #TODO: this takes a huggggge chunk of time, needs to be optimized
        ### Final Sub-Pixel Shift To Final Location: ###
        shiftx2_tensor = torch.Tensor([float(shifts_vec2[0])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shifty2_tensor = torch.Tensor([float(shifts_vec2[1])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shiftx2_tensor += shiftx1
        shifty2_tensor += shifty1
        displacement_matrix2 = torch.exp(-(1j * 2 * np.pi * ky * shifty2_tensor + 1j * 2 * np.pi * kx * shiftx2_tensor))
        tensor2_fft_displaced = tensor2_fft * displacement_matrix2
        tensor2_displaced = torch.fft.ifftn(tensor2_fft_displaced, dim=[-1, -2]).real
        tensor2_displaced_crop = crop_torch_batch(tensor2_displaced, [W2 - 2 * abs(shiftx_int), H2 - 2 * abs(shifty_int)], crop_style='center')
        tensor1_crop = crop_torch_batch(tensor1_crop, [W2 - 2 * abs(shiftx_int), H2 - 2 * abs(shifty_int)], crop_style='center')
    elif flag_CC_finetunning == 'FFT_CC':
        # tic()
        tensor1_crop_fft = torch.fft.fftn(tensor1_crop, dim=[-2, -1])
        tensor2_displaced_crop_fft = torch.fft.fftn(tensor2_displaced_crop, dim=[-2, -1])
        CC2 = torch.fft.ifftn(tensor1_crop_fft * tensor2_displaced_crop_fft.conj(), dim=[-2, -1]).abs()
        CC2 = fftshift_torch(CC2)
        # toc('FFT CC for crops after initial shift')
        ### Final Sub-Pixel Shift To Final Location: ###
        shifts_vec2, z_vec = return_shifts_using_parabola_fit_torch(CC2)
        shiftx_int = int(shifts_vec2[0])
        shifty_int = int(shifts_vec2[1])
        shiftx2_tensor = torch.Tensor([float(shifts_vec2[0])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shifty2_tensor = torch.Tensor([float(shifts_vec2[1])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shiftx2_tensor += shiftx1
        shifty2_tensor += shifty1
        # tic()
        displacement_matrix2 = torch.exp(-(1j * 2 * np.pi * ky * shifty2_tensor + 1j * 2 * np.pi * kx * shiftx2_tensor))
        tensor2_fft_displaced = tensor2_fft * displacement_matrix2
        tensor2_displaced = torch.fft.ifftn(tensor2_fft_displaced, dim=[-1, -2]).real
        tensor2_displaced_crop = crop_torch_batch(tensor2_displaced, [W2 - 2 * abs(shiftx_int), H2 - 2 * abs(shifty_int)], crop_style='center')
        tensor1_crop = crop_torch_batch(tensor1_crop, [W2 - 2 * abs(shiftx_int), H2 - 2 * abs(shifty_int)], crop_style='center')
        # toc('sub pixel FFT shift final')

    toc()
    # imshow_torch(tensor1_crop)
    # imshow_torch(tensor2_displaced_crop)
    return shiftx2_tensor, shifty2_tensor, tensor2_displaced_crop

def perform_gimbaless_cross_correlation_on_image_batch(input_tensor, number_of_samples_per_inner_batch):
    # ####################################################################################################
    # ### Testing: ###
    # shift_layer = Shift_Layer_Torch()
    # number_of_samples = 25
    # number_of_samples_per_inner_batch = 5
    # number_of_batches = number_of_samples // number_of_samples_per_inner_batch
    # SNR = 10
    # #(*). Temp - speckle pattern:
    # speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(10, 70,0,1,1,0)
    # speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    # #(*). Temp - real image:
    # speckle_pattern = read_image_default_torch()
    # speckle_pattern = RGB2BW(speckle_pattern)
    # speckle_pattern = crop_torch_batch(speckle_pattern, 512).squeeze(0)
    # C, H, W = speckle_pattern.shape
    # ### Get Input Tensor: ###
    # input_tensor = torch.cat([speckle_pattern] * number_of_samples, 0)
    # current_shift = 0
    # ### Actually Shift: ###
    # # (1). Batch Processing:
    # real_shifts_vec = torch.randn(number_of_samples)
    # differential_shifts_vec = real_shifts_vec[1:number_of_samples] - real_shifts_vec[0:number_of_samples-1]
    # differential_shifts_vec = differential_shifts_vec * -1
    # real_shifts_to_center_frame = real_shifts_vec - real_shifts_vec[number_of_samples//2]
    # real_shifts_to_center_frame = torch.cat([real_shifts_to_center_frame[0:number_of_samples//2], real_shifts_to_center_frame[number_of_samples//2+1:]])
    # input_tensor = shift_layer.forward(input_tensor, real_shifts_vec.cpu().numpy(), real_shifts_vec.cpu().numpy())
    # input_tensor = crop_torch_batch(input_tensor, 512)
    # input_tensor = input_tensor.unsqueeze(0)
    # #

    ### Add Noise To Tensor: ###
    B,C,H,W = input_tensor.shape
    number_of_samples = C
    number_of_batches = number_of_samples // number_of_samples_per_inner_batch
    final_outputs = []
    for i in np.arange(number_of_batches):
        start_index = i*number_of_samples_per_inner_batch
        stop_index = start_index + number_of_samples_per_inner_batch
        current_mean_frame = align_image_batch_to_center_frame_cross_correlation(input_tensor[:,start_index:stop_index]).permute([1,0,2,3]).mean(1,True)
        final_outputs.append(current_mean_frame)
    final_outputs = torch.cat(final_outputs,1)

    # imshow_torch(input_tensor[:,12])
    # imshow_torch(final_outputs[0])

    return final_outputs



class Shift_Layer_Torch(nn.Module):
    # Initialize this with a module
    def __init__(self):
        super(Shift_Layer_Torch, self).__init__()
        self.kx = None
        self.ky = None

    def forward(self, input_image, shiftx, shifty, fft_image=None):
        ### Check if we need to create a new kvec: ###
        if self.kx is None:
            flag_create_new_kvec = True
        else:
            if self.kx.shape[-1] != input_image.shape[-1] or self.kx.shape[-2] != input_image.shape[-2]:
                flag_create_new_kvec = True
            else:
                flag_create_new_kvec = False

        ### If we do, create it: ###
        if flag_create_new_kvec:
            # Get Input Dimensions:
            self.ndims = len(input_image.shape)
            if self.ndims == 4:
                B,C,H,W = input_image.shape
            elif self.ndims == 3:
                C,H,W = input_image.shape
                B = 1
            elif self.ndims == 2:
                H,W = input_image.shape
                B = 1
                C = 1
            self.B = B
            self.C = C
            self.H = H
            self.W = W
            # Get tilt phases k-space:
            x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
            y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
            delta_f1 = 1 / W
            delta_f2 = 1 / H
            f_x = x * delta_f1
            f_y = y * delta_f2
            # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
            f_x = np.fft.fftshift(f_x)
            f_y = np.fft.fftshift(f_y)
            # Build k-space meshgrid:
            [kx, ky] = np.meshgrid(f_x, f_y)
            # Frequency vec to tensor:
            self.kx = torch.Tensor(kx)
            self.ky = torch.Tensor(ky)
            # Expand to original image dimensions:
            if self.ndims == 3:
                self.kx = self.kx.unsqueeze(0)
                self.ky = self.ky.unsqueeze(0)
            if self.ndims == 4:
                self.kx = self.kx.unsqueeze(0).unsqueeze(0)
                self.ky = self.ky.unsqueeze(0).unsqueeze(0)
            # K-vecs to tensor device:
            self.kx = self.kx.to(input_image.device)
            self.ky = self.ky.to(input_image.device)

        ### Expand shiftx & shifty to match input image shape (and if shifts are more than a scalar i assume you want to multiply B or C): ###
        #TODO: enable accepting pytorch tensors/vectors etc'
        if type(shiftx) != list and type(shiftx) != tuple and type(shiftx) != np.ndarray and type(shiftx) != torch.Tensor:
            shiftx = [shiftx]
            shifty = [shifty]
        if self.ndims == 3:
            if type(shiftx) != torch.Tensor:
                shiftx = torch.Tensor(shiftx).to(input_image.device).unsqueeze(-1).unsqueeze(-1)
                shifty = torch.Tensor(shifty).to(input_image.device).unsqueeze(-1).unsqueeze(-1)
            else:
                shiftx = shiftx.to(input_image.device).unsqueeze(-1).unsqueeze(-1)
                shifty = shifty.to(input_image.device).unsqueeze(-1).unsqueeze(-1)
            shiftx = shiftx.to(input_image.device)
            shifty = shifty.to(input_image.device)
        elif self.ndims == 4:
            if type(shiftx) != torch.Tensor:
                shiftx = torch.Tensor(shiftx).to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                shifty = torch.Tensor(shifty).to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                shiftx = shiftx.to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                shifty = shifty.to(input_image.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shiftx = shiftx.to(input_image.device)
            shifty = shifty.to(input_image.device)

        ### Displace: ###
        # print(shiftx)
        # print(self.kx.shape)
        displacement_matrix = torch.exp(-(1j * 2 * np.pi * self.ky * shifty + 1j * 2 * np.pi * self.kx * shiftx))
        if fft_image is None:
            fft_image = torch.fft.fftn(input_image, dim=[-1, -2])
        fft_image_displaced = fft_image * displacement_matrix
        # input_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()
        input_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).real

        return input_image_displaced


def Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(input_tensor):
    ### the function expects input_tensor to be a numpy array of shape [T,H,W]: ###

    ### Initialization: ###
    if type(input_tensor) == np.ndarray:
        input_tensor = torch.Tensor(input_tensor)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    shift_layer = Shift_Layer_Torch()
    B, C, H, W = input_tensor.shape
    number_of_samples = B

    def fit_polynomial_torch(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[:, 2] + y[:, 0] - 2 * y[:, 1]) / 2
        b = -(y[:, 0] + 2 * a * x[1] - y[:, 1] - a)
        c = y[:, 1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    #########################################################################################################
    ### Get CC: ###
    # input tensor [T,H,W]
    B, T, H, W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1, -2])
    # (1). Regular Cross Corerlation:
    output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * input_tensor_fft[:, T // 2, :, :].conj(),
                                dim=[-1, -2]).real
    # output_CC = torch.cat()
    #########################################################################################################
    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    # (1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    # (2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > H] -= W
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * W
    output_CC_flattened_indices_i0 = i1 + i0 * W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, H * W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    #########################################################################################################

    #########################################################################################################
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # print(shiftx)
    # print(real_shifts_to_center_frame)
    #########################################################################################################

    #########################################################################################################
    ### Align Images: ###
    shifted_tensors = shift_layer.forward(input_tensor.permute([1, 0, 2, 3]), -shiftx, -shifty)  # TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0, True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    return shifted_tensors[:, 0, :, :].numpy()


def align_image_batch_to_center_frame_cross_correlation(input_tensor):
    ### the function expects input_tensor to be a NUMPY array of shape [T,H,W]: ###

    # # ####################################################################################################
    # # ### Testing: ###
    # shift_layer = Shift_Layer_Torch()
    # number_of_samples = 11
    # SNR = 10
    # speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(10, 70,0,1,1,0)
    # speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    # #(*). Temp - real image:
    # speckle_pattern = read_image_default_torch()
    # speckle_pattern = RGB2BW(speckle_pattern)
    # speckle_pattern = crop_torch_batch(speckle_pattern, 512).squeeze(0)
    # C, H, W = speckle_pattern.shape
    # ### Get Input Tensor: ###
    # input_tensor = torch.cat([speckle_pattern] * number_of_samples, 0)
    # current_shift = 0
    # ### Actually Shift: ###
    # # (1). Batch Processing:
    # real_shifts_vec = torch.randn(number_of_samples) * 10
    # differential_shifts_vec = real_shifts_vec[1:number_of_samples] - real_shifts_vec[0:number_of_samples-1]
    # differential_shifts_vec = differential_shifts_vec * -1
    # real_shifts_to_center_frame = real_shifts_vec - real_shifts_vec[number_of_samples//2]
    # real_shifts_to_center_frame = torch.cat([real_shifts_to_center_frame[0:number_of_samples//2], real_shifts_to_center_frame[number_of_samples//2+1:]])
    # input_tensor = shift_layer.forward(input_tensor, real_shifts_vec.cpu().numpy(), real_shifts_vec.cpu().numpy())
    # input_tensor = crop_torch_batch(input_tensor, 512)
    # input_tensor = input_tensor.unsqueeze(0)
    #
    # ### Add Noise To Tensor: ###
    # B,C,H,W = input_tensor.shape
    # number_of_samples = B
    # noise_map = torch.randn_like(input_tensor) * 1/np.sqrt(SNR)
    # input_tensor += noise_map
    # ###################################################################################################

    ### Initialization: ###
    if type(input_tensor) == np.ndarray:
        input_tensor = torch.Tensor(input_tensor)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    shift_layer = Shift_Layer_Torch()
    B, C, H, W = input_tensor.shape
    number_of_samples = B

    def fit_polynomial_torch(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[:,2] + y[:,0] - 2 * y[:,1]) / 2
        b = -(y[:,0] + 2 * a * x[1] - y[:,1] - a)
        c = y[:,1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    #########################################################################################################
    ### Get CC: ###
    #input tensor [T,H,W]
    B,T,H,W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1,-2])
    #(1). Regular Cross Corerlation:
    output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * input_tensor_fft[:, T//2, :, :].conj(), dim=[-1, -2]).real
    # output_CC = torch.cat()
    #########################################################################################################
    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H*W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    #(1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    #(2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > H] -= W
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0*W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1*W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1*W
    output_CC_flattened_indices_i0 = i1 + i0*W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, H*W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    #########################################################################################################

    #########################################################################################################
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # print(shiftx)
    # print(real_shifts_to_center_frame)
    #########################################################################################################

    #########################################################################################################
    ### Align Images: ###
    shifted_tensors = shift_layer.forward(input_tensor.permute([1,0,2,3]), -shiftx, -shifty) #TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0,True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    return shifted_tensors

def Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(input_tensor, reference_tensor=None):
    ### the function expects input_tensor to be a numpy array of shape [T,H,W]: ###

    ### Initialization: ###
    if type(input_tensor) == np.ndarray:
        input_tensor = torch.Tensor(input_tensor)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    shift_layer = Shift_Layer_Torch()
    B, C, H, W = input_tensor.shape
    number_of_samples = B

    def fit_polynomial_torch(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[:, 2] + y[:, 0] - 2 * y[:, 1]) / 2
        b = -(y[:, 0] + 2 * a * x[1] - y[:, 1] - a)
        c = y[:, 1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    #########################################################################################################
    ### Get CC: ###
    # input tensor [T,H,W]
    B, T, H, W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1, -2])

    ### Get reference tensor fft: ###
    if reference_tensor is None:
        reference_tensor_fft = input_tensor_fft[:, T // 2:T//2+1, :, :]
    else:
        reference_tensor_fft = torch.fft.fftn(reference_tensor, dim=[-1,-2])

    # (*). Circular Cross Corerlation Using FFT:
    output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * reference_tensor_fft.conj(), dim=[-1, -2]).real
    # output_CC = torch.cat()

    #########################################################################################################
    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    # (1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    # (2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > H] -= W
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * W
    output_CC_flattened_indices_i0 = i1 + i0 * W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, H * W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    #########################################################################################################

    #########################################################################################################
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # print(shiftx)
    # print(real_shifts_to_center_frame)
    #########################################################################################################

    #########################################################################################################
    ### Align Images: ###
    shifted_tensors = shift_layer.forward(input_tensor.permute([1, 0, 2, 3]), -shiftx, -shifty)  # TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0, True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    return shifted_tensors


def Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation(input_tensor, reference_tensor=None, weights_tensor=None):
    ### the function expects input_tensor to be a numpy array of shape [T,H,W]: ###

    ### Initialization: ###
    if type(input_tensor) == np.ndarray:
        input_tensor = torch.Tensor(input_tensor)
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0) #TODO: get rid of this for Anvil's pre-processing
    shift_layer = Shift_Layer_Torch()
    B, C, H, W = input_tensor.shape
    number_of_samples = B

    def fit_polynomial_torch(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[:, 2] + y[:, 0] - 2 * y[:, 1]) / 2
        b = -(y[:, 0] + 2 * a * x[1] - y[:, 1] - a)
        c = y[:, 1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    #########################################################################################################
    ### Get CC: ###
    # input tensor [T,H,W]
    B, T, H, W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1, -2])

    ### Get reference tensor fft: ###
    if reference_tensor is None:
        reference_tensor_fft = input_tensor_fft[:, T//2 : T//2+1, :, :]
        reference_tensor = input_tensor[:, T//2 : T//2+1, :, :]
    else:
        reference_tensor_fft = torch.fft.fftn(reference_tensor, dim=[-1,-2])

    # (*). Circular Cross Corerlation Using FFT:
    # output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * reference_tensor_fft.conj(), dim=[-1, -2]).real
    with torch.no_grad():
        output_CC = get_Weighted_Circular_Cross_Correlation_Batch_torch(input_tensor, reference_tensor, weights_tensor=weights_tensor)[:,0].unsqueeze(0)  #TODO: switch to Anvil instead of all this shit
    # output_CC = torch.cat()

    #########################################################################################################
    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    # (1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    # (2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > H] -= W
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= H
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * W
    output_CC_flattened_indices_i0 = i1 + i0 * W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, H * W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    #########################################################################################################

    #########################################################################################################
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # print(shiftx)
    # print(real_shifts_to_center_frame)
    # #TODO: delete, just for trying:
    shiftx = shiftx * 1
    shifty = shifty * 1
    #########################################################################################################

    #########################################################################################################
    ### Align Images: ###
    # plot_torch(shiftx)
    shifted_tensors = shift_layer.forward(input_tensor.permute([1, 0, 2, 3]), -shiftx, -shifty)  # TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0, True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    return shifted_tensors, shiftx, shifty


def Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation_MultipleReference(input_tensor, reference_tensor=None, weights_tensor=None):
    ### the function expects input_tensor to be a numpy array of shape [T,H,W]: ###
    shiftx_list = []
    shifty_list = []
    T,H,W = input_tensor.shape
    for t in np.arange(T):
        shifted_tensor, shiftx_current, shifty_current = Align_TorchBatch_ToCenterFrame_WeightedCircularCrossCorrelation(input_tensor[t:t+1],
                                                                                                                         reference_tensor[t:t+1],
                                                                                                                         weights_tensor)
        shiftx_list.append(shiftx_current)
        shifty_list.append(shifty_current)

    return torch.cat(shiftx_list), torch.cat(shifty_list)


def get_Normalized_Cross_Correlation_torch(tensor1, tensor2, correlation_size):
    #(*). Normalized Cross Correlation!!!!
    ### Trims: ###
    B, C, H, W = tensor2.shape
    Correlations = torch.zeros((B, C, correlation_size, correlation_size)).to(tensor1.device)
    trim = int(np.floor(correlation_size / 2))
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
    DispUD = np.arange(-RimD, RimU + 1)
    DispLR = np.arange(-RimL, RimR + 1)

    ### B-ROI: ###
    Broi = tensor2[:, :, RimU:H - RimD, RimL:W - RimR]
    Broibar = Broi.mean(2,True).mean(3,True)
    Broiup = (Broi - Broibar)
    Broidown = ((Broi - Broibar) ** 2).sum(2,True).sum(3,True)

    ### Get Cross-Correlation: ###
    for iin in np.arange(len(DispUD)):
        for jjn in np.arange(len(DispLR)):
            shift_y = DispUD[iin]
            shift_x = DispLR[jjn]
            A_upper_left = [B_upper_left[0] + shift_y, B_upper_left[1] + shift_x]
            Atmp = tensor1[:, :, A_upper_left[0]:A_upper_left[0] + ALy, A_upper_left[1]:A_upper_left[1] + ALx]
            Abar = Atmp.mean(2,True).mean(3,True)
            Aup = (Atmp - Abar)
            Adown = ((Atmp - Abar) ** 2).sum(2,True).sum(3,True)
            Correlations[:,:,iin, jjn] = (Broiup * Aup).sum(2,False).sum(2,False) / torch.sqrt(Broidown * Adown).squeeze(2).squeeze(2)

    return Correlations

def get_Normalized_Cross_Correlation_numpy(mat1, mat2, correlation_size):
    ### Trims: ###
    Correlations = np.zeros((correlation_size, correlation_size))
    trim = int(np.floor(correlation_size / 2))
    RimU = trim
    RimD = trim
    RimL = trim
    RimR = trim
    H, W = mat2.shape

    ### A,B Length: ###
    BLy = H - RimU - RimD
    BLx = W - RimL - RimR
    ALy = BLy
    ALx = BLx

    ### Displacement: ###
    B_upper_left = [RimU, RimL]  # the location of the upper-left corner of the Broi matrix
    DispUD = np.arange(-RimD, RimU + 1)
    DispLR = np.arange(-RimL, RimR + 1)

    ### B-ROI: ###
    Broi = mat2[RimU:H - RimD, RimL:W - RimR]
    Broibar = Broi.mean()
    Broiup = (Broi - Broibar)
    Broidown = ((Broi - Broibar) ** 2).sum()

    ### Get Cross-Correlation: ###
    for iin in np.arange(len(DispUD)):
        for jjn in np.arange(len(DispLR)):
            shift_y = DispUD[iin]
            shift_x = DispLR[jjn]
            A_upper_left = [B_upper_left[0] + shift_y, B_upper_left[1] + shift_x]
            Atmp = mat1[A_upper_left[0]:A_upper_left[0] + ALy, A_upper_left[1]:A_upper_left[1] + ALx]
            Abar = Atmp.mean()
            Aup = (Atmp - Abar)
            Adown = ((Atmp - Abar) ** 2).sum()
            Correlations[iin, jjn] = sum(Broiup * Aup) / np.sqrt(Broidown * Adown)

    return Correlations

def get_Circular_Cross_Correlation_and_Shifts_Numpy(input_image_1, input_image_2):
    input_image_1_fft = fft2(input_image_1)
    input_image_2_fft = fft2(input_image_2)
    cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate())))
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in cross_correlation.shape])
    i0, i1 = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    shifts = np.array((i0, i1), dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(cross_correlation.shape)[shifts > midpoints]
    return cross_correlation, shifts


def get_Circular_Cross_Correlation_torch(input_tensor_1, input_tensor_2, flag_fftshift=False):
    # input_image_1_fft = torch_fft2(input_tensor_1)
    # input_image_2_fft = torch_fft2(input_tensor_2)
    # cross_correlation = torch_ifft2((input_image_1_fft * input_image_2_fft.conj())).real
    cross_correlation = torch_ifft2((torch_fft2(input_tensor_1) * torch_fft2(input_tensor_2).conj())).real
    if flag_fftshift:
        (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor_1)
        cross_correlation = fftshift_torch(cross_correlation, shape_len-2)
    return cross_correlation

def get_Circular_Convolution_torch(input_tensor_1, input_tensor_2):
    input_image_1_fft = torch_fft2(input_tensor_1)
    input_image_2_fft = torch_fft2(input_tensor_2)
    cross_correlation = torch_ifft2((input_image_1_fft * input_image_2_fft)).real
    return cross_correlation


def get_Weighted_Circular_Cross_Correlation_torch(input_tensor_1, input_tensor_2, weights_tensor=None):
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor_2)
    weights_tensor = weights_tensor / weights_tensor.sum()

    input_tensor_1_conv_weights = get_Circular_Cross_Correlation_torch(input_tensor_1, weights_tensor, True)  # TODO: why cross-correlation instead of convolution?!?!?!

    input_tensor_2_weighted_mean = (weights_tensor * input_tensor_2).sum([-1, -2], True)  # since weights_tensor is normalized this is a weighted MEAN
    input_tensor_2_minus_weighted_mean = input_tensor_2 - input_tensor_2_weighted_mean  # TODO: change to more meaningful names!
    input_tensor_2_minus_weighted_mean_times_weights = weights_tensor * input_tensor_2_minus_weighted_mean
    input_tensor_2_minus_weighted_mean_times_weights_sum = input_tensor_2_minus_weighted_mean_times_weights.sum([-1, -2], True)

    input_tensor_1_conv_input_tensor_2_white = get_Circular_Cross_Correlation_torch(input_tensor_1, input_tensor_2_minus_weighted_mean_times_weights, True)
    W_Cov_XY = input_tensor_1_conv_input_tensor_2_white - input_tensor_1_conv_weights * input_tensor_2_minus_weighted_mean_times_weights_sum
    W_Cov_XX = get_Circular_Cross_Correlation_torch(input_tensor_1 ** 2, weights_tensor, True) - input_tensor_1_conv_weights ** 2
    W_Cov_YY = (input_tensor_2_minus_weighted_mean ** 2 + weights_tensor).sum([-1, -2], True)
    Denom = torch.sqrt((W_Cov_XX * W_Cov_YY).clip(0))
    Denom = Denom + 1e-6  #just to make sure there aren't any zeros

    #TODO: in the original code there is a tolerance and wherever any Denom element is below the tolerance -> the cross correlation value is zero
    cross_correlation = W_Cov_XY / Denom
    return cross_correlation

from RapidBase.Utils.Classical_DSP.Convolution_Utils import convn_layer_torch
from RapidBase.Utils.Tensor_Manipulation.Pytorch_Numpy_Utils import canny_edge_detection
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import BW2RGB, torch_get_valid_center
def get_Weighted_Circular_Cross_Correlation_Batch_torch(input_tensor, reference_tensor, weights_tensor=None):
    (B, T, C, H, W), shape_len, shape_vec = get_full_shape_torch(input_tensor)

    ### TODO: delete later on: ###
    input_tensor = input_tensor[0].unsqueeze(1)  #turn into [T,C,H,W] instead of [1,T,H,W] - will be taken care of later with Anvil

    ### Get Default Weights Tensor If None Was Inputed: ###
    if weights_tensor is None:
        ### Different Candidate For Cross Correlation Weights: ###
        #TODO: try structure tensor, or maybe something else?
        weights_tensor = torch.ones_like(input_tensor)
        input_tensor_mean = input_tensor.mean(0) #Assuming [T,C,H,W]
        input_tensor_median = input_tensor.median(0)[0]
        input_tensor_std = (input_tensor - input_tensor_mean).std(0)
        input_tensor_relative_std = input_tensor_std / input_tensor_mean
        convn_layer_torch_object = convn_layer_torch()
        input_tensor_gradient_x = convn_layer_torch_object.forward(input_tensor_mean, torch.Tensor([-1,0,1]), -1)
        input_tensor_gradient_y = convn_layer_torch_object.forward(input_tensor_mean, torch.Tensor([-1,0,1]), -2)
        input_tensor_gradient_magnitude = (input_tensor_gradient_x**2 + input_tensor_gradient_y**2).sqrt()
        input_tensor_BGS = input_tensor - input_tensor_mean
        input_tensor_BGS_median = input_tensor_BGS.median(0)[0]
        canny_edge_detection_layer = canny_edge_detection(10)
        blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = \
            canny_edge_detection_layer.forward(BW2RGB(input_tensor_mean.unsqueeze(0)), 3)

        weights_tensor = early_threshold
        # weights_tensor = thin_edges
        # weights_tensor = thresholded
        # weights_tensor = (thresholded > 2).float()
        weights_tensor = weights_tensor * torch_get_valid_center(weights_tensor, 3)

    ### Normalized Weights Tensor: ###
    weights_tensor = weights_tensor / weights_tensor.sum()

    ### Do Weight Cross Correlation Stuff (i will understand it exactly later): ###
    input_tensor_1_conv_weights = get_Circular_Cross_Correlation_torch(input_tensor, weights_tensor, True)  # TODO: why cross-correlation instead of convolution?!?!?!

    input_tensor_2_weighted_mean = (weights_tensor * reference_tensor).sum([-1, -2], True)  # since weights_tensor is normalized this is a weighted MEAN
    input_tensor_2_minus_weighted_mean = reference_tensor - input_tensor_2_weighted_mean  # TODO: change to more meaningful names!
    input_tensor_2_minus_weighted_mean_times_weights = weights_tensor * input_tensor_2_minus_weighted_mean
    input_tensor_2_minus_weighted_mean_times_weights_sum = input_tensor_2_minus_weighted_mean_times_weights.sum([-1, -2], True)

    input_tensor_1_conv_input_tensor_2_white = get_Circular_Cross_Correlation_torch(input_tensor, input_tensor_2_minus_weighted_mean_times_weights, True)
    W_Cov_XY = input_tensor_1_conv_input_tensor_2_white - input_tensor_1_conv_weights * input_tensor_2_minus_weighted_mean_times_weights_sum
    W_Cov_XX = get_Circular_Cross_Correlation_torch(input_tensor ** 2, weights_tensor, True) - input_tensor_1_conv_weights ** 2
    W_Cov_YY = (input_tensor_2_minus_weighted_mean ** 2 + weights_tensor).sum([-1, -2], True)
    Denom = torch.sqrt((W_Cov_XX * W_Cov_YY).clip(0))
    Denom = Denom + 1e-6  # just to make sure there aren't any zeros

    # TODO: in the original code there is a tolerance and wherever any Denom element is below the tolerance -> the cross correlation value is zero
    cross_correlation = W_Cov_XY / Denom
    regular_cross_correlation = get_Circular_Cross_Correlation_torch(input_tensor, reference_tensor, True)

    ### Use fftshift for now to make up for above used fftshift, i'll understand what's going on and take care of it later: ###
    cross_correlation = fftshift_torch(cross_correlation, -2)
    return cross_correlation


def get_Circular_Cross_Correlation_and_Shifts_DiscreteShift_torch(input_image_1, input_image_2):
    input_image_1_fft = torch_fft2(input_image_1)
    input_image_2_fft = torch_fft2(input_image_2)
    cross_correlation = abs(torch_ifft2((input_image_1_fft * input_image_2_fft.conj())))
    cross_correlation_shape = (cross_correlation.shape[2], cross_correlation.shape[3])
    H, W = cross_correlation_shape
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in cross_correlation_shape])

    max_index = torch.argmax(cross_correlation)
    # TODO: write general, multidimentional torch_unravel function just like np.unravel_index
    i0 = max_index//W
    i1 = max_index - i0*W
    shifts = np.array((i0, i1), dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(cross_correlation_shape)[shifts > midpoints]
    return cross_correlation, shifts

def get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(input_image_1, input_image_2):
    input_image_1_fft = torch_fft2(input_image_1)
    input_image_2_fft = torch_fft2(input_image_2)
    cross_correlation = abs(torch_ifft2((input_image_1_fft * input_image_2_fft.conj())))
    cross_correlation_shape = (cross_correlation.shape[2], cross_correlation.shape[3])
    H, W = cross_correlation_shape
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in cross_correlation_shape])

    #TODO: replace this with the more tested stuff from batch_speckle_correlation_1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO: or maybe use this? why is this so much shorter???? because it's in numpy?

    ### Using Indexing!!!: ###
    max_index = torch.argmax(cross_correlation)
    i0 = max_index.item() // W
    i1 = max_index.item() - i0 * W
    shifts = np.array((i0, i1), dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(cross_correlation_shape)[shifts > midpoints]
    row_indices = np.array([i0-1,i0,i0+1])
    col_indices = np.array([i1-1,i1,i1+1])
    # row_indices[row_indices > midpoints[0]] -= cross_correlation_shape[0]
    row_indices[row_indices < 0] += H
    row_indices[row_indices >= cross_correlation_shape[0]] -= cross_correlation_shape[0]
    # col_indices[col_indices > midpoints[1]] -= cross_correlation_shape[1]
    col_indices[col_indices < 0] += W
    col_indices[col_indices >= cross_correlation_shape[1]] -= cross_correlation_shape[1]

    ### Get Sub-Matrix Of Cross Correlation: ###
    # #(1). By getting sub matrix of cross correlation:
    # # CC_sub_matrix = cross_correlation_fftshift[:, :, i0-1:i0+2, i1-1:i1+2]
    # # CC_sub_matrix = cross_correlation[:, :, i0-1:i0+2, i1-1:i1+2]
    # W = 3
    # H = 3
    # x_vec = [-1, 0, 1]
    # y_vec =[-1, 0, 1]
    # fitting_points_x = CC_sub_matrix[:, :, 1, :]
    # fitting_points_y = CC_sub_matrix[:, :, :, 1]
    #(2). By Directly Getting the Sub Vectors:
    x_vec = [-1, 0, 1]
    y_vec =[-1, 0, 1]
    fitting_points_x = cross_correlation[:, :, i0, col_indices]
    fitting_points_y = cross_correlation[:, :, row_indices, i1]

    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)

    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2

    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2

    shifts_total = (delta_shiftx, delta_shifty)
    z_max_vec = (x_parabola_max, y_parabola_max)

    ### Correct Shifts: ###
    shifts_sub_pixel = (shifts[0] + delta_shifty, shifts[1] + delta_shiftx)

    return cross_correlation, shifts, shifts_sub_pixel

def shift_matrix_subpixel_torch(original_image_torch, shiftx, shifty):
    ndims = len(original_image_torch.shape)
    H = original_image_torch.shape[-2]
    W = original_image_torch.shape[-1]
    # Get tilt phases k-space:
    x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
    y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
    delta_f1 = 1 / W
    delta_f2 = 1 / H
    f_x = x * delta_f1
    f_y = y * delta_f2
    # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
    f_x = np.fft.fftshift(f_x)
    f_y = np.fft.fftshift(f_y)
    # Build k-space meshgrid:
    [kx, ky] = np.meshgrid(f_x, f_y)
    # Frequency vec to tensor:
    kx = torch.Tensor(kx)
    ky = torch.Tensor(ky)
    if ndims == 3:
        kx = kx.unsqueeze(0)
        ky = ky.unsqueeze(0)
    if ndims == 4:
        kx = kx.unsqueeze(0).unsqueeze(0)
        ky = ky.unsqueeze(0).unsqueeze(0)

    ### Expand shiftx & shifty to match input image shape (and if shifts are more thann a scalar i assume you want to multiply B or C): ###
    if type(shiftx) != list and type(shiftx) != tuple:
        shiftx = [shiftx]
        shifty = [shifty]
    if ndims == 3:
        shiftx = torch.Tensor(shiftx).unsqueeze(0).unsqueeze(0)
        shifty = torch.Tensor(shifty).unsqueeze(0).unsqueeze(0)
    elif ndims == 4:
        shiftx = torch.Tensor(shiftx).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shifty = torch.Tensor(shifty).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    ### Displace input image: ###
    displacement_matrix = torch.exp(-(1j * 2 * np.pi * ky * shifty + 1j * 2 * np.pi * kx * shiftx)).to(original_image_torch.device)
    fft_image = torch.fft.fftn(original_image_torch, dim=[-1,-2])
    fft_image_displaced = fft_image * displacement_matrix
    # original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1,-2]).real
    original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1,-2]).abs()

    return original_image_displaced

def cross_correlation_alignment_super_parameter(input_dict, original_frame_current):
    crop_size_after_cross_correlation = input_dict['crop_size_after_cross_correlation']
    crop_size_before_cross_correlation = input_dict['crop_size_before_cross_correlation']
    counter = input_dict['counter']
    number_of_frames_to_average = input_dict['number_of_frames_to_average']

    if counter == 0:
        ### Initialize: ###
        input_dict['aligned_frames_list'] = []
        input_dict['original_frames_list'] = []
        input_dict['original_frames_cropped_list'] = []
        input_dict['clean_frames_list'] = []
        input_dict['inferred_shifts_x_list'] = []
        input_dict['inferred_shifts_y_list'] = []

        aligned_frames_list = []
        original_frames_list = []
        original_frames_cropped_list = []
        clean_frames_list = []

        original_frames_list.append(original_frame_current)
        aligned_frames_list.append(crop_torch_batch(original_frame_current, crop_size_before_cross_correlation, crop_style='center'))
        clean_frames_list.append(crop_torch_batch(original_frame_current, crop_size_after_cross_correlation, crop_style='center'))
        original_frames_cropped_list.append(crop_torch_batch(original_frame_current, crop_size_after_cross_correlation, crop_style='center'))

    else:
        ### Get Images To Align Together: ###
        aligned_frames_list = input_dict['aligned_frames_list']
        original_frames_list = input_dict['original_frames_list']
        original_frames_cropped_list = input_dict['original_frames_cropped_list']
        clean_frames_list = input_dict['clean_frames_list']
        number_of_frames_to_average = input_dict['number_of_frames_to_average']

        ### Append Current Frame To Dict: ###
        original_frames_list.append(original_frame_current)

        ### Get Reference & Current Frames: ###
        original_frames_reference = aligned_frames_list[0]
        # original_frames_reference = original_frames_list[0]
        # original_frames_reference = original_frames_list[-2]  #TODO: at the end this is probably the right thing to do and to keep track of total shifts
        original_frame_current = original_frames_list[-1]

        ### Crop Images: ###
        original_frames_reference_cropped = crop_torch_batch(original_frames_reference, crop_size_before_cross_correlation, crop_style='center')
        original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_before_cross_correlation, crop_style='center')

        ### Use Cross Correlation To Find Translation Between Images: ###
        cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(
            original_frames_reference_cropped, original_frame_current_cropped)


        first_stage_shift_x = shifts_sub_pixel[0]
        first_stage_shift_y = shifts_sub_pixel[1]

        ### Align The Images Using FFT Translation: ###
        original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped,
                                                                            shifts_sub_pixel[1], shifts_sub_pixel[0])

        ### Do Cross Correlation Again: ###
        original_frames_reference_cropped_2 = crop_torch_batch(original_frames_reference_cropped, crop_size_after_cross_correlation, crop_style='center')
        original_frame_current_cropped_warped_2 = crop_torch_batch(original_frame_current_cropped_warped, crop_size_after_cross_correlation, crop_style='center')
        cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(
            original_frames_reference_cropped_2, original_frame_current_cropped_warped_2)
        original_frame_current_cropped_warped = shift_matrix_subpixel_torch(original_frame_current_cropped,
                                                                            shifts_sub_pixel[1]+first_stage_shift_y, shifts_sub_pixel[0]+first_stage_shift_x)

        ### Update Aligned Frames Lists: ###
        aligned_frames_list.append(original_frame_current_cropped_warped)

        ### Crop Again To Avoid InValid Regions: ###
        original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, crop_size_after_cross_correlation, crop_style='center')
        original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, crop_size_after_cross_correlation, crop_style='center')

        ### Append Cropped Frames: ###
        original_frames_cropped_list.append(original_frame_current_cropped)

        ### Average Aligned Frames: ###
        clean_frame_current = torch.zeros((1, 1, original_frame_current_cropped_warped[0].shape[-2],
                                           original_frame_current_cropped_warped[0].shape[-1])).cuda()
        for i in np.arange(len(aligned_frames_list)):
            # clean_frame_current += aligned_frames_list[i]
            clean_frame_current += crop_torch_batch(aligned_frames_list[i], crop_size_after_cross_correlation, crop_style='center')
        clean_frame_current = clean_frame_current / len(aligned_frames_list)
        clean_frames_list.append(clean_frame_current)

    ### Update Lists: ###
    if len(aligned_frames_list) > number_of_frames_to_average:
        aligned_frames_list.pop(0)
        original_frames_list.pop(0)
        original_frames_cropped_list.pop(0)
        clean_frames_list.pop(0)

    ### Update Super Dictionary: ###
    input_dict['aligned_frames_list'] = aligned_frames_list
    input_dict['original_frames_list'] = original_frames_list
    input_dict['clean_frames_list'] = clean_frames_list
    input_dict['original_frames_cropped_list'] = original_frames_cropped_list
    input_dict['counter'] += 1
    return input_dict

def cross_correlation_alignment_super_parameter_efficient(input_dict, original_frame_current):
    crop_size_after_cross_correlation = input_dict['crop_size_after_cross_correlation']
    crop_size_before_cross_correlation = input_dict['crop_size_before_cross_correlation']
    counter = input_dict['counter']
    number_of_frames_to_average = input_dict['number_of_frames_to_average']

    if counter == 0:
        ### Initialize: ###
        input_dict['aligned_frames_list'] = []
        input_dict['original_frames_list'] = []
        input_dict['original_frames_cropped_list'] = []
        input_dict['clean_frames_list'] = []
        input_dict['inferred_shifts_x_list'] = []
        input_dict['inferred_shifts_y_list'] = []
        input_dict['original_frames_fft_list'] = []

        aligned_frames_list = []
        original_frames_list = []
        original_frames_cropped_list = []
        clean_frames_list = []
        original_frames_fft_list = []

        original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_before_cross_correlation, crop_style='center')
        original_frame_current_cropped_final = crop_torch_batch(original_frame_current, crop_size_after_cross_correlation, crop_style='center')
        original_frame_current_cropped_fft = torch_fft2(original_frame_current_cropped)

        original_frames_list.append(original_frame_current)
        aligned_frames_list.append(original_frame_current_cropped)
        original_frames_fft_list.append(original_frame_current_cropped_fft)
        clean_frames_list.append(original_frame_current_cropped_final)
        original_frames_cropped_list.append(original_frame_current_cropped_final)

        ### Prepare Stuff For Sub-Pixel Shifts: ###
        B, C, H, W = original_frame_current_cropped.shape
        # Get tilt phases k-space:
        x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
        y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
        delta_f1 = 1 / W
        delta_f2 = 1 / H
        f_x = x * delta_f1
        f_y = y * delta_f2
        # Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
        f_x = np.fft.fftshift(f_x)
        f_y = np.fft.fftshift(f_y)
        # Build k-space meshgrid:
        [kx, ky] = np.meshgrid(f_x, f_y)
        # Frequency vec to tensor:
        kx = torch.Tensor(kx)
        ky = torch.Tensor(ky)
        kx = kx.unsqueeze(0).unsqueeze(0)
        ky = ky.unsqueeze(0).unsqueeze(0)
        input_dict['kx'] = kx.cuda()
        input_dict['ky'] = ky.cuda()

    else:
        ### Get Images To Align Together: ###
        aligned_frames_list = input_dict['aligned_frames_list']
        original_frames_list = input_dict['original_frames_list']
        original_frames_cropped_list = input_dict['original_frames_cropped_list']
        clean_frames_list = input_dict['clean_frames_list']
        number_of_frames_to_average = input_dict['number_of_frames_to_average']
        original_frames_fft_list = input_dict['original_frames_fft_list']

        ### Append Current Frame To Dict: ###
        original_frames_list.append(original_frame_current)

        ### Get Reference & Current Frames: ###
        original_frames_reference = aligned_frames_list[0]
        original_frame_current = original_frames_list[-1]

        ### Crop Images: ###
        original_frames_reference_cropped = crop_torch_batch(original_frames_reference, crop_size_before_cross_correlation, crop_style='center')
        original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_before_cross_correlation, crop_style='center')

        ### Append FFT Image To List: ###
        original_frame_current_cropped_fft = torch_fft2(original_frame_current_cropped)
        original_frames_fft_list.append(original_frame_current_cropped_fft)

        ### Get Reference Tensor: ###
        tensor1_fft = original_frames_fft_list[0]
        ### Get Tensor To Shift Compared To Reference Tensor: ###
        tensor2_fft = original_frames_fft_list[-1]
        ### Use FFT To Get Shifts From CC -> get tensor2 aligned to tensor1: ###
        cross_correlation = torch.fft.ifftn(tensor1_fft * tensor2_fft.conj(), dim=[-2, -1]).abs()  # CC=IFFT(FFT(A)*FFT_CONJ(B))
        cross_correlation_shape = (cross_correlation.shape[2], cross_correlation.shape[3])
        H, W = cross_correlation_shape
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in cross_correlation_shape])
        ### Using Indexing!!!: ###
        max_index = torch.argmax(cross_correlation)
        i0 = max_index.item() // W
        i1 = max_index.item() - i0 * W
        shifts = np.array((i0, i1), dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(cross_correlation_shape)[shifts > midpoints]
        row_indices = np.array([i0 - 1, i0, i0 + 1])
        col_indices = np.array([i1 - 1, i1, i1 + 1])
        row_indices[row_indices < 0] += H
        row_indices[row_indices >= cross_correlation_shape[0]] -= cross_correlation_shape[0]
        col_indices[col_indices < 0] += W
        col_indices[col_indices >= cross_correlation_shape[1]] -= cross_correlation_shape[1]
        # Get SubPixel Accuracy By Directly Getting the Sub Vectors:
        x_vec = [-1, 0, 1]
        y_vec = [-1, 0, 1]
        fitting_points_x = cross_correlation[:, :, i0, col_indices]
        fitting_points_y = cross_correlation[:, :, row_indices, i1]
        # fit a parabola over the CC values: #
        [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
        [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
        # find the sub-pixel max value and location using the parabola coefficients: #
        delta_shiftx = -b_x / (2 * a_x)
        x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
        delta_shifty = -b_y / (2 * a_y)
        y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
        shifts_total = (delta_shiftx, delta_shifty)
        z_max_vec = (x_parabola_max, y_parabola_max)
        ### Correct Shifts: ###
        shifts_vec = (shifts[0] + delta_shifty, shifts[1] + delta_shiftx)
        ### Shift Tensor2 For Maximum Valid Percent: ###
        shiftx_torch = torch.Tensor([float(shifts_vec[0])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        shifty_torch = torch.Tensor([float(shifts_vec[1])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        kx = input_dict['kx']
        ky = input_dict['ky']
        displacement_matrix = torch.exp(-(1j * 2 * np.pi * ky * shifty_torch + 1j * 2 * np.pi * kx * shiftx_torch))
        original_frame_current_cropped_warped = torch.fft.ifftn(tensor2_fft * displacement_matrix, dim=[-1, -2]).real

        ### Update Aligned Frames Lists: ###
        aligned_frames_list.append(original_frame_current_cropped_warped)

        ### Crop Again To Avoid InValid Regions: ###
        original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, crop_size_after_cross_correlation, crop_style='center')
        original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, crop_size_after_cross_correlation, crop_style='center')

        ### Append Cropped Frames: ###
        original_frames_cropped_list.append(original_frame_current_cropped)

        ### Average Aligned Frames: ###
        clean_frame_current = torch.zeros((1, 1, original_frame_current_cropped_warped[0].shape[-2],
                                           original_frame_current_cropped_warped[0].shape[-1])).cuda()
        for i in np.arange(len(aligned_frames_list)):
            # clean_frame_current += aligned_frames_list[i]
            clean_frame_current += crop_torch_batch(aligned_frames_list[i], crop_size_after_cross_correlation, crop_style='center')
        clean_frame_current = clean_frame_current / len(aligned_frames_list)
        clean_frames_list.append(clean_frame_current)

    ### Update Lists: ###
    if len(aligned_frames_list) > number_of_frames_to_average:
        aligned_frames_list.pop(0)
        original_frames_list.pop(0)
        original_frames_cropped_list.pop(0)
        clean_frames_list.pop(0)

    ### Update Super Dictionary: ###
    input_dict['aligned_frames_list'] = aligned_frames_list
    input_dict['original_frames_list'] = original_frames_list
    input_dict['clean_frames_list'] = clean_frames_list
    input_dict['original_frames_fft_list'] = original_frames_fft_list
    input_dict['original_frames_cropped_list'] = original_frames_cropped_list
    input_dict['counter'] += 1
    return input_dict

############################################################################################################################################################





############################################################################################################################################################
### Speckle Movie Stuff: ###
def analyze_full_speckle_movie(input_array):
    #########################################################################################################
    T,H,W = input_array.shape
    number_of_samples = T
    input_tensor = torch.Tensor(input_array)
    audio_stretch_factor = 20
    #########################################################################################################

    ########################################################################################################
    ### TODO: delete. Make Speckle Patterns: ###
    original_song = np.load('/home/mafat/PycharmProjects/ofra_haza.npy') / 255 * 1
    # play_obj = sa.play_buffer((original_song*255).astype(np.int16), 1, 2, 22050)
    X_pixels_per_cycle = 250
    number_of_samples = 1000 * 100
    x = np.linspace(0, number_of_samples, number_of_samples)
    Fx = 1 / X_pixels_per_cycle
    # bla = np.sin(2 * np.pi * (Fx * x)) * 5
    bla = original_song.cumsum()
    bla = torch.Tensor(bla)
    bla = bla[0:number_of_samples]
    # bla = torch.ones_like(bla).cuda() * 5
    shift_layer = Shift_Layer_Torch()
    speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(3, 70, 0, 1, 1, 0)
    speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    C, H, W = speckle_pattern.shape
    ### Get Input Tensor: ###
    # input_tensor = torch.zeros((number_of_samples, H, W)).cuda()
    # input_tensor[0] = speckle_pattern
    input_tensor = torch.cat([speckle_pattern] * number_of_samples, 0)
    current_shift = 0
    ### Actually Shift: ###
    # (1). Batch Processing:
    input_tensor = shift_layer.forward(input_tensor, bla.cpu().numpy(), bla.cpu().numpy())
    input_tensor = crop_torch_batch(input_tensor, 64)
    # #########################################################################################################

    #########################################################################################################
    ### Loop Over Different Parts Of The Movie (in case it's too long) & Analyze: ###
    number_of_samples_per_batch = 1024
    number_of_batches = number_of_samples / number_of_samples_per_batch
    shift_x_list = []
    shift_y_list = []
    for batch_counter in np.arange(number_of_batches):
        print(batch_counter)
        start_index = int(batch_counter * number_of_samples_per_batch)
        stop_index = int((batch_counter + 1) * number_of_samples_per_batch)
        current_tensor = input_tensor[start_index:stop_index, :, :]
        shift_x_current, shift_y_current = batch_speckle_correlation_1(current_tensor.cuda())
        shift_x_list.append(shift_x_current)
        shift_y_list.append(shift_y_current)
        play_obj = sd.play((shift_x_current.cpu().numpy() * 255 * audio_stretch_factor).clip(-255, 255).astype(np.int16), 22050)
        del current_tensor
        torch.cuda.empty_cache()
    ###
    # for batch_counter in np.arange(len(shift_x_list)):
    #     play_obj = sd.play(
    #         (shift_x_list[batch_counter].cpu().numpy() * 255 * audio_stretch_factor).clip(-255, 255).astype(np.int16), 22050)
    # ###
    shift_x_final = torch.cat(shift_x_list)
    shift_y_final = torch.cat(shift_y_list)
    play_obj = sa.play_buffer((shift_x_final.cpu().numpy() * 255 * audio_stretch_factor).clip(-255, 255).astype(np.int16), 1, 2, 22050)
    play_obj = sd.play((shift_x_final.cpu().numpy() * 255 * audio_stretch_factor).clip(-255, 255).astype(np.int16), 22050)
    #########################################################################################################

    return shift_x_final, shift_y_final

def local_sum(input_tensor, outer_frame_to_disregard=5):
    T,H,W = input_tensor.shape
    start_index_rows = outer_frame_to_disregard
    stop_index_rows = H - outer_frame_to_disregard
    start_index_cols = outer_frame_to_disregard
    stop_index_cols = W - outer_frame_to_disregard
    return input_tensor[:, start_index_rows:stop_index_rows, start_index_cols:stop_index_cols].sum(-1).sum(-1)

def batch_speckle_correlation_1(input_tensor):
    def fit_polynomial_torch(x, y):
        # solve for 2nd degree polynomial deterministically using three points
        a = (y[:,2] + y[:,0] - 2 * y[:,1]) / 2
        b = -(y[:,0] + 2 * a * x[1] - y[:,1] - a)
        c = y[:,1] - b * x[1] - a * x[1] ** 2
        return [c, b, a]

    # #########################################################################################################
    # ### TODO: delete. Make Speckle Patterns: ###
    # original_song = np.load('/home/mafat/PycharmProjects/ofra_haza.npy') / 255 * 5
    # # play_obj = sa.play_buffer((original_song*255).astype(np.int16), 1, 2, 22050)
    # X_pixels_per_cycle = 250
    # number_of_samples = 1000*50
    # x = np.linspace(0, number_of_samples, number_of_samples)
    # Fx = 1 / X_pixels_per_cycle
    # # bla = np.sin(2 * np.pi * (Fx * x)) * 5
    # bla = original_song.cumsum()
    # bla = torch.Tensor(bla).cuda()
    # bla = bla[0:number_of_samples]
    # # bla = torch.ones_like(bla).cuda() * 5
    # shift_layer = Shift_Layer_Torch()
    # speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(3, 64, 0, 1, 1, 0)
    # speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    # C,H,W = speckle_pattern.shape
    # ### Get Input Tensor: ###
    # # input_tensor = torch.zeros((number_of_samples, H, W)).cuda()
    # # input_tensor[0] = speckle_pattern
    # input_tensor = torch.cat([speckle_pattern]*number_of_samples, 0) + 5
    # current_shift = 0
    # ### Actually Shift: ###
    # #(1). Batch Processing:
    # input_tensor = shift_layer.forward(input_tensor, bla.cpu().numpy(), bla.cpu().numpy())
    # # np.save('/home/mafat/PycharmProjects/speckle_movie.npy', input_tensor.numpy())
    # # #(2). Serial Processing
    # # tic()
    # # for i in np.arange(1, number_of_samples):
    # #     current_shift = bla[i]
    # #     input_tensor[i:i+1] = shift_layer.forward(speckle_pattern, current_shift, current_shift)
    # # toc()
    # # for i in np.arange(10):
    # #     batch_size = 1024
    # #     start_index = batch_size*i
    # #     stop_index = batch_size*(i+1)
    # #     sd.play(original_song[start_index:stop_index], 22050)
    # #     sd.wait()
    # #########################################################################################################

    #########################################################################################################
    ### Get CC: ###
    #input tensor [T,H,W]
    T,H,W = input_tensor.shape
    cross_correlation_shape = (H,W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1,-2])
    #(1). Regular Cross Corerlation:
    output_CC = torch.fft.ifftn(input_tensor_fft[0:T - 1, :, :] * input_tensor_fft[1:T, :, :].conj(), dim=[-1, -2]).real
    # TODO: here be careful, must advance in steps of 2 frames, each step will look 1 frame forward and 1 frame backward. number of input frames must be odd
    # # #(2). Normalized Cross Correlation:
    # output_CC = torch.fft.ifftn(input_tensor_fft[0:T - 1, :, :] * input_tensor_fft[1:T, :, :].conj(), dim=[-1, -2]).real
    # zero_padding_size = max_shift = 3
    # zeros_map = torch.ones((T//2, H - 2*zero_padding_size, W - 2*zero_padding_size))
    # zeros_map = torch.nn.functional.pad(zeros_map, [zero_padding_size,zero_padding_size,zero_padding_size,zero_padding_size], value=0)
    # input_tensor[0::2] = input_tensor[0::2] * zeros_map  #TODO: probably instead of using indices use a better predefined zeros_map with T frames
    # xcorr_odd = torch.fft.ifftn(input_tensor_fft[0::2,:,:] * input_tensor_fft[1::2,:,:].conj(), dim=[-1,-2]).real  #TODO: no reason part B of the multiplication should be different in the two rows, understand this
    # xcorr_even = torch.fft.ifftn(input_tensor_fft[2::2,:,:] * input_tensor_fft[1:T-1:2,:,:].conj(), dim=[-1,-2]).real
    # input_tensor_even = input_tensor[1:T-1:2]  #A in script, has T-1 samples!!!!
    # input_tensor_odd = input_tensor[0::2]  #T in script, has T samples!!!
    # local_sum_A = local_sum(input_tensor_even, zero_padding_size).unsqueeze(-1).unsqueeze(-1)
    # local_sum_A2 = local_sum(input_tensor_even**2, zero_padding_size).unsqueeze(-1).unsqueeze(-1)
    # M = H - 2*max_shift
    # N = W - 2*max_shift
    # diff_local_sums_A = local_sum_A2 - (local_sum_A**2)/(M*N)
    # denom_A = torch.sqrt(diff_local_sums_A.clamp(0))
    # local_sum_B = input_tensor_odd.sum([-1,-2]).unsqueeze(-1).unsqueeze(-1)
    # local_sum_B2 = (input_tensor_odd**2).sum([-1,-2]).unsqueeze(-1).unsqueeze(-1)
    # diff_local_sums_B = local_sum_B2 - (local_sum_B**2)/(M*N)
    # denom_B = torch.sqrt(diff_local_sums_B.clamp(0))
    # CC_odd = (xcorr_odd[0:-1] - local_sum_A*local_sum_B[0:-1]/(M*N)) / (denom_A * denom_B[0:-1])
    # CC_even = (xcorr_even - local_sum_A*local_sum_B[1:]/(M*N)) / (denom_A * denom_B[1:])
    # output_CC = torch.cat([CC_odd, CC_even], 0)
    # output_CC = torch.cat((output_CC[0:1], output_CC), 0 ) #TODO: delete!!!! this is just temporary
    # #TODO: don't forget to flip the sign of the odd(even?) shifts!!!!
    #########################################################################################################

    #########################################################################################################
    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T-1, H*W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W
    i1 = output_CC_flattened_indices - i0 * W
    i0[i0 > midpoints[0]] -= H
    i1[i1 > midpoints[1]] -= W
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Correct For Wrap-Arounds: ###
    #(1). Below Zero:
    i0_minus1[i0_minus1 < 0] += H
    i1_minus1[i1_minus1 < 0] += W
    i0_plus1[i0_plus1 < 0] += H
    i1_plus1[i1_plus1 < 0] += W
    i0[i0 < 0] += H
    i1[i1 < 0] += W
    #(2). Above Max:
    i0[i0 > H] -= H
    i1[i1 > H] -= W
    i0_plus1[i0_plus1 > H] -= H
    i1_plus1[i1_plus1 > W] -= W
    i0_minus1[i0_minus1 > W] -= W
    i1_minus1[i1_minus1 > W] -= W
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0*W
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1*W
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1*W
    output_CC_flattened_indices_i0 = i1 + i0*W
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * W
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * W
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, H*W)
    output_CC_flattened_values_i0 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0.long())
    output_CC_flattened_values_i0_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_minus1)
    output_CC_flattened_values_i0_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i0_plus1)
    output_CC_flattened_values_i1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1)
    output_CC_flattened_values_i1_minus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_minus1)
    output_CC_flattened_values_i1_plus1 = torch.gather(output_CC, 1, output_CC_flattened_indices_i1_plus1)
    #########################################################################################################

    #########################################################################################################
    ### Get Sub Pixel Shifts: ###
    fitting_points_x = torch.cat([output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat([output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    # fit a parabola over the CC values: #
    [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    # find the sub-pixel max value and location using the parabola coefficients: #
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shift_x = i1_original.squeeze() + delta_shiftx
    shift_y = i0_original.squeeze() + delta_shifty

    # shift_x[1::2] = shift_x[1::2]*-1
    # play_obj = sa.play_buffer((shift_x.cpu().numpy() * 255 * 50).astype(np.int16), 1, 2, 22050)
    #########################################################################################################


    # #########################################################################################################
    # ### Find Shifts Serial Loop: ###
    # row_shifts_list = []
    # col_shifts_list = []
    # for frame_index in np.arange(output_CC.shape[0]):
    #     col_indices_current = col_indices[frame_index]
    #     row_indices_current = row_indices[frame_index]
    #     integer_shift_current_row = i0[frame_index]
    #     integer_shift_current_col = i1[frame_index]
    #     ### Get Sub-Matrix Of Cross Correlation: ###
    #     # (2). By Directly Getting the Sub Vectors:
    #     x_vec = [-1, 0, 1]
    #     y_vec = [-1, 0, 1]
    #     fitting_points_x = output_CC[frame_index, row_indices_current[1], col_indices_current]
    #     fitting_points_y = output_CC[frame_index, row_indices_current, col_indices_current[1]]
    #
    #     # fit a parabola over the CC values: #
    #     [c_x, b_x, a_x] = fit_polynomial_torch(x_vec, fitting_points_x)
    #     [c_y, b_y, a_y] = fit_polynomial_torch(y_vec, fitting_points_y)
    #
    #     # find the sub-pixel max value and location using the parabola coefficients: #
    #     delta_shiftx = -b_x / (2 * a_x)
    #     x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    #
    #     delta_shifty = -b_y / (2 * a_y)
    #     y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    #
    #     shifts_total = (delta_shiftx, delta_shifty)
    #     z_max_vec = (x_parabola_max, y_parabola_max)
    #
    #     ### Correct Shifts: ###
    #     row_shifts_list.append(integer_shift_current_row + delta_shifty)
    #     col_shifts_list.append(integer_shift_current_col + delta_shiftx)
    #     # shifts_sub_pixel = (integer_shift_current_row + delta_shifty, integer_shift_current_col + delta_shiftx)
    # row_shifts_vec = np.array(row_shifts_list)
    # col_shifts_vec = np.array(col_shifts_list)
    # play_obj = sa.play_buffer((col_shifts_vec*255).astype(np.int16), 1, 2, 22050)
    # #########################################################################################################
    return shift_x, shift_y

def analyze_real_speckles_movies():
    filenames_list = path_get_all_filenames_from_folder('/home/mafat/nussinson/output/split_session/pre_analysis')
    shift_x_list = []
    shift_y_list = []

    input_tensor1 = np.load(filenames_list[0])  # [1024, 64, 64]
    input_tensor1 = torch.Tensor(input_tensor1).cuda()[1:-1]
    input_tensor2 = np.load(filenames_list[1])  # [1024, 64, 64]
    input_tensor2 = torch.Tensor(input_tensor2).cuda()[1:-1]
    input_tensor3 = np.load(filenames_list[2])  # [1024, 64, 64]
    input_tensor3 = torch.Tensor(input_tensor3).cuda()[1:-1]
    input_tensor = torch.cat((input_tensor1, input_tensor2, input_tensor3),0)

    for counter, filename in enumerate(filenames_list):
        input_tensor = np.load(filename) #[1024, 64, 64]
        input_tensor = torch.Tensor(input_tensor).cuda()
        input_tensor = input_tensor[1:]
        if counter == 0:
            previous_last_tensor = torch.zeros_like(input_tensor[0:1]).cuda()
        input_tensor = torch.cat((previous_last_tensor, input_tensor), 0)
        shift_x, shift_y = batch_speckle_correlation_1(input_tensor)
        previous_last_tensor = input_tensor[-2].unsqueeze(0)
        shift_x_list.append(shift_x)
        shift_y_list.append(shift_y)
    shift_x_final = torch.cat(shift_x_list)
    shift_y_final = torch.cat(shift_y_list)
    shift_x_final = shift_x_final.cpu().numpy()
    shift_y_final = shift_y_final.cpu().numpy()
    shift_x_final = shift_x_final[1:] #TODO: problem with the very first term!
    shift_y_final = shift_y_final[1:] #TODO: problem with the very first term!
    shift_x_final = (shift_x_final * 255 * 50).astype(np.int16)
    shift_y_final = (shift_y_final * 255 * 50).astype(np.int16)
    play_obj = sd.play(shift_x_final, 6000)
############################################################################################################################################################


############################################################################################################################################################
def similarity_torch(input_image_1, input_image_2):
    """Return similarity transformed image im1 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.
    A similarity transformation is an affine transformation with isotropic
    scale and without shear.
    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.
    """

    if input_image_1.shape != input_image_2.shape:
        raise ValueError('images must have same shapes')

    ### Get The FFTs Of Both Images: ###
    input_image_1_fft_abs = fftshift_torch(torch.abs(torch_fft2(input_image_1)))
    input_image_2_fft_abs = fftshift_torch(torch.abs(torch_fft2(input_image_2)))

    ### Filter Images By HighPass Filter By Multiplying In Fourier Space: ###
    h = highpass(input_image_1_fft_abs.shape)
    input_image_1_fft_abs *= h
    input_image_2_fft_abs *= h

    ### Transform FFTs To LogPolar Base: ###
    input_image_1_fft_abs_LogPolar, log_base = logpolar(input_image_1_fft_abs)
    input_image_2_fft_abs_LogPolar, log_base = logpolar(input_image_2_fft_abs)

    ### Calculate The Phase Cross Correlation Of The Log Transformed Images, Then Get The Max Value To Extract (Angle,Scale): ###
    input_image_1_fft_abs_LogPolar = torch_fft2(input_image_1_fft_abs_LogPolar)
    input_image_2_fft_abs_LogPolar = torch_fft2(input_image_2_fft_abs_LogPolar)
    r0 = abs(input_image_1_fft_abs_LogPolar) * abs(input_image_2_fft_abs_LogPolar)
    phase_cross_correlation = abs(torch_ifft2((input_image_1_fft_abs_LogPolar * input_image_2_fft_abs_LogPolar.conjugate()) / r0))
    max_index = torch.argmax(phase_cross_correlation)
    max_index = np.atleast_1d(max_index.cpu().numpy())[0]
    numpy_shape = phase_cross_correlation.shape.cpu().numpy()
    i0, i1 = np.unravel_index(max_index, numpy_shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    ### Correct Angle For Wrap-Arounds: ###
    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0

    ### Scale & Rotate Second Image To Match Reference Image Before Finding Translation: ###
    input_image_2_scaled_rotated = ndii.zoom(input_image_2, 1.0 / scale)
    input_image_2_scaled_rotated = ndii.rotate(input_image_2_scaled_rotated, angle)

    ### Make Sure Both Images Have The Same Dimensions By Using Inserting The Smaller Matrix Into A Larger One: ### #TODO: what's more efficient? this or padding?
    if input_image_2_scaled_rotated.shape < input_image_1.shape:
        t = np.zeros_like(input_image_1)
        t[: input_image_2_scaled_rotated.shape[0],
        : input_image_2_scaled_rotated.shape[1]] = input_image_2_scaled_rotated
        input_image_2_scaled_rotated = t
    elif input_image_2_scaled_rotated.shape > input_image_1.shape:
        input_image_2_scaled_rotated = input_image_2_scaled_rotated[: input_image_1.shape[0], : input_image_1.shape[1]]

    ### Get Translation Using Phase Cross Correlation: ###
    input_image_1_fft = torch_fft2(input_image_1)  # TODO: can save on calculation here since i calculated this above
    input_image_2_fft = torch_fft2(
        input_image_2_scaled_rotated)  # TODO: can probably save on calculation here if i use FFT interpolation for zoom+rotation above
    phase_cross_correlation = abs(torch_ifft2(
        (input_image_1_fft * input_image_2_fft.conjugate()) / (abs(input_image_1_fft) * abs(input_image_2_fft))))
    max_index = torch.argmax(phase_cross_correlation)
    max_index = np.atleast_1d(max_index.cpu().numpy())[0]
    numpy_shape = phase_cross_correlation.shape.cpu().numpy()
    t0, t1 = np.unravel_index(max_index, numpy_shape)
    ### Correct For FFT_Shift Wrap-Arounds: ###
    if t0 > input_image_1_fft.shape[0] // 2:
        t0 -= input_image_1_fft.shape[0]
    if t1 > input_image_1_fft.shape[1] // 2:
        t1 -= input_image_1_fft.shape[1]

    ### Shift Second Image According To Found Translation: ###
    input_image_2_scaled_rotated_translated = ndii.shift(input_image_2_scaled_rotated, [t0,
                                                                                        t1])  # TODO: see what is faster, bilinaer interpolation or (multiplication+FFT)?

    ### Correct Parameters For ndimage's Internal Processing: ###
    if angle > 0.0:
        d = int(int(input_image_2.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(input_image_2.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (input_image_2.shape[1] - 1) / (int(input_image_2.shape[1] / scale) - 1)

    return input_image_2_scaled_rotated_translated, scale, angle, [-t0, -t1]

def similarity_numpy(input_image_1=None, input_image_2=None):
    """Return similarity transformed image input_image_2 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.
    A similarity transformation is an affine transformation with isotropic
    scale and without shear.
    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.
    """
    shift_x = np.array([0])
    shift_y = np.array([0])
    scale = np.array([1.])
    rotation_angle = np.array([5])  # [degrees]
    input_image_1 = read_image_default_torch()
    input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)
    input_image_1 = input_image_1[0, 0].numpy()
    input_image_2 = input_image_2[0, 0].numpy()
    input_image_1 = crop_numpy_batch(input_image_1, 1000)
    input_image_2 = crop_numpy_batch(input_image_2, 1000)
    input_image_1_original = input_image_1
    input_image_2_original = input_image_2

    if input_image_1.shape != input_image_2.shape:
        raise ValueError('images must have same shapes')
    if len(input_image_1.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_image_1 = difference_of_gaussians(input_image_1, 5, 20)  # TODO: the more you smear the better?
    input_image_2 = difference_of_gaussians(input_image_2, 5, 20)

    input_image_1_fft_abs = fftshift(abs(fft2(input_image_1)))
    input_image_2_fft_abs = fftshift(abs(fft2(input_image_2)))

    h = highpass(input_image_1_fft_abs.shape)
    input_image_1_fft_abs *= h
    input_image_2_fft_abs *= h
    del h

    # ### Numpy LogPolar: ###
    # input_image_1_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar(input_image_1_fft_abs)
    # input_image_2_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar(input_image_2_fft_abs)
    # ### Torch LogPolar: ###
    # input_image_1_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0))
    # input_image_2_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_image_2_fft_abs).unsqueeze(0).unsqueeze(0))
    # imshow(input_image_1_fft_abs_LogPolar_numpy); figure(); imshow_torch(input_image_1_fft_abs_LogPolar_torch)
    ### Scipy-Torch LogPolar: ###
    input_image_1_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_2_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_2_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    # imshow(input_image_1_fft_abs_LogPolar_numpy); figure(); imshow_torch(input_image_1_fft_abs_LogPolar_torch)
    # grid_diff = x - x_torch.cpu().numpy()[0,:,:,0]
    # ### Scipy LogPolar: ###
    # input_shape = input_image_1_fft_abs.shape
    # radius_to_use = input_shape[0] // 8  # only take lower frequencies. #TODO: Notice! the choice of the log-polar conversion is for your own choosing!!!!!!
    # input_image_1_fft_abs_LogPolar_numpy = warp_polar(input_image_1_fft_abs, radius=radius_to_use, output_shape=input_shape, scaling='log', order=0)
    # input_image_2_fft_abs_LogPolar_numpy = warp_polar(input_image_2_fft_abs, radius=radius_to_use, output_shape=input_shape, scaling='log', order=0)
    # ### External LopPolar: ###
    # # logpolar_grid_tensor = logpolar_grid(input_image_1_fft_abs.shape, ulim=(None, np.log(2.) / 2.), vlim=(0, np.pi), out=None, device='cuda').cuda()
    # logpolar_grid_tensor = logpolar_grid(input_image_1_fft_abs.shape, ulim=(np.log(0.01), np.log(1)), vlim=(0, np.pi), out=None, device='cuda').cuda()
    # input_image_1_fft_abs_LogPolar_torch = F.grid_sample(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0).cuda(), logpolar_grid_tensor.unsqueeze(0))
    # input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
    # imshow_torch(input_image_1_fft_abs_LogPolar_torch)

    input_image_1_fft_abs_LogPolar_numpy = fft2(input_image_1_fft_abs_LogPolar_numpy)
    input_image_2_fft_abs_LogPolar_numpy = fft2(input_image_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_image_1_fft_abs_LogPolar_numpy) * abs(input_image_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(
        ifft2((input_image_1_fft_abs_LogPolar_numpy * input_image_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(
            ifft2((input_image_2_fft_abs_LogPolar_numpy * input_image_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
        # phase_cross_correlation[0,0] = 0
        i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
        angle = -180.0 * i0 / phase_cross_correlation.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError('images are not compatible. Scale change > 1.8')

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    angle = angle * 2
    print(angle)

    input_image_2_scaled_rotated = ndii.zoom(input_image_2_original, 1.0 / scale)
    input_image_2_scaled_rotated = ndii.rotate(input_image_2_scaled_rotated, -angle)

    if input_image_2_scaled_rotated.shape < input_image_1.shape:
        t = np.zeros_like(input_image_1)
        t[: input_image_2_scaled_rotated.shape[0],
        : input_image_2_scaled_rotated.shape[1]] = input_image_2_scaled_rotated
        input_image_2_scaled_rotated = t
    elif input_image_2_scaled_rotated.shape > input_image_1.shape:
        input_image_2_scaled_rotated = input_image_2_scaled_rotated[: input_image_1.shape[0], : input_image_1.shape[1]]

    # imshow(input_image_2_scaled_rotated); figure(); imshow(input_image_1_original)
    # imshow(input_image_2_scaled_rotated - input_image_1_original)

    ### Crop Images Before Translational Cross Correlation: ###
    input_image_1_original_crop = crop_numpy_batch(input_image_1_original, 800)
    input_image_2_scaled_rotated_crop = crop_numpy_batch(input_image_2_scaled_rotated, 800)
    # imshow(input_image_2_scaled_rotated_crop); figure(); imshow(input_image_1_original_crop)
    # imshow(input_image_2_scaled_rotated_crop - input_image_1_original_crop)

    input_image_1_fft = fft2(input_image_1_original_crop)
    input_image_2_fft = fft2(input_image_2_scaled_rotated_crop)
    # TODO: probably need to center crop images here!
    phase_cross_correlation = abs(
        ifft2((input_image_1_fft * input_image_2_fft.conjugate()) / (abs(input_image_1_fft) * abs(input_image_2_fft))))
    # phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate())))
    t0, t1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)

    if t0 > input_image_1_fft_abs.shape[0] // 2:
        t0 -= input_image_1_fft.shape[0]
    if t1 > input_image_1_fft_abs.shape[1] // 2:
        t1 -= input_image_1_fft.shape[1]

    input_image_2_scaled_rotated_shifted = ndii.shift(input_image_2_scaled_rotated, [t0, t1])
    # figure(); imshow(input_image_2_scaled_rotated_shifted); figure(); imshow(input_image_1_original_crop)
    # figure(); imshow(input_image_2_scaled_rotated_shifted - input_image_1_original_crop)

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int(int(input_image_2.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(input_image_2.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (input_image_2.shape[1] - 1) / (int(input_image_2.shape[1] / scale) - 1)

    return input_image_2_scaled_rotated_shifted, scale, angle, [-t0, -t1]


def similarity_numpy_2(input_image_1=None, input_image_2=None):
    """Return similarity transformed image input_image_2 and transformation parameters.
    Transformation parameters are: isotropic scale factor, rotation angle (in
    degrees), and translation vector.
    A similarity transformation is an affine transformation with isotropic
    scale and without shear.
    Limitations:
    Image shapes must be equal and square.
    All image areas must have same scale, rotation, and shift.
    Scale change must be less than 1.8.
    No subpixel precision.
    """
    shift_x = np.array([-20])
    shift_y = np.array([40])
    scale = np.array([1.0])
    rotation_angle = np.array([15])  # [degrees]
    input_image_1 = read_image_default_torch()
    input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)
    input_image_1 = input_image_1[0, 0].numpy()
    input_image_2 = input_image_2[0, 0].numpy()
    input_image_1 = crop_numpy_batch(input_image_1, 1000)
    input_image_2 = crop_numpy_batch(input_image_2, 1000)
    input_image_1_original = input_image_1
    input_image_2_original = input_image_2
    # imshow(input_image_1_original); figure(); imshow(input_image_2_original)

    if input_image_1.shape != input_image_2.shape:
        raise ValueError('images must have same shapes')
    if len(input_image_1.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_image_1 = difference_of_gaussians(input_image_1, 5, 20)  # TODO: the more you smear the better?
    input_image_2 = difference_of_gaussians(input_image_2, 5, 20)

    input_image_1_fft_abs = fftshift(abs(fft2(input_image_1)))
    input_image_2_fft_abs = fftshift(abs(fft2(input_image_2)))

    h = highpass(input_image_1_fft_abs.shape)
    input_image_1_fft_abs *= h
    input_image_2_fft_abs *= h
    del h

    ### Scipy-Torch LogPolar: ###
    # TODO: add possibility of divide factor for radius
    input_image_1_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_2_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_2_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]

    ### Calculate Cross Correlation: ###
    input_image_1_fft_abs_LogPolar_numpy = fft2(input_image_1_fft_abs_LogPolar_numpy)
    input_image_2_fft_abs_LogPolar_numpy = fft2(input_image_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_image_1_fft_abs_LogPolar_numpy) * abs(input_image_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(
        ifft2((input_image_1_fft_abs_LogPolar_numpy * input_image_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(
            ifft2((input_image_2_fft_abs_LogPolar_numpy * input_image_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
        # phase_cross_correlation[0,0] = 0
        i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
        angle = -180.0 * i0 / phase_cross_correlation.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError('images are not compatible. Scale change > 1.8')

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    angle = angle * 2
    print(angle)
    print(scale)

    ### TODO: if to find the rotation and scaling i am actually finding the cross correlation peak of the fft_abs_LogPolar, why can i use that and
    ### TODO: shift the fft_LogPolar just like i do the when finding pure translation, and then use the inverse_LogPolar transform (i'm sure it exists)?????
    # TODO: isn't the above BETTER then rotating the original image around an unknown center of rotation?!?!?!....

    ### Warp Second Image Towards Reference Image: ###
    # #(1). Scipy:
    input_image_2_scaled_rotated = ndii.zoom(input_image_2_original, 1.0 / scale)
    input_image_2_scaled_rotated = ndii.rotate(input_image_2_scaled_rotated, -angle)
    # (2). Torch Affine:
    input_image_1_original_torch = torch.Tensor(input_image_1_original).unsqueeze(0).unsqueeze(0)
    input_image_2_original_torch = torch.Tensor(input_image_2_original).unsqueeze(0).unsqueeze(0)
    input_image_2_original_torch_rotated = warp_tensor_affine_matrix(input_image_2_original_torch, 0, 0, scale, -angle)
    imshow_torch(input_image_1_original_torch - input_image_2_original_torch_rotated)
    ### Crop: ###
    input_image_1_original = input_image_1_original_torch.numpy()[0, 0]
    input_image_2_scaled_rotated = input_image_2_original_torch_rotated.numpy()[0, 0]
    imshow(input_image_1_original - input_image_2_scaled_rotated)

    if input_image_2_scaled_rotated.shape < input_image_1.shape:
        t = np.zeros_like(input_image_1)
        t[: input_image_2_scaled_rotated.shape[0],
        : input_image_2_scaled_rotated.shape[1]] = input_image_2_scaled_rotated
        input_image_2_scaled_rotated = t
    elif input_image_2_scaled_rotated.shape > input_image_1.shape:
        input_image_2_scaled_rotated = input_image_2_scaled_rotated[: input_image_1.shape[0], : input_image_1.shape[1]]

    # imshow(input_image_2_scaled_rotated); figure(); imshow(input_image_1_original)
    # imshow(input_image_2_scaled_rotated - input_image_1_original)

    ### Crop Images Before Translational Cross Correlation: ###
    # input_image_1_original_crop = crop_numpy_batch(input_image_1_original, 800)
    # input_image_2_scaled_rotated_crop = crop_numpy_batch(input_image_2_scaled_rotated, 800)
    input_image_1_original_crop = input_image_1_original
    input_image_2_scaled_rotated_crop = input_image_2_scaled_rotated
    # imshow(input_image_2_scaled_rotated_crop); figure(); imshow(input_image_1_original_crop)
    # imshow(input_image_2_scaled_rotated_crop - input_image_1_original_crop)

    input_image_1_fft = fft2(input_image_1_original_crop)
    input_image_2_fft = fft2(input_image_2_scaled_rotated_crop)
    # TODO: probably need to center crop images here!
    phase_cross_correlation = abs(
        ifft2((input_image_1_fft * input_image_2_fft.conjugate()) / (abs(input_image_1_fft) * abs(input_image_2_fft))))
    # phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate())))
    t0, t1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)

    if t0 > input_image_1_fft_abs.shape[0] // 2:
        t0 -= input_image_1_fft.shape[0]
    if t1 > input_image_1_fft_abs.shape[1] // 2:
        t1 -= input_image_1_fft.shape[1]

    input_image_2_scaled_rotated_shifted = ndii.shift(input_image_2_scaled_rotated, [t0, t1])
    # figure(); imshow(input_image_2_scaled_rotated_shifted); figure(); imshow(input_image_1_original_crop)
    # figure(); imshow(input_image_2_scaled_rotated_shifted - input_image_1_original_crop)

    # correct parameters for ndimage's internal processing
    ### TODO: it does bring the images one on top of each other....but it doesn't give the correct [t0,t1]....meaning something's wrong perhapse below?
    if angle > 0.0:
        d = int(int(input_image_2.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(input_image_2.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (input_image_2.shape[1] - 1) / (int(input_image_2.shape[1] / scale) - 1)

    return input_image_2_scaled_rotated_shifted, scale, angle, [-t0, -t1]

def logpolar_test():
    shift_x = np.array([0])
    shift_y = np.array([0])
    scale = np.array([1])
    rotation_angle = np.array([9])  # [degrees]
    input_image_1 = read_image_default_torch()
    input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)
    input_image_1 = input_image_1[0, 0].numpy()
    input_image_2 = input_image_2[0, 0].numpy()
    input_image_1 = crop_numpy_batch(input_image_1, 1000)
    input_image_2 = crop_numpy_batch(input_image_2, 1000)
    input_image_1_original = input_image_1
    input_image_2_original = input_image_2

    if input_image_1.shape != input_image_2.shape:
        raise ValueError('images must have same shapes')
    if len(input_image_1.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_image_1 = difference_of_gaussians(input_image_1, 5, 20)  # TODO: the more you smear the better?
    input_image_2 = difference_of_gaussians(input_image_2, 5, 20)

    input_image_1_fft_abs = fftshift(abs(fft2(input_image_1)))
    input_image_2_fft_abs = fftshift(abs(fft2(input_image_2)))

    h = highpass(input_image_1_fft_abs.shape)
    input_image_1_fft_abs *= h
    input_image_2_fft_abs *= h
    del h

    ### Numpy LogPolar: ###
    input_image_1_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar(input_image_1_fft_abs)
    input_image_2_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar(input_image_2_fft_abs)

    ### Torch LogPolar: ###
    input_image_1_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_2_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_image_2_fft_abs).unsqueeze(0).unsqueeze(0))
    # imshow(input_image_1_fft_abs_LogPolar_numpy); imshow_torch(input_image_1_fft_abs_LogPolar_torch)
    # grid_diff = x - x_torch.cpu().numpy()[0,:,:,0]

    ### Scipy LogPolar: ###
    input_shape = input_image_1_fft_abs.shape
    radius_to_use = input_shape[
                        0] // 1  # only take lower frequencies. #TODO: Notice! the choice of the log-polar conversion is for your own choosing!!!!!!
    input_image_1_fft_abs_LogPolar_scipy = warp_polar(input_image_1_fft_abs, radius=radius_to_use,
                                                      output_shape=input_shape, scaling='log', order=0)
    input_image_2_fft_abs_LogPolar_scipy = warp_polar(input_image_2_fft_abs, radius=radius_to_use,
                                                      output_shape=input_shape, scaling='log', order=0)
    imshow(input_image_1_fft_abs_LogPolar_scipy);
    imshow_torch(input_image_1_fft_abs_LogPolar_torch)

    ### External LopPolar: ###
    # logpolar_grid_tensor = logpolar_grid(input_image_1_fft_abs.shape, ulim=(None, np.log(2.) / 2.), vlim=(0, np.pi), out=None, device='cuda').cuda()
    logpolar_grid_tensor = logpolar_grid(input_image_1_fft_abs.shape, ulim=(np.log(0.01), np.log(1)),
                                         vlim=(0, 2 * np.pi), out=None, device='cuda').cuda()
    input_image_1_fft_abs_LogPolar_torch_grid = F.grid_sample(
        torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0).cuda(), logpolar_grid_tensor.unsqueeze(0))
    input_image_1_fft_abs_LogPolar_numpy_grid = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    # imshow_torch(input_image_1_fft_abs_LogPolar_torch)

    ### Calculate Phase Cross Correlation: ###
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_scipy
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_scipy

    input_image_1_fft_abs_LogPolar_numpy = fft2(input_image_1_fft_abs_LogPolar_numpy)
    input_image_2_fft_abs_LogPolar_numpy = fft2(input_image_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_image_1_fft_abs_LogPolar_numpy) * abs(input_image_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(
        ifft2((input_image_1_fft_abs_LogPolar_numpy * input_image_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(
            ifft2((input_image_2_fft_abs_LogPolar_numpy * input_image_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
        # phase_cross_correlation[0,0] = 0
        i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
        angle = -180.0 * i0 / phase_cross_correlation.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError('images are not compatible. Scale change > 1.8')

    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    print(angle)
    1

def scipy_registration():
    shift_x = np.array([130])
    shift_y = np.array([-40])
    scale = np.array([0.9])
    rotation_angle = np.array([19])  # [degrees]
    input_image_1 = read_image_default_torch()
    input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)
    input_image_1 = input_image_1[0, 0].numpy()
    input_image_2 = input_image_2[0, 0].numpy()
    input_image_1 = crop_numpy_batch(input_image_1, 1000)
    input_image_2 = crop_numpy_batch(input_image_2, 1000)

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_image_1 = difference_of_gaussians(input_image_1, 5, 20)  # TODO: the more you smear the better?
    input_image_2 = difference_of_gaussians(input_image_2, 5, 20)

    ### Window Images To Avoid Effects From Image Edges: ###
    # (*). Probably not as important!!!!
    input_image_1_windowed = input_image_1 * window('hann', input_image_1.shape)
    input_image_2_windowed = input_image_2 * window('hann', input_image_1.shape)

    ### Work With Shifted FFT Magnitudes: ###
    input_image_1_FFT_abs = np.abs(fftshift(fft2(input_image_1_windowed)))
    input_image_2_FFT_abs = np.abs(fftshift(fft2(input_image_2_windowed)))

    ### Create Log-Polar Transformed FFT Mag Images and Register: ###
    shape = input_image_1_FFT_abs.shape
    radius = shape[0] // 1  # only take lower frequencies
    input_image_1_FFT_abs_LogPolar = warp_polar(input_image_1_FFT_abs, radius=radius, output_shape=shape, scaling='log',
                                                order=0)
    input_image_2_FFT_abs_LogPolar = warp_polar(input_image_2_FFT_abs, radius=radius, output_shape=shape, scaling='log',
                                                order=0)

    # #TODO: delete
    # input_image_1_FFT_abs_torch = torch.Tensor(input_image_1_FFT_abs).unsqueeze(0).unsqueeze(0)
    # input_image_2_FFT_abs_torch = torch.Tensor(input_image_2_FFT_abs).unsqueeze(0).unsqueeze(0)
    # bla_1, log_base, (radius2, theta2), (x,y), flow_grid = logpolar_scipy_torch(input_image_1_FFT_abs_torch)
    # bla_2, log_base, (radius2, theta2), (x,y), flow_grid = logpolar_scipy_torch(input_image_2_FFT_abs_torch)

    ### TODO: this works!!!!!
    # input_image_1_FFT_abs_LogPolar = bla_1.cpu().numpy()[0,0]
    # input_image_2_FFT_abs_LogPolar = bla_2.cpu().numpy()[0,0]
    # imshow(input_image_1_FFT_abs_LogPolar); imshow_torch(bla_1)
    # TODO: perhapse i should also use on half the FFT and use fftshift as in the above example?!??!?!
    input_image_1_FFT_abs_LogPolar = input_image_1_FFT_abs_LogPolar[:shape[0] // 2, :]  # only use half of FFT
    input_image_2_FFT_abs_LogPolar = input_image_2_FFT_abs_LogPolar[:shape[0] // 2, :]
    shifts, error, phasediff = skimage_phase_cross_correlation(input_image_1_FFT_abs_LogPolar,
                                                               input_image_2_FFT_abs_LogPolar,
                                                               upsample_factor=10)

    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)

    ### TODO: add option of finding translation afterwards
    ### TODO: encapsulate this in a function which finds rotation and scaling robustly!!!!
    ### TODO: time this and ask dudi if he can run this on the cpu?....is there a GPU version of OpenCV or skimage?!??!?!

    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # ax = axes.ravel()
    # ax[0].set_title("Original Image FFT\n(magnitude; zoomed)")
    # center = np.array(shape) // 2
    # ax[0].imshow(input_image_1_FFT_abs[center[0] - radius:center[0] + radius,
    #              center[1] - radius:center[1] + radius],
    #              cmap='magma')
    # ax[1].set_title("Modified Image FFT\n(magnitude; zoomed)")
    # ax[1].imshow(input_image_2_FFT_abs[center[0] - radius:center[0] + radius,
    #              center[1] - radius:center[1] + radius],
    #              cmap='magma')
    # ax[2].set_title("Log-Polar-Transformed\nOriginal FFT")
    # ax[2].imshow(input_image_1_FFT_abs_LogPolar, cmap='magma')
    # ax[3].set_title("Log-Polar-Transformed\nModified FFT")
    # ax[3].imshow(input_image_2_FFT_abs_LogPolar, cmap='magma')
    # fig.suptitle('Working in frequency domain can recover rotation and scaling')
    # plt.show()

    print(f"Expected value for cc rotation in degrees: {rotation_angle}")
    print(f"Recovered value for cc rotation: {recovered_angle}")
    print()
    print(f"Expected value for scaling difference: {scale}")
    print(f"Recovered value for scaling difference: {shift_scale}")
############################################################################################################################################################



############################################################################################################################################################
### LogPolar transform based FFT Affine Registration: ###
def torch_2D_hann_window(window_shape):
    hann_tensor = torch.Tensor(window('hann', (window_shape[2],window_shape[3]))).unsqueeze(0).unsqueeze(0)
    return hann_tensor

import torch.nn as nn

class FFT_LogPolar_Rotation_Scaling_Registration_Layer(nn.Module):
    def __init__(self, *args):
        super(FFT_LogPolar_Rotation_Scaling_Registration_Layer, self).__init__()
        self.B = None
        self.C = None
        self.H = None
        self.W = None
        self.X0 = None
        self.Y0 = None
        self.gaussian_blur_layer_low_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=5 ** 2, dim=2).cuda()
        self.gaussian_blur_layer_high_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=20 ** 2, dim=2).cuda()
        self.hann_window = None
        self.radius_div_factor = 8
        self.flow_grid = None
        self.flag_use_hann_window = True

    def forward(self, input_image_1, input_image_2):
        ### TODO: !!!!!!!: ###
        #(1). Play with the gaussian filter parameters of the high and low sigmas and the kernel size etc'
        #(2). play with the LogPolar transform. maybe make it LinearPolar, maybe change the range of angles to a very small one, maybe use something other the nearest neighbor

        ### First, Band-Pass Filter Both Images: ###
        # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
        input_image_1_DoG = self.gaussian_blur_layer_low_sigma(input_image_1) - self.gaussian_blur_layer_high_sigma(input_image_1)
        input_image_2_DoG = self.gaussian_blur_layer_low_sigma(input_image_2) - self.gaussian_blur_layer_high_sigma(input_image_2)

        ### Window Images To Avoid Effects From Image Edges: ###
        if self.flag_use_hann_window:
            if self.hann_window is None:
                self.hann_window = torch_2D_hann_window(input_image_1_DoG.shape).cuda()
            input_image_1_windowed = input_image_1_DoG * self.hann_window
            input_image_2_windowed = input_image_2_DoG * self.hann_window
        else:
            input_image_1_windowed = input_image_1_DoG
            input_image_2_windowed = input_image_2_DoG

        ### Work With Shifted FFT Magnitudes: ###
        input_image_1_window_FFT = torch_fftshift(torch_fft2(input_image_1_windowed))
        input_image_2_window_FFT = torch_fftshift(torch_fft2(input_image_2_windowed))
        input_image_1_FFT_abs_torch = torch.abs(input_image_1_window_FFT)
        input_image_2_FFT_abs_torch = torch.abs(input_image_2_window_FFT)

        ### Create Log-Polar Transformed FFT Mag Images and Register: ###
        shape = (input_image_1_FFT_abs_torch.shape[2], input_image_1_FFT_abs_torch.shape[3])
        radius = shape[0] // self.radius_div_factor  # only take lower frequencies
        if self.flow_grid is None:
            input_image_1_FFT_abs_LogPolar, log_base, (radius2, theta2), (x, y), self.flow_grid = torch_LogPolar_transform_ScipyFormat(input_image_1_FFT_abs_torch, self.radius_div_factor)
            input_image_2_FFT_abs_LogPolar = nn.functional.grid_sample(input_image_2_FFT_abs_torch.cuda(), self.flow_grid, mode='nearest')
        else:
            input_image_1_FFT_abs_LogPolar = nn.functional.grid_sample(input_image_1_FFT_abs_torch.cuda(), self.flow_grid, mode='nearest')
            input_image_2_FFT_abs_LogPolar = nn.functional.grid_sample(input_image_2_FFT_abs_torch.cuda(), self.flow_grid, mode='nearest')

        ### Get Cross Correlation & Shifts: ###
        input_image_1_FFT_abs_LogPolar = input_image_1_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]  # only use half of FFT
        input_image_2_FFT_abs_LogPolar = input_image_2_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]
        cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(input_image_1_FFT_abs_LogPolar,
                                                                                                                  input_image_2_FFT_abs_LogPolar)

        ### Use translation parameters to calculate rotation and scaling parameters: ###
        # shiftr, shiftc = shifts[:2]
        shiftr, shiftc = shifts_sub_pixel[:2]
        recovered_angle = (360 / shape[0]) * shiftr
        klog = shape[1] / np.log(radius)
        recovered_scale = torch.exp(shiftc / klog)
        recovered_scale = (input_image_2.shape[-1] - 1) / (int(input_image_2.shape[-1] / recovered_scale) - 1)

        return recovered_angle, recovered_scale, input_image_1_window_FFT, input_image_2_window_FFT

def get_FFT_LogPolar_Rotation_Scaling_torch(input_image_1=None, input_image_2=None, radius_div_factor=8):
    # ### Prepare Data: ###
    # shift_x = np.array([30])
    # shift_y = np.array([-30])
    # scale = np.array([1.05])
    # rotation_angle = np.array([-10])  # [degrees]
    # radius_div_factor = 8
    # input_image_1 = read_image_default_torch()
    # input_image_1 = RGB2BW(input_image_1)
    # input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)
    # input_image_1 = input_image_1[0, 0].numpy()
    # input_image_2 = input_image_2[0, 0].numpy()
    # input_image_1 = crop_numpy_batch(input_image_1, 1000)
    # input_image_2 = crop_numpy_batch(input_image_2, 1000)

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    #TODO: this doesn't work exactly like difference_of_gaussians from skimage...understand what's the difference?...perhapse kernel_size, perhapse gaussian definition?...
    gaussian_blur_layer_low_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=5**2, dim=2)
    gaussian_blur_layer_high_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=20**2, dim=2)
    input_image_1_DoG = gaussian_blur_layer_low_sigma(input_image_1) - gaussian_blur_layer_high_sigma(input_image_1)
    input_image_2_DoG = gaussian_blur_layer_low_sigma(input_image_2) - gaussian_blur_layer_high_sigma(input_image_2)
    # input_image_1_DoG = difference_of_gaussians(input_image_1, 5, 20)  # TODO: the more you smear the better?
    # input_image_2_DoG = difference_of_gaussians(input_image_2, 5, 20)

    ### Window Images To Avoid Effects From Image Edges: ###
    # (*). Probably not as important!!!!
    input_image_1_windowed = input_image_1_DoG * torch_2D_hann_window(input_image_1_DoG.shape)
    input_image_2_windowed = input_image_2_DoG * torch_2D_hann_window(input_image_1_DoG.shape)

    ### Work With Shifted FFT Magnitudes: ###
    input_image_1_FFT_abs_torch = torch.abs(torch_fftshift(torch_fft2(input_image_1_windowed)))
    input_image_2_FFT_abs_torch = torch.abs(torch_fftshift(torch_fft2(input_image_2_windowed)))

    ### Create Log-Polar Transformed FFT Mag Images and Register: ###
    shape = (input_image_1_FFT_abs_torch.shape[2], input_image_1_FFT_abs_torch.shape[3])
    radius = shape[0] // radius_div_factor  # only take lower frequencies
    input_image_1_FFT_abs_LogPolar, log_base, (radius2, theta2), (x, y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_image_1_FFT_abs_torch, radius_div_factor)
    input_image_2_FFT_abs_LogPolar, log_base, (radius2, theta2), (x, y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_image_2_FFT_abs_torch, radius_div_factor)

    ### Get Cross Correlation & Shifts: ###
    input_image_1_FFT_abs_LogPolar = input_image_1_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]  # only use half of FFT
    input_image_2_FFT_abs_LogPolar = input_image_2_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]
    ### Numpy Version: ###
    cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(input_image_1_FFT_abs_LogPolar,
                                                                                                              input_image_2_FFT_abs_LogPolar)

    ### Use translation parameters to calculate rotation and scaling parameters: ###
    # shiftr, shiftc = shifts[:2]
    shiftr, shiftc = shifts_sub_pixel[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = torch.exp(shiftc / klog)

    shift_scale = (input_image_2.shape[-1] - 1) / (int(input_image_2.shape[-1] / shift_scale) - 1)
    # print(f"Expected value for cc rotation in degrees: {rotation_angle}")
    # print(f"Recovered value for cc rotation: {recovered_angle}")
    # print()
    # print(f"Expected value for scaling difference: {scale}")
    # print(f"Recovered value for scaling difference: {shift_scale}")

    return recovered_angle, shift_scale

def get_FFT_LogPolar_Rotation_Scaling_scipy(input_image_1=None, input_image_2=None):
    # ### Prepare Data: ###
    # shift_x = np.array([30])
    # shift_y = np.array([-30])
    # scale = np.array([1.05])
    # rotation_angle = np.array([-10])  # [degrees]
    # input_image_1 = read_image_default_torch()
    # input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)
    # input_image_1 = input_image_1[0, 0].numpy()
    # input_image_2 = input_image_2[0, 0].numpy()
    # input_image_1 = crop_numpy_batch(input_image_1, 1000)
    # input_image_2 = crop_numpy_batch(input_image_2, 1000)

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_image_1_DoG = difference_of_gaussians(input_image_1, 5, 20)  # TODO: the more you smear the better?
    input_image_2_DoG = difference_of_gaussians(input_image_2, 5, 20)

    ### Window Images To Avoid Effects From Image Edges: ###
    # (*). Probably not as important!!!!
    input_image_1_windowed = input_image_1_DoG * window('hann', input_image_1_DoG.shape)
    input_image_2_windowed = input_image_2_DoG * window('hann', input_image_1_DoG.shape)

    ### Work With Shifted FFT Magnitudes: ###
    input_image_1_FFT_abs = np.abs(fftshift(fft2(input_image_1_windowed)))
    input_image_2_FFT_abs = np.abs(fftshift(fft2(input_image_2_windowed)))

    ### Create Log-Polar Transformed FFT Mag Images and Register: ###
    shape = input_image_1_FFT_abs.shape
    radius_div_factor = 16
    radius = shape[0] // radius_div_factor  # only take lower frequencies
    ### Scipy Version: ###
    input_image_1_FFT_abs_LogPolar = scipy_LogPolar_transform(input_image_1_FFT_abs, radius_div_factor)
    input_image_2_FFT_abs_LogPolar = scipy_LogPolar_transform(input_image_2_FFT_abs, radius_div_factor)
    # ### Torch Version: ###
    # input_image_1_FFT_abs_torch = torch.Tensor(input_image_1_FFT_abs).unsqueeze(0).unsqueeze(0)
    # input_image_2_FFT_abs_torch = torch.Tensor(input_image_2_FFT_abs).unsqueeze(0).unsqueeze(0)
    # input_image_1_FFT_abs_LogPolar_torch, log_base, (radius2, theta2), (x,y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_image_1_FFT_abs_torch, radius_div_factor)
    # input_image_2_FFT_abs_LogPolar_torch, log_base, (radius2, theta2), (x,y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_image_2_FFT_abs_torch, radius_div_factor)
    # input_image_1_FFT_abs_LogPolar = input_image_1_FFT_abs_LogPolar_torch.cpu().numpy()[0,0]
    # input_image_2_FFT_abs_LogPolar = input_image_2_FFT_abs_LogPolar_torch.cpu().numpy()[0,0]

    # TODO: perhapse i should also use on half the FFT and use fftshift as in the above example?!??!?!
    input_image_1_FFT_abs_LogPolar = input_image_1_FFT_abs_LogPolar[:shape[0] // 2, :]  # only use half of FFT
    input_image_2_FFT_abs_LogPolar = input_image_2_FFT_abs_LogPolar[:shape[0] // 2, :]
    ## SKimage Version: ###
    shifts, error, phasediff = skimage_phase_cross_correlation(input_image_1_FFT_abs_LogPolar,
                                                               input_image_2_FFT_abs_LogPolar,
                                                               upsample_factor=10)  #TODO: X10 precision using FFT, probably shift to parabola fit or something, timeit
    # ### Numpy Version: ###
    def numpy_to_torch(input_image, flag_unsqueeze=False):
        # Assumes "Natural" RGB form to there's no BGR->RGB conversion, can also accept BW image!:
        if input_image.ndim == 2:
            # Make Sure That We Get the Image in the correct format [H,W,C]:
            input_image = np.expand_dims(input_image, axis=2)  # [H,W]->[H,W,1]
        if input_image.ndim == 3:
            input_image = np.transpose(input_image, (2, 0, 1))  # [H,W,C]->[C,H,W]
        elif input_image.ndim == 4:
            input_image = np.transpose(input_image, (0, 3, 1, 2))  # [T,H,W,C] -> [T,C,H,W]
        input_image = torch.from_numpy(input_image.astype(np.float)).float()  # to float32

        if flag_unsqueeze:
            input_image = input_image.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        return input_image
    cross_correlation, shifts_discrete, shifts = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(
                                                                                            numpy_to_torch(input_image_1_FFT_abs_LogPolar).unsqueeze(0),
                                                                                            numpy_to_torch(input_image_2_FFT_abs_LogPolar).unsqueeze(0))

    ### Use translation parameters to calculate rotation and scaling parameters: ###
    shiftr, shiftc = shifts[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)

    # print(f"Expected value for cc rotation in degrees: {rotation_angle}")
    # print(f"Recovered value for cc rotation: {recovered_angle}")
    # print()
    # print(f"Expected value for scaling difference: {scale}")
    # print(f"Recovered value for scaling difference: {shift_scale}")

    return recovered_angle, shift_scale

def get_FFT_LogPolar_Rotation_Scaling_Translation_numpy(input_image_1=None, input_image_2=None):
    ### Prepare Data: ###
    #OHHHHHH - very important!!!! in warp_tensors_affine i translate->scale->rotate!!!! this explains why i get the tensors on top of each other but doesn't get the
    # "correct" translation....because it's not "correct"!!!!!.... i still need to think about the delta_r correction factor below
    # warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
    # shift_x = np.array([-5])
    # shift_y = np.array([5])
    # scale = np.array([1.0])
    # rotation_angle = np.array([15.5])  # [degrees]
    # input_image_1 = read_image_default_torch()
    # input_image_2 = warp_tensors_affine_layer.forward(input_image_1, shift_x, shift_y, scale, rotation_angle)
    # input_image_1 = input_image_1[0, 0].numpy()
    # input_image_2 = input_image_2[0, 0].numpy()
    # input_image_1 = crop_numpy_batch(input_image_1, 700)
    # input_image_2 = crop_numpy_batch(input_image_2, 700)

    ### Get Angle & Scale: ###
    (rotation_angle, scale) = get_FFT_LogPolar_Rotation_Scaling_scipy(input_image_1, input_image_2) #this uses PHASE cross correlation which is bad
    # print(rotation_angle)

    # ### Rescale & Rotate Second Image: ###
    # input_image_2_scaled_rotated = ndii.zoom(input_image_2, 1/scale)
    # input_image_2_scaled_rotated = ndii.rotate(input_image_2_scaled_rotated, -rotation_angle)
    # ### Equalize Image Sizes After Scipy Zoom & Rotate Operations (which change image size): ###
    # if input_image_2_scaled_rotated.shape < input_image_1.shape:
    #     t = numpy.zeros_like(input_image_1)
    #     t[: input_image_2_scaled_rotated.shape[0],
    #     : input_image_2_scaled_rotated.shape[1]] = input_image_2_scaled_rotated
    #     input_image_2_scaled_rotated = t
    # elif input_image_2_scaled_rotated.shape > input_image_1.shape:
    #     input_image_2_scaled_rotated = input_image_2_scaled_rotated[: input_image_1.shape[0], : input_image_1.shape[1]]
    # imshow(input_image_2_scaled_rotated); figure(); imshow(input_image_1)
    # imshow(input_image_2_scaled_rotated - input_image_1)


    ### Rescale & Rotate Second Image Torch: ###
    input_image_2_torch = torch.Tensor(input_image_2).unsqueeze(0).unsqueeze(0)
    input_image_2_scaled_rotated = warp_tensors_affine(input_image_2_torch, [0], [0], [1/scale], [-rotation_angle])
    input_image_2_scaled_rotated = input_image_2_scaled_rotated.cpu().numpy()[0,0,:,:]
    # imshow(input_image_2_scaled_rotated); figure(); imshow(input_image_1)
    # imshow(input_image_2_scaled_rotated - input_image_1)

    ### Crop Images Before Translational Cross Correlation: ###
    # input_image_1_original_crop = crop_numpy_batch(input_image_1, np.inf)
    # input_image_2_scaled_rotated_crop = crop_numpy_batch(input_image_2_scaled_rotated, np.inf)
    input_image_1_original_crop = crop_numpy_batch(input_image_1, 800)
    input_image_2_scaled_rotated_crop = crop_numpy_batch(input_image_2_scaled_rotated, 800)
    # imshow(input_image_2_scaled_rotated_crop); figure(); imshow(input_image_1_original_crop)
    # imshow(input_image_2_scaled_rotated_crop - input_image_1_original_crop)

    input_image_1_fft = fft2(input_image_1_original_crop)
    input_image_2_fft = fft2(input_image_2_scaled_rotated_crop)
    # TODO: probably need to center crop images here!
    # phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate()) / (abs(input_image_1_fft) * abs(input_image_2_fft))))
    phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate())))
    # phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate())))
    delta_y, delta_x = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)

    if delta_y > input_image_1_fft.shape[0] // 2:
        delta_y -= input_image_1_fft.shape[0]
    if delta_x > input_image_2_fft.shape[1] // 2:
        delta_x -= input_image_2_fft.shape[1]

    #######################
    #TODO: perhapse i should do something like plot the correction factor as a function of delta_x, delta_y, theta, scale???
    delta_r = np.sqrt(delta_y**2 + delta_x**2)
    delta_theta = math.atan2(-delta_y, -delta_x)
    theta_corrected = rotation_angle - delta_theta * 180/np.pi
    delta_x_correction = delta_r * math.sin(math.radians(theta_corrected)) * 1/scale
    delta_y_correction = delta_r * math.cos(math.radians(theta_corrected)) * 1/scale
    #TODO: besides the above correction and the below correction (which doesn't seem to work....), the should be some correction factor
    #TODO: depending on the angle (again, the below factor doesn't work). perhapse something like math.sin(angle) * delta_x
    #######################

    input_image_2_scaled_rotated_shifted = ndii.shift(input_image_2_scaled_rotated_crop, [delta_y, delta_x])
    # figure(); imshow(input_image_2_scaled_rotated_shifted); figure(); imshow(input_image_1_original_crop)
    # figure(); imshow(input_image_2_scaled_rotated_shifted - input_image_1_original_crop)

    # correct parameters for ndimage's internal processing
    if rotation_angle > 0.0:
        d = int(int(input_image_2.shape[1] / scale) * math.sin(math.radians(rotation_angle)))
        delta_y, delta_x = delta_x, d + delta_y
    elif rotation_angle < 0.0:
        d = int(int(input_image_2.shape[0] / scale) * math.sin(math.radians(rotation_angle)))
        delta_y, delta_x = d + delta_x, d + delta_y
    scale = (input_image_2.shape[1] - 1) / (int(input_image_2.shape[1] / scale) - 1)

    return rotation_angle, scale, (delta_x, delta_y), input_image_2_scaled_rotated_shifted

def get_FFT_LogPolar_Rotation_Scaling_Translation_torch():
    ### Prepare Data: ###
    shift_x = np.array([-30])
    shift_y = np.array([30])
    scale = np.array([1.05])
    rotation_angle = np.array([-10])  # [degrees]
    input_image_1 = read_image_default_torch()
    input_image_1 = RGB2BW(input_image_1)
    input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)
    input_image_1 = crop_torch_batch(input_image_1, 1000)
    input_image_2 = crop_torch_batch(input_image_2, 1000)
    input_image_1 = nn.AvgPool2d(2)(input_image_1)
    input_image_2 = nn.AvgPool2d(2)(input_image_2)

    ### Get Angle & Scale: ###
    (rotation_angle, scale) = get_FFT_LogPolar_Rotation_Scaling_torch(input_image_1, input_image_2)

    ### Rescale & Rotate Second Image Torch: ###
    input_image_2_scaled_rotated = warp_tensors_affine(input_image_2, [0], [0], [1/scale], [-rotation_angle])
    # imshow(input_image_2_scaled_rotated); figure(); imshow(input_image_1)
    # imshow(input_image_2_scaled_rotated - input_image_1)

    ### Crop Images Before Translational Cross Correlation: ###
    # input_image_1_original_crop = crop_numpy_batch(input_image_1, np.inf)
    # input_image_2_scaled_rotated_crop = crop_numpy_batch(input_image_2_scaled_rotated, np.inf)
    input_image_1_original_crop = crop_torch_batch(input_image_1, 800)
    input_image_2_scaled_rotated_crop = crop_torch_batch(input_image_2_scaled_rotated, 800)
    # imshow(input_image_2_scaled_rotated_crop); figure(); imshow(input_image_1_original_crop)
    # imshow(input_image_2_scaled_rotated_crop - input_image_1_original_crop)

    ### Get Cross Correlation & Shift: ###
    cross_correlation_mat, shifts = get_Circular_Cross_Correlation_and_Shifts_DiscreteShift_torch(input_image_1_original_crop, input_image_2_scaled_rotated_crop)

    ### Shift Second Image To Be On Top Of First One: ###
    input_image_2_scaled_rotated_crop_translated = shift_matrix_subpixel_torch(input_image_2_scaled_rotated_crop, shifts[1], shifts[0])
    imshow_torch(input_image_2_scaled_rotated_crop_translated - input_image_1_original_crop)

class FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer(nn.Module):
    def __init__(self, *args):
        super(FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer, self).__init__()
        self.gaussian_blur_layer_low_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=5 ** 2, dim=2)
        self.gaussian_blur_layer_high_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=20 ** 2, dim=2)
        self.hann_window = None
        self.radius_div_factor = 8
        # self.inner_crop_after_rotation_and_scaling = (800, 800)
        self.inner_crop_after_rotation_and_scaling = (np.inf, np.inf)
        self.flow_grid = None
        self.flag_use_hann_window = True
        self.fft_logpolar_rotation_scaling_registration_layer = FFT_LogPolar_Rotation_Scaling_Registration_Layer().cuda()
        self.warp_tensors_affine_layer = Warp_Tensors_Affine_Layer().cuda()  #TODO: change to fft rotation and translation later
        self.warp_tensors_shift_layer = Shift_Layer_Torch().cuda()
        self.downsample_factor = None

    def forward(self, input_image_1, input_image_2, downsample_factor=1, flag_return_shifted_image=True, flag_interpolation_mode='bilinear'):
        ### DownSample Input Images If Wanted: ###
        if self.downsample_factor is None:
            self.downsample_factor = downsample_factor
            self.downsample_layer = nn.AvgPool2d(self.downsample_factor)
        if self.downsample_factor > 1:
            input_image_1_downsampled = self.downsample_layer(input_image_1)
            input_image_2_downsampled = self.downsample_layer(input_image_2)
        else:
            input_image_1_downsampled = input_image_1
            input_image_2_downsampled = input_image_2

        ### Get Angle & Scale: ###
        (recovered_angle, recovered_scale, input_image_window_FFT_1, input_image_window_FFT_2) =\
            self.fft_logpolar_rotation_scaling_registration_layer(input_image_1_downsampled, input_image_2_downsampled)

        ### Rescale & Rotate Second Image Torch: ###
        input_image_2_scaled_rotated = self.warp_tensors_affine_layer.forward(input_image_2,
                                                                              np.float32(0), np.float32(0),
                                                                              np.float32(1 / recovered_scale),
                                                                              -recovered_angle,
                                                                              flag_interpolation_mode=flag_interpolation_mode)

        ### Crop Images Before Translational Cross Correlation: ###
        input_image_1_original_crop = crop_torch_batch(input_image_1, self.inner_crop_after_rotation_and_scaling)
        input_image_2_scaled_rotated_crop = crop_torch_batch(input_image_2_scaled_rotated, self.inner_crop_after_rotation_and_scaling)
        # input_image_1_original_crop = input_image_1
        # input_image_2_scaled_rotated_crop = input_image_2_scaled_rotated

        ### Get Cross Correlation & Shift: ###
        cross_correlation_mat, recovered_translation_discrete, recovered_translation = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(input_image_1_original_crop,
                                                                                                                     input_image_2_scaled_rotated_crop)

        if flag_return_shifted_image:
            input_image_2_displaced = self.warp_tensors_shift_layer.forward(input_image_2_scaled_rotated_crop,
                                                                          recovered_translation[1], recovered_translation[0],
                                                                          fft_image=None)
        else:
            input_image_2_displaced = None

        return recovered_angle, recovered_scale, recovered_translation, input_image_2_displaced

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
            self.padding.append(i//2)
            self.padding.append(i//2)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(F.pad(input,self.padding), weight=self.weight, groups=self.groups)

def Gaussian_Blur_Wrapper_Torch(gaussian_blur_layer, input_tensor, number_of_channels_per_frame, frames_dim=0, channels_dim=1):
    #TODO: this is extremely inefficient!!!! never have for loops (without parallelization) in script!!! take care of this!!!!
    #TODO: simply have the gaussian blur work on all channels together or something. in case of wanting several blurs within the batch we can do that
    ### indices pre processing: ###
    frames_dim_channels = input_tensor.shape[frames_dim]
    channels_dim_channels = input_tensor.shape[channels_dim]
    flag_frames_concatenated_along_channels = (frames_dim == channels_dim)
    if frames_dim == channels_dim:  # frames concatenated along channels dim
        number_of_frames = int(channels_dim_channels / number_of_channels_per_frame)
    else:
        number_of_frames = frames_dim_channels
    output_tensor = torch.zeros_like(input_tensor)

    if len(input_tensor.shape) == 4: #[B,C,H,W]
        B,C,H,W = input_tensor.shape

        if flag_frames_concatenated_along_channels:  #[B,C*T,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[:,i*number_of_channels_per_frame*(i+1)*number_of_channels_per_frame,:,:] = \
                    gaussian_blur_layer(input_tensor[:,i*number_of_channels_per_frame*(i+1)*number_of_channels_per_frame,:,:])
        else: #[T,C,H,W] or [B,C,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[i:i+1,:,:,:] = gaussian_blur_layer(input_tensor[i:i+1,:,:,:])

    elif len(input_tensor.shape) == 5: #[B,T,C,H,W]
        B, T, C, H, W = input_tensor.shape

        for i in np.arange(number_of_frames):
            output_tensor[:,i,:,:,:] = gaussian_blur_layer(input_tensor[:,i,:,:,:])

    return output_tensor

def FFT_alignment_super_parameter(input_dict, original_frame_current):
    crop_size_after_cross_correlation = input_dict['crop_size_after_cross_correlation']
    crop_size_before_cross_correlation = input_dict['crop_size_before_cross_correlation']
    counter = input_dict['counter']
    number_of_frames_to_average = input_dict['number_of_frames_to_average']

    if counter == 0:
        ### Initialize: ###
        input_dict['aligned_frames_list'] = []
        input_dict['original_frames_list'] = []
        input_dict['original_frames_cropped_list'] = []
        input_dict['clean_frames_list'] = []
        input_dict['inferred_shifts_x_list'] = []
        input_dict['inferred_shifts_y_list'] = []

        aligned_frames_list = []
        original_frames_list = []
        original_frames_cropped_list = []
        clean_frames_list = []

        original_frames_list.append(original_frame_current)
        aligned_frames_list.append(crop_torch_batch(original_frame_current, crop_size_before_cross_correlation, crop_style='center'))
        clean_frames_list.append(crop_torch_batch(original_frame_current, crop_size_after_cross_correlation, crop_style='center'))
        original_frames_cropped_list.append(crop_torch_batch(original_frame_current, crop_size_after_cross_correlation, crop_style='center'))

    else:
        ### Get Images To Align Together: ###
        aligned_frames_list = input_dict['aligned_frames_list']
        original_frames_list = input_dict['original_frames_list']
        original_frames_cropped_list = input_dict['original_frames_cropped_list']
        clean_frames_list = input_dict['clean_frames_list']
        number_of_frames_to_average = input_dict['number_of_frames_to_average']

        ### Append Current Frame To Dict: ###
        original_frames_list.append(original_frame_current)

        ### Get Reference & Current Frames: ###
        original_frames_reference = aligned_frames_list[0]
        # original_frames_reference = original_frames_list[0]
        # original_frames_reference = original_frames_list[-2]  #TODO: at the end this is probably the right thing to do and to keep track of total shifts
        original_frame_current = original_frames_list[-1]

        ### Crop Images: ###
        original_frames_reference_cropped = crop_torch_batch(original_frames_reference, crop_size_before_cross_correlation, crop_style='center')
        original_frame_current_cropped = crop_torch_batch(original_frame_current, crop_size_before_cross_correlation, crop_style='center')

        ### Use Cross Correlation To Find Translation Between Images: ###
        initial_scale_rotation_registration_downsample_factor = 1
        affine_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer()
        recovered_angle, recovered_scale, recovered_translation, original_frame_current_cropped_warped = affine_registration_layer.forward(
            original_frames_reference_cropped,
            original_frame_current_cropped,
            downsample_factor=initial_scale_rotation_registration_downsample_factor,
            flag_return_shifted_image=True)


        ### Update Aligned Frames Lists: ###
        aligned_frames_list.append(original_frame_current_cropped_warped)

        ### Crop Again To Avoid InValid Regions: ###
        original_frame_current_cropped = crop_torch_batch(original_frame_current_cropped, crop_size_after_cross_correlation, crop_style='center')
        original_frame_current_cropped_warped = crop_torch_batch(original_frame_current_cropped_warped, crop_size_after_cross_correlation, crop_style='center')

        ### Append Cropped Frames: ###
        original_frames_cropped_list.append(original_frame_current_cropped)

        ### Average Aligned Frames: ###
        clean_frame_current = torch.zeros((1, 1, original_frame_current_cropped_warped[0].shape[-2],
                                           original_frame_current_cropped_warped[0].shape[-1])).cuda()
        for i in np.arange(len(aligned_frames_list)):
            # clean_frame_current += aligned_frames_list[i]
            clean_frame_current += crop_torch_batch(aligned_frames_list[i], crop_size_after_cross_correlation, crop_style='center')
        clean_frame_current = clean_frame_current / len(aligned_frames_list)
        clean_frames_list.append(clean_frame_current)

    ### Update Lists: ###
    if len(aligned_frames_list) > number_of_frames_to_average:
        aligned_frames_list.pop(0)
        original_frames_list.pop(0)
        original_frames_cropped_list.pop(0)
        clean_frames_list.pop(0)

    ### Update Super Dictionary: ###
    input_dict['aligned_frames_list'] = aligned_frames_list
    input_dict['original_frames_list'] = original_frames_list
    input_dict['clean_frames_list'] = clean_frames_list
    input_dict['original_frames_cropped_list'] = original_frames_cropped_list
    input_dict['counter'] += 1
    return input_dict

class FFT_registration_layer(nn.Module):
    def __init__(self):
        super(FFT_registration_layer, self).__init__()
        self.input_dict = EasyDict()
        self.input_dict.number_of_frames_to_average = 5
        self.input_dict.crop_size_before_cross_correlation = (2048, 934)
        self.input_dict.crop_size_after_cross_correlation = (2048, 934)
        self.input_dict.counter = 0
    def forward(self, input_tensor):
        with torch.no_grad():
            self.input_dict, clean_frame_tensor = FFT_alignment_super_parameter(self.input_dict, input_tensor)
            return clean_frame_tensor




# ### Prepare Data: ###
# warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
# shift_x = np.array([0])
# shift_y = np.array([0])
# scale = np.array([1.00])
# rotation_angle = np.array([-2])  # [degrees]
# input_image_1 = read_image_default_torch().cuda()
# input_image_1 = RGB2BW(input_image_1)
# input_image_2 = warp_tensors_affine_layer.forward(input_image_1, shift_x, shift_y, scale, rotation_angle)
# input_image_1 = crop_torch_batch(input_image_1, 700)
# input_image_2 = crop_torch_batch(input_image_2, 700)
# input_image_1 = nn.AvgPool2d(1)(input_image_1)
# input_image_2 = nn.AvgPool2d(1)(input_image_2)
# input_image_1_numpy = input_image_1.cpu().numpy()[0,0]
# input_image_2_numpy = input_image_2.cpu().numpy()[0,0]
#
# ### Using Numpy Version - works but still not good for rotations<2, the more rotation the more accurate a: ###
# rotation, scale, translation, input_image_2_scaled_rotated_shifted = get_FFT_LogPolar_Rotation_Scaling_Translation_numpy(input_image_1_numpy, input_image_2_numpy)
# print(rotation)
#
# ### Using pytorch Version - doesn't work the same as numpy version!!!!!! fix it!!!!: ###
# fft_logpolar_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer().cuda()
# recovered_angle, recovered_scale, recovered_translation, input_image_1_displaced = fft_logpolar_registration_layer.forward(input_image_1,
#                                                                                                                      input_image_2,
#                                                                                                                      downsample_factor=4)
# print(recovered_translation)
# print(recovered_scale)
# print(recovered_angle)
# print(recovered_angle*input_image_1.shape[-1]/360*np.pi)
#
# ### Using Min-SAD: ###
#

################################################################################################################################################




############################################################################################################################################################
### Min-SAD Based Registration: ###

def get_scale_rotation_translation_minSAD_affine(input_image_1, input_image_2):
    ############################################################################################################################################################
    ### Sizes: ###
    H_initial = 500  # initial crop size
    W_initial = 500  # initial crop size
    H_final = 500
    W_final = 500
    initial_index_H_final = np.int((H_initial - H_final) / 2)
    initial_index_W_final = np.int((W_initial - W_final) / 2)
    final_index_H_final = initial_index_H_final + H_final
    final_index_W_final = initial_index_W_final + W_final

    ### Get Image: ###
    previous_image = read_image_default_torch()
    previous_image = RGB2BW(previous_image)
    previous_image = previous_image[:,:,0:H_initial,0:W_initial].cuda()
    # previous_image = crop_torch_batch(previous_image, (H_initial, W_initial)).cuda()

    ### GT parameters: ###
    GT_shift_x = np.float32(0.4)
    GT_shift_y = np.float32(-0.5)
    GT_rotation_angle = np.float32(10)
    GT_scale = np.float32(1.04)

    ### Initialize Warp Objects: ###
    affine_layer_torch = Warp_Tensors_Affine_Layer()

    ### Warp Image According To Above Parameters: ###
    current_image = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bilinear')
    # current_image_2 = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bicubic')

    ### Final Crop: ###
    # current_image = crop_torch_batch(current_image, (H_final, W_final))
    # previous_image = crop_torch_batch(previous_image, (H_final, W_final))
    # current_image = current_image[:, :, 0:H_final, 0:W_final]
    # previous_image = previous_image[:, :, 0:H_final, 0:W_final]
    input_image_1 = previous_image
    input_image_2 = current_image
    ############################################################################################################################################################

    ############################################################################################################################################################
    ### Parameters Space: ###
    shifts_vec = [-1, 0, 1]
    rotation_angle_vec = [-1, 0, 1]
    scale_factor_vec = [0.95, 1, 1.05]
    # shifts_vec = my_linspace(-3, 3, 5)
    # rotation_angle_vec = [-4, -2, 0, 2, 4]
    # scale_factor_vec = [0.95, 1, 1.05]
    final_crop_size = (800,800)

    ### Initialize Warp Objects: ###
    affine_layer_torch = Warp_Tensors_Affine_Layer()

    ### Pre-Allocate Flow Grid: ###
    L_shifts = len(shifts_vec)
    L_rotation_angle = len(rotation_angle_vec)
    L_scale = len(scale_factor_vec)
    number_of_possible_warps = L_shifts * L_shifts * L_rotation_angle * L_scale
    B, C, H, W = input_image_1.shape
    Flow_Grids = torch.zeros(number_of_possible_warps, H, W, 2).cuda()

    crop_H_start = (H-final_crop_size[0]) // 2
    crop_H_final = crop_H_start + final_crop_size[0]
    crop_W_start = (W - final_crop_size[1]) // 2
    crop_W_final = crop_W_start + final_crop_size[1]

    ### Loop over all Possible Parameters: ###
    SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec), len(scale_factor_vec))).to(input_image_2.device)
    SAD_matrix_numpy = SAD_matrix.cpu().numpy()

    ### Build All Possible Grids: ## #TODO: accelerate this!!!!
    counter = 0
    for shift_x_counter, current_shift_x in enumerate(shifts_vec):
        for shift_y_counter, current_shift_y in enumerate(shifts_vec):
            for rotation_counter, current_rotation_angle in enumerate(rotation_angle_vec):
                for scale_counter, current_scale_factor in enumerate(scale_factor_vec):
                    ### Warp Previous Image Many Ways To Find Best Fit: ###
                    current_shift_x = np.float32(current_shift_x)
                    current_shift_y = np.float32(current_shift_y)
                    current_scale_factor = np.float32(current_scale_factor)
                    current_rotation_angle = np.float32(current_rotation_angle)
                    input_image_2_warped, current_flow_grid = affine_layer_torch.forward(input_image_2,
                                                                                          current_shift_x,
                                                                                          current_shift_y,
                                                                                          current_scale_factor,
                                                                                          current_rotation_angle,
                                                                                          return_flow_grid=True,
                                                                                          flag_interpolation_mode='bilinear')
                    Flow_Grids[counter] = current_flow_grid
                    counter = counter + 1


    ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
    possible_warps_tensor = input_image_1.repeat((number_of_possible_warps, 1, 1, 1))

    ### Get All Possible Warps Of Previous Image: ###
    possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, Flow_Grids, mode='bilinear')
    # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

    ### Get Min SAD: ###
    # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
    SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_2)[:,:, :, :].mean(-1, True).mean(-2, True)
    min_index_2 = torch.argmin(SAD_matrix_2)
    min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
    min_indices = np.unravel_index(min_index_2, SAD_matrix_numpy.shape)
    shift_x_inference = shifts_vec[min_indices[0]]
    shift_y_inference = shifts_vec[min_indices[1]]
    rotation_angle_inference = rotation_angle_vec[min_indices[2]]
    scale_factor_inference = scale_factor_vec[min_indices[3]]

    ### Correct Sub-Pixel: ###
    #(1). Get Min Indices:
    SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (L_shifts, L_shifts, L_rotation_angle, L_scale))
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    shift_x_index = min_indices[0]
    shift_y_index = min_indices[1]
    rotation_index = min_indices[2]
    scale_index = min_indices[3]
    #(2). Get Indices For Parabola Fitting:
    shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
    shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
    rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
    scale_indices = np.array([scale_index - 1, scale_index, scale_index + 1])
    shift_x_indices[shift_x_indices < 0] += L_shifts
    shift_x_indices[shift_x_indices >= L_shifts] -= L_shifts
    shift_y_indices[shift_y_indices < 0] += L_shifts
    shift_y_indices[shift_y_indices >= L_shifts] -= L_shifts
    rotation_indices[rotation_indices < 0] += L_rotation_angle
    rotation_indices[rotation_indices >= L_rotation_angle] -= L_rotation_angle
    scale_indices[scale_indices < 0] += L_scale
    scale_indices[scale_indices >= L_scale] -= L_scale

    #### Fot Shift X Parabola: ###
    fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
    delta_shift_x = -b_x / (2 * a_x)
    shift_x_sub_pixel = shifts_vec[shift_x_index] + delta_shift_x * (shifts_vec[2] - shifts_vec[1])

    ### Fit Shit Y Parabola: ###
    fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
    delta_shift_y = -b_x / (2 * a_x)
    shift_y_sub_pixel = shifts_vec[shift_y_index] + delta_shift_y * (shifts_vec[2] - shifts_vec[1])

    ### Fit Rotation Parabola: ###
    fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
    delta_rotation = -b_x / (2 * a_x)
    rotation_sub_pixel = rotation_angle_vec[rotation_index] + delta_rotation * (rotation_angle_vec[2] - rotation_angle_vec[1])

    ### Fit Scale Parabola: ###
    fitting_points_scale = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_index, scale_indices]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_scale)
    delta_scale = -b_x / (2 * a_x)
    scale_sub_pixel = scale_factor_vec[scale_index] + delta_scale * (scale_factor_vec[2] - scale_factor_vec[1])

    print('recovered shift x: ' + str(shift_x_sub_pixel))
    print('recovered shift y: ' + str(shift_y_sub_pixel))
    print('recovered rotation: ' + str(rotation_sub_pixel))
    print('recovered scale: ' + str(scale_sub_pixel))

class minSAD_Bilinear_Affine_Registration_Layer(nn.Module):
    def __init__(self, *args):
        super(minSAD_Bilinear_Affine_Registration_Layer, self).__init__()

        ### Parameters Space: ###
        self.shifts_vec = [-1, 0, 1]
        self.rotation_angle_vec = [-1, 0, 1]
        self.scale_factor_vec = [1 - 1e-6, 1, 1 + 1e-6]

        ### Initialize Warp Objects: ###
        self.affine_layer_torch = Warp_Tensors_Affine_Layer()

        ### Pre-Allocate Flow Grid: ###
        self.L_shifts = len(self.shifts_vec)
        self.L_rotation_angle = len(self.rotation_angle_vec)
        self.L_scale = len(self.scale_factor_vec)
        self.number_of_possible_warps = self.L_shifts * self.L_shifts * self.L_rotation_angle * self.L_scale

        self.B = None
        self.C = None
        self.H = None
        self.W = None

    def forward(self, input_image_1, input_image_2):
        ############################################################################################################################################################

        # shifts_vec = my_linspace(-3, 3, 5)
        # rotation_angle_vec = [-4, -2, 0, 2, 4]
        # scale_factor_vec = [0.95, 1, 1.05]
        final_crop_size = (800, 800)

        if self.B is None:
            ### Initialize: ###
            self.B, self.C, self.H, self.W = input_image_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec),len(self.scale_factor_vec))).to(input_image_2.device)
            SAD_matrix_numpy = self.SAD_matrix.cpu().numpy()


        crop_H_start = (self.H - final_crop_size[0]) // 2
        crop_H_final = crop_H_start + final_crop_size[0]
        crop_W_start = (self.W - final_crop_size[1]) // 2
        crop_W_final = crop_W_start + final_crop_size[1]

        ### Build All Possible Grids: ## #TODO: accelerate this!!!!
        counter = 0
        for shift_x_counter, current_shift_x in enumerate(self.shifts_vec):
            for shift_y_counter, current_shift_y in enumerate(self.shifts_vec):
                for rotation_counter, current_rotation_angle in enumerate(self.rotation_angle_vec):
                    for scale_counter, current_scale_factor in enumerate(self.scale_factor_vec):
                        ### Warp Previous Image Many Ways To Find Best Fit: ###
                        current_shift_x = np.float32(current_shift_x)
                        current_shift_y = np.float32(current_shift_y)
                        current_scale_factor = np.float32(current_scale_factor)
                        current_rotation_angle = np.float32(current_rotation_angle)
                        input_image_2_warped, current_flow_grid = self.affine_layer_torch.forward(input_image_2,
                                                                                             current_shift_x,
                                                                                             current_shift_y,
                                                                                             current_scale_factor,
                                                                                             current_rotation_angle,
                                                                                             return_flow_grid=True,
                                                                                             flag_interpolation_mode='bilinear')
                        self.Flow_Grids[counter] = current_flow_grid
                        counter = counter + 1

        ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
        possible_warps_tensor = input_image_1.repeat((self.number_of_possible_warps, 1, 1, 1))

        ### Get All Possible Warps Of Previous Image: ###
        possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, self.Flow_Grids, mode='bilinear')
        # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

        ### Get Min SAD: ###
        # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
        SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_2)[:, :, :, :].mean(-1, True).mean(-2, True)
        min_index_2 = torch.argmin(SAD_matrix_2)
        min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
        min_indices = np.unravel_index(min_index_2, SAD_matrix_numpy.shape)
        shift_x_inference = self.shifts_vec[min_indices[0]]
        shift_y_inference = self.shifts_vec[min_indices[1]]
        rotation_angle_inference = self.rotation_angle_vec[min_indices[2]]
        scale_factor_inference = self.scale_factor_vec[min_indices[3]]

        ### Correct Sub-Pixel: ###
        # (1). Get Min Indices:
        SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (self.L_shifts, self.L_shifts, self.L_rotation_angle, self.L_scale))
        x_vec = [-1, 0, 1]
        y_vec = [-1, 0, 1]
        shift_x_index = min_indices[0]
        shift_y_index = min_indices[1]
        rotation_index = min_indices[2]
        scale_index = min_indices[3]
        # (2). Get Indices For Parabola Fitting:   #TODO: make sure this is correct!!!!!! use the cross correlation parabola fit corrections!!!!
        shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
        shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
        rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
        scale_indices = np.array([scale_index - 1, scale_index, scale_index + 1])
        shift_x_indices[shift_x_indices < 0] += self.L_shifts
        shift_x_indices[shift_x_indices >= self.L_shifts] -= self.L_shifts
        shift_y_indices[shift_y_indices < 0] += self.L_shifts
        shift_y_indices[shift_y_indices >= self.L_shifts] -= self.L_shifts
        rotation_indices[rotation_indices < 0] += self.L_rotation_angle
        rotation_indices[rotation_indices >= self.L_rotation_angle] -= self.L_rotation_angle
        scale_indices[scale_indices < 0] += self.L_scale
        scale_indices[scale_indices >= self.L_scale] -= self.L_scale

        #### Fot Shift X Parabola: ###
        fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index, scale_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
        delta_shift_x = -b_x / (2 * a_x)
        shift_x_sub_pixel = self.shifts_vec[shift_x_index] + delta_shift_x * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Shit Y Parabola: ###
        fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index, scale_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
        delta_shift_y = -b_x / (2 * a_x)
        shift_y_sub_pixel = self.shifts_vec[shift_y_index] + delta_shift_y * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Rotation Parabola: ###
        fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices, scale_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
        delta_rotation = -b_x / (2 * a_x)
        rotation_sub_pixel = self.rotation_angle_vec[rotation_index] + delta_rotation * (self.rotation_angle_vec[2] - self.rotation_angle_vec[1])

        ### Fit Scale Parabola: ###
        fitting_points_scale = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_index, scale_indices]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_scale)
        delta_scale = -b_x / (2 * a_x)
        scale_sub_pixel = self.scale_factor_vec[scale_index] + delta_scale * (self.scale_factor_vec[2] - self.scale_factor_vec[1])

        return shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, scale_sub_pixel


class minSAD_FFT_Affine_Registration_Layer(nn.Module):
    def __init__(self, *args):
        super(minSAD_FFT_Affine_Registration_Layer, self).__init__()

        ### Parameters Space: ###
        # self.shifts_vec = [-1, 0, 1]
        # self.rotation_angle_vec = [-1, 0, 1]
        self.shifts_vec = np.arange(-1,2)
        self.rotation_angle_vec = np.arange(-1,2)
        self.rotation_angle_vec = my_linspace(-1,1+2/10,10)

        ### Initialize Warp Objects: ###
        # self.affine_layer_torch = Warp_Tensors_Affine_Layer()
        self.affine_layer_torch = FFT_Translation_Rotation_Layer()

        ### Pre-Allocate Flow Grid: ###
        self.L_shifts = len(self.shifts_vec)
        self.L_rotation_angle = len(self.rotation_angle_vec)
        self.number_of_possible_warps = self.L_shifts * self.L_shifts * self.L_rotation_angle

        self.B = None
        self.C = None
        self.H = None
        self.W = None

    def forward(self, input_image_1, input_image_2):
        ############################################################################################################################################################

        # shifts_vec = my_linspace(-3, 3, 5)
        # rotation_angle_vec = [-4, -2, 0, 2, 4]
        # scale_factor_vec = [0.95, 1, 1.05]
        final_crop_size = (800, 800)

        if self.B is None:
            ### Initialize: ###
            self.B, self.C, self.H, self.W = input_image_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec))).to(input_image_2.device)
            SAD_matrix_numpy = self.SAD_matrix.cpu().numpy()


        crop_H_start = (self.H - final_crop_size[0]) // 2
        crop_H_final = crop_H_start + final_crop_size[0]
        crop_W_start = (self.W - final_crop_size[1]) // 2
        crop_W_final = crop_W_start + final_crop_size[1]

        ### Build All Possible Grids: ##
        # TODO: accelerate this!!!! parallelize
        # TODO: instead of saving the entire images to gpu memory - at each iteration calculate the SAD!!!!
        counter = 0
        image2_warped_list = []
        for shift_x_counter, current_shift_x in enumerate(self.shifts_vec):
            for shift_y_counter, current_shift_y in enumerate(self.shifts_vec):
                for rotation_counter, current_rotation_angle in enumerate(self.rotation_angle_vec):
                    ### Warp Previous Image Many Ways To Find Best Fit: ###
                    current_shift_x = np.float32(current_shift_x)
                    current_shift_y = np.float32(current_shift_y)
                    current_rotation_angle = np.float32(current_rotation_angle)

                    current_shift_x_tensor = torch.Tensor([current_shift_x])
                    current_shift_y_tensor = torch.Tensor([current_shift_y])
                    current_rotation_angle_tensor = torch.Tensor([current_rotation_angle]) * np.pi/180

                    ### Perform Affine Transform In FFT Space: ###
                    input_image_2_warped = self.affine_layer_torch.forward(input_image_2,
                                                                             current_shift_x_tensor,
                                                                             current_shift_y_tensor,
                                                                             current_rotation_angle_tensor)
                    image2_warped_list.append(input_image_2_warped)
                    counter = counter + 1

        ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
        input_image_1_stacked = input_image_1.repeat((self.number_of_possible_warps, 1, 1, 1))
        image2_different_warps_tensor = torch.cat(image2_warped_list, 0)
        input_image_1_stacked = crop_torch_batch(input_image_1_stacked, (int(self.W*0.9), int(self.H*0.9))).cuda()
        image2_different_warps_tensor = crop_torch_batch(image2_different_warps_tensor, (int(self.W*0.9), int(self.H*0.9))).cuda()

        ### Get Min SAD: ###
        # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
        differences_mat = torch.abs(input_image_1_stacked - image2_different_warps_tensor)
        SAD_matrix_2 = differences_mat.mean(-1, True).mean(-2, True)
        min_index_2 = torch.argmin(SAD_matrix_2)
        min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
        min_indices = np.unravel_index(min_index_2, (len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec)))
        shift_x_inference = self.shifts_vec[min_indices[0]]
        shift_y_inference = self.shifts_vec[min_indices[1]]
        rotation_angle_inference = self.rotation_angle_vec[min_indices[2]]

        ### Correct Sub-Pixel: ###
        # (1). Get Min Indices:
        SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (self.L_shifts, self.L_shifts, self.L_rotation_angle))
        x_vec = [-1, 0, 1]
        y_vec = [-1, 0, 1]
        shift_x_index = min_indices[0]
        shift_y_index = min_indices[1]
        rotation_index = min_indices[2]
        # (2). Get Indices For Parabola Fitting:   #TODO: make sure this is correct!!!!!! use the cross correlation parabola fit corrections!!!!
        shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
        shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
        rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
        shift_x_indices[shift_x_indices < 0] += self.L_shifts
        shift_x_indices[shift_x_indices >= self.L_shifts] -= self.L_shifts
        shift_y_indices[shift_y_indices < 0] += self.L_shifts
        shift_y_indices[shift_y_indices >= self.L_shifts] -= self.L_shifts
        rotation_indices[rotation_indices < 0] += self.L_rotation_angle
        rotation_indices[rotation_indices >= self.L_rotation_angle] -= self.L_rotation_angle

        #### Fot Shift X Parabola: ###
        fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
        delta_shift_x = -b_x / (2 * a_x)
        shift_x_sub_pixel = self.shifts_vec[shift_x_index] + delta_shift_x * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Shit Y Parabola: ###
        fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
        delta_shift_y = -b_x / (2 * a_x)
        shift_y_sub_pixel = self.shifts_vec[shift_y_index] + delta_shift_y * (self.shifts_vec[2] - self.shifts_vec[1])

        ### Fit Rotation Parabola: ###
        fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
        delta_rotation = -b_x / (2 * a_x)
        rotation_sub_pixel = self.rotation_angle_vec[rotation_index] + delta_rotation * (self.rotation_angle_vec[2] - self.rotation_angle_vec[1])
        rotation_sub_pixel = rotation_sub_pixel   # for radian: * np.pi/180

        ### Transform Second Image To Align With First: ###
        input_image_2_aligned = self.affine_layer_torch.forward(input_image_2,
                                                               -shift_x_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               -shift_y_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               -rotation_sub_pixel.unsqueeze(0).unsqueeze(0))
        input_image_2_aligned = crop_torch_batch(input_image_2_aligned, (self.W, self.H)).cuda()

        return shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, input_image_2_aligned


class minSAD_FFT_Affine_Registration_MemorySave_Layer(nn.Module):
    def __init__(self, *args):
        super(minSAD_FFT_Affine_Registration_MemorySave_Layer, self).__init__()

        ### Parameters Space: ###
        # self.shifts_vec = [-1, 0, 1]
        # self.rotation_angle_vec = [-1, 0, 1]
        #TODO: allow to insert this as input into the forward
        self.shifts_vec = my_linspace(-1,2,5)
        # self.rotation_angle_vec = np.arange(-1,2)
        self.rotation_angle_vec = my_linspace(-2,2+2/20,10)

        ### Initialize Warp Objects: ###
        # self.affine_layer_torch = Warp_Tensors_Affine_Layer()
        self.affine_layer_torch = FFT_Translation_Rotation_Layer()

        ### Pre-Allocate Flow Grid: ###
        self.L_shifts = len(self.shifts_vec)
        self.L_rotation_angle = len(self.rotation_angle_vec)
        self.number_of_possible_warps = self.L_shifts * self.L_shifts * self.L_rotation_angle

        self.B = None
        self.C = None
        self.H = None
        self.W = None

    def forward(self, input_image_1, input_image_2, shifts_vec=None, rotation_angle_vec=None):
        ############################################################################################################################################################

        # shifts_vec = my_linspace(-3, 3, 5)
        # rotation_angle_vec = [-4, -2, 0, 2, 4]
        # scale_factor_vec = [0.95, 1, 1.05]


        ### Take care of internal variables according to input search grid: ###
        if shifts_vec is None:
            shifts_vec = self.shifts_vec
        if rotation_angle_vec is None:
            rotation_angle_vec = self.rotation_angle_vec
        self.L_shifts = len(shifts_vec)
        self.L_rotation_angle = len(rotation_angle_vec)
        self.number_of_possible_warps = self.L_shifts * self.L_shifts * self.L_rotation_angle

        if self.B is None:
            ### Initialize: ###
            self.B, self.C, self.H, self.W = input_image_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec))).to(input_image_2.device)
            SAD_matrix_numpy = self.SAD_matrix.cpu().numpy()

        ### Build All Possible Grids: ##
        # TODO: accelerate this!!!! parallelize
        # TODO: instead of saving the entire images to gpu memory - at each iteration calculate the SAD!!!!
        counter = 0
        image2_warped_list = []
        SAD_list = []
        for shift_x_counter, current_shift_x in enumerate(shifts_vec):
            for shift_y_counter, current_shift_y in enumerate(shifts_vec):
                for rotation_counter, current_rotation_angle in enumerate(rotation_angle_vec):
                    ### Warp Previous Image Many Ways To Find Best Fit: ###
                    current_shift_x = np.float32(current_shift_x)
                    current_shift_y = np.float32(current_shift_y)
                    current_rotation_angle = np.float32(current_rotation_angle)

                    current_shift_x_tensor = torch.Tensor([current_shift_x])
                    current_shift_y_tensor = torch.Tensor([current_shift_y])
                    current_rotation_angle_tensor = torch.Tensor([current_rotation_angle]) * np.pi/180

                    ### Perform Affine Transform In FFT Space: ###
                    input_image_2_warped = self.affine_layer_torch.forward(input_image_2,
                                                                             current_shift_x_tensor,
                                                                             current_shift_y_tensor,
                                                                             current_rotation_angle_tensor)


                    ### Center Crop Images To Avoid Frame Artifacts: ###
                    diagonal = np.sqrt(self.H**2 + self.W**2)
                    max_translation = np.max(np.abs(shifts_vec))
                    max_rotation_angle = np.max(np.abs(rotation_angle_vec))
                    max_displacement = max_translation + diagonal * np.tan(max_rotation_angle*np.pi/180) + 10 #+10 for backup
                    input_image_1_cropped = crop_torch_batch(input_image_1, (int(self.H - max_displacement), int(self.W - max_displacement))).cuda()
                    input_image_2_warped_cropped = crop_torch_batch(input_image_2_warped, (int(self.H - max_displacement), int(self.W - max_displacement))).cuda()

                    ### Calculate Outliers To Avoid: ###
                    current_SAD = (input_image_2_warped_cropped - input_image_1_cropped).abs()
                    current_SAD_mean = current_SAD.mean(-1, True).mean(-2, True)

                    # ### Calculate SAD: ###
                    # quantile_above_which_to_ignore = 0.999
                    # current_SAD_STD = current_SAD.std(-1).std(-1)
                    # current_SAD_quantile = current_SAD.quantile(quantile_above_which_to_ignore)
                    # # current_SAD_outlier_logical_mask = current_SAD > current_SAD_STD * 3
                    # current_SAD_outlier_logical_mask = current_SAD > current_SAD_quantile
                    # current_SAD_no_outliers = current_SAD[current_SAD_outlier_logical_mask]
                    # current_SAD_no_outliers_mean = current_SAD_no_outliers.mean()
                    # current_SAD_mean = current_SAD_no_outliers_mean

                    ### Add SAD to list: ###
                    SAD_list.append(current_SAD_mean)
                    counter = counter + 1

        ### Get Min SAD: ###
        SAD_matrix_2 = torch.cat(SAD_list, 0)
        min_index_2 = torch.argmin(SAD_matrix_2)
        min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
        min_indices = np.unravel_index(min_index_2, (len(shifts_vec), len(shifts_vec), len(rotation_angle_vec)))
        shift_x_inference = shifts_vec[min_indices[0]]
        shift_y_inference = shifts_vec[min_indices[1]]
        rotation_angle_inference = rotation_angle_vec[min_indices[2]]

        ### Correct Sub-Pixel: ###
        # (1). Get Min Indices:
        SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (self.L_shifts, self.L_shifts, self.L_rotation_angle))
        x_vec = [-1, 0, 1]
        y_vec = [-1, 0, 1]
        shift_x_index = min_indices[0]
        shift_y_index = min_indices[1]
        rotation_index = min_indices[2]
        # (2). Get Indices For Parabola Fitting:   #TODO: make sure this is correct!!!!!! use the cross correlation parabola fit corrections!!!!
        shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
        shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
        rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
        shift_x_indices[shift_x_indices < 0] += self.L_shifts
        shift_x_indices[shift_x_indices >= self.L_shifts] -= self.L_shifts
        shift_y_indices[shift_y_indices < 0] += self.L_shifts
        shift_y_indices[shift_y_indices >= self.L_shifts] -= self.L_shifts
        rotation_indices[rotation_indices < 0] += self.L_rotation_angle
        rotation_indices[rotation_indices >= self.L_rotation_angle] -= self.L_rotation_angle

        #### Fot Shift X Parabola: ###
        fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
        delta_shift_x = -b_x / (2 * a_x)
        shift_x_sub_pixel = shifts_vec[shift_x_index] + delta_shift_x * (shifts_vec[2] - shifts_vec[1])

        ### Fit Shit Y Parabola: ###
        fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
        delta_shift_y = -b_x / (2 * a_x)
        shift_y_sub_pixel = shifts_vec[shift_y_index] + delta_shift_y * (shifts_vec[2] - shifts_vec[1])

        ### Fit Rotation Parabola: ###
        fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices]
        y_vec = [-1, 0, 1]
        [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
        delta_rotation = -b_x / (2 * a_x)
        rotation_sub_pixel = rotation_angle_vec[rotation_index] + delta_rotation * (rotation_angle_vec[2] - rotation_angle_vec[1])
        rotation_sub_pixel = rotation_sub_pixel   # for radian: * np.pi/180

        ### Transform Second Image To Align With First: ###
        input_image_2_aligned = self.affine_layer_torch.forward(input_image_2,
                                                               -shift_x_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               -shift_y_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               rotation_sub_pixel.unsqueeze(0).unsqueeze(0) * np.pi/180)
        input_image_2_aligned = crop_torch_batch(input_image_2_aligned, (self.H, self.W)).cuda()

        return shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, input_image_2_aligned


def get_scale_rotation_translation_minSAD_affine(input_image_1, input_image_2):
    ############################################################################################################################################################
    ### Sizes: ###
    H_initial = 500  # initial crop size
    W_initial = 500  # initial crop size
    H_final = 500
    W_final = 500
    initial_index_H_final = np.int((H_initial - H_final) / 2)
    initial_index_W_final = np.int((W_initial - W_final) / 2)
    final_index_H_final = initial_index_H_final + H_final
    final_index_W_final = initial_index_W_final + W_final

    ### Get Image: ###
    previous_image = read_image_default_torch()
    previous_image = RGB2BW(previous_image)
    previous_image = previous_image[:,:,0:H_initial,0:W_initial].cuda()
    # previous_image = crop_torch_batch(previous_image, (H_initial, W_initial)).cuda()

    ### GT parameters: ###
    GT_shift_x = np.float32(0.2)
    GT_shift_y = np.float32(-0.4)
    GT_rotation_angle = np.float32(0.1)
    GT_scale = np.float32(1.00)

    ### Initialize Warp Objects: ###
    affine_layer_torch = Warp_Tensors_Affine_Layer()

    ### Warp Image According To Above Parameters: ###
    current_image = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bilinear')
    # current_image_2 = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bicubic')

    ### Final Crop: ###
    # current_image = crop_torch_batch(current_image, (H_final, W_final))
    # previous_image = crop_torch_batch(previous_image, (H_final, W_final))
    # current_image = current_image[:, :, 0:H_final, 0:W_final]
    # previous_image = previous_image[:, :, 0:H_final, 0:W_final]
    input_image_1 = previous_image
    input_image_2 = current_image
    ############################################################################################################################################################

    ############################################################################################################################################################
    ### Parameters Space: ###
    shifts_vec = [-1, 0, 1]
    rotation_angle_vec = [-1, 0, 1]
    scale_factor_vec = [1-1e-6, 1, 1+1e-6]
    # shifts_vec = my_linspace(-3, 3, 5)
    # rotation_angle_vec = [-4, -2, 0, 2, 4]
    # scale_factor_vec = [0.95, 1, 1.05]
    final_crop_size = (800,800)

    ### Initialize Warp Objects: ###
    affine_layer_torch = Warp_Tensors_Affine_Layer()

    ### Pre-Allocate Flow Grid: ###
    L_shifts = len(shifts_vec)
    L_rotation_angle = len(rotation_angle_vec)
    L_scale = len(scale_factor_vec)
    number_of_possible_warps = L_shifts * L_shifts * L_rotation_angle * L_scale
    B, C, H, W = input_image_1.shape
    Flow_Grids = torch.zeros(number_of_possible_warps, H, W, 2).cuda()

    crop_H_start = (H-final_crop_size[0]) // 2
    crop_H_final = crop_H_start + final_crop_size[0]
    crop_W_start = (W - final_crop_size[1]) // 2
    crop_W_final = crop_W_start + final_crop_size[1]

    ### Loop over all Possible Parameters: ###
    SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec), len(scale_factor_vec))).to(input_image_2.device)
    SAD_matrix_numpy = SAD_matrix.cpu().numpy()

    ### Build All Possible Grids: ## #TODO: accelerate this!!!!
    counter = 0
    for shift_x_counter, current_shift_x in enumerate(shifts_vec):
        for shift_y_counter, current_shift_y in enumerate(shifts_vec):
            for rotation_counter, current_rotation_angle in enumerate(rotation_angle_vec):
                for scale_counter, current_scale_factor in enumerate(scale_factor_vec):
                    ### Warp Previous Image Many Ways To Find Best Fit: ###
                    current_shift_x = np.float32(current_shift_x)
                    current_shift_y = np.float32(current_shift_y)
                    current_scale_factor = np.float32(current_scale_factor)
                    current_rotation_angle = np.float32(current_rotation_angle)
                    input_image_2_warped, current_flow_grid = affine_layer_torch.forward(input_image_2,
                                                                                          current_shift_x,
                                                                                          current_shift_y,
                                                                                          current_scale_factor,
                                                                                          current_rotation_angle,
                                                                                          return_flow_grid=True,
                                                                                          flag_interpolation_mode='bilinear')
                    Flow_Grids[counter] = current_flow_grid
                    counter = counter + 1


    ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
    possible_warps_tensor = input_image_1.repeat((number_of_possible_warps, 1, 1, 1))

    ### Get All Possible Warps Of Previous Image: ###
    possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, Flow_Grids, mode='bilinear')
    # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

    ### Get Min SAD: ###
    # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
    SAD_matrix_2 = torch.abs(possible_warps_tensor - input_image_2)[:,:, :, :].mean(-1, True).mean(-2, True)
    min_index_2 = torch.argmin(SAD_matrix_2)
    min_index_2 = np.atleast_1d(min_index_2.cpu().numpy())[0]
    min_indices = np.unravel_index(min_index_2, SAD_matrix_numpy.shape)
    shift_x_inference = shifts_vec[min_indices[0]]
    shift_y_inference = shifts_vec[min_indices[1]]
    rotation_angle_inference = rotation_angle_vec[min_indices[2]]
    scale_factor_inference = scale_factor_vec[min_indices[3]]

    ### Correct Sub-Pixel: ###
    #(1). Get Min Indices:
    SAD_matrix_2_reshaped = torch.reshape(SAD_matrix_2.squeeze(), (L_shifts, L_shifts, L_rotation_angle, L_scale))
    x_vec = [-1, 0, 1]
    y_vec = [-1, 0, 1]
    shift_x_index = min_indices[0]
    shift_y_index = min_indices[1]
    rotation_index = min_indices[2]
    scale_index = min_indices[3]
    #(2). Get Indices For Parabola Fitting:
    shift_x_indices = np.array([shift_x_index - 1, shift_x_index, shift_x_index + 1])
    shift_y_indices = np.array([shift_y_index - 1, shift_y_index, shift_y_index + 1])
    rotation_indices = np.array([rotation_index - 1, rotation_index, rotation_index + 1])
    scale_indices = np.array([scale_index - 1, scale_index, scale_index + 1])
    shift_x_indices[shift_x_indices < 0] += L_shifts
    shift_x_indices[shift_x_indices >= L_shifts] -= L_shifts
    shift_y_indices[shift_y_indices < 0] += L_shifts
    shift_y_indices[shift_y_indices >= L_shifts] -= L_shifts
    rotation_indices[rotation_indices < 0] += L_rotation_angle
    rotation_indices[rotation_indices >= L_rotation_angle] -= L_rotation_angle
    scale_indices[scale_indices < 0] += L_scale
    scale_indices[scale_indices >= L_scale] -= L_scale

    #### Fot Shift X Parabola: ###
    fitting_points_shift_x = SAD_matrix_2_reshaped[shift_x_indices, shift_y_index, rotation_index, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_x)
    delta_shift_x = -b_x / (2 * a_x)
    shift_x_sub_pixel = shifts_vec[shift_x_index] + delta_shift_x * (shifts_vec[2] - shifts_vec[1])

    ### Fit Shit Y Parabola: ###
    fitting_points_shift_y = SAD_matrix_2_reshaped[shift_x_index, shift_y_indices, rotation_index, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_shift_y)
    delta_shift_y = -b_x / (2 * a_x)
    shift_y_sub_pixel = shifts_vec[shift_y_index] + delta_shift_y * (shifts_vec[2] - shifts_vec[1])

    ### Fit Rotation Parabola: ###
    fitting_points_rotation = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_indices, scale_index]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_rotation)
    delta_rotation = -b_x / (2 * a_x)
    rotation_sub_pixel = rotation_angle_vec[rotation_index] + delta_rotation * (rotation_angle_vec[2] - rotation_angle_vec[1])

    ### Fit Scale Parabola: ###
    fitting_points_scale = SAD_matrix_2_reshaped[shift_x_index, shift_y_index, rotation_index, scale_indices]
    y_vec = [-1, 0, 1]
    [c_x, b_x, a_x] = fit_polynomial(y_vec, fitting_points_scale)
    delta_scale = -b_x / (2 * a_x)
    scale_sub_pixel = scale_factor_vec[scale_index] + delta_scale * (scale_factor_vec[2] - scale_factor_vec[1])

    print('recovered shift x: ' + str(shift_x_sub_pixel))
    print('recovered shift y: ' + str(shift_y_sub_pixel))
    print('recovered rotation: ' + str(rotation_sub_pixel))
    print('recovered scale: ' + str(scale_sub_pixel))



def Gimbaless_3D_1(input_tensor):
    #### Get Mat Size: ###
    B,C,H,W = input_tensor.shape

    ######################################################################################################
    ### Prediction Grid Definition: ###
    prediction_block_size = 16 #for one global prediction for everything simply use W or H
    overlap_size = 0 #0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
    temporal_lowpass_number_of_steps = 5

    ### create filter: ###  TODO: can be preallocated
    params = EasyDict()
    params.dif_len = 9
    # spatial_lowpass_before_temporal_derivative_filter_x = Gimbaless_create_filter(params)
    spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((B,1,1,params.dif_len))
    spatial_lowpass_before_temporal_derivative_filter_y = spatial_lowpass_before_temporal_derivative_filter_x.permute([0,1,3,2])
    spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
    spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])

    ### Preallocate filters: ###
    temporal_derivative_filter = [-1,1]
    temporal_derivative_filter = reshape(temporal_derivative_filter, 1,1,2)
    temporal_derivative_fft_filter = fft(temporal_derivative_filter, T, 3)

    # grad_x_kernel = [1,0,-1]/2;
    # grad_y_kernel = [1,0,-1]'/2;
    grad_x_filter_fft = fft(grad_x_kernel, W, 2);
    grad_y_filter_fft = fft(grad_y_kernel, H, 1);

    temporal_averaging_filter_before_spatial_gradient = ones(1,1,temporal_lowpass_number_of_steps)/temporal_lowpass_number_of_steps;
    temporal_averaging_filter_before_spatial_gradient_fft = fft(temporal_dtemporal_averaging_filter_before_spatial_gradienterivative_filter, T, 3);
    ######################################################################################################


#
#
# ######################################################################################################
# ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###
#
# %(*). basically a [-1,0,1] filter
# %(*). after calculating the gradient per 2D matrix we can average the
# %results on however consecutive matrices we want...whether 2 or T or whatever
# %(1).
# px = convn(input_tensor, grad_x_kernel, 'same');
# % px2 = filter(grad_x_kernel, [1], input_tensor, [], 1);
# sum(sum(sum(abs(px-px2))))
#
# py = convn(input_tensor, grad_y_kernel, 'same');
# px = px(:,:,1:end-1); % T-1 displacements to output
# py = py(:,:,1:end-1);
#
# % %(2). average out consecutive frames to smooth spatial gradients a bit:
# % input_tensor_averaged = (input_tensor(:,:,1:end-1) + input_tensor(:,:,2:end))/2;
# % px = convn(input_tensor_averaged, grad_x_kernel, 'same');
# % py = convn(input_tensor_averaged, grad_y_kernel, 'same');
#
# % %(3).
# % px = (input_tensor(:,3:end,:) - input_tensor(:,1:end-2,:))/2;
# % py = (input_tensor(3:end,:,:) - input_tensor(1:end-2,:,:))/2;
# % px = padarray(px, [0,1]);  %pad array to fid later cropping instead of if conditions...not necessary
# % py = padarray(py, [1,0]);
#
# % %(3). average images out to get a better reading of the spatial gradient:
# % %TODO: perhapse can use cumsum?
# % input_tensor_averaged = convn(input_tensor, temporal_averaging_filter_before_spatial_gradient, 'same');
# % px = convn(input_tensor_averaged, grad_x_kernel, 'same');
# % py = convn(input_tensor_averaged, grad_y_kernel, 'same');
#
# % %(4). FFT Filtering 1D:
# % input_tensor_fft_x = fft(input_tensor,[],2); %TODO:  I don't even need to do fft2...i can simply use fft 1D!!!!
# % input_tensor_fft_y = fft(input_tensor,[],1);
# % px = ifft2(input_tensor_fft_x .* grad_x_filter_fft);
# % py = ifft2(input_tensor_fft_y .* grad_y_filter_fft);
#
# % %(5). FFT Filtering 1D (spatial and temporal):
# % input_tensor_temporal_fft = fft(input_tensor, [], 3);
# % input_tensor_averaged = ifft(input_tensor_temporal_fft .* temporal_averaging_filter_before_spatial_gradient_fft, [], 3);
# % input_tensor_fft_x = fft(input_tensor_averaged, [], 2); %TODO:  I don't even need to do fft2...i can simply use fft 1D!!!!
# % input_tensor_fft_y = fft(input_tensor_averaged, [], 1);
# % px = ifft2(input_tensor_fft_x .* grad_x_filter_fft);
# % py = ifft2(input_tensor_fft_y .* grad_y_filter_fft);
#
# %(6). Generalized Gradient Method. %TODO: can i use fftn to calculate the 3D FFT and
# %then multiply-filter? i don't think so....doing spatial fft and temporal
# %fft is not the same as using the 3D-FFT...but maybe i can use the 3D fft
# %for something more interesting? in the end optical flow is the same as
# %maxwell's equations...we're looking for the 4D generalized gradient to go to zero
#
# % ### Cut-Out Invalid Parts (Center-Crop): ###
# % %TODO: perhapse don't(!!!!) center crop now, but later
# % start_index = floor(params.dif_len/2);
# % px = px(start_index:end-(start_index-1), start_index:end-(start_index-1), 2:end);
# % py = py(start_index:end-(start_index-1), start_index:end-(start_index-1), 2:end);
# ######################################################################################################
#
#
# ######################################################################################################
# ### Calculate Temporal Derivative: ###
#
# ### low-pass correction of the image due to the gradient operator: ###
# %(1). Get the temporal difference between each two consecutive frames after lowpass spatial filting (the original was conv2(kernel, A-B)):
# %TODO: there's something about the fact that we are using spatial
# %convolution and then temporal convolution/substraction which supposedly
# %calls out for optimization. which is probably why using something like
# %HALIDE helped....i think it should tell us more then anything that we
# %aren't writting things smart or the algorithm is not general
# %enough....shouldn't we be able to use multi-dimensional gradient which can
# %be implemented using FFT3 or something??!??!!?!?!
# ABtx = convn(input_tensor, spatial_lowpass_before_temporal_derivative_filter_x, 'same');
# ABty = convn(input_tensor, spatial_lowpass_before_temporal_derivative_filter_y, 'same');
# ABtx = ABtx(:,:,2:end) - ABtx(:,:,1:end-1);
# ABty = ABty(:,:,2:end) - ABty(:,:,1:end-1);
#
# % %(2). FFT Domain: (very inefficient as it is....maybe it can be made more efficient....)
# % ABtx = ifft(input_tensor_fft_x .* spatial_lowpass_before_temporal_derivative_filter_fft_x, [], 2);
# % ABty = ifft(input_tensor_fft_y .* spatial_lowpass_before_temporal_derivative_filter_fft_y, [], 1);
# % ABtx_fft = fft(ABtx, [], 3);
# % ABty_fft = fft(ABty, [], 3);
# % ABtx = real(ifft(ABtx_fft .* temporal_derivative_fft_filter, [], 3));
# % ABty = real(ifft(ABty_fft .* temporal_derivative_fft_filter, [], 3));
#
# % ### Cut-Out Invalid Parts (Center-Crop): ###
# % %TODO: perhapse don't(!!!!) center crop now, but later
# % ABtx = ABtx(start_index:end-(start_index-1), start_index:end-(start_index-1), :);
# % ABty = ABty(start_index:end-(start_index-1), start_index:end-(start_index-1), :);
# ######################################################################################################
#
#
# ######################################################################################################
# ### Find Shift Between Frames: ###
#
# %(*) inv([A,B;C,D]) = 1/(AD-BC)*[D,-B;-C,A]
# ### Get elements of the first matrix in the equation we're solving: ###
# %TODO: perhapse make this total sum into local sums so as to allow local prediction?!?!?!
# %(1). Global sum over entire matrix (one prediction for everything):
# pxpy = squeeze(sum(px.*py,[1,2]));
# px2 = squeeze(sum(px.^2,[1,2]));
# py2 = squeeze(sum(py.^2,[1,2]));
# % %(2). Local Prediction:
# % pxpy = squeeze(fast_binning_2D_PixelBinning(px.*py, prediction_block_size, overlap_size));
# % px2 = squeeze(fast_binning_2D_PixelBinning(px.^2, prediction_block_size, overlap_size));
# % py2 = squeeze(fast_binning_2D_PixelBinning(py.^2, prediction_block_size, overlap_size));
#
# ### Invert the above matrix explicitly in a hard coded fashion until i find
# ### a more elegant way to invert matrices in parallel: ###
# % P = [px2,pxpy; py2,pxpy] -> matrix we are explicitely inverting
# % inv_P = 1./(A.*D-B.*C) .* [D,-B;-C,A];
# common_factor = 1./(px2.*py2 - pxpy.*pxpy);
# inv_P_xx = common_factor .* py2;
# inv_P_xy = -common_factor .* pxpy;
# inv_P_yx = -common_factor .* pxpy;
# inv_P_yy = common_factor .* px2;
#
# ### Solve the needed equation explicitly: ###
# % d = inv_P * [squeeze(sum(ABtx.*px, [1,2])); squeeze(sum(ABty.*py,[1,2]))];
# %(1). Global sum over entire matrix (one prediction for everything):
# %TODO: i'm sure the "squeeze" has the potential of wasting time and is only
# %there so that the multiplication dimensions hold...probably do something about it
# delta_x = inv_P_xx .* squeeze(sum(ABtx.*px, [1,2])) + ...
#     inv_P_xy .* squeeze(sum(ABty.*py, [1,2]));
# delta_y = inv_P_yx .* squeeze(sum(ABtx.*px, [1,2])) + ...
#     inv_P_yy .* squeeze(sum(ABty.*py, [1,2]));
# % % %(2). Local Prediction:
# % delta_x = inv_P_xx .* squeeze(fast_binning_2D_PixelBinning(ABtx.*px, prediction_block_size, overlap_size)) + ...
# %     inv_P_xy .* squeeze(fast_binning_2D_PixelBinning(ABty.*py, prediction_block_size, overlap_size));
# % delta_y = inv_P_yx .* squeeze(fast_binning_2D_PixelBinning(ABtx.*px, prediction_block_size, overlap_size)) + ...
# %     inv_P_yy .* squeeze(fast_binning_2D_PixelBinning(ABty.*py, prediction_block_size, overlap_size));
# % toc()
#
# %TODO: ABtx(:)'*px(:) is equivalent to element-wise multiplication and sum!
# %TODO: inv(p'*p) can be implemented hard-coded as it's only 2x2 matrix.
# %need to time what's faster!
# ######################################################################################################





# ### Prepare Data: ###
# warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
# fft_translation_rotation_layer = FFT_Translation_Rotation_Layer()
# shift_x = np.array([0.])
# shift_y = np.array([0.])
# scale = np.array([1.00])
# rotation_angle = np.array([0.5])  # [degrees]
# input_image_1 = read_image_default_torch().cuda()
# input_image_1 = RGB2BW(input_image_1)
# input_image_2 = warp_tensors_affine_layer.forward(input_image_1, shift_x, shift_y, scale, rotation_angle)
# # input_image_2 = fft_translation_rotation_layer.forward(input_image_1, shift_x, shift_y, rotation_angle)
# input_image_1 = crop_torch_batch(input_image_1, 700)
# input_image_2 = crop_torch_batch(input_image_2, 700)
# input_image_1 = nn.AvgPool2d(1)(input_image_1)
# input_image_2 = nn.AvgPool2d(1)(input_image_2)
# input_image_1_numpy = input_image_1.cpu().numpy()[0,0]
# input_image_2_numpy = input_image_2.cpu().numpy()[0,0]
#
# #TODO: when using the bilinear warp function to rotate the image then minSAD registration returns correct results.
# #TODO: when using the fft translation rotation layer to rotate the image then minSAD registration doesn't(!!!) return correct results. understand this!!!!!
#
# ### Using Min-SAD: ###
# # minsad_fft_affine_registration_layer = minSAD_FFT_Affine_Registration_Layer()
# minsad_fft_affine_registration_layer = minSAD_FFT_Affine_Registration_MemorySave_Layer()
# shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, input_image_2_aligned = minsad_fft_affine_registration_layer.forward(input_image_1, input_image_2)
#
# imshow_torch(input_image_1)
# imshow_torch(input_image_2)
# imshow_torch(input_image_2_aligned)
# print((shift_x_sub_pixel, shift_y_sub_pixel))
# print(rotation_sub_pixel)
# ##########################################################################################################################################################





# ##########################################################################################################################################################
### BORIS EFCOM: ###
def z_score_filter_avg_BORIS(nparray, harris_thresholdold=3.5):
    #z-score filter
    mean_int = np.mean(nparray)
    if (mean_int == 0):
        return 0
    std_int = np.std(nparray)
    z_scores = (nparray - mean_int) / std_int

    count = 0
    avg = 0
    for k in range(0,len(z_scores)):
        if abs(z_scores[k]) < harris_thresholdold:
            avg += z_scores[k]
            count += 1
    return int(avg/count)

def calc_images_translation_BORIS(img1, img2, numX, numY, stp):
#nimg1 and nimg2 are images to compare. numX, and numY are the number of search ROIs per image. step is the size of the ROI.
    tX = img1.shape[0]
    tY = img1.shape[1]
    # logging.debug("Dividing image to " + str(numX*numY) +" parts. Croping a " + str(2*stp) + "*" + str(2*stp) + " search pattern") #debug
    horizontal = np.zeros(numX*numY, int)
    vertical = np.zeros(numX*numY, int)
    for i in range(0,numX):
        for j in range(0,numY):
            #divide image into segments and calc positions of ROIs to search
            imTMP = img1[int((tX/numX)*i) + int(tX/(2*numX))-stp:int((tX/numX)*i) + int(tX/(2*numX))+stp, int((tY/numY)*j) + int(tY/(2*numY))-stp:int((tY/numY)*j) + int(tY/(2*numY))+stp]
            #do the work
            result = cv2.matchTemplate((img2*255).clip(0,255).astype(np.uint8), (imTMP*255).clip(0,255).astype(np.uint8), cv2.TM_CCORR_NORMED)
            _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
            yshift = maxLoc[0] - (int((tY/numY)*j) + int(tY/(2*numY))-stp)
            xshift = maxLoc[1] - (int((tX/numX)*i) + int(tX/(2*numX))-stp)
            vertical = np.append(vertical, yshift)
            horizontal = np.append(horizontal, xshift)
            # logging.debug ("Crop i:" + str(i) + " j:" + str(j) + " is shifted by x:" + str(xshift) + " y:"+str(yshift)) #debug

    #select the best x,y values from all ROIs
    return (z_score_filter_avg_BORIS(horizontal), z_score_filter_avg_BORIS(vertical))

def translate_BORIS(image, x, y):
    # move and image, define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted

def get_images_translation_and_translate_BORIS(image1, image2):
    transx, transy = calc_images_translation_BORIS(image1, image2, 4, 4, 10)
    image2_shifted = translate_BORIS(image2, transx, transy)
    return image2_shifted

def get_image_translation_and_translate_batch_BORIS(image_batch):
    T,H,W = image_batch.shape
    reference_tensor = image_batch[T//2,:,:]
    aligned_tensors = []
    for time_index in np.arange(T):
        current_tensor = image_batch[time_index,:,:]
        current_tensor_shifted = get_images_translation_and_translate_BORIS(reference_tensor, current_tensor)
        aligned_tensors.append(np.expand_dims(current_tensor_shifted,0))

    aligned_tensors = np.concatenate(aligned_tensors, 0)
    return aligned_tensors
# ##########################################################################################################################################################



# ##########################################################################################################################################################
### another way of tracking shifts using opencv: ###
def Ctypi(Movie):
    full_frame_flg = True
    ctypi_fr_sz = 64
    ctypi_fr_sz_2 = np.int(ctypi_fr_sz / 2)

    feature_pts = cv2.goodFeaturesToTrack(np.uint8(np.float32(Movie[0]) / 3), maxCorners=200,
                                          qualityLevel=0.001, minDistance=3, blockSize=3)
    x = feature_pts.reshape((feature_pts.shape[0], 2))
    y_pred = KMeans(n_clusters=10, random_state=170).fit_predict(x)
    yind = np.where(y_pred == np.where(np.bincount(y_pred) == max(np.bincount(y_pred)))[0][0])[0]
    cnt_ = np.mean(x[yind, :], axis=0).astype('int')
    Movie = Movie[:, cnt_[1] - ctypi_fr_sz_2:cnt_[1] + ctypi_fr_sz_2, cnt_[0] - ctypi_fr_sz_2:cnt_[0] + ctypi_fr_sz_2]

    return Movie
# ##########################################################################################################################################################


import torch
import numpy as np
from RapidBase.import_all import *



### Cross Correlation: ###
from _alignments.canny_edge_detection import canny_edge_detection


def get_CircularCrossCorrelation_batch_torch(input_tensor, reference_tensor=None, flag_normalize_CC_values=False, flag_fftshift=False):
    T,C,H,W = input_tensor.shape
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1,-2])
    if reference_tensor==None:
        reference_tensor = input_tensor[T//2:T//2+1]
    reference_tensor_fft = torch.fft.fftn(reference_tensor, dim=[-1,-2])
    output_CC = torch.fft.ifftn(input_tensor_fft[:, :, :, :] * reference_tensor_fft.conj(),
                                dim=[-1, -2]).real

    # output_CC = torch.cat()
    if flag_normalize_CC_values:
        ### Get Normalized Cross Correlation (max possible value = 1): ###
        A_sum = reference_tensor.sum(dim=[-1, -2])
        A_sum2 = (reference_tensor ** 2).sum(dim=[-1, -2])
        sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
        sigmaB = (input_tensor).std(dim=[-1, -2]) * (H * W - 1) ** (1 / 2)
        B_mean = (input_tensor).mean(dim=[-1, -2])
        output_CC = (output_CC - (A_sum * B_mean).unsqueeze(-1).unsqueeze(-1)) / (sigmaA * sigmaB).unsqueeze(-1).unsqueeze(-1)

    if flag_fftshift:
        output_CC = fftshift_torch(output_CC, first_spatial_dim=2)

    return output_CC


# has shifts
def Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(input_tensor):
    #TODO: make sure it can accept all the different inputs

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
    T, C, H, W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1, -2])
    # (1). Circular Cross Corerlation:
    output_CC = get_CircularCrossCorrelation_batch_torch(input_tensor, flag_normalize_CC_values=False)
    #########################################################################################################

    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W   #(*). discrete row (H dimension) max indices
    i1 = output_CC_flattened_indices - i0 * W  #(*). discrete col (W dimension) max indices
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

    return (shifty,shiftx), shifted_tensors   #TODO: always remember to return in (H,W) when it's all said and done



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
    ###A getting the normalized cross correlation
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


def get_Normalized_Cross_Correlation_FFTImplementation_torch(input_tensor1, input_tensor2, correlation_size):
    # (*). Normalized Cross Correlation!!!!
    ### Trims: ###
    C, H, W = input_tensor1.shape

    zero_padding_size = correlation_size // 2
    zeros_map = torch.ones((1, H - 2 * zero_padding_size, W - 2 * zero_padding_size))
    zeros_map = torch.nn.functional.pad(zeros_map,
                                        [zero_padding_size, zero_padding_size, zero_padding_size, zero_padding_size],
                                        value=0)
    input_tensor1 = input_tensor1 * zeros_map  # TODO: probably instead of using indices use a better predefined zeros_map with T frames
    input_tensor_fft = torch.fft.fftn(input_tensor1, dim=[-1, -2])
    reference_tensor_fft = torch.fft.fftn(input_tensor2, dim=[-1, -2])
    NCC = torch.fft.ifftn(input_tensor_fft * reference_tensor_fft.conj(), dim=[-1, -2]).real

    return NCC


def get_Normalized_Cross_Correlation_FFTImplementation_batch_torch(input_tensor, correlation_size, reference_tensor=None, flag_normalize_CC_values=True):
    #(*). Normalized Cross Correlation!!!!
    ### Trims: ###
    T,C,H,W = input_tensor.shape
    if reference_tensor == None:
        reference_tensor = input_tensor[T//2:T//2+1].clone()

    zero_padding_size = correlation_size//2
    zeros_map = torch.ones((T, H - 2*zero_padding_size, W - 2*zero_padding_size))
    zeros_map = torch.nn.functional.pad(zeros_map, [zero_padding_size,zero_padding_size,zero_padding_size,zero_padding_size], value=0)
    zeros_map = zeros_map.unsqueeze(1)
    input_tensor = input_tensor * zeros_map  #TODO: probably instead of using indices use a better predefined zeros_map with T frames
    input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1,-2])
    reference_tensor_fft = torch.fft.fftn(reference_tensor, dim=[-1,-2])
    NCC = torch.fft.ifftn(input_tensor_fft * reference_tensor_fft.conj(), dim=[-1,-2]).real
    # NCC = fftshift_torch(NCC,2)  #uncomment this to get it in the "correct form"

    if flag_normalize_CC_values:
        ### Get Normalized Cross Correlation (max possible value = 1): ###
        A_sum = reference_tensor.sum(dim=[-1,-2])
        A_sum2 = (reference_tensor ** 2).sum(dim=[-1,-2])
        sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
        sigmaB = (input_tensor).std(dim=[-1,-2]) * (H * W - 1) ** (1 / 2)
        B_mean = (input_tensor).mean(dim=[-1,-2])
        NCC = (NCC - A_sum * B_mean) / (sigmaA * sigmaB)

    return NCC


def get_Normalized_Cross_Correlation_batch_torch(input_tensor, correlation_size):
    #(*). Normalized Cross Correlation!!!!
    ### Trims: ###
    T, C, H, W = input_tensor.shape
    reference_tensor = input_tensor[T//2:T//2+1]

    NCC = torch.zeros((T,C,correlation_size, correlation_size))
    #TODO: i really think the funciton below is ALREADY parallel, so check it out, myabe i don't need the loop at all
    for i in np.arange(T):
        current_tensor = input_tensor[i:i+1]
        NCC[i:i+1,:,:,:] = get_Normalized_Cross_Correlation_torch(current_tensor, reference_tensor, correlation_size)

    return NCC


def Align_TorchBatch_ToCenterFrame_NormalizedCrossCorrelation(input_tensor, correlation_size):
    #TODO: make sure it can accept all the different inputs

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
    T, C, H, W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    # (1). Normalized Cross Corerlation:
    correlation_size = 5
    output_CC = get_Normalized_Cross_Correlation_batch_torch(input_tensor, correlation_size)
    # output_CC = torch.cat()
    #########################################################################################################

    ### Get Correct Max Indices For Fit: ###
    #TODO: add what to do when max index is at the edges
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, correlation_size * correlation_size), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // correlation_size   #(*). discrete row (H dimension) max indices
    i1 = output_CC_flattened_indices - i0 * correlation_size  #(*). discrete col (W dimension) max indices
    i0_original = i0 + 0
    i1_original = i1 + 0
    i0_minus1 = i0 - 1
    i0_plus1 = i0 + 1
    i1_minus1 = i1 - 1
    i1_plus1 = i1 + 1
    ### Get Flattened Indices From Row/Col Indices: ###
    output_CC_flattened_indices_i1 = i1 + i0 * correlation_size
    output_CC_flattened_indices_i1_minus1 = i1_minus1 + i0_minus1 * correlation_size
    output_CC_flattened_indices_i1_plus1 = i1_plus1 + i0_plus1 * correlation_size
    output_CC_flattened_indices_i0 = i1 + i0 * correlation_size
    output_CC_flattened_indices_i0_minus1 = i1 + i0_minus1 * correlation_size
    output_CC_flattened_indices_i0_plus1 = i1 + i0_plus1 * correlation_size
    #########################################################################################################

    #########################################################################################################
    ### Get Proper Values For Fit: ###
    # output_CC_flattened_indices = output_CC_flattened_indices.unsqueeze(-1)
    output_CC = output_CC.contiguous().view(-1, correlation_size * correlation_size)
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
    #TODO: parabola max isn't correct!!!!!!
    delta_shiftx = -b_x / (2 * a_x)
    x_parabola_max = a_x + b_x * delta_shiftx + c_x * delta_shiftx * 2
    delta_shifty = -b_y / (2 * a_y)
    y_parabola_max = a_y + b_y * delta_shifty + c_y * delta_shifty * 2
    # Add integer shift:
    shiftx = i1_original.squeeze() + delta_shiftx
    shifty = i0_original.squeeze() + delta_shifty
    # Substract Center Cross Correlation: ###
    shiftx = shiftx - correlation_size//2
    shifty = shifty - correlation_size//2
    # print(shiftx)
    # print(real_shifts_to_center_frame)
    #########################################################################################################

    #########################################################################################################
    ### Align Images: ###
    shifted_tensors = shift_layer.forward(input_tensor, -shiftx, -shifty)  # TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0, True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    return (shifty,shiftx), shifted_tensors   #TODO: always remember to return in (H,W) when it's all said and done


def Align_TorchBatch_ToCenterFrame_NormalizedCrossCorrelationFFTImplementation(input_tensor, correlation_size):
    #TODO: make sure it can accept all the different inputs

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
    T, C, H, W = input_tensor.shape
    cross_correlation_shape = (H, W)
    midpoints = (np.fix(H / 2), np.fix(W / 2))
    # (1). Normalized Cross Corerlation:
    correlation_size = 5
    output_CC = get_Normalized_Cross_Correlation_FFTImplementation_batch_torch(input_tensor, correlation_size)
    # output_CC = torch.cat()
    #########################################################################################################

    ### Get Correct Max Indices For Fit: ###
    output_CC_flattened_indices = torch.argmax(output_CC.contiguous().view(T, H * W), dim=-1).unsqueeze(-1)
    i0 = output_CC_flattened_indices // W  # (*). discrete row (H dimension) max indices
    i1 = output_CC_flattened_indices - i0 * W  # (*). discrete col (W dimension) max indices
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
    fitting_points_x = torch.cat(
        [output_CC_flattened_values_i1_minus1, output_CC_flattened_values_i1, output_CC_flattened_values_i1_plus1], -1)
    fitting_points_y = torch.cat(
        [output_CC_flattened_values_i0_minus1, output_CC_flattened_values_i0, output_CC_flattened_values_i0_plus1], -1)
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
    shifted_tensors = shift_layer.forward(input_tensor, -shiftx, -shifty)  # TODO: instead of permuting make shift_layer accept [1,T,H,W]
    mean_frame_averaged = shifted_tensors.mean(0, True)
    # print(input_tensor.shape)
    # imshow_torch(input_tensor[:,number_of_samples//2])
    # imshow_torch(mean_frame_averaged[number_of_samples//2])
    #########################################################################################################

    return (shifty,shiftx), shifted_tensors   #TODO: always remember to return in (H,W) when it's all said and done


def test_circular_cross_correlation():
    T = 128
    H = 256
    W = 256
    speckle_size = 5
    shift_layer = Shift_Layer_Torch()
    speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(speckle_size, H, 0, 1, 1, 0)
    speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    C,H,W = speckle_pattern.shape
    ### Get Input Tensor by stacking the different speckle patterns: ###
    input_tensor = torch.cat([speckle_pattern]*T, 0) + 5
    current_shift = 0
    ### Actually Shift: ###
    shifts_vec = torch.randn(T) * 1
    real_shifts_to_reference_tensor = shifts_vec - shifts_vec[T//2]
    input_tensor = shift_layer.forward(input_tensor, shifts_vec.cpu().numpy(), shifts_vec.cpu().numpy())
    input_tensor = input_tensor.unsqueeze(1)

    ### Get Cross Correlation: ###
    (shift_y,shift_x), shifted_tensors = Align_TorchBatch_ToCenterFrame_CircularCrossCorrelation(input_tensor)

    ### Present Shift Inference Results: ###
    shift_x = shift_x.cpu().numpy()
    shift_y = shift_y.cpu().numpy()
    plot(shift_x)
    plot(shift_y)
    plot(real_shifts_to_reference_tensor)
    legend(['shift_x', 'shift_y', 'real_shifts'])

    ### Show aligned video: ###
    imshow_torch_video(shifted_tensors, 30)

def test_normalized_cross_correlation():
    T = 128
    H = 256
    W = 256
    speckle_size = 5
    shift_layer = Shift_Layer_Torch()
    speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(speckle_size, H, 0, 1, 1, 0)
    speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    C,H,W = speckle_pattern.shape
    ### Get Input Tensor by stacking the different speckle patterns: ###
    input_tensor = torch.cat([speckle_pattern]*T, 0) + 5
    current_shift = 0
    ### Actually Shift: ###
    shifts_vec = torch.randn(T) * 0.5
    shifts_vec = shifts_vec.clip(-0.8, 0.8)
    real_shifts_to_reference_tensor = shifts_vec - shifts_vec[T//2]
    input_tensor = shift_layer.forward(input_tensor, shifts_vec.cpu().numpy(), shifts_vec.cpu().numpy())
    input_tensor = input_tensor.unsqueeze(1)
    input_tensor = crop_torch_batch(input_tensor, (H-5,H-5))

    ### Get Cross Correlation: ###
    (shift_y, shift_x), shifted_tensors = Align_TorchBatch_ToCenterFrame_NormalizedCrossCorrelation(input_tensor, 7)

    ### Present Shift Inference Results: ###
    shift_x = shift_x.cpu().numpy()
    shift_y = shift_y.cpu().numpy()
    plot(shift_x)
    plot(shift_y)
    plot(real_shifts_to_reference_tensor)
    legend(['shift_x', 'shift_y', 'real_shifts'])

    ### Show aligned video: ###
    imshow_torch_video(shifted_tensors, 30)

def test_normalized_cross_correlation_FFTImplementation():
    T = 128
    H = 256
    W = 256
    speckle_size = 5
    shift_layer = Shift_Layer_Torch()
    speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(speckle_size, H, 0, 1, 1, 0)
    speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    C,H,W = speckle_pattern.shape
    ### Get Input Tensor by stacking the different speckle patterns: ###
    input_tensor = torch.cat([speckle_pattern]*T, 0) + 5
    current_shift = 0
    ### Actually Shift: ###
    shifts_vec = torch.randn(T) * 0.5
    shifts_vec = shifts_vec.clip(-0.8, 0.8)
    real_shifts_to_reference_tensor = shifts_vec - shifts_vec[T//2]
    input_tensor = shift_layer.forward(input_tensor, shifts_vec.cpu().numpy(), shifts_vec.cpu().numpy())
    input_tensor = input_tensor.unsqueeze(1)
    input_tensor = crop_torch_batch(input_tensor, (H-5,H-5))

    ### Get Cross Correlation: ###
    # NCC = get_Normalized_Cross_Correlation_FFTImplementation_batch_torch(input_tensor, 7)
    (shift_y, shift_x), shifted_tensors = Align_TorchBatch_ToCenterFrame_NormalizedCrossCorrelation(input_tensor, 7)

    ### Present Shift Inference Results: ###
    shift_x = shift_x.cpu().numpy()
    shift_y = shift_y.cpu().numpy()
    plot(shift_x)
    plot(shift_y)
    plot(real_shifts_to_reference_tensor)
    legend(['shift_x', 'shift_y', 'real_shifts'])

    ### Show aligned video: ###
    imshow_torch_video(shifted_tensors, 30)


def get_Weighted_Circular_Cross_Correlation_Batch_torch(input_tensor, reference_tensor, weights_tensor=None):

    ### TODO: delete later on: ###
    input_tensor = input_tensor[0].unsqueeze(1)  #turn into [T,C,H,W] instead of [1,T,H,W] - will be taken care of later with Anvil

    ### Get Default Weights Tensor If None Was Inputed: ###
    if weights_tensor is None:
        ### Different Candidate For Cross Correlation Weights: ###
        #TODO: try structure tensor, or maybe something else?
        input_tensor_mean = input_tensor.mean(0) #Assuming [T,C,H,W]
        canny_edge_detection_layer = canny_edge_detection(10)
        blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = \
            canny_edge_detection_layer.forward(BW2RGB(input_tensor_mean.unsqueeze(0)), 3)
        weights_tensor = early_threshold
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
    ### Use fftshift for now to make up for above used fftshift, i'll understand what's going on and take care of it later: ###
    cross_correlation = fftshift_torch(cross_correlation, -2)
    return cross_correlation





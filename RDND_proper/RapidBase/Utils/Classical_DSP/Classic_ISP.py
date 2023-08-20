import torch

# from RapidBase.import_all import *
import numpy as np
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_image_flatten, get_full_shape_torch, pad_with_zeros_frame_torch, torch_get_ND, torch_unsqueeze_like

def get_Sigma(I):
    dx = torch.diff(I, dim=-1)
    dx_flattened = torch_image_flatten(dx)
    s = 1.4826 * (dx_flattened - dx_flattened.median(-1)[0].unsqueeze(-1)).abs().median(-1)[0]  #TODO: why not do it on a column-wise basis instead of a single number?
    return s

def D1boxfilter(I_col, r):
    #(*). Notice!!!!: the secret to Matlab->Pytorch indexing is: in matlab the first possible index is 1 as opposed to 0, moreover,
    # in matlab when indexing an array start_index:stop_index it gets transferred to start_index-1:stop_index in pytorch!!!!
    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(I_col)
    col_size = H #input shape is ...,H,1
    imDst = torch.zeros_like(I_col)
    imCum = torch.cumsum(I_col, -2)
    imDst[..., 0:r+1, :] = imCum[..., r:2*r+1, :]  #for the first r terms in the beginning simply use the cumulative sum of the first r "valid" (with at least r terms) sum
    imDst[..., r+1:H-r, :] = imCum[..., 2*r+1:H, :] - imCum[..., 0:H-2*r-1, :]  #moving sum (box filter) of 2r+1 terms
    imDst[..., H-r:H, :] = torch.cat([imCum[...,H-1:H, :]]*r, dim=-2) - imCum[..., H-2*r-1:H-r-1, :]  #
    return imDst

def verticalBoxfilter(I, r):
    H, W = I.shape[-2:]
    H_image = torch.ones((H,1))   # torch_image_flatten(I) - TODO: in the end make this torch way
    Nv = D1boxfilter(H_image, r)
    mI = D1boxfilter(I, r) / Nv
    return mI

def weightedLocalLinearFilter(x, y, w, r, eps):
    ww = verticalBoxfilter(w, r)
    wx = verticalBoxfilter(w * x, r) / ww
    wy = verticalBoxfilter(w * y, r) / ww
    wxy = verticalBoxfilter(w * x * y, r) / ww
    wxx = verticalBoxfilter(w * x * x, r) / ww
    a = (wxy - wx*wy + eps) / (wxx - wx*wx + eps)
    b = wy - wx * a
    mean_a = verticalBoxfilter(a, r)
    mean_b = verticalBoxfilter(b, r)
    u = (y - mean_b) / mean_a
    return u, a, b

### Vertical Weighted Local Ridge Regression Stage: ###
def edgeIndicator(L, xi):
    r = 5  #TODO: perhapse make this a variable?!!?!?!?
    mean_1 = (verticalBoxfilter(L.transpose(-1,-2), r)).transpose(-1,-2)
    mean_2 = (verticalBoxfilter((L*L).transpose(-1,-2), r)).transpose(-1,-2)
    Var = mean_2 - mean_1*mean_1
    m = Var.mean([-1,-2], True)
    Dire = torch.exp(-Var/(xi*m)) + 1e-10
    return Dire, m

def linearInverseOperator(a, b, c, f):
    ### Using Guass Elimination To Inverse a 3-point laplacian matrix: ###
    DL = a.shape[-1]

    ### Calculate the forward computation: ###
    c[:, 0] = c[:, 0] / b[:, 0]
    f[:, 0] = f[:, 0] / b[:, 0]

    # (*). this is recursive, so i think that this cannot be sped up... i need to think about it.....
    for k in np.arange(1, DL):
        c[:, k] = c[:, k] / (b[:, k] - c[:, k - 1] * a[:, k])
        f[:, k] = (f[:, k] - f[:, k - 1] * a[:, k]) / (b[:, k] - c[:, k - 1] * a[:, k])

    ### Calculate the backward computation: ###
    u = torch.zeros_like(f)
    u[:, DL - 1] = f[:, DL - 1]
    for k in np.arange(DL - 2, -1, -1):
        u[:, k] = f[:, k] - c[:, k] * u[:, k + 1]

    return u


def linearInverseOperator_parallel(a, b, c, f):
    ### Using Guass Elimination To Inverse a 3-point laplacian matrix: ###
    DL = a.shape[-1]

    ### Calculate the forward computation: ###
    c[..., 0] = c[..., 0] / b[..., 0]
    f[..., 0] = f[..., 0] / b[..., 0]

    # (*). this is recursive, so i think that this cannot be sped up... i need to think about it.....
    for k in np.arange(1, DL):
        c[..., k] = c[..., k] / (b[..., k] - c[..., k - 1] * a[..., k])
        f[..., k] = (f[..., k] - f[..., k - 1] * a[..., k]) / (b[..., k] - c[..., k - 1] * a[..., k])

    ### Calculate the backward computation: ###
    u = torch.zeros_like(f)
    u[..., DL - 1] = f[..., DL - 1]
    for k in np.arange(DL - 2, -1, -1):
        u[..., k] = f[..., k] - c[..., k] * u[..., k + 1]

    return u

### Horizontal edge preserving smoothing stage: ###
def fGHS(g, w, lambda_weight, sigma_n):
    H, W = g.shape[-2:]
    # g = torch_image_flatten(    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    # H[-1,0] = 0
    # H[-1,1] = 0g)
    # w = torch_image_flatten(w)
    dg = torch.diff(g, n=1, dim=-1)
    dg = torch.exp(- dg ** 2 / torch_unsqueeze_like(sigma_n, dg, -1) ** 2)
    dg_01 = pad_with_zeros_frame_torch(dg, dim=-1, padding_size=(0,1))
    dg_10 = pad_with_zeros_frame_torch(dg, dim=-1, padding_size=(1,0))
    dga = -lambda_weight * dg_10
    dgc = -lambda_weight * dg_01
    dg = w + lambda_weight * (dg_10 + dg_01)
    u = linearInverseOperator_parallel(dga, dg, dgc, w*g)
    return u


def perform_Column_Correction_on_tensor(input_tensor, lambda_weight=50, number_of_iterations=3):
    # folder_path = '/home/mafat/PycharmProjects/QS_alg_flow_main/Experiments/12.4.2022 - natznatz experiments/3_night_500fps_20000frames_640x320/Results/seq0'
    # filename = 'temporal_average.png'
    # input_tensor = read_image_torch(os.path.join(folder_path, filename))
    # input_tensor = RGB2BW(input_tensor)[0,0]

    ### Scale image into (0,1) Range: ###
    normalization_factor = input_tensor.max() #TODO: change to max_per_image
    input_tensor_original = input_tensor / normalization_factor

    (B,T,C,H,W), shape_len, shape_vec = get_full_shape_torch(input_tensor)

    ### Loop Over Iterations: ###
    output_tensor = torch.clone(input_tensor_original)
    for iteration_index in np.arange(number_of_iterations):
        print(iteration_index)
        Dire, _ = edgeIndicator(output_tensor, 0.3)
        sigma_n = get_Sigma(output_tensor)

        ones_vec = torch.ones(1,W).to(input_tensor.device)
        ones_vec = torch_get_ND(ones_vec, input_dims='HW', number_of_dims=shape_len)

        output_tensor = fGHS(output_tensor, ones_vec, lambda_weight, 2*sigma_n)

        output_tensor, a, b = weightedLocalLinearFilter(output_tensor, input_tensor_original, Dire, np.int(np.floor(H/6)), 0.01)
    print('done column correction')

    ### Scale output back: ###
    output_tensor = output_tensor * normalization_factor
    input_tensor_original = input_tensor_original * normalization_factor
    residual_tensor = (input_tensor_original - output_tensor)

    # imshow_torch(input_tensor_original)
    # imshow_torch(output_tensor)
    # imshow_torch(residual_tensor)

    return output_tensor, residual_tensor

def perform_RCC_on_tensor(input_tensor, lambda_weight=50, number_of_iterations=3):
    input_tensor, residual_tensor_column = perform_Column_Correction_on_tensor(input_tensor, lambda_weight=lambda_weight, number_of_iterations=number_of_iterations)
    input_tensor, residual_tensor_rows = perform_Column_Correction_on_tensor(input_tensor.transpose(-1, -2), lambda_weight=lambda_weight, number_of_iterations=number_of_iterations)
    return input_tensor.transpose(-1,-2), residual_tensor_rows.transpose(-1,-2) + residual_tensor_column
# perform_RCC_on_tensor(None)











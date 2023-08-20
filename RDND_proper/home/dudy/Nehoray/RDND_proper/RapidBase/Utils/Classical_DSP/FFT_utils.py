

# from RapidBase.Utils.IO.Path_and_Reading_utils import read_image_default
import numpy as np
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import roll_n
import torch

def torch_fft2(input_image):
    return torch.fft.fftn(input_image, dim=[-1,-2])
def torch_ifft2(input_image):
    return torch.fft.ifftn(input_image, dim=[-1,-2])


def get_frequency_vec_torch(N, FPS):
    return FPS * torch.linspace(-0.5, 0.5 - 1 / N, N)

def upsample_image_numpy(original_image, upsample_factor=1):
    ### Assuming image is H=W and is even size TODO: take care of general case later
    # bla = read_image_default()[:,:,0]
    original_image_fft = np.fft.fft2(original_image)
    original_image_fft = np.fft.fftshift(original_image_fft)
    original_image_fft_padded = np.pad(original_image_fft, int(original_image.shape[0]/2))
    original_image_fft_padded = np.fft.fftshift(original_image_fft_padded)
    original_image_upsampled = np.fft.ifft2(original_image_fft_padded)
    # imshow(bla)
    # figure()
    # imshow(np.abs(original_image_upsampled))
    return original_image_upsampled


def fftshift_torch(X, first_spatial_dim=-2):
    ### FFT-Shift for all dims from dim=first_spatial_dim onwards: ###
    # batch*channel*...*2
    # real, imag = X.chunk(chunks=2, dim=-1)
    # real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    shape_len = len(X.shape)
    if first_spatial_dim < 0:
        first_spatial_dim = shape_len - abs(first_spatial_dim)

    for dim in range(first_spatial_dim, len(X.shape)):
        X = roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim] / 2)))

    # real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    # X = torch.cat((real,imag),dim=-1)
    return X

def torch_fftshift(X, first_spatial_dim=-2):
    ### FFT-Shift for all dims from dim=first_spatial_dim onwards: ###
    # batch*channel*...*2
    # real, imag = X.chunk(chunks=2, dim=-1)
    # real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    if first_spatial_dim < 0:
        first_spatial_dim = len(X.shape) - np.abs(first_spatial_dim)

    for dim in range(first_spatial_dim, len(X.shape)):
        X = roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim] / 2)))

    # real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    # X = torch.cat((real,imag),dim=-1)
    return X

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift_torch_specific_dim(X, dim):
    return roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim]/2)))

def upsample_image_torch(original_image, upsample_factor=2):
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




import os.path
from math import floor, ceil
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def estimate_noise_in_image_EMVA(input_tensor, QE=None, G_DL_to_e_EMVA=None, G_e_to_DL_EMVA=None,
                                 readout_noise_electrons=None, full_well_electrons=None, N_bits_EMVA=None,
                                 N_bits_final=None):
    ### Temp: delete: ###
    QE = 0.57
    G_DL_to_e_EMVA = 0.468
    G_e_to_DL_EMVA = 2.417
    readout_noise_electrons = 4.53
    full_well_electrons = 7948
    N_bits_EMVA = 12
    N_bits_final = 8

    ### Get Correct Gain Factors: ###
    G_e_to_DL_final = G_e_to_DL_EMVA * (2 * N_bits_EMVA / 2 * N_bits_final)
    G_DL_to_e_final = G_DL_to_e_EMVA * (2 * N_bits_final / 2 * N_bits_EMVA)

    ### Get Proper Readout Noise: ###
    readout_noise_DL_EMVA = readout_noise_electrons * G_DL_to_e_EMVA
    readout_noise_DL_final = readout_noise_electrons * G_DL_to_e_final

    ### Get Shot Noise Per Pixel: ###
    photons_per_DL_EMVA = 1 / QE * G_e_to_DL_EMVA
    photons_per_DL_final = 1 / QE * G_e_to_DL_EMVA * (2 * N_bits_EMVA / 2 * N_bits_final)
    photons_per_pixel_EMVA = input_tensor * photons_per_DL_EMVA
    photons_per_pixel_final = input_tensor * photons_per_DL_final
    photon_shot_noise_per_pixel_EMVA = torch.sqrt(photons_per_pixel_EMVA)
    photon_shot_noise_per_pixel_final = torch.sqrt(photons_per_pixel_final)
    electron_shot_noise_per_pixel_EMVA = photon_shot_noise_per_pixel_EMVA * QE
    electron_shot_noise_per_pixel_final = photon_shot_noise_per_pixel_final * QE
    DL_shot_noise_per_pixel_EMVA = electron_shot_noise_per_pixel_EMVA * G_DL_to_e_EMVA
    DL_shot_noise_per_pixel_final = electron_shot_noise_per_pixel_final * G_DL_to_e_final

    ### Get Final Noise Levels Per DL: ###
    total_noise_per_pixel_EMVA = DL_shot_noise_per_pixel_EMVA + readout_noise_DL_EMVA
    total_noise_per_pixel_final = DL_shot_noise_per_pixel_final + readout_noise_DL_final

    return total_noise_per_pixel_final

def read_bin_images_batch(filenames, height, width):
    dtype = np.dtype('B')
    video_arr = np.expand_dims(np.zeros((height, width)), axis=0)
    for bin_path in filenames:
        print(f'reading image {os.path.basename(bin_path)}')
        with open(bin_path, 'rb') as f:
            numpy_data = np.fromfile(f, dtype)
        image_arr = np.reshape(numpy_data, (1, height, width))
        video_arr = np.concatenate((video_arr, image_arr))
    video_tensor = torch.from_numpy(video_arr[1:])
    return video_tensor


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


def read_bin_video_to_numpy(bin_path, H, W, C, dtype=np.float32):
    with open(bin_path, 'rb') as f:
        numpy_data = np.fromfile(f, dtype=dtype)
    video_arr = np.reshape(numpy_data, (-1, H, W, C))
    return video_arr


# Stretch array to see contrast clearly
# bla_stretched = scale_array_stretch_hist(bla, (0.1, 0.99))

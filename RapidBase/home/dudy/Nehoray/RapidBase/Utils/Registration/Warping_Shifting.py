


import torch.fft
import torch.nn as nn
import numpy as np
from RapidBase.Utils.Classical_DSP.FFT_utils import fftshift_torch_specific_dim, fftshift_torch, torch_fft2, torch_fftshift, torch_ifft2

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

# ### Test: ###
# original_image = read_image_stack_default()
# original_image = original_image[:,:,:,0]
# original_image_torch = torch.Tensor(original_image).unsqueeze(1) #to the form of [B,C,H,W]
# shiftx = (10,20,30)
# shifty = (30,20,10)
# layer = Shift_Layer_Torch()
# output_tensor = layer.forward(original_image_torch, shiftx, shifty)
# imshow_torch(original_image_torch[1])
# imshow_torch(output_tensor[1])


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
    kx = kx.to(original_image_torch.device)
    ky = ky.to(original_image_torch.device)

    ### Expand shiftx & shifty to match input image shape (and if shifts are more thann a scalar i assume you want to multiply B or C): ###
    if type(shiftx) != list and type(shiftx) != tuple and type(shiftx) != np.ndarray and type(shiftx) != torch.Tensor:
        shiftx = [shiftx]
        shifty = [shifty]
    if ndims == 3:
        if type(shiftx) != torch.Tensor:
            shiftx = torch.Tensor(shiftx).to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1)
            shifty = torch.Tensor(shifty).to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1)
        else:
            shiftx = shiftx.to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1)
            shifty = shifty.to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1)
        shiftx = shiftx.to(original_image_torch.device)
        shifty = shifty.to(original_image_torch.device)
    elif ndims == 4:
        if type(shiftx) != torch.Tensor:
            shiftx = torch.Tensor(shiftx).to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shifty = torch.Tensor(shifty).to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            shiftx = shiftx.to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shifty = shifty.to(original_image_torch.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shiftx = shiftx.to(original_image_torch.device)
        shifty = shifty.to(original_image_torch.device)

    ### Displace input image: ###
    displacement_matrix = torch.exp(-(1j * 2 * np.pi * ky * shifty + 1j * 2 * np.pi * kx * shiftx)).to(original_image_torch.device)
    fft_image = torch.fft.fftn(original_image_torch, dim=[-1,-2])
    fft_image_displaced = fft_image * displacement_matrix
    # original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1,-2]).real
    original_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1,-2]).abs()

    return original_image_displaced


def shift_matrix_subpixel(original_image, shiftx, shifty):

    H,W,C = original_image.shape[-3:]
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
    # get displacement matrix:
    displacement_matrix = np.exp(-(1j * 2 * np.pi * ky * shifty + 1j * 2 * np.pi * kx * shiftx))

    # displacement_matrix = np.atleast_3d(displacement_matrix)
    # original_image = np.atleast_3d(original_image)
    displacement_matrix = displacement_matrix.squeeze()
    original_image = original_image.squeeze()

    # get displaced speckles:
    fft_image = np.fft.fft2(original_image)
    fft_image_displaced = fft_image * displacement_matrix  # the shift or from one image to another, not for the original phase screen
    original_image_displaced = np.fft.ifft2(fft_image_displaced).real

    original_image_displaced = np.atleast_3d(original_image_displaced)
    return original_image_displaced



def blur_image_motion_blur(original_image, shiftx, shifty, N):
    shiftx_one_iteration = shiftx/N
    shifty_one_iteration = shifty/N
    blurred_image = np.zeros_like(original_image)
    for i in np.arange(N):
        blurred_image += 1/N*shift_matrix_subpixel(original_image, shiftx_one_iteration*(i+1), shifty_one_iteration*(i+1))
    return blurred_image

def blur_image_motion_blur_torch(original_image, shiftx, shifty, N, warp_object):
    shiftx_one_iteration = shiftx/N
    shifty_one_iteration = shifty/N
    blurred_image = torch.zeros_like(original_image).to(original_image.device)
    for i in np.arange(N):
        blurred_image += 1/N*warp_object.forward(original_image, shiftx_one_iteration*i, shifty_one_iteration*i)
        # blurred_image += 1/N*shift_matrix_subpixel(original_image, shiftx_one_iteration*i, shifty_one_iteration*i)
    return blurred_image

def blur_image_motion_blur_affine_torch(original_image, shiftx, shifty, scale_factor, rotation_angle, N, warp_object):
    shiftx_one_iteration = shiftx / N
    shifty_one_iteration = shifty / N
    scale_factor_one_iteration = scale_factor / N
    rotation_angle_one_iteration = rotation_angle / N
    blurred_image = torch.zeros_like(original_image)
    for i in np.arange(N):
        blurred_image += 1 / N * warp_object.forward(original_image,
                                                     shiftx_one_iteration * i, shifty_one_iteration * i,
                                                     scale_factor_one_iteration, rotation_angle_one_iteration)
    return blurred_image


from RapidBase.Utils.MISCELENEOUS import get_random_number_in_range
def blur_batch_numpy(original_image_batch, blur_size, N):
    for i in np.arange(original_image_batch.shape[0]):
        shiftx = get_random_number_in_range(-blur_size, blur_size)
        shifty = get_random_number_in_range(-blur_size, blur_size)
        original_image_batch[i,:,:,:] = blur_image_motion_blur(original_image_batch[i,:,:,:], shiftx, shifty, N)
    return original_image_batch

def blur_batch_torch(original_image_batch, blur_size, N, warp_object, device='cpu'):
    shiftx = torch.Tensor(get_random_number_in_range(-blur_size, blur_size, (original_image_batch.shape[0]))).to(device)
    shifty = torch.Tensor(get_random_number_in_range(-blur_size, blur_size, (original_image_batch.shape[0]))).to(device)
    shiftx = shiftx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    shifty = shifty.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    ones_tensor = torch.ones_like(original_image_batch).to(device)
    shiftx = ones_tensor * shiftx
    shifty = ones_tensor * shifty
    # original_image_batch = warp_object.forward(original_image_batch, shiftx, shifty, 'bilinear')
    original_image_batch = blur_image_motion_blur_torch(original_image_batch, shiftx, shifty, N, warp_object)
    return original_image_batch

def shift_image_integer_pixels(original_image, delta_x, delta_y):
    shifted_image = np.roll(original_image, delta_y, 0)
    shifted_image = np.roll(shifted_image, delta_x, 1)
    return shifted_image

def shift_image_integer_pixels_torch(original_image, delta_x, delta_y):
    shifted_image = torch.roll(original_image, [delta_x, delta_y], dims=[-2,-1])
    return shifted_image



class FFT_Rotate_Layer(nn.Module):
    def __init__(self, *args):
        super(FFT_Rotate_Layer, self).__init__()
        self.padded_mat = None

    def forward(self, input_tensors, thetas):
        # #################################################################
        # ### Just For Test: ###
        # input_tensors = read_image_default_torch()
        # input_tensors = crop_torch_batch(input_tensors, (256,256))
        # thetas = np.array([5,10,15])
        # thetas = thetas * np.pi/180
        # #################################################################

        #################################################################
        ### Pytorch: ###
        ### Gte Matrix Shape: ###
        B,T,H,W = input_tensors.shape

        ### Embed thetas vec in a(1, 1, T) matrix for later multiplication: ###
        if type(thetas) == list or type(thetas) == tuple:
            thetas_final = torch.zeros(B,len(thetas),1,1).to(input_tensors.device)
            thetas_final[:,:,0,0] = torch.Tensor(thetas)
            thetas = thetas_final
        elif type(thetas) == torch.Tensor:
            thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
            thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
            thetas = thetas_final
        elif type(thetas) == np.ndarray:
            thetas = torch.Tensor(thetas)
            thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
            thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
            thetas = thetas_final

        ### Choose Rotation Angle and assign variables needed for later: % % %
        a = torch.tan(thetas / 2)
        b = -torch.sin(thetas)
        # % a = (scale * cos(theta) - 1) / (-scale * sin(theta));
        # % b = -scale * sin(theta);

        ### Embedd Image Into Larger Matrix Able To Embedd Largest Rotation of 45 degrees. cant i handle M*sqrt(2),N*sqrt(2)???: ###
        self.padded_mat[:, :, H // 2:H // 2 + H, W // 2:W // 2 + W] = input_tensors

        ### Initialize Internal Variables: ###
        if self.padded_mat is None:
            self.padded_mat = torch.zeros(B, T, 2*H, 2*W).to(input_tensors.device)
            self.B, self.T, self.H, self.W = self.padded_mat.shape

            ### Build Needed Vectors: ###
            self.Nx_vec = np.arange(-np.fix(self.W / 2), np.ceil(self.W / 2))
            self.Ny_vec = np.arange(-np.fix(self.H / 2), np.ceil(self.H / 2))
            self.Nx_vec = torch.Tensor(self.Nx_vec).to(input_tensors.device)
            self.Ny_vec = torch.Tensor(self.Ny_vec).to(input_tensors.device)
            self.Nx_vec = fftshift_torch_specific_dim(self.Nx_vec, 0)
            self.Ny_vec = fftshift_torch_specific_dim(self.Ny_vec, 0)
            self.Nx_length = len(self.Nx_vec)
            self.Ny_length = len(self.Ny_vec)
            self.Nx_vec = self.Nx_vec.unsqueeze(0)
            self.Ny_vec = self.Ny_vec.unsqueeze(1)
            self.Nx_vec = self.Nx_vec.unsqueeze(0).unsqueeze(0)
            self.Ny_vec = self.Ny_vec.unsqueeze(0).unsqueeze(0)

            ### Initialize Matrices to contain results after column / row shifting: ###
            self.Ix = torch.zeros((self.B, self.T, self.Nx_length, self.Ny_length)).to(input_tensors.device)
            self.Iy = torch.zeros((self.B, self.T, self.Nx_length, self.Ny_length)).to(input_tensors.device)
            self.I_final = torch.zeros((self.B, self.T, self.Nx_length, self.Ny_length)).to(input_tensors.device)

            ### Prepare Matrices to avoid looping: ###
            self.Nx_vec_mat = torch.repeat_interleave(self.Nx_vec, self.H, -2)
            self.Ny_vec_mat = torch.repeat_interleave(self.Ny_vec, self.W, -1)
            self.column_mat = torch.zeros((self.B, self.T, self.Ny_length, self.Nx_length)).to(input_tensors.device)
            self.row_mat = torch.zeros((self.B, self.T, self.Ny_length, self.Nx_length)).to(input_tensors.device)
            for k in np.arange(self.H):
                self.column_mat[:,:,k,:] = k - np.floor(self.H/2)
            for k in np.arange(self.W):
                self.row_mat[:,:,:,k] = k - np.floor(self.W/2)
            self.Ny_vec_mat_final = self.Ny_vec_mat * self.row_mat * (-2 * 1j * np.pi) / self.W
            self.Nx_vec_mat_final = self.Nx_vec_mat * self.column_mat * (-2 * 1j * np.pi) / self.H

        ### Use Parallel Computing instead of looping: ###
        Ix_parallel = (torch.fft.ifftn(torch.fft.fftn(self.padded_mat, dim=[-1]) * torch.exp(self.Nx_vec_mat_final * a), dim=[-1])).real
        Iy_parallel = (torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(self.Ny_vec_mat_final * b * -1).conj(), dim=[-2])).real
        input_mat_rotated = (torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(self.Nx_vec_mat_final * a), dim=[-1])).real
        # imshow_torch(input_mat_rotated[0,-1])

        return input_mat_rotated
        #################################################################


class FFT_Translation_Rotation_Layer(nn.Module):
    def __init__(self, *args):
        super(FFT_Translation_Rotation_Layer, self).__init__()
        self.padded_mat = None
        self.kx = None
        self.ky = None

    def forward(self, input_tensors, shifts_x, shifts_y, thetas):
        # ##################################################################################################################################
        # ### Just For Test: ###
        # input_tensors = read_image_default_torch()
        # input_tensors = crop_torch_batch(input_tensors, (256,256))
        # thetas = np.array([5,10,15])
        # thetas = thetas * np.pi/180
        # ##################################################################################################################################

        ##################################################################################################################################
        ### Pytorch: ###
        ### Get Matrix Shape: ###
        B,T,H,W = input_tensors.shape
        H_original = H
        W_original = W
        self.ndims = len(input_tensors.shape)

        ### Embed thetas vec in a(1, 1, T) matrix for later multiplication: ###
        if type(thetas) == list or type(thetas) == tuple:
            thetas_final = torch.zeros(B,len(thetas),1,1).to(input_tensors.device)
            thetas_final[:,:,0,0] = torch.Tensor(thetas)
            thetas = thetas_final
        elif type(thetas) == torch.Tensor:
            thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
            thetas_final[:, :, 0, 0] = thetas
            thetas = thetas_final
        elif type(thetas) == np.ndarray:
            thetas = torch.Tensor(thetas)
            thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
            thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
            thetas = thetas_final
        ### Expand shiftx & shifty to match input image shape (and if shifts are more than a scalar i assume you want to multiply B or C): ###
        if type(shifts_x) != list and type(shifts_x) != tuple and type(shifts_x) != np.ndarray and type(shifts_x) != torch.Tensor:
            shifts_x = [shifts_x]
            shifts_y = [shifts_y]
        if self.ndims == 3:
            if type(shifts_x) != torch.Tensor:
                shifts_x = torch.Tensor(shifts_x).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
                shifty = torch.Tensor(shifts_y).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
            else:
                shifts_x = shifts_x.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
                shifts_y = shifts_y.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
            shifts_x = shifts_x.to(input_tensors.device)
            shifts_y = shifts_y.to(input_tensors.device)
        elif self.ndims == 4:
            if type(shifts_x) != torch.Tensor:
                shifts_x = torch.Tensor(shifts_x).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                shifts_y = torch.Tensor(shifts_y).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                #TODO: use the to_3d, to_4d functions!!!!
                shifts_x = shifts_x.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
                shifts_y = shifts_y.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
            shifts_x = shifts_x.to(input_tensors.device)
            shifts_y = shifts_y.to(input_tensors.device)
        ##################################################################################################################################


        ##################################################################################################################################
        ### Check if we need to create a new kvec: ###
        if self.kx is None:
            flag_create_new_kvec = True
        else:
            if self.kx.shape[-1] != input_tensors.shape[-1]*2 or self.kx.shape[-2] != input_tensors.shape[-2]*2:
                flag_create_new_kvec = True
            else:
                flag_create_new_kvec = False

        ### Initialize Internal Variables: ###
        if flag_create_new_kvec:
            self.padded_mat = torch.zeros(B, T, 2*H, 2*W).to(input_tensors.device)
            self.B, self.T, self.H, self.W = self.padded_mat.shape

            ### Build Needed Vectors: ###
            self.Nx_vec = np.arange(-np.fix(self.W / 2), np.ceil(self.W / 2))
            self.Ny_vec = np.arange(-np.fix(self.H / 2), np.ceil(self.H / 2))
            self.Nx_vec = torch.Tensor(self.Nx_vec).to(input_tensors.device)
            self.Ny_vec = torch.Tensor(self.Ny_vec).to(input_tensors.device)
            self.Nx_vec = fftshift_torch_specific_dim(self.Nx_vec, 0)
            self.Ny_vec = fftshift_torch_specific_dim(self.Ny_vec, 0)
            self.Nx_length = len(self.Nx_vec)
            self.Ny_length = len(self.Ny_vec)
            self.Nx_vec = self.Nx_vec.unsqueeze(0)
            self.Ny_vec = self.Ny_vec.unsqueeze(1)
            self.Nx_vec = self.Nx_vec.unsqueeze(0).unsqueeze(0)
            self.Ny_vec = self.Ny_vec.unsqueeze(0).unsqueeze(0)

            ### Prepare Matrices to avoid looping: ###
            self.Nx_vec_mat = torch.repeat_interleave(self.Nx_vec, self.H, -2)
            self.Ny_vec_mat = torch.repeat_interleave(self.Ny_vec, self.W, -1)
            self.column_mat = torch.zeros((self.B, self.T, self.Ny_length, self.Nx_length)).to(input_tensors.device)
            self.row_mat = torch.zeros((self.B, self.T, self.Ny_length, self.Nx_length)).to(input_tensors.device)
            for k in np.arange(self.H):
                self.column_mat[:,:,k,:] = k - np.floor(self.H/2)
            for k in np.arange(self.W):
                self.row_mat[:,:,:,k] = k - np.floor(self.W/2)
            self.Ny_vec_mat_final = self.Ny_vec_mat * self.row_mat * (-2 * 1j * np.pi) / self.W
            self.Nx_vec_mat_final = self.Nx_vec_mat * self.column_mat * (-2 * 1j * np.pi) / self.H

            ### Translation Variables: ### #TODO: i assume the translation is done after the rotation, and so on the padded and rotated image, which is larger then original!!!!
            ### Get tilt phases k-space: ###
            x = np.arange(-np.fix(self.W / 2), np.ceil(self.W / 2), 1)
            y = np.arange(-np.fix(self.H / 2), np.ceil(self.H / 2), 1)
            delta_f1 = 1 / self.W
            delta_f2 = 1 / self.H
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
            self.kx = self.kx.to(input_tensors.device)
            self.ky = self.ky.to(input_tensors.device)
        ##################################################################################################################################


        ##################################################################################################################################
        ### Choose Rotation Angle and assign variables needed for later: % % %
        a = torch.tan(thetas / 2)
        b = -torch.sin(thetas)
        # % a = (scale * cos(theta) - 1) / (-scale * sin(theta));
        # % b = -scale * sin(theta);

        ### Embedd Image Into Larger Matrix Able To Embedd Largest Rotation of 45 degrees. cant i handle M*sqrt(2),N*sqrt(2)???: ###
        self.padded_mat[:, :, H_original // 2:H_original // 2 + H_original, W_original // 2:W_original // 2 + W_original] = input_tensors

        ### Use Parallel Computing instead of looping: ###
        Ix_parallel = (torch.fft.ifftn(torch.fft.fftn(self.padded_mat, dim=[-1]) * torch.exp(self.Nx_vec_mat_final * a), dim=[-1])).real
        Iy_parallel = (torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(self.Ny_vec_mat_final * b * -1).conj(), dim=[-2])).real
        input_mat_rotated = (torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(self.Nx_vec_mat_final * a), dim=[-1])).real
        # imshow_torch(input_mat_rotated[0,-1])

        ### Displace: ###
        displacement_matrix = torch.exp(-(1j * 2 * np.pi * self.ky * shifts_y + 1j * 2 * np.pi * self.kx * shifts_x))
        fft_image = torch.fft.fftn(input_mat_rotated, dim=[-1, -2])
        fft_image_displaced = fft_image * displacement_matrix
        # input_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()
        input_tensors_rotated_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).real
        ##################################################################################################################################

        return input_tensors_rotated_displaced

def FFT_Rotation_Translation_Function(input_tensors, thetas, shifts_x, shifts_y):
    # ##################################################################################################################################
    # ### Just For Test: ###
    # input_tensors = read_image_default_torch()
    # input_tensors = crop_torch_batch(input_tensors, (256,256))
    # thetas = np.array([5,10,15])
    # thetas = thetas * np.pi/180
    # shifts_x = np.array([-10,0,10])
    # shifts_y = np.array([-10,0,10])
    # ##################################################################################################################################

    ##################################################################################################################################
    ### Pytorch: ###
    ### Get Matrix Shape: ###
    B, T, H, W = input_tensors.shape
    H_original = H
    W_original = W
    ndims = len(input_tensors.shape)  #TODO: right now i assume i only get B,T,H,W

    ### Embed thetas vec in a(1, 1, T) matrix for later multiplication: ###
    if type(thetas) == list or type(thetas) == tuple:
        thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
        thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
        thetas = thetas_final
    elif type(thetas) == torch.Tensor:
        thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
        thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
        thetas = thetas_final
    elif type(thetas) == np.ndarray:
        thetas = torch.Tensor(thetas)
        thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
        thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
        thetas = thetas_final
    ### Expand shiftx & shifty to match input image shape (and if shifts are more than a scalar i assume you want to multiply B or C): ###
    if type(shifts_x) != list and type(shifts_x) != tuple and type(shifts_x) != np.ndarray and type(
            shifts_x) != torch.Tensor:
        shifts_x = [shifts_x]
        shifts_y = [shifts_y]
    if ndims == 3:
        if type(shifts_x) != torch.Tensor:
            shifts_x = torch.Tensor(shifts_x).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
            shifts_y = torch.Tensor(shifts_y).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
        else:
            shifts_x = shifts_x.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
            shifts_y = shifts_y.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1)
        shifts_x = shifts_x.to(input_tensors.device)
        shifts_y = shifts_y.to(input_tensors.device)
    elif ndims == 4:
        if type(shifts_x) != torch.Tensor:
            shifts_x = torch.Tensor(shifts_x).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shifts_y = torch.Tensor(shifts_y).to(input_tensors.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            shifts_x = shifts_x.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shifts_y = shifts_y.to(input_tensors.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        shifts_x = shifts_x.to(input_tensors.device)
        shifts_y = shifts_y.to(input_tensors.device)
    ##################################################################################################################################

    ##################################################################################################################################
    ### Initialize Internal Variables: ###
    padded_mat = torch.zeros(B, T, 2 * H, 2 * W).to(input_tensors.device)
    B, T, H, W = padded_mat.shape

    ### Build Needed Vectors: ###
    Nx_vec = np.arange(-np.fix(W / 2), np.ceil(W / 2))
    Ny_vec = np.arange(-np.fix(H / 2), np.ceil(H / 2))
    Nx_vec = torch.Tensor(Nx_vec).to(input_tensors.device)
    Ny_vec = torch.Tensor(Ny_vec).to(input_tensors.device)
    Nx_vec = fftshift_torch_specific_dim(Nx_vec, 0)
    Ny_vec = fftshift_torch_specific_dim(Ny_vec, 0)
    Nx_length = len(Nx_vec)
    Ny_length = len(Ny_vec)
    Nx_vec = Nx_vec.unsqueeze(0)
    Ny_vec = Ny_vec.unsqueeze(1)
    Nx_vec = Nx_vec.unsqueeze(0).unsqueeze(0)
    Ny_vec = Ny_vec.unsqueeze(0).unsqueeze(0)

    ### Prepare Matrices to avoid looping: ###
    Nx_vec_mat = torch.repeat_interleave(Nx_vec, H, -2)
    Ny_vec_mat = torch.repeat_interleave(Ny_vec, W, -1)
    column_mat = torch.zeros((B, T, Ny_length, Nx_length)).to(input_tensors.device)
    row_mat = torch.zeros((B, T, Ny_length, Nx_length)).to(input_tensors.device)
    for k in np.arange(H):
        column_mat[:, :, k, :] = k - np.floor(H / 2)
    for k in np.arange(W):
        row_mat[:, :, :, k] = k - np.floor(W / 2)
    Ny_vec_mat_final = Ny_vec_mat * row_mat * (-2 * 1j * np.pi) / W
    Nx_vec_mat_final = Nx_vec_mat * column_mat * (-2 * 1j * np.pi) / H

    ### Translation Variables: ### #TODO: i assume the translation is done after the rotation, and so on the padded and rotated image, which is larger then original!!!!
    ### Get tilt phases k-space: ###
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
    # Expand to original image dimensions:
    if ndims == 3:
        kx = kx.unsqueeze(0)
        ky = ky.unsqueeze(0)
    if ndims == 4:
        kx = kx.unsqueeze(0).unsqueeze(0)
        ky = ky.unsqueeze(0).unsqueeze(0)
    # K-vecs to tensor device:
    kx = kx.to(input_tensors.device)
    ky = ky.to(input_tensors.device)
    ##################################################################################################################################

    ##################################################################################################################################
    ### Choose Rotation Angle and assign variables needed for later: % % %
    a = torch.tan(thetas / 2)
    b = -torch.sin(thetas)
    # % a = (scale * cos(theta) - 1) / (-scale * sin(theta));
    # % b = -scale * sin(theta);

    ### Embedd Image Into Larger Matrix Able To Embedd Largest Rotation of 45 degrees. cant i handle M*sqrt(2),N*sqrt(2)???: ###
    padded_mat[:, :, H_original // 2:H_original // 2 + H_original, W_original // 2:W_original // 2 + W_original] = input_tensors

    ### Use Parallel Computing instead of looping: ###
    Ix_parallel = (torch.fft.ifftn(torch.fft.fftn(padded_mat, dim=[-1]) * torch.exp(Nx_vec_mat_final * a),dim=[-1])).real
    Iy_parallel = (torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(Ny_vec_mat_final * b * -1).conj(),dim=[-2])).real
    input_mat_rotated = (torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    # imshow_torch(input_mat_rotated[0,-1])

    ### Displace: ###
    displacement_matrix = torch.exp(-(1j * 2 * np.pi * ky * shifts_y + 1j * 2 * np.pi * kx * shifts_x))
    fft_image = torch.fft.fftn(input_mat_rotated, dim=[-1, -2])
    fft_image_displaced = fft_image * displacement_matrix
    # input_image_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).abs()
    input_tensors_rotated_displaced = torch.fft.ifftn(fft_image_displaced, dim=[-1, -2]).real
    # imshow_torch(input_tensors_rotated_displaced[0,-1])
    ##################################################################################################################################

    return input_tensors_rotated_displaced

def FFT_Rotate_Function(input_tensors, thetas):
        # #################################################################
        # ### Just For Test: ###
        # input_tensors = read_image_default_torch()
        # input_tensors = crop_torch_batch(input_tensors, (256,256))
        # thetas = np.array([5,10,15])
        # thetas = thetas * np.pi/180
        # #################################################################

        #################################################################
        ### Pytorch: ###
        ### Gte Matrix Shape: ###
        B, T, H, W = input_tensors.shape

        ### Embed thetas vec in a(1, 1, T) matrix for later multiplication: ###
        if type(thetas) == list or type(thetas) == tuple:
            thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
            thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
            thetas = thetas_final
        elif type(thetas) == torch.Tensor:
            thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
            thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
            thetas = thetas_final
        elif type(thetas) == np.ndarray:
            thetas = torch.Tensor(thetas)
            thetas_final = torch.zeros(B, len(thetas), 1, 1).to(input_tensors.device)
            thetas_final[:, :, 0, 0] = torch.Tensor(thetas)
            thetas = thetas_final

        ### Embedd Image Into Larger Matrix Able To Embedd Largest Rotation of 45 degrees. cant i handle M*sqrt(2),N*sqrt(2)???: ###
        temp = torch.zeros(B, T, 2 * H, 2 * W).to(input_tensors.device)
        temp[:, :, H // 2:H // 2 + H, W // 2:W // 2 + W] = input_tensors
        input_mat = temp

        ### Choose Rotation Angle and assign variables needed for later: % % %
        B, T, H, W = input_mat.shape
        a = torch.tan(thetas / 2)
        b = -torch.sin(thetas)
        # % a = (scale * cos(theta) - 1) / (-scale * sin(theta));
        # % b = -scale * sin(theta);

        ### Build Needed Vectors: ###
        Nx_vec = np.arange(-np.fix(W / 2), np.ceil(W / 2))
        Ny_vec = np.arange(-np.fix(H / 2), np.ceil(H / 2))
        Nx_vec = torch.Tensor(Nx_vec).to(input_tensors.device)
        Ny_vec = torch.Tensor(Ny_vec).to(input_tensors.device)
        Nx_vec = fftshift_torch_specific_dim(Nx_vec, 0)
        Ny_vec = fftshift_torch_specific_dim(Ny_vec, 0)
        Nx_length = len(Nx_vec)
        Ny_length = len(Ny_vec)
        Nx_vec = Nx_vec.unsqueeze(0)
        Ny_vec = Ny_vec.unsqueeze(1)
        Nx_vec = Nx_vec.unsqueeze(0).unsqueeze(0)
        Ny_vec = Ny_vec.unsqueeze(0).unsqueeze(0)

        ### Initialize Matrices to contain results after column / row shifting: ###
        Ix = torch.zeros((B, T, Nx_length, Ny_length)).to(input_tensors.device)
        Iy = torch.zeros((B, T, Nx_length, Ny_length)).to(input_tensors.device)
        I_final = torch.zeros((B, T, Nx_length, Ny_length)).to(input_tensors.device)

        ### Prepare Matrices to avoid looping: ###
        Nx_vec_mat = torch.repeat_interleave(Nx_vec, H, -2)
        Ny_vec_mat = torch.repeat_interleave(Ny_vec, W, -1)
        column_mat = torch.zeros((B, T, Ny_length, Nx_length)).to(input_tensors.device)
        row_mat = torch.zeros((B, T, Ny_length, Nx_length)).to(input_tensors.device)
        for k in np.arange(H):
            column_mat[:, :, k, :] = k - np.floor(H / 2)
        for k in np.arange(W):
            row_mat[:, :, :, k] = k - np.floor(W / 2)
        Ny_vec_mat_final = Ny_vec_mat * row_mat * (-2 * 1j * np.pi) / W
        Nx_vec_mat_final = Nx_vec_mat * column_mat * (-2 * 1j * np.pi) / H

        ### Use Parallel Computing instead of looping: ###
        Ix_parallel = (torch.fft.ifftn(torch.fft.fftn(input_mat, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
        Iy_parallel = (torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(Ny_vec_mat_final * b * -1).conj(), dim=[-2])).real
        input_mat_rotated = (torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
        # imshow_torch(input_mat_rotated[0,-1])

        return input_mat_rotated
        #################################################################


### TODO: check rotation layer and rotation-translation layer with different dimensions on each side!!!: ###






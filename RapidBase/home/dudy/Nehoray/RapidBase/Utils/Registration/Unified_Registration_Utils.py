try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

from numpy.fft import fft2

# from RapidBase.Utils.Tensor_Manipulation.Pytorch_Numpy_Utils import get_shape

from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *

#TODO: delete
from RapidBase.import_all import *

######################################################################################################################################################################################
### Utils/Auxiliary: ###
def local_sum(input_tensor, outer_frame_to_disregard=5):
    T,H,W = input_tensor.shape
    start_index_rows = outer_frame_to_disregard
    stop_index_rows = H - outer_frame_to_disregard
    start_index_cols = outer_frame_to_disregard
    stop_index_cols = W - outer_frame_to_disregard
    return input_tensor[:, start_index_rows:stop_index_rows, start_index_cols:stop_index_cols].sum(-1).sum(-1)



# from RapidBase.import_all import *
def create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels=10, N=512, polarization=0, flag_gauss_circ=1, flag_normalize=1, flag_imshow=0):
    #    import numpy
    #    from numpy import arange
    #    from numpy.random import *

    # #Parameters:
    # speckle_size_in_pixels = 10
    # N = 512
    # polarization = 0
    # flag_gauss_circ = 0

    # Calculations:
    wf = (N / speckle_size_in_pixels)

    if flag_gauss_circ == 1:
        x = np.arange(-N / 2, N / 2, 1)
        distance_between_the_two_beams_x = 0
        distance_between_the_two_beams_y = 0
        [X, Y] = np.meshgrid(x, x)
        beam_one = np.exp(- ((X - distance_between_the_two_beams_x / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        beam_two = np.exp(- ((X + distance_between_the_two_beams_x / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2) / wf ** 2)
        total_beam = beam_one + beam_two
        total_beam = total_beam / np.sqrt(sum(sum(abs(total_beam) ** 2)))
    else:
        x = np.arange(-N / 2, N / 2, 1) * 1
        y = x
        [X, Y] = np.meshgrid(x, y)
        c = 0
        distance_between_the_two_beams_y = 0
        beam_one = ((X - distance_between_the_two_beams_y / 2) ** 2 + (Y - distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        beam_two = ((X + distance_between_the_two_beams_y / 2) ** 2 + (Y + distance_between_the_two_beams_y / 2) ** 2 <= wf ** 2)
        total_beam = beam_one + beam_two

    # Polarization:
    if (polarization > 0 & polarization < 1):
        beam_one = total_beam * np.exp(2 * np.pi * 1j * 10 * np.random.randn(N, N))
        beam_two = total_beam * np.exp(2 * np.pi * 1j * 10 * np.random.randn(N, N))
        speckle_pattern1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beam_one)))
        speckle_pattern2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(beam_two)))
        speckle_pattern_total_intensity = (1 - polarization) * abs(speckle_pattern1) ** 2 + polarization * abs(speckle_pattern2) ** 2
    else:
        total_beam = total_beam * np.exp(2 * np.pi * 1j * (10 * np.random.randn(N, N)))
        speckle_pattern1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(total_beam)))
        speckle_pattern2 = np.empty_like(speckle_pattern1)
        speckle_pattern_total_intensity = np.abs(speckle_pattern1) ** 2

    # if flag_normalize == 1: bla = bla-bla.min() bla=bla/bla.max()
    # if flag_imshow == 1: imshow(speckle_pattern_total_intensity) colorbar()

    return speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2



######################################################################################################################################################################################
#
#
#
#
#
######################################################################################################################################################################################
### Translation/Shifting Warping: ###
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
                B, C, H, W = input_image.shape
            elif self.ndims == 3:
                C, H, W = input_image.shape
                B = 1
            elif self.ndims == 2:
                H, W = input_image.shape
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
        # TODO: enable accepting pytorch tensors/vectors etc'
        if type(shiftx) != list and type(shiftx) != tuple and type(shiftx) != np.ndarray and type(
                shiftx) != torch.Tensor:
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

def blur_image_motion_blur_torch(original_image, shiftx, shifty, N, warp_object):
    shiftx_one_iteration = shiftx/N
    shifty_one_iteration = shifty/N
    blurred_image = torch.zeros_like(original_image).to(original_image.device)
    for i in np.arange(N):
        blurred_image += 1/N*warp_object.forward(original_image, shiftx_one_iteration*i, shifty_one_iteration*i)
        # blurred_image += 1/N*shift_matrix_subpixel(original_image, shiftx_one_iteration*i, shifty_one_iteration*i)
    return blurred_image

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

def shift_image_integer_pixels_torch(original_image, delta_x, delta_y):
    shifted_image = torch.roll(original_image, [delta_x, delta_y], dims=[-2,-1])
    return shifted_image


### TODO: temp test, to be commented out: ###
def test_single_BW_image_shift():
    input_tensor = read_image_default_torch()
    input_tensor = RGB2BW(input_tensor)
    translation_HW = (20, 80)

    #(1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0,0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    translation_matrix = np.float32([ [1,0,translation_HW[1]], [0,1,translation_HW[0]] ])
    input_tensor_numpy_translated = cv2.warpAffine(input_tensor_numpy, translation_matrix, (num_cols, num_rows))
    #(2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_HW[1], translation_HW[0])
    #(3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_HW[1], translation_HW[0])

    # Compare Results: #
    figure()
    imshow(input_tensor_numpy_translated)
    figure()
    imshow_torch(input_tensor_translated)
    figure()
    imshow_torch(input_tensor_translated - torch.Tensor(input_tensor_numpy_translated).unsqueeze(0).unsqueeze(0))
    figure()
    imshow_torch(input_tensor_translated - input_tensor_translated_layer)


def test_single_RGB_image_shift():
    input_tensor = read_image_default_torch()
    translation_HW = (20, 80)

    # (1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    translation_matrix = np.float32([[1, 0, translation_HW[1]], [0, 1, translation_HW[0]]])
    input_tensor_numpy_translated = cv2.warpAffine((input_tensor_numpy*255).astype(uint8), translation_matrix, (num_cols, num_rows))
    # (2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_HW[1], translation_HW[0])
    # (3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_HW[1], translation_HW[0])

    # Compare Results: #
    figure()
    imshow(input_tensor_numpy_translated)
    figure()
    imshow_torch(input_tensor_translated)
    figure()
    imshow_torch(input_tensor_translated - torch.Tensor(input_tensor_numpy_translated).unsqueeze(0).unsqueeze(0))
    figure()
    imshow_torch(input_tensor_translated - input_tensor_translated_layer)

def test_Multiple_BW_images_single_shift():
    input_tensor = read_image_stack_default_torch()
    translation_H = [20]
    translation_W = [80]

    # (1). OpenCV implementation:
    input_tensor_numpy = (input_tensor.cpu().permute([2,3,0,1]).numpy()*255).astype(np.uint8)
    input_tensor_1_numpy = input_tensor_numpy[:,:,0,:]
    input_tensor_2_numpy = input_tensor_numpy[:,:,1,:]
    input_tensor_3_numpy = input_tensor_numpy[:,:,2,:]
    num_rows, num_cols = input_tensor_numpy.shape[0:2]
    translation_matrix = np.float32([[1, 0, translation_W[0]], [0, 1, translation_H[0]]])
    input_tensor_1_numpy_translated = cv2.warpAffine(input_tensor_1_numpy, translation_matrix, (num_cols, num_rows))
    input_tensor_2_numpy_translated = cv2.warpAffine(input_tensor_2_numpy, translation_matrix, (num_cols, num_rows))
    input_tensor_3_numpy_translated = cv2.warpAffine(input_tensor_3_numpy, translation_matrix, (num_cols, num_rows))
    # (2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_W, translation_H)
    # (3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_W, translation_H)

    # Compare Results: #
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated / 255).unsqueeze(0).permute([0, 3, 1, 2]))
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[0])
    imshow_torch(torch.Tensor(input_tensor_2_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[1])
    imshow_torch(input_tensor_translated)
    imshow_torch(input_tensor_translated - input_tensor_translated_layer)

def test_Multiple_BW_images_single_shift():
    input_tensor = read_image_stack_default_torch()
    translation_H = [20,40,60]
    translation_W = [80,80,80]

    # (1). OpenCV implementation:
    input_tensor_numpy = (input_tensor.cpu().permute([2,3,0,1]).numpy()*255).astype(np.uint8)
    input_tensor_1_numpy = input_tensor_numpy[:,:,0,:]
    input_tensor_2_numpy = input_tensor_numpy[:,:,1,:]
    input_tensor_3_numpy = input_tensor_numpy[:,:,2,:]
    num_rows, num_cols = input_tensor_numpy.shape[0:2]
    translation_matrix_1 = np.float32([[1, 0, translation_W[0]], [0, 1, translation_H[0]]])
    translation_matrix_2 = np.float32([[1, 0, translation_W[1]], [0, 1, translation_H[1]]])
    translation_matrix_3 = np.float32([[1, 0, translation_W[2]], [0, 1, translation_H[2]]])
    input_tensor_1_numpy_translated = cv2.warpAffine(input_tensor_1_numpy, translation_matrix_1, (num_cols, num_rows))
    input_tensor_2_numpy_translated = cv2.warpAffine(input_tensor_2_numpy, translation_matrix_2, (num_cols, num_rows))
    input_tensor_3_numpy_translated = cv2.warpAffine(input_tensor_3_numpy, translation_matrix_3, (num_cols, num_rows))
    # (2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_W, translation_H)
    # (3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_W, translation_H)

    # Compare Results: #
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated / 255).unsqueeze(0).permute([0, 3, 1, 2]))
    imshow_torch(torch.Tensor(input_tensor_2_numpy_translated / 255).unsqueeze(0).permute([0, 3, 1, 2]))
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[0])
    imshow_torch(torch.Tensor(input_tensor_2_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[1])
    imshow_torch(input_tensor_translated)
    imshow_torch(input_tensor_translated - input_tensor_translated_layer)

def test_Affine_Transforms():
    ### Read Frame: ###
    input_tensor = read_image_default_torch()
    input_tensor = RGB2BW(input_tensor)
    translation_HW = (20, 80)

    # (1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0, 0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    translation_matrix = np.float32([[1, 0, translation_HW[1]], [0, 1, translation_HW[0]]])
    input_tensor_numpy_translated = cv2.warpAffine(input_tensor_numpy, translation_matrix, (num_cols, num_rows))
    # (2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_HW[1], translation_HW[0])
    # (3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_HW[1], translation_HW[0])



######################################################################################################################################################################################
#
#
#
#
######################################################################################################################################################################################
### Rotation Warping Operation: ###
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

        ### Choose Rotation Angle and assign variables needed for later: # # #
        a = torch.tan(thetas / 2)
        b = -torch.sin(thetas)
        # # a = (scale * cos(theta) - 1) / (-scale * sin(theta));
        # # b = -scale * sin(theta);

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

        ### Embedd Image Into Larger Matrix Able To Embedd Largest Rotation of 45 degrees. cant i handle M*sqrt(2),N*sqrt(2)???: ###
        self.padded_mat[:, :, H // 2:H // 2 + H, W // 2:W // 2 + W] = input_tensors

        ### Use Parallel Computing instead of looping: ###
        Ix_parallel = (torch.fft.ifftn(torch.fft.fftn(self.padded_mat, dim=[-1]) * torch.exp(self.Nx_vec_mat_final * a), dim=[-1])).real
        Iy_parallel = (torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(self.Ny_vec_mat_final * b * -1).conj(), dim=[-2])).real
        input_mat_rotated = (torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(self.Nx_vec_mat_final * a), dim=[-1])).real
        # imshow_torch(input_mat_rotated[0,-1])

        return input_mat_rotated

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

    ### Choose Rotation Angle and assign variables needed for later: # # #
    B, T, H, W = input_mat.shape
    a = torch.tan(thetas / 2)
    b = -torch.sin(thetas)
    # # a = (scale * cos(theta) - 1) / (-scale * sin(theta));
    # # b = -scale * sin(theta);

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
    Ix_parallel = (
        torch.fft.ifftn(torch.fft.fftn(input_mat, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    Iy_parallel = (
        torch.fft.ifftn(torch.fft.fftn(Ix_parallel, dim=[-2]) * torch.exp(Ny_vec_mat_final * b * -1).conj(),
                        dim=[-2])).real
    input_mat_rotated = (
        torch.fft.ifftn(torch.fft.fftn(Iy_parallel, dim=[-1]) * torch.exp(Nx_vec_mat_final * a), dim=[-1])).real
    # imshow_torch(input_mat_rotated[0,-1])

    return input_mat_rotated

### TODO: temp test, to be commented out: ###
def test_single_BW_image_single_rotation():
    input_tensor = read_image_default_torch()
    B,C,H,W = input_tensor.shape
    (cX, cY) = (W // 2, H // 2)
    input_tensor = RGB2BW(input_tensor)
    rotation_angle = 20 #[deg]

    #(1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0,0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    affine_matrix = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
    input_tensor_numpy_rotation = cv2.warpAffine(input_tensor_numpy, affine_matrix, (num_cols, num_rows))
    #(2). FFT Shift Function:
    input_tensor_rotation = FFT_Rotate_Function(input_tensor, torch.Tensor([rotation_angle*np.pi/180]))
    #(3). FFT Shift Layer:
    fft_rotate_layer = FFT_Rotate_Layer()
    input_tensor_rotation_layer = fft_rotate_layer.forward(input_tensor, torch.Tensor([rotation_angle*np.pi/180]))

    # Compare Results: #
    #TODO: results don't look right!!! something is wrong with my FFT layer, it look like there's a skew!!!
    input_tensor_rotation = crop_torch_batch(input_tensor_rotation, (H, W))
    input_tensor_rotation_layer = crop_torch_batch(input_tensor_rotation_layer, (H, W))
    figure()
    imshow_torch(torch.Tensor(input_tensor_numpy_rotation).unsqueeze(0).unsqueeze(0))
    imshow_torch(input_tensor_rotation)
    imshow_torch(input_tensor_rotation_layer)
    figure()
    imshow_torch(input_tensor_rotation + torch.Tensor(input_tensor_numpy_rotation).unsqueeze(0).unsqueeze(0))
    figure()
    imshow_torch(input_tensor_rotation_layer - input_tensor_rotation)

def test_SingleRGB_image_SingleRotation():
    input_tensor = read_image_default_torch()
    translation_HW = (20, 80)

    # (1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    translation_matrix = np.float32([[1, 0, translation_HW[1]], [0, 1, translation_HW[0]]])
    input_tensor_numpy_translated = cv2.warpAffine((input_tensor_numpy*255).astype(uint8), translation_matrix, (num_cols, num_rows))
    # (2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_HW[1], translation_HW[0])
    # (3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_HW[1], translation_HW[0])

    # Compare Results: #
    figure()
    imshow(input_tensor_numpy_translated)
    figure()
    imshow_torch(input_tensor_translated)
    figure()
    imshow_torch(input_tensor_translated - torch.Tensor(input_tensor_numpy_translated).unsqueeze(0).unsqueeze(0))
    figure()
    imshow_torch(input_tensor_translated - input_tensor_translated_layer)

def test_Multiple_BW_images_SingleRotation():
    input_tensor = read_image_stack_default_torch()
    translation_H = [20]
    translation_W = [80]

    # (1). OpenCV implementation:
    input_tensor_numpy = (input_tensor.cpu().permute([2,3,0,1]).numpy()*255).astype(np.uint8)
    input_tensor_1_numpy = input_tensor_numpy[:,:,0,:]
    input_tensor_2_numpy = input_tensor_numpy[:,:,1,:]
    input_tensor_3_numpy = input_tensor_numpy[:,:,2,:]
    num_rows, num_cols = input_tensor_numpy.shape[0:2]
    translation_matrix = np.float32([[1, 0, translation_W[0]], [0, 1, translation_H[0]]])
    input_tensor_1_numpy_translated = cv2.warpAffine(input_tensor_1_numpy, translation_matrix, (num_cols, num_rows))
    input_tensor_2_numpy_translated = cv2.warpAffine(input_tensor_2_numpy, translation_matrix, (num_cols, num_rows))
    input_tensor_3_numpy_translated = cv2.warpAffine(input_tensor_3_numpy, translation_matrix, (num_cols, num_rows))
    # (2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_W, translation_H)
    # (3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_W, translation_H)

    # Compare Results: #
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated / 255).unsqueeze(0).permute([0, 3, 1, 2]))
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[0])
    imshow_torch(torch.Tensor(input_tensor_2_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[1])
    imshow_torch(input_tensor_translated)
    imshow_torch(input_tensor_translated - input_tensor_translated_layer)

def test_Multiple_BW_images_MultipleRotations():
    input_tensor = read_image_stack_default_torch()
    translation_H = [20,40,60]
    translation_W = [80,80,80]

    # (1). OpenCV implementation:
    input_tensor_numpy = (input_tensor.cpu().permute([2,3,0,1]).numpy()*255).astype(np.uint8)
    input_tensor_1_numpy = input_tensor_numpy[:,:,0,:]
    input_tensor_2_numpy = input_tensor_numpy[:,:,1,:]
    input_tensor_3_numpy = input_tensor_numpy[:,:,2,:]
    num_rows, num_cols = input_tensor_numpy.shape[0:2]
    translation_matrix_1 = np.float32([[1, 0, translation_W[0]], [0, 1, translation_H[0]]])
    translation_matrix_2 = np.float32([[1, 0, translation_W[1]], [0, 1, translation_H[1]]])
    translation_matrix_3 = np.float32([[1, 0, translation_W[2]], [0, 1, translation_H[2]]])
    input_tensor_1_numpy_translated = cv2.warpAffine(input_tensor_1_numpy, translation_matrix_1, (num_cols, num_rows))
    input_tensor_2_numpy_translated = cv2.warpAffine(input_tensor_2_numpy, translation_matrix_2, (num_cols, num_rows))
    input_tensor_3_numpy_translated = cv2.warpAffine(input_tensor_3_numpy, translation_matrix_3, (num_cols, num_rows))
    # (2). FFT Shift Function:
    input_tensor_translated = shift_matrix_subpixel_torch(input_tensor, translation_W, translation_H)
    # (3). FFT Shift Layer:
    shift_layer_torch = Shift_Layer_Torch()
    input_tensor_translated_layer = shift_layer_torch.forward(input_tensor, translation_W, translation_H)

    # Compare Results: #
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated / 255).unsqueeze(0).permute([0, 3, 1, 2]))
    imshow_torch(torch.Tensor(input_tensor_2_numpy_translated / 255).unsqueeze(0).permute([0, 3, 1, 2]))
    imshow_torch(torch.Tensor(input_tensor_1_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[0])
    imshow_torch(torch.Tensor(input_tensor_2_numpy_translated/255).unsqueeze(0).permute([0,3,1,2]) - input_tensor_translated_layer[1])
    imshow_torch(input_tensor_translated)
    imshow_torch(input_tensor_translated - input_tensor_translated_layer)
######################################################################################################################################################################################
#
#
#
#
#
#
######################################################################################################################################################################################
### Rotation+Translation Operation: ###
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
        ### Choose Rotation Angle and assign variables needed for later: # # #
        a = torch.tan(thetas / 2)
        b = -torch.sin(thetas)
        # # a = (scale * cos(theta) - 1) / (-scale * sin(theta));
        # # b = -scale * sin(theta);

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
    ### Choose Rotation Angle and assign variables needed for later: # # #
    a = torch.tan(thetas / 2)
    b = -torch.sin(thetas)
    # # a = (scale * cos(theta) - 1) / (-scale * sin(theta));
    # # b = -scale * sin(theta);

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

    return input_tensors_rotated_displaced

### TODO: temp test, to be commented out: ###
def test_single_BW_image_single_rotation():
    input_tensor = read_image_default_torch()
    B,C,H,W = input_tensor.shape
    (cX, cY) = (W // 2, H // 2)
    input_tensor = RGB2BW(input_tensor)
    rotation_angle = 20 #[deg]
    shift_H = torch.Tensor([0])
    shift_W = torch.Tensor([0])

    #(1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0,0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    affine_matrix = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
    input_tensor_numpy_rotation = cv2.warpAffine(input_tensor_numpy, affine_matrix, (num_cols, num_rows))
    #(2). FFT Shift Function:
    input_tensor_rotation = FFT_Rotation_Translation_Function(input_tensor, torch.Tensor([rotation_angle*np.pi/180]), shift_W, shift_H)
    #(3). FFT Shift Layer:
    fft_rotate_layer = FFT_Translation_Rotation_Layer()
    input_tensor_rotation_layer = fft_rotate_layer.forward(input_tensor, shift_W, shift_H, torch.Tensor([rotation_angle*np.pi/180]))

    # Compare Results: #
    #TODO: results don't look right!!! something is wrong with my FFT layer, it look like there's a skew!!!
    input_tensor_rotation = crop_torch_batch(input_tensor_rotation, (H, W))
    input_tensor_rotation_layer = crop_torch_batch(input_tensor_rotation_layer, (H, W))
    figure()
    imshow_torch(torch.Tensor(input_tensor_numpy_rotation).unsqueeze(0).unsqueeze(0))
    imshow_torch(input_tensor_rotation)
    imshow_torch(input_tensor_rotation_layer)
    figure()
    imshow_torch(input_tensor_rotation_layer + torch.Tensor(input_tensor_numpy_rotation).unsqueeze(0).unsqueeze(0))
    figure()
    imshow_torch(input_tensor_rotation_layer - input_tensor_rotation)
######################################################################################################################################################################################
#
#
#
#
#
#
#
#
######################################################################################################################################################################################
### Affine (Translation+Rotation+Scaling) Operations: ###
def warp_tensor_affine(input_tensor, shift_x, shift_y, scale, rotation_angle):
    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensor.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)


    # #(1). Scale then Shift:
    # ### Scale: ###
    # X0 *= 1/scale
    # Y0 *= 1/scale
    # ### Shift: ###
    # X0 += shift_x * scale
    # Y0 += shift_y * scale

    #(2). Shift then Scale:
    ### Shift: ###
    X0 += shift_x * 1
    Y0 += shift_y * 1
    ### Scale: ###
    X0 *= 1 / scale
    Y0 *= 1 / scale

    ### Rotation: ###
    # X0_centered = X0 - X0.max() / 2
    # Y0_centered = Y0 - Y0.max() / 2
    # X0_centered = X0 - W / 2
    # Y0_centered = Y0 - H / 2
    X0_centered = X0 - (X0.max()-X0.min())/2
    Y0_centered = Y0 - (Y0.max()-Y0.min())/2
    rotation_angle = torch.Tensor([rotation_angle])
    X0_new = np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered - np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    Y0_new = np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered + np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensor, flow_grid, mode='bilinear')
    return output_tensor


def warp_tensor_affine_inverse(input_tensor, shift_x, shift_y, scale, rotation_angle):
    shift_x = -shift_x
    shift_y = -shift_y
    scale = scale
    rotation_angle = -rotation_angle

    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensor.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)

    ### Rotation: ###
    # X0_centered = X0 - X0.max() / 2
    # Y0_centered = Y0 - Y0.max() / 2
    # X0_centered = X0 - W / 2
    # Y0_centered = Y0 - H / 2
    X0_centered = X0 - (X0.max() - X0.min()) / 2
    Y0_centered = Y0 - (Y0.max() - Y0.min()) / 2
    rotation_angle = torch.Tensor([rotation_angle])
    X0_new = np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered - np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    Y0_new = np.sin(rotation_angle * np.pi / 180).unsqueeze(-1) * X0_centered + np.cos(rotation_angle * np.pi / 180).unsqueeze(-1) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new

    #(1). Scale then Shift:
    ### Scale: ###
    X0 *= scale
    Y0 *= scale
    ### Shift: ###
    X0 += shift_x * 1
    Y0 += shift_y * 1

    # #(2). Shift then Scale:
    # ### Shift: ###
    # X0 += shift_x * 1
    # Y0 += shift_y * 1
    # ### Scale: ###
    # X0 *= 1 / scale
    # Y0 *= 1 / scale


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensor, flow_grid, mode='bilinear')
    return output_tensor


def param2theta(param, w, h):
    # h = 50
    # w = 50
    # param = cv2.getRotationMatrix2D((h,w),45,1)

    param = np.linalg.inv(param)
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta


def warp_tensor_affine_matrix(input_tensor, shift_x, shift_y, scale, rotation_angle):
    B,C,H,W = input_tensor.shape
    height, width = input_tensor.shape[2], input_tensor.shape[3]
    center = (width / 2, height / 2)
    affine_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    affine_matrix[0, 2] -= shift_x
    affine_matrix[1, 2] -= shift_y

    ### To Theta: ###
    full_T = np.zeros((3,3))
    full_T[2,2] = 1
    full_T[0:2,:] = affine_matrix
    theta = param2theta(full_T,W,H)

    affine_matrix_tensor = torch.Tensor(theta)
    affine_matrix_tensor = affine_matrix_tensor.unsqueeze(0) #TODO: multiply batch dimension to batch_size. also and more importantly - generalize to shift_x and affine parameters being of batch_size (or maybe even [batch_size, number_of_channels]) and apply to each tensor and channel

    ### Get Grid From Affine Matrix: ###
    output_grid = torch.nn.functional.affine_grid(affine_matrix_tensor, torch.Size((B,C,H,W)))

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensor, output_grid, mode='bilinear')
    return output_tensor


def warp_numpy_affine(input_mat, shift_x, shift_y, scale, rotation_angle):
    ### Pure OpenCV: ###
    # # shift_scale_rotate(img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    # height, width = input_mat.shape[:2]ft_x, shift_y, scale, rotation_angle):
    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensor.shape
    # center = (width / 2, height / 2)
    # affine_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    # affine_matrix[0, 2] += shift_x * width
    # affine_matrix[1, 2] += shift_y * height
    # output_mat = cv2.warpAffine(input_mat, affine_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101, borderValue=None)


    ### IMGAUG: ###
    imgaug_parameters = get_IMGAUG_parameters()
    # Affine:
    imgaug_parameters['flag_affine_transform'] = True
    imgaug_parameters['affine_scale'] = scale
    imgaug_parameters['affine_translation_percent'] = None
    imgaug_parameters['affine_translation_number_of_pixels'] = {"x":shift_x, "y":shift_y}
    imgaug_parameters['affine_rotation_degrees'] = rotation_angle
    imgaug_parameters['affine_shear_degrees'] = 0
    imgaug_parameters['affine_order'] = cv2.INTER_LINEAR
    imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
    imgaug_parameters['probability_of_affine_transform'] = 1
    ### Get Augmenter: ###
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)
    deterministic_Augmenter_Object = imgaug_transforms.to_deterministic()
    output_mat = deterministic_Augmenter_Object.augment_image(input_mat)

    return output_mat


def vec_to_pytorch_format(input_vec, dim_to_put_scalar_in=0):
    if dim_to_put_scalar_in == 1: #channels
        output_tensor = torch.Tensor(input_vec).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    if dim_to_put_scalar_in == 0: #batches
        output_tensor = torch.Tensor(input_vec).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return output_tensor


def get_max_correct_form(input_tensor, dim_to_leave):
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
            values, indices = input_tensor.max(2)
        if len(values.shape) > 1:
            values, indices = values.max(2)
        values = values.unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [B,C,1,1]

    return values


def get_min_correct_form(input_tensor, dim_to_leave):
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
        values = values.unsqueeze(2).unsqueeze(3)  # Return to pytorch format: [B,C,1,1]

    return values


class Warp_Tensors_Affine_Layer(nn.Module):
    def __init__(self, *args):
        super(Warp_Tensors_Affine_Layer, self).__init__()
        self.B = None
        self.C = None
        self.H = None
        self.W = None
        self.X0 = None
        self.Y0 = None
    def forward(self, input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False, flag_interpolation_mode='bilinear'):
        flag_new_meshgrid = self.X0 is None
        if self.X0 is not None:
            flag_new_meshgrid = flag_new_meshgrid and (self.X0.shape[-1] != input_tensors.shape[-1] or self.X0.shape[-2] != input_tensors.shape[-2])
        if flag_new_meshgrid:
            self.B, self.C, self.H, self.W = input_tensors.shape
            # (1). Create meshgrid:
            [self.X0, self.Y0] = np.meshgrid(np.arange(self.W), np.arange(self.H))
            self.X0 = np.float32(self.X0)
            self.Y0 = np.float32(self.Y0)
            # (2). Turn meshgrid to be tensors:
            self.X0 = torch.Tensor(self.X0).to(input_tensors.device)
            self.Y0 = torch.Tensor(self.Y0).to(input_tensors.device)
            self.X0 = self.X0.unsqueeze(0)
            self.Y0 = self.Y0.unsqueeze(0)
            # (3). Duplicate meshgrid for each batch Example
            self.X0 = torch.cat([self.X0] * self.B, 0)
            self.Y0 = torch.cat([self.Y0] * self.B, 0)

        ### Make Sure All Inputs Are In The Correct Format: ###
        if type(shift_x) == np.ndarray or type(shift_x) == np.float32 or type(shift_x) == np.float64:
            shift_x = np.atleast_1d(shift_x)
            shift_y = np.atleast_1d(shift_y)
            shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
            shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
        elif type(shift_x) == torch.Tensor:
            shift_x_tensor = torch.atleast_3d(shift_x).to(input_tensors.device)
            shift_y_tensor = torch.atleast_3d(shift_y).to(input_tensors.device)

        if type(rotation_angle) == np.ndarray or type(rotation_angle) == np.float32 or type(rotation_angle) == np.float64:
            rotation_angle = np.atleast_1d(rotation_angle)
            rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
        elif type(rotation_angle) == torch.Tensor:
            rotation_angle_tensor = torch.atleast_3d(rotation_angle).to(input_tensors.device)

        if type(scale) == np.ndarray or type(scale) == np.float32 or type(scale) == np.float64:
            scale = np.atleast_1d(scale)
            scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2).to(input_tensors.device)
        elif type(scale) == torch.Tensor:
            scale_tensor = torch.atleast_3d(scale).to(input_tensors.device)

        # (2). Shift then Scale:
        ### Shift: ###
        X1 = self.X0 + shift_x_tensor * 1
        Y1 = self.Y0 + shift_y_tensor * 1
        ### Scale: ###
        X1 *= 1 / scale_tensor
        Y1 *= 1 / scale_tensor

        ### Rotation: ###
        # X0_max = get_max_correct_form(X1, 0).squeeze(2)
        # X0_min = get_min_correct_form(X1, 0).squeeze(2)
        # Y0_max = get_max_correct_form(Y1, 0).squeeze(2)
        # Y0_min = get_min_correct_form(Y1, 0).squeeze(2)
        X0_max = X1[0,-1,-1]
        X0_min = X1[0,0,0]
        Y0_max = Y1[0, -1, -1]
        Y0_min = Y1[0, 0, 0]
        X0_centered = X1 - (X0_max - X0_min) / 2   #TODO: make sure this is correct, perhapse we need an odd number of elements and rotate around it? don't know
        Y0_centered = Y1 - (Y0_max - Y0_min) / 2
        ### TODO: maybe i can speed things up here?!?!?
        X0_new = torch.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - torch.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
        Y0_new = torch.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + torch.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
        X1 = X0_new
        Y1 = Y0_new

        ### Normalize Meshgrid to 1 to conform with grid_sample: ###
        #TODO: i don't know how long it takes to do unsqueeze(-1)
        X1 = X1.unsqueeze(-1)
        Y1 = Y1.unsqueeze(-1)
        X1 = X1 / ((self.W - 1) / 2)
        Y1 = Y1 / ((self.H - 1) / 2)
        flow_grid = torch.cat([X1, Y1], 3)

        ### Warp: ###
        if flag_interpolation_mode == 'bilinear':
            output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid.to(input_tensors.device), mode='bilinear', align_corners=True)
        elif flag_interpolation_mode == 'bicubic':
            output_tensor = bicubic_interpolate(input_tensors, X1, Y1)

        if return_flow_grid:
            return output_tensor, flow_grid
        else:
            return output_tensor


class Warp_Tensors_Inverse_Affine_Layer(nn.Module):
    def __init__(self, *args):
        super(Warp_Tensors_Inverse_Affine_Layer, self).__init__()
        self.B = None
        self.C = None
        self.H = None
        self.W = None
        self.X0 = None
        self.Y0 = None

    def forward(self, input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
        if self.X0 is None:
            self.B, self.C, self.H, self.W = input_tensors.shape
            # (1). Create meshgrid:
            [self.X0, self.Y0] = np.meshgrid(np.arange(self.W), np.arange(self.H))
            self.X0 = np.float32(self.X0)
            self.Y0 = np.float32(self.Y0)
            # (2). Turn meshgrid to be tensors:
            self.X0 = torch.Tensor(self.X0)
            self.Y0 = torch.Tensor(self.Y0)
            self.X0 = self.X0.unsqueeze(0)
            self.Y0 = self.Y0.unsqueeze(0)
            # (3). Duplicate meshgrid for each batch Example
            self.X0 = torch.cat([self.X0] * self.B, 0)
            self.Y0 = torch.cat([self.Y0] * self.B, 0)

        shift_x = np.array(shift_x)
        shift_y = np.array(shift_y)
        scale = np.array(scale)
        rotation_angle = np.array(rotation_angle)

        shift_x = -shift_x
        shift_y = -shift_y
        scale = scale
        rotation_angle = -rotation_angle

        ### Rotation: ###
        X0_max = get_max_correct_form(self.X0, 0).squeeze(2)
        X0_min = get_min_correct_form(self.X0, 0).squeeze(2)
        Y0_max = get_max_correct_form(self.Y0, 0).squeeze(2)
        Y0_min = get_min_correct_form(self.Y0, 0).squeeze(2)
        X0_centered = self.X0 - (X0_max - X0_min) / 2
        Y0_centered = self.Y0 - (Y0_max - Y0_min) / 2
        rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
        X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - np.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
        Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + np.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
        X1 = X0_new
        Y1 = Y0_new

        # (2). Scale then Shift:
        ### Scale: ###
        scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
        X1 *= scale_tensor
        Y1 *= scale_tensor
        ### Shift: ###
        shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
        shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
        X1 += shift_x_tensor * 1
        Y1 += shift_y_tensor * 1

        ### Normalize Meshgrid to 1 to conform with grid_sample: ###
        X1 = X1.unsqueeze(-1)
        Y1 = Y1.unsqueeze(-1)
        X1 = X1 / ((self.W - 1) / 2)
        Y1 = Y1 / ((self.H - 1) / 2)
        flow_grid = torch.cat([X1, Y1], 3)

        ### Warp: ###
        output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid, mode='bilinear')

        if return_flow_grid:
            return output_tensor, flow_grid
        else:
            return output_tensor


def warp_tensors_affine(input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensors.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)

    ##############################################################################
    #(2). Shift then Scale:
    ### Shift: ###
    shift_x = np.array(shift_x)
    shift_y = np.array(shift_y)
    shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
    shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
    X0 += shift_x_tensor * 1
    Y0 += shift_y_tensor * 1

    ### Scale: ###
    scale = np.array(scale)
    scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
    X0 *= 1 / scale_tensor
    Y0 *= 1 / scale_tensor

    ### Rotation: ###
    X0_max = get_max_correct_form(X0,0).squeeze(2)
    X0_min = get_min_correct_form(X0,0).squeeze(2)
    Y0_max = get_max_correct_form(Y0,0).squeeze(2)
    Y0_min = get_min_correct_form(Y0,0).squeeze(2)
    X0_centered = X0 - (X0_max-X0_min)/2
    Y0_centered = Y0 - (Y0_max-Y0_min)/2
    rotation_angle = np.array(rotation_angle)
    rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
    X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - np.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
    Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + np.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new
    ##############################################################################

    # ##############################################################################
    # ### Rotation: ###
    # X0_max = get_max_correct_form(X0, 0).squeeze(2)
    # X0_min = get_min_correct_form(X0, 0).squeeze(2)
    # Y0_max = get_max_correct_form(Y0, 0).squeeze(2)
    # Y0_min = get_min_correct_form(Y0, 0).squeeze(2)
    # X0_centered = X0 - (X0_max - X0_min) / 2
    # Y0_centered = Y0 - (Y0_max - Y0_min) / 2
    # rotation_angle = np.array(rotation_angle)
    # rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
    # X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - np.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
    # Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + np.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
    # X0 = X0_new
    # Y0 = Y0_new
    #
    # X0 = X0_centered
    # Y0 = Y0_centered
    #
    # ### Scale: ###
    # scale = np.array(scale)
    # scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
    # X0 *= 1 / scale_tensor
    # Y0 *= 1 / scale_tensor
    #
    # ### Shift: ###
    # shift_x = np.array(shift_x)
    # shift_y = np.array(shift_y)
    # shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
    # shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
    # X0 += shift_x_tensor * 1
    # Y0 += shift_y_tensor * 1
    # ##############################################################################


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid, mode='bilinear', align_corners=True)

    # imshow_torch(output_tensor - input_tensors)

    if return_flow_grid:
        return output_tensor, flow_grid
    else:
        return output_tensor

# TODO Elisheva
def warp_tensors_affine_elisheva(input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
    #TODO: result doesn't agree with OpenCV implementation!!! there seems to be a a difference in the x component... maybe it's the boarder mode
    B,C,H,W = input_tensors.shape
    #(1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    #(2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    #(3). Duplicate meshgrid for each batch Example
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)

    X0_max = get_max_correct_form(X0, 0).squeeze(2)
    X0_min = get_min_correct_form(X0, 0).squeeze(2)
    Y0_max = get_max_correct_form(Y0, 0).squeeze(2)
    Y0_min = get_min_correct_form(Y0, 0).squeeze(2)
    X0 = X0 - (X0_max - X0_min) / 2
    Y0 = Y0 - (Y0_max - Y0_min) / 2


    #(2). Shift then Scale:
    ### Shift: ###
    shift_x = np.array(shift_x)
    shift_y = np.array(shift_y)
    shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
    shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
    X0 += shift_x_tensor * 1
    Y0 += shift_y_tensor * 1
    ### Scale: ###
    scale = np.array(scale)
    scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
    X0 *= 1 / scale_tensor
    Y0 *= 1 / scale_tensor

    ### Rotation: ###
    rotation_angle = np.array(rotation_angle)
    rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
    X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0 - np.sin(rotation_angle_tensor * np.pi / 180) * Y0
    Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0 + np.cos(rotation_angle_tensor * np.pi / 180) * Y0
    X0 = X0_new
    Y0 = Y0_new


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid, mode='bilinear')

    if return_flow_grid:
        return output_tensor, flow_grid
    else:
        return output_tensor


def warp_tensors_affine_inverse(input_tensors, shift_x, shift_y, scale, rotation_angle, return_flow_grid=False):
    shift_x = np.array(shift_x)
    shift_y = np.array(shift_y)
    scale = np.array(scale)
    rotation_angle = np.array(rotation_angle)

    shift_x = -shift_x
    shift_y = -shift_y
    scale = scale
    rotation_angle = -rotation_angle

    B, C, H, W = input_tensors.shape
    # (1). Create meshgrid:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X0 = np.float32(X0)
    Y0 = np.float32(Y0)
    # (2). Turn meshgrid to be tensors:
    X0 = torch.Tensor(X0)
    Y0 = torch.Tensor(Y0)
    X0 = X0.unsqueeze(0)
    Y0 = Y0.unsqueeze(0)
    # (3). Duplicate meshgrid for each batch Example
    X0 = torch.cat([X0] * B, 0)
    Y0 = torch.cat([Y0] * B, 0)


    ### Rotation: ###
    X0_max = get_max_correct_form(X0, 0).squeeze(2)
    X0_min = get_min_correct_form(X0, 0).squeeze(2)
    Y0_max = get_max_correct_form(Y0, 0).squeeze(2)
    Y0_min = get_min_correct_form(Y0, 0).squeeze(2)
    X0_centered = X0 - (X0_max - X0_min) / 2
    Y0_centered = Y0 - (Y0_max - Y0_min) / 2
    rotation_angle_tensor = torch.Tensor(rotation_angle).unsqueeze(1).unsqueeze(2)
    X0_new = np.cos(rotation_angle_tensor * np.pi / 180) * X0_centered - np.sin(rotation_angle_tensor * np.pi / 180) * Y0_centered
    Y0_new = np.sin(rotation_angle_tensor * np.pi / 180) * X0_centered + np.cos(rotation_angle_tensor * np.pi / 180) * Y0_centered
    X0 = X0_new
    Y0 = Y0_new


    # (2). Scale then Shift:
    ### Scale: ###
    scale_tensor = torch.Tensor(scale).unsqueeze(1).unsqueeze(2)
    X0 *= scale_tensor
    Y0 *= scale_tensor
    ### Shift: ###
    shift_x_tensor = torch.Tensor(shift_x).unsqueeze(1).unsqueeze(2)
    shift_y_tensor = torch.Tensor(shift_y).unsqueeze(1).unsqueeze(2)
    X0 += shift_x_tensor * 1
    Y0 += shift_y_tensor * 1


    ### Normalize Meshgrid to 1 to conform with grid_sample: ###
    X0 = X0.unsqueeze(-1)
    Y0 = Y0.unsqueeze(-1)
    X0 = X0 / ((W - 1) / 2)
    Y0 = Y0 / ((H - 1) / 2)
    flow_grid = torch.cat([X0, Y0], 3)

    ### Warp: ###
    output_tensor = torch.nn.functional.grid_sample(input_tensors, flow_grid, mode='bilinear')

    if return_flow_grid:
        return output_tensor, flow_grid
    else:
        return output_tensor

def warp_numpy_affine_IMGAUG(input_mat, shift_x, shift_y, scale, rotation_angle):
    ### Pure OpenCV: ###
    # # shift_scale_rotate(img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None):
    # height, width = input_mat.shape[:2]
    # center = (width / 2, height / 2)
    # affine_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)
    # affine_matrix[0, 2] += shift_x * width
    # affine_matrix[1, 2] += shift_y * height
    # output_mat = cv2.warpAffine(input_mat, affine_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101, borderValue=None)


    ### IMGAUG: ###
    imgaug_parameters = get_IMGAUG_parameters()
    # Affine:
    imgaug_parameters['flag_affine_transform'] = True
    imgaug_parameters['affine_scale'] = scale
    imgaug_parameters['affine_translation_percent'] = None
    imgaug_parameters['affine_translation_number_of_pixels'] = {"x":shift_x, "y":shift_y}
    imgaug_parameters['affine_rotation_degrees'] = rotation_angle
    imgaug_parameters['affine_shear_degrees'] = 0
    imgaug_parameters['affine_order'] = cv2.INTER_LINEAR
    imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
    imgaug_parameters['probability_of_affine_transform'] = 1
    # # Perspective:
    # imgaug_parameters['flag_perspective_transform'] = False
    # imgaug_parameters['flag_perspective_transform_keep_size'] = True
    # imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
    # imgaug_parameters['probability_of_perspective_trans#Perspective:
    # imgaug_parameters['flag_perspective_transform'] = False
    # imgaug_parameters['flag_perspective_transform_keep_size'] = True
    # imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
    # imgaug_parameters['probability_of_perspective_transform'] = 1form'] = 1
    ### Get Augmenter: ###
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)
    deterministic_Augmenter_Object = imgaug_transforms.to_deterministic()
    output_mat = deterministic_Augmenter_Object.augment_image(input_mat)

    return output_mat

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

### TODO: temp test, to be commented out: ###
from RapidBase.Anvil._transforms.matrix_transformations import *
def test_single_BW_image_SingleAffineTransform():
    input_tensor = read_image_default_torch()
    # input_tensor = input_tensor[:,:,0:-1,0:-1]
    B,C,H,W = input_tensor.shape
    (cX, cY) = (W // 2, H // 2)
    input_tensor = RGB2BW(input_tensor)
    rotation_angle = torch.Tensor([0]) #[deg]
    shift_H = torch.Tensor([0])
    shift_W = torch.Tensor([10])
    scale_factor = torch.Tensor([1.0])

    #(*). Anvil and OpenCV agree as long as:
    # the (H,W) are odd,
    # and the center to transformation_matrix_2D is H/2 and not H//2,
    # i enter both degrees,
    # i use bilinear!!!
    # and scale=1 !!! (probably again an issue with how you define the center, etc')


    ### Use Avrahams transform matrices from RapidBase.Anvil. which is intended for torch.AffineGrid (not for us!!!!) because i don't know how to use cv2.warpAffine for now: ###
    invertable_transformation = transformation_matrix_2D((H / 2, W / 2), rotation_angle, scale_factor, (shift_H, shift_W))
    invertable_transformation_numpy = invertable_transformation.cpu().numpy()
    invertable_transformation_numpy = invertable_transformation_numpy[0:2,:]

    #(1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0,0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    input_tensor_numpy_affine = cv2.warpAffine(input_tensor_numpy, invertable_transformation_numpy, (num_cols, num_rows), cv2.INTER_LINEAR)
    #(2). FFT Shift Function:
    input_tensor_warped_affine_function = warp_tensors_affine(input_tensor, shift_x=shift_W+1, shift_y=-shift_H, scale=scale_factor, rotation_angle=rotation_angle)
    # input_tensor_warped_affine_function = warp_tensors_affine_elisheva(input_tensor, shift_x=-shift_W, shift_y=-shift_H, scale=scale_factor, rotation_angle=rotation_angle)
    #(3). FFT Shift Layer:
    warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
    input_tensor_warped_layer = warp_tensors_affine_layer.forward(input_tensor, shift_W, shift_H, scale_factor, torch.Tensor([rotation_angle]))
    #(4). Affine Grid (Anvil):
    input_tensor_Anvil = affine_warp_matrix(input_tensor, shift_H, shift_W, rotation_angle, scale_factor, warp_method='bilinear')
    #(5). Circshift:
    input_tensor_circshift = torch.roll(input_tensor, (shift_W.int().item()), -1)

    # Compare Results: #
    #TODO: results don't look right!!! something is wrong with my FFT layer, it look like there's a skew!!!
    input_tensor_warped_affine_function = crop_torch_batch(input_tensor_warped_affine_function, (H, W))
    input_tensor_warped_layer = crop_torch_batch(input_tensor_warped_layer, (H, W))
    input_tensor_Anvil = crop_torch_batch(input_tensor_Anvil, (H, W))
    input_tensor_OpenCV = torch.Tensor(input_tensor_numpy_affine).unsqueeze(0).unsqueeze(0)

    # imshow_torch(torch.Tensor(input_tensor_numpy_rotation).unsqueeze(0).unsqueeze(0), title_str='OpenCV warpAffine')
    # imshow_torch(input_tensor_warped_affine_function, title_str='Dudy warp affine function')
    # imshow_torch(input_tensor_warped_layer, title_str='Dudy warp affine layer')
    # imshow_torch(input_tensor_Anvil, title_str='Anvil (params2theta + torch.affine_grid')
    # imshow_torch(input_tensor_circshift, title_str='Circshift')
    imshow_torch(input_tensor_warped_layer - input_tensor_OpenCV, title_str='affine layer - openCV')
    imshow_torch(input_tensor_warped_layer - input_tensor_Anvil, title_str='affine layer - Anvil')
    imshow_torch(input_tensor_warped_affine_function - input_tensor, title_str='Affine Function - Input_tensor')
    imshow_torch(input_tensor_warped_layer - input_tensor_circshift, title_str='affine layer - Circshift')
    imshow_torch(input_tensor - input_tensor_circshift, title_str='Input tensor - Circshift')
    imshow_torch(input_tensor_OpenCV - input_tensor, title_str='OpenCV - Input_Tensor')
    imshow_torch(input_tensor_OpenCV - input_tensor_circshift, title_str='OpenCV - Circshift')
    imshow_torch(input_tensor_Anvil - input_tensor_OpenCV, title_str='Anvil - OpenCV')

# test_single_BW_image_SingleAffineTransform()


### General Warp/Deformation (Per-Pixel-Flow):  ###
class Warp_Object(nn.Module):
    # Initialize this with a module
    def __init__(self):
        super(Warp_Object, self).__init__()
        self.X = None
        self.Y = None

    # Elementwise sum the output of a submodule to its input
    def forward(self, input_image, delta_x, delta_y, flag_bicubic_or_bilinear='bilinear'):
        # delta_x = map of x deltas from meshgrid, shape=[B,H,W] or [B,C,H,W].... same for delta_Y
        B, C, H, W = input_image.shape
        BXC = B * C

        ### ReOrder delta_x, delta_y: ###
        #TODO: this expects delta_x,delta_y to be image sized tensors. but sometimes i just wanna pass in a single number per image
        #(1). Dim=3 <-> [B,H,W], I Interpret As: Same Flow On All Channels:
        if len(delta_x.shape) == 3:  # [B,H,W] - > [B,H,W,1]
            delta_x = delta_x.unsqueeze(-1)
            delta_y = delta_x.unsqueeze(-1)
            flag_same_on_all_channels = True
        #(2). Dim=4 <-> [B,C,H,W], Different Flow For Each Channel:
        elif (len(delta_x.shape) == 4 and delta_x.shape[1]==C):  # [B,C,H,W] - > [BXC,H,W,1] (because pytorch's function only warps all channels of a tensor the same way so in order to warp each channel seperately we need to transfer channels to batch dim)
            delta_x = delta_x.view(B * C, H, W).unsqueeze(-1)
            delta_y = delta_y.view(B * C, H, W).unsqueeze(-1)
            flag_same_on_all_channels = False
        #(3). Dim=4 but C=1 <-> [B,1,H,W], Same Flow On All Channels:
        elif len(delta_x.shape) == 4 and delta_x.shape[1]==1:
            delta_x = delta_x.permute([0,2,3,1]) #[B,1,H,W] -> [B,H,W,1]
            delta_y = delta_y.permute([0,2,3,1])
            flag_same_on_all_channels = True
        #(4). Dim=4 but C=1 <-> [B,H,W,1], Same Flow On All Channels:
        elif len(delta_x.shape) == 4 and delta_x.shape[3] == 1:
            flag_same_on_all_channels = True


        ### Create "baseline" meshgrid (as the function ultimately accepts a full map of locations and not just delta's): ###
        #(*). ultimately X.shape=[BXC, H, W, 1]/[B,H,W,1]... so check if the input shape has changed and only then create a new meshgrid:
        flag_input_changed_from_last_time = (self.X is None) or (self.X.shape[0]!=BXC and flag_same_on_all_channels==False) or (self.X.shape[0]!=B and flag_same_on_all_channels==True) or (self.X.shape[1]!=H) or (self.X.shape[2]!=W)
        if flag_input_changed_from_last_time:
            print('new meshgrid')
            [X, Y] = np.meshgrid(np.arange(W), np.arange(H))  #X.shape=[H,W]
            if flag_same_on_all_channels:
                X = torch.Tensor([X] * B).unsqueeze(-1) #X.shape=[B,H,W,1]
                Y = torch.Tensor([Y] * B).unsqueeze(-1)
            else:
                X = torch.Tensor([X] * BXC).unsqueeze(-1) #X.shape=[BXC,H,W,1]
                Y = torch.Tensor([Y] * BXC).unsqueeze(-1)
            X = X.to(input_image.device)
            Y = Y.to(input_image.device)
            self.X = X
            self.Y = Y

        ### Add Difference (delta) Maps to Meshgrid: ###
        ### Previous Try: ###
        # X += delta_x
        # Y += delta_y
        # X = (X - W / 2) / (W / 2 - 1)
        # Y = (Y - H / 2) / (H / 2 - 1)
        # ### Previous Use: ###
        # new_X = ((self.X + delta_x) - W / 2) / (W / 2 - 1)
        # new_Y = ((self.Y + delta_y) - H / 2) / (H / 2 - 1)
        ### New Use: ###
        new_X = 2 * ((self.X + delta_x)) / max(W-1,1) - 1
        new_Y = 2 * ((self.Y + delta_y)) / max(H-1,1) - 1


        #TODO: there's no more need to use my bicubic interpolation since new pytorch versions simply have a flag for it
        #TODO: however, it seems that bicubic interpolation requires of allocations and so theoretically if i want max speed i might need to do it myself
        if flag_bicubic_or_bilinear == 'bicubic':
            #input_image.shape=[B,C,H,W] , new_X.shape=[B,H,W,1] OR new_X.shape=[BXC,H,W,1]
            warped_image = bicubic_interpolate(input_image, new_X, new_Y)
            return warped_image
        else:
            bilinear_grid = torch.cat([new_X,new_Y],dim=3)
            if flag_same_on_all_channels:
                #input_image.shape=[B,C,H,W] , bilinear_grid.shape=[B,H,W,2]
                input_image_to_bilinear = input_image
                warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, bilinear_grid)
                return warped_image
            else:
                #input_image.shape=[BXC,1,H,W] , bilinear_grid.shape=[BXC,H,W,2]
                input_image_to_bilinear = input_image.reshape(-1, int(H), int(W)).unsqueeze(1) #[B,C,H,W]->[B*C,1,H,W]
                warped_image = torch.nn.functional.grid_sample(input_image_to_bilinear, bilinear_grid)
                warped_image = warped_image.view(B,C,H,W)
                return warped_image

#TODO: this object is simply a wrapper on the Warp_Object because of the constant mix-up betwen (X,Y) and (H,W).
#TODO: decide on one convention and only use that!, let it be (H,W)
class Warp_Object_OpticalFlow(Warp_Object):
    # Initialize this with a module
    def __init__(self):
        super(Warp_Object_OpticalFlow, self).__init__()
        self.X = None
        self.Y = None

    # Elementwise sum the output of a submodule to its input
    def forward(self, input_image, optical_flow_map, flag_bicubic_or_bilinear='bilinear', flag_input_type='XY'):
        if flag_input_type == 'XY':
            return super().forward(input_image, optical_flow_map[:,0:1,:,:], optical_flow_map[:,1:2,:,:], flag_bicubic_or_bilinear)
        elif flag_input_type == 'HW':
            return super().forward(input_image, optical_flow_map[:, 1:2, :, :], optical_flow_map[:, 0:1, :, :], flag_bicubic_or_bilinear)
######################################################################################################################################################################################
#
#
#
#
#
#
#
#
######################################################################################################################################################################################
### Turbulence Approximation Using PieceWise Constant Deformation: ###
def get_turbulence_flow_field_numpy(H,W,Cn2):
    ### Parameters: ###
    h = 100
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(H / PatchSize))
    PatchNumCol = int(np.round(W / PatchSize))
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))

    ### Get Random Motion Field: ###
    ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)

    ### Resize (small) Random Motion Field To Image Size: ###
    ShiftMatX = cv2.resize(ShiftMatX0, (W, H), interpolation=cv2.INTER_CUBIC)
    ShiftMatY = cv2.resize(ShiftMatY0, (W, H), interpolation=cv2.INTER_CUBIC)

    return ShiftMatX, ShiftMatY


class Turbulence_Flow_Field_Generation_Layer(nn.Module):
    def __init__(self,H,W, batch_size, Cn2=2e-13):
        super(Turbulence_Flow_Field_Generation_Layer, self).__init__()
        ### Parameters: ###
        h = 100
        wvl = 5.3e-7
        IFOV = 4e-7
        R = 1000
        VarTiltX = 3.34e-6
        VarTiltY = 3.21e-6
        k = 2 * np.pi / wvl
        r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
        PixSize = IFOV * R
        PatchSize = 2 * r0 / PixSize

        ### Get Current Image Shape And Appropriate Meshgrid: ###
        PatchNumRow = int(np.round(H / PatchSize))
        PatchNumCol = int(np.round(W / PatchSize))
        [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))

        ### Get Random Motion Field: ###
        [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
        [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
        X_large = torch.Tensor(X_large).unsqueeze(-1)
        Y_large = torch.Tensor(Y_large).unsqueeze(-1)
        X_large = (X_large-W/2) / (W/2-1)
        Y_large = (Y_large-H/2) / (H/2-1)

        new_grid = torch.cat([X_large, Y_large],2)
        new_grid = torch.Tensor([new_grid.numpy()]*batch_size)

        self.new_grid = new_grid
        self.batch_size = batch_size
        self.PatchNumRow = PatchNumRow
        self.PatchNumCol = PatchNumCol
        self.VarTiltX = VarTiltX
        self.VarTiltY = VarTiltY
        self.R = R
        self.PixSize = PixSize
        self.H = H
        self.W = W

    def forward(self):
        ### TODO: fix this because for Cn2 which is low enough we get self.PatchNumRow & self.PatchNumCol = 0 and this breaks down.....
        ShiftMatX0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltX * self.R / self.PixSize)
        ShiftMatY0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltY * self.R / self.PixSize)
        ShiftMatX0 = ShiftMatX0 * self.W
        ShiftMatY0 = ShiftMatY0 * self.H

        ShiftMatX = torch.nn.functional.interpolate(ShiftMatX0, size=(self.new_grid.shape[2],self.new_grid.shape[1]), mode='bicubic')
        ShiftMatY = torch.nn.functional.interpolate(ShiftMatY0, size=(self.new_grid.shape[2],self.new_grid.shape[1]), mode='bicubic')
        # ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')
        # ShiftMatY = torch.nn.functional.grid_sample(ShiftMatY0, self.new_grid, mode='bilinear', padding_mode='reflection')

        ShiftMatX = ShiftMatX.squeeze()
        ShiftMatY = ShiftMatY.squeeze()

        return ShiftMatX, ShiftMatY


class Turbulence_Deformation_Layer(nn.Module):
    def __init__(self,H, W, Batch_size, Cn2=2e-13):
        super(Turbulence_Deformation_Layer, self).__init__()
        self.W = W
        self.H = H
        self.turbulence_flow_field_generation_layer = Turbulence_Flow_Field_Generation_Layer(H,W,Batch_size,Cn2)
        self.warp_object = Warp_Object()
    def forward(self, input_tensor):
        #TODO: need to check if everything is consistent with the rest of the implementations below
        delta_x_map, delta_y_map = self.turbulence_flow_field_generation_layer.forward()
        output_tensor = self.warp_object.forward(input_tensor, delta_x_map.unsqueeze(0).unsqueeze(0)/self.W, delta_y_map.unsqueeze(0).unsqueeze(0)/self.H)
        return output_tensor, delta_x_map, delta_y_map

#TODO: temp, delete later
def test_turbulence_deformation_layer():
    input_tensor = read_image_default_torch()
    B, C, H, W = input_tensor.shape
    Cn2 = 1e-14
    turbulence_deformation_layer = Turbulence_Deformation_Layer(H,W,B,Cn2)
    input_tensor_warped = turbulence_deformation_layer.forward(input_tensor)

def test_turbulence_field():
    #Torch:
    input_tensor = read_image_default_torch()
    B,C,H,W = input_tensor.shape
    Cn2 = 1e-14
    turbulence_flow_field_layer = Turbulence_Flow_Field_Generation_Layer(H,W,B,Cn2)
    delta_x_map, delta_y_map = turbulence_flow_field_layer.forward()
    imshow_torch(delta_x_map)

    #Numpy:
    delta_x_map_numpy, delta_y_map_numpy = get_turbulence_flow_field_numpy(H,W,Cn2)
    imshow_torch(torch.Tensor(delta_x_map_numpy))


def test_turbulence_field_warping():
    input_tensor = read_image_default_torch()
    input_tensor = RGB2BW(input_tensor)
    input_tensor_numpy = torch_to_numpy(input_tensor)[0]
    B, C, H, W = input_tensor.shape
    Cn2 = 1e-14
    delta_x_map_numpy, delta_y_map_numpy = get_turbulence_flow_field_numpy(H,W,Cn2)

    #(1). OpenCV Remap:
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    X = X0 + delta_x_map_numpy
    Y = Y0 + delta_y_map_numpy
    input_tensor_numpy_warped = cv2.remap(input_tensor_numpy, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_LINEAR)
    #(2). Pytorch Grid Sample:
    X_torch = torch.Tensor(X).unsqueeze(0).unsqueeze(-1)
    Y_torch = torch.Tensor(Y).unsqueeze(0).unsqueeze(-1)
    X_torch = 2 * X_torch / max(W - 1, 1) - 1
    Y_torch = 2 * Y_torch / max(H - 1, 1) - 1
    new_grid = torch.cat((X_torch, Y_torch), -1)
    #TODO: upgrade pytorch version to one which has mode='bicubic'
    input_tensor_warped = torch.nn.functional.grid_sample(input_tensor, new_grid, mode='bilinear', align_corners=False)
    #(3). My Warp_Object:
    warp_layer = Warp_Object()
    delta_x_map_torch = torch.Tensor(delta_x_map_numpy).unsqueeze(0).unsqueeze(0)
    delta_y_map_torch = torch.Tensor(delta_y_map_numpy).unsqueeze(0).unsqueeze(0)
    input_tensor_warped_layer = warp_layer.forward(input_tensor, delta_x_map_torch, delta_y_map_torch)
    #(4). Compare:
    #TODO: the images don't equal exactly, maybe it's because of a difference between c2.INTER_LINEAR and 'bilinear' in pytorch? i don't know
    imshow_torch(input_tensor_warped-numpy_to_torch(input_tensor_numpy_warped))
    imshow_torch(input_tensor_warped_layer-numpy_to_torch(input_tensor_numpy_warped).unsqueeze(0))
######################################################################################################################################################################################
#
#
#
#
#
#
#
######################################################################################################################################################################################
### Cross Correlation: ###
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

def register_images_CrossCorrelation_numpy(input_image1, input_image2, search_area, inner_crop_size_to_use=None):
    # TODO: turn to pytorch instead of numpy
    ### Get Only Valid Inner Crop Of The Images If Wanted: ###
    if inner_crop_size_to_use is None:
        img1 = input_image1
        img2 = input_image2
    else:
        img1 = crop_torch_batch(input_image1, inner_crop_size_to_use, 'center')
        img2 = crop_torch_batch(input_image2, inner_crop_size_to_use, 'center')
    if type(img1) == torch.Tensor:
        img1 = img1.numpy().squeeze(0).squeeze(0)
    if type(img2) == torch.Tensor:
        img2 = img2.numpy().squeeze(0).squeeze(0)

    ### Get Shape: ###
    H, W = img2.shape

    ### Get FFTs: ###
    f1 = cv2.dft(img1.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f2 = cv2.dft(img2.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f1_shf = np.fft.fftshift(f1)
    f2_shf = np.fft.fftshift(f2)

    ### Get Complex numbers outputs instead of 2 output channels: ###
    f1_shf_cplx = f1_shf[:, :, 0] + 1j * f1_shf[:, :, 1]
    f2_shf_cplx = f2_shf[:, :, 0] + 1j * f2_shf[:, :, 1]

    ### Get absolute values needed for calculations: ###
    f1_shf_abs = np.abs(f1_shf_cplx)
    f2_shf_abs = np.abs(f2_shf_cplx)
    total_abs = f1_shf_abs * f2_shf_abs
    P_real = (np.real(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.imag(f1_shf_cplx) * np.imag(f2_shf_cplx))
    P_imag = (np.imag(f1_shf_cplx) * np.real(f2_shf_cplx) -
              np.real(f1_shf_cplx) * np.imag(f2_shf_cplx))
    P_complex = P_real + 1j * P_imag

    ### Get inverse FFT of normalized multiplication of FFTs (phase only part of FFT cross correlation calculation): ###
    P_inverse = np.abs(np.fft.ifft2(P_complex))  # inverse FFT
    P_inverse = np.fft.fftshift(P_inverse)

    ### Get Normalized Cross Correlation (max possible value = 1): ###
    A_sum = img2.sum()
    A_sum2 = (img2 ** 2).sum()
    sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
    sigmaB = (img1).std() * (H * W - 1) ** (1 / 2)
    B_mean = (img1).mean()
    P_inverse = (P_inverse - A_sum * B_mean) / (sigmaA * sigmaB)

    # ### Get CC Max: ###
    # max_index = np.argmax(P_inverse)
    # max_row = max_index // W
    # max_col = max_index # W
    # max_value = P_inverse[max_row, max_col]
    # center_row = H // 2
    # center_col = W // 2

    ### Search CC Max Around Center: ###
    center_row = H // 2
    center_col = W // 2
    max_value = -np.inf
    for index_row in np.arange(center_row - search_area, center_row + search_area):
        for index_col in np.arange(center_col - search_area, center_col + search_area):
            current_value = P_inverse[index_row, index_col]
            if current_value > max_value:
                max_value = current_value
                max_row = index_row
                max_col = index_col

    ### Assign CC Values around max: ###
    CC = zeros((search_area, search_area))
    for index_col in np.arange(0, search_area):
        for index_row in np.arange(0, search_area):
            CC[index_row, index_col] = P_inverse[max_row - search_area // 2 + index_row,
                                                 max_col - search_area // 2 + index_col]

    ### Use parabola fit to interpolate sub-pixel shift: ###
    shifts_total, z_max_vec = return_shifts_using_parabola_fit_numpy(CC)
    shifts_total = [shifts_total[0], shifts_total[1]]
    shifts_total[1] -= (center_row - max_row)
    shifts_total[0] -= (center_col - max_col)
    shift_x = shifts_total[0]
    shift_y = shifts_total[1]

    ### If shift values or z-value doesn't make sense -> limit shift or disregard it: ###
    if abs(shift_x) > search_area:
        shift_x = 0
    if abs(shift_x) > search_area:
        shift_y = 0

    ### Shift Clean Image: ###
    img2_warped = shift_matrix_subpixel(np.expand_dims(input_image2.cpu().numpy().squeeze(0).squeeze(0), -1), -shift_x,
                                        -shift_y).squeeze()

    return (shift_x, shift_y), z_max_vec, img2_warped


def batch_speckle_correlation_1(input_tensor):
    #TODO: this is a somewhat special case where instead of comparing all the images to a single reference_tensor we compare CONSECUTIVE PAIRS.
    # we do this in order to get max possible accuracy of shift between consecutive pairs for high accuracy applications like speckle correlation.
    # the reason we don't always do this even in stabilization is first a matter of code complications/implementaion and second the adding
    # up of low frequency biases which can throw up accuracy in stabilization if not taken care of very very carefully (so fuck it)

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
######################################################################################################################################################################################
#
#
#
#
#
#
#
######################################################################################################################################################################################
### Phase Cross Correlation: ###
### Using Phase Only Correlation: ###
def register_images_PhaseCorrelation_numpy(input_image1, input_image2, search_area, inner_crop_size_to_use=None):
    ### Get Only Valid Inner Crop Of The Images If Wanted: ###
    if inner_crop_size_to_use is None:
        img1 = input_image1
        img2 = input_image2
    else:
        img1 = crop_torch_batch(input_image1, inner_crop_size_to_use, 'center')
        img2 = crop_torch_batch(input_image2, inner_crop_size_to_use, 'center')

    if type(img1) == torch.Tensor:
        img1 = img1.numpy().squeeze(0).squeeze(0)
    if type(img2) == torch.Tensor:
        img2 = img2.numpy().squeeze(0).squeeze(0)

    ### Get Shape: ###
    H, W = img2.shape

    ### Get FFTs: ###
    f1 = cv2.dft(img1.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f2 = cv2.dft(img2.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f1_shf = np.fft.fftshift(f1)
    f2_shf = np.fft.fftshift(f2)

    ### Get Complex numbers outputs instead of 2 output channels: ###
    f1_shf_cplx = f1_shf[:, :, 0] + 1j * f1_shf[:, :, 1]
    f2_shf_cplx = f2_shf[:, :, 0] + 1j * f2_shf[:, :, 1]

    ### Get absolute values needed for calculations: ###
    f1_shf_abs = np.abs(f1_shf_cplx)
    f2_shf_abs = np.abs(f2_shf_cplx)
    total_abs = f1_shf_abs * f2_shf_abs
    P_real = (np.real(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.imag(f1_shf_cplx) * np.imag(f2_shf_cplx)) / (total_abs + 1e-3)
    P_imag = (np.imag(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.real(f1_shf_cplx) * np.imag(f2_shf_cplx)) / (total_abs + 1e-3)
    P_complex = P_real + 1j * P_imag

    ### Get inverse FFT of normalized multiplication of FFTs (phase only part of FFT cross correlation calculation): ###
    P_inverse = np.abs(np.fft.ifft2(P_complex))  # inverse FFT
    P_inverse = np.fft.fftshift(P_inverse)

    ### Get Normalized Cross Correlation (max possible value = 1): ###
    A_sum = img2.sum()
    A_sum2 = (img2 ** 2).sum()
    sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
    sigmaB = (img1).std() * (H * W - 1) ** (1 / 2)
    B_mean = (img1).mean()
    P_inverse = (P_inverse - A_sum * B_mean) / (sigmaA * sigmaB)

    # ### Get CC Max: ###
    # max_index = np.argmax(P_inverse)  #TODO: instead of using expensive argmax simply search small area around zero shift within search_area
    # max_row = max_index // W
    # max_col = max_index # W
    # max_value = P_inverse[max_row, max_col]
    # center_row = H // 2
    # center_col = W // 2

    ### Search CC Max Around Center: ###
    center_row = H // 2
    center_col = W // 2
    max_value = -np.inf
    for index_row in np.arange(center_row - search_area, center_row + search_area):
        for index_col in np.arange(center_col - search_area, center_col + search_area):
            current_value = P_inverse[index_row, index_col]
            if current_value > max_value:
                max_value = current_value
                max_row = index_row
                max_col = index_col

    ### Assign CC Values around max: ###
    CC = zeros((3, 3))
    for index_col in np.arange(0, 3):
        for index_row in np.arange(0, 3):
            CC[index_row, index_col] = P_inverse[
                max_row - search_area // 2 + index_row, max_col - search_area // 2 + index_col]

    ### Use parabola fit to interpolate sub-pixel shift: ###
    shifts_total, z_max_vec = return_shifts_using_parabola_fit_numpy(CC)
    shifts_total = [shifts_total[0], shifts_total[1]]
    shifts_total[0] -= (center_row - max_row)
    shifts_total[1] -= (center_col - max_col)
    shift_x = shifts_total[1]
    shift_y = shifts_total[0]

    # max_id = [0, 0]
    # max_val = 0
    # for idy in np.arange(search_area):
    #     for idx in np.arange(search_area):
    #         if P_inverse[idy, idx] > max_val:
    #             max_val = P_inverse[idy, idx]
    #             max_id = [idy, idx]
    # shift_x = search_area - max_id[0]
    # shift_y = search_area - max_id[1]
    # print(shift_x, shift_y)

    ### If shift values or z-value doesn't make sense -> limit shift or disregard it: ###
    if abs(shift_x) > search_area:
        shift_x = 0
    if abs(shift_x) > search_area:
        shift_y = 0

    ### Shift Clean Image: ###
    img2_warped = shift_matrix_subpixel_torch(input_image2, shift_x, shift_y)

    return (-shift_x, -shift_y), z_max_vec, img2_warped


### Using Phase Only Correlation: ###
def register_images_PhaseCorrelation(input_image1, input_image2, search_area, inner_crop_size_to_use=None):
    ### Get Only Valid Inner Crop Of The Images If Wanted: ###
    if inner_crop_size_to_use is None:
        img1 = input_image1
        img2 = input_image2
    else:
        img1 = crop_torch_batch(input_image1, inner_crop_size_to_use, 'center')
        img2 = crop_torch_batch(input_image2, inner_crop_size_to_use, 'center')

    if type(img1) == torch.Tensor:
        img1 = img1.numpy().squeeze(0).squeeze(0)
    if type(img2) == torch.Tensor:
        img2 = img2.numpy().squeeze(0).squeeze(0)

    ### Get Shape: ###
    H, W = img2.shape

    ### Get FFTs: ###
    f1 = cv2.dft(img1.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f2 = cv2.dft(img2.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    f1_shf = np.fft.fftshift(f1)
    f2_shf = np.fft.fftshift(f2)

    ### Get Complex numbers outputs instead of 2 output channels: ###
    f1_shf_cplx = f1_shf[:, :, 0] + 1j * f1_shf[:, :, 1]
    f2_shf_cplx = f2_shf[:, :, 0] + 1j * f2_shf[:, :, 1]

    ### Get absolute values needed for calculations: ###
    f1_shf_abs = np.abs(f1_shf_cplx)
    f2_shf_abs = np.abs(f2_shf_cplx)
    total_abs = f1_shf_abs * f2_shf_abs
    P_real = (np.real(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.imag(f1_shf_cplx) * np.imag(f2_shf_cplx)) / (total_abs + 1e-3)
    P_imag = (np.imag(f1_shf_cplx) * np.real(f2_shf_cplx) +
              np.real(f1_shf_cplx) * np.imag(f2_shf_cplx)) / (total_abs + 1e-3)
    P_complex = P_real + 1j * P_imag

    ### Get inverse FFT of normalized multiplication of FFTs (phase only part of FFT cross correlation calculation): ###
    P_inverse = np.abs(np.fft.ifft2(P_complex))  # inverse FFT
    P_inverse = np.fft.fftshift(P_inverse)

    ### Get Normalized Cross Correlation (max possible value = 1): ###
    A_sum = img2.sum()
    A_sum2 = (img2 ** 2).sum()
    sigmaA = (A_sum2 - A_sum ** 2 / (H * W)) ** (1 / 2)
    sigmaB = (img1).std() * (H * W - 1) ** (1 / 2)
    B_mean = (img1).mean()
    P_inverse = (P_inverse - A_sum * B_mean) / (sigmaA * sigmaB)

    # ### Get CC Max: ###
    # max_index = np.argmax(P_inverse)  #TODO: instead of using expensive argmax simply search small area around zero shift within search_area
    # max_row = max_index // W
    # max_col = max_index # W
    # max_value = P_inverse[max_row, max_col]
    # center_row = H // 2
    # center_col = W // 2

    ### Search CC Max Around Center: ###
    center_row = H // 2
    center_col = W // 2
    max_value = -np.inf
    for index_row in np.arange(center_row - search_area, center_row + search_area):
        for index_col in np.arange(center_col - search_area, center_col + search_area):
            current_value = P_inverse[index_row, index_col]
            if current_value > max_value:
                max_value = current_value
                max_row = index_row
                max_col = index_col

    ### Assign CC Values around max: ###
    CC = zeros((3, 3))
    for index_col in np.arange(0, 3):
        for index_row in np.arange(0, 3):
            CC[index_row, index_col] = P_inverse[
                max_row - search_area // 2 + index_row, max_col - search_area // 2 + index_col]

    ### Use parabola fit to interpolate sub-pixel shift: ###
    shifts_total, z_max_vec = return_shifts_using_parabola_fit_numpy(CC)
    shifts_total = [shifts_total[0], shifts_total[1]]
    shifts_total[0] -= (center_row - max_row)
    shifts_total[1] -= (center_col - max_col)
    shift_x = shifts_total[1]
    shift_y = shifts_total[0]

    # max_id = [0, 0]
    # max_val = 0
    # for idy in np.arange(search_area):
    #     for idx in np.arange(search_area):
    #         if P_inverse[idy, idx] > max_val:
    #             max_val = P_inverse[idy, idx]
    #             max_id = [idy, idx]
    # shift_x = search_area - max_id[0]
    # shift_y = search_area - max_id[1]
    # print(shift_x, shift_y)

    ### If shift values or z-value doesn't make sense -> limit shift or disregard it: ###
    if abs(shift_x) > search_area:
        shift_x = 0
    if abs(shift_x) > search_area:
        shift_y = 0

    ### Shift Clean Image: ###
    img2_warped = shift_matrix_subpixel_torch(input_image2, shift_x, shift_y)

    return (-shift_x, -shift_y), z_max_vec, img2_warped
######################################################################################################################################################################################
#
#
#
#
#
#
#
#
#
######################################################################################################################################################################################
### FFT-LogPolar Transform Registration: ###
def similarity_torch(input_tensor_1, input_tensor_2):
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

    if input_tensor_1.shape != input_tensor_2.shape:
        raise ValueError('images must have same shapes')

    #TODO: this doesn't include difference_of_gaussians as the numpy version includes!!!!!

    ### Get The FFTs Of Both Images: ###
    input_tensor_1_fft_abs = fftshift_torch(torch.abs(torch_fft2(input_tensor_1)))
    input_tensor_2_fft_abs = fftshift_torch(torch.abs(torch_fft2(input_tensor_2)))

    ### Filter Images By HighPass Filter By Multiplying In Fourier Space: ###
    h = highpass(input_tensor_1_fft_abs.shape)
    input_tensor_1_fft_abs *= h
    input_tensor_2_fft_abs *= h

    ### Transform FFTs To LogPolar Base: ###
    input_tensor_1_fft_abs_LogPolar, log_base = logpolar(input_tensor_1_fft_abs)
    input_tensor_2_fft_abs_LogPolar, log_base = logpolar(input_tensor_2_fft_abs)

    ### Calculate The Phase Cross Correlation Of The Log Transformed Images, Then Get The Max Value To Extract (Angle,Scale): ###
    input_tensor_1_fft_abs_LogPolar = torch_fft2(input_tensor_1_fft_abs_LogPolar)
    input_tensor_2_fft_abs_LogPolar = torch_fft2(input_tensor_2_fft_abs_LogPolar)
    r0 = abs(input_tensor_1_fft_abs_LogPolar) * abs(input_tensor_2_fft_abs_LogPolar)
    phase_cross_correlation = abs(torch_ifft2((input_tensor_1_fft_abs_LogPolar * input_tensor_2_fft_abs_LogPolar.conjugate()) / r0))
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
    input_tensor_2_scaled_rotated = ndii.zoom(input_tensor_2, 1.0 / scale)
    input_tensor_2_scaled_rotated = ndii.rotate(input_tensor_2_scaled_rotated, angle)

    ### Make Sure Both Images Have The Same Dimensions By Using Inserting The Smaller Matrix Into A Larger One: ### #TODO: what's more efficient? this or padding?
    if input_tensor_2_scaled_rotated.shape < input_tensor_1.shape:
        t = np.zeros_like(input_tensor_1)
        t[: input_tensor_2_scaled_rotated.shape[0],
        : input_tensor_2_scaled_rotated.shape[1]] = input_tensor_2_scaled_rotated
        input_tensor_2_scaled_rotated = t
    elif input_tensor_2_scaled_rotated.shape > input_tensor_1.shape:
        input_tensor_2_scaled_rotated = input_tensor_2_scaled_rotated[: input_tensor_1.shape[0], : input_tensor_1.shape[1]]

    ### Get Translation Using Phase Cross Correlation: ###
    input_tensor_1_fft = torch_fft2(input_tensor_1)  # TODO: can save on calculation here since i calculated this above
    input_tensor_2_fft = torch_fft2(input_tensor_2_scaled_rotated)
    phase_cross_correlation = abs(torch_ifft2(
        (input_tensor_1_fft * input_tensor_2_fft.conjugate()) / (abs(input_tensor_1_fft) * abs(input_tensor_2_fft))))
    max_index = torch.argmax(phase_cross_correlation)
    max_index = np.atleast_1d(max_index.cpu().numpy())[0]
    numpy_shape = phase_cross_correlation.shape.cpu().numpy()
    t0, t1 = np.unravel_index(max_index, numpy_shape)
    ### Correct For FFT_Shift Wrap-Arounds: ###
    if t0 > input_tensor_1_fft.shape[0] // 2:
        t0 -= input_tensor_1_fft.shape[0]
    if t1 > input_tensor_1_fft.shape[1] // 2:
        t1 -= input_tensor_1_fft.shape[1]

    ### Shift Second Image According To Found Translation: ###
    input_tensor_2_scaled_rotated_translated = ndii.shift(input_tensor_2_scaled_rotated, [t0, t1])

    ### Correct Parameters For ndimage's Internal Processing: ###
    if angle > 0.0:
        d = int(int(input_tensor_2.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(input_tensor_2.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (input_tensor_2.shape[1] - 1) / (int(input_tensor_2.shape[1] / scale) - 1)

    return input_tensor_2_scaled_rotated_translated, scale, angle, [-t0, -t1]

def similarity_RotationScalingTranslation_numpy(input_tensor_1=None, input_tensor_2=None):
    """Return similarity transformed image input_tensor_2 and transformation parameters.
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
    ### TODO: write a dictionary of parameters: difference_of_gaussians_sigma_tuple, fft_highpass_radius,
    # log-polar transform parameters like angles_vec_to_map and log-radius jumps and log-base etc'

    ### Save copies of the original images for checkup: ###
    input_tensor_1_original = np.copy(input_tensor_1)
    input_tensor_2_original = np.copy(input_tensor_2)

    if input_tensor_1.shape != input_tensor_2.shape:
        raise ValueError('images must have same shapes')
    if len(input_tensor_1.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_tensor_1 = difference_of_gaussians(input_tensor_1, 5, 20)  # TODO: the more you smear the better?
    input_tensor_2 = difference_of_gaussians(input_tensor_2, 5, 20)

    input_tensor_1_fft_abs = fftshift(abs(fft2(input_tensor_1)))
    input_tensor_2_fft_abs = fftshift(abs(fft2(input_tensor_2)))

    h = highpass(input_tensor_1_fft_abs.shape)
    input_tensor_1_fft_abs *= h
    input_tensor_2_fft_abs *= h
    del h

    #TODO: test this out once and for-all and delete all the repeated stuff later
    #TODO: understand the difference between logpolar_torch and lopolar_scipy_torch

    # ### Numpy LogPolar: ###
    # input_tensor_1_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar(input_tensor_1_fft_abs)
    # input_tensor_2_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar(input_tensor_2_fft_abs)
    # ### Torch LogPolar: ###
    # input_tensor_1_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_tensor_1_fft_abs).unsqueeze(0).unsqueeze(0))
    # input_tensor_2_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_tensor_2_fft_abs).unsqueeze(0).unsqueeze(0))
    # imshow(input_tensor_1_fft_abs_LogPolar_numpy); figure(); imshow_torch(input_tensor_1_fft_abs_LogPolar_torch)
    ### Scipy-Torch LogPolar: ###
    input_tensor_1_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_tensor_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_tensor_2_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_tensor_2_fft_abs).unsqueeze(0).unsqueeze(0))
    input_tensor_1_fft_abs_LogPolar_numpy = input_tensor_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_tensor_2_fft_abs_LogPolar_numpy = input_tensor_2_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    # imshow(input_tensor_1_fft_abs_LogPolar_numpy); figure(); imshow_torch(input_tensor_1_fft_abs_LogPolar_torch)
    # grid_diff = x - x_torch.cpu().numpy()[0,:,:,0]
    # ### Scipy LogPolar: ###
    # input_shape = input_tensor_1_fft_abs.shape
    # radius_to_use = input_shape[0] // 8  # only take lower frequencies. #TODO: Notice! the choice of the log-polar conversion is for your own choosing!!!!!!
    # input_tensor_1_fft_abs_LogPolar_numpy = warp_polar(input_tensor_1_fft_abs, radius=radius_to_use, output_shape=input_shape, scaling='log', order=0)
    # input_tensor_2_fft_abs_LogPolar_numpy = warp_polar(input_tensor_2_fft_abs, radius=radius_to_use, output_shape=input_shape, scaling='log', order=0)
    # ### External LopPolar: ###
    # # logpolar_grid_tensor = logpolar_grid(input_tensor_1_fft_abs.shape, ulim=(None, np.log(2.) / 2.), vlim=(0, np.pi), out=None, device='cuda').cuda()
    # logpolar_grid_tensor = logpolar_grid(input_tensor_1_fft_abs.shape, ulim=(np.log(0.01), np.log(1)), vlim=(0, np.pi), out=None, device='cuda').cuda()
    # input_tensor_1_fft_abs_LogPolar_torch = F.grid_sample(torch.Tensor(input_tensor_1_fft_abs).unsqueeze(0).unsqueeze(0).cuda(), logpolar_grid_tensor.unsqueeze(0))
    # input_tensor_1_fft_abs_LogPolar_numpy = input_tensor_1_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
    # imshow_torch(input_tensor_1_fft_abs_LogPolar_torch)

    input_tensor_1_fft_abs_LogPolar_numpy = fft2(input_tensor_1_fft_abs_LogPolar_numpy)
    input_tensor_2_fft_abs_LogPolar_numpy = fft2(input_tensor_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_tensor_1_fft_abs_LogPolar_numpy) * abs(input_tensor_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(
        ifft2((input_tensor_1_fft_abs_LogPolar_numpy * input_tensor_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(ifft2((input_tensor_2_fft_abs_LogPolar_numpy * input_tensor_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
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

    input_tensor_2_scaled_rotated = ndii.zoom(input_tensor_2_original, 1.0 / scale)
    input_tensor_2_scaled_rotated = ndii.rotate(input_tensor_2_scaled_rotated, -angle)

    if input_tensor_2_scaled_rotated.shape < input_tensor_1.shape:
        t = np.zeros_like(input_tensor_1)
        t[: input_tensor_2_scaled_rotated.shape[0],
        : input_tensor_2_scaled_rotated.shape[1]] = input_tensor_2_scaled_rotated
        input_tensor_2_scaled_rotated = t
    elif input_tensor_2_scaled_rotated.shape > input_tensor_1.shape:
        input_tensor_2_scaled_rotated = input_tensor_2_scaled_rotated[: input_tensor_1.shape[0], : input_tensor_1.shape[1]]

    # imshow(input_tensor_2_scaled_rotated); figure(); imshow(input_tensor_1_original)
    # imshow(input_tensor_2_scaled_rotated - input_tensor_1_original)

    ### Crop Images Before Translational Cross Correlation: ###
    #TODO: important! understand how much to cut tensors before cross correlation automatically, if at all
    input_tensor_1_original_crop = crop_numpy_batch(input_tensor_1_original, 800)
    input_tensor_2_scaled_rotated_crop = crop_numpy_batch(input_tensor_2_scaled_rotated, 800)
    # imshow(input_tensor_2_scaled_rotated_crop); figure(); imshow(input_tensor_1_original_crop)
    # imshow(input_tensor_2_scaled_rotated_crop - input_tensor_1_original_crop)

    input_tensor_1_fft = fft2(input_tensor_1_original_crop)
    input_tensor_2_fft = fft2(input_tensor_2_scaled_rotated_crop)
    # TODO: probably need to center crop images here!
    phase_cross_correlation = abs(
        ifft2((input_tensor_1_fft * input_tensor_2_fft.conjugate()) / (abs(input_tensor_1_fft) * abs(input_tensor_2_fft))))
    # phase_cross_correlation = abs(ifft2((input_tensor_1_fft * input_tensor_2_fft.conjugate())))
    t0, t1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)

    if t0 > input_tensor_1_fft_abs.shape[0] // 2:
        t0 -= input_tensor_1_fft.shape[0]
    if t1 > input_tensor_1_fft_abs.shape[1] // 2:
        t1 -= input_tensor_1_fft.shape[1]

    input_tensor_2_scaled_rotated_shifted = ndii.shift(input_tensor_2_scaled_rotated, [t0, t1])
    # figure(); imshow(input_tensor_2_scaled_rotated_shifted); figure(); imshow(input_tensor_1_original_crop)
    # figure(); imshow(input_tensor_2_scaled_rotated_shifted - input_tensor_1_original_crop)

    # correct parameters for ndimage's internal processing
    if angle > 0.0:
        d = int(int(input_tensor_2.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(input_tensor_2.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (input_tensor_2.shape[1] - 1) / (int(input_tensor_2.shape[1] / scale) - 1)

    return input_tensor_2_scaled_rotated_shifted, scale, angle, [-t0, -t1]


def similarity_RotationScalingTranslation_numpy_2(input_tensor_1=None, input_tensor_2=None):
    """Return similarity transformed image input_tensor_2 and transformation parameters.
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

    input_tensor_1_original = np.copy(input_tensor_1)
    input_tensor_2_original = np.copy(input_tensor_2)
    # imshow(input_tensor_1_original); figure(); imshow(input_tensor_2_original)

    if input_tensor_1.shape != input_tensor_2.shape:
        raise ValueError('images must have same shapes')
    if len(input_tensor_1.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_tensor_1 = difference_of_gaussians(input_tensor_1, 5, 20)  # TODO: the more you smear the better?
    input_tensor_2 = difference_of_gaussians(input_tensor_2, 5, 20)

    input_tensor_1_fft_abs = fftshift(abs(fft2(input_tensor_1)))
    input_tensor_2_fft_abs = fftshift(abs(fft2(input_tensor_2)))

    h = highpass(input_tensor_1_fft_abs.shape)
    input_tensor_1_fft_abs *= h
    input_tensor_2_fft_abs *= h
    del h

    ### Scipy-Torch LogPolar: ###
    # TODO: add possibility of divide factor for radius
    input_tensor_1_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_tensor_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_tensor_2_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (
    x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_tensor_2_fft_abs).unsqueeze(0).unsqueeze(0))
    input_tensor_1_fft_abs_LogPolar_numpy = input_tensor_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_tensor_2_fft_abs_LogPolar_numpy = input_tensor_2_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]

    ### Calculate Cross Correlation: ###
    input_tensor_1_fft_abs_LogPolar_numpy = fft2(input_tensor_1_fft_abs_LogPolar_numpy)
    input_tensor_2_fft_abs_LogPolar_numpy = fft2(input_tensor_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_tensor_1_fft_abs_LogPolar_numpy) * abs(input_tensor_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(
        ifft2((input_tensor_1_fft_abs_LogPolar_numpy * input_tensor_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(
            ifft2((input_tensor_2_fft_abs_LogPolar_numpy * input_tensor_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
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

    ### TODO: if to find the rotation and scaling i am actually finding the cross correlation peak of the fft_abs_LogPolar, why can i use that and
    ### TODO: shift the fft_LogPolar just like i do the when finding pure translation, and then use the inverse_LogPolar transform (i'm sure it exists)?????
    # TODO: isn't the above BETTER then rotating the original image around an unknown center of rotation?!?!?!....

    ### Warp Second Image Towards Reference Image: ###
    # #(1). Scipy:
    input_tensor_2_scaled_rotated = ndii.zoom(input_tensor_2_original, 1.0 / scale)
    input_tensor_2_scaled_rotated = ndii.rotate(input_tensor_2_scaled_rotated, -angle)
    # (2). Torch Affine:
    input_tensor_1_original_torch = torch.Tensor(input_tensor_1_original).unsqueeze(0).unsqueeze(0)
    input_tensor_2_original_torch = torch.Tensor(input_tensor_2_original).unsqueeze(0).unsqueeze(0)
    input_tensor_2_original_torch_rotated = warp_tensor_affine_matrix(input_tensor_2_original_torch, 0, 0, scale, -angle)
    # imshow_torch(input_tensor_1_original_torch - input_tensor_2_original_torch_rotated)
    ### Crop: ###
    input_tensor_1_original = input_tensor_1_original_torch.numpy()[0, 0]
    input_tensor_2_scaled_rotated = input_tensor_2_original_torch_rotated.numpy()[0, 0]
    # imshow(input_tensor_1_original - input_tensor_2_scaled_rotated)

    if input_tensor_2_scaled_rotated.shape < input_tensor_1.shape:
        t = np.zeros_like(input_tensor_1)
        t[: input_tensor_2_scaled_rotated.shape[0],
        : input_tensor_2_scaled_rotated.shape[1]] = input_tensor_2_scaled_rotated
        input_tensor_2_scaled_rotated = t
    elif input_tensor_2_scaled_rotated.shape > input_tensor_1.shape:
        input_tensor_2_scaled_rotated = input_tensor_2_scaled_rotated[: input_tensor_1.shape[0], : input_tensor_1.shape[1]]

    # imshow(input_tensor_2_scaled_rotated); figure(); imshow(input_tensor_1_original)
    # imshow(input_tensor_2_scaled_rotated - input_tensor_1_original)

    ### Crop Images Before Translational Cross Correlation: ###
    # input_tensor_1_original_crop = crop_numpy_batch(input_tensor_1_original, 800)
    # input_tensor_2_scaled_rotated_crop = crop_numpy_batch(input_tensor_2_scaled_rotated, 800)
    input_tensor_1_original_crop = input_tensor_1_original
    input_tensor_2_scaled_rotated_crop = input_tensor_2_scaled_rotated
    # imshow(input_tensor_2_scaled_rotated_crop); figure(); imshow(input_tensor_1_original_crop)
    # imshow(input_tensor_2_scaled_rotated_crop - input_tensor_1_original_crop)

    input_tensor_1_fft = fft2(input_tensor_1_original_crop)
    input_tensor_2_fft = fft2(input_tensor_2_scaled_rotated_crop)
    # TODO: probably need to center crop images here!
    phase_cross_correlation = abs(
        ifft2((input_tensor_1_fft * input_tensor_2_fft.conjugate()) / (abs(input_tensor_1_fft) * abs(input_tensor_2_fft))))
    # phase_cross_correlation = abs(ifft2((input_tensor_1_fft * input_tensor_2_fft.conjugate())))
    t0, t1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)

    if t0 > input_tensor_1_fft_abs.shape[0] // 2:
        t0 -= input_tensor_1_fft.shape[0]
    if t1 > input_tensor_1_fft_abs.shape[1] // 2:
        t1 -= input_tensor_1_fft.shape[1]

    input_tensor_2_scaled_rotated_shifted = ndii.shift(input_tensor_2_scaled_rotated, [t0, t1])
    # figure(); imshow(input_tensor_2_scaled_rotated_shifted); figure(); imshow(input_tensor_1_original_crop)
    # figure(); imshow(input_tensor_2_scaled_rotated_shifted - input_tensor_1_original_crop)

    # correct parameters for ndimage's internal processing
    ### TODO: it does bring the images one on top of each other....but it doesn't give the correct [t0,t1]....meaning something's wrong perhapse below?
    if angle > 0.0:
        d = int(int(input_tensor_2.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(input_tensor_2.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (input_tensor_2.shape[1] - 1) / (int(input_tensor_2.shape[1] / scale) - 1)

    return input_tensor_2_scaled_rotated_shifted, scale, angle, [-t0, -t1]


def scipy_Rotation_Scaling_registration(input_tensor_1, input_tensor_2):

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_tensor_1 = difference_of_gaussians(input_tensor_1, 5, 20)  # TODO: the more you smear the better?
    input_tensor_2 = difference_of_gaussians(input_tensor_2, 5, 20)

    ### Window Images To Avoid Effects From Image Edges: ###
    # (*). Probably not as important!!!!
    input_tensor_1_windowed = input_tensor_1 * window('hann', input_tensor_1.shape)
    input_tensor_2_windowed = input_tensor_2 * window('hann', input_tensor_1.shape)

    ### Work With Shifted FFT Magnitudes: ###
    input_tensor_1_FFT_abs = np.abs(fftshift(fft2(input_tensor_1_windowed)))
    input_tensor_2_FFT_abs = np.abs(fftshift(fft2(input_tensor_2_windowed)))

    ### Create Log-Polar Transformed FFT Mag Images and Register: ###
    shape = input_tensor_1_FFT_abs.shape
    radius = shape[0] // 1  # only take lower frequencies
    input_tensor_1_FFT_abs_LogPolar = warp_polar(input_tensor_1_FFT_abs, radius=radius, output_shape=shape, scaling='log',
                                                order=0)
    input_tensor_2_FFT_abs_LogPolar = warp_polar(input_tensor_2_FFT_abs, radius=radius, output_shape=shape, scaling='log',
                                                order=0)

    # #TODO: delete
    # input_tensor_1_FFT_abs_torch = torch.Tensor(input_tensor_1_FFT_abs).unsqueeze(0).unsqueeze(0)
    # input_tensor_2_FFT_abs_torch = torch.Tensor(input_tensor_2_FFT_abs).unsqueeze(0).unsqueeze(0)
    # bla_1, log_base, (radius2, theta2), (x,y), flow_grid = logpolar_scipy_torch(input_tensor_1_FFT_abs_torch)
    # bla_2, log_base, (radius2, theta2), (x,y), flow_grid = logpolar_scipy_torch(input_tensor_2_FFT_abs_torch)

    ### TODO: this works!!!!! - duplicate this "cutting half" process to all the other functions!!!!!
    # input_tensor_1_FFT_abs_LogPolar = bla_1.cpu().numpy()[0,0]
    # input_tensor_2_FFT_abs_LogPolar = bla_2.cpu().numpy()[0,0]
    # imshow(input_tensor_1_FFT_abs_LogPolar); imshow_torch(bla_1)
    # TODO: perhapse i should also use on half the FFT and use fftshift as in the above example?!??!?!
    input_tensor_1_FFT_abs_LogPolar = input_tensor_1_FFT_abs_LogPolar[:shape[0] // 2, :]  # only use half of FFT
    input_tensor_2_FFT_abs_LogPolar = input_tensor_2_FFT_abs_LogPolar[:shape[0] // 2, :]
    shifts, error, phasediff = skimage_phase_cross_correlation(input_tensor_1_FFT_abs_LogPolar,
                                                               input_tensor_2_FFT_abs_LogPolar,
                                                               upsample_factor=10)

    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = np.exp(shiftc / klog)

    ### TODO: add option of finding translation afterwards
    ### TODO: encapsulate this in a function which finds rotation and scaling robustly!!!!

    # fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    # ax = axes.ravel()
    # ax[0].set_title("Original Image FFT\n(magnitude; zoomed)")
    # center = np.array(shape) // 2
    # ax[0].imshow(input_tensor_1_FFT_abs[center[0] - radius:center[0] + radius,
    #              center[1] - radius:center[1] + radius],
    #              cmap='magma')
    # ax[1].set_title("Modified Image FFT\n(magnitude; zoomed)")
    # ax[1].imshow(input_tensor_2_FFT_abs[center[0] - radius:center[0] + radius,
    #              center[1] - radius:center[1] + radius],
    #              cmap='magma')
    # ax[2].set_title("Log-Polar-Transformed\nOriginal FFT")
    # ax[2].imshow(input_tensor_1_FFT_abs_LogPolar, cmap='magma')
    # ax[3].set_title("Log-Polar-Transformed\nModified FFT")
    # ax[3].imshow(input_tensor_2_FFT_abs_LogPolar, cmap='magma')
    # fig.suptitle('Working in frequency domain can recover rotation and scaling')
    # plt.show()

    return recovered_angle, shift_scale

def torch_2D_hann_window(window_shape):
    hann_tensor = torch.Tensor(window('hann', (window_shape[2],window_shape[3]))).unsqueeze(0).unsqueeze(0)
    return hann_tensor

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

    def forward(self, input_tensor_1, input_tensor_2):
        ### TODO: !!!!!!!: ###
        #(1). Play with the gaussian filter parameters of the high and low sigmas and the kernel size etc'
        #(2). play with the LogPolar transform. maybe make it LinearPolar, maybe change the range of angles to a very small one, maybe use something other the nearest neighbor

        ### First, Band-Pass Filter Both Images: ###
        # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
        input_tensor_1_DoG = self.gaussian_blur_layer_low_sigma(input_tensor_1) - self.gaussian_blur_layer_high_sigma(input_tensor_1)
        input_tensor_2_DoG = self.gaussian_blur_layer_low_sigma(input_tensor_2) - self.gaussian_blur_layer_high_sigma(input_tensor_2)

        ### Window Images To Avoid Effects From Image Edges: ###
        if self.flag_use_hann_window:
            if self.hann_window is None:
                self.hann_window = torch_2D_hann_window(input_tensor_1_DoG.shape).cuda()
            input_tensor_1_windowed = input_tensor_1_DoG * self.hann_window
            input_tensor_2_windowed = input_tensor_2_DoG * self.hann_window
        else:
            input_tensor_1_windowed = input_tensor_1_DoG
            input_tensor_2_windowed = input_tensor_2_DoG

        ### Work With Shifted FFT Magnitudes: ###
        input_tensor_1_window_FFT = torch_fftshift(torch_fft2(input_tensor_1_windowed))
        input_tensor_2_window_FFT = torch_fftshift(torch_fft2(input_tensor_2_windowed))
        input_tensor_1_FFT_abs_torch = torch.abs(input_tensor_1_window_FFT)
        input_tensor_2_FFT_abs_torch = torch.abs(input_tensor_2_window_FFT)

        ### Create Log-Polar Transformed FFT Mag Images and Register: ###
        shape = (input_tensor_1_FFT_abs_torch.shape[2], input_tensor_1_FFT_abs_torch.shape[3])
        radius = shape[0] // self.radius_div_factor  # only take lower frequencies
        if self.flow_grid is None:
            input_tensor_1_FFT_abs_LogPolar, log_base, (radius2, theta2), (x, y), self.flow_grid = torch_LogPolar_transform_ScipyFormat(input_tensor_1_FFT_abs_torch, self.radius_div_factor)
            input_tensor_2_FFT_abs_LogPolar = nn.functional.grid_sample(input_tensor_2_FFT_abs_torch.cuda(), self.flow_grid, mode='nearest')
        else:
            input_tensor_1_FFT_abs_LogPolar = nn.functional.grid_sample(input_tensor_1_FFT_abs_torch.cuda(), self.flow_grid, mode='nearest')
            input_tensor_2_FFT_abs_LogPolar = nn.functional.grid_sample(input_tensor_2_FFT_abs_torch.cuda(), self.flow_grid, mode='nearest')

        ### Get Cross Correlation & Shifts: ###
        input_tensor_1_FFT_abs_LogPolar = input_tensor_1_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]  # only use half of FFT
        input_tensor_2_FFT_abs_LogPolar = input_tensor_2_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]
        cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(input_tensor_1_FFT_abs_LogPolar,
                                                                                                                  input_tensor_2_FFT_abs_LogPolar)

        ### Use translation parameters to calculate rotation and scaling parameters: ###
        # shiftr, shiftc = shifts[:2]
        shiftr, shiftc = shifts_sub_pixel[:2]
        recovered_angle = (360 / shape[0]) * shiftr
        klog = shape[1] / np.log(radius)
        recovered_scale = torch.exp(shiftc / klog)
        recovered_scale = (input_tensor_2.shape[-1] - 1) / (int(input_tensor_2.shape[-1] / recovered_scale) - 1)

        return recovered_angle, recovered_scale, input_tensor_1_window_FFT, input_tensor_2_window_FFT

def get_FFT_LogPolar_Rotation_Scaling_torch(input_tensor_1, input_tensor_2, radius_div_factor=8):

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    #TODO: this doesn't work exactly like difference_of_gaussians from skimage...understand what's the difference?...perhapse kernel_size, perhapse gaussian definition?...
    gaussian_blur_layer_low_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=5**2, dim=2)
    gaussian_blur_layer_high_sigma = Gaussian_Blur_Layer(channels=1, kernel_size=25, sigma=20**2, dim=2)
    input_tensor_1_DoG = gaussian_blur_layer_low_sigma(input_tensor_1) - gaussian_blur_layer_high_sigma(input_tensor_1)
    input_tensor_2_DoG = gaussian_blur_layer_low_sigma(input_tensor_2) - gaussian_blur_layer_high_sigma(input_tensor_2)
    # input_tensor_1_DoG = difference_of_gaussians(input_tensor_1, 5, 20)  # TODO: the more you smear the better?
    # input_tensor_2_DoG = difference_of_gaussians(input_tensor_2, 5, 20)

    ### Window Images To Avoid Effects From Image Edges: ###
    # (*). Probably not as important!!!!
    input_tensor_1_windowed = input_tensor_1_DoG * torch_2D_hann_window(input_tensor_1_DoG.shape)
    input_tensor_2_windowed = input_tensor_2_DoG * torch_2D_hann_window(input_tensor_1_DoG.shape)

    ### Work With Shifted FFT Magnitudes: ###
    input_tensor_1_FFT_abs_torch = torch.abs(torch_fftshift(torch_fft2(input_tensor_1_windowed)))
    input_tensor_2_FFT_abs_torch = torch.abs(torch_fftshift(torch_fft2(input_tensor_2_windowed)))

    ### Create Log-Polar Transformed FFT Mag Images and Register: ###
    shape = (input_tensor_1_FFT_abs_torch.shape[2], input_tensor_1_FFT_abs_torch.shape[3])
    radius = shape[0] // radius_div_factor  # only take lower frequencies
    input_tensor_1_FFT_abs_LogPolar, log_base, (radius2, theta2), (x, y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_tensor_1_FFT_abs_torch, radius_div_factor)
    input_tensor_2_FFT_abs_LogPolar, log_base, (radius2, theta2), (x, y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_tensor_2_FFT_abs_torch, radius_div_factor)

    ### Get Cross Correlation & Shifts: ###
    input_tensor_1_FFT_abs_LogPolar = input_tensor_1_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]  # only use half of FFT
    input_tensor_2_FFT_abs_LogPolar = input_tensor_2_FFT_abs_LogPolar[:, :, :shape[0] // 2, :]
    ### Numpy Version: ###
    cross_correlation, shifts, shifts_sub_pixel = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(input_tensor_1_FFT_abs_LogPolar,
                                                                                                              input_tensor_2_FFT_abs_LogPolar)

    ### Use translation parameters to calculate rotation and scaling parameters: ###
    # shiftr, shiftc = shifts[:2]
    shiftr, shiftc = shifts_sub_pixel[:2]
    recovered_angle = (360 / shape[0]) * shiftr
    klog = shape[1] / np.log(radius)
    shift_scale = torch.exp(shiftc / klog)

    shift_scale = (input_tensor_2.shape[-1] - 1) / (int(input_tensor_2.shape[-1] / shift_scale) - 1)
    # print(f"Expected value for cc rotation in degrees: {rotation_angle}")
    # print(f"Recovered value for cc rotation: {recovered_angle}")
    # print()
    # print(f"Expected value for scaling difference: {scale}")
    # print(f"Recovered value for scaling difference: {shift_scale}")

    return recovered_angle, shift_scale

def get_FFT_LogPolar_Rotation_Scaling_scipy(input_tensor_1, input_tensor_2):
    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_tensor_1_DoG = difference_of_gaussians(input_tensor_1, 5, 20)  # TODO: the more you smear the better?
    input_tensor_2_DoG = difference_of_gaussians(input_tensor_2, 5, 20)

    ### Window Images To Avoid Effects From Image Edges: ###
    # (*). Probably not as important!!!!
    input_tensor_1_windowed = input_tensor_1_DoG * window('hann', input_tensor_1_DoG.shape)
    input_tensor_2_windowed = input_tensor_2_DoG * window('hann', input_tensor_1_DoG.shape)

    ### Work With Shifted FFT Magnitudes: ###
    input_tensor_1_FFT_abs = np.abs(fftshift(fft2(input_tensor_1_windowed)))
    input_tensor_2_FFT_abs = np.abs(fftshift(fft2(input_tensor_2_windowed)))

    ### Create Log-Polar Transformed FFT Mag Images and Register: ###
    shape = input_tensor_1_FFT_abs.shape
    radius_div_factor = 16
    radius = shape[0] // radius_div_factor  # only take lower frequencies
    ### Scipy Version: ###
    input_tensor_1_FFT_abs_LogPolar = scipy_LogPolar_transform(input_tensor_1_FFT_abs, radius_div_factor)
    input_tensor_2_FFT_abs_LogPolar = scipy_LogPolar_transform(input_tensor_2_FFT_abs, radius_div_factor)
    # ### Torch Version: ###
    # input_tensor_1_FFT_abs_torch = torch.Tensor(input_tensor_1_FFT_abs).unsqueeze(0).unsqueeze(0)
    # input_tensor_2_FFT_abs_torch = torch.Tensor(input_tensor_2_FFT_abs).unsqueeze(0).unsqueeze(0)
    # input_tensor_1_FFT_abs_LogPolar_torch, log_base, (radius2, theta2), (x,y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_tensor_1_FFT_abs_torch, radius_div_factor)
    # input_tensor_2_FFT_abs_LogPolar_torch, log_base, (radius2, theta2), (x,y), flow_grid = torch_LogPolar_transform_ScipyFormat(input_tensor_2_FFT_abs_torch, radius_div_factor)
    # input_tensor_1_FFT_abs_LogPolar = input_tensor_1_FFT_abs_LogPolar_torch.cpu().numpy()[0,0]
    # input_tensor_2_FFT_abs_LogPolar = input_tensor_2_FFT_abs_LogPolar_torch.cpu().numpy()[0,0]

    # TODO: perhapse i should also use on half the FFT and use fftshift as in the above example?!??!?!
    input_tensor_1_FFT_abs_LogPolar = input_tensor_1_FFT_abs_LogPolar[:shape[0] // 2, :]  # only use half of FFT
    input_tensor_2_FFT_abs_LogPolar = input_tensor_2_FFT_abs_LogPolar[:shape[0] // 2, :]
    ## SKimage Version: ###
    shifts, error, phasediff = skimage_phase_cross_correlation(input_tensor_1_FFT_abs_LogPolar,
                                                               input_tensor_2_FFT_abs_LogPolar,
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
                                                                                            numpy_to_torch(input_tensor_1_FFT_abs_LogPolar).unsqueeze(0),
                                                                                            numpy_to_torch(input_tensor_2_FFT_abs_LogPolar).unsqueeze(0))

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

def get_FFT_LogPolar_Rotation_Scaling_Translation_numpy(input_tensor_1, input_tensor_2):
    #TODO: this DOES NOT properly do Rotation-Scaling registration,
    # but rather uses the scipy version of Rotation-Scaling and then does a cross-correlation plus some ed-hok error corrections which don't work!!!!

    ### Prepare Data: ###
    #OHHHHHH - very important!!!! in warp_tensors_affine i translate->scale->rotate!!!! this explains why i get the tensors on top of each other but doesn't get the
    # "correct" translation....because it's not "correct"!!!!!.... i still need to think about the delta_r correction factor below

    ### Get Angle & Scale: ###
    (rotation_angle, scale) = get_FFT_LogPolar_Rotation_Scaling_scipy(input_tensor_1, input_tensor_2) #this uses PHASE cross correlation which is bad
    # print(rotation_angle)

    # ### Rescale & Rotate Second Image Scipy: ###
    # input_tensor_2_scaled_rotated = ndii.zoom(input_tensor_2, 1/scale)
    # input_tensor_2_scaled_rotated = ndii.rotate(input_tensor_2_scaled_rotated, -rotation_angle)
    # ### Equalize Image Sizes After Scipy Zoom & Rotate Operations (which change image size): ###
    # if input_tensor_2_scaled_rotated.shape < input_tensor_1.shape:
    #     t = numpy.zeros_like(input_tensor_1)
    #     t[: input_tensor_2_scaled_rotated.shape[0],
    #     : input_tensor_2_scaled_rotated.shape[1]] = input_tensor_2_scaled_rotated
    #     input_tensor_2_scaled_rotated = t
    # elif input_tensor_2_scaled_rotated.shape > input_tensor_1.shape:
    #     input_tensor_2_scaled_rotated = input_tensor_2_scaled_rotated[: input_tensor_1.shape[0], : input_tensor_1.shape[1]]
    # imshow(input_tensor_2_scaled_rotated); figure(); imshow(input_tensor_1)
    # imshow(input_tensor_2_scaled_rotated - input_tensor_1)


    ### Rescale & Rotate Second Image Torch: ###
    input_tensor_2_torch = torch.Tensor(input_tensor_2).unsqueeze(0).unsqueeze(0)
    input_tensor_2_scaled_rotated = warp_tensors_affine(input_tensor_2_torch, np.array([0]), np.array([0]), np.array([1/scale]), np.array([-rotation_angle]))
    input_tensor_2_scaled_rotated = input_tensor_2_scaled_rotated.cpu().numpy()[0,0,:,:]
    # imshow(input_tensor_2_scaled_rotated); figure(); imshow(input_tensor_1)
    # imshow(input_tensor_2_scaled_rotated - input_tensor_1)

    ### Crop Images Before Translational Cross Correlation: ###
    # input_tensor_1_original_crop = crop_numpy_batch(input_tensor_1, np.inf)
    # input_tensor_2_scaled_rotated_crop = crop_numpy_batch(input_tensor_2_scaled_rotated, np.inf)
    input_tensor_1_original_crop = crop_numpy_batch(input_tensor_1, 800)
    input_tensor_2_scaled_rotated_crop = crop_numpy_batch(input_tensor_2_scaled_rotated, 800)
    # imshow(input_tensor_2_scaled_rotated_crop); figure(); imshow(input_tensor_1_original_crop)
    # imshow(input_tensor_2_scaled_rotated_crop - input_tensor_1_original_crop)

    input_tensor_1_fft = fft2(input_tensor_1_original_crop)
    input_tensor_2_fft = fft2(input_tensor_2_scaled_rotated_crop)
    # TODO: probably need to center crop images here!
    # phase_cross_correlation = abs(ifft2((input_tensor_1_fft * input_tensor_2_fft.conjugate()) / (abs(input_tensor_1_fft) * abs(input_tensor_2_fft))))
    phase_cross_correlation = abs(ifft2((input_tensor_1_fft * input_tensor_2_fft.conjugate())))
    # phase_cross_correlation = abs(ifft2((input_tensor_1_fft * input_tensor_2_fft.conjugate())))
    delta_y, delta_x = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)

    if delta_y > input_tensor_1_fft.shape[0] // 2:
        delta_y -= input_tensor_1_fft.shape[0]
    if delta_x > input_tensor_2_fft.shape[1] // 2:
        delta_x -= input_tensor_2_fft.shape[1]

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

    input_tensor_2_scaled_rotated_shifted = ndii.shift(input_tensor_2_scaled_rotated_crop, [delta_y, delta_x])
    # figure(); imshow(input_tensor_2_scaled_rotated_shifted); figure(); imshow(input_tensor_1_original_crop)
    # figure(); imshow(input_tensor_2_scaled_rotated_shifted - input_tensor_1_original_crop)

    # correct parameters for ndimage's internal processing
    if rotation_angle > 0.0:
        d = int(int(input_tensor_2.shape[1] / scale) * math.sin(math.radians(rotation_angle)))
        delta_y, delta_x = delta_x, d + delta_y
    elif rotation_angle < 0.0:
        d = int(int(input_tensor_2.shape[0] / scale) * math.sin(math.radians(rotation_angle)))
        delta_y, delta_x = d + delta_x, d + delta_y
    scale = (input_tensor_2.shape[1] - 1) / (int(input_tensor_2.shape[1] / scale) - 1)

    return rotation_angle, scale, (delta_x, delta_y), input_tensor_2_scaled_rotated_shifted

def get_FFT_LogPolar_Rotation_Scaling_Translation_torch(input_tensor_1, input_tensor_2):

    ### Get Angle & Scale: ###
    (rotation_angle, scale) = get_FFT_LogPolar_Rotation_Scaling_torch(input_tensor_1, input_tensor_2)

    ### Rescale & Rotate Second Image Torch: ###
    input_tensor_2_scaled_rotated = warp_tensors_affine(input_tensor_2, [0], [0], [1/scale], [-rotation_angle])
    # imshow(input_tensor_2_scaled_rotated); figure(); imshow(input_tensor_1)
    # imshow(input_tensor_2_scaled_rotated - input_tensor_1)

    ### Crop Images Before Translational Cross Correlation: ###
    # input_tensor_1_original_crop = crop_numpy_batch(input_tensor_1, np.inf)
    # input_tensor_2_scaled_rotated_crop = crop_numpy_batch(input_tensor_2_scaled_rotated, np.inf)
    input_tensor_1_original_crop = crop_torch_batch(input_tensor_1, 800)
    input_tensor_2_scaled_rotated_crop = crop_torch_batch(input_tensor_2_scaled_rotated, 800)
    # imshow(input_tensor_2_scaled_rotated_crop); figure(); imshow(input_tensor_1_original_crop)
    # imshow(input_tensor_2_scaled_rotated_crop - input_tensor_1_original_crop)

    ### Get Cross Correlation & Shift: ###
    cross_correlation_mat, shifts = get_Circular_Cross_Correlation_and_Shifts_DiscreteShift_torch(input_tensor_1_original_crop, input_tensor_2_scaled_rotated_crop)

    ### Shift Second Image To Be On Top Of First One: ###
    input_tensor_2_scaled_rotated_crop_translated = shift_matrix_subpixel_torch(input_tensor_2_scaled_rotated_crop, shifts[1], shifts[0])
    imshow_torch(input_tensor_2_scaled_rotated_crop_translated - input_tensor_1_original_crop)

    return rotation_angle, scale, shifts

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

    def forward(self, input_tensor_1, input_tensor_2, downsample_factor=1, flag_return_shifted_image=True, flag_interpolation_mode='bilinear'):
        ### DownSample Input Images If Wanted: ###
        if self.downsample_factor is None:
            self.downsample_factor = downsample_factor
            self.downsample_layer = nn.AvgPool2d(self.downsample_factor)
        if self.downsample_factor > 1:
            input_tensor_1_downsampled = self.downsample_layer(input_tensor_1)
            input_tensor_2_downsampled = self.downsample_layer(input_tensor_2)
        else:
            input_tensor_1_downsampled = input_tensor_1
            input_tensor_2_downsampled = input_tensor_2

        ### Get Angle & Scale: ###
        (recovered_angle, recovered_scale, input_image_window_FFT_1, input_image_window_FFT_2) =\
            self.fft_logpolar_rotation_scaling_registration_layer(input_tensor_1_downsampled, input_tensor_2_downsampled)

        ### Rescale & Rotate Second Image Torch: ###
        input_tensor_2_scaled_rotated = self.warp_tensors_affine_layer.forward(input_tensor_2,
                                                                              np.float32(0), np.float32(0),
                                                                              np.float32(1 / recovered_scale),
                                                                              -recovered_angle,
                                                                              flag_interpolation_mode=flag_interpolation_mode)

        ### Crop Images Before Translational Cross Correlation: ###
        input_tensor_1_original_crop = crop_torch_batch(input_tensor_1, self.inner_crop_after_rotation_and_scaling)
        input_tensor_2_scaled_rotated_crop = crop_torch_batch(input_tensor_2_scaled_rotated, self.inner_crop_after_rotation_and_scaling)
        # input_tensor_1_original_crop = input_tensor_1
        # input_tensor_2_scaled_rotated_crop = input_tensor_2_scaled_rotated

        ### Get Cross Correlation & Shift: ###
        cross_correlation_mat, recovered_translation_discrete, recovered_translation = get_Circular_Cross_Correlation_and_Shifts_ParabolaFit_torch(input_tensor_1_original_crop,
                                                                                                                     input_tensor_2_scaled_rotated_crop)

        if flag_return_shifted_image:
            input_tensor_2_displaced = self.warp_tensors_shift_layer.forward(input_tensor_2_scaled_rotated_crop,
                                                                          recovered_translation[1], recovered_translation[0],
                                                                          fft_image=None)
        else:
            input_tensor_2_displaced = None

        return recovered_angle, recovered_scale, recovered_translation, input_tensor_2_displaced

def test_logpolar_transforms():
    shift_x = np.array([0])
    shift_y = np.array([0])
    scale = np.array([1])
    rotation_angle = np.array([9])  # [degrees]
    input_tensor_1 = read_image_default_torch()
    input_tensor_2 = warp_tensors_affine(input_tensor_1, shift_x, shift_y, scale, rotation_angle)
    input_tensor_1 = input_tensor_1[0, 0].numpy()
    input_tensor_2 = input_tensor_2[0, 0].numpy()
    input_tensor_1 = crop_numpy_batch(input_tensor_1, 1000)
    input_tensor_2 = crop_numpy_batch(input_tensor_2, 1000)
    input_tensor_1_original = input_tensor_1
    input_tensor_2_original = input_tensor_2

    if input_tensor_1.shape != input_tensor_2.shape:
        raise ValueError('images must have same shapes')
    if len(input_tensor_1.shape) != 2:
        raise ValueError('images must be 2 dimensional')

    ### First, Band-Pass Filter Both Images: ###
    # (*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_tensor_1 = difference_of_gaussians(input_tensor_1, 5, 20)  # TODO: the more you smear the better?
    input_tensor_2 = difference_of_gaussians(input_tensor_2, 5, 20)

    input_tensor_1_fft_abs = fftshift(abs(fft2(input_tensor_1)))
    input_tensor_2_fft_abs = fftshift(abs(fft2(input_tensor_2)))

    h = highpass(input_tensor_1_fft_abs.shape)
    input_tensor_1_fft_abs *= h
    input_tensor_2_fft_abs *= h
    del h


    ### Numpy LogPolar: ###
    input_tensor_1_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar_numpy(input_tensor_1_fft_abs)
    input_tensor_2_fft_abs_LogPolar_numpy, log_base, (radius, theta), (x, y) = logpolar_numpy(input_tensor_2_fft_abs)

    ### Torch LogPolar: ###
    input_tensor_1_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), \
    (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_tensor_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_tensor_2_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch),\
    (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_tensor_2_fft_abs).unsqueeze(0).unsqueeze(0))

    ### TODO: logpolar_numpy and logpolar_torch ARE NOT THE SAME!!!!: ###
    # imshow(input_tensor_1_fft_abs_LogPolar_numpy)
    # imshow_torch(input_tensor_1_fft_abs_LogPolar_torch)
    # imshow_torch(input_tensor_1_fft_abs_LogPolar_torch.cpu() - numpy_to_torch(input_tensor_1_fft_abs_LogPolar_numpy).unsqueeze(0))

    ### Scipy/Skimage LogPolar: ###
    input_shape = input_tensor_1_fft_abs.shape
    radius_to_use = input_shape[0] // 1  # only take lower frequencies. #TODO: Notice! the choice of the log-polar conversion is for your own choosing!!!!!!
    input_tensor_1_fft_abs_LogPolar_scipy = warp_polar(input_tensor_1_fft_abs, radius=radius_to_use,
                                                      output_shape=input_shape, scaling='log', order=0)
    input_tensor_2_fft_abs_LogPolar_scipy = warp_polar(input_tensor_2_fft_abs, radius=radius_to_use,
                                                      output_shape=input_shape, scaling='log', order=0)
    # imshow(input_tensor_1_fft_abs_LogPolar_scipy)
    # imshow_torch(input_tensor_1_fft_abs_LogPolar_torch)

    ### External LopPolar: ###
    logpolar_grid_tensor = logpolar_grid(input_tensor_1_fft_abs.shape, ulim=(np.log(0.01), np.log(1)),
                                         vlim=(0, 2 * np.pi), out=None, device='cuda').cuda()
    input_tensor_1_fft_abs_LogPolar_torch_grid = F.grid_sample(torch.Tensor(input_tensor_1_fft_abs).unsqueeze(0).unsqueeze(0).cuda(), logpolar_grid_tensor.unsqueeze(0))
    input_tensor_1_fft_abs_LogPolar_numpy_grid = input_tensor_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    # imshow_torch(input_tensor_1_fft_abs_LogPolar_torch_grid)
    # figure(); imshow(input_tensor_1_fft_abs_LogPolar_scipy)

    ### Calculate Phase Cross Correlation: ###
    input_tensor_1_fft_abs_LogPolar_numpy = input_tensor_1_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_tensor_2_fft_abs_LogPolar_numpy = input_tensor_2_fft_abs_LogPolar_torch.cpu().numpy()[0, 0]
    input_tensor_1_fft_abs_LogPolar_numpy = input_tensor_1_fft_abs_LogPolar_scipy
    input_tensor_2_fft_abs_LogPolar_numpy = input_tensor_2_fft_abs_LogPolar_scipy

    input_tensor_1_fft_abs_LogPolar_numpy = fft2(input_tensor_1_fft_abs_LogPolar_numpy)
    input_tensor_2_fft_abs_LogPolar_numpy = fft2(input_tensor_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_tensor_1_fft_abs_LogPolar_numpy) * abs(input_tensor_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(
        ifft2((input_tensor_1_fft_abs_LogPolar_numpy * input_tensor_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = np.unravel_index(np.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(
            ifft2((input_tensor_2_fft_abs_LogPolar_numpy * input_tensor_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
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


def test_FFTLogPolar_registration():
    ### Paramters: ###
    warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
    shift_x = np.array([0])
    shift_y = np.array([0])
    scale_factor_warp = np.array([1.00])
    rotation_angle = np.array([-2])  # [degrees]
    rotation_angles = my_linspace(-10,11,21)

    ### Prepare Data: ###
    input_tensor_1 = read_image_default_torch()
    input_tensor_1 = RGB2BW(input_tensor_1)
    input_tensor_1 = crop_torch_batch(input_tensor_1, 700).cuda()
    input_tensor_1_numpy = input_tensor_1.cpu().numpy()[0, 0]

    warped_tensors_list_numpy = []
    warped_tensors_list_torch = []
    inferred_rotation_list_1 = []
    inferred_scaling_list_1 = []
    inferred_shift_x_list_1 = []
    inferred_shift_y_list_1 = []
    inferred_rotation_list_2 = []
    inferred_scaling_list_2 = []
    inferred_shift_x_list_2 = []
    inferred_shift_y_list_2 = []

    for i in np.arange(len(rotation_angles)):
        print(i)
        ### Warp Tensor: ###
        rotation_angle = np.array([rotation_angles[i]])
        input_tensor_2, _ = warp_tensors_affine_layer.forward(input_tensor_1, shift_x, shift_y, scale_factor_warp, rotation_angle)
        input_tensor_2 = crop_torch_batch(input_tensor_2, 700)
        input_tensor_2_numpy = input_tensor_2.cpu().numpy()[0,0]
        warped_tensors_list_numpy.append(input_tensor_2_numpy)
        warped_tensors_list_torch.append(input_tensor_2)

        ### Get Parameters: ###
        #(1).
        rotation, scale, translation, input_tensor_2_scaled_rotated_shifted = get_FFT_LogPolar_Rotation_Scaling_Translation_numpy(
            input_tensor_1_numpy, input_tensor_2_numpy)
        inferred_rotation_list_1.append(rotation)
        inferred_scaling_list_1.append(scale)
        inferred_shift_x_list_1.append(translation[0])
        inferred_shift_y_list_1.append(translation[1])
        # #(2).
        fft_logpolar_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer().cuda()
        # recovered_angle, recovered_scale, recovered_translation, input_tensor_1_displaced = fft_logpolar_registration_layer.forward(
        #     input_tensor_1,
        #     input_tensor_2,
        #     downsample_factor=1)
        # inferred_rotation_list_2.append(recovered_angle)
        # inferred_scaling_list_2.append(recovered_scale)
        # inferred_shift_x_list_2.append(recovered_translation[1])
        # inferred_shift_y_list_2.append(recovered_translation[0])

    ### Present Outputs: ###
    #(1). Rotation
    bla = torch.zeros(len(inferred_rotation_list_1))
    for i in np.arange(len(bla)):
        bla[i] = inferred_rotation_list_1[i]
    figure()
    plot(rotation_angles, rotation_angles)
    plot(rotation_angles, inferred_rotation_list_1)
    plot(rotation_angles, (bla.numpy() - rotation_angles))
    # plot(rotation_angles, inferred_rotation_list_2)
    plt.legend(['GT angles', 'numpy function', 'torch layer'])
    plt.title('rotation plot')
    # (1). Translation
    figure()
    plot(rotation_angles, shift_x[0]*np.ones_like(rotation_angles))
    plot(rotation_angles, inferred_shift_x_list_2)
    plt.xlabel('Rotation Angle')
    plt.legend(['GT Shift X', 'torch layer'])
    plt.title('Translation plot')
    figure()
    plot(rotation_angles, shift_y[0] * np.ones_like(rotation_angles))
    plot(rotation_angles, inferred_shift_y_list_2)
    plt.legend(['GT Shift Y', 'torch layer'])
    plt.title('Translation Y plot')
    plt.xlabel('Rotation Angle')


    # imshow_torch_video(torch.cat(warped_tensors_list_torch))  #TODO: speed up this functio

    ### Functions To Check And Compare: ###
    #TODO: as far as i can see the FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer is very close except for the
    # translation stuff!...so probably don't really need all the rest right now except for understanding and testing?
    #
    # get_FFT_LogPolar_Rotation_Scaling_Translation_torch
    # get_FFT_LogPolar_Rotation_Scaling_Translation_numpy
    # get_FFT_LogPolar_Rotation_Scaling_scipy
    # get_FFT_LogPolar_Rotation_Scaling_torch
    # FFT_LogPolar_Rotation_Scaling_Registration_Layer
    # scipy_Rotation_Scaling_registration
    # similarity_RotationScalingTranslation_numpy
    # similarity_RotationScalingTranslation_numpy_2
    # similarity_torch

    # ### Using Numpy Version - works but still not good for rotations<2, the more rotation the more accurate a: ###
    # rotation, scale, translation, input_tensor_2_scaled_rotated_shifted = get_FFT_LogPolar_Rotation_Scaling_Translation_numpy(input_tensor_1_numpy, input_tensor_2_numpy)
    # print(rotation)
    #
    # ### Using pytorch Version - doesn't work the same as numpy version!!!!!! fix it!!!!: ###
    # fft_logpolar_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer().cuda()
    # recovered_angle, recovered_scale, recovered_translation, input_tensor_1_displaced = fft_logpolar_registration_layer.forward(input_tensor_1,
    #                                                                                                                      input_tensor_2,
    #                                                                                                                      downsample_factor=4)
    # print(recovered_translation)
    # print(recovered_scale)
    # print(recovered_angle)
    # print(recovered_angle*input_tensor_1.shape[-1]/360*np.pi)
# test_FFTLogPolar_registration()
###########################################ב###########################################################################################################################################
#
#
#
#
#
#
#
#
######################################################################################################################################################################################
### min-SAD Registration: ###
def get_scale_rotation_translation_minSAD_affine(input_tensor_1, input_tensor_2):
    # ############################################################################################################################################################
    # ### Sizes: ###
    # H_initial = 500  # initial crop size
    # W_initial = 500  # initial crop size
    # H_final = 500
    # W_final = 500
    # initial_index_H_final = np.int((H_initial - H_final) / 2)
    # initial_index_W_final = np.int((W_initial - W_final) / 2)
    # final_index_H_final = initial_index_H_final + H_final
    # final_index_W_final = initial_index_W_final + W_final
    #
    # ### Get Image: ###
    # previous_image = read_image_default_torch()
    # previous_image = RGB2BW(previous_image)
    # previous_image = previous_image[:,:,0:H_initial,0:W_initial].cuda()
    # # previous_image = crop_torch_batch(previous_image, (H_initial, W_initial)).cuda()
    #
    # ### GT parameters: ###
    # GT_shift_x = np.float32(0.4)
    # GT_shift_y = np.float32(-0.5)
    # GT_rotation_angle = np.float32(10)
    # GT_scale = np.float32(1.04)
    #
    # ### Initialize Warp Objects: ###
    # affine_layer_torch = Warp_Tensors_Affine_Layer()
    #
    # ### Warp Image According To Above Parameters: ###
    # current_image = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bilinear')
    # # current_image_2 = affine_layer_torch.forward(previous_image, GT_shift_x, GT_shift_y, GT_scale, GT_rotation_angle, flag_interpolation_mode='bicubic')
    #
    # ### Final Crop: ###
    # # current_image = crop_torch_batch(current_image, (H_final, W_final))
    # # previous_image = crop_torch_batch(previous_image, (H_final, W_final))
    # # current_image = current_image[:, :, 0:H_final, 0:W_final]
    # # previous_image = previous_image[:, :, 0:H_final, 0:W_final]
    # input_tensor_1 = previous_image
    # input_tensor_2 = current_image
    # ############################################################################################################################################################

    ############################################################################################################################################################
    ### Parameters Space: ###
    shifts_vec = [-1, 0, 1]
    rotation_angle_vec = [-1, 0, 1]
    scale_factor_vec = [0.95, 1, 1.05]
    # shifts_vec = my_linspace(-3, 3, 5)
    # rotation_angle_vec = [-4, -2, 0, 2, 4]
    # scale_factor_vec = [0.95, 1, 1.05]
    final_crop_size = (800,800)  #TODO: this is edhok from when i was testing, make this automatic

    ### Initialize Warp Objects: ###
    affine_layer_torch = Warp_Tensors_Affine_Layer()

    ### Pre-Allocate Flow Grid: ###
    L_shifts = len(shifts_vec)
    L_rotation_angle = len(rotation_angle_vec)
    L_scale = len(scale_factor_vec)
    number_of_possible_warps = L_shifts * L_shifts * L_rotation_angle * L_scale
    B, C, H, W = input_tensor_1.shape
    Flow_Grids = torch.zeros(number_of_possible_warps, H, W, 2).cuda()

    crop_H_start = (H-final_crop_size[0]) // 2
    crop_H_final = crop_H_start + final_crop_size[0]
    crop_W_start = (W - final_crop_size[1]) // 2
    crop_W_final = crop_W_start + final_crop_size[1]

    ### Loop over all Possible Parameters: ###
    SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec), len(scale_factor_vec))).to(input_tensor_2.device)
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
                    input_tensor_2_warped, current_flow_grid = affine_layer_torch.forward(input_tensor_2,
                                                                                          current_shift_x,
                                                                                          current_shift_y,
                                                                                          current_scale_factor,
                                                                                          current_rotation_angle,
                                                                                          return_flow_grid=True,
                                                                                          flag_interpolation_mode='bilinear')
                    Flow_Grids[counter] = current_flow_grid
                    counter = counter + 1


    ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
    possible_warps_tensor = input_tensor_1.repeat((number_of_possible_warps, 1, 1, 1))

    ### Get All Possible Warps Of Previous Image: ###
    possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, Flow_Grids, mode='bilinear')
    # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

    ### Get Min SAD: ###
    # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_tensor_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
    SAD_matrix_2 = torch.abs(possible_warps_tensor - input_tensor_2)[:,:, :, :].mean(-1, True).mean(-2, True)
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

    def forward(self, input_tensor_1, input_tensor_2):
        ############################################################################################################################################################

        # shifts_vec = my_linspace(-3, 3, 5)
        # rotation_angle_vec = [-4, -2, 0, 2, 4]
        # scale_factor_vec = [0.95, 1, 1.05]
        final_crop_size = (800, 800)  #TODO: again, get rid of this edhok shit and make things automatic

        if self.B is None:
            ### Initialize: ###
            self.B, self.C, self.H, self.W = input_tensor_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec),len(self.scale_factor_vec))).to(input_tensor_2.device)
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
                        input_tensor_2_warped, current_flow_grid = self.affine_layer_torch.forward(input_tensor_2,
                                                                                             current_shift_x,
                                                                                             current_shift_y,
                                                                                             current_scale_factor,
                                                                                             current_rotation_angle,
                                                                                             return_flow_grid=True,
                                                                                             flag_interpolation_mode='bilinear')
                        self.Flow_Grids[counter] = current_flow_grid
                        counter = counter + 1

        ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
        possible_warps_tensor = input_tensor_1.repeat((self.number_of_possible_warps, 1, 1, 1))

        ### Get All Possible Warps Of Previous Image: ###
        possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, self.Flow_Grids, mode='bilinear')
        # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

        ### Get Min SAD: ###
        # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_tensor_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
        SAD_matrix_2 = torch.abs(possible_warps_tensor - input_tensor_2)[:, :, :, :].mean(-1, True).mean(-2, True)
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

    def forward(self, input_tensor_1, input_tensor_2):
        ############################################################################################################################################################

        # shifts_vec = my_linspace(-3, 3, 5)
        # rotation_angle_vec = [-4, -2, 0, 2, 4]
        # scale_factor_vec = [0.95, 1, 1.05]
        final_crop_size = (800, 800)

        if self.B is None:
            ### Initialize: ###
            self.B, self.C, self.H, self.W = input_tensor_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(self.shifts_vec), len(self.shifts_vec), len(self.rotation_angle_vec))).to(input_tensor_2.device)
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
                    input_tensor_2_warped = self.affine_layer_torch.forward(input_tensor_2,
                                                                             current_shift_x_tensor,
                                                                             current_shift_y_tensor,
                                                                             current_rotation_angle_tensor)
                    image2_warped_list.append(input_tensor_2_warped)
                    counter = counter + 1

        ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
        input_tensor_1_stacked = input_tensor_1.repeat((self.number_of_possible_warps, 1, 1, 1))
        image2_different_warps_tensor = torch.cat(image2_warped_list, 0)
        input_tensor_1_stacked = crop_torch_batch(input_tensor_1_stacked, (int(self.W*0.9), int(self.H*0.9))).cuda()
        image2_different_warps_tensor = crop_torch_batch(image2_different_warps_tensor, (int(self.W*0.9), int(self.H*0.9))).cuda()

        ### Get Min SAD: ###
        # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_tensor_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
        differences_mat = torch.abs(input_tensor_1_stacked - image2_different_warps_tensor)
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
        input_tensor_2_aligned = self.affine_layer_torch.forward(input_tensor_2,
                                                               -shift_x_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               -shift_y_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                               -rotation_sub_pixel.unsqueeze(0).unsqueeze(0))
        input_tensor_2_aligned = crop_torch_batch(input_tensor_2_aligned, (self.W, self.H)).cuda()

        return shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, input_tensor_2_aligned

#TODO: either turn this into two functions minSAD_Bilinear and minSAD_FFT or simply have minSAD_Affine... and have the
# warp method for calculating the SAD as a flag
#TODO: there's no scaling_vec here!!!! add this!!!!
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
        self.affine_layer_torch = Warp_Tensors_Affine_Layer()
        # self.affine_layer_torch = FFT_Translation_Rotation_Layer()

        ### Pre-Allocate Flow Grid: ###
        self.L_shifts = len(self.shifts_vec)
        self.L_rotation_angle = len(self.rotation_angle_vec)
        self.number_of_possible_warps = self.L_shifts * self.L_shifts * self.L_rotation_angle

        self.B = None
        self.C = None
        self.H = None
        self.W = None

    def forward(self, input_tensor_1, input_tensor_2, shifts_vec=None, rotation_angle_vec=None):
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
            self.B, self.C, self.H, self.W = input_tensor_1.shape
            self.Flow_Grids = torch.zeros(self.number_of_possible_warps, self.H, self.W, 2).cuda()
            self.SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec))).to(input_tensor_2.device)
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

                    # ### Perform Affine Transform In FFT Space: ###
                    # input_tensor_2_warped = self.affine_layer_torch.forward(input_tensor_2,
                    #                                                          current_shift_x_tensor,
                    #                                                          current_shift_y_tensor,
                    #                                                          current_rotation_angle_tensor)
                    ### Perform Affine Transform Using Bilinear Interpolation: ###
                    input_tensor_2_warped = self.affine_layer_torch.forward(input_tensor_2,
                                                                            current_shift_x_tensor,
                                                                            current_shift_y_tensor,
                                                                            torch.Tensor([1.0]),
                                                                            current_rotation_angle_tensor)

                    ### Center Crop Images To Avoid Frame Artifacts: ###
                    diagonal = np.sqrt(self.H**2 + self.W**2)
                    max_translation = np.max(np.abs(shifts_vec))
                    max_rotation_angle = np.max(np.abs(rotation_angle_vec))
                    max_displacement = max_translation + diagonal * np.tan(max_rotation_angle*np.pi/180) + 10 #+10 for backup
                    input_tensor_1_cropped = crop_torch_batch(input_tensor_1, (int(self.H - max_displacement), int(self.W - max_displacement))).cuda()
                    input_tensor_2_warped_cropped = crop_torch_batch(input_tensor_2_warped, (int(self.H - max_displacement), int(self.W - max_displacement))).cuda()

                    ### Calculate Outliers To Avoid: ###
                    current_SAD = (input_tensor_2_warped_cropped - input_tensor_1_cropped).abs()
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
        # #(1). FFT Layer
        # input_tensor_2_aligned = self.affine_layer_torch.forward(input_tensor_2,
        #                                                        -shift_x_sub_pixel.unsqueeze(0).unsqueeze(0),
        #                                                        -shift_y_sub_pixel.unsqueeze(0).unsqueeze(0),
        #                                                        rotation_sub_pixel.unsqueeze(0).unsqueeze(0) * np.pi/180)
        # (1). Bilinear Layer
        input_tensor_2_aligned = self.affine_layer_torch.forward(input_tensor_2,
                                                                 -shift_x_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                                 -shift_y_sub_pixel.unsqueeze(0).unsqueeze(0),
                                                                 torch.Tensor([1.0]).unsqueeze(0).unsqueeze(0),
                                                                 rotation_sub_pixel.unsqueeze(0).unsqueeze(0) * np.pi / 180)
        input_tensor_2_aligned = crop_torch_batch(input_tensor_2_aligned, (self.H, self.W)).cuda()

        return shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, input_tensor_2_aligned


def get_scale_rotation_translation_minSAD_affine(input_tensor_1, input_tensor_2):
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
    input_tensor_1 = previous_image
    input_tensor_2 = current_image
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
    B, C, H, W = input_tensor_1.shape
    Flow_Grids = torch.zeros(number_of_possible_warps, H, W, 2).cuda()

    crop_H_start = (H-final_crop_size[0]) // 2
    crop_H_final = crop_H_start + final_crop_size[0]
    crop_W_start = (W - final_crop_size[1]) // 2
    crop_W_final = crop_W_start + final_crop_size[1]

    ### Loop over all Possible Parameters: ###
    SAD_matrix = torch.zeros((len(shifts_vec), len(shifts_vec), len(rotation_angle_vec), len(scale_factor_vec))).to(input_tensor_2.device)
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
                    input_tensor_2_warped, current_flow_grid = affine_layer_torch.forward(input_tensor_2,
                                                                                          current_shift_x,
                                                                                          current_shift_y,
                                                                                          current_scale_factor,
                                                                                          current_rotation_angle,
                                                                                          return_flow_grid=True,
                                                                                          flag_interpolation_mode='bilinear')
                    Flow_Grids[counter] = current_flow_grid
                    counter = counter + 1


    ### Do Things "Efficiently" Using Pytroch's Parallel Programming: ###
    possible_warps_tensor = input_tensor_1.repeat((number_of_possible_warps, 1, 1, 1))

    ### Get All Possible Warps Of Previous Image: ###
    possible_warps_tensor = torch.nn.functional.grid_sample(possible_warps_tensor, Flow_Grids, mode='bilinear')
    # possible_warps_tensor = bicubic_interpolate(possible_warps_tensor, Flow_Grids[:,:,:,0:1], Flow_Grids[:,:,:,1:2])

    ### Get Min SAD: ###
    # SAD_matrix_2 = torch.abs(possible_warps_tensor - input_tensor_1)[:,:, crop_H_start:crop_H_final, crop_W_start:crop_W_final].mean(-1, True).mean(-2, True)
    SAD_matrix_2 = torch.abs(possible_warps_tensor - input_tensor_2)[:,:, :, :].mean(-1, True).mean(-2, True)
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


def test_minSAD_registration():
    ### Paramters: ###
    warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
    shift_x = np.array([0])
    shift_y = np.array([-0])
    scale_factor_warp = np.array([1.00])
    rotation_angles = my_linspace(-1, 1, 21)

    ### Prepare Data: ###
    input_tensor_1 = read_image_default_torch()
    input_tensor_1 = RGB2BW(input_tensor_1)
    input_tensor_1 = crop_torch_batch(input_tensor_1, 700).cuda()
    input_tensor_1_numpy = input_tensor_1.cpu().numpy()[0, 0]

    warped_tensors_list_numpy = []
    warped_tensors_list_torch = []
    inferred_rotation_list_1 = []
    inferred_scaling_list_1 = []
    inferred_shift_x_list_1 = []
    inferred_shift_y_list_1 = []
    inferred_rotation_list_2 = []
    inferred_scaling_list_2 = []
    inferred_shift_x_list_2 = []
    inferred_shift_y_list_2 = []

    for i in np.arange(len(rotation_angles)):
        print(i)
        ### Warp Tensor: ###
        rotation_angle = np.array([rotation_angles[i]])
        input_tensor_2 = warp_tensors_affine_layer.forward(input_tensor_1, shift_x, shift_y, scale_factor_warp, rotation_angle*np.pi/180)
        input_tensor_2 = crop_torch_batch(input_tensor_2, 700)
        input_tensor_2_numpy = input_tensor_2.cpu().numpy()[0, 0]
        warped_tensors_list_numpy.append(input_tensor_2_numpy)
        warped_tensors_list_torch.append(input_tensor_2)

        ### Get Parameters: ###
        # (1).
        # minsad_fft_affine_registration_layer = minSAD_FFT_Affine_Registration_Layer()
        minsad_fft_affine_registration_layer = minSAD_FFT_Affine_Registration_MemorySave_Layer()
        minsad_fft_affine_registration_layer.shifts_vec = my_linspace(-1, 2, 5)
        minsad_fft_affine_registration_layer.rotation_angle_vec = my_linspace(-2, 2 + 2 / 20, 10)
        shift_x_sub_pixel, shift_y_sub_pixel, rotation_sub_pixel, input_tensor_2_aligned = minsad_fft_affine_registration_layer.forward(input_tensor_1, input_tensor_2)

        inferred_rotation_list_1.append(rotation_sub_pixel)
        inferred_shift_x_list_1.append(shift_x_sub_pixel)
        inferred_shift_y_list_1.append(shift_y_sub_pixel)


    ### Present Outputs: ###
    # (1). Rotation
    figure()
    plot(rotation_angles, rotation_angles)
    plot(rotation_angles, -np.array(inferred_rotation_list_1))
    plt.legend(['GT angles', 'torch layer'])
    plt.title('rotation plot')
    # # (1). Translation
    # figure()
    # plot(rotation_angles, shift_x[0] * np.ones_like(rotation_angles))
    # plot(rotation_angles, inferred_shift_x_list_1)
    # plt.xlabel('Rotation Angle')
    # plt.legend(['GT Shift X', 'torch layer'])
    # plt.title('Translation plot')
    # figure()
    # plot(rotation_angles, shift_y[0] * np.ones_like(rotation_angles))
    # plot(rotation_angles, inferred_shift_y_list_2)
    # plt.legend(['GT Shift Y', 'torch layer'])
    # plt.title('Translation Y plot')
    # plt.xlabel('Rotation Angle')

# test_minSAD_registration()
######################################################################################################################################################################################
#
#
#
#
#
#
#
#
#
######################################################################################################################################################################################
### Gimbaless Section: ###
def Gimbaless_3D_1(input_tensor=None):
    #### Get Mat Size: ###
    B,T,C,H,W = input_tensor.shape

    ######################################################################################################
    ### Prediction Grid Definition: ###
    prediction_block_size = int16(H/1) #for one global prediction for everything simply use W or H
    overlap_size = 0 #0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
    temporal_lowpass_number_of_steps = 9

    ### Ctypi Parameters: ###
    reg_step = .1
    filt_len = 11
    dif_ord = 2
    dif_len = 7
    CorW = 7
    dYExp = 0
    dXExp = 0
    reg_lst = np.arange(0, 0.5, 0.1)

    ### Define convn layers: ###
    torch_convn_layer = convn_layer_torch()

    ### create filter: ###  TODO: can be preallocated
    params = EasyDict()
    params.dif_len = 9
    # spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((1,1,1,params.dif_len))  #Read this from file
    dict_from_matlab = scipy.io.loadmat('/home/mafat/PycharmProjects/DATA_FOLDER/spatial_lowpass_9tap.mat') #Read this from file
    spatial_lowpass_before_temporal_derivative_filter_x = dict_from_matlab['spatial_lowpass_before_temporal_derivative_filter'].flatten()
    spatial_lowpass_before_temporal_derivative_filter_x = torch_get_4D(torch.Tensor(spatial_lowpass_before_temporal_derivative_filter_x))
    spatial_lowpass_before_temporal_derivative_filter_y = spatial_lowpass_before_temporal_derivative_filter_x.permute([0,1,3,2])
    spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
    spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])

    ### Preallocate filters: ###
    #(1). temporal derivative:
    temporal_derivative_filter = torch.Tensor(np.array([-1,1]))
    temporal_derivative_filter = torch.reshape(temporal_derivative_filter, (1,2,1,1))
    temporal_derivative_fft_filter = torch.fft.fftn(temporal_derivative_filter, s=T, dim=[1])
    #(2). spatial derivative:
    grad_x_kernel = torch_get_4D(torch.Tensor(np.array([-1,0,1])/2), 'W')
    grad_y_kernel = grad_x_kernel.permute([0,1,3,2])
    grad_x_filter_fft = torch.fft.fftn(grad_x_kernel, s=W, dim=[-1])
    grad_y_filter_fft = torch.fft.fftn(grad_y_kernel, s=H, dim=[-2])
    #(3). temporal averaging:
    temporal_averaging_filter_before_spatial_gradient = torch.ones((1,temporal_lowpass_number_of_steps,1,1,1))/temporal_lowpass_number_of_steps
    temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

    ######################################################################################################
    ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###\
    #(1).
    px = torch_convn_layer.forward(input_tensor, grad_x_kernel.flatten(), -1)
    py = torch_convn_layer.forward(input_tensor, grad_y_kernel.flatten(), -2)
    # #(*). Pair-Wise Shifts:
    # px = px[:,0:T-1] # T-1 displacements to output
    # py = py[:,0:T-1]
    #(*). Center Frame Reference Shifts:
    px = px
    py = py

    # #(2). average out consecutive frames to smooth spatial gradients a bit:
    # input_tensor_averaged = (input_tensor[:,0:T-1] + input_tensor[:,1:T])/2
    # px = torch_convn_layer.forward(input_tensor, grad_x_kernel.flatten(), -1)
    # py = torch_convn_layer.forward(input_tensor, grad_y_kernel.flatten(), -2)

    # #(3). HardCoded:
    # px = (input_tensor[:,:,:,:,2:W] - input_tensor[:,:,:,:,0:W-2])/2
    # py = (input_tensor[:,:,:,2:H,:] - input_tensor[:,:,:,0:H-2,:])/2
    # px = torch.nn.functional.pad(px, [1,1,0,0])  #pad array to fid later cropping instead of if conditions...not necessary
    # py = torch.nn.functional.pad(py, [0,0,1,1])

    # #(3). average images out to get a better reading of the spatial gradient:
    # input_tensor_averaged = torch_convn_layer.forward(input_tensor, temporal_averaging_filter_before_spatial_gradient.flatten(), 1)
    # px = torch_convn_layer.forward(input_tensor, grad_x_kernel.flatten(), -1)
    # py = torch_convn_layer.forward(input_tensor, grad_y_kernel.flatten(), -2)

    # #(4). FFT Filtering 1D (TODO: can i use 1D fft or should i do fft2?):
    # input_tensor_fft_x = torch.fft.fftn(input_tensor,dim=[-1])
    # input_tensor_fft_y = torch.fft.fftn(input_tensor,dim=[-2])
    # px = torch.fft.ifftn(input_tensor_fft_x * grad_x_filter_fft, dim=[-1])
    # py = torch.fft.ifftn(input_tensor_fft_y * grad_y_filter_fft, dim=[-2])

    ### Cut-Out Invalid Parts (Center-Crop): ###
    #TODO: perhapse don't(!!!!) center crop now, but later
    start_index = np.int16(floor(params.dif_len/2))
    px = px[:,:,:,start_index:H-(start_index-1), start_index:W-(start_index-1)]
    py = py[:,:,:,start_index:H-(start_index-1), start_index:W-(start_index-1)]
    ######################################################################################################

    ######################################################################################################
    ### Calculate Temporal Derivative: ###
    ABtx = torch_convn_layer.forward(input_tensor, spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
    ABty = torch_convn_layer.forward(input_tensor, spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)
    # #(1). Pair-Wise Shifts:
    # ABtx = ABtx[:,1:T] - ABtx[:,0:T-1]
    # ABty = ABty[:,1:T] - ABty[:,0:T-1]
    #(2). Center Tensor Reference Shifts:
    ABtx = ABtx - ABtx[:,T//2:T//2+1]
    ABty = ABty - ABty[:,T//2:T//2+1]

    # #(2). FFT Domain: (very inefficient as it is....maybe it can be made more efficient....)
    # ABtx = torch.fft.fftn(input_tensor_fft_x * spatial_lowpass_before_temporal_derivative_filter_fft_x, dim=[-1])
    # ABty = torch.fft.fftn(input_tensor_fft_y * spatial_lowpass_before_temporal_derivative_filter_fft_y, dim=[-2])
    # ABtx_fft = torch.fft.fftn(ABtx, dim=[1])
    # ABty_fft = torch.fft.fftn(ABty, dim=[1])
    # ABtx = torch.real(torch.fft.ifftn(ABtx_fft * temporal_derivative_fft_filter, dim=[1]))
    # ABty = torch.real(torch.fft.ifftn(ABty_fft * temporal_derivative_fft_filter, dim=[1]))

    ### Cut-Out Invalid Parts (Center-Crop): ###
    ABtx = ABtx[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
    ABty = ABty[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
    #####################################################################################################

    ######################################################################################################
    ### Find Shift Between Frames: ###

    #(*) inv([A,B;C,D]) = 1/(AD-BC)*[D,-B;-C,A]
    ### Get elements of the first matrix in the equation we're solving: ###
    # Local Prediction:
    B2,T2,C2,H2,W2 = px.shape
    prediction_block_size = min(prediction_block_size, H2)
    pxpy = fast_binning_2D_PixelBinning(px*py, prediction_block_size, overlap_size)
    px2 = fast_binning_2D_PixelBinning(px**2, prediction_block_size, overlap_size)
    py2 = fast_binning_2D_PixelBinning(py**2, prediction_block_size, overlap_size)

    ### Invert the above matrix explicitly in a hard coded fashion until i find
    ### a more elegant way to invert matrices in parallel: ###
    # P = [px2,pxpy; py2,pxpy] -> matrix we are explicitely inverting
    # inv_P = 1./(A.*D-B.*C) .* [D,-B;-C,A];
    common_factor = 1./(px2*py2 - pxpy*pxpy)
    inv_P_xx = common_factor * py2
    inv_P_xy = -common_factor * pxpy
    inv_P_yx = -common_factor * pxpy
    inv_P_yy = common_factor * px2

    ### Solve the needed equation explicitly: ###
    # d = inv_P * [squeeze(sum(ABtx.*px, [1,2])); squeeze(sum(ABty.*py,[1,2]))];
    # Local Prediction:
    delta_x = inv_P_xx * (fast_binning_2D_PixelBinning((ABtx*px), prediction_block_size, overlap_size)) + \
              inv_P_xy * (fast_binning_2D_PixelBinning((ABty*py), prediction_block_size, overlap_size))
    delta_y = inv_P_yx * (fast_binning_2D_PixelBinning((ABtx*px), prediction_block_size, overlap_size)) + \
              inv_P_yy * (fast_binning_2D_PixelBinning((ABty*py), prediction_block_size, overlap_size))

    return delta_x, delta_y

def Gimbaless_3D_2(input_tensor=None, flag_pairwise_or_reference='reference'):
    #### Get Mat Size: ###
    B, T, C, H, W = input_tensor.shape

    ######################################################################################################
    ### Prediction Grid Definition: ###
    prediction_block_size = int16(H / 1)  # for one global prediction for everything simply use W or H
    overlap_size = 0  # 0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
    temporal_lowpass_number_of_steps = 9

    ### Ctypi Parameters: ###
    #TODO: delete
    reg_step = .1
    filt_len = 11
    dif_ord = 2
    dif_len = 7
    CorW = 7
    dYExp = 0
    dXExp = 0
    reg_lst = np.arange(0, 0.5, 0.1)

    ### Define convn layers: ###
    torch_convn_layer = convn_layer_torch()

    ### create filter: ###  TODO: can be preallocated
    params = EasyDict()
    params.dif_len = 9
    # spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((1,1,1,params.dif_len))  #Read this from file
    dict_from_matlab = scipy.io.loadmat('/home/mafat/PycharmProjects/DATA_FOLDER/spatial_lowpass_9tap.mat')  # Read this from file
    spatial_lowpass_before_temporal_derivative_filter_x = dict_from_matlab['spatial_lowpass_before_temporal_derivative_filter'].flatten()
    spatial_lowpass_before_temporal_derivative_filter_x = torch_get_4D(torch.Tensor(spatial_lowpass_before_temporal_derivative_filter_x))
    spatial_lowpass_before_temporal_derivative_filter_y = spatial_lowpass_before_temporal_derivative_filter_x.permute([0, 1, 3, 2])
    spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
    spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])

    ### Preallocate filters: ###
    # (1). temporal derivative:
    temporal_derivative_filter = torch.Tensor(np.array([-1, 1]))
    temporal_derivative_filter = torch.reshape(temporal_derivative_filter, (1, 2, 1, 1))
    temporal_derivative_fft_filter = torch.fft.fftn(temporal_derivative_filter, s=T, dim=[1])
    # (2). spatial derivative:
    grad_x_kernel = torch_get_4D(torch.Tensor(np.array([-1, 0, 1]) / 2), 'W')
    grad_y_kernel = grad_x_kernel.permute([0, 1, 3, 2])
    grad_x_filter_fft = torch.fft.fftn(grad_x_kernel, s=W, dim=[-1])
    grad_y_filter_fft = torch.fft.fftn(grad_y_kernel, s=H, dim=[-2])
    # (3). temporal averaging:
    temporal_averaging_filter_before_spatial_gradient = torch.ones((1, temporal_lowpass_number_of_steps, 1, 1, 1)) / temporal_lowpass_number_of_steps
    temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

    ######################################################################################################
    ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###\
    # (1).
    px = torch_convn_layer.forward(input_tensor, grad_x_kernel.flatten(), -1)
    py = torch_convn_layer.forward(input_tensor, grad_y_kernel.flatten(), -2)
    if flag_pairwise_or_reference == 'pairs':
        # #(*). Pair-Wise Shifts:
        px = px[:,0:T-1] # T-1 displacements to output
        py = py[:,0:T-1]
    else:
        # (*). Center Frame Reference Shifts:
        px = px
        py = py

    ### Cut-Out Invalid Parts (Center-Crop): ###
    # TODO: perhapse don't(!!!!) center crop now, but later
    start_index = np.int16(floor(params.dif_len / 2))
    px = px[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
    py = py[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
    ######################################################################################################

    ######################################################################################################
    ### Calculate Temporal Derivative: ###
    ABtx = torch_convn_layer.forward(input_tensor, spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
    ABty = torch_convn_layer.forward(input_tensor, spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)
    if flag_pairwise_or_reference == 'pairwise':
        #(1). Pair-Wise Shifts:
        ABtx = ABtx[:,1:T] - ABtx[:,0:T-1]
        ABty = ABty[:,1:T] - ABty[:,0:T-1]
    else:
        # (2). Center Tensor Reference Shifts:
        ABtx = ABtx - ABtx[:, T // 2:T // 2 + 1]
        ABty = ABty - ABty[:, T // 2:T // 2 + 1]

    ### Cut-Out Invalid Parts (Center-Crop): ###
    ABtx = ABtx[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
    ABty = ABty[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
    #####################################################################################################

    ######################################################################################################
    ### Find Shift Between Frames: ###
    # (*) inv([A,B;C,D]) = 1/(AD-BC)*[D,-B;-C,A]
    ### Get elements of the first matrix in the equation we're solving: ###
    # Local Prediction:
    B2, T2, C2, H2, W2 = px.shape
    prediction_block_size = min(prediction_block_size, H2)
    pxpy = fast_binning_2D_PixelBinning(px * py, prediction_block_size, overlap_size)
    px2 = fast_binning_2D_PixelBinning(px ** 2, prediction_block_size, overlap_size)
    py2 = fast_binning_2D_PixelBinning(py ** 2, prediction_block_size, overlap_size)

    ### Invert the above matrix explicitly in a hard coded fashion until i find
    ### a more elegant way to invert matrices in parallel: ###
    # P = [px2,pxpy; py2,pxpy] -> matrix we are explicitely inverting
    # inv_P = 1./(A.*D-B.*C) .* [D,-B;-C,A];
    common_factor = 1. / (px2 * py2 - pxpy * pxpy)
    inv_P_xx = common_factor * py2
    inv_P_xy = -common_factor * pxpy
    inv_P_yx = -common_factor * pxpy
    inv_P_yy = common_factor * px2

    ### Solve the needed equation explicitly: ###
    # d = inv_P * [squeeze(sum(ABtx.*px, [1,2])); squeeze(sum(ABty.*py,[1,2]))];
    # Local Prediction:
    delta_x = inv_P_xx * (fast_binning_2D_PixelBinning((ABtx * px), prediction_block_size, overlap_size)) + \
              inv_P_xy * (fast_binning_2D_PixelBinning((ABty * py), prediction_block_size, overlap_size))
    delta_y = inv_P_yx * (fast_binning_2D_PixelBinning((ABtx * px), prediction_block_size, overlap_size)) + \
              inv_P_yy * (fast_binning_2D_PixelBinning((ABty * py), prediction_block_size, overlap_size))

    return (delta_x, delta_x)


class Gimbaless_Layer_Torch(nn.Module):

    # Initialize this with a module
    def __init__(self):
        super(Gimbaless_Layer_Torch, self).__init__()

        ### Prediction Grid Definition: ###
        self.prediction_block_size = int16(4 / 1)  # for one global prediction for everything simply use W or H
        self.overlap_size = 0  # 0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
        self.temporal_lowpass_number_of_steps = 9

        ### Ctypi Parameters: ###
        self.reg_step = .1
        self.filt_len = 11
        self.dif_ord = 2
        self.dif_len = 7
        self.CorW = 7
        self.dYExp = 0
        self.dXExp = 0
        self.reg_lst = np.arange(0, 0.5, 0.1)

        ### Define convn layers: ###
        self.torch_convn_layer = convn_layer_torch()

        ### create filter: ###
        self.params = EasyDict()
        self.params.dif_len = 9
        # spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((1,1,1,params.dif_len))  #Read this from file
        self.dict_from_matlab = scipy.io.loadmat(
            '/home/mafat/PycharmProjects/DATA_FOLDER/spatial_lowpass_9tap.mat')  # Read this from file
        self.spatial_lowpass_before_temporal_derivative_filter_x = self.dict_from_matlab[
            'spatial_lowpass_before_temporal_derivative_filter'].flatten()
        self.spatial_lowpass_before_temporal_derivative_filter_x = torch_get_4D(
            torch.Tensor(self.spatial_lowpass_before_temporal_derivative_filter_x))
        self.spatial_lowpass_before_temporal_derivative_filter_y = self.spatial_lowpass_before_temporal_derivative_filter_x.permute(
            [0, 1, 3, 2])

        ### Preallocate filters: ###
        # (1). temporal derivative:
        self.temporal_derivative_filter = torch.Tensor(np.array([-1, 1]))
        self.temporal_derivative_filter = torch.reshape(self.temporal_derivative_filter, (1, 2, 1, 1))
        # (2). spatial derivative:
        self.grad_x_kernel = torch_get_4D(torch.Tensor(np.array([-1, 0, 1]) / 2), 'W')
        self.grad_y_kernel = self.grad_x_kernel.permute([0, 1, 3, 2])
        # (3). temporal averaging:
        self.temporal_averaging_filter_before_spatial_gradient = torch.ones(
            (1, self.temporal_lowpass_number_of_steps, 1, 1, 1)) / self.temporal_lowpass_number_of_steps

        self.spatial_lowpass_before_temporal_derivative_filter_fft_x = None

    def forward(self, input_tensor, reference_tensor=None, flag_pairwise_or_reference='reference'):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.spatial_lowpass_before_temporal_derivative_filter_fft_x is None:
            ### Get FFT Filters: ###
            self.spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(
                self.spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
            self.spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(
                self.spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])
            self.temporal_derivative_fft_filter = torch.fft.fftn(self.temporal_derivative_filter, s=T, dim=[1])
            self.grad_x_filter_fft = torch.fft.fftn(self.grad_x_kernel, s=W, dim=[-1])
            self.grad_y_filter_fft = torch.fft.fftn(self.grad_y_kernel, s=H, dim=[-2])
            self.temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(
                self.temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

        ######################################################################################################
        ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###\
        # (1).
        px = self.torch_convn_layer.forward(input_tensor, self.grad_x_kernel.flatten(), -1)
        py = self.torch_convn_layer.forward(input_tensor, self.grad_y_kernel.flatten(), -2)
        if flag_pairwise_or_reference == 'pairse':
            # #(*). Pair-Wise Shifts:
            px = px[:, 0:T - 1]  # T-1 displacements to output
            py = py[:, 0:T - 1]
        else:
            # (*). Center Frame Reference Shifts:
            px = px
            py = py

        ### Cut-Out Invalid Parts (Center-Crop): ###
        # TODO: perhapse don't(!!!!) center crop now, but later. i can also hard-code this into the fast_binning function start and stop indices
        start_index = np.int16(floor(self.params.dif_len / 2))
        px = px[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        py = py[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ######################################################################################################

        ######################################################################################################
        ### Calculate Temporal Derivative: ###
        ABtx = self.torch_convn_layer.forward(input_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
        ABty = self.torch_convn_layer.forward(input_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)
        if flag_pairwise_or_reference == 'pairwise':
            # (1). Pair-Wise Shifts:
            ABtx = ABtx[:, 1:T] - ABtx[:, 0:T - 1]
            ABty = ABty[:, 1:T] - ABty[:, 0:T - 1]
        else:
            # (2). Center Tensor Reference Shifts:
            ABtx = ABtx - ABtx[:, T // 2:T // 2 + 1] #TODO: replace this with possible external reference tensor, but make sure it undergoes same filtering!!!!
            ABty = ABty - ABty[:, T // 2:T // 2 + 1]

        ### Cut-Out Invalid Parts (Center-Crop): ###
        ABtx = ABtx[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ABty = ABty[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        #####################################################################################################

        ######################################################################################################
        ### Find Shift Between Frames: ###
        # (*) inv([A,B;C,D]) = 1/(AD-BC)*[D,-B;-C,A]
        ### Get elements of the first matrix in the equation we're solving: ###
        # Local Prediction:
        B2, T2, C2, H2, W2 = px.shape
        prediction_block_size = min(self.prediction_block_size, H2)
        pxpy = fast_binning_2D_PixelBinning(px * py, self.prediction_block_size, self.overlap_size)
        px2 = fast_binning_2D_PixelBinning(px ** 2, self.prediction_block_size, self.overlap_size)
        py2 = fast_binning_2D_PixelBinning(py ** 2, self.prediction_block_size, self.overlap_size)

        ### Invert the above matrix explicitly in a hard coded fashion until i find
        ### a more elegant way to invert matrices in parallel: ###
        # P = [px2,pxpy; py2,pxpy] -> matrix we are explicitely inverting
        # inv_P = 1./(A.*D-B.*C) .* [D,-B;-C,A];
        common_factor = 1. / (px2 * py2 - pxpy * pxpy)
        inv_P_xx = common_factor * py2
        inv_P_xy = -common_factor * pxpy
        inv_P_yx = -common_factor * pxpy
        inv_P_yy = common_factor * px2

        ### Solve the needed equation explicitly: ###
        # d = inv_P * [squeeze(sum(ABtx.*px, [1,2])); squeeze(sum(ABty.*py,[1,2]))];
        # Local Prediction:
        A = (fast_binning_2D_PixelBinning((ABtx * px), self.prediction_block_size, self.overlap_size))
        B = (fast_binning_2D_PixelBinning((ABty * py), self.prediction_block_size, self.overlap_size))
        delta_x = inv_P_xx * A + inv_P_xy * B
        delta_y = inv_P_yx * A + inv_P_yy * B

        return (delta_x, delta_y)


class Gimbaless_Rotation_Layer_Torch(nn.Module):

    # Initialize this with a module
    def __init__(self):
        super(Gimbaless_Rotation_Layer_Torch, self).__init__()

        ### Prediction Grid Definition: ###
        self.prediction_block_size = int16(4 / 1)  # for one global prediction for everything simply use W or H
        self.overlap_size = 0  # 0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
        self.temporal_lowpass_number_of_steps = 9

        ### Ctypi Parameters: ###
        self.reg_step = .1
        self.filt_len = 11
        self.dif_ord = 2
        self.dif_len = 7
        self.CorW = 7
        self.dYExp = 0
        self.dXExp = 0
        self.reg_lst = np.arange(0, 0.5, 0.1)

        ### Define convn layers: ###
        self.torch_convn_layer = convn_layer_torch()

        ### create filter: ###
        self.params = EasyDict()
        self.params.dif_len = 9
        # spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((1,1,1,params.dif_len))  #Read this from file
        self.dict_from_matlab = scipy.io.loadmat(r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/spatial_lowpass_9tap.mat')  # Read this from file
        self.spatial_lowpass_before_temporal_derivative_filter_x = self.dict_from_matlab['spatial_lowpass_before_temporal_derivative_filter'].flatten()
        self.spatial_lowpass_before_temporal_derivative_filter_x = torch_get_4D(torch.Tensor(self.spatial_lowpass_before_temporal_derivative_filter_x))
        self.spatial_lowpass_before_temporal_derivative_filter_y = self.spatial_lowpass_before_temporal_derivative_filter_x.permute([0, 1, 3, 2])

        ### Preallocate filters: ###
        # (1). temporal derivative:
        self.temporal_derivative_filter = torch.Tensor(np.array([-1, 1]))
        self.temporal_derivative_filter = torch.reshape(self.temporal_derivative_filter, (1, 2, 1, 1))
        # (2). spatial derivative:
        self.grad_x_kernel = torch_get_4D(torch.Tensor(np.array([-1, 0, 1]) / 2), 'W')
        self.grad_y_kernel = self.grad_x_kernel.permute([0, 1, 3, 2])
        # (3). temporal averaging:
        self.temporal_averaging_filter_before_spatial_gradient = torch.ones((1, self.temporal_lowpass_number_of_steps, 1, 1, 1)) / self.temporal_lowpass_number_of_steps

        self.spatial_lowpass_before_temporal_derivative_filter_fft_x = None
        self.X = None

    def forward(self, input_tensor, reference_tensor=None):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.spatial_lowpass_before_temporal_derivative_filter_fft_x is None:
            ### Get FFT Filters: ###
            self.spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
            self.spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])
            self.temporal_derivative_fft_filter = torch.fft.fftn(self.temporal_derivative_filter, s=T, dim=[1])
            self.grad_x_filter_fft = torch.fft.fftn(self.grad_x_kernel, s=W, dim=[-1])
            self.grad_y_filter_fft = torch.fft.fftn(self.grad_y_kernel, s=H, dim=[-2])
            self.temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(self.temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

        if self.X is None:
            derivative_filter_lowpass_equivalent_length = len(self.spatial_lowpass_before_temporal_derivative_filter_x)
            shift_filter_length = 11 #TODO: later on with the full version there will be an equivalent shift filter
            invalid_frame_size = np.round((derivative_filter_lowpass_equivalent_length + shift_filter_length)/4)
            x_start = -input_tensor.shape[-1]/2 + 0.5 + invalid_frame_size
            x_stop = input_tensor.shape[-1]/2 - 0.5 - invalid_frame_size
            y_start = -input_tensor.shape[-2] / 2 + 0.5 + invalid_frame_size
            y_stop = input_tensor.shape[-2] / 2 - 0.5 - invalid_frame_size
            x_vec = my_linspace_step2(x_start, x_stop, 1)
            y_vec = my_linspace_step2(y_start, y_stop, 1)
            [X,Y] = np.meshgrid(x_vec, y_vec)
            self.X = torch.tensor(X).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.Y = torch.tensor(Y).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.X_flattened = torch.flatten(self.X, -3, -1)
            self.Y_flattened = torch.flatten(self.Y, -3, -1)

        ######################################################################################################
        ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###
        # # (1). Only input tensor
        # px = self.torch_convn_layer.forward(input_tensor, self.grad_x_kernel.flatten(), -1)
        # py = self.torch_convn_layer.forward(input_tensor, self.grad_y_kernel.flatten(), -2)
        # (2). input_tensor + reference_tensor
        px = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_x_kernel.flatten(), -1)
        py = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_y_kernel.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        # TODO: perhapse don't(!!!!) center crop now, but later. i can also hard-code this into the fast_binning function start and stop indices
        start_index = np.int16(floor(self.params.dif_len / 2))
        px = px[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        py = py[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ######################################################################################################

        ######################################################################################################
        ### Calculate Temporal Derivative: ###
        ABtx = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
        ABty = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        ABtx = ABtx[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ABty = ABty[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        #####################################################################################################


        ######################################################################################################
        ### Find Shift Between Frames Matrix Notation: ###
        #(*). i assume that gimbaless rotation is use for the ENTIRE FRAME only (no local predictions), so global sums only
        B2, T2, C2, H2, W2 = px.shape
        px = torch.flatten(px, -3, -1)
        py = torch.flatten(py, -3, -1)
        ABtx = torch.flatten(ABtx, -3, -1)
        ABty = torch.flatten(ABty, -3, -1)

        momentum_operator_tensor = px * self.Y_flattened - py * self.X_flattened
        p = torch.cat([px.unsqueeze(-1), py.unsqueeze(-1), momentum_operator_tensor.unsqueeze(-1)], -1).squeeze(0)
        p_mat = torch.bmm(torch.transpose(p, -1, -2), p)
        d1 = torch.linalg.inv(p_mat)
        d2 = torch.cat([(ABtx*px).sum(-1, True), (ABty*py).sum(-1, True), (ABtx*(px*self.Y_flattened) - ABty*(py*self.X_flattened)).sum(-1, True)], -1).squeeze(0).unsqueeze(-1)
        d_vec = torch.matmul(d1, d2).squeeze(-1)
        delta_x = d_vec[:, 0]
        delta_y = d_vec[:, 1]
        delta_theta = d_vec[:, 2]

        return delta_x, delta_y, delta_theta


    def forward_weighted(self, input_tensor, reference_tensor=None, W=None):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.spatial_lowpass_before_temporal_derivative_filter_fft_x is None:
            ### Get FFT Filters: ###
            self.spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
            self.spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])
            self.temporal_derivative_fft_filter = torch.fft.fftn(self.temporal_derivative_filter, s=T, dim=[1])
            self.grad_x_filter_fft = torch.fft.fftn(self.grad_x_kernel, s=W, dim=[-1])
            self.grad_y_filter_fft = torch.fft.fftn(self.grad_y_kernel, s=H, dim=[-2])
            self.temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(self.temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

        if self.X is None:
            derivative_filter_lowpass_equivalent_length = len(self.spatial_lowpass_before_temporal_derivative_filter_x)
            shift_filter_length = 11 #TODO: later on with the full version there will be an equivalent shift filter
            invalid_frame_size = np.round((derivative_filter_lowpass_equivalent_length + shift_filter_length)/4)
            x_start = -input_tensor.shape[-1]/2 + 0.5 + invalid_frame_size
            x_stop = input_tensor.shape[-1]/2 - 0.5 - invalid_frame_size
            y_start = -input_tensor.shape[-2] / 2 + 0.5 + invalid_frame_size
            y_stop = input_tensor.shape[-2] / 2 - 0.5 - invalid_frame_size
            x_vec = my_linspace_step2(x_start, x_stop, 1)
            y_vec = my_linspace_step2(y_start, y_stop, 1)
            [X,Y] = np.meshgrid(x_vec, y_vec)
            self.X = torch.tensor(X).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.Y = torch.tensor(Y).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.X_flattened = torch.flatten(self.X, -3, -1)
            self.Y_flattened = torch.flatten(self.Y, -3, -1)

        ######################################################################################################
        ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###
        # # (1). Only input tensor
        # px = self.torch_convn_layer.forward(input_tensor, self.grad_x_kernel.flatten(), -1)
        # py = self.torch_convn_layer.forward(input_tensor, self.grad_y_kernel.flatten(), -2)
        # (2). input_tensor + reference_tensor
        px = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_x_kernel.flatten(), -1)
        py = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_y_kernel.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        # TODO: perhapse don't(!!!!) center crop now, but later. i can also hard-code this into the fast_binning function start and stop indices
        start_index = np.int16(floor(self.params.dif_len / 2))
        px = px[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        py = py[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ######################################################################################################

        ######################################################################################################
        ### Calculate Temporal Derivative: ###
        ABtx = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
        ABty = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        ABtx = ABtx[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ABty = ABty[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        #####################################################################################################


        ######################################################################################################
        ### Find Shift Between Frames Matrix Notation: ###
        #(*). i assume that gimbaless rotation is use for the ENTIRE FRAME only (no local predictions), so global sums only
        B2, T2, C2, H2, W2 = px.shape
        px = torch.flatten(px, -3, -1)
        py = torch.flatten(py, -3, -1)
        ABtx = torch.flatten(ABtx, -3, -1)
        ABty = torch.flatten(ABty, -3, -1)

        momentum_operator_tensor = px * self.Y_flattened - py * self.X_flattened
        W_flattened = torch.flaten(W, -3, -1)
        p = torch.cat([px.unsqueeze(-1), py.unsqueeze(-1), momentum_operator_tensor.unsqueeze(-1)], -1).squeeze(0)
        Wp = torch.cat([(W_flattened*px).unsqueeze(-1), (W_flattened*py).unsqueeze(-1), (W_flattened*momentum_operator_tensor).unsqueeze(-1)], -1).squeeze(0)
        p_mat = torch.bmm(torch.transpose(Wp, -1, -2), p)
        d1 = torch.linalg.inv(p_mat)
        d2 = torch.cat([(W_flattened*ABtx*px).sum(-1, True), (W_flattened*ABty*py).sum(-1, True), (W_flattened*ABtx*(px*self.Y_flattened) - W_flattened*ABty*(py*self.X_flattened)).sum(-1, True)], -1).squeeze(0).unsqueeze(-1)
        d_vec = torch.matmul(d1, d2).squeeze(-1)
        delta_x = d_vec[:, 0]
        delta_y = d_vec[:, 1]
        delta_theta = d_vec[:, 2]

        return delta_x, delta_y, delta_theta


class Gimbaless_Homography_Layer_Torch(nn.Module):

    # Initialize this with a module
    def __init__(self):
        super(Gimbaless_Homography_Layer_Torch, self).__init__()

        ### Prediction Grid Definition: ###
        self.prediction_block_size = int16(4 / 1)  # for one global prediction for everything simply use W or H
        self.overlap_size = 0  # 0 = no overlap, prediction_block_size-1 = maximum overlap (dense prediction)
        self.temporal_lowpass_number_of_steps = 9

        ### Ctypi Parameters: ###
        self.reg_step = .1
        self.filt_len = 11
        self.dif_ord = 2
        self.dif_len = 7
        self.CorW = 7
        self.dYExp = 0
        self.dXExp = 0
        self.reg_lst = np.arange(0, 0.5, 0.1)

        ### Define convn layers: ###
        self.torch_convn_layer = convn_layer_torch()

        ### create filter: ###
        self.params = EasyDict()
        self.params.dif_len = 9
        # spatial_lowpass_before_temporal_derivative_filter_x = torch.ones((1,1,1,params.dif_len))  #Read this from file
        self.dict_from_matlab = scipy.io.loadmat(r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/spatial_lowpass_9tap.mat')  # Read this from file
        self.spatial_lowpass_before_temporal_derivative_filter_x = self.dict_from_matlab['spatial_lowpass_before_temporal_derivative_filter'].flatten()
        self.spatial_lowpass_before_temporal_derivative_filter_x = torch_get_4D(torch.Tensor(self.spatial_lowpass_before_temporal_derivative_filter_x))
        self.spatial_lowpass_before_temporal_derivative_filter_y = self.spatial_lowpass_before_temporal_derivative_filter_x.permute([0, 1, 3, 2])

        ### Preallocate filters: ###
        # (1). temporal derivative:
        self.temporal_derivative_filter = torch.Tensor(np.array([-1, 1]))
        self.temporal_derivative_filter = torch.reshape(self.temporal_derivative_filter, (1, 2, 1, 1))
        # (2). spatial derivative:
        self.grad_x_kernel = torch_get_4D(torch.Tensor(np.array([-1, 0, 1]) / 2), 'W')
        self.grad_y_kernel = self.grad_x_kernel.permute([0, 1, 3, 2])
        # (3). temporal averaging:
        self.temporal_averaging_filter_before_spatial_gradient = torch.ones((1, self.temporal_lowpass_number_of_steps, 1, 1, 1)) / self.temporal_lowpass_number_of_steps

        self.spatial_lowpass_before_temporal_derivative_filter_fft_x = None
        self.X = None

    def forward(self, input_tensor, reference_tensor=None):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.spatial_lowpass_before_temporal_derivative_filter_fft_x is None:
            ### Get FFT Filters: ###
            self.spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
            self.spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])
            self.temporal_derivative_fft_filter = torch.fft.fftn(self.temporal_derivative_filter, s=T, dim=[1])
            self.grad_x_filter_fft = torch.fft.fftn(self.grad_x_kernel, s=W, dim=[-1])
            self.grad_y_filter_fft = torch.fft.fftn(self.grad_y_kernel, s=H, dim=[-2])
            self.temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(self.temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

        if self.X is None:
            derivative_filter_lowpass_equivalent_length = len(self.spatial_lowpass_before_temporal_derivative_filter_x)
            shift_filter_length = 11 #TODO: later on with the full version there will be an equivalent shift filter
            invalid_frame_size = np.round((derivative_filter_lowpass_equivalent_length + shift_filter_length)/4)
            x_start = -input_tensor.shape[-1]/2 + 0.5 + invalid_frame_size
            x_stop = input_tensor.shape[-1]/2 - 0.5 - invalid_frame_size
            y_start = -input_tensor.shape[-2] / 2 + 0.5 + invalid_frame_size
            y_stop = input_tensor.shape[-2] / 2 - 0.5 - invalid_frame_size
            x_vec = my_linspace_step2(x_start, x_stop, 1)
            y_vec = my_linspace_step2(y_start, y_stop, 1)
            [X,Y] = np.meshgrid(x_vec, y_vec)
            self.X = torch.tensor(X).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.Y = torch.tensor(Y).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.X_flattened = torch.flatten(self.X, -3, -1)
            self.Y_flattened = torch.flatten(self.Y, -3, -1)

        ######################################################################################################
        ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###
        # # (1). Only input tensor
        # px = self.torch_convn_layer.forward(input_tensor, self.grad_x_kernel.flatten(), -1)
        # py = self.torch_convn_layer.forward(input_tensor, self.grad_y_kernel.flatten(), -2)
        # (2). input_tensor + reference_tensor
        I_x = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_x_kernel.flatten(), -1)
        I_y = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_y_kernel.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        # TODO: perhapse don't(!!!!) center crop now, but later. i can also hard-code this into the fast_binning function start and stop indices
        start_index = np.int16(floor(self.params.dif_len / 2))
        I_x = I_x[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        I_y = I_y[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ######################################################################################################

        ######################################################################################################
        ### Calculate Temporal Derivative: ###
        I_t_xconv = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
        I_t_yconv = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        I_t_xconv = I_t_xconv[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        I_t_yconv = I_t_yconv[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        #####################################################################################################


        ######################################################################################################
        ### Find Shift Between Frames Matrix Notation: ###
        #(*). i assume that gimbaless rotation is use for the ENTIRE FRAME only (no local predictions), so global sums only
        B2, T2, C2, H2, W2 = I_x.shape
        I_x = torch.flatten(I_x, -3, -1)
        I_y = torch.flatten(I_y, -3, -1)
        I_t_xconv = torch.flatten(I_t_xconv, -3, -1)
        I_t_yconv = torch.flatten(I_t_yconv, -3, -1)
        M1_operator = I_x * self.Y_flattened - I_y * self.X_flattened
        M2_operator = -I_x * self.X_flattened**2 - I_y * self.X_flattened * self.Y_flattened
        M3_operator = -I_x * self.X_flattened * self.Y_flattened - I_y * self.Y_flattened**2

        p = torch.cat([I_x.unsqueeze(-1), I_y.unsqueeze(-1), M1_operator.unsqueeze(-1), M2_operator.unsuqeeze(-1), M3_operator.unsqueeze(-1)], -1).squeeze(0)
        p_mat = torch.bmm(torch.transpose(p, -1, -2), p)
        d1 = torch.linalg.inv(p_mat)
        d2 = torch.cat([(I_t_xconv*I_x).sum(-1, True),
                        (I_t_yconv*I_y).sum(-1, True),
                        (I_t_xconv*(I_x*self.Y_flattened) - I_t_yconv*(I_y*self.X_flattened)).sum(-1, True),
                        (I_t_xconv*(-I_x*self.X_flattened**2) + I_t_yconv*(-I_y*self.X_flattened*self.Y_flattened)).sum(-1, True),
                        (I_t_xconv*(-I_x*self.X_flattened*self.Y_flattened) + I_t_yconv*(-I_y*self.Y_flattened**2)).sum(-1, True)], -1).squeeze(0).unsqueeze(-1)
        d_vec = torch.matmul(d1, d2).squeeze(-1)
        delta_x = d_vec[:, 0]
        delta_y = d_vec[:, 1]
        delta_theta = d_vec[:, 2]
        delta_h7 = d_vec[:, 3]
        delta_h8 = d_vec[:, 4]

        return delta_x, delta_y, delta_theta, delta_h7, delta_h8


    def forward_weighted(self, input_tensor, reference_tensor=None, W=None):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.spatial_lowpass_before_temporal_derivative_filter_fft_x is None:
            ### Get FFT Filters: ###
            self.spatial_lowpass_before_temporal_derivative_filter_fft_x = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_x, s=W, dim=[-1])
            self.spatial_lowpass_before_temporal_derivative_filter_fft_y = torch.fft.fftn(self.spatial_lowpass_before_temporal_derivative_filter_y, s=H, dim=[-2])
            self.temporal_derivative_fft_filter = torch.fft.fftn(self.temporal_derivative_filter, s=T, dim=[1])
            self.grad_x_filter_fft = torch.fft.fftn(self.grad_x_kernel, s=W, dim=[-1])
            self.grad_y_filter_fft = torch.fft.fftn(self.grad_y_kernel, s=H, dim=[-2])
            self.temporal_averaging_filter_before_spatial_gradient_fft = torch.fft.fftn(self.temporal_averaging_filter_before_spatial_gradient, s=T, dim=[1])

        if self.X is None:
            derivative_filter_lowpass_equivalent_length = len(self.spatial_lowpass_before_temporal_derivative_filter_x)
            shift_filter_length = 11 #TODO: later on with the full version there will be an equivalent shift filter
            invalid_frame_size = np.round((derivative_filter_lowpass_equivalent_length + shift_filter_length)/4)
            x_start = -input_tensor.shape[-1]/2 + 0.5 + invalid_frame_size
            x_stop = input_tensor.shape[-1]/2 - 0.5 - invalid_frame_size
            y_start = -input_tensor.shape[-2] / 2 + 0.5 + invalid_frame_size
            y_stop = input_tensor.shape[-2] / 2 - 0.5 - invalid_frame_size
            x_vec = my_linspace_step2(x_start, x_stop, 1)
            y_vec = my_linspace_step2(y_start, y_stop, 1)
            [X,Y] = np.meshgrid(x_vec, y_vec)
            self.X = torch.tensor(X).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.Y = torch.tensor(Y).to(input_tensor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.X_flattened = torch.flatten(self.X, -3, -1)
            self.Y_flattened = torch.flatten(self.Y, -3, -1)

        ######################################################################################################
        ### Calculate Spatial Gradient (maybe average matrices to get better gradient SNR): ###
        # # (1). Only input tensor
        # I_x = self.torch_convn_layer.forward(input_tensor, self.grad_x_kernel.flatten(), -1)
        # I_y = self.torch_convn_layer.forward(input_tensor, self.grad_y_kernel.flatten(), -2)
        # (2). input_tensor + reference_tensor
        I_x = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_x_kernel.flatten(), -1)
        I_y = self.torch_convn_layer.forward((input_tensor + reference_tensor) / 2, self.grad_y_kernel.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        # TODO: perhapse don't(!!!!) center crop now, but later. i can also hard-code this into the fast_binning function start and stop indices
        start_index = np.int16(floor(self.params.dif_len / 2))
        I_x = I_x[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        I_y = I_y[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        ######################################################################################################

        ######################################################################################################
        ### Calculate Temporal Derivative: ###
        I_t_xconv = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_x.flatten(), -1)
        I_t_yconv = self.torch_convn_layer.forward(input_tensor - reference_tensor, self.spatial_lowpass_before_temporal_derivative_filter_y.flatten(), -2)

        ### Cut-Out Invalid Parts (Center-Crop): ###
        I_t_xconv = I_t_xconv[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        I_t_yconv = I_t_yconv[:, :, :, start_index:H - (start_index - 1), start_index:W - (start_index - 1)]
        #####################################################################################################


        ######################################################################################################
        ### Find Shift Between Frames Matrix Notation: ###
        #(*). i assume that gimbaless rotation is use for the ENTIRE FRAME only (no local predictions), so global sums only
        B2, T2, C2, H2, W2 = I_x.shape
        I_x = torch.flatten(I_x, -3, -1)
        I_y = torch.flatten(I_y, -3, -1)
        I_t_xconv = torch.flatten(I_t_xconv, -3, -1)
        I_t_yconv = torch.flatten(I_t_yconv, -3, -1)
        M1_operator = I_x * self.Y_flattened - I_y * self.X_flattened
        M2_operator = -I_x * self.X_flattened ** 2 - I_y * self.X_flattened * self.Y_flattened
        M3_operator = -I_x * self.X_flattened * self.Y_flattened - I_y * self.Y_flattened ** 2
        W_flattened = torch.flaten(W, -3, -1)

        p = torch.cat([I_x.unsqueeze(-1),
                       I_y.unsqueeze(-1),
                       M1_operator.unsqueeze(-1),
                       M2_operator.unsuqeeze(-1),
                       M3_operator.unsqueeze(-1)], -1).squeeze(0)
        Wp = torch.cat([(W_flattened*I_x).unsqueeze(-1),
                        (W_flattened*I_y).unsqueeze(-1),
                        (W_flattened*M1_operator).unsqueeze(-1),
                        (W_flattened*M2_operator).unsqueeze(-1),
                        (W_flattened*M3_operator).unsqueeze(-1)], -1).squeeze(0)
        p_mat = torch.bmm(torch.transpose(Wp, -1, -2), p)
        d1 = torch.linalg.inv(p_mat)
        d2 = torch.cat([(W_flattened*I_t_xconv*I_x).sum(-1, True),
                        (W_flattened*I_t_yconv*I_y).sum(-1, True),
                        (W_flattened*I_t_xconv*(I_x*self.Y_flattened) - W_flattened*I_t_yconv*(I_y*self.X_flattened)).sum(-1, True),
                        (W_flattened*I_t_xconv*(-I_x * self.X_flattened ** 2) + W_flattened*I_t_yconv*(-I_y * self.X_flattened * self.Y_flattened)).sum(-1, True),
                        (W_flattened*I_t_xconv*(-I_x * self.X_flattened * self.Y_flattened) + W_flattened*I_t_yconv*(-I_y * self.Y_flattened ** 2)).sum(-1, True)], -1).squeeze(0).unsqueeze(-1)
        d_vec = torch.matmul(d1, d2).squeeze(-1)
        delta_x = d_vec[:, 0]
        delta_y = d_vec[:, 1]
        delta_theta = d_vec[:, 2]
        delta_h7 = d_vec[:, 3]
        delta_h8 = d_vec[:, 4]

        return delta_x, delta_y, delta_theta


def testing1_Gimbaless_Rotation_3D_1():
    ### for testing purposes: ###
    T = 128
    H = 64
    W = 64
    speckle_size = 3
    shift_layer = Shift_Layer_Torch()
    speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(speckle_size, H, 0, 1, 1, 0)
    speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    C, H, W = speckle_pattern.shape
    ### Get Input Tensor by stacking the different speckle patterns: ###
    input_tensor = torch.cat([speckle_pattern] * T, 0) + 5
    current_shift = 0
    ### Actually Rotate: ###
    shift_H = 0
    shift_W = 0
    rotation_degrees = np.linspace(0,1,128)
    rotation_rads = rotation_degrees * np.pi/180
    input_tensor_warped = affine_transform_interpolated(input_tensor.unsqueeze(1).unsqueeze(0),
                                                        construct_tensor(shift_H),
                                                        construct_tensor(shift_W),
                                                        construct_tensor(rotation_degrees * np.pi / 180),
                                                        construct_tensor(1),
                                                        warp_method='bicubic',
                                                        expand=False).squeeze(0).squeeze(1)
    input_tensor_warped[T//2:T//2+1,:,:] = input_tensor[T//2:T//2+1,:,:]
    input_tensor = input_tensor_warped
    # ### Actually Shift: ###
    # shifts_vec = torch.randn(T) * 0.1
    # real_shifts_to_reference_tensor = shifts_vec - shifts_vec[T // 2]
    # # real_shifts_to_reference_tensor = shifts_vec[1:T] - shifts_vec[0:T-1]
    # input_tensor = shift_layer.forward(input_tensor, shifts_vec.cpu().numpy(), shifts_vec.cpu().numpy())
    ### Unsqueeze to B,T,C,H,W : ###
    input_tensor = input_tensor.unsqueeze(1)  # [T,H,W]->[T,C,H,W]
    input_tensor = input_tensor.unsqueeze(0)  # [T,C,H,W] -> [B,T,C,H,W]

    ### Initialize Gimbaless Layer: ###
    gimbaless_layer = Gimbaless_Rotation_Layer_Torch()

    ### Use Gimbaless: ###
    (delta_x, delta_y, delta_rotation) = gimbaless_layer.forward(input_tensor, input_tensor[:,T//2:T//2+1])

    ### Plot Results: ###
    plot_torch(torch.tensor(rotation_degrees), -delta_rotation)
    plt.plot(rotation_degrees, rotation_rads, 'g')
    plt.plot(-delta_rotation - rotation_rads)

# testing1_Gimbaless_Rotation_3D_1()


def test_Gimbaless_3D_2():
    ### for testing purposes: ###
    T = 128
    H = 64
    W = 64
    speckle_size = 3
    shift_layer = Shift_Layer_Torch()
    speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(speckle_size, H, 0, 1, 1, 0)
    speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
    C, H, W = speckle_pattern.shape
    ### Get Input Tensor by stacking the different speckle patterns: ###
    input_tensor = torch.cat([speckle_pattern] * T, 0) + 5
    current_shift = 0
    ### Actually Shift: ###
    shifts_vec = torch.randn(T) * 0.1
    real_shifts_to_reference_tensor = shifts_vec - shifts_vec[T // 2]
    # real_shifts_to_reference_tensor = shifts_vec[1:T] - shifts_vec[0:T-1]
    input_tensor = shift_layer.forward(input_tensor, shifts_vec.cpu().numpy(), shifts_vec.cpu().numpy())
    input_tensor = input_tensor.unsqueeze(1)  # [T,H,W]->[T,C,H,W]
    input_tensor = input_tensor.unsqueeze(0)  # [T,C,H,W] -> [B,T,C,H,W]

    ### Initialize Gimbaless Layer: ###
    gimbaless_layer = Gimbaless_Layer_Torch()

    ### Use Gimbaless: ###
    (delta_x, delta_y) = gimbaless_layer(input_tensor)

    ### Get Average For Plotting Purposes: ###
    delta_x = delta_x.mean([-1, -2]).flatten()
    delta_y = delta_y.mean([-1, -2]).flatten()

    ### Plot Results: ###
    plot_torch(-delta_x)
    plt.plot(real_shifts_to_reference_tensor)

# test_Gimbaless_3D_2()
#####################################################################################################


class FFT_OLA_PerPixel_Layer_Torch(nn.Module):

    # Initialize this with a module
    def __init__(self):
        super(FFT_OLA_PerPixel_Layer_Torch, self).__init__()

        ### Basic Parameters: ###
        self.samples_per_frame = 128
        self.overlap_samples_per_frame = float(int16((self.samples_per_frame) * 1/2 - 0))
        self.non_overlapping_samples_per_frame = self.samples_per_frame - self.overlap_samples_per_frame

        ### Frame Window: ###
        self.frame_window = scipy.io.loadmat(
            '/home/mafat/PycharmProjects/DATA_FOLDER/spatial_lowpass_9tap.mat')['spatial_lowpass_before_temporal_derivative_filter'].flatten()  # Read this from file
        self.frame_window = torch_get_5D(self.frame_window, 'T')
        self.frame_window_length = len(self.frame_window)

        ### Filter Itself: ###
        self.filter_Numerator = scipy.io.loadmat(
            '/home/mafat/PycharmProjects/DATA_FOLDER/spatial_lowpass_9tap.mat')['spatial_lowpass_before_temporal_derivative_filter'].flatten()  # Read this from file
        self.filter_length = len(self.filter_Numerator)
        self.filter_Numerator = torch_get_5D(self.filter_Numerator, 'T')

        self.FFT_size = pow(2, np.ceil(np.log(self.samples_per_frame + self.filter_length - 1)/np.log(2)))
        self.filter_fft = torch.fft.fftn(self.filter_Numerator, s=self.FFT_size, dim=[-1])
        self.filter_fft = torch_get_5D(self.filter_fft, 'T')

        # Initialize lookahead buffer for overlap add operation:
        self.lookahead_buffer_for_overlap_add = None

    def forward(self, input_tensor):
        #### Get Mat Size: ###
        B, T, C, H, W = input_tensor.shape

        if self.lookahead_buffer_for_overlap_add is None:
           self.lookahead_buffer_for_overlap_add = torch.zeros(1, self.FFT_size, 1, H, W).to(input_tensor.device)
           self.zeros_mat = torch.zeros(1, self.overlap_samples_per_frame, 1, H, W).to(input_tensor.device)

        ### Actually Filter: ###
        #(1). window signal:
        input_tensor = input_tensor * self.frame_window
        #(2). calculate buffered and windowed frame fft:
        input_tensor_fft = torch.fft.fftn(input_tensor, s=self.FFT_size, dim=[1])
        #(3). calculate time domain filtered signal:
        filtered_signal = torch.fft.ifftn(input_tensor_fft * self.filter_fft, dim=[1])

        ### Overlap-Add Method: ###
        #(1). overlap add:
        self.lookahead_buffer_for_overlap_add = self.lookahead_buffer_for_overlap_add + filtered_signal
        #(2). get current valid part of the overlap-add:
        filtered_signal_final_valid = self.lookahead_buffer_for_overlap_add[:, 0:self.non_overlapping_samples_per_frame]
        #(3). shift lookahead buffer tail to beginning of buffer for next overlap-add:
        self.lookahead_buffer_for_overlap_add = torch.cat((self.lookahead_buffer_for_overlap_add[:, self.non_overlapping_samples_per_frame+1:T],
                                                      self.zeros_mat), 1)

        return filtered_signal_final_valid



# ######################################################################################################

#
#

#
#
######################################################################################################################################################################################





##########################################################################################################################################################################################################################################################################################################################################################################################
import torch
import torch.nn.functional as F
# loading images
# import image_registration


# import the necessary packages
import numpy as np
import cv2
import torch.fft
# from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

import imutils

### use ORB to detect features and match features: ###
def align_images_OpenCVFeatures(image, template, maxFeatures=500, keepPercent=0.2, debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
    # invariant features
    orb = cv2.ORB_create(maxFeatures)
    # orb = cv2.xfeatures2d.SIFT_create()
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)

    # match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
    # the "more similar" the features are)
    matches = sorted(matches, key=lambda x: x.distance)

    # keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # allocate memory for the keypoints (x,y-coordinates) from the
    # top matches -- we'll use these coordinates to compute our homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)  #TODO: is there some other method?

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # return the aligned image
    return aligned


def align_images_ECC(img1, img2):
    # Find the width and height of the color image
    height = img1.shape[0]
    width = img1.shape[1]

    # Allocate space for aligned image
    im_aligned = np.zeros((height, width, 1), dtype=np.uint8)

    # Define motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY  # TODO: what other types are there

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

    # Warp the blue and green channels to the red channel
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(img1),
                                             get_gradient(img2),
                                             warp_matrix,
                                             warp_mode,
                                             criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use Perspective warp when the transformation is a Homography
        im_aligned = cv2.warpPerspective(img2,
                                         warp_matrix,
                                         (width, height),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use Affine warp when the transformation is not a Homography
        im_aligned = cv2.warpAffine(img2,
                                    warp_matrix,
                                    (width, height),
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im_aligned




### Using Feature Detection and Homography matrix: ###
def register_images_FeatureHomography(img1, img2):
    img1 = img1.squeeze(0).squeeze(0)
    img2 = img2.squeeze(0).squeeze(0)
    img1 = uint8((img1 * 255).clamp(0, 255))
    img2 = uint8((img2 * 255).clamp(0, 255))
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 # matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img2, homography, (width, height))
    transformed_img = transformed_img / 255

    shift_x = homography[0, 2]
    shift_y = homography[1, 2]
    shifts_vec = (-shift_x, -shift_y)

    return shifts_vec, transformed_img


### Using Optical Flow: ###
def register_images_OpticalFlow(img_source, img_to_align, mc_alg='DeepFlow'):
    # Applies to img_to_align a transformation which converts it into img_source.
    # Args:
    # 	img_to_align: HxWxC image
    # 	img_source: HxWxC image
    # 	mc_alg: selects between DeepFlow, SimpleFlow, and TVL1. DeepFlow runs by default.
    # Returns:
    # 	HxWxC aligned image

    img_to_align_numpy = torch.zeros_like(img_to_align)
    img_source_numpy = torch.zeros_like(img_source)
    img_to_align_numpy[:] = img_to_align[:]
    img_source_numpy[:] = img_source[:]

    ### Tensor To Numpy: ###
    if type(img_to_align_numpy) == torch.Tensor:
        img_to_align_numpy = img_to_align_numpy.cpu().numpy().squeeze(0).squeeze(0)
    if type(img_source_numpy) == torch.Tensor:
        img_source_numpy = img_source_numpy.cpu().numpy().squeeze(0).squeeze(0)

    ### Some of the algorithms demand uint8, do it: ###
    # img_source_copy = (img_source_copy * 255).clip(0, 255)
    # img_to_align_copy = (img_to_align_copy * 255).clip(0, 255)
    # img_source_copy = img_source_copy.astype(np.uint8)
    # img_to_align_copy = img_to_align_copy.astype(np.uint8)

    img0 = img_to_align_numpy[:, :]
    img1 = img_source_numpy[:, :]
    out_img = None

    # Align frames according to selection in mc_alg
    flow = estimate_invflow(img0, img1, mc_alg)

    ### Get Global Shift: ###
    shift_x = flow[0].mean()
    shift_y = flow[1].mean()
    shifts_vec = (shift_x, shift_y)

    # rectifier
    out_img = warp_flow(img_to_align_numpy, flow)

    return shifts_vec, out_img, flow


def estimate_AffineParamters_FromOpticalFlow(frame1, frame2, optical_flow):
    1  # TODO: for shifts can simply use averages. for more complex homographies more elaborate fitting strategies are needed


def estimate_invflow(img0, img1, me_algo):
    # Estimates inverse optical flow by using the me_algo algorithm.
    # # # img0, img1 have to be uint8 grayscale
    # assert img0.dtype == 'uint8' and img1.dtype == 'uint8'

    # Create estimator object
    if me_algo == "DeepFlow":
        of_estim = cv2.optflow.createOptFlow_DeepFlow()
    elif me_algo == "SimpleFlow":
        of_estim = cv2.optflow.createOptFlow_SimpleFlow()
    elif me_algo == "TVL1":
        of_estim = cv2.DualTVL1OpticalFlow_create()
    else:
        raise Exception("Incorrect motion estimation algorithm")

    # Run flow estimation (inverse flow)
    flow = of_estim.calc(img1, img0, None)
    #	flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def warp_flow(img, flow):
    # Applies to img the transformation described by flow.
    assert len(flow.shape) == 3 and flow.shape[-1] == 2

    hf, wf = flow.shape[:2]
    # flow 		= -flow
    flow[:, :, 0] += np.arange(wf)
    flow[:, :, 1] += np.arange(hf)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res







########################################################################################################################################################################



def register_images_batch(images_batch,
                          registration_algorithm,
                          search_area=5,
                          inner_crop_size_to_use=1000,
                          downsample_factor=1,  # binning / average_pooling
                          flag_do_initial_SAD=True, initial_SAD_search_area=5, center_frame_index=None):
    ### Initial handling of indices according to input type: ###
    if type(images_batch) == list:
        if center_frame_index is None:
            center_frame_index = len(images_batch) // 2
        number_of_images = len(images_batch)
    if type(images_batch) == torch.Tensor:
        # if [T,C,H,W] i assume T is frames index, if [B,T,C,H,W]
        if len(images_batch.shape) == 4:
            frames_dim = 0
            number_of_images = images_batch.shape[0]
            if center_frame_index is None:
                center_frame_index = number_of_images // 2
            center_frame = images_batch[center_frame_index:center_frame_index + 1, :, :, :]
        elif len(images_batch.shape) == 5:
            frames_dim = 1
            number_of_images = images_batch.shape[1]
            if center_frame_index is None:
                center_frame_index = number_of_images // 2
            center_frame = images_batch[:, center_frame_index:center_frame_index + 1, :, :, :]

    ### Loop over images and register each of them in reference to reference_frame: ###
    # TODO: this is compatible with algorithms which work on two images at a time, some algorithms, like maor, act on all algorithms together.
    warped_frames_list = []
    shifts_array_list = []
    for frame_index in np.arange(number_of_images):
        ### Register Individual Images Pairs
        if frame_index != center_frame_index:
            ### Get current frame: ###
            if len(center_frame.shape) == 4:
                current_frame = images_batch[frame_index:frame_index + 1, :, :, :]
            elif len(center_frame.shape) == 5:
                current_frame = images_batch[:, frame_index:frame_index + 1, :, :, :]

            ### Register current frame to reference frame: ###
            shifts_array, current_frame_warped = register_images(center_frame.data, current_frame.data,
                                                                 registration_algorithm,
                                                                 search_area=search_area,
                                                                 inner_crop_size_to_use=inner_crop_size_to_use,
                                                                 downsample_factor=downsample_factor,
                                                                 # binning / average_pooling
                                                                 flag_do_initial_SAD=flag_do_initial_SAD,
                                                                 initial_SAD_search_area=initial_SAD_search_area)
        else:
            current_frame = center_frame
            current_frame_warped = center_frame
            shifts_array = (0, 0)

        ### Append To Lists: ###
        warped_frames_list.append(current_frame_warped)
        shifts_array_list.append(shifts_array)

    return shifts_array_list, warped_frames_list



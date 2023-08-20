import sys
sys.path.append('/home/avraham/mafat/Anvil/RDND')
from RapidBase.import_all import *


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


### TODO: temp test, to be commented out: ###
def test_single_BW_image_SingleAffineTransform():
    input_tensor = read_image_default_torch()
    B,C,H,W = input_tensor.shape
    (cX, cY) = (W // 2, H // 2)
    input_tensor = RGB2BW(input_tensor)
    rotation_angle = torch.Tensor([20]) #[deg]
    shift_H = torch.Tensor([0])
    shift_W = torch.Tensor([0])
    scale_factor = torch.Tensor([1])

    #(1). OpenCV implementation:
    input_tensor_numpy = input_tensor.cpu().numpy()[0,0]
    num_rows, num_cols = input_tensor_numpy.shape[-2:]
    affine_matrix = cv2.getRotationMatrix2D((cX, cY), rotation_angle, 1.0)
    input_tensor_numpy_rotation = cv2.warpAffine(input_tensor_numpy, affine_matrix, (num_cols, num_rows))
    #(2). FFT Shift Function:
    input_tensor_warped = warp_tensors_affine(input_tensor, shift_x=shift_W, shift_y=shift_H, scale=scale_factor, rotation_angle=rotation_angle)
    #(3). FFT Shift Layer:
    warp_tensors_affine_layer = Warp_Tensors_Affine_Layer()
    input_tensor_warped_layer = warp_tensors_affine_layer.forward(input_tensor, shift_W, shift_H, scale_factor, torch.Tensor([rotation_angle]))

    # Compare Results: #
    #TODO: results don't look right!!! something is wrong with my FFT layer, it look like there's a skew!!!
    input_tensor_warped = crop_torch_batch(input_tensor_warped, (H, W))
    input_tensor_warped_layer = crop_torch_batch(input_tensor_warped_layer, (H, W))
    figure()
    imshow_torch(torch.Tensor(input_tensor_numpy_rotation).unsqueeze(0).unsqueeze(0))
    imshow_torch(input_tensor_warped)
    imshow_torch(input_tensor_warped_layer)
    figure()
    imshow_torch(input_tensor_warped_layer - torch.Tensor(input_tensor_numpy_rotation).unsqueeze(0).unsqueeze(0))
    figure()
    imshow_torch(input_tensor_warped_layer - input_tensor_warped)


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
test_circular_cross_correlation()

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
    shift_x = np.array([5])
    shift_y = np.array([-5])
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
        input_tensor_2 = warp_tensors_affine_layer.forward(input_tensor_1, shift_x, shift_y, scale_factor_warp, rotation_angle)
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
        #(2).
        fft_logpolar_registration_layer = FFT_LogPolar_Rotation_Scaling_Translation_Registration_Layer().cuda()
        recovered_angle, recovered_scale, recovered_translation, input_tensor_1_displaced = fft_logpolar_registration_layer.forward(
            input_tensor_1,
            input_tensor_2,
            downsample_factor=1)
        inferred_rotation_list_2.append(recovered_angle)
        inferred_scaling_list_2.append(recovered_scale)
        inferred_shift_x_list_2.append(recovered_translation[1])
        inferred_shift_y_list_2.append(recovered_translation[0])

    ### Present Outputs: ###
    #(1). Rotation
    figure()
    plot(rotation_angles, rotation_angles)
    plot(rotation_angles, inferred_rotation_list_1)
    plot(rotation_angles, inferred_rotation_list_2)
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
    # (1). Translation
    figure()
    plot(rotation_angles, shift_x[0] * np.ones_like(rotation_angles))
    plot(rotation_angles, inferred_shift_x_list_1)
    plt.xlabel('Rotation Angle')
    plt.legend(['GT Shift X', 'torch layer'])
    plt.title('Translation plot')
    figure()
    plot(rotation_angles, shift_y[0] * np.ones_like(rotation_angles))
    plot(rotation_angles, inferred_shift_y_list_2)
    plt.legend(['GT Shift Y', 'torch layer'])
    plt.title('Translation Y plot')
    plt.xlabel('Rotation Angle')






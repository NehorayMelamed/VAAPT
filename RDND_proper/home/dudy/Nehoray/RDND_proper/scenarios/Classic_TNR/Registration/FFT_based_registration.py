from RapidBase.import_all import *
from torch.utils.data import DataLoader
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Registration.Registration_StarTrack_Functions import warp_tensors_affine, \
    warp_tensor_affine_matrix
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration.Transforms_Grids import *

warp_object = Warp_Object()

try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

def torch_fft2(input_image):
    return torch.fft.fftn(input_image, dim=[-1,-2])
def torch_ifft2(input_image):
    return torch.fft.ifftn(input_image, dim=[-1,-2])


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
    i0, i1 = numpy.unravel_index(max_index, numpy_shape)
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
        t = numpy.zeros_like(input_image_1)
        t[: input_image_2_scaled_rotated.shape[0], : input_image_2_scaled_rotated.shape[1]] = input_image_2_scaled_rotated
        input_image_2_scaled_rotated = t
    elif input_image_2_scaled_rotated.shape > input_image_1.shape:
        input_image_2_scaled_rotated = input_image_2_scaled_rotated[: input_image_1.shape[0], : input_image_1.shape[1]]
    
    ### Get Translation Using Phase Cross Correlation: ###
    input_image_1_fft = torch_fft2(input_image_1)  #TODO: can save on calculation here since i calculated this above
    input_image_2_fft = torch_fft2(input_image_2_scaled_rotated)  #TODO: can probably save on calculation here if i use FFT interpolation for zoom+rotation above
    phase_cross_correlation = abs(torch_ifft2((input_image_1_fft * input_image_2_fft.conjugate()) / (abs(input_image_1_fft) * abs(input_image_2_fft))))
    max_index = torch.argmax(phase_cross_correlation)
    max_index = np.atleast_1d(max_index.cpu().numpy())[0]
    numpy_shape = phase_cross_correlation.shape.cpu().numpy()
    t0, t1 = numpy.unravel_index(max_index, numpy_shape)
    ### Correct For FFT_Shift Wrap-Arounds: ###
    if t0 > input_image_1_fft.shape[0] // 2:
        t0 -= input_image_1_fft.shape[0]
    if t1 > input_image_1_fft.shape[1] // 2:
        t1 -= input_image_1_fft.shape[1]

    ### Shift Second Image According To Found Translation: ###
    input_image_2_scaled_rotated_translated = ndii.shift(input_image_2_scaled_rotated, [t0, t1])  #TODO: see what is faster, bilinaer interpolation or (multiplication+FFT)?

    ### Correct Parameters For ndimage's Internal Processing: ###
    if angle > 0.0:
        d = int(int(input_image_2.shape[1] / scale) * math.sin(math.radians(angle)))
        t0, t1 = t1, d + t0
    elif angle < 0.0:
        d = int(int(input_image_2.shape[0] / scale) * math.sin(math.radians(angle)))
        t0, t1 = d + t1, d + t0
    scale = (input_image_2.shape[1] - 1) / (int(input_image_2.shape[1] / scale) - 1)

    return input_image_2_scaled_rotated_translated, scale, angle, [-t0, -t1]


from numpy.fft import fft2, ifft2, fftshift
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
    input_image_1 = input_image_1[0,0].numpy()
    input_image_2 = input_image_2[0,0].numpy()
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
    input_image_1_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_2_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_2_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
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
    phase_cross_correlation = abs(ifft2((input_image_1_fft_abs_LogPolar_numpy * input_image_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(ifft2((input_image_2_fft_abs_LogPolar_numpy * input_image_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
        # phase_cross_correlation[0,0] = 0
        i0, i1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)
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
        t = numpy.zeros_like(input_image_1)
        t[: input_image_2_scaled_rotated.shape[0], : input_image_2_scaled_rotated.shape[1]] = input_image_2_scaled_rotated
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
    #TODO: probably need to center crop images here!
    phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate()) / (abs(input_image_1_fft) * abs(input_image_2_fft))))
    # phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate())))
    t0, t1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)

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
    input_image_1 = input_image_1[0,0].numpy()
    input_image_2 = input_image_2[0,0].numpy()
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
    #TODO: add possibility of divide factor for radius
    input_image_1_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_2_fft_abs_LogPolar_torch, log_base, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_scipy_torch(torch.Tensor(input_image_2_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_torch.cpu().numpy()[0,0]

    ### Calculate Cross Correlation: ###
    input_image_1_fft_abs_LogPolar_numpy = fft2(input_image_1_fft_abs_LogPolar_numpy)
    input_image_2_fft_abs_LogPolar_numpy = fft2(input_image_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_image_1_fft_abs_LogPolar_numpy) * abs(input_image_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(ifft2((input_image_1_fft_abs_LogPolar_numpy * input_image_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(ifft2((input_image_2_fft_abs_LogPolar_numpy * input_image_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
        # phase_cross_correlation[0,0] = 0
        i0, i1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)
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
    #TODO: isn't the above BETTER then rotating the original image around an unknown center of rotation?!?!?!....

    ### Warp Second Image Towards Reference Image: ###
    # #(1). Scipy:
    input_image_2_scaled_rotated = ndii.zoom(input_image_2_original, 1.0 / scale)
    input_image_2_scaled_rotated = ndii.rotate(input_image_2_scaled_rotated, -angle)
    #(2). Torch Affine:
    input_image_1_original_torch = torch.Tensor(input_image_1_original).unsqueeze(0).unsqueeze(0)
    input_image_2_original_torch = torch.Tensor(input_image_2_original).unsqueeze(0).unsqueeze(0)
    input_image_2_original_torch_rotated = warp_tensor_affine_matrix(input_image_2_original_torch, 0, 0, scale, -angle)
    imshow_torch(input_image_1_original_torch - input_image_2_original_torch_rotated)
    ### Crop: ###
    input_image_1_original = input_image_1_original_torch.numpy()[0,0]
    input_image_2_scaled_rotated = input_image_2_original_torch_rotated.numpy()[0,0]
    imshow(input_image_1_original - input_image_2_scaled_rotated)

    if input_image_2_scaled_rotated.shape < input_image_1.shape:
        t = numpy.zeros_like(input_image_1)
        t[: input_image_2_scaled_rotated.shape[0], : input_image_2_scaled_rotated.shape[1]] = input_image_2_scaled_rotated
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
    #TODO: probably need to center crop images here!
    phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate()) / (abs(input_image_1_fft) * abs(input_image_2_fft))))
    # phase_cross_correlation = abs(ifft2((input_image_1_fft * input_image_2_fft.conjugate())))
    t0, t1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)

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
    input_image_1_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0))
    input_image_2_fft_abs_LogPolar_torch, log_base_torch, (radius_torch, theta_torch), (x_torch, y_torch), flow_grid = logpolar_torch(torch.Tensor(input_image_2_fft_abs).unsqueeze(0).unsqueeze(0))
    # imshow(input_image_1_fft_abs_LogPolar_numpy); imshow_torch(input_image_1_fft_abs_LogPolar_torch)
    # grid_diff = x - x_torch.cpu().numpy()[0,:,:,0]

    ### Scipy LogPolar: ###
    input_shape = input_image_1_fft_abs.shape
    radius_to_use = input_shape[0] // 1 # only take lower frequencies. #TODO: Notice! the choice of the log-polar conversion is for your own choosing!!!!!!
    input_image_1_fft_abs_LogPolar_scipy = warp_polar(input_image_1_fft_abs, radius=radius_to_use,
                                                      output_shape=input_shape, scaling='log', order=0)
    input_image_2_fft_abs_LogPolar_scipy = warp_polar(input_image_2_fft_abs, radius=radius_to_use,
                                                      output_shape=input_shape, scaling='log', order=0)
    imshow(input_image_1_fft_abs_LogPolar_scipy); imshow_torch(input_image_1_fft_abs_LogPolar_torch)

    ### External LopPolar: ###
    # logpolar_grid_tensor = logpolar_grid(input_image_1_fft_abs.shape, ulim=(None, np.log(2.) / 2.), vlim=(0, np.pi), out=None, device='cuda').cuda()
    logpolar_grid_tensor = logpolar_grid(input_image_1_fft_abs.shape, ulim=(np.log(0.01), np.log(1)), vlim=(0, 2*np.pi), out=None, device='cuda').cuda()
    input_image_1_fft_abs_LogPolar_torch_grid = F.grid_sample(torch.Tensor(input_image_1_fft_abs).unsqueeze(0).unsqueeze(0).cuda(), logpolar_grid_tensor.unsqueeze(0))
    input_image_1_fft_abs_LogPolar_numpy_grid = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
    # imshow_torch(input_image_1_fft_abs_LogPolar_torch)

    ### Calculate Phase Cross Correlation: ###
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_torch.cpu().numpy()[0,0]
    input_image_1_fft_abs_LogPolar_numpy = input_image_1_fft_abs_LogPolar_scipy
    input_image_2_fft_abs_LogPolar_numpy = input_image_2_fft_abs_LogPolar_scipy

    input_image_1_fft_abs_LogPolar_numpy = fft2(input_image_1_fft_abs_LogPolar_numpy)
    input_image_2_fft_abs_LogPolar_numpy = fft2(input_image_2_fft_abs_LogPolar_numpy)
    r0 = abs(input_image_1_fft_abs_LogPolar_numpy) * abs(input_image_2_fft_abs_LogPolar_numpy)
    phase_cross_correlation = abs(ifft2((input_image_1_fft_abs_LogPolar_numpy * input_image_2_fft_abs_LogPolar_numpy.conjugate()) / r0))
    # phase_cross_correlation[0, 0] = 0
    i0, i1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)
    angle = 180.0 * i0 / phase_cross_correlation.shape[0]
    scale = log_base ** i1

    if scale > 1.8:
        phase_cross_correlation = abs(
            ifft2((input_image_2_fft_abs_LogPolar_numpy * input_image_1_fft_abs_LogPolar_numpy.conjugate()) / r0))
        # phase_cross_correlation[0,0] = 0
        i0, i1 = numpy.unravel_index(numpy.argmax(phase_cross_correlation), phase_cross_correlation.shape)
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



from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
from scipy.fftpack import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation as skimage_phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

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
    #(*). This is EXTREMELY IMPORTANT! Without This It Doesn't Work!!!!!
    input_image_1 = difference_of_gaussians(input_image_1, 5, 20)  #TODO: the more you smear the better?
    input_image_2 = difference_of_gaussians(input_image_2, 5, 20)

    ### Window Images To Avoid Effects From Image Edges: ###
    #(*). Probably not as important!!!!
    input_image_1_windowed = input_image_1 * window('hann', input_image_1.shape)
    input_image_2_windowed = input_image_2 * window('hann', input_image_1.shape)

    ### Work With Shifted FFT Magnitudes: ###
    input_image_1_FFT_abs = np.abs(fftshift(fft2(input_image_1_windowed)))
    input_image_2_FFT_abs = np.abs(fftshift(fft2(input_image_2_windowed)))

    ### Create Log-Polar Transformed FFT Mag Images and Register: ###
    shape = input_image_1_FFT_abs.shape
    radius = shape[0] // 1  # only take lower frequencies
    input_image_1_FFT_abs_LogPolar = warp_polar(input_image_1_FFT_abs, radius=radius, output_shape=shape, scaling='log', order=0)
    input_image_2_FFT_abs_LogPolar = warp_polar(input_image_2_FFT_abs, radius=radius, output_shape=shape, scaling='log', order=0)

    # #TODO: delete
    # input_image_1_FFT_abs_torch = torch.Tensor(input_image_1_FFT_abs).unsqueeze(0).unsqueeze(0)
    # input_image_2_FFT_abs_torch = torch.Tensor(input_image_2_FFT_abs).unsqueeze(0).unsqueeze(0)
    # bla_1, log_base, (radius2, theta2), (x,y), flow_grid = logpolar_scipy_torch(input_image_1_FFT_abs_torch)
    # bla_2, log_base, (radius2, theta2), (x,y), flow_grid = logpolar_scipy_torch(input_image_2_FFT_abs_torch)

    ### TODO: this works!!!!!
    # input_image_1_FFT_abs_LogPolar = bla_1.cpu().numpy()[0,0]
    # input_image_2_FFT_abs_LogPolar = bla_2.cpu().numpy()[0,0]
    # imshow(input_image_1_FFT_abs_LogPolar); imshow_torch(bla_1)
    #TODO: perhapse i should also use on half the FFT and use fftshift as in the above example?!??!?!
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

# scipy_registration()
# similarity_numpy()
# similarity_numpy_2()
# logpolar_test()


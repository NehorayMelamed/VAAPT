"""
Sampling grids for 2D canonical coordinate systems.
Each coordinate system is defined by a pair of coordinates u, v.
Each grid function maps from a grid in (u, v) coordinates to a collection points in Cartesian coordinates.
"""

import numpy as np
import torch
import math
try:
    import scipy.ndimage.interpolation as ndii
except ImportError:
    import ndimage.interpolation as ndii

def torch_fft2(input_image):
    return torch.fft.fftn(input_image, dim=[-1,-2])
def torch_ifft2(input_image):
    return torch.fft.ifftn(input_image, dim=[-1,-2])

def logpolar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base."""
    # image = np.zeros((100,100))
    shape = image.shape
    center = shape[0] / 2, shape[1] / 2
    angles = shape[0]
    radii = shape[1]
    theta = np.empty((angles, radii), dtype='float64')
    theta.T[:] = np.linspace(0, np.pi, angles, endpoint=False) * -1.0
    # d = radii
    d = np.hypot(shape[0] - center[0], shape[1] - center[1])
    log_base = 10.0 ** (math.log10(d) / (radii))
    radius = np.empty_like(theta)
    radius[:] = (np.power(log_base, np.arange(radii, dtype='float64')) - 1.0)
    x = radius * np.sin(theta) + center[0]
    y = radius * np.cos(theta) + center[1]
    output = np.empty_like(x)
    ndii.map_coordinates(image, [x, y], output=output)
    return output, log_base, (radius, theta), (x,y)

def logpolar_numpy(image, angles=None, radii=None):
    return logpolar(image, angles=None, radii=None)

def logpolar_torch(input_image):
    """Return log-polar transformed image and log base."""
    ### Initial Parameters: ###
    # input_image = torch.zeros((1,1,100,100)).cuda()
    B,C,H,W = input_image.shape
    center = (H/2, W/2)
    number_of_angles = H
    number_of_radii = W
    radii_vec = np.arange(number_of_radii)

    ### Initialize Theta Meshgrid: ###
    theta = torch.zeros((number_of_angles, number_of_radii)).cuda()   #use .to(input_image.device) + turn into layer to avoid this instantiation
    theta.T[:] = torch.Tensor(np.linspace(0, np.pi, number_of_angles, endpoint=False) * -1.0).cuda()

    ### Initialize Radius Meshgrid: ###
    # d = radii
    d = np.sqrt((H-center[0])**2 + (W-center[1])**2)  #TODO: if i only want to take lower value <-> longer frequencies, i need to divide by something. some divide by 8
    log_base = 10.0 ** (math.log10(d) / (number_of_radii))
    radius = (torch.pow(log_base, torch.Tensor(radii_vec).cuda()) - 1.0)

    ### Get (X,Y) Meshgrid For Subsequent Interpolation From Cartesian To LogPolar Coordinates: ###
    x = radius * torch.sin(theta) + center[0]
    y = radius * torch.cos(theta) + center[1]
    # #TODO: below is temp, need to understand
    x = (x) / max(x.shape[1], 1)
    y = (y) / max(x.shape[0], 1)
    x = (x-0.5) * 2
    y = (y-0.5) * 2

    ### Interpolate image To LogPolar Coordinates: ###
    x = x.unsqueeze(0).unsqueeze(-1)
    y = y.unsqueeze(0).unsqueeze(-1)
    ### TODO: understand how to concat and what's going on here!
    # flow_grid = torch.cat((x,y), -1).cuda()
    flow_grid = torch.cat((y,x), -1).cuda()  #TODO: maybe there's a faster way then actually concating?
    output = torch.nn.functional.grid_sample(input_image.cuda(), flow_grid, mode='bilinear')
    return output, log_base, (radius, theta), (x,y), flow_grid

def logpolar_scipy_torch(input_image, radius_div_factor=1):
    """Return log-polar transformed image and log base."""
    ### Initial Parameters: ###
    B,C,H,W = input_image.shape
    center = (H/2, W/2)
    number_of_angles = H
    number_of_radii = W
    radii_vec = np.arange(number_of_radii)

    ### Initialize Theta Meshgrid: ###
    theta = torch.zeros((number_of_angles, number_of_radii)).cuda()
    # theta.T[:] = torch.Tensor(np.linspace(0, np.pi, number_of_angles, endpoint=False) * -1.0).cuda()
    theta.T[:] = torch.Tensor(np.linspace(0, 2*np.pi, number_of_angles, endpoint=False)).cuda()

    ### Initialize Radius Meshgrid: ###
    # d = radii
    d = np.sqrt((H-center[0])**2 + (W-center[1])**2)  #TODO: if i only want to take lower value <-> longer frequencies, i need to divide by something. some divide by 8
    log_base = 10.0 ** (math.log10(d) / (number_of_radii))
    # radius = (torch.pow(log_base, torch.Tensor(radii_vec).cuda()) - 1.0)
    radius = torch.exp(torch.Tensor(radii_vec).cuda()/number_of_radii*np.log(number_of_radii / radius_div_factor))

    # ### New Try: ###
    # radius_base = np.log(H) / radius_div_factor
    # radius = torch.exp(torch.Tensor(radii_vec).cuda() / H * radius_base)

    ### Get (X,Y) Meshgrid For Subsequent Interpolation From Cartesian To LogPolar Coordinates: ###
    x = radius * torch.sin(theta) + center[0]
    y = radius * torch.cos(theta) + center[1]
    # #TODO: below is temp, need to understand
    x = (x) / max(x.shape[1], 1)
    y = (y) / max(x.shape[0], 1)
    x = (x-0.5) * 2
    y = (y-0.5) * 2

    ### Interpolate image To LogPolar Coordinates: ###
    x = x.unsqueeze(0).unsqueeze(-1)
    y = y.unsqueeze(0).unsqueeze(-1)
    ### TODO: understand how to concat and what's going on here!
    # flow_grid = torch.cat((x,y), -1).cuda()
    flow_grid = torch.cat((y,x), -1).cuda()
    output = torch.nn.functional.grid_sample(input_image.cuda(), flow_grid, mode='nearest')
    return output, log_base, (radius, theta), (x,y), flow_grid

def highpass(shape):
    #TODO: make the highpass configurable in radius!!!!
    """Return highpass filter to be multiplied with fourier transform."""
    x = np.outer(
        np.cos(np.linspace(-math.pi / 2.0, math.pi / 2.0, shape[0])),
        np.cos(np.linspace(-math.pi / 2.0, math.pi / 2.0, shape[1])),
    )
    return (1.0 - x) * (2.0 - x)

def highpass_torch(shape):
    # TODO: make the highpass configurable in radius!!!!
    """Return highpass filter to be multiplied with fourier transform."""
    x = np.outer(
        np.cos(np.linspace(-math.pi / 2.0, math.pi / 2.0, shape[0])),
        np.cos(np.linspace(-math.pi / 2.0, math.pi / 2.0, shape[1])),
    )
    high_pass_fft_filter = torch.Tensor((1.0 - x) * (2.0 - x)).cuda()
    return high_pass_fft_filter


def identity_grid(output_size, ulim=(-1, 1), vlim=(-1, 1), out=None, device=None):
    """Cartesian coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = vs
    return torch.stack([xs, ys], 2, out=out)

def get_meshgrid_2D_torch(input_tensor, flag_type='default'):
    H,W = input_tensor.shape[-2:]
    if flag_type == 'default':
        x_vec = torch.arange(W).to(input_tensor.device)
        y_vec = torch.arange(H).to(input_tensor.device)
        Y, X = torch.meshgrid((x_vec, y_vec))
    elif flag_type == 'centered_pixel':
        x_vec = torch.arange(W).to(input_tensor.device)
        y_vec = torch.arange(H).to(input_tensor.device)
        Y, X = torch.meshgrid((x_vec, y_vec))
        X -= W//2
        Y -= H//2
    elif flag_type == 'centered':
        x_vec = torch.arange(W).to(input_tensor.device)
        y_vec = torch.arange(H).to(input_tensor.device)
        Y, X = torch.meshgrid((x_vec, y_vec))
        X -= W/2
        Y -= H/2
    elif flag_type == 'grid_sample':
        x_vec = torch.linspace(-1, 1, W, device=input_tensor.device)
        y_vec = torch.linspace(-1, 1, H, device=input_tensor.device)
        Y, X = torch.meshgrid([x_vec, y_vec])

    return X,Y

def polar_grid(output_size, ulim=(0, np.sqrt(2.)), vlim=(-np.pi, np.pi), out=None, device=None):
    """Polar coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us * torch.cos(vs)
    ys = us * torch.sin(vs)
    return torch.stack([xs, ys], 2, out=out)


def logpolar_grid(output_size, ulim=(None, np.log(2.) / 2.), vlim=(-np.pi, np.pi), out=None, device=None):
    """Log-polar coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), radial coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    rs = torch.exp(us)
    xs = rs * torch.cos(vs)
    ys = rs * torch.sin(vs)
    return torch.stack([xs, ys], 2, out=out)


def shearx_grid(output_size, ulim=(-1, 1), vlim=(-5, 5), out=None, device=None):
    """Horizontal shear coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian y-coordinate limits
        vlim: (float, float), x/y ratio limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    ys = us
    xs = us * vs
    return torch.stack([xs, ys], 2, out=out)


def sheary_grid(output_size, ulim=(-1, 1), vlim=(-5, 5), out=None, device=None):
    """Vertical shear coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), Cartesian x-coordinate limits
        vlim: (float, float), y/x ratio limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    xs = us
    ys = us * vs
    return torch.stack([xs, ys], 2, out=out)


def scalex_grid(output_size, ulim=(None, 0), vlim=(-1, 1), out=None, device=None):
    """Horizontal scale coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic x-coordinate limits
        vlim: (float, float), Cartesian y-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu / 2), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv // 2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    xs = torch.exp(us)
    ys = vs

    if nv % 2 == 0:
        xs = torch.cat([xs, -xs])
        ys = torch.cat([ys, ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0] - 1, 1), -xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0] - 1, 1), ys])
    return torch.stack([xs, ys], 2, out=out)


def scaley_grid(output_size, ulim=(None, 0), vlim=(-1, 1), out=None, device=None):
    """Vertical scale coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), logarithmic y-coordinate limits
        vlim: (float, float), Cartesian x-coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    if ulim[0] is None:
        ulim = (-np.log(nu / 2), ulim[1])
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv // 2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    ys = torch.exp(us)
    xs = vs

    if nv % 2 == 0:
        xs = torch.cat([xs, xs])
        ys = torch.cat([ys, -ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0] - 1, 1), xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0] - 1, 1), -ys])
    return torch.stack([xs, ys], 2, out=out)


def hyperbolic_grid(output_size, ulim=(-np.sqrt(0.5), np.sqrt(0.5)), vlim=(-np.log(6.), np.log(6.)), out=None,
                    device=None):
    """Hyperbolic coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), hyperbolic angular coordinate limits
        vlim: (float, float), hyperbolic log-radial coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv // 2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    rs = torch.exp(vs)
    xs = us * rs
    ys = us / rs

    if nv % 2 == 0:
        xs = torch.cat([xs, xs])
        ys = torch.cat([ys, -ys])
    else:
        xs = torch.cat([xs, xs.narrow(0, xs.shape[0] - 1, 1), xs])
        ys = torch.cat([ys, ys.narrow(0, ys.shape[0] - 1, 1), -ys])
    return torch.stack([xs, ys], 2, out=out)


def perspectivex_grid(output_size, ulim=(1, 8), vlim=(-0.99 * np.pi / 2, 0.99 * np.pi / 2), out=None, device=None):
    """Horizontal perspective coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), x^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv // 2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    xl = -1 / us.flip([1])
    xr = 1 / us
    yl = -xl * torch.tan(vs)
    yr = xr * torch.tan(vs)

    if nv % 2 == 0:
        xs = torch.cat([xl, xr])
        ys = torch.cat([yl, yr])
    else:
        xs = torch.cat([xl, xl.narrow(0, xl.shape[0] - 1, 1), xr])
        ys = torch.cat([yl, yl.narrow(0, yl.shape[0] - 1, 1), yr])
    return torch.stack([xs, ys], 2, out=out)


def perspectivey_grid(output_size, ulim=(1, 8), vlim=(-0.99 * np.pi / 2, 0.99 * np.pi / 2), out=None, device=None):
    """Vertical perspective coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), y^{-1} "radial" coordinate limits
        vlim: (float, float), angular coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv // 2, device=device)
    vs, us = torch.meshgrid([vrange, urange])

    yl = -1 / us.flip([1])
    yr = 1 / us
    xl = -yl * torch.tan(vs)
    xr = yr * torch.tan(vs)

    if nv % 2 == 0:
        xs = torch.cat([xl, xr])
        ys = torch.cat([yl, yr])
    else:
        xs = torch.cat([xl, xl.narrow(0, xl.shape[0] - 1, 1), xr])
        ys = torch.cat([yl, yl.narrow(0, yl.shape[0] - 1, 1), yr])
    return torch.stack([xs, ys], 2, out=out)


def spherical_grid(output_size, ulim=(-np.pi / 4, np.pi / 4), vlim=(-np.pi / 4, np.pi / 4), out=None, device=None):
    """Spherical coordinate system.

    Args:
        output_size: (int, int), number of sampled values for the v-coordinate and u-coordinate respectively
        ulim: (float, float), latitudinal coordinate limits
        vlim: (float, float), longitudinal coordinate limits
        out: torch.FloatTensor, output tensor
        device: string or torch.device, device for torch.tensor

    Returns:
        torch.FloatTensor, shape (output_size[0], output_size[1], 2), tensor where entry (i,j) gives the
        (x, y)-coordinates of the grid point.
    """
    nv, nu = output_size
    urange = torch.linspace(ulim[0], ulim[1], nu, device=device)
    vrange = torch.linspace(vlim[0], vlim[1], nv, device=device)
    vs, us = torch.meshgrid([vrange, urange])
    su, cu = torch.sin(us), torch.cos(us)
    sv, cv = torch.sin(vs), torch.cos(vs)
    xs = cu * sv / (np.sqrt(2.) - cu * cv)
    ys = su / (np.sqrt(2.) - cu * cv)
    return torch.stack([xs, ys], 2, out=out)




from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
from scipy.fftpack import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation as skimage_phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

def scipy_LogPolar_transform(input_image, radius_div_factor):
    current_shape = input_image.shape
    radius = current_shape[0] // radius_div_factor
    #(*). use skimage/scipy warp_polar transform
    #TODO: maybe use this warp_polar transform on a meshgrid (or something like it) to get the flow-field for pytorch grid_sample????
    output_image = warp_polar(input_image, radius=radius, output_shape=current_shape, scaling='log',
                                                order=0)  #order=0 <-> nearest neighbor? linear?
    return output_image

def torch_LogPolar_transform_ScipyFormat(input_image, radius_div_factor):
    #TODO: turn this into an object to avoid needlessly building the meshgrid etc'
    """Return log-polar transformed image and log base."""
    ### Initial Parameters: ###
    # input_image = torch.zeros((1,1,100,100)).cuda()
    B, C, H, W = input_image.shape
    center = (H / 2, W / 2)
    number_of_angles = H
    number_of_radii = W
    radii_vec = np.arange(number_of_radii)

    ### Initialize Theta Meshgrid: ###
    theta = torch.zeros((number_of_angles, number_of_radii)).cuda()
    # theta.T[:] = torch.Tensor(np.linspace(0, np.pi, number_of_angles, endpoint=False) * -1.0).cuda()
    theta.T[:] = torch.Tensor(np.linspace(0, 2 * np.pi, number_of_angles, endpoint=False)).cuda()

    ### Initialize Radius Meshgrid: ###
    # d = radii
    d = np.sqrt((H - center[0]) ** 2 + (W - center[1]) ** 2)
    log_base = 10.0 ** (math.log10(d) / (number_of_radii))
    # radius = (torch.pow(log_base, torch.Tensor(radii_vec).cuda()) - 1.0)
    radius = torch.exp(torch.Tensor(radii_vec).cuda() / number_of_radii * np.log(number_of_radii // radius_div_factor))

    ### Get (X,Y) Meshgrid For Subsequent Interpolation From Cartesian To LogPolar Coordinates: ###
    x = radius * torch.sin(theta) + center[0]
    y = radius * torch.cos(theta) + center[1]
    x = (x) / max(x.shape[1], 1)
    y = (y) / max(x.shape[0], 1)
    x = (x - 0.5) * 2
    y = (y - 0.5) * 2

    ### Interpolate image To LogPolar Coordinates: ###
    x = x.unsqueeze(0).unsqueeze(-1)
    y = y.unsqueeze(0).unsqueeze(-1)
    ### TODO: understand how to concat and what's going on here!
    # flow_grid = torch.cat((x,y), -1).cuda()
    flow_grid = torch.cat((y, x), -1).cuda()
    output = torch.nn.functional.grid_sample(input_image.cuda(), flow_grid, mode='nearest')
    return output, log_base, (radius, theta), (x, y), flow_grid
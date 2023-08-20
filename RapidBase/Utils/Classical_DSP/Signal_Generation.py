

import numpy as np
import torch
import torch.linalg

# from RapidBase.import_all import *


def square_wave_2D(H, W, X_pixels_per_cycle, Y_pixels_per_cycle):
    x = np.linspace(0, W, W)
    y = np.linspace(0, H, H)
    [X,Y] = np.meshgrid(x, y)
    Fx = 1/X_pixels_per_cycle
    Fy = 1/Y_pixels_per_cycle
    sine_2D = np.sin(2*np.pi*(Fx*X + Fy*Y))
    square_2D = (sine_2D > 0).astype(np.float)
    return square_2D

def sine_wave_2D(H, W, X_pixels_per_cycle, Y_pixels_per_cycle):
    x = np.linspace(0, W, W)
    y = np.linspace(0, H, H)
    [X,Y] = np.meshgrid(x, y)
    Fx = 1/X_pixels_per_cycle
    Fy = 1/Y_pixels_per_cycle
    sine_2D = np.sin(2*np.pi*(Fx*X + Fy*Y))
    return sine_2D

from RapidBase.Utils.Registration.Transforms_Grids import get_meshgrid_2D_torch
def generate_ellipse_2D_torch(X, Y, flag_full=True, center=(0,0), ellipse_axes=(1,1)):
    ### Temp: delete: ###
    input_tensor = torch.ones((1,1,512,512))
    X,Y = get_meshgrid_2D_torch(input_tensor)
    center = (200,120)
    ellipse_axes = (15,25)
    flag_full = False

    ### Create ellipse: ###
    x0,y0 = center
    a,b = ellipse_axes
    if flag_full:
        output_tensor = (((X-x0)/a)**2 + ((Y-y0)/b)**2 <= 1)
    else:
        tolerance = 20/min(a**2,b**2)  #the 20 is ed-hok to draw the entire ellipse without breaks
        output_tensor = (((X - x0) / a) ** 2 + ((Y - y0) / b) ** 2 <= 1 + tolerance).float() * (((X - x0) / a) ** 2 + ((Y - y0) / b) ** 2 >= 1 - tolerance).float()

    # imshow_torch(output_tensor)
    return output_tensor



def generate_gaussian_2D_torch(X, Y, center=(0,0), covariance_matrix=torch.eye(2)):
    ### Temp: delete: ###
    temp_tensor = torch.ones((1, 1, 512, 512))
    X, Y = get_meshgrid_2D_torch(temp_tensor)
    center = (200, 300)
    covariance_matrix = torch.zeros((2,2))
    covariance_matrix[0,0] = 50*3
    covariance_matrix[0,1] = 6*3
    covariance_matrix[1,0] = 6*3
    covariance_matrix[1,1] = 50*3

    ### Create gaussian: ###
    x0,y0 = center
    covariance_inverse = torch.inverse(covariance_matrix)
    a = covariance_inverse[0, 0]
    b = covariance_inverse[0, 1]
    c = covariance_inverse[1, 0]
    d = covariance_inverse[1, 1]
    # XY_pairs = torch.cat((torch.flatten(X).unsqueeze(-1), torch.flatten(Y).unsqueeze(-1)), -1).float().unsqueeze(1)
    # output_tensor = torch.exp(-0.5 * torch.bmm(torch.bmm(XY_pairs, covariance_inverse), XY_pairs.transpose(-1,-2)))
    input_tensor = 1 * torch.exp(-0.5*(a*(X-x0)**2 + b*(X-x0)*(Y-y0) + c*(X-x0)*(Y-y0) + d*(Y-y0)**2))
    input_tensor = (input_tensor>=0.5).float()
    # imshow_torch(input_tensor)

    # ### Use Ellipse Fit: ###
    # output_frame, ellipse_centroid, ellipse_axes_length = fit_ellipse_2D_using_moment_of_intertia_torch(input_tensor)
    # output_frame, ellipse_centroid, ellipse_axes_length = fit_ellipse_2D_using_scipy_regionprops_torch(input_tensor)

    return input_tensor


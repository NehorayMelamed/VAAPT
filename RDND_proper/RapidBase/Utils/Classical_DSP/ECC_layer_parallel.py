import cv2
import numpy as np
import scipy.ndimage
import torch.nn.functional

from RapidBase.import_all import *


def get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string, H, W):
    #     %J = get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string)
    # % This function computes the jacobian J of H_matrix transform with respect
    # % to parameters. In case of homography/euclidean transform, the jacobian depends on
    # % the parameter values, while in affine/translation case is totally invariant.
    # %
    # % Input variables:
    # % x_vec:           the x-coordinate values of the horizontal side of ROI (i.e. [xmin:xmax]),
    # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
    # % H_matrix:         the H_matrix transform (used only in homography and euclidean case),
    # % transform_string:    the type of adopted transform
    # % {'affine''homography','translation','euclidean'}
    # %
    # % Output:
    # % J:            The jacobian matrix J

    ### Get vec sizes: ###
    x_vec_length = len(x_vec)
    y_vec_length = len(y_vec)

    ### Initialize the jacobians: ###
    x_vec_unsqueezed = numpy_unsqueeze(x_vec, 0)
    y_vec_unsqueezed = numpy_unsqueeze(y_vec, -1)
    Jx = np.repeat(x_vec_unsqueezed, y_vec_length, 0)
    Jy = np.repeat(y_vec_unsqueezed, x_vec_length, -1)
    J0 = 0 * Jx  # could also use zeros_like
    J1 = J0 + 1  # could also use ones_like

    ### Flatten Arrays: ###
    # J0 = torch_flatten_image(J0).squeeze()
    # J1 = torch_flatten_image(J1).squeeze()
    # Jx = torch_flatten_image(Jx)
    # Jy = torch_flatten_image(Jy)

    if str.lower(transform_string) == 'homography':
        ### Concatenate the flattened jacobians: ###
        # TODO: aren't these all simply ones?!?! why?! we're just copying the H_matrix, which i can do without all of this
        xy = np.concatenate([np.transpose(numpy_flatten(Jx, True, 'F')),
                             np.transpose(numpy_flatten(Jy, True, 'F')),
                             np.ones((1, x_vec_length * y_vec_length))], 0)  # TODO: before axis was -1

        ### 3x3 matrix transformation: ###
        A = H_matrix
        A[2, 2] = 1

        ### new coordinates after H_matrix: ###
        xy_prime = np.matmul(A, xy)  # matrix multiplication

        ### division due to homogeneous coordinates: ###
        xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
        xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]
        den = np.transpose(xy_prime[2, :])  # TODO: understand if this is needed
        den = np.reshape(den, (H, W), order='F')
        Jx = Jx / den  # element-wise
        Jy = Jy / den  # element-wise

        ### warped jacobian(???): ###
        Jxx_prime = Jx
        Jxx_prime = Jxx_prime * np.reshape(xy_prime[0, :], (H, W), order='F')  # element-wise
        Jyx_prime = Jy
        Jyx_prime = Jyx_prime * np.reshape(xy_prime[0, :], (H, W), order='F')

        Jxy_prime = Jx
        Jxy_prime = Jxy_prime * np.reshape(xy_prime[1, :], (H, W), order='F')  # element-wise
        Jyy_prime = Jy
        Jyy_prime = Jyy_prime * np.reshape(xy_prime[1, :], (H, W), order='F')

        ### Get final jacobian of the H_matrix with respect to the different parameters: ###
        J_up = np.concatenate([Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0], -1)
        J_down = np.concatenate([J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)

    elif str.lower(transform_string) == 'affine':
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        J_up = np.concatenate([Jx, J0, Jy, J0, J1, J0], -1)
        J_down = np.concatenate([J0, Jx, J0, Jy, J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)

    elif str.lower(transform_string) == 'translation':
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        J_up = np.concatenate([J1, J0], -1)
        J_down = np.concatenate([J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)

    elif str.lower(transform_string) == 'euclidean':
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        mycos = H_matrix[1, 1]
        mysin = H_matrix[2, 1]

        Jx_prime = -mysin * Jx - mycos * Jy
        Jy_prime = mycos * Jx - mysin * Jy

        J_up = np.concatenate([Jx_prime, J1, J0], -1)
        J_down = np.concatenate([Jy_prime, J0, J1], -1)
        J = np.concatenate([J_up, J_down], 0)

    return J


def get_jacobian_for_warp_transform_torch(x_vec, y_vec, H_matrix, transform_string, H, W):
    #     %J = get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string)
    # % This function computes the jacobian J of H_matrix transform with respect
    # % to parameters. In case of homography/euclidean transform, the jacobian depends on
    # % the parameter values, while in affine/translation case is totally invariant.
    # %
    # % Input variables:
    # % x_vec:           the x-coordinate values of the horizontal side of ROI (i.e. [xmin:xmax]),
    # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
    # % H_matrix:         the H_matrix transform (used only in homography and euclidean case),
    # % transform_string:    the type of adopted transform
    # % {'affine''homography','translation','euclidean'}
    # %
    # % Output:
    # % J:            The jacobian matrix J

    ### Get vec sizes: ###
    x_vec_length = len(x_vec)
    y_vec_length = len(y_vec)

    ### Initialize the jacobians: ###
    # TODO: i can absolutely calculate this in advance!!!!
    x_vec_unsqueezed = x_vec.unsqueeze(0)
    y_vec_unsqueezed = y_vec.unsqueeze(-1)
    Jx = torch.repeat_interleave(x_vec_unsqueezed, y_vec_length, 0)
    Jy = torch.repeat_interleave(y_vec_unsqueezed, x_vec_length, 1)
    # Jx = np.repeat(x_vec_unsqueezed, y_vec_length, 0)
    # Jy = np.repeat(y_vec_unsqueezed, x_vec_length, -1)
    J0 = 0 * Jx  # could also use zeros_like
    J1 = J0 + 1  # could also use ones_like

    ### Flatten Arrays: ###
    # J0 = torch_flatten_image(J0).squeeze()
    # J1 = torch_flatten_image(J1).squeeze()
    # Jx = torch_flatten_image(Jx)
    # Jy = torch_flatten_image(Jy)

    if str.lower(transform_string) == 'homography':
        ### Concatenate the flattened jacobians: ###
        # TODO: aren't these all simply ones?!?! why?! we're just copying the H_matrix, which i can do without all of this
        # TODO: i can simply calculate this in advanced!
        # TODO: understand if i can do something else instead of flattening and unflattening!!!
        xy = torch.cat([torch.transpose(torch_flatten_image(Jx, True, 'F'), -1, -2),
                        torch.transpose(torch_flatten_image(Jy, True, 'F'), -1, -2),
                        torch.ones((1, x_vec_length * y_vec_length)).to(H_matrix.device)], 0)  # TODO: before axis was -1

        ### 3x3 matrix transformation: ###
        A = H_matrix
        A[2, 2] = 1

        ### new coordinates after H_matrix: ###
        xy_prime = torch.matmul(A, xy)  # matrix multiplication

        ### division due to homogeneous coordinates: ###
        xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
        xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]
        den = torch.transpose(xy_prime[2:3, :], -1, -2)  # TODO: understand if this is needed
        den = torch_reshape_flattened_image(den, (H, W), order='F')
        Jx = Jx / den  # element-wise
        Jy = Jy / den  # element-wise

        ### warped jacobian(???): ###
        Jxx_prime = Jx
        Jxx_prime = Jxx_prime * torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')  # element-wise
        Jyx_prime = Jy
        Jyx_prime = Jyx_prime * torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')

        Jxy_prime = Jx
        Jxy_prime = Jxy_prime * torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')  # element-wise
        Jyy_prime = Jy
        Jyy_prime = Jyy_prime * torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')

        ### Get final jacobian of the H_matrix with respect to the different parameters: ###
        J_up = torch.cat([Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0], -1)
        J_down = torch.cat([J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1], -1)
        J = torch.cat([J_up, J_down], 0)

    elif str.lower(transform_string) == 'affine':
        # TODO: can be calculated in advance!!!
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        J_up = torch.cat([Jx, J0, Jy, J0, J1, J0], -1)
        J_down = torch.cat([J0, Jx, J0, Jy, J0, J1], -1)
        J = torch.cat([J_up, J_down], 0)

    elif str.lower(transform_string) == 'translation':
        # TODO: can be calculated in advance!
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        J_up = torch.cat([J1, J0], -1)
        J_down = torch.cat([J0, J1], -1)
        J = torch.cat([J_up, J_down], 0)

    elif str.lower(transform_string) == 'euclidean':
        Jx = Jx.squeeze()
        Jy = Jy.squeeze()
        mycos = H_matrix[1, 1]
        mysin = H_matrix[2, 1]

        Jx_prime = -mysin * Jx - mycos * Jy
        Jy_prime = mycos * Jx - mysin * Jy

        J_up = torch.cat([Jx_prime, J1, J0], -1)
        J_down = torch.cat([Jy_prime, J0, J1], -1)
        J = torch.cat([J_up, J_down], 0)

    return J


def spatial_interpolation_numpy(input_image, H_matrix, interpolation_method, transform_string, x_vec, y_vec, H, W):
    # %OUT = spatial_interpolation_numpy(IN, H_matrix, STR, transform_string, x_vec, y_vec)
    # % This function implements the 2D spatial interpolation of image IN
    # %(inverse warping). The coordinates defined by x_vec,y_vec are projected through
    # % H_matrix thus resulting in new subpixel coordinates. The intensity values in
    # % new pixel coordinates are computed via bilinear interpolation
    # % of image IN. For other valid interpolation methods look at the help
    # % of Matlab function INTERP2.
    # %
    # % Input variables:
    # % IN:           the input image which must be warped,
    # % H_matrix:         the H_matrix transform,
    # % STR:          the string corresponds to interpolation method: 'linear',
    # %               'cubic' etc (for details look at the help file of
    # %               Matlab function INTERP2),
    # % transform_string:    the type of adopted transform: {'translation','euclidean','affine','homography'}
    # % x_vec:           the x-coordinate values of horizontal side of ROI (i.e. [xmin:xmax]),
    # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
    # %
    # % Output:
    # % OUT:          The warped (interpolated) image

    ### Correct H_matrix If Needed: ###
    if transform_string == 'affine' or transform_string == 'euclidean':
        if H_matrix.shape[0] == 2:
            H_matrix = np.concatenate([H_matrix, np.zeros(1, 3)], 0)
    if transform_string == 'translation':
        H_matrix = np.concatenate([np.eye(2), H_matrix], -1)
        H_matrix = np.concatenate([H_matrix, np.zeros((1, 3))], 0)

    # ###############################################
    # ### TODO: temp, delete: ####
    # ### HOW TO USE DIFFERENT INTERPOLATION FUNCTIONS: ###
    # #(1). scipy.ndimage.map_coordinates:
    # a = np.arange(12.).reshape((4, 3))
    # coordinates =  [[0.5, 2], [0.5, 1]]  #a list of lists or a list of tuples where to predict
    # result = scipy.ndimage.map_coordinates(a, coordinates, order=1)
    # #(2). scipy.interpolate.griddata:
    # def func(x, y):
    #     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
    # grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    # rng = np.random.default_rng()
    # coordinates_given = rng.random((1000, 2))  #[N,2] array
    # values_given = func(coordinates_given[:, 0], coordinates_given[:, 1])  #[N] array
    # points_to_predict = (grid_x, grid_y)  #a tuple of (X,Y) meshgrids!!!
    # grid_z2 = scipy.interpolate.griddata(coordinates_given, values_given, points_to_predict, method='cubic')
    # #(3). scipy.interpolate.interp2d:  seems to accept only x_vec's and meshgrids
    # x_vec = np.arange(-5.01, 5.01, 0.25)
    # y_vec = np.arange(-5.01, 5.01, 0.25)
    # X, Y = np.meshgrid(x_vec, y_vec)
    # Z_meshgrid = np.sin(X ** 2 + Y ** 2)
    # f_interpolation_function = scipy.interpolate.interp2d(x_vec, y_vec, Z_meshgrid, kind='cubic')
    # x_vec_new = np.arange(-5.01, 5.01, 1e-2)
    # y_vec_new = np.arange(-5.01, 5.01, 1e-2)
    # znew = f_interpolation_function(x_vec_new, y_vec_new)
    # #(4). scipy.interpolate.interpn:
    # def value_func_3d(x, y, z):
    #     return 2 * x + 3 * y - z
    # x_vec = np.linspace(0, 4, 5)
    # y_vec = np.linspace(0, 5, 6)
    # z_vec = np.linspace(0, 6, 7)
    # coordinates_tuple_given = (x_vec, y_vec, z_vec)
    # values_given = value_func_3d(*np.meshgrid(*coordinates_tuple_given, indexing='ij'))
    # new_point = np.array([2.21, 3.12, 1.15])
    # final_results = scipy.interpolate.interpn(coordinates_tuple_given, values_given, new_point)
    # #(5). cv2.remap:
    #
    # ###############################################

    ### create meshgrid and flattened coordinates array ([x,y,1] basis): ###
    [xx, yy] = np.meshgrid(x_vec, y_vec)
    xy = np.concatenate([np.transpose(numpy_flatten(xx, True, 'F')), np.transpose(numpy_flatten(yy, True, 'F')), np.ones((1, len(numpy_flatten(yy, True, 'F'))))], 0)

    ### 3X3 matrix transformation: ###
    A = H_matrix
    A[-1, -1] = 1

    ### new coordinates: ###
    xy_prime = np.matmul(A, xy)

    ### division due to homogenous coordinates: ###
    # TODO: if we do not use homography THERE'S NO NEED TO CALCULATE xy_prime[2,:] and so the above is also not relevant and it simply becomes a matter of affine warp
    if transform_string == 'homography':
        xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
        xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]

    ### Ignore third row: ###
    xy_prime = xy_prime[0:2, :]

    ### Turn to float32 instead of float64: ###
    xy_prime = xy_prime.astype(float32)

    ### Subpixel interpolation: ###
    # out = cv2.remap(input_image, np.reshape(xy_prime[0,:]+1, (H,W)), np.reshape(xy_prime[1,:]+1, (H,W)), cv2.INTER_CUBIC)
    final_X_grid = np.reshape(xy_prime[0, :], (H, W), order='F')
    final_Y_grid = np.reshape(xy_prime[1, :], (H, W), order='F')
    if interpolation_method == 'linear':
        out = cv2.remap(input_image, final_X_grid, final_Y_grid, cv2.INTER_LINEAR)
    elif interpolation_method == 'nearest':
        out = cv2.remap(input_image, final_X_grid, final_Y_grid, cv2.INTER_NEAREST)

    ### Make sure to output the same number of dimensions as input: ###
    if len(out.shape) == 2 and len(input_image.shape) == 3:
        out = numpy_unsqueeze(out, -1)

    return out


def spatial_interpolation_torch(input_image, H_matrix, interpolation_method, transform_string, x_vec, y_vec, H, W):
    # %OUT = spatial_interpolation_numpy(IN, H_matrix, STR, transform_string, x_vec, y_vec)
    # % This function implements the 2D spatial interpolation of image IN
    # %(inverse warping). The coordinates defined by x_vec,y_vec are projected through
    # % H_matrix thus resulting in new subpixel coordinates. The intensity values in
    # % new pixel coordinates are computed via bilinear interpolation
    # % of image IN. For other valid interpolation methods look at the help
    # % of Matlab function INTERP2.
    # %
    # % Input variables:
    # % IN:           the input image which must be warped,
    # % H_matrix:         the H_matrix transform,
    # % STR:          the string corresponds to interpolation method: 'linear',
    # %               'cubic' etc (for details look at the help file of
    # %               Matlab function INTERP2),
    # % transform_string:    the type of adopted transform: {'translation','euclidean','affine','homography'}
    # % x_vec:           the x-coordinate values of horizontal side of ROI (i.e. [xmin:xmax]),
    # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
    # %
    # % Output:
    # % OUT:          The warped (interpolated) image

    ### Correct H_matrix If Needed: ###
    if transform_string == 'affine' or transform_string == 'euclidean':
        if H_matrix.shape[0] == 2:
            H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)
    if transform_string == 'translation':
        H_matrix = torch.cat([torch.eye(2), H_matrix], -1)
        H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)

    ### create meshgrid and flattened coordinates array ([x,y,1] basis): ###
    [yy, xx] = torch.meshgrid(y_vec, x_vec)
    xy = torch.cat([torch.transpose(torch_flatten_image(xx, True, 'F'), -1, -2),
                    torch.transpose(torch_flatten_image(yy, True, 'F'), -1, -2),
                    torch.ones((1, len(torch_flatten_image(yy, True, 'F')))).to(input_image.device)], 0).to(input_image.device)

    # ### Try Simple Transformation: ###
    # xx_new = copy.deepcopy(xx)
    # yy_new = copy.deepcopy(yy)
    # xx_new = (H_matrix[0,0] * xx_new + H_matrix[0,1] * yy_new + H_matrix[0,2]) / (H_matrix[-1,0] * xx_new + H_matrix[-1,1] * yy_new + H_matrix[-1,2])
    # yy_new = (H_matrix[1,0] * xx_new + H_matrix[1,1] * yy_new + H_matrix[1,2]) / (H_matrix[-1,0] * xx_new + H_matrix[-1,1] * yy_new + H_matrix[-1,2])

    ### 3X3 matrix transformation: ###
    A = H_matrix
    A[-1, -1] = 1

    ### new coordinates: ###
    xy_prime = torch.matmul(A, xy)

    ### division due to homogenous coordinates: ###
    # TODO: if we do not use homography THERE'S NO NEED TO CALCULATE xy_prime[2,:] and so the above is also not relevant and it simply becomes a matter of affine warp
    if transform_string == 'homography':
        xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
        xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]

    ### Ignore third row: ###
    xy_prime = xy_prime[0:2, :]

    ### Turn to float32 instead of float64: ###
    xy_prime = xy_prime.type(torch.float32)

    ### Subpixel interpolation: ###
    # out = cv2.remap(input_image, np.reshape(xy_prime[0,:]+1, (H,W)), np.reshape(xy_prime[1,:]+1, (H,W)), cv2.INTER_CUBIC)
    final_X_grid = torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')
    final_Y_grid = torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')
    new_X = 2 * final_X_grid / max(W - 1, 1) - 1
    new_Y = 2 * final_Y_grid / max(H - 1, 1) - 1
    bilinear_grid = torch.cat([torch_get_4D(new_X, 'CH'), torch_get_4D(new_Y, 'CH')], dim=3)
    out = torch.nn.functional.grid_sample(input_image.unsqueeze(0).unsqueeze(0), bilinear_grid, mode='bicubic').squeeze(0).squeeze(0)

    # ### Subpixel Interpolation 2: ###
    # new_X = 2 * xx_new / max(W - 1, 1) - 1
    # new_Y = 2 * yy_new / max(H - 1, 1) - 1
    # bilinear_grid = torch.cat([torch_get_4D(new_X, 'CH'), torch_get_4D(new_Y, 'CH')], dim=3)
    # out = torch.nn.functional.grid_sample(input_image.unsqueeze(0).unsqueeze(0), bilinear_grid, mode='bicubic').squeeze(0).squeeze(0)

    # if interpolation_method == 'linear':
    #     out = cv2.remap(input_image, final_X_grid, final_Y_grid, cv2.INTER_LINEAR)
    # elif interpolation_method == 'nearest':
    #     out = cv2.remap(input_image, final_X_grid, final_Y_grid, cv2.INTER_NEAREST)

    # ### Make sure to output the same number of dimensions as input: ###
    # if len(out.shape) == 2 and len(input_image.shape) == 3:
    #     out = out.unsqueeze(0)

    return out


def image_jacobian_numpy(gx, gy, jac, number_of_parameters):
    # %G = image_jacobian_numpy(GX, GY, JAC, number_of_parameters)
    # % This function computes the jacobian G of the warped image wrt parameters.
    # % This matrix depends on the gradient of the warped image, as
    # % well as on the jacobian JAC of the warp transform wrt parameters.
    # % For a detailed definition of matrix G, see the paper text.
    # %
    # % Input variables:
    # % GX:           the warped image gradient in x (horizontal) direction,
    # % GY:           the warped image gradient in y (vertical) direction,
    # % JAC:            the jacobian matrix J of the warp transform wrt parameters,
    # % number_of_parameters:          the number of parameters.
    # %
    # % Output:
    # % G:            The jacobian matrix G.
    #

    ### Get image shape: ###
    if len(gx.shape) == 2:
        h, w = gx.shape
    elif len(gx.shape) == 3:
        h, w, c = gx.shape

    ### Repeat image gradients by the number of parameters: ###
    gx = np.concatenate([gx] * number_of_parameters, -1)
    gy = np.concatenate([gy] * number_of_parameters, -1)
    # gx = np.repeat(gx, number_of_parameters, -1)
    # gy = np.repeat(gy, number_of_parameters, -1)

    ### Get Jacobian of warped image with respect to parameters (chain rule i think): ###
    G = gx * jac[0:h, :] + gy * jac[h:, :]
    G = np.reshape(G, (h * w, number_of_parameters), order='F')
    return G


def image_jacobian_torch(gx, gy, jac, number_of_parameters):
    # %G = image_jacobian_numpy(GX, GY, JAC, number_of_parameters)
    # % This function computes the jacobian G of the warped image wrt parameters.
    # % This matrix depends on the gradient of the warped image, as
    # % well as on the jacobian JAC of the warp transform wrt parameters.
    # % For a detailed definition of matrix G, see the paper text.
    # %
    # % Input variables:
    # % GX:           the warped image gradient in x (horizontal) direction,
    # % GY:           the warped image gradient in y (vertical) direction,
    # % JAC:            the jacobian matrix J of the warp transform wrt parameters,
    # % number_of_parameters:          the number of parameters.
    # %
    # % Output:
    # % G:            The jacobian matrix G.
    #

    ### Get image shape: ###
    if len(gx.shape) == 2:
        h, w = gx.shape
    elif len(gx.shape) == 3:
        c, h, w = gx.shape

    ### Repeat image gradients by the number of parameters: ###
    gx = torch.cat([gx] * number_of_parameters, -1)
    gy = torch.cat([gy] * number_of_parameters, -1)
    # gx = np.repeat(gx, number_of_parameters, -1)
    # gy = np.repeat(gy, number_of_parameters, -1)

    ### Get Jacobian of warped image with respect to parameters (chain rule i think): ###
    G = gx * jac[0:h, :] + gy * jac[h:, :]  # TODO: understand if there's a better way then concatenating multiple times and then multiplying and then reshaping!!!
    G = torch_reshape_image(G, (h * w, number_of_parameters), order='F').contiguous()  # TODO: understand what this outputs and maybe we can avoid the torch_reshape_image

    return G


def correct_H_matrix_for_coming_level_numpy(H_matrix_in, transform_string, high_flag):
    # %H_matrix=correct_H_matrix_for_coming_level_numpy(H_matrix_in, transform_string, HIGH_FLAG)
    # % This function modifies appropriately the WARP values in order to apply
    # % the warp in the next level. If HIGH_FLAG is equal to 1, the function
    # % makes the warp appropriate for the next level of higher resolution.
    # If HIGH_FLAG is equal to 0, the function makes the warp appropriate for the previous level of lower resolution.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               'tranlation','affine' and 'homography'.
    # % HIGH_FLAG:    The flag which defines the 'next' level. 1 means that the
    # %               the next level is a higher resolution level,
    # %               while 0 means that it is a lower resolution level.
    # % Output:
    # % H_matrix:         the next-level warp transform

    H_matrix = H_matrix_in
    if high_flag == 'higher_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] / 2

        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

        if transform_string == 'translation':
            H_matrix = H_matrix * 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

    elif high_flag == 'lower_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] * 2

        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2

        if transform_string == 'translation':
            H_matrix = H_matrix / 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2

    return H_matrix


def correct_H_matrix_for_coming_level_torch(H_matrix_in, transform_string, high_flag):
    # %H_matrix=correct_H_matrix_for_coming_level_numpy(H_matrix_in, transform_string, HIGH_FLAG)
    # % This function modifies appropriately the WARP values in order to apply
    # % the warp in the next level. If HIGH_FLAG is equal to 1, the function
    # % makes the warp appropriate for the next level of higher resolution.
    # If HIGH_FLAG is equal to 0, the function makes the warp appropriate for the previous level of lower resolution.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               'tranlation','affine' and 'homography'.
    # % HIGH_FLAG:    The flag which defines the 'next' level. 1 means that the
    # %               the next level is a higher resolution level,
    # %               while 0 means that it is a lower resolution level.
    # % Output:
    # % H_matrix:         the next-level warp transform

    H_matrix = H_matrix_in
    if high_flag == 'higher_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] / 2

        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

        if transform_string == 'translation':
            H_matrix = H_matrix * 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] * 2

    elif high_flag == 'lower_resolution':
        if transform_string == 'homography':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2
            H_matrix[-1, 0:2] = H_matrix[-1, 0:2] * 2

        if transform_string == 'affine':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2

        if transform_string == 'translation':
            H_matrix = H_matrix / 2

        if transform_string == 'euclidean':
            H_matrix[0:2, -1] = H_matrix[0:2, -1] / 2

    return H_matrix


def update_transform_params_numpy(H_matrix_in, delta_p, transform_string):
    # % H_matrix_out=update_transform_params_numpy(H_matrix_in,DELTA_P,transform_string)
    # % This function updates the parameter values by adding the correction values
    # % of DELTA_P to the current warp values in H_matrix_in.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % DELTA_P:      the current correction parameter vector,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               {'translation','euclidean','affine','homography'}.
    # % Output:
    # % H_matrix:         the new (updated) warp transform

    if transform_string == 'homography':
        delta_p = np.concatenate([delta_p, np.zeros((1, 1))], 0)  # TODO: understand what's this
        H_matrix_out = H_matrix_in + np.reshape(delta_p, (3, 3), order='F')
        H_matrix_out[2, 2] = 1

    if transform_string == 'affine':
        H_matrix_out = np.zeros((2, 3))
        H_matrix_out[0:2, :] = H_matrix_in[0:2, :] + np.reshape(delta_p, (2, 3))
        H_matrix_out = np.concatenate([H_matrix_out, np.zeros((1, 3))], 0)
        H_matrix_out[2, 2] = 1

    if transform_string == 'translation':
        H_matrix_out = H_matrix_in + delta_p

    if transform_string == 'euclidean':
        theta = sign(H_matrix_in[1, 0]) * np.arccos(H_matrix_in[0, 0]) + delta_p[0]
        tx = H_matrix_in[0, 2] + delta_p[1]
        ty = H_matrix_in[1, 2] + delta_p[2]
        H_matrix_out = np.eye(3)
        H_matrix_out[0, :] = np.array([np.cos(theta), -sin(theta), tx])
        H_matrix_out[1, :] = np.array([np.sin(theta), cos(theta), ty])

    return H_matrix_out


def update_transform_params_torch(H_matrix_in, delta_p, transform_string):
    # % H_matrix_out=update_transform_params_numpy(H_matrix_in,DELTA_P,transform_string)
    # % This function updates the parameter values by adding the correction values
    # % of DELTA_P to the current warp values in H_matrix_in.
    # %
    # % Input variables:
    # % H_matrix_in:      the current warp transform,
    # % DELTA_P:      the current correction parameter vector,
    # % transform_string:    the type of adopted transform, accepted strings:
    # %               {'translation','euclidean','affine','homography'}.
    # % Output:
    # % H_matrix:         the new (updated) warp transform

    if transform_string == 'homography':
        # T = H_matrix_in.shape[0]
        # delta_p = torch.cat([delta_p, torch.zeros((T, 1, 1)).to(H_matrix_in.device)], 1)  # TODO: understand what's this
        # H_matrix_out = H_matrix_in + torch_reshape_image(delta_p, (3, 3), order='F')
        # H_matrix_out[2, 2] = 1

        H_matrix_out = H_matrix_in
        H_matrix_out[:, 0, 0] += delta_p[:, 0, 0]
        H_matrix_out[:, 1, 0] += delta_p[:, 1, 0]
        H_matrix_out[:, 2, 0] += delta_p[:, 2, 0]
        H_matrix_out[:, 0, 1] += delta_p[:, 3, 0]
        H_matrix_out[:, 1, 1] += delta_p[:, 4, 0]
        H_matrix_out[:, 2, 1] += delta_p[:, 5, 0]
        H_matrix_out[:, 0, 2] += delta_p[:, 6, 0]
        H_matrix_out[:, 1, 2] += delta_p[:, 7, 0]
        H_matrix_out[:, 2, 2] = 1

    if transform_string == 'affine':
        H_matrix_out = torch.zeros((2, 3)).to(H_matrix_in.device)
        H_matrix_out[0:2, :] = H_matrix_in[0:2, :] + torch_reshape_image(delta_p, (2, 3), order='F')
        H_matrix_out = torch.cat([H_matrix_out, torch.zeros((1, 3))], 0)
        H_matrix_out[2, 2] = 1

    if transform_string == 'translation':
        H_matrix_out = H_matrix_in + delta_p

    if transform_string == 'euclidean':
        theta = sign(H_matrix_in[1, 0]) * torch.arccos(H_matrix_in[0, 0]) + delta_p[0]
        tx = H_matrix_in[0, 2] + delta_p[1]
        ty = H_matrix_in[1, 2] + delta_p[2]
        H_matrix_out = torch.eye(3).to(H_matrix_in.device)
        H_matrix_out[0, :] = torch.tensor([torch.cos(theta), -torch.sin(theta), tx])
        H_matrix_out[1, :] = torch.tensor([torch.sin(theta), torch.cos(theta), ty])

    return H_matrix_out


def ECC_numpy(input_tensor, reference_tensor, number_of_levels, number_of_iterations_per_level, transform_string, delta_p_init=None):
    # %ECC image alignment algorithm
    # %[RESULTS, H_matrix, WARPEDiMAGE] = ECC(input_tensor, reference_tensor, number_of_levels, number_of_iterations_per_level, transform_string, DELTA_P_INIT)
    # %
    # % This m-file implements the ECC image alignment algorithm as it is
    # % presented in the paper "G.D.Evangelidis, E.Z.Psarakis, Parametric Image Alignment
    # % using Enhanced Correlation Coefficient.IEEE Trans. on PAMI, vol.30, no.10, 2008"
    # %
    # % ------------------
    # % Input variables:
    # % input_tensor:        the profile needs to be warped in order to be similar to reference_tensor,
    # % reference_tensor:     the profile needs to be reached,
    # % number_of_iterations_per_level:          the number of iterations per level; the algorithm is executed
    # %               (number_of_iterations_per_level-1) times
    # % number_of_levels:       the number of number_of_levels in pyramid scheme (set number_of_levels=1 for a
    # %               non pyramid implementation), the level-index 1
    # %               corresponds to the highest (original) image resolution
    # % transform_string:    the image transformation {'translation', 'euclidean', 'affine', 'homography'}
    # % DELTA_P_INIT: the initial transformation matrix for original images (optional); The identity
    # %               transformation is the default value (see 'transform initialization'
    # %               subroutine in the code). In case of affine or euclidean transform,
    # %               DELTA_P_INIT must be a 2x3 matrix, in homography case it must be a 3x3 matrix,
    # %               while with translation transform it must be a 2x1 vector.
    # %
    # % For example, to initialize the warp with a rotation by x radians, DELTA_P_INIT must
    # % be [cos(x) sin(x) 0 ; -sin(x) cos(x) 0] for affinity
    # % [cos(x) sin(x) 0 ; -sin(x) cos(x) 0 ; 0 0 1] for homography.
    # %
    # %
    # % Output:
    # %
    # % RESULTS:   A struct of size number_of_levels x number_of_iterations_per_level with the following fields:
    # %
    # % RESULTS(m,n).H_matrix:     the warp needs to be applied to IMAGE at n-th iteration of m-th level,
    # % RESULTS(m,n).rho:      the enhanced correlation coefficient value at n-th iteration of m-th level,
    # % H_matrix :              the final estimated transformation [usually also stored in RESULTS(1,number_of_iterations_per_level).H_matrix ].
    # % WARPEDiMAGE:        the final warped image (it should be similar to reference_tensor).
    # %
    # % The first stored .H_matrix and .rho values are due to the initialization. In
    # % case of poor final alignment results check the warp initialization
    # % and/or the overlap of the images.

    # %% transform initialization
    # % In case of translation transform the initialiation matrix is of size 2x1:
    # %  delta_p_init = [p1;
    # %                  p2]
    # % In case of affine transform the initialiation matrix is of size 2x3:
    # %
    # %  delta_p_init = [p1, p3, p5;
    # %                  p2, p4, p6]
    # %
    # % In case of euclidean transform the initialiation matrix is of size 2x3:
    # %
    # %  delta_p_init = [p1, p3, p5;
    # %                  p2, p4, p6]
    # %
    # % where p1=cos(theta), p2 = sin(theta), p3 = -p2, p4 =p1
    # %
    # % In case of homography transform the initialiation matrix is of size 3x3:
    # %  delta_p_init = [p1, p4, p7;
    # %                 p2, p5, p8;
    # %                 p3, p6,  1]

    ### Initialize Parameters: ###
    break_flag = 0
    transform_string = str.lower(transform_string)
    H_input, W_input, C_input = input_tensor.shape
    H_reference, W_reference, C_reference = reference_tensor.shape

    ### Initialize New Images For Algorithm To Change: ###
    initImage = input_tensor
    initTemplate = reference_tensor
    input_tensor = RGB2BW(input_tensor).astype(float).squeeze()
    reference_tensor = RGB2BW(reference_tensor).astype(float).squeeze()
    reference_tensor_output_list = [0] * number_of_levels
    input_tensor_output_list = [0] * number_of_levels

    # pyramid images
    # The following for-loop creates pyramid images in cells current_level_input_tensor and reference_tensor_output_list with varying names
    # The variables input_tensor_output_list1} and reference_tensor_output_list{1} are the images with the highest resolution

    ### Smoothing of original images: ###
    # TODO: in the matlab version they overwrite the gaussian blur, so they're not actually blurring anything
    # reference_tensor_output_list[0] = cv2.GaussianBlur(reference_tensor, [7,7], 0.5)
    # input_tensor_output_list[0] = cv2.GaussianBlur(input_tensor, [7,7], 0.5)
    reference_tensor_output_list[0] = reference_tensor
    input_tensor_output_list[0] = input_tensor
    for level_index in np.arange(1, number_of_levels):
        H, W = input_tensor_output_list[level_index - 1].shape
        input_tensor_output_list[level_index] = cv2.resize(input_tensor_output_list[level_index - 1], dsize=(W // 2, H // 2))
        reference_tensor_output_list[level_index] = cv2.resize(reference_tensor_output_list[level_index - 1], dsize=(W // 2, H // 2))

    ### Initialize H_matrix matrix: ###
    # (1). Translation:
    if transform_string == 'translation':
        number_of_parameters = 2  # number of parameters
        if delta_p_init is None:
            H_matrix = np.zeros(2, 1)
        else:
            H_matrix = delta_p_init
    # (2). Euclidean:
    elif transform_string == 'euclidean':
        number_of_parameters = 3  # number of parameters
        if delta_p_init is None:
            H_matrix = np.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = np.concatenate([delta_p_init, np.zeros(1, 3)], 0)
    # (3). Affine:
    elif transform_string == 'affine':
        number_of_parameters = 6  # number of parameters
        if delta_p_init is None:
            H_matrix = np.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = np.concatenate([delta_p_init, np.zeros((1, 3))], 0)
    # (4). Homography:
    elif transform_string == 'homography':
        number_of_parameters = 8  # number of parameters
        if delta_p_init is None:
            H_matrix = np.eye(3)
        else:
            H_matrix = delta_p_init

    ### in case of pyramid implementation, the initial transformation must be appropriately modified: ###
    for level_index in np.arange(0, number_of_levels - 1):
        H_matrix = correct_H_matrix_for_coming_level_numpy(H_matrix, transform_string, 'lower_resolution')

    ### Run ECC algorithm for each level of the pyramid: ###
    for level_index in np.arange(number_of_levels, 0, -1):  # start with lowest resolution (highest level of the pyramid)
        ### Get Current Level input_tensor and reference_tensor: ###
        current_level_input_tensor = input_tensor_output_list[level_index - 1]
        current_level_reference_tensor = reference_tensor_output_list[level_index - 1]
        if len(current_level_reference_tensor.shape) == 3:
            A, B, C = current_level_reference_tensor.shape
            H, W, C = current_level_reference_tensor.shape
        elif len(current_level_reference_tensor.shape) == 2:
            A, B = current_level_reference_tensor.shape
            H, W = current_level_reference_tensor.shape

        ### Get input_tensor gradients: ###
        [vy, vx] = np.gradient(current_level_input_tensor, axis=[0, 1])
        # imshow(vx)

        ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
        # Here we just ignore some image margins.
        # Margin is equal to 5 percent of the mean of [height,width].
        m0 = mean([A, B])
        # margin = floor(m0 * .05 / (2 ** (level_index - 1)))
        margin = 0  # no - margin - modify these two lines if you want to exclude a margin
        x_vec = np.arange(margin, B - margin)
        y_vec = np.arange(margin, A - margin)
        current_level_reference_tensor = current_level_reference_tensor[margin:A - margin, margin:B - margin].astype(float)

        ### ECC, Forward Additive Algorithm: ###
        for iteration_index in np.arange(number_of_iterations_per_level):
            print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))
            wim = spatial_interpolation_numpy(current_level_input_tensor, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)  # inverse(backward) warping

            ### define a mask to deal with warping outside the image borders: ###
            # (they may have negative values due to the subtraction of the mean value)
            # TODO: there must be an easier way to do this!!! no way i need all these calculations
            ones_map = spatial_interpolation_numpy(np.ones_like(current_level_input_tensor), H_matrix, 'nearest', transform_string, x_vec, y_vec, H, W)  # inverse(backward) warping
            numOfElem = (ones_map != 0).sum()

            meanOfWim = (wim * (ones_map != 0)).sum() / numOfElem
            meanOfTemp = (current_level_reference_tensor * (ones_map != 0)).sum() / numOfElem

            wim = wim - meanOfWim  # zero - mean image; is useful for brightness change compensation, otherwise you can comment this line
            tempzm = current_level_reference_tensor - meanOfTemp  # zero - mean reference_tensor

            wim[ones_map == 0] = 0  # for pixels outside the overlapping area
            tempzm[ones_map == 0] = 0

            # ### Save current transform: ###
            # # TODO: find an appropriate data structure / object for this
            # if transform_string == 'affine' or transform_string == 'euclidean':
            #     results[level_index, iteration_index].H_matrix = H_matrix[0:2, :]
            # else:
            #     results[level_index, iteration_index].H_matrix = H_matrix
            # results[level_index, iteration_index].rho = dot(current_level_reference_tensor[:], wim[:]) / norm(tempzm[:]) / norm(wim[:])

            ### Break the loop if reached max number of iterations per level: ###
            if iteration_index == number_of_iterations_per_level:  # the algorithm is executed (number_of_iterations_per_level-1) times
                break

            ### Gradient Image interpolation (warped gradients): ###
            vx_warped = spatial_interpolation_numpy(vx, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)
            vy_warped = spatial_interpolation_numpy(vy, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)

            ### Compute the jacobian of warp transform_string: ###
            J = get_jacobian_for_warp_transform_numpy(x_vec + 1, y_vec + 1, H_matrix, transform_string, H, W)

            ### Compute the jacobian of warped image wrt parameters (matrix G in the paper): ###
            G = image_jacobian_numpy(vx_warped, vy_warped, J, number_of_parameters)

            ### Coompute Hessian and its inverse: ###
            C = np.matmul(np.transpose(G), G)  # matrix multiplication, C = Hessian matrix
            cond = np.linalg.cond(C)
            i_C = np.linalg.inv(C)

            ### Compute projections of images into G: ###
            Gt = np.transpose(G) @ numpy_flatten(tempzm, True, 'F')
            Gw = np.transpose(G) @ numpy_flatten(wim, True, 'F')

            ### ECC Closed Form Solution: ###
            # (1). compute lambda parameter:
            num = (np.linalg.norm(numpy_flatten(wim, True, 'F'))) ** 2 - np.transpose(Gw) @ i_C @ Gw
            den = (np.dot(numpy_flatten(tempzm, True, 'F').squeeze(), numpy_flatten(wim, True, 'F').squeeze())) - np.transpose(Gt) @ i_C @ Gw
            lambda_correction = num / den
            # (2). compute error vector:
            imerror = lambda_correction * tempzm - wim
            # (3). compute the projection of error vector into Jacobian G:
            Ge = np.transpose(G) @ numpy_flatten(imerror, True, 'F')
            # (4). compute the optimum parameter correction vector:
            delta_p = np.matmul(i_C, Ge)

            ### Update Parameters: ###
            H_matrix = update_transform_params_numpy(H_matrix, delta_p, transform_string)

            print(H_matrix)
            # print(delta_p)

        ### END OF INTERNAL LOOP

        ### break loop of reached errors: ###
        if break_flag == 1:
            break

        ### modify the parameters appropriately for next pyramid level: ###
        if level_index > 1 and break_flag == 0:
            H_matrix = correct_H_matrix_for_coming_level_numpy(H_matrix, transform_string, 'higher_resolution')

    ### END OF PYRAMID number_of_levels LOOP:

    # ### this conditional part is only executed when algorithm stops due to Hessian singularity: ###
    # if break_flag == 1:
    #     for jj in np.arange(level_index-1):
    #         H_matrix = correct_H_matrix_for_coming_level_numpy(H_matrix, transform_string, 'higher_resolution')

    ### Get final H_matrix: ###
    final_warp = H_matrix

    ### return the final warped image using the whole support area (including margins): ###
    nx2 = np.arange(0, B)
    ny2 = np.arange(0, A)
    warpedImage = np.zeros_like(initImage)
    for ii in np.arange(C_input):
        warpedImage[:, :, ii] = spatial_interpolation_numpy(initImage[:, :, ii], final_warp, 'linear', transform_string, nx2, ny2, H, W)
    H_matrix = final_warp

    return H_matrix, warpedImage


def ECC_torch(input_tensor, reference_tensor, number_of_levels, number_of_iterations_per_level, transform_string, delta_p_init=None):
    ### Initialize Parameters: ###
    break_flag = 0
    transform_string = str.lower(transform_string)
    C_input, H_input, W_input = input_tensor.shape
    C_reference, H_reference, W_reference = reference_tensor.shape

    ### Initialize New Images For Algorithm To Change: ###
    initImage = input_tensor
    initTemplate = reference_tensor
    input_tensor = RGB2BW(input_tensor).type(torch.float32).squeeze()
    reference_tensor = RGB2BW(reference_tensor).type(torch.float32).squeeze()
    reference_tensor_output_list = [0] * number_of_levels
    input_tensor_output_list = [0] * number_of_levels

    ### Smoothing of original images: ###
    reference_tensor_output_list[0] = reference_tensor
    input_tensor_output_list[0] = input_tensor
    for level_index in np.arange(1, number_of_levels):
        ### Shape: ###
        H, W = input_tensor_output_list[level_index - 1].shape

        # ### Gaussian Blur: ###
        # gaussian_blur_layer = Gaussian_Blur_Layer(1, kernel_size=7, sigma=0.5)
        # input_tensor_output_list[level_index] = gaussian_blur_layer.forward(input_tensor_output_list[level_index])
        # reference_tensor_output_list[level_index] = gaussian_blur_layer.forward(reference_tensor_output_list[level_index])

        ### Interpolate: ###
        input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)
        reference_tensor_output_list[level_index] = torch.nn.functional.interpolate(reference_tensor_output_list[level_index - 1], scale_factor=0.5)

    ### Initialize H_matrix matrix: ###
    # (1). Translation:
    if transform_string == 'translation':
        number_of_parameters = 2  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.zeros((2, 1))
        else:
            H_matrix = delta_p_init
    # (2). Euclidean:
    elif transform_string == 'euclidean':
        number_of_parameters = 3  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
    # (3). Affine:
    elif transform_string == 'affine':
        number_of_parameters = 6  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.eye(3)
            H_matrix[-1, -1] = 0
        else:
            H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
    # (4). Homography:
    elif transform_string == 'homography':
        number_of_parameters = 8  # number of parameters
        if delta_p_init is None:
            H_matrix = torch.eye(3)
        else:
            H_matrix = delta_p_init

    ### Send To Device: ###
    H_matrix = H_matrix.to(input_tensor.device).type(torch.float32)

    ### in case of pyramid implementation, the initial transformation must be appropriately modified: ###
    for level_index in np.arange(0, number_of_levels - 1):
        H_matrix = correct_H_matrix_for_coming_level_torch(H_matrix, transform_string, 'lower_resolution')

    ### Run ECC algorithm for each level of the pyramid: ###
    for level_index in np.arange(number_of_levels, 0, -1):  # start with lowest resolution (highest level of the pyramid)
        ### Get Current Level input_tensor and reference_tensor: ###
        current_level_input_tensor = input_tensor_output_list[level_index - 1]
        current_level_reference_tensor = reference_tensor_output_list[level_index - 1]
        if len(current_level_reference_tensor.shape) == 3:
            C, A, B = current_level_reference_tensor.shape
            C, H, W = current_level_reference_tensor.shape
        elif len(current_level_reference_tensor.shape) == 2:
            A, B = current_level_reference_tensor.shape
            H, W = current_level_reference_tensor.shape

        ### Get input_tensor gradients: ###
        [vy, vx] = torch.gradient(current_level_input_tensor, dim=[0, 1])

        ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
        # Here we just ignore some image margins.
        # Margin is equal to 5 percent of the mean of [height,width].
        m0 = mean([A, B])
        # margin = floor(m0 * .05 / (2 ** (level_index - 1)))
        margin = 0  # no - margin - modify these two lines if you want to exclude a margin
        x_vec = torch.arange(margin, B - margin).to(input_tensor.device)
        y_vec = torch.arange(margin, A - margin).to(input_tensor.device)
        current_level_reference_tensor = current_level_reference_tensor[margin:A - margin, margin:B - margin].type(torch.float32)

        ### ECC, Forward Additive Algorithm: ###
        for iteration_index in np.arange(number_of_iterations_per_level):
            print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))
            wim = spatial_interpolation_torch(current_level_input_tensor, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)  # inverse(backward) warping

            ### define a mask to deal with warping outside the image borders: ###
            # (they may have negative values due to the subtraction of the mean value)
            # TODO: there must be an easier way to do this!!! no way i need all these calculations
            # TODO: i can probably simply calculate the center crop size and use that right? i should probably be able to roughly calculate the center crop as a function of H_matrix
            ones_map = spatial_interpolation_torch(torch.ones_like(current_level_input_tensor), H_matrix, 'nearest', transform_string, x_vec, y_vec, H, W)  # inverse(backward) warping
            numOfElem = (ones_map != 0).sum()

            meanOfWim = (wim * (ones_map != 0)).sum() / numOfElem
            meanOfTemp = (current_level_reference_tensor * (ones_map != 0)).sum() / numOfElem

            wim = wim - meanOfWim  # zero - mean image; is useful for brightness change compensation, otherwise you can comment this line
            tempzm = current_level_reference_tensor - meanOfTemp  # zero - mean reference_tensor

            wim[ones_map == 0] = 0  # for pixels outside the overlapping area
            tempzm[ones_map == 0] = 0

            # ### Save current transform: ###
            # # TODO: find an appropriate data structure / object for this
            # if transform_string == 'affine' or transform_string == 'euclidean':
            #     results[level_index, iteration_index].H_matrix = H_matrix[0:2, :]
            # else:
            #     results[level_index, iteration_index].H_matrix = H_matrix
            # results[level_index, iteration_index].rho = dot(current_level_reference_tensor[:], wim[:]) / norm(tempzm[:]) / norm(wim[:])

            ### Break the loop if reached max number of iterations per level: ###
            if iteration_index == number_of_iterations_per_level:  # the algorithm is executed (number_of_iterations_per_level-1) times
                break

            ### Gradient Image interpolation (warped gradients): ###
            vx_warped = spatial_interpolation_torch(vx, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)
            vy_warped = spatial_interpolation_torch(vy, H_matrix, 'linear', transform_string, x_vec, y_vec, H, W)

            ### Compute the jacobian of warp transform_string: ###
            J = get_jacobian_for_warp_transform_torch(x_vec + 1, y_vec + 1, H_matrix, transform_string, H, W)

            ### Compute the jacobian of warped image wrt parameters (matrix G in the paper): ###
            G = image_jacobian_torch(vx_warped, vy_warped, J, number_of_parameters)

            ### Coompute Hessian and its inverse: ###
            C = torch.matmul(torch.transpose(G, -1, -2), G)  # matrix multiplication, C = Hessian matrix
            cond = torch.linalg.cond(C)
            i_C = torch.linalg.inv(C)

            ### Compute projections of images into G: ###
            Gt = torch.transpose(G, -1, -2) @ torch_flatten_image(tempzm, True, 'F')
            Gw = torch.transpose(G, -1, -2) @ torch_flatten_image(wim, True, 'F')

            ### ECC Closed Form Solution: ###
            # (1). compute lambda parameter:
            num = (torch.linalg.norm(torch_flatten_image(wim, True, 'F'))) ** 2 - torch.transpose(Gw, -1, -2) @ i_C @ Gw
            den = (torch.dot(torch_flatten_image(tempzm, True, 'F').squeeze(), torch_flatten_image(wim, True, 'F').squeeze())) - torch.transpose(Gt, -1, -2) @ i_C @ Gw
            lambda_correction = num / den
            # (2). compute error vector:
            imerror = lambda_correction * tempzm - wim
            # (3). compute the projection of error vector into Jacobian G:
            Ge = torch.transpose(G, -1, -2) @ torch_flatten_image(imerror, True, 'F')
            # (4). compute the optimum parameter correction vector:
            delta_p = torch.matmul(i_C, Ge)

            ### Update Parameters: ###
            H_matrix = update_transform_params_torch(H_matrix, delta_p, transform_string)

            print(H_matrix)
            # print(delta_p)

        ### END OF INTERNAL LOOP

        ### break loop of reached errors: ###
        if break_flag == 1:
            break

        ### modify the parameters appropriately for next pyramid level: ###
        if level_index > 1 and break_flag == 0:
            H_matrix = correct_H_matrix_for_coming_level_torch(H_matrix, transform_string, 'higher_resolution')

    ### END OF PYRAMID number_of_levels LOOP:

    # ### this conditional part is only executed when algorithm stops due to Hessian singularity: ###
    # if break_flag == 1:
    #     for jj in np.arange(level_index-1):
    #         H_matrix = correct_H_matrix_for_coming_level_numpy(H_matrix, transform_string, 'higher_resolution')

    ### Get final H_matrix: ###
    final_warp = H_matrix

    ### return the final warped image using the whole support area (including margins): ###
    nx2 = torch.arange(0, B).to(input_tensor.device)
    ny2 = torch.arange(0, A).to(input_tensor.device)
    warpedImage = torch.zeros_like(initImage)
    for ii in torch.arange(C_input):
        warpedImage[ii, :, :] = spatial_interpolation_torch(initImage[ii, :, :], final_warp, 'linear', transform_string, nx2, ny2, H, W)
    H_matrix = final_warp

    return H_matrix, warpedImage


class ECC_Layer_Torch(nn.Module):
    # Initialize this with a module
    def __init__(self, reference_tensor, number_of_iterations_per_level, number_of_levels=1, transform_string='homography', delta_p_init=None):
        super(ECC_Layer_Torch, self).__init__()
        self.X = None
        self.Y = None
        self.device = reference_tensor.device

        ### Initialize Parameters: ###
        transform_string = str.lower(transform_string)
        # C_reference, H_reference, W_reference = reference_tensor.shape

        ### Initialize New Images For Algorithm To Change: ###
        initTemplate = reference_tensor
        reference_tensor = RGB2BW(reference_tensor).type(torch.float32).squeeze()

        ### Initialize H_matrix matrix: ###
        H_matrix, number_of_parameters = self.initialize_H_matrix(delta_p_init=None, transform_string=transform_string, device=self.device)

        ### in case of pyramid implementation, the initial transformation must be appropriately modified: ###
        for level_index in np.arange(0, number_of_levels - 1):
            H_matrix = correct_H_matrix_for_coming_level_torch(H_matrix, transform_string, 'lower_resolution')

        ### Assign To Internal Attributes: ###
        self.number_of_levels = number_of_levels
        self.number_of_iterations_per_level = number_of_iterations_per_level
        self.transform_string = transform_string
        self.number_of_parameters = number_of_parameters
        self.H_matrix = H_matrix
        self.reference_tensor_output_list = None

    def initialize_H_matrix(self, delta_p_init=None, transform_string='homography', device='cpu'):
        # (1). Translation:
        if transform_string == 'translation':
            number_of_parameters = 2  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.zeros((2, 1))
            else:
                H_matrix = delta_p_init
        # (2). Euclidean:
        elif transform_string == 'euclidean':
            number_of_parameters = 3  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
                H_matrix[-1, -1] = 0
            else:
                H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
        # (3). Affine:
        elif transform_string == 'affine':
            number_of_parameters = 6  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
                H_matrix[-1, -1] = 0
            else:
                H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
        # (4). Homography:
        elif transform_string == 'homography':
            number_of_parameters = 8  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
            else:
                H_matrix = delta_p_init

        ### Send To Device: ###
        H_matrix = H_matrix.to(self.device).type(torch.float32)

        return H_matrix, number_of_parameters

    def spatial_interpolation_torch(self, input_image, H_matrix, interpolation_method, transform_string, x_vec, y_vec, X_mat, Y_mat, H, W):
        # %OUT = spatial_interpolation_numpy(IN, H_matrix, STR, transform_string, x_vec, y_vec)
        # % This function implements the 2D spatial interpolation of image IN
        # %(inverse warping). The coordinates defined by x_vec,y_vec are projected through
        # % H_matrix thus resulting in new subpixel coordinates. The intensity values in
        # % new pixel coordinates are computed via bilinear interpolation
        # % of image IN. For other valid interpolation methods look at the help
        # % of Matlab function INTERP2.
        # %
        # % Input variables:
        # % IN:           the input image which must be warped,
        # % H_matrix:         the H_matrix transform,
        # % STR:          the string corresponds to interpolation method: 'linear',
        # %               'cubic' etc (for details look at the help file of
        # %               Matlab function INTERP2),
        # % transform_string:    the type of adopted transform: {'translation','euclidean','affine','homography'}
        # % x_vec:           the x-coordinate values of horizontal side of ROI (i.e. [xmin:xmax]),
        # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
        # %
        # % Output:
        # % OUT:          The warped (interpolated) image

        #################################################################################################################################
        # (1). Older, Slower Method:
        # ### Correct H_matrix If Needed: ###
        # if transform_string == 'affine' or transform_string == 'euclidean':
        #     if H_matrix.shape[0] == 2:
        #         H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)
        # if transform_string == 'translation':
        #     H_matrix = torch.cat([torch.eye(2), H_matrix], -1)
        #     H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)
        #
        # ### create meshgrid and flattened coordinates array ([x,y,1] basis): ###
        # xy = torch.cat([torch.transpose(torch_flatten_image(X_mat, True, 'F'), -1, -2),
        #                 torch.transpose(torch_flatten_image(Y_mat, True, 'F'), -1, -2),
        #                 torch.ones((1, len(torch_flatten_image(Y_mat, True, 'F')))).to(input_image.device)], 0).to(input_image.device)
        #
        # ### 3X3 matrix transformation: ###
        # A = H_matrix
        # A[-1, -1] = 1
        #
        # ### new coordinates: ###
        # xy_prime = torch.matmul(A, xy)
        #
        # ### division due to homogenous coordinates: ###
        # if transform_string == 'homography':
        #     xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
        #     xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]
        #
        # ### Ignore third row: ###
        # xy_prime = xy_prime[0:2, :]
        #
        # ### Turn to float32 instead of float64: ###
        # xy_prime = xy_prime.type(torch.float32)
        #
        # ### Subpixel interpolation: ###
        # # out = cv2.remap(input_image, np.reshape(xy_prime[0,:]+1, (H,W)), np.reshape(xy_prime[1,:]+1, (H,W)), cv2.INTER_CUBIC)
        # final_X_grid = torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')
        # final_Y_grid = torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')
        # new_X = 2 * final_X_grid / max(W - 1, 1) - 1
        # new_Y = 2 * final_Y_grid / max(H - 1, 1) - 1
        # bilinear_grid = torch.cat([torch_get_4D(new_X, 'CH'), torch_get_4D(new_Y, 'CH')], dim=3)
        # out = torch.nn.functional.grid_sample(input_image.unsqueeze(0).unsqueeze(0), bilinear_grid, mode='bicubic').squeeze(0).squeeze(0)
        #################################################################################################################################

        #################################################################################################################################
        # (2). New Faster Method (currently assumes homography transform):
        ### Try Simple Transformation: ###
        xx_new = 2 * (H_matrix[0, 0] * X_mat + H_matrix[0, 1] * Y_mat + H_matrix[0, 2]) / (H_matrix[-1, 0] * X_mat + H_matrix[-1, 1] * Y_mat + H_matrix[-1, 2]) / max(W - 1, 1) - 1
        yy_new = 2 * (H_matrix[1, 0] * X_mat + H_matrix[1, 1] * Y_mat + H_matrix[1, 2]) / (H_matrix[-1, 0] * X_mat + H_matrix[-1, 1] * Y_mat + H_matrix[-1, 2]) / max(H - 1, 1) - 1

        ### Subpixel Interpolation 2: ###
        # xx_new = 2 * xx_new / max(W - 1, 1) - 1
        # yy_new = 2 * yy_new / max(H - 1, 1) - 1
        bilinear_grid = torch.cat([torch_get_4D(xx_new, 'CH'), torch_get_4D(yy_new, 'CH')], dim=3)  # TODO: get rid of the torch_get_4D and maybe find a faster way then torch.cat()
        out = torch.nn.functional.grid_sample(input_image.unsqueeze(0).unsqueeze(0), bilinear_grid, mode='bilinear').squeeze(0).squeeze(0)  # TODO: get rid of the unsqueeze by being consistent with the format
        #################################################################################################################################

        return out

    def get_jacobian_for_warp_transform_torch(self, x_vec, y_vec, H_matrix, Jx, Jy, J0, J1, transform_string, H, W):
        #     %J = get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string)
        # % This function computes the jacobian J of H_matrix transform with respect
        # % to parameters. In case of homography/euclidean transform, the jacobian depends on
        # % the parameter values, while in affine/translation case is totally invariant.
        # %
        # % Input variables:
        # % x_vec:           the x-coordinate values of the horizontal side of ROI (i.e. [xmin:xmax]),
        # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
        # % H_matrix:         the H_matrix transform (used only in homography and euclidean case),
        # % transform_string:    the type of adopted transform
        # % {'affine''homography','translation','euclidean'}
        # %
        # % Output:
        # % J:            The jacobian matrix J

        ### Get vec sizes: ###
        x_vec_length = len(x_vec)
        y_vec_length = len(y_vec)

        ### Flatten Arrays: ###
        # J0 = torch_flatten_image(J0).squeeze()
        # J1 = torch_flatten_image(J1).squeeze()
        # Jx = torch_flatten_image(Jx)
        # Jy = torch_flatten_image(Jy)

        if str.lower(transform_string) == 'homography':
            ### New, Better Way: ###
            xy_prime_reshaped_X = (H_matrix[0, 0] * Jx + H_matrix[0, 1] * Jy + H_matrix[0, 2]) / (H_matrix[-1, 0] * Jx + H_matrix[-1, 1] * Jy + H_matrix[-1, 2])
            xy_prime_reshaped_Y = (H_matrix[1, 0] * Jx + H_matrix[1, 1] * Jy + H_matrix[1, 2]) / (H_matrix[-1, 0] * Jx + H_matrix[-1, 1] * Jy + H_matrix[-1, 2])
            den = (H_matrix[-1, 0] * Jx + H_matrix[-1, 1] * Jy + H_matrix[-1, 2])
            Jx = Jx / den  # element-wise
            Jy = Jy / den  # element-wise

            # ############################################################################################
            # ### Concatenate the flattened jacobians: ###
            # # TODO: aren't these all simply ones?!?! why?! we're just copying the H_matrix, which i can do without all of this
            # # TODO: i can simply calculate this in advanced!
            # # TODO: understand if i can do something else instead of flattening and unflattening!!!
            # xy = torch.cat([torch.transpose(torch_flatten_image(Jx, True, 'F'), -1, -2),  #TODO: all of this can be calculated in advance!!!, but maybe simply take a different approach
            #                 torch.transpose(torch_flatten_image(Jy, True, 'F'), -1, -2), #TODO: plus, Jx and Jy are known in advance and have very simple structure
            #                 torch.ones((1, x_vec_length * y_vec_length)).to(H_matrix.device)], 0)  # TODO: before axis was -1
            #
            # ### 3x3 matrix transformation: ###
            # A = H_matrix
            # A[2, 2] = 1
            #
            # ### new coordinates after H_matrix: ###
            # xy_prime = torch.matmul(A, xy)  # matrix multiplication
            #
            # ### division due to homogeneous coordinates: ###
            # xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
            # xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]
            # den = torch.transpose(xy_prime[2:3, :], -1, -2)  # TODO: understand if this is needed
            # den = torch_reshape_flattened_image(den, (H, W), order='F')
            # Jx = Jx / den  # element-wise
            # Jy = Jy / den  # element-wise
            #
            # ### warped jacobian(???): ###
            # xy_prime_reshaped_X = torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')    #TODO: understand what's faster, calculating and storing and accessing in advance or calculating on the spot
            # xy_prime_reshaped_Y = torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')
            # ##################################################################################################################

            ### Assign Jacobian Elements: ###
            Jxx_prime = Jx
            Jxx_prime = Jxx_prime * xy_prime_reshaped_X  # element-wise.
            Jyx_prime = Jy
            Jyx_prime = Jyx_prime * xy_prime_reshaped_X

            Jxy_prime = Jx
            Jxy_prime = Jxy_prime * xy_prime_reshaped_Y  # element-wise
            Jyy_prime = Jy
            Jyy_prime = Jyy_prime * xy_prime_reshaped_Y

            ### Get final jacobian of the H_matrix with respect to the different parameters: ###
            # #TODO: maybe there's a better way then concatenating huge amounts of memory like this?...maybe i can simply understand where this goes and calculate what's needed instead
            # J_up = torch.cat([Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0], -1)
            # J_down = torch.cat([J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1], -1)
            # J = torch.cat([J_up, J_down], 0)
            J_list = [Jx, Jy, J0, J1, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime]

        elif str.lower(transform_string) == 'affine':
            # TODO: can be calculated in advance!!!
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            J_up = torch.cat([Jx, J0, Jy, J0, J1, J0], -1)
            J_down = torch.cat([J0, Jx, J0, Jy, J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        elif str.lower(transform_string) == 'translation':
            # TODO: can be calculated in advance!
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            J_up = torch.cat([J1, J0], -1)
            J_down = torch.cat([J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        elif str.lower(transform_string) == 'euclidean':
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            mycos = H_matrix[1, 1]
            mysin = H_matrix[2, 1]

            Jx_prime = -mysin * Jx - mycos * Jy
            Jy_prime = mycos * Jx - mysin * Jy

            J_up = torch.cat([Jx_prime, J1, J0], -1)
            J_down = torch.cat([Jy_prime, J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        return J_list

    def image_jacobian_torch(self, gx, gy, J_list, number_of_parameters):
        # %G = image_jacobian_numpy(GX, GY, JAC, number_of_parameters)
        # % This function computes the jacobian G of the warped image wrt parameters.
        # % This matrix depends on the gradient of the warped image, as
        # % well as on the jacobian JAC of the warp transform wrt parameters.
        # % For a detailed definition of matrix G, see the paper text.
        # %
        # % Input variables:
        # % GX:           the warped image gradient in x (horizontal) direction,
        # % GY:           the warped image gradient in y (vertical) direction,
        # % JAC:            the jacobian matrix J of the warp transform wrt parameters,
        # % number_of_parameters:          the number of parameters.
        # %
        # % Output:
        # % G:            The jacobian matrix G.
        #

        ### Get image shape: ###
        if len(gx.shape) == 2:
            h, w = gx.shape
        elif len(gx.shape) == 3:
            c, h, w = gx.shape

        # ### Unroll All Variables V1: ###
        # #TODO: if i remember correctly J0,J1 are simply zeros and ones. i can skip multiplications!!!!
        # [Jx, Jy, J0, J1, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime] = J_list
        # G0 = gx * Jx + gy * J0
        # G1 = gx * J0 + gy * Jx
        # G2 = -gx * Jxx_prime - gy * Jxy_prime
        # G3 = gx * Jy + gy * J0
        # G4 = gx * J0 + gy * Jy
        # G5 = -gx * Jyx_prime - gy * Jyy_prime
        # G6 = gx * J1 + gy * J0
        # G7 = gx * J0 + gy * J1
        ### Unroll All Variables V2 (using the fact the J0,J1 are simply zeros and ones and disregarding them): ###
        [Jx, Jy, J0, J1, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime] = J_list
        G0 = gx * Jx
        G1 = gy * Jx
        G2 = -gx * Jxx_prime - gy * Jxy_prime
        G3 = gx * Jy
        G4 = gy * Jy
        G5 = -gx * Jyx_prime - gy * Jyy_prime
        G6 = gx
        G7 = gy

        # TODO: understand if making a list here takes time?!??
        G_list = [G0,
                  G1,
                  G2,
                  G3,
                  G4,
                  G5,
                  G6,
                  G7]

        ### PreCalculate C=(Gt*G): ###
        # TODO: would be smart to combine everything here together in the same moemory run
        C = torch.zeros((8, 8)).to(gx.device)
        C[0, 0] = (G0 * G0).sum()
        C[0, 1] = (G0 * G1).sum()
        C[0, 2] = (G0 * G2).sum()
        C[0, 3] = (G0 * G3).sum()
        C[0, 4] = (G0 * G4).sum()
        C[0, 5] = (G0 * G5).sum()
        C[0, 6] = (G0 * G6).sum()
        C[0, 7] = (G0 * G7).sum()
        #
        C[1, 0] = (G1 * G0).sum()
        C[1, 1] = (G1 * G1).sum()
        C[1, 2] = (G1 * G2).sum()
        C[1, 3] = (G1 * G3).sum()
        C[1, 4] = (G1 * G4).sum()
        C[1, 5] = (G1 * G5).sum()
        C[1, 6] = (G1 * G6).sum()
        C[1, 7] = (G1 * G7).sum()
        #
        C[2, 0] = (G2 * G0).sum()
        C[2, 1] = (G2 * G1).sum()
        C[2, 2] = (G2 * G2).sum()
        C[2, 3] = (G2 * G3).sum()
        C[2, 4] = (G2 * G4).sum()
        C[2, 5] = (G2 * G5).sum()
        C[2, 6] = (G2 * G6).sum()
        C[2, 7] = (G2 * G7).sum()
        #
        C[3, 0] = (G3 * G0).sum()
        C[3, 1] = (G3 * G1).sum()
        C[3, 2] = (G3 * G2).sum()
        C[3, 3] = (G3 * G3).sum()
        C[3, 4] = (G3 * G4).sum()
        C[3, 5] = (G3 * G5).sum()
        C[3, 6] = (G3 * G6).sum()
        C[3, 7] = (G3 * G7).sum()
        #
        C[4, 0] = (G4 * G0).sum()
        C[4, 1] = (G4 * G1).sum()
        C[4, 2] = (G4 * G2).sum()
        C[4, 3] = (G4 * G3).sum()
        C[4, 4] = (G4 * G4).sum()
        C[4, 5] = (G4 * G5).sum()
        C[4, 6] = (G4 * G6).sum()
        C[4, 7] = (G4 * G7).sum()
        #
        C[5, 0] = (G5 * G0).sum()
        C[5, 1] = (G5 * G1).sum()
        C[5, 2] = (G5 * G2).sum()
        C[5, 3] = (G5 * G3).sum()
        C[5, 4] = (G5 * G4).sum()
        C[5, 5] = (G5 * G5).sum()
        C[5, 6] = (G5 * G6).sum()
        C[5, 7] = (G5 * G7).sum()
        #
        C[6, 0] = (G6 * G0).sum()
        C[6, 1] = (G6 * G1).sum()
        C[6, 2] = (G6 * G2).sum()
        C[6, 3] = (G6 * G3).sum()
        C[6, 4] = (G6 * G4).sum()
        C[6, 5] = (G6 * G5).sum()
        C[6, 6] = (G6 * G6).sum()
        C[6, 7] = (G6 * G7).sum()
        #
        C[7, 0] = (G7 * G0).sum()
        C[7, 1] = (G7 * G1).sum()
        C[7, 2] = (G7 * G2).sum()
        C[7, 3] = (G7 * G3).sum()
        C[7, 4] = (G7 * G4).sum()
        C[7, 5] = (G7 * G5).sum()
        C[7, 6] = (G7 * G6).sum()
        C[7, 7] = (G7 * G7).sum()

        # ### Repeat image gradients by the number of parameters: ###
        # gx_repeated = torch.cat([gx] * number_of_parameters, -1)
        # gy_repeated = torch.cat([gy] * number_of_parameters, -1)
        #
        # G = gx_repeated * jac[0:h, :] + gy_repeated * jac[h:, :]  # TODO: understand if there's a better way then concatenating multiple times and then multiplying and then reshaping!!!
        # G = torch_reshape_image(G, (h * w, number_of_parameters), order='F').contiguous()  # TODO: understand what this outputs and maybe we can avoid the torch_reshape_image

        return G_list, C

    def initialize_things_for_first_run(self, reference_tensor):
        ### Initialize Image Pyramids: ###
        self.reference_tensor_output_list = [0] * self.number_of_levels
        self.H_list = [0] * self.number_of_levels
        self.W_list = [0] * self.number_of_levels
        self.x_vec_list = [0] * self.number_of_levels
        self.y_vec_list = [0] * self.number_of_levels
        self.X_mat_list = [0] * self.number_of_levels
        self.Y_mat_list = [0] * self.number_of_levels
        self.Jx_list = [0] * self.number_of_levels
        self.Jy_list = [0] * self.number_of_levels
        self.J0_list = [0] * self.number_of_levels
        self.J1_list = [0] * self.number_of_levels

        ### Get Image Pyramid: ###
        # (1). First Level (Highest Resolution):
        self.warped_image = torch.zeros_like(reference_tensor)
        H, W = reference_tensor.shape[-2:]
        self.reference_tensor_output_list[0] = reference_tensor
        self.H_list[0] = H
        self.W_list[0] = W
        self.x_vec_list[0] = torch.arange(0, W).to(reference_tensor.device)
        self.y_vec_list[0] = torch.arange(0, H).to(reference_tensor.device)
        [yy, xx] = torch.meshgrid(self.y_vec_list[0], self.x_vec_list[0])
        self.X_mat_list[0] = xx
        self.Y_mat_list[0] = yy
        x_vec_length = len(self.x_vec_list[0])
        y_vec_length = len(self.y_vec_list[0])
        self.x_vec_unsqueezed = self.x_vec_list[0].unsqueeze(0)
        self.y_vec_unsqueezed = self.y_vec_list[0].unsqueeze(-1)
        self.Jx_list[0] = torch.repeat_interleave(self.x_vec_unsqueezed, y_vec_length, 0)
        self.Jy_list[0] = torch.repeat_interleave(self.y_vec_unsqueezed, x_vec_length, 1)
        self.J0_list[0] = 0 * self.Jx_list[0]  # could also use zeros_like  #TODO: obviously this, like others, can and should be created beforehand!!!
        self.J1_list[0] = self.J0_list[0] + 1  # could also use ones_like
        # (2). Subsequence Levels (Lower Resolutions):
        for level_index in np.arange(1, self.number_of_levels):
            ### Interpolate: ###
            self.reference_tensor_output_list[level_index] = torch.nn.functional.interpolate(self.reference_tensor_output_list[level_index - 1], scale_factor=0.5)

            ### Get Meshgrids & Vecs: ###
            H_current = self.H_list[level_index]
            W_current = self.W_list[level_index]
            x_vec = torch.arange(0, W_current).to(reference_tensor.device)
            y_vec = torch.arange(0, H_current).to(reference_tensor.device)
            self.x_vec_list[level_index] = x_vec
            self.y_vec_list[level_index] = y_vec
            [yy, xx] = torch.meshgrid(self.y_vec_list[level_index], self.x_vec_list[level_index])
            self.X_mat_list[level_index] = xx
            self.Y_mat_list[level_index] = yy

            ### Get Jacobian Auxiliary Tensors: ###
            x_vec_length = len(self.x_vec_list[level_index])
            y_vec_length = len(self.y_vec_list[level_index])
            x_vec_unsqueezed = self.x_vec_list[level_index].unsqueeze(0)
            y_vec_unsqueezed = self.y_vec_list[level_index].unsqueeze(-1)
            self.Jx_list[level_index] = torch.repeat_interleave(x_vec_unsqueezed, y_vec_length, 0)
            self.Jy_list[level_index] = torch.repeat_interleave(y_vec_unsqueezed, x_vec_length, 1)
            self.J0_list[level_index] = 0 * self.Jx_list[level_index]  # could also use zeros_like  #TODO: obviously this, like others, can and should be created beforehand!!!
            self.J1_list[level_index] = self.J0_list[level_index] + 1  # could also use ones_like

    def forward(self, input_tensor, reference_tensor, delta_p_threshold=2e-3, flag_print=False):
        ### Initialize Things For Subsequence Runs: ###
        if self.reference_tensor_output_list is None:
            self.initialize_things_for_first_run(reference_tensor)

        ### Get Image Pyramid For input_tensor: ###
        # (1). First Level (Highest Resolution):
        input_tensor_output_list = [0] * self.number_of_levels
        input_tensor_vx_output_list = [0] * self.number_of_levels
        input_tensor_vy_output_list = [0] * self.number_of_levels
        input_tensor_output_list[0] = input_tensor  # TODO: only this and the below lines need to be created at each new forward, all the rest can and should be created beforehand
        [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2, -1])
        input_tensor_vx_output_list[0] = vx
        input_tensor_vy_output_list[0] = vy
        # (2). Subsequence Levels (Lower Resolutions):
        for level_index in np.arange(1, self.number_of_levels):
            ### Shape: ###
            H, W = input_tensor_output_list[level_index - 1].shape

            ### Interpolate: ###
            input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

            ### Get Gradients: ###
            # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
            [vy, vx] = torch.gradient(input_tensor_output_list[level_index], dim=[0, 1])  # TODO: only this needs to stay in the forward loop
            input_tensor_vx_output_list[level_index] = vx
            input_tensor_vy_output_list[level_index] = vy

        ### Run ECC algorithm for each level of the pyramid: ###
        for level_index in np.arange(self.number_of_levels, 0, -1):  # start with lowest resolution (highest level of the pyramid)
            ### Get Current Level input_tensor and reference_tensor: ###
            current_level_input_tensor = input_tensor_output_list[level_index - 1]
            current_level_reference_tensor = self.reference_tensor_output_list[level_index - 1]
            if len(current_level_reference_tensor.shape) == 3:
                C, A, B = current_level_reference_tensor.shape
                C, H, W = current_level_reference_tensor.shape
            elif len(current_level_reference_tensor.shape) == 2:
                A, B = current_level_reference_tensor.shape
                H, W = current_level_reference_tensor.shape

            ### Get input_tensor gradients: ###
            vx = input_tensor_vx_output_list[level_index - 1]
            vy = input_tensor_vy_output_list[level_index - 1]

            ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
            x_vec = self.x_vec_list[level_index - 1]
            y_vec = self.y_vec_list[level_index - 1]
            X_mat = self.X_mat_list[level_index - 1]
            Y_mat = self.Y_mat_list[level_index - 1]
            current_level_reference_tensor = current_level_reference_tensor.type(torch.float32)

            ### Get Current Level Jacobian Auxiliary Tensors: ###
            # TODO: i wonder if all these assignments take anything up...i think NOT!
            Jx = self.Jx_list[level_index - 1]
            Jy = self.Jy_list[level_index - 1]
            J0 = self.J0_list[level_index - 1]
            J1 = self.J1_list[level_index - 1]

            ### ECC, Forward Additive Algorithm: ###
            H_matrix = self.H_matrix
            for iteration_index in np.arange(self.number_of_iterations_per_level):
                if flag_print:
                    print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))
                current_level_input_tensor_warped = self.spatial_interpolation_torch(current_level_input_tensor, H_matrix, 'linear', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W)  # inverse(backward) warping

                ########################################################################################################################################################################
                ### define a mask to deal with warping outside the image borders: ###
                # (they may have negative values due to the subtraction of the mean value)
                # TODO: there must be an easier way to do this!!! no way i need all these calculations
                # TODO: i can probably simply calculate the center crop size and use that right? i should probably be able to roughly calculate the center crop as a function of H_matrix
                # TODO: notice i current don't use the interpolation_method input variable!!! i simply perform bicubic interpolation
                # #########(1). Using a logical mask to mask out non-valid pixels:  ###########
                # # (1.1). Warp ones map according to H_matrix:  #TODO: maybe i can interpolate binary pattern?!!?
                # ones_map = self.spatial_interpolation_torch(torch.ones_like(current_level_input_tensor), H_matrix, 'nearest', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W)  # inverse(backward) warping
                # ones_map = (ones_map != 0)
                # zeros_map = (ones_map == 0)
                # # (1.2). Get mean of windows #TODO: maybe i don't need to lower the mean because the images are so similiar to each other?!!?!?
                # numOfElem = ones_map.sum()
                # meanOfWim = (current_level_input_tensor_warped * ones_map).sum() / numOfElem
                # meanOfTemp = (current_level_reference_tensor * ones_map).sum() / numOfElem
                # # (1.3). Substract windows mean from windows:
                # current_level_input_tensor_warped = current_level_input_tensor_warped - meanOfWim  # zero - mean image; is useful for brightness change compensation, otherwise you can comment this line
                # current_level_reference_tensor_zero_mean = current_level_reference_tensor - meanOfTemp  # zero - mean reference_tensor
                # # (1.4). Zero-Out pixels outside overlapping regions:  #TODO: maybe this is all that's needed!
                # current_level_input_tensor_warped[zeros_map] = 0
                # current_level_reference_tensor_zero_mean[zeros_map] = 0
                # ###########(2). Only Cropping/Indexing, assuming no need for substracting window mean: ############  #TODO: see if this works and/or is faster
                # h_start, h_end, w_start, w_end = crop_size_after_homography(H_matrix, H, W, add_just_in_case=5)
                # current_level_reference_tensor_zero_mean = current_level_reference_tensor * 1.0
                # current_level_input_tensor_warped[...,0:h_start, :] = 0
                # current_level_reference_tensor_zero_mean[...,0:h_start, :] = 0
                # current_level_input_tensor_warped[..., :, 0:w_start] = 0
                # current_level_reference_tensor_zero_mean[..., :, 0:w_start] = 0
                # current_level_input_tensor_warped[..., :, -(W-w_end):] = 0
                # current_level_reference_tensor_zero_mean[..., :, -(W-w_end):] = 0
                # current_level_input_tensor_warped[..., -(H-h_end):, :] = 0
                # current_level_reference_tensor_zero_mean[..., -(H-h_end):, :] = 0
                ###### DON'T DO ANYTHING: #######
                current_level_reference_tensor_zero_mean = current_level_reference_tensor
                ####################################################################################################################################

                ### Gradient Image interpolation (warped gradients): ###
                vx_warped = self.spatial_interpolation_torch(vx, H_matrix, 'linear', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W)
                vy_warped = self.spatial_interpolation_torch(vy, H_matrix, 'linear', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W)

                ### Compute the jacobian of warp transform_string: ###
                J_list = self.get_jacobian_for_warp_transform_torch(x_vec + 1, y_vec + 1, H_matrix, Jx, Jy, J0, J1, self.transform_string, H, W)

                ### Compute the jacobian of warped image wrt parameters (matrix G in the paper): ###
                G_list, C = self.image_jacobian_torch(vx_warped, vy_warped, J_list, self.number_of_parameters)
                G0, G1, G2, G3, G4, G5, G6, G7 = G_list

                ### Coompute Hessian and its inverse: ###
                # C = torch.matmul(G_transposed, G)  # matrix multiplication, C = Hessian matrix.  #TODO: understand if this can be done efficiently, maybe using einops
                # C = MM_with_transpose_einsum(G, G)
                # cond = torch.linalg.cond(C)
                i_C = torch.linalg.inv(C)

                ### Compute projections of images into G: ###
                # (*). Calculate Gt:
                # TODO: why not simply calculate this in the above self.image_jacobian_torch function ???? - oh, okay, because they are being used individual to calculate C,
                # so i need to see if i can still unify the calculations!!!!
                Gt = torch.zeros((8, 1)).to(input_tensor.device)
                Gw = torch.zeros((8, 1)).to(input_tensor.device)
                Gt[0] = (G0 * current_level_reference_tensor_zero_mean).sum()
                Gt[1] = (G1 * current_level_reference_tensor_zero_mean).sum()
                Gt[2] = (G2 * current_level_reference_tensor_zero_mean).sum()
                Gt[3] = (G3 * current_level_reference_tensor_zero_mean).sum()
                Gt[4] = (G4 * current_level_reference_tensor_zero_mean).sum()
                Gt[5] = (G5 * current_level_reference_tensor_zero_mean).sum()
                Gt[6] = (G6 * current_level_reference_tensor_zero_mean).sum()
                Gt[7] = (G7 * current_level_reference_tensor_zero_mean).sum()
                # (*). Calculate Gw:
                Gw[0] = (G0 * current_level_input_tensor_warped).sum()
                Gw[1] = (G1 * current_level_input_tensor_warped).sum()
                Gw[2] = (G2 * current_level_input_tensor_warped).sum()
                Gw[3] = (G3 * current_level_input_tensor_warped).sum()
                Gw[4] = (G4 * current_level_input_tensor_warped).sum()
                Gw[5] = (G5 * current_level_input_tensor_warped).sum()
                Gw[6] = (G6 * current_level_input_tensor_warped).sum()
                Gw[7] = (G7 * current_level_input_tensor_warped).sum()

                ### ECC Closed Form Solution: ###
                # (1). compute lambda parameter:
                # TODO: maybe the norm of the warped tensor remains approximately the same and i can skip this stage????
                num = (torch.linalg.norm(current_level_input_tensor_warped)) ** 2 - torch.transpose(Gw, -1, -2) @ i_C @ Gw
                den = (current_level_input_tensor_warped * current_level_reference_tensor_zero_mean).sum() - torch.transpose(Gt, -1, -2) @ i_C @ Gw
                lambda_correction = num / den
                # (2). compute error vector:
                imerror = lambda_correction * current_level_reference_tensor_zero_mean - current_level_input_tensor_warped
                # (3). compute the projection of error vector into Jacobian G:
                Ge = torch.zeros((8, 1)).to(input_tensor.device)
                Ge[0] = (G0 * imerror).sum()
                Ge[1] = (G1 * imerror).sum()
                Ge[2] = (G2 * imerror).sum()
                Ge[3] = (G3 * imerror).sum()
                Ge[4] = (G4 * imerror).sum()
                Ge[5] = (G5 * imerror).sum()
                Ge[6] = (G6 * imerror).sum()
                Ge[7] = (G7 * imerror).sum()
                # (4). compute the optimum parameter correction vector:
                delta_p = torch.matmul(i_C, Ge)
                delta_p_norm = torch.norm(delta_p)

                ### Update Parameters: ###
                H_matrix = update_transform_params_torch(H_matrix, delta_p, self.transform_string)

                ### Break the loop if reached max number of iterations per level: ###
                flag_delta_p_small_enough = delta_p_norm <= delta_p_threshold
                flag_end_iterations = (iteration_index == self.number_of_iterations_per_level or flag_delta_p_small_enough)
                if flag_end_iterations:  # the algorithm is executed (number_of_iterations_per_level-1) times
                    break

                # print(H_matrix)
                # print(delta_p)

            ### END OF INTERNAL ITERATIONS (PER LEVEL) LOOP
        ### END OF PYRAMID number_of_levels LOOP:

        ### Get final H_matrix: ###
        final_warp = H_matrix

        ### return the final warped image using the whole support area (including margins): ###
        # TODO: fix this to allow for B,C,H,W ; B,T,C,H,W; C,H,W; H,W
        nx2 = self.x_vec_list[0]
        ny2 = self.y_vec_list[0]
        H, W = input_tensor.shape[-2:]
        C_input = 1
        self.warped_image = spatial_interpolation_torch(input_tensor[:, :], final_warp, 'linear', self.transform_string, nx2, ny2, H, W)
        # for ii in torch.arange(C_input):
        #     warpedImage[ii, :, :] = spatial_interpolation_torch(input_tensor[ii, :, :], final_warp, 'linear', self.transform_string, nx2, ny2, H, W)
        H_matrix = final_warp

        return H_matrix, self.warped_image


class ECC_Layer_Torch_Batch(nn.Module):
    # Initialize this with a module
    def __init__(self, input_tensor, reference_tensor, number_of_iterations_per_level, number_of_levels=1, transform_string='homography', delta_p_init=None):
        super(ECC_Layer_Torch_Batch, self).__init__()
        self.X = None
        self.Y = None
        self.device = reference_tensor.device

        ### Initialize Parameters: ###
        transform_string = str.lower(transform_string)
        # C_reference, H_reference, W_reference = reference_tensor.shape
        T,C,H,W = input_tensor.shape

        ### Initialize New Images For Algorithm To Change: ###
        initTemplate = reference_tensor
        reference_tensor = RGB2BW(reference_tensor).type(torch.float32).squeeze()

        ### Initialize H_matrix matrix: ###
        H_matrix, number_of_parameters = self.initialize_H_matrix(delta_p_init=None, transform_string=transform_string, device=self.device)

        ### in case of pyramid implementation, the initial transformation must be appropriately modified: ###
        for level_index in np.arange(0, number_of_levels - 1):
            H_matrix = correct_H_matrix_for_coming_level_torch(H_matrix, transform_string, 'lower_resolution')

        ### Assign To Internal Attributes: ###
        self.number_of_levels = number_of_levels
        self.number_of_iterations_per_level = number_of_iterations_per_level
        self.transform_string = transform_string
        self.number_of_parameters = number_of_parameters
        self.H_matrix = H_matrix.unsqueeze(0).repeat(T,1,1)
        self.reference_tensor_output_list = None

    def initialize_H_matrix(self, delta_p_init=None, transform_string='homography', device='cpu'):
        # (1). Translation:
        if transform_string == 'translation':
            number_of_parameters = 2  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.zeros((2, 1))
            else:
                H_matrix = delta_p_init
        # (2). Euclidean:
        elif transform_string == 'euclidean':
            number_of_parameters = 3  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
                H_matrix[-1, -1] = 0
            else:
                H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
        # (3). Affine:
        elif transform_string == 'affine':
            number_of_parameters = 6  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
                H_matrix[-1, -1] = 0
            else:
                H_matrix = torch.cat([delta_p_init, torch.zeros((1, 3))], 0)
        # (4). Homography:
        elif transform_string == 'homography':
            number_of_parameters = 8  # number of parameters
            if delta_p_init is None:
                H_matrix = torch.eye(3)
            else:
                H_matrix = delta_p_init

        ### Send To Device: ###
        H_matrix = H_matrix.to(self.device).type(torch.float32)

        return H_matrix, number_of_parameters

    def spatial_interpolation_torch(self, input_image, H_matrix, interpolation_method, transform_string, x_vec, y_vec, X_mat, Y_mat, H, W, bilinear_grid=None):
        # %OUT = spatial_interpolation_numpy(IN, H_matrix, STR, transform_string, x_vec, y_vec)
        # % This function implements the 2D spatial interpolation of image IN
        # %(inverse warping). The coordinates defined by x_vec,y_vec are projected through
        # % H_matrix thus resulting in new subpixel coordinates. The intensity values in
        # % new pixel coordinates are computed via bilinear interpolation
        # % of image IN. For other valid interpolation methods look at the help
        # % of Matlab function INTERP2.
        # %
        # % Input variables:
        # % IN:           the input image which must be warped,
        # % H_matrix:         the H_matrix transform,
        # % STR:          the string corresponds to interpolation method: 'linear',
        # %               'cubic' etc (for details look at the help file of
        # %               Matlab function INTERP2),
        # % transform_string:    the type of adopted transform: {'translation','euclidean','affine','homography'}
        # % x_vec:           the x-coordinate values of horizontal side of ROI (i.e. [xmin:xmax]),
        # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
        # %
        # % Output:
        # % OUT:          The warped (interpolated) image

        #################################################################################################################################
        # (1). Older, Slower Method:
        # ### Correct H_matrix If Needed: ###
        # if transform_string == 'affine' or transform_string == 'euclidean':
        #     if H_matrix.shape[0] == 2:
        #         H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)
        # if transform_string == 'translation':
        #     H_matrix = torch.cat([torch.eye(2), H_matrix], -1)
        #     H_matrix = torch.cat([H_matrix, torch.zeros((1, 3))], 0)
        #
        # ### create meshgrid and flattened coordinates array ([x,y,1] basis): ###
        # xy = torch.cat([torch.transpose(torch_flatten_image(X_mat, True, 'F'), -1, -2),
        #                 torch.transpose(torch_flatten_image(Y_mat, True, 'F'), -1, -2),
        #                 torch.ones((1, len(torch_flatten_image(Y_mat, True, 'F')))).to(input_image.device)], 0).to(input_image.device)
        #
        # ### 3X3 matrix transformation: ###
        # A = H_matrix
        # A[-1, -1] = 1
        #
        # ### new coordinates: ###
        # xy_prime = torch.matmul(A, xy)
        #
        # ### division due to homogenous coordinates: ###
        # if transform_string == 'homography':
        #     xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
        #     xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]
        #
        # ### Ignore third row: ###
        # xy_prime = xy_prime[0:2, :]
        #
        # ### Turn to float32 instead of float64: ###
        # xy_prime = xy_prime.type(torch.float32)
        #
        # ### Subpixel interpolation: ###
        # # out = cv2.remap(input_image, np.reshape(xy_prime[0,:]+1, (H,W)), np.reshape(xy_prime[1,:]+1, (H,W)), cv2.INTER_CUBIC)
        # final_X_grid = torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')
        # final_Y_grid = torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')
        # new_X = 2 * final_X_grid / max(W - 1, 1) - 1
        # new_Y = 2 * final_Y_grid / max(H - 1, 1) - 1
        # bilinear_grid = torch.cat([torch_get_4D(new_X, 'CH'), torch_get_4D(new_Y, 'CH')], dim=3)
        # out = torch.nn.functional.grid_sample(input_image.unsqueeze(0).unsqueeze(0), bilinear_grid, mode='bicubic').squeeze(0).squeeze(0)
        #################################################################################################################################

        #################################################################################################################################
        # (2). New Faster Method (currently assumes homography transform):
        ### Try Simple Transformation: ###
        if bilinear_grid is None:
            H_matrix_corrected = H_matrix.unsqueeze(-1)
            denom = (H_matrix_corrected[:, 2:3, 0:1] * X_mat + H_matrix_corrected[:, 2:3, 1:2] * Y_mat + H_matrix_corrected[:, 2:3, 2:3])
            xx_new = 2 * (H_matrix_corrected[:, 0:1, 0:1] * X_mat + H_matrix_corrected[:, 0:1, 1:2] * Y_mat + H_matrix_corrected[:, 0:1, 2:3]) / denom / max(W - 1, 1) - 1
            yy_new = 2 * (H_matrix_corrected[:, 1:2, 0:1] * X_mat + H_matrix_corrected[:, 1:2, 1:2] * Y_mat + H_matrix_corrected[:, 1:2, 2:3]) / denom / max(H - 1, 1) - 1
            ### Subpixel Interpolation 2: ###
            bilinear_grid = torch.cat([xx_new, yy_new], dim=3)

        out = torch.nn.functional.grid_sample(input_image, bilinear_grid, mode='bilinear')
        #################################################################################################################################

        return out, bilinear_grid

    def get_jacobian_for_warp_transform_torch(self, x_vec, y_vec, H_matrix, Jx, Jy, J0, J1, transform_string, H, W):
        #     %J = get_jacobian_for_warp_transform_numpy(x_vec, y_vec, H_matrix, transform_string)
        # % This function computes the jacobian J of H_matrix transform with respect
        # % to parameters. In case of homography/euclidean transform, the jacobian depends on
        # % the parameter values, while in affine/translation case is totally invariant.
        # %
        # % Input variables:
        # % x_vec:           the x-coordinate values of the horizontal side of ROI (i.e. [xmin:xmax]),
        # % y_vec:           the y-coordinate values of vertical side of ROI (i.e. [ymin:ymax]),
        # % H_matrix:         the H_matrix transform (used only in homography and euclidean case),
        # % transform_string:    the type of adopted transform
        # % {'affine''homography','translation','euclidean'}
        # %
        # % Output:
        # % J:            The jacobian matrix J

        ### Get vec sizes: ###
        x_vec_length = len(x_vec)
        y_vec_length = len(y_vec)

        ### Flatten Arrays: ###
        # J0 = torch_flatten_image(J0).squeeze()
        # J1 = torch_flatten_image(J1).squeeze()
        # Jx = torch_flatten_image(Jx)
        # Jy = torch_flatten_image(Jy)

        if str.lower(transform_string) == 'homography':
            ### New, Better Way: ###
            #TODO: i also think the both H*Jx and H*X_mat can be calculated at the same time when going to CUDA!!!!
            H_matrix_corrected = H_matrix.unsqueeze(-1)
            den = (H_matrix_corrected[:, 2:3, 0:1] * Jx + H_matrix_corrected[:, 2:3, 1:2] * Jy + H_matrix_corrected[:, 2:3, 2:3])  #TODO: this is used three times here!!!!! calculate once and use it!!!!!
            denom_inverse = 1/den
            xy_prime_reshaped_X = (H_matrix_corrected[:, 0:1, 0:1] * Jx + H_matrix_corrected[:, 0:1, 1:2] * Jy + H_matrix_corrected[:, 0:1, 2:3]) * denom_inverse
            xy_prime_reshaped_Y = (H_matrix_corrected[:, 1:2, 0:1] * Jx + H_matrix_corrected[:, 1:2, 1:2] * Jy + H_matrix_corrected[:, 1:2, 2:3]) * denom_inverse
            Jx = Jx * denom_inverse  # element-wise
            Jy = Jy * denom_inverse  # element-wise

            # ############################################################################################
            # ### Concatenate the flattened jacobians: ###
            # # TODO: aren't these all simply ones?!?! why?! we're just copying the H_matrix, which i can do without all of this
            # # TODO: i can simply calculate this in advanced!
            # # TODO: understand if i can do something else instead of flattening and unflattening!!!
            # xy = torch.cat([torch.transpose(torch_flatten_image(Jx, True, 'F'), -1, -2),  #TODO: all of this can be calculated in advance!!!, but maybe simply take a different approach
            #                 torch.transpose(torch_flatten_image(Jy, True, 'F'), -1, -2), #TODO: plus, Jx and Jy are known in advance and have very simple structure
            #                 torch.ones((1, x_vec_length * y_vec_length)).to(H_matrix.device)], 0)  # TODO: before axis was -1
            #
            # ### 3x3 matrix transformation: ###
            # A = H_matrix
            # A[2, 2] = 1
            #
            # ### new coordinates after H_matrix: ###
            # xy_prime = torch.matmul(A, xy)  # matrix multiplication
            #
            # ### division due to homogeneous coordinates: ###
            # xy_prime[0, :] = xy_prime[0, :] / xy_prime[2, :]  # element-wise
            # xy_prime[1, :] = xy_prime[1, :] / xy_prime[2, :]
            # den = torch.transpose(xy_prime[2:3, :], -1, -2)  # TODO: understand if this is needed
            # den = torch_reshape_flattened_image(den, (H, W), order='F')
            # Jx = Jx / den  # element-wise
            # Jy = Jy / den  # element-wise
            #
            # ### warped jacobian(???): ###
            # xy_prime_reshaped_X = torch_reshape_flattened_image(xy_prime[0, :], (H, W), order='F')    #TODO: understand what's faster, calculating and storing and accessing in advance or calculating on the spot
            # xy_prime_reshaped_Y = torch_reshape_flattened_image(xy_prime[1, :], (H, W), order='F')
            # ##################################################################################################################

            ### Assign Jacobian Elements: ###
            # #### V1: ####
            # #TODO: HERE IS WELL!!! superfluous calculations!!!!!
            # Jxx_prime = Jx
            # Jxx_prime = Jxx_prime * xy_prime_reshaped_X  # element-wise.
            # Jyx_prime = Jy
            # Jyx_prime = Jyx_prime * xy_prime_reshaped_X
            #
            # Jxy_prime = Jx
            # Jxy_prime = Jxy_prime * xy_prime_reshaped_Y  # element-wise
            # Jyy_prime = Jy
            # Jyy_prime = Jyy_prime * xy_prime_reshaped_Y
            #### V2: ####
            Jxx_prime = Jx * xy_prime_reshaped_X  # element-wise.
            Jyx_prime = Jy * xy_prime_reshaped_X
            Jxy_prime = Jx * xy_prime_reshaped_Y  # element-wise
            Jyy_prime = Jy * xy_prime_reshaped_Y

            ### Get final jacobian of the H_matrix with respect to the different parameters: ###
            # #TODO: maybe there's a better way then concatenating huge amounts of memory like this?...maybe i can simply understand where this goes and calculate what's needed instead
            # J_up = torch.cat([Jx, J0, -Jxx_prime, Jy, J0, - Jyx_prime, J1, J0], -1)
            # J_down = torch.cat([J0, Jx, -Jxy_prime, J0, Jy, -Jyy_prime, J0, J1], -1)
            # J = torch.cat([J_up, J_down], 0)
            J_list = [Jx, Jy, J0, J1, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime]

        elif str.lower(transform_string) == 'affine':
            # TODO: can be calculated in advance!!!
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            J_up = torch.cat([Jx, J0, Jy, J0, J1, J0], -1)
            J_down = torch.cat([J0, Jx, J0, Jy, J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        elif str.lower(transform_string) == 'translation':
            # TODO: can be calculated in advance!
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            J_up = torch.cat([J1, J0], -1)
            J_down = torch.cat([J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        elif str.lower(transform_string) == 'euclidean':
            Jx = Jx.squeeze()
            Jy = Jy.squeeze()
            mycos = H_matrix[1, 1]
            mysin = H_matrix[2, 1]

            Jx_prime = -mysin * Jx - mycos * Jy
            Jy_prime = mycos * Jx - mysin * Jy

            J_up = torch.cat([Jx_prime, J1, J0], -1)
            J_down = torch.cat([Jy_prime, J0, J1], -1)
            J = torch.cat([J_up, J_down], 0)

        return J_list

    def image_jacobian_torch(self, gx, gy, J_list, number_of_parameters):
        # %G = image_jacobian_numpy(GX, GY, JAC, number_of_parameters)
        # % This function computes the jacobian G of the warped image wrt parameters.
        # % This matrix depends on the gradient of the warped image, as
        # % well as on the jacobian JAC of the warp transform wrt parameters.
        # % For a detailed definition of matrix G, see the paper text.
        # %
        # % Input variables:
        # % GX:           the warped image gradient in x (horizontal) direction,
        # % GY:           the warped image gradient in y (vertical) direction,
        # % JAC:            the jacobian matrix J of the warp transform wrt parameters,
        # % number_of_parameters:          the number of parameters.
        # %
        # % Output:
        # % G:            The jacobian matrix G.
        #

        ### Get image shape: ###
        if len(gx.shape) == 2:
            h, w = gx.shape
        elif len(gx.shape) == 3:
            c, h, w = gx.shape

        # ### Unroll All Variables V1: ###
        # #TODO: if i remember correctly J0,J1 are simply zeros and ones. i can skip multiplications!!!!
        # [Jx, Jy, J0, J1, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime] = J_list
        # G0 = gx * Jx + gy * J0
        # G1 = gx * J0 + gy * Jx
        # G2 = -gx * Jxx_prime - gy * Jxy_prime
        # G3 = gx * Jy + gy * J0
        # G4 = gx * J0 + gy * Jy
        # G5 = -gx * Jyx_prime - gy * Jyy_prime
        # G6 = gx * J1 + gy * J0
        # G7 = gx * J0 + gy * J1
        ### Unroll All Variables V2 (using the fact the J0,J1 are simply zeros and ones and disregarding them): ###
        [Jx, Jy, J0, J1, Jxx_prime, Jxy_prime, Jyx_prime, Jyy_prime] = J_list
        G0 = gx * Jx
        G1 = gy * Jx
        G2 = -gx * Jxx_prime - gy * Jxy_prime
        G3 = gx * Jy
        G4 = gy * Jy
        G5 = -gx * Jyx_prime - gy * Jyy_prime
        G6 = gx
        G7 = gy

        # TODO: understand if making a list here takes time?!??
        G_list = [G0,
                  G1,
                  G2,
                  G3,
                  G4,
                  G5,
                  G6,
                  G7]

        ### PreCalculate C=(Gt*G): ###
        # TODO: would be smart to combine everything here together in the same moemory run
        T,C,H,W = gx.shape
        C = torch.zeros((T, 8, 8)).to(gx.device)
        C[:, 0, 0] = (G0 * G0).sum([-1,-2,-3])
        C[:, 0, 1] = (G0 * G1).sum([-1,-2,-3])
        C[:, 0, 2] = (G0 * G2).sum([-1,-2,-3])
        C[:, 0, 3] = (G0 * G3).sum([-1,-2,-3])
        C[:, 0, 4] = (G0 * G4).sum([-1,-2,-3])
        C[:, 0, 5] = (G0 * G5).sum([-1,-2,-3])
        C[:, 0, 6] = (G0 * G6).sum([-1,-2,-3])
        C[:, 0, 7] = (G0 * G7).sum([-1,-2,-3])
        #
        C[:, 1, 0] = (G1 * G0).sum([-1,-2,-3])
        C[:, 1, 1] = (G1 * G1).sum([-1,-2,-3])
        C[:, 1, 2] = (G1 * G2).sum([-1,-2,-3])
        C[:, 1, 3] = (G1 * G3).sum([-1,-2,-3])
        C[:, 1, 4] = (G1 * G4).sum([-1,-2,-3])
        C[:, 1, 5] = (G1 * G5).sum([-1,-2,-3])
        C[:, 1, 6] = (G1 * G6).sum([-1,-2,-3])
        C[:, 1, 7] = (G1 * G7).sum([-1,-2,-3])
        #
        C[:, 2, 0] = (G2 * G0).sum([-1,-2,-3])
        C[:, 2, 1] = (G2 * G1).sum([-1,-2,-3])
        C[:, 2, 2] = (G2 * G2).sum([-1,-2,-3])
        C[:, 2, 3] = (G2 * G3).sum([-1,-2,-3])
        C[:, 2, 4] = (G2 * G4).sum([-1,-2,-3])
        C[:, 2, 5] = (G2 * G5).sum([-1,-2,-3])
        C[:, 2, 6] = (G2 * G6).sum([-1,-2,-3])
        C[:, 2, 7] = (G2 * G7).sum([-1,-2,-3])
        #
        C[:, 3, 0] = (G3 * G0).sum([-1,-2,-3])
        C[:, 3, 1] = (G3 * G1).sum([-1,-2,-3])
        C[:, 3, 2] = (G3 * G2).sum([-1,-2,-3])
        C[:, 3, 3] = (G3 * G3).sum([-1,-2,-3])
        C[:, 3, 4] = (G3 * G4).sum([-1,-2,-3])
        C[:, 3, 5] = (G3 * G5).sum([-1,-2,-3])
        C[:, 3, 6] = (G3 * G6).sum([-1,-2,-3])
        C[:, 3, 7] = (G3 * G7).sum([-1,-2,-3])
        #
        C[:, 4, 0] = (G4 * G0).sum([-1,-2,-3])
        C[:, 4, 1] = (G4 * G1).sum([-1,-2,-3])
        C[:, 4, 2] = (G4 * G2).sum([-1,-2,-3])
        C[:, 4, 3] = (G4 * G3).sum([-1,-2,-3])
        C[:, 4, 4] = (G4 * G4).sum([-1,-2,-3])
        C[:, 4, 5] = (G4 * G5).sum([-1,-2,-3])
        C[:, 4, 6] = (G4 * G6).sum([-1,-2,-3])
        C[:, 4, 7] = (G4 * G7).sum([-1,-2,-3])
        #
        C[:, 5, 0] = (G5 * G0).sum([-1,-2,-3])
        C[:, 5, 1] = (G5 * G1).sum([-1,-2,-3])
        C[:, 5, 2] = (G5 * G2).sum([-1,-2,-3])
        C[:, 5, 3] = (G5 * G3).sum([-1,-2,-3])
        C[:, 5, 4] = (G5 * G4).sum([-1,-2,-3])
        C[:, 5, 5] = (G5 * G5).sum([-1,-2,-3])
        C[:, 5, 6] = (G5 * G6).sum([-1,-2,-3])
        C[:, 5, 7] = (G5 * G7).sum([-1,-2,-3])
        #
        C[:, 6, 0] = (G6 * G0).sum([-1,-2,-3])
        C[:, 6, 1] = (G6 * G1).sum([-1,-2,-3])
        C[:, 6, 2] = (G6 * G2).sum([-1,-2,-3])
        C[:, 6, 3] = (G6 * G3).sum([-1,-2,-3])
        C[:, 6, 4] = (G6 * G4).sum([-1,-2,-3])
        C[:, 6, 5] = (G6 * G5).sum([-1,-2,-3])
        C[:, 6, 6] = (G6 * G6).sum([-1,-2,-3])
        C[:, 6, 7] = (G6 * G7).sum([-1,-2,-3])
        #
        C[:, 7, 0] = (G7 * G0).sum([-1,-2,-3])
        C[:, 7, 1] = (G7 * G1).sum([-1,-2,-3])
        C[:, 7, 2] = (G7 * G2).sum([-1,-2,-3])
        C[:, 7, 3] = (G7 * G3).sum([-1,-2,-3])
        C[:, 7, 4] = (G7 * G4).sum([-1,-2,-3])
        C[:, 7, 5] = (G7 * G5).sum([-1,-2,-3])
        C[:, 7, 6] = (G7 * G6).sum([-1,-2,-3])
        C[:, 7, 7] = (G7 * G7).sum([-1,-2,-3])

        # ### Repeat image gradients by the number of parameters: ###
        # gx_repeated = torch.cat([gx] * number_of_parameters, -1)
        # gy_repeated = torch.cat([gy] * number_of_parameters, -1)
        #
        # G = gx_repeated * jac[0:h, :] + gy_repeated * jac[h:, :]  # TODO: understand if there's a better way then concatenating multiple times and then multiplying and then reshaping!!!
        # G = torch_reshape_image(G, (h * w, number_of_parameters), order='F').contiguous()  # TODO: understand what this outputs and maybe we can avoid the torch_reshape_image

        return G_list, C

    def initialize_things_for_first_run(self, input_tensor, reference_tensor):
        ### Initialize Image Pyramids: ###
        self.reference_tensor_output_list = [0] * self.number_of_levels
        self.H_list = [0] * self.number_of_levels
        self.W_list = [0] * self.number_of_levels
        self.x_vec_list = [0] * self.number_of_levels
        self.y_vec_list = [0] * self.number_of_levels
        self.X_mat_list = [0] * self.number_of_levels
        self.Y_mat_list = [0] * self.number_of_levels
        self.Jx_list = [0] * self.number_of_levels
        self.Jy_list = [0] * self.number_of_levels
        self.J0_list = [0] * self.number_of_levels
        self.J1_list = [0] * self.number_of_levels

        ### Get Image Pyramid: ###
        # (1). First Level (Highest Resolution):
        self.input_tensor_warped = torch.zeros_like(input_tensor)
        H, W = reference_tensor.shape[-2:]
        T,C,H,W = input_tensor.shape
        self.reference_tensor_output_list[0] = reference_tensor
        self.H_list[0] = H
        self.W_list[0] = W
        self.x_vec_list[0] = torch.arange(0, W).to(reference_tensor.device)
        self.y_vec_list[0] = torch.arange(0, H).to(reference_tensor.device)
        [yy, xx] = torch.meshgrid(self.y_vec_list[0], self.x_vec_list[0])
        self.X_mat_list[0] = xx.unsqueeze(-1).unsqueeze(0).repeat(T,1,1,1)  #TODO: make sure this is correct
        self.Y_mat_list[0] = yy.unsqueeze(-1).unsqueeze(0).repeat(T,1,1,1)
        x_vec_length = len(self.x_vec_list[0])
        y_vec_length = len(self.y_vec_list[0])
        self.x_vec_unsqueezed = self.x_vec_list[0].unsqueeze(0)
        self.y_vec_unsqueezed = self.y_vec_list[0].unsqueeze(-1)
        self.Jx_list[0] = torch.repeat_interleave(self.x_vec_unsqueezed, y_vec_length, 0).unsqueeze(0).unsqueeze(0).repeat(T,1,1,1)
        self.Jy_list[0] = torch.repeat_interleave(self.y_vec_unsqueezed, x_vec_length, 1).unsqueeze(0).unsqueeze(0).repeat(T,1,1,1)
        self.J0_list[0] = 0 * self.Jx_list[0]  # could also use zeros_like  #TODO: obviously this, like others, can and should be created beforehand!!!
        self.J1_list[0] = self.J0_list[0] + 1  # could also use ones_like
        # (2). Subsequence Levels (Lower Resolutions):
        for level_index in np.arange(1, self.number_of_levels):
            ### Interpolate: ###
            self.reference_tensor_output_list[level_index] = torch.nn.functional.interpolate(self.reference_tensor_output_list[level_index - 1], scale_factor=0.5)

            ### Get Meshgrids & Vecs: ###
            H_current = self.H_list[level_index]
            W_current = self.W_list[level_index]
            x_vec = torch.arange(0, W_current).to(reference_tensor.device)
            y_vec = torch.arange(0, H_current).to(reference_tensor.device)
            self.x_vec_list[level_index] = x_vec
            self.y_vec_list[level_index] = y_vec
            [yy, xx] = torch.meshgrid(self.y_vec_list[level_index], self.x_vec_list[level_index])
            self.X_mat_list[level_index] = xx.unsqueeze(-1).unsqueeze(0).repeat(T,1,1,1)
            self.Y_mat_list[level_index] = yy.unsqueeze(-1).unsqueeze(0).repeat(T,1,1,1)

            ### Get Jacobian Auxiliary Tensors: ###
            x_vec_length = len(self.x_vec_list[level_index])
            y_vec_length = len(self.y_vec_list[level_index])
            x_vec_unsqueezed = self.x_vec_list[level_index].unsqueeze(0)
            y_vec_unsqueezed = self.y_vec_list[level_index].unsqueeze(-1)
            self.Jx_list[level_index] = torch.repeat_interleave(x_vec_unsqueezed, y_vec_length, 0).unsqueeze(0).unsqueeze(0).repeat(T,1,1,1)
            self.Jy_list[level_index] = torch.repeat_interleave(y_vec_unsqueezed, x_vec_length, 1).unsqueeze(0).unsqueeze(0).repeat(T,1,1,1)
            self.J0_list[level_index] = 0 * self.Jx_list[level_index]  # could also use zeros_like  #TODO: obviously this, like others, can and should be created beforehand!!!
            self.J1_list[level_index] = self.J0_list[level_index] + 1  # could also use ones_like

    def forward(self, input_tensor, reference_tensor, delta_p_threshold=2e-3, flag_print=False):
        ### Initialize Things For Subsequence Runs: ###
        if self.reference_tensor_output_list is None:
            self.initialize_things_for_first_run(input_tensor, reference_tensor)

        ### Get Image Pyramid For input_tensor: ###
        # (1). First Level (Highest Resolution):
        input_tensor_output_list = [0] * self.number_of_levels
        input_tensor_vx_output_list = [0] * self.number_of_levels
        input_tensor_vy_output_list = [0] * self.number_of_levels
        input_tensor_output_list[0] = input_tensor  # TODO: only this and the below lines need to be created at each new forward, all the rest can and should be created beforehand
        [vy, vx] = torch.gradient(input_tensor_output_list[0], dim=[-2, -1])
        input_tensor_vx_output_list[0] = vx
        input_tensor_vy_output_list[0] = vy
        # (2). Subsequence Levels (Lower Resolutions):
        for level_index in np.arange(1, self.number_of_levels):
            ### Shape: ###
            H, W = input_tensor_output_list[level_index - 1].shape

            ### Interpolate: ###
            input_tensor_output_list[level_index] = torch.nn.functional.interpolate(input_tensor_output_list[level_index - 1], scale_factor=0.5)  # TODO: only this needs to stay in the forward loop

            ### Get Gradients: ###
            # TODO: maybe i can switch this over to be the reference_tensor and then i don't have to calculate for each frame!!!!
            [vy, vx] = torch.gradient(input_tensor_output_list[level_index], dim=[0, 1])  # TODO: only this needs to stay in the forward loop
            input_tensor_vx_output_list[level_index] = vx
            input_tensor_vy_output_list[level_index] = vy

        ### Run ECC algorithm for each level of the pyramid: ###
        for level_index in np.arange(self.number_of_levels, 0, -1):  # start with lowest resolution (highest level of the pyramid)
            ### Get Current Level input_tensor and reference_tensor: ###
            current_level_input_tensor = input_tensor_output_list[level_index - 1]
            current_level_reference_tensor = self.reference_tensor_output_list[level_index - 1]
            if len(current_level_reference_tensor.shape) == 4:
                T, C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 3:
                C, H, W = current_level_input_tensor.shape
            elif len(current_level_reference_tensor.shape) == 2:
                H, W = current_level_input_tensor.shape

            ### Get input_tensor gradients: ###
            vx = input_tensor_vx_output_list[level_index - 1]
            vy = input_tensor_vy_output_list[level_index - 1]

            ### Define the rectangular Region of Interest (ROI) by x_vec and y_vec (you can modify the ROI): ###
            x_vec = self.x_vec_list[level_index - 1]
            y_vec = self.y_vec_list[level_index - 1]
            X_mat = self.X_mat_list[level_index - 1]
            Y_mat = self.Y_mat_list[level_index - 1]
            current_level_reference_tensor = current_level_reference_tensor.type(torch.float32)

            ### Get Current Level Jacobian Auxiliary Tensors: ###
            # TODO: i wonder if all these assignments take anything up...i think NOT!
            Jx = self.Jx_list[level_index - 1]
            Jy = self.Jy_list[level_index - 1]
            J0 = self.J0_list[level_index - 1]
            J1 = self.J1_list[level_index - 1]

            ### ECC, Forward Additive Algorithm: ###
            H_matrix = self.H_matrix
            for iteration_index in np.arange(self.number_of_iterations_per_level):
                if flag_print:
                    print('Level: ' + str(level_index) + ', Iteration: ' + str(iteration_index))
                current_level_input_tensor_warped, bilinear_grid = self.spatial_interpolation_torch(current_level_input_tensor, H_matrix, 'linear', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W)  # inverse(backward) warping

                ########################################################################################################################################################################
                ### define a mask to deal with warping outside the image borders: ###
                # (they may have negative values due to the subtraction of the mean value)
                # TODO: there must be an easier way to do this!!! no way i need all these calculations
                # TODO: i can probably simply calculate the center crop size and use that right? i should probably be able to roughly calculate the center crop as a function of H_matrix
                # TODO: notice i current don't use the interpolation_method input variable!!! i simply perform bicubic interpolation
                # #########(1). Using a logical mask to mask out non-valid pixels:  ###########
                # # (1.1). Warp ones map according to H_matrix:  #TODO: maybe i can interpolate binary pattern?!!?
                # ones_map = self.spatial_interpolation_torch(torch.ones_like(current_level_input_tensor), H_matrix, 'nearest', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W)  # inverse(backward) warping
                # ones_map = (ones_map != 0)
                # zeros_map = (ones_map == 0)
                # # (1.2). Get mean of windows #TODO: maybe i don't need to lower the mean because the images are so similiar to each other?!!?!?
                # numOfElem = ones_map.sum()
                # meanOfWim = (current_level_input_tensor_warped * ones_map).sum() / numOfElem
                # meanOfTemp = (current_level_reference_tensor * ones_map).sum() / numOfElem
                # # (1.3). Substract windows mean from windows:
                # current_level_input_tensor_warped = current_level_input_tensor_warped - meanOfWim  # zero - mean image; is useful for brightness change compensation, otherwise you can comment this line
                # current_level_reference_tensor_zero_mean = current_level_reference_tensor - meanOfTemp  # zero - mean reference_tensor
                # # (1.4). Zero-Out pixels outside overlapping regions:  #TODO: maybe this is all that's needed!
                # current_level_input_tensor_warped[zeros_map] = 0
                # current_level_reference_tensor_zero_mean[zeros_map] = 0
                # ###########(2). Only Cropping/Indexing, assuming no need for substracting window mean: ############  #TODO: see if this works and/or is faster
                # h_start, h_end, w_start, w_end = crop_size_after_homography(H_matrix, H, W, add_just_in_case=5)
                # current_level_reference_tensor_zero_mean = current_level_reference_tensor * 1.0
                # current_level_input_tensor_warped[...,0:h_start, :] = 0
                # current_level_reference_tensor_zero_mean[...,0:h_start, :] = 0
                # current_level_input_tensor_warped[..., :, 0:w_start] = 0
                # current_level_reference_tensor_zero_mean[..., :, 0:w_start] = 0
                # current_level_input_tensor_warped[..., :, -(W-w_end):] = 0
                # current_level_reference_tensor_zero_mean[..., :, -(W-w_end):] = 0
                # current_level_input_tensor_warped[..., -(H-h_end):, :] = 0
                # current_level_reference_tensor_zero_mean[..., -(H-h_end):, :] = 0
                ###### DON'T DO ANYTHING: #######
                current_level_reference_tensor_zero_mean = current_level_reference_tensor
                ####################################################################################################################################

                ### Gradient Image interpolation (warped gradients): ###
                vx_warped, bilinear_grid = self.spatial_interpolation_torch(vx, H_matrix, 'linear', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W, bilinear_grid)
                vy_warped, bilinear_grid = self.spatial_interpolation_torch(vy, H_matrix, 'linear', self.transform_string, x_vec, y_vec, X_mat, Y_mat, H, W, bilinear_grid)

                ### Compute the jacobian of warp transform_string: ###
                J_list = self.get_jacobian_for_warp_transform_torch(x_vec + 1, y_vec + 1, H_matrix, Jx, Jy, J0, J1, self.transform_string, H, W)

                ### Compute the jacobian of warped image wrt parameters (matrix G in the paper): ###
                G_list, C = self.image_jacobian_torch(vx_warped, vy_warped, J_list, self.number_of_parameters)
                G0, G1, G2, G3, G4, G5, G6, G7 = G_list

                ### Coompute Hessian and its inverse: ###
                # C = torch.matmul(G_transposed, G)  # matrix multiplication, C = Hessian matrix.  #TODO: understand if this can be done efficiently, maybe using einops
                # C = MM_with_transpose_einsum(G, G)
                # cond = torch.linalg.cond(C)
                # i_C = torch.linalg.inv_ex(C)
                i_C = torch.linalg.inv(C)

                ### Compute projections of images into G: ###
                # (*). Calculate Gt:
                # TODO: why not simply calculate this in the above self.image_jacobian_torch function ???? - oh, okay, because they are being used individual to calculate C,
                # so i need to see if i can still unify the calculations!!!!
                Gt = torch.zeros((T, 8, 1)).to(input_tensor.device)
                Gw = torch.zeros((T, 8, 1)).to(input_tensor.device)
                Gt[:, 0] = (G0 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                Gt[:, 1] = (G1 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                Gt[:, 2] = (G2 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                Gt[:, 3] = (G3 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                Gt[:, 4] = (G4 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                Gt[:, 5] = (G5 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                Gt[:, 6] = (G6 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                Gt[:, 7] = (G7 * current_level_reference_tensor_zero_mean).sum([-1,-2])
                # (*). Calculate Gw:
                Gw[:, 0] = (G0 * current_level_input_tensor_warped).sum([-1,-2])
                Gw[:, 1] = (G1 * current_level_input_tensor_warped).sum([-1,-2])
                Gw[:, 2] = (G2 * current_level_input_tensor_warped).sum([-1,-2])
                Gw[:, 3] = (G3 * current_level_input_tensor_warped).sum([-1,-2])
                Gw[:, 4] = (G4 * current_level_input_tensor_warped).sum([-1,-2])
                Gw[:, 5] = (G5 * current_level_input_tensor_warped).sum([-1,-2])
                Gw[:, 6] = (G6 * current_level_input_tensor_warped).sum([-1,-2])
                Gw[:, 7] = (G7 * current_level_input_tensor_warped).sum([-1,-2])

                ### ECC Closed Form Solution: ###
                # (1). compute lambda parameter:
                # TODO: maybe the norm of the warped tensor remains approximately the same and i can skip this stage????
                num = (torch.linalg.norm(current_level_input_tensor_warped, dim=(-1,-2))).unsqueeze(-1) ** 2 - torch.transpose(Gw, -1, -2) @ i_C @ Gw
                den = (current_level_input_tensor_warped * current_level_reference_tensor_zero_mean).sum([-1, 2]).unsqueeze(-1) - torch.transpose(Gt, -1, -2) @ i_C @ Gw
                lambda_correction = (num / den).unsqueeze(-1)
                # (2). compute error vector:
                imerror = lambda_correction * current_level_reference_tensor_zero_mean - current_level_input_tensor_warped
                # (3). compute the projection of error vector into Jacobian G:
                Ge = torch.zeros((T, 8, 1)).to(input_tensor.device)
                Ge[:, 0] = (G0 * imerror).sum([-1,-2])
                Ge[:, 1] = (G1 * imerror).sum([-1,-2])
                Ge[:, 2] = (G2 * imerror).sum([-1,-2])
                Ge[:, 3] = (G3 * imerror).sum([-1,-2])
                Ge[:, 4] = (G4 * imerror).sum([-1,-2])
                Ge[:, 5] = (G5 * imerror).sum([-1,-2])
                Ge[:, 6] = (G6 * imerror).sum([-1,-2])
                Ge[:, 7] = (G7 * imerror).sum([-1,-2])
                # (4). compute the optimum parameter correction vector:
                delta_p = torch.matmul(i_C, Ge)
                delta_p_norm = torch.norm(delta_p, dim=1)

                ### Update Parameters: ###
                H_matrix = update_transform_params_torch(H_matrix, delta_p, self.transform_string)

                ### Break the loop if reached max number of iterations per level: ###
                # flag_delta_p_small_enough = delta_p_norm <= delta_p_threshold
                flag_end_iterations = (iteration_index == self.number_of_iterations_per_level)
                if flag_end_iterations:  # the algorithm is executed (number_of_iterations_per_level-1) times
                    break

                # print(H_matrix)
                # print(delta_p)

            ### END OF INTERNAL ITERATIONS (PER LEVEL) LOOP
        ### END OF PYRAMID number_of_levels LOOP:

        ### Get final H_matrix: ###
        final_warp = H_matrix

        ### return the final warped image using the whole support area (including margins): ###
        # TODO: fix this to allow for B,C,H,W ; B,T,C,H,W; C,H,W; H,W
        nx2 = self.x_vec_list[0]
        ny2 = self.y_vec_list[0]
        H, W = input_tensor.shape[-2:]
        C_input = 1
        self.warped_image, bilinear_grid = self.spatial_interpolation_torch(input_tensor, final_warp, 'linear', self.transform_string, nx2, ny2, self.X_mat_list[0], self.Y_mat_list[0], H, W)
        # for ii in torch.arange(C_input):
        #     warpedImage[ii, :, :] = spatial_interpolation_torch(input_tensor[ii, :, :], final_warp, 'linear', self.transform_string, nx2, ny2, H, W)
        H_matrix = final_warp

        return H_matrix, self.warped_image


def MM_with_transpose_einsum(input_tensor_1, input_tensor_2):
    # [input_tensor_1] = [H,W]
    # [input_tensor_2] = [H,W]
    # output_tensor = torch.transpose(input_tensor, -1, -2) * input_tensor_2
    return torch.einsum('k i, k j -> i j', input_tensor_1, input_tensor_2)


def BMM_with_transpose_einsum(input_tensor_1, input_tensor_2):
    # [input_tensor_1] = [B,H,W]
    # [input_tensor_2] = [B.H,W]
    # output_tensor = torch.transpose(input_tensor, -1, -2) * input_tensor_2
    return torch.einsum('b k i, b k j -> b i j', input_tensor_1, input_tensor_2)


from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import BW2RGB


def ECC_demo_numpy():
    # ### Read Input: ###
    # reference_tensor = read_image_default_torch() * 255
    # reference_tensor = RGB2BW(reference_tensor).type(torch.uint8)
    # ### Warp Input: ###
    # input_tensor = shift_matrix_subpixel_torch(reference_tensor, 1, 1)

    # path1 = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/image_1_before_homography.png'
    # path2 = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/image_1_after_homography.png'
    path1 = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/palantir_image_1.jpeg'
    path2 = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/palantir_image_2.jpeg'
    reference_tensor = read_image_torch(path1)
    input_tensor = read_image_torch(path2)

    ### Other Parameters: ###
    transform_string = 'homography'
    # delta_p_init = None
    delta_p_init = np.eye(3)
    delta_p_init[0, -1] = 0
    delta_p_init[1, -1] = 0

    ### ECC Layer: ###
    input_tensor = read_image_torch(path1)
    reference_tensor = read_image_torch(path2)
    reference_tensor = RGB2BW(reference_tensor[0]).cuda()[0].unsqueeze(0).unsqueeze(0)
    input_tensor = RGB2BW(input_tensor[0]).cuda()[0].unsqueeze(0).unsqueeze(0).repeat(20,1,1,1)
    ECC_layer_object = ECC_Layer_Torch_Batch(input_tensor, reference_tensor, number_of_iterations_per_level=50, number_of_levels=1, transform_string='homography', delta_p_init=delta_p_init)
    tic()
    H_matrix, warpedImage = ECC_layer_object.forward(input_tensor, reference_tensor, delta_p_threshold=0, flag_print=False)
    toc()
    # imshow_torch(input_tensor)
    # imshow_torch(reference_tensor)
    # imshow_torch(warpedImage)
    imshow_torch(warpedImage - input_tensor)
    imshow_torch((warpedImage - reference_tensor)[1])


ECC_demo_numpy()









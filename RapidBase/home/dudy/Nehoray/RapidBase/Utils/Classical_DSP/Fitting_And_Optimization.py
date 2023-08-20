import itertools

import numpy as np
from RapidBase.Utils.Tensor_Manipulation.linspace_arange import my_linspace
import matplotlib.pyplot as plt
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch, plot_torch
from RapidBase.Utils.IO.tic_toc import tic, toc
from RapidBase.Utils.Classical_DSP.Convolution_Utils import convn_torch

def polyfit_torch_FitSimplestDegree(x, y, max_degree_to_check=2, flag_get_prediction=True, flag_get_residuals=False):
    poly_residual = []
    for i in np.arange(max_degree_to_check):
        current_poly_degree = i
        coefficients, prediction, residual_values = polyfit_torch(x, y, current_poly_degree, flag_get_prediction=True, flag_get_residuals=True)
        poly_residual.append(residual_values.abs().mean().item() / (len(residual_values) - current_poly_degree - 1)**2)
    poly_residual_torch = torch.tensor(poly_residual)

    ### Choose Degre: ###
    #(1). simply choose the minimum:
    best_polynomial_degree = torch.argmin(poly_residual_torch, dim=0)
    #(2). choose the first index which starts to decrease slowly:
    poly_residual_torch_diff = torch.diff(poly_residual_torch)
    #(3). Fit to a distribution and find it's effective width or something:
    #TODO: do it

    ### Get "Best" Fit: ###
    coefficients, prediction, residual_values = polyfit_torch(x, y, best_polynomial_degree, flag_get_prediction=flag_get_prediction, flag_get_residuals=flag_get_residuals)

    return coefficients, prediction, residual_values, poly_residual_torch, best_polynomial_degree

def polyfit_torch_FitSimplestDegree_parallel(x, y, max_degree_to_check=2, flag_get_prediction=True, flag_get_residuals=False):
    poly_residual = []
    #TODO: make this parallel all the way!!!!
    for i in np.arange(max_degree_to_check):
        current_poly_degree = i
        coefficients, prediction, residual_values = polyfit_torch_parallel(x, y, current_poly_degree, flag_get_prediction=True, flag_get_residuals=True)
        poly_residual.append(residual_values.abs().mean().item() / (len(residual_values) - current_poly_degree - 1)**2)
    poly_residual_torch = torch.tensor(poly_residual)

    ### Choose Degre: ###
    #(1). simply choose the minimum:
    best_polynomial_degree = torch.argmin(poly_residual_torch, dim=0)
    #(2). choose the first index which starts to decrease slowly:
    poly_residual_torch_diff = torch.diff(poly_residual_torch)
    #(3). Fit to a distribution and find it's effective width or something:
    #TODO: do it

    ### Get "Best" Fit: ###
    coefficients, prediction, residual_values = polyfit_torch(x, y, best_polynomial_degree, flag_get_prediction=flag_get_prediction, flag_get_residuals=flag_get_residuals)

    return coefficients, prediction, residual_values, poly_residual_torch, best_polynomial_degree

def polyfit_torch_parallel(x, y, polynom_degree, flag_get_prediction=True, flag_get_residuals=False):
    ####################################################################
    ### New Version - Parallel: ###
    ### Assuming x is a flattened, 1D torch tensor: ###
    A = torch.ones_like(x).to(y.device).float()
    x = x.unsqueeze(-1)
    A = A.unsqueeze(-1)

    ### Polyfit using least-squares solution: ###
    for current_degree in np.arange(1, polynom_degree + 1):
        A = torch.cat((A, (x ** current_degree)), -1)
    returned_solution = torch.linalg.lstsq(A, y)
    coefficients = returned_solution.solution
    rank = returned_solution.rank
    residuals = returned_solution.residuals
    singular_values = returned_solution.singular_values

    ### Predict y using smooth polynom: ###
    if flag_get_prediction:
        x = x.squeeze(-1)
        prediction = 0
        for current_degree in np.arange(0, polynom_degree + 1):
            prediction += coefficients[:, current_degree:current_degree + 1] * x ** current_degree
    else:
        prediction = None

    ### calculate residual: ###
    if flag_get_residuals:
        residual_values = prediction - y
        residual_std = residual_values.std(-1)
    else:
        residual_values = None
        residual_std = None

    # index = 6
    # plot_torch(x[index,:], y[index,:])
    # plot_torch(x[index,:], prediction[index,:])

    return coefficients, prediction, residual_values, residual_std
    ####################################################################

def polyfit_torch(x, y, polynom_degree, flag_get_prediction=True, flag_get_residuals=False):
    # ### Possible Values: ###
    # polynom_degree = 2
    # x = torch.arange(0,5,0.1)
    # y = 1*1 + 2.1*x - 3.3*x**2 + 0.2*torch.randn_like(x)

    # ### Temp: ###
    # full_filename = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/y.pt'
    # y = torch.load(full_filename)
    # x = torch.arange(len(y)).to(y.device)
    ####################################################################

    ### Old Version - Unparallel: ###
    ### Assuming x is a flattened, 1D torch tensor: ###
    A = torch.ones(len(x)).to(y.device)
    x = x.unsqueeze(-1)
    A = A.unsqueeze(-1)

    ### Polyfit using least-squares solution: ###
    for current_degree in np.arange(1, polynom_degree + 1):
        A = torch.cat((A, (x ** current_degree)), -1)

    ### Perform Least Squares: ###
    returned_solution = torch.linalg.lstsq(A, y)
    # returned_solution_2 = (torch.linalg.inv(A.T @ A) @ A.T) @ y
    coefficients = returned_solution.solution
    rank = returned_solution.rank
    residuals = returned_solution.residuals
    singular_values = returned_solution.singular_values

    ### Predict y using smooth polynom: ###
    if flag_get_prediction:
        x = x.squeeze()
        prediction = 0
        for current_degree in np.arange(0, polynom_degree + 1):
            prediction += coefficients[current_degree] * x ** current_degree
    else:
        prediction = None

    ### calculate residual: ###
    if flag_get_residuals:
        residual_values = prediction - y
        residual_std = residual_values.std()
    else:
        residual_values = None
        residual_std = None
    ####################################################################


    return coefficients, prediction, residual_values



def polyval_torch(coefficients, x):
    x = x.squeeze()
    polynom_degree = len(coefficients) - 1
    prediction = 0
    for current_degree in np.arange(0, polynom_degree + 1):
        prediction += coefficients[current_degree] * x ** current_degree
    return prediction



def fit_polynomial(x, y):
    # solve for 2nd degree polynomial deterministically using three points
    a = (y[2] + y[0] - 2*y[1])/2
    b = -(y[0] + 2*a*x[1] - y[1] - a)
    c = y[1] - b*x[1] - a*x[1]**2
    return [c, b, a]

def fit_polynomial_torch(x, y):
    # solve for 2nd degree polynomial deterministically using three points
    a = (y[0,0,2] + y[0,0,0] - 2*y[0,0,1])/2
    b = -(y[0,0,0] + 2*a*x[1] - y[0,0,1] - a)
    c = y[0,0,1] - b*x[1] - a*x[1]**2
    return [c, b, a]


#TODO: create a pytorch version
def return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(y_vec, x_vec=None):
    # (*). Assumed 1D Input!!!!!!!!!!!!!!!
    ### take care of input: ###
    y_vec = np.ndarray.flatten(y_vec)
    if x_vec is None:
        x_vec = my_linspace(0 ,len(y_vec) ,len(y_vec))
    x_vec = np.ndarray.flatten(x_vec)

    ### get max index around which to interpolate: ###
    max_index = np.argmax(y_vec)
    if max_index == 0:  # TODO: maybe if max is at beginning of array return 0
        indices_to_fit = np.arange(0 ,2)
    elif max_index == len(y_vec) - 1:  # TODO: maybe if max is at end of array return last term
        indices_to_fit = np.arange(len(x_vec ) - 1 -2, len(x_vec ) -1 + 1)
    else:
        indices_to_fit = np.arange(max_index -1, max_index +1 + 1)

    ### Actually Fit: ###
    x_vec_to_fit = x_vec[indices_to_fit]
    y_vec_to_fit = y_vec[indices_to_fit]  # use only 3 points around max to make sub-pixel fit
    P = np.polynomial.polynomial.polyfit(x_vec_to_fit, y_vec_to_fit, 2)
    x_max = -P[1] / (2 * P[2])
    y_max = np.polynomial.polynomial.polyval(x_max ,P)

    return x_max, y_max

def return_shifts_using_parabola_fit_numpy(CC):
    W = CC.shape[-1]
    H = CC.shape[-2]
    max_index = np.argmax(CC)
    max_row = max_index // W
    max_col = max_index % W
    center_row = H // 2
    center_col = W // 2

    if max_row == 0 or max_row == H or max_col == 0 or max_col == W:
        z_max_vec = (0 ,0)
        shifts_total = (0 ,0)
    else:
        # fitting_points_x = CC[max_row:max_row + 1, max_col - 1:max_col + 2]
        # fitting_points_y = CC[max_row - 1:max_row + 2, max_col:max_col + 1]
        fitting_points_x = CC[max_row ,:]
        fitting_points_y = CC[:, max_col]

        ### Use PolyFit: ###
        x_vec = np.arange(-(H // 2), H // 2 + 1)
        shiftx, x_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_x, x_vec=x_vec)
        shifty, y_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_y, x_vec=x_vec)
        shiftx = shiftx - (max_row - center_row)
        shifty = shifty - (max_col - center_col)
        shifts_total = (shiftx, shifty)
        z_max_vec = (x_parabola_max, y_parabola_max)

    # ### Fast, minimum amount of operations: ###
    # y1 = fitting_points_x[:,0]
    # y2 = fitting_points_x[:,1]
    # y3 = fitting_points_x[:,2]
    # shiftx = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # y1 = fitting_points_y[0,:]
    # y2 = fitting_points_y[1,:]
    # y3 = fitting_points_y[2,:]
    # shifty = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # shifts_total = (shiftx, shifty)
    return shifts_total, z_max_vec


def return_shifts_using_parabola_fit_torch(CC):
    W = CC.shape[-1]
    H = CC.shape[-2]
    max_index = np.argmax(CC)
    max_row = max_index // W
    max_col = max_index % W
    max_value = CC[0 ,0 ,max_row, max_col]
    center_row = H // 2
    center_col = W // 2
    if max_row == 0 or max_row == H or max_col == 0 or max_col == W:
        z_max_vec = (0 ,0)
        shifts_total = (0 ,0)
    else:
        # fitting_points_x = CC[:,:, max_row:max_row+1, max_col-1:max_col+2]
        # fitting_points_y = CC[:,:, max_row-1:max_row+2, max_col:max_col+1]
        fitting_points_x = CC[: ,:, max_row, :]
        fitting_points_y = CC[: ,:, :, max_col]
        fitting_points_x = fitting_points_x.cpu().numpy()
        fitting_points_y = fitting_points_y.cpu().numpy()

        x_vec = np.arange(-(W // 2), W // 2 + 1)
        y_vec = np.arange(-(H // 2), H // 2 + 1)
        shiftx, x_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_x,
                                                                                               x_vec=x_vec)
        shifty, y_parabola_max = return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(fitting_points_y,
                                                                                               x_vec=y_vec)
        shifts_total = (shiftx, shifty)
        z_max_vec = (x_parabola_max, y_parabola_max)

    # ### Fast Way to only find shift: ###
    # y1 = fitting_points_x[:,:,:,0]
    # y2 = fitting_points_x[:,:,:,1]
    # y3 = fitting_points_x[:,:,:,2]
    # shiftx = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # y1 = fitting_points_y[:,:,0,:]
    # y2 = fitting_points_y[:,:,1,:]
    # y3 = fitting_points_y[:,:,2,:]
    # shifty = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # shifts_total = (float(shiftx[0][0][0]), float(shifty[0][0][0]))
    return shifts_total, z_max_vec


def return_shifts_using_paraboloid_fit(CC):
    ### Get sequence of (x,y) locations: ###
    cloc = 1  # Assuming 3X3. TODO: generalize to whatever size
    rloc = 1
    x = np.ndarray([1, cloc - 1, cloc - 1, cloc, cloc, cloc, cloc + 1, cloc + 1,
                    cloc + 1]) - 1  # the -1 is because we immigrated from matlab (1 based) to python (0 based)
    y = np.ndarray([rloc - 1, rloc, rloc + 1, rloc - 1, rloc, rloc + 1, rloc - 1, rloc, rloc + 1]) - 1

    ### Get corresponding z(x,y) values: ###
    cross_correlation_samples = np.zeros((len(x)))
    for k in np.arange(len(x)):
        cross_correlation_samples[k] = CC[x[k], y[k]]

        ### Fit paraboloid surface and get corresponding paraboloid coefficients: ###
    [coeffs] = fit_polynom_surface(x, y, cross_correlation_samples, 2)
    shifty = (-(coeffs(2) * coeffs(5) - 2 * coeffs(3) * coeffs(4)) / (coeffs(5) ^ 2 - 4 * coeffs(3) * coeffs(6)))
    shiftx = ((2 * coeffs(2) * coeffs(6) - coeffs(4) * coeffs(5)) / (coeffs(5) ^ 2 - 4 * coeffs(3) * coeffs(6)))

    ### Find Z_max at found (shiftx,shifty): ###
    z_max = evaluate_2d_polynom_surface(shiftx, shifty, coeffs)

    return (shiftx, shifty)

def center_of_mass(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention

    Parameters
    ----------
    z : np.array

    Returns
    -------
    (x,y) : tuple of floats
        The x and y locations of the center of mass of the parsed square
    """

    s = np.sum(z)
    if s != 0:
        z *= 1 / s
    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))
    return cx, cy

def evaluate_2d_polynom_surface(x, y, coeffs_mat):
    original_size = np.shape(x)
    x = x[:]
    y = y[:]
    z_values = np.zeros(x.shape)

    ### Combined x^i*y^j: ###
    for current_deg_x in np.arange(np.size(coeffs_mat)):
        for current_deg_y in np.arange(np.size(coeffs_mat) - current_deg_x + 1):
            z_values = z_values + coeffs_mat[current_deg_x, current_deg_y] * (x ** current_deg_x) * (
                        y ** current_deg_y)

    z_values = np.reshape(z_values, original_size)
    return z_values

def fit_polynom_surface(x, y, z, order):
    #  Fit a polynomial f(x,y) so that it provides a best fit
    #  to the data z.
    #  Uses SVD which is robust even if the data is degenerate.  Will always
    #  produce a least-squares best fit to the data even if the data is
    #  overspecified or underspecified.
    #  x, y, z are column vectors specifying the points to be fitted.
    #  The three vectors must be the same length.
    #  Order is the order of the polynomial to fit.
    #  Coeffs returns the coefficients of the polynomial.  These are in
    #  increasing power of y for each increasing power of x, e.g. for order 2:
    #  zbar = coeffs(1) + coeffs(2).*y + coeffs(3).*y^2 + coeffs(4).*x +
    #  coeffs(5).*x.*y + coeffs(6).*x^2
    #  Use eval2dPoly to evaluate the polynomial.

    [sizexR, sizexC] = np.shape(x)
    [sizeyR, sizeyC] = np.shape(y)
    [sizezR, sizezC] = np.shape(z)
    numVals = sizexR

    ### scale to prevent precision problems: ###
    scalex = 1.0 / max(abs(x))
    scaley = 1.0 / max(abs(y))
    scalez = 1.0 / max(abs(z))
    xs = x * scalex
    ys = y * scaley
    zs = z * scalez

    ### number of combinations of coefficients in resulting polynomial: ###
    numCoeffs = (order + 2) * (order + 1) / 2

    ### Form array to process with SVD: ###
    A = np.zeros(numVals, numCoeffs)

    column = 1
    for xpower in np.arange(order):
        for ypower in np.arange(order - xpower):
            A[:, column] = (xs ** xpower) * (ys ** ypower)
            column = column + 1

    ### Perform SVD: ###
    [u, s, v] = np.linalg.svd(A)

    ### pseudo-inverse of diagonal matrix s: ###
    eps = 2e-16
    sigma = eps ** (1 / order)  # minimum value considered non-zero
    qqs = np.diag(s)
    qqs[abs(qqs) >= sigma] = 1 / qqs[abs(qqs) >= sigma]
    qqs[abs(qqs) < sigma] = 0
    qqs = np.diag(qqs)
    if numVals > numCoeffs:
        qqs[numVals, 1] = 0  # add empty rows

    ### calculate solution: ###
    coeffs = v * np.transpose(qqs) * np.transpose(u) * zs

    ### scale the coefficients so they are correct for the unscaled data: ###
    column = 1
    for xpower in np.arange(order):
        for ypower in np.arange(order - xpower):
            coeffs[column] = coeffs(column) * (scalex ** xpower) * (scaley ** ypower) / scalez
            column = column + 1


from skimage.measure import label, regionprops, regionprops_table, find_contours
from RapidBase.Utils.IO.Imshow_and_Plots import draw_bounding_boxes_with_labels_on_image_XYXY, draw_circle_with_label_on_image, draw_ellipse_with_label_on_image, draw_trajectories_on_images, draw_text_on_image, draw_polygons_on_image, draw_polygon_points_on_image, draw_circles_with_labels_on_image, draw_ellipses_with_labels_on_image
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import make_tuple_int, get_COM_and_MOI_tensor_torch, logical_mask_to_indices_torch


def fit_ellipse_2D_using_scipy_regionprops_torch(input_tensor):
    ### Use regionprops to get ellipse properties: ###
    input_tensor_numpy = input_tensor.cpu().numpy()
    input_tensor_label_numpy = input_tensor.bool().cpu().numpy().astype(np.uint8)
    regionpprops_output = regionprops(input_tensor_label_numpy, input_tensor_numpy)
    ellipse_a = regionpprops_output[0].axis_major_length
    ellipse_b = regionpprops_output[0].axis_minor_length
    ellipse_axes_length = (ellipse_a, ellipse_b)
    ellipse_area = regionpprops_output[0].area
    ellipse_BB = regionpprops_output[0].bbox
    ellipse_centroid = regionpprops_output[0].centroid
    ellipse_central_moments = regionpprops_output[0].moments_central
    ellipse_orientation = regionpprops_output[0].orientation
    ellipse_intertia_tensor = regionpprops_output[0].inertia_tensor
    ellipse_intertia_eigenvalues = regionpprops_output[0].inertia_tensor_eigvals

    ### Plot Regionprops: ###
    ellipse_axes_length = (ellipse_axes_length[0] / np.sqrt(2), ellipse_axes_length[1] / np.sqrt(2))
    # ellipse_axes_length = (ellipse_axes_length[0] / 2, ellipse_axes_length[1] / 2)
    ellipse_axes_length = (ellipse_axes_length[1], ellipse_axes_length[0])
    ellipse_centroid = (ellipse_centroid[1], ellipse_centroid[0])

    ### Draw ellipse on numpy image: ###
    output_frame = draw_ellipse_with_label_on_image(input_tensor_numpy * 255,
                                                    center_coordinates=make_tuple_int(ellipse_centroid),
                                                    ellipse_angle=ellipse_orientation * 180 / np.pi,
                                                    axes_lengths=make_tuple_int(ellipse_axes_length),
                                                    ellipse_label='bla', line_thickness=3)

    # ### Plot: ###
    # plt.imshow(output_frame)
    # plt.imshow(input_tensor_numpy)

    return output_frame, ellipse_centroid, ellipse_axes_length

def fit_ellipse_2D_using_moment_of_intertia_torch(input_tensor):
    ### Use moment of inertia (MOI): ###
    cx, cy, cx2, cy2, cxy, MOI_tensor = get_MOI_tensor_torch(input_tensor.unsqueeze(0).unsqueeze(0))
    # torch.inverse(MOI_tensor)
    eigen_values, eigen_vectors = torch.linalg.eig(MOI_tensor)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real
    major_axis = eigen_vectors[:, 0]
    minor_axis = eigen_vectors[:, 1]
    ellipse_orientation = torch.arctan(minor_axis[1] / minor_axis[0])  #TODO: make sure to understand this better

    ellipse_centroid = (cx.item(), cy.item())
    ellipse_axes_length = (eigen_values[0].item(), eigen_values[1].item())
    ellipse_axes_length = (ellipse_axes_length[0] / 2 / np.sqrt(2), ellipse_axes_length[1] / 2 / np.sqrt(2))
    ellipse_axes_length = (ellipse_axes_length[1], ellipse_axes_length[0])

    ### Draw ellipse on numpy image: ###
    output_frame = draw_ellipse_with_label_on_image(input_tensor.cpu().numpy() * 255,
                                                    center_coordinates=make_tuple_int(ellipse_centroid),
                                                    ellipse_angle=ellipse_orientation.item() * 180 / np.pi,
                                                    axes_lengths=make_tuple_int(ellipse_axes_length),
                                                    ellipse_label='bla', line_thickness=3)

    # ### Plot: ###
    # plt.imshow(output_frame)
    # plt.imshow(input_tensor.cpu().numpy())
    # # imshow_torch(input_tensor)

    return output_frame, ellipse_centroid, ellipse_axes_length


def fit_ellipse_2D_outer_ring_using_least_squares(input_tensor):
    alpha = 5
    beta = 3
    N = 500
    DIM = 2

    np.random.seed(2)

    # Generate random points on the unit circle by sampling uniform angles
    theta = np.random.uniform(0, 2 * np.pi, (N, 1))
    eps_noise = 0.2 * np.random.normal(size=[N, 1])
    circle = np.hstack([np.cos(theta), np.sin(theta)])

    # Stretch and rotate circle to an ellipse with random linear tranformation
    B = np.random.randint(-3, 3, (DIM, DIM))
    noisy_ellipse = circle.dot(B) + eps_noise

    # Extract x coords and y coords of the ellipse as column vectors
    X = noisy_ellipse[:, 0:1]
    Y = noisy_ellipse[:, 1:]

    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()

    # Print the equation of the ellipse in standard form
    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'.format(x[0], x[1], x[2], x[3], x[4]))

    # Plot the noisy data
    plt.scatter(X, Y, label='Data Points')

    # Plot the original ellipse from which the data was generated
    phi = np.linspace(0, 2 * np.pi, 1000).reshape((1000, 1))
    c = np.hstack([np.cos(phi), np.sin(phi)])
    ground_truth_ellipse = c.dot(B)
    plt.plot(ground_truth_ellipse[:, 0], ground_truth_ellipse[:, 1], 'k--', label='Generating Ellipse')

    # Plot the least squares ellipse
    x_coord = np.linspace(-5, 5, 300)
    y_coord = np.linspace(-5, 5, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



def center_of_mass(z):
    """Return the center of mass of an array with coordinates in the
    hyperspy convention

    Parameters
    ----------
    z : np.array

    Returns
    -------
    (x,y) : tuple of floats
        The x and y locations of the center of mass of the parsed square
    """

    s = np.sum(z)
    if s != 0:
        z *= 1 / s
    dx = np.sum(z, axis=0)
    dy = np.sum(z, axis=1)
    h, w = z.shape
    cx = np.sum(dx * np.arange(w))
    cy = np.sum(dy * np.arange(h))
    return cx, cy



from scipy.optimize import curve_fit
from scipy.stats import chi2, norm, lognorm, loglaplace, laplace, poisson
from scipy.special import factorial
from scipy.stats import poisson
def fit_Gaussian(x, y):
    # N = 100
    # x = my_linspace(0, 5, N)
    # A = 1.5
    # mu = 2
    # sigma = 1.2
    # noise_sigma = 0.1
    # y = A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + noise_sigma*np.random.randn(N)

    def gaus(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / n)  # note this correction

    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
    y_fitted = gaus(x, *popt)

    # plot(x,y)
    # plot(x,y_fitted,'--')
    # legend(['real','fitted'])

    # hist, bin_edges = np.histogram(bla,100)
    # bin_centers = (bin_edges[1:]+bin_edges[0:-1])/2
    # x = bin_centers
    # y = hist
    popt[1] = popt[1]
    popt[2] = popt[2]

    return y_fitted, popt, pcov

def fit_LogGaussian(x, y):
    # sqrt(1 / (2 * pi)) / (s * x) * exp(- (log(x - m) ^ 2) / (2 * s ^ 2))
    # N = 100
    # x = linspace(3, 5, N)
    # A = 1.5
    # mu = 2
    # sigma = 1.2
    # noise_sigma = 0.1
    # y = A * np.exp(-(np.log(x - mu) ** 2 / (2 * sigma ** 2))) / (sigma * x) + noise_sigma * np.random.randn(N)

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction

    def gaus(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) / (sigma * x)

    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
    y_fitted = gaus(x, *popt)

    # plot(x, y)
    # plot(x, y_fitted, '--')
    # legend(['real', 'fitted'])

    return y_fitted

def fit_Laplace(x, y):
    # 1 / (2 * b) * exp(-abs(x - u) / b)
    # N = 100
    # x = linspace(0, 5, N)
    # A = 1.5
    # mu = 2
    # sigma = 1.2
    # noise_sigma = 0.1
    # y = A * np.exp(-np.abs(x - mu) / sigma) + noise_sigma * np.random.randn(N)

    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction

    def gaus(x, a, x0, sigma):
        return a * np.exp(-abs(x - x0) / sigma)

    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
    y_fitted = gaus(x, *popt)
    # plot(x, y)
    # plot(x, y_fitted, '--')
    # legend(['real', 'fitted'])

    return y_fitted, popt, pcov

def fit_Poisson(x, y):
    # N = 255
    # x = np.arange(0,255)
    # A = 1.5
    # mu = 143
    # sigma = 1.2
    # noise_sigma = 0.
    # y = A * np.exp(-mu) * (mu**x) / scipy.special.factorial(x) + noise_sigma * np.random.randn(N)

    def fit_function(x, lamb, A):
        '''poisson function, parameter lamb is the fit parameter'''
        return A * poisson.pmf(x, lamb)

    # fit with curve_fit
    parameters, cov_matrix = curve_fit(fit_function, x, y)

    y_fitted = fit_function(x, *parameters)

    # plot(x, y)
    # plot(x, y_fitted, '--')
    # legend(['real', 'fitted'])

    return y_fitted


import numbers


################################################################################################################
### Simple Model/Function/Objective Optimization Using Pytorch Gradient-Based Optimization: ###
import torch.nn as nn
import torch
class Model(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((3,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        self.real_coefficients = [1, -0.2, 0.3]

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-b * X) + c),
        """
        a, b, c = self.coefficients
        return a * torch.exp(-b * X) + c
        # return self.a * torch.exp(-self.k*X) + self.b

    def real_forward(self, X):
        a, b, c = self.real_coefficients
        return a * torch.exp(-b * X) + c


class FitFunctionTorch_Gaussian(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a * exp(-b * X) + c),
        """
        a, b, c, x0 = self.coefficients
        return a * torch.exp(-b * (X - x0) ** 2) + c


class FitFunctionTorch_Laplace(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * torch.exp(-b * (X - x0).abs()) + c


class FitFunctionTorch_Maxwell(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * (X ** 2) * torch.exp(-b * (X - x0).abs()) + c


class FitFunctionTorch_LogNormal(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a / (X ** 2) * torch.exp(-b * (torch.log(X) - x0).abs()) + c


class FitFunctionTorch_FDistribution(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, d1=None, d2=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if d1 is not None:
                self.coefficients[1] = d1
            if d2 is not None:
                self.coefficients[2] = d2
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, d1, d2, x0 = self.coefficients
        return a / X * torch.sqrt((d1 * X) ** d1 / (d1 * X + d2) ** (d1 + d2))


class FitFunctionTorch_Rayleigh(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * X * torch.exp(-b * (X - x0) ** 2) + c


class FitFunctionTorch_Lorenzian(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * 1 / (1 + ((X - x0) / b) ** 2) + c


class FitFunctionTorch_Sin(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 1).sample((4,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0 = self.coefficients
        return a * torch.sin(b * (X - x0)) + c

    def function_string(self):
        return 'a * sin(b*(x-x0)) + c,     variables: a,b,c,x0 '


class FitFunctionTorch_DecayingSin(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None, x1=None, t=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((6,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0

    def forward(self, X):
        a, b, c, x0, x1, t = self.coefficients
        return a * torch.sin(b * (X - x0)) * torch.exp(-t * (X - x1).abs()) + c

    def function_string(self):
        return 'a * sin(b*(x-x0)) * exp(-t|x-x1|) + c,    '


class FitFunctionTorch_DecayingSinePlusLine(nn.Module):
    """Custom Pytorch model for gradient optimization. """

    def __init__(self, a=None, b=None, c=None, x0=None, x1=None, t=None, d=None):
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((7,))
        # make weights torch parameters
        self.coefficients = nn.Parameter(weights)
        with torch.no_grad():
            if a is not None:
                self.coefficients[0] = a
            if b is not None:
                self.coefficients[1] = b
            if c is not None:
                self.coefficients[2] = c
            if x0 is not None:
                self.coefficients[3] = x0
            if x1 is not None:
                self.coefficients[4] = x1
            if t is not None:
                self.coefficients[5] = t
            if d is not None:
                self.coefficients[6] = d

    def forward(self, X):
        a, b, c, x0, x1, t, d = self.coefficients
        return a * torch.sin(b * (X - x0)) * torch.exp(-t * (X - x1).abs()) + c * X + d

    def function_string(self):
        return 'a * sin(b*(x-x0)) * exp(-t|x-x1|) + c*x + d,    '


def Coefficients_Optimizer_Torch(x, y,
                                 model,
                                 learning_rate=1e-3,
                                 loss_function=torch.nn.L1Loss(reduction='sum'),
                                 max_number_of_iterations=500,
                                 tolerance=1e-3,
                                 print_frequency=10):
    # learning_rate = 1e-6
    # loss_func = torch.nn.MSELoss(reduction='sum')

    # (*). x is the entire dataset inputs with shape [N,Mx]. N=number of observations, Mx=number of dimensions
    # (*). y is the entire dataset outputs/noisy-results with shape [N,My], My=number of output dimensions

    ### Define Optimizer: ###
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    flag_continue_optimization = True
    time_step = 1
    while flag_continue_optimization:
        ### Basic Optimization Step: ###
        # print(time_step)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        if time_step % print_frequency == print_frequency - 1:
            print('time step:  ' + str(time_step) + ';    current loss: ' + str(loss.item()))
        loss.backward()
        optimizer.step()

        ### Check whether to stop optimization: ###
        if loss < tolerance:
            flag_continue_optimization = False
            print('reached tolerance')
        if time_step == max_number_of_iterations:
            flag_continue_optimization = False
            print('reached max number of iterations')

        ### Advance time step: ###
        time_step += 1

    return model, model.coefficients


def get_initial_guess_for_sine(x, y_noisy):
    y_noisy_meaned = y_noisy - y_noisy.mean()
    y_noisy_smooth = convn_torch(y_noisy_meaned, torch.ones(10) / 10, dim=0).squeeze()
    y_noisy_smooth_sign_change = y_noisy_smooth[1:] * y_noisy_smooth[0:-1] < 0
    y_number_of_zero_crossings = y_noisy_smooth_sign_change.sum()
    zero_crossing_period = y_noisy_smooth_sign_change.float().nonzero().squeeze().diff().float().mean()
    full_cycle_period = zero_crossing_period * 2
    b_initial = (full_cycle_period) * (x[1] - x[0]) / (2 * pi)
    c_initial = y_noisy.mean()
    a_initial = (y_noisy.max() - y_noisy.min()) / 2
    return a_initial, b_initial, c_initial


# import csaps
# import interpol  #TODO: this brings out an error! cannot use this! maybe try linux!
from RapidBase.MISC_REPOS.torchcubicspline.torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline


def Testing_model_fit():
    ### Get Random Signal: ###
    x = torch.linspace(-5, 5, 100)
    y = torch.randn(100).cumsum(0)
    y = y - torch.linspace(0, y[-1].item(), y.size(0))
    # y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # plot_torch(y)

    ### Test Spline Fit Over Above Signal: ###
    # (1). CSAPS:
    x_numpy = x.cpu().numpy()
    y_numpy = y.squeeze().cpu().numpy()
    y_numpy_smooth_spline = csaps.csaps(xdata=x_numpy,
                                        ydata=y_numpy,
                                        xidata=x_numpy,
                                        weights=None,
                                        smooth=0.85)
    plt.plot(x_numpy, y_numpy)
    plt.plot(x_numpy, y_numpy_smooth_spline)
    # (2). TorchCubicSplines
    t = x
    x = y
    coeffs = natural_cubic_spline_coeffs(t, x.unsqueeze(-1))
    spline = NaturalCubicSpline(coeffs)
    x_estimate = spline.evaluate(t)
    plot_torch(t, x);
    plot_torch(t + 0.5, x_estimate)
    # (3). Using torch's Fold/Unfold Layers:
    # TODO: come up with a general solution for the division factor per index!!!!!!
    kernel_size = 20
    stride_size = 10
    number_of_index_overlaps = (kernel_size // stride_size)
    # y_image = y.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_image = torch.tensor(y_numpy_smooth_spline).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_image.shape
    y_image_unfolded = torch.nn.Unfold(kernel_size=(kernel_size, 1), dilation=1, padding=0, stride=(stride_size, 1))(y_image)
    # TODO: add polynomial fit for each segment/interval of the y_image_unfolded before folding back (as cheating, instead of proper boundary conditions in the least squares)
    y_image_folded = torch.nn.Fold(y_image.shape[-2:], kernel_size=(kernel_size, 1), dilation=1, padding=0, stride=(stride_size, 1))(y_image_unfolded)
    plot_torch(y)
    plot_torch(y_image.squeeze())
    plot_torch(y_image_folded.squeeze() / 2)
    # (4). Using Tensors Object's unfold method, need to add averaging of "in between" values or perform conditional least squares like i did in my function!!!: ###
    patch_size = 20
    interval_size = 20
    y_unfolded = y.unfold(-1, patch_size, interval_size)
    # y_unfolded = torch.tensor(y_numpy_smooth_spline).float().unfold(-1, patch_size, interval_size)
    x_unfolded = x.unfold(-1, patch_size, interval_size)
    outputs_list = []
    for patch_index in np.arange(y_unfolded.shape[0]):
        coefficients, prediction, residual_values = polyfit_torch(x_unfolded[patch_index], y_unfolded[patch_index], 2, True, True)
        outputs_list.append(prediction.unsqueeze(0))
    final_output = torch.cat(outputs_list, -1)
    final_output
    plot_torch(x, y);
    plot_torch(x, final_output)
    # (4). Loess (still, the same shit applies...i need to get a handle on folding/unfolding and boundary conditions):

    ### Get Sine Signal: ###
    x = torch.linspace(-10, 10, 1000)
    y = 1.4 * torch.sin(x) + 0.1 * x + 3
    y_noisy = y + torch.randn_like(x) * 0.1
    a_initial, b_initial, c_initial = get_initial_guess_for_sine(x, y_noisy)
    # plot_torch(x,y)
    # plot_torch(x,y_noisy)
    # plt.show()

    ### Fit Sine Signal: ###
    # model_to_fit = FitFunctionTorch_Sin(a_initial, b_initial, c_initial, None)
    # model_to_fit = FitFunctionTorch_DecayingSin(a_initial, b_initial, c_initial, None, None, 0)
    model_to_fit = FitFunctionTorch_DecayingSinePlusLine(a_initial, b_initial, None, None, None, 0)
    device = 'cuda'
    x = x.to(device)
    y_noisy = y_noisy.to(device)
    model_to_fit = model_to_fit.to(device)
    tic()
    model, coefficients = Coefficients_Optimizer_Torch(x, y_noisy,
                                                       model_to_fit,
                                                       learning_rate=0.5e-3,
                                                       loss_function=torch.nn.L1Loss(),
                                                       max_number_of_iterations=5000,
                                                       tolerance=1e-3, print_frequency=10)
    toc('optimization')
    print(model.coefficients)
    print(model_to_fit.function_string())
    plot_torch(x, y_noisy)
    plot_torch(x, model_to_fit.forward((x)))
    plt.legend(['input noisy', 'fitted model'])
    plt.show()


# Testing_model_fit()
################################################################################################################


################################################################################################################
### RANSAC: ###
def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)


def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError('Input data must be at least 2D.')


def _norm_along_axis(x, axis):
    """NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`."""
    return np.sqrt(np.einsum('ij,ij->i', x, x))


def _norm_along_axis_Torch(x, axis):
    """NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`."""
    return torch.sqrt(torch.einsum('ij,ij->i', x, x))


class BaseModel(object):
    def __init__(self):
        self.params = None


class LineModelND(BaseModel):
    """Total least squares estimator for N-dimensional lines.

    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.

    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::

        X = origin + lambda * direction

    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.

    # Examples
    # --------
    # >>> x = np.linspace(1, 2, 25)
    # >>> y = 1.5 * x + 3
    # >>> lm = LineModelND()
    # >>> lm.estimate(np.array([x, y]).T)
    # True
    # >>> tuple(np.round(lm.params, 5))
    # (array([1.5 , 5.25]), array([0.5547 , 0.83205]))
    # >>> res = lm.residuals(np.array([x, y]).T)
    # >>> np.abs(np.round(res, 9))
    # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #        0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> np.round(lm.predict_y(x[:5]), 3)
    # array([4.5  , 4.562, 4.625, 4.688, 4.75 ])
    # >>> np.round(lm.predict_x(y[:5]), 3)
    # array([1.   , 1.042, 1.083, 1.125, 1.167])
    #
    # """

    def estimate(self, data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)

        origin = data.mean(axis=0)
        data = data - origin

        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            norm = np.linalg.norm(direction)
            if norm != 0:  # this should not happen to be norm 0
                direction /= norm
        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _, _, v = np.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        self.params = (origin, direction)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        _check_data_atleast_2D(data)
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params
        res = (data - origin) - \
              ((data - origin) @ direction)[..., np.newaxis] * direction
        return _norm_along_axis(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.

        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        data : (n, m) array
            Predicted coordinates.

        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError('Line parallel to axis %s' % axis)

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data


class LineModelND_Torch(BaseModel):
    """Total least squares estimator for N-dimensional lines.

    In contrast to ordinary least squares line estimation, this estimator
    minimizes the orthogonal distances of points to the estimated line.

    Lines are defined by a point (origin) and a unit vector (direction)
    according to the following vector equation::

        X = origin + lambda * direction

    Attributes
    ----------
    params : tuple
        Line model parameters in the following order `origin`, `direction`.

    # Examples
    # --------
    # >>> x = np.linspace(1, 2, 25)
    # >>> y = 1.5 * x + 3
    # >>> lm = LineModelND()
    # >>> lm.estimate(np.array([x, y]).T)
    # True
    # >>> tuple(np.round(lm.params, 5))
    # (array([1.5 , 5.25]), array([0.5547 , 0.83205]))
    # >>> res = lm.residuals(np.array([x, y]).T)
    # >>> np.abs(np.round(res, 9))
    # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #        0., 0., 0., 0., 0., 0., 0., 0.])
    # >>> np.round(lm.predict_y(x[:5]), 3)
    # array([4.5  , 4.562, 4.625, 4.688, 4.75 ])
    # >>> np.round(lm.predict_x(y[:5]), 3)
    # array([1.   , 1.042, 1.083, 1.125, 1.167])
    #
    # """

    def estimate(self, data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        _check_data_atleast_2D(data)

        origin = data.mean(0)
        data = data - origin

        if data.shape[0] == 2:  # well determined
            direction = data[1] - data[0]
            norm = torch.linalg.norm(direction)
            if norm != 0:  # this should not happen to be norm 0
                direction /= norm
        elif data.shape[0] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _, _, v = torch.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        self.params = (origin, direction)

        return True

    def residuals(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        ### Make sure data is fine (at least 2d): ###
        _check_data_atleast_2D(data)

        ### Make sure we have some parameters which define the model so we can calculate the parameters: ###
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        ### Get residuals from model data points (in this case it's the distance between the data point to the direction line): ###
        origin, direction = params

        # ### Before Bug: ###
        # res = (data - origin) - ((data - origin) @ direction).unsqueeze(-1) * direction
        ### After Bug: ###
        bug_patch = torch.matmul(direction.unsqueeze(0), (data - origin).T).squeeze()
        res = (data - origin) - (bug_patch).unsqueeze(-1) * direction

        # bla1 = ((data-origin) @ direction)
        # bla2 = torch.mm(data-origin, direction.unsqueeze(1)).squeeze()
        return _norm_along_axis_Torch(res, axis=1)

    def predict(self, x, axis=0, params=None):
        """Predict intersection of the estimated line model with a hyperplane
        orthogonal to a given axis.

        Parameters
        ----------
        x : (n, 1) array
            Coordinates along an axis.
        axis : int
            Axis orthogonal to the hyperplane intersecting the line.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        data : (n, m) array
            Predicted coordinates.

        Raises
        ------
        ValueError
            If the line is parallel to the given axis.
        """
        ### Make sure params is defined so we can work with something here: ###
        if params is None:
            if self.params is None:
                raise ValueError('Parameters cannot be None')
            params = self.params
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')

        origin, direction = params

        if direction[axis] == 0:
            # line parallel to axis
            raise ValueError('Line parallel to axis %s' % axis)

        l = (x - origin[axis]) / direction[axis]
        data = origin + l[..., np.newaxis] * direction
        return data


def _dynamic_max_trials(n_inliers, n_samples, min_samples_for_model, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.
    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples_for_model : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.
    Returns
    -------
    trials : int
        Number of trials.
    """
    if n_inliers == 0:
        return np.inf

    nom = 1 - probability
    if nom == 0:
        return np.inf

    inlier_ratio = n_inliers / float(n_samples)
    denom = 1 - inlier_ratio ** min_samples_for_model
    if denom == 0:
        return 1
    elif denom == 1:
        return np.inf

    nom = np.log(nom)
    denom = np.log(denom)
    if denom == 0:
        return 0

    return int(np.ceil(nom / denom))


def check_random_state(seed):
    """Turn seed into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int or np.random.RandomState
           If `seed` is None, return the RandomState singleton used by `np.random`.
           If `seed` is an int, return a new RandomState instance seeded with `seed`.
           If `seed` is already a RandomState instance, return it.

    Raises
    ------
    ValueError
        If `seed` is of the wrong type.

    """
    # Function originally from scikit-learn's module sklearn.utils.validation
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def ransac(data, model_class, min_samples_for_model, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1, random_state=None, initial_inliers=None):
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:

    1. Select `min_samples_for_model` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.

    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.

    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:

         * ``success = estimate(*data)``
         * ``residuals(*data)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples_for_model : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to a high value
        such as 0.99, e is the current fraction of inliers w.r.t. the
        total number of samples, and m is the min_samples_for_model value.
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation


    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, https://en.wikipedia.org/wiki/RANSAC

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    # >>> t = np.linspace(0, 2 * np.pi, 50)
    # >>> xc, yc = 20, 30
    # >>> a, b = 5, 10
    # >>> x = xc + a * np.cos(t)
    # >>> y = yc + b * np.sin(t)
    # >>> data = np.column_stack([x, y])
    # >>> np.random.seed(seed=1234)
    # >>> data += np.random.normal(size=data.shape)
    #
    # Add some faulty data:
    #
    # >>> data[0] = (100, 100)
    # >>> data[1] = (110, 120)
    # >>> data[2] = (120, 130)
    # >>> data[3] = (140, 130)
    #
    # Estimate ellipse model using all available data:
    #
    # >>> model = EllipseModel()
    # >>> model.estimate(data)
    # True
    # >>> np.round(model.params)  # doctest: +SKIP
    # array([ 72.,  75.,  77.,  14.,   1.])
    #
    # Estimate ellipse model using RANSAC:
    #
    # >>> ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
    # >>> abs(np.round(ransac_model.params))
    # array([20., 30.,  5., 10.,  0.])
    # >>> inliers # doctest: +SKIP
    # array([False, False, False, False,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True], dtype=bool)
    # >>> sum(inliers) > 40
    # True
    #
    # RANSAC can be used to robustly estimate a geometric transformation. In this section,
    # we also show how to use a proportion of the total samples, rather than an absolute number.
    #
    # >>> from skimage.transform import SimilarityTransform
    # >>> np.random.seed(0)
    # >>> src = 100 * np.random.rand(50, 2)
    # >>> model0 = SimilarityTransform(scale=0.5, rotation=1, translation=(10, 20))
    # >>> dst = model0(src)
    # >>> dst[0] = (10000, 10000)
    # >>> dst[1] = (-100, 100)
    # >>> dst[2] = (50, 50)
    # >>> ratio = 0.5  # use half of the samples
    # >>> min_samples_for_model = int(ratio * len(src))
    # >>> model, inliers = ransac((src, dst), SimilarityTransform, min_samples_for_model, 10,
    # ...                         initial_inliers=np.ones(len(src), dtype=bool))
    # >>> inliers
    # array([False, False, False,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True,  True,  True,  True,  True,
    #         True,  True,  True,  True,  True])

    # """

    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    random_state = check_random_state(random_state)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data,)
    total_number_of_samples = len(data[0])

    if not (0 < min_samples_for_model < total_number_of_samples):
        raise ValueError("`min_samples_for_model` must be in range (0, <number-of-samples>)")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != total_number_of_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), total_number_of_samples))

    # for the first run use initial guess of inliers
    random_indices_to_sample = (initial_inliers if initial_inliers is not None
                                else random_state.choice(total_number_of_samples, min_samples_for_model, replace=False))

    for num_trials in range(max_trials):
        # do sample selection according data pairs
        samples = [d[random_indices_to_sample] for d in data]
        # for next iteration choose random sample set and be sure that no samples repeat
        random_indices_to_sample = random_state.choice(total_number_of_samples, min_samples_for_model, replace=False)

        # optional check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals ** 2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        if (
                # more inliers
                sample_inlier_num > best_inlier_num
                # same number of inliers but less "error" in terms of residuals
                or (sample_inlier_num == best_inlier_num
                    and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     total_number_of_samples,
                                                     min_samples_for_model,
                                                     stop_probability)
            if (best_inlier_num >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum
                    or num_trials >= dynamic_max_trials):
                break

    # estimate final model using all inliers
    if best_inliers is not None:
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)

    return best_model, best_inliers


def ransac_Torch(data, model_class, min_samples_for_model, residual_threshold,
                 is_data_valid=None, is_model_valid=None,
                 max_trials=100, stop_sample_num=torch.inf, stop_residuals_sum=0,
                 stop_probability=1, random_state=None, initial_inliers=None):
    ### Initialize Paramters: ###
    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    ### Get Random State: ###
    random_state = check_random_state(random_state)

    ### in case data is not pair of input and output, make it so: ###
    if not isinstance(data, (tuple, list)):
        data = (data,)
    total_number_of_samples = len(data[0])

    ### Check Inputs: ###
    if not (0 < min_samples_for_model < total_number_of_samples):
        raise ValueError("`min_samples_for_model` must be in range (0, <number-of-samples>)")
    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")
    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")
    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")
    if initial_inliers is not None and len(initial_inliers) != total_number_of_samples:
        raise ValueError("RANSAC received a vector of initial inliers (length %i)"
                         " that didn't match the number of samples (%i)."
                         " The vector of initial inliers should have the same length"
                         " as the number of samples and contain only True (this sample"
                         " is an initial inlier) and False (this one isn't) values."
                         % (len(initial_inliers), total_number_of_samples))

    ### for the first run use initial guess of inliers: ###
    random_indices_to_sample = (initial_inliers if initial_inliers is not None
                                else random_state.choice(total_number_of_samples, min_samples_for_model, replace=False))

    for num_trials in range(max_trials):
        ### do sample selection according data pairs: ###
        samples = [d[random_indices_to_sample] for d in data]

        ### for next iteration choose random sample set and be sure that no samples repeat: ###
        random_indices_to_sample = random_state.choice(total_number_of_samples, min_samples_for_model, replace=False)

        ### optional check if random sample set is valid: ###
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        ### Initialize new model and estimate parameters from current random sample set: ###
        sample_model = model_class()
        success = sample_model.estimate(*samples)
        # (*) backwards compatibility
        if success is not None and not success:
            continue

        ### optional check if estimated model is valid: ###
        if is_model_valid is not None and not is_model_valid(sample_model, *samples):
            continue

        ### Get residuals from model which was fit: ###
        sample_model_residuals = torch.abs(sample_model.residuals(*data))

        ### Get Consensus set (Inliers) and Residuals sum: ###
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = (sample_model_residuals ** 2).sum()

        ### choose as new best model if number of inliers is maximal: ###
        sample_inlier_num = (sample_model_inliers).sum()
        flag_current_model_with_more_inliers = (sample_inlier_num > best_inlier_num)
        flag_current_model_with_less_error = (sample_inlier_num == best_inlier_num and sample_model_residuals_sum < best_inlier_residuals_sum)
        flag_current_model_is_the_best_so_far = flag_current_model_with_more_inliers or flag_current_model_with_less_error
        # (*). if new model is the best then record it as such:
        if flag_current_model_is_the_best_so_far:
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            dynamic_max_trials = _dynamic_max_trials(best_inlier_num,
                                                     total_number_of_samples,
                                                     min_samples_for_model,
                                                     stop_probability)

            ### Test whether we've reached a point where the model is good enough: ###
            if (best_inlier_num.float() >= stop_sample_num
                    or best_inlier_residuals_sum <= stop_residuals_sum
                    or num_trials >= dynamic_max_trials):
                break

    ### estimate final model using all inliers: ###
    if best_inliers is not None:
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        best_model.estimate(*data_inliers)

    return best_model, best_inliers
#############################################################################################




#############################################################################################
### KORNIA RANSAC: ###
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from kornia.geometry import (
    find_fundamental,
    find_homography_dlt,
    find_homography_dlt_iterated,
    symmetrical_epipolar_distance,
)
# from kornia.geometry.homography import symmetric_transfer_error


class RANSAC_Homography_Kornia(nn.Module):
    """Module for robust geometry estimation with RANSAC.

    https://en.wikipedia.org/wiki/Random_sample_consensus

    Args:
        model_type: type of model to estimate, e.g. "homography" or "fundamental".
        inliers_threshold: threshold for the correspondence to be an inlier.
        batch_size: number of generated samples at once.
        max_iterations: maximum batches to generate. Actual number of models to try is ``batch_size * max_iterations``.
        confidence: desired confidence of the result, used for the early stopping.
        max_local_iterations: number of local optimization (polishing) iterations.
    """
    supported_models = ['homography', 'fundamental']

    def __init__(self,
                 model_type: str = 'homography',
                 inlier_threshold: float = 2.0,
                 batch_size: int = 2048,
                 max_iter: int = 10,
                 confidence: float = 0.99,
                 max_number_of_polish_iterations: int = 5):
        super().__init__()
        self.inlier_threshold = inlier_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_type = model_type
        self.confidence = confidence
        self.max_number_of_polish_iterations = max_number_of_polish_iterations
        self.model_type = model_type
        if model_type == 'homography':
            self.error_fn = symmetric_transfer_error  # type: ignore
            self.minimal_solver = find_homography_dlt  # type: ignore
            self.polisher_solver = find_homography_dlt_iterated  # type: ignore
            self.minimal_sample_size = 4
        elif model_type == 'fundamental':
            self.error_fn = symmetrical_epipolar_distance  # type: ignore
            self.minimal_solver = find_fundamental  # type: ignore
            self.minimal_sample_size = 8
            # ToDo: implement 7pt solver instead of 8pt minimal_solver
            # https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp#L498
            self.polisher_solver = find_fundamental  # type: ignore
        else:
            raise NotImplementedError(f"{model_type} is unknown. Try one of {self.supported_models}")

    def sample(self,
               sample_size: int,
               pop_size: int,
               batch_size: int,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches to get benefit of the parallel
        processing, esp.

        on GPU
        """
        rand = torch.rand(batch_size, pop_size, device=device)
        _, out = rand.topk(k=sample_size, dim=1)
        return out

    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> float:
        """Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus."""
        if n_inl == num_tc:
            return 1.0
        return math.log(1.0 - conf) / math.log(1. - math.pow(n_inl / num_tc, sample_size))

    def estimate_model_from_sampled_data(self,
                                      kp1: torch.Tensor,
                                      kp2: torch.Tensor) -> torch.Tensor:
        batch_size, sample_size = kp1.shape[:2]
        H = self.minimal_solver(kp1,
                                kp2,
                                torch.ones(batch_size,
                                           sample_size,
                                           dtype=kp1.dtype,
                                           device=kp1.device))
        return H

    def verify(self,
               kp1: torch.Tensor,
               kp2: torch.Tensor,
               models: torch.Tensor, inlier_threshold: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if len(kp1.shape) == 2:
            kp1 = kp1[None]
        if len(kp2.shape) == 2:
            kp2 = kp2[None]
        batch_size = models.shape[0]
        errors = self.error_fn(kp1.expand(batch_size, -1, 2),
                               kp2.expand(batch_size, -1, 2),
                               models)
        inl = (errors <= inlier_threshold)
        models_score = inl.to(kp1).sum(dim=1)
        best_model_idx = models_score.argmax()
        best_current_model_score = models_score[best_model_idx].item()
        model_best = models[best_model_idx].clone()
        inliers_best = inl[best_model_idx]
        return model_best, inliers_best, best_current_model_score

    def remove_bad_samples(self, kp1: torch.Tensor, kp2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        # ToDo: add (model-specific) verification of the samples,
        # E.g. constraints on not to be a degenerate sample
        return kp1, kp2

    def remove_bad_models(self, models: torch.Tensor) -> torch.Tensor:
        # ToDo: add more and better degenerate model rejection
        # For now it is simple and hardcoded
        main_diagonal = torch.diagonal(models,
                                       dim1=1,
                                       dim2=2)
        mask = main_diagonal.abs().min(dim=1)[0] > 1e-4
        return models[mask]

    def polish_model(self,
                     kp1: torch.Tensor,
                     kp2: torch.Tensor,
                     inliers: torch.Tensor) -> torch.Tensor:
        # TODO: Replace this with MAGSAC++ polisher
        kp1_inl = kp1[inliers][None]
        kp2_inl = kp2[inliers][None]
        num_inl = kp1_inl.size(1)
        model = self.polisher_solver(kp1_inl,
                                     kp2_inl,
                                     torch.ones(1,
                                                num_inl,
                                                dtype=kp1_inl.dtype,
                                                device=kp1_inl.device))
        return model

    def forward(self,
                kp1: torch.Tensor,
                kp2: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Main forward method to execute the RANSAC algorithm.

        Args:
            kp1 (torch.Tensor): source image keypoints :math:`(N, 2)`.
            kp2 (torch.Tensor): distance image keypoints :math:`(N, 2)`.
            weights (torch.Tensor): optional correspondences weights. Not used now

        Returns:
            - Estimated model, shape of :math:`(1, 3, 3)`.
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
            """
        if not isinstance(kp1, torch.Tensor):
            raise TypeError(f"Input kp1 is not torch.Tensor. Got {type(kp1)}")
        if not isinstance(kp2, torch.Tensor):
            raise TypeError(f"Input kp2 is not torch.Tensor. Got {type(kp2)}")
        if not len(kp1.shape) == 2:
            raise ValueError(f"Invalid kp1 shape, we expect Nx2 Got: {kp1.shape}")
        if not len(kp2.shape) == 2:
            raise ValueError(f"Invalid kp2 shape, we expect Nx2 Got: {kp2.shape}")
        if not (kp1.shape[0] == kp2.shape[0]) or (kp1.shape[0] < self.minimal_sample_size):
            raise ValueError(f"kp1 and kp2 should be \
                             equal shape at at least [{self.minimal_sample_size}, 2], \
                             got {kp1.shape}, {kp2.shape}")

        best_score_total: float = float(self.minimal_sample_size)
        num_tc: int = len(kp1)
        best_model_total = torch.zeros(3, 3, dtype=kp1.dtype, device=kp1.device)
        inliers_best_total: torch.Tensor = torch.zeros(num_tc, 1, device=kp1.device, dtype=torch.bool)
        for i in range(self.max_iter):
            # Sample minimal samples in batch to estimate models
            idxs = self.sample(self.minimal_sample_size, num_tc, self.batch_size, kp1.device)
            kp1_sampled = kp1[idxs]
            kp2_sampled = kp2[idxs]

            kp1_sampled, kp2_sampled = self.remove_bad_samples(kp1_sampled, kp2_sampled)
            # Estimate models
            models = self.estimate_model_from_sampled_data(kp1_sampled, kp2_sampled)
            models = self.remove_bad_models(models)
            if (models is None) or (len(models) == 0):
                continue
            # Score the models and select the best one
            model, inliers, current_model_score = self.verify(kp1, kp2, models, self.inlier_threshold)
            # Store far-the-best model and (optionally) do a local optimization
            if current_model_score > best_score_total:
                # Local optimization
                for local_optimization_step in range(self.max_number_of_polish_iterations):
                    model_after_local_optimization = self.polish_model(kp1, kp2, inliers)
                    if (model_after_local_optimization is None) or (len(model_after_local_optimization) == 0):
                        continue
                    _, inliers_local_optimization, score_local_optimization = self.verify(kp1, kp2, model_after_local_optimization, self.inlier_threshold)
                    # print (f"Orig score = {best_current_model_score}, LO score = {score_local_optimization} TC={num_tc}")
                    if score_local_optimization > current_model_score:
                        model = model_after_local_optimization.clone()[0]
                        inliers = inliers_local_optimization.clone()
                        current_model_score = score_local_optimization
                    else:
                        break
                # Now storing the best model
                best_model_total = model.clone()
                inliers_best_total = inliers.clone()
                best_score_total = current_model_score

                # Should we already stop?
                new_max_iter = int(self.max_samples_by_conf(int(best_score_total),
                                                            num_tc,
                                                            self.minimal_sample_size,
                                                            self.confidence))
                # print (f"New max_iter = {new_max_iter}")
                # Stop estimation, if the model is very good
                if (i + 1) * self.batch_size >= new_max_iter:
                    break
        # local optimization with all inliers for better precision
        return best_model_total, inliers_best_total


def closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False, clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)

import itertools
from RapidBase.Utils.IO.tic_toc import *
class RANSAC_NDLine_Kornia(nn.Module):
    """Module for robust geometry estimation with RANSAC.
    Args:
        model_type: type of model to estimate, e.g. "homography" or "fundamental".
        inliers_threshold: threshold for the correspondence to be an inlier.
        batch_size: number of generated samples at once.
        max_iterations: maximum batches to generate. Actual number of models to try is ``batch_size * max_iterations``.
        confidence: desired confidence of the result, used for the early stopping.
        max_local_iterations: number of local optimization (polishing) iterations.
    """

    def __init__(self,
                 model_type: str = 'homography',
                 inlier_threshold: float = 2.0,
                 model_number_of_inliners_threshold: int = 350,
                 max_number_of_models: int = 16,
                 params = None,
                 number_of_frames = 100,
                 batch_size: int = 2048,
                 max_iter: int = 10,
                 confidence: float = 0.99,
                 max_number_of_polish_iterations: int = 5):
        super().__init__()
        self.inlier_threshold = inlier_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.model_type = model_type
        self.confidence = confidence
        self.max_number_of_polish_iterations = max_number_of_polish_iterations
        self.model_type = model_type

        ### Populate Proper Models For RANSAC Estimation: ###
        self.error_fn = self.residuals_error_function
        self.initial_solver = self.estimate_model
        self.polisher_solver = self.estimate_model
        self.minimal_sample_size = 2
        self.model_number_of_inliners_threshold = model_number_of_inliners_threshold
        self.max_number_of_models = max_number_of_models
        self.params = params
        self.number_of_frames = number_of_frames

    def estimate_model(self, sampled_data):
        """Estimate line model from data.

        This minimizes the sum of shortest (orthogonal) distances
        from the given data points to the estimated line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimensionality dim >= 2.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """

        #TODO: i need to understand how to make this BATCH operations...first of all data.mean(1) probably, second i need to understand to make whether i'm over or well determined
        ### Get Holding Point For Data & Substract It: ###
        origin = sampled_data.mean(1, True)
        data = sampled_data - origin

        ### Estimate Model From Data: ###
        if data.shape[1] == 2:  # well determined
            direction = data[:, 1:2] - data[:, 0:1]  #Get ND-Line vector direction by vector substration
            norm = torch.linalg.norm(direction, dim=-1, keepdim=True)  #Get direction vector norm to normalize it to unit vector
            direction /= norm  #Return unit vector
        elif data.shape[1] > 2:  # over-determined
            # Note: with full_matrices=1 Python dies with joblib parallel_for.
            _, _, v = torch.linalg.svd(data, full_matrices=False)
            direction = v[0]
        else:  # under-determined
            raise ValueError('At least 2 input points needed.')

        line_params = (origin, direction)

        return line_params

    def residuals_error_function(self, data, params=None):
        """Determine residuals of data to model.

        For each point, the shortest (orthogonal) distance to the line is
        returned. It is obtained by projecting the data onto the line.

        Parameters
        ----------
        data : (N, dim) array
            N points in a space of dimension dim.
        params : (2, ) array, optional
            Optional custom parameter set in the form (`origin`, `direction`).

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """
        #TODO: understand how to make this batch operations....for instance matmul into batch mat mul
        ### Get Model Parameters: ###
        origin, direction = params

        ### Get residuals from model data points (in this case it's the distance between the data point to the direction line): ###
        whitened_data = data - origin
        bug_patch = torch.bmm(direction, whitened_data.transpose(-1,-2)).squeeze()
        residuals_around_direction = whitened_data - bug_patch.unsqueeze(-1) * direction

        ### Get Norm Along Axis: ###
        # TODO: NumPy < 1.8 does not support the `axis` argument for `np.linalg.norm`...maybe possilbe in pytorch?
        # TODO: why use the torch.einsum???....isn't it easier and not that slow to do it in a straight-forward manner?
        # norm_around_axis = torch.sqrt(torch.einsum('ij,ij->i', residuals_around_direction, residuals_around_direction))
        norm_around_axis = (residuals_around_direction**2).sum(-1).sqrt()

        return norm_around_axis

    def logical_mask_to_indices_torch(self, input_logical_mask, flag_return_tensor_or_list_of_tuples='tensor'):
        # flag_return_tensor_or_list_of_tuples = 'tensor' / 'list_of_tuples'
        if flag_return_tensor_or_list_of_tuples == 'tensor':
            return input_logical_mask.nonzero(as_tuple=False)
        else:
            return input_logical_mask.nonzero(as_tuple=True)

    def sample(self,
               sample_size: int,
               total_data_size: int,
               batch_size: int,
               device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Minimal sampler, but unlike traditional RANSAC we sample in batches to get benefit of the parallel
        processing, esp.

        on GPU
        """
        rand = torch.rand(batch_size, total_data_size, device=device)
        _, out = rand.topk(k=sample_size, dim=1)  #Probably not absolutely necessary but also probably not the bottleneck
        return out

    def get_all_possible_indices_combinations(self,
                     data,
                     sample_size: int,
                     total_data_size: int,
                     batch_size: int,
                     device: torch.device = torch.device('cpu')):
        ### Initialize Lists: ###
        self.valid_models_indices_list = []
        self.valid_models_scores_list = []
        self.valid_models_parameters_list = []
        self.valid_models_TXY_tensors_list = []
        self.valid_models_holding_vecs_list = []
        self.valid_models_direction_vecs_list = []
        self.valid_models_logical_mask_list = []
        self.valid_models_inliners_logical_mask_list = []
        self.model_t_vec_arange_list = []

        ### Perform In A Stupid Way By Looping: ###
        block_size = 64
        stride = 32
        number_of_blocks_H = int(np.floor(self.params.H / stride))
        number_of_blocks_W = int(np.floor(self.params.W / stride))
        indices_mat = np.full((number_of_blocks_H, number_of_blocks_W), object)
        points_mat = np.full((number_of_blocks_H, number_of_blocks_W), object)
        total_points = torch.empty((0, 2, 3)).to(data.device)
        data_indices_total = torch.empty((0, 2)).type(torch.long).to(data.device)
        #TODO: perhapse split this into 4 non-overlapping passes on the data?
        for i in np.arange(number_of_blocks_H):
            for j in np.arange(number_of_blocks_W):
                torch.cuda.empty_cache()
                # print('['+str(i)+', '+str(j)+']')
                # flag_block_active = self.params.points_grid_list[0][i][j].numel() > 0

                # if flag_block_active:
                ### Get Block Boundaries: ###
                start_H = i * stride
                stop_H = start_H + block_size
                start_W = j * stride
                stop_W = start_W + block_size
                # print('('+str(i)+','+str(j)+')')
                if stop_H < self.params.H and stop_W < self.params.W:
                    ### Get Logical Mask Of Points In Current Block: ###
                    # gtic()
                    current_block_logical_mask = (data[:, 1] >= start_W).float() * (data[:, 1] < stop_W).float() * (data[:, 2] >= start_H).float() * (data[:, 2] < stop_H).float()
                    current_block_logical_mask = current_block_logical_mask > 0
                    # current_block_logical_mask_indices = self.logical_mask_to_indices_torch(current_block_logical_mask, flag_return_tensor_or_list_of_tuples='tensor')
                    # print(current_block_logical_mask.sum())
                    # gtoc('get relevant outliers logical mask inside current grid block')

                    ### Get Relevant (T,X,Y) Points And Indices: ###
                    # gtic()
                    current_block_TXY_points = data[current_block_logical_mask]
                    # indices_mat[i, j] = data[current_block_logical_mask]
                    # gtoc('use logical mask to actually extract outliers from entire points')

                    ### Get Indices For RANSAC Within Current Block: ###
                    # (1). Get random indices from start and finish of temporal domain:
                    # gtic()
                    current_grid_T_indices = current_block_TXY_points[:, 0]
                    current_grid_T_indices_first_part = self.logical_mask_to_indices_torch(current_grid_T_indices <= 2).squeeze(-1).tolist()
                    current_grid_T_indices_last_part = self.logical_mask_to_indices_torch(current_grid_T_indices >= 98).squeeze(-1).tolist()
                    # current_grid_T_values_first_part = current_grid_T_indices[current_grid_T_indices <= 10].unique().type(torch.int).tolist()
                    # current_grid_T_values_last_part = current_grid_T_indices[current_grid_T_indices >= 90].unique().type(torch.int).tolist()
                    # gtoc('get T indices from first 2 frames and last two frames to create all possible combinations')

                    ### Look For Trajectories: ###
                    current_grid_T_indices_unique = torch.unique_consecutive(current_grid_T_indices)
                    flag_enough_points_for_RANSAC = len(current_grid_T_indices_unique) > 90 and \
                                                    len(current_grid_T_indices_first_part) > 0 and \
                                                    len(current_grid_T_indices_last_part) > 0
                    if flag_enough_points_for_RANSAC:
                        # gtic()
                        max_T_difference = current_grid_T_indices_unique.diff().max()
                        # gtoc('get max time between subseqeuent outliers, if it is larger than 2 then there is no hope so just continue')
                        if max_T_difference <= 10:
                            ### Get All Combinations: ###
                            # list = [current_grid_T_indices_first_part, current_grid_T_indices_last_part]
                            # all_points_combinations = [torch.tensor(p).unsqueeze(0) for p in itertools.product(*list)]
                            # all_points_combinations = torch.cat(all_points_combinations).to(data.device)
                            # gtic()
                            all_points_combinations = torch.tensor([(x, y) for x in current_grid_T_indices_first_part for y in current_grid_T_indices_last_part]).to(data.device)
                            # data_indices_total = torch.cat([data_indices_total, current_block_logical_mask_indices[all_points_combinations].squeeze(-1).long()], 0)
                            # gtoc('get all possible combinations for RANSAC')

                            ### Get Only A Subset Of The Data Iteratively To Avoid Memory OverFlow: ###
                            #TODO: sometimes i get to many points even in the small block, do something about it. maybe binning
                            #(1). iterative
                            #(2). center of mass
                            number_of_trajectories_per_batch = 5000
                            total_number_of_possible_trajectories = all_points_combinations.shape[0]
                            total_number_of_batches = np.ceil(total_number_of_possible_trajectories/number_of_trajectories_per_batch)
                            for trajectory_batch_index in np.arange(total_number_of_batches):
                                ### Get Current Batch: ###
                                start_index = int(trajectory_batch_index * number_of_trajectories_per_batch)
                                stop_index = int(min(start_index + number_of_trajectories_per_batch, total_number_of_possible_trajectories))
                                current_points_combinations = all_points_combinations[start_index:stop_index]

                                ### Get Points From Data: ###
                                valid_points_in_current_block = current_block_TXY_points[current_points_combinations]

                                ### Estimate Models From Sampled Batch: ###
                                # gtic()
                                models = self.estimate_model_from_sampled_data(valid_points_in_current_block)  # (origin, direction)
                                # gtoc('get all models (slope+intercept) for all possible combinations)')

                                ### Remove Bad Models If Needed: ###
                                models = self.remove_bad_models(models)

                                ### If No Models Remain Simply Continue To Next Iteration: ###
                                if (models is None) or (len(models) == 0):
                                    continue

                                ### Score the models and select the best one: ###
                                points_not_assigned_to_lines_yet, valid_models_logical_masks_list = self.verify_and_choose_best_model_from_current_batch(current_block_TXY_points,
                                                                                                                        models,
                                                                                                                        self.inlier_threshold,
                                                                                                                        self.model_number_of_inliners_threshold,
                                                                                                                        self.max_number_of_models)
                                # TODO: add some code which takes valid_models_logical_masks_list or something else and substracts it from the coming outliers to save on calculations
                                #  gotcha I'm on it :)
                                #  valid_models_logical_masks_list_sum = valid_models_logical_masks_list.sum(0) > 0
                                torch.cuda.empty_cache()

        #     print('******************************************************')
        #     print('******************************************************')
        #     print('******************************************************')
        # print('done')

        return points_not_assigned_to_lines_yet

    def sample_smart(self,
                     data,
                     sample_size: int,
                     total_data_size: int,
                     batch_size: int,
                     device: torch.device = torch.device('cpu')):
        ########################################################################################################################

        # ### After Getting All Possible Points -> Randomly Pick A Few: ###
        # #TODO: this is a VERY BAD WAY OF DOING THIS in every single way!!!!, i should probably only collect indices in the original "data" and do it using CUDA etc' etc'...
        # #TODO: also, perhapse, to speed things up, i SHOULD use the outer iterations loop to only randomize a small batch size each time, and at each time know some points out etc'...
        # def get_random_number_in_range(min_num, max_num, array_size=(1)):
        #     return (np.random.random(array_size) * (max_num - min_num) + min_num).astype('float32')
        # random_numbers = torch.tensor(get_random_number_in_range(0,total_points.shape[0], batch_size)).type(torch.int).long()
        # total_points = total_points[random_numbers]
        ####################################################################################################################################

        ####################################################################################################################################
        # ### Sample Smart - i want ~full trajectories for the small times sub-sequences i'm using. i want something which goes approximately from beginning to end of the sub-sequence: ###
        # flag_continue_sampling = True
        # total_points = torch.empty((0,2)).to(data.device)
        # while flag_continue_sampling:
        #     #(1). Get Last Index Of First %10 of temporal domain:
        #     last_index_of_first_fraction_of_temporal_domain = (data[:,0] <= 5).sum().item()  #TODO: make this a parameter which is a fraction of the total amount of number of frames
        #     first_index_of_last_fraction_of_temporal_domain = (data[:,0] <= 95).sum().item()
        #     N_first_possibilities = last_index_of_first_fraction_of_temporal_domain
        #     N_last_possibilities = (total_data_size - first_index_of_last_fraction_of_temporal_domain)
        #     N_total_possibillities = N_first_possibilities * N_last_possibilities
        #     #TODO: these are to total number of possibilities without any constraints...what i should do i divide the data apriori between different regions and
        #     # only allow sampling between those regions. that will dramatically lower the number of possibilities
        #
        #     #(2). Get random indices from start and finish:
        #     first_indices = torch.randint(0, last_index_of_first_fraction_of_temporal_domain, (batch_size, 1))
        #     second_indices = torch.randint(first_index_of_last_fraction_of_temporal_domain, total_data_size, (batch_size, 1))
        #     out = torch.cat([first_indices, second_indices], -1)
        #
        #     #(3). Cross Out A-Priori Impossible Data (points very far away from each other):
        #     data_first_indices = data[first_indices][:,:,1:].squeeze(1)
        #     data_second_indices = data[second_indices][:,:,1:].squeeze(1)
        #     data_points_distance = torch.norm(data_first_indices - data_second_indices, dim=-1)
        #     valid_points_logical_mask = data_points_distance < 20 #TODO: make this a parameter
        #     out = out[valid_points_logical_mask].to(data.device)
        #     total_points = torch.cat([total_points, out], 0)
        #
        #     flag_continue_sampling = total_points.shape[0] < batch_size
        # # print('done')
        ############################################################################################################

        return data_indices_total

    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> float:
        """Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus."""
        if n_inl == num_tc:
            return 1.0
        return math.log(1.0 - conf) / math.log(1. - math.pow(n_inl / num_tc, sample_size))

    def estimate_model_from_sampled_data(self, sampled_data: torch.Tensor) -> torch.Tensor:
        ### Get Input Data Shape: ###
        batch_size, sample_size = sampled_data.shape[:2]

        ### Estimate Model From Data: ###
        line_params = self.initial_solver(sampled_data)  #returns (origin, direction)

        return line_params   #(origin, direction)

    def Decide_If_Trajectory_Valid_Drone_Torch(self, xyz_line, t_vec_arange, direction_vec, number_of_frames, params, flag_return_string=True):
        ### Get variables from params dictionary: ###
        DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible = params['DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible']
        DroneTrjDec_maximum_projection_onto_XY_plane = params['DroneTrjDec_maximum_projection_onto_XY_plane']
        DroneTrjDec_minimum_projection_onto_XY_plane = params['DroneTrjDec_minimum_projection_onto_XY_plane']
        DroneTrjDec_max_time_gap_in_trajectory = params['DroneTrjDec_max_time_gap_in_trajectory']
        DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible = params['DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible']
        SeqT = params['SeqT']
        FrameRate = params['FPS']
        DroneTrjDec_allowed_BB_within_frame_H_by_fraction = params['DroneTrjDec_allowed_BB_within_frame_H_by_fraction']
        DroneTrjDec_allowed_BB_within_frame_W_by_fraction = params['DroneTrjDec_allowed_BB_within_frame_W_by_fraction']
        H = params.H
        W = params.W

        ### make sure there are enough time stamps to even look into trajectory: ###
        # t_vec is the smooth, incrementally increasing, t_vec.   t_vec = np.arange(min(xyz_line[:,0]), max(xyz_line[:,0])))
        flag_trajectory_valid = len(t_vec_arange) > (number_of_frames * DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible)
        output_message = ''
        if flag_trajectory_valid == False:
            if flag_return_string:
                output_message = 'number of trajectory time steps too little. len(t_vec) = ' + str(len(t_vec_arange)) + ', minimum required: ' + str(number_of_frames * DroneTrjDec_minimum_fraction_of_total_time_drone_is_visible)
        if flag_trajectory_valid:
            ### make sure velocity is between min and max values: ###
            b_abs = direction_vec.abs()

            ### the portion of the unit (t,x,y) direction vec's projection on the (x,y) plane: ###
            v = (b_abs[1] ** 2 + b_abs[2] ** 2) ** 0.5

            ### make sure the drone's "velocity" in the (x,y) is between some values, if it's too stationary it's weird and if it's too fast it's weird: ###
            # TODO: for experiments DroneTrjDec_minimum_projection_onto_XY_plane should be 0, IN REAL LIFE it should be >0 because drones usually move
            flag_trajectory_valid = ((v <= DroneTrjDec_maximum_projection_onto_XY_plane) & (v >= DroneTrjDec_minimum_projection_onto_XY_plane))
            if flag_trajectory_valid == False:
                if flag_return_string:
                    output_message = 'drone trajectory projection onto XY plane not in the appropriate range'
            if flag_trajectory_valid:

                ### make sure max difference between time stamps, which is also the maximum consecutive frames suspect disappeared, is not too high: ###
                original_t_vec = xyz_line[:, 0]
                original_t_vec_unique = original_t_vec.unique()  # SLIGHTLY different from t_vec....t_vec is built using np.arange(min_events_T, max_events_T), this is simply number of unique values
                flag_trajectory_valid = original_t_vec_unique.diff().max() <= DroneTrjDec_max_time_gap_in_trajectory
                if flag_trajectory_valid == False:
                    if flag_return_string:
                        output_message = 'max   time different between time stamps is too large'
                if flag_trajectory_valid:

                    ### Make sure total time for valid points is above the minimum we decided upon: ###
                    total_amount_of_valid_original_t_vec_samples = len(original_t_vec_unique)
                    minimum_total_valid_time_range_for_suspect = (DroneTrjDec_minimum_fraction_of_trajectory_time_drone_is_visible * number_of_frames)  # this is stupid, SeqT*FrameRate = number of frames
                    flag_trajectory_valid = minimum_total_valid_time_range_for_suspect < total_amount_of_valid_original_t_vec_samples

                    if flag_trajectory_valid == False:
                        if flag_return_string:
                            output_message = 'total valid time for suspect is not enough ' + \
                                             ', original len(t_vec): ' + \
                                             str(total_amount_of_valid_original_t_vec_samples) + \
                                             ', whereas minimum amount of time wanted is: ' \
                                             + str(minimum_total_valid_time_range_for_suspect)

        # flag_trajectory_valid = True  #TODO: temp!, delete
        if flag_trajectory_valid:
            output_message = 'Trajectory Valid'
        return flag_trajectory_valid, output_message

    def verify_and_choose_best_model_from_current_batch(self,
                                                        data: torch.Tensor,
                                                        line_params: torch.Tensor,
                                                        inlier_threshold: float,
                                                        model_number_of_inliners_threshold: float,
                                                        max_number_of_models: float) -> Tuple[torch.Tensor, torch.Tensor, float]:

        ### Get Errors/Residuals To Estimate Models Validity: ###
        lines_origins, lines_directions = line_params
        # gtic()
        errors = self.error_fn(data, line_params)
        # gtoc('get error map ')

        ### Get All Models Within Threshold: ###
        # gtic()
        inliners_logical_mask = (errors <= inlier_threshold)
        inliners_logical_mask_float = inliners_logical_mask.float() #TODO: i can save this by using bitwise operators
        # gtoc('get error function logical mask and turn to float')

        # ### Order Tensor According To Scores To Be Able To Effectively Loop Over Different Trajectories: ###
        # inliners_logical_mask_float_sum = inliners_logical_mask_float.sum(-1)
        # inliners_logical_mask_float_sum_sorted_values, inliners_logical_mask_float_sum_sorted_indices = torch.sort(inliners_logical_mask_float_sum, 0, descending=True)

        ### Loop And Find Valid Models: ###
        flag_continue_searching_for_models = True
        number_of_models_counter = 0
        model_index = 0
        valid_models_logical_masks_list = []
        while flag_continue_searching_for_models:
            # print(model_index)
            ### get rid of all model score lower then a certain threshold to save on future calculations: ###
            #TODO: if trajectory is NOT valid...no need really to sum over everything again...i can simply move on to the next index
            # gtic()
            inliners_logical_mask_float_sum = inliners_logical_mask_float.sum(-1)
            inliners_logical_mask_float = inliners_logical_mask_float[inliners_logical_mask_float_sum >= model_number_of_inliners_threshold]
            # gtoc('sum up logical mask and keep only models with larger than threshold number of outliers')

            ### Order Tensor According To Scores To Be Able To Effectively Loop Over Different Trajectories: ###
            if inliners_logical_mask_float.numel() > 0:
                inliners_logical_mask_float_sum = inliners_logical_mask_float.sum(-1)  #TODO....no need to sum everything up again....just get rid of invalid indices from the above calculated tensor
                best_current_model_score, best_model_idx = torch.max(inliners_logical_mask_float_sum, 0)
                if best_current_model_score < model_number_of_inliners_threshold:
                    flag_continue_searching_for_models = False
                    continue
                else:
                    ### Get Best Model Score & Parameters: ###
                    lines_origins_best = lines_origins[best_model_idx]
                    lines_directions_best = lines_directions[best_model_idx]

                    ### Get Current Best Model Data: ###
                    # gtic()
                    model_best_parameters = (lines_origins_best, lines_directions_best)
                    inliers_best_logical_mask = inliners_logical_mask[best_model_idx]
                    # inliers_best_logical_mask = (inliners_logical_mask_float>0)[best_model_idx]
                    model_TXY_data_tensor = data[inliers_best_logical_mask]
                    model_t_vec_arange = torch.arange(model_TXY_data_tensor[0, 0], model_TXY_data_tensor[-1, 0], device='cuda:0')
                    #TODO: maybe i can also calculate this for everyone at the same time and it will save time??? - it is ordered ...
                    # ... along the T dimension...so no need for min() and max()...simply first and last elements.
                    # model_t_vec_arange = torch.arange(model_TXY_data_tensor[:, 0].min(), model_TXY_data_tensor[:, 0].max()).to(data.device)
                    # gtoc('get best model data')

                    ### Check If Trajectory Valid: ###
                    # gtic()
                    ### make sure max difference between time stamps, which is also the maximum consecutive frames suspect disappeared, is not too high: ###
                    original_t_vec = model_TXY_data_tensor[:, 0]
                    original_t_vec_unique = torch.unique_consecutive(original_t_vec)  # SLIGHTLY different from t_vec....t_vec is built using np.arange(min_events_T, max_events_T), this is simply number of unique values
                    flag_is_trajectory_valid = original_t_vec_unique.diff().max() <= self.params.DroneTrjDec_max_time_gap_in_trajectory
                    # gtoc('max time gap for best model')
                    # # #TODO: check all trajectories validity at the same time!!!!!!!!!! instead of every time....maybe not....
                    # flag_is_trajectory_valid, output_message = self.Decide_If_Trajectory_Valid_Drone_Torch(model_TXY_data_tensor,
                    #                                                                                            model_t_vec_arange,
                    #                                                                                            lines_directions_best.squeeze(0),
                    #                                                                                            self.number_of_frames,
                    #                                                                                            self.params,
                    #                                                                                            flag_return_string=True)

                    ### If Current Best Model Is Good Enough --> Add It, Otherwise We Are Done: ###
                    if best_current_model_score > model_number_of_inliners_threshold:
                        if flag_is_trajectory_valid:
                            ### Get Current Inlinear Logical Mask: ###
                            valid_trajectory_logical_mask = inliners_logical_mask_float[best_model_idx:best_model_idx+1]
                            valid_models_logical_masks_list.append(valid_trajectory_logical_mask)

                            ### Substract Model Inliners From The Other Models Before Searching For Another One: ###
                            inliners_logical_mask_float = (inliners_logical_mask_float - valid_trajectory_logical_mask).clip(0)


                            ### Add Current Model To Models List: ###
                            self.valid_models_indices_list.append(best_model_idx)
                            self.valid_models_scores_list.append(best_current_model_score)
                            self.valid_models_inliners_logical_mask_list.append(inliers_best_logical_mask.unsqueeze(0))
                            self.valid_models_parameters_list.append(model_best_parameters)
                            self.valid_models_TXY_tensors_list.append(model_TXY_data_tensor)
                            self.model_t_vec_arange_list.append(model_t_vec_arange)
                            self.valid_models_holding_vecs_list.append(model_best_parameters[0])
                            self.valid_models_direction_vecs_list.append(model_best_parameters[1])
                            number_of_models_counter += 1

                            # ### If We Have Too Many Models --> STOP: ###
                            # if number_of_models_counter > max_number_of_models:
                            #     print('Reached max number of models')
                            #     flag_continue_searching_for_models = False
                        else:
                            ### Trajectory is not valid, so to not get stuck on it simply zero it out: ###
                            #TODO: i can calculate in parallel all the things relevant to see if a trajectory is valid on everyone together!!!!!
                            inliners_logical_mask_float[best_model_idx:best_model_idx + 1] = 0
                    else:
                        flag_continue_searching_for_models = False
            else:
                flag_continue_searching_for_models = False
            ### Uptick model_index: ###
            model_index += 1

        ### Get All Points Not Assigned To Trajectories: ###
        indices_remaining = inliners_logical_mask_float.sum(0) > 0
        points_not_assigned_to_lines_yet = data[indices_remaining]
        if len(valid_models_logical_masks_list) == 0:
            final_valid_models_logical_masks_list = None
        else:
            final_valid_models_logical_masks_list = torch.cat(valid_models_logical_masks_list)
        return points_not_assigned_to_lines_yet, final_valid_models_logical_masks_list

    def remove_bad_samples(self, kp1: torch.Tensor) -> Tuple[torch.Tensor]:
        return kp1

    def remove_bad_models(self, models: torch.Tensor) -> torch.Tensor:
        ### Remove Near Identical Line Models: ###

        return models

    def polish_model(self,
                     kp1: torch.Tensor,
                     inliers: torch.Tensor) -> torch.Tensor:
        kp1_inl = kp1[inliers][None]
        model = self.polisher_solver(kp1_inl)
        return model

    def forward(self,
                  input_tensor: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Main forward method to execute the RANSAC algorithm.

        Args:
            input_tensor (torch.Tensor): source image keypoints :math:`(N, 2)`.
            weights (torch.Tensor): optional correspondences weights. Not used now

        Returns:
            - Estimated model, shape of :math:`(1, 3, 3)`.
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
            """

        ### Initialize Parameters: ###
        # TODO: I can use weights when I sum up the number of outliers; I can instead sum outliers * weights
        best_score_total = 0
        total_data_size = len(input_tensor)
        best_model_total = (0, 0)
        inliers_best_total = 0

        ### Initialize Lists: ###
        self.valid_models_indices_list = []
        self.valid_models_scores_list = []
        self.valid_models_parameters_list = []
        self.valid_models_TXY_tensors_list = []
        self.valid_models_holding_vecs_list = []
        self.valid_models_direction_vecs_list = []
        self.valid_models_logical_mask_list = []
        self.valid_models_inliners_logical_mask_list = []
        self.model_t_vec_arange_list = []

        ### Get All Possible Indices Combinations From Current Data And Look For Trajectories: ###
        points_not_assigned_to_lines_yet = self.get_all_possible_indices_combinations(input_tensor, self.minimal_sample_size, total_data_size, self.batch_size, input_tensor.device)

        # local optimization with all inliers for better precision
        return self.valid_models_indices_list, self.valid_models_inliners_logical_mask_list, \
               self.valid_models_scores_list, self.valid_models_TXY_tensors_list, \
               self.valid_models_holding_vecs_list, self.valid_models_direction_vecs_list, \
               points_not_assigned_to_lines_yet, self.model_t_vec_arange_list


    def forward_2(self,
                input_tensor: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Main forward method to execute the RANSAC algorithm.

        Args:
            input_tensor (torch.Tensor): source image keypoints :math:`(N, 2)`.
            weights (torch.Tensor): optional correspondences weights. Not used now

        Returns:
            - Estimated model, shape of :math:`(1, 3, 3)`.
            - The inlier/outlier mask, shape of :math:`(1, N)`, where N is number of input correspondences.
            """
        
        ### Initialize Parameters: ###
        #TODO: i can use weights when i sum up the number of outliers, i can instead sum outliers*weights
        best_score_total = 0
        total_data_size = len(input_tensor)
        best_model_total = (0, 0)
        inliers_best_total = 0

        ### Initialize Lists: ###
        self.valid_models_indices_list = []
        self.valid_models_scores_list = []
        self.valid_models_parameters_list = []
        self.valid_models_TXY_tensors_list = []
        self.valid_models_holding_vecs_list = []
        self.valid_models_direction_vecs_list = []
        self.valid_models_logical_mask_list = []
        self.valid_models_inliners_logical_mask_list = []
        self.model_t_vec_arange_list = []


        ### Loop Over Number Of Iterations: ###
        for i in np.arange(self.max_iter):
            #TODO: rewrite in two versions, this one below and one which simply goes over everything...all trajectories!!!
            ### Lower Points Already Found To Be In Trajectories From Total Points: ###
            if len(self.valid_models_inliners_logical_mask_list) > 0:
                combined_logical_mask = torch.cat(self.valid_models_inliners_logical_mask_list).sum(0) > 0
                input_tensor = input_tensor[combined_logical_mask]
                self.valid_models_inliners_logical_mask_list = []
                #TODO: add some heuristic that if the size of input_tensor doesn't change that it probably means i've gone over ~everything and we're good to go

            ### Get All Possible Indices Combinations From Current Data: ###
            all_possible_combinations = self.get_all_possible_indices_combinations(input_tensor, self.minimal_sample_size, total_data_size, self.batch_size, input_tensor.device)

            ### Get Random Permutation Within The Possibilities: ###
            random_permutations = np.random.permutation(all_possible_combinations.shape[0])
            all_possible_combinations = all_possible_combinations[random_permutations]

            #### Sample From All Possible Combinations: ####
            # ### Simple, Random Sampling: ###
            # idxs = self.sample(self.minimal_sample_size, total_data_size, self.batch_size, input_tensor.device)
            # input_tensor_sampled = input_tensor[idxs]  #notice!! [idx]=[B,2], [input_tensor]=[N,3] and yet [input_tensor[idxs]] = [B,2,3]...beautiful
            # ### Sample Minimal Samples In Batch To Estimate Models: ###
            # input_tensor_sampled = self.sample_smart(input_tensor, self.minimal_sample_size, total_data_size, self.batch_size, input_tensor.device)
            ### Sample Current Batch From all_possible_combinations: ###
            indices_start = i*self.batch_size
            indices_stop = min(indices_start+self.batch_size, len(all_possible_combinations))
            current_indices = all_possible_combinations[indices_start:indices_stop]

            ### Sample Data: ###
            input_tensor_sampled = input_tensor[current_indices]

            ### Remove Bad Samples If Needed: ###
            input_tensor_sampled = self.remove_bad_samples(input_tensor_sampled)

            ### Estimate Models From Sampled Batch: ###
            models = self.estimate_model_from_sampled_data(input_tensor_sampled)  #(origin, direction)

            ### Remove Bad Models If Needed: ###
            models = self.remove_bad_models(models)

            ### If No Models Remain Simply Continue To Next Iteration: ###
            if (models is None) or (len(models) == 0):
                continue

            ### Score the models and select the best one: ###
            # best_model_of_current_batch, inliers, current_model_score
            # valid_models_indices_list, valid_models_inliners_logical_mask_list, valid_models_scores_list
            points_not_assigned_to_lines_yet = self.verify_and_choose_best_model_from_current_batch(input_tensor,
                                                                                                     models,
                                                                                                     self.inlier_threshold,
                                                                                                     self.model_number_of_inliners_threshold,
                                                                                                     self.max_number_of_models)
            print('Done This RANSAC Round')

            # ### Store Best-So-Far Model: ###
            # if current_model_score > best_score_total:
            #     ### Now storing the best model: ###
            #     best_model_total = (best_model_of_current_batch[0].clone(), best_model_of_current_batch[1].clone())
            #     inliers_best_total = inliers.clone()
            #     best_score_total = current_model_score
            #
            #     ### Should we already stop?: ###
            #     new_max_iter = int(self.max_samples_by_conf(int(best_score_total),
            #                                                 total_data_size,
            #                                                 self.minimal_sample_size,
            #                                                 self.confidence))
            #
            #     ### Stop estimation, if the model is very good: ###
            #     if (i + 1) * self.batch_size >= new_max_iter:
            #         break
                    
        # local optimization with all inliers for better precision
        return self.valid_models_indices_list, self.valid_models_inliners_logical_mask_list, \
               self.valid_models_scores_list, self.valid_models_TXY_tensors_list, \
               self.valid_models_holding_vecs_list, self.valid_models_direction_vecs_list, \
               points_not_assigned_to_lines_yet, self.model_t_vec_arange_list

#############################################################################################





#############################################################################################
### Fit 2D: ###

#############################################################################################


#############################################################################################
### Point Cloud Functions ###


#############################################################################################



#############################################################################################
### Contours: ###

#############################################################################################





import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import torch


def np_get_diag_inds(diag, mat_sz):
    # the smallet value for the first index, if we are in the lower part of the matrix choose -diag, else 0
    # the smallet value for the second index, if we are in the lower part of the matrix choose 0, else diag
    ind_start = np.array([max(-diag, 0), max(diag, 0)], dtype=np.uint)

    # how many steps can we take along the diagonal? well we stop once we hit any edge (note i'm not assuming a squre matrix where things are easy)i.e
    total_diag_length = np.min(mat_sz - ind_start)

    return np.vstack((np.arange(ind_start[0], ind_start[0] + total_diag_length, dtype=np.uint),
                      np.arange(ind_start[1], ind_start[1] + total_diag_length, dtype=np.uint)))
    # first_ind_end = min(mat_sz[0], first_ind_start+ ) # last valus of the first index
    # first_ind = np.arange( max(-diag,0), diag + )


def torch_get_diag_inds(diag, mat_sz, cuda_device='cpu'):
    # the smallet value for the first index, if we are in the lower part of the matrix choose -diag, else 0
    # the smallet value for the second index, if we are in the lower part of the matrix choose 0, else diag
    ind_start = np.array([max(-diag, 0), max(diag, 0)], dtype=np.int32)

    # how many steps can we take along the diagonal? well we stop once we hit any edge (note i'm not assuming a squre matrix where things are easy)i.e
    total_diag_length = np.min(mat_sz - ind_start)
    # note that these preliminary calcs are done on the cpu only the inds are build on the gpu

    return torch.vstack(
        (torch.arange(ind_start[0], ind_start[0] + total_diag_length, dtype=torch.long, device=cuda_device),
         torch.arange(ind_start[1], ind_start[1] + total_diag_length, dtype=torch.long, device=cuda_device)))
    # first_ind_end = min(mat_sz[0], first_ind_start+ ) # last v


def make_1d_spline(t, Y, smooth=1, W=None):
    # this routine make a smoothing cubic spline as documented by
    # https://en.wikipedia.org/wiki/Smoothing_spline,
    #  De Boor, C. (2001). A Practical Guide to Splines (Revised Edition)
    # and the article : Reinsch, Christian H (1967). "Smoothing by Spline Functions". Numerische Mathematik. 10 (3): 177–183. doi:10.1007/BF02162161
    # which gives a more detailed derivation of the matrices involved (also see my lyx note for a long full derivtion of this shit)
    # note that there are two similar formulations as detailed in wikipedia. the matrices used in them are almost idential they differ by a p (1-p) factor)
    # i.e we use here we use "De Boor's approach", as detailed in wikipedia

    # the construction of the smoothing spline involves solving a linear system on the coefficients  of the linear system detialed in

    # inputs
    # x: a 1d tensor of length N detailing the "t" position of the data. note that i use t, as let's say in parametric 3d curve this some parameter that parmetrizes the curve
    # i.e we have (x(t), y(y), z(t))
    # a tensor whos last dim hase length N. The spline  and whose other dims we are fitting.
    #
    # y: a 2d tensor of the data to fit along the last dimention i.e the length of the last dimention must be eqaul to size x
    #
    # W: optinal paramter : the weights vector of the same length as x. It is defined in a different way than De Boor,
    #    in my deffintion this the wieght of each point is w^2*(y-a)^2, where y is the value of the data at the point
    #    a is the fitted value of the data, hence the wieght of points increses with w. this is in contrast to De Boor these are 1. / sqrt(w) of my deffintions,
    #    which to me is counter intuetive
    #
    # smooth: the smoothing  paramter. where 0 is no smooth and 1 is compelte smoth (which makes a fit that doesnt take the y data into account at all)
    #         this is differnt from the  original csaps as there 1 is no smothing, which is cunter intutive to me.
    #
    #
    cuda_device = t.device

    pcount = t.shape[0]
    dt = torch.diff(t)
    # this is the (sparse) matrix R as defined in De Boor, C. (2001). A Practical Guide to Splines (Revised Edition)
    R_diag_inds = torch.hstack((torch_get_diag_inds(-1, [pcount - 2, pcount - 2], cuda_device),
                                torch_get_diag_inds(0, [pcount - 2, pcount - 2], cuda_device),
                                torch_get_diag_inds(1, [pcount - 2, pcount - 2], cuda_device)))
    R = torch.sparse_coo_tensor(R_diag_inds, torch.hstack(
        (dt[1:-1], 2 * (dt[1:] + dt[:-1]), dt[1:-1])), (pcount - 2, pcount - 2))
    # we now caclute the matrix Q_transpose (i.e Qt is Q transpose) as defined in De Boor, C. (2001). A Practical Guide to Splines (Revised Edition)
    dt_recip = 1. / dt
    Q_diag_inds = torch.hstack((torch_get_diag_inds(0, [pcount - 2, pcount], cuda_device),
                                torch_get_diag_inds(1, [pcount - 2, pcount], cuda_device),
                                torch_get_diag_inds(2, [pcount - 2, pcount], cuda_device)))
    Qt = torch.sparse_coo_tensor(Q_diag_inds, torch.hstack(
        (dt_recip[:-1], - (dt_recip[1:] + dt_recip[:-1]), dt_recip[1:])), (pcount - 2, pcount))
    if (W is not None):
        QwQ = Qt @ torch.sparse_coo_tensor(torch_get_diag_inds(0,
                                                               [pcount, pcount], cuda_device), W)
        QwQ = QwQ @ (QwQ.t())
    else:
        QwQ = Qt @ (Qt.t())
    p = 1 - smooth
    # we can now start solving for the coffesiants of thr polinomuial f( xi+ z )= ai +bi*z+ci*z^2+di*z^3
    # Solve linear system for the 2nd derivatives
    # Qt.mm(Y.t())
    # c is the vector of 2nd derivaties
    c = torch.empty((Y.shape[0], Y.shape[1]), dtype=Y.dtype, device=cuda_device)

    c[:, 1:-1] = torch.linalg.solve((6. * (1. - p)
                                     * QwQ + p * R).to_dense(), 3 * p * Qt.mm(Y.t())).T
    c[:, 0] = 0
    c[:, -1] = 0

    # see my (AKA yuri) attached lyx notes as to how to get a from c
    if (W is not None):
        a = Y - W * ((((2 * (1 - p) / p) * Qt.t()) @ (c[:, 1:-1].T)).T)
    else:
        a = Y - ((((2 * (1 - p) / p) * Qt.t()) @ (c[:, 1:-1].T)).T)

    # d is simply cacalued by the finite differnce equation for the 2 derivative.
    d = torch.diff(c, 1) / dt
    # b can by cacluted from the eqation  f( xi+ z )= ai +bi*z+ci*z^2+di*z^3, where we demnad that the 1st derivtive is contiues
    b = torch.diff(a) / dt - dt * (c[:, :-1] + dt * d)
    return torch.stack([a[:, :-1], b, c[:, :-1], d])


def eval_1d_spline(t, Coefficients, t_interp, deriv=0):
    interp_inds = torch.searchsorted(t[:-1], t_interp, side="right") - 1
    interp_inds[interp_inds < 0] = 0
    dist_to_grid_point = t_interp - t[interp_inds]

    if (deriv == 0):
        interpulated_values = Coefficients[0, ..., interp_inds] + \
                              dist_to_grid_point[None,] * (Coefficients[1, ..., interp_inds] + \
                                                           dist_to_grid_point[None,] * (
                                                                       Coefficients[2, ..., interp_inds] + \
                                                                       dist_to_grid_point[None,] * Coefficients[
                                                                           3, ..., interp_inds]))

    elif (deriv == 1):
        interpulated_values = (Coefficients[1, ..., interp_inds] + \
                               dist_to_grid_point[None,] * (2 * Coefficients[2, ..., interp_inds] + \
                                                            3 * dist_to_grid_point[None,] * Coefficients[
                                                                3, ..., interp_inds]))
    elif (deriv == 2):
        interpulated_values = 2 * Coefficients[2, ..., interp_inds] + \
                              6 * dist_to_grid_point[None,] * Coefficients[3, ..., interp_inds]

    elif (deriv == 3):
        interpulated_values = 6 * Coefficients[3, ..., interp_inds]

    else:
        interpulated_values = torch.zeros_like(Coefficients[0, ..., interp_inds])

    return interpulated_values


def csaps_smoothn_torch(t_original=None, Y=None, t_final=None, smoothing=0, W=None):
    ### make sure [Y] is at least [B,N]: ###
    if len(Y.shape) == 1:
        Y = Y.unsqueeze(0)

    ### Create simple linear grid vec is None is input: ###
    if t_original is None:
        t_original = torch.arange(Y.shape[1]).to(Y.device)
    if t_final is None:
        t_final = t_original

    ### Get Coefficients: ###
    Coefficients = make_1d_spline(t_original, Y, smooth=smoothing, W=W)

    ### Get Final Smooth Estimate: ###
    Y_smooth = eval_1d_spline(t_original, Coefficients, t_final, deriv=0)

    return Y_smooth



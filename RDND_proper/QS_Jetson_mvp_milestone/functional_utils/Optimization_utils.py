import torch

from RapidBase.import_all import *
import numbers

################################################################################################################
### Simple Model/Function/Objective Optimization: ###
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
        return a * torch.exp(-b*X) + c


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
        return a * torch.exp(-b * (X-x0)**2) + c


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
        return a * torch.exp(-b * (X-x0).abs()) + c


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
        return a * (X**2) * torch.exp(-b * (X-x0).abs()) + c


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
        return a / (X**2) * torch.exp(-b * (torch.log(X)-x0).abs()) + c


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
        return a / X * torch.sqrt((d1*X)**d1 / (d1*X+d2)**(d1+d2))

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
        return a * X * torch.exp(-b * (X-x0)**2) + c

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
        return a * 1/(1 + ((X-x0)/b)**2) + c


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
        return a * torch.sin(b*(X-x0)) + c

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
        return a * torch.sin(b*(X-x0)) * torch.exp(-t*(X-x1).abs()) + c

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
        return a * torch.sin(b*(X-x0)) * torch.exp(-t*(X-x1).abs()) + c*X + d

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

    #(*). x is the entire dataset inputs with shape [N,Mx]. N=number of observations, Mx=number of dimensions
    #(*). y is the entire dataset outputs/noisy-results with shape [N,My], My=number of output dimensions

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
        if time_step % print_frequency == print_frequency-1:
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


import csaps
# import interpol  #TODO: this brings out an error! cannot use this! maybe try linux!
from RapidBase.MISC_REPOS.torchcubicspline.torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
def Testing_model_fit():
    ### Get Random Signal: ###
    x = torch.linspace(-5,5,100)
    y = torch.randn(100).cumsum(0)
    y = y - torch.linspace(0, y[-1].item(), y.size(0))
    # y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # plot_torch(y)

    ### Test Spline Fit Over Above Signal: ###
    #(1). CSAPS:
    x_numpy = x.cpu().numpy()
    y_numpy = y.squeeze().cpu().numpy()
    y_numpy_smooth_spline = csaps.csaps(xdata=x_numpy,
                     ydata=y_numpy,
                     xidata=x_numpy,
                     weights=None,
                     smooth=0.85)
    plt.plot(x_numpy, y_numpy)
    plt.plot(x_numpy, y_numpy_smooth_spline)
    #(2). TorchCubicSplines
    t = x
    x = y
    coeffs = natural_cubic_spline_coeffs(t, x.unsqueeze(-1))
    spline = NaturalCubicSpline(coeffs)
    x_estimate = spline.evaluate(t)
    plot_torch(t,x);
    plot_torch(t+0.5,x_estimate)
    #(3). Using torch's Fold/Unfold Layers:
    #TODO: come up with a general solution for the division factor per index!!!!!!
    kernel_size = 20
    stride_size = 10
    number_of_index_overlaps = (kernel_size//stride_size)
    # y_image = y.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_image = torch.tensor(y_numpy_smooth_spline).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_image.shape
    y_image_unfolded = torch.nn.Unfold(kernel_size=(kernel_size,1), dilation=1, padding=0, stride=(stride_size,1))(y_image)
    #TODO: add polynomial fit for each segment/interval of the y_image_unfolded before folding back (as cheating, instead of proper boundary conditions in the least squares)
    y_image_folded = torch.nn.Fold(y_image.shape[-2:], kernel_size=(kernel_size,1), dilation=1, padding=0, stride=(stride_size,1))(y_image_unfolded)
    plot_torch(y)
    plot_torch(y_image.squeeze())
    plot_torch(y_image_folded.squeeze()/2)
    #(4). Using Tensors Object's unfold method, need to add averaging of "in between" values or perform conditional least squares like i did in my function!!!: ###
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
    #(4). Loess (still, the same shit applies...i need to get a handle on folding/unfolding and boundary conditions):


    ### Get Sine Signal: ###
    x = torch.linspace(-10, 10, 1000)
    y = 1.4 * torch.sin(x) + 0.1*x + 3
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
        bug_patch = torch.matmul(direction.unsqueeze(0), (data-origin).T).squeeze()
        res = (data-origin) - (bug_patch).unsqueeze(-1)*direction

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
        data = (data, )
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
        data = (data, )
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
        #(*) backwards compatibility
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
        #(*). if new model is the best then record it as such:
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





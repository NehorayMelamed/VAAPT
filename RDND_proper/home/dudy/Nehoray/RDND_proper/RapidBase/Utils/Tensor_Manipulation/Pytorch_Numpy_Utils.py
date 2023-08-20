from RapidBase.Basic_Import_Libs import *
import einops
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_get_ND

### Structure Tensor orientation & coherence - a way to find clear structure in an image and might help us when deciding upon semantic map: ###
def get_structure_tensor(inputIMG, w=10):
    # w = 10  # window size is WxW
    # C_Thr = 0.43  # threshold for coherency
    # LowThr = 35  # threshold1 for orientation, it ranges from 0 to 180
    # HighThr = 57  # threshold2 for orientation, it ranges from 0 to 180


    img = inputIMG.astype(np.float32)
    # GST components calculation (start)
    # J =  (J11 J12 J12 J22) - GST
    imgDiffX = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
    imgDiffY = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    imgDiffXY = cv2.multiply(imgDiffX, imgDiffY)

    imgDiffXX = cv2.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv2.multiply(imgDiffY, imgDiffY)
    J11 = cv2.boxFilter(imgDiffXX, cv2.CV_32F, (w, w))
    J22 = cv2.boxFilter(imgDiffYY, cv2.CV_32F, (w, w))
    J12 = cv2.boxFilter(imgDiffXY, cv2.CV_32F, (w, w))
    # GST components calculations (stop)
    # eigenvalue calculation (start)
    # lambda1 = J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2)
    # lambda2 = J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2)
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv2.multiply(tmp2, tmp2)
    tmp3 = cv2.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = tmp1 + tmp4  # biggest eigenvalue
    lambda2 = tmp1 - tmp4  # smallest eigenvalue
    # eigenvalue calculation (stop)
    # Coherency calculation (start)
    # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
    # Coherency is anisotropy degree (consistency of local orientation)
    imgCoherencyOut = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)
    # Coherency calculation (stop)
    # orientation angle calculation (start)
    # tan(2*Alpha) = 2*J12/(J22 - J11)
    # Alpha = 0.5 atan2(2*J12/(J22 - J11))
    imgOrientationOut = cv2.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
    imgOrientationOut = 0.5 * imgOrientationOut
    # orientation angle calculation (stop)
    return imgCoherencyOut, imgOrientationOut


class Gaussian_Blur_Layer(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussian_Blur_Layer, self).__init__()
        if type(kernel_size) is not list and type(kernel_size) is not tuple:
            kernel_size = [kernel_size] * dim
        if type(sigma) is not list and type(sigma) is not tuple:
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        self.padding = []
        for i in kernel_size:
            self.padding.append(i//2)
            self.padding.append(i//2)

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(F.pad(input,self.padding), weight=self.weight, groups=self.groups)


def Gaussian_Blur_Wrapper_Torch(gaussian_blur_layer, input_tensor, number_of_channels_per_frame, frames_dim=0, channels_dim=1):
    #TODO: this is extremely inefficient!!!! never have for loops (without parallelization) in script!!! take care of this!!!!
    #TODO: simply have the gaussian blur work on all channels together or something. in case of wanting several blurs within the batch we can do that
    ### indices pre processing: ###
    frames_dim_channels = input_tensor.shape[frames_dim]
    channels_dim_channels = input_tensor.shape[channels_dim]
    flag_frames_concatenated_along_channels = (frames_dim == channels_dim)
    if frames_dim == channels_dim:  # frames concatenated along channels dim
        number_of_frames = int(channels_dim_channels / number_of_channels_per_frame)
    else:
        number_of_frames = frames_dim_channels
    output_tensor = torch.zeros_like(input_tensor)

    if len(input_tensor.shape) == 4: #[B,C,H,W]
        B,C,H,W = input_tensor.shape

        if flag_frames_concatenated_along_channels:  #[B,C*T,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[:,i*number_of_channels_per_frame*(i+1)*number_of_channels_per_frame,:,:] = \
                    gaussian_blur_layer(input_tensor[:,i*number_of_channels_per_frame*(i+1)*number_of_channels_per_frame,:,:])
        else: #[T,C,H,W] or [B,C,H,W]
            for i in np.arange(number_of_frames):
                output_tensor[i:i+1,:,:,:] = gaussian_blur_layer(input_tensor[i:i+1,:,:,:])

    elif len(input_tensor.shape) == 5: #[B,T,C,H,W]
        B, T, C, H, W = input_tensor.shape

        for i in np.arange(number_of_frames):
            output_tensor[:,i,:,:,:] = gaussian_blur_layer(input_tensor[:,i,:,:,:])

    return output_tensor


### Canny Edge Detection - in case we decide the sobel edge detection isn't good enough: ###
import scipy
from scipy.signal import gaussian
class canny_edge_detection(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False):
        super(canny_edge_detection, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda
        if use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))


        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

        self.gaussian_filter_horizontal = self.gaussian_filter_horizontal.to(device)
        self.gaussian_filter_vertical = self.gaussian_filter_vertical.to(device)
        self.sobel_filter_horizontal = self.sobel_filter_horizontal.to(device)
        self.sobel_filter_vertical = self.sobel_filter_vertical.to(device)
        self.directional_filter = self.directional_filter.to(device)


    def forward(self, img, threshold=None):
        #TODO: when all the functions will be consistent with anvil make sure to transfer this from BW2RGB
        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation = torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)]).to(img.device)
        # if self.use_cuda:
        #     pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD
        if threshold is None:
            threshold = self.threshold

        thresholded = thin_edges.clone()
        thresholded[thin_edges<threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold


class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        pool = lambda x: F.max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            self.return_indices,
        )
        return einops.rearrange(
            pool(einops.rearrange(input, "n c w h -> n (w h) c")),
            "n (w h) c -> n c w h",
            n=n,
            w=w,
            h=h,
        )


def round_to_nearest_half(input_tensor):
    if np.isscalar(input_tensor):
        return round((input_tensor*2)) / 2
    else:
        return (input_tensor*2).round()/2

def first_nonzero_torch(x, axis=0):
    #TODO: make first_zero_torch()
    nonz = (x > 0)
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)

from RapidBase.Utils.Classical_DSP.Convolution_Utils import convn_torch
def peak_detect_pytorch(input_vec, window_size=11, dim=0, flag_use_ratio_threshold=False, ratio_threshold=2.5, flag_plot=False):
    # This is a specific test case for debug, delete if not needed
    # window_size = 11
    # flag_plot = False
    # input_vec = torch.randn(100).cumsum(0)
    # input_vec = input_vec - torch.linspace(0, input_vec[-1].item(), input_vec.size(0))
    # input_vec = input_vec.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # input_vec = input_vec.repeat(1, 4, 2, 3)
    # dim = 0

    # Get Frame To Concat
    if dim < 0:
        dim = len(input_vec.shape) - np.abs(dim)
    to_concat = torch.zeros([1]).to(input_vec.device)
    for dim_index in np.arange(len(input_vec.shape) - 1):
        to_concat = to_concat.unsqueeze(0)
    for dim_index in np.arange(len(input_vec.shape)):
        if dim_index != dim:
            to_concat = torch.cat([to_concat] * input_vec.shape[dim_index], dim_index)

    #  Phase one: find peaks mask by comparing each element to its neighbors over the chosen dimension
    I1 = torch.arange(0, input_vec.shape[dim]-2).to(input_vec.device)
    I2 = torch.arange(1, input_vec.shape[dim]-1).to(input_vec.device)
    I3 = torch.arange(2, input_vec.shape[dim]-0).to(input_vec.device)
    shape_len = len(input_vec.shape)
    shape_characters = 'BTCHW'
    current_shape_characters = shape_characters[-shape_len:]
    current_shape_characters = current_shape_characters[dim]
    I1 = torch_get_ND(I1, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    I2 = torch_get_ND(I2, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    I3 = torch_get_ND(I3, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    shape_vec = torch.tensor(input_vec.shape)
    shape_vec[dim] = 1
    shape_vec = tuple(shape_vec.cpu().numpy())
    I1 = I1.repeat(shape_vec)
    I2 = I2.repeat(shape_vec)
    I3 = I3.repeat(shape_vec)
    a1 = torch.gather(input_vec, dim, I1)  # input_vec[:, :, :-2]
    a2 = torch.gather(input_vec, dim, I2)  # input_vec[:, :, 1:-1]
    a3 = torch.gather(input_vec, dim, I3)  # input_vec[:, :, 2:]
    peak_mask = torch.cat([to_concat, (a1 < a2) & (a3 < a2), to_concat], dim=dim)

    # Phase two: find peaks that are also local maximas
    # This section performs generic rearrangement for the torch max_pool1d_with_indices function.
    # This function requires an up to 3d input and operates max pool over the last dimension only.
    # Therefore, dimensioned are swapped and unified for the operation and later restored
    dims_string = 'b t c h w'
    initial_dims_string = dims_string[-shape_len*2+1:]
    dims_dict = dict(zip(initial_dims_string.split(' '), input_vec.shape))
    new_dims_string = initial_dims_string[0:2*dim] + initial_dims_string[dim*2+1:] + ' ' + initial_dims_string[2*dim:2*dim+1]
    new_dims_string = str.replace(new_dims_string, ' ', '')

    if len(new_dims_string) > 3:
        first_dim_string = new_dims_string[:-2]
        new_first_dim_string = str.replace(first_dim_string, '', ' ')[1:-1]
        new_first_dim_string = f'({new_first_dim_string})'
        rest_dims_string = new_dims_string[-2:]
        new_rest_dim_string = str.replace(rest_dims_string, '', ' ')[1:-1]
        new_dims_string = f'{new_first_dim_string} {new_rest_dim_string}'
    else:
        new_dims_string = str.replace(new_dims_string, '', ' ')[1:-1]

    total_dims_string_1 = initial_dims_string + ' -> ' + new_dims_string
    total_dims_string_2 = new_dims_string + ' -> ' + initial_dims_string
    # rearrange and do max pool
    input_vec2 = einops.rearrange(input_vec, total_dims_string_1)
    input_vec2_maxpool_values, input_vec2_maxpool_indices = torch.nn.functional.max_pool1d_with_indices(input_vec2, window_size, 1, padding=window_size // 2)
    dims_to_separate_dict = {k: v for k, v in dims_dict.items() if k in first_dim_string}
    input_vec2_maxpool_indices = einops.rearrange(input_vec2_maxpool_indices, total_dims_string_2, **dims_to_separate_dict)

    # ### Get Window Mean: ###
    # ratio_threshold = 3
    # window_mean = convn_torch(input_vec2, torch.ones(window_size)/window_size, dim=-1)
    # max_value_enough_above_value_logical_mask = (input_vec2_maxpool_values/window_mean > ratio_threshold)
    # max_value_enough_above_value_logical_mask = einops.rearrange(max_value_enough_above_value_logical_mask, total_dims_string_2, **dims_to_separate_dict)

    ### Get Window Median: ###
    if flag_use_ratio_threshold:
        input_vec2_unfolded = input_vec2.unfold(dimension=-1, size=window_size, step=1)
        input_vec2_unfolded_median = input_vec2_unfolded.median(-1)[0]
        to_concat_median1 = torch.ones(window_size//2).to(input_vec.device).unsqueeze(0).unsqueeze(0)
        to_concat_median2 = torch.ones(window_size//2).to(input_vec.device).unsqueeze(0).unsqueeze(0)
        input_vec2_unfolded_median = torch.cat((to_concat_median1, input_vec2_unfolded_median, to_concat_median2), -1)
        max_value_enough_above_value_logical_mask = (input_vec2_maxpool_values / input_vec2_unfolded_median > ratio_threshold)
        max_value_enough_above_value_logical_mask = einops.rearrange(max_value_enough_above_value_logical_mask, total_dims_string_2, **dims_to_separate_dict)
    
    # keep only peaks, each contains the value of the local maximum in the surrounding window
    filtered_peaks_with_window_maxima_value = input_vec2_maxpool_indices * peak_mask

    ### Only Keep Those Peaks Which Are Sufficiently Above Window Mean: ###
    if flag_use_ratio_threshold:
        filtered_peaks_with_window_maxima_value = filtered_peaks_with_window_maxima_value * max_value_enough_above_value_logical_mask

    # make a tensor of indices over the wanted dimension, repeated in the other dimensions
    arange_in_correct_dim_tensor = torch.arange(0, input_vec.shape[dim])
    arange_in_correct_dim_tensor = torch_get_ND(arange_in_correct_dim_tensor, input_dims=current_shape_characters, number_of_dims=len(input_vec.shape))
    dims_to_repeat = input_vec.shape
    dims_to_repeat = torch.tensor(dims_to_repeat)
    dims_to_repeat[dim] = 1
    arange_in_correct_dim_tensor = arange_in_correct_dim_tensor.repeat(*dims_to_repeat).to(input_vec.device)
    # keep only the picks that are also maximal in their local window
    # Assumption: no picks in the 0 index (not possible due to pick definition)
    # to avoid true values at the 0 indices, change the 0 values in filtered_picks_with_window_maxima_value to -1
    filtered_peaks_with_window_maxima_value = torch.where((filtered_peaks_with_window_maxima_value == 0),
                                                          torch.ones_like(filtered_peaks_with_window_maxima_value) * (-1),
                                                          filtered_peaks_with_window_maxima_value)
    # get a mask of the picks that are also window maximas
    maxima_peaks = (filtered_peaks_with_window_maxima_value == arange_in_correct_dim_tensor)

    # Several plot options for different dimensionalities
    # Plot Results: this is a specific case of chw tensor, chosen dim w
    # i = 1
    # j = 1
    # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[i, j])[maxima_peaks[i, j]]
    # plt.figure()
    # plt.plot(input_vec[i, j].numpy())
    # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[i, j, maxima_peaks_to_plot].numpy(), '.')
    # plt.title('local maxima peaks')

    # Plot Results: this is a specific case of tchw tensor, chosen dim t
    # i = 1
    # j = 1
    # k = 1
    # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[maxima_peaks[:, i, j, k]])[:, i, j, k]
    # plt.figure()
    # plt.plot(input_vec[:, i, j, k].numpy())
    # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[maxima_peaks_to_plot, i, j, k].numpy(), '.')
    # plt.title('local maxima peaks')

    # Plot Results: this is a specific case of btchw tensor, chosen dim t
    # i = 1
    # j = 1
    # k = 1
    # l = 1
    # maxima_peaks_to_plot = (arange_in_correct_dim_tensor[i])[maxima_peaks[i, :, j, k, l]][:, j, k, l]
    # plt.figure()
    # plt.plot(input_vec[i, :, j, k, l].numpy())
    # plt.plot(maxima_peaks_to_plot.numpy(), input_vec[i, maxima_peaks_to_plot, j, k, l].numpy(), '.')
    # plt.title('local maxima peaks')

    return maxima_peaks, arange_in_correct_dim_tensor


class Morphology(nn.Module):
    '''
    Base class for morpholigical operators
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure.
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)

        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1)  # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x  # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x  # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError

        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False)  # (B, Cout, L)
        else:
            x = torch.logsumexp(x * self.beta, dim=2, keepdim=False) / self.beta  # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x


class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')


class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

# Get needed padding to keep image size the same after a convolution layer with a certain kernel size and dilation (TODO: what about strid?1!?!)
def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def torch_get_where_condition_holds(condition):
    return (condition).nonzero(as_tuple=False)  #reutrns a tensor of size (number_of_matches X tensor_number_of_dimensions)

def torch_get_where_condition_holds_as_list_of_tuples(condition):
    matches_list_per_dimension = (condition).nonzero(as_tuple=False)  #reutrns a tensor of size (number_of_matches X tensor_number_of_dimensions)
    matches_list_of_tuples = []
    number_of_dimensions = len(matches_list_per_dimension)
    for match_index in matches_list_per_dimension[0].shape[0]:
        current_list = []
        for dim_index in number_of_dimensions:
            current_list.append(matches_list_per_dimension[dim_index][match_index])
        current_list = tuple(current_list)
        matches_list_of_tuples.append(current_list)

###############################################kornia~=0.5.11#########################################################################################################################################################################################################################################
####################
# Auxiliaries:
####################
def get_circle_kernel(kernel_size, radius):
    x = np.arange(-kernel_size / 2, kernel_size / 2, 1)
    [X, Y] = np.meshgrid(x, x)

    circular_kernel = (X ** 2 + Y ** 2) < radius ** 2
    return circular_kernel.astype(np.uint8)


def get_gaussian_kernel(kernel_size, gaussian_sigma):
    x = np.arange(-kernel_size / 2, kernel_size / 2, 1)
    [X, Y] = np.meshgrid(x, x)
    gaussian_kernel = np.exp(- ((X) ** 2 + (Y) ** 2) / gaussian_sigma ** 2)
    gaussian_kernel = gaussian_kernel / np.sqrt(sum(sum(abs(gaussian_kernel) ** 2)))
    return gaussian_kernel


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np =  make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def get_network_output_size_given_input_size(network, input_size):
    f = network.forward(torch.Tensor(1,*input_size))
    return int(np.prod(f.size()[1:]))


########################################################################################################################################################################################################################################################################################











########################################################################################################################################################################################################################################################################################
####################
# Save & Load Models/Checkpoint:
####################
#(1). State Dictionary
def save_network_state_dict(network, save_full_filename):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    #transfer parameters to cpu so that we will be able to save them to disk:
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_full_filename)


def load_state_dict_to_network_from_path(network, load_path,  flag_strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    network.load_state_dict(torch.load(load_path), strict=flag_strict) #this changes


def load_state_dict_to_network_robust(current_network, pretrained_state_dictionary):
    current_network_state_dictionary = current_network.state_dict()
    #(1). filter out unnecessary keys:
    pretrained_state_dictionary_filtered  = {k:v for k,v in pretrained_state_dictionary.items() if k in current_network_state_dictionary.keys()}
    #(2). overwrite entries in existing state dictionary:
    current_network_state_dictionary.update(pretrained_state_dictionary_filtered)
    #(3). load the new state dictionary to current_model:
    current_network.load_state_dict(current_network_state_dictionary)



#(2). Network:
def save_network_simple(network, save_filename):
    if isinstance(network, nn.DataParallel):
        network = network.module
    torch.save(network, save_filename)

def load_network_simple(load_path):
    return torch.load(load_path)

def load_network_robust(current_network, load_path):
    loaded_file = torch.load(load_path)
    if type(loaded_file) == collections.OrderedDict:
        load_state_dict_to_network_robust(current_network, loaded_file)
        return current_network
    else:
        return loaded_file


# #Example Of Use:
# folder_name = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS'
# complete_model_save_file_name = 'complete_model.pt'
# state_dictionary_save_file_name = 'state_dictionary.pth'
#
# complete_model_save_full_file_name = os.path.join(folder_name,complete_model_save_file_name)
# state_dictionary_save_full_file_name = os.path.join(folder_name,state_dictionary_save_file_name)
#
# current_model = torchvision.models.resnet18(True)
# current_model_state_dictionary = current_model.state_dict()
#
# torch.save(current_model,complete_model_save_full_file_name)
# torch.save(current_model_state_dictionary, state_dictionary_save_full_file_name)
#
# loaded_model_from_complete_model = load_network_robust(None, complete_model_save_full_file_name)
# loaded_model_from_state_dictionary = load_network_robust(current_model,state_dictionary_save_full_file_name)
# loaded_model_from_complete_model == current_model
# loaded_model_from_state_dictionary == current_model
#
# bla = torch.load(complete_model_save_full_file_name)
# bla == current_model


#(3. General/Custom Checkpoint:
def save_checkpoint_specific(network, optimizer=None, OPT=None, iteration_number=None, flag_is_best_model=False, full_filename='checkpoint.pth'):
    checkpoint_dictionary = {'model': network, 'state_dict': network.state_dict(), 'optimizer':optimizer, 'OPT':OPT, 'iteration': iteration_number}
    torch.save(checkpoint_dictionary, full_filename)

    if flag_is_best_model:
        shutil.copyfile(filename, 'model_best.pth.tar') #shutil.copyfile


def save_checkpoint_general(network, full_filename = 'checkpoint.pth', flag_save_network_state_dict_or_both='both', flag_is_best_model=False, **kwargs):
    if flag_save_network_state_dict_or_both == 'network':
        checkpoint_dictionary = {'model': network}
    elif flag_save_network_state_dict_or_both == 'state_dict':
        checkpoint_dictionary = {'state_dict': network.state_dict()}
    elif flag_save_network_state_dict_or_both == 'both':
        checkpoint_dictionary = {'model': network, 'state_dict': network.state_dict}

    for key in kwargs.keys():
        checkpoint_dictionary[key] = kwargs[key]

    torch.save(checkpoint_dictionary, full_filename)

    if flag_is_best_model:
        shutil.copyfile(full_filename, 'model_best.pth.tar')



def load_checkpoint(checkpoint_filename):
    if os.path.isfile(checkpoint_filename):
        print(" --> loading checkpoint '{}'".format(checkpoint_filename))
        checkpoint = torch.load(checkpoint_filename)
    return checkpoint



# #Use Case of the above save/load functions:
# #Parameters:
# blabla = torchvision.models.vgg19(pretrained=True)
# bla = torchvision.models.vgg11()
# base_path = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS'
# model_name = 'bla_model.pt' #what's the difference between .pt & .pth  ????
# full_filename = os.path.join(base_path,model_name)
#
# ### Within Script: ###
# #Save & Load Entire Model within script:
# torch.save(bla, full_filename)
# bla2 = torch.load(full_filename)
#
# #Save & Load State Dictionary within script:
# bla_state_dictionary = bla.state_dict()
# for key, param in bla_state_dictionary.items():
#     bla_state_dictionary[key] = param.cpu()
# torch.save(bla_state_dictionary, full_model_name)
# bla2 = bla.load_state_dict(full_filename, strict=True) #State-Dict
#
# ### Using Function Wrappers: ###
# #Save & Load using checkpoint:
# save_checkpoint(bla, filename=full_filename)
# network_model, optimizer, checkpoint_epoch = load_checkpoint(full_filename)
# #Save & Load using network state dictionaries:
# save_network_state_dict(network, full_filename)
# bla.load_state_dict(torch.load(full_filename), strict=True)
# bla = load_state_dict_to_network(full_filename, bla) #Doesn't Work
# #Save & Load using entire network save&load:
# save_network_simple(bla,full_filename)
# bla2 = load_network_simple(full_filename)


# model = torch.load('./model_resnet50.pth.tar')
# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
# )
# preprocess = transforms.Compose([
#     transforms.Scale(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize
# ])
#
# img = Image.open(IMG_URL)
# img_tensor = preprocess(img)
# img_tensor.unsqueeze_(0)
# output = model(Variable(img_tensor))
########################################################################################################################################################################################################################################################################################






###################################################################################################
#### Revisit BatchNorm After Model Learning: ###
import torch
import torch.nn as nn


## A module dedicated to computing the true population statistics
## after the training is done following the original Batch norm paper

# Example of usage:

# Note: you might want to traverse the dataset a couple of times to get
# a better estimate of the population statistics
# Make sure your trainloader has shuffle=True and drop_last=True

# net.apply(adjust_bn_layers_to_compute_population_stats)
# for i in range(10):
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             _ = net(inputs.cuda())
# net.apply(restore_original_settings_of_bn_layers)


# Why this works --
# https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm1d
# if you set momentum property in batchnorm layer, it will
# compute cumulutive average or just simple average of observed
# values

def adjust_bn_layers_to_compute_population_stats_old(module):
    if isinstance(module, nn.BatchNorm2d):
        # Removing the stats computed using exponential running average
        # and resetting count
        module.reset_running_stats()

        # Doing this so that we can restore it later
        module._old_momentum = module.momentum

        # Setting the momentum to none makes the statistics computation go to cumulative moving average mode....i don't know if that's the right thing to do
        module.momentum = None

        #Note: now the batchnorm2d acts the following way: x_new = (1-momentum)*x_estimated_static + momentum*x_observed_value
        #the suggestion most people have is to set the momentum to something higher than the original 0.1, so i'm setting it to 0.5, which supposedly makes the network prefer curent value....so i don't really understand why that's the answer....
        #it would make sense if you just want to make the situation better when evaluating because it would be closer the .train() mode but if you want to collect statistics?
        # module.momentum = 0.5

        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats

        module.training = True
        module.track_running_stats = True




def adjust_bn_layers_to_compute_population_stats(module):
    if isinstance(module, nn.BatchNorm2d):
        # Removing the stats computed using exponential running average
        # and resetting count
        module.reset_running_stats()

        # Doing this so that we can restore it later
        module._old_momentum = module.momentum

        #Note: now the batchnorm2d acts the following way: x_new = (1-momentum)*x_estimated_static + momentum*x_observed_value
        #the suggestion most people have is to set the momentum to something higher than the original 0.1, so i'm setting it to 0.5, which supposedly makes the network prefer curent value....so i don't really understand why that's the answer....
        #it would make sense if you just want to make the situation better when evaluating because it would be closer the .train() mode but if you want to collect statistics?
        module.momentum = 0.5

        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats

        module.training = True # net.eval() makes this False, net.train() makes this True
        module.track_running_stats = True #net.eval() and net.train() doesn't change this


def restore_original_settings_of_bn_layers(module):
    if isinstance(module, nn.BatchNorm2d):
        # Restoring old settings
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats


def set_bn_layers_momentum(value=0.5):
    #TODO: understand how i can use this and pass in variables...as it is now the vanilla form of calling it doesn't work: network.apply(set_bn_layers_momentum(0.5)) - doesn't work!
    def set_bn_layers_with_specific_value(module):
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = value
    return set_bn_layers_with_specific_value
##########################################################################################################################################






########################################################################################################################################################################################################################################################################################
####################
# Get Network Layers:
####################
# Get Network Layers:
def get_network_model_up_to_block_indices(network, block_indices):
    1

def get_network_model_up_to_layer_name(network, layer_name):
    1

def get_network_layer_using_block_indices(network, block_indices):
    1

def get_network_layer_using_layer_name(network, layer_name):
    1

def get_network_model_up_to_layer_index(network, layer_index):
    #TODO: change this to accomodate block_indices and .features attributes (if hasattr(network, features) -> blablabla), if block_indices!=None --> get_network_layer_using_block_hierarcy(model,block_indices)
    # return nn.Sequential(*list(network.children())[layer_index])
    return nn.Sequential(*get_network_layers_list_flat(network)[:layer_index]) #if layer_index is larger than number of layers no error pops up - we simply return the entire model

def get_network_model_up_to_layer_index_from_end(network, layer_index_from_end):
    return nn.Sequential(*list(network.children())[:-layer_index_from_end])

def get_network_layer(network, layer_index):
    return get_network_layers_list_flat(network)[layer_index]

def flatten_list(input_list):
    # return [item for sublist in input_list for item in sublist]
    # return list(itertools.chain.from_iterable(input_list))
    total_list = []
    def get_merged_list(current_list):
        if type(current_list) == list:
            for element in current_list:
                if type(element) == list:
                    get_merged_list(element)
                else:
                    total_list.append(element)

    get_merged_list(input_list)
    return total_list

def get_network_layers_list(network):
    return list(network.children())

def get_network_layers_list_flat(network,verbose=0):
    # return flatten_list(list(network.children()))
    all_layers = []
    if list(network.children()) == []:
        all_layers = [network]

    def get_layers(network):
        for layer in network.children():
            if hasattr(layer, 'children'):  # if sequential layer, apply recursively to layers in sequential layer
                get_layers(layer)
            else:
                if verbose: print('no children')
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)
                if verbose: print(type(layer))

    get_layers(network)
    if verbose:
        for layer in all_layers:
            print(layer)
        len(all_layers)
    return all_layers



# #TODO: still doesn't work
# def get_network_layers_list_flat_test(network,verbose=0):
#     # return flatten_list(list(network.children()))
#     all_layers = []
#     def get_layers(network):
#             try:  # if sequential layer, apply recursively to layers in sequential layer
#                 get_layers(network.children())
#             except:  # if leaf node, add it to list
#                 all_layers.append(network)
#     get_layers(network)
#     return all_layers


def get_number_of_conv_layers(network):
    counter = 0
    for current_module in get_network_layers_list_flat(network):
        if isinstance(current_module, nn.Conv2d):
            counter += 1
    return counter

def get_network_named_layers_list_flat(network, verbose=0):
    all_layers = []
    all_layers_names = [] #names hierarchies
    def get_layers(network, pre_string=''):
        for layer in network.named_children():
            if hasattr(layer[1], 'named_children'):  # if sequential layer, apply recursively to layers in sequential layer
                get_layers(layer[1], pre_string + '/' + layer[0])
            else:
                if verbose: print('no children')
            if list(layer[1].named_children()) == []:  # if leaf node, add it to list
                all_layers.append(layer[1])
                all_layers_names.append(pre_string + '/' + layer[0])
                if verbose: print(type(layer))

    get_layers(network)
    if verbose:
        for layer in all_layers:
            print(layer)
        len(all_layers)
    for i,layer_name in enumerate(all_layers_names):
        all_layers_names[i] = layer_name[1:]
    return all_layers, all_layers_names



def get_network_variable_size_through_the_layers(network, input_variable):
    # Assuming the flow through the model is entirely known through the model.children() property.
    # Also probably assuming the network is built as convolutional/features part and classifier head.
    x = input_variable
    for i, layer in enumerate(get_network_layers_list_flat(pretrained_model)):
        print('index ' + str(i) + ',input to layer shape: ' + str(x.shape) + ', ' + str(layer))
        if 'Linear' in str(layer):
            x = x.view(x.size(0), -1)
        x = layer(x)
    print('index ' + str(i) + ',final shape: ' + str(x.shape))
########################################################################################################################################################################################################################################################################################






########################################################################################################################################################################################################################################################################################
####################
# Network Summary, Description, Graph:
####################
# Network String Description:
def get_network_description(network):
    if isinstance(network, nn.DataParallel):
        network = network.module
    network_string_description = str(network)
    number_of_parameters = sum(map(lambda x: x.numel(), network.parameters()))
    return network_string_description, number_of_parameters

def print_network_description(network):
    network_string_description, number_of_parameters = get_network_description(network)
    print(network_string_description)
    print('\n')
    print('Number of parameters in G: {:,d}'.format(number_of_parameters))


def data_from_dataloader_to_GPU(data_from_dataloader, device):
    for k, v in data_from_dataloader.items():
        data_from_dataloader[k] = data_from_dataloader[k].to(device)
    return data_from_dataloader

from operator import itemgetter
import operator
def get_sublist_from_list(input_list, indices):
    f = operator.itemgetter(*indices)
    return f(input_list)
    # list(itemgetter(*idx)(input_list))
    # map(input_list.__getitem__, idx)
    # list(input_list[_] for _ in idx)  # a generator expression passed to a list constructor.
    # map(lambda _: input_list[_], idx)  # using 'map'
    # for i, x in enumerate(input_list) if i in idx]
    # filter(lambda x: l.index(x) in idx, input_list)


def get_model_gradient_norm(model):
    total_norm = 0
    # average_value = 0
    # max_value = 0
    # counter = 0
    # for p in model.parameters():
    #     param_norm = p.grad.data.norm(2)
    #     total_norm += param_norm.item() ** 2
    #     average_value += p.grad.data.mean()
    #     max_value = max(max_value, p.grad.data.max())
    #     counter += 1
    # total_norm = total_norm ** (1. / 2)
    # average_value = average_value / counter

    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        total_norm = p.grad.data.norm(2).item()

    return total_norm

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict


#one needs to add the dot.exe executable to environment variables. to do that you need to locate the graphviz folder and add it:
import os
os.environ["PATH"] += os.pathsep + 'C:/Users\dkarl\AppData\Local\conda\conda\envs\dudy_test\Library/bin/graphviz/'
#TODO: try other implementations of make_dot which can make things prettier and easier to read
# Print Network Graph to PDF file which is automatically opened:
# from RapidBase.Utils.IO.Path_and_Reading_utils import path_get_current_working_directory
# def print_network_graph_to_pdf(network, input_size, filename='Network_Graph', directory=path_get_current_working_directory()):
#     inputs = torch.randn(1,*input_size)
#     outputs = network(Variable(inputs))
#     network_graph = make_dot(outputs, params=dict(network.named_parameters()))
#     network_graph.view(filename,directory)


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
# input_size = (1,28,28)
# network = Net()
#
# input_size = (3,224,224)
# network = torchvision.models.vgg19_bn()

#Print Network Summary (KERAS style):
def print_network_summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary
# print_network_summary(torchvision.models.resnet18(True))
########################################################################################################################################################################################################################################################################################










########################################################################################################################################################################################################################################################################################
####################
# Gradients:
####################
def enable_gradients(network):
    for p in network.parameters():
        p.requires_grad = True
def unfreeze_gradients(network):
    for p in network.parameters():
        p.requires_grad = True
    return network
def disable_gradients(network):
    for p in network.parameters():
        p.requires_grad = False
def freeze_gradients(network):
    for p in network.parameters():
        p.requires_grad = False
    return network

def freeze_gradients_up_to_layer_index(network, layer_index):
    counter = 0
    for child in network.children():
        if counter < layer_index:
            for param in child.parameters():
                param.requires_grad = False
        counter += 1

def get_Network_Layers_Devices(Network):
    Generator_Network_layers_list = get_network_layers_list_flat(Generator_Network)
    for layer in Generator_Network_layers_list:
        if len(list(layer.parameters()))>0:
            print(layer.parameters().__next__().device)




def plot_grad_flow(Network, full_path):
    #Use this function AFTER doing a .backward() which updates the .grad property of the layers
    average_abs_gradients_list = []
    layers_list = []
    for parameter_name, parameter in Network.named_parameters():
        if (parameter.requires_grad) and ("bias" not in parameter_name) and ('bnrm' not in parameter_name) and type(parameter.grad)!=type(None): #we usually don't really care about the bias term
            layers_list.append(parameter_name.split('.weight'))
            average_abs_gradients_list.append(parameter.grad.abs().mean())

    ### Devide layers into sizable chunks to present: ###
    layers_per_plot = 5
    number_of_plots = np.ceil(len(average_abs_gradients_list)/layers_per_plot)
    for plot_index in np.arange(number_of_plots):
        plot_index = int(plot_index)
        figure()
        start_index = plot_index * layers_per_plot
        stop_index = min((plot_index+1)*layers_per_plot, len(average_abs_gradients_list))
        number_of_lines = stop_index-start_index+1
        plt.plot(average_abs_gradients_list[start_index:stop_index], alpha=0.3, color="b")
        # plt.scatter(arange(len(average_abs_gradients_list)),average_abs_gradients_list)
        plt.hlines(0, 0, number_of_lines, linewidth=1, color="k" )
        # plt.xticks(range(0,len(average_abs_gradients_list), 1), layers_list, rotation="horizontal")
        plt.xlim(xmin=0, xmax=number_of_lines-1)
        plt.ylim(ymin=0, ymax=max(average_abs_gradients_list).item())
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        x_vec = np.arange(0,number_of_lines,1)
        for i,layer_name in enumerate(layers_list[start_index:stop_index]):
            matplotlib.pylab.text(x_vec[i], average_abs_gradients_list[i], layers_list[start_index+i], fontdict=None, withdash=False, rotation='vertical', verticalalignment ='bottom')
        plt.title("Gradient flow")
        plt.grid(True)
        plt.savefig(full_path + '_' + str(plot_index) + '_GradFlow.png')
        1
# #Use Example:
# my_loss.backward()
# plot_grad_flow(network)


# #A way of getting middle layer output from the forward hook (without the need for a second, seperate forward pass):
# network = torchvision.models.vgg19()
# inputs = torch.randn(1,3,244,244)
# layer_index = 5
# global_outputs_hooked = []
# def hook(module, input, output):
#     global_outputs_hooked.append(output)
# get_network_layer(network,layer_index).register_forward_hook(hook)
# output1 = network(inputs)
# output2 = network(inputs)
# print(outputs)
####################################################################################################################################################################################################################################################################################################################################################################################################################################








#########################################################################################################################################
##############            CUDA Auxiliaries:           ##################
#########################################################################################################################################
def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__,
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
												   type(obj.data).__name__,
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "",
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass
	print("Total size:", total_size)
########################################################################################################################################









#########################################################################################################################################
##############            WEIGHT INITIALIZATIONS:           ##################
#########################################################################################################################################
########################################
# Weight Initializations:
########################################
def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))






#########################################################################################################################################
##############            NETWORK VISUALIZAITONS:           ##################
#########################################################################################################################################
#(1). Simple Layer Output Visualizations:
def plot_layer_outputs(network, input_variable, layer_index, filter_indices=None, crop_percent=0.8, flag_colorbar=1,
                       delta_subplots_spacing=0.1):
    #TODO: Switch to hook+.forward mode!!!!
    network_model_up_to_chosen_layer = get_network_model_up_to_layer_index(network, layer_index)
    output = network_model_up_to_chosen_layer(input_variable)
    output_numpy = output.numpy() #Be Careful - can't the .detach() cause problems?
    output_numpy = output_numpy[0]  # Assuming i only input 1 input variable and this squeezes the batch_size dimension which would be exactly 1
    #The torch output is of form: [N,C,H,W] so i don't need to transpose and i can simply send it to plot_multiple_images as is, and it gets [C,H,W] and so it plots each channel as grayscale
    if filter_indices == None:
        filter_indices = arange(length(output_numpy))
    filter_indices = list(filter_indices)
    super_title = 'Layer ' + str(layer_index) + ' Outputs, Layer Name: ' + str(network_model_up_to_chosen_layer[-1])
    titles_string_list = str(list(filter_indices))[1:-1].split(' ')
    plot_multiple_images(output_numpy, filter_indices, crop_percent, delta_subplots_spacing, flag_colorbar, super_title, titles_string_list)



# #Examples Of Use:
# image_directory = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images'
# image_name = 'cat_dog.png'
# full_image_path = os.path.join(image_directory,image_name)
# input_image = read_image_cv2(full_image_path)
# input_image = preprocess_image(input_image, resize_im=True)
# input_image.shape
#
# pretrained_model = torchvision.models.vgg16(True)
# layer_index = 7
# pretrained_model_up_to_chosen_layer  = get_network_model_up_to_layer_index(pretrained_model, layer_index)
# pretrained_model_chosen_layer_output = pretrained_model_up_to_chosen_layer(input_image)
# pretrained_model_chosen_layer_output.shape
# bla = pretrained_model_chosen_layer_output.detach().numpy()
# bla = bla[0]
# bla.shape
# plot_images_mat(bla,list(arange(10)))
# plot_layer_outputs(pretrained_model, input_image, layer_index=7, filter_indices=arange(10), delta_subplots_spacing=0.05)




def plot_network_conv_filters(network, number_of_conv_layers):
    1








#########################################################################################################################################
##############            NETWORK ARCHITECTURE AUGMENTATIONS:           ##################
#########################################################################################################################################
#TODO LIST:
#(1). Add deformable convolutions instead of current convolutions WHERE WANTED
#(2). Change from BatchNorm to SynchronizedBatchNorm
#(3). Change from BatchNorm to InstanceNorm / GroupNorm
#(4). Change dilation rate
#(5). Change from full 3x3 (for example) convolution to seperable convolution
#(6). Insert Dropout2d Layers where wanted
#(7). Make the "evil twin" network of a CNN -> a DCNN (Deconvolutional Neural Network)




















###########################################################################################################################################################################################################################################################
def get_Network_Layers_Devices(Network):
    Generator_Network_layers_list = get_network_layers_list_flat(Generator_Network)
    for layer in Generator_Network_layers_list:
        if len(list(layer.parameters()))>0:
            print(layer.parameters().__next__().device)


def Generator_and_Discriminator_to_GPU(Generator_Network, Discriminator_Network, netF):
    global Generator_device, netF_device, discriminator_device
    Generator_Network = Generator_Network.to(Generator_device)
    if netF:
        netF = netF.to(netF_device)
    Discriminator_Network = Discriminator_Network.to(discriminator_device)
    return Generator_Network, Discriminator_Network


def Generator_to_GPU(Generator_Network,netF=None,device=None,netF_device=None):
    current_device = device
    Generator_Network = Generator_Network.to(current_device)
    if netF:
        netF = netF.to(netF_device)
    return Generator_Network

def Discriminator_to_GPU(Discriminator_Network, device):
    Discriminator_Network = Discriminator_Network.to(device)
    return Discriminator_Network

def Generator_to_train_mode(Generator_Network):
    Generator_Network.train()
    Generator_Network = unfreeze_gradients(Generator_Network)
    return Generator_Network

def Generator_to_eval_mode(Generator_Network):
    Generator_Network.eval()
    Generator_Network = freeze_gradients(Generator_Network)
    return Generator_Network

def Discriminator_to_train_mode(Discriminator_Network):
    Discriminator_Network.train()
    Discriminator_Network = unfreeze_gradients(Discriminator_Network)
    return Discriminator_Network

def Discriminator_to_eval_mode(Discriminator_Network):
    Discriminator_Network.eval()
    Discriminator_Network = freeze_gradients(Discriminator_Network)
    return Discriminator_Network

# def load_Generator_from_checkpoint(models_folder='C:/Users\dkarl\PycharmProjects\dudykarl\TNR\Model New Checkpoints',
#                                    load_Generator_filename='Generator_Network_TEST1_Step1020.pth',
#                                    Generator_Network=None):
#     # Get filename and postfix (.pt / .pth):
#     if len(load_Generator_filename.split('.'))==1: #file doesn't have a suffix (.pth or .pt)
#         load_Generator_filename += '.pth'
#         filename_Generator_type = '.pth'
#     else:
#         filename_Generator_type = '.' + load_Generator_filename.split('.')[-1]
#
#     # Get Generator Full Path:
#     # path_Generator = os.path.join(models_folder , str(load_Generator_filename))
#     path_Generator = os.path.join(models_folder , load_Generator_filename.split('_TEST1')[0])
#     path_Generator = os.path.join(path_Generator , str(load_Generator_filename) )
#
#
#     # If We Inserted a Network Then Load It There:
#     if not Generator_Network:
#         Generator_Network = get_Original_Generator()
#     # Otherwise, Load New Generator:
#     if path_Generator.split('.')[1] == 'pth':
#         Generator_Network.load_state_dict(torch.load(path_Generator))
#     elif path_Generator.split('.')[1] == 'pt':
#         Generator_Network = torch.load(path_Generator)
#     return Generator_Network


# def save_Generator_parts_to_checkpoint(Generator_Network,folder,basic_filename,flag_save_dict_or_whole='dict'):
#     # Save Generator:
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#     basic_filename = basic_filename.split('.')[0] #in case i pass in filename.pth for generality and easiness sake
#     if flag_save_dict_or_whole == 'dict' or 'both':
#         path_Generator = os.path.join(folder,str(basic_filename))  + '.pth'
#         #to get state_dict() it must be on cpu...so to do things straightforwardly - pass network to cpu and then back to gpu:
#         Generator_Network = Generator_to_CPU(Generator_Network)
#         torch.save(Generator_Network.state_dict(), path_Generator)
#         Generator_Network = Generator_to_GPU(Generator_Network)
#     if flag_save_dict_or_whole == 'whole' or 'both':
#         path_Generator = os.path.join(folder,str(basic_filename))  + '.pt'
#         Geneator_Network = Generator_to_CPU(Generator_Network)
#         torch.save(Generator_Network, path_Generator)
#         Generator_Network = Generator_to_GPU(Generator_Network)


def load_Discriminator_from_checkpoint(folder,filename,Discriminator_Network):
    Discriminator_load_path = os.path.join(folder,filename)
    if filename.split('.')[1] == 'pt':
        Discriminator_Network = torch.load(Discriminator_load_path)
    elif filename.split('.')[1] == 'pth':
        Discriminator_Network.load_state_dict(torch.load(Discriminator_load_path))
    return Discriminator_Network

def save_Discriminator_to_checkpoint(Discriminator_Network,folder,filename,flag_save_dict_or_whole='dict'):
    Discriminator_save_path = os.path.join(folder,filename)
    Discriminator_Network = Discriminator_to_CPU(Discriminator_Network)
    if flag_save_dict_or_whole == 'dict' or 'both':
        torch.save(Discriminator_Network.state_dict(), Discriminator_save_path+'.pth')
    if flag_save_dict_or_whole == 'whole' or 'both':
        torch.save(Discriminator_Network, Discriminator_save_path+'.pt')
    Discriminator_Network = Discriminator_to_GPU(Discriminator_Network)

def Generator_to_CPU(Generator_Network, netF=None):
    Generator_Network = Generator_Network.to('cpu')
    # Generator_Network.hidden_states_to_device(device_cpu)
    # Generator_Network.reset_hidden_states()
    if netF:
        netF = netF.to('cpu')
    return Generator_Network

def Generator_Loss_Functions_to_CPU():
    global DirectPixels_Loss_Function, FeatureExtractor_Loss_Function, Contextual_Loss_Function, GradientSensitive_Loss_Function, Gram_Loss_Function
    DirectPixels_Loss_Function.to('cpu')
    FeatureExtractor_Loss_Function.to('cpu')
    Contextual_Loss_Function.to('cpu')
    GradientSensitive_Loss_Function.to('cpu')
    Gram_Loss_Function.to('cpu')

def Discriminator_to_CPU(Discriminator_Network):
    Discriminator_Network = Discriminator_Network.cpu()
    return Discriminator_Network

def Discriminator_Loss_Functions_to_CPU():
    global GAN_Validity_Loss_Function, GradientPenalty_Loss_Function, Relativistic_GAN_Validity_Loss_Function
    GAN_Validity_Loss_Function.to('cpu')
    GradientPenalty_Loss_Function.to('cpu')
    Relativistic_GAN_Validity_Loss_Function.to('cpu')


def get_Network_Layers_Devices(Network):
    Generator_Network_layers_list = get_network_layers_list_flat(Generator_Network)
    for layer in Generator_Network_layers_list:
        if len(list(layer.parameters()))>0:
            print(layer.parameters().__next__().device)

def freeze_non_temporal_layers(Network):
    for name, value in Generator_Network.named_parameters():
        if 'cell' not in str.lower(name):
            value.requires_grad = False
    return Network

def plot_grad_flow(Generator_Network):
    #Use this function AFTER doing a .backward() which updates the .grad property of the layers
    average_abs_gradients_list = []
    layers_list = []
    for parameter_name, parameter in Generator_Network.named_parameters():
        if (parameter.requires_grad) and ("bias" not in parameter_name) and type(parameter.grad)!=type(None): #we usually don't really care about the bias term
            layers_list.append(parameter_name.split('.weight'))
            average_abs_gradients_list.append(parameter.grad.abs().mean())
    plt.plot(average_abs_gradients_list, alpha=0.3, color="b")
    # plt.scatter(arange(len(average_abs_gradients_list)),average_abs_gradients_list)
    plt.hlines(0, 0, len(average_abs_gradients_list)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(average_abs_gradients_list), 1), layers_list, rotation="horizontal")
    plt.xlim(xmin=0, xmax=len(average_abs_gradients_list))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    x_vec = np.arange(0,len(average_abs_gradients_list),1)
    for i,layer_name in enumerate(layers_list):
        plt.text(x_vec[i], average_abs_gradients_list[i], layers_list[i], fontdict=None, withdash=False, rotation='vertical', verticalalignment ='bottom')
    plt.title("Gradient flow")
    plt.grid(True)
# #Use Example:



class EMA(nn.Module):
    #Exponential moving average
    def __init__(self, mu):
        super(EMA, self).__init__()
        self.mu = mu
        self.flag_first_time_passed = False
    def forward(self, x, last_average):
        if self.flag_first_time_passed==False:
            new_average = x
            self.flag_first_time_passed = True
        else:
            new_average = self.mu * x + (1 - self.mu) * last_average
        return new_average





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






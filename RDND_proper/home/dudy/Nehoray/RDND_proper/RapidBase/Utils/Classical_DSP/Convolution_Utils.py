




from typing import Tuple, List
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
import numpy as np
import cv2
# from kornia.filters.kernels import normalize_kernel2d
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import torch_get_5D, torch_get_3D, torch_get_2D, torch_get_4D

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.size()) < 2:
        raise TypeError("input should be at least 2D tensor. Got {}"
                        .format(input.size()))
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Computes padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) >= 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymetric padding :(

    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        if kernel_size[i] % 2 == 0:
            padding = max(computed_tmp - 1, 0)
        else:
            padding = computed_tmp
        out_padding[2 * i + 0] = padding
        out_padding[2 * i + 1] = computed_tmp
    return out_padding


def filter2D_torch(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'reflect',
             normalized: bool = False) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2D_torch(input, kernel)
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input image is not torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input kernel is not torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 3 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    # pad the input tensor
    height, width = tmp_kernel.shape[-2:]
    padding_shape: List[int] = _compute_padding([height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    B, C, H, W = output.shape
    residual_shape_1 = output.shape[-1] - input.shape[-1]
    residual_shape_2 = output.shape[-2] - input.shape[-2]
    output = output[:, :, 0:H - residual_shape_2, 0:W - residual_shape_1]

    return output.view(b, c, h, w)



def filter3D_torch(input: torch.Tensor, kernel: torch.Tensor,
             border_type: str = 'replicate',
             normalized: bool = False) -> torch.Tensor:
    r"""Convolve a tensor with a 3d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, D, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kD, kH, kW)`  or :math:`(B, kD, kH, kW)`.
        border_type (str): the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``,
          ``'replicate'`` or ``'circular'``. Default: ``'replicate'``.
        normalized (bool): If True, kernel will be L1 normalized.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, D, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 5., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]],
        ...    [[0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.],
        ...     [0., 0., 0., 0., 0.]]
        ... ]]])
        >>> kernel = torch.ones(1, 3, 3, 3)
        >>> filter3D_torch(input, kernel)
        tensor([[[[[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]],
        <BLANKLINE>
                  [[0., 0., 0., 0., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 5., 5., 5., 0.],
                   [0., 0., 0., 0., 0.]]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(input)))

    if not isinstance(kernel, torch.Tensor):
        raise TypeError("Input border_type is not torch.Tensor. Got {}"
                        .format(type(kernel)))

    if not isinstance(border_type, str):
        raise TypeError("Input border_type is not string. Got {}"
                        .format(type(kernel)))

    if not len(input.shape) == 5:
        raise ValueError("Invalid input shape, we expect BxCxDxHxW. Got: {}"
                         .format(input.shape))

    if not len(kernel.shape) == 4 and kernel.shape[0] != 1:
        raise ValueError("Invalid kernel shape, we expect 1xDxHxW. Got: {}"
                         .format(kernel.shape))

    # prepare kernel
    b, c, d, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        bk, dk, hk, wk = kernel.shape
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(
            bk, dk, hk * wk)).view_as(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1, -1)

    # pad the input tensor
    depth, height, width = tmp_kernel.shape[-3:]
    padding_shape: List[int] = _compute_padding([depth, height, width])
    input_pad: torch.Tensor = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.size(0), input_pad.size(-3), input_pad.size(-2), input_pad.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv3d(input_pad, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    B,C,D,H,W = output.shape
    residual_shape_1 = output.shape[-1] - input.shape[-1]
    residual_shape_2 = output.shape[-2] - input.shape[-2]
    residual_shape_3 = output.shape[-3] - input.shape[-3]
    output = output[:,:,0:D-residual_shape_3,0:H-residual_shape_2,0:W-residual_shape_1]

    return output.view(b, c, d, h, w)


# class convn_layer_torch(nn.Module):
#     """
#     Apply gaussian smoothing on a
#     1d, 2d or 3d tensor. Filtering is performed seperately for each channel
#     in the input using a depthwise convolution.
#     Arguments:
#         kernel_size (int, sequence): Size of the gaussian kernel.
#         sigma (float, sequence): Standard deviation of the gaussian kernel.
#         dim (int, optional): The number of dimensions of the data.
#             Default value is 2 (spatial).
#     """
#     def __init__(self, kernel, dim=2):
#         super(convn_layer_torch, self).__init__()
#
#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#
#         self.register_buffer('weight', kernel)
#         self.groups = channels
#
#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(
#                 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
#             )
#
#         self.padding = []
#         for i in kernel_size:
#             self.padding.append(i//2)
#             self.padding.append(i//2)
#
#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """
#         return self.conv(F.pad(input, self.padding), weight=self.weight, groups=self.groups)

import matplotlib.pyplot as plt
class convn_layer_Filter3D_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """
    def __init__(self):
        super(convn_layer_Filter3D_torch, self).__init__()

    def forward(self, input_tensor, kernel, dim):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        ### Get input tensor to proper size (B,T,C,H,W)/(B,C,H,W,D): ###
        original_shape = input_tensor.shape
        input_tensor = torch_get_5D(input_tensor)

        ### Get explicit dimension index: ###
        if dim >= 0:
            dim = dim - len(original_shape)
        dim = len(input_tensor.shape) - np.abs(dim)

        ### Filter: ###
        if dim>=2 and dim<=4:
            ### Get kernel to be 4D: (1,Kx,Ky,Kz) or (B,Kx,Ky,Kz) ###
            if dim == 4:
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            elif dim == 3:
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            elif dim == 2:
                kernel_final = kernel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            output_tensor = filter3D_torch(input_tensor, kernel_final)
        else:
            if dim == 1:
                input_tensor = input_tensor.permute([0, 2, 3, 4, 1])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                output_tensor = filter3D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([0, 4, 1, 2, 3])
            elif dim == 0:
                input_tensor = input_tensor.permute([1, 2, 3, 4, 0])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                output_tensor = filter3D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([4, 0, 1, 2, 3])

        if len(original_shape) == 3:
            output_tensor = output_tensor[0,0]
        if len(original_shape) == 4:
            output_tensor = output_tensor[0]

        # plt.figure()
        # plt.imshow(output_tensor[0,0].permute([1,2,0]).cpu().numpy())
        # plt.figure()
        # plt.imshow(input_tensor[0,0].permute([1,2,0]).cpu().numpy())
        # plt.show()
        return output_tensor

class convn_layer_Filter2D_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """
    def __init__(self):
        super(convn_layer_Filter2D_torch, self).__init__()

    def forward(self, input_tensor, kernel, dim):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """

        ### Get input tensor to proper size: ###
        original_shape = input_tensor.shape
        input_tensor = torch_get_4D(input_tensor)
        B,C,H,W = input_tensor.shape

        ### Get explicit dimension index: ###
        if dim >= 0:
            dim = dim - len(original_shape)
        dim = len(input_tensor.shape) - np.abs(dim)

        ### Filter: ###
        if dim>=2 and dim<=3:
            ### Get kernel to be 4D: (1,Kx,Ky) or (B,Kx,Ky) ###
            if dim == 3:
                kernel_final = kernel.unsqueeze(0).unsqueeze(0)
            elif dim == 2:
                kernel_final = kernel.unsqueeze(0).unsqueeze(-1)
            output_tensor = filter2D_torch(input_tensor, kernel_final)
        else:
            if dim == 1:
                input_tensor = input_tensor.permute([0,2,3,1])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0)
                output_tensor = filter2D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([0,3,1,2])
            elif dim == 0:
                input_tensor = input_tensor.permute([1, 2, 3, 0])
                kernel_final = kernel.unsqueeze(0).unsqueeze(0)
                output_tensor = filter2D_torch(input_tensor, kernel_final)
                output_tensor = output_tensor.permute([3, 0, 1, 2])


        if len(original_shape) == 2:
            output_tensor = output_tensor[0,0]
        if len(original_shape) == 3:
            output_tensor = output_tensor[0]

        # plt.figure()
        # plt.imshow(output_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.figure()
        # plt.imshow(input_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.show()
        return output_tensor


class convn_layer_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """

    def __init__(self):
        super(convn_layer_torch, self).__init__()
        self.convn_layer_Filter2D_torch = convn_layer_Filter2D_torch()
        self.convn_layer_Filter3D_torch = convn_layer_Filter3D_torch()

    def forward(self, input_tensor, kernel, dim):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        #TODO: allow the kernel input to be a list or a tuple!!!!!

        ### Choose which function to use according to input size and dimension
        # (assuming timing(Filter1D)<timing(Filter2D)<timing(Filter3D) in any case): ###
        original_shape = input_tensor.shape
        if dim < 0:
            dim = len(original_shape) - np.abs(dim)

        if len(original_shape) == 1:
            output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 2:
            output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 3:
            if dim == 1 or dim == 2:
                output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
            elif dim == 0:
                output_tensor = self.convn_layer_Filter3D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 4:
            if dim == 2 or dim == 3:
                output_tensor = self.convn_layer_Filter2D_torch.forward(input_tensor, kernel=kernel, dim=dim)
            elif dim == 0 or dim == 1:
                output_tensor = self.convn_layer_Filter3D_torch.forward(input_tensor, kernel=kernel, dim=dim)
        elif len(original_shape) == 5:
            output_tensor = self.convn_layer_Filter3D_torch.forward(input_tensor, kernel=kernel, dim=dim)

        # plt.figure()
        # plt.imshow(output_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.figure()
        # plt.imshow(input_tensor[0].permute([1,2,0]).cpu().numpy())
        # plt.show()
        return output_tensor

def convn_torch(input_tensor, kernel, dim):
    original_shape = input_tensor.shape
    if dim < 0:
        dim = len(original_shape) - np.abs(dim)

    #TODO: turn convn_layer_Filter2D_torch into function!!!!!
    if len(original_shape) == 1:
        output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 2:
        output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 3:
        if dim == 1 or dim == 2:
            output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
        elif dim == 0:
            output_tensor = convn_layer_Filter3D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 4:
        if dim == 2 or dim == 3:
            output_tensor = convn_layer_Filter2D_torch().forward(input_tensor, kernel=kernel, dim=dim)
        elif dim == 0 or dim == 1:
            output_tensor = convn_layer_Filter3D_torch().forward(input_tensor, kernel=kernel, dim=dim)
    elif len(original_shape) == 5:
        output_tensor = convn_layer_Filter3D_torch().forward(input_tensor, kernel=kernel, dim=dim)

    return output_tensor


class convn_fft_layer_torch(nn.Module):
    """
    1D convolution of a n-dimensional tensor over wanted dimension
    """
    def __init__(self):
        super(convn_fft_layer_torch, self).__init__()

    def forward(self, input_tensor, kernel, dim, flag_return_real=True):
        """
        Arguments:
            input (torch.Tensor): Input to apply filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        original_shape = input_tensor.shape
        if dim < 0:
            dim = len(input_tensor.shape) - abs(dim)
        kernel_fft = torch.fft.fftn(kernel, s=input_tensor.shape[dim], dim=-1)
        number_of_dimensions_to_add_in_the_end = len(original_shape) - dim - 1
        number_of_dimensions_to_add_in_the_beginning = len(original_shape) - number_of_dimensions_to_add_in_the_end - 1
        for i in np.arange(number_of_dimensions_to_add_in_the_end):
            kernel_fft = kernel_fft.unsqueeze(-1)
        for i in np.arange(number_of_dimensions_to_add_in_the_beginning):
            kernel_fft = kernel_fft.unsqueeze(0)

        output_tensor = torch.fft.ifftn(torch.fft.fftn(input_tensor, dim=dim) * kernel_fft, dim=dim)
        if flag_return_real:
            return output_tensor.real
        else:
            return output_tensor


def read_image_torch(path, flag_convert_to_rgb=1, flag_normalize_to_float=0):
    image = cv2.imread(path, cv2.IMREAD_COLOR);
    if flag_convert_to_rgb == 1:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB);
    if flag_normalize_to_float == 1:
        image = image.astype(np.float32) / 255; #normalize to range [0,1] instead of [0,255]
    if image.ndim == 2:
        # Make Sure That We Get the Image in the correct format [H,W,C]:
        image = np.expand_dims(image, axis=2);
    if image.shape[2] > 4:
        # In the weird case where the image has more then 3 channels only take the first 3 channels to make it conform with 3-channels image format:
        image = image[:,:,:3];

    image = np.transpose(image,[2,0,1])
    image = torch.Tensor(image);
    image = image.unsqueeze(0)
    return image




# ### TODO: temp ###
# default_image_filename_to_load1 = r'/home/mafat/DataSets/DIV2K/DIV2K/Flickr2K/000001.png'
# kernel = torch.ones((11))/11
# input_tensor = read_image_torch(default_image_filename_to_load1)/255
# # input_tensor = torch.randn(3,3,3,100,100)
# convn_layer = convn_layer_torch()
# convn_layer_2D = convn_layer_Filter2D_torch()
# convn_layer_3D = convn_layer_Filter3D_torch()
# convn_layer_fft = convn_fft_layer_torch()
# output_tensor_fft_space = convn_layer_fft(input_tensor, kernel, -1)
# output_tensor_real_space = convn_layer(input_tensor, kernel, -1)
#
# plt.figure(); plt.imshow(output_tensor_fft_space[0].permute([1,2,0]).cpu().numpy())
# plt.figure(); plt.imshow(output_tensor_real_space[0].permute([1,2,0]).cpu().numpy())
# plt.figure(); plt.imshow((output_tensor_real_space-output_tensor_fft_space)[0].permute([1,2,0]).cpu().numpy())
# plt.figure(); plt.imshow(input_tensor[0].permute([1,2,0]).cpu().numpy())








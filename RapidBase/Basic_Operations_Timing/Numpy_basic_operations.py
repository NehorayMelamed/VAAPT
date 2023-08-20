# from RapidBase.import_all import *
import torch
import torch.fft
import torch.nn as nn
import numpy as np

T = 1024
H = 512
W = 512
input_tensor = np.random.randn(H,W,T)

#(1).
output = np.fft.fftn(input_tensor, axes=[0,1])
output = np.fft.ifftn(input_tensor, axes=[0,1])
#(2).
output = np.fft.fftn(input_tensor, axes=[0]) #temporal

#(3).
output = np.fft.fftn(input_tensor, axes=[-1]) #columns
output = np.fft.fftn(input_tensor, axes=[-2]) #rows

#(4).
multiply_mat = np.random.randn(H,W,T)
output = input_tensor * multiply_mat

#(5).
multiply_mat = np.random.randn(H,W)
output = input_tensor * np.atleast_3d(multiply_mat)  #[T,H,W] * [H,W]

#(6).
center_crop_rows = 400
center_crop_cols = 400
mat_in_rows_excess = H - center_crop_rows
mat_in_cols_excess = W - center_crop_cols
start = 0
end_x = W
end_y = H
start_index_cols = int(start + mat_in_cols_excess / 2)
start_index_rows = int(start + mat_in_rows_excess / 2)
stop_index_cols = start_index_cols + center_crop_cols
stop_index_rows = start_index_rows + center_crop_rows
output = input_tensor[start_index_rows:stop_index_rows, start_index_cols:stop_index_cols,:]

#(7)
F_cols = np.random.randn(1,W)
output = input_tensor * np.expand_dims(F_cols,-1)

#(8).
F_rows = np.random.randn(H,1)
output = input_tensor * np.expand_dims(F_rows,-1)

#(9).
[X, Y] = np.meshgrid(np.arange(W), np.arange(H))
XY_meshgrid_tensor = np.concatenate([np.expand_dims(X,-1), np.expand_dims(Y,-1)], -1)  #[2,H,W]

#(10).
scalar = 5
[X_mesh, Y_mesh] = np.meshgrid(np.arange(W), np.arange(H))
X_mesh_T_stacked = np.concatenate([np.expand_dims(X_mesh,-1)]*T, -1)
output = np.exp(-2*np.pi*1j*scalar*np.expand_dims(X_mesh,-1)) * input_tensor
output = np.exp(-2*np.pi*1j*scalar*X_mesh_T_stacked) * input_tensor

#(11).
start_index = 13
stop_index = 108
stride = 3
output = input_tensor[0:-1,:,:]
output = input_tensor[1:-2,:,:]
output = input_tensor[start_index:stop_index,:,:]
output = input_tensor[0::2,:,:]
output = input_tensor[start_index:stop_index:2,:,:]
output = input_tensor[start_index:stop_index:stride,:,:]

#(12).
output = input_tensor.sum()
output = input_tensor.sum(0).sum(0)
output = input_tensor.sum(0)
#(*). Max over pixels:
max_values_per_row = input_tensor.max(1)
max_values_per_col = input_tensor.max(0)
max_values_per_image = max_values_per_row.max(0)
max_values_per_image = max_values_per_col.max(0)
#(*). Max over time:
max_values_per_pixel_over_time = input_tensor.max(-1)
#(*). Sums:
sum_per_row = input_tensor.sum(1)
sum_per_col = input_tensor.sum(0)
#(*). Cumsum:
cumsum_per_row = input_tensor.cumsum(1)
cumsum_per_col = input_tensor.cumsum(0)


#(13).
#TODO: avearge pooling can also be implemented using difference of cumsum operations, which very likely will be faster!!!!!
def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 4, with shape (m, hi, wi, ci).
        sub_shape (tuple): window size: (f1, f2).
        stride (int): stride of windows in both 2nd and 3rd dimensions.
    Returns:
        subs (view): strided window view.
    This is used to facilitate a vectorized 3d convolution.
    The input array <arr> has shape (m, hi, wi, ci), and is transformed
    to a strided view with shape (m, ho, wo, f, f, ci). where:
        m: number of records.
        hi, wi: height and width of input image.
        ci: channels of input image.
        f: kernel size.
    The convolution kernel has shape (f, f, ci, co).
    Then the vectorized 3d convolution can be achieved using either an einsum()
    or a tensordot():
        conv = np.einsum('myxfgc,fgcz->myxz', arr_view, kernel)
        conv = np.tensordot(arr_view, kernel, axes=([3, 4, 5], [0, 1, 2]))
    See also skimage.util.shape.view_as_windows()
    '''
    sm, sh, sw, sc = arr.strides
    m, hi, wi, ci = arr.shape
    f1, f2 = sub_shape
    view_shape = (m, 1+(hi-f1)//stride, 1+(wi-f2)//stride, f1, f2, ci)
    strides = (sm, stride*sh, stride*sw, sh, sw, sc)
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs
def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 4D data.
    Args:
        mat (ndarray): input array to do pooling on the mid 2 dimensions.
            Shape of the array is (m, hi, wi, ci). Where m: number of records.
            hi, wi: height and width of input image. ci: channels of input image.
        f (int): pooling kernel size in row/column.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.
    See also unpooling().
    '''
    m, hi, wi, ci = mat.shape
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1
    if pad:
        ny = _ceil(hi, stride)
        nx = _ceil(wi, stride)
        size = (m, (ny-1)*stride+f, (nx-1)*stride+f, ci)
        mat_pad = np.full(size, 0)
        mat_pad[:, :hi, :wi, ...] = mat
    else:
        mat_pad = mat[:, :(hi-f)//stride*stride+f, :(wi-f)//stride*stride+f, ...]
    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(3, 4), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(3, 4), keepdims=return_max_pos)
    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result, axis=(3,4))
        return result, pos
    else:
        return result
binning_factor = 2
binning_layer_X2 = nn.AvgPool2d(2)
binning_layer_X4 = nn.AvgPool2d(4)
binning_layer_X8 = nn.AvgPool2d(8)
binning_layer_X16 = nn.AvgPool2d(16)
binning_layer_X32 = nn.AvgPool2d(32)
output_X2 = poolingOverlap(np.expand_dims(input_tensor,0),2,2)
output_X4 = poolingOverlap(np.expand_dims(input_tensor,0),4,4)
output_X8 = poolingOverlap(np.expand_dims(input_tensor,0),8,8)
output_X16 = poolingOverlap(np.expand_dims(input_tensor,0),16,16)


#(14).
T1 = 1024
T2 = 1023
input_tensor1 = np.random.randn(H,W,T1)
input_tensor2 = np.random.randn(H,W,T2)
output = torch.cat([input_tensor1, input_tensor2], -1)

#(15).
output_flattened = np.reshape(input_tensor, [H*W,T])
# output_flatten_view = input_tensor.view(-1, H*W)
# output_unflattened = output_flattened.view(-1,H,W)
# output_unflattened = torch.reshape(input_tensor, [T,H,W])

#(16).
#TODO: i think it's numpy.take or something
# input_tensor = np.random.randn(H,W,T) #[H,W,T]
# index_mat = torch.randint_like(input_tensor, 1)  #[H,1,T]
# output = torch.gather(input_tensor, dim=2, index=index_mat.long()) #[H,1,T]

#(18).
#TODO: use scipy or opencv
upsampling_layer = nn.Upsample(scale_factor=2, mode='bicubic')
downsampling_layer = nn.Upsample(scale_factor=1/2, mode='bicubic')
output_upsampled = upsampling_layer(input_tensor)
output_downsampled = downsampling_layer(input_tensor)

#(19).
#TODO: use scipy or opencv
#for bilinear use grid_sample
#for bicubic i have built a function








# from RapidBase.import_all import *
import torch
import torch.fft
import torch.nn as nn
import numpy as np

T = 1024
H = 512
W = 512
input_tensor = torch.randn(T,H,W)

#(1).
output = torch.fft.fftn(input_tensor, dim=[-1,-2])
output = torch.fft.ifftn(input_tensor, dim=[-1,-2])

#(2).
output = torch.fft.fftn(input_tensor, dim=[0]) #temporal

#(3).
output = torch.fft.fftn(input_tensor, dim=[-1]) #columns
output = torch.fft.fftn(input_tensor, dim=[-2]) #rows

#(4).
multiply_mat = torch.randn(T,H,W)
output = input_tensor * multiply_mat

#(5).
multiply_mat = torch.randn(H,W)
output = input_tensor * multiply_mat  #[T,H,W] * [H,W]

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
output = input_tensor[:,start_index_rows:stop_index_rows, start_index_cols:stop_index_cols]

#(7)
F_cols = torch.randn(1,W)
output = input_tensor * F_cols.unsqueeze(1)

#(8).
F_rows = torch.randn(1,H)
output = input_tensor * F_rows.unsqueeze(2)

#(9).
[X, Y] = np.meshgrid(np.arange(W), np.arange(H))
X = torch.Tensor(X)
Y = torch.Tensor(Y)
XY_meshgrid_tensor = torch.cat([X.unsqueeze(0), Y.unsqueeze(0)], 0)  #[2,H,W]
stacked_meshgrid_tensor = torch.cat([XY_meshgrid_tensor.unsqueeze(0)] * T, 0) #[T,2,H,W]

#(10).
scalar = 5
[X_mesh, Y_mesh] = np.meshgrid(np.arange(W), np.arange(H))
X_mesh = torch.Tensor(X_mesh)
Y_mesh = torch.Tensor(Y_mesh)
X_mesh_T_stacked = torch.cat([X_mesh.unsqueeze(0)]*T, 0)
output = torch.exp(-2*np.pi*1j*scalar*X_mesh) * input_tensor
output = torch.exp(-2*np.pi*1j*scalar*X_mesh_T_stacked) * input_tensor

#(11).
start_index = 13
stop_index = 108
stride = 3
output = input_tensor[:,:,0:-1]
output = input_tensor[:,:,1:-2]
output = input_tensor[:,:,start_index:stop_index]
output = input_tensor[:,:,0::2]
output = input_tensor[:,:,start_index:stop_index:2]
output = input_tensor[:,:,start_index:stop_index:stride]

#(12).
output = input_tensor.sum()
output = input_tensor.sum([-1,-2], keepdim=True)
output = input_tensor.sum(0)
#(*). Max over pixels:
max_values_per_row, max_indices_per_row = input_tensor.max(-1)
max_values_per_col, max_indices_per_col = input_tensor.max(-2)
max_values_per_image, max_indices_per_image = max_values_per_row.max(-1)
max_values_per_image, max_indices_per_image = max_values_per_col.max(-1)
#(*). Max over time:
max_values_per_pixel_over_time, max_indices_per_pixel_over_time = input_tensor.max(0)
#(*). Sums:
sum_per_row = input_tensor.sum(2, keepdim=True)
sum_per_col = input_tensor.sum(1, keepdim=True)
#(*). Cumsum:
cumsum_per_row = input_tensor.cumsum(2)
cumsum_per_col = input_tensor.cumsum(1)


#(13).
binning_factor = 2
binning_layer_X2 = nn.AvgPool2d(2)
binning_layer_X4 = nn.AvgPool2d(4)
binning_layer_X8 = nn.AvgPool2d(8)
binning_layer_X16 = nn.AvgPool2d(16)
binning_layer_X32 = nn.AvgPool2d(32)
output_X2 = binning_layer_X2(input_tensor)
output_X4 = binning_layer_X4(input_tensor)
output_X8 = binning_layer_X8(input_tensor)
output_X16 = binning_layer_X16(input_tensor)
output_X32 = binning_layer_X32(input_tensor)

#(14).
T1 = 1024
T2 = 1023
input_tensor1 = torch.randn(T1,H,W)
input_tensor2 = torch.randn(T2,H,W)
output = torch.cat([input_tensor1, input_tensor2], 0)

#(15).
output_flattened = torch.flatten(input_tensor, 1, 2)
output_flattened = torch.reshape(input_tensor, [T,H*W])
output_flatten_view = input_tensor.view(-1, H*W)
output_unflattened = output_flattened.view(-1,H,W)
output_unflattened = torch.reshape(input_tensor, [T,H,W])

#(16).
input_tensor = torch.randn(T,H,W) #[T,H,W]
index_mat = torch.randint_like(input_tensor, 1)  #[T,H,1]
output = torch.gather(input_tensor, dim=2, index=index_mat.long()) #[T,H,1]

#(18).
upsampling_layer = nn.Upsample(scale_factor=2, mode='bicubic')
downsampling_layer = nn.Upsample(scale_factor=1/2, mode='bicubic')
output_upsampled = upsampling_layer(input_tensor)
output_downsampled = downsampling_layer(input_tensor)

#(19).
#for bilinear use grid_sample
#for bicubic i have built a function








from RapidBase.import_all import *
import timeit
H = 512*2
W = 512*2
T = 128*1
NUM_WARMUP_ITERS = 10
timings = []
names = []
totaltimes = []
torchdevice = 0
sizes = [512, 1024, 2048, 4096]
#
# def torch_ifft2_fft2(data):
#     # data2 = torch.fft.fftn(data, dim=[-1,-2])
#     # return torch.fft.ifftn(data2, dim=[-1,-2])
#     return torch.fft.fftn(data, dim=[-1,-2])
#
# def torch_CC(input_tensor):
#     # data2 = torch.fft.fftn(input_tensor, dim=[-1,-2])
#     # return torch.fft.ifftn(data2, dim=[-1,-2])
#     input_tensor_fft = torch.fft.fftn(input_tensor, dim=[-1,-2])
#     input_tensor_CC = torch.fft.fftn(input_tensor_fft[0:T-1]*input_tensor_fft[1:T]).real
#     input_tensor_CC_max = torch.argmax(input_tensor_CC, dim=-1)
#     return input_tensor_CC
#
# def torch_fft_1D(data):
#     # data2 = torch.fft.fftn(data, dim=[-1,-2])
#     # return torch.fft.ifftn(data2, dim=[-1,-2])
#     return torch.fft.fftn(data, dim=[0])
#
# ### Pytorch Benchmark: ###
# name = 'PyTorch\nGPU'
# device_type = 'cuda'
# device = torch.device('cuda:{}'.format(torchdevice))
# names.append(name)
# print('{} available: {}'.format(name, torch.cuda.is_available()))
# torchtimes = []
# datas = [torch.randn(size=(x,x,2), device=device) for x in sizes]
# input_tensor = torch.randn(T,H,W).cuda()
# print(f'Working on device: {datas[0].device}')
# # for x, data in zip(sizes, datas):
# #     print('{} {}x{}'.format(" ".join(name.split('\n')), x, x))
# #     # t = %timeit -o torch_ifft2_fft2(data)
# #
# #     # start = timeit.timeit()
# #     # torch_ifft2_fft2(data)
# #     # end = timeit.timeit()
#
# # result = timeit.timeit(lambda: torch_ifft2_fft2(input_tensor), number=100)
# input_tensor = input_tensor.cpu()
# start = torch.cuda.Event(enable_timing=True)
# finish = torch.cuda.Event(enable_timing=True)
# start.record()
# # bla = torch_ifft2_fft2(input_tensor)
# # bla = torch_fft_1D(input_tensor)
# # bla = torch_CC(input_tensor)
# input_tensor = input_tensor.cuda()
# finish.record()
# torch.cuda.synchronize()
# print(f"total time = {start.elapsed_time(finish)}")
# # torchtimes.append(result*1000/100)
# # totaltimes.append(torchtimes)
# # print(totaltimes)
#
#
### 2D FFT: ###
from RapidBase.import_all import *
import timeit
import torch
import torch.fft
H = 512*1
W = 512*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W) + 1j*torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    output_tensor = torch.fft.fftn(input_tensor, dim=[-1,-2])
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
output_tensor = torch.fft.fftn(input_tensor, dim=[-1,-2])
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")

### 1D FFT (Temporal): ###
import timeit
import torch
import torch.fft
H = 320*1
W = 640*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W) + 1j*torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    output_tensor = torch.fft.fftn(input_tensor, dim=[0])
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
output_tensor = torch.fft.fftn(input_tensor, dim=[0])
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")

### Grid Sample: ###
import timeit
import torch
import torch.fft
import torch.nn.functional as F
H = 320*1
W = 640*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(1,T,H,W)
tensor_grid = torch.zeros(1,H,W,2)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    tensor_grid = tensor_grid.cuda()
    output_tensor = F.grid_sample(input_tensor, tensor_grid, 'nearest')
    input_tensor = input_tensor.cpu()
    tensor_grid = tensor_grid.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
tensor_grid = tensor_grid.cuda()
start.record()
output_tensor = F.grid_sample(input_tensor, tensor_grid, 'nearest')
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")

### Binning: ###
def fast_binning_2D_overap_flexible(input_tensor, binning_size_tuple, overlap_size_tuple):
    # input_tensor = torch.randn((100,512,512))
    # binning_size = (10,20)
    # overlap_size = (0,0)

    ### Expecfting tuple: ###
    binning_size_H, binning_size_W = binning_size_tuple
    overlap_size_H, overlap_size_W = overlap_size_tuple

    T,H,W = input_tensor.shape
    step_size_H = binning_size_H - overlap_size_H
    step_size_W = binning_size_W - overlap_size_W
    H_final = 1 + np.int16((H - binning_size_H) / step_size_H)
    W_final = 1 + np.int16((W - binning_size_W) / step_size_W)

    column_cumsum = torch.cat((torch.zeros((T, H, 1)).to(input_tensor.device), torch.cumsum(input_tensor, 2)), 2) #pad to the left, maybe there's a faster way
    T1,H1,W1 = column_cumsum.shape
    column_binning = column_cumsum[:, :, binning_size_W:W1:step_size_W] - column_cumsum[:, :, 0:W1 - binning_size_W:step_size_W]

    row_cumsum = torch.cat((torch.zeros(T,1,W_final).to(input_tensor.device), torch.cumsum(column_binning,1)), 1)
    T2,H2,W2 = row_cumsum.shape
    binned_matrix_final = row_cumsum[:, binning_size_H:H2:step_size_H,:] - row_cumsum[:, 0:H2-binning_size_H:step_size_H]

    return binned_matrix_final

import timeit
import torch
import numpy as np
import torch.fft
H = 320*1
W = 640*1
T = 512*1
binning_size_tuple = (1,5)
overlap_size_tuple = (0,4)
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    output_tensor = fast_binning_2D_overap_flexible(input_tensor, binning_size_tuple, overlap_size_tuple)
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
output_tensor = fast_binning_2D_overap_flexible(input_tensor, binning_size_tuple, overlap_size_tuple)
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")



### 1D FFT (Row-Wise): ###
import timeit
import torch
import torch.fft
H = 320*1
W = 640*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W) + 1j*torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    output_tensor = torch.fft.fftn(input_tensor, dim=[-1])
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
output_tensor = torch.fft.fftn(input_tensor, dim=[-1])
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")

### 1D FFT (Col-Wise): ###
import timeit
import torch
import torch.fft
H = 512*1
W = 512*1
T = 128*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W) + 1j*torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    output_tensor = torch.fft.fftn(input_tensor, dim=[-2])
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
output_tensor = torch.fft.fftn(input_tensor, dim=[-2])
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")


### 1D Multiplication (Col-Wise): ###
import timeit
import torch
import torch.fft
H = 512*1
W = 512*1
T = 128*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W) + 1j*torch.randn(T,H,W)
F_cols = torch.randn(1,W,1)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    F_cols = F_cols.cuda()
    output_tensor = input_tensor * F_cols
    input_tensor = input_tensor.cpu()
    F_cols = F_cols.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
F_cols = F_cols.cuda()
start.record()
output_tensor = input_tensor * F_cols
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")


### Pixel-Wise Multiplication: ###
import timeit
import torch
import torch.fft
H = 1024*1
W = 1024*1
T = 256*1
NUM_WARMUP_ITERS = 10
input_tensor_1 = torch.randn(T,H,W) + 1j*torch.randn(T,H,W)
input_tensor_2 = torch.randn(T,H,W) + 1j*torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor_1 = input_tensor_1.cuda()
    input_tensor_2 = input_tensor_2.cuda()
    output_tensor = input_tensor_1 * input_tensor_2
    input_tensor_1 = input_tensor_1.cpu()
    input_tensor_2 = input_tensor_2.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor_1 = input_tensor_1.cuda()
input_tensor_2 = input_tensor_2.cuda()
start.record()
output_tensor = input_tensor_1 * input_tensor_2
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")


### 1D Max (Over Time): ###
import timeit
import torch
import torch.fft
H = 1024*1
W = 1024*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    max_values, max_indices = torch.max(input_tensor, 0)
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
max_values, max_indices = torch.max(input_tensor, 0)
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")


### EDVR: ####
import torch
# from RDND_proper.models.BasicSR.basicsr.models.archs.edvr_arch import *
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import *
H = 256
W = 256
model = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=False).cuda()
input_tensor = torch.randn(1,5,3,H,W).cuda()
with torch.no_grad():
    output_tensor = model.forward(input_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
start.record()
with torch.no_grad():
    output_tensor = model.forward(input_tensor)
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")


### 1D Mean (Over Time): ###
import timeit
import torch
import torch.fft
H = 1024*1
W = 1024*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    mean_values = torch.mean(input_tensor, 0)
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
mean_values = torch.mean(input_tensor, 0)
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")



### 2D Mean (Over Space): ###
import timeit
import torch
import torch.fft
H = 1024*1
W = 1024*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    mean_values = torch.mean(input_tensor, [-1,-2])
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
mean_values = torch.mean(input_tensor, [-1,-2])
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")



### Exponentiate Values (Over Space): ###
import timeit
import torch
import torch.fft
H = 1024*1
W = 1024*1
T = 512*1
NUM_WARMUP_ITERS = 10
input_tensor = torch.randn(T,H,W)
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    exp_tensor = torch.exp(2.0*1j*input_tensor)
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
exp_tensor = torch.exp(2.0*1j*input_tensor)
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")




### Gimbaless: ###
from RapidBase.import_all import *
from RapidBase.Utils.Registration.Unified_Registration_Utils import Gimbaless_Rotation_Layer_Torch, create_speckles_of_certain_size_in_pixels
gimbaless_layer = Gimbaless_Rotation_Layer_Torch().cuda()
T = 128
H = 512
W = 512
speckle_size = 3
shift_layer = Shift_Layer_Torch()
speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(speckle_size, H, 0, 1, 1, 0)
speckle_pattern = torch.Tensor(speckle_pattern_total_intensity).unsqueeze(0)
C, H, W = speckle_pattern.shape
### Get Input Tensor by stacking the different speckle patterns: ###
input_tensor = torch.cat([speckle_pattern] * T, 0) + 5
input_tensor = input_tensor.unsqueeze(1)  # [T,H,W]->[T,C,H,W]
input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.cuda()
### Use Gimbaless: ###
reference_tensor = input_tensor[:,T//2:T//2+1]
(delta_x, delta_y, delta_rotation) = gimbaless_layer.forward(input_tensor, reference_tensor)
NUM_WARMUP_ITERS = 10
for _ in range(NUM_WARMUP_ITERS):
    input_tensor = input_tensor.cuda()
    (delta_x, delta_y, delta_rotation) = gimbaless_layer.forward(input_tensor, reference_tensor)
    input_tensor = input_tensor.cpu()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True, blocking=True)
finish = torch.cuda.Event(enable_timing=True, blocking=True)
input_tensor = input_tensor.cuda()
start.record()
(delta_x, delta_y, delta_rotation) = gimbaless_layer.forward(input_tensor, reference_tensor)
torch.cuda.synchronize()
finish.record()
total_time = start.elapsed_time(finish)
print(f"Total time to execute = {total_time} ms")


### TODO - test for row-wise multipllication/conv, column-wise multiplication/conv,






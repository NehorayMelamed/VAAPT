import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.autograd import Function
# import correlation_cuda

#TODO: yoav changed here: correlation cuda isnt working, this is a pytorch alternative
class Correlation(nn.Module):
    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        # self.pad_size = pad_size
        # self.kernel_size = kernel_size
        # self.stride1 = stride1
        # self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                           torch.arange(0, 2 * self.max_hdisp + 1)])
        hei, wid = in1.shape[2], in1.shape[3]
        #TODO: there must be a faster way of doing this than using a for loop
        output = torch.cat([torch.mean(in1 * in2_pad[:, :, dy:dy+hei, dx:dx+wid], 1, keepdim=True)
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))], 1)
        return output
#
# class CorrelationFunction(Function):
#
#     def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
#         super(CorrelationFunction, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply
#         # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)
#
#     def forward(self, input1, input2):
#         self.save_for_backward(input1, input2)
#
#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()
#             output = input1.new()
#
#             correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
#                 self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)
#
#         return output
#
#     def backward(self, grad_output):
#         input1, input2 = self.saved_tensors
#
#         with torch.cuda.device_of(input1):
#             rbot1 = input1.new()
#             rbot2 = input2.new()
#
#             grad_input1 = input1.new()
#             grad_input2 = input2.new()
#
#             correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
#                 self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)
#
#         return grad_input1, grad_input2
#
#
# class Correlation(Module):
#     def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
#         super(Correlation, self).__init__()
#         self.pad_size = pad_size
#         self.kernel_size = kernel_size
#         self.max_displacement = max_displacement
#         self.stride1 = stride1
#         self.stride2 = stride2
#         self.corr_multiply = corr_multiply
#
#     def forward(self, input1, input2):
#
#         result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)
#
#         return result
#

# import RapidBase.import_all
# from RapidBase.import_all import *
# from MagIC_I_args import *

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# # # #   U T I L I T I E S   # # # # #


def replication_padding():
    return False


def BN_epsilon():
    return 1.0e-3


def BN_momentum():
    return 0.1

# @torch.jit.script_method
# def reluM(x, M=torch.FloatTensor([8]), leak=torch.FloatTensor([0])):
#     return x.clamp(0.0, M.numpy().data)
#     # return x.max(leak * x).min(leak * (x - M) + M)

def reluM(x):
    return x.clamp(0)

# def reluMsymm(x, M=8.0, leak=0.0):
#     return x.clamp(-M, M)
#     # return x.max(leak * (x + M) - M).min(leak * (x - M) + M)

def reluMsymm(x):
    return x.clamp(-8, 8)

def load_keras_weight(filename, device=torch.device('cpu')):
    return torch.from_numpy(np.load(filename)).float().to(device)


# # # #   B A S I C   C O M P U T A T I O N   U N I T S   # # # # #


class Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2D, self).__init__()

        self.padder = nn.ReplicationPad2d(kernel_size // 2) if replication_padding() else nn.ZeroPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.bnrm = nn.BatchNorm2d(out_channels, eps=BN_epsilon(), momentum=BN_momentum())

    def forward(self, x):
        return reluM(self.bnrm(self.conv(self.padder(x))))

    def load_keras_params(self, drc, conv_name, bnrm_name):
        self.conv.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + conv_name + '__kernel_0.npy', device=self.conv.weight.device).permute(3, 2, 1, 0))
        self.bnrm.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__gamma_0.npy', device=self.bnrm.weight.device))
        self.bnrm.bias = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__beta_0.npy', device=self.bnrm.bias.device))


class Conv2D_Grouped(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, groups=1):
        super(Conv2D_Grouped, self).__init__()

        self.padder = nn.ReplicationPad2d(kernel_size // 2) if replication_padding() else nn.ZeroPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True, groups=groups)
        self.bnrm = nn.BatchNorm2d(out_channels, eps=BN_epsilon(), momentum=BN_momentum())

    def forward(self, x):
        return reluM(self.bnrm(self.conv(self.padder(x))))

    def load_keras_params(self, drc, conv_name, bnrm_name):
        self.conv.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + conv_name + '__kernel_0.npy', device=self.conv.weight.device).permute(3, 2, 1, 0))
        self.bnrm.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__gamma_0.npy', device=self.bnrm.weight.device))
        self.bnrm.bias = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__beta_0.npy', device=self.bnrm.bias.device))


class PointwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        self.bnrm = nn.BatchNorm2d(out_channels, eps=BN_epsilon(), momentum=BN_momentum())

    def forward(self, x):
        return reluM(self.bnrm(self.conv(x)))

    def load_keras_params(self, drc, conv_name, bnrm_name):
        self.conv.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + conv_name + '__kernel_0.npy', device=self.conv.weight.device).permute(3, 2, 1, 0))
        self.bnrm.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__gamma_0.npy', device=self.bnrm.weight.device))
        self.bnrm.bias = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__beta_0.npy', device=self.bnrm.bias.device))


class DepthwiseConv2D(nn.Module):

    def __init__(self, num_channels, kernel_size):
        super(DepthwiseConv2D, self).__init__()

        self.num_channels = num_channels

        self.padder = nn.ReplicationPad2d(kernel_size // 2) if replication_padding() else nn.ZeroPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(self.num_channels, self.num_channels, kernel_size, groups=self.num_channels, bias=True)
        self.bnrm = nn.BatchNorm2d(self.num_channels, eps=BN_epsilon(), momentum=BN_momentum())

    def forward(self, x):
        return reluMsymm(self.bnrm(self.conv(self.padder(x))))

    def load_keras_params(self, drc, conv_name, bnrm_name):
        self.conv.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + conv_name + '__depthwise_kernel_0.npy', device=self.conv.weight.device).permute(2, 3, 1, 0))
        self.bnrm.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__gamma_0.npy', device=self.bnrm.weight.device))
        self.bnrm.bias = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__beta_0.npy', device=self.bnrm.bias.device))


class DepthwiseConv1D(nn.Module):

    def __init__(self, num_channels, kernel_size):
        super(DepthwiseConv1D, self).__init__()

        self.num_channels = num_channels

        padding = (kernel_size // 2, kernel_size // 2, 0, 0)
        self.padder = nn.ReplicationPad2d(padding) if replication_padding() else nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(self.num_channels, self.num_channels, (1, kernel_size), groups=self.num_channels, bias=True)
        self.bnrm = nn.BatchNorm2d(self.num_channels, eps=BN_epsilon(), momentum=BN_momentum())

    def forward(self, x):
        return reluMsymm(self.bnrm(self.conv(self.padder(x))))

    def load_keras_params(self, drc, conv_name, bnrm_name):
        self.conv.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + conv_name + '__depthwise_kernel_0.npy', device=self.conv.weight.device).permute(2, 3, 1, 0))
        self.bnrm.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__gamma_0.npy', device=self.bnrm.weight.device))
        self.bnrm.bias = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__beta_0.npy', device=self.bnrm.bias.device))


class DepthwiseIIR(nn.Module):

    def __init__(self, num_channels):
        super(DepthwiseIIR, self).__init__()

        self.num_channels = num_channels

        self.w_curr_inp = nn.Parameter(torch.rand([1, num_channels, 1, 1]))
        self.w_prev_inp = nn.Parameter(torch.rand([1, num_channels, 1, 1]))
        self.w_prev_out = nn.Parameter(torch.rand([1, num_channels, 1, 1]))

        self.bnrm = nn.BatchNorm2d(self.num_channels, eps=BN_epsilon(), momentum=BN_momentum())

    def forward(self, x):
        y = self.w_curr_inp * x

        initial_cond_term = (self.w_prev_inp + self.w_prev_out) * x[:, :, 0:1]
        y[:, :, 0:1] = y[:, :, 0:1] + initial_cond_term

        feedback_coeff = self.w_prev_inp + self.w_prev_out * self.w_curr_inp
        feedback_term = torch.zeros((x.size(0), x.size(1), 1, x.size(3)), dtype=x.dtype, device=x.device)
        for row in range(1, x.size(2)):
            initial_cond_term = self.w_prev_out * initial_cond_term
            feedback_term = self.w_prev_out * feedback_term + x[:, :, (row-1):row]

            y[:, :, row:(row+1)] = y[:, :, row:(row+1)] + feedback_coeff * feedback_term + initial_cond_term

        return reluMsymm(self.bnrm(y))

    def forward_feedback(self, x):
        y = torch.zeros_like(x)

        y[:, :, 0:1] = (self.w_curr_inp + self.w_prev_inp + self.w_prev_out) * x[:, :, 0:1]
        for row in range(1, x.size(2)):
            y[:, :, row:(row+1)] = self.w_curr_inp * x[:, :, row:(row+1)] + \
                                   self.w_prev_inp * x[:, :, (row-1):row] + \
                                   self.w_prev_out * y[:, :, (row-1):row]

        return reluMsymm(self.bnrm(y))

    def load_keras_params(self, drc, conv_name, bnrm_name):
        # IIR not verified yet - this method currently a placeholder

        # self.conv.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + conv_name + '__kernel_0.npy', device=self.conv.weight.device).permute(3, 2, 1, 0))
        self.bnrm.weight = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__gamma_0.npy', device=self.bnrm.weight.device))
        self.bnrm.bias = nn.Parameter(load_keras_weight(drc + 'weight__' + bnrm_name + '__beta_0.npy', device=self.bnrm.bias.device))



class ShuffleConv(nn.Module):

    def __init__(self, num_channels, shuffle_split, kernel_size, num_states=0, parametric_steps=0):
        super(ShuffleConv, self).__init__()

        if np.mod(num_channels, shuffle_split):
            print('ShuffleConv error!')

        self.shuffle_split = shuffle_split
        self.shuffle_channels = num_channels // self.shuffle_split

        self.num_states = num_states
        self.parametric_steps = parametric_steps

        self.conv_2d = Conv2D_Grouped(num_channels, num_channels, kernel_size, groups=shuffle_split)

        self.output_transform = PointwiseConv(num_channels, num_channels)

    def forward(self, x):
        out = self.conv_2d(x)
        return self.output_transform(out)


# class ShuffleConv(nn.Module):
#
#     def __init__(self, num_channels, shuffle_split, kernel_size, num_states=0, parametric_steps=0):
#         super(ShuffleConv, self).__init__()
#
#         if np.mod(num_channels, shuffle_split):
#             print('ShuffleConv error!')
#
#         self.shuffle_split = shuffle_split
#         self.shuffle_channels = num_channels // self.shuffle_split
#
#         self.num_states = num_states
#         self.parametric_steps = parametric_steps
#
#         self.conv_2d = nn.ModuleList()
#         for sp in np.arange(self.shuffle_split):
#             self.conv_2d.append(Conv2D(self.shuffle_channels, self.shuffle_channels, kernel_size))
#
#         self.output_transform = PointwiseConv(num_channels + self.num_states, num_channels)
#
#         #TODO: this is for side-chain, get rid of this
#         if self.num_states > 0:
#             self.state_compression = PointwiseConv(18, self.num_states)
#             self.estimator = nn.Sequential(DepthwiseConv2D(self.num_states, 3),
#                                            PointwiseConv(self.num_states, self.num_states))
#             self.discriminator = nn.Sequential(DepthwiseConv2D(self.num_states, 3),
#                                                PointwiseConv(self.num_states, self.num_states))
#
#             self.selector = nn.ModuleList()
#             for ps in range(parametric_steps + 1):
#                 self.selector.append(nn.Sequential(DepthwiseConv2D(self.num_states, 3),
#                                                    PointwiseConv(self.num_states, self.num_states)))
#
#             self.state_vector = torch.empty((1, self.num_states), dtype=torch.float)
#
#     def forward(self, x, alfa=[0]):
#         out = torch.zeros_like(x)
#
#         #TODO: this is basically grouped convolution...simply use that instead
#         for sp in range(self.shuffle_split):
#             # curr_chan = sp*self.shuffle_channels + np.array(range(0, self.shuffle_channels))
#             # out[:, curr_chan] = self.conv_2d[sp](x[:, curr_chan])
#             start_index = int(sp*self.shuffle_channels)
#             stop_index = int(sp*self.shuffle_channels+self.shuffle_channels)
#             out[:, start_index:stop_index] = self.conv_2d[int(sp)](x[:, start_index:stop_index])
# 
#         if self.num_states > 0:
#             x_for_state = self.state_compression(x)
#             state_estimate = self.estimator(x_for_state)
# 
#             discrim = self.discriminator(x_for_state)
#             self.state_vector = (state_estimate * discrim).sum([2, 3]) / (discrim.sum([2, 3]) + 1.0e-5)
# 
#             # for chn in range(discrim.size(1)):
#             #     plt.imshow(discrim[0, chn].cpu().detach().numpy(), vmin=0, vmax=0.1)
#             #     plt.show()
# 
#             if len(alfa) > 1:
#                 selection = torch.zeros_like(x_for_state)
#                 for img in range(x_for_state.size(0)):
#                     selection[img:(img+1)] = self.selector[alfa[img]](x_for_state[img:(img+1)])
#             else:
#                 selection = self.selector[alfa[0]](x_for_state)
# 
#             state_map = selection * self.state_vector.unsqueeze(2).unsqueeze(3)
# 
#             out = torch.cat([out, state_map], 1)
# 
#         return self.output_transform(out)


# # # #   E N C O D E R S   /   D E C O D E R S   # # # # #


class BB_ShuffleConv(nn.Module):

    def __init__(self, active_inputs, num_states=0, parametric_steps=0):
        super(BB_ShuffleConv, self).__init__()

        self.input_transform = PointwiseConv(active_inputs, 18)

        self.shuffle_conv = nn.ModuleList()
        self.shuffle_conv.append(ShuffleConv(18, 3, 3))
        self.shuffle_conv.append(ShuffleConv(18, 3, 3))
        self.shuffle_conv.append(ShuffleConv(18, 3, 3, num_states, parametric_steps))
        self.shuffle_conv.append(ShuffleConv(18, 3, 3))

    def forward(self, x):
        x = self.input_transform(x)

        x = x + self.shuffle_conv[1](self.shuffle_conv[0](x))
        x = x + self.shuffle_conv[3](self.shuffle_conv[2](x))

        return x

    def load_keras_params(self, drc):
        self.input_transform.load_keras_params(drc, 'enc1x_conv0', 'batch_normalization')
        self.shuffle_conv[0].conv_2d[0].load_keras_params(drc, 'enc1x_conv1a', 'batch_normalization_1')
        self.shuffle_conv[0].conv_2d[1].load_keras_params(drc, 'enc1x_conv1b', 'batch_normalization_2')
        self.shuffle_conv[0].conv_2d[2].load_keras_params(drc, 'enc1x_conv1c', 'batch_normalization_3')
        self.shuffle_conv[0].output_transform.load_keras_params(drc, 'enc1x_conv1d', 'batch_normalization_4')

        self.shuffle_conv[1].conv_2d[0].load_keras_params(drc, 'enc1x_conv2a', 'batch_normalization_5')
        self.shuffle_conv[1].conv_2d[1].load_keras_params(drc, 'enc1x_conv2b', 'batch_normalization_6')
        self.shuffle_conv[1].conv_2d[2].load_keras_params(drc, 'enc1x_conv2c', 'batch_normalization_7')
        self.shuffle_conv[1].output_transform.load_keras_params(drc, 'enc1x_conv2d', 'batch_normalization_8')

        self.shuffle_conv[2].conv_2d[0].load_keras_params(drc, 'enc1x_conv3a', 'batch_normalization_9')
        self.shuffle_conv[2].conv_2d[1].load_keras_params(drc, 'enc1x_conv3b', 'batch_normalization_10')
        self.shuffle_conv[2].conv_2d[2].load_keras_params(drc, 'enc1x_conv3c', 'batch_normalization_11')
        self.shuffle_conv[2].output_transform.load_keras_params(drc, 'enc1x_conv3d', 'batch_normalization_12')

        self.shuffle_conv[3].conv_2d[0].load_keras_params(drc, 'enc1x_conv4a', 'batch_normalization_13')
        self.shuffle_conv[3].conv_2d[1].load_keras_params(drc, 'enc1x_conv4b', 'batch_normalization_14')
        self.shuffle_conv[3].conv_2d[2].load_keras_params(drc, 'enc1x_conv4c', 'batch_normalization_15')
        self.shuffle_conv[3].output_transform.load_keras_params(drc, 'enc1x_conv4d', 'batch_normalization_16')


class BB_DepthwiseSeparableConvolution(nn.Module):

    def __init__(self):
        super(BB_DepthwiseSeparableConvolution, self).__init__()

        in_channels = 18
        out_channels = 36

        self.expand = PointwiseConv(in_channels, out_channels)

        self.process_unit = nn.ModuleList()
        for pu in range(4):
            self.process_unit.append(nn.Sequential(
                DepthwiseConv2D(out_channels, 3),
                PointwiseConv(out_channels, out_channels)))

    def forward(self, x):
        x = self.expand(x)

        x = x + self.process_unit[1](self.process_unit[0](x))
        x = x + self.process_unit[3](self.process_unit[2](x))

        return x

    def load_keras_params(self, drc):
        self.expand.load_keras_params(drc, 'enc4x_conv0', 'batch_normalization_18')

        self.process_unit[0][0].load_keras_params(drc, 'enc4x_dwconv1', 'batch_normalization_19')
        self.process_unit[0][1].load_keras_params(drc, 'enc4x_conv1', 'batch_normalization_20')

        self.process_unit[1][0].load_keras_params(drc, 'enc4x_dwconv2', 'batch_normalization_21')
        self.process_unit[1][1].load_keras_params(drc, 'enc4x_conv2', 'batch_normalization_22')

        self.process_unit[2][0].load_keras_params(drc, 'enc4x_dwconv3', 'batch_normalization_23')
        self.process_unit[2][1].load_keras_params(drc, 'enc4x_conv3', 'batch_normalization_24')

        self.process_unit[3][0].load_keras_params(drc, 'enc4x_dwconv4', 'batch_normalization_25')
        self.process_unit[3][1].load_keras_params(drc, 'enc4x_conv4', 'batch_normalization_26')


class BB_DepthwiseAndSpatialSeparableConvolution(nn.Module):

    def __init__(self):
        super(BB_DepthwiseAndSpatialSeparableConvolution, self).__init__()

        in_channels = 36
        out_channels = 72

        self.expand = PointwiseConv(in_channels, out_channels)

        self.process_unit = nn.ModuleList()
        for pu in range(4):
            self.process_unit.append(nn.Sequential(
                DepthwiseIIR(out_channels),
                DepthwiseConv1D(out_channels, 5),
                PointwiseConv(out_channels, out_channels)))

    def forward(self, x):
        x = self.expand(x)

        x = x + self.process_unit[1](self.process_unit[0](x))
        x = x + self.process_unit[3](self.process_unit[2](x))

        return x

    def load_keras_params(self, drc):
        # BB_DepthwiseAndSpatialSeparableConvolution not verified yet - this method currently a placeholder

        self.expand.load_keras_params(drc, 'enc16x_conv0', 'batch_normalization_28')

        # self.process_unit[0][0].load_keras_params(drc, 'enc4x_dwconv1', 'batch_normalization_19')
        # self.process_unit[0][1].load_keras_params(drc, 'enc4x_conv1', 'batch_normalization_20')


class BB_QUICK(nn.Module):

    def __init__(self):
        super(BB_QUICK, self).__init__()

        in_channels = 36
        out_channels = 72

        self.expand = PointwiseConv(in_channels, out_channels)

        self.process_unit = nn.ModuleList()
        for pu in range(4):
            self.process_unit.append(nn.Sequential(
                DepthwiseConv2D(out_channels, 3),
                PointwiseConv(out_channels, out_channels)))

    def forward(self, x):
        x = self.expand(x)

        x = x + self.process_unit[1](self.process_unit[0](x))
        x = x + self.process_unit[3](self.process_unit[2](x))

        return x

    def load_keras_params(self, drc):
        self.expand.load_keras_params(drc, 'enc16x_conv0', 'batch_normalization_28')

        self.process_unit[0][0].load_keras_params(drc, 'enc16x_dwconv1', 'batch_normalization_29')
        self.process_unit[0][1].load_keras_params(drc, 'enc16x_conv1', 'batch_normalization_30')

        self.process_unit[1][0].load_keras_params(drc, 'enc16x_dwconv2', 'batch_normalization_31')
        self.process_unit[1][1].load_keras_params(drc, 'enc16x_conv2', 'batch_normalization_32')

        self.process_unit[2][0].load_keras_params(drc, 'enc16x_dwconv3', 'batch_normalization_33')
        self.process_unit[2][1].load_keras_params(drc, 'enc16x_conv3', 'batch_normalization_34')

        self.process_unit[3][0].load_keras_params(drc, 'enc16x_dwconv4', 'batch_normalization_35')
        self.process_unit[3][1].load_keras_params(drc, 'enc16x_conv4', 'batch_normalization_36')


class UpscaleAndCombine(nn.Module):

    def __init__(self, in_channels_low, in_channels_low_compressed, in_channels_hi, out_channels):
        super(UpscaleAndCombine, self).__init__()

        self.low_compress = PointwiseConv(in_channels_low, in_channels_low_compressed)

        kernel_size = 3
        self.conv_2d = Conv2D(in_channels_low_compressed + in_channels_hi, out_channels, kernel_size)

        self.upscale = nn.UpsamplingNearest2d(scale_factor=4)

    def forward(self, x_low, x_hi):
        x_low = self.low_compress(x_low)

        x_hi = torch.cat([self.upscale(x_low)[:, :, 0:x_hi.size(2), 0:x_hi.size(3)], x_hi], 1)

        return self.conv_2d(x_hi)


# # # #   MMM  AAA  GGG  III  CCC  iii   # # # # #


class MagIC_I_float(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, num_compressed_channels_1x=6, num_compressed_channels_4x=12, args=None, **kw):
        super(MagIC_I_float, self).__init__()

        if args is not None:
            for current_key,current_value in zip(args.keys(),args.values()):
                setattr(self, current_key, current_value)
        else:
            # set operation mode
            self.enable_batchnorms = kw['enable_batchnorms'] if ('enable_batchnorms' in kw) else False
            is_quick_mode = kw['is_quick_mode'] if ('is_quick_mode' in kw) else False
            num_states = 6 if (('has_state' in kw) and kw['has_state']) else 0
            parametric_steps = kw['parametric_steps'] if 'parametric_steps' in kw else 0

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.num_compressed_channels_1x = num_compressed_channels_1x
            self.num_compressed_channels_4x = num_compressed_channels_4x
            self.is_quick_mode = is_quick_mode
            self.num_states = num_states
            self.parametric_steps = parametric_steps


        # # set operation mode
        # self.enable_batchnorms = kw['enable_batchnorms'] if ('enable_batchnorms' in kw) else False
        # is_quick_mode = kw['is_quick_mode'] if ('is_quick_mode' in kw) else False
        # num_states = 6 if (('has_state' in kw) and kw['has_state']) else 0
        # parametric_steps = kw['parametric_steps'] if 'parametric_steps' in kw else 0

        # create encoder objects
        self.enc_1x = BB_ShuffleConv(in_channels, self.num_states, self.parametric_steps)
        self.enc_4x = BB_DepthwiseSeparableConvolution()
        self.enc_16x = BB_QUICK() if self.is_quick_mode else BB_DepthwiseAndSpatialSeparableConvolution()

        # create channel compression objects
        self.compress_1x = PointwiseConv(18, num_compressed_channels_1x)
        self.compress_4x = PointwiseConv(36, num_compressed_channels_4x)

        # create decoder objects
        self.dec_4x = UpscaleAndCombine(72, 12, num_compressed_channels_4x, 12)
        self.dec_1x = UpscaleAndCombine(12, 6, num_compressed_channels_1x, out_channels)

        # create pooling object
        self.pool = nn.MaxPool2d(4, ceil_mode=True)

        if not self.enable_batchnorms:
            self.disable_batchnorms()

        self.get_args_to_save_function = MagIC_I_get_args_to_save

    def forward(self, x):
        x = self.enc_1x(x)

        x_mid = self.enc_4x(self.pool(x))

        x_low = self.enc_16x(self.pool(x_mid))

        x_mid = self.dec_4x(x_low, self.compress_4x(x_mid))

        x = self.dec_1x(x_mid, self.compress_1x(x))

        return x

    # def parameters(self):
    #     if self.enable_batchnorms:
    #         return super(MagIC_I_float, self).parameters()
    #     else:
    #         non_bnrm_params = []
    #         for nm, prm in self.named_parameters():
    #             is_bnrm = 'bnrm' in nm
    #             is_state_module = any(xxx in nm for xxx in ['state_compression', 'estimator', 'discriminator', 'selector'])
    #             if (not is_bnrm) or is_state_module:
    #                 non_bnrm_params.append(prm)
    #         return non_bnrm_params

    def train(self, mode=True):
        super(MagIC_I_float, self).train(mode)
        if mode and not self.enable_batchnorms:
            for sub_module in self.modules():
                if type(sub_module).__name__ == 'BatchNorm2d':
                    sub_module.eval()

        # re-enable batchnorms for global statistics
        for sc in range(len(self.enc_1x.shuffle_conv)):
            if self.enc_1x.shuffle_conv[sc].num_states > 0:
                self.enc_1x.shuffle_conv[sc].state_compression.train(mode)
                self.enc_1x.shuffle_conv[sc].estimator.train(mode)
                self.enc_1x.shuffle_conv[sc].selector.train(mode)
                self.enc_1x.shuffle_conv[sc].discriminator.train(mode)

    def disable_batchnorms(self):
        for sub_module in self.modules():
            if type(sub_module).__name__ == 'BatchNorm2d':
                sub_module.weight = nn.Parameter(torch.ones_like(sub_module.weight))
                sub_module.bias = nn.Parameter(torch.zeros_like(sub_module.bias))
                sub_module.momentum = 0.0

        # re-enable batchnorms for global statistics
        for sc in range(len(self.enc_1x.shuffle_conv)):
            if self.enc_1x.shuffle_conv[sc].num_states > 0:
                for sub_module in self.enc_1x.shuffle_conv[sc].state_compression.modules():
                    if type(sub_module).__name__ == 'BatchNorm2d':
                        sub_module.momentum = 0.1
                for sub_module in self.enc_1x.shuffle_conv[sc].estimator.modules():
                    if type(sub_module).__name__ == 'BatchNorm2d':
                        sub_module.momentum = 0.1
                for sub_module in self.enc_1x.shuffle_conv[sc].selector.modules():
                    if type(sub_module).__name__ == 'BatchNorm2d':
                        sub_module.momentum = 0.1
                for sub_module in self.enc_1x.shuffle_conv[sc].discriminator.modules():
                    if type(sub_module).__name__ == 'BatchNorm2d':
                        sub_module.momentum = 0.1

    def load_keras_params(self, drc):
        self.enc_1x.load_keras_params(drc)
        self.enc_4x.load_keras_params(drc)
        self.enc_16x.load_keras_params(drc)

        self.compress_1x.load_keras_params(drc, 'enc1x_conv5', 'batch_normalization_17')
        self.compress_4x.load_keras_params(drc, 'enc4x_conv5', 'batch_normalization_27')

        self.dec_4x.low_compress.load_keras_params(drc, 'dec4x_conv0', 'batch_normalization_37')
        self.dec_4x.conv_2d.load_keras_params(drc, 'dec4x_conv1', 'batch_normalization_38')

        self.dec_1x.low_compress.load_keras_params(drc, 'dec1x_conv0', 'batch_normalization_39')
        self.dec_1x.conv_2d.load_keras_params(drc, 'dec1x_conv1', 'batch_normalization_40')


# # # #   MMM  AAA  GGG  III  CCC  iii  [full-resolution only]   # # # # #


class MagIC_I_float_top(nn.Module):

    def __init__(self, in_channels, out_channels, **kw):
        super(MagIC_I_float_top, self).__init__()

        # set operation mode
        num_states = 6 if (('has_state' in kw) and kw['has_state']) else 0

        # create processing objects
        self.enc_1x = BB_ShuffleConv(in_channels, num_states)
        self.compress = PointwiseConv(18, 6)
        self.combine = Conv2D(6, out_channels, 3)

    def forward(self, x):
        x = self.enc_1x(x)
        x = self.compress(x)
        x = self.combine(x)

        return x


# def MagIC_I_parameters():
#
#     in_channels = 1
#     out_channels = 1
#     num_compressed_channels_1x = 6
#     num_compressed_channels_4x = 12
#     is_quick_mode = True
#     has_state = True
#
#     if(has_state):
#         num_states = 6
#     else:
#         num_states = 0
#
#     parametric_steps = 0
#     enable_batchnorms = False
#
#     ### Into Dictionary: ###
#     args = EasyDict()
#     args['in_channels'] = in_channels
#     args['out_channels'] = out_channels
#     args['num_compressed_channels_1x'] = num_compressed_channels_1x
#     args['num_compressed_channels_4x'] = num_compressed_channels_4x
#     args['is_quick_mode'] = is_quick_mode
#     args['has_state'] = has_state
#     args['parametric_steps'] = parametric_steps
#     args['enable_batchnorms'] = enable_batchnorms
#     args['num_states'] = num_states
#
#     return args


def MagIC_I_get_args_to_save(model):
    dictionary_to_save = {}
    ### Get current network args as defined in above parameters() function: ###
    reference_args_dictionary = MagIC_I_parameters()
    for current_key in reference_args_dictionary.keys():
        dictionary_to_save[current_key] = getattr(model, current_key)
    return dictionary_to_save


def MagIC_I_init(args):

    return MagIC_I_float(args)
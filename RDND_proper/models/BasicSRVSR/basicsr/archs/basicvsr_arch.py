import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from RDND_proper.models.BasicSRVSR.basicsr.utils.registry import ARCH_REGISTRY
from RDND_proper.models.BasicSRVSR.basicsr.archs.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import PCDAlignment, TSAFusion
from RDND_proper.models.BasicSRVSR.basicsr.archs.spynet_arch import SpyNet

class BasicVSR_SimpleFFX2_1(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, number_of_input_channels=3, number_of_input_frames=5, number_of_output_channels=1, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.number_of_input_channels = number_of_input_channels
        self.number_of_input_frames = number_of_input_frames
        self.forward_trunk = ConvResidualBlocks(number_of_input_channels*number_of_input_frames, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 1, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, number_of_output_channels, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B,T,C,H,W = x.shape
        center_frame = x[:, T//2, :, :, :]
        x = x.view(B,T*C,H,W)
        current_frame_features = self.forward_trunk(x)

        out = torch.cat([current_frame_features], dim=1)
        out = self.lrelu(self.fusion(out))
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = F.interpolate(center_frame, scale_factor=2, mode='bicubic',
                             align_corners=False)  # TODO: maybe change to bicubic
        out += base
        return out


class BasicVSR_SimpleFFX2_MultipleOutputImages_1(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, number_of_input_channels=3, number_of_input_frames=5, number_of_output_channels=1, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.number_of_input_channels = number_of_input_channels
        self.number_of_input_frames = number_of_input_frames
        self.forward_trunk = ConvResidualBlocks(number_of_input_channels*number_of_input_frames, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 1, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, number_of_output_channels*number_of_input_frames, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B,T,C,H,W = x.shape
        center_frame = x[:, T//2, :, :, :]
        x = x.view(B,T*C,H,W)
        current_frame_features = self.forward_trunk(x)

        out = torch.cat([current_frame_features], dim=1)
        out = self.lrelu(self.fusion(out))
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        # out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = F.interpolate(x, scale_factor=2, mode='bicubic',
                             align_corners=False)  # TODO: maybe change to bicubic
        out += base

        return out

class BasicVSR_SimpleRecurrent_1(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, number_of_input_channels=3, number_of_input_frames=5, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.number_of_input_channels = number_of_input_channels
        self.number_of_input_frames = number_of_input_frames
        self.forward_trunk = ConvResidualBlocks(num_feat+3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 1, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        B, T, C, H, W = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, C, H, W) #[0,...,T-2]
        x_2 = x[:, 1:, :, :, :].reshape(-1, C, H, W)  #[1,...,T-1]

        flows_backward = self.spynet(x_1, x_2).view(B, T - 1, 2, H, W)
        flows_forward = self.spynet(x_2, x_1).view(B, T - 1, 2, H, W)  #SpyNet accept as inputs (reference_frame, frame_to_warp)

        return flows_forward, flows_backward

    def forward(self, input_tensor):
        #TODO: get rid of this and insert into forward loop
        flows_forward, flows_backward = self.get_flow(input_tensor)
        B, T, C, H, W = input_tensor.size()

        ### Loop Forward: ###
        output_tensors_list = []
        previous_frame_tensor = 0
        previous_output_features = 0
        current_frame_features = input_tensor.new_zeros(B, self.num_feat, H, W)  #TODO: understand exactly what this does. i think i need this simply zeros
        for i in range(0, T):
            ### Get Current Frame: ###
            current_frame_tensor = input_tensor[:, i, :, :, :]

            ### Get Previous Features To Concat To Final Reconstruction Head: ###
            # (*). in BasicVSR the output_tensors_list are built from the backward branch. i want to create something recurrent so i enter PREVIOUS output
            if i == 0:
                previous_output_tensor = F.interpolate(current_frame_tensor, scale_factor=4, mode='bilinear', align_corners=False)
            else:
                previous_output_tensor = output_tensors_list[i - 1]

            ### Get Current Flow & Warp Features: ###
            if i > 0:
                ### Calculate Flow: ###
                flow_forward = self.spynet(current_frame_tensor, previous_frame_tensor)
                flow_backward = self.spynet(previous_frame_tensor, current_frame_tensor)
                #TODO: add occlusions inference branch

                ### Warp Previous Features To Current Image According To Optical Flow: ###
                current_frame_features = flow_warp(previous_output_features, flow_forward.permute(0, 2, 3, 1))

            ### Concat current frame with warped features: ###
            current_frame_features = torch.cat([current_frame_tensor, current_frame_features], dim=1)
            current_frame_features = self.forward_trunk(current_frame_features)

            ### Assign Previous Features For Next Iteration: ###
            previous_output_features = current_frame_features

            ### Upsample: ###
            out = torch.cat([current_frame_features], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(current_frame_tensor, scale_factor=4, mode='bilinear', align_corners=False) #TODO: maybe change to bicubic
            out += base

            ### Assign current output to output tensors list for later use: ###
            output_tensors_list[i] = out
            previous_frame_tensor = current_frame_tensor

        return torch.stack(output_tensors_list, dim=1)


class BasicVSR_SimpleRecurrent_BiDirectional_1(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, number_of_input_channels=3, number_of_input_frames=5, num_feat=64, num_block=15, number_of_looks_forward_frames=2, spynet_path=None):
        super().__init__()


        # alignment
        self.spynet = SpyNet(spynet_path)

        # variables:
        self.num_feat = num_feat
        self.number_of_looks_forward_frames = number_of_looks_forward_frames
        self.number_of_input_channels = number_of_input_channels
        self.number_of_input_frames = number_of_input_frames
        self.forward_trunk = ConvResidualBlocks(num_feat+3, num_feat, num_block)
        self.backward_trunk = ConvResidualBlocks(num_feat+3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        B, T, C, H, W = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, C, H, W) #[0,...,T-2]
        x_2 = x[:, 1:, :, :, :].reshape(-1, C, H, W)  #[1,...,T-1]

        flows_backward = self.spynet(x_1, x_2).view(B, T - 1, 2, H, W)
        flows_forward = self.spynet(x_2, x_1).view(B, T - 1, 2, H, W)  #SpyNet accept as inputs (reference_frame, frame_to_warp)

        return flows_forward, flows_backward

    def forward(self, input_tensor):
        #TODO: get rid of this and insert into forward loop
        flows_forward, flows_backward = self.get_flow(input_tensor)
        B, T, C, H, W = input_tensor.size()

        ### Loop Forward: ###
        output_tensors_list = []
        previous_frame_tensor = 0
        previous_output_features = 0
        current_frame_features = input_tensor.new_zeros(B, self.num_feat, H, W)  #TODO: understand exactly what this does. i think i need this simply zeros
        for i in range(0, T-self.number_of_looks_forward_frames):
            ### Get Current Frame: ###
            current_frame_tensor = input_tensor[:, i, :, :, :]

            ### Get Previous Features To Concat To Final Reconstruction Head: ###
            # (*). in BasicVSR the output_tensors_list are built from the backward branch. i want to create something recurrent so i enter PREVIOUS output
            if i == 0:
                previous_output_tensor = F.interpolate(current_frame_tensor, scale_factor=4, mode='bilinear', align_corners=False)
            else:
                previous_output_tensor = output_tensors_list[i - 1]

            ### Get Current Flow & Warp Features: ###
            if i > 0:
                ### Calculate Flow: ###
                flow_forward = self.spynet(current_frame_tensor, previous_frame_tensor)
                flow_backward = self.spynet(previous_frame_tensor, current_frame_tensor)
                #TODO: add occlusions inference branch

                ### Warp Previous Features To Current Image According To Optical Flow: ###
                current_frame_features = flow_warp(previous_output_features, flow_forward.permute(0, 2, 3, 1))

            ### Loop Backward Up To Current Frame: ###
            backward_features = torch.zeros_like(current_frame_features)
            for j in np.arange(i+1, i-self.number_of_looks_forward_frames+1, -1):
                # remember, we're going backwards!, so temporally earlier frame is "later" when counting looking backwards
                LB_current_frame = input_tensor[:, j-1, :, :, :]  #current_frame
                LB_previous_frame = input_tensor[:, j, :, :, :]  #previous_frame
                flow_backward = self.spynet(LB_current_frame, LB_previous_frame)
                backward_features = flow_warp(backward_features, flow_backward.permute(0, 2, 3, 1))
                backward_features = torch.cat([LB_current_frame, backward_features], dim=1)
                backward_features = self.backward_trunk(backward_features)


            ### Concat current frame with warped features: ###
            current_frame_features = torch.cat([current_frame_tensor, current_frame_features], dim=1)
            current_frame_features = self.forward_trunk(current_frame_features)

            ### Assign Previous Features For Next Iteration: ###
            previous_output_features = current_frame_features

            ### Upsample: ###
            out = torch.cat([backward_features, current_frame_features], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(current_frame_tensor, scale_factor=4, mode='bilinear', align_corners=False) #TODO: maybe change to bicubic
            out += base

            ### Assign current output to output tensors list for later use: ###
            output_tensors_list[i] = out
            previous_frame_tensor = current_frame_tensor

        return torch.stack(output_tensors_list, dim=1)



@ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


@ARCH_REGISTRY.register()
class IconVSR(nn.Module):
    """IconVSR, proposed also in the BasicVSR paper
    """

    def __init__(self,
                 num_feat=64,
                 num_block=15,
                 keyframe_stride=5,
                 temporal_padding=2,
                 spynet_path=None,
                 edvr_path=None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride

        # keyframe_branch
        self.edvr = EDVRFeatureExtractor(temporal_padding * 2 + 1, num_feat, edvr_path)
        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(2 * num_feat + 3, num_feat, num_block)

        # reconstruction
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def pad_spatial(self, x):
        """ Apply padding spatially.

        Since the PCD module in EDVR requires that the resolution is a multiple
        of 4, we apply padding to the input LR images if their resolution is
        not divisible by 4.

        Args:
            x (Tensor): Input LR sequence with shape (n, t, c, h, w).
        Returns:
            Tensor: Padded LR sequence with shape (n, t, c, h_pad, w_pad).
        """
        n, t, c, h, w = x.size()

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        # padding
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def get_keyframe_feature(self, x, keyframe_idx):
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)

        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def forward(self, x):
        b, n, _, h_input, w_input = x.size()

        x = self.pad_spatial(x)
        h, w = x.shape[3:]

        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)  # last frame is a keyframe

        # compute flow and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            feat_prop = torch.cat([x_i, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]


class EDVRFeatureExtractor(nn.Module):

    def __init__(self, num_input_frame, num_feat, load_path):

        super(EDVRFeatureExtractor, self).__init__()

        self.center_frame_idx = num_input_frame // 2

        # extract pyramid features
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, 5, num_feat=64)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=8)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_input_frame, center_frame_idx=self.center_frame_idx)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, x):
        b, n, c, h, w = x.size()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, n, -1, h, w)
        feat_l2 = feat_l2.view(b, n, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, n, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(n):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        # TSA fusion
        return self.fusion(aligned_feat)

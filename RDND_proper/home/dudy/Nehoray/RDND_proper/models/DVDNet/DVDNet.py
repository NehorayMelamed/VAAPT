import torch
import torch.nn as nn
import numpy as np

from RapidBase.import_all import *
import RapidBase.TrainingCore.datasets



class DVDNet_dudy_BW(nn.Module):
    def __init__(self, temp_psz=5, mc_algo='DeepFlow'):
        super(DVDNet_dudy_BW, self).__init__()
        self.temporal_batch_size = temp_psz
        self.model_temporal = DVDnet_temporal_BW(temp_psz)
        self.model_spatial = DVDnet_spatial_BW()
        self.mc_algo = mc_algo
        #TODO: make special function to load weights to conform with general environment. probably one checkpoint dictionary which will have spatial and temporal keys

    def forward(self, inputs):
        ### Assign Inputs: ###
        input_frames = inputs[0]
        noise_std = inputs[1]

        ### Input should be [B,T,C,H,W] ###
        # init arrays to handle contiguous frames and related patches
        B, numframes, C, H, W = input_frames.shape
        center_frame_index = int((self.temporal_batch_size - 1) // 2)
        inframes = list()
        inframes_wrpd = np.empty((B, self.temporal_batch_size, H, W, C))
        denframes = torch.empty((B, numframes, C, H, W)).to(input_frames.device)
        output_dict = EasyDict()

        # build noise map from noise std---assuming Gaussian noise
        # noise_map = torch.Tensor([noise_std]).expand((B, 1, C, H, W))
        noise_map = torch.Tensor([noise_std]).expand((B, C, H, W)).to(input_frames.device)

        ### Spatially Denoise Frames: ###
        inframes = []
        for frame_index in range(self.temporal_batch_size):
            # real_frame_index = max(0, frame_index - center_frame_index)
            inframes.append(spatial_denoise(self.model_spatial, input_frames[:,frame_index,:,:,:], noise_map))

        ### Assign Spatial Part Results To Dict: ###
        output_dict.spatial_part_output_frames = inframes

        ### Convert To Numpy: ###
        inframes_wrpd[:, center_frame_index, :, :, :] = np.atleast_3d(variable_to_cv2_image(inframes[center_frame_index], conv_rgb_to_bgr=False))

        ### register frames w.r.t central frame and warp: ###
        for idx in range(self.temporal_batch_size):
            if not idx == center_frame_index:
                img_to_warp = np.atleast_3d(variable_to_cv2_image_BW(inframes[idx]))
                for batch_index in np.arange(B):
                    inframes_wrpd[batch_index, idx, :, :, :] = np.atleast_3d(align_frames(img_to_warp[batch_index], \
                                                      inframes_wrpd[batch_index, center_frame_index, :, :, :], \
                                                      mc_alg=self.mc_algo))

        # denoise with temporal model
        # temp_pszxHxWxC to temp_pszxCxHxW
        #TODO: accomodate batches
        inframes_t = normalize(inframes_wrpd.transpose(0, 1, 4, 2, 3))
        inframes_t = torch.from_numpy(inframes_t).contiguous().view((B, self.temporal_batch_size * C, H, W)).to(input_frames.device)
        denframes = temporal_denoise(self.model_temporal, inframes_t, noise_map)
        torch.cuda.empty_cache()

        ### Assign Temporal Part Results To Dict: ###
        output_dict.temporal_part_output_frame = denframes

        return output_dict


class DVDNet_dudy(nn.Module):
    def __init__(self, temp_psz=5, mc_algo='DeepFlow'):
        super(DVDNet_dudy, self).__init__()
        self.temporal_batch_size = temp_psz
        self.model_temporal = DVDnet_temporal(temp_psz)
        self.model_spatial = DVDnet_spatial()
        self.mc_algo = mc_algo
        #TODO: make special function to load weights to conform with general environment. probably one checkpoint dictionary which will have spatial and temporal keys

    def forward(self, inputs):
        ### Assign Inputs: ###
        input_frames = inputs[0]
        noise_std = inputs[1]

        ### Input should be [B,T,C,H,W] ###
        # init arrays to handle contiguous frames and related patches
        B, numframes, C, H, W = input_frames.shape
        center_frame_index = int((self.temporal_batch_size - 1) // 2)
        inframes = list()
        inframes_wrpd = np.empty((B, self.temporal_batch_size, H, W, C))
        denframes = torch.empty((B, numframes, C, H, W)).to(input_frames.device)
        output_dict = EasyDict()

        # build noise map from noise std---assuming Gaussian noise
        # noise_map = torch.Tensor([noise_std]).expand((B, 1, C, H, W))
        noise_map = torch.Tensor([noise_std]).expand((B, C, H, W)).to(input_frames.device)

        ### Spatially Denoise Frames: ###
        inframes = []
        for frame_index in range(self.temporal_batch_size):
            # real_frame_index = max(0, frame_index - center_frame_index)
            inframes.append(spatial_denoise(self.model_spatial, input_frames[:,frame_index,:,:,:], noise_map))

        ### Assign Spatial Part Results To Dict: ###
        output_dict.spatial_part_output_frames = inframes

        ### Convert To Numpy: ###
        inframes_wrpd[:, center_frame_index, :, :, :] = variable_to_cv2_image(inframes[center_frame_index], conv_rgb_to_bgr=False)

        ### register frames w.r.t central frame and warp: ###
        for idx in range(self.temporal_batch_size):
            if not idx == center_frame_index:
                img_to_warp = variable_to_cv2_image(inframes[idx], conv_rgb_to_bgr=False)
                for batch_index in np.arange(B):
                    inframes_wrpd[batch_index, idx, :, :, :] = align_frames(img_to_warp[batch_index], \
                                                      inframes_wrpd[batch_index, center_frame_index, :, :, :], \
                                                      mc_alg=self.mc_algo)

        # denoise with temporal model
        # temp_pszxHxWxC to temp_pszxCxHxW
        #TODO: accomodate batches
        inframes_t = normalize(inframes_wrpd.transpose(0, 1, 4, 2, 3))
        inframes_t = torch.from_numpy(inframes_t).contiguous().view((B, self.temporal_batch_size * C, H, W)).to(input_frames.device)
        denframes = temporal_denoise(self.model_temporal, inframes_t, noise_map)
        torch.cuda.empty_cache()

        ### Assign Temporal Part Results To Dict: ###
        output_dict.temporal_part_output_frame = denframes

        return output_dict



class DVDNet(nn.Module):
    def __init__(self, temp_psz=5, mc_algo='DeepFlow'):
        super(DVDNet, self).__init__()
        self.temporal_batch_size = temp_psz
        self.model_temporal = DVDnet_temporal(temp_psz)
        self.model_spatial = DVDnet_spatial()
        self.mc_algo = mc_algo
        #TODO: make special function to load weights to conform with general environment. probably one checkpoint dictionary which will have spatial and temporal keys

    def forward(self, input_frames, noise_std):
        ### Input should be [B,T,C,H,W] ###
        # init arrays to handle contiguous frames and related patches
        B, numframes, C, H, W = input_frames.shape
        center_frame_index = int((self.temporal_batch_size - 1) // 2)
        inframes = list()
        inframes_wrpd = np.empty((B, self.temporal_batch_size, H, W, C))
        denframes = torch.empty((B, numframes, C, H, W)).to(input_frames.device)

        # build noise map from noise std---assuming Gaussian noise
        # noise_map = torch.Tensor([noise_std]).expand((B, 1, C, H, W))
        noise_map = torch.Tensor([noise_std]).expand((B, C, H, W)).to(input_frames.device)

        ### Spatially Denoise Frames: ###
        inframes = []
        for frame_index in range(self.temporal_batch_size):
            # real_frame_index = max(0, frame_index - center_frame_index)
            inframes.append(spatial_denoise(self.model_spatial, input_frames[:,frame_index,:,:,:], noise_map))

        ### Convert To Numpy: ###
        inframes_wrpd[:, center_frame_index, :, :, :] = variable_to_cv2_image(inframes[center_frame_index], conv_rgb_to_bgr=False)

        ### register frames w.r.t central frame and warp: ###
        for idx in range(self.temporal_batch_size):
            if not idx == center_frame_index:
                img_to_warp = variable_to_cv2_image(inframes[idx], conv_rgb_to_bgr=False)
                for batch_index in np.arange(B):
                    inframes_wrpd[batch_index, idx, :, :, :] = align_frames(img_to_warp[batch_index], \
                                                      inframes_wrpd[batch_index, center_frame_index, :, :, :], \
                                                      mc_alg=self.mc_algo)

        # denoise with temporal model
        # temp_pszxHxWxC to temp_pszxCxHxW
        #TODO: accomodate batches
        inframes_t = normalize(inframes_wrpd.transpose(0, 1, 4, 2, 3))
        # inframes_t = torch.from_numpy(inframes_t).contiguous().view((1, self.temporal_batch_size * C, H, W)).to(input_frames.device)
        inframes_t = torch.from_numpy(inframes_t).contiguous().view((B, self.temporal_batch_size * C, H, W)).to(input_frames.device)
        denframes = temporal_denoise(self.model_temporal, inframes_t, noise_map)
        torch.cuda.empty_cache()
        return denframes


class DVDnet_spatial_BW(nn.Module):
    """ Definition of the spatial denoiser of DVDnet.
	Inputs of forward():
		x: array of input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, C, H, W], C (noise map for each channel)
	"""

    def __init__(self):
        super(DVDnet_spatial_BW, self).__init__()

        self.down_kernel_size = (2, 2)
        self.down_stride = 2
        self.kernel_size = 3
        self.padding = 1

        # RGB image
        self.num_input_channels = 6
        self.middle_features = 96
        self.num_conv_layers = 12
        self.down_input_channels = 4*1  # 3(RGB) * 4 = 12....for BW use 1*4=1
        self.downsampled_channels = 5 #4 channels for downsampled and shuffled input image + 1 cchannel for noise (which is strided, not shuffled)
        self.output_features = 4*1

        self.downscale = nn.Unfold(kernel_size=self.down_kernel_size, stride=self.down_stride)

        layers = []
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels,
                                out_channels=self.middle_features,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.middle_features,
                                    out_channels=self.middle_features,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.middle_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features,
                                out_channels=self.output_features,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))

        self.conv_relu_bn = nn.Sequential(*layers)
        self.pixelshuffle = nn.PixelShuffle(2)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):

        N, _, H, W = x.size()  # compute size of input

        # Downscale input using nn.Unfold
        x1 = self.downscale(x)
        x1 = x1.reshape(N, self.down_input_channels, H // 2, W // 2)

        # Concat downscaled input with downsampled noise map
        x1 = torch.cat((noise_map[:, :, ::2, ::2], x1), 1)

        # Conv + ReLU + BN
        x1 = self.conv_relu_bn(x1)

        # Upscale back to original resolution
        x1 = self.pixelshuffle(x1)

        # Residual learning
        x = x - x1
        return x


class DVDnet_temporal_BW(nn.Module):
    """ Definition of the temporal denoiser of DVDnet.
	Inputs of constructor:
		num_input_frames: int. number of frames to denoise
	Inputs of forward():
		x: array of input frames of dim [num_input_frames, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [1, C, H, W], C (noise map for each channel)
	"""

    def __init__(self, num_input_frames):
        super(DVDnet_temporal_BW, self).__init__()
        self.num_input_frames = num_input_frames
        self.num_input_channels = int((num_input_frames + 1) * 1)  # num_input_frames RGB frames + noisemap
        self.num_feature_maps = 96
        self.num_conv_layers = 4
        self.output_features = 12
        self.down_kernel_size = 5
        self.down_stride = 2
        self.down_padding = 2
        self.conv1x1_kernel_size = 1
        self.conv1x1_stride = 1
        self.conv1x1_padding = 0
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        self.down_conv = nn.Sequential(nn.Conv2d(in_channels=self.num_input_channels,
                                                 out_channels=self.num_feature_maps,
                                                 kernel_size=self.down_kernel_size,
                                                 padding=self.down_padding,
                                                 stride=self.down_stride,
                                                 bias=False),
                                       nn.BatchNorm2d(self.num_feature_maps),
                                       nn.ReLU(inplace=True))
        self.conv1x1 = nn.Conv2d(in_channels=self.num_feature_maps,
                                 out_channels=self.num_feature_maps,
                                 kernel_size=self.conv1x1_kernel_size,
                                 padding=self.conv1x1_padding,
                                 stride=self.conv1x1_stride,
                                 bias=False)
        layers = []
        for _ in range(self.num_conv_layers):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps,
                                    out_channels=self.num_feature_maps,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))

        self.block_conv = nn.Sequential(*layers)
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.num_feature_maps,
                                                out_channels=self.num_feature_maps,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride,
                                                bias=False),
                                      nn.BatchNorm2d(self.num_feature_maps),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=self.num_feature_maps,
                                                out_channels=self.output_features,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride,
                                                bias=False))
        self.pixelshuffle = nn.PixelShuffle(2)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):
        x1 = torch.cat((noise_map, x), 1)
        x1 = self.down_conv(x1)
        x2 = self.conv1x1(x1)
        x1 = self.block_conv(x1)
        x1 = self.out_conv(x1 + x2)
        x1 = self.pixelshuffle(x1)
        return x1



class DVDnet_spatial(nn.Module):
    """ Definition of the spatial denoiser of DVDnet.
	Inputs of forward():
		x: array of input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, C, H, W], C (noise map for each channel)
	"""

    def __init__(self):
        super(DVDnet_spatial, self).__init__()

        self.down_kernel_size = (2, 2)
        self.down_stride = 2
        self.kernel_size = 3
        self.padding = 1

        # RGB image
        self.num_input_channels = 6
        self.middle_features = 96
        self.num_conv_layers = 12
        self.down_input_channels = 12  # 3(RGB) * 4 = 12....for BW use 1*4=1
        self.downsampled_channels = 15
        self.output_features = 12

        self.downscale = nn.Unfold(kernel_size=self.down_kernel_size, stride=self.down_stride)

        layers = []
        layers.append(nn.Conv2d(in_channels=self.downsampled_channels,
                                out_channels=self.middle_features,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.middle_features,
                                    out_channels=self.middle_features,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.middle_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features,
                                out_channels=self.output_features,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False))

        self.conv_relu_bn = nn.Sequential(*layers)
        self.pixelshuffle = nn.PixelShuffle(2)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):

        N, _, H, W = x.size()  # compute size of input

        # Downscale input using nn.Unfold
        x1 = self.downscale(x)
        x1 = x1.reshape(N, self.down_input_channels, H // 2, W // 2)

        # Concat downscaled input with downsampled noise map
        x1 = torch.cat((noise_map[:, :, ::2, ::2], x1), 1)

        # Conv + ReLU + BN
        x1 = self.conv_relu_bn(x1)

        # Upscale back to original resolution
        x1 = self.pixelshuffle(x1)

        # Residual learning
        x = x - x1
        return x


class DVDnet_temporal(nn.Module):
    """ Definition of the temporal denoiser of DVDnet.
	Inputs of constructor:
		num_input_frames: int. number of frames to denoise
	Inputs of forward():
		x: array of input frames of dim [num_input_frames, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [1, C, H, W], C (noise map for each channel)
	"""

    def __init__(self, num_input_frames):
        super(DVDnet_temporal, self).__init__()
        self.num_input_frames = num_input_frames
        self.num_input_channels = int((num_input_frames + 1) * 3)  # num_input_frames RGB frames + noisemap
        self.num_feature_maps = 96
        self.num_conv_layers = 4
        self.output_features = 12
        self.down_kernel_size = 5
        self.down_stride = 2
        self.down_padding = 2
        self.conv1x1_kernel_size = 1
        self.conv1x1_stride = 1
        self.conv1x1_padding = 0
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        self.down_conv = nn.Sequential(nn.Conv2d(in_channels=self.num_input_channels,
                                                 out_channels=self.num_feature_maps,
                                                 kernel_size=self.down_kernel_size,
                                                 padding=self.down_padding,
                                                 stride=self.down_stride,
                                                 bias=False),
                                       nn.BatchNorm2d(self.num_feature_maps),
                                       nn.ReLU(inplace=True))
        self.conv1x1 = nn.Conv2d(in_channels=self.num_feature_maps,
                                 out_channels=self.num_feature_maps,
                                 kernel_size=self.conv1x1_kernel_size,
                                 padding=self.conv1x1_padding,
                                 stride=self.conv1x1_stride,
                                 bias=False)
        layers = []
        for _ in range(self.num_conv_layers):
            layers.append(nn.Conv2d(in_channels=self.num_feature_maps,
                                    out_channels=self.num_feature_maps,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=False))
            layers.append(nn.BatchNorm2d(self.num_feature_maps))
            layers.append(nn.ReLU(inplace=True))

        self.block_conv = nn.Sequential(*layers)
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.num_feature_maps,
                                                out_channels=self.num_feature_maps,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride,
                                                bias=False),
                                      nn.BatchNorm2d(self.num_feature_maps),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=self.num_feature_maps,
                                                out_channels=self.output_features,
                                                kernel_size=self.kernel_size,
                                                padding=self.padding,
                                                stride=self.stride,
                                                bias=False))
        self.pixelshuffle = nn.PixelShuffle(2)
        # Init weights
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, noise_map):
        x1 = torch.cat((noise_map, x), 1)
        x1 = self.down_conv(x1)
        x2 = self.conv1x1(x1)
        x1 = self.block_conv(x1)
        x1 = self.out_conv(x1 + x2)
        x1 = self.pixelshuffle(x1)
        return x1





import numpy as np
import cv2


# Parameters of the motion estimation algorithms
def warp_flow(img, flow):
    # Applies to img the transformation described by flow.
    assert len(flow.shape) == 3 and flow.shape[-1] == 2

    hf, wf = flow.shape[:2]
    # flow 		= -flow
    flow[:, :, 0] += np.arange(wf)
    flow[:, :, 1] += np.arange(hf)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def estimate_invflow(img0, img1, me_algo):
    # Estimates inverse optical flow by using the me_algo algorithm.
    # # # img0, img1 have to be uint8 grayscale
    assert img0.dtype == 'uint8' and img1.dtype == 'uint8'

    # Create estimator object
    if me_algo == "DeepFlow":
        of_estim = cv2.optflow.createOptFlow_DeepFlow()
    elif me_algo == "SimpleFlow":
        of_estim = cv2.optflow.createOptFlow_SimpleFlow()
    elif me_algo == "TVL1":
        of_estim = cv2.DualTVL1OpticalFlow_create()
    else:
        raise Exception("Incorrect motion estimation algorithm")

    # Run flow estimation (inverse flow)
    flow = of_estim.calc(img1, img0, None)
    #	flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def align_frames(img_to_align, img_source, mc_alg='DeepFlow'):
    # Applies to img_to_align a transformation which converts it into img_source.
    # Args:
    # 	img_to_align: HxWxC image
    # 	img_source: HxWxC image
    # 	mc_alg: selects between DeepFlow, SimpleFlow, and TVL1. DeepFlow runs by default.
    # Returns:
    # 	HxWxC aligned image

    # make sure images are uint8 in the [0, 255] range
    if img_source.max() <= 1.0:
        img_source = (img_source * 255).clip(0, 255)
    img_source = img_source.astype(np.uint8)
    if img_to_align.max() <= 1.0:
        img_to_align = (img_to_align * 255).clip(0, 255)
    img_to_align = img_to_align.astype(np.uint8)

    img0 = img_to_align[:, :, 0]
    img1 = img_source[:, :, 0]
    out_img = None

    # Align frames according to selection in mc_alg
    flow = estimate_invflow(img0, img1, mc_alg)

    # rectifier
    out_img = warp_flow(img_to_align, flow)

    return out_img


import numpy as np
import torch
import torch.nn.functional as F
from RDND_proper.models.DVDNet.dvdnet.utils import *


def temporal_denoise(model, noisyframe, sigma_noise):
    '''Encapsulates call to temporal model adding padding if necessary
	'''
    # Handle odd sizes
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2] % 2
    expanded_w = sh_im[-1] % 2
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
    sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

    # denoise
    out = torch.clamp(model(noisyframe, sigma_noise), 0., 1.)

    if expanded_h:
        out = out[:, :, :-1, :]
    if expanded_w:
        out = out[:, :, :, :-1]

    return out


def spatial_denoise(model, noisyframe, noise_map):
    '''Encapsulates call to spatial model adding padding if necessary
	'''
    # Handle odd sizes
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2] % 2
    expanded_w = sh_im[-1] % 2
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
    noise_map = F.pad(input=noise_map, pad=padexp, mode='reflect')

    # denoise
    out = torch.clamp(model(noisyframe, noise_map), 0., 1.)

    if expanded_h:
        out = out[:, :, :-1, :]
    if expanded_w:
        out = out[:, :, :, :-1]

    return out


def denoise_seq_dvdnet(seq, noise_std, temp_psz, model_temporal, model_spatial, mc_algo):
    # r"""Denoises a sequence of frames with DVDnet.
    #
    # Args:
    # 	seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
    # 	noise_std: Tensor. Standard deviation of the added noise
    # 	temp_psz: size of the temporal patch
    # 	model_temp: instance of the PyTorch model of the temporal denoiser
    # 	spatial_temp: instance of the PyTorch model of the spatial denoiser
    # 	mc_algo: motion compensation algorithm to apply
    # """
    # init arrays to handle contiguous frames and related patches
    numframes, _, C, H, W = seq.shape
    center_frame_index = int((temp_psz - 1) // 2)
    inframes = list()
    inframes_wrpd = np.empty((temp_psz, H, W, C))
    denframes = torch.empty((numframes, C, H, W)).to(seq.device)

    # build noise map from noise std---assuming Gaussian noise
    noise_map = noise_std.expand((1, C, H, W))

    for fridx in range(numframes):
        # load input frames
        # denoise each frame with spatial denoiser when appending
        if not inframes:
            # if list not yet created, fill it with temp_patchsz frames
            for idx in range(temp_psz):
                relidx = max(0, idx - center_frame_index)
                inframes.append(spatial_denoise(model_spatial, seq[relidx], noise_map))
        else:
            del inframes[0]
            relidx = min(numframes - 1, fridx + center_frame_index)
            inframes.append(spatial_denoise(model_spatial, seq[relidx], noise_map))

        # save converted central frame
        # OpenCV images are HxWxC uint8 images
        inframes_wrpd[center_frame_index] = variable_to_cv2_image(inframes[center_frame_index], conv_rgb_to_bgr=False)

        # register frames w.r.t central frame
        # need to convert them to OpenCV images first
        for idx in range(temp_psz):
            if not idx == center_frame_index:
                img_to_warp = variable_to_cv2_image(inframes[idx], conv_rgb_to_bgr=False)
                inframes_wrpd[idx] = align_frames(img_to_warp, \
                                                  inframes_wrpd[center_frame_index], \
                                                  mc_alg=mc_algo)
        # denoise with temporal model
        # temp_pszxHxWxC to temp_pszxCxHxW
        inframes_t = normalize(inframes_wrpd.transpose(0, 3, 1, 2))
        inframes_t = torch.from_numpy(inframes_t).contiguous().view((1, temp_psz * C, H, W)).to(seq.device)

        # append result to output list
        denframes[fridx] = temporal_denoise(model_temporal, inframes_t, noise_map)

    # free memory up
    del inframes
    del inframes_wrpd
    del inframes_t
    torch.cuda.empty_cache()

    # convert to appropiate type and return
    return denframes











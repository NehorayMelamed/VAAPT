"""
Definition of the FastDVDnet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class ShufflePixels(nn.Module):
    # Initialize this with a module
    def __init__(self, upscale_factor=2):
        super(ShufflePixels, self).__init__()
        self.upscale_factor = upscale_factor;

    # Elementwise sum the output of a submodule to its input
    def forward(self, x):
        return ShufflePixels_Function(x,self.upscale_factor)

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class CvBlock_2(CvBlock):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock_2, self).__init__(in_ch, out_ch)
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.PReLU(),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.PReLU()
		)


class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)


class InputCvBlock_2(InputCvBlock):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock_2, self).__init__(num_in_frames, out_ch)
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames * (3 + 1), num_in_frames * self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.PReLU(),
			nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.PReLU()
		)

class InputCvBlock_General(InputCvBlock):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

	def __init__(self, num_in_frames, number_of_input_channels, out_ch):
		super(InputCvBlock_General, self).__init__(num_in_frames, out_ch)
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames * (number_of_input_channels + 1), num_in_frames * self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.PReLU(),
			nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.PReLU()
		)

class InputCvBlock_General_NoNoiseMap(InputCvBlock):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

	def __init__(self, num_in_frames, number_of_input_channels, out_ch):
		super(InputCvBlock_General_NoNoiseMap, self).__init__(num_in_frames, out_ch)
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames * number_of_input_channels, num_in_frames * self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.PReLU(),
			nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.PReLU()
		)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock_2(DownBlock):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock_2, self).__init__(in_ch, out_ch)
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
			nn.PReLU(),
			CvBlock_2(out_ch, out_ch)
		)


class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock_2(UpBlock):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock_2, self).__init__(in_ch, out_ch)
		self.convblock = nn.Sequential(
			CvBlock_2(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock_2(OutputCvBlock):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock_2, self).__init__(in_ch, out_ch)
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.PReLU(),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
		)

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		# x = in1 - x
		x = in1 + x

		return x

class DenBlock_2(DenBlock):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(DenBlock_2, self).__init__(num_input_frames)
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock_2(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_2(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock_2(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock_2(in_ch=self.chs_lyr0, out_ch=3)

		self.reset_params()


class DenBlock_SuperResolution(DenBlock):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(DenBlock_SuperResolution, self).__init__(num_input_frames)
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock_2(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_2(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock_2(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock_2(in_ch=self.chs_lyr0, out_ch=16)

		self.reset_params()

class DenBlock_General(DenBlock):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=1):
		super(DenBlock_General, self).__init__(number_of_input_channels)
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock_General(num_in_frames=number_of_input_frames, number_of_input_channels=number_of_input_channels, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_2(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock_2(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock_2(in_ch=self.chs_lyr0, out_ch=number_of_output_channels)

		self.reset_params()

class DenBlock_General_NoNoiseMap(DenBlock):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=1):
		super(DenBlock_General_NoNoiseMap, self).__init__(number_of_input_channels)
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock_General_NoNoiseMap(num_in_frames=number_of_input_frames, number_of_input_channels=number_of_input_channels, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_2(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock_2(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock_2(in_ch=self.chs_lyr0, out_ch=number_of_output_channels)

		self.reset_params()

	def forward(self, in0, in1, in2):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		# x = in1 - x
		# x = in1 + x

		return x

class DenBlock_General_NoNoiseMap_NoLimitOnFrames(DenBlock):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=1):
		super(DenBlock_General_NoNoiseMap_NoLimitOnFrames, self).__init__(number_of_input_channels)
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.number_of_input_frames = number_of_input_frames
		self.number_of_input_channels = number_of_input_channels
		self.number_of_output_channels = number_of_output_channels

		self.inc = InputCvBlock_General_NoNoiseMap(num_in_frames=self.number_of_input_frames,
												   number_of_input_channels=self.number_of_input_channels, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_2(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock_2(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_2(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock_2(in_ch=self.chs_lyr0, out_ch=self.number_of_output_channels)
		self.channel_correction_layer = nn.Conv2d(in_channels=self.number_of_input_channels*self.number_of_input_frames,
												  out_channels=self.number_of_output_channels, kernel_size=3 ,padding=1)
		self.reset_params()

	def forward(self, input_frames, flag_center_residual=True, flag_input_residual=False, center_image_external=None):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		center_index = int(input_frames.shape[1]//2)
		x_center = input_frames[:,center_index:center_index+1,:,:]

		# x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		x0 = self.inc(input_frames)
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		# x = x
		# x = x_center - x
		if flag_input_residual:
			x = x + self.channel_correction_layer(input_frames)
		if flag_center_residual:
			x = x + x_center
		if center_image_external is not None:
			x = x + center_image_external

		return x


class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3)
		self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map = 0):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, m, :, :, :] for m in range(self.num_input_frames))
		noise_map = torch.randn(x.shape[0], 1, x.shape[3],x.shape[4])
		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x20, x21, x22, noise_map)

		return x

class FastDVDnet_6f(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=6):
		super(FastDVDnet_6f, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3)
		self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map = 0):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		n = x.shape[1]
		xs = []
		noise_map = torch.zeros(x.shape[0], 1, x.shape[3], x.shape[4]).to(x.device)
		for i in range(n-2):
			x0 = x[:, i, :, :, :]
			x1 = x[:, i+1, :, :, :]
			x2 = x[:, i+2, :, :, :]
			xs.append(self.temp1(x0, x1, x2, noise_map).squeeze(0))

		xs.insert(0,xs[0])
		xs.insert(n-1,xs[n-2])
		xs = torch.stack(xs).mean(axis=1)
		return xs
		# (x0, x1, x2, x3, x4, x5) = tuple(x[:, m, :, :, :] for m in range(self.num_input_frames))
		# noise_map = torch.zeros(x.shape[0], 1, x.shape[3], x.shape[4]).to(x.device)
		# # First stage
		# x20 = self.temp1(x0, x1, x2, noise_map)
		# x21 = self.temp1(x1, x2, x3, noise_map)
		# x22 = self.temp1(x2, x3, x4, noise_map)
		# x23 = self.temp1(x3, x4, x5, noise_map)
		#
		# #Second stage
		# x30 = self.temp2(x20, x21, x22, noise_map)
		# x31 = self.temp2(x21, x22, x23, noise_map)
		#
		# return (x30+x31)/2

class FastDVDnet_dudy(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=3, num_input_frames=5):
		super(FastDVDnet_dudy, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3)
		self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x_input):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		x, noise_map = x_input
		(x0, x1, x2, x3, x4) = tuple(x[:, self.in_channels*m:self.in_channels*m+self.in_channels, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.temp2(x20, x21, x22, noise_map)

		return x


class FastDVDnet_dudy_2(FastDVDnet_dudy):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=3, num_input_frames=5):
		super(FastDVDnet_dudy_2, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock_2(num_input_frames=3)
		self.temp2 = DenBlock_2(num_input_frames=3)
		# Init weights
		self.reset_params()


class FastDVDnet_dudy_2_Recursive(FastDVDnet_dudy_2):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=3, num_input_frames=5):
		super(FastDVDnet_dudy_2_Recursive, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock_2(num_input_frames=3)
		self.temp2 = DenBlock_2(num_input_frames=3)
		# Init weights
		self.reset_params()

		self.number_of_channels_from_last_iteration = 1  #simply the last clean results

	def forward(self, x_input):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4, x_last_clean) = tuple(x_input[:, self.in_channels*m:self.in_channels*m+self.in_channels, :, :]
									 				for m in range(self.num_input_frames + 1))

		# First stage
		x_last_clean = x_last_clean[:,0:1,:,:].clamp(-0.5,1.5)
		x20 = self.temp1(x0, x1, x2, x_last_clean)
		x21 = self.temp1(x1, x2, x3, x_last_clean)
		x22 = self.temp1(x2, x3, x4, x_last_clean)

		#Second stage
		x = self.temp2(x20, x21, x22, x_last_clean)

		return x


class FastDVDnet_SuperResolution(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=1, num_input_frames=5):
		super(FastDVDnet_SuperResolution, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock_General(number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=1)
		self.temp2 = DenBlock_General(number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=1)
		self.final_block = DenBlock_General(number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=16)
		self.pixel_shuffle = ShufflePixels(4)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x_input):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		x, noise_map = x_input
		noise_map = noise_map.squeeze(1)
		(x0, x1, x2, x3, x4) = tuple(x[:, self.in_channels*m:self.in_channels*m+self.in_channels, 0, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		#Second stage
		x = self.final_block(x20, x21, x22, noise_map)

		#Shuffle Pixels to get full resolution:
		x = self.pixel_shuffle(x)

		return x


class FastDVDnet_SuperResolution_NoNoiseMap(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=1, num_input_frames=5):
		super(FastDVDnet_SuperResolution_NoNoiseMap, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock_General_NoNoiseMap(number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=1)
		self.final_block = DenBlock_General_NoNoiseMap(number_of_input_frames=3, number_of_input_channels=1, number_of_output_channels=1)
		self.pixel_shuffle = ShufflePixels(4)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x_input):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		x, noise_map = x_input
		noise_map = noise_map.squeeze(1)
		(x0, x1, x2, x3, x4) = tuple(x[:, self.in_channels*m:self.in_channels*m+self.in_channels, 0, :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2)
		x21 = self.temp1(x1, x2, x3)
		x22 = self.temp1(x2, x3, x4)

		#Second stage
		x = self.final_block(x20, x21, x22)

		# #Shuffle Pixels to get full resolution:
		# x = self.pixel_shuffle(x)
		x += x2

		return x


class FastDVDnet_SuperResolution_NoNoiseMap_NoLimitOnFrames(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=1, num_input_frames=5):
		super(FastDVDnet_SuperResolution_NoNoiseMap_NoLimitOnFrames, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=5, number_of_input_channels=1, number_of_output_channels=1)
		self.final_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=5, number_of_input_channels=1, number_of_output_channels=16)
		self.pixel_shuffle = ShufflePixels(4)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x_input):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		x, noise_map = x_input
		# noise_map = noise_map.squeeze(1)
		(x0, x1, x2, x3, x4) = tuple(x[:, self.in_channels*m:self.in_channels*m+self.in_channels, 0, :, :] for m in range(self.num_input_frames))

		# # First stage
		# x20 = self.temp1(x0, x1, x2)
		# x21 = self.temp1(x1, x2, x3)
		# x22 = self.temp1(x2, x3, x4)

		#Second stage
		x = self.final_block(torch.cat((x0,x1,x2,x3,x4), 1)) * 1

		# #Shuffle Pixels to get full resolution:
		# x += x2
		x = self.pixel_shuffle(x)

		return x


# class FastDVDnet_Deblur_NoNoiseMap_NoLimitOnFrames(nn.Module):
# 	""" Definition of the FastDVDnet model.
# 	Inputs of forward():
# 		xn: input frames of dim [N, C, H, W], (C=3 RGB)
# 		noise_map: array with noise map of dim [N, 1, H, W]
# 	"""
#
# 	def __init__(self, in_channels=3, out_channels=3,  num_input_frames=5):
# 		super(FastDVDnet_Deblur_NoNoiseMap_NoLimitOnFrames, self).__init__()
# 		self.num_input_frames = num_input_frames
# 		self.in_channels = in_channels
# 		self.out_channels = out_channels
# 		# Define models of each denoising stage
# 		self.temp1 = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=self.num_input_frames,
# 			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
# 		self.final_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=self.num_input_frames,
# 			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
# 		self.pixel_shuffle = ShufflePixels(4)
# 		# Init weights
# 		self.reset_params()
#
# 	@staticmethod
# 	def weight_init(m):
# 		if isinstance(m, nn.Conv2d):
# 			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!
#
# 	def reset_params(self):
# 		for _, m in enumerate(self.modules()):
# 			self.weight_init(m)
#
# 	def forward(self, x):
# 		'''Args:
# 			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
# 			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
# 		'''
# 		(x0, x1, x2, x3, x4) = tuple(x[:, self.in_channels*m:self.in_channels*m+self.in_channels, 0, :, :] for m in range(self.num_input_frames))
#
# 		# First stage
# 		x20 = self.temp1(torch.cat((x0,x1,x2), 1))
# 		x21 = self.temp1(torch.cat((x1,x2,x3), 1))
# 		x22 = self.temp1(torch.cat((x2,x3,x4), 1))
#
# 		#Second stage
# 		# x = self.final_block(torch.cat((x0,x1,x2,x3,x4), 1)) * 1
# 		x = self.final_block(torch.cat((x20,x21,x22), 1)) * 1
#
# 		# #Shuffle Pixels to get full resolution:
# 		x -= x2
# 		# x = self.pixel_shuffle(x)
#
# 		return x

class FastDVDnet_Deblur_NoNoiseMap_NoLimitOnFrames(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=3, out_channels=3,  num_input_frames_per_block=5, total_number_of_frames=5, interim_number_of_channels=15):
		super(FastDVDnet_Deblur_NoNoiseMap_NoLimitOnFrames, self).__init__()
		self.num_input_frames_per_block = num_input_frames_per_block
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.total_number_of_frames = total_number_of_frames
		self.interim_number_of_channels = interim_number_of_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=self.num_input_frames_per_block,
			number_of_input_channels=self.in_channels, number_of_output_channels=self.interim_number_of_channels)
		self.final_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=self.num_input_frames_per_block,
			number_of_input_channels=self.interim_number_of_channels, number_of_output_channels= self.out_channels)
		# self.pixel_shuffle = ShufflePixels(4)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		(x0, x1, x2, x3, x4) = tuple(x[:, self.in_channels*m:self.in_channels*m+self.in_channels, 0, :, :] for m in range(self.total_number_of_frames))

		# First stage
		x20 = self.temp1(torch.cat((x0, x1, x2), 1), flag_center_residual=True, flag_input_residual=False, center_image_external=None)
		x21 = self.temp1(torch.cat((x1, x2, x3), 1), flag_center_residual=True, flag_input_residual=False, center_image_external=None)
		x22 = self.temp1(torch.cat((x2, x3, x4), 1), flag_center_residual=True, flag_input_residual=False, center_image_external=None)
		# Second stage
		# x = self.final_block(torch.cat((x0,x1,x2,x3,x4), 1)) * 1
		x = self.final_block(torch.cat((x20, x21, x22), 1), flag_center_residual=False, flag_input_residual=False, center_image_external=None) * 1

		# #Shuffle Pixels to get full resolution:
		x += x2
		# x = self.pixel_shuffle(x)

		return x


class FastDVDnet_SuperResolution_NoNoiseMap_NoLimitOnFrames(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, in_channels=3, out_channels=3,  num_input_frames_per_block=5, total_number_of_frames=5,
				 interim_number_of_channels=15, downsample_factor=2):
		super(FastDVDnet_SuperResolution_NoNoiseMap_NoLimitOnFrames, self).__init__()
		self.num_input_frames_per_block = num_input_frames_per_block
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.total_number_of_frames = total_number_of_frames
		self.interim_number_of_channels = interim_number_of_channels
		self.downsample_factor = downsample_factor
		# Define models of each denoising stage
		self.temp1 = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=self.num_input_frames_per_block,
			number_of_input_channels=self.in_channels, number_of_output_channels=self.interim_number_of_channels)
		self.final_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=self.num_input_frames_per_block,
			number_of_input_channels=self.interim_number_of_channels, number_of_output_channels=self.out_channels*(self.downsample_factor**2))
		self.pixel_shuffle = nn.PixelShuffle(downsample_factor)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		(x0, x1, x2, x3, x4) = tuple(x[:, self.in_channels*m:self.in_channels*m+self.in_channels, 0, :, :] for m in range(self.total_number_of_frames))

		# First stage
		x20 = self.temp1(torch.cat((x0,x1,x2), 1), flag_input_residual=True, center_image_external=None)
		x21 = self.temp1(torch.cat((x1,x2,x3), 1), flag_input_residual=True, center_image_external=None)
		x22 = self.temp1(torch.cat((x2,x3,x4), 1), flag_input_residual=True, center_image_external=None)
		#Second stage
		# x = self.final_block(torch.cat((x0,x1,x2,x3,x4), 1)) * 1
		x = self.final_block(torch.cat((x20,x21,x22), 1), flag_input_residual=True, center_image_external=None) * 1

		# #Shuffle Pixels to get full resolution:
		# #(1). idea number 1
		# x += x2
		# x = self.pixel_shuffle(x)
		#(2). idea number 2 drawing from BasicVSR way of doing things:
		x = self.pixel_shuffle(x)  #upscale using pixel shuffle
		base = F.interpolate(x2, scale_factor=self.downsample_factor, mode='bilinear', align_corners=False) #upscale using bilinear interpolation
		x = x + base

		return x


class FastDVDnet_Deblur(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
	"""
	def __init__(self, in_channels=3, out_channels=3,  num_input_frames=5):
		super(FastDVDnet_Deblur, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		self.out_channels = out_channels
		# Define models of each denoising stage
		self.temp1 = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=3,
			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
		self.temp2 = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=3,
			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')   #TODO: wait what?!?!?!

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		(x0, x1, x2, x3, x4) = tuple(x[:, m, :, :, :] for m in range(self.num_input_frames))
		# First stage
		x20 = self.temp1(torch.cat((x0,x1,x2), 1))
		x21 = self.temp1(torch.cat((x1,x2,x3), 1))
		x22 = self.temp1(torch.cat((x2,x3,x4), 1))

		#Second stage
		res = self.temp2(torch.cat((x20,x21,x22), 1)) #* 1

		res += x2

		return res


class FastDVDnet_Omer(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
	"""
	def __init__(self, in_channels=3, out_channels=3,  num_input_frames=9):
		super(FastDVDnet_Omer, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		self.out_channels = out_channels
		# Define models of each denoising stage
		self.first_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=3,
			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
		self.second_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=3,
																	   number_of_input_channels=self.in_channels,
																	   number_of_output_channels=self.out_channels)
		self.third_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=3,
																	   number_of_input_channels=self.in_channels,
																	   number_of_output_channels=self.out_channels)
		self.final_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=3,
			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
		# self.reset_params()
		# Init weights

	# @staticmethod
	# def weight_init(m):
	# 	if isinstance(m, nn.Conv2d):
	# 		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # TODO: wait what?!?!?!
	#
	# def reset_params(self):
	# 	for _, m in enumerate(self.modules()):
	# 		self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		(x0, x1, x2, x3, x4, x5, x6, x7, x8) = tuple(x[:, m, :, :, :] for m in range(self.num_input_frames))
		# First stage
		first_layer = []
		for i in range (self.num_input_frames - 2):
			x2i = self.first_block(torch.cat((x[:, i, :, :, :],x[:, i+1, :, :, :],x[:, i+2, :, :, :]),1))
			first_layer.append(x2i)

		#Second stage
		second_layer = []
		for i in range(self.num_input_frames - 2*2):
			x3i = self.second_block(torch.cat((first_layer[i], first_layer[i+1], first_layer[i+2]),1))
			second_layer.append(x3i)

		third_layer = []
		for i in range(self.num_input_frames - 2 * 3):
			x4i = self.third_block(torch.cat((second_layer[i], second_layer[i + 1], second_layer[i + 2]), 1))
			third_layer.append(x4i)

		c = torch.cat((third_layer[0], third_layer[1], third_layer[2]), 1)
		res = self.final_block(c)
		# res += x4

		return res

class FastDVDnet_Omer2(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
	"""
	def __init__(self, in_channels=3, out_channels=3,  num_input_frames=9):
		super(FastDVDnet_Omer2, self).__init__()
		self.num_input_frames = num_input_frames
		self.in_channels = in_channels
		self.out_channels = out_channels
		# Define models of each denoising stage
		self.first_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=5,
			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
		self.final_block = DenBlock_General_NoNoiseMap_NoLimitOnFrames(number_of_input_frames=3,
			number_of_input_channels= self.in_channels, number_of_output_channels= self.out_channels)
		# self.reset_params()
		# Init weights

	# @staticmethod
	# def weight_init(m):
	# 	if isinstance(m, nn.Conv2d):
	# 		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # TODO: wait what?!?!?!
	#
	# def reset_params(self):
	# 	for _, m in enumerate(self.modules()):
	# 		self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		(x0, x1, x2, x3, x4, x5, x6, x7, x8) = tuple(x[:, m, :, :, :] for m in range(self.num_input_frames))
		# First stage
		first_layer = []
		for i in range (0,5,2):
			x2i = self.first_block(torch.cat((x[:, i, :, :, :],x[:, i+1, :, :, :],x[:, i+2, :, :, :]
											  ,x[:, i+3, :, :, :],x[:, i+4, :, :, :]),1))
			first_layer.append(x2i)

		c = torch.cat((first_layer[0], first_layer[1], first_layer[2]), 1)
		res = self.final_block(c)
		# res += x4

		return res

# model = FastDVDnet_Deblur(in_channels=3, out_channels = 3,  num_input_frames=5)


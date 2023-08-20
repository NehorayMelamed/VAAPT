"""
Code adapted by amonod
Definition of the FastDVDnet model
Derived from original FastDVDnet code
This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch, batch_norm=True):
		super(CvBlock, self).__init__()
		if batch_norm:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(out_ch),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(out_ch),
				nn.ReLU(inplace=True)
			)
		else:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.ReLU(inplace=True)
			)

	def forward(self, x):
		return self.convblock(x)


class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, num_channels, out_ch, use_noise_map=True, batch_norm=True):
		super().__init__()
		self.interm_ch = 30
		if use_noise_map:
			input_channels = num_in_frames * (num_channels + 1)
		else:	# blind: no noise maps
			input_channels = num_in_frames * num_channels
		if batch_norm:
			self.convblock = nn.Sequential(
				nn.Conv2d(input_channels, num_in_frames * self.interm_ch,
						  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
				nn.BatchNorm2d(num_in_frames * self.interm_ch),
				nn.ReLU(inplace=True),
				nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(out_ch),
				nn.ReLU(inplace=True)
			)
		else:
			self.convblock = nn.Sequential(
				nn.Conv2d(input_channels, num_in_frames * self.interm_ch,
						  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.ReLU(inplace=True)
			)

	def forward(self, x):
		return self.convblock(x)


class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch, batch_norm=True):
		super(DownBlock, self).__init__()
		if batch_norm:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
				nn.BatchNorm2d(out_ch),
				nn.ReLU(inplace=True),
				CvBlock(out_ch, out_ch, batch_norm)
			)
		else:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
				nn.ReLU(inplace=True),
				CvBlock(out_ch, out_ch, batch_norm)
			)

	def forward(self, x):
		return self.convblock(x)


class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch, batch_norm=True):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch, batch_norm),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
			nn.PixelShuffle(2)
		)

	def forward(self, x):
		return self.convblock(x)


class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch, batch_norm=True):
		super(OutputCvBlock, self).__init__()
		if batch_norm:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(in_ch),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
			)
		else:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
			)

	def forward(self, x):
		return self.convblock(x)


class DenBlock(nn.Module):
	""" Definition of a U-net like denoising block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
		num_channels: int. number of channels per frame
		residual: bool. True: learn the residual. False: learn the denoised central image directly.
		noise_maps: str. None: blind network. 'per_frame': one noise map per frame. 'per_channel_per_frame': one noise map per channel per frame
		bias_free_conv: bool. True: do not learn biases in the convolution layers
		bias_free_BN: bool. True: do not learn bias in the Batch Norm operation (gamma=1, beta=0 ie only use feature mean and std)
	Inputs of forward():
		xn: input frames of dim [B, C, H, W], (C=3 RGB)
		noise_mapn: if any, array with noise map of dim [B, 1, H, W] / [B, C, H, W]
	"""

	def __init__(self, num_input_frames=3, num_channels=3, use_noise_map=True, batch_norm=True):
		super().__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.use_noise_map = use_noise_map

		self.inc = InputCvBlock(num_in_frames=num_input_frames, num_channels=num_channels, out_ch=self.chs_lyr0, use_noise_map=use_noise_map, batch_norm=batch_norm)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1, batch_norm=batch_norm)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2, batch_norm=batch_norm)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1, batch_norm=batch_norm)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0, batch_norm=batch_norm)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=num_channels, batch_norm=batch_norm)

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
			inX: Tensor, [B, C, H, W] in the [0., 1.] range
			noise_mapX: Tensor [B, 1, H, W] / [B, C, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if self.use_noise_map:
			x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		else:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1 + x2)
		# Estimation
		x = self.outc(x0 + x1)

		# residual
		x = in1 - x

		return x


class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5, num_channels=3, use_noise_map=True, batch_norm=True):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		self.num_channels = num_channels
		self.use_noise_map = use_noise_map
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3, num_channels=self.num_channels, use_noise_map=use_noise_map, batch_norm=batch_norm)
		self.temp2 = DenBlock(num_input_frames=3, num_channels=self.num_channels, use_noise_map=use_noise_map, batch_norm=batch_norm)
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
		'''Args:
			x: Tensor, [B, N*C, H, W] in the [0., 1.] range
			noise_map: Tensor [B, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		C = self.num_channels
		(x0, x1, x2, x3, x4) = tuple(x[:, C*m:C*(m+1), :, :] for m in range(self.num_input_frames))

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map)
		x21 = self.temp1(x1, x2, x3, noise_map)
		x22 = self.temp1(x2, x3, x4, noise_map)

		# Second stage
		x = self.temp2(x20, x21, x22, noise_map)

		return x

def pytorch_fastdvdnet_video_denoiser(video, model, noise_level, model_device, output_device=torch.device("cpu")):
	"""
	pytorch_denoiser
	Inputs:
		video            noisy image / image sequence
		model            FastDVDnet model
		noise_level      noise level to be used in the input noise map
		output_device           torch.device("cuda:X") or torch.device("cpu")
	Output:
		denoised_video   denoised video
	"""
	assert isinstance(model, FastDVDnet), 'invalid model specified'

	# image size
	assert len(video.shape) == 5, 'expected a 5D tensor (B, N, C, H, W)'
	B, N, C, H, W = video.shape
	assert C == 1 or C == 3, 'expected third dimension to be the channel dimension (= 1 / 3)'

	denoised_video = torch.empty((B, N, C, H, W), device=output_device)

	# pad by reflecting the first and last two images of the video because fastDVDnet takes 5 input frames
	video_pad = torch.empty((B, N + 4, C, H, W), device=video.device)
	video_pad[:, 2:-2] = video
	video_pad[:, :2] = video[:, 1:3].flip((1,))
	video_pad[:, -2:] = video[:, -3:-1].flip((1,))
	# pad to fit the 2 U-Net downsamplings if needed
	pad_H, pad_W = 4 - H % 4 if H % 4 else 0, 4 - W % 4 if W % 4 else 0
	padding = (0, pad_W, 0, pad_H)
	N2, H2, W2, = N + 4,  H + pad_H, W + pad_W
	video_pad = F.pad(video_pad.view(B*N2, C, H, W), padding, mode='reflect').view(B, N2, C, H2, W2)
	noise_map = noise_level * torch.ones((B, 1, H2, W2), device=model_device)
	for i in range(2, N2 - 2):
		# denoise each group of 5 frames
		frame_seq = video_pad[:, i-2: i+2+1].view(B, -1, H2, W2)
		denoised_video[:, i - 2] = model(frame_seq.to(model_device), noise_map)[..., :H, :W].to(output_device)

	return denoised_video


model.load_state_dict(torch.load(dvd_path)['params'])
        print('Loading pretrained VRT denoise 6 frames')
dvd_path = '/raid/Checkpoints/dvdnet/fastdvdnet_nodp.pth'
a = torch.load(dvd_path)
device = torch.device(8)
model = FastDVDnet().to(device)
x = torch.randn(1,17,3,192,192).to(device)
noise_level = 10
model.load_state_dict(a)
y = pytorch_fastdvdnet_video_denoiser(x , model, noise_level, device, device)

t = x[:, 0: 5]
B, N, C, H, W = x.shape
noise_map = noise_level * torch.ones((B, 1, H, W))

y = model(x, noise_map)


with open('/raid/scripts/readme.txt', 'w') as f:
    f.write('readme')

f = 'scenarios/new_DGX/pnp/pnp_try.py'

f = 'scenarios/new_DGX/pnp/pnp_try.py'

with open("filetowrite.py","wb") as fout:
    with open("filetoread.txt","rb") as fin:
         fout.write(fin.read())
import shutil

# use copyfile()
shutil.copyfile(f, '/raid/scripts/readme.txt')

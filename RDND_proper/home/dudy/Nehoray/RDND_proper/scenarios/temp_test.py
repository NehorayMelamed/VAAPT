import os

import torch
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
from numpy.random import *
from matplotlib.pyplot import *

import skimage
import imutils
import skimage.measure as skimage_measure
import skimage.metrics as skimage_metrics
# import image_registration

##### (1). Get clean image, add noise, and test some metrics: ######
### Get Original Image: ###
folder_name = r'C:\DataSets\Div2K\Flickr2K'
image_name = '000074.png'
image_full_filename = os.path.join(folder_name, image_name)
original_image = cv2.imread(image_full_filename)
original_image = cv2.cvtColor(original_image,cv2.COLOR_RGB2GRAY)
original_image = original_image/original_image.max()
print(original_image.max())
print(original_image.shape)

### Add some noise:: ###
SNR = 2
H,W = original_image.shape
noisy_image = original_image + 1/sqrt(SNR)*randn(H,W)
# noisy_image = convert_range(noisy_image, (0,1))
imshow_new(original_image)
imshow_new(noisy_image)

### Metrics: ###
### Denoise Metrics: ###
# PSNR, SSIM, MSSIM, MSE, SNR, Eigen-Values, Edge Metrics
metrics_SSIM = skimage_metrics.structural_similarity(original_image, noisy_image, win_size=9, multichannel=True)
metrics_NMSE = skimage_metrics.normalized_root_mse(original_image, noisy_image)
metrics_MSE = skimage_metrics.mean_squared_error(original_image, noisy_image)
metrics_SNR_linear = 1/metrics_MSE
metrics_SNR_dB = 10*log10(metrics_SNR_linear)
metrics_PSNR = skimage_metrics.peak_signal_noise_ratio(original_image, noisy_image)
metrics_VOI = skimage_metrics.variation_of_information((original_image*255).astype(np.uint8), (noisy_image*255).astype(np.uint8))
### Blur Metrics: ###
metric_contrast_measure_delta = contrast_measure_delta(original_image, noisy_image)
metric_blur_measurement = blur_measurement(original_image, noisy_image)

### Shifting: ###
original_image_displaced = shift_matrix_subpixel(original_image, 0, 10)
imshow_new(original_image)
imshow_new(original_image_displaced)

### blur image: ###
original_image_blurred = blur_image_motion_blur(original_image, 20, 20, N=10)
imshow_new(original_image)
imshow_new(original_image_blurred)

### Load Image Dataset: ###
train_dataset = TrainingCore.datasets.SimpleImage_Dataset(path='C:\DataSets\Div2K\DIV2K_train_HR', crop=None, use_ram=False)
### DataLoaders: ###
train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)




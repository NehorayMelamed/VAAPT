from RapidBase.import_all import *
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

















# video_to_images(os.path.join(noisy_images_folder, 'Beirut_BW_10.avi'))

##### (1). Get clean image, add noise, and test some metrics: ######
### Get Original Image: ###
folder_name = r'/home/mafat/DataSets/Example Videos/Beirut_BW_10'
image_name = '0000.png'
image_full_filename = os.path.join(folder_name, image_name)
original_image = cv2.imread(image_full_filename)
original_image = cv2.cvtColor(original_image,cv2.COLOR_RGB2GRAY)
original_image = original_image/original_image.max()
print(original_image.max())
print(original_image.shape)

### Add some noise:: ###
SNR = 2
H,W = original_image.shape
noisy_image = original_image + 1/sqrt(SNR)*np.random.randn(H,W)
# noisy_image = convert_range(noisy_image, (0,1))
imshow_new(original_image)
imshow_new(noisy_image)

### Get Image to Image metrics: ###
metrics_dict = get_metrics_image_pair(noisy_image, original_image)

### Compare Video Streams: ###
clean_images_folder = r'/home/mafat/DataSets/Example Videos/Beirut_BW_Clean'
noisy_images_folder = r'/home/mafat/DataSets/Example Videos/Beirut_BW_10'
output_dict_average, output_dict_history = get_metrics_video(clean_images_folder, noisy_images_folder)

### Video to images: ###
video_path = r'/home/mafat/DataSets/Example Videos/Beirut.mp4'
video_folder = os.path.split(video_path)[0]
video_name = os.path.split(video_path)[1]
video_stream = cv2.VideoCapture(video_path)
# video_stream.open()
counter = 0
while video_stream.isOpened():
    flag_frame_available, current_frame = video_stream.read()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    # current_frame = current_frame / current_frame.max()
    cv2.imwrite(os.path.join(video_folder, str.replace(video_name,'.mp4','_'+string_rjust(counter,4)+'.png')), current_frame)
    counter += 1


# ### Load Model: ###
# from RDND_proper.models.DVDNet.DVDNet import *
#
#
#
# ### Load Image Dataset: ###
# train_dataset = TrainingCore.datasets.SimpleImage_Dataset(path='C:\DataSets\Div2K\DIV2K_train_HR', crop=None, use_ram=False)
# ### DataLoaders: ###
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)




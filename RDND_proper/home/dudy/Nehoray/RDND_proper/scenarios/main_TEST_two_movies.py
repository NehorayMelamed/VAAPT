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





##########################################################################################################################################################
### Get {Original,Noisy} Sequence Of Images & Get Metrics - FOR MOVIES: ###
### Paths: ###
original_images_folder = r'/home/mafat/Pytorch_Checkpoints/Inference/ffdnet_gray/Dataset_MultipleImagesFromSingleImage_AddNoise/LWIR_video_10'
noisy_images_folder = r'/home/mafat/DataSets/Example Videos/LWIR_video_10'

### Compare Original-Noisy Video Streams: ###
output_dict_average_original_noisy, output_dict_history_original_noisy = get_metrics_video(original_images_folder, noisy_images_folder, np.inf)
### Create Seperate Folder: ###
create_folder_if_needed(os.path.join(original_images_folder, 'Results'))
### Save Graphs For All Metrics: ###
for key in output_dict_history_original_noisy.keys():
    try:
        plot_multiple([np.array(output_dict_history_original_noisy.inner_dict[key])],
                      legend_labels=['original-noisy: ' + decimal_notation(output_dict_average_original_noisy.inner_dict[key],2)],
                      super_title=key + ' over time', x_label='frame-counter', y_label=key)
        plt.savefig(os.path.join(original_images_folder, 'Results', key + ' over time.png'))
        plt.close()
    except:
        1
### Save Video Of Original-Result Side By Side: ###
side_by_side_movie_full_filename = os.path.join(original_images_folder, 'Results', 'Side_By_Side_Movie.avi')
fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
single_image_example = read_single_image_from_folder(original_images_folder)
original_image_filenames = read_image_filenames_from_folder(original_images_folder)
noisy_image_filenames = read_image_filenames_from_folder(noisy_images_folder)
H,W,C = single_image_example.shape
video_writer = cv2.VideoWriter(side_by_side_movie_full_filename, fourcc, 25.0, (2*W, H))
for frame_counter in np.arange(len(original_image_filenames)):
    original_estimate_frame = read_image_cv2(original_image_filenames[frame_counter])
    if original_estimate_frame.shape[2] == 1:
        original_estimate_frame = np.concatenate([original_estimate_frame,original_estimate_frame,original_estimate_frame], 2)

    noisy_frame = read_image_cv2(noisy_image_filenames[frame_counter])
    if noisy_frame.shape[2] == 1:
        concat_frame = np.concatenate([noisy_frame,noisy_frame,noisy_frame], 2)

    concat_frame = np.concatenate([noisy_frame, original_estimate_frame], 1)
    if concat_frame.shape[2] == 1:
        concat_frame = np.concatenate([concat_frame,concat_frame,concat_frame], 2)
    video_writer.write(concat_frame)

video_writer.release()
##########################################################################################################################################################








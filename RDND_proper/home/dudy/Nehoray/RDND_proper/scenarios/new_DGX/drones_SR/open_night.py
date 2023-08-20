import sys
sys.path.extend(['/home/omerl/rdnd', '/home/omerl/rdnd'])

from RapidBase.import_all import *
from torch.utils.data import DataLoader
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.callbacks import InferenceCallback_Denoising_Base
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.losses import Loss_Simple
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_RLSP_Recursive, PostProcessing_FastDVDNet
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.tensorboard_callbacks import TB_Denoising_Recursive
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import get_default_IO_dict

from os import path as osp
from pathlib import Path
from datetime import datetime

### Initialize Network State Dict: ###
######################################################################################################################
### Paths: ###
#(1). General Paths:
datasets_main_folder = '/raid/datasets'
videos_folder = osp.join(datasets_main_folder, 'night/videos')


def read_frames_from_binary_file_stream(f, number_of_frames_to_read=-1, number_of_frames_to_skip=0, params=None):
    ### Get parameters from params dict: ###
    utype = np.uint16
    roi = [320,640]
    ### Read Frames: ###
    Mov = np.fromfile(f, dtype=utype, count=number_of_frames_to_read * roi[0] * roi[1], offset=number_of_frames_to_skip*roi[0]*roi[1]*2)
    Mov_len = np.int(len(Mov) / roi[0] / roi[1])
    number_of_elements = Mov_len * roi[0] * roi[1]
    Mov = np.reshape(Mov[0:number_of_elements], [Mov_len, roi[0], roi[1]])
    Mov = Mov[:,2:,2:]  #Get rid of bad frame
    return Mov

save_dir = osp.join(datasets_main_folder, 'night/fixed_images')
movies = os.listdir(videos_folder)
num_frames = 2500
quantiles = (0.01, 0.99)

for movie in movies:
        file_name = os.path.join(videos_folder,movie)
        print ('Saving ', file_name)
        f = open(file_name, "rb")
        mov = torch.tensor(read_frames_from_binary_file_stream(f, num_frames).astype(np.float))
        q1 = mov[0].quantile(quantiles[0])
        q2 = mov[0].quantile(quantiles[1])
        input_tensor = scale_array_from_range(mov.clip(q1,q2), (q1,q2))
        curr_save_dir = osp.join(save_dir, movie)
        os.makedirs(curr_save_dir, exist_ok=True)
        for frame in range(num_frames):
            # save_image_torch(curr_save_dir, str(frame) + '.png', input_tensor[frame].unsqueeze(0))
            torchvision.utils.save_image(input_tensor[frame], osp.join(
                curr_save_dir, str(frame).zfill(8) + '.png'))
            if frame%200 == 1:
                print('Saving frame ', frame)
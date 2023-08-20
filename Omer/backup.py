import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from RDND_proper.RapidBase.Utils.IO.tic_toc import toc
from RapidBase.Utils.IO.Imshow_and_Plots import imshow_torch_video
from RapidBase.Utils.IO.tic_toc import tic
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import numpy_to_torch
# from RapidBase.import_all import *
from SHABACK_POC_NEW.Omer.to_neo.RVRT.models.rvrt_models import RVRT_deblur_16_frames


def deblur(model, video):
    video = video / video.max()
    video = video.unsqueeze(0)
    print("Running model")
    with torch.no_grad():
        output = model.test_video(video)
    print('Inference finished')
    return video, output[0]

def fast_deblur(model, video):
    video = video / video.max()
    video = video.unsqueeze(0)
    print("Running model")
    with torch.no_grad():
        output = model._forward(video)
    print('Inference finished')
    return video, output[0]


def video_read_video_to_numpy_tensor(input_video_path: str, frame_index_to_start, frame_index_to_end):
    if os.path.isfile(input_video_path) is False:
        raise FileNotFoundError("Failed to convert video to numpy array")
    print("Downloading target sub video to torch... it may take a few moments")

    # return skvideo.io.vread(input_video_path)
    ### Get video stream object: ###
    video_stream = cv2.VideoCapture(input_video_path)
    # video_stream.open()
    all_frames = []
    frame_index = 0
    while video_stream.isOpened():
        flag_frame_available, current_frame = video_stream.read()
        if frame_index < frame_index_to_start:
            frame_index += 1
            continue
        elif frame_index == frame_index_to_end:
            break

        if flag_frame_available:
            all_frames.append(current_frame)
            frame_index += 1
        else:
            break
    video_stream.release()
    # print("\n\n\n\npre stack")
    full_arr = np.stack(all_frames)
    # print("post stack")
    return full_arr


# Paths
model_deblur_path = 'to_neo/RVRT_deblur_shoval_train.py_blur20_TEST1_Step60000.tar'
# inference_path ='/raid/datasets/shabak/blur_cars/truck'
# output_path = '/raid/inference/RVRT_deblur_inference5'

# Setting Params
train_device = 0
test_device = 0
# num_frames = 16
# w = 600 #2048
# h =  800 #2048


#### Read video  ####
# video_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/netivay_israel/more/ch01_000000000230000004444.mp4"
# video_as_torch = video_read_video_to_numpy_tensor(video_path, 2, 18)
# video_as_torch = numpy_to_torch(video_as_torch)
# video_as_torch = video_as_torch.cuda()
# video_as_torch.to("cuda:0")
# video = read_video_default_torch(video_dir=video_path, size=(w,h), flag_convert_to_rgb=0)
# # video = video[:num_frames]
# imshow_torch_video(video_as_torch/255, FPS=5)


#### Load model  ####
model = RVRT_deblur_16_frames(train_device=train_device, test_device=test_device, checkpoint_path=model_deblur_path,
                              pretrained='self')
model = model.cuda()


# model = RVRT_deblur_16_frames(checkpoint_path=model_deblur_path,
#                               pretrained='self')
pt_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/ecc_segmantion_layer/input_vehicle.pt"
input_video_as_torch = torch.load(pt_path)
#### Read from images in directory #####
# path_directory = "/home/dudy/Nehoray/SHABACK_POC_NEW/Omer/data/video_blur_directory_1"
# video_create_movie_from_images_in_folder(images_path=path_directory, movie_name="blur_video.avi")
# video_as_torch = video_read_video_to_numpy_tensor("/home/dudy/Nehoray/SHABACK_POC_NEW/Omer/data/blur_video.avi", 1, 50)
# video_as_torch = numpy_to_torch(video_as_torch)
# video_as_torch = video_as_torch.cuda()
# input_video_as_torch = video_as_torch
#### Actually deblur ####
# input_video_as_torch = video_as_torch[:, :, 0:800, 0:600][:12]

tic()

# with torch.no_grad():
#     # x = model._forward(video_as_torch.unsqueeze(0))
input_tensor, output_tensor = deblur(model=model, video=input_video_as_torch)  #TODO: why is the output_tensor on the CPU?!?!?
print(1)
toc()

torch.save(output_tensor, "data/output_shaback_vehicle_deblur_tensor.pt")

plt.imshow(output_tensor[0].permute(1, 2, 0).cpu())
plt.show()

final_tensor = torch.cat([input_tensor.cpu().squeeze(0), output_tensor], -1)
imshow_torch_video(final_tensor, FPS=1)
# bla = RGB2BW(input_tensor.cpu().squeeze(0)-output_tensor)
# imshow_torch_video(bla, FPS=1)
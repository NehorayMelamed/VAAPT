import os

import cv2
import numpy as np
import torch

from PARAMETER import RVRT_deblur_shoval_train_py_blur20_TEST1_Step60000
# from RapidBase.import_all import *
from RVRT.models.rvrt_models import RVRT_deblur_16_frames


class Deblur:
    def __init__(self, input_torch_video,
                 model_deblur_path=RVRT_deblur_shoval_train_py_blur20_TEST1_Step60000,
                 train_device=0,
                 test_device=0):
        self.input_torch_video = input_torch_video.cuda()
        self.model_deblur_path = model_deblur_path
        self.train_device = train_device
        self.test_device = test_device

    def get_video_torch_deblur_result(self):
        model = RVRT_deblur_16_frames(train_device=self.train_device, test_device=self.test_device,
                                      checkpoint_path=self.model_deblur_path,
                                      pretrained='self')
        model = model.cuda()

        input_tensor, output_tensor = self.deblur(model=model, video=self.input_torch_video)

        # ToDo: u may do " output_tensor.permute(1, 2, 0).cpu() "  for output_tensor
        return input_tensor, output_tensor

    def show_video_deblur_result(self):
        raise NotImplemented

    def deblur(self, model, video):
        video = video / video.max()
        video = video.unsqueeze(0)
        print("Running model")
        with torch.no_grad():
            output = model.test_video(video)
        print('Inference finished')
        return video, output[0]

    def fast_deblur(self, model, video):
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
# inference_path ='/raid/datasets/shabak/blur_cars/truck'
# output_path = '/raid/inference/RVRT_deblur_inference5'

# Setting Params

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
# model = RVRT_deblur_16_frames(train_device=train_device, test_device=test_device, checkpoint_path=model_deblur_path,
#                               pretrained='self')
# model = model.cuda()


# model = RVRT_deblur_16_frames(checkpoint_path=model_deblur_path,
#                               pretrained='self')


#### Actually deblur ####
# tic()
# video_as_torch = video_as_torch[:,:,0:800,0:600]
#
#
# input_tensor, output_tensor = deblur(model=model, video=video_as_torch)  #TODO: why is the output_tensor on the CPU?!?!?
# print(1)
# toc()
#
# final_tensor = torch.cat([input_tensor.cpu().squeeze(0), output_tensor], -1)
# imshow_torch_video(final_tensor, FPS=1)
# bla = RGB2BW(input_tensor.cpu().squeeze(0)-output_tensor)
# imshow_torch_video(bla, FPS=1)

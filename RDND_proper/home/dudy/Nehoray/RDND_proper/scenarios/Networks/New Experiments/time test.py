import torch
import os
from NEW_TASHTIT.import_Tashtit import *

from torch.utils.data import DataLoader
from RapidBase.TrainingCore.training_utils import *
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.losses import Loss_Simple
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_RLSP_Recursive
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.tensorboard_callbacks import TB_Denoising_Recursive
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import get_default_IO_dict
import RapidBase.TrainingCore.training_utils
import RapidBase.TrainingCore

from RDND_proper.models.BasicSRVSR.basicsr.archs.basicvsr_arch import *
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import *
from RDND_proper.models.Restormer.Restormer import Restormer
from RDND_proper.models.SwinIR.SwinIR import SwinIR
from RDND_proper.models.FastDVDNet.models import *

def test_model_time(model, input_dim):
    input_tensor = torch.randn(input_dim).cuda()
    with torch.no_grad():
        output_tensor = model.forward((input_tensor,0))
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, blocking=True)
    finish = torch.cuda.Event(enable_timing=True, blocking=True)
    start.record()
    with torch.no_grad():
        output_tensor = model.forward((input_tensor,0))
    torch.cuda.synchronize()
    finish.record()
    total_time = start.elapsed_time(finish)
    print(f"Total time to execute = {total_time} ms")

H = 256
W = 256

### EDVR ###

# model_edvr = EDVR(
#         num_in_ch=3, num_out_ch=3,
#         num_feat=128,
#         num_frame=5, deformable_groups=8,
#         num_extract_block=5, num_reconstruct_block=40,
#         with_predeblur= False, with_tsa=True, hr_in=False).cuda()
#
#
# test_model_time(model_edvr, edvr_inp_dim)


model = FastDVDnet_Omer(in_channels=3, out_channels = 3,  num_input_frames=9).cuda()
input_dim = (1,5,3,H,W)
input_tensor = torch.randn(input_dim).cuda()
with torch.no_grad():
    output_tensor = model.forward(input_tensor)

# test_model_time(model_edvr, edvr_inp_dim)
# model_edvr = EDVR2(
#         num_in_ch=3, num_out_ch=3,
#         num_feat=128,
#         num_frame=5, deformable_groups=8,
#         num_extract_block=5, num_reconstruct_block=40,
#         with_predeblur= False, with_tsa=True, hr_in=False).cuda()
# edvr_inp_dim = (1,5,3,H,W)
# input_tensor = torch.randn(edvr_inp_dim).cuda()
# with torch.no_grad():
#     output_tensor = model_edvr.forward((input_tensor,0))





# model = Restormer().cuda()
# model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
#                         img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
#                         num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
#                         mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv').cuda()


# EDVR stats on 3x256x256 input frames:

# - Normal arch : ~500ns
# - Calc features only (befor PCD): ~
# - Calc features only (befor PCD): ~
# - Calc features only (befor PCD): ~

#
#
#
# def calc(x):
#     i,j = x
#     x = 2**i + j
#     y = 3**i + j
#
#     return x,y
#
# import multiprocessing
# pool = multiprocessing.Pool(5)
#
# out1, out2 = zip(*pool.map(calc, [(i,i+1) for i in range(10)]))
#
#
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True, blocking=True)
# finish = torch.cuda.Event(enable_timing=True, blocking=True)
# start.record()
# out1, out2 = zip(*pool.map(calc, range(0,10)))
# torch.cuda.synchronize()
# finish.record()
# total_time = start.elapsed_time(finish)
# print(f"Total time to execute = {total_time} ms")
#
#
#
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True, blocking=True)
# finish = torch.cuda.Event(enable_timing=True, blocking=True)
# start.record()
# for i in range (0,10):
#
#     out1, out2 = calc(i)
# torch.cuda.synchronize()
# finish.record()
# total_time = start.elapsed_time(finish)
# print(f"Total time to execute = {total_time} ms")
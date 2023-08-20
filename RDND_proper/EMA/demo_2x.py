import cv2
import math
import sys
import torch
import numpy as np
import argparse
from imageio import mimsave

'''==========import from our code=========='''
sys.path.append('.')
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ours', type=str)
args = parser.parse_args()
assert args.model in ['ours', 'ours_small'], 'Model not exists!'



def generate_intermediate_frame(img_path1, img_path2, output_dir):
    import cv2
    import sys
    import torch
    import numpy as np
    import shutil
    from imageio import mimsave

    '''==========import from our code=========='''
    sys.path.append('.')
    import config as cfg
    from Trainer import Model
    from benchmark.utils.padder import InputPadder

    '''==========Model setting=========='''
    TTA = True
    model_type = 'ours'  # You can change this as per your requirement
    if model_type == 'ours_small':
        TTA = False
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F = 16,
            depth = [2, 2, 2, 2, 2]
        )
    else:
        cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
        cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
            F = 32,
            depth = [2, 2, 2, 4, 4]
        )
    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()

    print(f'=========================Start Generating=========================')

    I0 = cv2.imread(img_path1)
    I2 = cv2.imread(img_path2)

    I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
    I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)

    padder = InputPadder(I0_.shape, divisor=32)
    I0_, I2_ = padder.pad(I0_, I2_)

    mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
    mimsave(f'{output_dir}/output_new_frame.gif', images, duration=3)

    # Save input images to the output directory
    shutil.copy(img_path1, output_dir)
    shutil.copy(img_path2, output_dir)

    print(f'=========================Done=========================')
    return True
# generate_intermediate_frame('example/img1.jpg', 'example/img2.jpg', 'example')


# '''==========Model setting=========='''
# TTA = True
# if args.model == 'ours_small':
#     TTA = False
#     cfg.MODEL_CONFIG['LOGNAME'] = 'ours_small'
#     cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
#         F = 16,
#         depth = [2, 2, 2, 2, 2]
#     )
# else:
#     cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
#     cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
#         F = 32,
#         depth = [2, 2, 2, 4, 4]
#     )
# model = Model(-1)
# model.load_model()
# model.eval()
# model.device()
#
#
# print(f'=========================Start Generating=========================')
#
# I0 = cv2.imread('example/img1.jpg')`
# I2 = cv2.imread('example/img2.jpg')
#
# I0_ = (torch.tensor(I0.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
# I2_ = (torch.tensor(I2.transpose(2, 0, 1)).cuda() / 255.).unsqueeze(0)
#
# padder = InputPadder(I0_.shape, divisor=32)
# I0_, I2_ = padder.pad(I0_, I2_)
#
# mid = (padder.unpad(model.inference(I0_, I2_, TTA=TTA, fast_TTA=TTA))[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
# images = [I0[:, :, ::-1], mid[:, :, ::-1], I2[:, :, ::-1]]
# mimsave('example/out_2x.gif', images, duration=3)
#
#
# print(f'=========================Done=========================')
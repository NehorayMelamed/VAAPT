from RapidBase.import_all import *

import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from RDND_proper.models.RAFT.core.raft import RAFT
from RDND_proper.models.RAFT.core.utils import flow_viz
from RDND_proper.models.RAFT.core.utils.utils import InputPadder

DEVICE = 'cuda'
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.atleast_3d(torch.from_numpy(img)).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()

args = EasyDict()
args.model = '/home/mafat/PycharmProjects/IMOD/models/RAFT/Checkpoints/raft-sintel.pth'
args.model = '/home/mafat/PycharmProjects/IMOD/models/RAFT/Checkpoints/raft-small.pth'
args.dropout = 0
args.alternate_corr = False
args.small = True
args.mixed_precision = False

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model))
model = model.module
model.to(DEVICE)
model.eval()

imfile1 = '/home/mafat/DataSets/DIV2K/DIV2K/DIV2K_train_HR_BW/0798.png'
image1 = load_image(imfile1)
image1 = BW2RGB(image1)
image2 = shift_matrix_subpixel_torch(image1, 15, 8)
image1 = crop_torch_or_numpy_batch(image1, (512,512))
image2 = crop_torch_or_numpy_batch(image2, (512,512))

padder = InputPadder(image1.shape)
image1, image2 = padder.pad(image1, image2)

with torch.no_grad():
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    # viz(image1, flow_up)

with torch.no_grad():
    images = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg'))

    images = sorted(images)
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        viz(image1, flow_up)






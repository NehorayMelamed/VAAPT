import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.models.archs.rrdbnet_arch import RRDBNet
from basicsr.models.archs.edvr_arch import EDVR
from RapidBase.import_all import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        '/home/mafat/PycharmProjects/IMOD/models/GitHub/BasicSR/experiments/pretrained_models/EDVR/EDVR_L_deblur_REDS_official-ca46bd8c.pth'  # noqa: E501
    )
    parser.add_argument(
        '--folder',
        type=str,
        default='/home/mafat/PycharmProjects/IMOD/models/GitHub/BasicSR/datasets/Set14/LRbicx4',
        help='input test image folder')
    parser.add_argument(
        '--device', type=str, default='cuda', help='Options: cuda, cpu.')
    args = parser.parse_args()

    device = torch.device(args.device)

    # set up model
    model = EDVR(
        num_in_ch=3, num_out_ch=3, num_feat=128, num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40, with_predeblur=True, with_tsa=True, hr_in=True)


    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)


    os.makedirs('results/ESRGAN', exist_ok=True)
    for idx, path in enumerate(
            sorted(glob.glob(os.path.join(args.folder, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                            (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        ### replicate baboon image for testing as input video frames: ###
        img = img.unsqueeze(1)
        img = torch.cat((img,img,img,img,img), 1)
        img = img[:,:,:,0:112,0:112]

        # inference
        with torch.no_grad():
            output = model(img)
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(f'results/ESRGAN/{imgname}_ESRGAN.png', output)


if __name__ == '__main__':
    main()

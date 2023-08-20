# VRT Models

import os.path as osp
import torch
from RDND_proper.models.network_vrt import VRT as net

base_path = '/home/mafat/Desktop/Omer'
vrt_path = osp.join(base_path, 'VRT')
checkpoints_path = osp.join(vrt_path, 'model_zoo', 'vrt')

def VRT_deblur(pretrained = True, pretrained_dataset = 'REDS'):

    model = net(upscale=1,
                img_size=[6, 192, 192],
                window_size=[6, 8, 8],
                depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10],
                embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                pa_frames=2,
                deformable_groups=16)

    if pretrained == True:
        if pretrained_dataset == 'REDS':
            model_path = osp.join(checkpoints_path, '007_VRT_videodeblurring_REDS.pth')
        elif pretrained_dataset == 'GoPro':
            model_path = osp.join(checkpoints_path, '006_VRT_videodeblurring_GoPro.pth')
        else:
            print ('No such dataset ' + pretrained_dataset)
            return
        model.load_state_dict(torch.load(model_path)['params'])

    return model


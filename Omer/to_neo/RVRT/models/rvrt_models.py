# RVRT Models

import torch
from ..models.network_rvrt import RVRT


# pretrained paths

denoise_davis_path = '/raid/Checkpoints/RVRT_cp/006_RVRT_videodenoising_DAVIS_16frames (1).pth'
sr_reds_path = '/raid/Checkpoints/RVRT_cp/001_RVRT_videosr_bi_REDS_30frames (1).pth'
deblur_dvd_path = '/raid/Checkpoints/RVRT_cp/004_RVRT_videodeblurring_DVD_16frames (1).pth'



class TestArgs():
    def __init__(self, tile=[30,192,192], tile_overlap=[2, 20, 20], scale=1, nonblind_denoising =True, window_size=[6,8,8]):
        #TODO check values of tile and tile_overlap
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.scale = scale
        self.nonblind_denoising = nonblind_denoising
        self.window_size = window_size


def RVRT_SR_30_frames(pretrained='REDS', checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2)):

    args = TestArgs(tile=[30,128, 128], scale=4, window_size=[2, 8, 8], nonblind_denoising=False)
    sr_model = RVRT(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                attention_window=[3, 3], cpu_cache_length=100, test_args=args,
                    train_device=train_device, test_device=test_device).to(train_device)
    #TODO add to rvrt test args, train,testdevice

    if pretrained=='REDS':
        sr_model.load_state_dict(torch.load(sr_reds_path)['params'])

    elif pretrained=='self':
        pass #TODO add self trained loading

    return sr_model

def RVRT_denoise_16_frames(pretrained='DAVIS', checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2)):

    args = TestArgs(tile=[30,192,192], scale=1, window_size=[2, 8, 8], nonblind_denoising=True)
    denoise_model = RVRT(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], nonblind_denoising=True, cpu_cache_length=100, test_args=args,
                    train_device=train_device, test_device=test_device).to(train_device)

    if pretrained=='DAVIS':
        denoise_model.load_state_dict(torch.load(denoise_davis_path)['params'])

    elif pretrained=='self':
        denoise_model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        return denoise_model

    return denoise_model

def RVRT_deblur_16_frames(pretrained='DVD', checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2)):

    args = TestArgs(tile=[30,192,192], scale=1, window_size=[2, 8, 8], nonblind_denoising=False)
    deblur_model = RVRT(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100, test_args=args,
                    train_device=train_device, test_device=test_device).to(train_device)

    if pretrained=='DVD':
        deblur_model.load_state_dict(torch.load(deblur_dvd_path)['params'])
        return deblur_model

    elif pretrained=='self':
        deblur_model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0')['model_state_dict'])
        return deblur_model

# compare gpu compute on both on another file
# VRT Models

import torch
import KAIR_light.models.network_vrt
import KAIR_light.models.network_vrt_new
from models.FastDVDNet.models import FastDVDnet_dudy, FastDVDnet_6f
from models.FastDVDNet.dvdnet_pnp import FastDVDnet, pytorch_fastdvdnet_video_denoiser
from RVRT.models.network_rvrt import RVRT

# model = RVRT(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
#                     depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
#                     inputconv_groups=[1, 3, 3, 3, 3, 3], deformable_groups=12, attention_heads=12,
#                     attention_window=[3, 3], cpu_cache_length=100).to(2)
#
# model2 = RVRT(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
#                     depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
#                     inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
#                     attention_window=[3, 3], cpu_cache_length=100).to(1)
#
# model_denoise = RVRT(upscale=1, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
#                     depths=[2, 2, 2], embed_dims=[192, 192, 192], num_heads=[6, 6, 6],
#                     inputconv_groups=[1, 3, 4, 6, 8, 4], deformable_groups=12, attention_heads=12,
#                     attention_window=[3, 3], nonblind_denoising=True, cpu_cache_length=100).to(3)

# model_path = '/raid/Checkpoints/RVRT_cp/006_RVRT_videodenoising_DAVIS_16frames (1).pth'
#
#
# # a  = torch.load(model_path)['params']
# # model_denoise.load_state_dict(a)
#
#
#
# # x = torch.randn(1, 30, 3, 64, 64).to(1)
# # x = torch.randn(1, 16, 3, 64, 64).to(1)
# x3 = torch.randn(1, 16, 4, 64,64).to(3)
#
# with torch.no_grad():
#     y  = model_denoise(x3)
#
#
# # with torch.no_grad():
# #     y  = model2(x)

class TestArgs():
    def __init__(self, tile=[12,192,192], tile_overlap=[2, 20, 20], scale=1, nonblind_denoising =True, window_size=[6,8,8]):
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.scale = scale
        self.nonblind_denoising = nonblind_denoising
        self.window_size = window_size

def VRT_denoise_deblur(checkpoints=['', ''], train_devices=[torch.device(i) for i in range(2)],
                 test_devices=[torch.device(i) for i in range(2,4)], pretrained=[False, False]):

    model = KAIR_light.models.network_vrt.VRT_denoise_deblur(checkpoints, train_devices,test_devices, pretrained)
    print('Loading pretrained VRT denoise-deblur 6 frames')
    return model

def VRT_denoise_SR(checkpoints=['', ''], train_devices=[torch.device(i) for i in range(2)],
                 test_devices=[torch.device(i) for i in range(2,4)], pretrained=[False, False], freeze=[False,False],
                   all_pretrained = False):

    model = KAIR_light.models.network_vrt.VRT_denoise_SR(checkpoints, train_devices,test_devices, pretrained, freeze, all_pretrained)
    print('Loading pretrained VRT denoise-deblur 6 frames')
    return model

def VRT_denoise_SR_2_losses(checkpoints=['', ''], train_devices=[torch.device(i) for i in range(2)],
                 test_devices=[torch.device(i) for i in range(2,4)], pretrained=[False, False], freeze=[False,False],
                   all_pretrained = False):

    model = KAIR_light.models.network_vrt.VRT_denoise_SR_2_losses(checkpoints, train_devices,test_devices, pretrained, freeze, all_pretrained)
    print('Loading pretrained VRT denoise-deblur 6 frames')
    return model

def VRT_SR_6_frames_checkpoint(pretrained=True, checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2), test_args=TestArgs(nonblind_denoising=False, window_size=[6,8,8])):

    model = KAIR_light.models.network_vrt.VRT(upscale=4, img_size=[6,64,64], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12,
                nonblind_denoising=False, train_device=train_device, test_device=test_device, test_args=test_args).to(train_device)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT denoise 6 frames')

    else:  # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])
        print('Loading pretrained VRT denoise 6 frames')
    return model

def VRT_SR_6_frames_checkpoint_img_sz_160(pretrained=True, checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2),denoise_prior=False, pnp=False):

    test_args = TestArgs(tile=[60,160,160], nonblind_denoising=False, scale=4)
    model = KAIR_light.models.network_vrt.VRT(upscale=4, img_size=[6,160,160], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12,
                nonblind_denoising=False, train_device=train_device, test_device=test_device, test_args=test_args, denoise_prior=denoise_prior, pnp=pnp).to(train_device)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])

    else:  # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])

    print('Loading pretrained VRT SR 6 frames')

    if denoise_prior:
        if pnp:
            model.dvdnet = FastDVDnet().to(train_device)
            dvd_path = '/raid/Checkpoints/dvdnet/fastdvdnet_nodp.pth'
            model.dvdnet.load_state_dict(torch.load(dvd_path))
        else:
            model.dvdnet = FastDVDnet_6f().to(train_device)
        # conv_first
        model.conv_first = torch.nn.Conv3d(3 * (1 + 2 * 4) + 1,
                                    120, kernel_size=(1, 3, 3), padding=(0, 1, 1)).to(train_device)
        model = model.to(train_device)

    return model

def VRT_deblur_6_frames_checkpoint(pretrained=True, checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2), test_args=TestArgs(nonblind_denoising=False)):

    model = KAIR_light.models.network_vrt.VRT(upscale=1, img_size=[6, 192, 192], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16,
                nonblind_denoising=False, train_device=train_device, test_device=test_device, test_args=test_args).to(train_device)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT denoise 6 frames')

    else:  # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])
        print('Loading pretrained VRT denoise 6 frames')
    return model

def VRT_denoise_6_frames_checkpoint(pretrained=True, checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2), test_args=TestArgs(),
                                    denoise_prior=False, pnp=False):

    model = KAIR_light.models.network_vrt.VRT(upscale=1, img_size=[6, 192, 192], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16,
                nonblind_denoising=True, train_device=train_device, test_device=test_device, test_args=test_args,
                                              denoise_prior=denoise_prior, pnp=pnp).to(train_device)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT denoise 6 frames')

    else: # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])
        print('Loading pretrained VRT denoise 6 frames')

    if denoise_prior:
        if pnp:
            model.dvdnet = FastDVDnet().to(train_device)
            dvd_path = '/raid/Checkpoints/dvdnet/fastdvdnet_nodp.pth'
            model.dvdnet.load_state_dict(torch.load(dvd_path))
        else:
            model.dvdnet = FastDVDnet_6f().to(train_device)
        model = model.to(train_device)

    return model

def VRT_denoise_dudy(pretrained=True, checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2), test_args=TestArgs(), noises_flags=[], use_conf=True):

    model = KAIR_light.models.network_vrt.VRT_dudy(upscale=1, img_size=[6, 192, 192], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16,
                nonblind_denoising=True, train_device=train_device, test_device=test_device, test_args=test_args, noises_flags=noises_flags, use_conf
                                                   =use_conf).to(train_device)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT denoise 6 frames')

    else: # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])
        print('Loading pretrained VRT denoise 6 frames')
    model.conv_last = torch.nn.Conv3d(96, 1+sum(noises_flags), kernel_size=(1, 3, 3), padding=(0, 1, 1))
    return model

def VRT_denoise_6_frames_8_GPUs(pretrained=True, checkpoint_path='', devices =[torch.device(i) for i in range (8,16)]):

    model = KAIR_light.models.network_vrt.VRT_6_frames_8_GPUs(upscale=1, img_size=[6, 192, 192], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16,
                nonblind_denoising=True, devices=devices)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT denoise 6 frames on 8 GPUs')

    else: # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:0")['model_state_dict'])
        print('Loading pretrained VRT denoise 6 frames on 8 GPUs')
    return model

def VRT_deblur_6_frames_8_GPUs(pretrained=True, checkpoint_path='', devices=[torch.device(i) for i in range (8)]):

    model = KAIR_light.models.network_vrt.VRT_6_frames_8_GPUs(upscale=1, img_size=[6, 192, 192], window_size=[6, 8, 8],
                         depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                         indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                         num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16, devices=devices)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT deblur 6 frames on 8 GPUs')

    return model

def VRT_deblur_6_frames(pretrained=True, checkpoint_path='', device=torch.device(0)):

    model = KAIR_light.models.network_vrt.VRT_6_frames(upscale=1, img_size=[6, 192, 192], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16).to(device)


    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT deblur 6 frames')

    return model

def VRT_denoise_6_frames(pretrained=True, checkpoint_path='', device=torch.device(0)):

    model = KAIR_light.models.network_vrt.VRT_6_frames(upscale=1, img_size=[6, 192, 192], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16,
                nonblind_denoising=True).to(device)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT denoise 6 frames')
    return model

def VRT_denoise_deblur_6_frames_2_GPUs(checkpoints, devices=[torch.device(i) for i in range (2)]):

    print('Loading pretrained VRT denoise 6 frames & pretrained VRT deblur 6 frames')
    model = KAIR_light.models.network_vrt.VRT_denoise_deblur_6_frames_2_GPUs(checkpoints, devices)
    return model

def VRT_denoise_deblur_6_frames_16_GPUs(checkpoints):

    print('Loading pretrained VRT denoise 6 frames & pretrained VRT deblur 6 frames')
    model = KAIR_light.models.network_vrt.VRT_denoise_deblur_6_frames_16_GPUs(checkpoints)
    return model

def VRT_denoise_denoise_6_frames_16_GPUs(checkpoints):

    print('Loading pretrained VRT denoise 6 frames & pretrained VRT denoise 6 frames')
    model = KAIR_light.models.network_vrt.VRT_denoise_denoise_6_frames_16_GPUs(checkpoints)
    return model

def VRT_denoise_11_frames(pretrained=True, checkpoint_path='', device=torch.device(0)):

    model = KAIR_light.models.network_vrt.VRT_11_frames(upscale=1, img_size=[11, 128, 128], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
                indep_reconsts=[9, 10], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=16,
                nonblind_denoising=True).to(device)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT denoise 11 frames')
    return model

def VRT_denoise_deblur_11_frames_2_GPUs(checkpoints, devices=[torch.device(i) for i in range (2)]):

    print('Loading pretrained VRT denoise 11 frames & pretrained VRT deblur 11 frames')
    model = KAIR_light.models.network_vrt.VRT_denoise_deblur_11_frames_2_GPUs(checkpoints, devices)
    return model

def VRT_SR_6_frames(pretrained=True, checkpoint_path=''):

    model = KAIR_light.models.network_vrt.VRT_6_frames(upscale=4, img_size=[6, 64, 64], window_size=[6, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
                indep_reconsts=[11, 12], embed_dims=[120, 120, 120, 120, 120, 120, 120, 180, 180, 180, 180, 180, 180],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=2, deformable_groups=12)

    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])
        print('Loading pretrained VRT SR 6 frames')
    return model




def VRT_FI_6_frames(pretrained=True, checkpoint_path='',
                train_device=torch.device(1), test_device=torch.device(2), test_args=TestArgs(nonblind_denoising=False)):

    model = KAIR_light.models.network_vrt_new.VRT(upscale=1, out_chans=3, img_size=[4, 192, 192], window_size=[4, 8, 8], depths=[8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4],
    indep_reconsts=[], embed_dims=[96, 96, 96, 96, 96, 96, 96, 120, 120, 120, 120],
    num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], pa_frames=0, train_device=train_device, test_device=test_device,
                                                  test_args=TestArgs(tile=[12,192,192], tile_overlap=[2, 20, 20], scale=1, nonblind_denoising =False, window_size=[4,8,8])).to(train_device)
    # TestArgs(tile=[0, 256, 256], tile_overlap=[0, 20, 20], scale=1, nonblind_denoising=False, window_size=[4, 8, 8]
    print('Loading pretrained VRT FI 6 frames')
    if pretrained == True:
        model.load_state_dict(torch.load(checkpoint_path)['params'])

    else:  # Loading our checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda:" + str(train_device))['model_state_dict'])

    return model
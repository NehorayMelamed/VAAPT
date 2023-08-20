"""
General useful functions that are all seamless at your fingertips.
@ Omer Leibovitch
"""
### General imports ###
from RapidBase.TrainingCore.trainer import *

### scp functions ###
import paramiko
import scp
import time
from os import path as osp

def get_DGX_paths(paths_file='/raid/omer_paths.txt'):

    with open(paths_file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        return lines[0], lines[1], lines[2]

def get_personal_info(info_file='/raid/omer_info.txt'):
    with open(info_file) as f:
        lines = f.readlines()
    lines = [line.rstrip().split('=')[-1] for line in lines]
    return lines[0], lines[1], lines[2], lines[3]

### Constants ###
DGX_image_path, DGX_text_path, DGX_temp_path = get_DGX_paths()
local_path, user, DGX_user, password = get_personal_info()

def get_IP_by_user(user: str):

    w = os.popen("w").read()
    users = w.split('\n')
    user_lines = [x for x in users if user in x]
    lines = user_lines[0].split(' ')
    lines = [x for x in lines if '.' in x]
    IP = lines[0]
    # return '10.10.12.5' #change this please
    return IP

def create_SSH_client(user: str, IP: str, password: str):
    """
     A helper function that return ssh client.

     Example:
         user = 'elisheva'
         password = 'mafat123'
    """
    port = 22
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(IP, port, user, password)
    return client

def copy_to_local(DGX_path: str, local_path: str,
                user: str, DGX_user: str, password: str, recursive: bool, print_info: bool = False):
    """
     Copy files from DGX to local machine through scp.

     Example (copy to local):
         DGX_path = '/raid/a.pt'
         local_path = '/home/'
         user = 'elisheva'
         DGX_user = 'omerl'
         password = 'mafat123'
         recursive = True

    Example (copy to self):
         DGX_path = '/raid/a.pt'
         local_path = '.'
         user = 'omerl'
         DGX_user = 'omerl'
         password = 'mafat123'
         recursive = True
    """
    IP = get_IP_by_user(DGX_user)
    if print_info:
        print('Copying to {} machine'.format(IP))
        print('Copy started from {} to {}'.format(DGX_path, local_path))
    start_time = time.time()
    ssh = create_SSH_client(user, IP, password)
    scp_client = scp.SCPClient(ssh.get_transport())
    scp_client.put(DGX_path, local_path, recursive=recursive)
    end_time = time.time()
    if print_info:
        print('Copy finished in {:.2f} seconds'.format(end_time - start_time))
    # Send a commend or a bash script:
        # command = './rdnd-venv/bin/python main.py'
        # bash_script = ''
        # stdin, stdout, stderr = ssh.exec_command(command)

def imshow_torch_seamless(image, flag_colorbar: bool = True, title_str: str = '' , print_info: bool = False,
                          close_after_next_imshow: bool = False):

    torch.save(image, DGX_image_path)
    with open(DGX_text_path, "w") as file:
        file.write(str(flag_colorbar) + '\n')
        file.write(title_str + '\n')
        file.write(str(close_after_next_imshow) + '\n')
        file.write('image')
    with open(DGX_temp_path, "w") as file:
        file.write(str(time.time()))

    recursive = False
    copy_to_local(DGX_image_path, local_path, user, DGX_user, password, recursive, print_info)
    copy_to_local(DGX_text_path, local_path, user, DGX_user, password, recursive, print_info)
    copy_to_local(DGX_temp_path, local_path, user, DGX_user, password, recursive, print_info)

def imshow_torch_video_seamless(input_tensor, number_of_frames=30, FPS=3, flag_BGR2RGB=True, frame_stride=1, print_info: bool = False,
                                close_after_next_imshow: bool = False):

    torch.save(input_tensor, '/raid/a.pt')
    with open(DGX_text_path, "w") as file:
        file.write(str(number_of_frames) + '\n')
        file.write(str(FPS) + '\n')
        file.write(str(flag_BGR2RGB) + '\n')
        file.write(str( frame_stride) + '\n')
        file.write(str(close_after_next_imshow) + '\n')
        file.write('video')
    with open(DGX_temp_path, "w") as file:
        file.write(str(time.time()))

    recursive = False
    copy_to_local(DGX_image_path, local_path, user, DGX_user, password, recursive, print_info)
    copy_to_local(DGX_text_path, local_path, user, DGX_user, password, recursive, print_info)
    copy_to_local(DGX_temp_path, local_path, user, DGX_user, password, recursive, print_info)

def move_images_dir(input_dir, output_dir, start_index, end_index):

    for i in range(start_index, end_index+1):
        img = os.path.join(input_dir,str(i).zfill(8)+'.png')
        command = 'mv {} {}'.format(img, output_dir)
        os.system(command)



# input_dir = '/raid/datasets/FixedThirdEye/r0_2'
# output_dir = '/raid/datasets/FixedThirdEye/004'
# start_index = 468
# end_index = 511
# move_images_dir(input_dir, output_dir, start_index, end_index)

# DGX_path = '/raid/datasets/FixedThirdEye/r0_3'
# DGX_path = '/raid/datasets/night/fixed_images'
# DGX_path = '/raid/old_checkpoints/Denoise_Dronse_SBNUC_harder.py/Denoise_Dronse_SBNUC_harder.py_TEST1_Step5325.tar'
# DGX_path = '/raid/datasets/REDS/val_sharp/000/crf25.mp4'
# DGX_path = '/raid/datasets/REDS/val_sharp/000/crf25.mp4'
DGX_path = '/raid/Pytorch_Checkpoints/2022-08-10/Inference/VRT_denoise_third_eye_inf.py/RapidBase__TrainingCore__datasets__DataSet_Videos_In_Folders_LoadCorruptedGT/third_fixed_images__validation/000000'

### Constants ###
DGX_image_path, DGX_text_path, DGX_temp_path = get_DGX_paths()
local_path, user, DGX_user, password = get_personal_info()

recursive = True
copy_to_local(DGX_path, local_path, user, DGX_user, password, recursive, True)


# a = torch.randn(3,160,160)
# input_tensor = read_image_default_torch()
# input_video_tensor = read_video_default_torch(size=256)
# input_video_tensor = read_video_default_torch(size=512, video_num=2)
# imshow_torch_video_seamless(input_video_tensor, number_of_frames=30, FPS=11, flag_BGR2RGB=True, frame_stride=1, print_info=True, close_after_next_imshow=True)
#
# imshow_torch_seamless(a, True, 'dudy3ee')
# imshow_torch_seamless(input_tensor, True, 'dudy3ee3', close_after_next_imshow=False)


### google-drive downloading###
import gdown

def download_google_drive():
    """
         Downloads an url or a directory from google-drive.
    """
    ### a file ###
    url = "https://drive.google.com/u/0/uc?id=1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd&export=download"
    output = '/raid/datasets/binning_datasets/DroneCrowd'
    gdown.download(url, output, quiet=False)
    ### a folder ###
    folder = 'https://drive.google.com/file/d/1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd/view'
    output = '/raid/datasets/binning_datasets/DroneCrowd'
    gdown.download_folder(url=folder, output=output, quiet=False)


### a folder ###
folder = 'https://drive.google.com/drive/folders/1ptezGuRhY6qJuaLSJNAbypgG9kDol172'
output = '.'
gdown.download_folder(url=folder, output=output, quiet=False)

from os import path as osp
import torch.nn as nn

def save_dataloader_binning(data_loader, sf: int, num_videos: int, num_frames: int,
                         output_dir: str, dataset_name: str):
    """
          Saves torch dataloader with specific sf.
    """

    down_sampling_layer = nn.AvgPool2d(sf)
    current_img = 0
    for batch in data_loader:
        print ('Saving image ', current_img +1)
        current_img_str = str(current_img).zfill(3)
        frames = batch['output_frames_original']
        save_dir = osp.join(output_dir, dataset_name, current_img_str)
        current_frame = 0
        for frame in range(num_frames):
            img = frames[0][frame]
            down_sampled_img = down_sampling_layer(img)
            current_frame_str = str(current_frame).zfill(8)
            save_image_torch(save_dir, current_frame_str + '.png', (down_sampled_img /down_sampled_img .max() * 255).clip(0,255)
                             ,flag_convert_bgr2rgb=True)
            current_frame += 1
        current_img += 1
        if current_img==num_videos:
            return



# sf = 2
# num_videos = 10
# num_frames = 30
# output_dir = '/raid/datasets/binning_datasets/'
# dataset_name = 'VisDroneX2'
# save_dataloader_binning(test_dataloader, sf, num_videos, num_frames, output_dir, dataset_name)


# clean_dir = '/raid/datasets/REDS/val_sharp'
# crf = 23
# flags = '-c:v libx264 -crf ' + str(crf)
# output_dir = '/raid/datasets/REDS/val_sharp_comp_' + str(crf)
#
# num_videos = len(os.listdir(clean_dir))
# os.makedirs(output_dir, exist_ok=True)
# for idx in range(num_videos):
#
# str_dir = str(idx).zfill(3)
# video_dir = osp.join(clean_dir, str_dir)
# output_video = video_dir + '.mp4'
# make_lossless_command = 'ffmpeg -framerate 10 -i {}/*.png {} {}'.\
#     format(video_dir, '000000%2d.png', flags,output_video)
# make_lossless_command = 'ffmpeg -framerate 10 -i /raid/datasets/REDS/val_sharp/000/000000%2d.png -c:v libx264 -crf 0 lossless.mp4'
# os.system(make_lossless_command)
# os.system()
#
# print('Video {} saved'.format(idx))


def generate_compressed_videos(clean_dir, output_dir, crf):

    num_videos = len(os.listdir(full_dir))
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(num_videos):


        str_dir = str(idx).zfill(4)
        video_dir = os.path.join(full_dir, str_dir, 'Concatenated_Outputs')
        output_video = 'output_video_' + str_dir + '.mp4'
        command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} {}'.format(
            fps, video_dir, flags, os.path.join(output_dir, output_video))
        os.system(command)
        print ('Video {} saved'.format(idx))


def generate_concat_videos(full_dir, output_dir, fps, flags):

    num_videos = len(os.listdir(full_dir))
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(num_videos):
        str_dir = str(idx).zfill(4)
        video_dir = os.path.join(full_dir, str_dir, 'Concatenated_Outputs')
        output_video = 'output_video_' + str_dir + '.mp4'
        command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} {}'.format(
            fps, video_dir, flags, os.path.join(output_dir, output_video))
        os.system(command)
        print ('Video {} saved'.format(idx))
# parameters


# inference_dir = '/raid/Pytorch_Checkpoints/2022-06-27/Inference/VRT_18_frames_reg_pelicanD_inf.py_better_concat/'
# save_dir = 'RapidBase__TrainingCore__datasets__DataSet_Videos_In_Folders_LoadCorruptedGT/fixed_images__validation/0000'
# full_dir = os.path.join(inference_dir, save_dir)
# # dir_name = 'dejeg_deblur_REDS_videos_15_FPS'
# output_dir = os.path.join('/raid/datasets/send_dudy/VRT_18_frames_pelicanD_inference_fixed')
# fps = 15
# # flags = '-c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p'
# flags = '-c:v libx264 -crf 0 -pix_fmt yuv420p'
# flags_new = '-c:v libx264 -crf 0'
# # flags_new = '-c:v huffyuv -c:a libmp3lame -b:a 320k'
# compressed_flags = '-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -minrate 1M -maxrate 1M -bufsize 2M'
#
# generate_concat_videos(full_dir, output_dir, fps, flags)
# # generate_single_video(full_dir, output_dir, 'a.avi',fps, flags)

def generate_single_video(full_dir, output_dir, output_name, fps, flags):
    video_dir = full_dir
    os.makedirs(output_dir, exist_ok=True)
    command = 'ffmpeg -framerate {} -pattern_type glob -i "{}/*.png" {} {}'.format(
        fps, video_dir, flags, os.path.join(output_dir, output_name))
    os.system(command)

def generate_dataset_videos(dataset_dir, output_dir, fps, flags):
    dataset_subdirs = os.listdir(dataset_dir)
    os.makedirs(output_dir, exist_ok=True)
    for subdir in dataset_subdirs:
        full_subdir = os.path.join(dataset_dir, subdir)
        generate_single_video(full_subdir, output_dir, subdir+'.mp4', fps, flags)
        print('Video {} saved'.format(subdir))

# # parameters
# inference_dir = '/raid/Pytorch_Checkpoints/2022-06-12/Inference/drones_SR_noised.py/'
# save_dir = 'RapidBase__TrainingCore__datasets__DataSet_Videos_In_Folders_LoadCorruptedGT_validation/16425'
# full_dir = os.path.join(inference_dir, save_dir)
# output_dir = '/raid/datasets/send/drones_SR_noised'
# fps = 15
# flags = '-c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p'
# compressed_flags = '-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -minrate 2M -maxrate 2M -bufsize 2M'
# generate_single_video(full_dir, output_dir, 'drones_SR_noised', fps, flags)
#
#
#
# dataset_dir = '/home/omerl/REDS_noised/PP5/Noisy/'
# output_dir = '/home/omerl/REDS_noised/PP5/Noisy_vid_2M'
# fps = 15
# flags = '-c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p'
# compressed_flags = '-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -minrate 2M -maxrate 2M -bufsize 2M'
# generate_dataset_videos(dataset_dir, output_dir, fps, compressed_flags)

# # # curr_dir = os.getcwd()
# videos_folder = '/raid/datasets/Mitspe/videos'
# save_folder = '/raid/datasets/Mitspe/new_images'
# videos = os.listdir(videos_folder)
# for video in videos:
#     video = video
#     video_stream = cv2.VideoCapture(os.path.join(videos_folder, video))
#     # video_stream.open()
#     video_num = video_num = video.split('_')[-1].split('.')[0]
#     save_sub_folder = os.path.join(save_folder, video_num)
#     os.makedirs(save_sub_folder, exist_ok=True)
#     counter = 0
#     while video_stream.isOpened():
#         flag_frame_available, current_frame = video_stream.read()
#         if flag_frame_available:
#             # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
#             image_name = str(counter).zfill(8) + '.png'
#             cv2.imwrite(os.path.join(save_sub_folder, image_name), current_frame)
#             counter += 1
#             print(counter)
#         else:
#             break
#         # video_stream.release()

def save_dataloader(dataloader, num_videos=20, num_frames=50, val=False, data_dir = '../Data4', dataset_name = 'dataset'):

    current_img = 0

    for batch in dataloader:
        # print ('Saving image ', current_img +1)
        outputs_dict = batch
        #
        current_img_str = str(current_img).zfill(3)
        #
        # dataset_kind = 'Noisy' # Save noisy images
        # if val:
        #     dataset_kind += '_val'
        #
        # batch_noisy = outputs_dict['output_frames_noisy']
        # save_dir = osp.join(data_dir, dataset_name, dataset_kind, current_img_str)
        # current_frame = 0
        # for frame in range(num_frames):
        #     img = batch_noisy[0][frame]
        #     current_frame_str = str(current_frame).zfill(8)
        #     save_image_torch(save_dir, current_frame_str + '.png', (img/img.max() * 255).clip(0,255),flag_convert_bgr2rgb=True)
        #     current_frame += 1
        #
        dataset_kind = 'Clean' # Save clean images
        if val:
            dataset_kind += '_val'

        batch_orig = outputs_dict['output_frames_original']
        save_dir = osp.join(data_dir, dataset_name, dataset_kind, current_img_str)
        current_frame = 0
        for frame in range(num_frames):
            img = batch_orig[0][frame]
            current_frame_str = str(current_frame).zfill(8)
            save_image_torch(save_dir, current_frame_str + '.png', (img/img.max() * 255).clip(0,255),flag_convert_bgr2rgb=True)
            current_frame += 1

        current_img += 1

        if current_img == num_videos:
            return



# current_img = 0
#
# for batch in dataloader:
#     # print ('Saving image ', current_img +1)
#     outputs_dict = batch
#     #
#     current_img_str = str(current_img).zfill(3)
#     #
#     # dataset_kind = 'Noisy' # Save noisy images
#     # if val:
#     #     dataset_kind += '_val'
#     #
#     # batch_noisy = outputs_dict['output_frames_noisy']
#     # save_dir = osp.join(data_dir, dataset_name, dataset_kind, current_img_str)
#     # current_frame = 0
#     # for frame in range(num_frames):
#     #     img = batch_noisy[0][frame]
#     #     current_frame_str = str(current_frame).zfill(8)
#     #     save_image_torch(save_dir, current_frame_str + '.png', (img/img.max() * 255).clip(0,255),flag_convert_bgr2rgb=True)
#     #     current_frame += 1
#     #
#     dataset_kind = 'Clean' # Save clean images
#     if val:
#         dataset_kind += '_val'
#
#     batch_orig = outputs_dict['output_frames_original']
#     save_dir = osp.join(data_dir, dataset_name, dataset_kind, current_img_str)
#     current_frame = 0
#     for frame in range(num_frames):
#         img = batch_orig[0][frame]
#         current_frame_str = str(current_frame).zfill(8)
#         save_image_torch(save_dir, current_frame_str + '.png', (img/img.max() * 255).clip(0,255),flag_convert_bgr2rgb=True)
#         current_frame += 1
#
#     current_img += 1
#
#     if current_img == num_videos:
#         return
#
#
#
#
#
#
# save_dataloader(test_dataloader, data_dir = '/raid/datasets/binning_datasets/VisDroneX8',
#                 dataset_name='sequences', num_frames=30)
#
#
# if __name__ == '__main__':
#
# # curr_dir = os.getcwd()
# videos_folder = '/raid/datasets/kobe'
# save_folder = '/raid/datasets/kobe_images'
# videos = os.listdir(videos_folder)
# for video in videos:
#     video_stream = cv2.VideoCapture(os.path.join(videos_folder, video))
#     # video_stream.open()
#     # video_num = video.split('_')[-1].split('.')[0]
#     video_num = video
#     save_sub_folder = os.path.join(save_folder, video_num)
#     os.makedirs(save_sub_folder, exist_ok=True)
#     counter = 0
#     while video_stream.isOpened():
#         flag_frame_available, current_frame = video_stream.read()
#         if flag_frame_available:
#             # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
#             image_name = str(counter).zfill(8) + '.png'
#             cv2.imwrite(os.path.join(save_sub_folder, image_name), current_frame)
#             counter += 1
#             print(counter)
#         else:
#             break
#         if counter == 50000:
#             break
#         # video_stream.release()
#
#
#
#
#
#
#
#
#
#
#





#     print (3)
#     generate_single_video(full_dir, output_dir, output_name, fps, compressed_flags)
#
# import torch
# import torchvision
# video_array = torch.randn(6,3,192,192)
# fps = 15
# video_codec = 'h264'
# filename = 'temp.mp4'
# torchvision.io.write_video(filename, video_array, fps, video_codec)



# ffmpeg -framerate 10 -i %08d.png -c:v libx264 -crf 0 lossless.mp4
# ffmpeg -i lossless.mp4 -vcodec libx264 -crf 25 crf25.mp4
# ffmpeg -ss 00:00:00 -t 00:00:10 -i crf25.mp4 -r 10 ../try/crf25_%2d.png
# Util functions
from RapidBase.TrainingCore.datasets import get_default_IO_dict
from RapidBase.TrainingCore.trainer import *
from os import path as osp


def imshow_dataloader(dataloader, index, noisy=False):
    i = 0
    for batch in dataloader:
        if i==index:
            if not noisy:
                orig_img = batch['output_frames_original'] / 255
                imshow_torch(orig_img[0][2])
            else:
                noisy_img = batch['output_frames_noisy']/255
                imshow_torch(noisy_img[0][2])
            break
        i += 1

def get_img(dataloader, index, noisy=False):
    i = 0
    for batch in dataloader:
        if i == index:
            if not noisy:
                orig_img = batch['output_frames_original']
                return orig_img
            else:
                noisy_img = batch['output_frames_noisy'] / 255
                return noisy_img
        i += 1

def save_dataloader(dataloader, num_imgs=20, num_frames=50, val=False, data_dir = '../Data', dataset_name = 'dataset'):

    current_img = 0

    for batch in dataloader:
        print ('Saving image ', current_img +1)
        outputs_dict = batch

        current_img_str = str(current_img).zfill(3)

        dataset_kind = 'Noisy' # Save noisy images
        if val:
            dataset_kind += '_val'

        batch_noisy = outputs_dict['output_frames_noisy']
        save_dir = osp.join(data_dir, dataset_name, dataset_kind, current_img_str)
        current_frame = 0
        for frame in range(num_frames):
            img = batch_noisy[0][frame]
            current_frame_str = str(current_frame).zfill(8)
            save_image_torch(save_dir, current_frame_str + '.png', img.clip(0,255),flag_convert_bgr2rgb=True)
            current_frame += 1

        dataset_kind = 'Clean' # Save clean images
        if val:
            dataset_kind += '_val'

        batch_orig = outputs_dict['output_frames_original']
        save_dir = osp.join(data_dir, dataset_name, dataset_kind, current_img_str)
        current_frame = 0
        for frame in range(num_frames):
            img = batch_orig[0][frame]
            current_frame_str = str(current_frame).zfill(8)
            save_image_torch(save_dir, current_frame_str + '.png', img.clip(0,255),flag_convert_bgr2rgb=True)
            current_frame += 1

        current_img += 1

        if current_img == num_imgs:
            return



def create_meta_file(data_dir, meta_file = 'meta_file.txt', save_dir = '.'):
    num_imgs = len(os.listdir(data_dir))
    zf = num_imgs//10 + 1# 000 for <100 images
    with open(osp.join(save_dir, meta_file), 'w') as output_file:
        num_frames = 30
        for n in range(num_imgs):
            vid_str = str(n).zfill(zf)
            vid_dir = osp.join(data_dir, vid_str)
            frames = os.listdir(vid_dir)
            frame_dir = osp.join(vid_dir, frames[0])
            img = Image.open(frame_dir)
            num_frames = str(len(frames))
            output_file.write(vid_str + ' ' + num_frames + ' ' + '(' +
                              str(img.size[0]) + ',' + str(img.size[1]) + ','
                              + '3)' + ' ' + '0'.zfill(8) + '\n')# for RGB

def create_lmdb():
    pass
    # with open('ALPR.txt', 'w') as out:
    #     for n in range(num_imgs):
    #         out.write(str(n).zfill(3) + ' ' + str(num_frames) + ' ' + '(256,400,3)' + '\n')


# data_dir = '../Data/coco30_srx4_shift=8_dim=[512,400]_frames=50/Noisy'
# meta_file = 'a.txt'
# save_dir = '../KAIR/data/meta_info'
# create_meta_file(data_dir, meta_file, save_dir)



def string_rjust(original_number, final_number_size):
    return str.rjust(str(original_number), final_number_size, '0')


import os
import cv2

# video_path = '../../shoval/h20t_dataset/test1/DCIM/DJI_202204031230_001/DJI_20220403130240_0001_S.MP4'
video_path = '/media/newhd/datasets/Drones_Random_Experiments/RGB_drone_WhiteNight2.mp4'
video_stream = cv2.VideoCapture(video_path)
save_folder = '../comp_drones11/001'
os.makedirs(save_folder, exist_ok=True)
# video_stream.open()
counter = 0
while video_stream.isOpened():
    flag_frame_available, current_frame = video_stream.read()
    if flag_frame_available:
        # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        image_name = str(counter).zfill(8)+ '.png'
        cv2.imwrite(os.path.join(save_folder, image_name), current_frame)
        counter += 1
        print(counter)
video_stream.release()

import os
import cv2

# curr_dir = os.getcwd()
videos_folder = '../../shoval/h20t_dataset/test1/DCIM/DJI_202204031230_001'
save_folder = '/raid/datasets/drones_comp'
video_types = ['S', 'T']
videos = os.listdir(videos_folder)
for video in videos:
    video_type, video_num = video.split('_')[-1][0], video.split('_')[-2]
    if video_type in video_types:
        video_stream = cv2.VideoCapture(os.path.join(videos_folder, video))
        # video_stream.open()
        video_num = video.split('_')[-2]
        save_sub_folder = os.path.join(save_folder, video_type, video_num)
        os.makedirs(save_sub_folder, exist_ok=True)
        counter = 0
        while video_stream.isOpened():
            flag_frame_available, current_frame = video_stream.read()
            if flag_frame_available:
                # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
                image_name = str(counter).zfill(8)+ '.png'
                cv2.imwrite(os.path.join(save_sub_folder, image_name), current_frame, [cv2.IMWRITE_JPEG_LUMA_QUALITY, 5])
                counter += 1
                print(counter)
            else:
                break
        # video_stream.release()


if __name__ == '__main__'

    # curr_dir = os.getcwd()
    videos_folder = '../output_videos/drones_RGB'
    save_folder = '/raid/datasets/drones_RGB_comp_0.5_0.5_2/clean'
    videos = os.listdir(videos_folder)
    for video in videos:
        video = video
        video_stream = cv2.VideoCapture(os.path.join(videos_folder, video))
        # video_stream.open()
        video_num = video_num = video.split('_')[-1].split('.')[0]
        save_sub_folder = os.path.join(save_folder, video_num)
        os.makedirs(save_sub_folder, exist_ok=True)
        counter = 0
        while video_stream.isOpened():
            flag_frame_available, current_frame = video_stream.read()
            if flag_frame_available:
                # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
                image_name = str(counter).zfill(8)+ '.png'
                cv2.imwrite(os.path.join(save_sub_folder, image_name), current_frame, [cv2.IMWRITE_JPEG_LUMA_QUALITY, 5])
                counter += 1
                print(counter)
            else:
                break
            # video_stream.release()
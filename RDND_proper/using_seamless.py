# from Seamless import *
#
# clean_dir = '/raid/datasets/REDS/val_sharp'
# crf = 35
# flags = '-c:v libx264 -crf ' + str(crf)
# output_dir = '/raid/datasets/REDS/val_sharp_comp_' + str(crf)
# num_videos = len(os.listdir(clean_dir))
# os.makedirs(output_dir, exist_ok=True)
# for idx in range(num_videos):
#     str_dir = str(idx).zfill(3)
#     video_dir = osp.join(clean_dir, str_dir)
#     make_lossless_command = 'ffmpeg -framerate 10 -i {}/000000%2d.png -c:v libx264 -crf 0 lossless.mp4'.format(video_dir)
#     os.system(make_lossless_command)
#     make_crf_vid_command = 'ffmpeg -i lossless.mp4 -vcodec libx264 -crf {} crf{}.mp4'.format(crf, crf)
#     os.system(make_crf_vid_command)
#     delete_loseless_command = 'rm lossless.mp4'
#     os.system(delete_loseless_command)
#     output_crf_dir = osp.join(output_dir, str_dir)
#     os.makedirs(output_crf_dir , exist_ok=True)
#     make_crf_images_command = 'ffmpeg -ss 00:00:00 -t 00:00:10 -i crf{}.mp4 -r 10 {}/%08d.png -start_number 0'.format(crf, output_crf_dir , crf)
#     os.system(make_crf_images_command)
#     files = sorted(os.listdir(output_crf_dir))
#     for index,file in enumerate(files):
#         os.rename(osp.join(output_crf_dir, file), os.path.join(output_crf_dir, ''.join([str(index).zfill(8), '.jpg'])))
#     delete_crf_command = 'rm crf{}.mp4'.format(crf)
#     os.system(delete_crf_command)
#     print('Video {} saved'.format(idx))

# import os
# from os import path as osp
#
# dir = '/raid/datasets/Mitspe/new_images/C0003'
# fixed_dir = '/raid/datasets/Mitspe/fixed'
# files = sorted(os.listdir(dir))
# start_idx = 457
# curr_dir = 0
# idx = start_idx
# offset = 3
# dir_offset = 500
# dir_sz = 30
#
# while idx<len(files):
#     str_dir = str(curr_dir).zfill(3)
#     video_dir = osp.join(fixed_dir, str_dir)
#     os.makedirs(video_dir, exist_ok=True)
#     for i in range(dir_sz):
#         img = files[idx]
#         cp_img = osp.join(dir, img)
#         cp_command = 'cp {} {}'.format(cp_img, video_dir)
#         os.system(cp_command)
#         idx += offset
#         if idx>=len(files):
#             break
#     curr_dir += 1
#     idx += dir_offset

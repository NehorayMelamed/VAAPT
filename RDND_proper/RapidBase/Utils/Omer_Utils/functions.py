
from RapidBase.TrainingCore.trainer import *


### scp functions ###

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

def copy_to_local(dgx_path='/raid/a.pt', local_path='/home/', local_ip='192.168.21.88'
                  , port='22', username='elisheva', password='mafat123'):
    print ('Starting scp from {} to {}'.format(dgx_path, local_path))
    start = time.time()
    ssh = createSSHClient(local_ip, port, username, password)
    scp = SCPClient(ssh.get_transport())
    scp.put(dgx_path, local_path, recursive=True)
    end = time.time()
    print('Finished scp in {} seconds'.format(end-start))

# copy_to_local()

# video generation functions
https://github.com/wkentaro/gdown

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

def generate_concat_videos(full_dir, output_dir, fps, flags):

    num_videos = len(os.listdir(full_dir))
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(num_videos):
        str_dir = str(idx).zfill(4)
        video_dir = os.path.join(full_dir, str_dir, 'Concatenated_Outputs')
        output_video = 'output_video_' + str_dir + '.mp4'
        command = 'ffmpeg -framerate {} -i "{}/{}_everything_concat_Frame%03d.png" {} {}'.format(
            fps, video_dir, str_dir, flags, os.path.join(output_dir, output_video))
        os.system(command)
        print ('Video {} saved'.format(idx))
# parameters


inference_dir = '/raid/Pytorch_Checkpoints/2022-06-27/Inference/VRT_18_frames_reg_pelicanD_inf.py_better_concat/'
save_dir = 'RapidBase__TrainingCore__datasets__DataSet_Videos_In_Folders_LoadCorruptedGT/fixed_images__validation/0000'
full_dir = os.path.join(inference_dir, save_dir)
# dir_name = 'dejeg_deblur_REDS_videos_15_FPS'
output_dir = os.path.join('/raid/datasets/send_dudy/VRT_18_frames_pelicanD_inference_fixed')
fps = 15
# flags = '-c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p'
flags = '-c:v libx264 -crf 0 -pix_fmt yuv420p'
flags_new = '-c:v libx264 -crf 0'
# flags_new = '-c:v huffyuv -c:a libmp3lame -b:a 320k'
compressed_flags = '-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -minrate 1M -maxrate 1M -bufsize 2M'

generate_concat_videos(full_dir, output_dir, fps, flags)
# generate_single_video(full_dir, output_dir, 'a.avi',fps, flags)

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

# parameters
inference_dir = '/raid/Pytorch_Checkpoints/2022-06-12/Inference/drones_SR_noised.py/'
save_dir = 'RapidBase__TrainingCore__datasets__DataSet_Videos_In_Folders_LoadCorruptedGT_validation/16425'
full_dir = os.path.join(inference_dir, save_dir)
output_dir = '/raid/datasets/send/drones_SR_noised'
fps = 15
flags = '-c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p'
compressed_flags = '-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -minrate 2M -maxrate 2M -bufsize 2M'
generate_single_video(full_dir, output_dir, 'drones_SR_noised', fps, flags)



dataset_dir = '/home/omerl/REDS_noised/PP5/Noisy/'
output_dir = '/home/omerl/REDS_noised/PP5/Noisy_vid_2M'
fps = 15
flags = '-c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p'
compressed_flags = '-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -minrate 2M -maxrate 2M -bufsize 2M'
generate_dataset_videos(dataset_dir, output_dir, fps, compressed_flags)

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
            save_image_torch(save_dir, current_frame_str + '.png', (img/img.max() * 255).clip(0,255),flag_convert_bgr2rgb=True)
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
            save_image_torch(save_dir, current_frame_str + '.png', (img/img.max() * 255).clip(0,255),flag_convert_bgr2rgb=True)
            current_frame += 1

        current_img += 1

        if current_img == num_videos:
            return


if __name__ == '__main__':

# curr_dir = os.getcwd()
videos_folder = '/raid/datasets/liveVideos'
save_folder = '/raid/datasets/liveVideosImages'
videos = os.listdir(videos_folder)
for video in videos:
    video_stream = cv2.VideoCapture(os.path.join(videos_folder, video))
    # video_stream.open()
    # video_num = video.split('_')[-1].split('.')[0]
    video_num = video
    save_sub_folder = os.path.join(save_folder, video_num)
    os.makedirs(save_sub_folder, exist_ok=True)
    counter = 0
    while video_stream.isOpened():
        flag_frame_available, current_frame = video_stream.read()
        if flag_frame_available:
            # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
            image_name = str(counter).zfill(8) + '.png'
            cv2.imwrite(os.path.join(save_sub_folder, image_name), current_frame)
            counter += 1
            print(counter)
        else:
            break
        if counter == 500:
            break
        # video_stream.release()
















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
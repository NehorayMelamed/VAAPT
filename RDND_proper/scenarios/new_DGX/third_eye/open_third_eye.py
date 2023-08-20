import sys
sys.path.extend(['/home/omerl/rdnd', '/home/omerl/rdnd'])

from RapidBase.import_all import *
from torch.utils.data import DataLoader
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.callbacks import InferenceCallback_Denoising_Base
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.losses import Loss_Simple
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_RLSP_Recursive, PostProcessing_FastDVDNet
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.tensorboard_callbacks import TB_Denoising_Recursive
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import get_default_IO_dict

from os import path as osp
import kornia

def get_ThirdEye_tensor(full_filename):
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_2.raw2'
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_3.raw2'
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_4.raw2'
    # full_filename = '/home/mafat/Datasets/ThirdEye/Thermal_RAW_16bit_Movies_640_480/r0_5.raw2'

    W = 640
    H = 480

    scene_infile = open(full_filename, 'rb')
    frames_list = []

    flag_continue = True
    while flag_continue:
        scene_image_array = np.fromfile(scene_infile, dtype=np.uint16, count=W * H)
        if scene_image_array.size == H * W:
            image = Image.frombuffer("I", [W, H],
                                     scene_image_array.astype('I'),
                                     'raw', 'I', 0, 1)
            image = np.array(image)
            # image = image - image.min()
            frames_list.append(torch.tensor(image).unsqueeze(0))
        else:
            flag_continue = False

    final_tensor = torch.cat(frames_list, 0).unsqueeze(1).float()

    # ### Stretch Histogram: ###
    # image, (Q1, Q2) = scale_array_stretch_hist(final_tensor[0], min_max_values_to_scale_to=(0, 1),
    #                                            quantiles=(0.01, 0.99), flag_return_quantile=True)
    # final_tensor = scale_array_from_range(final_tensor.clamp(Q1, Q2), (Q1, Q2))

    # ### Simply Lower DC: ###
    # final_tensor = final_tensor - final_tensor.min()

    return final_tensor


### Paths: ###
datasets_main_folder = '/raid/datasets'
# videos_folder = osp.join(datasets_main_folder, 'ThirdEye/Thermal_RAW_16bit_Movies_640_480')
videos_folder = osp.join(datasets_main_folder, 'third_fixed_images')
files = os.listdir(videos_folder)
save_dir = '/raid/datasets/vrt_third/sharp_deblur_25_7/GT_X4'
movies = os.listdir(videos_folder)
upsampling_layer = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=None)
upsample = True

# s = 109
# img = input_tensor[s][0]
# imshow_torch(img)
# plt.show()


movies = '/raid/datasets/vrt_third/sharp_deblur_25_7/GT'
os.listdir(movies)


# imshow_torch_video_seamless(x,print_info=True)
folders = os.listdir(movies)
for movie in folders:
    file_name = os.path.join(movies, movie)
    # if movie.split('.')[-1] != 'raw2':
    #     continue
    print('Saving ', file_name)
    input_tensor = read_video_default_torch(file_name,flag_convert_to_rgb=0, size = (480,640))
    if upsample:
        input_tensor = upsampling_layer(input_tensor)
    curr_save_dir = osp.join(save_dir, movie.split('.')[0])
    os.makedirs(curr_save_dir, exist_ok=True)
    start_index = int(sorted(os.listdir(file_name))[0].split('.')[0])
    for frame in range(input_tensor.shape[0]):
        # save_image_torch(curr_save_dir, str(frame) + '.png', input_tensor[frame].unsqueeze(0))
        torchvision.utils.save_image(input_tensor[frame][0], osp.join(
            curr_save_dir, str(start_index+frame).zfill(8) + '.png'))
        if frame%200 == 1:
            print('Saving frame ', frame)



for movie in movies:
    file_name = os.path.join(videos_folder,movie)
    # if movie.split('.')[-1] != 'raw2':
    #     continue
    print('Saving ', file_name)
    input_tensor = get_ThirdEye_tensor(file_name)
    if upsample:
        input_tensor = upsampling_layer(input_tensor)
    curr_save_dir = osp.join(save_dir, movie.split('.')[0])
    os.makedirs(curr_save_dir, exist_ok=True)
    for frame in range(input_tensor.shape[0]):
        # save_image_torch(curr_save_dir, str(frame) + '.png', input_tensor[frame].unsqueeze(0))
        torchvision.utils.save_image(input_tensor[frame][0], osp.join(
            curr_save_dir, str(frame).zfill(8) + '.png'))
        if frame%200 == 1:
            print('Saving frame ', frame)


# imshow_torch(t[0][0])
# plt.show()
# copy from open_night, crate dataset, and run similiar stuff to white_night.
# fix stuff from txt file


# PPP, sigma = 100, 0.4
# PPP, sigma = 50, 0.2


'''
  input_tensor_edges = kornia.filters.canny(scale_array_stretch_hist(input_tensor[0:1]))[1]
    input_tensor_edges_dilated = kornia.morphology.dilation(input_tensor_edges, torch.ones(3,3))
    input_tensor_blur = kornia.filters.gaussian_blur2d(input_tensor[0:1], (11,11), (3,3))
    bla = input_tensor_blur - input_tensor[0:1]
    imshow_torch(bla*(1-input_tensor_edges_dilated)); plt.show()
    bla[~input_tensor_edges_dilated.bool()].std()
    bla.std()
    imshow_torch(bla)
    plt.show()

    print ((255*input_tensor).std(), input_tensor.median())
'''
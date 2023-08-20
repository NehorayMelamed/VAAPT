import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils

from RDND_proper.models.VRT_models import *
from utils.utils_omer import *


train_dict = EasyDict()

# Paths
base_path = '/home/mafat/Desktop/Omer'
vrt_path = osp.join(base_path, 'VRT')
data_path = osp.join(vrt_path, 'testsets')
reds_train_path = 'REDS4/blur'
reds_test_path = 'REDS4/GT'
checkpoints_path = osp.join(vrt_path, 'model_zoo', 'vrt')
train_folder = osp.join(data_path, reds_train_path)
test_folder = osp.join(data_path, reds_test_path)

# Set seed
TrainingCore.training_utils.set_seed(seed=42)

def get_IO_dicts():
    ### IO: ###
    IO_dict = get_default_IO_dict()
    IO_dict.number_of_image_frames_to_generate = 6
    IO_dict.flag_noise_images_to_RAM = False
    ### Post Loading Stuff: ###
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.flag_to_RGB_before_noise = False
    IO_dict.flag_to_BW_after_noise = False
    IO_dict.flag_to_RGB_after_noise = False
    IO_dict.flag_how_to_concat = 'T'
    ### Noise Parameters: ###
    IO_dict.SNR = np.inf #150
    IO_dict.sigma_to_dataset = 0 # 1 / np.sqrt(IO_dict.SNR) * 255
    IO_dict.SNR_to_model = IO_dict.SNR
    ### Shift / Directional-Blur Parameters: ###
    IO_dict.shift_size = 1 * 4
    IO_dict.blur_size = 0
    ### Gaussian Blur Parameters: ###
    IO_dict.blur_number_of_channels = 1
    IO_dict.blur_kernel_size = 3
    IO_dict.blur_sigma = 1
    ### Super Resolution: ###
    IO_dict.downsampling_factor = 4
    ### Universal Training Parameters: ###
    IO_dict.batch_size = 1
    ### Cropping: ###
    IO_dict.flag_crop_mode = 'center' # center/random
    # dim = [1280,720]
    dim = [128,128]
    IO_dict.initial_crop_size = dim
    IO_dict.final_crop_size = dim
    ### Test/Validation Dict: ###
    IO_dict_test = EasyDict(IO_dict)
    IO_dict_test.flag_crop_mode = 'center'
    IO_dict_test.initial_crop_size = dim
    IO_dict_test.final_crop_size = dim

    return IO_dict, IO_dict_test

IO_dict, IO_dict_test = get_IO_dicts()
#######################################################################################################################################

#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly_Bicubic(train_folder, IO_dict)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly_Bicubic(test_folder, IO_dict_test)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# imshow_dataloader(train_dataloader, 0, noisy=False)
# imshow_dataloader(train_dataloader, 17, noisy=True)
# train_dataset = torchvision.datasets.ImageFolder(root=train_folder)
# train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=False,  num_workers=4)

########

model = VRT_deblur(pretrained = True, pretrained_dataset='REDS').cuda()
model_path = osp.join(checkpoints_path, '007_VRT_videodeblurring_REDS.pth')
model.load_state_dict(torch.load(model_path)['params'])

x = get_img(train_dataloader, 0, noisy=False).cuda()

x64 = (x[:,:,:,0:64,0:64] / 255)

res = model(x64) #


# for idx, layer in enumerate(model.children()):
#     print (layer, idx)
#     layer.requires_grad = False
#
#
# for idx, layer in enumerate(model.parameters()):
#     # print (layer, idx)
#     layer.grad = None
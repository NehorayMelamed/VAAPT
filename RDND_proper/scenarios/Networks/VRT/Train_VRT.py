from RapidBase.TrainingCore.datasets import get_default_IO_dict
import RapidBase.TrainingCore as cr

from os import path as osp

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
coco30_path = osp.join(base_path, 'coco30')
# Set seed
cr.training_utils.set_seed(seed=42)


IO_dict = get_IO_dict(sf=1, shift=8, blur=8, blur_steps=5, snr=np.inf, sigma= 0.5,
 w=512, h=400, offset=32, crop_mode='center')

IO_dict = get_IO_dict(sf=4, shift=8, blur=0, snr=np.inf, sigma= 0,
                       w=512, h=400, offset=16, crop_mode='center')
#######################################################################################################################################

#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
train_dataset = cr.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_DirectionalBlur(coco30_path, IO_dict)
# train_dataset = cr.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly_Bicubic(coco30_path, IO_dict)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=False)




val = False
data_dir = '../Data'
# dataset_name = 'coco30_srx4_shift=8_dim=[512,400]_frames=50'
dataset_name = 'coco30_shift=8_blur=8_steps=5_dim=[512,400]_frames=50_new'
save_dataloader(train_dataloader, num_imgs=20, num_frames=50, val=val, data_dir=data_dir, dataset_name= dataset_name)















for a in train_dataloader:
    b = a
    break

bla = train_dataset[2]

a = train_dataset[3]
b = a['output_frames_original']
c = a['output_frames_noisy']
imshow_torch(c[0]/255)
imshow_torch(b[0]/255)


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

from RapidBase.import_all import *
import os
import torch
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
from numpy.random import *
from matplotlib.pyplot import *

import skimage
import imutils
import skimage.measure as skimage_measure
import skimage.metrics as skimage_metrics
import image_registration
import RapidBase.TrainingCore.datasets

import logging
from collections import OrderedDict
from RDND_proper.models.KAIR_FFDNet_and_more.utils import utils_logger
from RDND_proper.models.KAIR_FFDNet_and_more.utils import utils_image as util
from RDND_proper.models.KAIR_FFDNet_and_more.models.network_ffdnet import FFDNet as net
from RDND_proper.models.KAIR_FFDNet_and_more.models.network_ffdnet import FFDNet_dudy, FFDNet_dudy2

NUM_IN_FRAMES = 5
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage(root_path=r'/home/mafat/DataSets/DataSets/Div2K/Official_Test_Images/Original_Images',
                                                                    number_of_image_frame_to_generate=NUM_IN_FRAMES,
                                                                    base_transform=None, batch_transform=None,
                                                                    image_loader=ImageLoaderCV,
                                                                    max_number_of_images=np.inf,
                                                                    crop_size=1000,
                                                                    flag_to_RAM=True, flag_recursive=False, flag_normalize_by_255=True, flag_crop_mode='center',
                                                                    flag_explicitely_make_tensor=False, allowed_extentions=IMG_EXTENSIONS, flag_to_BW=True,
                                                                    flag_base_tranform=False, flag_turbulence_transform=False, Cn2=5e-13,
                                                                    flag_batch_transform=False,
                                                                    flag_how_to_concat='T')
### DataLoaders: ###
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)


### Get Model Input: ###
input_tensor = train_dataloader.__iter__().__next__()['output_frames'].cuda()
input_tensor = torch.cat((input_tensor,input_tensor,input_tensor), 2) #BW->RGB


### FFDNet Script Parameters: ###   #TODO: turn this into something i can mainstream
### Main Parameters: ###
main_path = r'/home/mafat/PycharmProjects/IMOD/models/KAIR_FFDNet_and_more'
model_name = 'ffdnet_gray'  # 'ffdnet_gray' | 'ffdnet_color' | 'ffdnet_color_clip' | 'ffdnet_gray_clip'
task_current = 'dn'  # 'dn' for denoising | 'sr' for super-resolution !!!!!
noise_level_img = 15  # noise level for noisy image
noise_level_model = noise_level_img  # noise level for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Dataset Parameters: ###
testset_name = 'bsd68'  # test set,  'bsd68' | 'cbsd68' | 'set12'
need_degradation = True  # default: True
show_img = False  # default: False

### Set Channels According To Specific Scenario: ###
sf = 1  # unused for denoising
if 'color' in model_name:
    n_channels = 3  # setting for color image
    nc = 96  # setting for color image
    nb = 12  # setting for color image
else:
    n_channels = 1  # setting for grayscale image
    nc = 64  # setting for grayscale image
    nb = 15  # setting for grayscale image
if 'clip' in model_name:
    use_clip = True  # clip the intensities into range of [0, 1]
else:
    use_clip = False

### Get Paths: ###
model_pool = 'model_zoo'  # fixed
testsets = 'testsets'  # fixed
results = 'results'  # fixed
result_name = testset_name + '_' + model_name
border = sf if task_current == 'sr' else 0  # shave boader to calculate PSNR and SSIM
model_path = os.path.join(main_path, model_pool, model_name + '.pth')


### Low-quality, High-quality and Estimated Images Paths: ###
L_path = os.path.join(main_path, testsets, testset_name)  # L_path, for Low-quality images
H_path = L_path  # H_path, for High-quality images
E_path = os.path.join(main_path, results, result_name)  # E_path, for Estimated images
util.mkdir(E_path)
if H_path == L_path:
    need_degradation = True
need_H = True if H_path is not None else False


### Set Up Logger: ###
logger_name = result_name
utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name + '.log'))
logger = logging.getLogger(logger_name)


### Load Model: ###
model = net(in_nc=n_channels, out_nc=n_channels, nc=nc, nb=nb, act_mode='R')
model.load_state_dict(torch.load(model_path), strict=True)  #TODO: strict???
model = model.to(device)


### Test Mode: turn to eval mode and freeze gradients: ###
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False


### Initialize logger: ###
logger.info('Model path: {:s}'.format(model_path))
test_results = OrderedDict()
test_results['psnr'] = []
test_results['ssim'] = []
logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
logger.info(L_path)

### Get Images Filenames: ###
L_paths = util.get_image_paths(L_path)
H_paths = util.get_image_paths(H_path) if need_H else None


### Loop over low quality images to be restored and pass through model: ###
for idx, img in enumerate(L_paths):
    1

    ### Get low quality image and normalize it: ###
    img_name, ext = os.path.splitext(os.path.basename(img))
    img_L = util.imread_uint(img, n_channels=n_channels)
    img_L = util.uint2single(img_L)

    ### If we use degredation live do it: ###
    if need_degradation:  # degradation process
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img / 255., img_L.shape)
        if use_clip:
            img_L = util.uint2single(util.single2uint(img_L))

    ### Plot low quality image if wanted: ###
    util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

    ### Turn loq quality array to tensor: ###
    img_L = util.single2tensor4(img_L)
    img_L = img_L.to(device)

    ### Get sigma map: ###
    sigma = torch.full((1, 1, 1, 1), noise_level_model / 255.).type_as(img_L)

    ### Pass input image and sigma map through model and get estimated image: ###
    img_E = model(img_L, sigma)  #TODO: change from seperate input to one input tensor or dictionary at most according to infrastructure
    img_E = util.tensor2uint(img_E)

    ### If we have high quality image to read -> get it and get metrics: ###
    if need_H:
        ### Get high quality image: ###
        img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
        img_H = img_H.squeeze()

        ### PSNR and SSIM: ###
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        ssim = util.calculate_ssim(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name + ext, psnr, ssim))
        util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

    ### Save Results: ###
    util.imsave(img_E, os.path.join(E_path, img_name + ext))









with torch.no_grad():
    output_tensor = model_DVDNet.forward(input_tensor, noise_std=torch.Tensor([args.noise_sigma]).cuda())



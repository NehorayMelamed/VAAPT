import sys
import os.path
import math
import argparse
import time
import random
import cv2
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from RDND_proper.models.select_model import define_Model

'''
# --------------------------------------------
# training code for VRT
# --------------------------------------------
'''

#fix some code
#add end2end
#train weekend
# add metrics somehow
# add dvdnet try
# look dataset
# look arhava

'''
# ----------------------------------------
# Step--1 (prepare opt)
# ----------------------------------------
'''

dist = False
json_path = 'options/vrt_omer/train_vrt_denoised_blur.json'
opt = option.parse(json_path, is_train=True)
# ----------------------------------------
# distributed settings
# ----------------------------------------
if dist:
    init_dist('pytorch')
opt['rank'], opt['world_size'] = get_dist_info()

if opt['rank'] == 0:
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

# ----------------------------------------
# save opt to  a '../option.json' file
# ----------------------------------------
if opt['rank'] == 0:
    option.save(opt)
# ----------------------------------------
# return None for missing key
# ----------------------------------------
opt = option.dict_to_nonedict(opt)

# ----------------------------------------
# configure logger
# ----------------------------------------
if opt['rank'] == 0:
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

# ----------------------------------------
# seed
# ----------------------------------------
seed = opt['train']['manual_seed']
if seed is None:
    seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

'''
# ----------------------------------------
# Step--2 (creat dataloader)
# ----------------------------------------
'''

# ----------------------------------------
# 1) create_dataset
# 2) create_dataloader for train and test
# ----------------------------------------
for phase, dataset_opt in opt['datasets'].items():
    if phase == 'train':
        train_set = define_Dataset(dataset_opt)
        train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
        if opt['rank'] == 0:
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        if opt['dist']:
            train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'],
                                               drop_last=True, seed=seed)
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                      shuffle=False,
                                      num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                      drop_last=True,
                                      pin_memory=True,
                                      sampler=train_sampler)
        else:
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)

    elif phase == 'test':
        test_set = define_Dataset(dataset_opt)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=1,
                                 drop_last=False, pin_memory=True)
    else:
        raise NotImplementedError("Phase [%s] is not recognized." % phase)


# for a in test_loader:
#     a=a

# from utils.utils_omer import imshow_torch

# for a in test_loader:
#     lr = a['L']
#     hr = a['H']
#     break
# x = lr[0][0][:3]
# y = hr[0][0][:3]
# y = test_loader.dataset[1]['H']
# x = test_loader.dataset[1]['L']
# imshow_torch(y[5][:3])
#
# imshow_torch(x[0][:3])
# imshow_torch(y[0][:3])
# imshow_torch(a[0][0])

'''
# ----------------------------------------
# Step--3 (initialize model)
# ----------------------------------------
'''
model = define_Model(opt)
# net = model.netG
# x = torch.randn(1,6,3,64,64)
# net(x)
model.init_train()
if opt['rank'] == 0:
    logger.info(model.info_network())
    logger.info(model.info_params())
# load pth here and make sure it produces something52

# model_path = '../VRT/model_zoo/vrt/007_VRT_videodeblurring_REDS.pth'
# if os.path.exists(model_path):
#     print(f'loading model from {model_path}')
#
# pretrained_model = torch.load(model_path)

# new_dict = {'moudle.'+ k : v for k,v in pretrained_model['params'].items()}

from RDND_proper.models.vrt_models import VRT_deblur

net_G = VRT_deblur(pretrained=True).cuda()
model.netG = net_G

'''
# ----------------------------------------
# Step--4 (main training)
# ----------------------------------------
'''
current_step = 0
first_save = True
is_training = True
for epoch in range(1000000):  # keep running
    for i, train_data in enumerate(train_loader):

        current_step += 1

        if is_training:
            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if opt['use_static_graph'] and (current_step == opt['train']['fix_iter'] - 1):
                current_step += 1
                model.update_learning_rate(current_step)
                model.save(current_step)
                current_step -= 1
                logger.info('Saving models ahead of time when changing the computation graph with use_static_graph=True'
                            ' (we need it due to a bug with use_checkpoint=True in distributed training). The training '
                            'will be terminated by PyTorch in the next iteration. Just resume training with the same '
                            '.json config file.')

        # -------------------------------
        # 6) testing
        # -------------------------------
        if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['ssim'] = []
            test_results['psnr_y'] = []
            test_results['ssim_y'] = []

            for idx, test_data in enumerate(test_loader):
                model.feed_data(test_data)
                model.test()

                visuals = model.current_visuals()
                output = visuals['E']
                gt = visuals['H'] if 'H' in visuals else None
                folder = test_data['folder']

                test_results_folder = OrderedDict()
                test_results_folder['psnr'] = []
                test_results_folder['ssim'] = []
                test_results_folder['psnr_y'] = []
                test_results_folder['ssim_y'] = []

                for i in range(output.shape[0]):
                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    img = output[i, ...].clamp_(0, 1).numpy()
                    if img.ndim == 3:
                        img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                    img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                    if opt['val']['save_img']:
                        save_dir = opt['path']['images']
                        util.mkdir(save_dir)
                        # seq_ = os.path.basename(test_data['lq_path'][i][0]).split('.')[0] #This was bugged
                        seq_ = str(i)
                        os.makedirs(f'{save_dir}/{folder[0]}/{current_step:d}', exist_ok=True)
                        # Swich to folder of current step
                        cv2.imwrite(f'{save_dir}/{folder[0]}/{current_step:d}/{seq_}.png', img)
                        print('Saving frame ', str(i))

                        if first_save:
                            os.makedirs(f'{save_dir}/{folder[0]}_noised', exist_ok=True)
                            noised_img = test_data['L'][0][i][:3].clamp_(0, 1).numpy()
                            if noised_img.ndim == 3:
                                noised_img = np.transpose(noised_img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                            noised_img = (noised_img * 255.0).round().astype(np.uint8)  # float32 to uint8
                            cv2.imwrite(f'{save_dir}/{folder[0]}_noised/{seq_}.png', noised_img)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    img_gt = gt[i, ...].clamp_(0, 1).numpy()
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                    img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                    img_gt = np.squeeze(img_gt)

                    test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                    if img_gt.ndim == 3:  # RGB image
                        img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                        img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                        test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                        test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                    else:
                        test_results_folder['psnr_y'] = test_results_folder['psnr']
                        test_results_folder['ssim_y'] = test_results_folder['ssim']

                first_save = False
                psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
                ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])

                if gt is not None:
                    logger.info('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                                format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
                    test_results['psnr'].append(psnr)
                    test_results['ssim'].append(ssim)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                else:
                    logger.info('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

            # summarize psnr/ssim
            if gt is not None:
                ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                    epoch, current_step, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))

        if current_step > opt['train']['total_iter']:
            logger.info('Finish training.')
            model.save(current_step)
            sys.exit()



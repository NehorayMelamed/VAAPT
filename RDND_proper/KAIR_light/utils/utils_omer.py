# Utils for Omer

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.pylab import figure, colorbar, title
import numpy as np
import os
from os import path as osp

def imshow_torch(image, flag_colorbar=True, title_str=''):
    fig = figure()
    # plt.cla()
    plt.clf()
    # plt.close()

    if len(image.shape) == 4:
        pylab.imshow(np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0)).squeeze())  # transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 3:
        pylab.imshow(np.transpose(image.detach().cpu().numpy(),(1,2,0)).squeeze()) #transpose to turn from [C,H,W] to numpy regular [H,W,C]
    elif len(image.shape) == 2:
        pylab.imshow(image.detach().cpu().numpy())

    if flag_colorbar:
        colorbar()  #TODO: fix the bug of having multiple colorbars when calling this multiple times
    title(title_str)
    return fig

def change_img_prefix(folder, digits = 8, type = 'png'):

    for subdir, dirs, files in os.walk(folder):
        break
    for d in dirs:
        file = osp.join(folder, d)
        for ff in os.walk(file):
            for fff in ff[2]:
                file_path = osp.join(file, fff)
                new_file = fff.split('.')[0].zfill(digits)
                new_file_path = osp.join(file, new_file + '.' + type)
                os.rename(file_path, new_file_path)
            print('finished subfolder: ', d)

# imshow_torch(b[0][0][3])

folder = 'experiments/008_train_vrt_videodenoising_davis_omer_blur2/images/020'
folder = 'experiments/008_train_vrt_videodenoising_davis_omer_blur4/images/020'
folder = 'experiments/train_vrt_denoised_blur4.1/images/020'
change_img_prefix(folder)
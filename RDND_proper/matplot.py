import torch

from RapidBase.import_all import *
from Seamless import imshow_torch_seamless, imshow_torch_video_seamless
import kornia

input_tensor = read_image_default_torch()
input_tensor2 = torch.randn(1,3,100,100)
input_tensor2 = (input_tensor2 - input_tensor2.min())
input_tensor2 = (input_tensor2 / input_tensor2.max())

a = read_video_default_torch()
imshow_torch_seamless(input_tensor, print_info=True)
imshow_torch_seamless(input_tensor2, print_info=True)
imshow_torch_video_seamless(a,print_info=True)
# plt.show()
# imshow_torch(blur_tensor_kornia)
# plt.show()
# imshow_torch(blur_tensor_dudy)
# plt.show()
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pickle
#
# ax = plt.subplot(111)
# x = np.linspace(0, 10)
# y = np.exp(x)
# plt.plot(x, y)
# plt.show()
# with open('/raid/myplot4.pkl','wb') as fid:
#     pickle.dump(ax, fid)
#
# with open('/raid/img.pkl','wb') as fid:
#     pickle.dump(ax, fid)
#
# imshow_torch
#
# ax2 = plt.plot(img[0])
# torch.save(eli, '/raid/a.pt')
# with open('/raid/img2.pkl','wb') as fid:
#     pickle.dump(ax2, fid)
# import matplotlib.pyplot as plt
# import pickle
#
# with open('/raid/myplot4.pkl','rb') as fid:
#     ax = pickle.load(fid)
# plt.show()
#
#
# def seemless_imshow_torch(eli,...):
#     torch.save(eli, '/raid/a.pt')


# check batch_size
# see how much testing takes if u can do cuple tests on both.
# fix 4-> to 6 saving dirs
# fix thirdeye files
# make mv function from range to dir

import numpy as np
import matplotlib.pyplot as plt
import cv2

bin_image = read_image_default_torch()
imshow_torch_seamless(bin_image)
np_image = np.array(bin_image.numpy()[0] * 255, dtype=uint8).transpose([1,2,0])[:,:,0]
bin_image2 = cv2.cvtColor(np_image, cv2.COLOR_BayerBG2RGB)

bin_image3 = cv2.cvtColor(bin_image2, cv2.COLOR_RGB2GRAY)
bin_image3_torch = torch.transpose(torch.transpose(torch.Tensor(bin_image2),0,2),1,2)
imshow_torch_seamless(bin_image3_torch/255, print_info=True)
imshow_torch_video_seamless(bin_video, print_info=True)

bin_video = read_video_default_torch()
plt.imshow(bin_image, cmap='gray')
plt.show(block=True)

# shift_R = np.sqrt(shift_x**2 + shift_y**2)
# direction_rad = np.arctan(shift_y/shift_x)
# direction_degree = direction_rad * 180/np.pi
# warp_object = Warp_Object()

# plt.show()

# plt.show()
# imshow_torch(blur_tensor_dudy)
# plt.show()
#


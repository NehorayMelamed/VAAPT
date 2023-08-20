
from RapidBase.import_all import *
from colour_demosaicing import mosaicing_CFA_Bayer, demosaicing_CFA_Bayer_Menon2007
# 'demosaicing_CFA_Bayer_bilinear', 'demosaicing_CFA_Bayer_DDFAPD',
#     'demosaicing_CFA_Bayer_Malvar2004', 'demosaicing_CFA_Bayer_Menon2007',
#     'masks_CFA_Bayer', 'mosaicing_CFA_Bayer'


input_tensor_RGB = read_image_default_torch()
input_tensor_RGB = (input_tensor_RGB*5).round()
input_tensor_RGB_numpy = input_tensor_RGB.cpu().squeeze(0).permute([1,2,0]).numpy()
H,W,C = input_tensor_RGB_numpy.shape

### Turn input RGB image into the needed Bayer Pattern: ###
input_tensor_Bayer = mosaicing_CFA_Bayer(input_tensor_RGB_numpy, 'GRBG')  # to_linear = de-gamma
H_bayer, W_bayer = input_tensor_Bayer.shape

### Add noise to bayer pattern: ###
input_tensor_Bayer_noisy = input_tensor_Bayer + np.random.randn(H_bayer,W_bayer) * 5

### demosaic noisy bayer: ###
RGB_image_noisy = demosaicing_CFA_Bayer_Menon2007(input_tensor_Bayer_noisy, 'GRBG')
imshow(RGB_image_noisy/5)

### Mosaic noisy RGB back to Bayer: ###
Bayer_from_noisy_RGB = mosaicing_CFA_Bayer(RGB_image_noisy, 'GRBG')
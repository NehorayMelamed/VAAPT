from Seamless import *
import colour_demosaicing

y = read_video_default_torch()
imshow_torch_seamless(x,print_info=True)
imshow_torch_video_seamless(y,print_info=True)

x = torch.randn(1,3,64,64)
RGB = np.array([[[0, 1, 2],
                 [0, 1, 2]],
                [[0, 1, 2],
                 [0, 1, 2]]])

input_tensor = read_image_default_torch()[0]
imshow_torch_seamless(input_tensor,print_info=True)

input_np = torch.transpose(input_tensor,0,2)
mosaic = colour_demosaicing.mosaicing_CFA_Bayer(input_np, 'GRBG')
imshow_torch_seamless(x,print_info=True,title_str = 'mosaic')

x = torch.transpose(torch.tensor(mosaic).unsqueeze(0),1,2)

C,H,W = x.shape
rand_noise = torch.randn(C,H,W)

bayer_noised = x + rand_noise
bayer_noised = torch.clip(bayer_noised,0,1)

imshow_torch_seamless(bayer_noised,print_info=True,title_str = 'bayer_noised')

noisy_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(bayer_noised[0] , 'GRBG')

y = torch.transpose(torch.transpose(torch.tensor(noisy_rgb),0,2),1,2)

imshow_torch_seamless(y,print_info=True,title_str = 'noisy_rgb')


if flag_clip:
    noisy_rgb = np.maximum(noisy_rgb, 0)
    noisy_rgb = np.minimum(noisy_rgb, 1)

return noisy_rgb






def noise_RGB_through_bayer_no_rand(original_image, flag_clip=True, flag_random_analog_gain=False, gain=None):
    bayer = mosaicing_CFA_Bayer(original_image, 'GRBG')  # to_linear = de-gamma
    h, w = bayer.shape

    ### Compute noisy part of the image using analog gain: ###
    if flag_random_analog_gain:
        analog_gain = np.random.random(1) * 92 + 32
    else:
        analog_gain = 32 + 92/2
        if gain:
            analog_gain = gain
    std_image = compute_noise_stddev(bayer, analog_gain, use_mean=False)  ### Difference is use_mean=False  ???

    ### Apply Multiplicative random noise: ###
    rand_noise = np.random.randn(h, w)
    noise_image = std_image * rand_noise  # actually generate the noise

    ### Add noise to the bayer image: ###
    bayer = bayer + noise_image
    bayer = np.maximum(bayer, 0)
    bayer = np.minimum(bayer, 1)

    ### return to rgb domain: ###
    noisy_rgb = demosaicing_CFA_Bayer_Menon2007(bayer, 'GRBG')
    if flag_clip:
        noisy_rgb = np.maximum(noisy_rgb, 0)
        noisy_rgb = np.minimum(noisy_rgb, 1)

    return noisy_rgb
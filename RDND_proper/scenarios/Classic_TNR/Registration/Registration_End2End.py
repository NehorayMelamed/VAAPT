from RDND_proper.scenarios.Classic_TNR.Registration.FFT_based_registration_2 import *

### Prepare Data: ###
shift_x = np.array([-30])
shift_y = np.array([30])
scale = np.array([1.05])
rotation_angle = np.array([-10])  # [degrees]
input_image_1 = read_image_default_torch()
input_image_1 = RGB2BW(input_image_1)
input_image_2 = warp_tensors_affine(input_image_1, shift_x, shift_y, scale, rotation_angle)  #TODO: add bicubic interpolation
input_image_1 = crop_torch_batch(input_image_1, 1024)
input_image_2 = crop_torch_batch(input_image_2, 1024)
input_image_1 = nn.AvgPool2d(1)(input_image_1).cuda()
input_image_2 = nn.AvgPool2d(1)(input_image_2).cuda()

### Initial Scale-Rotation-Translation Discovery: ###
affine_registration_layer = get_rotation_scaling_and_translation_object()
initial_scale_rotation_registration_downsample_factor = 1
tic()
recovered_angle, recovered_scale, recovered_translation, input_image_1_displaced = affine_registration_layer.forward(input_image_1,
                                                                                                                     input_image_2,
                                                                                                                     downsample_factor=initial_scale_rotation_registration_downsample_factor,
                                                                                                                     flag_return_shifted_image=True)
toc()
imshow_torch(input_image_1_displaced - input_image_1)


# ### Second Min-SAD For Fine Tunning Registration: ###
# recovered_angle, recovered_scale, recovered_translation, input_image_1_displaced = affine_registration_layer.forward(input_image_1,
#                                                                                                                      input_image_1_displaced,
#                                                                                                                      downsample_factor=initial_scale_rotation_registration_downsample_factor,
#                                                                                                                      flag_return_shifted_image=True)
# toc()
# imshow_torch(input_image_1_displaced - input_image_1)














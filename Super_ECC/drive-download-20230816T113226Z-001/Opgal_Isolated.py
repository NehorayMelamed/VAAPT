# import os.path
#
# import torchvision.datasets.mnist
# from RapidBase.Utils.Registration.optical_flow_objects import *
# from RapidBase.Anvil.alignments import *
# from RapidBase.Anvil._alignments.cross_correlation_alignments import *
# from RapidBase.Anvil._alignments._cross_correlation.cross_correlation_stuff import *
# from RapidBase.Utils.Registration.Unified_Registration_Utils import FFT_OLA_PerPixel_Layer_Torch
#
# import os, ffmpeg
#
# from QS_Jetson_mvp_milestone.functional_utils.miscelenous_utils import create_folder_if_doesnt_exist
# from RapidBase.Anvil._alignments.cross_correlation_alignments import classic_weighted_circular_cc_shifts_calc
# from RapidBase.import_all import *
# from RapidBase.Anvil.import_all import *




from Opgal_Isolated_Utils import *








### Read all image filenames: ###
opgal_movies_list = [
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/OpGal\m_2Me_iwr_4ms_250fr',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OpGal\Vid_7_25Hz_IT500us_05Me',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OpGal\Vid_8_25Hz_IT500us_05Me_wVib',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OpGal\Vid_9_25Hz_750fr_05Me_IT500us',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OpGal\Vid_10_25Hz_750fr_05Me_IT500us',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/OpGal\m_2Me_iwr_2000u_250fr_50Hz',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/OpGal\m_35Me_iwr_2ms_250fr',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/OpGal\n1',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data/OpGal\n2',

    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_1',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_2',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_3',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_4',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_5',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_6',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_7',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_8',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_9',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_10_lowenergy',
    r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_11_lowenergy',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_with_cap',
    # r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_with_cap_zoom',
    ]



### Get NU Pattern: ###
folder_path = r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\tint150us_cap05_25fps_with_cap'
NU_full_filename = os.path.join(folder_path, 'NU_pattern.pt')
if os.path.exists(NU_full_filename) == False:
    ### Get current movie images filenames: ###
    all_filenames = path_get_all_filenames_from_folder(folder_path, flag_recursive=True, flag_full_filename=True)
    ### Get filenames into tensors: ###
    input_tensor_noise = read_images_from_filenames_list(all_filenames[0:500], flag_how_to_concat='T', flag_return_torch=True, max_number_of_images=500)
    input_tensor_noise = input_tensor_noise.cuda()
    input_tensor_NU = input_tensor_noise.mean(0,True)
    # input_tensor = scale_array_stretch_hist(input_tensor)
    # imshow_torch_video(input_tensor, FPS=25, frame_stride=3)
    # imshow_torch(input_tensor.mean(0,True))
    ### Save Tensor: ###
    torch.save(input_tensor_NU, NU_full_filename)
else:
    input_tensor_NU = torch.load(NU_full_filename)


max_number_of_images = 150


# # Compress input.mp4 to 50MB and save as output.mp4
# compress_video(r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2\very_noise_cleaned.avi',
#                r'C:\Users\dudyk\Desktop\dudy_karl\RDND\RapidBase\Data\OPGAL2/very_noise_cleaned2.avi', 50 * 1000)


### Loop over all movies: ###
for movie_index in np.arange(len(opgal_movies_list)):
    ### Get current movie images filenames: ###
    current_movie_filename = opgal_movies_list[movie_index]
    all_filenames = path_get_all_filenames_from_folder(current_movie_filename, flag_recursive=True, flag_full_filename=True)
    opal_movie_name = os.path.split(current_movie_filename)[-1]
    opgal_super_folder_path = os.path.split(current_movie_filename)[0]

    ### Get filenames into tensors: ###
    input_tensor = read_images_from_filenames_list(all_filenames[0:max_number_of_images], flag_how_to_concat='T', flag_return_torch=True, max_number_of_images=max_number_of_images)
    input_tensor = input_tensor.cuda()
    # input_tensor_original = input_tensor * 1.0
    # reference_tensor_original = input_tensor_original[0:1]

    ### Lower NU: ###
    input_tensor = input_tensor - input_tensor_NU
    input_tensor_original = input_tensor * 1.0
    reference_tensor_original = input_tensor_original[0:1]
    # input_tensor_NUC = input_tensor - input_tensor_NU
    # imshow_torch_video(scale_array_stretch_hist(input_tensor), FPS=25, frame_stride=2)
    # imshow_torch_video(scale_array_stretch_hist(input_tensor_NUC), FPS=25, frame_stride=2)
    # imshow_torch_video(torch.cat([scale_array_stretch_hist(input_tensor), scale_array_stretch_hist(input_tensor_NUC)],-1), FPS=25, frame_stride=1)

    ### Stretch histogram for better viewing: ###
    input_tensor_stretched, (q1, q2) = scale_array_stretch_hist(input_tensor, flag_return_quantile=True, quantiles=(0.05, 0.95))
    # imshow_torch_video(input_tensor_stretched, FPS=25, frame_stride=1)
    # imshow_torch(input_tensor_stretched[0])
    # imshow_torch(input_tensor_stretched[1])
    # imshow_torch(input_tensor_stretched[2])
    # imshow_torch(input_tensor_stretched[3])

    ### Get Mean Tensor Over All Frames: ###
    input_tensor_stretched_mean = input_tensor_stretched.mean(0, True)

    # ### RCC Correction On Entire Movie: ###
    # # input_tensor_low_FPS_CC_RCC, Residual = perform_RCC_on_tensor(input_tensor_low_FPS_CC.mean(0, True).cpu(), lambda_weight=50, number_of_iterations=1)
    # input_tensor_stretched_mean, RCC_Residual = perform_RCC_on_tensor(input_tensor_stretched.mean(0, True).cpu(), lambda_weight=50, number_of_iterations=3)
    # RCC_Residual = RCC_Residual.cuda()
    # input_tensor_stretched = input_tensor_stretched - RCC_Residual
    # input_tensor_stretched = input_tensor_stretched
    # # input_tensor_gradient_h, input_tensor_gradient_w, input_tensor_gradient_total = torch_gradient_stats(input_tensor_stretched_mean)
    # # imshow_torch(input_tensor_gradient_total)
    # # imshow_torch(input_tensor_stretched_mean)
    # # imshow_torch(RCC_Residual)
    # # imshow_torch_video(input_tensor_stretched, FPS=25)
    # # imshow_torch_video(input_tensor_stretched**2, FPS=25)

    # ### FFT-OLA on low spatial frequencies: ###   #TODO: FOR QUICKSHOT!
    # spatial_frequencies_nyquist_fraction = 0.1 #[Fraction]
    # temporal_cutoff_frequency = 60 #[Hz]
    # Fs = 500
    # N_temporal_filter_coefficients = 64
    # input_tensor_stretched = input_tensor_stretched.unsqueeze(0)
    # FFT_OLA_layer = FFT_OLA_PerPixel_Layer_Torch(samples_per_frame=input_tensor_stretched.shape[1], filter_name='hanning', filter_type='lowpass', N=N_temporal_filter_coefficients, Fs=Fs, low_cutoff=temporal_cutoff_frequency, high_cutoff=temporal_cutoff_frequency)
    # output_filtered = FFT_OLA_layer.forward_FFT_coefficients_filtering(input_tensor_stretched, spatial_frequencies_nyquist_fraction=spatial_frequencies_nyquist_fraction)

    ###########################################################################################################################################################################################
    ### Stabilize ALL (NO FPS DROP) Stretched images using regular cross correlation: ###
    ### Set Reference Tensor: ###
    reference_tensor_stretched = input_tensor_stretched[0:1]

    ### Perform Cross Correlation: ###
    # warped_matrix_stretched_CC, shifts_h_CC, shifts_w_CC, cc_CC = align_to_reference_frame_circular_cc(input_tensor_stretched[1:]**2,
    #                                                                                                    input_tensor_stretched[0:-1]**2,
    #                                                                                                    flag_zero_out_center_component_of_CC=True, normalize_over_matrix=True)
    # ### Perform Dudy Cross Correlation: ###
    # flag_gaussian_filter = False
    # flag_fftshift_before_median = False
    # flag_median_per_image = True
    # flag_mean_instead_of_median = True
    # flag_zero_out_zero_component_each_CC = False
    # shifts_H, shifts_W = Super_CC(input_tensor_stretched, reference_tensor_stretched,
    #                               flag_gaussian_filter=flag_gaussian_filter,
    #                               flag_fftshift_before_median=flag_fftshift_before_median,
    #                               flag_median_per_image=flag_median_per_image,
    #                               flag_mean_instead_of_median=flag_mean_instead_of_median,
    #                               flag_zero_out_zero_component_each_CC=flag_zero_out_zero_component_each_CC)

    ### Perform Dudy Cross Correlatio Torch_Layer: ###
    flag_gaussian_filter = True
    flag_fftshift_before_median = False
    flag_median_per_image = True
    flag_mean_instead_of_median = True
    flag_zero_out_zero_component_each_CC = False
    flag_stretch_tensors = False
    flag_W_matrix_method = 1
    flag_shift_CC_to_zero = True
    flag_CC_shift_method = 'bicubic'
    flag_round_shift_before_shifting_CC = False
    max_shift = 101
    max_shift_for_fit = 15
    W_matrix_circle_radius = 15
    warp_method = 'bilinear'
    Super_CC_layer = Super_CC_Layer(flag_gaussian_filter,
                                    flag_fftshift_before_median,
                                    flag_median_per_image,
                                    flag_mean_instead_of_median,
                                    flag_zero_out_zero_component_each_CC,
                                    flag_stretch_tensors=flag_stretch_tensors,
                                    flag_W_matrix_method=flag_W_matrix_method,
                                    flag_shift_CC_to_zero=flag_shift_CC_to_zero,
                                    flag_round_shift_before_shifting_CC=flag_round_shift_before_shifting_CC,
                                    flag_CC_shift_method=flag_CC_shift_method,
                                    max_shift=max_shift,
                                    max_shift_for_fit=max_shift_for_fit,
                                    W_matrix_circle_radius=W_matrix_circle_radius,
                                    warp_method=warp_method)
    shifts_H, shifts_W, input_tensor_stretched_warped_mean_RCC_corrected = Super_CC_layer.forward(input_tensor_stretched)

    ### Stretch Image: ###
    # H,W = input_tensor_stretched_warped_mean_RCC_corrected.shape[-2:]
    # input_tensor_stretched_warped_mean_RCC_corrected[torch.isnan(input_tensor_stretched_warped_mean_RCC_corrected)] = input_tensor_stretched_warped_mean_RCC_corrected[0,0,H//2,W//2]
    input_tensor_stretched_warped_mean_RCC_corrected_stretched = scale_array_stretch_hist(input_tensor_stretched_warped_mean_RCC_corrected, quantiles=(0.01,0.99))
    # input_tensor_stretched_warped_mean_RCC_corrected_stretched = scale_array_clahe(input_tensor_stretched_warped_mean_RCC_corrected)
    # imshow_torch(input_tensor_stretched_warped_mean_RCC_corrected_stretched)

    # ### Shift Matrix: ###
    # input_tensor_stretched_warped = shift_matrix_subpixel(input_tensor_stretched.unsqueeze(0), -shifts_H, -shifts_W, matrix_FFT=None, warp_method='fft').squeeze(0)
    # input_tensor_stretched_original_warped = shift_matrix_subpixel(input_tensor_stretched_original.unsqueeze(0), -shifts_H, -shifts_W, matrix_FFT=None, warp_method='fft').squeeze(0)

    # ### Average Frames: ###
    # number_of_frames_to_average = 11
    # conv_layer = convn_layer_torch()
    # input_tensor_stretched_warped_running_mean = conv_layer.forward(input_tensor_stretched_warped, kernel=torch.ones(number_of_frames_to_average) / number_of_frames_to_average, dim=0)

    # ### Perform Super-Resolution: ###
    # input_tensor_stretched_original_warped_upsampled = torch.nn.Upsample(scale_factor=2)(input_tensor_stretched_original_warped)
    # input_tensor_stretched_original_upsampled = torch.nn.Upsample(scale_factor=2)(input_tensor_stretched_original)
    # concat_tensor_upsampled = torch.cat([input_tensor_stretched_original_upsampled[0], input_tensor_stretched_original_warped_upsampled.mean(0)], -1)
    # # imshow_torch(scale_array_stretch_hist(concat_tensor_upsampled))
    # # imshow_torch(scale_array_stretch_hist(concat_tensor))

    # ### Define concat tensor for ease of handling: ###
    # concat_tensor = torch.cat([input_tensor_stretched_original_warped[0], input_tensor_stretched_original_warped.mean(0)],-1)
    # concat_tensor_stretched = scale_array_stretch_hist(concat_tensor)

    # imshow_torch_video(torch.cat([input_tensor_stretched_original, input_tensor_stretched_original_warped],-1), FPS=25)
    # imshow_torch(input_tensor_stretched_original_warped.mean(0))
    # imshow_torch(concat_tensor_stretched)
    # imshow_torch_video(torch.cat([input_tensor_stretched, input_tensor_stretched_warped],-1), FPS=25)
    # imshow_torch_video(torch.cat([input_tensor_stretched, input_tensor_stretched_warped_running_mean],-1), FPS=25)
    # imshow_torch_video(torch.cat([input_tensor_stretched_original, input_tensor_stretched_warped_running_mean],-1), FPS=25)
    # imshow_torch_video(input_tensor_stretched_warped_running_mean[1:] - input_tensor_stretched_warped_running_mean[0:-1], FPS=25)
    # imshow_torch_video(cc_CC[1:], FPS=25)
    # imshow_torch_video(torch_fftshift(cc_CC)[1:], FPS=25)
    ######################################################################################################################################################################################################








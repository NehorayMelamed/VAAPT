from RapidBase.import_all import *

from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *

### Paths: ###
##############
inference_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints/Inference/model_combined/test_for_bla")
original_frames_folder = os.path.join(datasets_main_folder, '/DataSets/Drones/drones_small')
###




#######################################################################################################################################
### Images: ###
clean_images, clean_image_filenames = read_images_and_filenames_from_folder(str(original_frames_folder),
                                          flag_recursive=True,
                                          crop_size=1024,  #minimum size to be read
                                          max_number_of_images=10,
                                          allowed_extentions=IMG_EXTENSIONS,
                                          flag_return_numpy_or_list='list',
                                          flag_how_to_concat='C',
                                          crop_style='random',
                                          flag_return_torch=False,
                                          transform=None,
                                          flag_to_BW=False,
                                          flag_random_first_frame=False)

experiment_name_addone = 'GeneralAffine'
# max_shifts_vec = np.flip(np.arange(10))
max_shifts_vec = my_linspace(0.2,0.5,10)
algorithm_names = ['MAOR', 'CrossCorrelation', 'NormalizedCrossCorrelation', 'PhaseCorrelation']  #TODO: Add optical flow step with confidence metric?
SNR_vec = [np.inf, 100, 10, 5, 4, 3, 2, 1]
crop_size_vec = [1024]
binning_vec = [1]
rotation_angle_vec = [0]  #[degrees]
scale_factor_vec = [1]
number_of_blur_iterations_per_pixel = 5
reference_to_current_image_SNR_ratio = np.sqrt(1)
reference_image_blur_ratio = 1
number_of_algorithms = 4
number_of_affine_parameters = 4  #delta_x, delta_y, rotation_angle, scale
number_of_blur_options = 2

shift_layer_torch = Shift_Layer_Torch()
warp_layer_torch = Warp_Tensors_Affine_Layer()
results_vec = np.zeros((len(clean_image_filenames), len(max_shifts_vec), len(rotation_angle_vec),
                        len(scale_factor_vec), len(crop_size_vec), len(binning_vec), len(SNR_vec), number_of_algorithms, number_of_blur_options, number_of_affine_parameters)) #last two entries: with/without blur; number of affine params
shifts_vec = np.zeros((len(clean_image_filenames), len(max_shifts_vec), len(rotation_angle_vec),
                       len(scale_factor_vec), len(crop_size_vec), len(binning_vec), len(SNR_vec), number_of_algorithms, number_of_blur_options, number_of_affine_parameters))
for image_counter, current_filename in enumerate(clean_image_filenames):
    ### Get Image: ###
    current_image = clean_images[image_counter]
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
    current_image = torch.Tensor(current_image).unsqueeze(0).unsqueeze(0)
    current_image = current_image / 255
    tic()

    for shift_counter, current_max_shift in enumerate(max_shifts_vec):
        for rotation_counter, current_rotation_angle in enumerate(rotation_angle_vec):
            for scale_counter, current_scale_factor in enumerate(scale_factor_vec):
                ### Shift: ###
                shiftx = current_max_shift
                shifty = current_max_shift

                # ### Print Warp Parameters: ###
                # print('shifts: ' + str(current_max_shift) + ', scale ' + str(current_scale_factor) + ', rotation angle ' + str(current_rotation_angle))

                ### Warp Image: ###
                for blur_counter in np.arange(number_of_blur_options):
                    max_pixels_shift = max(shiftx, shifty)
                    max_pixels_shift = max(shiftx, (current_rotation_angle * pi / 180) * (sqrt(2) * np.max(crop_size_vec)))
                    ### 0==no blur, 1==with blur: ###
                    if blur_counter == 0:
                        current_image_reference = current_image.data
                        current_image_shifted = warp_layer_torch.forward(current_image.data, shiftx, shifty, current_scale_factor, current_rotation_angle)
                    else:
                        ### Blur Reference Image (FOR NOW - IN THE SAME DIRECTION AS THE FINAL SHIFT!!!!!): ###
                        number_of_blur_iterations = int(max_pixels_shift * number_of_blur_iterations_per_pixel)
                        number_of_blur_iterations = max(number_of_blur_iterations, 1)
                        current_image_reference = blur_image_motion_blur_affine_torch(current_image.data,
                                                                                    shiftx*reference_image_blur_ratio,
                                                                                    shifty*reference_image_blur_ratio,
                                                                                    current_scale_factor, current_rotation_angle,
                                                                                    number_of_blur_iterations, warp_layer_torch)
                        ### Blur Second Image: ###
                        #(1). shift to reference image blur location:
                        current_image_initial_shifted = warp_layer_torch.forward(current_image.data,
                                                                            shiftx*reference_image_blur_ratio,
                                                                            shifty*reference_image_blur_ratio,
                                                                            current_scale_factor,
                                                                            current_rotation_angle)
                        #(2). continue to blur image:
                        current_image_blurred = blur_image_motion_blur_affine_torch(current_image_initial_shifted.data,
                                                                                    shiftx,
                                                                                    shifty,
                                                                                    current_scale_factor,
                                                                                    current_rotation_angle,
                                                                                    number_of_blur_iterations,
                                                                                    warp_layer_torch)

                    for crop_counter, current_crop_size in enumerate(crop_size_vec):
                        ### Center Crop: ###
                        current_image_shifted_cropped = crop_torch_batch(current_image_shifted.data, current_crop_size, crop_style='center')
                        current_image_reference_cropped = crop_torch_batch(current_image_reference.data, current_crop_size, crop_style='center')

                        # print(current_image.shape)
                        for SNR_counter, current_SNR in enumerate(SNR_vec):
                            # print(1/sqrt(current_SNR))
                            ### Add Noise To Image: ###
                            current_image_noisy = current_image_reference_cropped + 1/sqrt(current_SNR*reference_to_current_image_SNR_ratio)*torch.randn_like(current_image_reference_cropped)
                            current_image_shifted_noisy = current_image_shifted_cropped + 1/sqrt(current_SNR)*torch.randn_like(current_image_shifted_cropped)
                            # current_PSNR = PSNR_Loss()(current_image, current_image_noisy)
                            # current_PSNR = float(current_PSNR.numpy())
                            # actual_SNR_linear = (current_image.abs().mean() * sqrt(current_SNR))**2
                            # fig = imshow_torch_multiple((current_image.clip(0, 1), current_image_noisy.clip(0, 1)))
                            # title('PSNR ' + decimal_notation(current_PSNR,3) + ', Sigma ' + str(int(1/sqrt(current_SNR)*255)) + ', Spoken SNR ' + str(current_SNR) + ', Linear SNR ' + decimal_notation(actual_SNR_linear,2))
                            # image_folder_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD/scenarios')
                            # image_folder_path = os.path.join(image_folder_path, 'Registration')
                            # image_folder_path = os.path.join(image_folder_path, 'Example Images')
                            # path_make_path_if_none_exists(image_folder_path)
                            # current_filename = os.path.join(image_folder_path, 'PSNR ' + decimal_notation(current_PSNR,3) +
                            #                                 ', Sigma ' + str(int(1/sqrt(current_SNR)*255)) + ', Spoken SNR ' + str(current_SNR)+ ', Linear SNR ' + decimal_notation(actual_SNR_linear,2) + '.png')
                            # savefig(current_filename)
                            # close()

                            for binning_counter, current_binning in enumerate(binning_vec):
                                ### Do Initial Average Pooling / Binning: ###
                                average_pooling_layer = nn.AvgPool2d(current_binning)
                                current_image_noisy_final = average_pooling_layer(current_image_noisy)
                                current_image_shifted_noisy_final = average_pooling_layer(current_image_shifted_noisy)

                                for algorithm_counter in np.arange(0, number_of_algorithms):
                                    ### Loop Over The Difference Algorithms To Be Tested: ###
                                    try:
                                        ### Register Images (return numpy arrays): ###
                                        algorithm_name = algorithm_names[algorithm_counter]
                                        affine_parameters, clean_frame_previous_warped = register_images(current_image_noisy_final.data, current_image_shifted_noisy_final.data,
                                                        algorithm_name,
                                                        search_area=int(np.ceil(max_pixels_shift) + 2),
                                                        inner_crop_size_to_use=np.inf,
                                                        downsample_factor=1,  # binning / average_pooling
                                                        flag_do_initial_SAD=False, initial_SAD_search_area=5)


                                        ### If the output has less affine parameters then possible (for instance, shift only algorithm), then assume zeros for other parameters: ###
                                        affine_parameters_list = []
                                        for current_parameter in affine_parameters:
                                            if type(current_parameter) is not np.ndarray:
                                                affine_parameters_list.append(np.array(current_parameter))
                                            else:
                                                affine_parameters_list.append(current_parameter)
                                        if len(affine_parameters_list) == 2:
                                            affine_parameters_list.append(np.array([1.0])) # scale = 1
                                            affine_parameters_list.append(np.array([0.0])) # rotation = 0

                                        ### Print results in a pretty way: ###
                                        real_shifts_string = '(' + decimal_notation(shiftx, 3) + ', ' + decimal_notation(shifty,3) + ')'
                                        real_scale_string = decimal_notation(current_scale_factor, 1)
                                        real_rotation_string = decimal_notation(current_rotation_angle, 2)
                                        found_shifts_string = '(' + decimal_notation(affine_parameters_list[0], 3) + ', ' + decimal_notation(affine_parameters_list[1],3) + ')'
                                        found_scale_string = decimal_notation(affine_parameters_list[2],2)
                                        found_rotation_string = decimal_notation(affine_parameters_list[3],2)
                                        # print( '*'*10 +
                                        #        'Algorithm: ' + algorithm_names[algorithm_counter] + ', SNR: ' + str(current_SNR) + '\n'
                                        #       ', real shifts: ' + real_shifts_string + ', deduced shifts: ' + found_shifts_string + '\n' +
                                        #       ', real scale: ' + real_scale_string + ' , deduced scale: ' + found_scale_string + '\n' +
                                        #       ', real_rotation: ' + real_rotation_string + ', deduced rotation: ' + found_rotation_string + '\n' +
                                        #        '*'*10)

                                        ### Assign Results: ###
                                        current_indices = [image_counter, shift_counter, rotation_counter, scale_counter,
                                                    crop_counter, binning_counter, SNR_counter, algorithm_counter, blur_counter]
                                        results_vec[tuple([*current_indices,0])] = affine_parameters_list[0]
                                        results_vec[tuple([*current_indices,1])] = affine_parameters_list[1]
                                        results_vec[tuple([*current_indices,2])] = affine_parameters_list[2]
                                        results_vec[tuple([*current_indices,3])] = affine_parameters_list[3]
                                        shifts_vec[tuple([*current_indices,0])] = shiftx
                                        shifts_vec[tuple([*current_indices,1])] = shifty
                                        shifts_vec[tuple([*current_indices,2])] = current_scale_factor
                                        shifts_vec[tuple([*current_indices,3])] = current_rotation_angle
                                    except:
                                        1
    toc()

# ### Clean up data - there are sometimes huge outliers, deal with it later but for analysis purposes get rid of them: ###
# clean_results_vec = results_vec.__copy__()
# for shift_counter, current_max_shift in enumerate(max_shifts_vec):
#     for SNR_counter, current_SNR in enumerate(SNR_vec):
#         for algorithm_counter in np.arange(0, number_of_algorithms):
#             current_results_mat = clean_results_vec[:,shift_counter, SNR_counter, algorithm_counter,:]
#             current_shifts_mat = shifts_vec[:,shift_counter, SNR_counter, algorithm_counter,:]
#             current_median = np.median(np.abs(current_results_mat - current_shifts_mat))  #get median for current algorithm, current SNR, and current shift
#             indices_above_threshold = (np.abs(current_results_mat - current_shifts_mat) > current_median * 3)
#             current_results_mat[indices_above_threshold] = np.nan
#             clean_results_vec[:, shift_counter, SNR_counter, algorithm_counter, :] = current_results_mat
# percent_nan = np.sum(np.isnan(clean_results_vec)) / np.size(clean_results_vec)
# results_vec = clean_results_vec
#
# ### Analyze Results: ###
# results_vec = results_vec.clip(-6,6)
# mean_vec = np.nanmean(shifts_vec - results_vec, 0)
# mean_vec = mean_vec[..., -1]
# # std_vec = np.nanstd(shifts_vec - results_vec, 0)
# std_vec = np.nanmean(np.abs(shifts_vec - results_vec), 0)
# std_vec = std_vec[..., -1]

### Saved arrays names: ###
folder_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD/scenarios')
folder_path = os.path.join(folder_path, 'Registration')
path_make_path_if_none_exists(folder_path)

filename_parameters_dict = os.path.join(folder_path, experiment_name_addone +  '_parameter_dict_known_shifts.pkl')
filename_results = os.path.join(folder_path, experiment_name_addone + '_results_vec_known_shifts.npy')
filename_original_shifts = os.path.join(folder_path, experiment_name_addone + '_shifts_vec_known_shifts.npy')


### Save Parameters dict: ###
parameters_dict = EasyDict()
parameters_dict.max_shifts_vec = max_shifts_vec
parameters_dict.algorithm_names = algorithm_names
parameters_dict.SNR_vec = SNR_vec
parameters_dict.crop_size_vec = crop_size_vec
parameters_dict.binning_vec = binning_vec
parameters_dict.rotation_angle_vec = rotation_angle_vec
parameters_dict.scale_factor_vec = scale_factor_vec
parameters_dict.number_of_blur_iterations_per_pixel = number_of_blur_iterations_per_pixel
parameters_dict.reference_to_current_image_SNR_ratio = reference_to_current_image_SNR_ratio
parameters_dict.reference_image_blur_ratio = reference_image_blur_ratio
parameters_dict.number_of_algorithms = number_of_algorithms
parameters_dict.number_of_affine_parameters = number_of_affine_parameters
parameters_dict.number_of_blur_options = number_of_blur_options


### Save Arrays: ###
np.save(filename_results, results_vec)
np.save(filename_original_shifts, shifts_vec)
save_dict(parameters_dict, filename_parameters_dict)



# ### Load Arrays if wanted: ###
# mean_vec = np.load(filename_mean)
# std_vec = np.load(filename_std)
# results_vec = np.load(filename_results)
# shifts_vec = np.load(filename_original_shifts)







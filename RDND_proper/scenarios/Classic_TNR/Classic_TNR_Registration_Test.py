
from RapidBase.import_all import *

from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RapidBase.TrainingCore.training_utils import path_make_path_if_none_exists
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *



### Paths: ###
##############
inference_path = path_fix_path_for_linux("/home/mafat/Pytorch_Checkpoints/Inference/model_combined/test_for_bla")
original_frames_folder = os.path.join(datasets_main_folder, '/DataSets/Drones/drones_small')
###



#######################################################################################################################################
### IO: ###
IO_dict = EasyDict()
### Crop Parameters: ###
IO_dict.crop_X = 340
IO_dict.crop_Y = 270
### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_use_train_dataset_split = False
IO_dict.train_dataset_split_to_test_factor = 0.1
### Noise Parameters: ###
IO_dict.sigma_to_dataset = 1/sqrt(np.inf)*255
IO_dict.SNR_to_model = np.inf
IO_dict.sigma_to_model = 1/sqrt(IO_dict.SNR_to_model)*255
# IO_dict.sigma_to_model = 80
### Blur Parameters: ###
IO_dict.blur_size = 20
### Number of frames to load: ###
IO_dict.NUM_IN_FRAMES = 5 # temporal size of patch
### Training Flags: ###
IO_dict.non_valid_border_size = 30
### Universal Training Parameters: ###
IO_dict.batch_size = 8
IO_dict.number_of_epochs = 60000
#### Assign Device (for instance, we might want to do things on the gpu in the dataloader): ###
IO_dict.device = torch.device('cuda')
#######################################################################################################################################




#######################################################################################################################################
### Images: ###
clean_images, clean_image_filenames = read_images_and_filenames_from_folder(str(original_frames_folder),
                                          flag_recursive=True,
                                          crop_size=np.inf,
                                          max_number_of_images=100,
                                          allowed_extentions=IMG_EXTENSIONS,
                                          flag_return_numpy_or_list='list',
                                          flag_how_to_concat='C',
                                          crop_style='random',
                                          flag_return_torch=False,
                                          transform=None,
                                          flag_to_BW=False,
                                          flag_random_first_frame=False)


# max_shifts_vec = np.flip(np.arange(10))
max_shifts_vec = my_linspace(0,5,50)
# max_shifts_vec = [5]
algorithm_names = ['Maor Algorithm', 'Cross Correlation', 'OpenCV Features', 'Phase Only Correlation']
SNR_vec = [np.inf, 100, 10, 5, 4, 3, 2, 1]
shift_layer_torch = Shift_Layer_Torch()
crop_size_x = 512
number_of_algorithms = 4
results_vec = np.zeros((len(clean_image_filenames), len(max_shifts_vec), len(SNR_vec), number_of_algorithms, 2))
shifts_vec = np.zeros((len(clean_image_filenames), len(max_shifts_vec), len(SNR_vec), number_of_algorithms, 2))
for shift_counter, current_max_shift in enumerate(max_shifts_vec):
    tic()
    for image_counter, current_filename in enumerate(clean_image_filenames):
        ### Get Image: ###
        current_image = clean_images[image_counter]
        current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        current_image = torch.Tensor(current_image).unsqueeze(0).unsqueeze(0)
        current_image = current_image / 255

        ### Shift: ###
        # Random Shift: #
        # shiftx = get_random_number_in_range(-current_max_shift, current_max_shift)
        # shifty = get_random_number_in_range(-current_max_shift, current_max_shift)
        # Pre-Determined Shift (so i can plot a line of mean prediction with error bars): #
        shiftx = current_max_shift
        shifty = current_max_shift
        print('shifts: ' + str(shiftx) + ', ' + str(shifty))
        current_image_shifted = shift_layer_torch.forward(current_image, shiftx, shifty)

        ### Center Crop: ###
        current_image_shifted = crop_torch_batch(current_image_shifted, crop_size_x, crop_style='center')
        current_image = crop_torch_batch(current_image, crop_size_x, crop_style='center')


        # print(current_image.shape)
        for SNR_counter, current_SNR in enumerate(SNR_vec):
            # print(1/sqrt(current_SNR))
            ### Add Noise To Image: ###
            current_image_noisy = current_image + 1/sqrt(current_SNR)*torch.randn_like(current_image)
            current_image_shifted_noisy = current_image_shifted + 1/sqrt(current_SNR)*torch.randn_like(current_image)
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


            for algorithm_counter in np.arange(0, number_of_algorithms):
                try:
                    ### Register Images (return numpy arrays): ###
                    #TODO: sometimes i get wild outliers...i need to put max bounds on possible results. for instance only search a certain distance around 0
                    if algorithm_counter == 0:
                        shifts_array, clean_frame_previous_warped = register_images(current_image_noisy, current_image_shifted_noisy) # Maor Algorithm  #TODO: fix outputs to match all others
                    elif algorithm_counter == 1:
                        shifts_array, clean_frame_previous_warped = register_images2(current_image_noisy, current_image_shifted_noisy) # Using Cross Correlation
                    elif algorithm_counter == 2:
                        shifts_array, clean_frame_previous_warped = register_images3(current_image_noisy, current_image_shifted_noisy)  # Using Feature Detection and Homography matrix
                    elif algorithm_counter == 3:
                        shifts_array, clean_frame_previous_warped = register_images5(current_image_noisy, current_image_shifted_noisy)  # Using Phase Only Correlation (POC)
                        ### TODO: there is an ambiguity in sign, so until i solve it i simply assign the correct sign: ###
                        bla1 = np.abs(shifts_array[0]) * np.sign(shiftx)
                        bla2 = np.abs(shifts_array[1]) * np.sign(shifty)
                        shifts_array = [bla1, bla2]
                    # elif algorithm_counter == 4:
                    #     shifts_array, clean_frame_previous_warped = register_images6(current_image_noisy, current_image_shifted_noisy)  # Using Normalized Cross Correlation



                    # clean_frame_previous_warped = register_images4(current_image_noisy, current_image_shifted_noisy, 'DeepFlow')   #DeepFlow, SimpleFlow, and TVL1
                    # imshow_torch(current_image_noisy)
                    real_shifts_string = decimal_notation(shiftx, 3) + ', ' + decimal_notation(shifty,3)
                    found_shifts_string = decimal_notation(shifts_array[0], 3) + ', ' + decimal_notation(shifts_array[1],3)
                    print('Algorithm: ' + algorithm_names[algorithm_counter] + ', SNR: ' + str(current_SNR) + ', real shifts: ' + real_shifts_string + ', deduced shifts: ' + found_shifts_string)

                    ### Assign Results: ###
                    results_vec[image_counter, shift_counter, SNR_counter, algorithm_counter, 0] = shifts_array[0]
                    results_vec[image_counter, shift_counter, SNR_counter, algorithm_counter, 1] = shifts_array[1]
                    shifts_vec[image_counter, shift_counter, SNR_counter, algorithm_counter, 0] = shiftx
                    shifts_vec[image_counter, shift_counter, SNR_counter, algorithm_counter, 1] = shifty
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
# folder_path = path_get_current_working_directory()
folder_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD/scenarios')
folder_path = os.path.join(folder_path, 'Registration')
path_make_path_if_none_exists(folder_path)
# ### Random Shifts: ###
# filename_mean = os.path.join(folder_path,  'mean_vec.npy')
# filename_std = os.path.join(folder_path,  'std_vec.npy')
# filename_results = os.path.join(folder_path, 'results_vec.npy')
# filename_original_shifts = os.path.join(folder_path, 'shifts_vec.npy')
### Specific Shifts: ###
filename_mean = os.path.join(folder_path,  'mean_vec_known_shifts.npy')
filename_std = os.path.join(folder_path,  'std_vec_known_shifts.npy')
filename_results = os.path.join(folder_path, 'results_vec_known_shifts.npy')
filename_original_shifts = os.path.join(folder_path, 'shifts_vec_known_shifts.npy')
#TODO: save shifts and SNR vecs

### Save Arrays: ###
# np.save(filename_mean, mean_vec)
# np.save(filename_std, std_vec)
np.save(filename_results, results_vec)
np.save(filename_original_shifts, shifts_vec)

# ### Load Arrays if wanted: ###
# mean_vec = np.load(filename_mean)
# std_vec = np.load(filename_std)
# results_vec = np.load(filename_results)
# shifts_vec = np.load(filename_original_shifts)


# ### Show All Cuts of Mean(Shift) & STD(Shift) For Certain SNR and Algorithm Up To Shift=1[Pixel]: ###
# max_value = 10
# max_shift_to_check = 1
# for SNR_counter, current_SNR in enumerate(SNR_vec):
#     for algorithm_counter in np.arange(0, number_of_algorithms):
#         current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
#         current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
#         algorithm_name = algorithm_names[algorithm_counter]
#
#         first_index_larger_then_1 = find(max_shifts_vec > max_shift_to_check)[0][0][0]
#         current_std_vec = current_std_vec[0:first_index_larger_then_1]
#         current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
#         current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]
#
#         ### Mean Vec: ###
#         if algorithm_counter == 0:
#             fig, ax = plt.subplots(1, 1)
#         img = ax.plot(current_mean_vec.clip(0, max_value))
#         x_label_list = []
#         y_label_list = []
#         for i in np.arange(len(current_max_shifts_vec)):
#             x_label_list.append(str(current_max_shifts_vec[i]))
#         # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
#         # ax.set_xticklabels(x_label_list)
#         ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
#         ax.xaxis.set_ticklabels(x_label_list)
#         xlabel('Shift')
#         ylabel('Mean')
#         ylim([0, 1])
#         title('Mean(Shift): ' + algorithm_name + ', SNR: ' + str(np.inf))
#         legend(algorithm_names)
#     savefig(os.path.join(folder_path, 'Specific All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(current_SNR)) + ', Shift Smaller Than 1')
#     close()
#
#     for algorithm_counter in np.arange(0, number_of_algorithms):
#         current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
#         current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
#         algorithm_name = algorithm_names[algorithm_counter]
#
#         first_index_larger_then_1 = find(max_shifts_vec > max_shift_to_check)[0][0][0]
#         current_std_vec = current_std_vec[0:first_index_larger_then_1]
#         current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
#         current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]
#
#         ### Mean Vec: ###
#         if algorithm_counter == 0:
#             fig, ax = plt.subplots(1, 1)
#         img = ax.plot(current_std_vec.clip(0, max_value))
#         x_label_list = []
#         y_label_list = []
#         for i in np.arange(len(current_max_shifts_vec)):
#             x_label_list.append(str(current_max_shifts_vec[i]))
#         # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
#         # ax.set_xticklabels(x_label_list)
#         ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
#         ax.xaxis.set_ticklabels(x_label_list)
#         xlabel('Shift')
#         ylabel('STD')
#         ylim([0,1])
#         title('STD(Shift): ' + algorithm_name + ', SNR: ' + str(current_SNR))
#         legend(algorithm_names)
#     savefig(os.path.join(folder_path, 'Specific All Algorithms Plot STD(Shift): ' + ', SNR: ' + str(current_SNR)) + ', Shift Smaller Than 1')
#     close()
#
#
#
# ### Show All Cuts of Mean(Shift) & STD(Shift) For Certain SNR and Algorithm Up To Shift=5[Pixel]: ###
# max_value = 10
# max_shift_to_check = 5
# for SNR_counter, current_SNR in enumerate(SNR_vec):
#     for algorithm_counter in np.arange(0, number_of_algorithms):
#         current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
#         current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
#         algorithm_name = algorithm_names[algorithm_counter]
#
#         first_index_larger_then_1 = find(max_shifts_vec >= max_shift_to_check-0.1)[0][0][0]
#         current_std_vec = current_std_vec[0:first_index_larger_then_1]
#         current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
#         current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]
#
#         ### Mean Vec: ###
#         if algorithm_counter == 0:
#             fig, ax = plt.subplots(1, 1)
#         img = ax.plot(current_mean_vec.clip(0, max_value))
#         x_label_list = []
#         y_label_list = []
#         for i in np.arange(len(current_max_shifts_vec)):
#             x_label_list.append(str(current_max_shifts_vec[i]))
#         # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
#         # ax.set_xticklabels(x_label_list)
#         ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
#         ax.xaxis.set_ticklabels(x_label_list)
#         xlabel('Shift')
#         ylabel('Mean')
#         ylim([0, 5])
#         title('Mean(Shift): ' + algorithm_name + ', SNR: ' + str(np.inf))
#         legend(algorithm_names)
#     savefig(os.path.join(folder_path, 'Specific All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(current_SNR)))
#     close()
#
#     for algorithm_counter in np.arange(0, number_of_algorithms):
#         current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
#         current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
#         algorithm_name = algorithm_names[algorithm_counter]
#
#         first_index_larger_then_1 = find(max_shifts_vec >= max_shift_to_check-0.1)[0][0][0]
#         current_std_vec = current_std_vec[0:first_index_larger_then_1]
#         current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
#         current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]
#
#         ### Mean Vec: ###
#         if algorithm_counter == 0:
#             fig, ax = plt.subplots(1, 1)
#         img = ax.plot(current_std_vec.clip(0, max_value))
#         x_label_list = []
#         y_label_list = []
#         for i in np.arange(len(current_max_shifts_vec)):
#             x_label_list.append(str(current_max_shifts_vec[i]))
#         # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
#         # ax.set_xticklabels(x_label_list)
#         ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
#         ax.xaxis.set_ticklabels(x_label_list)
#         xlabel('Shift')
#         ylabel('STD')
#         ylim([0,5])
#         title('STD(Shift): ' + algorithm_name + ', SNR: ' + str(current_SNR))
#         legend(algorithm_names)
#     savefig(os.path.join(folder_path, 'Specific All Algorithms Plot STD(Shift): ' + ', SNR: ' + str(current_SNR)))
#     close()
#
#
#
# ##### Show All 1D Plots On Same Graph: #####
# ### All algorithms mean(shift) for snr=inf: ###
# max_value = 10
# for algorithm_counter in np.arange(number_of_algorithms):
#     current_mean_vec = mean_vec[:,:,algorithm_counter]
#     current_std_vec = std_vec[:,:,algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     ##################################################################################
#     ### Show Vecs For No Noise: ###
#     current_std_vec_1 = current_std_vec[:,0]
#     current_mean_vec_1 = current_mean_vec[:,0]
#
#     ### Mean Vec: ###
#     if algorithm_counter == 0:
#         fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_mean_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('Mean')
#     title('Shift(Mean): ' + algorithm_name + ', SNR: ' + str(np.inf))
#     legend(algorithm_names)
# savefig(os.path.join(folder_path, 'All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(np.inf)))
# close()
#
# ### All algorithms std(shift) for snr=inf: ###
# for algorithm_counter in np.arange(number_of_algorithms):
#     current_mean_vec = mean_vec[:,:,algorithm_counter]
#     current_std_vec = std_vec[:,:,algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     ##################################################################################
#     ### Show Vecs For No Noise: ###
#     current_std_vec_1 = current_std_vec[:,0]
#     current_mean_vec_1 = current_mean_vec[:,0]
#
#     ### Mean Vec: ###
#     if algorithm_counter == 0:
#         fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_std_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('STD')
#     title('STD(Shift): ' + algorithm_name + ', SNR: ' + str(np.inf))
#     legend(algorithm_names)
# savefig(os.path.join(folder_path, 'All Algorithms Plot STD(Shift): ' + ', SNR: ' + str(np.inf)))
# close()
#
# ### All algorithms mean(shift) for snr=specific: ###
# for algorithm_counter in np.arange(number_of_algorithms):
#     current_mean_vec = mean_vec[:,:,algorithm_counter]
#     current_std_vec = std_vec[:,:,algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     ##################################################################################
#     ### Show Vecs For No Noise: ###
#     SNR_index = 3
#     current_PSNR = SNR_vec[SNR_index]
#     current_std_vec_1 = current_std_vec[:, SNR_index]
#     current_mean_vec_1 = current_mean_vec[:, SNR_index]
#
#     ### Mean Vec: ###
#     if algorithm_counter == 0:
#         fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_mean_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('Mean')
#     title('Shift(Mean): ' + ', SNR: ' + str(current_PSNR))
#     legend(algorithm_names)
# savefig(os.path.join(folder_path, 'All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(current_PSNR)))
# close()
#
# ### All algorithms std(shift) for snr=specific: ###
# for algorithm_counter in np.arange(number_of_algorithms):
#     current_mean_vec = mean_vec[:,:,algorithm_counter]
#     current_std_vec = std_vec[:,:,algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     ##################################################################################
#     ### Show Vecs For No Noise: ###
#     SNR_index = 3
#     current_PSNR = SNR_vec[SNR_index]
#     current_std_vec_1 = current_std_vec[:, SNR_index]
#     current_mean_vec_1 = current_mean_vec[:, SNR_index]
#
#     ### Mean Vec: ###
#     if algorithm_counter == 0:
#         fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_std_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('STD')
#     title('STD(Shift): ' + ', SNR: ' + str(current_PSNR))
#     legend(algorithm_names)
# savefig(os.path.join(folder_path, 'All Algorithms Plot STD(Shift): ' + ', SNR: ' + str(current_PSNR)))
# close()
#
#
# #### All algorithms Together - Mean(Shift) for certain SNR: #####
# for algorithm_counter in np.arange(number_of_algorithms):
#     current_mean_vec = mean_vec[:,:,algorithm_counter]
#     current_std_vec = std_vec[:,:,algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     ##################################################################################
#     ### Show Vecs For No Noise: ###
#     SNR_index = 3
#     current_PSNR = SNR_vec[SNR_index]
#     current_std_vec_1 = current_std_vec[:, SNR_index]
#     current_mean_vec_1 = current_mean_vec[:, SNR_index]
#
#     ### Mean Vec: ###
#     if algorithm_counter == 0:
#         fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_mean_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('Mean')
#     title('Mean(Shift): ' + algorithm_name + ', SNR: ' + str(current_PSNR))
#     legend(algorithm_names)
# savefig(os.path.join(folder_path, 'All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(np.inf)))
# close()
#
#
#
#
# ##### Show 1D Plots: #####
# for algorithm_counter in np.arange(number_of_algorithms):
#     current_mean_vec = mean_vec[:,:,algorithm_counter]
#     current_std_vec = std_vec[:,:,algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     ##################################################################################
#     ### Show Vecs For No Noise: ###
#     current_std_vec_1 = current_std_vec[:,0]
#     current_mean_vec_1 = current_mean_vec[:,0]
#
#     ### Mean Vec: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_mean_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('Mean')
#     title('Shift(Mean): ' + algorithm_name + ', SNR: ' + str(np.inf) + ', Algorithm: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Plot Mean(Shift): ' + algorithm_name + ', SNR: ' + str(np.inf) + ', Algorithm: ' + algorithm_name))
#     close()
#
#     ### STD Vec: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_std_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('STD')
#     title('STD(Shift): ' + algorithm_name + ', SNR: ' + str(np.inf) + ', Algorithm: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Plot STD(Shift): ' + algorithm_name + ', SNR: ' + str(np.inf) + ', Algorithm: ' + algorithm_name))
#     close()
#     ##################################################################################
#
#     ##################################################################################
#     ##### Show vecs for certain SNR: ######
#     ### Mean Vec: ###
#     SNR_index = 3
#     current_PSNR = SNR_vec[SNR_index]
#     current_std_vec_1 = current_std_vec[:, SNR_index]
#     current_mean_vec_1 = current_mean_vec[:, SNR_index]
#     fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_mean_vec_1.clip(0,max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('Mean')
#     title('Mean(Shift): ' + algorithm_name + ', SNR: ' + str(current_PSNR) + ', Algorithm: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Plot Mean(Shift): ' + algorithm_name + ', SNR: ' + str(current_PSNR) + ', Algorithm: ' + algorithm_name))
#     close()
#
#     ### STD Vec: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_std_vec_1.clip(0, max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(max_shifts_vec)):
#         x_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(max_shifts_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('Shift')
#     ylabel('STD')
#     title('STD(Shift): ' + algorithm_name + ', SNR: ' + str(current_PSNR) + ', Algorithm: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Plot STD(Shift): ' + algorithm_name + ', SNR: ' + str(current_PSNR) + ', Algorithm: ' + algorithm_name))
#     close()
#     ##################################################################################
#
#     ##################################################################################
#     ### Show Vecs As Function Of SNR For Certain Shifts: ###
#     shift_index = 4
#     current_std_vec_1 = current_std_vec[shift_index, :]
#     current_mean_vec_1 = current_mean_vec[shift_index, :]
#     current_shift = max_shifts_vec[shift_index]
#
#     ### Mean Vec: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_mean_vec_1.clip(0,max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(SNR_vec)):
#         x_label_list.append(str(SNR_vec[i]))
#     ax.set_xticks(np.arange(len(SNR_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('PSNR')
#     ylabel('Mean')
#     title('Mean(SNR): ' + algorithm_name + ', Shift: ' + str(current_shift) + ', Algorithm: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Plot Mean(SNR): ' + algorithm_name + ', Shift: ' + str(current_shift) + ', Algorithm: ' + algorithm_name) + '.png')
#     close()
#
#     ### STD Vec: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.plot(current_std_vec_1.clip(0,max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(SNR_vec)):
#         x_label_list.append(str(SNR_vec[i]))
#     ax.set_xticks(np.arange(len(SNR_vec)))
#     ax.set_xticklabels(x_label_list)
#     xlabel('PSNR')
#     ylabel('STD')
#     title('STD(SNR): ' + algorithm_name + ', Shift: ' + str(current_shift) + ', Algorithm: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Plot STD(SNR): ' + algorithm_name + ', Shift: ' + str(current_shift) + ', Algorithm: ' + algorithm_name + '.png'))
#     close()
#     ##################################################################################
#
#
#
# ##### ShowCase Entire Mat: #####
# for algorithm_counter in np.arange(number_of_algorithms):
#     current_mean_vec = mean_vec[:,:,algorithm_counter]
#     current_std_vec = std_vec[:,:,algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     ### STD Mat: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.imshow(current_std_vec.clip(0,max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(SNR_vec)):
#         x_label_list.append(str(SNR_vec[i]))
#     for i in np.arange(len(max_shifts_vec)):
#         y_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(SNR_vec)))
#     ax.set_yticks(np.arange(len(max_shifts_vec)))
#     ax.set_yticklabels(y_label_list)
#     ax.set_xticklabels(x_label_list)
#     xlabel('PSNR')
#     ylabel('Shift')
#     fig.colorbar(img)
#     title('Shift STD: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'STD Mat ' + algorithm_name))
#     close()
#
#     ### Mean Mat: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.imshow(current_mean_vec)
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(SNR_vec)):
#         x_label_list.append(str(SNR_vec[i]))
#     for i in np.arange(len(max_shifts_vec)):
#         y_label_list.append(str(max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(SNR_vec)))
#     ax.set_yticks(np.arange(len(max_shifts_vec)))
#     ax.set_yticklabels(y_label_list)
#     ax.set_xticklabels(x_label_list)
#     xlabel('PSNR')
#     ylabel('Shift')
#     fig.colorbar(img)
#     title('Shift Mean: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Mean Mat ' + algorithm_name))
#     close()
#
#
# ##### Show Case Up To 1 Pixel: #####
# for algorithm_counter in np.arange(number_of_algorithms):
#     max_shift_to_check = 1
#     current_mean_vec = mean_vec[:, :, algorithm_counter]
#     current_std_vec = std_vec[:, :, algorithm_counter]
#     algorithm_name = algorithm_names[algorithm_counter]
#
#     first_index_larger_then_1 = find(max_shifts_vec>max_shift_to_check)[0][0][0]
#     current_std_vec = current_std_vec[0:first_index_larger_then_1, :]
#     current_mean_vec = current_mean_vec[0:first_index_larger_then_1, :]
#     current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]
#     ### STD Mat: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.imshow(current_std_vec.clip(0,max_value))
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(SNR_vec)):
#         x_label_list.append(str(SNR_vec[i]))
#     for i in np.arange(len(current_max_shifts_vec)):
#         y_label_list.append(str(current_max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(SNR_vec)))
#     ax.set_yticks(np.arange(len(current_max_shifts_vec)))
#     ax.set_yticklabels(y_label_list)
#     ax.set_xticklabels(x_label_list)
#     xlabel('PSNR')
#     ylabel('Shift')
#     fig.colorbar(img)
#     title('Shift STD: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'STD Mat Up To Shift 1' + algorithm_name))
#     close()
#
#     ### Mean Mat: ###
#     fig, ax = plt.subplots(1, 1)
#     img = ax.imshow(current_mean_vec)
#     x_label_list = []
#     y_label_list = []
#     for i in np.arange(len(SNR_vec)):
#         x_label_list.append(str(SNR_vec[i]))
#     for i in np.arange(len(current_max_shifts_vec)):
#         y_label_list.append(str(current_max_shifts_vec[i]))
#     ax.set_xticks(np.arange(len(SNR_vec)))
#     ax.set_yticks(np.arange(len(current_max_shifts_vec)))
#     ax.set_yticklabels(y_label_list)
#     ax.set_xticklabels(x_label_list)
#     xlabel('PSNR')
#     ylabel('Shift')
#     fig.colorbar(img)
#     title('Shift Mean: ' + algorithm_name)
#     savefig(os.path.join(folder_path, 'Mean Mat Up To Shift 1' + algorithm_name))
#     close()







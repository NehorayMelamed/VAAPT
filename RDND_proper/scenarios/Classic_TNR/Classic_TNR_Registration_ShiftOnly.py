Registration: ###
recovered_angle, recovered_scale, recovered_translation, input_image_1_displaced = affine_registration_layer.forward(input_image_1,
                                                                                                                     input_image_1_displaced,
                                                                                                                     downsample_factor=initial_scale_rotation_registration_downsample_factor,
                                                                                                                     flag_return_shifted_image=True)
toc()
imshow_torch(input_image_1_displaced - input_image_1)

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
algorithm_names = ['Maor Algorithm', 'Cross Correlation', 'Normalized Cross Correlation', 'Phase Only Correlation', 'OpenCV Features']
SNR_vec = [np.inf, 100, 10, 5, 4, 3, 2, 1]
shift_layer_torch = Shift_Layer_Torch()
crop_size_x = 512
crop_size_vec = [16,32,64,124,256,512,1024]
binning_size_vec = [1,2,3,4,8]  #TODO: add this possibility
number_of_algorithms = 2
number_of_blur_iterations_per_pixel = 5
number_of_affine_parameters = 2 # (shiftx,shifty)
results_vec = np.zeros((len(clean_image_filenames), len(max_shifts_vec), len(SNR_vec),
                        len(crop_size_vec), number_of_algorithms, 2, number_of_affine_parameters)) #the 2 is blur/no-blur
shifts_vec = np.zeros((len(clean_image_filenames), len(max_shifts_vec), len(SNR_vec),
                       len(crop_size_vec), number_of_algorithms, 2, number_of_affine_parameters))

for image_counter, current_filename in enumerate(clean_image_filenames):
    ### Get Image: ###
    current_image = clean_images[image_counter]
    current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
    current_image = torch.Tensor(current_image).unsqueeze(0).unsqueeze(0)
    current_image = current_image / 255

    ### Shift Image: ###
    for shift_counter, current_max_shift in enumerate(max_shifts_vec):
        tic()
        shiftx = current_max_shift
        shifty = current_max_shift
        print('shifts: ' + str(shiftx) + ', ' + str(shifty))

        for blur_counter in np.arange(2):
            ### 0==no blur, 1==with blur: ###
            if blur_counter == 0:
                current_image_shifted = shift_layer_torch.forward(current_image, shiftx, shifty)
            else:
                current_number_of_blur_iterations = max(int(number_of_blur_iterations_per_pixel * current_max_shift), 1)
                #TODO: add blur to the original image !
                #TODO: add possibility of different blur / SNR for reference and current image!!!
                current_image_shifted = blur_image_motion_blur_torch(current_image, shiftx, shifty, current_number_of_blur_iterations, shift_layer_torch)

            for crop_counter, current_crop_size in enumerate(crop_size_vec):
                ### Center Crop: ###
                current_image_shifted = crop_torch_batch(current_image_shifted, current_crop_size, crop_style='center')
                current_image = crop_torch_batch(current_image, current_crop_size, crop_style='center')

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

                            # imshow_torch(current_image_noisy)
                            real_shifts_string = decimal_notation(shiftx, 3) + ', ' + decimal_notation(shifty,3)
                            found_shifts_string = decimal_notation(shifts_array[0], 3) + ', ' + decimal_notation(shifts_array[1],3)
                            print('Algorithm: ' + algorithm_names[algorithm_counter] + ', SNR: ' + str(current_SNR) + ', real shifts: ' + real_shifts_string + ', deduced shifts: ' + found_shifts_string)

                            ### Assign Results: ###
                            results_vec[image_counter, shift_counter, SNR_counter, crop_counter, algorithm_counter, blur_counter, 0] = shifts_array[0]
                            results_vec[image_counter, shift_counter, SNR_counter, crop_counter, algorithm_counter, blur_counter, 1] = shifts_array[1]
                            shifts_vec[image_counter, shift_counter, SNR_counter, crop_counter, algorithm_counter, blur_counter, 0] = shiftx
                            shifts_vec[image_counter, shift_counter, SNR_counter, crop_counter, algorithm_counter, blur_counter, 1] = shifty
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
filename_mean = os.path.join(folder_path,  'ShiftOnly_mean_vec_known_shifts.npy')
filename_std = os.path.join(folder_path,  'ShiftOnly_std_vec_known_shifts.npy')
filename_results = os.path.join(folder_path, 'ShiftOnly_results_vec_known_shifts.npy')
filename_original_shifts = os.path.join(folder_path, 'ShiftOnly_shifts_vec_known_shifts.npy')
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







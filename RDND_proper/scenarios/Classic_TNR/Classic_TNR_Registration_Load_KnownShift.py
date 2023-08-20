from RapidBase.Utils.MISCELENEOUS import path_make_path_if_none_exists
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
# max_shifts_vec = np.flip(np.arange(10))
max_shifts_vec = my_linspace(0,5,50)
# max_shifts_vec = [5]
algorithm_names = ['Maor Algorithm', 'Cross Correlation', 'OpenCV Features', 'Phase Only Correlation']
SNR_vec = [np.inf, 100, 10, 5, 4, 3, 2, 1]
shift_layer_torch = Shift_Layer_Torch()
crop_size_x = 512
number_of_algorithms = 4

### Load arrays names: ###
# folder_path = path_get_current_working_directory()
folder_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD/scenarios')
folder_path = os.path.join(folder_path, 'Registration')
path_make_path_if_none_exists(folder_path)
### Specific Shifts: ###
filename_mean = os.path.join(folder_path,  'mean_vec_known_shifts.npy')
filename_std = os.path.join(folder_path,  'std_vec_known_shifts.npy')
filename_results = os.path.join(folder_path, 'results_vec_known_shifts.npy')
filename_original_shifts = os.path.join(folder_path, 'shifts_vec_known_shifts.npy')
#TODO: save shifts and SNR vecs

### Load Arrays if wanted: ###
mean_vec = np.load(filename_mean)
std_vec = np.load(filename_std)
results_vec = np.load(filename_results)
shifts_vec = np.load(filename_original_shifts)

### Clean up data - there are sometimes huge outliers, deal with it later but for analysis purposes get rid of them: ###
# results_vec[image_counter, shift_counter, SNR_counter, algorithm_counter, 0]
#TODO: add percent of outliers
clean_results_vec = results_vec.__copy__()
for shift_counter, current_max_shift in enumerate(max_shifts_vec):
    for SNR_counter, current_SNR in enumerate(SNR_vec):
        for algorithm_counter in np.arange(0, number_of_algorithms):
            current_results_mat = clean_results_vec[:,shift_counter, SNR_counter, algorithm_counter,:]
            current_shifts_mat = shifts_vec[:,shift_counter, SNR_counter, algorithm_counter,:]
            current_median = np.median(np.abs(current_results_mat - current_shifts_mat))  #get median for current algorithm, current SNR, and current shift
            indices_above_threshold = (np.abs(current_results_mat - current_shifts_mat) > current_median * 3)
            current_results_mat[indices_above_threshold] = np.nan
            clean_results_vec[:, shift_counter, SNR_counter, algorithm_counter, :] = current_results_mat
percent_nan = np.sum(np.isnan(clean_results_vec)) / np.size(clean_results_vec)
results_vec = clean_results_vec

### Analyze Results: ###
results_vec = results_vec.clip(-6,6)
mean_vec = np.nanmean(shifts_vec - results_vec, 0)
mean_vec = mean_vec[..., -1]
# std_vec = np.nanstd(shifts_vec - results_vec, 0)
std_vec = np.nanmean(np.abs(shifts_vec - results_vec), 0)
std_vec = std_vec[..., -1]


### Mean + Error Bars: ###
max_value = 10
max_shift_to_check = 1
for SNR_counter, current_SNR in enumerate(SNR_vec):
    for algorithm_counter in np.arange(0, number_of_algorithms + 1):
        if algorithm_counter <= 3:
            current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
            current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
            algorithm_name = algorithm_names[algorithm_counter]

            first_index_larger_then_1 = find(max_shifts_vec > max_shift_to_check)[0][0][0]
            # current_std_vec = current_std_vec[0:first_index_larger_then_1]
            # current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
            current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]

            current_results_vec = results_vec[:, 0:first_index_larger_then_1, SNR_counter, algorithm_counter, 0]
            current_mean_vec = np.nanmean(current_results_vec, 0)
            current_std_vec = np.nanstd(current_results_vec ,0)

            ### Mean Vec: ###
            # if algorithm_counter == 0:
            #     fig, ax = plt.subplots(1, 1)
            # img = ax.plot(current_max_shifts_vec, current_mean_vec.clip(0, max_value))
            # img = ax.errorbar(current_max_shifts_vec, current_mean_vec.clip(0, max_value), current_std_vec)
            fill_between(current_max_shifts_vec, current_mean_vec.clip(0, max_value) - current_std_vec,
                             current_mean_vec.clip(0, max_value) + current_std_vec, alpha=0.25)
        else:
            img = fill_between(current_max_shifts_vec, current_max_shifts_vec + 10**-3, current_max_shifts_vec - 10**-3, alpha=0.25)  #Ideal Straight Line
            # img = plot(current_max_shifts_vec, current_max_shifts_vec, 'g--')  #Ideal Straight Line

        ### Add Plot Stuff: ###
        x_label_list = []
        y_label_list = []
        for i in np.arange(len(current_max_shifts_vec)):
            x_label_list.append(str(current_max_shifts_vec[i]))
        # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
        # ax.set_xticklabels(x_label_list)
        # ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
        # ax.xaxis.set_ticklabels(x_label_list)
        xticks(np.arange(len(current_max_shifts_vec))/10, x_label_list)
        xlabel('Shift')
        ylabel('Mean')
        ylim([0, 1])
        title('Mean+STD(Shift): ' + ', SNR: ' + str(current_SNR))
        legend(algorithm_names + ['Ideal Line'])

    path_make_path_if_none_exists(os.path.join(folder_path, 'Known Shift', 'Mean With Error Bar'))
    savefig(os.path.join(folder_path, 'Known Shift', 'Mean With Error Bar', 'Known Shift ErrorBar All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(current_SNR)) + ', Shift Smaller Than 1')
    close()





### Show All Cuts of Mean(Shift) & STD(Shift) For Certain SNR and Algorithm Up To Shift=1[Pixel]: ###
max_value = 10
max_shift_to_check = 1
for SNR_counter, current_SNR in enumerate(SNR_vec):
    for algorithm_counter in np.arange(0, number_of_algorithms):
        current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
        current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
        algorithm_name = algorithm_names[algorithm_counter]

        first_index_larger_then_1 = find(max_shifts_vec > max_shift_to_check)[0][0][0]
        current_std_vec = current_std_vec[0:first_index_larger_then_1]
        current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
        current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]

        # current_results_vec = results_vec[:, 0:first_index_larger_then_1, SNR_counter, algorithm_counter, 0]
        # current_mean_vec = np.nanmean(current_results_vec, 0)
        # current_std_vec = np.nanstd(current_results_vec ,0)

        ### Mean Vec: ###
        if algorithm_counter == 0:
            fig, ax = plt.subplots(1, 1)
        img = ax.plot(current_max_shifts_vec, current_mean_vec.clip(0, max_value))

        ### Add Plot Stuff: ###
        x_label_list = []
        y_label_list = []
        for i in np.arange(len(current_max_shifts_vec)):
            x_label_list.append(str(current_max_shifts_vec[i]))
        # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
        # ax.set_xticklabels(x_label_list)
        # ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
        # ax.xaxis.set_ticklabels(x_label_list)
        xticks(np.arange(len(current_max_shifts_vec))/10, x_label_list)
        xlabel('Shift')
        ylabel('Mean')
        ylim([0, 1])
        title('Mean(Shift): ' + ', SNR: ' + str(current_SNR))
        legend(algorithm_names)

    path_make_path_if_none_exists(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately Up To 1 Pixel'))
    savefig(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately Up To 1 Pixel', 'Known Shift Specific All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(current_SNR)) + ', Shift Smaller Than 1')
    close()

    for algorithm_counter in np.arange(0, number_of_algorithms):
        current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
        current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
        algorithm_name = algorithm_names[algorithm_counter]

        first_index_larger_then_1 = find(max_shifts_vec > max_shift_to_check)[0][0][0]
        current_std_vec = current_std_vec[0:first_index_larger_then_1]
        current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
        current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]

        ### Mean Vec: ###
        if algorithm_counter == 0:
            fig, ax = plt.subplots(1, 1)
        img = ax.plot(current_std_vec.clip(0, max_value))
        x_label_list = []
        y_label_list = []
        for i in np.arange(len(current_max_shifts_vec)):
            x_label_list.append(str(current_max_shifts_vec[i]))
        # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
        # ax.set_xticklabels(x_label_list)
        # ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
        # ax.xaxis.set_ticklabels(x_label_list)
        xticks(np.arange(len(current_max_shifts_vec)) / 10, x_label_list)
        xlabel('Shift')
        ylabel('STD')
        ylim([0,1])
        title('STD(Shift): ' + algorithm_name + ', SNR: ' + str(current_SNR))
        legend(algorithm_names)
    path_make_path_if_none_exists(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately Up To 1 Pixel'))
    savefig(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately Up To 1 Pixel', 'Known Shift Specific All Algorithms Plot STD(Shift): ' + ', SNR: ' + str(current_SNR)) + ', Shift Smaller Than 1')
    close()



### Show All Cuts of Mean(Shift) & STD(Shift) For Certain SNR and Algorithm Up To Shift=5[Pixel]: ###
max_value = 10
max_shift_to_check = 5
for SNR_counter, current_SNR in enumerate(SNR_vec):
    for algorithm_counter in np.arange(0, number_of_algorithms):
        current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
        current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
        algorithm_name = algorithm_names[algorithm_counter]

        first_index_larger_then_1 = find(max_shifts_vec >= max_shift_to_check-0.1)[0][0][0]
        current_std_vec = current_std_vec[0:first_index_larger_then_1]
        current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
        current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]

        ### Mean Vec: ###
        if algorithm_counter == 0:
            fig, ax = plt.subplots(1, 1)
        img = ax.plot(current_mean_vec.clip(0, max_value))
        x_label_list = []
        y_label_list = []
        for i in np.arange(len(current_max_shifts_vec)):
            x_label_list.append(str(current_max_shifts_vec[i]))
        # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
        # ax.set_xticklabels(x_label_list)
        # ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
        # ax.xaxis.set_ticklabels(x_label_list)
        xticks(np.arange(len(current_max_shifts_vec)) / 10, x_label_list)
        xlabel('Shift')
        ylabel('Mean')
        ylim([0, 5])
        title('Mean(Shift): ' + algorithm_name + ', SNR: ' + str(np.inf))
        legend(algorithm_names)
    path_make_path_if_none_exists(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately'))
    savefig(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately' ,'Known Shift Specific All Algorithms Plot Mean(Shift): ' + ', SNR: ' + str(current_SNR)))
    close()

    for algorithm_counter in np.arange(0, number_of_algorithms):
        current_mean_vec = mean_vec[:, SNR_counter, algorithm_counter]
        current_std_vec = std_vec[:, SNR_counter, algorithm_counter]
        algorithm_name = algorithm_names[algorithm_counter]

        first_index_larger_then_1 = find(max_shifts_vec >= max_shift_to_check-0.1)[0][0][0]
        current_std_vec = current_std_vec[0:first_index_larger_then_1]
        current_mean_vec = current_mean_vec[0:first_index_larger_then_1]
        current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]

        ### Mean Vec: ###
        if algorithm_counter == 0:
            fig, ax = plt.subplots(1, 1)
        img = ax.plot(current_std_vec.clip(0, max_value))
        x_label_list = []
        y_label_list = []
        for i in np.arange(len(current_max_shifts_vec)):
            x_label_list.append(str(current_max_shifts_vec[i]))
        # ax.set_xticks(np.arange(len(current_max_shifts_vec)))
        # ax.set_xticklabels(x_label_list)
        ax.xaxis.set_ticks(np.arange(len(current_max_shifts_vec)))
        ax.xaxis.set_ticklabels(x_label_list)
        xlabel('Shift')
        ylabel('STD')
        ylim([0,5])
        title('STD(Shift): ' + algorithm_name + ', SNR: ' + str(current_SNR))
        legend(algorithm_names)
    path_make_path_if_none_exists(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately'))
    savefig(os.path.join(folder_path, 'Known Shift', 'Mean and STD Seperately Up To 1 Pixel', 'Known Shift Specific All Algorithms Plot STD(Shift): ' + ', SNR: ' + str(current_SNR)))
    close()



##### Show All 1D Plots On Same Graph: #####
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



##### ShowCase Entire Mat: #####
for algorithm_counter in np.arange(number_of_algorithms):
    current_mean_vec = mean_vec[:,:,algorithm_counter]
    current_std_vec = std_vec[:,:,algorithm_counter]
    algorithm_name = algorithm_names[algorithm_counter]

    ### STD Mat: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(current_std_vec.clip(0,max_value))
    x_label_list = []
    y_label_list = []
    for i in np.arange(len(SNR_vec)):
        x_label_list.append(str(SNR_vec[i]))
    for i in np.arange(len(max_shifts_vec)):
        y_label_list.append(str(max_shifts_vec[i]))
    ax.set_xticks(np.arange(len(SNR_vec)))
    ax.set_yticks(np.arange(len(max_shifts_vec)))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Shift')
    fig.colorbar(img)
    title('Shift STD: ' + algorithm_name)
    savefig(os.path.join(folder_path, 'STD Mat ' + algorithm_name))
    close()

    ### Mean Mat: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(current_mean_vec)
    x_label_list = []
    y_label_list = []
    for i in np.arange(len(SNR_vec)):
        x_label_list.append(str(SNR_vec[i]))
    for i in np.arange(len(max_shifts_vec)):
        y_label_list.append(str(max_shifts_vec[i]))
    ax.set_xticks(np.arange(len(SNR_vec)))
    ax.set_yticks(np.arange(len(max_shifts_vec)))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Shift')
    fig.colorbar(img)
    title('Shift Mean: ' + algorithm_name)
    savefig(os.path.join(folder_path, 'Mean Mat ' + algorithm_name))
    close()


##### Show Case Up To 1 Pixel: #####
for algorithm_counter in np.arange(number_of_algorithms):
    max_shift_to_check = 1
    current_mean_vec = mean_vec[:, :, algorithm_counter]
    current_std_vec = std_vec[:, :, algorithm_counter]
    algorithm_name = algorithm_names[algorithm_counter]

    first_index_larger_then_1 = find(max_shifts_vec>max_shift_to_check)[0][0][0]
    current_std_vec = current_std_vec[0:first_index_larger_then_1, :]
    current_mean_vec = current_mean_vec[0:first_index_larger_then_1, :]
    current_max_shifts_vec = max_shifts_vec[0:first_index_larger_then_1]
    ### STD Mat: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(current_std_vec.clip(0,max_value))
    x_label_list = []
    y_label_list = []
    for i in np.arange(len(SNR_vec)):
        x_label_list.append(str(SNR_vec[i]))
    for i in np.arange(len(current_max_shifts_vec)):
        y_label_list.append(str(current_max_shifts_vec[i]))
    ax.set_xticks(np.arange(len(SNR_vec)))
    ax.set_yticks(np.arange(len(current_max_shifts_vec)))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Shift')
    fig.colorbar(img)
    title('Shift STD: ' + algorithm_name)
    savefig(os.path.join(folder_path, 'STD Mat Up To Shift 1' + algorithm_name))
    close()

    ### Mean Mat: ###
    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(current_mean_vec)
    x_label_list = []
    y_label_list = []
    for i in np.arange(len(SNR_vec)):
        x_label_list.append(str(SNR_vec[i]))
    for i in np.arange(len(current_max_shifts_vec)):
        y_label_list.append(str(current_max_shifts_vec[i]))
    ax.set_xticks(np.arange(len(SNR_vec)))
    ax.set_yticks(np.arange(len(current_max_shifts_vec)))
    ax.set_yticklabels(y_label_list)
    ax.set_xticklabels(x_label_list)
    xlabel('PSNR')
    ylabel('Shift')
    fig.colorbar(img)
    title('Shift Mean: ' + algorithm_name)
    savefig(os.path.join(folder_path, 'Mean Mat Up To Shift 1' + algorithm_name))
    close()







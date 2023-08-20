
# from RapidBase.Utils.IO.Path_and_Reading_utils import save_image_numpy, gray2color_torch, to_range

from RapidBase.import_all import *

from torch.utils.data import DataLoader

from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
import RapidBase.TrainingCore.datasets
import RapidBase.TrainingCore.training_utils
from RDND_proper.scenarios.Classic_TNR.Classical_TNR_misc import *

from RDND_proper.scenarios.Classic_TNR.Noise_Estimate_misc import *
from RDND_proper.scenarios.Classic_TNR.Registration_misc import *

### Get Paths To Images: ###
experiment_path = r'/home/mafat/DataSets/KAYA_EXPERIMENT/Experiment 09.03.2021'
gain_folder = os.path.join(experiment_path, 'Integration sphere', 'gain')
offset_folder = os.path.join(experiment_path, 'Integration sphere', 'offset')

configuration_folder = 'fps = 90 exposure = 11 ms'
# configuration_folder = 'fps = 150 exposure = 6.6 ms'
# configuration_folder = 'fps = 210 exposure = 4.7 ms'
# configuration_folder = 'fps = 270 exposure = 3.7 ms'
# configuration_folder = 'fps = 330 exposure = 2.9 ms'
# configuration_folder = 'fps = 390 exposure = 2.5 ms'
# configuration_folder = 'fps = 450 exposure = 2.2 ms'
# configuration_folder = 'fps = 510 exposure = 1.9 ms'
# configuration_folder = 'fps = 570 exposure = 1.7 ms'
# configuration_folder = 'fps = 630 exposure = 1.5 ms'

final_path_gain = os.path.join(gain_folder, configuration_folder)
final_path_offset = os.path.join(offset_folder, configuration_folder)

gain_images_filenames_list = path_get_files_recursively(final_path_gain, '.raw', True)
offset_images_filenames_list = path_get_files_recursively(final_path_offset, '.raw', True)

### Get Images: ###
gain_images_list = []
offset_images_list = []
for frame_counter in np.arange(len(gain_images_filenames_list)):
    gain_image_current = read_image_general(gain_images_filenames_list[frame_counter], io_dict=None)
    offset_image_current = read_image_general(offset_images_filenames_list[frame_counter], io_dict=None)

    gain_images_list.append(gain_image_current)
    offset_images_list.append(offset_image_current)

# ### Save Images In Regular Format: ###
# for frame_counter in np.arange(len(gain_images_filenames_list)):
#     gain_image_filename_current = gain_images_filenames_list[frame_counter]
#     offset_image_filename_current = offset_images_filenames_list[frame_counter]
#     folder_path_gain, gain_image_filename_current = os.path.split(gain_image_filename_current)
#     folder_path_offset, offset_image_filename_current = os.path.split(offset_image_filename_current)
#     gain_image_filename_current = gain_image_filename_current.replace('.raw','.png')
#     offset_image_filename_current = offset_image_filename_current.replace('.raw','.png')
#     gain_image_current = gain_images_list[frame_counter]
#     offset_image_current = offset_images_list[frame_counter]
#     gain_image_current = np.uint8(gain_image_current)
#     offset_image_current = np.uint8(offset_image_current)
#     folder_path1 = os.path.join(folder_path_gain, 'png_images')
#     folder_path2 = os.path.join(folder_path_offset, 'png_images')
#     path_make_path_if_none_exists(folder_path1)
#     path_make_path_if_none_exists(folder_path2)
#     save_image_numpy(folder_path1, gain_image_filename_current, gain_image_current, False, False, False)
#     save_image_numpy(folder_path2, offset_image_filename_current, offset_image_current, False, False, False)

# ### Create Movie To Show This: ###
# H,W,C = gain_images_list[0].shape
# final_gain_movie = os.path.join(final_path_gain, 'gain_movie.avi')
# final_offset_movie = os.path.join(final_path_offset, 'offset_movie.avi')
# fourcc = cv2.VideoWriter_fourcc(*'MP42')
# video_writer_gain = cv2.VideoWriter(final_gain_movie, fourcc, 3, (W, H))
# video_writer_offset = cv2.VideoWriter(final_offset_movie, fourcc, 3, (W, H))
# for frame_counter in np.arange(len(gain_images_list)):
#     gain_image_current = gain_images_list[frame_counter]
#     offset_image_current = offset_images_list[frame_counter]
#
#     gain_image_current = np.concatenate([gain_image_current, gain_image_current, gain_image_current], -1)
#     offset_image_current = np.concatenate([offset_image_current, offset_image_current, offset_image_current], -1)
#
#     gain_image_current = np.uint8(gain_image_current)
#     offset_image_current = np.uint8(offset_image_current)
#
#     video_writer_gain.write(gain_image_current)
#     video_writer_offset.write(offset_image_current)
# video_writer_gain.release()
# video_writer_offset.release()


### Get Average Gain and Offset Images Images: ###
gain_image_average = 0
offset_image_average = 0
frame_index_to_start_from = 1  #it seems there's a problem with the first one
for frame_counter in np.arange(frame_index_to_start_from, len(gain_images_list)):
    gain_image_current = gain_images_list[frame_counter]
    offset_image_current = offset_images_list[frame_counter]
    gain_image_average += gain_image_current
    offset_image_average += offset_image_current
gain_image_average = gain_image_average / frame_counter
offset_image_average = offset_image_average / frame_counter
gain_image_average_no_offset = gain_image_average - offset_image_average

### Correct Images: ###
gain_images_corrected_list = []
offset_images_corrected_list = []
for frame_counter in np.arange(0, len(gain_images_list)):
    gain_image_current = gain_images_list[frame_counter]
    offset_image_current = offset_images_list[frame_counter]

    offset_image_corrected_current = offset_image_current - offset_image_average
    gain_image_corrected_current = (gain_image_current - offset_image_average) / gain_image_average_no_offset * gain_image_average_no_offset.mean()

    gain_images_corrected_list.append(gain_image_corrected_current)
    offset_images_corrected_list.append(offset_image_corrected_current)

# ### Save Corrected Images: ###
# for frame_counter in np.arange(len(gain_images_filenames_list)):
#     gain_image_filename_current = gain_images_filenames_list[frame_counter]
#     offset_image_filename_current = offset_images_filenames_list[frame_counter]
#     folder_path_gain, gain_image_filename_current = os.path.split(gain_image_filename_current)
#     folder_path_offset, offset_image_filename_current = os.path.split(offset_image_filename_current)
#     gain_image_filename_current = gain_image_filename_current.replace('.raw','.png')
#     offset_image_filename_current = offset_image_filename_current.replace('.raw','.png')
#     gain_image_corrected_current = gain_images_corrected_list[frame_counter]
#     offset_image_corrected_current = offset_images_corrected_list[frame_counter]
#     gain_image_corrected_current = np.uint8(gain_image_corrected_current)
#     offset_image_corrected_current = np.uint8(offset_image_corrected_current)
#     folder_path1 = os.path.join(folder_path_gain, 'corrected_images_gain')
#     folder_path2 = os.path.join(folder_path_offset, 'corrected_images_offset')
#     path_make_path_if_none_exists(folder_path1)
#     path_make_path_if_none_exists(folder_path2)
#     save_image_numpy(folder_path1, gain_image_filename_current, gain_image_corrected_current, False, False, False)
#     save_image_numpy(folder_path2, offset_image_filename_current, offset_image_corrected_current, False, False, False)


### Get Histogram Of Gain Matrix: ###
histogram_values, histogram_bins = np.histogram(np.ndarray.flatten(gain_image_average_no_offset), 50)
histogram_bin_centers = (histogram_bins[0:-1] + histogram_bins[1:])/2
plot(histogram_bin_centers, histogram_values)

### Get FFTs of Column and Row Averages: ###
offset_image_current_rows_PSD_average = 0
offset_image_current_cols_PSD_average = 0
for frame_counter in np.arange(0, len(gain_images_corrected_list)):
    gain_image_current = gain_images_corrected_list[frame_counter]
    offset_image_current = offset_images_corrected_list[frame_counter]

    offset_image_current_rows = offset_image_current.mean(1).squeeze() #average over columns
    offset_image_current_cols = offset_image_current.mean(0).squeeze() #average over rows

    offset_image_current_rows_fft = np.fft.fft(offset_image_current_rows)
    offset_image_current_cols_fft = np.fft.fft(offset_image_current_cols)

    offset_image_current_rows_fft = np.fft.fftshift(offset_image_current_rows_fft)
    offset_image_current_cols_fft = np.fft.fftshift(offset_image_current_cols_fft)

    offset_image_current_rows_PSD = np.abs(offset_image_current_rows_fft) ** 2
    offset_image_current_cols_PSD = np.abs(offset_image_current_cols_fft) ** 2

    offset_image_current_rows_PSD_average += offset_image_current_rows_PSD
    offset_image_current_cols_PSD_average += offset_image_current_cols_PSD

    offset_image_current_rows_PSD_log = np.log(offset_image_current_rows_PSD)
    offset_image_current_cols_PSD_log = np.log(offset_image_current_cols_PSD)
offset_image_current_rows_PSD_average = offset_image_current_rows_PSD_average / frame_counter
offset_image_current_cols_PSD_average = offset_image_current_cols_PSD_average / frame_counter

### Plot PSDs: ###
figure()
plot(np.log(offset_image_current_rows_PSD_average))
title('rows log PSD')
figure()
plot(np.log(offset_image_current_cols_PSD_average))
title('cols log PSD')
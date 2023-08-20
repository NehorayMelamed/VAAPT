# from RapidBase.import_all import *
from RapidBase.TrainingCore.datasets import *
from RDND_proper.scenarios.Classic_TNR.Registration.Registration_misc import *

def fftshift_torch(X, first_spatial_dim=2):
    ### FFT-Shift for all dims from dim=first_spatial_dim onwards: ###
    # batch*channel*...*2
    # real, imag = X.chunk(chunks=2, dim=-1)
    # real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)

    for dim in range(first_spatial_dim, len(X.shape)):
        X = roll_n(X, axis=dim, n=int(np.ceil(X.shape[dim] / 2)))

    # real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    # X = torch.cat((real,imag),dim=-1)
    return X

def torch_Cross_Correlation_FFT_small(tensor1, tensor2, tensor1_fft, tensor2_fft, kx, ky):
    ### Get Cross Correlation From FFT: ###
    CC = torch.fft.ifftn(tensor1_fft * tensor2_fft.conj(), dim=[-2, -1]).abs()  #CC=IFFT(FFT(A)*FFT_CONJ(B))
    CC = fftshift_torch(CC) #can get rid of this and use indexing instead

    ### Get Sub-Pixel Shifts Using Parabola Fit: ###
    shifts_vec, z_vec = return_shifts_using_parabola_fit_torch(CC)

    ### Shift Tensor2 For Maximum Valid Percent: ###
    shiftx_torch = torch.Tensor([float(shifts_vec[0])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    shifty_torch = torch.Tensor([float(shifts_vec[1])]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    displacement_matrix = torch.exp(-(1j * 2 * np.pi * ky * shifty_torch + 1j * 2 * np.pi * kx * shiftx_torch))
    tensor2_displaced = torch.fft.ifftn(tensor2_fft * displacement_matrix, dim=[-1, -2]).real  # f(x+t) = IFFT(FFT(x)*exp(i*k_x*t)), k_x=predefined vector, t=scalar

    return shifts_vec, tensor2_displaced  #tensor2_displaced to be "on top of" tensor1

def fit_polynomial(x, y):
    # solve for 2nd degree polynomial deterministically using three points
    a = (y[2] + y[0] - 2*y[1])/2
    b = -(y[0] + 2*a*x[1] - y[1] - a)
    c = y[1] - b*x[1] - a*x[1]**2
    return [c, b, a]

def return_shifts_using_parabola_fit_torch(CC):
    W = CC.shape[-1]
    H = CC.shape[-2]
    x_vec = np.arange(-(W // 2), W // 2 + 1)
    y_vec = np.arange(-(H // 2), H // 2 + 1)
    max_index = np.argmax(CC)
    max_row = max_index // W
    max_col = max_index % W
    max_value = CC[0,0,max_row, max_col]
    center_row = H // 2
    center_col = W // 2
    if max_row == 0 or max_row == H or max_col == 0 or max_col == W:
        z_max_vec = (0,0)
        shifts_total = (0,0)
    else:
        fitting_points_x = CC[:,:, max_row, :]
        fitting_points_y = CC[:,:, :, max_col]

        # fit a parabola over the CC values: #
        [c_x,b_x,a_x] = fit_polynomial(x_vec, fitting_points_x)
        [c_y,b_y,a_y] = fit_polynomial(y_vec, fitting_points_y)

        # find the sub-pixel max value and location using the parabola coefficients: #
        shiftx = -b_x/(2*a_x)
        x_parabola_max = a_x + b_x*shiftx + c_x*shiftx*2

        shifty = -b_y / (2 * a_y)
        y_parabola_max = a_y + b_y * shifty + c_y * shifty * 2

        shifts_total = (shiftx, shifty)
        z_max_vec = (x_parabola_max, y_parabola_max)

    # ### Fast Way to only find shift: ###
    # y1 = fitting_points_x[:,:,:,0]
    # y2 = fitting_points_x[:,:,:,1]
    # y3 = fitting_points_x[:,:,:,2]
    # shiftx = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # y1 = fitting_points_y[:,:,0,:]
    # y2 = fitting_points_y[:,:,1,:]
    # y3 = fitting_points_y[:,:,2,:]
    # shifty = (y1 - y3) / (2 * (y1 + y3 - 2 * y2))
    # shifts_total = (float(shiftx[0][0][0]), float(shifty[0][0][0]))
    return shifts_total, z_max_vec

def return_sub_pixel_accuracy_around_max_using_parabola_fit_numpy(y_vec, x_vec=None):
    #(*). Assumed 1D Input!!!!!!!!!!!!!!!
    ### take care of input: ###
    y_vec = np.ndarray.flatten(y_vec)
    if x_vec is None:
        x_vec = my_linspace(0,len(y_vec),len(y_vec))
    x_vec = np.ndarray.flatten(x_vec)

    ### get max index around which to interpolate: ###
    max_index = np.argmax(y_vec)
    if max_index == 0: #TODO: maybe if max is at beginning of array return 0
        indices_to_fit = np.arange(0,2)
    elif max_index == len(y_vec) - 1: #TODO: maybe if max is at end of array return last term
        indices_to_fit = np.arange(len(x_vec)-1-2, len(x_vec)-1 + 1)
    else:
        indices_to_fit = np.arange(max_index-1, max_index+1 + 1)

    ### Actually Fit: ###
    x_vec_to_fit = x_vec[indices_to_fit]
    y_vec_to_fit = y_vec[indices_to_fit]  #use only 3 points around max to make sub-pixel fit
    P = np.polynomial.polynomial.polyfit(x_vec_to_fit, y_vec_to_fit, 2)
    x_max = -P[1] / (2 * P[2])
    y_max = np.polynomial.polynomial.polyval(x_max,P)

    return x_max, y_max



### Paths: ###
##############
### Movies Paths: ###
original_frames_folder = os.path.join(datasets_main_folder, '/KAYA_EXPERIMENT')
original_frames_folder = os.path.join(original_frames_folder, 'Experiment 09.03.2021/Illumination v= 30.1  a = 0.01 lux=  4/fps = 450 exposure = 2.2 ms')
# original_frames_folder = os.path.join(original_frames_folder, 'Experiment 09.03.2021/Outdoor lux=3/fps = 450 exposure = 2.2 ms')
# original_frames_folder = os.path.join(original_frames_folder, 'Experiment 09.03.2021/Outdoor lux=3/fps = 630 exposure = 1.5 ms')
### Results Paths: ###
results_path = '/home/mafat/PycharmProjects/IMOD/scenarios/TNR_Results/RealMovies_Kaya_FinalAlgo'
# results_path_name_addition = 'KAYA_OFIR_EXPERIMENT_FPS450_EXP2.2'
results_path_name_addition = 'KAYA_OFIR_EXPERIMENT_FPS630_EXP1.5_Indoor'
results_path = os.path.join(results_path, results_path_name_addition)
path_make_path_if_none_exists(results_path)
###
### Gain Offset Matrices Paths: ###
gain_offset_folder = r'/home/mafat/DataSets/KAYA_EXPERIMENT/Experiment 09.03.2021/Integration sphere'
# gain_matrix_path = os.path.join(gain_offset_folder, 'fps__450_exposure__2.2_ms_gain.mat')
# offset_matrix_path = os.path.join(gain_offset_folder, 'fps__450_exposure__2.2_ms_offset.mat')
gain_matrix_path = os.path.join(gain_offset_folder, 'fps__630_exposure__1.5_ms_gain.mat')
offset_matrix_path = os.path.join(gain_offset_folder, 'fps__630_exposure__1.5_ms_offset.mat')
###

### Read Gain/Offset Matrices: ###
gain_average = load_image_mat(gain_matrix_path).squeeze()
offset_average = load_image_mat(offset_matrix_path).squeeze()
gain_avg_minus_offset_avg = gain_average - offset_average

#######################################################################################################################################
### Initialize Parameters: ###
clean_frame_last = 0
histograms_array = 0

### Choose Methos: ###
Algo_Dict = EasyDict()
image_index_to_start_from = 0
number_of_images_to_generate_each_time = 9
number_of_images = 100
flag_initial_shift_method = 'sub_pixel_fft' # 'integer_roll', 'sub_pixel_fft'
flag_CC_finetunning = 'none'  # 'none', 'manual_CC'
flag_use_clean_or_original_frame_for_reference = 'original'  # 'original', 'clean'
registration_algorithm = 'CrossCorrelation'  #DeepFlow, CrossCorrelation, PhaseCorrelation, FeatureHomography, MAOR, NormalizedCrossCorrelation
max_search_area = 11 #for algorithms like NCC
downsample_kernel_size_FrameCombine = 1
downsample_kernel_size_NoiseEstimation = 1
downsample_factor_Registration = 1
crop_size_initial_W = np.inf
crop_size_initial_H = np.inf

### Set Up Avg Pooling Layers: ###
downsample_layer_FrameCombine = nn.AvgPool2d(downsample_kernel_size_FrameCombine)
downsample_layer_NoiseEstimatione = nn.AvgPool2d(downsample_kernel_size_NoiseEstimation)
downsample_layer_Registration = nn.AvgPool2d(downsample_factor_Registration)

#######################################################################################################################################
### IO: ###
IO_dict = EasyDict()
### Noise Parameters: ###
SNR = np.inf
IO_dict.sigma_to_dataset = 1/sqrt(SNR)*255
IO_dict.SNR_to_model = SNR
IO_dict.sigma_to_model = 1/sqrt(IO_dict.SNR_to_model)*255
# IO_dict.sigma_to_model = 80
### Blur Parameters: ###
IO_dict.blur_size = 20
### Number of frames to load: ###
IO_dict.NUM_IN_FRAMES = 5 # temporal size of patch
### Training Flags: ###
IO_dict.non_valid_border_size = 30
### Universal Training Parameters: ###
IO_dict.batch_size = 1
IO_dict.number_of_epochs = 60000
#### Assign Device (for instance, we might want to do things on the gpu in the dataloader): ###
IO_dict.device = torch.device('cuda')
#######################################################################################################################################

####################################
### DataSets: ###
original_frames_dataset = DataSet_SingleVideo_RollingIndex_AddNoise(root_folder=original_frames_folder,
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video_to_scan=100,  #TODO: add possibility to randomize start index from a possibly long video
                                                         number_of_images_per_video_to_load=1,
                                                         max_number_of_videos=np.inf,
                                                         crop_size=np.inf,
                                                         flag_to_RAM=True,
                                                         flag_explicitely_make_tensor=True,
                                                         flag_normalize_by_255=False,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False,
                                                         flag_to_RGB=False, #Stack BW frames to get "psudo-RGB", TODO: get rid of this, this is only for reference checkpoint
                                                         Train_dict=IO_dict,
                                                         allowed_extentions=['png', 'tif', '.raw'])


### Take care of initializations and such for first frame: ###
normalization_factor_after_NUC = 20
original_frames_output_dict = original_frames_dataset[image_index_to_start_from]
original_frame_current = original_frames_output_dict.original_frames.data[:,0:1,:,:]
original_frame_current = (original_frame_current - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)
original_frame_current = original_frame_current / normalization_factor_after_NUC
original_frame_current_cropped = crop_torch_batch(original_frame_current, (crop_size_initial_W, crop_size_initial_H), crop_style='center')
original_frame_current_cropped = downsample_layer_Registration(original_frame_current_cropped)

### Prepare Stuff For Sub-Pixel Shifts: ###
#TODO: if you already know image size you don't have to wait for an incoming image, of course
B, C, H, W = original_frame_current_cropped.shape
# Get tilt phases k-space:
x = np.arange(-np.fix(W / 2), np.ceil(W / 2), 1)
y = np.arange(-np.fix(H / 2), np.ceil(H / 2), 1)
delta_f1 = 1 / W
delta_f2 = 1 / H
f_x = x * delta_f1
f_y = y * delta_f2
# Use fftshift on the 1D vectors for effeciency sake to not do fftshift on the final 2D array:
f_x = np.fft.fftshift(f_x)
f_y = np.fft.fftshift(f_y)
# Build k-space meshgrid:
[kx, ky] = np.meshgrid(f_x, f_y)
# Frequency vec to tensor:
kx = torch.Tensor(kx)
ky = torch.Tensor(ky)
kx = kx.unsqueeze(0).unsqueeze(0)
ky = ky.unsqueeze(0).unsqueeze(0)

### Initialize More Things: ###
B,C,H,W = original_frame_current.shape
crop_size_initial_W = min(crop_size_initial_W, W)
crop_size_initial_H = min(crop_size_initial_H, H)
shifts_array = torch.zeros((1, 1, crop_size_initial_W, crop_size_initial_W, 2))

### Lists: ###
clean_frames_list = []
original_frames_list = []
original_frames_fft_list = []
aligned_frames_list = []
simple_average_frames_list = []
real_shifts_list = []
inferred_shiftx_list = []
inferred_shifty_list = []

### Initialize Running Lists: ###
original_frames_list.append(original_frame_current_cropped)
original_frames_fft_list.append(torch.fft.fftn(original_frame_current_cropped, dim=[-2,-1]))
aligned_frames_list.append(original_frame_current_cropped)
clean_frames_list.append(original_frame_current_cropped)
clean_frame_current = original_frame_current_cropped
clean_frame_simple_averaging = original_frame_current_cropped
shiftx_sum = 0
shifty_sum = 0

### Get Video Recorder: ###
#(1). File Name:
experiment_name = results_path_name_addition + \
                  '_' + registration_algorithm + \
                  '_DSCombine' + str(downsample_kernel_size_FrameCombine) + \
                  '_DSReg' + str(downsample_factor_Registration)
#(2). Paths:
save_path = os.path.join(results_path, experiment_name)
save_path1 = os.path.join(save_path, 'original_frames')
save_path2 = os.path.join(save_path, 'clean_frame_estimate')
save_path3 = os.path.join(save_path, 'noisy_frames')
path_make_path_if_none_exists(save_path)
path_make_path_if_none_exists(save_path1)
path_make_path_if_none_exists(save_path2)
path_make_path_if_none_exists(save_path3)
path_make_path_if_none_exists(os.path.join(save_path, 'Results'))
video_name = os.path.join(save_path, 'Results', 'Results.avi')
#(3). Initialize Video Object:
video_width = crop_size_initial_W * 2
video_height = crop_size_initial_W * 1
video_object = cv2.VideoWriter(video_name, 0, 10, (crop_size_initial_W * 2, crop_size_initial_H))

for frame_index in np.arange(1, number_of_images-number_of_images_to_generate_each_time//2-1):
    # print(frame_index)

    ### Get Next Frame: ###
    original_frames_output_dict = original_frames_dataset[frame_index]
    current_frames_original = original_frames_output_dict.original_frames[:,0:1,:,:]

    tic()
    ### Make Gain/Offset Correction: ### #TODO: insert into dataset object
    current_frames_original = (current_frames_original - offset_average) * gain_avg_minus_offset_avg.mean() / (gain_avg_minus_offset_avg)
    current_frames_original = current_frames_original / normalization_factor_after_NUC
    current_frames_original = current_frames_original.clamp(0)

    ### Choose ROI: ###
    original_frame_current_cropped = crop_torch_batch(current_frames_original, (crop_size_initial_W, crop_size_initial_H), crop_style='center')
    
    ### Add current crop to batch: ###
    original_frames_list.append(original_frame_current_cropped)
    original_frames_fft_list.append(torch.fft.fftn(original_frame_current_cropped, dim=[-2,-1]))

    ### Get Reference Tensor: ###
    tensor1 = original_frames_list[0]
    tensor1_fft = original_frames_fft_list[0]

    ### Get Tensor To Shift Compared To Reference Tensor: ###
    tensor2 = original_frames_list[-1]
    tensor2_fft = original_frames_fft_list[-1]

    ### Use FFT To Get Shifts From CC -> get tensor2 aligned to tensor1: ###
    shifts_vec, tensor2_displaced = torch_Cross_Correlation_FFT_small(tensor1, tensor2,
                                                                      tensor1_fft, tensor2_fft,
                                                                      kx, ky,
                                                                      flag_initial_shift_method, flag_CC_finetunning)

    ### Append Aligned Frame To List: ###
    aligned_frames_list.append(tensor2_displaced)

    ### Average Frames After Alignment: ###
    if len(original_frames_list) <= number_of_images_to_generate_each_time:
        ### Add Current Aligned Frame To Clean Frame Moving-Average/Kalman-Filter Style: ###
        T = frame_index + 1
        clean_frame_current = (1-1/T) * clean_frame_current + 1/T * tensor2_displaced
        clean_frames_list.append(clean_frame_current)
    else:
        ### Substract Oldest Frame From Clean Frame & Add Newest One: ###
        clean_frame_current -= 1/T * aligned_frames_list[0]
        clean_frame_current += 1/T * tensor2_displaced
        clean_frames_list.append(clean_frame_current)

        ### Pop Oldest Elements In Lists: ###
        aligned_frames_list.pop(0)
        original_frames_list.pop(0)
        original_frames_fft_list.pop(0)
        clean_frames_list.pop(0)

    ### Get Inferred Shifts To Keep Track Of Stuff: ###
    inferred_shiftx_list.append(shifts_vec[0])
    inferred_shifty_list.append(shifts_vec[1])
    shiftx_sum += shifts_vec[0]
    shifty_sum += shifts_vec[1]

    toc(str(frame_index))

    ### Save Wanted Images: ###
    save_image_numpy(save_path1, string_rjust(frame_index,4) + '_center_frame_original.png', original_frame_current_cropped[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True, flag_convert_to_uint8=True)
    save_image_numpy(save_path2, string_rjust(frame_index,4) + '_clean_frame_estimate.png', clean_frame_current[0].cpu().numpy().transpose(1, 2, 0), False, flag_scale=True, flag_convert_to_uint8=True)

    ### Record Noisy-Clean Video: ###
    stretch_factor = 255
    image_to_write = torch.cat((original_frame_current_cropped, clean_frame_current), -1)
    image_to_write = torch.cat((image_to_write,image_to_write,image_to_write),1)
    image_to_write = image_to_write[0].cpu().numpy().transpose(1,2,0)
    image_to_write = (image_to_write * stretch_factor).clip(0,255).astype(np.uint8)
    video_object.write(image_to_write)


### Stop Video Writer Object: ###
video_object.release()


# ### Get Metrics For Video: ###
# output_dict_CleanOriginal_average, output_dict_CleanOriginal_history = \
#     get_metrics_video_lists(original_frames_list, clean_frames_list, number_of_images=np.inf)
# output_dict_SimpleAverageOriginal_average, output_dict_SimpleAverageOriginal_history = \
#     get_metrics_video_lists(original_frames_list, simple_average_frames_list, number_of_images=np.inf)
# path_make_path_if_none_exists(os.path.join(save_path,'Results'))
# for key in output_dict_CleanOriginal_history.keys():
#     try:
#         y2 = np.array(output_dict_CleanOriginal_history.inner_dict[key])
#         y4 = np.array(output_dict_SimpleAverageOriginal_history.inner_dict[key])
#         plot_multiple([y2, y4],
#                       legend_labels=['cleaned-noisy: ' + decimal_notation(y2.mean(), 2),
#                                      'cleaned-SimpleAverage: ' + decimal_notation(y4.mean(), 2)],
#                       super_title=key + ' over time', x_label='frame-counter', y_label=key)
#         plt.savefig(os.path.join(save_path, 'Results', key + ' over time.png'))
#         plt.close()
#     except:
#         1
# plt.close()

# ### Get Metrics For Shifts: ###
# real_shifts_x = [x[0] for x in real_shifts_list]
# real_shifts_y = [x[1] for x in real_shifts_list]
# inferred_shifts_x = [x[0] for x in inferred_shifts_list]
# inferred_shifts_y = [x[1] for x in inferred_shifts_list]
# inferred_shifts_x = np.array(inferred_shifts_x)
# inferred_shifts_y = np.array(inferred_shifts_y)
# real_shifts_x = np.array(real_shifts_x).squeeze()
# real_shifts_y = np.array(real_shifts_y).squeeze()
# shift_x_std = (real_shifts_x - inferred_shifts_x).std()
# shift_y_std = (real_shifts_y - inferred_shifts_y).std()
#
# plot(real_shifts_x)
# plot(inferred_shifts_x)
# legend(['real_shift_x','inferred_shift_x'])
# title('Shift X STD: ' + str(shift_x_std))
# plt.savefig(os.path.join(save_path, 'Results', 'Shifts_Graph_x.png'))
# plt.close()
#
# plot(real_shifts_y)
# plot(inferred_shifts_y)
# legend(['real_shift_y','inferred_shift_y'])
# title('Shift Y STD: ' + str(shift_y_std))
# plt.savefig(os.path.join(save_path, 'Results', 'Shifts_Graph_y.png'))
# plt.close()





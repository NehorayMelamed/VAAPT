from RapidBase.import_all import *


movie_full_filename = '/home/mafat/Videos/4K Video Downloader/Walking and flying in Toride, Ibaraki・4K HDR.mkv'
movie_full_filename = '/home/mafat/DataSets/Example Videos/Beirut.mp4'
movie_save_filename = 'Berlin_ForPresentation2.avi'

### Where To Save Results: ###
super_folder_to_save_everything_at = '/home/mafat/DataSets/Synthetic_Movies'
if not os.path.exists(super_folder_to_save_everything_at):
    os.makedirs(super_folder_to_save_everything_at)
movie_save_full_filename = os.path.join(super_folder_to_save_everything_at, movie_save_filename)

### Auxiliary Layers: ###
blur_number_of_channels = 1
blur_kernel_size = 7
blur_sigma = 1
gaussian_blur_layer = Gaussian_Blur_Layer(blur_number_of_channels, blur_kernel_size, blur_sigma)
average_pooling_layer = nn.AvgPool2d(16,1)
downsample_layer = nn.AvgPool2d(2,2)
crop_size_vec = [1000,1000]

### Camera Parameters: ### (*). INSTEAD OF ABOVE NOISE PARAMETERS!!!!
mean_gray_level_per_pixel = [1.5,1.5]
electrons_per_gray_level = 14.53
photons_per_electron = 1 #TODO: i don't think this is the QE but whether we have a photo-multiplier or not....check this out
gray_levels_per_electron = 1/electrons_per_gray_level
electrons_per_photon = 1/photons_per_electron

### Kaya Camera Noise: ###
Camera_Noise_Folder = os.path.join(datasets_main_folder,'/KAYA_CAMERA_NOISE/noise')
noise_image_filenames = path_get_files_recursively(Camera_Noise_Folder, '', True)

### Video Reader: ###
video_stream = cv2.VideoCapture(movie_full_filename)
flag_frame_available, frame = video_stream.read()

### Video Writer: ###
H,W,C = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'MP42')  # Be sure to use lower case
video_writer = cv2.VideoWriter(movie_save_full_filename, fourcc, 25.0, (crop_size_vec[0]*2, crop_size_vec[0]))

number_of_images = 200
start_frame = 1
for frame_index in np.arange(start_frame):
    print(frame_index)
    flag_frame_available, original_frame = video_stream.read()

for frame_index in np.arange(number_of_images):
    print(frame_index)

    ### Read Movie Frame: ###
    flag_frame_available, original_frame = video_stream.read()
    original_frame = original_frame / 255
    original_frame = BW2RGB(RGB2BW(original_frame))
    noisy_frame = original_frame
    noisy_frame = RGB2BW(noisy_frame)

    ### Read Camera Readout-Noise: ###
    camera_noise_frame = read_image_general(noise_image_filenames[frame_index])

    ### Get Non-Uniformity Map: ###
    speckle_pattern_total_intensity, speckle_pattern1, speckle_pattern2 = create_speckles_of_certain_size_in_pixels(400, crop_size_vec[0], 0, 1, 1)
    NU_map = speckle_pattern_total_intensity
    # NU_map = NU_map.clip(0,1)
    NU_map = to_range(NU_map, 0.7, 1)

    ### Numpy to Torch: ###
    original_frame = numpy_to_torch(original_frame, True)
    noisy_frame = numpy_to_torch(noisy_frame, True)
    NU_map = numpy_to_torch(NU_map, True)
    camera_noise_frame = numpy_to_torch(camera_noise_frame, True)

    ### Blur: ###
    noisy_frame = average_pooling_layer(noisy_frame)

    ### Crop Frames To Same Size: ###
    original_frame = crop_torch_batch(original_frame, crop_size_vec, 'center')
    noisy_frame = crop_torch_batch(noisy_frame, crop_size_vec, 'center')
    camera_noise_frame = crop_torch_batch(camera_noise_frame, crop_size_vec, 'center')
    NU_map = crop_torch_batch(NU_map, crop_size_vec, 'center')

    ### Add Noise: ###
    #(1). Shot Noise:
    input_image_mean = noisy_frame.mean() + 1e-5
    current_mean_gray_level_per_pixel = 2
    noisy_frame = noisy_frame * (current_mean_gray_level_per_pixel) / input_image_mean
    noisy_frame_photons = noisy_frame * electrons_per_gray_level * photons_per_electron
    shot_noise = torch.sqrt(noisy_frame_photons) * torch.randn(*noisy_frame_photons.shape)
    shot_noise = shot_noise * electrons_per_photon * gray_levels_per_electron
    noisy_frame = noisy_frame + shot_noise
    #(2). Kaya Camera Noise:
    noisy_frame = noisy_frame + camera_noise_frame
    #(3). General Synthetic Noise:

    #(4). Normalize:
    noisy_frame = noisy_frame / 10

    ### Add NU: ###
    noisy_frame = noisy_frame * NU_map

    ### Concatenate Original and Noisy: ###
    concatenated_frame = torch.cat([BW2RGB(noisy_frame), original_frame], -1)
    concatenated_frame_numpy = torch_to_numpy(concatenated_frame).squeeze(0)
    concatenated_frame_numpy = (concatenated_frame_numpy*255).clip(0,255).astype(np.uint8)

    ### Make Final Video: ###
    video_writer.write(concatenated_frame_numpy)

video_writer.release()
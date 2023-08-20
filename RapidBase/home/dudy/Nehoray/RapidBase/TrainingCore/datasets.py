"""
This file contains the datasets hierarchy as shown in the bellow tree:

<MultipleImages>
    <Images_From_Folder>
        <MultipleImagesFromSingleImage>
            <Shifts>
                <AGN>
                    <SuperResolution_DownSampleOnTheFly>
                        <Bicubic>
                        </Bicubic>
                    </SuperResolution_DownSampleOnTheFly>
                    <SR_SameSize_DownSampleOnTheFly>
                        <Bicubic>
                        </Bicubic>
                    </SR_SameSize_DownSampleOnTheFly>
                    <SuperResolution_LoadLRHR_1>
                    </SuperResolution_LoadLRHR_1>
                </AGN>
            </Shifts>
        </MultipleImagesFromSingleImage>
    </Images_From_Folder>
    <Videos_In_Folders>
    </Videos_In_Folders>
</MultipleImages>
"""
import numpy as np

from RapidBase.TrainingCore.Basic_DataSets import *
from torch.utils.data import Dataset

# from RapidBase.Utils.Add_Noise import add_noise_to_images_full

class TestArgs():
    def __init__(self, tile=[12,192,192], tile_overlap=[2, 20, 20], scale=1, nonblind_denoising =True, window_size=[6,8,8]):
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.scale = scale
        self.nonblind_denoising = nonblind_denoising
        self.window_size = window_size

def split_dataset(train_dataset, test_dataset, IO_dict):
    if IO_dict.flag_only_split_indices:  # THIS ASSUMES THE TRAIN AND TEST DATASETS ARE THE SAME DATASET AND YOU JUST WANNA SPLIT THEM
        test_dataset_length_new = int(np.round(len(train_dataset) * IO_dict.train_dataset_split_to_test_factor))
        train_dataset_length_new = len(train_dataset) - test_dataset_length_new
        # TODO: understand exactly what's going on here and maybe don't do a random split but a deterministic one?
        # TODO: to do that use torch.utils.data.Subset...... Subset(train_dataset, train_indices), Subset(train_dataset, test_indices)
        train_dataset, train_dataset_split = torch.utils.data.random_split(train_dataset, [train_dataset_length_new,
                                                                                           test_dataset_length_new])
        test_dataset, test_dataset_split = torch.utils.data.random_split(test_dataset, [test_dataset_length_new,
                                                                                        train_dataset_length_new])
    else:
        test_dataset_length_new = int(np.round(len(train_dataset) * IO_dict.train_dataset_split_to_test_factor))
        train_dataset_length_new = len(train_dataset) - test_dataset_length_new
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,
                                                                    [train_dataset_length_new, test_dataset_length_new])

    train_dataset.indices = np.arange(0, train_dataset_length_new)
    test_dataset.indices = np.arange(train_dataset_length_new, train_dataset_length_new + test_dataset_length_new)
    IO_dict.num_mini_batches_trn = train_dataset_length_new // IO_dict.batch_size
    IO_dict.num_mini_batches_val = test_dataset_length_new // IO_dict.batch_size
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def get_default_noise_dict():
    # Initialize Dict:
    IO_dict = EasyDict()

    # Misc:
    IO_dict.max_number_of_noise_images = np.inf
    IO_dict.flag_recursive = True

    # Paths:
    # (1). External Readout Noise Path:
    IO_dict.readout_noise_external_image_path = ''
    IO_dict.readout_noise_search_pattern = '*'
    # (2). External Non-Uniformity Pattern Path:
    IO_dict.NU_external_image_offset_path = ''
    IO_dict.NU_external_image_offset_search_pattern = '*'
    IO_dict.NU_external_image_gain_path = ''
    IO_dict.NU_external_image_gain_search_pattern = '*'
    IO_dict.allowed_extentions = IMG_EXTENSIONS

    # External readout noise parameters
    IO_dict.flag_noise_images_to_RAM = False
    # IO_dict.noise_images_path = None
    IO_dict.max_number_of_noise_images = np.inf


    # Flags: Each flag requires definition to a set of correlated parameters
    IO_dict.flag_add_per_pixel_readout_noise = False
    IO_dict.flag_add_shot_noise = False
    IO_dict.flag_add_dark_current_noise = False
    IO_dict.flag_add_per_pixel_NU = False
    IO_dict.flag_add_row_NU = False
    IO_dict.flag_add_col_NU = False
    IO_dict.flag_add_blob_NU = False
    IO_dict.flag_add_external_image_NU = False
    IO_dict.flag_add_row_readout_noise = False
    IO_dict.flag_add_col_readout_noise = False
    IO_dict.flag_add_external_readout_noise = False
    IO_dict.flag_add_dead_pixels_noise = False
    IO_dict.flag_add_white_pixels_noise = False
    IO_dict.flag_quantize_images = False

    # Parameters:
    # Shot noise parameters
    #TODO: make all the following numbers a range!!!: PPP, dark_current_background, blob_NU_blob_size, per_pixel_readout_noise_sigma,
    # row_readout_noise_sigma, col_readout_noise_sigma, dead_pixels_fraction, white_pixels_fraction
    IO_dict.QE = 1  # Quantum Efficiency
    IO_dict.Ge = 14.53  # Electrons per Gray level
    # Dark current noise parameters
    IO_dict.dark_current_background = 10
    # Pixel non uniformity parameters
    IO_dict. _sigmas_polynomial_params = [1, 0.05]#, 0.00001]  # per Pixel Non Uniformity. polynomial coefficients
    # Row non uniformity params
    IO_dict.RRNU_sigmas_polynomial_params = [1, 0.05]#, 0.00001]  # Row Readout Non Uniformity. polynomial coefficients
    IO_dict.flag_row_NU_random = True
    IO_dict.row_NU_PSD = None
    IO_dict.flag_same_row_NU_on_all_channels = True
    IO_dict.flag_same_row_NU_on_all_batch = False
    # Column non uniformity parameters
    IO_dict.CRNU_sigmas_polynomial_params = [1, 0.05]#, 0.00001]  # Col Readout Non Uniformity. polynomial coefficients
    IO_dict.flag_col_NU_random = True
    IO_dict.col_NU_PSD = None
    IO_dict.flag_same_col_NU_on_all_channels = True
    IO_dict.flag_same_col_NU_on_all_batch = False
    # Blob non uniformity parameters
    IO_dict.BRNU_sigmas_polynomial_params = [0, 0.2, 0]  # Blob Readout Non Uniformity. polynomial coefficients
    IO_dict.blob_NU_blob_size = 10
    IO_dict.flag_same_blob_NU_on_all_channels = True
    IO_dict.flag_same_blob_NU_on_all_batch = True
    # External image non uniformity parameters
    IO_dict.NU_external_image_offset = None
    IO_dict.NU_external_image_gain = None
    # Per pixel readout noise (AWGN) parameters
    IO_dict.per_pixel_readout_noise_sigma = 10
    # Row readout noise parameters
    IO_dict.flag_row_readout_noise_random = True
    IO_dict.row_readout_noise_PSD = None
    IO_dict.row_readout_noise_sigma = 10
    IO_dict.flag_same_row_readout_noise_on_all_channels = False
    # Column readout noise parameters
    IO_dict.flag_col_readout_noise_random = True
    IO_dict.col_readout_noise_PSD = None
    IO_dict.col_readout_noise_sigma = 10
    IO_dict.flag_same_col_readout_noise_on_all_channels = False
    # Dead pixels parameters
    IO_dict.dead_pixels_fraction = 0.001
    IO_dict.flag_same_dead_pixels_on_all_channels = True
    IO_dict.flag_same_dead_pixels_on_all_batch = True
    IO_dict.dead_pixels_value = 0 * 255
    # White pixels parameters
    IO_dict.white_pixels_fraction = 0.001
    IO_dict.flag_same_white_pixels_on_all_channels = True
    IO_dict.flag_same_white_pixels_on_all_batch = True
    IO_dict.white_pixels_value = 1 * 255

    return IO_dict

def get_default_transforms_IO_dict():
    ### Initialize: ###
    IO_dict = EasyDict()

    ### Transforms: ###
    IO_dict.flag_base_transform = False
    IO_dict.flag_batch_transform = False
    IO_dict.flag_turbulence_transform = False
    IO_dict.base_transform = None
    IO_dict.batch_transform = None

    ### Shift / Directional-Blur Parameters: ###
    IO_dict.warp_method = 'bilinear'  # 'nearest', 'bilinear', 'bicubic', 'fft'
    IO_dict.shift_size = 1  # TODO: switch to ranges instead of single numbers
    IO_dict.rotation_angle_size = 1
    IO_dict.scale_delta = 0
    IO_dict.blur_fraction = 0
    IO_dict.shift_mode = 'seperate'  # 'seperate'=each axis randomizes shift seperately,  'constant_size'=constant shift size, direction is randomized
    IO_dict.number_of_blur_steps_per_pixel = 5  # TODO: when using general affine transform find the largest shift and use that

    ### Super Resolution: ###
    # (1). Gaussian Blur Parameters (Low-Pass):
    IO_dict.gaussian_blur_number_of_channels = 3
    IO_dict.gaussian_blur_kernel_size = 3
    IO_dict.gaussian_blur_sigma = 1
    # (2). Upsample/Downsample:
    IO_dict.flag_upsample_noisy_input_to_same_size_as_original = False
    IO_dict.upsample_method = 'bilinear'  # 'nearest', 'bilinear', 'bicubic', 'fft'
    IO_dict.downsample_method = 'binning'  # 'binning', 'nearest', 'bilinear', 'bicubic', 'fft'
    IO_dict.downsampling_factor = 1

    return IO_dict

def get_default_IO_dict():
    IO_dict = EasyDict()

    ### Use Train/Test Split Instead Of Distinct Test Set: ###
    IO_dict.flag_use_train_dataset_split = True
    IO_dict.flag_only_split_indices = True  #
    IO_dict.train_dataset_split_to_test_factor = 0.05
    IO_dict.no_GT = False
    IO_dict.is_moving_images = False
    IO_dict.num_moving_images = 0
    IO_dict.bayer = False
    IO_dict.vrt_debayer = False
    IO_dict.blur_range = False

    ### Transforms / Warping: ###
    IO_dict.transforms_dict = get_default_transforms_IO_dict()  # add a key within the dictionary which contains the noise related parameters

    ### Loading Images: ###
    #(1). Folder names:
    #(1.1). Root Folder:
    IO_dict.root_folder = ''
    IO_dict.corrupted_folder = ''
    #(2). how to search:
    IO_dict.string_pattern_to_search = '*'
    IO_dict.Corrupted_string_pattern_to_search = '*'
    IO_dict.GT_string_pattern_to_search = '*'
    IO_dict.allowed_extentions = IMG_EXTENSIONS
    IO_dict.flag_recursive = True
    #(3). how to load:
    IO_dict.flag_to_RAM = False
    IO_dict.image_loader = ImageLoaderCV
    IO_dict.max_number_of_images = np.inf  # max number of images to search for
    IO_dict.max_number_of_noise_images = np.inf # max number of noise images to search for
    IO_dict.max_number_of_videos = np.inf # max number of videos to search for
    IO_dict.number_of_images_per_video_to_scan = np.inf # max number of images per video to search for / scan
    IO_dict.number_of_image_frames_to_generate = 3
    IO_dict.number_of_images_per_video_to_load = 25
    IO_dict.movie_frame_to_start_from = 100

    ### Post Loading Stuff: ###
    #TODO: understand which of these are still relevant
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.flag_to_RGB_before_noise = False
    IO_dict.flag_to_BW_after_noise = False
    IO_dict.flag_to_RGB_after_noise = False
    IO_dict.flag_noise_to_BW = False
    IO_dict.flag_noise_to_RGB = False
    IO_dict.flag_how_to_concat = 'T'  # 'C'<->[B,C*T,H,W],   'T'<->[B,T,C,H,W]
    IO_dict.flag_normalize_by_255 = False
    IO_dict.flag_make_tensor_before_batch_transform = True  # flag_make_tensor_before_batch_transform

    ### Noise Addition: ###
    IO_dict.noise_dict = get_default_noise_dict()  # add a key within the dictionary which contains the noise related parameters

    ### Miscellenous: ###
    IO_dict.flag_normalize_images_to_PPP = False
    IO_dict.PPP = 100  # Photons per Pixel.
    IO_dict.flag_residual_learning = False
    IO_dict.flag_clip_noisy_tensor_after_noising = False

    ### Training Flags: ###
    IO_dict.non_valid_border_size = 10
    #(1). Universal Training Parameters:
    IO_dict.batch_size = 1
    IO_dict.number_of_epochs = 60000
    #(2). Assign Device (for instance, we might want to do things on the gpu in the dataloader):
    devices = [0,1,2,3]
    IO_dict.model_devices = [torch.device(device) for device in devices]
    IO_dict.device = torch.device(0)
    IO_dict.flag_all_inputs_to_gpu = True

    ### Cropping: ###
    IO_dict.flag_crop_mode = 'random'  # 'center', 'random'
    IO_dict.initial_crop_size = [128 * 2, 128 * 2]
    IO_dict.final_crop_size = [128 * 2, 128 * 2]
    IO_dict.final_crop_size = IO_dict.final_crop_size

    ### Test args: ###
    IO_dict.test_args = TestArgs()
    return IO_dict


class Dataset_MultipleImagesFromSingleImage_AGN(Dataset_Images_From_Folder):
    """
        This class generates a video from a single image.
    """
    def __init__(self, root_path, IO_dict):
        super().__init__(root_path, IO_dict)

    def __getitem__(self, index):
        ### Read current frame (perform base transform on the individual image if wanted), multiply it and stack it: ###
        outputs_dict = self.get_multiple_images_from_single_image(self.image_filenames_list, self.images_list, index)

        ### Get random transformation factors between following frames: ###
        outputs_dict = self.get_random_transformation_vectors(outputs_dict)

        ### Perform Functionality (shifts, blur, etc'): ###
        outputs_dict = self.specific_dataset_functionality_before_cropping(outputs_dict)

        ### Perform Final Crop On Original Images: ###
        outputs_dict = self.perform_final_crop_on_original_images(outputs_dict)

        ### Normalize image graylevels according to camera ###
        outputs_dict = self.normalize_tensor_graylevels_to_photons(outputs_dict)

        ### Perform Functionality After Crop & Before Adding Noise (downsampling mostly): ###
        outputs_dict = self.specific_dataset_functionality_before_adding_noise(outputs_dict)

        ### Assign Original Images To Output Dict: ###
        outputs_dict = self.assign_original_images_to_outputs_dict(outputs_dict)

        ### Add Noise & Get Noise Map: ###
        outputs_dict = self.add_noise_to_tensors_full(outputs_dict)

        ### Perform Functionality After Adding Noise (For instance assign the correct form of stuff etc'): ###
        outputs_dict = self.specific_dataset_functionality_after_adding_noise(outputs_dict)

        ### Assign Final Outputs: ###
        outputs_dict = self.assign_final_outputs_after_noise(outputs_dict)

        return outputs_dict


class Dataset_MultipleImagesFromSingleImage_LoadCorruptedGT(Dataset_MultipleImagesFromSingleImage_AGN):
    def __init__(self, root_path, IO_dict):
        super().__init__(root_path, IO_dict)
        ### Assign Attributes: ###
        self.root_path = root_path
        IO_dict.root_path = root_path

        ### Assign class variables: ###
        if IO_dict.flag_crop_mode == 'random':
            IO_dict.flag_crop_mode = 'uniform'
        self.IO_dict = IO_dict
        assign_attributes_from_dict(self, IO_dict)

        ### Get filenames and images: ###
        # (1). HR:
        IO_dict.string_pattern_to_search = IO_dict.GT_string_pattern_to_search
        self.HR_image_filenames_list, self.HR_images_list = self.get_filenames_and_images_from_folder(IO_dict)
        # (2). LR:
        IO_dict.string_pattern_to_search = IO_dict.Corrupted_string_pattern_to_search
        self.LR_image_filenames_list, self.LR_images_list = self.get_filenames_and_images_from_folder(IO_dict)

        self.default_noise_dict = get_default_noise_dict()

    def __len__(self):
        return len(self.HR_image_filenames_list)

    def __getitem__(self, index):
        ### Read current frame (perform base transform on the individual image if wanted), multiply it and stack it: ###
        HR_outputs_dict = self.get_multiple_images_from_single_image(self.HR_image_filenames_list, self.HR_images_list,
                                                                     index)
        LR_outputs_dict = self.get_multiple_images_from_single_image(self.LR_image_filenames_list, self.LR_images_list,
                                                                     index)

        ### Perform Specific Functionality For Dataset Object: ###
        #(*). use a temporary dict to hold the original values:
        original_IO_dict = EasyDict()
        original_IO_dict.update(self.IO_dict)
        #####
        #(1). use the default noise IO dict (which means = add NO noise) to the GT:
        self.IO_dict.noise_dict = get_default_noise_dict()  # add a key within the dictionary which contains the noise related parameters
        self.IO_dict.transforms_dict.downsampling_factor = 1  # don't downsample here
        self.IO_dict.blur_fraction = 0 # don't blur here
        HR_outputs_dict = self.get_random_transformation_vectors(HR_outputs_dict)
        HR_outputs_dict = self.specific_dataset_functionality_before_cropping(HR_outputs_dict)
        HR_outputs_dict = self.perform_final_crop_on_original_images(HR_outputs_dict)
        HR_outputs_dict = self.specific_dataset_functionality_before_adding_noise(HR_outputs_dict)
        HR_outputs_dict = self.assign_original_images_to_outputs_dict(HR_outputs_dict)
        HR_outputs_dict = self.add_noise_to_tensors_full(HR_outputs_dict)
        HR_outputs_dict = self.specific_dataset_functionality_after_adding_noise(HR_outputs_dict)
        HR_outputs_dict = self.assign_final_outputs_after_noise(HR_outputs_dict)
        #####

        #####
        #(1). add the original noise_dict to the IO_dict which will be used for the corrupted images:
        self.IO_dict.noise_dict = original_IO_dict.noise_dict
        #(2). add the original downsampling factor:
        self.IO_dict.transforms_dict.downsampling_factor = original_IO_dict.transforms_dict.downsampling_factor
        #(3). warp the images the same way as we did the GT:
        LR_outputs_dict.tranforms_dict = HR_outputs_dict.transforms_dict
        LR_outputs_dict.update(LR_outputs_dict.tranforms_dict)
        #(4). add the original blur fraction to the corrupted frames:
        self.IO_dict.blur_fraction = original_IO_dict.transforms_dict.blur_fraction
        LR_outputs_dict = self.specific_dataset_functionality_before_cropping(LR_outputs_dict)
        LR_outputs_dict = self.perform_final_crop_on_original_images(LR_outputs_dict)
        LR_outputs_dict = self.specific_dataset_functionality_before_adding_noise(LR_outputs_dict)
        LR_outputs_dict = self.assign_original_images_to_outputs_dict(LR_outputs_dict)
        LR_outputs_dict = self.add_noise_to_tensors_full(LR_outputs_dict)
        LR_outputs_dict = self.specific_dataset_functionality_after_adding_noise(LR_outputs_dict)
        LR_outputs_dict = self.assign_final_outputs_after_noise(LR_outputs_dict)
        #####

        ### Adjust GT to HR Image We Read: ###
        LR_outputs_dict.output_frames_original = HR_outputs_dict.output_frames_original
        LR_outputs_dict.center_frame_original = HR_outputs_dict.center_frame_original

        return LR_outputs_dict


class Dataset_MultipleImagesFromSingleImage_AGN_BlurLowPass(Dataset_MultipleImagesFromSingleImage_AGN):
    def __init__(self, root_path, IO_dict):
        super().__init__(root_path, IO_dict)
        self.warp_object = Shift_Layer_Torch()
        self.gaussian_blur_layer = Gaussian_Blur_Layer(IO_dict.gaussian_blur_number_of_channels, IO_dict.gaussian_blur_kernel_size,
                                                       IO_dict.gaussian_blur_sigma)

    def specific_dataset_functionality_before_cropping(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        ### Shift Image: ###
        outputs_dict = self.shift_and_blur_images(outputs_dict, IO_dict)

        ### Blur Image: ###
        T, C, H, W = outputs_dict.output_frames_original.shape
        outputs_dict.output_frames_before_adding_noise = Gaussian_Blur_Wrapper_Torch(self.gaussian_blur_layer,
                                                                                     outputs_dict.output_frames_before_adding_noise,
                                                                                     number_of_channels_per_frame=C,
                                                                                     frames_dim=0,
                                                                                     channels_dim=1)
        return outputs_dict


class DataSet_Videos_In_Folders_AGN(DataSet_Videos_In_Folders):
    def __init__(self, root_folder, IO_dict):
        super().__init__(root_folder, IO_dict)

    def __getitem__(self, index):
        ### Get Frames: ###
        outputs_dict = self.base_get_current_sub_folder_video_batch(self.image_filenames_list_of_lists,
                                                                    self.images_list, index)

        ### Assign outputs_dict: ###
        self.number_of_image_frames_to_generate = min(self.number_of_image_frames_to_generate,
                                                      outputs_dict.output_frames_original.shape[0])

        ### Get random transformation factors between following frames: ###
        outputs_dict = self.get_random_transformation_vectors(outputs_dict)

        ### Perform Functionality (shifts, blur, etc'): ###
        outputs_dict = self.specific_dataset_functionality_before_cropping(outputs_dict)

        ### Perform Final Crop On Original Images: ###
        outputs_dict = self.perform_final_crop_on_original_images(outputs_dict)

        ### Normalize image graylevels according to camera ###
        outputs_dict = self.normalize_tensor_graylevels_to_photons(outputs_dict)

        ### Perform Functionality After Crop & Before Adding Noise (downsampling mostly): ###
        outputs_dict = self.specific_dataset_functionality_before_adding_noise(outputs_dict)

        ### Assign Original Images To Output Dict: ###
        outputs_dict = self.assign_original_images_to_outputs_dict(outputs_dict)

        ### Add Noise & Get Noise Map: ###
        outputs_dict = self.add_noise_to_tensors_full(outputs_dict)

        ### Perform Functionality After Adding Noise (For instance assign the correct form of stuff etc'): ###
        outputs_dict = self.specific_dataset_functionality_after_adding_noise(outputs_dict)

        ### Assign Final Outputs: ###
        outputs_dict = self.assign_final_outputs_after_noise(outputs_dict)

        return outputs_dict

    def __len__(self):
        return len(self.image_filenames_list_of_lists)


class DataSet_Videos_In_Folders_LoadCorruptedGT(DataSet_Videos_In_Folders):
    def __init__(self, root_folder, IO_dict=None):
        super().__init__(root_folder, IO_dict)
        ### Assign Object Variables: ###
        self.root_folder = root_folder
        self.IO_dict = IO_dict
        assign_attributes_from_dict(self, IO_dict)

        ### Get Videos In Folders FileNames (& Images if flag_use_RAM is True): ###
        #(1). GT:
        self.IO_dict.string_pattern_to_search = self.GT_string_pattern_to_search
        self.GT_image_filenames_list_of_lists, self.GT_images_folders, self.GT_images_list = \
            self.get_videos_in_folders_filenames_and_images(root_folder, self.IO_dict)

        #(2). Corrupted:
        self.IO_dict.string_pattern_to_search = self.Corrupted_string_pattern_to_search
        if self.IO_dict.corrupted_folder is None or self.IO_dict.corrupted_folder == '':
            self.IO_dict.corrupted_folder = root_folder #if noisy_root_folder is none, this means i'm using the same basic path but with different strings to seaerch
        self.Noisy_image_filenames_list_of_lists, self.Noisy_images_folders, self.Noisy_images_list = \
            self.get_videos_in_folders_filenames_and_images(self.IO_dict.corrupted_folder, self.IO_dict)

        ### Clean Up: ###
        self.Noisy_image_filenames_list_of_lists = clean_up_filenames_list(self.Noisy_image_filenames_list_of_lists)
        self.GT_image_filenames_list_of_lists = clean_up_filenames_list(self.GT_image_filenames_list_of_lists)
        self.image_filenames_list_of_lists = self.GT_image_filenames_list_of_lists


    def __getitem__(self, index):
        ### Get Frames: ###
        self.IO_dict.flag_crop_mode = 'random'
        GT_outputs_dict = self.base_get_current_sub_folder_video_batch(self.GT_image_filenames_list_of_lists, self.GT_images_list, index, start_index=-1, IO_dict=self.IO_dict)
        self.IO_dict.flag_crop_mode = 'predetermined'
        Noisy_outputs_dict = self.base_get_current_sub_folder_video_batch(self.Noisy_image_filenames_list_of_lists, self.Noisy_images_list, index,
                                                                          start_index=GT_outputs_dict.start_index, start_H=GT_outputs_dict.start_H, start_W=GT_outputs_dict.start_W)
        ### Check if moving images ###

        if self.IO_dict.is_moving_images:

            GT_outputs_dict.output_frames_original = torch.cat([GT_outputs_dict.output_frames_original.squeeze(0)] * self.IO_dict.num_moving_images, axis=0).unsqueeze(1)
            Noisy_outputs_dict.output_frames_original = torch.cat([Noisy_outputs_dict.output_frames_original.squeeze(0)] * self.IO_dict.num_moving_images, axis=0).unsqueeze(1)

        ### Assign outputs_dict: ###
        self.number_of_image_frames_to_generate = min(self.number_of_image_frames_to_generate, Noisy_outputs_dict.output_frames_original.shape[0])
        self.original_IO_dict = copy.deepcopy(self.IO_dict)
        # self.original_IO_dict.update()
        # self.original_IO_dict.update_copy(self.IO_dict.copy())

        ### Perform Functionality (shifts, blur, etc') On GT: ###
        self.IO_dict.noise_dict = get_default_noise_dict()
        self.IO_dict.transforms_dict.downsampling_factor = 1  # don't downsample here
        GT_outputs_dict = self.get_random_transformation_vectors(GT_outputs_dict)
        GT_outputs_dict = self.specific_dataset_functionality_before_cropping(GT_outputs_dict)
        GT_outputs_dict = self.perform_final_crop_on_original_images(GT_outputs_dict)
        GT_outputs_dict = self.normalize_tensor_graylevels_to_photons(GT_outputs_dict)
        GT_outputs_dict = self.specific_dataset_functionality_before_adding_noise(GT_outputs_dict)
        GT_outputs_dict = self.assign_original_images_to_outputs_dict(GT_outputs_dict)
        GT_outputs_dict = self.add_noise_to_tensors_full(GT_outputs_dict, GT=True)
        GT_outputs_dict = self.specific_dataset_functionality_after_adding_noise(GT_outputs_dict)
        GT_outputs_dict = self.assign_final_outputs_after_noise(GT_outputs_dict)

        ### Perform Functionality (shifts, blur, etc') On Noisy: ###
        self.IO_dict.noise_dict = self.original_IO_dict.noise_dict
        self.IO_dict.transforms_dict.downsampling_factor = self.original_IO_dict.transforms_dict.downsampling_factor
        if self.IO_dict.no_GT:
            self.IO_dict.transforms_dict.downsampling_factor = 1
        if not self.IO_dict.flag_normalize_images_to_PPP:
            GT_outputs_dict.PPP = None
        Noisy_outputs_dict.transforms_dict = GT_outputs_dict.transforms_dict
        Noisy_outputs_dict = self.specific_dataset_functionality_before_cropping(Noisy_outputs_dict)
        Noisy_outputs_dict = self.perform_final_crop_on_original_images(Noisy_outputs_dict)
        Noisy_outputs_dict = self.normalize_tensor_graylevels_to_photons(Noisy_outputs_dict, external_PPP=GT_outputs_dict.PPP)
        Noisy_outputs_dict = self.specific_dataset_functionality_before_adding_noise(Noisy_outputs_dict)
        Noisy_outputs_dict = self.assign_original_images_to_outputs_dict(Noisy_outputs_dict)
        Noisy_outputs_dict = self.add_noise_to_tensors_full(Noisy_outputs_dict, GT=False)
        Noisy_outputs_dict = self.specific_dataset_functionality_after_adding_noise(Noisy_outputs_dict)
        Noisy_outputs_dict = self.assign_final_outputs_after_noise(Noisy_outputs_dict)

        ### Assign GT To GT Images Loaded: ###
        Noisy_outputs_dict.output_frames_original = GT_outputs_dict.output_frames_original
        Noisy_outputs_dict.center_frame_original = GT_outputs_dict.center_frame_original

        return Noisy_outputs_dict

    def __len__(self):
        return len(self.image_filenames_list_of_lists)




#TODO: create a base class DataSet_SingleVideo_RollingIndex
class DataSet_SingleVideo_RollingIndex_AGN(DataSet_Videos_In_Folders):
    def __init__(self, root_folder, IO_dict):
        super().__init__(root_folder, IO_dict)

        ### Get Filenames & Images Of Movie File: ###    #TODO: add possibility of movie file (maybe read to images or use opencv functionality to read movie)
        if os.path.isdir(root_folder):
            #(*). Got Folder Of Images/Filesnames:
            self.image_filenames_list_of_lists, self.image_folders, self.images_list = \
                self.get_movie_filenames_and_images(root_folder, IO_dict)  #TODO: i think this is unecessary because Videos_In_Folders already has a method for reading the filenames and images
            self.flag_is_movie_file = False
        elif is_video_file(root_folder):
            #(*). Got A Single Movie File:
            H,W,FPS,number_of_movie_frames = video_get_movie_file_properties(root_folder)
            self.number_of_movie_frames = number_of_movie_frames
            self.Movie_Reader = cv2.VideoCapture(root_folder)
            self.flag_is_movie_file = True
            self.Movie_Reader.set(cv2.CAP_PROP_POS_FRAMES, self.IO_dict.movie_frame_to_start_from)  #Set movie frame start frame
            self.current_movie_images_list = []
            self.IO_dict.start_index = self.IO_dict.movie_frame_to_start_from

            ### Initialize to avoid raising errors: ###
            self.images_folders = []
            self.image_filenames_list_of_lists = []
            self.images_list = []

        ### Assign Object Variables: ###
        self.root_folder = root_folder
        self.IO_dict = IO_dict
        self.index = 0

    def __len__(self):
        if self.flag_is_movie_file:
            return int16(self.number_of_movie_frames)
        else:
            return len(self.image_filenames_list_of_lists[0])-self.IO_dict.number_of_images_per_video_to_load

    def base_get_current_sub_folder_video_batch(self, image_filenames_list_of_lists, images_list, index, start_index=-1, start_H=-1, start_W=-1, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Read Images: ###
        if self.flag_is_movie_file:
            #(*). Movie File:
            if len(self.current_movie_images_list) == 0:
                #(*). Read first batch of frames:
                for frame_index in np.arange(self.IO_dict.number_of_images_per_video_to_load):
                    flag_frame_available, current_movie_frame = self.Movie_Reader.read()
                    self.current_movie_images_list.append(np.atleast_3d(current_movie_frame))
            else:
                #(*). Advance one frame!:
                flag_frame_available, current_movie_frame = self.Movie_Reader.read()
                if flag_frame_available:
                    del self.current_movie_images_list[0]
                    self.current_movie_images_list.append(current_movie_frame)
                else:
                    1

            ### Concat Everything To One Tensor: ###
            current_folder_images_numpy = numpy.concatenate([self.current_movie_images_list], axis=0)
        else:
            #(*). Images Folder:
            current_folder_images_numpy, IO_dict = self.get_current_sub_folder_video_frames(image_filenames_list_of_lists,
                                                                                            images_list,
                                                                                            dataset_index=self.index,
                                                                                            frame_start_index=index,
                                                                                            start_H=-1, start_W=-1,
                                                                                            IO_dict=IO_dict)

        ### Crop All Images Consistently: ###
        # output_frames_original = crop_torch_or_numpy_batch(current_folder_images_numpy, IO_dict.initial_crop_size, crop_style=IO_dict.flag_crop_mode)
        IO_dict.initial_crop_size = to_list_of_certain_size(IO_dict.initial_crop_size,2)
        if IO_dict.flag_crop_mode == 'random':
            start_H, stop_H = get_random_start_stop_indices_for_crop(min(IO_dict.initial_crop_size[0], current_folder_images_numpy.shape[-3]), current_folder_images_numpy.shape[-3])
            start_W, stop_W = get_random_start_stop_indices_for_crop(min(IO_dict.initial_crop_size[1], current_folder_images_numpy.shape[-2]), current_folder_images_numpy.shape[-2])
            IO_dict.start_H = start_H
            IO_dict.start_W = start_W
        elif IO_dict.flag_crop_mode == 'center':
            if len(current_folder_images_numpy.shape) == 1:
                1
            T,H,W,C = current_folder_images_numpy.shape
            IO_dict.initial_crop_size[0] = min(IO_dict.initial_crop_size[0], H)
            IO_dict.initial_crop_size[1] = min(IO_dict.initial_crop_size[1], W)
            mat_in_rows_excess = H - IO_dict.initial_crop_size[0]
            mat_in_cols_excess = W - IO_dict.initial_crop_size[1]
            start = 0
            start_W = int(start + mat_in_cols_excess / 2)
            start_H = int(start + mat_in_rows_excess / 2)
            stop_W = start_W + IO_dict.initial_crop_size[1]
            stop_H = start_H + IO_dict.initial_crop_size[0]
            IO_dict.start_H = start_H
            IO_dict.start_W = start_W


        output_frames_original = crop_torch_or_numpy_batch(current_folder_images_numpy, crop_size_tuple_or_scalar=IO_dict.initial_crop_size,
                                            crop_style='predetermined', start_H=IO_dict.start_H, start_W=IO_dict.start_W)   # crop_style = 'random', 'predetermined', 'center'

        ### Augment Returned Image Frames: ###
        output_frames_original = self.transform_batch(output_frames_original)

        ### Numpy To Torch: ###
        if type(output_frames_original) is not torch.Tensor:
            output_frames_original = numpy_to_torch(output_frames_original)

        ### Normalize and Color-Convert: ###
        output_frames_original = self.normalize_and_convert_color_MultipleFrames(output_frames_original)

        ### Final Crop: ###
        output_frames_original = crop_torch_or_numpy_batch(output_frames_original, self.IO_dict.final_crop_size, crop_style='center')

        outputs_dict = EasyDict()
        outputs_dict.output_frames_original = output_frames_original
        outputs_dict.start_index = IO_dict.start_index
        outputs_dict.start_H = IO_dict.start_H
        outputs_dict.start_W = IO_dict.start_W
        return outputs_dict


    def get_movie_filenames_and_images(self, root_folder, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Initialize lists: ###
        image_filenames_list_of_lists = []
        image_folders = []
        images = []

        ### Loop Over Sub Folders And Read: ###
        folder_counter = 0
        for directory_path, directory_name, file_names_in_sub_directory in os.walk(root_folder):
            # If number of videos/directories so far exceeds maximum then break:
            if folder_counter > IO_dict.max_number_of_videos and IO_dict.max_number_of_videos != -1:
                break

            # Add new element to lists, representing a sub directory
            image_folders.append(directory_path)
            image_filenames_list_of_lists.append(read_image_filenames_from_folder(directory_path,
                                                                                  number_of_images=IO_dict.number_of_images_per_video_to_scan,
                                                                                  allowed_extentions=IO_dict.allowed_extentions,
                                                                                  flag_recursive=False))
            if IO_dict.flag_to_RAM:
                # Load Numpy Array Concatanted Along Channels To RAM
                images.append(read_images_from_folder(directory_path,
                                                      False,
                                                      np.inf,
                                                      # don't crop right now, crop when actually outputing frames
                                                      IO_dict.number_of_images_per_video_to_scan,
                                                      # Read all images from a possibly long video, when loading only load wanted number of images randomly
                                                      IO_dict.allowed_extentions,
                                                      flag_return_numpy_or_list='numpy',
                                                      flag_how_to_concat='T',
                                                      crop_style='random',
                                                      flag_return_torch=True,
                                                      transform=IO_dict.base_transform,
                                                      flag_to_BW=False))
            ### Uptick folder counter: ###
            folder_counter += 1

        return image_filenames_list_of_lists, image_folders, images

    def get_current_images_from_movie_folder(self, image_filenames_list_of_lists, images_list, index, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        if IO_dict.flag_to_RAM:
            # start_index, stop_index = get_random_start_stop_indices_for_crop(self.number_of_images_per_video, len(self.images_list[0]))
            current_folder_images_numpy = images_list[0][index:index + IO_dict.number_of_images_per_video_to_load]
        else:
            #(*). load specific images from index to index + self.number_of_images_per_video_to_load
            specific_subfolder_filenames_list = image_filenames_list_of_lists[0][index:index + IO_dict.number_of_images_per_video_to_load]  #Single Video Supported For Now!!!!
            current_folder_images_numpy = \
                read_images_from_filenames_list(specific_subfolder_filenames_list,
                                                flag_return_numpy_or_list='numpy',
                                                crop_size=np.inf,
                                                max_number_of_images=IO_dict.number_of_images_per_video_to_load,
                                                allowed_extentions=IO_dict.allowed_extentions,
                                                flag_how_to_concat=IO_dict.flag_how_to_concat,
                                                crop_style='center',
                                                flag_return_torch=IO_dict.flag_return_torch_on_load,
                                                transform=IO_dict.base_transform,
                                                flag_random_first_frame=False, #TODO: make this contingent upon which mode you're in
                                                first_frame_index=-1)  #TODO: make this contingent upon which mode you're in ???

        return current_folder_images_numpy

    def __getitem__(self, index):
        outputs_dict = EasyDict()

        # ### Read Images: ###
        # output_frames_original = self.get_current_images_from_movie_folder(self.image_filenames_list_of_lists, self.images_list, index)

        ### Get Frames: ###
        outputs_dict = self.base_get_current_sub_folder_video_batch(self.image_filenames_list_of_lists,
                                                                    self.images_list, index)

        ### Assign outputs_dict: ###
        self.number_of_image_frames_to_generate = min(self.number_of_image_frames_to_generate,
                                                      outputs_dict.output_frames_original.shape[0])

        ### Get random transformation factors between following frames: ###
        outputs_dict = self.get_random_transformation_vectors(outputs_dict)

        ### Perform Functionality (shifts, blur, etc'): ###
        outputs_dict = self.specific_dataset_functionality_before_cropping(outputs_dict)

        ### Perform Final Crop On Original Images: ###
        outputs_dict = self.perform_final_crop_on_original_images(outputs_dict)

        ### Normalize image graylevels according to camera ###
        outputs_dict = self.normalize_tensor_graylevels_to_photons(outputs_dict)

        ### Perform Functionality After Crop & Before Adding Noise (downsampling mostly): ###
        outputs_dict = self.specific_dataset_functionality_before_adding_noise(outputs_dict)

        ### Assign Original Images To Output Dict: ###
        outputs_dict = self.assign_original_images_to_outputs_dict(outputs_dict)

        ### Add Noise & Get Noise Map: ###
        outputs_dict = self.add_noise_to_tensors_full(outputs_dict)

        ### Perform Functionality After Adding Noise (For instance assign the correct form of stuff etc'): ###
        outputs_dict = self.specific_dataset_functionality_after_adding_noise(outputs_dict)

        ### Assign Final Outputs: ###
        outputs_dict = self.assign_final_outputs_after_noise(outputs_dict)

        return outputs_dict
from PIL import Image
import torchvision

#Augmentations:
import PIL
import albumentations
from albumentations import *
from albumentations.augmentations.functional import to_gray


import os
import torch.utils.data as data
import torch
import numpy as np
from easydict import EasyDict as edict
import cv2
from tqdm import tqdm
from easydict import EasyDict

from RapidBase.import_all import *
# from RapidBase.Basic_Import_Libs import *
# from RapidBase.Utils.Path_and_Reading_utils import *
# import RapidBase.Utils.Albumentations_utils
# import RapidBase.Utils.IMGAUG_utils
# from RapidBase.Utils.IMGAUG_utils import *
# from RapidBase.Utils.Pytorch_Numpy_Utils import RGB2BW_torch
# from RapidBase.Utils.Add_Noise import add_noise_to_images_full
# from RapidBase.Utils.Warping_Shifting import Shift_Layer_Torch
# from RapidBase.Utils.Path_and_Reading_utils import read_image_filenames_from_folder
import colour_demosaicing

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',  #in ImageNet files there appears to be, at least in my computer, a problem with png and gif
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP', '.TIFF', '.tiff', '.tif'
]


IMG_EXTENSIONS_PNG = [
    '.png',
    '.PNG',  #in ImageNet files there appears to be, at least in my computer, a problem with png and gif
]




class Dataset_MultipleImages(data.Dataset):
    def __init__(self, root_folder=None, IO_dict=None):
        super().__init__()
        self.IO_dict = IO_dict
        assign_attributes_from_dict(self, IO_dict)

        ### Upsampling & Downsampling Layer: ###
        IO_dict.transforms_dict.upsample_method = 'bilinear'  # 'nearest', 'bilinear', 'bicubic', 'fft'
        IO_dict.transforms_dict.downsample_method = 'binning'  # 'binning', 'nearest', 'bilinear', 'bicubic', 'fft'

        ### Whether to do SR-SameSize: ###
        self.flag_upsample_noisy_input_to_same_size_as_original = False  #by default

        ### UpSample Method: ###
        if IO_dict.transforms_dict.upsample_method == 'bilinear':
            self.upsampling_layer = nn.Upsample(scale_factor=self.IO_dict.transforms_dict.downsampling_factor, mode='bilinear',
                                                align_corners=None)
        elif IO_dict.transforms_dict.upsample_method == 'bicubic':
            self.upsampling_layer = nn.Upsample(scale_factor=self.IO_dict.transforms_dict.downsampling_factor, mode='bicubic',
                                                align_corners=None)
        elif IO_dict.transforms_dict.upsample_method == 'nearest':
            self.upsampling_layer = nn.Upsample(scale_factor=self.IO_dict.transforms_dict.downsampling_factor, mode='nearest',
                                                align_corners=None)

        ### DownSample Method: ###
        if IO_dict.transforms_dict.upsample_method == 'bilinear':
            self.downsampling_layer = nn.Upsample(scale_factor=1/self.IO_dict.transforms_dict.downsampling_factor, mode='bilinear',
                                                align_corners=None)
        elif IO_dict.transforms_dict.upsample_method == 'bicubic':
            self.downsampling_layer = nn.Upsample(scale_factor=1/self.IO_dict.transforms_dict.downsampling_factor, mode='bicubic',
                                                align_corners=None)
        elif IO_dict.transforms_dict.upsample_method == 'nearest':
            self.downsampling_layer = nn.Upsample(scale_factor=1/self.IO_dict.transforms_dict.downsampling_factor, mode='nearest',
                                                align_corners=None)
        elif IO_dict.transforms_dict.downsample_method == 'binning':
            self.downsampling_layer = nn.AvgPool2d(self.IO_dict.transforms_dict.downsampling_factor)
        self.average_pooling_layer = nn.AvgPool2d(self.IO_dict.transforms_dict.downsampling_factor)


        ### Warp/Transforms Layers: ###
        #TODO: add affine transform layers when it's done, add general deformation (turbulence) transform layer 
        self.warp_object = Shift_Layer_Torch()

    def get_random_transformation_vectors(self, outputs_dict, IO_dict=None):
        """
        This function creates vectors of random transformations (translation, rotation, scaling) in the ranges
        specified in IO_dict and assigns them to outputs_dict
        :param outputs_dict: specific data information
        :param IO_dict: general processing information
        :return: outputs_dict with transformation vectors
        """
        if IO_dict is None:
            IO_dict = self.IO_dict

        T, C, H, W = outputs_dict.output_frames_original.shape
        current_device = outputs_dict.output_frames_original.device
        transforms_dict = EasyDict()

        ### Shift/Translation: ###
        shift_size = IO_dict.transforms_dict.shift_size
        transforms_dict.shift_x_vec = torch.Tensor(get_random_number_in_range(-shift_size, shift_size, T)).to(current_device)
        transforms_dict.shift_y_vec = torch.Tensor(get_random_number_in_range(-shift_size, shift_size, T)).to(current_device)
        # TODO add random thetas and scaling factors as well

        ### Put all the transforms dict variables into outputs_dict: ###
        outputs_dict.transforms_dict = transforms_dict
        outputs_dict.update(outputs_dict.transforms_dict)

        return outputs_dict

    def shift_and_blur_images(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Initial: ###
        outputs_dict.output_frames_before_adding_noise = outputs_dict.output_frames_original
        T,C,H,W = outputs_dict.output_frames_original.shape
        blur_fraction = IO_dict.transforms_dict.blur_fraction
        shift_size = IO_dict.transforms_dict.shift_size
        directional_blur_size = shift_size * blur_fraction

        ### Get total shifts: ###
        N_blur_steps_total = max(self.IO_dict.transforms_dict.number_of_blur_steps_per_pixel * directional_blur_size, 1)
        ### divide it into a blur step and a shift step: ###
        #TODO: when shift layers will be completed this won't be necessary
        shiftx = outputs_dict.transforms_dict.shift_x_vec.cpu().numpy()
        shifty = outputs_dict.transforms_dict.shift_y_vec.cpu().numpy()
        if self.IO_dict.blur_range:
            blur_fraction = torch.rand(1).item() * blur_fraction
        shiftx_blur = shiftx * blur_fraction
        shifty_blur = shifty * blur_fraction
        shiftx_preblur_shift = shiftx - shiftx_blur
        shifty_preblur_shift = shifty - shifty_blur

        ### Shift the image to get the "perfect" image: ###
        #(1). Shift original image so that the blurry(noisy) image and original image are at the same place:
        if any(shiftx_preblur_shift != 0):
            outputs_dict.output_frames_original = self.warp_object(outputs_dict.output_frames_original, shiftx, shifty)
        else:
            outputs_dict.output_frames_original = outputs_dict.output_frames_original
        #(2). Shift -> blur to get the blurred image: ###
        outputs_dict.output_frames_before_adding_noise = self.warp_object(
            outputs_dict.output_frames_before_adding_noise, shiftx_preblur_shift, shifty_preblur_shift)
        #TODO: use a blur layer when it is done being written!!!
        outputs_dict.output_frames_before_adding_noise = blur_image_motion_blur_torch(
            outputs_dict.output_frames_before_adding_noise, shiftx_blur, shifty_blur, N_blur_steps_total, self.warp_object)

        ### Assign/Initiailize HR image here: ###
        outputs_dict.output_frames_original_HR = outputs_dict.output_frames_original.data

        ### Keep track of shifts: ###
        #(1). Correct for downsampling factor which will come later
        outputs_dict.transforms_dict.shift_x_vec = torch.Tensor(outputs_dict.transforms_dict.shift_x_vec / IO_dict.transforms_dict.downsampling_factor)
        outputs_dict.transforms_dict.shift_y_vec = torch.Tensor(outputs_dict.transforms_dict.shift_y_vec / IO_dict.transforms_dict.downsampling_factor)
        L = len(outputs_dict.transforms_dict.shift_x_vec)
        outputs_dict.transforms_dict.shift_x_to_reference_vec = outputs_dict.transforms_dict.shift_x_vec - outputs_dict.transforms_dict.shift_x_vec[L // 2]
        outputs_dict.transforms_dict.shift_y_to_reference_vec = outputs_dict.transforms_dict.shift_y_vec - outputs_dict.transforms_dict.shift_y_vec[L // 2]

        ### Produce Full Optical Flow From Shifts: ###
        outputs_dict.optical_flow_delta_x = outputs_dict.transforms_dict.shift_x_vec.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1) * torch.ones_like(outputs_dict.output_frames_original)
        outputs_dict.optical_flow_delta_y = outputs_dict.transforms_dict.shift_y_vec.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1) * torch.ones_like(outputs_dict.output_frames_original)
        outputs_dict.optical_flow_GT = torch.cat((outputs_dict.optical_flow_delta_x, outputs_dict.optical_flow_delta_y), 1)  # [T=N_frames,C=2(x,y),H,W]
        outputs_dict.optical_flow_delta_x_to_reference = outputs_dict.optical_flow_delta_x - outputs_dict.optical_flow_delta_x[T//2:T//2+1]
        outputs_dict.optical_flow_delta_y_to_reference = outputs_dict.optical_flow_delta_y - outputs_dict.optical_flow_delta_y[T//2:T//2+1]
        outputs_dict.optical_flow_GT = torch.cat((outputs_dict.optical_flow_delta_x_to_reference, outputs_dict.optical_flow_delta_y_to_reference), 1)
        return outputs_dict

    def downsample_before_adding_noise(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Perform DownSampling: ###
        if IO_dict.transforms_dict.downsampling_factor > 1:
            output_frames_after_downsampling = self.downsampling_layer(outputs_dict.output_frames_before_adding_noise)
        else:
            output_frames_after_downsampling = outputs_dict.output_frames_before_adding_noise
        outputs_dict.output_frames_before_adding_noise = output_frames_after_downsampling
        return outputs_dict

    def upsample_tensors_where_needed(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Perform Upsampling For Deblur-SR Input To Model: ###
        if self.IO_dict.transforms_dict.flag_upsample_noisy_input_to_same_size_as_original:
            outputs_dict.output_frames_noisy = self.upsampling_layer(outputs_dict.output_frames_noisy)

        ### Upsample the noise map when doing Super-Resolution (the noise map is added to the input low resolution): ###
        if self.IO_dict.transforms_dict.downsampling_factor > 1:
            noise_instance_map_HR = self.upsampling_layer(outputs_dict.noise_instance_map)
            noise_sigma_map_HR = self.upsampling_layer(outputs_dict.noise_sigma_map)

            T_noisy, C_noisy, H_noisy, W_noisy = outputs_dict.output_frames_noisy.shape
            T_original, C_original, H_original, W_original = outputs_dict.output_frames_original.shape

            ### this is to support SR & SR_Deblur: ###
            if H_original > H_noisy:
                outputs_dict.output_frames_noisy_HR = self.upsampling_layer(outputs_dict.output_frames_noisy)
            else:
                outputs_dict.output_frames_noisy_HR = outputs_dict.output_frames_noisy  # Add upsampled noise to final HR frames

        else:  # self.IO_dict.transforms_dict.downsampling_factor == 1:
            outputs_dict.output_frames_noisy_HR = outputs_dict.output_frames_noisy
            noise_instance_map_HR = outputs_dict.noise_instance_map
            noise_sigma_map_HR = outputs_dict.noise_sigma_map

        ### Assign Center Frame Of HR Images: ###
        outputs_dict.center_frame_noisy_HR = outputs_dict.output_frames_noisy_HR[
                                             IO_dict.number_of_image_frames_to_generate // 2:IO_dict.number_of_image_frames_to_generate // 2 + 1,
                                             :, :]

        ### Assign the HR noise maps to outputs_dict: ###
        outputs_dict.noise_sigma_map_HR = noise_sigma_map_HR
        outputs_dict.noise_instance_map_HR = noise_instance_map_HR

        return outputs_dict

    def get_filenames_and_images_from_folder(self, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        if IO_dict.flag_to_RAM:
            numpy_images_list, image_filenames_list = read_images_and_filenames_from_folder(IO_dict.root_path,
                                                                                            IO_dict.flag_recursive,
                                                                                            IO_dict.initial_crop_size,
                                                                                            IO_dict.max_number_of_images,
                                                                                            IO_dict.allowed_extentions,
                                                                                            flag_return_numpy_or_list='list',
                                                                                            flag_how_to_concat='T',
                                                                                            crop_style=IO_dict.flag_crop_mode,
                                                                                            flag_to_BW=IO_dict.flag_to_BW_before_noise,
                                                                                            string_pattern_to_search=IO_dict.string_pattern_to_search)

        else:
            image_filenames_list = get_image_filenames_from_folder(IO_dict.root_path,
                                                                   IO_dict.max_number_of_images,
                                                                   IO_dict.allowed_extentions,
                                                                   IO_dict.flag_recursive,
                                                                   IO_dict.string_pattern_to_search)
            numpy_images_list = None

        return image_filenames_list, numpy_images_list

    def get_noise_filenames_and_images(self, IO_dict=None):
        #TODO: see if this is still necessary
        if IO_dict is None:
            IO_dict = self.IO_dict

        if IO_dict.flag_noise_images_to_RAM:
            noise_numpy_images_list, noise_image_filenames_list = read_images_and_filenames_from_folder(
                IO_dict.noise_images_path,
                flag_recursive=True,
                crop_size=np.inf,
                max_number_of_images=IO_dict.max_number_of_noise_images,
                allowed_extentions=IO_dict.allowed_extentions,
                flag_return_numpy_or_list='list',
                flag_how_to_concat='T',
                crop_style=IO_dict.flag_crop_mode,
                flag_to_BW=IO_dict.flag_to_BW,
                string_pattern_to_search=IO_dict.string_pattern_to_search)

        else:
            noise_image_filenames_list = get_image_filenames_from_folder(IO_dict.noise_images_path,
                                                                         IO_dict.max_number_of_noise_images,
                                                                         IO_dict.allowed_extentions,
                                                                         IO_dict.flag_recursive,
                                                                         IO_dict.string_pattern_to_search)
            noise_numpy_images_list = None
        ### Assign Noise Images & Filenames List To Internal Attributes: ###
        return noise_image_filenames_list, noise_numpy_images_list

    def specific_dataset_functionality_before_cropping(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        outputs_dict = self.shift_and_blur_images(outputs_dict, IO_dict)
        return outputs_dict

    def specific_dataset_functionality_before_adding_noise(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        outputs_dict = self.downsample_before_adding_noise(outputs_dict, IO_dict)
        return outputs_dict

    def specific_dataset_functionality_after_adding_noise(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        outputs_dict = self.upsample_tensors_where_needed(outputs_dict, IO_dict)
        return outputs_dict

    def normalize_tensor_graylevels_to_photons(self, outputs_dict, IO_dict=None, external_PPP=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        if IO_dict.flag_normalize_images_to_PPP:
            if external_PPP is None:
                IO_dict.PPP = to_list_of_certain_size(IO_dict.PPP, 2)
                current_PPP = get_random_number_in_range(IO_dict.PPP[0], IO_dict.PPP[1])
                outputs_dict.PPP = current_PPP
            else:
                current_PPP = external_PPP
            output_frames_original = outputs_dict.output_frames_original
            original_images_mean = output_frames_original.mean()
            PPP_normalization_factor = current_PPP / (original_images_mean.abs() + 1e-6)
            # normalized_images = output_frames_original * current_PPP / (original_images_mean.abs() + 1e-6)
            outputs_dict.output_frames_original = output_frames_original * PPP_normalization_factor
            outputs_dict.output_frames_before_adding_noise = outputs_dict.output_frames_before_adding_noise * PPP_normalization_factor
            outputs_dict.output_frames_original_HR = outputs_dict.output_frames_original_HR * PPP_normalization_factor
        # else:
        #     outputs_dict.PPP = None
        return outputs_dict

    def add_noise_and_get_frames_final(self, outputs_dict, output_frames_original_to_add_noise_to, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        ### Assign outputs_dict (Including Adding Noise): ###
        outputs_dict = self.get_outputs_from_original_images_additive_gaussian_noise(outputs_dict, output_frames_original_to_add_noise_to, IO_dict)
        return outputs_dict

    def assign_original_images_to_outputs_dict(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        ### Assign Original Image To Outputs_dict: ###
        outputs_dict.center_frame_original = outputs_dict.output_frames_original[IO_dict.number_of_image_frames_to_generate // 2, :,:, :]

        return outputs_dict

    def perform_final_crop_on_original_images(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        outputs_dict.output_frames_original = crop_torch_or_numpy_batch(outputs_dict.output_frames_original, self.IO_dict.final_crop_size, crop_style='center')
        outputs_dict.output_frames_before_adding_noise = crop_torch_or_numpy_batch(outputs_dict.output_frames_before_adding_noise, self.IO_dict.final_crop_size, crop_style='center')
        return outputs_dict

    def transfer_Tstack_to_Cstack(self, outputs_dict):
        #TODO: understand whether i can ismply loop over dictionary attributes, check if they're tensors and Cstack_to_Tstack
        outputs_dict.output_frames_original = Tstack_to_Cstack(outputs_dict.output_frames_original)
        outputs_dict.center_frame_original = Tstack_to_Cstack(outputs_dict.center_frame_original)

        if hasattr(outputs_dict, 'output_frames_before_adding_noise'):
            outputs_dict.output_frames_before_adding_noise = Tstack_to_Cstack(outputs_dict.output_frames_before_adding_noise)
        if hasattr(outputs_dict, 'noise_sigma_map'):
            outputs_dict.noise_sigma_map = Tstack_to_Cstack(outputs_dict.noise_sigma_map)
        if hasattr(outputs_dict, 'noise_instance_map'):
            outputs_dict.noise_instance_map = Tstack_to_Cstack(outputs_dict.noise_instance_map)
        if hasattr(outputs_dict, 'center_frame_noisy'):
            outputs_dict.center_frame_noisy = Tstack_to_Cstack(outputs_dict.center_frame_noisy)
        if hasattr(outputs_dict, 'output_frames_noisy'):
            outputs_dict.output_frames_noisy = Tstack_to_Cstack(outputs_dict.output_frames_noisy)
        if hasattr(outputs_dict, 'output_frames_noisy_HR'):
            outputs_dict.output_frames_noisy_HR = Tstack_to_Cstack(outputs_dict.output_frames_noisy_HR)
        if hasattr(outputs_dict, 'center_frame_noisy_HR'):
            outputs_dict.center_frame_noisy_HR = Tstack_to_Cstack(outputs_dict.center_frame_noisy_HR)
        if hasattr(outputs_dict, 'noise_map_to_add_HR'):
            outputs_dict.noise_map_to_add_HR = Tstack_to_Cstack(outputs_dict.noise_map_to_add_HR)
        if hasattr(outputs_dict, 'output_frames_original_HR'):
            outputs_dict.output_frames_original_HR = Tstack_to_Cstack(outputs_dict.output_frames_original_HR)
        return outputs_dict

    def transfer_Cstack_to_Tstack(self, outputs_dict):
        #TODO: understand whether i can ismply loop over dictionary attributes, check if they're tensors and Cstack_to_Tstack
        outputs_dict.output_frames_original = Cstack_to_Tstack(outputs_dict.output_frames_original)
        outputs_dict.center_frame_original = Cstack_to_Tstack(outputs_dict.center_frame_original)

        if hasattr(outputs_dict, 'output_frames_before_adding_noise'):
            outputs_dict.output_frames_before_adding_noise = Cstack_to_Tstack(outputs_dict.output_frames_before_adding_noise)
        if hasattr(outputs_dict, 'noise_sigma_map'):
            outputs_dict.noise_sigma_map = Cstack_to_Tstack(outputs_dict.noise_sigma_map)
        if hasattr(outputs_dict, 'noise_instance_map'):
            outputs_dict.noise_instance_map = Cstack_to_Tstack(outputs_dict.noise_instance_map)
        if hasattr(outputs_dict, 'center_frame_noisy'):
            outputs_dict.center_frame_noisy = Cstack_to_Tstack(outputs_dict.center_frame_noisy)
        if hasattr(outputs_dict, 'output_frames_noisy'):
            outputs_dict.output_frames_noisy = Cstack_to_Tstack(outputs_dict.output_frames_noisy)
        if hasattr(outputs_dict, 'output_frames_noisy_HR'):
            outputs_dict.output_frames_noisy_HR = Cstack_to_Tstack(outputs_dict.output_frames_noisy_HR)
        if hasattr(outputs_dict, 'center_frame_noisy_HR'):
            outputs_dict.center_frame_noisy_HR = Cstack_to_Tstack(outputs_dict.center_frame_noisy_HR)
        if hasattr(outputs_dict, 'noise_map_to_add_HR'):
            outputs_dict.noise_map_to_add_HR = Cstack_to_Tstack(outputs_dict.noise_map_to_add_HR)
        if hasattr(outputs_dict, 'output_frames_original_HR'):
            outputs_dict.output_frames_original_HR = Cstack_to_Tstack(outputs_dict.output_frames_original_HR)

        return outputs_dict

    def get_approximate_running_average(self, output_frames_original, output_frames_noisy, outputs_dict, noise_map_to_add, IO_dict=None):
        #TODO: understand what to do with this one!!!!, i ignore it for now
        if IO_dict is None:
            IO_dict = self.IO_dict
        #TODO: probably need to make sure this uses noise_map_to_add = outputs_dict.HR_noise_map or something...probably should even add any input besides outputs_dict
        output_frames_noisy_running_average = torch.zeros_like(output_frames_original)
        for i in np.arange(output_frames_noisy.shape[0]):
            output_frames_noisy_running_average[i:i + 1, :, :, :] = output_frames_original[i:i + 1, :, :,:] + 1 / np.sqrt(i + 1) * noise_map_to_add[i:i + 1, :, :, :]
        outputs_dict.output_frames_noisy_running_average = output_frames_noisy_running_average  # [T,C,H,W]

        return outputs_dict, output_frames_noisy_running_average

    def get_noise_sigma_map(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        #TODO: this still isn't used, understand whether i want to use sigma_map or noise instance itself
        ### Readout Noise Sigma: ###
        noise_sigma_map = self.readout_noise_sigma_map  #TODO: add this calculation

        #### Shot Noise Sigma Map: ###
        noise_sigma_map += outputs_dict.shot_noise_sigma_map

        ### Add Noise Sigma Map To Outputs Dict: ###
        outputs_dict.noise_sigma_map = noise_sigma_map

        return outputs_dict


    def get_item_noise_image(self, indices, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        ### Read Noise Images: ###
        if IO_dict.flag_noise_images_to_RAM:
            # current_noise_images = self.noise_images_list[indices]
            current_noise_images = np.concatenate(get_sublist_from_list(self.noise_images_list, indices), -1)
        else:
            specific_subfolder_filenames_list = [self.noise_image_filenames_list[index] for index in indices]
            current_noise_images = read_images_from_filenames_list(specific_subfolder_filenames_list,
                                                                   flag_return_numpy_or_list='numpy',
                                                                   crop_size=np.inf,
                                                                   max_number_of_images=np.inf,
                                                                   allowed_extentions=IO_dict.allowed_extentions,
                                                                   flag_how_to_concat='T',
                                                                   crop_style='center',
                                                                   flag_return_torch=False,
                                                                   transform=None,
                                                                   flag_random_first_frame=False, flag_to_BW=False)

        # (*). Now that i've finished the base transform part (which could require PIL.Image object) now i need this in numpy format: (*)#
        ### Normalize To 1 and convert to float32: ###
        if IO_dict.image_loader == ImageLoaderPIL:
            current_noise_images = np.array(current_noise_images)

        ### Normalize and Color-Convert: ###
        current_noise_images = self.normalize_and_convert_color(current_noise_images)

        ### Crop Center: ###
        # (*). Important to be able to stack frames of different initial sizes
        current_noise_images = crop_numpy_batch(current_noise_images, IO_dict.initial_crop_size, IO_dict.flag_crop_mode)
        ######################################################################################

        if len(current_noise_images.shape) == 3:
            ### To pytorch convention: [H,W,C]->[C,H,W] ###
            current_noise_images = np.ascontiguousarray(np.transpose(current_noise_images, [2, 0, 1]))
        elif len(current_noise_images.shape) == 4:
            ### To pytorch convention: [T,H,W,C]->[T,C,H,W] ###
            current_noise_images = np.ascontiguousarray(np.transpose(current_noise_images, [0, 3, 1, 2]))
        current_noise_images = torch.Tensor(current_noise_images)

        return current_noise_images

    def multiply_and_stack_single_image(self, current_image, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        if type(current_image) == torch.Tensor:
            output_frames = torch.cat([current_image.unsqueeze(0)] * IO_dict.number_of_image_frames_to_generate, axis=0)  # [T,C,H,W]
        else:
            output_frames = np.concatenate([np.expand_dims(current_image, 0)] * IO_dict.number_of_image_frames_to_generate, axis=0) #[T,H,W,C]

        return output_frames

    def get_single_image_from_folder(self, image_filenames_list, images_list, index, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        ### Load Image: ###
        filename = image_filenames_list[index]
        if IO_dict.flag_to_RAM:
            current_image = images_list[index]
        else:
            current_image = IO_dict.image_loader(filename)
        return current_image

    def augment_and_crop_single_image(self, current_image, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### If there is a transform apply it to the SINGLE IMAGE (whether it be from imgaug, albumentations or torchvision): ###
        if IO_dict.transforms_dict.flag_base_transform:
            if IO_dict.transforms_dict.base_transform is not None:
                if type(IO_dict.transforms_dict.base_transform) == iaa.Sequential:
                    deterministic_Augmenter_Object = IO_dict.transforms_dict.base_transform.to_deterministic()
                    current_image = deterministic_Augmenter_Object.augment_image(current_image);
                elif type(IO_dict.transforms_dict.base_transform) == albumentations.Compose:
                    augmented_images_dictionary = IO_dict.transforms_dict.base_transform(image=current_image);
                    current_image = augmented_images_dictionary['image']
                elif type(IO_dict.transforms_dict.base_transform) == torchvision.transforms.transforms.Compose:
                    current_image = IO_dict.transforms_dict.base_transform(current_image)
                else:  # Custrom Transform
                    current_image = IO_dict.transforms_dict.base_transform(current_image)

        # (*). Now that i've finished the base transform part (which could require PIL.Image object) now i need this in numpy format: (*)#
        ### Normalize To 1 and convert to float32: ###
        if IO_dict.image_loader == ImageLoaderPIL:
            current_image = np.array(current_image)

        ### Normalize and Color-Convert: ###
        current_image = self.normalize_and_convert_color(current_image)

        ### Add Turbulence Deformation: ###
        if IO_dict.transforms_dict.flag_turbulence_transform:
            current_image = turbulence_deformation_single_image(current_image, Cn2=IO_dict.transforms_dict.Cn2)

        ### Crop Center: ###
        # (*). Important to be able to stack frames of different initial sizes
        current_image = crop_numpy_batch(current_image, IO_dict.initial_crop_size, IO_dict.flag_crop_mode)
        ######################################################################################

        if IO_dict.flag_make_tensor_before_batch_transform:
            ### To pytorch convention: [H,W,C]->[C,H,W] ###
            current_image = np.ascontiguousarray(np.transpose(current_image, [2, 0, 1]))
            # if current_image.dtype == 'uint8' or current_image.dtype == 'uint16':
            #     current_image = current_image.astype(int)
            current_image = torch.Tensor(current_image) #[:3,:,:]

        return current_image

    def get_multiple_images_from_single_image(self, image_filenames_list, images_list, index):
        ### Initialize outputs_dict for the first time: ### #TODO: do this outside the function
        outputs_dict = EasyDict()

        ### Read current frame (perform base transform on the individual image if wanted), multiply it and stack it: ###
        current_image = self.get_single_image_from_folder(image_filenames_list, images_list, index)
        current_image = self.augment_and_crop_single_image(current_image)
        output_frames_original = self.multiply_and_stack_single_image(current_image)

        ### Perform batch_transform on the entire batch: ###
        output_frames_original = self.transform_batch(output_frames_original)

        ### Numpy To Torch: ###
        if type(output_frames_original) is not torch.Tensor:
            output_frames_original = numpy_to_torch(output_frames_original)

        ### Assign output frames original to outputs_dict: ###
        outputs_dict.output_frames_original = output_frames_original

        return outputs_dict

    def normalize_and_convert_color(self, input_frames, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        ### Normalize Image: ###
        output_frames = input_frames

        ### Stack BW frames to make "psudo-RGB" frames if wanted: ###
        if IO_dict.flag_to_RGB_before_noise:
            output_frames = BW2RGB(output_frames)

        ### RGB -> BW: ###
        if IO_dict.flag_to_BW_before_noise:
            output_frames = RGB2BW(output_frames)

        return output_frames

    def normalize_and_convert_color_MultipleFrames(self, input_frames, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Initialize Image: ###
        output_frames = input_frames

        ### Stack BW frames to make "psudo-RGB" frames if wanted: ###
        if IO_dict.flag_to_RGB_before_noise:
            output_frames = BW2RGB_MultipleFrames(output_frames, IO_dict.flag_how_to_concat)

        ### RGB -> BW: ###
        if IO_dict.flag_to_BW_before_noise:
            output_frames = RGB2BW_MultipleFrames(output_frames, IO_dict.flag_how_to_concat)

        return output_frames

    def RGB2BW_for_whole_dict(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        outputs_dict.output_frames_original = RGB2BW_MultipleFrames(outputs_dict.output_frames_original, IO_dict.flag_how_to_concat)
        outputs_dict.output_frames_noisy = RGB2BW_MultipleFrames(outputs_dict.output_frames_noisy, IO_dict.flag_how_to_concat)
        outputs_dict.output_frames_noisy_HR = RGB2BW_MultipleFrames(outputs_dict.output_frames_noisy_HR, IO_dict.flag_how_to_concat)
        outputs_dict.center_frame_original = RGB2BW_MultipleFrames(outputs_dict.center_frame_original, IO_dict.flag_how_to_concat)
        return outputs_dict

    def BW2RGB_for_whole_dict(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        outputs_dict.output_frames_original = BW2RGB_MultipleFrames(outputs_dict.output_frames_original, IO_dict.flag_how_to_concat)
        outputs_dict.output_frames_noisy = BW2RGB_MultipleFrames(outputs_dict.output_frames_noisy, IO_dict.flag_how_to_concat)
        outputs_dict.output_frames_noisy_HR = BW2RGB_MultipleFrames(outputs_dict.output_frames_noisy_HR, IO_dict.flag_how_to_concat)
        outputs_dict.center_frame_original = BW2RGB_MultipleFrames(outputs_dict.center_frame_original, IO_dict.flag_how_to_concat)
        return outputs_dict

    def transform_batch(self, input_frames, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
        if IO_dict.transforms_dict.flag_batch_transform:
            if IO_dict.transforms_dict.batch_transform is not None:
                if type(IO_dict.transforms_dict.batch_transform) == iaa.Sequential:
                    deterministic_Augmenter_Object = IO_dict.transform.to_deterministic()
                    output_frames = deterministic_Augmenter_Object.augment_image(input_frames)
                elif type(IO_dict.transforms_dict.batch_transform) == albumentations.Compose:
                    augmented_images_dictionary = IO_dict.transforms_dict.transform(image=input_frames)
                    output_frames = augmented_images_dictionary['image']
                elif type(IO_dict.transforms_dict.batch_transform) == torchvision.transforms.transforms.Compose:
                    output_frames = IO_dict.transforms_dict.transform(input_frames)
                else:
                    output_frames = IO_dict.transforms_dict.transform(input_frames)
        else:
            output_frames = input_frames
        return output_frames

    def add_noise_to_tensors_full(self, outputs_dict, IO_dict=None, GT=False):
        if IO_dict is None:
            IO_dict = self.IO_dict

        images_to_noise = outputs_dict.output_frames_before_adding_noise
        if IO_dict.bayer and (not GT):
            mosaic_images_to_noise = []
            for image in images_to_noise:
                x = torch.einsum('...ijk->...jki', image)
                mosaic = colour_demosaicing.mosaicing_CFA_Bayer(x, 'GRBG')
                y = torch.tensor(mosaic).unsqueeze(0)
                mosaic_images_to_noise.append(y)
            images_to_noise = torch.stack(mosaic_images_to_noise)
        noisy_images, noise_instance_map, noise_sigma_map = add_noise_to_images_full(images_to_noise, IO_dict.noise_dict, flag_check_c_or_t=False)
        # imshow_torch_seamless(noisy_images[0], print_info=True, title_str='mosaic')
        # from Seamless import imshow_torch_seamless,imshow_torch_video_seamless
        if IO_dict.bayer and (not GT):
            if  IO_dict.vrt_debayer:
                noisy_images  = BW2RGB(noisy_images).type(torch.cuda.FloatTensor)
            else:
                demosaic_noisy_images = []
                for image in noisy_images:
                    demosaic = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(image.squeeze(0), 'GRBG')
                    y = torch.einsum('...jki->...ijk', torch.tensor(demosaic))
                    demosaic_noisy_images.append(y)
                noisy_images = torch.stack(demosaic_noisy_images).type(torch.cuda.FloatTensor)
            # imshow_torch_seamless(y / y.max(), print_info=True, title_str='noisy_rgb')
        # one demosaic with vrt maybe>
        # imshow_torch_video_seamless(noisy_images/noisy_images.max(), print_info=True)

        outputs_dict.output_frames_noisy = noisy_images
        # imshow_torch_video_seamless(noisy_images/noisy_images.max(), print_info=True)
        outputs_dict.noise_sigma_map = noise_sigma_map
        outputs_dict.noise_instance_map = noise_instance_map

        return outputs_dict

    def assign_final_outputs_after_noise(self, outputs_dict, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Transform Image To RGB/BW: ###
        if IO_dict.flag_to_RGB_after_noise:
            outputs_dict = self.BW2RGB_for_whole_dict(outputs_dict)
        if IO_dict.flag_to_BW_after_noise:
            outputs_dict = self.RGB2BW_for_whole_dict(outputs_dict)

        # ### Change Noise Maps Levels To Imitate Running-Average: ###
        # outputs_dict, output_frames_noisy_running_average = self.get_approximate_running_average(
        #     outputs_dict.output_frames_original,
        #     outputs_dict.output_frames_noisy,
        #     outputs_dict,
        #     outputs_dict.noise_map_to_add_HR)  #noise_map_to_add_HR
        ############################################################################

        ############################################################################
        ### Get Center Frames: ###
        outputs_dict.center_frame_noisy = outputs_dict.output_frames_noisy[IO_dict.number_of_image_frames_to_generate // 2, :, :,:]
        # outputs_dict.center_frame_pseudo_running_mean = outputs_dict.center_frame_original + self.final_noise_sigma_running_average * torch.randn(*outputs_dict.center_frame_original.shape)
        outputs_dict.center_frame_actual_mean = outputs_dict.output_frames_noisy.mean(0)
        ############################################################################

        ############################################################################
        if IO_dict.flag_how_to_concat == 'C':
            ### Turn T-stacking (which is easier to handle) to C-stacking (which is what the model and the rest of it expect): ###
            outputs_dict = self.transfer_Tstack_to_Cstack(outputs_dict)
        ############################################################################

        return outputs_dict




from RapidBase.Utils.MISCELENEOUS import assign_attributes_from_dict
class Dataset_Images_From_Folder(Dataset_MultipleImages):
    def __init__(self, root_path, IO_dict):
        super().__init__(IO_dict=IO_dict)

        ### Assign class variables: ###
        # if IO_dict.flag_crop_mode == 'random':
        #     IO_dict.flag_crop_mode = 'uniform'
        self.IO_dict = IO_dict
        assign_attributes_from_dict(self, IO_dict)

        ### Get filenames and images: ###
        self.root_path = root_path
        self.IO_dict.root_path = root_path
        self.image_filenames_list, self.images_list = self.get_filenames_and_images_from_folder(IO_dict)

    def __getitem__(self, index):
        return self.get_single_image_from_folder(self.image_filenames_list, self.images_list, index)

    def __len__(self):
        return len(self.image_filenames_list)






### Get torch dataset object more appropriate for loading numpy arrays and implementing geometric distortions + center cropping: ###
class DataSet_FromNumpyOrList(data.Dataset):
    def __init__(self, numpy_array, transform=None, crop_size=100,
                 flag_normalize_by_255=False, flag_crop_mode='center',
                 flag_make_tensor_before_batch_transform=False,
                 flag_base_tranform=False, flag_turbulence_transform=False, Cn2=5e-13):

        ### Assign class variables: ###
        self.images_list = numpy_array
        self.transform = transform
        self.crop_size = crop_size
        self.flag_make_tensor_before_batch_transform = flag_make_tensor_before_batch_transform
        self.flag_turbulence_transform = flag_turbulence_transform
        self.Cn2 = Cn2
        self.flag_base_tranform = flag_base_tranform
        self.flag_normalize_by_255 = flag_normalize_by_255
        if flag_crop_mode == 'random':
            flag_crop_mode = 'uniform'
        self.flag_crop_mode = flag_crop_mode

    def __getitem__(self, index):
        current_image = self.images_list[index]

        ### If there is a transform apply it to the SINGLE IMAGE (whether it be from imgaug, albumentations or torchvision): ###
        if self.flag_base_transform:
            if self.transform is not None:
                if type(self.transform) == iaa.Sequential:
                    deterministic_Augmenter_Object = self.transform.to_deterministic()
                    current_image = deterministic_Augmenter_Object.augment_image(current_image);
                elif type(self.transform) == albumentations.Compose:
                    augmented_images_dictionary = self.transform(image=current_image);
                    current_image = augmented_images_dictionary['image']
                elif type(self.transform) == torchvision.transforms.transforms.Compose:
                    current_image = self.transform(current_image)

        # (*). Now that i've finished the base transform part (which could require PIL.Image object) now i need this in numpy format: (*)#
        ### Normalize To 1 and convert to float32: ###
        if self.loader == ImageLoaderPIL:
            current_image = np.array(current_image)
        if self.flag_normalize_by_255:
            current_image = current_image / 255
        current_image = np.float32(current_image)

        ### Add Turbulence Deformation: ###
        if self.flag_turbulence_transform:
            current_image = turbulence_deformation_single_image(current_image, Cn2=self.Cn2)

        ### Crop Center: ###
        current_image = self.cropping_transform.augment_image(current_image)
        ######################################################################################

        ### Append Color Conversions If So Wanted: ###
        if self.flag_output_channel_average:
            current_image_intensity = np.sum(current_image, axis=2, keepdims=True)
            current_image = np.concatenate((current_image, current_image_intensity), axis=2)

        ### To pytorch convention: [H,W,C]->[C,H,W] ###
        current_image = np.ascontiguousarray(np.transpose(current_image, [2, 0, 1]))

        if self.flag_make_tensor_before_batch_transform:
            current_image = torch.Tensor(current_image)

    def __len__(self):
        return len(self.image_filenames_list)



### Get torch dataset of "videos". each "video" is a sub folder within the root folder: ##
class DataSet_Videos_In_Folders(Dataset_MultipleImages):
    def __init__(self,
                 root_folder,
                 IO_dict):
        ### Note: ###
        #as of now i only use 'CV' as loader and disregard 'PIL'
        #as of now i read all images in a specific sub folder at once and using a transform should be thought about before rushed
        #options for transforms: (1). when loading each image pass the transform to the reader and transform it there
        #                        (2). load images concatenated along 'C' dimension and make sure my transforms still work as wanted
        #                        (3). load images concatenated along 'B'/'T' dimension and make, transform on batches, and reshape to concat along 'C' if wanted


        ### Assign Object Variables: ###
        super().__init__(root_folder, IO_dict)
        self.root_folder = root_folder
        # self.IO_dict = IO_dict
        # assign_attributes_from_dict(self, IO_dict)

        if os.path.isdir(root_folder):
            ### Get Videos In Folders FileNames (& Images if flag_use_RAM is True): ###
            self.image_filenames_list_of_lists, self.images_folders, self.images_list = \
                self.get_videos_in_folders_filenames_and_images(root_folder, self.IO_dict)

            ### Delete First Element Because It's Garbage: ###
            if self.image_filenames_list_of_lists[0] == []:
                del self.image_filenames_list_of_lists[0]
                del self.images_folders[0]
                if self.IO_dict.flag_to_RAM:
                    del self.images_list[0]

            # del image_filenames_list_of_lists[0]
            # del image_folders_list[0]
            # if flag_to_RAM:
            #     del images[0]


    def get_images_in_folders_filenames_and_images(self,
                                                   root_folder,
                                                   IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        if IO_dict.flag_to_RAM:
            numpy_images_list, image_filenames_list = read_images_and_filenames_from_folder(
                root_folder,
                IO_dict.flag_recursive,
                IO_dict.crop_size,  # Train_dict.initial_crop_size
                np.inf,
                IO_dict.allowed_extentions,
                flag_return_numpy_or_list='list',
                flag_how_to_concat='T',
                crop_style='center',
                flag_to_BW=IO_dict.flag_to_BW,
                string_pattern_to_search=IO_dict.string_pattern_to_search)

        else:
            image_filenames_list = get_image_filenames_from_folder(root_folder,
                                                                   np.inf,
                                                                   IO_dict.allowed_extentions,
                                                                   IO_dict.flag_recursive,
                                                                   IO_dict.string_pattern_to_search)
            numpy_images_list = None

        return image_filenames_list, numpy_images_list

    def get_videos_in_folders_filenames_and_images(self,
                                                   root_folder,
                                                   IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Initialize lists: ###
        image_filenames_list_of_lists = []
        image_folders_list = []
        images = []

        ### Loop Over Sub Folders And Read: ###
        folder_counter = 0
        # TODO: add possibility of recursive/non-recursive search - when i'm using a for loop i can't have a real signal for "leaf-folder" accept perhapse "are there any more folders beneath you?"
        for directory_path, directory_name, file_names_in_sub_directory in os.walk(root_folder):
            # If number of videos/directories so far exceeds maximum then break:
            if folder_counter > IO_dict.max_number_of_videos and IO_dict.max_number_of_videos != -1:
                break

            # Add new element to lists, representing a sub directory
            current_folder_filenames_list = read_image_filenames_from_folder(directory_path,
                                                                             number_of_images=IO_dict.number_of_images_per_video_to_scan,
                                                                             allowed_extentions=IO_dict.allowed_extentions,
                                                                             flag_recursive=False,
                                                                             string_pattern_to_search=IO_dict.string_pattern_to_search)
            if len(current_folder_filenames_list)>=1 and (len(current_folder_filenames_list) >= IO_dict.number_of_images_per_video_to_load or IO_dict.number_of_images_per_video_to_load==np.inf):
                image_folders_list.append(directory_path)
                image_filenames_list_of_lists.append(current_folder_filenames_list)

                if IO_dict.flag_to_RAM:
                    # Load Numpy Array Concatanted Along Proper Dim To RAM
                    images.append(read_images_from_folder(directory_path, False,
                                                          np.inf,
                                                          # don't crop right now, crop when actually outputing frames
                                                          IO_dict.number_of_images_per_video_to_scan,
                                                          # Read all images from a possibly long video, when loading only load wanted number of images randomly
                                                          IO_dict.allowed_extentions,
                                                          flag_return_numpy_or_list='numpy',
                                                          flag_how_to_concat='T',
                                                          crop_style='random',
                                                          flag_return_torch=False,
                                                          transform=IO_dict.transforms_dict.base_transform,
                                                          flag_to_BW=IO_dict.flag_to_BW_before_noise,
                                                          string_pattern_to_search=IO_dict.string_pattern_to_search))
            ### Uptick folder counter: ###
            folder_counter += 1

        return image_filenames_list_of_lists, image_folders_list, images

    def multiply_and_stack_single_image(self, current_image, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        if type(current_image) == torch.Tensor:
            output_frames = torch.cat([current_image.unsqueeze(0)] * IO_dict.number_of_image_frames_to_generate, axis=0)  # [T,C,H,W]
        else:
            output_frames = np.concatenate([np.expand_dims(current_image, 0)] * IO_dict.number_of_image_frames_to_generate, axis=0) #[T,H,W,C]

        return output_frames

    def get_current_sub_folder_video_frames(self, image_filenames_list_of_lists, images_list, dataset_index, frame_start_index=-1, start_H=-1, start_W=-1, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict
            flag_return_dict = False
        else:
            flag_return_dict = True

        #TODO: this is still too complicated, fix this and make the flags more simple and clear
        self.IO_dict.convert_color_on_the_fly = ((IO_dict.flag_to_BW_before_noise or IO_dict.flag_to_RGB_before_noise) and IO_dict.flag_to_RAM is False)
        self.convert_color_on_the_fly = self.IO_dict.convert_color_on_the_fly

        ### Get start and stop indices: ###
        if frame_start_index > -1:
            #(*). Got first index from outside, use it
            stop_index = frame_start_index + IO_dict.number_of_images_per_video_to_load
        else:
            #(*). Random first index
            frame_start_index, stop_index = get_random_start_stop_indices_for_crop(IO_dict.number_of_images_per_video_to_load, len(image_filenames_list_of_lists[dataset_index]))

        if IO_dict.flag_to_RAM:
            #TODO: if i'm loading to memory and i only need BW, for instance, only save BW instead of converting on-the-fly...etc'
            current_folder_images_numpy = images_list[dataset_index][:, :, frame_start_index:stop_index]  #we're getting numpy here
        else:
            specific_subfolder_filenames_list = image_filenames_list_of_lists[dataset_index]
            current_folder_images_numpy, start_index, start_H, start_W = \
                read_images_from_filenames_list(specific_subfolder_filenames_list,
                                                flag_return_numpy_or_list='numpy',
                                                crop_size=np.inf,  #TODO: before this was IO_dict.initial_crop_size which made things messy...i change to np.inf because instead of messy code simply load the entire images and then crop the BATCH!!! instead of individual images
                                                max_number_of_images=IO_dict.number_of_images_per_video_to_load,
                                                allowed_extentions=IMG_EXTENSIONS,
                                                flag_how_to_concat='T',
                                                crop_style=IO_dict.flag_crop_mode,
                                                start_H=start_H,
                                                start_W=start_W,
                                                flag_return_torch=False,
                                                transform=IO_dict.transforms_dict.base_transform,
                                                flag_random_first_frame=False,
                                                first_frame_index=frame_start_index,
                                                flag_return_position_indices=True,
                                                flag_to_BW=IO_dict.flag_to_BW_before_noise * self.convert_color_on_the_fly,  #TODO: take care of all this shit!!!
                                                flag_to_RGB=IO_dict.flag_to_RGB_before_noise * self.convert_color_on_the_fly)

        if type(current_folder_images_numpy) == np.array or type(current_folder_images_numpy) == np.ndarray:
            current_folder_images_numpy = current_folder_images_numpy.astype(np.float32)
        else:
            current_folder_images_numpy = current_folder_images_numpy.type(torch.float32)

        if flag_return_dict:
            IO_dict.start_index = start_index
            return current_folder_images_numpy, IO_dict
        else:
            return current_folder_images_numpy


    def base_get_current_sub_folder_video_batch(self, image_filenames_list_of_lists, images_list, index, start_index=-1, start_H=-1, start_W=-1, IO_dict=None):
        if IO_dict is None:
            IO_dict = self.IO_dict

        ### Read Images: ###
        current_folder_images_numpy, IO_dict = self.get_current_sub_folder_video_frames(image_filenames_list_of_lists, images_list, index, start_index, start_H=-1, start_W=-1, IO_dict=IO_dict)

        ### Crop All Images Consistently: ###
        # output_frames_original = crop_torch_or_numpy_batch(current_folder_images_numpy, IO_dict.initial_crop_size, crop_style=IO_dict.flag_crop_mode)
        if IO_dict.flag_crop_mode == 'random':
            start_H, stop_H = get_random_start_stop_indices_for_crop(min(IO_dict.initial_crop_size[0], current_folder_images_numpy.shape[-3]), current_folder_images_numpy.shape[-3])
            start_W, stop_W = get_random_start_stop_indices_for_crop(min(IO_dict.initial_crop_size[1], current_folder_images_numpy.shape[-2]), current_folder_images_numpy.shape[-2])
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

    def __getitem__(self, index):
        outputs_dict = self.base_get_current_sub_folder_video_batch(index)
        return outputs_dict

    def __len__(self):
        return len(self.image_filenames_list_of_lists)







### Get torch dataset from a folder with binary files to be read! - TODO: complete this, there are things to do: ###
class ImageFolderRecursive_MaxNumberOfImages_BinaryFiles(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""

    def __init__(self, root, transform=None, max_number_of_images=-1, min_crop_size=100, image_data_type_string='uint8'):
        #TODO: why isn't the inherited class' (data.Dataset) __init__ function being activated?
        images = []

        counter = 0;
        # root = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet'
        # max_number_of_images = 100
        for dp, dn, fn in os.walk(root):
            if counter > max_number_of_images and max_number_of_images != -1:
                break;
            for f in fn:
                filename = os.path.join(dp, f);
                counter += 1;
                if counter > max_number_of_images and max_number_of_images != -1:
                    break;
                else:
                    #TODO: add condition that file ends with .bin!!!!
                    images.append('{}'.format(filename))
        # shuffle(filename_list)

        self.root = root
        self.imgs = images
        self.transform = transform
        self.loader = loader
        self.min_crop_size = min_crop_size;
        self.image_data_type_string = image_data_type_string


    def __getitem__(self, index):
        filename = self.imgs[index]

        try:
            fid_Read_X = open(filename, 'rb')
            frame_height = np.fromfile(fid_Read_X, 'float32', count=1);
            frame_width = np.fromfile(fid_Read_X, 'float32', count=1);
            number_of_channels = np.fromfile(fid_Read_X, 'float32', count=1);
            mat_shape = [frame_height, frame_width, number_of_channels]  # list or tuple?!?!?!?!!?
            total_images_number_of_elements = frame_height*frame_width*number_of_channels;
            img = np.fromfile(fid_Read_X, self.image_data_type_string, count=total_images_number_of_elements);
            img = img.reshape(mat_shape)
            s = img.size
            if min(s) < self.min_crop_size:
                raise ValueError('image smaller than a block (32,32)')
            # img = img.resize((int(s[0]/2), int(s[1]/2)), Image.ANTIALIAS)
        except:
            print("problem loading " + filename)
            img = Image.new('RGB', [self.min_crop_size, self.min_crop_size])

            # return torch.zeros((3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)



#####################################################################################################################################################
def turbulence_deformation_single_image(I, Cn2=5e-13, flag_clip=False):
    ### TODO: why the .copy() ??? ###
    # I = I.copy()

    # I = read_image_default()
    # I = crop_center(I,150,150)
    # imshow(I)

    ### Parameters: ###
    h = 100
    # Cn2 = 2e-13
    # Cn2 = 7e-17
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(I.shape[0] / PatchSize))
    PatchNumCol = int(np.round(I.shape[1] / PatchSize))
    shape = I.shape
    [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    if I.dtype == 'uint8':
        mv = 255
    else:
        mv = 1

    ### Get Random Motion Field: ###
    ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)

    ### Resize (small) Random Motion Field To Image Size: ###
    ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

    ### Add Rescaled Flow Field To Meshgrid: ###
    X = X0 + ShiftMatX
    Y = Y0 + ShiftMatY

    ### Resample According To Motion Field: ###
    I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)

    ### Clip Result: ###
    if flag_clip:
        I = np.minimum(I, mv)
        I = np.maximum(I, 0)

    # imshow(I)

    return I




def get_turbulence_flow_field(H,W, batch_size, Cn2=2e-13):
    ### TODO: why the .copy() ??? ###
    # I = I.copy()

    ### Parameters: ###
    h = 100
    # Cn2 = 2e-13
    # Cn2 = 7e-17
    wvl = 5.3e-7
    IFOV = 4e-7
    R = 1000
    VarTiltX = 3.34e-6
    VarTiltY = 3.21e-6
    k = 2 * np.pi / wvl
    r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
    PixSize = IFOV * R
    PatchSize = 2 * r0 / PixSize

    ### Get Current Image Shape And Appropriate Meshgrid: ###
    PatchNumRow = int(np.round(H / PatchSize))
    PatchNumCol = int(np.round(W / PatchSize))
    shape = I.shape
    [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
    if I.dtype == 'uint8':
        mv = 255
    else:
        mv = 1

    ### Get Random Motion Field: ###
    [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
    [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
    X_large = torch.Tensor(X_large).unsqueeze(-1)
    Y_large = torch.Tensor(Y_large).unsqueeze(-1)
    X_large = (X_large-W/2) / (W/2-1)
    Y_large = (Y_large-H/2) / (H/2-1)

    new_grid = torch.cat([X_large,Y_large],2)
    new_grid = torch.Tensor([new_grid.numpy()]*batch_size)

    ShiftMatX0 = torch.randn(batch_size, PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
    ShiftMatY0 = torch.randn(batch_size, PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)
    ShiftMatX0 = ShiftMatX0 * W
    ShiftMatY0 = ShiftMatY0 * H

    ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0.unsqueeze(1), new_grid, mode='bilinear', padding_mode='reflection')
    ShiftMatY = torch.nn.functional.grid_sample(ShiftMatX0.unsqueeze(1), new_grid, mode='bilinear', padding_mode='reflection')

    ShiftMatX = ShiftMatX.squeeze()
    ShiftMatY = ShiftMatY.squeeze()

    # ### Resize (small) Random Motion Field To Image Size: ###
    # ShiftMatX0 = F.adaptive_avg_pool2d(ShiftMatX0, torch.Size([H,W]))
    # ShiftMatY0 = F.adaptive_avg_pool2d(ShiftMatY0, torch.Size([H,W]))

    return ShiftMatX, ShiftMatY








class get_turbulence_flow_field_object:
    def __init__(self,H,W, batch_size, Cn2=2e-13):
        ### Parameters: ###
        h = 100
        # Cn2 = 2e-13
        # Cn2 = 7e-17
        wvl = 5.3e-7
        IFOV = 4e-7
        R = 1000
        VarTiltX = 3.34e-6
        VarTiltY = 3.21e-6
        k = 2 * np.pi / wvl
        r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
        PixSize = IFOV * R
        PatchSize = 2 * r0 / PixSize

        ### Get Current Image Shape And Appropriate Meshgrid: ###
        PatchNumRow = int(np.round(H / PatchSize))
        PatchNumCol = int(np.round(W / PatchSize))
        shape = I.shape
        [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
        if I.dtype == 'uint8':
            mv = 255
        else:
            mv = 1

        ### Get Random Motion Field: ###
        [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
        [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
        X_large = torch.Tensor(X_large).unsqueeze(-1)
        Y_large = torch.Tensor(Y_large).unsqueeze(-1)
        X_large = (X_large-W/2) / (W/2-1)
        Y_large = (Y_large-H/2) / (H/2-1)

        new_grid = torch.cat([X_large,Y_large],2)
        new_grid = torch.Tensor([new_grid.numpy()]*batch_size)

        self.new_grid = new_grid;
        self.batch_size = batch_size
        self.PatchNumRow = PatchNumRow
        self.PatchNumCol = PatchNumCol
        self.VarTiltX = VarTiltX
        self.VarTiltY = VarTiltY
        self.R = R
        self.PixSize = PixSize
        self.H = H
        self.W = W

    def get_flow_field(self):
        ShiftMatX0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltX * self.R / self.PixSize)
        ShiftMatY0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltY * self.R / self.PixSize)
        ShiftMatX0 = ShiftMatX0 * self.W
        ShiftMatY0 = ShiftMatY0 * self.H

        ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')
        ShiftMatY = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')

        ShiftMatX = ShiftMatX.squeeze()
        ShiftMatY = ShiftMatY.squeeze()

        return ShiftMatX, ShiftMatY


# ##################
# # Lambda: #
#
# def func_images(images, random_state, parents, hooks):
#     images[:, ::2, :, :] = 0
#     return images
#
# def func_heatmaps(heatmaps, random_state, parents, hooks):
#     for heatmaps_i in heatmaps:
#         heatmaps.arr_0to1[::2, :, :] = 0
#     return heatmaps
#
# def func_keypoints(keypoints_on_images, random_state, parents, hooks):
#     return keypoints_on_images
#
#
# aug = iaa.Lambda(func_images = func_images,
#                  func_heatmaps = func_heatmaps,
#                  func_keypoints = func_keypoints)
# ##################





















# from PIL import Image
# import torchvision
#
# #Augmentations:
# import PIL
# import albumentations
# from albumentations import *
# from albumentations.augmentations.functional import to_gray
#
#
# import os
# import torch.utils.data as data
# import torch
# import numpy as np
# from easydict import EasyDict as edict
# import cv2
# from tqdm import tqdm
# from easydict import EasyDict
#
# from RapidBase.import_all import *
# # from RapidBase.Basic_Import_Libs import *
# # from RapidBase.Utils.Path_and_Reading_utils import *
# # import RapidBase.Utils.Albumentations_utils
# # import RapidBase.Utils.IMGAUG_utils
# # from RapidBase.Utils.IMGAUG_utils import *
# # from RapidBase.Utils.Pytorch_Numpy_Utils import RGB2BW_torch
# # from RapidBase.Utils.Add_Noise import add_noise_to_images_full
# # from RapidBase.Utils.Warping_Shifting import Shift_Layer_Torch
# # from RapidBase.Utils.Path_and_Reading_utils import read_image_filenames_from_folder
#
#
# IMG_EXTENSIONS = [
#     '.jpg',
#     '.JPG',
#     '.jpeg',
#     '.JPEG',
#     '.png',
#     '.PNG',  #in ImageNet files there appears to be, at least in my computer, a problem with png and gif
#     '.ppm',
#     '.PPM',
#     '.bmp',
#     '.BMP', '.TIFF', '.tiff', '.tif'
# ]
#
#
# IMG_EXTENSIONS_PNG = [
#     '.png',
#     '.PNG',  #in ImageNet files there appears to be, at least in my computer, a problem with png and gif
# ]
#
#
#
#
# class Dataset_MultipleImages(data.Dataset):
#     def __init__(self, root_folder=None, IO_dict=None):
#         super().__init__()
#         self.IO_dict = IO_dict
#         assign_attributes_from_dict(self, IO_dict)
#
#         ### Upsampling & Downsampling Layer: ###
#         IO_dict.transforms_dict.upsample_method = 'bilinear'  # 'nearest', 'bilinear', 'bicubic', 'fft'
#         IO_dict.transforms_dict.downsample_method = 'binning'  # 'binning', 'nearest', 'bilinear', 'bicubic', 'fft'
#
#         ### Whether to do SR-SameSize: ###
#         self.flag_upsample_noisy_input_to_same_size_as_original = False  #by default
#
#         ### UpSample Method: ###
#         if IO_dict.transforms_dict.upsample_method == 'bilinear':
#             self.upsampling_layer = nn.Upsample(scale_factor=self.IO_dict.transforms_dict.downsampling_factor, mode='bilinear',
#                                                 align_corners=None)
#         elif IO_dict.transforms_dict.upsample_method == 'bicubic':
#             self.upsampling_layer = nn.Upsample(scale_factor=self.IO_dict.transforms_dict.downsampling_factor, mode='bicubic',
#                                                 align_corners=None)
#         elif IO_dict.transforms_dict.upsample_method == 'nearest':
#             self.upsampling_layer = nn.Upsample(scale_factor=self.IO_dict.transforms_dict.downsampling_factor, mode='nearest',
#                                                 align_corners=None)
#
#         ### DownSample Method: ###
#         if IO_dict.transforms_dict.upsample_method == 'bilinear':
#             self.downsampling_layer = nn.Upsample(scale_factor=1/self.IO_dict.transforms_dict.downsampling_factor, mode='bilinear',
#                                                 align_corners=None)
#         elif IO_dict.transforms_dict.upsample_method == 'bicubic':
#             self.downsampling_layer = nn.Upsample(scale_factor=1/self.IO_dict.transforms_dict.downsampling_factor, mode='bicubic',
#                                                 align_corners=None)
#         elif IO_dict.transforms_dict.upsample_method == 'nearest':
#             self.downsampling_layer = nn.Upsample(scale_factor=1/self.IO_dict.transforms_dict.downsampling_factor, mode='nearest',
#                                                 align_corners=None)
#         elif IO_dict.transforms_dict.downsample_method == 'binning':
#             self.downsampling_layer = nn.AvgPool2d(self.IO_dict.transforms_dict.downsampling_factor)
#         self.average_pooling_layer = nn.AvgPool2d(self.IO_dict.transforms_dict.downsampling_factor)
#
#
#         ### Warp/Transforms Layers: ###
#         #TODO: add affine transform layers when it's done, add general deformation (turbulence) transform layer
#         self.warp_object = Shift_Layer_Torch()
#
#     def get_random_transformation_vectors(self, outputs_dict, IO_dict=None):
#         """
#         This function creates vectors of random transformations (translation, rotation, scaling) in the ranges
#         specified in IO_dict and assigns them to outputs_dict
#         :param outputs_dict: specific data information
#         :param IO_dict: general processing information
#         :return: outputs_dict with transformation vectors
#         """
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         T, C, H, W = outputs_dict.output_frames_original.shape
#         current_device = outputs_dict.output_frames_original.device
#         transforms_dict = EasyDict()
#
#         ### Shift/Translation: ###
#         shift_size = IO_dict.transforms_dict.shift_size
#         # transforms_dict.shift_x_vec = torch.Tensor(get_random_number_in_range(-shift_size, shift_size, T)).to(current_device)
#         # transforms_dict.shift_y_vec = torch.Tensor(get_random_number_in_range(-shift_size, shift_size, T)).to(current_device)
#         transforms_dict.shift_x_vec = torch.Tensor(get_random_number_in_range(0, shift_size, T)).to(
#             current_device)
#         transforms_dict.shift_y_vec = torch.Tensor(get_random_number_in_range(0, shift_size, T)).to(
#             current_device)
#         # TODO add random thetas and scaling factors as well
#
#         ### Put all the transforms dict variables into outputs_dict: ###
#         outputs_dict.transforms_dict = transforms_dict
#         outputs_dict.update(outputs_dict.transforms_dict)
#
#         return outputs_dict
#
#     def shift_and_blur_images(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### Initial: ###
#         outputs_dict.output_frames_before_adding_noise = outputs_dict.output_frames_original
#         T,C,H,W = outputs_dict.output_frames_original.shape
#         blur_fraction = IO_dict.transforms_dict.blur_fraction
#         shift_size = IO_dict.transforms_dict.shift_size
#         directional_blur_size = shift_size * blur_fraction
#
#         ### Get total shifts: ###
#         N_blur_steps_total = max(self.IO_dict.transforms_dict.number_of_blur_steps_per_pixel * directional_blur_size, 1)
#         ### divide it into a blur step and a shift step: ###
#         #TODO: when shift layers will be completed this won't be necessary
#         shiftx = outputs_dict.transforms_dict.shift_x_vec.cpu().numpy()
#         shifty = outputs_dict.transforms_dict.shift_y_vec.cpu().numpy()
#         shiftx_blur = shiftx * blur_fraction
#         shifty_blur = shifty * blur_fraction
#         shiftx_preblur_shift = shiftx - shiftx_blur
#         shifty_preblur_shift = shifty - shifty_blur
#
#         ### Shift the image to get the "perfect" image: ###
#         #(1). Shift original image so that the blurry(noisy) image and original image are at the same place:
#         if any(shiftx_preblur_shift != 0):
#             outputs_dict.output_frames_original = self.warp_object(outputs_dict.output_frames_original, shiftx, shifty)
#         else:
#             outputs_dict.output_frames_original = outputs_dict.output_frames_original
#         #(2). Shift -> blur to get the blurred image: ###
#         outputs_dict.output_frames_before_adding_noise = self.warp_object(
#             outputs_dict.output_frames_before_adding_noise, shiftx_preblur_shift, shifty_preblur_shift)
#         #TODO: use a blur layer when it is done being written!!!
#         outputs_dict.output_frames_before_adding_noise = blur_image_motion_blur_torch(
#             outputs_dict.output_frames_before_adding_noise, shiftx_blur, shifty_blur, N_blur_steps_total, self.warp_object)
#
#         ### Assign/Initiailize HR image here: ###
#         outputs_dict.output_frames_original_HR = outputs_dict.output_frames_original.data
#
#         ### Keep track of shifts: ###
#         #(1). Correct for downsampling factor which will come later
#         outputs_dict.transforms_dict.shift_x_vec = torch.Tensor(outputs_dict.transforms_dict.shift_x_vec / IO_dict.transforms_dict.downsampling_factor)
#         outputs_dict.transforms_dict.shift_y_vec = torch.Tensor(outputs_dict.transforms_dict.shift_y_vec / IO_dict.transforms_dict.downsampling_factor)
#         L = len(outputs_dict.transforms_dict.shift_x_vec)
#         outputs_dict.transforms_dict.shift_x_to_reference_vec = outputs_dict.transforms_dict.shift_x_vec - outputs_dict.transforms_dict.shift_x_vec[L // 2]
#         outputs_dict.transforms_dict.shift_y_to_reference_vec = outputs_dict.transforms_dict.shift_y_vec - outputs_dict.transforms_dict.shift_y_vec[L // 2]
#
#         ### Produce Full Optical Flow From Shifts: ###
#         outputs_dict.optical_flow_delta_x = outputs_dict.transforms_dict.shift_x_vec.unsqueeze(-1).unsqueeze(-1).unsqueeze(
#             -1) * torch.ones_like(outputs_dict.output_frames_original)
#         outputs_dict.optical_flow_delta_y = outputs_dict.transforms_dict.shift_y_vec.unsqueeze(-1).unsqueeze(-1).unsqueeze(
#             -1) * torch.ones_like(outputs_dict.output_frames_original)
#         outputs_dict.optical_flow_GT = torch.cat((outputs_dict.optical_flow_delta_x, outputs_dict.optical_flow_delta_y), 1)  # [T=N_frames,C=2(x,y),H,W]
#         outputs_dict.optical_flow_delta_x_to_reference = outputs_dict.optical_flow_delta_x - outputs_dict.optical_flow_delta_x[T//2:T//2+1]
#         outputs_dict.optical_flow_delta_y_to_reference = outputs_dict.optical_flow_delta_y - outputs_dict.optical_flow_delta_y[T//2:T//2+1]
#         outputs_dict.optical_flow_GT = torch.cat((outputs_dict.optical_flow_delta_x_to_reference, outputs_dict.optical_flow_delta_y_to_reference), 1)
#         return outputs_dict
#
#     def downsample_before_adding_noise(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### Perform DownSampling: ###
#         if IO_dict.transforms_dict.downsampling_factor > 1:
#             output_frames_after_downsampling = self.downsampling_layer(outputs_dict.output_frames_before_adding_noise)
#         else:
#             output_frames_after_downsampling = outputs_dict.output_frames_before_adding_noise
#         outputs_dict.output_frames_before_adding_noise = output_frames_after_downsampling
#         return outputs_dict
#
#     def upsample_tensors_where_needed(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### Perform Upsampling For Deblur-SR Input To Model: ###
#         if self.IO_dict.transforms_dict.flag_upsample_noisy_input_to_same_size_as_original:
#             outputs_dict.output_frames_noisy = self.upsampling_layer(outputs_dict.output_frames_noisy)
#
#         ### Upsample the noise map when doing Super-Resolution (the noise map is added to the input low resolution): ###
#         if self.IO_dict.transforms_dict.downsampling_factor > 1:
#             noise_instance_map_HR = self.upsampling_layer(outputs_dict.noise_instance_map)
#             noise_sigma_map_HR = self.upsampling_layer(outputs_dict.noise_sigma_map)
#
#             T_noisy, C_noisy, H_noisy, W_noisy = outputs_dict.output_frames_noisy.shape
#             T_original, C_original, H_original, W_original = outputs_dict.output_frames_original.shape
#
#             ### this is to support SR & SR_Deblur: ###
#             if H_original > H_noisy:
#                 outputs_dict.output_frames_noisy_HR = self.upsampling_layer(outputs_dict.output_frames_noisy)
#             else:
#                 outputs_dict.output_frames_noisy_HR = outputs_dict.output_frames_noisy  # Add upsampled noise to final HR frames
#
#         else:  # self.IO_dict.transforms_dict.downsampling_factor == 1:
#             outputs_dict.output_frames_noisy_HR = outputs_dict.output_frames_noisy
#             noise_instance_map_HR = outputs_dict.noise_instance_map
#             noise_sigma_map_HR = outputs_dict.noise_sigma_map
#
#         ### Assign Center Frame Of HR Images: ###
#         outputs_dict.center_frame_noisy_HR = outputs_dict.output_frames_noisy_HR[
#                                              IO_dict.number_of_image_frames_to_generate // 2:IO_dict.number_of_image_frames_to_generate // 2 + 1,
#                                              :, :]
#
#         ### Assign the HR noise maps to outputs_dict: ###
#         outputs_dict.noise_sigma_map_HR = noise_sigma_map_HR
#         outputs_dict.noise_instance_map_HR = noise_instance_map_HR
#
#         return outputs_dict
#
#     def get_filenames_and_images_from_folder(self, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         if IO_dict.flag_to_RAM:
#             numpy_images_list, image_filenames_list = read_images_and_filenames_from_folder(IO_dict.root_path,
#                                                                                             IO_dict.flag_recursive,
#                                                                                             IO_dict.initial_crop_size,
#                                                                                             IO_dict.max_number_of_images,
#                                                                                             IO_dict.allowed_extentions,
#                                                                                             flag_return_numpy_or_list='list',
#                                                                                             flag_how_to_concat='T',
#                                                                                             crop_style=IO_dict.flag_crop_mode,
#                                                                                             flag_to_BW=IO_dict.flag_to_BW_before_noise,
#                                                                                             string_pattern_to_search=IO_dict.string_pattern_to_search)
#
#         else:
#             image_filenames_list = get_image_filenames_from_folder(IO_dict.root_path,
#                                                                    IO_dict.max_number_of_images,
#                                                                    IO_dict.allowed_extentions,
#                                                                    IO_dict.flag_recursive,
#                                                                    IO_dict.string_pattern_to_search)
#             numpy_images_list = None
#
#         return image_filenames_list, numpy_images_list
#
#     def get_noise_filenames_and_images(self, IO_dict=None):
#         #TODO: see if this is still necessary
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         if IO_dict.flag_noise_images_to_RAM:
#             noise_numpy_images_list, noise_image_filenames_list = read_images_and_filenames_from_folder(
#                 IO_dict.noise_images_path,
#                 flag_recursive=True,
#                 crop_size=np.inf,
#                 max_number_of_images=IO_dict.max_number_of_noise_images,
#                 allowed_extentions=IO_dict.allowed_extentions,
#                 flag_return_numpy_or_list='list',
#                 flag_how_to_concat='T',
#                 crop_style=IO_dict.flag_crop_mode,
#                 flag_to_BW=IO_dict.flag_to_BW,
#                 string_pattern_to_search=IO_dict.string_pattern_to_search)
#
#         else:
#             noise_image_filenames_list = get_image_filenames_from_folder(IO_dict.noise_images_path,
#                                                                          IO_dict.max_number_of_noise_images,
#                                                                          IO_dict.allowed_extentions,
#                                                                          IO_dict.flag_recursive,
#                                                                          IO_dict.string_pattern_to_search)
#             noise_numpy_images_list = None
#         ### Assign Noise Images & Filenames List To Internal Attributes: ###
#         return noise_image_filenames_list, noise_numpy_images_list
#
#     def specific_dataset_functionality_before_cropping(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         outputs_dict = self.shift_and_blur_images(outputs_dict, IO_dict)
#         return outputs_dict
#
#     def specific_dataset_functionality_before_adding_noise(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         outputs_dict = self.downsample_before_adding_noise(outputs_dict, IO_dict)
#         return outputs_dict
#
#     def specific_dataset_functionality_after_adding_noise(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         outputs_dict = self.upsample_tensors_where_needed(outputs_dict, IO_dict)
#         return outputs_dict
#
#     def normalize_tensor_graylevels_to_photons(self, outputs_dict, IO_dict=None, external_PPP=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         if IO_dict.flag_normalize_images_to_PPP:
#             if external_PPP is None:
#                 IO_dict.PPP = to_list_of_certain_size(IO_dict.PPP, 2)
#                 current_PPP = get_random_number_in_range(IO_dict.PPP[0], IO_dict.PPP[1])
#                 outputs_dict.PPP = current_PPP
#             else:
#                 current_PPP = external_PPP
#             output_frames_original = outputs_dict.output_frames_original
#             original_images_mean = output_frames_original.mean()
#             PPP_normalization_factor = current_PPP / (original_images_mean.abs() + 1e-6)
#             # normalized_images = output_frames_original * current_PPP / (original_images_mean.abs() + 1e-6)
#             outputs_dict.output_frames_original = output_frames_original * PPP_normalization_factor
#             outputs_dict.output_frames_before_adding_noise = outputs_dict.output_frames_before_adding_noise * PPP_normalization_factor
#             outputs_dict.output_frames_original_HR = outputs_dict.output_frames_original_HR * PPP_normalization_factor
#         # else:
#         #     outputs_dict.PPP = None
#         return outputs_dict
#
#     def add_noise_and_get_frames_final(self, outputs_dict, output_frames_original_to_add_noise_to, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         ### Assign outputs_dict (Including Adding Noise): ###
#         outputs_dict = self.get_outputs_from_original_images_additive_gaussian_noise(outputs_dict, output_frames_original_to_add_noise_to, IO_dict)
#         return outputs_dict
#
#     def assign_original_images_to_outputs_dict(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         ### Assign Original Image To Outputs_dict: ###
#         outputs_dict.center_frame_original = outputs_dict.output_frames_original[IO_dict.number_of_image_frames_to_generate // 2, :,:, :]
#
#         return outputs_dict
#
#     def perform_final_crop_on_original_images(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         outputs_dict.output_frames_original = crop_torch_or_numpy_batch(outputs_dict.output_frames_original, self.IO_dict.final_crop_size, crop_style='center')
#         outputs_dict.output_frames_before_adding_noise = crop_torch_or_numpy_batch(outputs_dict.output_frames_before_adding_noise, self.IO_dict.final_crop_size, crop_style='center')
#         return outputs_dict
#
#     def transfer_Tstack_to_Cstack(self, outputs_dict):
#         #TODO: understand whether i can ismply loop over dictionary attributes, check if they're tensors and Cstack_to_Tstack
#         outputs_dict.output_frames_original = Tstack_to_Cstack(outputs_dict.output_frames_original)
#         outputs_dict.center_frame_original = Tstack_to_Cstack(outputs_dict.center_frame_original)
#
#         if hasattr(outputs_dict, 'output_frames_before_adding_noise'):
#             outputs_dict.output_frames_before_adding_noise = Tstack_to_Cstack(outputs_dict.output_frames_before_adding_noise)
#         if hasattr(outputs_dict, 'noise_sigma_map'):
#             outputs_dict.noise_sigma_map = Tstack_to_Cstack(outputs_dict.noise_sigma_map)
#         if hasattr(outputs_dict, 'noise_instance_map'):
#             outputs_dict.noise_instance_map = Tstack_to_Cstack(outputs_dict.noise_instance_map)
#         if hasattr(outputs_dict, 'center_frame_noisy'):
#             outputs_dict.center_frame_noisy = Tstack_to_Cstack(outputs_dict.center_frame_noisy)
#         if hasattr(outputs_dict, 'output_frames_noisy'):
#             outputs_dict.output_frames_noisy = Tstack_to_Cstack(outputs_dict.output_frames_noisy)
#         if hasattr(outputs_dict, 'output_frames_noisy_HR'):
#             outputs_dict.output_frames_noisy_HR = Tstack_to_Cstack(outputs_dict.output_frames_noisy_HR)
#         if hasattr(outputs_dict, 'center_frame_noisy_HR'):
#             outputs_dict.center_frame_noisy_HR = Tstack_to_Cstack(outputs_dict.center_frame_noisy_HR)
#         if hasattr(outputs_dict, 'noise_map_to_add_HR'):
#             outputs_dict.noise_map_to_add_HR = Tstack_to_Cstack(outputs_dict.noise_map_to_add_HR)
#         if hasattr(outputs_dict, 'output_frames_original_HR'):
#             outputs_dict.output_frames_original_HR = Tstack_to_Cstack(outputs_dict.output_frames_original_HR)
#         return outputs_dict
#
#     def transfer_Cstack_to_Tstack(self, outputs_dict):
#         #TODO: understand whether i can ismply loop over dictionary attributes, check if they're tensors and Cstack_to_Tstack
#         outputs_dict.output_frames_original = Cstack_to_Tstack(outputs_dict.output_frames_original)
#         outputs_dict.center_frame_original = Cstack_to_Tstack(outputs_dict.center_frame_original)
#
#         if hasattr(outputs_dict, 'output_frames_before_adding_noise'):
#             outputs_dict.output_frames_before_adding_noise = Cstack_to_Tstack(outputs_dict.output_frames_before_adding_noise)
#         if hasattr(outputs_dict, 'noise_sigma_map'):
#             outputs_dict.noise_sigma_map = Cstack_to_Tstack(outputs_dict.noise_sigma_map)
#         if hasattr(outputs_dict, 'noise_instance_map'):
#             outputs_dict.noise_instance_map = Cstack_to_Tstack(outputs_dict.noise_instance_map)
#         if hasattr(outputs_dict, 'center_frame_noisy'):
#             outputs_dict.center_frame_noisy = Cstack_to_Tstack(outputs_dict.center_frame_noisy)
#         if hasattr(outputs_dict, 'output_frames_noisy'):
#             outputs_dict.output_frames_noisy = Cstack_to_Tstack(outputs_dict.output_frames_noisy)
#         if hasattr(outputs_dict, 'output_frames_noisy_HR'):
#             outputs_dict.output_frames_noisy_HR = Cstack_to_Tstack(outputs_dict.output_frames_noisy_HR)
#         if hasattr(outputs_dict, 'center_frame_noisy_HR'):
#             outputs_dict.center_frame_noisy_HR = Cstack_to_Tstack(outputs_dict.center_frame_noisy_HR)
#         if hasattr(outputs_dict, 'noise_map_to_add_HR'):
#             outputs_dict.noise_map_to_add_HR = Cstack_to_Tstack(outputs_dict.noise_map_to_add_HR)
#         if hasattr(outputs_dict, 'output_frames_original_HR'):
#             outputs_dict.output_frames_original_HR = Cstack_to_Tstack(outputs_dict.output_frames_original_HR)
#
#         return outputs_dict
#
#     def get_approximate_running_average(self, output_frames_original, output_frames_noisy, outputs_dict, noise_map_to_add, IO_dict=None):
#         #TODO: understand what to do with this one!!!!, i ignore it for now
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         #TODO: probably need to make sure this uses noise_map_to_add = outputs_dict.HR_noise_map or something...probably should even add any input besides outputs_dict
#         output_frames_noisy_running_average = torch.zeros_like(output_frames_original)
#         for i in np.arange(output_frames_noisy.shape[0]):
#             output_frames_noisy_running_average[i:i + 1, :, :, :] = output_frames_original[i:i + 1, :, :,:] + 1 / np.sqrt(i + 1) * noise_map_to_add[i:i + 1, :, :, :]
#         outputs_dict.output_frames_noisy_running_average = output_frames_noisy_running_average  # [T,C,H,W]
#
#         return outputs_dict, output_frames_noisy_running_average
#
#     def get_noise_sigma_map(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         #TODO: this still isn't used, understand whether i want to use sigma_map or noise instance itself
#         ### Readout Noise Sigma: ###
#         noise_sigma_map = self.readout_noise_sigma_map  #TODO: add this calculation
#
#         #### Shot Noise Sigma Map: ###
#         noise_sigma_map += outputs_dict.shot_noise_sigma_map
#
#         ### Add Noise Sigma Map To Outputs Dict: ###
#         outputs_dict.noise_sigma_map = noise_sigma_map
#
#         return outputs_dict
#
#
#     def get_item_noise_image(self, indices, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         ### Read Noise Images: ###
#         if IO_dict.flag_noise_images_to_RAM:
#             # current_noise_images = self.noise_images_list[indices]
#             current_noise_images = np.concatenate(get_sublist_from_list(self.noise_images_list, indices), -1)
#         else:
#             specific_subfolder_filenames_list = [self.noise_image_filenames_list[index] for index in indices]
#             current_noise_images = read_images_from_filenames_list(specific_subfolder_filenames_list,
#                                                                    flag_return_numpy_or_list='numpy',
#                                                                    crop_size=np.inf,
#                                                                    max_number_of_images=np.inf,
#                                                                    allowed_extentions=IO_dict.allowed_extentions,
#                                                                    flag_how_to_concat='T',
#                                                                    crop_style='center',
#                                                                    flag_return_torch=False,
#                                                                    transform=None,
#                                                                    flag_random_first_frame=False, flag_to_BW=False)
#
#         # (*). Now that i've finished the base transform part (which could require PIL.Image object) now i need this in numpy format: (*)#
#         ### Normalize To 1 and convert to float32: ###
#         if IO_dict.image_loader == ImageLoaderPIL:
#             current_noise_images = np.array(current_noise_images)
#
#         ### Normalize and Color-Convert: ###
#         current_noise_images = self.normalize_and_convert_color(current_noise_images)
#
#         ### Crop Center: ###
#         # (*). Important to be able to stack frames of different initial sizes
#         current_noise_images = crop_torch_or_numpy_batch(current_noise_images, IO_dict.initial_crop_size, IO_dict.flag_crop_mode)
#         ######################################################################################
#
#         if len(current_noise_images.shape) == 3:
#             ### To pytorch convention: [H,W,C]->[C,H,W] ###
#             current_noise_images = np.ascontiguousarray(np.transpose(current_noise_images, [2, 0, 1]))
#         elif len(current_noise_images.shape) == 4:
#             ### To pytorch convention: [T,H,W,C]->[T,C,H,W] ###
#             current_noise_images = np.ascontiguousarray(np.transpose(current_noise_images, [0, 3, 1, 2]))
#         current_noise_images = torch.Tensor(current_noise_images)
#
#         return current_noise_images
#
#     def multiply_and_stack_single_image(self, current_image, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         if type(current_image) == torch.Tensor:
#             output_frames = torch.cat([current_image.unsqueeze(0)] * IO_dict.number_of_image_frames_to_generate, axis=0)  # [T,C,H,W]
#         else:
#             output_frames = np.concatenate([np.expand_dims(current_image, 0)] * IO_dict.number_of_image_frames_to_generate, axis=0) #[T,H,W,C]
#
#         return output_frames
#
#     def get_single_image_from_folder(self, image_filenames_list, images_list, index, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         ### Load Image: ###
#         filename = image_filenames_list[index]
#         if IO_dict.flag_to_RAM:
#             current_image = images_list[index]
#         else:
#             current_image = IO_dict.image_loader(filename)
#         return current_image
#
#     def augment_and_crop_single_image(self, current_image, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### If there is a transform apply it to the SINGLE IMAGE (whether it be from imgaug, albumentations or torchvision): ###
#         if IO_dict.transforms_dict.flag_base_transform:
#             if IO_dict.transforms_dict.base_transform is not None:
#                 if type(IO_dict.transforms_dict.base_transform) == iaa.Sequential:
#                     deterministic_Augmenter_Object = IO_dict.transforms_dict.base_transform.to_deterministic()
#                     current_image = deterministic_Augmenter_Object.augment_image(current_image);
#                 elif type(IO_dict.transforms_dict.base_transform) == albumentations.Compose:
#                     augmented_images_dictionary = IO_dict.transforms_dict.base_transform(image=current_image);
#                     current_image = augmented_images_dictionary['image']
#                 elif type(IO_dict.transforms_dict.base_transform) == torchvision.transforms.transforms.Compose:
#                     current_image = IO_dict.transforms_dict.base_transform(current_image)
#                 else:  # Custrom Transform
#                     current_image = IO_dict.transforms_dict.base_transform(current_image)
#
#         # (*). Now that i've finished the base transform part (which could require PIL.Image object) now i need this in numpy format: (*)#
#         ### Normalize To 1 and convert to float32: ###
#         if IO_dict.image_loader == ImageLoaderPIL:
#             current_image = np.array(current_image)
#
#         ### Normalize and Color-Convert: ###
#         current_image = self.normalize_and_convert_color(current_image)
#
#         ### Add Turbulence Deformation: ###
#         if IO_dict.transforms_dict.flag_turbulence_transform:
#             current_image = turbulence_deformation_single_image(current_image, Cn2=IO_dict.transforms_dict.Cn2)
#
#         ### Crop Center: ###
#         # (*). Important to be able to stack frames of different initial sizes
#         current_image = crop_torch_or_numpy_batch(current_image, IO_dict.initial_crop_size, IO_dict.flag_crop_mode)
#         ######################################################################################
#
#         if IO_dict.flag_make_tensor_before_batch_transform:
#             ### To pytorch convention: [H,W,C]->[C,H,W] ###
#             current_image = np.ascontiguousarray(np.transpose(current_image, [2, 0, 1]))
#             # if current_image.dtype == 'uint8' or current_image.dtype == 'uint16':
#             #     current_image = current_image.astype(int)
#             current_image = torch.Tensor(current_image) #[:3,:,:]
#
#         return current_image
#
#     def get_multiple_images_from_single_image(self, image_filenames_list, images_list, index):
#         ### Initialize outputs_dict for the first time: ### #TODO: do this outside the function
#         outputs_dict = EasyDict()
#
#         ### Read current frame (perform base transform on the individual image if wanted), multiply it and stack it: ###
#         current_image = self.get_single_image_from_folder(image_filenames_list, images_list, index)
#         current_image = self.augment_and_crop_single_image(current_image)
#         output_frames_original = self.multiply_and_stack_single_image(current_image)
#
#         ### Perform batch_transform on the entire batch: ###
#         output_frames_original = self.transform_batch(output_frames_original)
#
#         ### Numpy To Torch: ###
#         if type(output_frames_original) is not torch.Tensor:
#             output_frames_original = numpy_to_torch(output_frames_original)
#
#         ### Assign output frames original to outputs_dict: ###
#         outputs_dict.output_frames_original = output_frames_original
#
#         return outputs_dict
#
#     def normalize_and_convert_color(self, input_frames, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         ### Normalize Image: ###
#         output_frames = input_frames
#
#         ### Stack BW frames to make "psudo-RGB" frames if wanted: ###
#         if IO_dict.flag_to_RGB_before_noise:
#             output_frames = BW2RGB(output_frames)
#
#         ### RGB -> BW: ###
#         if IO_dict.flag_to_BW_before_noise:
#             output_frames = RGB2BW(output_frames)
#
#         return output_frames
#
#     def normalize_and_convert_color_MultipleFrames(self, input_frames, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### Initialize Image: ###
#         output_frames = input_frames
#
#         ### Stack BW frames to make "psudo-RGB" frames if wanted: ###
#         if IO_dict.flag_to_RGB_before_noise:
#             output_frames = BW2RGB_MultipleFrames(output_frames, IO_dict.flag_how_to_concat)
#
#         ### RGB -> BW: ###
#         if IO_dict.flag_to_BW_before_noise:
#             output_frames = RGB2BW_MultipleFrames(output_frames, IO_dict.flag_how_to_concat)
#
#         return output_frames
#
#     def RGB2BW_for_whole_dict(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         outputs_dict.output_frames_original = RGB2BW_MultipleFrames(outputs_dict.output_frames_original, IO_dict.flag_how_to_concat)
#         outputs_dict.output_frames_noisy = RGB2BW_MultipleFrames(outputs_dict.output_frames_noisy, IO_dict.flag_how_to_concat)
#         outputs_dict.output_frames_noisy_HR = RGB2BW_MultipleFrames(outputs_dict.output_frames_noisy_HR, IO_dict.flag_how_to_concat)
#         outputs_dict.center_frame_original = RGB2BW_MultipleFrames(outputs_dict.center_frame_original, IO_dict.flag_how_to_concat)
#         return outputs_dict
#
#     def BW2RGB_for_whole_dict(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         outputs_dict.output_frames_original = BW2RGB_MultipleFrames(outputs_dict.output_frames_original, IO_dict.flag_how_to_concat)
#         outputs_dict.output_frames_noisy = BW2RGB_MultipleFrames(outputs_dict.output_frames_noisy, IO_dict.flag_how_to_concat)
#         outputs_dict.output_frames_noisy_HR = BW2RGB_MultipleFrames(outputs_dict.output_frames_noisy_HR, IO_dict.flag_how_to_concat)
#         outputs_dict.center_frame_original = BW2RGB_MultipleFrames(outputs_dict.center_frame_original, IO_dict.flag_how_to_concat)
#         return outputs_dict
#
#     def transform_batch(self, input_frames, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#         if IO_dict.transforms_dict.flag_batch_transform:
#             if IO_dict.transforms_dict.batch_transform is not None:
#                 if type(IO_dict.transforms_dict.batch_transform) == iaa.Sequential:
#                     deterministic_Augmenter_Object = IO_dict.transform.to_deterministic()
#                     output_frames = deterministic_Augmenter_Object.augment_image(input_frames)
#                 elif type(IO_dict.transforms_dict.batch_transform) == albumentations.Compose:
#                     augmented_images_dictionary = IO_dict.transforms_dict.transform(image=input_frames)
#                     output_frames = augmented_images_dictionary['image']
#                 elif type(IO_dict.transforms_dict.batch_transform) == torchvision.transforms.transforms.Compose:
#                     output_frames = IO_dict.transforms_dict.transform(input_frames)
#                 else:
#                     output_frames = IO_dict.transforms_dict.transform(input_frames)
#         else:
#             output_frames = input_frames
#         return output_frames
#

#     def assign_final_outputs_after_noise(self, outputs_dict, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### Transform Image To RGB/BW: ###
#         if IO_dict.flag_to_RGB_after_noise:
#             outputs_dict = self.BW2RGB_for_whole_dict(outputs_dict)
#         if IO_dict.flag_to_BW_after_noise:
#             outputs_dict = self.RGB2BW_for_whole_dict(outputs_dict)
#
#         # ### Change Noise Maps Levels To Imitate Running-Average: ###
#         # outputs_dict, output_frames_noisy_running_average = self.get_approximate_running_average(
#         #     outputs_dict.output_frames_original,
#         #     outputs_dict.output_frames_noisy,
#         #     outputs_dict,
#         #     outputs_dict.noise_map_to_add_HR)  #noise_map_to_add_HR
#         ############################################################################
#
#         ############################################################################
#         ### Get Center Frames: ###
#         outputs_dict.center_frame_noisy = outputs_dict.output_frames_noisy[IO_dict.number_of_image_frames_to_generate // 2, :, :,:]
#         # outputs_dict.center_frame_pseudo_running_mean = outputs_dict.center_frame_original + self.final_noise_sigma_running_average * torch.randn(*outputs_dict.center_frame_original.shape)
#         outputs_dict.center_frame_actual_mean = outputs_dict.output_frames_noisy.mean(0)
#         ############################################################################
#
#         ############################################################################
#         if IO_dict.flag_how_to_concat == 'C':
#             ### Turn T-stacking (which is easier to handle) to C-stacking (which is what the model and the rest of it expect): ###
#             outputs_dict = self.transfer_Tstack_to_Cstack(outputs_dict)
#         ############################################################################
#
#         return outputs_dict
#
#
#
#
# from RapidBase.Utils.MISCELENEOUS import assign_attributes_from_dict
# class Dataset_Images_From_Folder(Dataset_MultipleImages):
#     def __init__(self, root_path, IO_dict):
#         super().__init__(IO_dict=IO_dict)
#
#         ### Assign class variables: ###
#         # if IO_dict.flag_crop_mode == 'random':
#         #     IO_dict.flag_crop_mode = 'uniform'
#         self.IO_dict = IO_dict
#         assign_attributes_from_dict(self, IO_dict)
#
#         ### Get filenames and images: ###
#         self.root_path = root_path
#         self.IO_dict.root_path = root_path
#         self.image_filenames_list, self.images_list = self.get_filenames_and_images_from_folder(IO_dict)
#
#     def __getitem__(self, index):
#         return self.get_single_image_from_folder(self.image_filenames_list, self.images_list, index)
#
#     def __len__(self):
#         return len(self.image_filenames_list)
#
#
#
#
#
#
# ### Get torch dataset object more appropriate for loading numpy arrays and implementing geometric distortions + center cropping: ###
# class DataSet_FromNumpyOrList(data.Dataset):
#     def __init__(self, numpy_array, transform=None, crop_size=100,
#                  flag_normalize_by_255=False, flag_crop_mode='center',
#                  flag_make_tensor_before_batch_transform=False,
#                  flag_base_tranform=False, flag_turbulence_transform=False, Cn2=5e-13):
#
#         ### Assign class variables: ###
#         self.images_list = numpy_array
#         self.transform = transform
#         self.crop_size = crop_size
#         self.flag_make_tensor_before_batch_transform = flag_make_tensor_before_batch_transform
#         self.flag_turbulence_transform = flag_turbulence_transform
#         self.Cn2 = Cn2
#         self.flag_base_tranform = flag_base_tranform
#         self.flag_normalize_by_255 = flag_normalize_by_255
#         if flag_crop_mode == 'random':
#             flag_crop_mode = 'uniform'
#         self.flag_crop_mode = flag_crop_mode
#
#     def __getitem__(self, index):
#         current_image = self.images_list[index]
#
#         ### If there is a transform apply it to the SINGLE IMAGE (whether it be from imgaug, albumentations or torchvision): ###
#         if self.flag_base_transform:
#             if self.transform is not None:
#                 if type(self.transform) == iaa.Sequential:
#                     deterministic_Augmenter_Object = self.transform.to_deterministic()
#                     current_image = deterministic_Augmenter_Object.augment_image(current_image);
#                 elif type(self.transform) == albumentations.Compose:
#                     augmented_images_dictionary = self.transform(image=current_image);
#                     current_image = augmented_images_dictionary['image']
#                 elif type(self.transform) == torchvision.transforms.transforms.Compose:
#                     current_image = self.transform(current_image)
#
#         # (*). Now that i've finished the base transform part (which could require PIL.Image object) now i need this in numpy format: (*)#
#         ### Normalize To 1 and convert to float32: ###
#         if self.loader == ImageLoaderPIL:
#             current_image = np.array(current_image)
#         if self.flag_normalize_by_255:
#             current_image = current_image / 255
#         current_image = np.float32(current_image)
#
#         ### Add Turbulence Deformation: ###
#         if self.flag_turbulence_transform:
#             current_image = turbulence_deformation_single_image(current_image, Cn2=self.Cn2)
#
#         ### Crop Center: ###
#         current_image = self.cropping_transform.augment_image(current_image)
#         ######################################################################################
#
#         ### Append Color Conversions If So Wanted: ###
#         if self.flag_output_channel_average:
#             current_image_intensity = np.sum(current_image, axis=2, keepdims=True)
#             current_image = np.concatenate((current_image, current_image_intensity), axis=2)
#
#         ### To pytorch convention: [H,W,C]->[C,H,W] ###
#         current_image = np.ascontiguousarray(np.transpose(current_image, [2, 0, 1]))
#
#         if self.flag_make_tensor_before_batch_transform:
#             current_image = torch.Tensor(current_image)
#
#     def __len__(self):
#         return len(self.image_filenames_list)
#
#
#
# ### Get torch dataset of "videos". each "video" is a sub folder within the root folder: ##
# class DataSet_Videos_In_Folders(Dataset_MultipleImages):
#     def __init__(self,
#                  root_folder,
#                  IO_dict):
#         ### Note: ###
#         #as of now i only use 'CV' as loader and disregard 'PIL'
#         #as of now i read all images in a specific sub folder at once and using a transform should be thought about before rushed
#         #options for transforms: (1). when loading each image pass the transform to the reader and transform it there
#         #                        (2). load images concatenated along 'C' dimension and make sure my transforms still work as wanted
#         #                        (3). load images concatenated along 'B'/'T' dimension and make, transform on batches, and reshape to concat along 'C' if wanted
#
#
#         ### Assign Object Variables: ###
#         super().__init__(root_folder, IO_dict)
#         self.root_folder = root_folder
#         # self.IO_dict = IO_dict
#         # assign_attributes_from_dict(self, IO_dict)
#
#         if os.path.isdir(root_folder):
#             ### Get Videos In Folders FileNames (& Images if flag_use_RAM is True): ###
#             self.image_filenames_list_of_lists, self.images_folders, self.images_list = \
#                 self.get_videos_in_folders_filenames_and_images(root_folder, self.IO_dict)
#
#             ### Delete First Element Because It's Garbage: ###
#             if self.image_filenames_list_of_lists[0] == []:
#                 del self.image_filenames_list_of_lists[0]
#                 del self.images_folders[0]
#                 if self.IO_dict.flag_to_RAM:
#                     del self.images_list[0]
#
#             # del image_filenames_list_of_lists[0]
#             # del image_folders_list[0]
#             # if flag_to_RAM:
#             #     del images[0]
#
#
#     def get_images_in_folders_filenames_and_images(self,
#                                                    root_folder,
#                                                    IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         if IO_dict.flag_to_RAM:
#             numpy_images_list, image_filenames_list = read_images_and_filenames_from_folder(
#                 root_folder,
#                 IO_dict.flag_recursive,
#                 IO_dict.crop_size,  # Train_dict.initial_crop_size
#                 np.inf,
#                 IO_dict.allowed_extentions,
#                 flag_return_numpy_or_list='list',
#                 flag_how_to_concat='T',
#                 crop_style='center',
#                 flag_to_BW=IO_dict.flag_to_BW,
#                 string_pattern_to_search=IO_dict.string_pattern_to_search)
#
#         else:
#             image_filenames_list = get_image_filenames_from_folder(root_folder,
#                                                                    np.inf,
#                                                                    IO_dict.allowed_extentions,
#                                                                    IO_dict.flag_recursive,
#                                                                    IO_dict.string_pattern_to_search)
#             numpy_images_list = None
#
#         return image_filenames_list, numpy_images_list
#
#     def get_videos_in_folders_filenames_and_images(self,
#                                                    root_folder,
#                                                    IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### Initialize lists: ###
#         image_filenames_list_of_lists = []
#         image_folders_list = []
#         images = []
#
#         ### Loop Over Sub Folders And Read: ###
#         folder_counter = 0
#         # TODO: add possibility of recursive/non-recursive search - when i'm using a for loop i can't have a real signal for "leaf-folder" accept perhapse "are there any more folders beneath you?"
#         for directory_path, directory_name, file_names_in_sub_directory in os.walk(root_folder):
#             # If number of videos/directories so far exceeds maximum then break:
#             if folder_counter > IO_dict.max_number_of_videos and IO_dict.max_number_of_videos != -1:
#                 break
#
#             # Add new element to lists, representing a sub directory
#             current_folder_filenames_list = read_image_filenames_from_folder(directory_path,
#                                                                              number_of_images=IO_dict.number_of_images_per_video_to_scan,
#                                                                              allowed_extentions=IO_dict.allowed_extentions,
#                                                                              flag_recursive=False,
#                                                                              string_pattern_to_search=IO_dict.string_pattern_to_search)
#             if len(current_folder_filenames_list)>=1 and (len(current_folder_filenames_list) >= IO_dict.number_of_images_per_video_to_load or IO_dict.number_of_images_per_video_to_load==np.inf):
#                 image_folders_list.append(directory_path)
#                 image_filenames_list_of_lists.append(current_folder_filenames_list)
#
#                 if IO_dict.flag_to_RAM:
#                     # Load Numpy Array Concatanted Along Proper Dim To RAM
#                     images.append(read_images_from_folder(directory_path, False,
#                                                           np.inf,
#                                                           # don't crop right now, crop when actually outputing frames
#                                                           IO_dict.number_of_images_per_video_to_scan,
#                                                           # Read all images from a possibly long video, when loading only load wanted number of images randomly
#                                                           IO_dict.allowed_extentions,
#                                                           flag_return_numpy_or_list='numpy',
#                                                           flag_how_to_concat='T',
#                                                           crop_style='random',
#                                                           flag_return_torch=False,
#                                                           transform=IO_dict.transforms_dict.base_transform,
#                                                           flag_to_BW=IO_dict.flag_to_BW_before_noise,
#                                                           string_pattern_to_search=IO_dict.string_pattern_to_search))
#             ### Uptick folder counter: ###
#             folder_counter += 1
#
#         return image_filenames_list_of_lists, image_folders_list, images
#
#     def get_current_sub_folder_video_frames(self, image_filenames_list_of_lists, images_list, dataset_index, frame_start_index=-1, start_H=-1, start_W=-1, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#             flag_return_dict = False
#         else:
#             flag_return_dict = True
#
#         #TODO: this is still too complicated, fix this and make the flags more simple and clear
#         self.IO_dict.convert_color_on_the_fly = ((IO_dict.flag_to_BW_before_noise or IO_dict.flag_to_RGB_before_noise) and IO_dict.flag_to_RAM is False)
#         self.convert_color_on_the_fly = self.IO_dict.convert_color_on_the_fly
#
#         ### Get start and stop indices: ###
#         if frame_start_index > -1:
#             #(*). Got first index from outside, use it
#             stop_index = frame_start_index + IO_dict.number_of_images_per_video_to_load
#         else:
#             #(*). Random first index
#             frame_start_index, stop_index = get_random_start_stop_indices_for_crop(IO_dict.number_of_images_per_video_to_load, len(image_filenames_list_of_lists[dataset_index]))
#
#         if IO_dict.flag_to_RAM:
#             #TODO: if i'm loading to memory and i only need BW, for instance, only save BW instead of converting on-the-fly...etc'
#             current_folder_images_numpy = images_list[dataset_index][:, :, frame_start_index:stop_index]  #we're getting numpy here
#         else:
#             specific_subfolder_filenames_list = image_filenames_list_of_lists[dataset_index]
#             current_folder_images_numpy, start_index, start_H, start_W = \
#                 read_images_from_filenames_list(specific_subfolder_filenames_list,
#                                                 flag_return_numpy_or_list='numpy',
#                                                 crop_size=np.inf,  #TODO: before this was IO_dict.initial_crop_size which made things messy...i change to np.inf because instead of messy code simply load the entire images and then crop the BATCH!!! instead of individual images
#                                                 max_number_of_images=IO_dict.number_of_images_per_video_to_load,
#                                                 allowed_extentions=IMG_EXTENSIONS,
#                                                 flag_how_to_concat='T',
#                                                 crop_style=IO_dict.flag_crop_mode,
#                                                 start_H=start_H,
#                                                 start_W=start_W,
#                                                 flag_return_torch=False,
#                                                 transform=IO_dict.transforms_dict.base_transform,
#                                                 flag_random_first_frame=False,
#                                                 first_frame_index=frame_start_index,
#                                                 flag_return_position_indices=True,
#                                                 flag_to_BW=IO_dict.flag_to_BW_before_noise * self.convert_color_on_the_fly,  #TODO: take care of all this shit!!!
#                                                 flag_to_RGB=IO_dict.flag_to_RGB_before_noise * self.convert_color_on_the_fly)
#
#         if type(current_folder_images_numpy) == np.array or type(current_folder_images_numpy) == np.ndarray:
#             current_folder_images_numpy = current_folder_images_numpy.astype(np.float32)
#         else:
#             current_folder_images_numpy = current_folder_images_numpy.type(torch.float32)
#
#         if flag_return_dict:
#             IO_dict.start_index = start_index
#             return current_folder_images_numpy, IO_dict
#         else:
#             return current_folder_images_numpy
#
#
#     def base_get_current_sub_folder_video_batch(self, image_filenames_list_of_lists, images_list, index, start_index=-1, start_H=-1, start_W=-1, IO_dict=None):
#         if IO_dict is None:
#             IO_dict = self.IO_dict
#
#         ### Read Images: ###
#         current_folder_images_numpy, IO_dict = self.get_current_sub_folder_video_frames(image_filenames_list_of_lists, images_list, index, start_index, start_H=-1, start_W=-1, IO_dict=IO_dict)
#
#         ### Crop All Images Consistently: ###
#         # output_frames_original = crop_torch_or_numpy_batch(current_folder_images_numpy, IO_dict.initial_crop_size, crop_style=IO_dict.flag_crop_mode)
#         if IO_dict.flag_crop_mode == 'random':
#             start_H, stop_H = get_random_start_stop_indices_for_crop(min(IO_dict.initial_crop_size[0], current_folder_images_numpy.shape[-3]), current_folder_images_numpy.shape[-3])
#             start_W, stop_W = get_random_start_stop_indices_for_crop(min(IO_dict.initial_crop_size[1], current_folder_images_numpy.shape[-2]), current_folder_images_numpy.shape[-2])
#             IO_dict.start_H = start_H
#             IO_dict.start_W = start_W
#         output_frames_original = crop_torch_or_numpy_batch(current_folder_images_numpy, crop_size_tuple_or_scalar=IO_dict.initial_crop_size,
#                                             crop_style='predetermined', start_H=IO_dict.start_H, start_W=IO_dict.start_W)   # crop_style = 'random', 'predetermined', 'center'
#
#         ### Augment Returned Image Frames: ###
#         output_frames_original = self.transform_batch(output_frames_original)
#
#         ### Numpy To Torch: ###
#         if type(output_frames_original) is not torch.Tensor:
#             output_frames_original = numpy_to_torch(output_frames_original)
#
#         ### Normalize and Color-Convert: ###
#         output_frames_original = self.normalize_and_convert_color_MultipleFrames(output_frames_original)
#
#         ### Final Crop: ###
#         output_frames_original = crop_torch_or_numpy_batch(output_frames_original, self.IO_dict.final_crop_size, crop_style='center')
#
#         outputs_dict = EasyDict()
#         outputs_dict.output_frames_original = output_frames_original
#         outputs_dict.start_index = IO_dict.start_index
#         outputs_dict.start_H = IO_dict.start_H
#         outputs_dict.start_W = IO_dict.start_W
#         return outputs_dict
#
#     def __getitem__(self, index):
#         outputs_dict = self.base_get_current_sub_folder_video_batch(index)
#         return outputs_dict
#
#     def __len__(self):
#         return len(self.image_filenames_list_of_lists)
#
#
#
#
#
#
#
# ### Get torch dataset from a folder with binary files to be read! - TODO: complete this, there are things to do: ###
# class ImageFolderRecursive_MaxNumberOfImages_BinaryFiles(data.Dataset):
#     """ ImageFolder can be used to load images where there are no labels."""
#
#     def __init__(self, root, transform=None, max_number_of_images=-1, min_crop_size=100, image_data_type_string='uint8'):
#         #TODO: why isn't the inherited class' (data.Dataset) __init__ function being activated?
#         images = []
#
#         counter = 0;
#         # root = 'C:/Users\dkarl\PycharmProjects\dudykarl\Imagenet'
#         # max_number_of_images = 100
#         for dp, dn, fn in os.walk(root):
#             if counter > max_number_of_images and max_number_of_images != -1:
#                 break;
#             for f in fn:
#                 filename = os.path.join(dp, f);
#                 counter += 1;
#                 if counter > max_number_of_images and max_number_of_images != -1:
#                     break;
#                 else:
#                     #TODO: add condition that file ends with .bin!!!!
#                     images.append('{}'.format(filename))
#         # shuffle(filename_list)
#
#         self.root = root
#         self.imgs = images
#         self.transform = transform
#         self.loader = loader
#         self.min_crop_size = min_crop_size;
#         self.image_data_type_string = image_data_type_string
#
#
#     def __getitem__(self, index):
#         filename = self.imgs[index]
#
#         try:
#             fid_Read_X = open(filename, 'rb')
#             frame_height = np.fromfile(fid_Read_X, 'float32', count=1);
#             frame_width = np.fromfile(fid_Read_X, 'float32', count=1);
#             number_of_channels = np.fromfile(fid_Read_X, 'float32', count=1);
#             mat_shape = [frame_height, frame_width, number_of_channels]  # list or tuple?!?!?!?!!?
#             total_images_number_of_elements = frame_height*frame_width*number_of_channels;
#             img = np.fromfile(fid_Read_X, self.image_data_type_string, count=total_images_number_of_elements);
#             img = img.reshape(mat_shape)
#             s = img.size
#             if min(s) < self.min_crop_size:
#                 raise ValueError('image smaller than a block (32,32)')
#             # img = img.resize((int(s[0]/2), int(s[1]/2)), Image.ANTIALIAS)
#         except:
#             print("problem loading " + filename)
#             img = Image.new('RGB', [self.min_crop_size, self.min_crop_size])
#
#             # return torch.zeros((3, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))
#
#         if self.transform is not None:
#             img = self.transform(img)
#         return img
#
#     def __len__(self):
#         return len(self.imgs)
#
#
#
# #####################################################################################################################################################
# def turbulence_deformation_single_image(I, Cn2=5e-13, flag_clip=False):
#     ### TODO: why the .copy() ??? ###
#     # I = I.copy()
#
#     # I = read_image_default()
#     # I = crop_center(I,150,150)
#     # imshow(I)
#
#     ### Parameters: ###
#     h = 100
#     # Cn2 = 2e-13
#     # Cn2 = 7e-17
#     wvl = 5.3e-7
#     IFOV = 4e-7
#     R = 1000
#     VarTiltX = 3.34e-6
#     VarTiltY = 3.21e-6
#     k = 2 * np.pi / wvl
#     r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
#     PixSize = IFOV * R
#     PatchSize = 2 * r0 / PixSize
#
#     ### Get Current Image Shape And Appropriate Meshgrid: ###
#     PatchNumRow = int(np.round(I.shape[0] / PatchSize))
#     PatchNumCol = int(np.round(I.shape[1] / PatchSize))
#     shape = I.shape
#     [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
#     if I.dtype == 'uint8':
#         mv = 255
#     else:
#         mv = 1
#
#     ### Get Random Motion Field: ###
#     ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
#     ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)
#
#     ### Resize (small) Random Motion Field To Image Size: ###
#     ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
#     ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
#
#     ### Add Rescaled Flow Field To Meshgrid: ###
#     X = X0 + ShiftMatX
#     Y = Y0 + ShiftMatY
#
#     ### Resample According To Motion Field: ###
#     I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)
#
#     ### Clip Result: ###
#     if flag_clip:
#         I = np.minimum(I, mv)
#         I = np.maximum(I, 0)
#
#     # imshow(I)
#
#     return I
#
#
#
#
# def get_turbulence_flow_field(H,W, batch_size, Cn2=2e-13):
#     ### TODO: why the .copy() ??? ###
#     # I = I.copy()
#
#     ### Parameters: ###
#     h = 100
#     # Cn2 = 2e-13
#     # Cn2 = 7e-17
#     wvl = 5.3e-7
#     IFOV = 4e-7
#     R = 1000
#     VarTiltX = 3.34e-6
#     VarTiltY = 3.21e-6
#     k = 2 * np.pi / wvl
#     r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
#     PixSize = IFOV * R
#     PatchSize = 2 * r0 / PixSize
#
#     ### Get Current Image Shape And Appropriate Meshgrid: ###
#     PatchNumRow = int(np.round(H / PatchSize))
#     PatchNumCol = int(np.round(W / PatchSize))
#     shape = I.shape
#     [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
#     if I.dtype == 'uint8':
#         mv = 255
#     else:
#         mv = 1
#
#     ### Get Random Motion Field: ###
#     [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
#     [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
#     X_large = torch.Tensor(X_large).unsqueeze(-1)
#     Y_large = torch.Tensor(Y_large).unsqueeze(-1)
#     X_large = (X_large-W/2) / (W/2-1)
#     Y_large = (Y_large-H/2) / (H/2-1)
#
#     new_grid = torch.cat([X_large,Y_large],2)
#     new_grid = torch.Tensor([new_grid.numpy()]*batch_size)
#
#     ShiftMatX0 = torch.randn(batch_size, PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
#     ShiftMatY0 = torch.randn(batch_size, PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)
#     ShiftMatX0 = ShiftMatX0 * W
#     ShiftMatY0 = ShiftMatY0 * H
#
#     ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0.unsqueeze(1), new_grid, mode='bilinear', padding_mode='reflection')
#     ShiftMatY = torch.nn.functional.grid_sample(ShiftMatX0.unsqueeze(1), new_grid, mode='bilinear', padding_mode='reflection')
#
#     ShiftMatX = ShiftMatX.squeeze()
#     ShiftMatY = ShiftMatY.squeeze()
#
#     # ### Resize (small) Random Motion Field To Image Size: ###
#     # ShiftMatX0 = F.adaptive_avg_pool2d(ShiftMatX0, torch.Size([H,W]))
#     # ShiftMatY0 = F.adaptive_avg_pool2d(ShiftMatY0, torch.Size([H,W]))
#
#     return ShiftMatX, ShiftMatY
#
#
#
#
#
#
#
#
# class get_turbulence_flow_field_object:
#     def __init__(self,H,W, batch_size, Cn2=2e-13):
#         ### Parameters: ###
#         h = 100
#         # Cn2 = 2e-13
#         # Cn2 = 7e-17
#         wvl = 5.3e-7
#         IFOV = 4e-7
#         R = 1000
#         VarTiltX = 3.34e-6
#         VarTiltY = 3.21e-6
#         k = 2 * np.pi / wvl
#         r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
#         PixSize = IFOV * R
#         PatchSize = 2 * r0 / PixSize
#
#         ### Get Current Image Shape And Appropriate Meshgrid: ###
#         PatchNumRow = int(np.round(H / PatchSize))
#         PatchNumCol = int(np.round(W / PatchSize))
#         shape = I.shape
#         [X0, Y0] = np.meshgrid(np.arange(W), np.arange(H))
#         if I.dtype == 'uint8':
#             mv = 255
#         else:
#             mv = 1
#
#         ### Get Random Motion Field: ###
#         [Y_small, X_small] = np.meshgrid(np.arange(PatchNumRow), np.arange(PatchNumCol))
#         [Y_large, X_large] = np.meshgrid(np.arange(H), np.arange(W))
#         X_large = torch.Tensor(X_large).unsqueeze(-1)
#         Y_large = torch.Tensor(Y_large).unsqueeze(-1)
#         X_large = (X_large-W/2) / (W/2-1)
#         Y_large = (Y_large-H/2) / (H/2-1)
#
#         new_grid = torch.cat([X_large,Y_large],2)
#         new_grid = torch.Tensor([new_grid.numpy()]*batch_size)
#
#         self.new_grid = new_grid;
#         self.batch_size = batch_size
#         self.PatchNumRow = PatchNumRow
#         self.PatchNumCol = PatchNumCol
#         self.VarTiltX = VarTiltX
#         self.VarTiltY = VarTiltY
#         self.R = R
#         self.PixSize = PixSize
#         self.H = H
#         self.W = W
#
#     def get_flow_field(self):
#         ShiftMatX0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltX * self.R / self.PixSize)
#         ShiftMatY0 = torch.randn(self.batch_size, 1, self.PatchNumRow, self.PatchNumCol) * (self.VarTiltY * self.R / self.PixSize)
#         ShiftMatX0 = ShiftMatX0 * self.W
#         ShiftMatY0 = ShiftMatY0 * self.H
#
#         ShiftMatX = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')
#         ShiftMatY = torch.nn.functional.grid_sample(ShiftMatX0, self.new_grid, mode='bilinear', padding_mode='reflection')
#
#         ShiftMatX = ShiftMatX.squeeze()
#         ShiftMatY = ShiftMatY.squeeze()
#
#         return ShiftMatX, ShiftMatY
#
#
# # ##################
# # # Lambda: #
# #
# # def func_images(images, random_state, parents, hooks):
# #     images[:, ::2, :, :] = 0
# #     return images
# #
# # def func_heatmaps(heatmaps, random_state, parents, hooks):
# #     for heatmaps_i in heatmaps:
# #         heatmaps.arr_0to1[::2, :, :] = 0
# #     return heatmaps
# #
# # def func_keypoints(keypoints_on_images, random_state, parents, hooks):
# #     return keypoints_on_images
# #
# #
# # aug = iaa.Lambda(func_images = func_images,
# #                  func_heatmaps = func_heatmaps,
# #                  func_keypoints = func_keypoints)
# # ##################

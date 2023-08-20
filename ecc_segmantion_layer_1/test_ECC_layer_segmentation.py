import matplotlib.pyplot as plt

from RapidBase.import_all import *
import ECC_layer_points_segmentation as ecc_file
import torch
import torchvision.transforms as transforms
import kornia
desired_size = (256, 256)  # Desired size for resizing


def get_files_from_directory(directory_path):
    file_list = []
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_list.append(filename)
    return file_list

def resize_images_and_masks(images_list, masks_list, desired_size):
    ### Upsample Images to constant size: ###
    upsample_layer = torch.nn.Upsample(size=desired_size, mode='bicubic')
    images_list_final = []
    masks_list_final = []
    for frame_index in np.arange(len(images_list)):
        ### get current image and mask: ###
        current_image = images_list[frame_index]
        current_mask = masks_list[frame_index][0]

        # Add dimensions for batch size and number of channels, if necessary
        if len(current_image.shape) == 2:  # Grayscale image without channel dim
            current_image = current_image.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        elif len(current_image.shape) == 3:  # Image with channels but without batch dim
            current_image = current_image.unsqueeze(0)  # add batch dim

        if len(current_mask.shape) == 2:  # Grayscale mask without channel dim
            current_mask = current_mask.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        elif len(current_mask.shape) == 3:  # Mask with channels but without batch dim
            current_mask = current_mask.unsqueeze(0)  # add batch dim

        ### upsample tensors: ###
        current_mask = current_mask.float()
        current_image = current_image.float()
        final_image = upsample_layer.forward(current_image)
        final_mask = upsample_layer.forward(current_mask)

        ### remove added dimensions (if you want to keep the original format in the list)
        final_image = final_image.squeeze(0)
        final_mask = final_mask.squeeze(0)

        ### append to final list: ###
        images_list_final.append(final_image.unsqueeze(0))
        masks_list_final.append(final_mask.unsqueeze(0))

    return images_list_final, masks_list_final


def crop_according_to_mask_and_resize(input_tensor_RGB, input_tensor_BW, segmentation_tensor):
    ### Upsample Images to constant size: ###
    images_bw_list_final = []
    images_rgb_list_final = []
    masks_list_final = []
    max_W = 0
    max_H = 0
    max_HW = 0
    ### Loop Over Frames and Get Max Bounding-Box: ###
    for frame_index in np.arange(len(input_tensor_RGB)):
        ### get current image and mask: ###
        current_image_RGB = input_tensor_RGB[frame_index]
        current_image_BW = input_tensor_BW[frame_index]
        current_segmentation_mask = segmentation_tensor[frame_index]
        current_logical_mask = (current_segmentation_mask == 255.0).nonzero()[:,1:]
        current_logical_mask_H_indices = current_logical_mask[:,0]
        current_logical_mask_W_indices = current_logical_mask[:,1]

        ### Get Current Bounding-Box: ###
        buffer_size = 20
        current_logical_mask_H_indices_min = current_logical_mask_H_indices.min()-buffer_size
        current_logical_mask_H_indices_max = current_logical_mask_H_indices.max()+buffer_size
        current_logical_mask_W_indices_min = current_logical_mask_W_indices.min()-buffer_size
        current_logical_mask_W_indices_max = current_logical_mask_W_indices.max()+buffer_size
        current_H = current_logical_mask_H_indices_max - current_logical_mask_H_indices_min
        current_W = current_logical_mask_W_indices_max - current_logical_mask_W_indices_min

        ### Get max H*W Untill now: ###
        current_HW = current_H * current_W
        if current_HW > max_HW:
            max_H = current_H
            max_W = current_W
            max_HW = current_HW



    ### Loop Over Frames, Get Crops And Resize Properly: ###
    upsample_layer = torch.nn.Upsample(size=(max_H, max_W), mode='bicubic')
    for frame_index in np.arange(len(input_tensor_RGB)):
        ### get current image and mask: ###
        current_image_RGB = input_tensor_RGB[frame_index]
        current_image_BW = input_tensor_BW[frame_index]
        current_segmentation_mask = segmentation_tensor[frame_index]
        current_logical_mask = (current_segmentation_mask == 255.0).nonzero()[:, 1:]
        current_logical_mask_H_indices = current_logical_mask[:, 0]
        current_logical_mask_W_indices = current_logical_mask[:, 1]

        ### Get Current Bounding-Box: ###
        current_logical_mask_H_indices_min = current_logical_mask_H_indices.min() - 2
        current_logical_mask_H_indices_max = current_logical_mask_H_indices.max() + 2
        current_logical_mask_W_indices_min = current_logical_mask_W_indices.min() - 2
        current_logical_mask_W_indices_max = current_logical_mask_W_indices.max() + 2
        current_H = current_logical_mask_H_indices_max - current_logical_mask_H_indices_min
        current_W = current_logical_mask_W_indices_max - current_logical_mask_W_indices_min
        ### Get Crop Tensor: ###
        current_crop_RGB = current_image_RGB[:,current_logical_mask_H_indices_min:current_logical_mask_H_indices_max, current_logical_mask_W_indices_min:current_logical_mask_W_indices_max]
        current_crop_BW = current_image_BW[:,current_logical_mask_H_indices_min:current_logical_mask_H_indices_max, current_logical_mask_W_indices_min:current_logical_mask_W_indices_max]

        ### Add dimensions for batch size and number of channels, if necessary: ###
        if len(current_crop_RGB.shape) == 2:  # Grayscale image without channel dim
            current_crop_RGB = current_crop_RGB.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        elif len(current_crop_RGB.shape) == 3:  # Image with channels but without batch dim
            current_crop_RGB = current_crop_RGB.unsqueeze(0)  # add batch dim

        ### Add dimensions for batch size and number of channels, if necessary: ###
        if len(current_crop_BW.shape) == 2:  # Grayscale image without channel dim
            current_crop_BW = current_crop_BW.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        elif len(current_crop_BW.shape) == 3:  # Image with channels but without batch dim
            current_crop_BW = current_crop_BW.unsqueeze(0)  # add batch dim

        ### Add dimensions for batch size and number of channels, if necessary: ###
        if len(current_segmentation_mask.shape) == 2:  # Grayscale mask without channel dim
            current_segmentation_mask = current_segmentation_mask.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        elif len(current_segmentation_mask.shape) == 3:  # Mask with channels but without batch dim
            current_segmentation_mask = current_segmentation_mask.unsqueeze(0)  # add batch dim

        ### upsample tensors: ###
        current_segmentation_mask = current_segmentation_mask.float()
        current_crop_BW = current_crop_BW.float()
        current_crop_RGB = current_crop_RGB.float()
        final_image_BW = upsample_layer.forward(current_crop_BW)
        final_image_RGB = upsample_layer.forward(current_crop_RGB)
        final_mask = upsample_layer.forward(current_segmentation_mask)

        ### remove added dimensions (if you want to keep the original format in the list)
        final_image_BW = final_image_BW
        final_image_RGB = final_image_RGB
        final_mask = final_mask

        ### append to final list: ###
        images_bw_list_final.append(final_image_BW)
        images_rgb_list_final.append(final_image_RGB)
        masks_list_final.append(final_mask)

    ### Concat To Tensor: ###
    images_bw_final_tensor = torch.cat(images_bw_list_final)
    images_rgb_final_tensor = torch.cat(images_rgb_list_final)
    segmentation_masks_final_tensor = torch.cat(masks_list_final)
    return images_bw_final_tensor, images_rgb_final_tensor, segmentation_masks_final_tensor


def main_interface(super_folder, desired_size=(256, 256), draw_image_string_pattern_to_search='*raw_image.png', segmentation_string_pattern_to_search="*segmentation.pt"):

    # Load Images And Segmentation Masks
    raw_images_filenames_list = path_get_files_from_folder(path=super_folder, flag_recursive=True,
                                                           string_pattern_to_search=draw_image_string_pattern_to_search)
    segmentation_mask_filenames_list = path_get_files_from_folder(path=super_folder, flag_recursive=True,
                                                                  string_pattern_to_search=segmentation_string_pattern_to_search)

    if len(segmentation_mask_filenames_list) != len(segmentation_mask_filenames_list):
        raise RuntimeError(f"Amount of segmentation files must equal the raw image files {len(segmentation_mask_filenames_list)} != {len(segmentation_mask_filenames_list)}")
    raw_images_list = []
    segmentation_tensor_list = []

    for image_index in np.arange(len(raw_images_filenames_list)):
        # Get filenames
        raw_image_filename = raw_images_filenames_list[image_index]
        segmentation_mask_filename = segmentation_mask_filenames_list[image_index]

        # Load file
        raw_image_tensor = numpy_to_torch(read_image_cv2(raw_image_filename))
        segmentation_mask_tensor = torch.tensor(torch.load(segmentation_mask_filename))

        # Append to list
        raw_images_list.append(raw_image_tensor.unsqueeze(0))
        segmentation_tensor_list.append(segmentation_mask_tensor.unsqueeze(0))

    raw_images_list, segmentation_tensor_list = resize_images_and_masks(raw_images_list, segmentation_tensor_list, desired_size)

    # Concat to full tensor
    input_tensor_RGB = torch.cat(raw_images_list)
    segmentation_mask_tensor = torch.cat(segmentation_tensor_list)

    input_tensor_RGB = input_tensor_RGB[4:]
    segmentation_mask_tensor = segmentation_mask_tensor[4:]

    input_tensor_original = input_tensor_RGB * 1
    input_tensor_BW = RGB2BW(input_tensor_RGB)

    # Show Segmentation Mask
    input_tensor_segmented = input_tensor_BW * 1
    input_tensor_segmented[segmentation_mask_tensor == 0] = 0

    # Transfer to CUDA
    input_tensor_BW = input_tensor_BW.cuda()
    input_tensor_RGB = input_tensor_RGB.cuda()
    segmentation_mask_tensor = segmentation_mask_tensor.cuda()

    # Perform ECC
    number_of_frames_per_batch = input_tensor_BW.shape[0]
    H, W = input_tensor_BW.shape[-2:]
    total_number_of_pixels = H * W
    # number_of_pixels_to_use = int(1 * segmentation_mask_tensor[0].sum())
    # number_of_batches = input_tensor_BW.shape[0] / number_of_frames_per_batch
    reference_tensor = input_tensor_BW[-1:]
    precision = torch.float
    ECC_layer_object = ecc_file.ECC_Layer_Torch_Points_Batch(input_tensor_BW[0:number_of_frames_per_batch],
                                                             reference_tensor,
                                                             number_of_iterations_per_level=800,
                                                             number_of_levels=1,
                                                             transform_string='homography',
                                                             number_of_pixels_to_use=total_number_of_pixels,
                                                             delta_p_init=None,
                                                             precision=precision)

    output_tensor, H_matrix = ECC_layer_object.forward_iterative(input_tensor_RGB,
                                                                 input_tensor_BW.type(precision),
                                                                 reference_tensor.type(precision),
                                                                 max_shift_threshold=0.3e-4,
                                                                 flag_print=False,
                                                                 delta_p_init=None,
                                                                 number_of_images_per_batch=number_of_frames_per_batch,
                                                                 flag_calculate_gradient_in_advance=False,
                                                                 segmentation_mask=segmentation_mask_tensor[-1:].bool())

    #ToDo save the outputs
    upsample_layer = torch.nn.Upsample(scale_factor=(3, 3), mode='bicubic')
    input_tensor_upsampled = upsample_layer.forward(input_tensor_BW)
    input_tensor_original_upsampled = upsample_layer.forward(input_tensor_original)
    output_tensor_upsampled = upsample_layer.forward(output_tensor)
    output_tensor_upsampled_time_averaged = output_tensor_upsampled[0:10].mean(0, True)
    output_tensor_upsampled_time_averaged_clahe = scale_array_clahe(output_tensor_upsampled_time_averaged, flag_stretch_histogram_first=False, grid_size=(16,16), clip_limit=40.0)
    output_tensor_upsampled_time_averaged_sharpen = kornia.enhance.sharpness(RGB2BW(output_tensor_upsampled_time_averaged)/255, factor=40)
    torch.save(output_tensor_upsampled_time_averaged, r'shaback.pt')
    imshow_torch(output_tensor_upsampled_time_averaged/255)
    imshow_torch(RGB2BW(output_tensor_upsampled_time_averaged)/255)
    imshow_torch(output_tensor_upsampled_time_averaged_clahe)
    imshow_torch(output_tensor_upsampled_time_averaged_sharpen)
    imshow_torch(input_tensor_RGB[3]/255)
    imshow_torch_video(output_tensor_upsampled/255, FPS=1)
    imshow_torch_video(input_tensor_RGB/255, FPS=1)
    # imshow_torch_imagesc(torch.cat([input_tensor_original_upsampled[3:4], output_tensor_upsampled_time_averaged], -1))






if __name__ == '__main__':
    super_folder ="/home/nehoray/PycharmProjects/Shaback/output/GROUNDED_SEGMENTATION/1"
    main_interface(super_folder)
from RapidBase.import_all import *
from RapidBase.Utils.Classical_DSP.Add_Noise import *




#########################   Make Still Images For Official Test DataSet: ##################
### Still Images: ###
clean_still_images_folder = r'C:\DataSets\Div2K\Official_Test_Images\Original_Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
max_shift_in_pixels = 0
min_crop_size = 1300
crop_size = np.inf
noise_gains = [0,30,65,100]
additive_noise_PSNR = [100,10,5,4,3,2,1]
number_of_time_steps = 10

### IMGAUG: ###
imgaug_parameters = get_IMGAUG_parameters()
# Affine:
imgaug_parameters['flag_affine_transform'] = True
imgaug_parameters['affine_scale'] = 1
imgaug_parameters['affine_translation_percent'] = None
imgaug_parameters['affine_translation_number_of_pixels'] = (0,max_shift_in_pixels)
imgaug_parameters['affine_rotation_degrees'] = 0
imgaug_parameters['affine_shear_degrees'] = 0
imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
imgaug_parameters['probability_of_affine_transform'] = 1
#Perspective:
imgaug_parameters['flag_perspective_transform'] = False
imgaug_parameters['flag_perspective_transform_keep_size'] = True
imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
imgaug_parameters['probability_of_perspective_transform'] = 1
### Get Augmenter: ###
imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)


### IMGAUG Cropping: ###
train_dataset = Dataset_Images_From_Folder(clean_still_images_folder,
                                           transform=imgaug_transforms,
                                           image_loader=ImageLoaderCV,
                                           max_number_of_images=10,
                                           crop_size=np.inf,
                                           flag_to_RAM=True,
                                           flag_recursive=False,
                                           flag_normalize_by_255=False,
                                           flag_crop_mode='center',
                                           flag_explicitely_make_tensor=False,
                                           allowed_extentions=IMG_EXTENSIONS,
                                           flag_base_transform=False,
                                           flag_turbulence_transform=False,
                                           Cn2=5e-13)
image_brightness_reduction_factor = 0.9
train_dataset[0]


### For Every Image Get Noisy Image (For Every Noise Gain): ###
for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
    ### Create new folder for this particular Noise: ###
    PSNR_folder = os.path.join(still_images_folder, 'Still_Images', 'PSNR_' + str(PSNR))
    path_make_directory_from_path(PSNR_folder)

    for image_index in arange(len(train_dataset)):
        ### Create a new folder for this particular image: ###
        image_within_noise_folder = os.path.join(PSNR_folder, train_dataset.image_filenames_list[image_index].split('.')[0].split('\\')[-1])
        path_make_directory_from_path(image_within_noise_folder)

        ### Read the image: ###
        current_frame = train_dataset[image_index]
        current_frame = numpy_from_torch_to_numpy_convention(current_frame)
        current_frame = current_frame * image_brightness_reduction_factor
        current_frame = auxiliary_take_care_of_input_image(current_frame, flag_input_normalized=False)

        for time_step in arange(number_of_time_steps):
            current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0, additive_noise_SNR=additive_noise_PSNR[PSNR_counter], flag_input_normalized=True)
            numpy_array = current_frame_noisy * 255
            numpy_array = numpy_array.clip(0, 255)
            numpy_array = numpy_array.astype('uint8')
            save_image_numpy(image_within_noise_folder, str(time_step).rjust(3,'0') + '.png', numpy_array, flag_convert_bgr2rgb=False, flag_scale=False)
















#########################   Make Semi-Dynamic Shifting Images For Official Test DataSet: ##################
### Only Shift Images: ###
super_shifts_folder = r'C:\DataSets\Div2K\Official_Test_Images\Shifted'
clean_still_images_folder = r'C:\DataSets\Div2K\Official_Test_Images\Original_Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
max_shifts_in_pixels = [1,2,5,10,20]
min_crop_size = 1300
crop_size = np.inf
additive_noise_PSNR = [100,10,5,4,3,2,1]
number_of_time_steps = 10



for max_shift_in_pixels in max_shifts_in_pixels:
    shift_folder = os.path.join(super_shifts_folder, 'Shift_' + str(max_shift_in_pixels))

    ### IMGAUG: ###
    imgaug_parameters = get_IMGAUG_parameters()
    # Affine:
    imgaug_parameters['flag_affine_transform'] = True
    imgaug_parameters['affine_scale'] = 1
    imgaug_parameters['affine_translation_percent'] = None
    imgaug_parameters['affine_translation_number_of_pixels'] = {"x": (-max_shift_in_pixels, max_shift_in_pixels), "y": (-max_shift_in_pixels, max_shift_in_pixels)}
    imgaug_parameters['affine_rotation_degrees'] = 0
    imgaug_parameters['affine_shear_degrees'] = 0
    imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
    imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
    imgaug_parameters['probability_of_affine_transform'] = 1
    #Perspective:
    imgaug_parameters['flag_perspective_transform'] = False
    imgaug_parameters['flag_perspective_transform_keep_size'] = True
    imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
    imgaug_parameters['probability_of_perspective_transform'] = 1
    ### Get Augmenter: ###
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)

    ### Define the dataset object with the new relevant imgaug_parameters: ###
    train_dataset = Dataset_Images_From_Folder(clean_still_images_folder,
                                               transform=imgaug_transforms,
                                               image_loader=ImageLoaderCV,
                                               max_number_of_images=10,
                                               crop_size=np.inf,
                                               flag_to_RAM=True,
                                               flag_recursive=False,
                                               flag_normalize_by_255=False,
                                               flag_crop_mode='center',
                                               flag_explicitely_make_tensor=False,
                                               allowed_extentions=IMG_EXTENSIONS,
                                               flag_base_transform=True,
                                               flag_turbulence_transform=False,
                                               Cn2=5e-13)

    for image_index in arange(len(train_dataset)):

        for time_step in arange(number_of_time_steps):
            ### Read the image (and randomise new shift!): ###
            current_frame = train_dataset[image_index]
            current_frame = numpy_from_torch_to_numpy_convention(current_frame)
            current_frame = current_frame * image_brightness_reduction_factor
            current_frame = auxiliary_take_care_of_input_image(current_frame, flag_input_normalized=False)

            ### For Every Image Get Noisy Image (For Every Noise Gain): ###
            for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
                ### Create new folder for this particular Noise: ###
                noise_gain_folder = os.path.join(shift_folder, 'PSNR_' + str(PSNR))
                image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames_list[image_index].split('.')[0].split('\\')[-1])
                path_make_directory_from_path(image_within_noise_folder)

                current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0,
                                                         additive_noise_SNR=additive_noise_PSNR[PSNR_counter],
                                                         flag_input_normalized=True)
                numpy_array = current_frame_noisy * 255
                numpy_array = numpy_array.clip(0, 255)
                numpy_array = numpy_array.astype('uint8')
                save_image_numpy(image_within_noise_folder, str(time_step).rjust(3, '0') + '.png', numpy_array, flag_convert_bgr2rgb=False, flag_scale=False)












#########################   Make Semi-Dynamic Affine Deformation Images For Official Test DataSet: ##################
### Still Images: ###
super_affines_folder = r'C:\DataSets\Div2K\Official_Test_Images\Affine'
clean_still_images_folder = r'C:\DataSets\Div2K\Official_Test_Images\Original_Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
min_crop_size = 1300
crop_size = np.inf
number_of_time_steps = 10

### Loop Variables: ###
additive_noise_PSNR = [100,10,5,4,3,2,1]
max_shifts_in_pixels = [0,5,20]
max_scales_deltas = [0,0.05]
max_degrees = [0,5]
max_perspective_change_factors = [0,0.05]

image_brightness_reduction_factor = 0.9


for max_shift_in_pixels in max_shifts_in_pixels:
    for max_scale in max_scales_deltas:
        for max_degree in max_degrees:
            for max_perspective_change_factor in max_perspective_change_factors:

                dot_change_char = '#'
                current_sub_folder_name = '_Shift_' + str(max_shift_in_pixels).replace('.',dot_change_char) + \
                                          '_Degrees_' + str(max_degree).replace('.',dot_change_char) + \
                                          '_Scale_' + str(max_scale).replace('.',dot_change_char) + \
                                          '_Perspective_' + str(max_perspective_change_factor).replace('.',dot_change_char)
                current_affine_folder = os.path.join(super_affines_folder, current_sub_folder_name)
                path_make_directory_from_path(current_affine_folder)

                ### IMGAUG: ###
                imgaug_parameters = get_IMGAUG_parameters()
                # Affine:
                imgaug_parameters['flag_affine_transform'] = True
                imgaug_parameters['affine_scale'] = (1-max_scale,1+max_scale)
                imgaug_parameters['affine_translation_percent'] = None
                imgaug_parameters['affine_translation_number_of_pixels'] = {"x": (-max_shift_in_pixels, max_shift_in_pixels), "y": (-max_shift_in_pixels, max_shift_in_pixels)}
                imgaug_parameters['affine_rotation_degrees'] = (-max_degree,max_degree)
                imgaug_parameters['affine_shear_degrees'] = 0
                imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
                imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
                imgaug_parameters['probability_of_affine_transform'] = 1
                #Perspective:
                imgaug_parameters['flag_perspective_transform'] = True
                imgaug_parameters['flag_perspective_transform_keep_size'] = True
                imgaug_parameters['perspective_transform_scale'] = (0.0, max_perspective_change_factor)
                imgaug_parameters['probability_of_perspective_transform'] = 1
                ### Get Augmenter: ###
                imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)


                ### Define the dataset object with the new relevant imgaug_parameters: ###
                train_dataset = Dataset_Images_From_Folder(clean_still_images_folder,
                                                           transform=imgaug_transforms,
                                                           image_loader=ImageLoaderCV,
                                                           max_number_of_images=10,
                                                           crop_size=np.inf,
                                                           flag_to_RAM=True,
                                                           flag_recursive=False,
                                                           flag_normalize_by_255=False,
                                                           flag_crop_mode='center',
                                                           flag_explicitely_make_tensor=False,
                                                           allowed_extentions=IMG_EXTENSIONS,
                                                           flag_base_transform=True,
                                                           flag_turbulence_transform=False,
                                                           Cn2=5e-13)


                tic()
                for image_index in arange(len(train_dataset)):

                    for time_step in arange(number_of_time_steps):
                        ### Read the image (and randomise new shift!): ###
                        current_frame = train_dataset[image_index]
                        current_frame = numpy_from_torch_to_numpy_convention(current_frame)
                        current_frame = current_frame * image_brightness_reduction_factor
                        current_frame = auxiliary_take_care_of_input_image(current_frame, flag_input_normalized=False)

                        ### For Every Image Get Noisy Image (For Every Noise Gain): ###
                        for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
                            ### Create new folder for this particular Noise: ###
                            noise_gain_folder = os.path.join(current_affine_folder, 'PSNR_' + str(PSNR))
                            image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames_list[image_index].split('.')[0].split('\\')[-1])
                            path_make_directory_from_path(image_within_noise_folder)

                            current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0,
                                                                     additive_noise_SNR=additive_noise_PSNR[
                                                                         PSNR_counter],
                                                                     flag_input_normalized=True)
                            numpy_array = current_frame_noisy * 255
                            numpy_array = numpy_array.clip(0, 255)
                            numpy_array = numpy_array.astype('uint8')
                            save_image_numpy(image_within_noise_folder, str(time_step).rjust(3, '0') + '.png', numpy_array, flag_convert_bgr2rgb=True, flag_scale=False)
                toc()















#########################   Make Semi-Dynamic "Turbulence" Only Deformation Images For Official Test DataSet: ##################
### Still Images: ###
super_affines_folder = r'C:\DataSets\Div2K\Official_Test_Images\Turbulence'
clean_still_images_folder = r'C:\DataSets\Div2K\Official_Test_Images\Original_Images'
still_images_folder = os.path.split(clean_still_images_folder)[0]
clean_still_images_filenames = get_all_filenames_from_folder(clean_still_images_folder)
min_crop_size = 1300
crop_size = 1000
additive_noise_PSNR = [100,10,5,4,3,2,1]
number_of_time_steps = 10


Cn2_vec = [3e-15,5e-15,1e-14,3e-14,1e-13]
# Cn2_vec = [3e-15]
image_brightness_reduction_factor = 0.9

def scientific_notation(input_number,number_of_digits_after_point=2):
    format_string = '{:.' + str(number_of_digits_after_point) + 'e}'
    return format_string.format(input_number)


for current_Cn2 in Cn2_vec:

    dot_change_char = '#'
    current_sub_folder_name = 'Cn2_' + str(current_Cn2)
    current_affine_folder = os.path.join(super_affines_folder, current_sub_folder_name)
    path_make_directory_from_path(current_affine_folder)

    ### IMGAUG: ###
    imgaug_parameters = get_IMGAUG_parameters()
    imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)

    ### Define the dataset object with the new relevant imgaug_parameters: ###
    train_dataset = Dataset_Images_From_Folder(clean_still_images_folder,
                                               transform=imgaug_transforms,
                                               image_loader=ImageLoaderCV,
                                               max_number_of_images=10,
                                               crop_size=np.inf,
                                               flag_to_RAM=True,
                                               flag_recursive=False,
                                               flag_normalize_by_255=False,
                                               flag_crop_mode='center',
                                               flag_explicitely_make_tensor=False,
                                               allowed_extentions=IMG_EXTENSIONS,
                                               flag_base_transform=False,
                                               flag_turbulence_transform=True,
                                               Cn2=5e-13)


    tic()
    for image_index in arange(len(train_dataset)):

        for time_step in arange(number_of_time_steps):
            ### Read the image (and randomise new shift!): ###
            current_frame = train_dataset[image_index]
            current_frame = numpy_from_torch_to_numpy_convention(current_frame)
            current_frame = current_frame * image_brightness_reduction_factor
            current_frame = auxiliary_take_care_of_input_image(current_frame, flag_input_normalized=False)

            ### For Every Image Get Noisy Image (For Every Noise Gain): ###
            for PSNR_counter, PSNR in enumerate(additive_noise_PSNR):
                ### Create new folder for this particular Noise: ###
                noise_gain_folder = os.path.join(current_affine_folder, 'PSNR_' + str(PSNR))
                image_within_noise_folder = os.path.join(noise_gain_folder, train_dataset.image_filenames_list[image_index].split('.')[0].split('\\')[-1])
                path_make_directory_from_path(image_within_noise_folder)

                current_frame_noisy = add_noise_to_image(current_frame, shot_noise_adhok_gain=0,
                                                         additive_noise_SNR=additive_noise_PSNR[PSNR_counter],
                                                         flag_input_normalized=True)
                numpy_array = current_frame_noisy * 255
                numpy_array = numpy_array.clip(0, 255)
                numpy_array = numpy_array.astype('uint8')
                save_image_numpy(image_within_noise_folder, str(time_step).rjust(3, '0') + '.png', numpy_array, flag_convert_bgr2rgb=True, flag_scale=False)
    toc()























#
# ### IMGAUG: ###
# imgaug_parameters = get_IMGAUG_parameters()
# # Affine:
# imgaug_parameters['flag_affine_transform'] = True
# imgaug_parameters['affine_scale'] = (0.9,1.1)
# imgaug_parameters['affine_translation_percent'] = None
# imgaug_parameters['affine_translation_number_of_pixels'] = (0,10)
# imgaug_parameters['affine_rotation_degrees'] = (0,10)
# imgaug_parameters['affine_shear_degrees'] = (0,10)
# imgaug_parameters['affine_order'] = cv2.INTER_CUBIC
# imgaug_parameters['affine_mode'] = cv2.BORDER_REFLECT_101
# imgaug_parameters['probability_of_affine_transform'] = 1
# #Perspective:
# imgaug_parameters['flag_perspective_transform'] = False
# imgaug_parameters['flag_perspective_transform_keep_size'] = True
# imgaug_parameters['perspective_transform_scale'] = (0.0, 0.05)
# imgaug_parameters['probability_of_perspective_transform'] = 1
# ### Get Augmenter: ###
# imgaug_transforms = get_IMGAUG_Augmenter(imgaug_parameters)
#
#
# ### IMGAUG Cropping: ###
# #(*). Note - this method probably isn't smart for generality sake because it requires to reinitialize the iaa.Crop object for different sizes images....
# min_crop_size = 140
# crop_size = 100
# root = 'F:/NON JPEG IMAGES\HR_images\General100'
# batch_size = 16
# train_dataset = ImageFolderRecursive_MaxNumberOfImages_Deformations(root,
#                                                                     transform=imgaug_transforms,
#                                                                     flag_base_transform=True,
#                                                                     flag_turbulence_transform = True,
#                                                                     max_number_of_images = -1,
#                                                                     min_crop_size=min_crop_size,
#                                                                     crop_size=crop_size,
#                                                                     extention='png',loader='CV')
# train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True, pin_memory=True)  ### Mind the shuffle variable!@#!#%@#%@$%
# tic()
# for i,data_from_dataloader in enumerate(train_dataloader):
#     bla = data_from_dataloader
#     # for image_counter in arange(len(data_from_dataloader)):
#     #     imshow_torch(data_from_dataloader[image_counter,:,:,:])
#     #     pause(0.2)
#     if i==0:
#         break
# toc()











# #### Affine Transform On GPU: ###
# crop_size = 100
# current_image = read_image_default()
# current_image = crop_tensor(current_image, crop_size,crop_size)
# current_image = np.transpose(current_image, [2,0,1])
# current_image_tensor = torch.Tensor(current_image).unsqueeze(0)
#
# #(1). Initial Grid:
# turbulence_object = get_turbulence_flow_field_object(crop_size,crop_size,batch_size,Cn2=7e-13)
# [X0, Y0] = np.meshgrid(np.arange(crop_size), np.arange(crop_size))
# X0 = float32(X0)
# Y0 = float32(Y0)
# X0 = torch.Tensor(X0)
# Y0 = torch.Tensor(Y0)
# X0 = X0.unsqueeze(0)
# Y0 = Y0.unsqueeze(0)
# X0 = torch.cat([X0]*batch_size,0)
# Y0 = torch.cat([Y0]*batch_size,0)
# #(3). Scale:
# scale_x = 1.2
# scale_y = 1
# scale_x = 1/scale_x
# scale_y = 1/scale_y
# X0 *= scale_x
# Y0 *= scale_y
# #(4). Rotate:
# max_rotation_angle = 45
# rotation_angles = ( torch.rand((batch_size,1))*max_rotation_angle-0.5*max_rotation_angle  ) * 2
# # rotation_matrix = cv2.getRotationMatrix2D((crop_size/2,crop_size/2), rotation_angle, 1)
# # flow_field = torch.nn.functional.affine_grid(torch.Tensor(rotation_matrix).unsqueeze(0), torch.Size([1,3,crop_size,crop_size]) )
#
# X0_centered = X0-X0.max()/2
# Y0_centered = Y0-Y0.max()/2
# X0_new = cos(rotation_angles*pi/180).unsqueeze(-1)*X0_centered - sin(rotation_angles*pi/180).unsqueeze(-1)*Y0_centered
# Y0_new = sin(rotation_angles*pi/180).unsqueeze(-1)*X0_centered + cos(rotation_angles*pi/180).unsqueeze(-1)*Y0_centered
# X0 = X0_new
# Y0 = Y0_new
#
# #(2). Shift:
# shift_x = 0
# shift_y = 0
# X0 += shift_x*scale_x
# Y0 += shift_y*scale_y
#
# ### Add Turbulence: ###
# # turbulence_flow_field_X, turbulence_flow_field_Y = get_turbulence_flow_field(crop_size,crop_size,batch_size,Cn2=6e-13)
# turbulence_flow_field_X, turbulence_flow_field_Y = turbulence_object.get_flow_field()
# turbulence_flow_field_X = turbulence_flow_field_X/crop_size
# turbulence_flow_field_Y = turbulence_flow_field_Y/crop_size
#
# X0 = X0 + turbulence_flow_field_X
# Y0 = Y0 + turbulence_flow_field_X
# X0 = X0.unsqueeze(-1)
# Y0 = Y0.unsqueeze(-1)
#
# #(*). Normalize To 1:
# X0 = X0/((crop_size-1)/2)
# Y0 = Y0/((crop_size-1)/2)
# flow_grid = torch.cat([X0,Y0],3)
#
# tic()
# # bli = torch.nn.functional.grid_sample(current_image_tensor,flow_grid,mode='bilinear')
# bli = torch.nn.functional.grid_sample(data_from_dataloader,flow_grid,mode='bilinear',padding_mode='reflection')
# toc()
#
# figure(1)
# imshow_torch(data_from_dataloader[0,:,:,:])
# figure(2)
# imshow_torch(bli[0,:,:,:])
#
# figure(1)
# imshow_torch(current_image_tensor[0,:,:,:])
# figure(2)
# imshow_torch(bli[0,:,:,:])

# imgaug_transforms = imgaug_transforms.to_deterministic()
# current_frame_transformed = imgaug_transforms.augment_images([current_frame])[0]













def create_noisy_dataset_with_optical_flow_from_existing_dataset_PARALLEL():
    ### Get all image filenames: ###
    images_super_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'
    max_counter = 1

    ### Initialize TVNet_layer to None: ###
    TVNet_layer = None

    ### Loop over sub-folders and for each sub-folder: ###
    counter = 0
    for dirpath, sub_folders_list, filenames in os.walk(images_super_folder):
        if counter>max_counter:
            break

        ### Read ALL IMAGES for current sub-folder: ###
        for folder_index, current_subfolder in enumerate(sub_folders_list):
            if counter>max_counter:
                break
            images = read_images_from_folder_to_numpy(os.path.join(dirpath,current_subfolder),flag_recursive=True)

            ### Turn Images to Tensor and send to wanted device: ###
            tic()
            images = torch.Tensor(np.transpose(images,[0,3,1,2])).to(Generator_device)

            ### get Optical Flow : ###
            if TVNet_layer is None:
                TVNet_layer = get_TVNet_instance(images, number_of_iterations=TVNet_number_of_iterations, number_of_pyramid_scales=TVNet_number_of_pyramid_scales, device=Generator_device, flag_trainable=False)
            delta_x, delta_y, x2_warped = TVNet_layer.forward(images[0:-1], images[1:], need_result=True)
            toc('Optical Flow')

            ### Add Zero Shift Maps as start of map because usually i assume the for the first input image i assume the "previous_image" is simply the same firs input image: ###
            delta_x = torch.cat([torch.zeros(1,1,delta_x.shape[2],delta_x.shape[3]).to(Generator_device), delta_x], dim=0)
            delta_y = torch.cat([torch.zeros(1,1,delta_y.shape[2],delta_y.shape[3]).to(Generator_device), delta_y], dim=0)

            ### Write down Opical Flow Maps into Binary File: ###
            #(1). Initialize Binary File FID:
            fid_filename_X = os.path.join(os.path.join(dirpath,current_subfolder), 'Optical_Flow' + '.bin')
            fid_Write_X = open(fid_filename_X, 'ab')
            np.array(delta_x.shape[2]).tofile(fid_Write_X) #Height
            np.array(delta_x.shape[3]).tofile(fid_Write_X) #Width
            np.array(2).tofile(fid_Write_X) #Number_of_channels=2 (u,v)
            #(2). Write Maps to FID:
            #   (2.1). Write in form [B,2,H,W] all at once:
            numpy_array_to_write = torch.cat([delta_x,delta_y], dim=1).cpu().numpy()
            #   (2.2). Write in form [1,2,H,W] sequentially:
            #TODO: IMPLEMENT!
            #   (2.3). Write Optical Flow Maps To Binary File:
            numpy_array_to_write.tofile(fid_Write_X)
            #(3). Release FID:
            fid_Write_X.close()
            ### Uptick counter by 1: ###
            counter += 1









def create_noisy_dataset_with_optical_flow_from_existing_dataset():
    ### Get all image filenames: ###
    images_super_folder = 'G:\Movie_Scenes_bin_files\CropHeight_256_CropWidth_256'
    max_counter = 10
    # image_filenames_list = get_image_filenames_from_folder(images_super_folder, flag_recursive=True)

    ### Initialize TVNet_layer to None: ###
    TVNet_layer = None

    ### Loop over sub-folders and for each sub-folder: ###
    counter = 0
    for dirpath, dn, filenames in os.walk(images_super_folder):
        if counter>max_counter:
            break
        for image_index, current_filename in enumerate(sorted(filenames)):
            if counter>max_counter:
                break
            ### Get image full filename & Read Image: ###
            full_filename = os.path.join(dirpath, current_filename)
            print(full_filename)
            current_image = read_image_torch(full_filename, flag_convert_to_rgb=1, flag_normalize_to_float=1)

            tic()
            ### Put current image in wanted device (maybe going to gpu will be worth it because it will be that much faster?...maybe i should load all images together into a batch format...send them to gpu and then do it?): ###
            current_image = current_image.to(Generator_device)

            ### If this is not the first image then we can compare it to the previous one using Optical Flow: ###
            if image_index>0:
                if TVNet_layer is None:
                    TVNet_layer = get_TVNet_instance(current_image, number_of_iterations=TVNet_number_of_iterations, number_of_pyramid_scales=TVNet_number_of_pyramid_scales, device=Generator_device, flag_trainable=False)
                delta_x, delta_y, x2_warped = TVNet_layer.forward(current_image, previous_image, need_result=True)
            toc('Optical Flow')

            ### Assign previous image with current image: ###
            previous_image = current_image
            counter += 1













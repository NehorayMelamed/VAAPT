import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import *
from RapidBase.import_all import *


######################################################################################################################
### Paths: ###
#(1). General Paths:
project_path = path_fix_path_for_linux('/home/mafat/PycharmProjects/IMOD')
#(2). Train Images:


#(*). External Noise Source:
Camera_Noise_Folder = os.path.join(datasets_main_folder, '/KAYA_CAMERA_NOISE/noise')  #readout noise images

Inference_folder_name_addon = ''
######################################################################################################################




#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW') #simple BW images
Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/real_RGB_noisy_images')  #RGB images
Train_Images_Folder = os.path.join(datasets_main_folder, '/Vidmeo90K/vimeo_septuplet/sequences/00001') #videos in folders
Train_Images_Folder = os.path.join(datasets_main_folder, '/GoPro/train') #videos in folders where noisy and GT are in different folders seperated by prefixes in their names
Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/Image_SuperResolution/RealSR (Final)/Nikon/Train/2') #images in folders where noisy and GT are in the SAME folder seperated by prefixes in their names
Train_Images_Folder = os.path.join(datasets_main_folder, '/Example Videos/Beirut_BW_Original') #Beirut Drone Movie
### DataSets: ###

#(1). MIFSI 
def bla_test_1():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_image_frames_to_generate = 3
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.initial_crop_size = (256+0, 300+0)
    IO_dict.final_crop_size = (256, 270)
    Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW') #simple BW images
    train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AGN(Train_Images_Folder, IO_dict)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present Results: ###
    imshow_torch(outputs_dict.output_frames_original / 255)
    imshow_torch(outputs_dict.center_frame_original / 255)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)

#(*). RGB:
def bla_test_2():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_image_frames_to_generate = 5
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.initial_crop_size = (128+0, 150+0)
    IO_dict.final_crop_size = (128, 130)
    Train_Images_Folder = path_fix_path_for_linux(
        '/home/mafat/DataSets/DIV2K/real_RGB_noisy_images')  # simple BW images
    train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AGN(Train_Images_Folder, IO_dict)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present Results: ###
    imshow_torch(outputs_dict.output_frames_original / 255)
    imshow_torch(outputs_dict.center_frame_original / 255)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)

#(3). MIFSI with shifts and AGN
def bla_test_3():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_image_frames_to_generate = 5
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.initial_crop_size = (128+5, 130+5)
    IO_dict.final_crop_size = (128, 130)

    IO_dict.shift_size = 5
    IO_dict.blur_size = 0
    IO_dict.flag_add_shot_noise = True
    IO_dict.flag_add_per_pixel_readout_noise = True

    Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW') #simple BW images
    train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_Shifts_AGN(Train_Images_Folder, IO_dict)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present results: ###
    imshow_torch(outputs_dict.output_frames_original / 255)
    imshow_torch(outputs_dict.output_frames_noisy / 255)
    imshow_torch(outputs_dict.center_frame_original)
    imshow_torch(outputs_dict.center_frame_noisy)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)

    imshow_torch(outputs_dict.HR_center_frame_original)
    imshow_torch(outputs_dict.HR_output_frames_noisy)

#(4). MIFSI with shifts, directional blur and AGN:
#(*). directional blur no noise:
def bla_test_4():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_image_frames_to_generate = 5
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.initial_crop_size = (128+5, 130+5)
    IO_dict.final_crop_size = (128, 130)

    IO_dict.shift_size = 15
    IO_dict.blur_size = 15
    IO_dict.flag_add_shot_noise = True
    IO_dict.flag_add_per_pixel_readout_noise = True

    Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW') #simple BW images
    train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AGN(Train_Images_Folder, IO_dict)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present results: ###
    imshow_torch(outputs_dict.output_frames_original[2] / 255)
    imshow_torch(outputs_dict.output_frames_noisy[2] / 255)
    imshow_torch(outputs_dict.center_frame_original)
    imshow_torch(outputs_dict.center_frame_noisy)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)


# #(5). MIFSI with shifts, circular lowpass blur and AGN:
# IO_dict.blur_sigma = 5
# IO_dict.blur_kernel_size = 10
# Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW') #simple BW images
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AGN_BlurLowPass(Train_Images_Folder, IO_dict)
#

#(6). MIFSI with shifts, AGN and SR DSOTF:
#(*). super resolution no noise:
def bla_test_5():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_image_frames_to_generate = 5
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.initial_crop_size = (128+5, 128+5)
    IO_dict.final_crop_size = (128, 128)

    IO_dict.shift_size = 5
    IO_dict.blur_size = 0
    IO_dict.flag_add_shot_noise = True
    IO_dict.flag_add_per_pixel_readout_noise = True
    IO_dict.downsampling_factor = 4 #notice that downsampling_factor needs to divide final_crop_size
    IO_dict.flag_upsample_noisy_input_to_same_size_as_original = True

    Train_Images_Folder = os.path.join(datasets_main_folder, '/DIV2K/DIV2K/DIV2K_train_HR_BW') #simple BW images
    train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AGN(Train_Images_Folder, IO_dict)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present results: ###
    imshow_torch(outputs_dict.output_frames_original / 255)
    imshow_torch(outputs_dict.output_frames_noisy / 255)
    imshow_torch(outputs_dict.center_frame_original)
    imshow_torch(outputs_dict.center_frame_noisy)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)




#(8). MIFSI with shifts, AGN and SR LoadLRHR:
def bla_test_6():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_image_frames_to_generate = 5
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.initial_crop_size = (128+5, 128+5)
    IO_dict.final_crop_size = (128, 128)

    IO_dict.shift_size = 15
    IO_dict.blur_fraction = 1
    IO_dict.flag_add_shot_noise = True
    IO_dict.flag_add_per_pixel_readout_noise = True
    IO_dict.downsampling_factor = 1 #notice that downsampling_factor needs to divide final_crop_size
    IO_dict.flag_upsample_noisy_input_to_same_size_as_original = True

    IO_dict.string_pattern_to_search = '*'
    IO_dict.Corrupted_string_pattern_to_search = '*HR*'
    IO_dict.GT_string_pattern_to_search = '*HR*'

    Train_Images_Folder = path_fix_path_for_linux(
        '/home/mafat/DataSets/DIV2K/Image_SuperResolution/RealSR (Final)/Nikon/Train/2')  # images in folders where noisy and GT are in the SAME folder seperated by prefixes in their names
    train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_LoadCorruptedGT(Train_Images_Folder,
                                                                                             IO_dict)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present results: ###
    imshow_torch(outputs_dict.output_frames_original[1] / 255)
    imshow_torch(outputs_dict.output_frames_noisy[1] / 255)
    imshow_torch(outputs_dict.center_frame_original)
    imshow_torch(outputs_dict.center_frame_noisy)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)



#
#(9). VIF:
def bla_test_7():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_image_frames_to_generate = 5
    IO_dict.number_of_images_per_video_to_load = np.inf
    IO_dict.number_of_images_per_video_to_scan = 10
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.flag_to_RAM = False
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.initial_crop_size = (128*4+5, 128*4+5)
    IO_dict.final_crop_size = (128*4, 128*4)

    IO_dict.shift_size = 0
    IO_dict.blur_fraction = 0
    IO_dict.flag_add_shot_noise = False
    IO_dict.flag_add_per_pixel_readout_noise = False
    IO_dict.downsampling_factor = 1 #notice that downsampling_factor needs to divide final_crop_size
    IO_dict.flag_upsample_noisy_input_to_same_size_as_original = True

    IO_dict.string_pattern_to_search = '*'
    IO_dict.Corrupted_string_pattern_to_search = '*HR*'
    IO_dict.GT_string_pattern_to_search = '*HR*'

    Train_Images_Folder = path_fix_path_for_linux(
        '/home/mafat/DataSets/Vidmeo90K/vimeo_septuplet/sequences/00001')  # videos in folders
    train_dataset = TrainingCore.datasets.DataSet_Videos_In_Folders_AGN(Train_Images_Folder, IO_dict)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present results: ###
    imshow_torch(outputs_dict.output_frames_original[1] / 255)
    imshow_torch(outputs_dict.output_frames_noisy[1] / 255)
    imshow_torch(outputs_dict.center_frame_original)
    imshow_torch(outputs_dict.center_frame_noisy)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)



def bla_test_8():
    #(*). BW:
    IO_dict = get_default_IO_dict()
    IO_dict.max_number_of_images = np.inf  # max number of images to search/map
    IO_dict.number_of_images_per_video_to_load = np.inf
    IO_dict.number_of_images_per_video_to_scan = 10
    IO_dict.flag_how_to_concat = 'T'
    IO_dict.flag_to_RAM = False
    IO_dict.flag_to_BW_before_noise = False
    IO_dict.initial_crop_size = (128*4+5, 128*4+5)
    IO_dict.final_crop_size = (128*4, 128*4)

    IO_dict.transforms_dict.shift_size = 0
    IO_dict.transforms_dict.blur_fraction = 0
    IO_dict.transforms_dict.downsampling_factor = 1 #notice that downsampling_factor needs to divide final_crop_size
    IO_dict.transforms_dict.flag_upsample_noisy_input_to_same_size_as_original = True

    IO_dict.string_pattern_to_search = '*'
    IO_dict.Corrupted_string_pattern_to_search = '*blur/*'
    IO_dict.GT_string_pattern_to_search = '*sharp*'

    IO_dict.noise_dict.flag_add_shot_noise = False
    IO_dict.noise_dict.flag_add_per_pixel_readout_noise = False
    IO_dict.noise_dict.flag_add_external_image_NU = False
    IO_dict.noise_dict.flag_add_external_readout_noise = True
    IO_dict.noise_dict.readout_noise_external_image_path = Camera_Noise_Folder
    IO_dict.noise_dict.readout_noise_search_pattern = '*'
    IO_dict.noise_dict.NU_external_image_offset_path = Camera_Noise_Folder
    IO_dict.noise_dict.NU_external_image_offset_search_pattern = '*'
    IO_dict.noise_dict.NU_external_image_gain_path = Camera_Noise_Folder
    IO_dict.noise_dict.NU_external_image_gain_search_pattern = '*'

    Train_Images_Folder = os.path.join(datasets_main_folder, '/GoPro/train')
    train_dataset = TrainingCore.datasets.DataSet_Videos_In_Folders_LoadCorruptedGT(Train_Images_Folder, IO_dict)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=False)

    ### Dataset object output: ###
    outputs_dict = train_dataset[0]
    ### Dataloader output: ###
    outputs_dict_dataloader = EasyDict(train_dataloader.__iter__().__next__())

    ### Present results: ###
    imshow_torch_video(outputs_dict.output_frames_noisy/255)
    imshow_torch_video(outputs_dict.output_frames_original/255)
    imshow_torch(outputs_dict.output_frames_original[1] / 255)
    imshow_torch(outputs_dict.output_frames_noisy[1] / 255)
    imshow_torch(outputs_dict.center_frame_original)
    imshow_torch(outputs_dict.center_frame_noisy)
    plot_torch(outputs_dict.shift_x_vec)  # TODO: prepare shifts_x_to_center_image
    plot_torch(outputs_dict.shift_y_vec)
    imshow_torch(outputs_dict.optical_flow_delta_x)
    imshow_torch(outputs_dict.optical_flow_delta_y)
    imshow_torch(outputs_dict.optical_flow_GT)

bla_test_8()
#######################################################################################################################################




#######################################################################################################################################
### Get item for testing: ###
# things that we want to check:
# 1. basic functionality of all the dataset objects without "complications": initial_crop_size=final_crop_size, H=W,
# 2. initial_crop_size!=final_crop_size
# 3. H!=W

# output_dict = train_dataset[0]


#######################################################################################################################################


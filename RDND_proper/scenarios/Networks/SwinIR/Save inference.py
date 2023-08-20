from NEW_TASHTIT.import_Tashtit import *

from torch.utils.data import DataLoader
from RapidBase.TrainingCore.training_utils import *
from RapidBase.TrainingCore.callbacks import *
from RapidBase.TrainingCore.datasets import *
from RapidBase.TrainingCore.losses import Loss_Simple
from RapidBase.TrainingCore.pre_post_callbacks import *
from RapidBase.TrainingCore.pre_post_callbacks import PreProcessing_RLSP_Recursive
from RapidBase.TrainingCore.tensorboard_callbacks import *
from RapidBase.TrainingCore.lr import *
from RapidBase.TrainingCore.optimizers import *
from RapidBase.TrainingCore.tensorboard_callbacks import TB_Denoising_Recursive
from RapidBase.TrainingCore.trainer import *
from RapidBase.TrainingCore.clip_gradients import *
import RapidBase.TrainingCore.datasets
from RapidBase.TrainingCore.datasets import get_default_IO_dict
import RapidBase.TrainingCore.training_utils
import RapidBase.TrainingCore

from pathlib import Path
import time

from RDND_proper.models.SwinIR.SwinIR import SwinIR
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import *
from RDND_proper.models.BasicSRVSR.basicsr.archs.basicvsr_arch import *
from RDND_proper.models.Restormer.Restormer import Restormer



### Initialize Network State Dict: ###
Train_dict = EasyDict()

######################################################################################################################
### Paths: ###
#(1). General Paths


base_path = path_fix_path_for_linux('/media/mmm/DATADRIVE6/Omer')
project_path = path_fix_path_for_linux(os.path.join(base_path, 'IMOD'))
inference_path_master = path_fix_path_for_linux(os.path.join(base_path, 'Results', 'Inference'))
TensorBoard_path = path_fix_path_for_linux(os.path.join(base_path, 'Results', 'TensorBoard'))
Network_checkpoint_folder = path_fix_path_for_linux(os.path.join(base_path, 'Results', 'Model Checkpoints'))
#(2). Train Images:
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/test')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Div2K/DIV2K_train_HR_BW')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Drones')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Thermal_Images')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u/')
# Train_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'Div2K/DIV2K_train_HR_BW'))
Train_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'test'))


#(3). Test Images
# Test_Images_Folder = os.path.join(datasets_main_folder, '/Vidmeo90K/vimeo_septuplet/sequences/00001/0021')
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Div2K/DIV2K_train_HR_BW')
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Drones')
# Test_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/Thermal_Images')
# Train_Images_Folder = path_fix_path_for_linux('C:/Users/Omer Leibovitch/Desktop/Work/Data/words/a01/a01-000u/')
# Test_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'Div2K/DIV2K_train_HR_BW'))
Test_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'test'))

#RGB images
Inference_folder_name_addon = ''
######################################################################################################################


######################################################################################################################
### Train/Inference/Debug Parameters: ###
Train_dict.is_training = False
Train_dict.flag_do_validation = False
Train_dict.flag_movie_inference = False

### Number of Epochs / Iterations: ###
Train_dict.number_of_epochs = 1e9
Train_dict.max_total_number_of_batches = 1e9

### Frequencies/TensorBoard: ###
Train_dict.debug_step = 200
Train_dict.save_models_frequency = 200
Train_dict.validation_set_frequency = 100
Train_dict.tensorboard_train_frequency = 100
Train_dict.number_of_validation_samples = 10
Train_dict.flag_write_TB = True
Train_dict.flag_plot_grad_flow = False

### Flags: ###
Train_dict.flag_residual_learning = False
Train_dict.flag_clip_noisy_tensor_after_noising = False
Train_dict.flag_learn_clean_or_running_average = 'clean' # 0=clean, 1=running_average
###
######################################################################################################################


######################################################################################################################
### Save Network: ###
#######################
# (1.1). Name to Save checkpoint as when training new network:
# don't use the string 'Step' at any point in the prefix!!@#!@$@#
Train_dict.Network_checkpoint_prefix = 'FastDVDNet_SuperResolution_1'

# (1.2). Addition to Checkpoint (& TensorBoard) When Loading Checkpoint:
Train_dict.flag_continue_checkpoint_naming_convention = False

Train_dict.network_free_flow_description = ''
######################################################################################################################
## Random Seed: ###   (Comment In To Set Random Seed!)
TrainingCore.training_utils.set_seed(seed=42)

######################################################################################################################
### Load Network: ###
#######################
Train_dict.load_Network_filename = r''
######################################################################################################################


########################################
### New Training Or From Checkpoint: ###
# 'new', 'checkpoint'                # Use New Network or Checkpoint
Train_dict.flag_use_new_or_checkpoint_network = 'new'
Train_dict.flag_use_new_or_checkpoint_optimizer = 'new'

Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint = False
########################################

##########################################################################################
### Adjust Network Checkpoint string according to script variables/prefixes/postfixes: ###
Train_dict.main_file_to_save_for_later = os.path.realpath('__file__')
Train_dict = TrainingCore.training_utils.PreTraining_Aux(Train_dict, Network_checkpoint_folder, project_path)
##########################################################################################

#######################################################################################################################################
### IO: ###
IO_dict = get_default_IO_dict()
### Use Train/Test Split Instead Of Distinct Test Set: ###
IO_dict.flag_use_train_dataset_split = True #TODO: add possibility of splitting the train set as far as images but still use the IO_dict_test instead o the IO_dict_train
IO_dict.flag_only_split_indices = True #test_dataset is still defined seperately with its own IO_dict_test but split on indices base
IO_dict.train_dataset_split_to_test_factor = 0.05
### Loading Images: ###
IO_dict.string_pattern_to_search = '*'
IO_dict.HR_string_pattern_to_search = '*HR*'
IO_dict.LR_string_pattern_to_search = '*LR*'
IO_dict.flag_recursive = True
IO_dict.max_number_of_images = np.inf
IO_dict.number_of_image_frames_to_generate = 3 #5
IO_dict.flag_to_RAM = False
IO_dict.flag_noise_images_to_RAM = False
### Post Loading Stuff: ###
IO_dict.flag_to_BW_before_noise = True
IO_dict.flag_to_RGB_before_noise = False
IO_dict.flag_to_BW_after_noise = False
IO_dict.flag_to_RGB_after_noise = False
IO_dict.flag_how_to_concat = 'C'
### Noise Parameters: ###
IO_dict.SNR = np.inf  # mean~10, std~2(readout)+sqrt(10)(shot)=5, effective_sigma~255/mean*5~127.5 -> SNR~4. this is only an approximation to better condition network
IO_dict.sigma_to_dataset = 1.5 #1 / np.sqrt(IO_dict.SNR) * 255
IO_dict.SNR_to_model = IO_dict.SNR
### Camera Parameters: ### (*). INSTEAD OF ABOVE NOISE PARAMETERS!!!!
IO_dict.mean_gray_level_per_pixel = [200, 200]
IO_dict.electrons_per_gray_level = 14.53
IO_dict.photons_per_electron = 1
IO_dict.gray_levels_per_electron = 1 / IO_dict.electrons_per_gray_level
IO_dict.electrons_per_photon = 1 / IO_dict.photons_per_electron
### Shift / Directional-Blur Parameters: ###
IO_dict.shift_size = 0.5*4 # * 1e-9
IO_dict.blur_size = 0
### Gaussian Blur Parameters: ###
IO_dict.blur_number_of_channels = 1
IO_dict.blur_kernel_size = 3
IO_dict.blur_sigma = 1
### Super Resolution: ###
IO_dict.downsampling_factor = 4
### Universal Training Parameters: ###
IO_dict.batch_size = 1
#### Decide whether to load all inputs to gpu immediately after dataloader: ###
IO_dict.flag_all_inputs_to_gpu = True
### Cropping: ###
IO_dict.flag_crop_mode = 'random'
scale = 16
IO_dict.initial_crop_size = [int(64*scale)+5,int(64*scale)+5]
IO_dict.final_crop_size = [int(64*scale),int(64*scale)]
# IO_dict.initial_crop_size = [240,240]
# IO_dict.final_crop_size = [240,240]

### Test/Validation Dict: ###
IO_dict_test = EasyDict(IO_dict)
IO_dict_test.flag_crop_mode = 'center'
IO_dict_test.initial_crop_size = np.inf
IO_dict_test.final_crop_size = np.inf
IO_dict_test.mean_gray_level_per_pixel = [10,10]
#######################################################################################################################################

#######################################################################################################################################
## DataSet & DataLoader Objects: ###
####################################
### DataSets: ###
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_LoadLRHR_1(Train_Images_Folder, IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_LoadLRHR_1(Test_Images_Folder, IO_dict_test)
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
# test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
test_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolution_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

train_dataset_deblur = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolutionDeblur_DownSampleOnTheFly(Train_Images_Folder, IO_dict)
test_dataset_deblur = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage_AWGNNoise_Shifts_SuperResolutionDeblur_DownSampleOnTheFly(Test_Images_Folder, IO_dict_test)

train_dataloader_deblur = DataLoader(dataset=train_dataset, batch_size=IO_dict.batch_size, shuffle=True)
test_dataloader_deblur = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

### Number Of Elements: ###
IO_dict.num_mini_batches_trn = len(train_dataset) // IO_dict.batch_size
IO_dict.num_mini_batches_val = len(test_dataset) // 1
IO_dict.train_dataset_length = len(train_dataset)
IO_dict.test_dataset_length = len(test_dataset)
######################################################################################################################

### Model path: ###
model_path_L = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
model_path_M = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
model_path_LW = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth')
model_path_div2k = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')
model_path_restormer = os.path.join(base_path,'Pretrained Checkpoints/Restormer/motion_deblurring.pth')
model_path_edvr_sr = os.path.join(base_path,'Pretrained Checkpoints/EDVR/EDVR_L_x4_SR_REDS_official-9f5f5039.pth')
model_path_basicvsr = os.path.join(base_path,'Pretrained Checkpoints/BasicVSR/BasicVSR_REDS4-543c8261.pth')
model_path_iconvsr = os.path.join(base_path,'Pretrained Checkpoints/BasicVSR/IconVSR_Vimeo90K_BDx4-cfcb7e00.pth')
model_path_edvr_compressed_deblur = os.path.join(base_path,'Pretrained Checkpoints/EDVR/EDVR_L_deblurcomp_REDS_official-0e988e5c.pth') # motion blur + video compression artifacts.
model_path_edvr_deblur = os.path.join(base_path,'Pretrained Checkpoints/EDVR/EDVR_L_deblur_REDS_official-ca46bd8c.pth') # standard deblurring (motion blur).

### Get Models: ###
model_L = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv').cuda()

model_L.load_state_dict(torch.load(model_path_L)['params_ema'])

model_LW =  SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').cuda()
model_LW.load_state_dict(torch.load(model_path_LW)['params'])

model_M = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv').cuda()
model_M.load_state_dict(torch.load(model_path_M)['params_ema'])

model_div2k = SwinIR(upscale=4, in_chans=3, img_size=48, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                        num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv').cuda()

model_div2k.load_state_dict(torch.load(model_path_div2k)['params'])

model_edvr_sr = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur= False, with_tsa=True, hr_in=False).cuda()
model_edvr_sr.load_state_dict(torch.load(model_path_edvr_sr)['params'])

model_basicvsr = BasicVSR(num_feat=64, num_block=30, spynet_path = None).cuda()
model_basicvsr.load_state_dict(torch.load(model_path_basicvsr)['params'])

model_restormer = Restormer().cuda()
model_restormer.load_state_dict(torch.load(model_path_restormer)['params'])

model_edvr_deblur = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
model_edvr_deblur.load_state_dict(torch.load(model_path_edvr_deblur)['params'])

model_edvr_compressed_deblur = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
model_edvr_compressed_deblur.load_state_dict(torch.load(model_path_edvr_compressed_deblur)['params'])


model_edvr_sr = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur= False, with_tsa=True, hr_in=False).cuda()
model_edvr_sr.load_state_dict(torch.load(model_path_edvr_sr)['params'])

model_basicvsr = BasicVSR(num_feat=64, num_block=30, spynet_path = None).cuda()
model_basicvsr.load_state_dict(torch.load(model_path_basicvsr)['params'])

model_iconvsr = IconVSR(num_feat=64, num_block=15, keyframe_stride=5, temporal_padding=3, spynet_path=None, edvr_path=None).cuda()
model_iconvsr.load_state_dict(torch.load(model_path_iconvsr)['params'], strict=False) # They left me no choice

upsampling_layer = nn.Upsample(scale_factor=IO_dict.downsampling_factor, mode='bilinear', align_corners=None)


'''
### Get Batch: ###
for idx,batch in enumerate(train_dataloader):
    print ('Starting Batch {}'.format(idx+1))
    outputs_dict = batch
    y = outputs_dict['output_frames_original']/255
    x = outputs_dict['output_frames_noisy']/255
    x_deblur = upsampling_layer(x)
    # x = x.squeeze(2)

    #plt.show(block=True)

    ### Test Models: ###

    with torch.no_grad():
        start_L = time.time()
        x_sr_L = model_L(x.cuda())
        end_L = time.time()
        model_L_time = end_L - start_L
        torch.cuda.empty_cache()
        print ('SwinIR Large took {} seconds'.format(model_L_time))

        start_LW = time.time()
        x_sr_LW = model_LW(x.cuda())
        end_LW = time.time()
        model_LW_time = end_LW - start_LW
        torch.cuda.empty_cache()
        print('SwinIR Light took {} seconds'.format(model_LW_time))

        start_M = time.time()
        x_sr_M = model_M(x.cuda())
        end_M = time.time()
        model_M_time = end_M - start_M
        torch.cuda.empty_cache()
        print('SwinIR Medium took {} seconds'.format(model_M_time))

        start_div = time.time()
        x_sr_div = model_div2k(x.cuda())
        end_div = time.time()
        model_div_time = end_div - start_div
        torch.cuda.empty_cache()
        print('SwinIR div2k took {} seconds'.format(model_div_time))

        start_restormer = time.time()
        try:
            x_restormer = model_restormer(x_deblur.cuda())
        except:
            print ('Skipping')
        end_restormer = time.time()
        model_restormer_time = end_restormer - start_restormer
        torch.cuda.empty_cache()
        print('Restormer took {} seconds'.format(model_restormer_time))

        '''
        # start_edvr_compressed_deblur = time.time()
        # x_edvr_compressed_deblur = model_edvr_compressed_deblur(BW2RGB(x_deblur.unsqueeze(2)).cuda())
        # end_edvr_compressed_deblur = time.time()
        # model_edvr_compressed_deblur_time = end_edvr_compressed_deblur - start_edvr_compressed_deblur
        # torch.cuda.empty_cache()
        # print('EDVR compressed deblur took {} seconds'.format(model_edvr_compressed_deblur_time))
        #
        # start_edvr_deblur = time.time()
        # x_edvr_deblur = model_edvr_deblur(x_deblur.unsqueeze(2).cuda())
        # end_edvr_deblur = time.time()
        # model_edvr_deblur_time = end_edvr_deblur - start_edvr_deblur
        # torch.cuda.empty_cache()
        # print('EDVR deblur took {} seconds'.format(model_edvr_deblur_time))
        #
        #
        # start_edvr_sr = time.time()
        # x_edvr_sr = model_edvr_sr(BW2RGB(x.unsqueeze(2).cuda()))
        # end_edvr_sr = time.time()
        # edvr_sr_time = end_edvr__sr - start_edvr_sr
        # torch.cuda.empty_cache()
        # print('EDVR SR took {} seconds'.format(edvr_sr_time))
        #
        # start_basicvsr = time.time()
        # x_basicvsr = model_basicvsr(BW2RGB(x.unsqueeze(2).cuda()))
        # end_basicvsr = time.time()
        # basicvsr_time = end_basicvsr - start_basicvsr
        # torch.cuda.empty_cache()
        # print('BasicVSR took {} seconds'.format(basicvsr_time))
        #
        # start_iconvsr = time.time()
        # x_iconvsr = model_iconvsr(BW2RGB(x.unsqueeze(2).cuda()))
        # end_iconvsr = time.time()
        # iconvsr_time = end_iconvsr - start_iconvsr
        # torch.cuda.empty_cache()
        # print('IconVSR took {} seconds'.format(iconvsr_time))
         '''


    ### Save Results: ###

    # curr_dir = "Swin Results/{}".format(idx+1)
    curr_dir = "../Pretrained Inference Results On DIV2K-HR-BW/{}".format(idx+1)
    Path(curr_dir).mkdir(parents=True, exist_ok=True)

    torchvision.utils.save_image((BW2RGB(RGB2BW(y))), curr_dir+'/Original.png')
    torchvision.utils.save_image((BW2RGB(RGB2BW(x))), curr_dir+'/Downsample.png')
    torchvision.utils.save_image((BW2RGB(RGB2BW(x_sr_L))), curr_dir+'/Swin Large.png')
    torchvision.utils.save_image((BW2RGB(RGB2BW(x_sr_M))), curr_dir+'/Swin Medium.png')
    torchvision.utils.save_image((BW2RGB(RGB2BW(x_sr_LW))), curr_dir+'/Swin Light.png')
    torchvision.utils.save_image((BW2RGB(RGB2BW(x_sr_div))), curr_dir+'/Swin div2k.png')
    torchvision.utils.save_image((BW2RGB(RGB2BW(x_restormer))), curr_dir + '/Restormer.png')
    '''
    # torchvision.utils.save_image((BW2RGB(RGB2BW(x_edvr_compressed_deblur))), curr_dir + '/EDVR compressed deblur.png')
    # torchvision.utils.save_image((BW2RGB(RGB2BW(x_edvr_deblur))), curr_dir + '/EDVR deblur.png')
    # torchvision.utils.save_image((BW2RGB(RGB2BW(x_edvr_sr))), curr_dir + 'EDVR SR.png')
    #
    # torchvision.utils.save_image((BW2RGB(RGB2BW(x_basicvsr))), curr_dir + '/BasicVSR.png')
    # torchvision.utils.save_image((BW2RGB(RGB2BW(x_iconvsr))), curr_dir+'/IconVSR.png')
    '''



# imshow_torch((BW2RGB(RGB2BW(x))))
# imshow_torch((BW2RGB(RGB2BW(y))))


#Fix other networks
# add datasets
# add metrics
# make modular/ fix repeating code
'''
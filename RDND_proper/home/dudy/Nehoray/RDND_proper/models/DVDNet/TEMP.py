import torch
import torch.nn as nn
import numpy as np

from RapidBase.import_all import *
import RapidBase.TrainingCore.datasets
from RDND_proper.models.DVDNet.DVDNet import *


### Test: ###
args = EasyDict()

### Model Parameters: ###
NUM_IN_FRAMES = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

### Paths: ###
args.model_spatial_file = r'/home/mafat\PycharmProjects\IMOD\models\DVDNet\dvdnet/model_spatial.pth'
args.model_temp_file = r'/home/mafat\PycharmProjects\IMOD\models\DVDNet\dvdnet/model_temp.pth'
args.test_path = r'/home/mafat/DataSets/DataSets/Div2K/Official_Test_Images/Still_Images/PSNR_100\NIND_droid_ISO200'  # path to sequence to denoise. TODO: understand how sequence should be loaded here
args.save_path = r'/home/mafat\Pytorch_Checkpoints\Checkpoints\Results'
args.save_path = str.replace(args.save_path, '//', '/').replace('\\','/')
args.model_spatial_file = str.replace(args.model_spatial_file, '//', '/').replace('\\','/')
args.model_temp_file = str.replace(args.model_temp_file, '//', '/').replace('\\','/')
args.test_path = str.replace(args.test_path, '//', '/').replace('\\','/')
args.suffix = ''  # suffix to add to output name ????

### Parameters: ###
args.max_num_fr_per_seq = 10
args.noise_sigma = 25  # TODO: make possible to switch smoothly from noise sigma to PSNR
args.save_noisy = True
args.no_gpu = False

args.noise_sigma /= 255.
args.cuda = not args.no_gpu and torch.cuda.is_available()

### If save_path does not exist, create it: ###
if not os.path.exists(args['save_path']):
    os.makedirs(args['save_path'])
logger = init_logger_test(args['save_path'])

### Sets data type according to CPU or GPU modes: ###
if args['cuda']:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### Create models: ####
model_DVDNet = DVDNet(temp_psz=NUM_IN_FRAMES, mc_algo=MC_ALGO)

### Load saved weights from disk: ###
state_spatial_dict = torch.load(args['model_spatial_file'], map_location=torch.device('cuda'))  #TODO: change when i will have CUDA
state_temp_dict = torch.load(args['model_temp_file'], map_location=torch.device('cuda'))

### Models/Modules were save in DataParallel -> Remove it / Take care of it: ###
state_spatial_dict = remove_dataparallel_wrapper(state_spatial_dict)
state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
model_DVDNet.model_spatial.load_state_dict(state_spatial_dict)
model_DVDNet.model_temporal.load_state_dict(state_temp_dict)

### Load to GPU: ###
model_DVDNet = model_DVDNet.cuda()
model_DVDNet.model_spatial = model_DVDNet.model_spatial.cuda()
model_DVDNet.model_temporal = model_DVDNet.model_temporal.cuda()

### Sets the model in evaluation mode (e.g. it removes BN): ###
model_DVDNet.model_spatial.eval()
model_DVDNet.model_temporal.eval()

### Load Image Dataset: ###
# train_dataset = TrainingCore.datasets.Dataset_MultipleImagesFromSingleImage(root_path=r'/home/mafat/DataSets/DataSets/Div2K/Official_Test_Images/Original_Images',
#                                                                     number_of_image_frame_to_generate=NUM_IN_FRAMES,
#                                                                     base_transform=None, batch_transform=None,
#                                                                     image_loader=ImageLoaderCV,
#                                                                     max_number_of_images=np.inf,
#                                                                     crop_size=1000,
#                                                                     flag_to_RAM=True, flag_recursive=False, flag_normalize_by_255=True, flag_crop_mode='center',
#                                                                     flag_explicitely_make_tensor=False, allowed_extentions=IMG_EXTENSIONS, flag_to_BW=True,
#                                                                     flag_base_tranform=False, flag_turbulence_transform=False, Cn2=5e-13,
#                                                                     flag_batch_transform=False,
#                                                                     flag_how_to_concat='T')
# train_dataset[0]
train_dataset = TrainingCore.datasets.DataSet_Videos_In_Folders(root_folder=r'/home/mafat/DataSets/DataSets/Div2K/DIV2K_train_HR_BW_Crops_Videos',
                                                        transform_per_image=None,
                                                         transform_on_all_images=None,
                                                         image_loader=ImageLoaderCV,
                                                         number_of_images_per_video=100,  #TODO: add possibility to randomize start index from a possibly long video
                                                         max_number_of_videos=np.inf,
                                                         crop_size=100,
                                                         flag_to_RAM=True,
                                                         flag_explicitely_make_tensor=True,
                                                         flag_normalize_by_255=True,
                                                         flag_how_to_concat='T',
                                                         flag_return_torch_on_load=False,
                                                         flag_to_BW=False)
### DataLoaders: ###
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)


### Pass Full Sequence / Video Through Model: ###
input_tensor = train_dataloader.__iter__().__next__()['output_frames'].cuda()
input_tensor_noisy = input_tensor + args.noise_sigma*torch.randn_like(input_tensor)
output_frames = []
for frame_index in np.arange(0,input_tensor_noisy.shape[1]-NUM_IN_FRAMES+1):
    current_input_tensor = input_tensor_noisy[:,frame_index:frame_index+NUM_IN_FRAMES,:,:,:]
    current_input_tensor = torch.cat((current_input_tensor, current_input_tensor, current_input_tensor), 2)  # BW->RGB
    with torch.no_grad():
        output_tensor = model_DVDNet.forward(current_input_tensor, noise_std=torch.Tensor([args.noise_sigma]).cuda())
        output_frames.append(torch_to_numpy(output_tensor.cpu()))

### Get Metrics: ###
from RDND_proper.scenarios.main_check_metrics import get_metrics_image_pair
cleaned_image = output_frames[0][0,:,:,0]
noisy_image = input_tensor_noisy[0,2,0,:,:].cpu().numpy()
original_image = input_tensor[0,2,0,:,:].cpu().numpy()
imshow_new(cleaned_image)
imshow_new(noisy_image)
imshow_new(original_image)
output_metrics_noisy = get_metrics_image_pair(noisy_image, original_image)
output_metrics_cleaned = get_metrics_image_pair(cleaned_image, original_image)



### Pass Single Input Through Model: ###
input_tensor = train_dataloader.__iter__().__next__()['output_frames'].cuda()
input_tensor = torch.cat((input_tensor,input_tensor,input_tensor), 2) #BW->RGB
with torch.no_grad():
    output_tensor = model_DVDNet.forward(input_tensor, noise_std=torch.Tensor([args.noise_sigma]).cuda())



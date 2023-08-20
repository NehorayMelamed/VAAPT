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
from RDND_proper.models.SwinIR.SwinIR import SwinIR
from RDND_proper.models.Restormer.Restormer import Restormer
from RDND_proper.models.BasicSRVSR.basicsr.archs.edvr_arch import *
from RDND_proper.models.BasicSRVSR.basicsr.archs.basicvsr_arch import *

######################################################################################################################
### Paths: ###

#(1). General Paths
base_path = path_fix_path_for_linux('/media/mmm/DATADRIVE6/Omer')
project_path = path_fix_path_for_linux(os.path.join(base_path, 'IMOD'))
inference_path_master = path_fix_path_for_linux(os.path.join(base_path, 'Inference'))
TensorBoard_path = path_fix_path_for_linux(os.path.join(base_path, 'TensorBoard'))
Network_checkpoint_folder = path_fix_path_for_linux(os.path.join(base_path, 'Model Checkpoints'))
#(2). Train Images:
Train_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'omers_stuff'))
# (3). Test Images
Test_Images_Folder = path_fix_path_for_linux(os.path.join(base_path, 'Data', 'omers stuff'))

def save_raw_in_jpg(Train_Images_Folder):

    files = os.listdir(Train_Images_Folder)
    img_size = (256,512)
    for file in files:
        img = np.fromfile(os.path.join(Train_Images_Folder,file),dtype = 'uint8').reshape(img_size)
        IMG = Image.fromarray(img)
        IMG.save(os.path.join(base_path, 'Data/Filmed cars',file.split('.')[0] + '.jpg'))
        print ('done' + file)

def load_image(infilename):
    """This function loads an image into memory when you give it
       the path of the image
    """
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="float32")
    return data

def get_input(video_dir, start_idx = 0, num_frames = 5, stride = 1, move = True, do_BW2RGB = True, crop_size = (256,400)):
    flag = 1 * move
    files = os.listdir(video_dir)
    t = transforms.CenterCrop(crop_size)
    pics = []
    for i in range (num_frames):
        img = load_image(os.path.join(video_dir, files[start_idx+ i*flag*stride]))
        img = torch.tensor(img) / 255
        if do_BW2RGB:
            img = BW2RGB(img)
        img  = t(img)
        pics.append(img)

    pics = torch.stack(pics)
    pics = torch.unsqueeze(pics,0)
    return pics

def get_rgb_input(video_dir, start_idx = 0, num_frames = 5, stride = 1, move = True, do_RGB2BW = False, crop_size = (256,400)):

    flag = 1 * move
    files = os.listdir(video_dir)
    t = transforms.CenterCrop(crop_size)
    pics = []
    for i in range(num_frames):
        img = load_image(os.path.join(video_dir, files[start_idx + i * flag * stride]))
        img = torch.tensor(img) / 255
        img = img.transpose(0, 2).transpose(1, 2)
        if do_RGB2BW:
            img = RGB2BW(img)
        img = t(img)
        pics.append(img)

    pics = torch.stack(pics)
    pics = torch.unsqueeze(pics, 0)
    return pics


plt.rcParams["figure.figsize"] = (20,20)

video_dir = os.path.join(base_path, 'Data/Filmed cars')
crop_size = (256,400)
num_frames = 9
do_BW2RGB = True
move = True
stride = 1
start_idx = 340

pics = get_input(video_dir,start_idx,num_frames, stride, move,do_BW2RGB, crop_size)
BW_pics = get_input(video_dir,start_idx+1 ,3, stride, move, False, crop_size)
pics_not_move = get_input(video_dir,start_idx+5//2,5, 1, False,do_BW2RGB, crop_size)

preds = []
for i in range (5):
    with torch.no_grad():
        resdiv = modeldiv(pics[:,0+i:5+i,:,:,:].cuda())
        preds.append(resdiv)
preds = torch.stack(preds,dim=1)

pics = pics[:,2:7,:,:,:]


with torch.no_grad():
    res = model(pics.cuda())
    resdiv = modeldiv(pics.cuda())
    res_swin = model_swin(BW_pics.cuda())
    resdiv_not_move = modeldiv(pics_not_move.cuda())
    one_img = model_swin(pics[0][2].unsqueeze(0).cuda())
    preds_img = modeldiv(preds.cuda())

imshow_torch(pics[0][2], 0, 'Original')
imshow_torch(res, 0, 'ALPR Model')
imshow_torch(resdiv, 0, 'Div2K Model')
imshow_torch(res_swin, 0, 'SwinIR Model')
imshow_torch(resdiv_not_move, 0, 'Div2K Model not move')
imshow_torch(preds_img, 0, 'Div2K ensemble')

plt.show(block=True)

video_dir = os.path.join(base_path, 'Data/car/newcar')
crop_size = (256,400)
num_frames = 5
do_RGB2BW = False
move = True
stride = 1
start_idx = 100



pics = get_rgb_input(video_dir,start_idx,num_frames, stride, move,do_RGB2BW , crop_size)
BW_pics = get_rgb_input(video_dir,start_idx+1 ,3, stride, move, True, crop_size).squeeze(2)
pics_not_move = get_rgb_input(video_dir,start_idx,num_frames, 1, False,do_RGB2BW , crop_size)

with torch.no_grad():
    res = model(pics.cuda())
    resdiv = modeldiv(pics.cuda())
    res_swin = model_swin(BW_pics.cuda())
    resdiv_not_move = modeldiv(pics_not_move.cuda())
    one_img = model_swin(pics[0][2].unsqueeze(0).cuda())

imshow_torch(pics[0][2], 0, 'Original')
imshow_torch(res, 0, 'ALPR Model')
imshow_torch(resdiv, 0, 'Div2K Model')
imshow_torch(res_swin, 0, 'SwinIR Model')
imshow_torch(one_img, 0, 'SwinIR Model one img')
imshow_torch(resdiv, 0, 'Div2K Model not move')

plt.show(block=True)


imshow_torch(pics[0][0])
imshow_torch(pics[0][1])
imshow_torch(pics[0][2])
imshow_torch(pics[0][3])
imshow_torch(pics[0][4])
plt.show(block=True)

## Get Models: ###
model = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
 # motion blur + video compression artifacts.

model_path = os.path.join(base_path,'Model Checkpoints/EDVR (Deblur compress pth) ALPR [400,256] SR-Deblur/'
                                    'EDVR (Deblur compress pth) ALPR [400,256] SR-Deblur_TEST1_Step41000.tar')
model.load_state_dict(torch.load(model_path)['model_state_dict'])


modeldiv = EDVR(
        num_in_ch=3, num_out_ch=3,
        num_feat=128,
        num_frame=5, deformable_groups=8,
        num_extract_block=5, num_reconstruct_block=40,
        with_predeblur=True, with_tsa=True, hr_in=True).cuda()
 # motion blur + video compression artifacts.

modeldiv_path = os.path.join(base_path,'Model Checkpoints/EDVR (Deblur compress pth) Div2K [320,320] SR-Deblur/'
                                    'EDVR (Deblur compress pth) Div2K [320,320] SR-Deblur_TEST1_Step14000.tar')
modeldiv.load_state_dict(torch.load(modeldiv_path)['model_state_dict'])






# model_path = os.path.join(base_path,'Model Checkpoints/BasicVSR (Vimeo90K BD pth) ALPR [400,256] SR/'
#                                          'BasicVSR (Vimeo90K BD pth) ALPR [400,256] SR_TEST1_Step44000.tar')
#
#
# ### Get Models: ###
# model = BasicVSR(num_feat=64, num_block=30, spynet_path = None).cuda()
# model.load_state_dict(torch.load(model_path)['model_state_dict'])
# Train_dict.Network_checkpoint_step = 0


model_path = os.path.join(base_path,'Pretrained Checkpoints/SwinIR/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
model_swin = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv').cuda()
model_swin.load_state_dict(torch.load(model_path)['params_ema'])




### To do:

#1 Test time augmentations - S2 AND S2+ (Self ebsemble and 4 rotations)

#2 EDVR training from scratch, learn how to train

# fastdvdnet pretrained, fastdvdnet SR with pixelshufle



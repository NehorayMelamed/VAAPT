# from RapidBase.import_all import *
# import RapidBase.TrainingCore.training_utils
# 

# from RapidBase.Models.Tiny_MagicNet import *
# from RapidBase.Models.Unets import *
# from RDND_proper.models.KAIR_FFDNet_and_more.models.network_ffdnet import FFDNet_dudy2


### Models: ###
from RDND_proper.scenarios.Deployment.Deployment_Temp import *
model_SmallUnet = Unet(5)
# model_Magic = MagIC_I_float(in_channels=5, out_channels=1, num_compressed_channels_1x=6, num_compressed_channels_4x=12)
model_FFFDNet = FFDNet_dudy2(in_nc=5, out_nc=1, nc=64, nb=15, act_mode='R')
# r18 = torchvision.models.resnet18(pretrained=True)

### Define Input: ###
input_tensor = torch.randn((1,5,512,512))



#####################################################################################################################
### Use TorchJit: ###
model_SmallUnet_scripted = torch.jit.script(model_SmallUnet)
# model_Magic_scripted = torch.jit.script(model_Magic)
model_FFFDNet_scripted = torch.jit.script(model_FFFDNet)
# r18_scripted = torch.jit.script(r18)

### Check Outputs: ###
print((model_SmallUnet_scripted(input_tensor) - model_SmallUnet(input_tensor)).abs().mean())
# print((model_Magic_scripted(input_tensor) - model_Magic(input_tensor)).abs().mean())
print((model_FFFDNet_scripted(input_tensor) - model_FFFDNet(input_tensor)).abs().mean())

### Save Scripted Models: ###
folder_path = '/home/mafat/PycharmProjects/IMOD/Deployment'
model_SmallUnet_scripted.save(os.path.join(folder_path, 'small_unet_scripted.pt'))
# model_Magic_scripted.save(os.path.join(folder_path, 'magic_unet_scripted.pt'))
model_FFFDNet_scripted.save(os.path.join(folder_path, 'ffdnet_scripted.pt'))
# r18_scripted.save(os.path.join(folder_path,'r18_scripted.pt'))





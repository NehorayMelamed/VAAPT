#GAN possible template:

#Imports:
#(1). Auxiliary:
import matplotlib
import argparse
import os
import numpy as np
import math
from PIL import Image
import glob
import random
import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int,importlib
from int_range import *
from int_arange import *
from mat_range import *
from matlab_arange import *
from my_linspace import *
from my_linspace_int import *
import get_center_number_of_pixels
from get_center_number_of_pixels import *
import get_speckle_sequences
from get_speckle_sequences import *
import tic_toc
from tic_toc import *
import search_file
from search_file import *
import show_matrices_video
from show_matrices_video import *
import klepto_functions
from klepto_functions import *
import ctypes  # An included library with Python install.
def message_box(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
import pymsgbox
from pymsgbox import alert as alert_box
from pymsgbox import confirm as confirm_box
from pymsgbox import prompt as prompt_box
#(2). TorchVision (add FastAi stuff which are much faster and more intuitive as far as augmentations)
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import vgg19
#(3). Torch Utils:
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
#(4). Torch NN:
import torch.nn as nn
import torch.nn.functional as F
import torch


#(5). Reminders on how to execute:
# load_variables_from_klepto_file(file_name):
# save_variables_to_klepto_file(file_name,variables_dict)
# update_klepto(file_name,variables_dict)
# save_tic(baseline_name_finale)
# save_toc(base_finale, post_finale)
# alert_box('alert text','alert title')
# response = pymsgbox.prompt('What is your name?')
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
#Auxiliary Functions:
class dictionary_to_class_attributes:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)

#Network Weights initialization function (TODO: check if there are other things people suggest):
def weights_init_normal(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)


#Feature Extractor Model To Be Used As Distance Penalty on Features and Not Pixels (Contexual Loss):
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        #Could Probably Use Something Else Which Is Quicker
        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer (TODO: totally a HyperParameter!!@##)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out


#Residual Block Model (Later to be replaced with a Residual in a Residual Block without the BN):
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        #Convolutional block:
        conv_block = [  nn.Conv2d(in_features, in_features, 3, 1, 1),
                        nn.BatchNorm2d(in_features),
                        nn.ReLU(),
                        nn.Conv2d(in_features, in_features, 3, 1, 1),
                        nn.BatchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        #Conv Block with residual connection:
        return x + self.conv_block(x)


#ImageDataset object inheriting from Dataset:
class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms=None, hr_transforms=None):
        self.lr_transform = transforms.Compose(lr_transforms)
        self.hr_transform = transforms.Compose(hr_transforms)

        #root is the path that contains all the images files:
        self.files = sorted(glob.glob(root + '/*.*'))


    #OverWriting the __getitem__  function to retrieve Image from a file.
    #TODO: understand what is the best and quickest way to get it (maybe FastAI) and what's the best way to do Augmentations like TensorFlow puts the emphasis on:
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {'lr': img_lr, 'hr': img_hr}

    def __len__(self):
        return len(self.files)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Create a Folder called images in your base folder:
os.makedirs('images', exist_ok=True)
# os.getcwd() - Auxiliary: get current working directory to view created folder



"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""


#Script Parameters (usually in the form of ArgParse but here i write them down explicitly for ease of debugging):
exec(save_tic(1));
#(1). Dataset and Input Parameters:
dataset_name = "img_align_celeba" #TODO: change this to my own dataset:
HR_image_height = 256;
HR_image_width = 256;
image_number_of_channels = 3; #number of image image_number_of_channels (1 for BW -> change to 3 for rgb)
#(2). Training Parameters:
number_of_epochs = 200; #number of training epochs
batch_size =  64;
lr = 0.0002; #adam learning rate (try SGD with lr cycles)
epoch_to_start_lr_decay = 100;
b1 = 0.5; #decay of first order momentum of gradient
b2 = 0.999; #decay of first order momentum of gradient
number_of_cpu_threads_in_BG = 8; #number of cpu threads to use during batch generation
latent_space_dimensionality = 100; #dimensionality of the latent space
#(3). Model Checking Parameters:s
number_of_batches_to_spot_check_generator = 400; #interval between image samples (??)
number_of_batches_to_checkpoint_model = -1;
#(4). Model Loading:
epoch_to_start_training_from = 0;
exec(save_toc(1,1)); # (saves a dictionary variable called saved_dictionary1)
OPT = dictionary_to_class_attributes(saved_dictionary1);



#Preliminary stuff:
# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(OPT.HR_image_height / 2**4), int(OPT.HR_image_width / 2**4)
patch = (OPT.batch_size, 1, patch_h, patch_w)

#Cude if possible:
cuda = True if torch.cuda.is_available() else False
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, number_of_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        #Again...no BN at the first layer... what about instance normalization again!@$?@%$
        self.conv1 = nn.Sequential(
                                    nn.Conv2d(in_channels, 64, 9, 1, padding=4), #Notice the large kernel size of 9 (i should try several filter sizes concatenated)@!$@#%
                                                                         #TODO: Again... all of these parameters like number of filter, kernel size etc' are HyperParameters
                                                                         #TODO: Notice the padding input parameter... it's equal to kernel_size//2 !@!##$#@$#
                                    nn.ReLU(inplace=True)
                                  )

        # Residual blocks Stacked on top of each other:
        res_blocks = []
        for _ in range(number_of_residual_blocks):
            res_blocks.append(ResidualBlock(in_features=64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))

        # Upsampling layers
        # PixelShuffle: an efficient way to effectively do subpixel convolution with a stride of 1/r.
        #               it take elements of shape (batch_size, r^2*number_of_channels, H, W) and outputs a tensor of shape (batch_size, number_of_channels, r*H, r*W).
        upsampling = []
        for out_features in range(2):
            upsampling += [ nn.Conv2d(64, 256, 3, 1, 1),
                            nn.BatchNorm2d(256),
                            nn.PixelShuffle(upscale_factor=2), #upsampled factor = r from the above equation
                            nn.ReLU(inplace=True)]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, 9, 1, 4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        #Basic discriminator block - Conv2d -> BN -> LeakyRelU
        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters)) #Wait...What? How come i need to insert number of output filters to BatchNorm2d??!?
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        #Construct Discriminator Network by concatenating discriminator blocks:
        layers = []
        in_filters = in_channels
        for out_filters, stride, normalize in [ (64, 1, False),
                                                (64, 2, True),
                                                (128, 1, True),
                                                (128, 2, True),
                                                (256, 1, True),
                                                (256, 2, True),
                                                (512, 1, True),
                                                (512, 2, True),]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters


        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Loss function
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize generator and discriminator, assign them to gpu if available:
# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator()
feature_extractor = FeatureExtractor()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize weights a certain way
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Weight Initialization or Previous Model Loading and continuing to train from there:
if OPT.epoch_to_start_training_from != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/generator_%d.pth'))     #what is this way of specifying the path string?   generator_%d.pth ?????
    discriminator.load_state_dict(torch.load('saved_models/discriminator_%d.pth'))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Optimizers (maybe add SGD with learning rate cycles etc' from FastAI):
optimizer_G = torch.optim.Adam(generator.parameters(), lr=OPT.lr, betas=(OPT.b1, OPT.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=OPT.lr, betas=(OPT.b1, OPT.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
batch_shape_LR = Tensor(OPT.batch_size, OPT.image_number_of_channels, OPT.HR_image_height//4, OPT.HR_image_width//4)
batch_shape_HR = Tensor(OPT.batch_size, OPT.image_number_of_channels, OPT.HR_image_height, OPT.HR_image_width)
# Adversarial ground truths
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Transforms & DataLoader
#(1). Transforms:
lr_transforms = [   transforms.Resize((OPT.HR_image_height//4, OPT.HR_image_height//4), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

hr_transforms = [   transforms.Resize((OPT.HR_image_height, OPT.HR_image_height), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
#(2). DataLoader:
# dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, lr_transforms=lr_transforms, hr_transforms=hr_transforms),
#                         batch_size=OPT.batch_size,
#                         shuffle=True,
#                         num_workers=OPT.number_of_cpu_threads_in_BG)
dataloader = DataLoader(ImageDataset("C:/Users\dkarl\PycharmProjects\dudy\KERAS\DIV2K_train_HR/", lr_transforms=lr_transforms, hr_transforms=hr_transforms),
                        batch_size=OPT.batch_size,
                        shuffle=True,
                        num_workers=OPT.number_of_cpu_threads_in_BG)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# ----------
#  Training
for current_epoch in range(OPT.number_of_epochs):
    for current_batch_index, current_image_batch in enumerate(dataloader):

        # Configure model input
        # TODO: hmmm... this is a weird way to do it... get a numpy array and put it into a tensor to be passed through the Network by tensor_variable.copy_(numpy_array)...
        # TODO: is this efficient?!?
        real_image_batch_LR = Variable(batch_shape_LR.copy_(current_image_batch['lr']))
        real_image_batch_HR = Variable(batch_shape_HR.copy_(current_image_batch['hr']))

        # Adversarial ground truths
        valid_image_label_value = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake_image_label_value = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        ########################
        ###  Train Generator ###
        optimizer_G.zero_grad()
        # Generate a high resolution image from low resolution input
        generated_image_batch_HR = generator(real_image_batch_LR)
        # Adversarial loss
        D_output_generated_image_batch_HR_validity = discriminator(generated_image_batch_HR)
        loss_GAN = criterion_GAN(D_output_generated_image_batch_HR_validity, valid_image_label_value) #Criterion loss is MSE loss to measure difference between label and prediction
        # Content loss
        fake_image_batch_extracted_features = feature_extractor(generated_image_batch_HR)
        real_image_batch_extracted_features = Variable(feature_extractor(real_image_batch_HR).data, requires_grad=False)
        loss_content =  criterion_content(fake_image_batch_extracted_features, real_image_batch_extracted_features) #Content is L1 to measure difference between high level layer outputs between real and fake feaures
        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN
        loss_G.backward()
        optimizer_G.step()


        ############################
        ###  Train Discriminator ###
        optimizer_D.zero_grad()
        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(real_image_batch_HR), valid_image_label_value)
        loss_fake = criterion_GAN(discriminator(generated_image_batch_HR.detach()), fake_image_label_value)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()


        ####################
        ###  LOG PROCESS ###
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
              (current_epoch, OPT.number_of_epochs, i, len(dataloader),
               loss_D.item(), loss_G.item()))

        batches_done = current_epoch * len(dataloader) + i
        if batches_done % OPT.number_of_batches_to_spot_check_generator == 0:
            # Save image sample
            save_image(torch.cat((generated_image_batch_HR.data, real_image_batch_HR_.data), -2),
                       'images/%d.png' % batches_done, normalize=True)

    if OPT.number_of_batches_to_checkpoint_model != -1 and current_epoch % OPT.number_of_batches_to_checkpoint_model == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % current_epoch)
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % current_epoch)







































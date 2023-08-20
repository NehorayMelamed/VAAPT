#GAN possible template:

#Imports:
#(1). Auxiliary:
import argparse
import os
import numpy as np
import math
from PIL import Image
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
from torchvision.models import resnet18
#(3). Torch Utils:
from torch.utils.data import DataLoader
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
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Create a Folder called images in your base folder:
os.makedirs('images', exist_ok=True)
# os.getcwd() - Auxiliary: get current working directory to view created folder


#Script Parameters (usually in the form of ArgParse but here i write them down explicitly for ease of debugging):
exec(save_tic(1));
#(1). Dataset and Input Parameters:
dataset_name = "img_align_celeba" #TODO: change this to my own dataset:
image_height = 256;
image_width = 256;
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
#(3). Model Checking Parameters:
number_of_batches_to_spot_check_generator = 400; #interval between image samples (??)
number_of_batches_to_checkpoint_model = -1;
#(4). Model Loading:
epoch_to_start_training_from = 0;
exec(save_toc(1,1)); # (saves a dictionary variable called saved_dictionary1)
OPT = dictionary_to_class_attributes(saved_dictionary1);


#Preliminary stuff:
#(1). make folders to put outputs
os.makedirs('images/%s' % OPT.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % OPT.dataset_name, exist_ok=True)
#(2). Auxiliary:
input_image_shape = (OPT.image_number_of_channels, OPT.image_size, OPT.image_size)
cuda = True if torch.cuda.is_available() else False
####################################################################################################################################################################################################################################################################################################################################################






####################################################################################################################################################################################################################################################################################################################################################
#DataSet Definition:
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        #Get Two Images From A single image file by cropping it down the middle giving the left-hand side and the right-hand side as two images!:
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)


# Configure dataloaders
transforms_ = [ transforms.Resize((OPT.image_height, OPT.image_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, transforms_=transforms_),
                        batch_size=OPT.batch_size, shuffle=True, num_workers=OPT.number_of_cpu_threads_in_BG)

val_dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, transforms_=transforms_, mode='val'),
                            batch_size=10, shuffle=True, num_workers=1)
####################################################################################################################################################################################################################################################################################################################################################




####################################################################################################################################################################################################################################################################################################################################################
# Auxiliary Classes/Blocks for the networks:
#(1). Downsampling UNet
class UNetDown(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        #DownSampling by stride=2 convolution:
        layers = [nn.Conv2d(number_of_input_channels, number_of_output_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(number_of_output_channels)) #Instance normalization instead of BN!
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#(2). Upsampling UNet:
class UNetUp(nn.Module):
    def __init__(self, number_of_input_channels, number_of_output_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        #Upsample by ConvTranspose2d (why not Upsampe layer? what's preferred? i think Upsampling and then Convolution right?!)
        layers = [  nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

#(3). Encoder:
class Encoder(nn.Module):
    def __init__(self, number_of_latent_space_dimensions):
        super(Encoder, self).__init__()

        #A pretrained model to output us an inner layer's feature space (content loss):
        resnet18_model = resnet18(pretrained=True)

        # Extracts features at the last fully-connected
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3]) #third layer from the end, a HyperParameter
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0) #understand how to incorporate Adaptive Average Pooling

        # Output is mu and log(var) for reparameterization trick used in VAEs
        # TODO: understand VAEs better
        self.fc_mu = nn.Linear(256, number_of_latent_space_dimensions)
        self.fc_logvar = nn.Linear(256, number_of_latent_space_dimensions)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar
####################################################################################################################################################################################################################################################################################################################################################




####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)


        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['B'].type(Tensor))
    real_B = Variable(imgs['A'].type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, 'images/%s/%s.png' % (OPT.dataset_name, batches_done), nrow=5, normalize=True)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Loss function
#(1). Basic Advarserial Loss (perhapse a way to generalize to multiple domains is to use a basic classifier multi label architecture with softmax CE)
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize generator and discriminator, assign them to gpu if available:
generator = GeneratorUNet()
discriminator = Discriminator()
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize weights a certain way
if opt.epoch_to_start_training_from != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (opt.dataset_name, opt.epoch)))
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
# ----------
#  Training
# ----------
for current_epoch in range(OPT.number_of_epochs):
    for current_batch_index, (current_image_batch, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # Configure input
        # Model inputs
        real_A = Variable(current_image_batch['B'].type(Tensor))
        real_B = Variable(current_image_batch['A'].type(Tensor))

        ########################
        ###  Train Generator ###
        optimizer_G.zero_grad()
        # Sample noise as generator input (consider passing in "speckles" and not totaly spatially random noise)
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, real_B)
        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()



        ############################
        ###  Train Discriminator ###
        optimizer_D.zero_grad()
        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)
        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()


        #####################
        ###  Log Progress ###
        # Determine approximate time left
        batches_done = current_epoch * len(dataloader) + i
        batches_left = OPT.number_of_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s" %
                         (current_epoch, OPT.number_of_epochs,
                          i, len(dataloader),
                          loss_D.item(), loss_G.item(),
                          loss_pixel.item(), loss_GAN.item(),
                          time_left))

        #If number of batches processed so far is a multiple of OPT.number_of_batches_to_spot_check_generator show certain number of pictures:
        #TODO: generalize this to arbitrary number of images:
        number_of_batches_so_far = current_epoch * len(dataloader) + current_batch_index
        if number_of_batches_so_far % OPT.number_of_batches_to_spot_check_generator == 0:
            save_image(generated_image_batch.data[:25], 'images/%d.png' % number_of_batches_so_far, nrow=5, normalize=True)

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if OPT.number_of_batches_to_checkpoint_model != -1 and current_epoch % OPT.number_of_batches_to_checkpoint_model == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (OPT.dataset_name, current_epoch))
            torch.save(discriminator.state_dict(), 'saved_models/%s/discriminator_%d.pth' % (OPT.dataset_name, current_epoch))

































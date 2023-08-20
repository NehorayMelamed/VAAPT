# GAN possible template:

# Imports:
# (1). Auxiliary:
import argparse
import os
import numpy as np
import math
from PIL import Image
import int_range, int_arange, mat_range, matlab_arange, my_linspace, my_linspace_int, importlib
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
# (2). TorchVision (add FastAi stuff which are much faster and more intuitive as far as augmentations)
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import resnet18
# (3). Torch Utils:
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
# (4). Torch NN:
import torch.nn as nn
import torch.nn.functional as F
import torch


# (5). Reminders on how to execute:
# load_variables_from_klepto_file(file_name):
# save_variables_to_klepto_file(file_name,variables_dict)
# update_klepto(file_name,variables_dict)
# save_tic(baseline_name_finale)
# save_toc(base_finale, post_finale)
# alert_box('alert text','alert title')
# response = pymsgbox.prompt('What is your name?')
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Auxiliary Functions:
class dictionary_to_class_attributes:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


# Network Weights initialization function (TODO: check if there are other things people suggest):
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Create a Folder called images in your base folder:
os.makedirs('images', exist_ok=True)
# os.getcwd() - Auxiliary: get current working directory to view created folder


# Script Parameters (usually in the form of ArgParse but here i write them down explicitly for ease of debugging):
exec(save_tic(1));
# (1). Dataset and Input Parameters:
dataset_name = "img_align_celeba"  # TODO: change this to my own dataset:
image_height = 256;
image_width = 256;
image_number_of_channels = 3;  # number of image image_number_of_channels (1 for BW -> change to 3 for rgb)
# (2). Training Parameters:
number_of_epochs = 200;  # number of training epochs
batch_size = 64;
lr = 0.0002;  # adam learning rate (try SGD with lr cycles)
epoch_to_start_lr_decay = 100;
b1 = 0.5;  # decay of first order momentum of gradient
b2 = 0.999;  # decay of first order momentum of gradient
number_of_cpu_threads_in_BG = 8;  # number of cpu threads to use during batch generation
latent_space_dimensionality = 100;  # dimensionality of the latent space
# (3). Model Checking Parameters:
number_of_batches_to_spot_check_generator = 400;  # interval between image samples (??)
number_of_epochs_to_checkpoint_model = -1;
# (4). Model Loading:
epoch_to_start_training_from = 0;
discriminator_to_generator_training_ratio = 5;
# (5). Model HyperParameters:
number_of_residual_blocks = 6;
selected_attributes_in_dataset = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
exec(save_toc(1, 1));  # (saves a dictionary variable called saved_dictionary1)
OPT = dictionary_to_class_attributes(saved_dictionary1);




# Preliminary stuff:
# (1). make folders to put outputs
os.makedirs('images/%s' % OPT.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % OPT.dataset_name, exist_ok=True)
# (2). Auxiliary:
number_of_selected_attributes_in_dataset = len(OPT.selected_attributed_in_dataset)
input_image_shape = (OPT.image_number_of_channels, OPT.image_size, OPT.image_size)
cuda = True if torch.cuda.is_available() else False


####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# DataSet & DataLoaders Definition:
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))

    def __getitem__(self, index):

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
dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, transforms_=transforms_, mode='train'),
                            batch_size=OPT.batch_size, shuffle=True, num_workers=OPT.number_of_cpu_thread_in_BG)
val_dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, transforms_=transforms_, mode='val'),
                            batch_size=16, shuffle=True, num_workers=OPT.number_of_cpu_thread_in_BG)###########################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Auxiliary Classes/Blocks for the networks:
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                      nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
#(1). Auxiliary Functions:
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

#(2). The Generator Itself:
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)


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
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Discriminator Definitioni:
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
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)


# Calculate output of image discriminator (PatchGAN)
patch = (1, OPT.image_height//2**3, OPT.image_width//2**3)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Generate and Sample Generated Images over time:
def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs['A'].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = G_BA(real_B)
    img_sample = torch.cat((real_A.data, fake_B.data,
                            real_B.data, fake_A.data), 0)
    save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=8, normalize=True)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Loss function
# (1).Auxiliary Loss Functions:
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)  # why is the fake label = 1 !?$@#!%
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates,
                              inputs=interpolates,
                              grad_outputs=fake,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# (2). Loss Definitions:
# Losses
adversarial_loss = torch.nn.MSELoss()
cycle_loss = torch.nn.L1Loss()
pixelwise_loss = torch.nn.L1Loss()
# Loss weights
#TODO: add loss weights variable even if the scrip's loss weights are all 1 !


### Multi-Class BCE Loss Function: ###
def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Initialize generator and discriminator, assign them to gpu if available:
G_AB = GeneratorUNet()
G_BA = GeneratorUNet()
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    adversarial_loss.cuda()
    cycle_loss.cuda()
    pixelwise_loss.cuda()
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Initialize weights a certain way
if OPT.epoch_to_start_training_from != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (OPT.dataset_name, OPT.epoch_to_start_training_from)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (OPT.dataset_name, OPT.epoch_to_start_training_from)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (OPT.dataset_name, OPT.epoch_to_start_training_from)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (OPT.dataset_name, OPT.epoch_to_start_training_from)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Optimizers (maybe add SGD with learning rate cycles etc' from FastAI):
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# ----------
#  Training
# ----------
for current_epoch in range(OPT.number_of_epochs):
    for current_batch_index, (current_image_batch, current_label_batch) in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = adversarial_loss(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = adversarial_loss(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Pixelwise translation loss
        loss_pixelwise = (pixelwise_loss(fake_A, real_A) + \
                          pixelwise_loss(fake_B, real_B)) / 2

        # Cycle loss
        loss_cycle_A = cycle_loss(G_BA(fake_B), real_A)
        loss_cycle_B = cycle_loss(G_AB(fake_A), real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + loss_cycle + loss_pixelwise

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = adversarial_loss(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = adversarial_loss(D_A(fake_A.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()
        # Real loss
        loss_real = adversarial_loss(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        loss_fake = adversarial_loss(D_B(fake_B.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = 0.5 * (loss_D_A + loss_D_B)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f] ETA: %s" %
            (epoch, opt.n_epochs,
             i, len(dataloader),
             loss_D.item(), loss_G.item(),
             loss_GAN.item(), loss_pixelwise.item(),
             loss_cycle.item(), time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))





























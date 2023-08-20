#GAN possible template:

#Imports:
#(1). Auxiliary:
import argparse
import os
import numpy as np
import math
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
class dictionary_to_class_attributes:
     def __init__(self, dictionary):
         for k, v in dictionary.items():
             setattr(self, k, v)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
#Create a Folder called images in your base folder:
os.makedirs('images', exist_ok=True)
# os.getcwd() - Auxiliary: get current working directory to view created folder


#Script Parameters (usually in the form of ArgParse but here i write them down explicitly for ease of debugging):
exec(save_tic(1));
number_of_epochs = 200; #number of training epochs
batch_size =  64;
lr = 0.0002; #adam learning rate (try SGD with lr cycles)
b1 = 0.5; #decay of first order momentum of gradient
b2 = 0.999; #decay of first order momentum of gradient
number_of_cpu_threads_in_BG = 8; #number of cpu threads to use during batch generation
latent_space_dimensionality = 100; #dimensionality of the latent space
image_size = 28; #size of each image dimension (width/height - 28 is for mnist, change this and add different aspect ratios)
image_number_of_channels = 1; #number of image image_number_of_channels (1 for BW -> change to 3 for rgb)
number_of_batches_to_spot_check_generator = 400; #interval between image samples (??)
exec(save_toc(1,1)); # (saves a dictionary variable called saved_dictionary1)
OPT = dictionary_to_class_attributes(saved_dictionary1);


#Preliminary stuff:
input_image_shape = (OPT.image_number_of_channels, OPT.image_size, OPT.image_size)
cuda = True if torch.cuda.is_available() else False
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #Basic GAN block - fully connected:
        def block(number_of_input_features, number_of_output_features, flag_BN_output=True):
            layers = [nn.Linear(number_of_input_features, number_of_output_features)]
            if flag_BN_output:
                layers.append(nn.BatchNorm1d(number_of_output_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True)) #remember of "inplace" input variable means
            return layers

        #Basic GAN generator network - a bunch of fully connected layers with Tanh at the end
        self.model = nn.Sequential(
            *block(OPT.latent_space_dimensionality, 128, flag_BN_output=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(input_image_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        #Forward propagate input z through the generator:
        img = self.model(z)
        #Reshape network output features to the shape of an image:
        img = img.view(img.size(0), *input_image_shape)
        return img
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #Basic GAN discriminator net - a bunch of fully connected layers
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        #The Discriminator receives an image (2d tensor) -> flatten it to make it suitable for discriminator input:
        img_flat = img.view(img.size(0), -1)
        #Forward propagate through the discriminator and get a number between 0 and 1 (sigmoide function - is there a better way?)
        validity = self.model(img_flat)

        return validity
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Loss function
#(1). Basic Advarserial Loss (perhapse a way to generalize to multiple domains is to use a basic classifier multi label architecture with softmax CE)
#TODO: i remember reading pytorch doesn't like one hot encoding - what do they do instead
adversarial_loss = torch.nn.BCELoss()
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Configure data loader with MNIST data:
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ])),
                   batch_size=OPT.batch_size,
                   shuffle=True)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize generator and discriminator, assign them to gpu if available:
generator = Generator()
discriminator = Discriminator()
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
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
        valid_image_label_value = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake_image_label_value = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_image_batch = Variable(current_image_batch.type(Tensor))

        ########################
        ###  Train Generator ###
        optimizer_G.zero_grad()
        # Sample noise as generator input (consider passing in "speckles" and not totaly spatially random noise)
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, OPT.latent_space_dimensionality))))
        # Generate a batch of images by passing the random noise through the generator
        generated_image_batch = generator(z)
        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(generated_image_batch), valid_image_label_value)
        # Back-Propagate:
        g_loss.backward()
        optimizer_G.step()


        ############################
        ###  Train Discriminator ###
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_image_batch), valid_image_label_value)
        fake_loss = adversarial_loss(discriminator(generated_image_batch.detach()), fake_image_label_value)
        d_loss = (real_loss + fake_loss) / 2
        # Back-Propagate:
        d_loss.backward()
        optimizer_D.step()


        #Verbosity - show current training stats:
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (current_epoch, OPT.number_of_epochs, current_batch_index, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        #If number of batches processed so far is a multiple of OPT.number_of_batches_to_spot_check_generator show certain number of pictures:
        #TODO: generalize this to arbitrary number of images:
        number_of_batches_so_far = current_epoch * len(dataloader) + current_batch_index
        if number_of_batches_so_far % OPT.number_of_batches_to_spot_check_generator == 0:
            save_image(generated_image_batch.data[:25], 'images/%d.png' % number_of_batches_so_far, nrow=5, normalize=True)











































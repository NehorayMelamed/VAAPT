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

#I understand what's being done here but i don't understand why:
def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt
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

# BEGAN hyper parameters
lambda_pt = 0.1
margin = max(1, OPT.batch_size / 64.)

#Preliminary stuff:
input_image_shape = (OPT.image_number_of_channels, OPT.image_size, OPT.image_size)
cuda = True if torch.cuda.is_available() else False
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        #I get an input vector z which i decide here will go into a Fully-Connected network with a OPT.latent_space_dimensionality hidden neurons and
        #will output basically 128 "channels" with a "new image size" of which is 1/4 the size of the original image.
        #TODO: the above are basically hyper parameters and i need to parameterize them!#@#!@: first_hidden_layer_output_channels=128, CNN_new_image_size=OPT.image_size//4;
        self.init_size = OPT.image_size // 4; #why divided by 4????!@$#!$!@#
        self.l1 = nn.Sequential(nn.Linear(OPT.latent_space_dimensionality, 128*self.init_size**2)) #why 128???#$@?$@#

        #CNN which takes a small "image" with many channels (after a fully connected layer) and outputs (after the appropriate amount of Upsample layers) an image the same size as the original
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, OPT.number_of_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        #Pass through initial fully connected layer which outputs 128 "channels"
        out = self.l1(z)
        #Reshape the new "image" (which is smaller with many more "channels")
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        #Pass through the CNN, which outputs a new image:
        img = self.conv_blocks(out)
        return img
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Downsampling (by using strided convolution):
        self.down = nn.Sequential(
            nn.Conv2d(OPT.number_of_channels, 64, 3, 2, 1), #Stride=2
            nn.ReLU())

        # Fully-connected layers
        self.downsampled_image_size = (OPT.image_size // 2)
        down_dim = 64 * (OPT.image_size // 2)**2  #64 channels after the strided convolutions X (downsampled_image_size)^2

        # Embedding
        self.embedding = nn.Linear(down_dim, 32) #TODO: make the embedding size of 32 a hyper parameter!!!!!

        # Fully Connected Network:
        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True)
        )

        # Upsampling:
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        embedding = self.embedding(out.view(out.size(0), -1))
        out = self.fc(embedding)
        out = self.up(out.view(out.size(0), 64, self.downsampled_image_size, self.downsampled_image_size))
        return out, embedding
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Loss function
#(1). Basic Advarserial Loss (perhapse a way to generalize to multiple domains is to use a basic classifier multi label architecture with softmax CE)
#TODO: i remember reading pytorch doesn't like one hot encoding - what do they do instead
adversarial_loss = torch.nn.BCELoss()

#(2). Reconstruction loss of AE (AutoEncoder):
pixelwise_loss = nn.MSELoss()
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
    pixelwise_loss.cuda()
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize weights a certain way
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
        D_reconstructed_images_from_generated_images, D_embeddings_from_generated_images = discriminator(generated_image_batch);
        # Loss measures generator's ability to fool the discriminator
        g_loss = pixelwise_loss(D_reconstructed_images_from_generated_images, generated_image_batch.detach()) + \
                 lambda_pt * pullaway_loss(D_embeddings_from_generated_images)
        # Back-Propagate:
        g_loss.backward()
        optimizer_G.step()




        ############################
        ###  Train Discriminator ###
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        #TODO: why isn't the generated_image_batch created again?!@#?!#? because it would be too slow!?$!$?@$
        D_reconstructed_images_from_real_images, D_embeddings_from_real_images = discriminator(real_image_batch)
        D_reconstructed_images_from_generated_images, D_embeddings_from_generated_images = discriminator(generated_image_batch.detach())
        d_loss_real = pixelwise_loss(D_reconstructed_images_from_real_images, real_image_batch);
        d_loss_fake = pixelwise_loss(D_reconstructed_images_from_generated_images, generated_image_batch);
        d_loss_total = d_loss_real;
        #TODO: what's going on here?
        if (margin - d_loss_fake.data).item() > 0:
            d_loss_total += margin - d_loss_fake; #why the minus sign? - because it's a pixelwise_loss from the generated images... when the discriminator is good it's embedding and reconstruction doesn't work
                                                  #and then the pixelwise_loss will be high....so if the discriminator is good this will be high and since we want to minimize the loss we put a minus sign here
        d_loss_total.backward()
        optimizer_D.step()




        #Verbosity - show current training stats:
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (current_epoch, OPT.number_of_epochs, i, len(dataloader),
                                                                         d_loss.item(), g_loss.item()))

        #If number of batches processed so far is a multiple of OPT.number_of_batches_to_spot_check_generator show certain number of pictures:
        #TODO: generalize this to arbitrary number of images:
        number_of_batches_so_far = current_epoch * len(dataloader) + current_batch_index
        if number_of_batches_so_far % OPT.number_of_batches_to_spot_check_generator == 0:
            save_image(generated_image_batch.data[:25], 'images/%d.png' % number_of_batches_so_far, nrow=5, normalize=True)











































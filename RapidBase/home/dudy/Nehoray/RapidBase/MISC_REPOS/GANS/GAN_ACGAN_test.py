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


#Save a grid of generated digits ranging from 0 to n_classes (TODO: make the number of saved images a hyperparameter and use something like a siemese network to check for mode collapse...although not so relevant in conditional GANs):
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, OPT.latent_space_dimensionallity))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generatorgenerator(z, labels) #the generator here is a global parameter
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)
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
number_of_classes = 10; #number of classes in the dataset!!@#!$@#$%#$%
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

        #Embedding layer which takes 10 inputs (or, more generally number_of_classes number of inputs) and embedds them in a space of size OPT.latent_space_dimensionality.
        #This makes it easy the "interpolate between classes" and do "arithmatic" like man+glasses-mustache
        self.label_emb = nn.Embedding(OPT.number_of_classes, OPT.latent_space_dimensionality)

        #Again - a fully connected layer which takes the output of the Embedding layer and outputs a certain number of elements which are to be reshaped into an "image"
        #Is it possible not to go through a fully connected layer? maybe there a Fully Convolutional way of doing this?
        self.init_size = OPT.image_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(OPT.latent_space_dimensionality, 128*self.init_size**2))

        #CNN for the generator
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
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        #Form the generator input by the torch.mul (matrix multiplication???):
        gen_input = torch.mul(self.label_emb(labels), noise)
        #Put the generator input into the fully connected layer and then the CNN:
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        downsampled_image_size = OPT.image_size // (2**4)

        # Output layers
        # first pass it through a fully connected layer followed by a sigmoid (why not a relu???)
        self.adv_layer = nn.Sequential( nn.Linear(128*downsampled_image_size**2, 1),
                                        nn.Sigmoid())
        # then pass it again through a fully connected layer which has number_of_classes output neurons (the discriminator wants to say what classes is this) followed by a softmax layer to make the output probabilities
        self.aux_layer = nn.Sequential( nn.Linear(128*downsampled_image_size**2, OPT.number_of_classes),
                                        nn.Softmax())

    def forward(self, img):
        #Pass input image through the CNN
        out = self.conv_blocks(img)
        #Flatten the CNN output:
        out = out.view(out.shape[0], -1)
        #Pass it through the two fully connected layer to predicted final classes using the output of the softmax layer:
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Loss function
#(1). Basic Advarserial Loss (perhapse a way to generalize to multiple domains is to use a basic classifier multi label architecture with softmax CE)
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()
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
    auxiliary_loss.cuda()

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
#Since ACGAN:
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# ----------
#  Training
# ----------
for current_epoch in range(OPT.number_of_epochs):
    for current_batch_index, (current_batch_images, current_batch_labels) in enumerate(dataloader):

        # Adversarial ground truths
        valid_image_label_value = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake_image_label_value = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_batch_images_tensor = Variable(current_batch_images.type(Tensor))
        real_batch_labels_tensors = Variable(current_batch_labels.type(LongTensor))

        ########################
        ###  Train Generator ###
        optimizer_G.zero_grad()
        # Sample noise as generator input (consider passing in "speckles" and not totaly spatially random noise)
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, OPT.latent_space_dimensionality))))
        generated_labels_batch = Variable(LongTensor(np.random.randint(0, OPT.number_of_classes,
                                                                 batch_size)))  # this time we also randomize the label (which will go through an Embedding layer on it's way to latent space)
        # Generate a batch of images by passing the random noise through the generator
        generated_image_batch = generator(z, generated_labels_batch)
        # Loss measures generator's ability to fool the discriminator (BUT!- this time we have to losses...the adversarial loss (whether the discriminator was fooled) and the auxiliary loss which should be called the Classification Loss
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + \
                        auxiliary_loss(pred_label, gen_labels))
        # Back-Propagate:
        g_loss.backward()
        optimizer_G.step()


        ############################
        ###  Train Discriminator ###
        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        #(*). Loss for real images:
        real_batch_discriminator_validity_prediction, real_batch_discriminator_label_prediction = discriminator(real_batch_images_tensor);
        d_real_loss = (
                       adversarial_loss(real_batch_discriminator_validity_prediction, valid_image_label_value) +
                       auxiliary_loss(real_batch_discriminator_label_prediction, real_batch_labels_tensors)
                       ) / 2
        #(*). Loss for fake images:
        fake_batch_discriminator_validity_prediction, fake_batch_discriminator_label_prediction = discriminator(generated_image_batch.detach());
        d_fake_loss = (
                       adversarial_loss(fake_batch_discriminator_validity_prediction, fake_image_label_value) +
                       auxiliary_loss(fake_batch_discriminator_label_prediction, generated_labels_batch)
                      ) / 2
        #(*). Total Discriminator Loss:
        d_loss = (d_real_loss + d_fake_loss) / 2;


        # Calculate discriminator accuracy
        discriminator_predictions = np.concatenate([real_batch_discriminator_label_prediction.data.cpu().numpy(), fake_batch_discriminator_label_prediction.data.cpu().numpy()], axis=0)
        ground_truth_labels = np.concatenate([current_batch_labels.data.cpu().numpy(), generated_labels_batch.data.cpu().numpy()], axis=0)
        discriminator_accuracy = np.mean(np.argmax(pred, axis=1) == gt)

        # Back Propagate:
        d_loss.backward()
        optimizer_D.step()

        #Verbosity - show current training stats:
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (current_epoch, OPT.number_of_epochs, i, len(dataloader), d_loss.item(), 100 * discriminator_accuracy, g_loss.item()))

        #If number of batches processed so far is a multiple of OPT.number_of_batches_to_spot_check_generator show certain number of pictures:
        #TODO: generalize this to arbitrary number of images:
        number_of_batches_so_far = current_epoch * len(dataloader) + current_batch_index
        if number_of_batches_so_far % OPT.number_of_batches_to_spot_check_generator == 0:
            save_image(generated_image_batch.data[:25], 'images/%d.png' % number_of_batches_so_far, nrow=5, normalize=True)











































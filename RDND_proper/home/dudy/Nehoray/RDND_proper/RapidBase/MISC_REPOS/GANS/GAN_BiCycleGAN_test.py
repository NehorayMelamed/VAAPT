#GAN possible template:

#Imports:
#(1). Auxiliary:
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



#Residual Block With Instance Normalization instead of Batch Normalization:
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


#ImageDataset object inheriting from Dataset:
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))



class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

#Learning Rate Decay:
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

# Keeping Track of Generated Images:
def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img_A, img_B in zip(imgs['A'], imgs['B']):
        # Repeat input image by number of channels
        real_A = img_A.view(1, *img_A.shape).repeat(8, 1, 1, 1)
        real_A = Variable(real_A.type(Tensor))
        # Get interpolated noise [-1, 1]
        sampled_z = np.repeat(np.linspace(-1, 1, 8)[:, np.newaxis], OPT.latent_space_dimensionality, 1)
        sampled_z = Variable(Tensor(sampled_z))
        # Generator samples
        '###### This is Sort of a  GAN and the generator receives both the real image and sampled noise!!!! #####'
        fake_B = generator(real_A, sampled_z)
        # Concatenate samples horisontally
        fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
        img_sample = torch.cat((img_A, fake_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Cocatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)

####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Create a Folder called images in your base folder:
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)
# os.getcwd() - Auxiliary: get current working directory to view created folder




#Script Parameters (usually in the form of ArgParse but here i write them down explicitly for ease of debugging):
exec(save_tic(1));
#(1). Dataset and Input Parameters:
dataset_name = "monet2photo" #TODO: change this to my own dataset:
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
#(5). Network Hyper Parameters:
number_of_residual_blocks = 9;
exec(save_toc(1,1)); # (saves a dictionary variable called saved_dictionary1)
OPT = dictionary_to_class_attributes(saved_dictionary1);


#Cude if possible:
image_shape = (OPT.number_of_channels, OPT.image_height, OPT.image_width)
cuda = True if torch.cuda.is_available() else False
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

#(4). Reparameterization:
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), OPT.latent_space_dimensionality))))
    z = sampled_z * std + mu
    return z
####################################################################################################################################################################################################################################################################################################################################################




####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
class GeneratorUNet(nn.Module):
    def __init__(self, number_of_latent_dimensions, image_shape):
        super(GeneratorUNet, self).__init__()

        number_of_input_channels, self.image_height, self.image_width = image_shape
        self.fc = nn.Linear(number_of_latent_dimensions, self.image_height * self.image_width)

        self.down1 = UNetDown(number_of_input_channels + 1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        #assuming number of output channels equals number of input channels
        #Input size into the nn.ConvTransposed2d is 128 because it is twice the size of the output of self.up6. because the UNetUp does
        #Concatenation on the channels and so it double the number of channels you input as "output size".
        final = [nn.ConvTranspose2d(128, number_of_input_channels, 4, stride=2, padding=1),
                 nn.Tanh()]
        self.final = nn.Sequential(*final)

    def forward(self, x, z):
        # Propogate noise through fc layer and reshape to img shape
        #TODO: check if instead a hidden fully connected layer maybe i can input a noise signal the shape of an image into a CNN?
        #TODO: has the above suggestion bean tried? it seems to be able to yield less diversity but maybe that's bullshit?
        z_ = self.fc(z).view(z.size(0), 1, self.image_height, self.image_width)
        d1 = self.down1(torch.cat((x, z_), 1))
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class MultiDiscriminator(nn.Module):
    def __init__(self, number_of_input_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(number_of_input_filters, number_of_output_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(number_of_input_filters, number_of_output_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(number_of_output_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList() # nn.ModuleList() ! - very nice
        for i in range(3):
            self.models.add_module('disc_%d' % i,
                nn.Sequential(
                    *discriminator_block(number_of_input_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                )
            )

        self.downsample = nn.AvgPool2d(number_of_input_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt)**2) for out in self.forward(x)])
        return loss

    #Very Interesting Way! :
    # i have 3 models in my models (which is a nn.ModuleList).
    # i loop over them, giving each model a downsampled version of the input, and concetenate the outputs of each model.
    # each scale is given different weights
    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(OPT.HR_image_height / 2**4), int(OPT.HR_image_width / 2**4)
patch = (OPT.batch_size, 1, patch_h, patch_w)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Loss function
#(1). Basic Advarserial Loss (perhapse a way to generalize to multiple domains is to use a basic classifier multi label architecture with softmax CE)
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
mae_loss = torch.nn.L1Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 10
lambda_latent = 0.5
lambda_kl = 0.01
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize generator and discriminator, assign them to gpu if available:
# Initialize generator, encoder and discriminators
generator = Generator(opt.latent_dim, img_shape)
encoder = Encoder(opt.latent_dim)
D_VAE = MultiDiscriminator()
D_LR = MultiDiscriminator()

if cuda:
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    mae_loss.cuda()
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize weights a certain way
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize weights a certain way
if opt.epoch_to_start_training_from != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (opt.dataset_name, opt.epoch)))
    encoder.load_state_dict(torch.load('saved_models/%s/encoder_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_VAE.load_state_dict(torch.load('saved_models/%s/D_VAE_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_LR.load_state_dict(torch.load('saved_models/%s/D_LR_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)
    D_LR.apply(weights_init_normal)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Optimizers (maybe add SGD with learning rate cycles etc' from FastAI):
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
####################################################################################################################################################################################################################################################################################################################################################




####################################################################################################################################################################################################################################################################################################################################################
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# Adversarial ground truths
valid_image_label_value = Variable(Tensor(np.ones((batch_size, *patch))), requires_grad=False)
fake_image_label_value = Variable(Tensor(np.zeros((batch_size, *patch))), requires_grad=False)
valid = 1
fake = 0
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# PreProcessing, Transforms, Augmentations and DataLoaders.
# TODO: transfer to the FastAI input pipeline which is supposed to be faster
transforms_ = [ transforms.Resize((OPT.image_height, OPT.image_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, transforms_=transforms_),
                            batch_size=OPT.batch_size, shuffle=True, num_workers=OPT.number_of_cpu_thread_in_BG)
val_dataloader = DataLoader(ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode='val'),
                            batch_size=8, shuffle=True, num_workers=1)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Learning Rate Schedualers, Tracking:
# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.number_of_epochs, opt.epoch_to_start_training_from, opt.epoch_to_start_lr_decay).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.number_of_epochs, opt.epoch_to_start_training_from, opt.epoch_to_start_lr_decay).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.number_of_epochs, opt.epoch_to_start_training_from, opt.epoch_to_start_lr_decay).step)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# ----------
#  Training
for current_epoch in range(OPT.number_of_epochs):
    for current_batch_index, current_image_batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(current_image_batch['A'].type(Tensor))
        real_B = Variable(current_image_batch['B'].type(Tensor))


        ###################################
        ###  Train Generator & Encoder! ###
        optimizer_E.zero_grad()
        optimizer_G.zero_grad()
        # Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)
        # Pixelwise loss of translated image by VAE
        loss_pixel = mae_loss(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1))
        # Adversarial loss
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

        # ---------
        # cLR-GAN
        # ---------
        # Produce output using sampled z (cLR-GAN)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), OPT.latent_space_dimensionality))))
        _fake_B = generator(real_A, sampled_z)
        # cLR Loss: Adversarial loss
        loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

        # ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------
        #TODO: i don't see any cycle consistency loss.... perhapse this isn't appropriate here?
        loss_GE = loss_VAE_GAN + \
                  loss_LR_GAN + \
                  lambda_pixel * loss_pixel + \
                  lambda_kl * loss_kl
        loss_GE.backward(retain_graph=True)
        optimizer_E.step()

        # ---------------------
        # Generator Only Loss
        # ---------------------
        # Latent L1 loss
        _mu, _ = encoder(_fake_B)
        loss_latent = lambda_latent * mae_loss(_mu, sampled_z) #what does this loss penalize?
        loss_latent.backward()
        optimizer_G.step()

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------
        optimizer_D_VAE.zero_grad()
        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + \
                     D_VAE.compute_loss(fake_B.detach(), fake)
        loss_D_VAE.backward()
        optimizer_D_VAE.step()

        # ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------
        optimizer_D_LR.zero_grad()
        loss_D_LR = D_VAE.compute_loss(real_B, valid) + \
                    D_VAE.compute_loss(_fake_B.detach(), fake)
        loss_D_LR.backward()
        optimizer_D_LR.step()

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = current_epoch * len(dataloader) + i
        batches_left = OPT.number_of_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, latent: %f] ETA: %s" %
            (current_epoch, OPT.number_of_epochs,
             i, len(dataloader),
             loss_D_VAE.item(), loss_D_LR.item(),
             loss_GE.item(), loss_pixel.item(),
             loss_latent.item(), time_left))

        if batches_done % opt.number_of_batches_to_spot_check_generator == 0:
            sample_images(batches_done)

    if OPT.number_of_batches_to_checkpoint_model != -1 and current_epoch % OPT.number_of_batches_to_checkpoint_model == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/%s/generator_%d.pth' % (OPT.dataset_name, current_epoch))
        torch.save(encoder.state_dict(), 'saved_models/%s/encoder_%d.pth' % (OPT.dataset_name, current_epoch))
        torch.save(D_VAE.state_dict(), 'saved_models/%s/D_VAE_%d.pth' % (OPT.dataset_name, current_epoch))
        torch.save(D_LR.state_dict(), 'saved_models/%s/D_LR_%d.pth' % (OPT.dataset_name, current_epoch))

































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

#Saves a generated sample from the test set:
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader)) #Iterator from DataLoader!@$#@!$@$!#$!@
    real_A = Variable(imgs['A'].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs['B'].type(Tensor))
    fake_A = G_BA(real_B)
    img_sample = torch.cat((real_A.data, fake_B.data,
                            real_B.data, fake_A.data), 0)
    save_image(img_sample, 'images/%s/%s.png' % (OPT.dataset_name, batches_done), nrow=5, normalize=True)

####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Create a Folder called images in your base folder:
os.makedirs('images', exist_ok=True)
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
cuda = True if torch.cuda.is_available() else False
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
class GeneratorResNet(nn.Module):
    def __init__(self, number_of_input_channels=3, number_of_output_channels=3, number_of_residual_blocks=9):
        super(GeneratorResNet, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2): #Two DownSamplings
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2): #Two UpSamplings
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class Discriminator(nn.Module):
    def __init__(self, number_of_input_channels=3):
        super(Discriminator, self).__init__()

        #TODO: Change name to include "DownSampling" for better clarity:
        def discriminator_block(number_of_input_filters, number_of_output_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(number_of_input_filters, number_of_output_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(number_of_output_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(number_of_input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(OPT.HR_image_height / 2**4), int(OPT.HR_image_width / 2**4)
patch = (OPT.batch_size, 1, patch_h, patch_w)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Loss function
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize generator and discriminator, assign them to gpu if available:
# Initialize generator and discriminator
G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
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
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
# Adversarial ground truths
valid_image_label_value = Variable(Tensor(np.ones((batch_size, *patch))), requires_grad=False)
fake_image_label_value = Variable(Tensor(np.zeros((batch_size, *patch))), requires_grad=False)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Transforms & DataLoader
#(1). Transforms:
# Image transformations
transforms_ = [ transforms.Resize(int(OPT.image_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((OPT.image_height, OPT.image_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
#(2). DataLoader:
# Training data loader
dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, transforms_=transforms_, unaligned=True),
                        batch_size=OPT.batch_size, shuffle=True, num_workers=OPT.n_cpu)
# Test data loader
val_dataloader = DataLoader(ImageDataset("../../data/%s" % OPT.dataset_name, transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=5, shuffle=True, num_workers=1)
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


        ########################
        ###  Train Generator ###
        optimizer_G.zero_grad()
        # Identity loss ????....i understand the cycle loss but why is this loss here?
        loss_id_A = criterion_identity(G_BA(real_B), real_A)
        loss_id_B = criterion_identity(G_AB(real_A), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid_image_label_value)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid_image_label_value)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        # Cycle loss
        recov_A = G_BA(fake_B)  #recov_A = G_BA(G_AB(real_A))
        loss_cycle_A = criterion_cycle(recov_A, real_A) #it seems to me unreasonable to have a direct pixels loss...at least not without a segmentation map and/or some stochastic nature. besides i think that seperating the generators is stupid...
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        # Total loss
        loss_G = loss_GAN + \
                 lambda_cyc * loss_cycle + \
                 lambda_id * loss_identity
        loss_G.backward()
        optimizer_G.step()



        #############################
        ###  Train Discriminators ###
        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid_image_label_value)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake_image_label_value)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid_image_label_value)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake_image_label_value)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        loss_D = (loss_D_A + loss_D_B) / 2



        ####################
        ###  LOG PROCESS ###
        # Determine approximate time left
        batches_done = current_epoch * len(dataloader) + current_batch_index
        batches_left = OPT.number_of_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
            (current_epoch, OPT.number_of_epochs,
             i, len(dataloader),
             loss_D.item(), loss_G.item(),
             loss_GAN.item(), loss_cycle.item(),
             loss_identity.item(), time_left))

        # If at sample interval save image
        if batches_done % OPT.number_of_batches_to_spot_check_generator == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.number_of_batches_to_checkpoint_model != -1 and epoch % opt.number_of_batches_to_checkpoint_model == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (OPT.dataset_name, current_epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (OPT.dataset_name, current_epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (OPT.dataset_name, current_epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (OPT.dataset_name, current_epoch))





































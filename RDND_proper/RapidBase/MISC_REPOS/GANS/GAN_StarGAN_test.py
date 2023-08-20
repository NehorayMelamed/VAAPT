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
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
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
number_of_epochs_to_checkpoint_model = -1;
#(4). Model Loading:
epoch_to_start_training_from = 0;
discriminator_to_generator_training_ratio = 5;
#(5). Model HyperParameters:
number_of_residual_blocks = 6;
selected_attributes_in_dataset = ['Black_Hair','Blond_Hair','Brown_Hair','Male','Young']
exec(save_toc(1,1)); # (saves a dictionary variable called saved_dictionary1)
OPT = dictionary_to_class_attributes(saved_dictionary1);



#Preliminary stuff:
#(1). make folders to put outputs
os.makedirs('images/%s' % OPT.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % OPT.dataset_name, exist_ok=True)
#(2). Auxiliary:
number_of_selected_attributes_in_dataset = len(OPT.selected_attributed_in_dataset)
input_image_shape = (OPT.image_number_of_channels, OPT.image_size, OPT.image_size)
cuda = True if torch.cuda.is_available() else False
####################################################################################################################################################################################################################################################################################################################################################






####################################################################################################################################################################################################################################################################################################################################################
#DataSet & DataLoaders Definition:
class CelebADataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train', attributes=None):
        self.transform = transforms.Compose(transforms_)

        self.selected_attrs = attributes
        self.files = sorted(glob.glob('%s/*.jpg' % root))
        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]
        self.label_path = glob.glob('%s/*.txt' % root)[0]
        self.annotations = self.get_annotations()

    def get_annotations(self):
        """Extracts annotations for CelebA"""
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, 'r')]
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == '1'))
            annotations[filename] = labels
        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split('/')[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[filename]
        label = torch.FloatTensor(np.array(label))

        return img, label

    def __len__(self):
        return len(self.files)


# Configure dataloaders
train_transforms = [transforms.Resize(int(1.12*OPT.image_height), Image.BICUBIC),
                    transforms.RandomCrop(OPT.image_height),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

dataloader = DataLoader(CelebADataset("../../data/%s" % OPT.dataset_name, transforms_=train_transforms, mode='train', attributes=OPT.selected_attributes_in_dataset),
                        batch_size=OPT.batch_size, shuffle=True, num_workers=OPT.number_of_cpu_threads_in_BG)

val_transforms = [  transforms.Resize((OPT.img_height, OPT.img_width), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

val_dataloader = DataLoader(CelebADataset("../../data/%s" % OPT.dataset_name, transforms_=val_transforms, mode='val', attributes=OPT.selected_attributes_in_dataset),
                        batch_size=10, shuffle=True, num_workers=1)
####################################################################################################################################################################################################################################################################################################################################################




####################################################################################################################################################################################################################################################################################################################################################
# Auxiliary Classes/Blocks for the networks:
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                        nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_features, in_features, 3, stride=1, padding=1, bias=False),
                        nn.InstanceNorm2d(in_features, affine=True, track_running_stats=True)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
####################################################################################################################################################################################################################################################################################################################################################




####################################################################################################################################################################################################################################################################################################################################################
# Generator Definition:
class GeneratorResNet(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), res_blocks=9, number_of_selected_attributes_in_dataset=5):
        super(GeneratorResNet, self).__init__()
        input_image_number_of_channels, img_size, _ = img_shape

        # Initial convolution block
        # input number of channels is: input_image_number_of_channels + number_of_selected_attributes_in_dataset
        model = [   nn.Conv2d(in_channels=input_image_number_of_channels + number_of_selected_attributes_in_dataset, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False),
                    nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        current_number_of_channels = 64 #nice way of keeping track of dimensions
        for _ in range(2):
            model += [  nn.Conv2d(current_number_of_channels, current_number_of_channels*2, 4, stride=2, padding=1, bias=False),
                        nn.InstanceNorm2d(current_number_of_channels*2, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True) ]
            current_number_of_channels *= 2

        # Residual blocks
        for _ in range(res_blocks):
            model += [ResidualBlock(current_number_of_channels)]

        # Upsampling
        for _ in range(2):
            model += [  nn.ConvTranspose2d(current_number_of_channels, current_number_of_channels//2, 4, stride=2, padding=1, bias=False),
                        nn.InstanceNorm2d(current_number_of_channels//2, affine=True, track_running_stats=True),
                        nn.ReLU(inplace=True) ]
            current_number_of_channels = current_number_of_channels // 2

        # Output layer
        model += [  nn.Conv2d(current_number_of_channels, input_image_number_of_channels, 7, stride=1, padding=3),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)


    def forward(self, x, label):
        #NOTICE: as we can see, the way the labels are incorporated into the generator is by simply resizing them into the shape of the image and inserting them as extra channels of the image.
        #        This seems to hint at the option of changing multiple things in the image to other categories according to semantic segmentation mapping.
        #Shape
        label = label.view(label.size(0), label.size(1), 1, 1) #simply add two additional dimensions
        label = label.repeat(1, 1, x.size(2), x.size(3)) #repeat the label value in the two additional dimensiosn
        x = torch.cat((x, label), 1)
        return self.model(x)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Discriminator Definitioni:
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 128, 128), number_of_selected_attributes_in_dataset=5, n_strided=6):
        super(Discriminator, self).__init__()
        channels, img_size, _ = img_shape

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [  nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                        nn.LeakyReLU(0.01)]
            return layers

        layers = discriminator_block(channels, 64)
        current_number_of_channels = 64
        for _ in range(n_strided - 1):
            layers.extend(discriminator_block(current_number_of_channels, current_number_of_channels*2))
            current_number_of_channels *= 2

        self.model = nn.Sequential(*layers)

        # Output 1: PatchGAN
        self.out1 = nn.Conv2d(current_number_of_channels, 1, 3, padding=1, bias=False)
        # Output 2: Class prediction
        kernel_size = img_size // 2**n_strided
        self.out2 = nn.Conv2d(current_number_of_channels, number_of_selected_attributes_in_dataset, kernel_size, bias=False)

    def forward(self, img):
        feature_repr = self.model(img)
        out_adv = self.out1(feature_repr)
        out_cls = self.out2(feature_repr)
        return out_adv, out_cls.view(out_cls.size(0), -1)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
#Generate and Sample Generated Images over time:
def sample_images(batches_done):
    """Saves a generated sample of domain translations"""
    val_imgs, val_labels = next(iter(val_dataloader))
    #From Numpy to PyTorch Tensor:
    val_imgs = Variable(val_imgs.type(Tensor))
    val_labels = Variable(val_labels.type(Tensor))
    img_samples = None
    #Sample 10 Images (TODO: change to the sqrt of image dataset size as explained in the lecture to check for mode collapse). (TODO: is mode collapse relevant in ConditionalGAN?????):
    for i in range(10):
        img, label = val_imgs[i], val_labels[i]
        # Repeat for number of label changes
        imgs = img.repeat(number_of_selected_attributes_in_dataset, 1, 1, 1)
        labels = label.repeat(number_of_selected_attributes_in_dataset, 1)
        # Make changes to labels
        for sample_i, changes in enumerate(label_changes):
            for col, val in changes:
                labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

        # Generate translations
        gen_imgs = generator(imgs, labels)
        # Concatenate images by width (nice way of doing it!)
        gen_imgs = torch.cat([x for x in gen_imgs.data], -1) #-1 = 1 from the end meaning columnwise
        img_sample = torch.cat((img.data, gen_imgs), -1)
        # Add as row to generated samples
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2) #-2 = 2 from the end meaning rowwise

    save_image(img_samples.view(1, *img_samples.shape), 'images/%s.png' % batches_done, normalize=True)
####################################################################################################################################################################################################################################################################################################################################################


####################################################################################################################################################################################################################################################################################################################################################
# Loss function
#(1).Auxiliary Loss Functions:
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = discriminator(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False) #why is the fake label = 1 !?$@#!%
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

#(2). Loss Definitions:
#TODO: maybe i can add a double cycle loss: G_AB(G_BA(G_AB(G_BA(B)))) = realB?
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
criterion_cycle = torch.nn.L1Loss()
# Loss weights
lambda_cls = 1
lambda_rec = 10
lambda_gp = 10
### Multi-Class BCE Loss Function: ###
def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize generator and discriminator, assign them to gpu if available:
generator = GeneratorResNet(img_shape=input_image_shape, res_blocks=OPT.number_of_residual_blocks, number_of_selected_attributes_in_dataset=number_of_selected_attributes_in_dataset)
discriminator = Discriminator(img_shape=input_image_shape, number_of_selected_attributes_in_dataset=number_of_selected_attributes_in_dataset)
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_cycle.cuda()

#Label Changes Definitions (understand this better!$#@$%):
label_changes = [
    ((0, 1), (1, 0), (2, 0)),   # Set to black hair
    ((0, 0), (1, 1), (2, 0)),   # Set to blonde hair
    ((0, 0), (1, 0), (2, 1)),   # Set to brown hair
    ((3, -1),),                 # Flip gender
    ((4, -1),)                   # Age flip
]
####################################################################################################################################################################################################################################################################################################################################################



####################################################################################################################################################################################################################################################################################################################################################
# Initialize weights a certain way
if opt.epoch_to_start_training_from != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load('saved_models/%s/generator_%d.pth' % (OPT.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load('saved_models/%s/discriminator_%d.pth' % (OPT.dataset_name, opt.epoch)))
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
    for current_batch_index, (current_image_batch, current_label_batch) in enumerate(dataloader):

        # Model inputs
        current_image_batch = Variable(current_image_batch.type(Tensor))
        current_label_batch = Variable(current_label_batch.type(Tensor))

        # Sample labels as generator inputs
        sampled_labels = Variable(Tensor(np.random.randint(0, 2, (batch_size, number_of_selected_attributes_in_dataset))))
        # Generate fake batch of images
        fake_imgs = generator(imgs, sampled_c)

        ############################
        ###  Train Discriminator ###
        optimizer_D.zero_grad()
        # Real images
        real_validity, pred_cls = discriminator(imgs)
        # Fake images
        fake_validity, _ = discriminator(fake_imgs.detach())
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data)
        # Adversarial loss
        loss_D_adv = - torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        # Classification loss
        loss_D_cls = criterion_cls(pred_cls, labels)
        # Total loss
        loss_D = loss_D_adv + lambda_cls * loss_D_cls
        # Back-Propagate:
        loss_D.backward()
        optimizer_D.step()


        ########################
        ###  Train Generator ###
        # Every n_critic times update generator
        if current_batch_index % OPT.discriminator_to_generator_training_ratio == 0:

            # Translate and reconstruct image
            fake_images = generator(current_image_batch, sampled_labels)
            recovered_images = generator(fake_images, current_label_batch) #Cyclic Generation - take input image, generate a new one, a try to re-generate the original images from the generated ones
            # Discriminator evaluates translated image
            discriminator_output_fake_validity, discriminator_output_fake_classes = discriminator(generated_images)
            # Adversarial loss
            loss_G_adv = -torch.mean(discriminator_output_fake_validity)
            # Classification loss
            loss_G_cls = criterion_cls(discriminator_output_fake_classes, sampled_labels)
            # Reconstruction loss
            loss_G_rec = criterion_cycle(recovered_images, current_image_batch)
            # Total loss
            loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec
            # Back-Propagate:
            loss_G.backward()
            optimizer_G.step()


            #####################
            ###  Log Progress ###
            # Determine approximate time left
            batches_done = current_epoch * len(dataloader) + i
            batches_left = OPT.number_of_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f] ETA: %s" %
                (current_epoch, OPT.number_of_epochs,
                 current_batch_index, len(dataloader),
                 loss_D_adv.item(), loss_D_cls.item(),
                 loss_G.item(), loss_G_adv.item(),
                 loss_G_cls.item(), loss_G_rec.item(),
                 time_left))

            # If at sample interval sample and save image
            if batches_done % OPT.number_of_batches_to_spot_check_generator == 0:
                sample_images(batches_done)



        if OPT.number_of_epochs_to_checkpoint_model != -1 and current_epoch % OPT.number_of_epochs_to_checkpoint_model == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
            torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)
        



































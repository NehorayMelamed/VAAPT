#Imports:
#(1). Auxiliary:
from __future__ import print_function
import PIL
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
from collections import OrderedDict
import sys
import math
from datetime import datetime
import cv2
from skimage.measure import compare_ssim
from shutil import get_terminal_size
import math
import pickle
import random
import numpy as np
import lmdb
import ctypes  # An included library with Python install.
from numpy import power as power
from numpy import exp as exp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import time #time.time()
import nltk
import collections
import re
from csv import reader
import tarfile
from pandas import read_csv
from pandas import Series
import collections
#counter = collections.Counter()
#from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.ar_model import AR
import argparse
import pydot
import psutil
# import graphviz
# import pydot_ng
length = len #use length instead of len.... make sure it doesn't cause problems

#Pylab imshow stuff:
import pylab
from pylab import imshow, pause, draw, title, axes, ylabel, ylim, yticks, xlabel, xlim, xticks
from pylab import colorbar, colormaps, subplot, suptitle, plot
#Message Box:
def message_box(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
import pymsgbox
from pymsgbox import alert as alert_box
from pymsgbox import confirm as confirm_box
from pymsgbox import prompt as prompt_box

#(2). TorchVision (add FastAi stuff which are much faster and more intuitive as far as augmentations)
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.models import vgg19
from torchvision.utils import make_grid
#(3). Torch Utils:
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
#(4). Torch NN:
import torch.nn as nn
import torch.nn.functional as F
import torch
#(5). More Torch Stuff:
import torch
from torch.nn.functional import normalize
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import torch.utils.data as data

#TensorFlow:
import tensorflow as tf
import tensorboard as TB
import tensorboardX as TBX
from tensorboardX import SummaryWriter



#ESRGAN:
from ESRGAN_utils import *
from ESRGAN_deep_utils import *
from ESRGAN_basic_Blocks_and_Layers import *
# from ESRGAN_Models import *
# from ESRGAN_Losses import *
# from ESRGAN_Optimizers import *
# from ESRGAN_Visualizers import *
# from ESRGAN_dataset import *
# from ESRGAN_OPT import *


import copy






#(2). Deep Visualizers Auxiliaries:
########################################################################################################################################################################################################
#### Saving Results To Image On Disk: ####
def save_gradient_images(gradient, save_directory, save_image_filename):
    """
        Exports the original gradient image
    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    #Normalize Gradient Values to Fill Dynamic Range to between [0,255]:
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    #Switch Gradient's Image Mode from [C,H,W] to [H,W,C]:
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    #Full File Name to Save:
    path_to_file = os.path.join(save_directory, save_image_filename + '.jpg')
    #Convert RBG to GBR:
    gradient = gradient[..., ::-1]
    #Actually Save Image:
    cv2.imwrite(path_to_file, gradient)


def save_class_activation_on_image(original_image, activation_map, save_directory, save_image_filename):
    """
        Saves cam activation map and activation map on the original image
    Args:
        original_image (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    # Grayscale activation map
    path_to_file = os.path.join(save_directory, save_image_filename+'_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join(save_directory, save_image_filename+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    original_image = cv2.resize(original_image, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(original_image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join(save_directory, save_image_filename+'_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))
############################################################################################################




############################################################################################################
##### PreProcess & Inverse PreProcessing of Images: ######
def preprocess_image_ImageNet(cv2im, flag_resize_image=True, resize_to_shape=(224,224), mean_vec=[0.485, 0.456, 0.406], std_vec=[0.229, 0.224, 0.225], flag_requires_grad=False):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean_vec = list(mean_vec);
    std_vec = list(std_vec)
    # Resize image
    if flag_resize_image:
        cv2im = cv2.resize(cv2im, resize_to_shape)
    input_image_numpy = np.float32(cv2im)
    if input_image_numpy.shape[0] == 3:
        input_image_numpy = input_image_numpy[..., [2, 1, 0]]  # Negative striding
    input_image_numpy = input_image_numpy.transpose(2, 0, 1)
    # Normalize the channels
    for channel, _ in enumerate(input_image_numpy):
        input_image_numpy[channel] /= 255
        input_image_numpy[channel] -= mean_vec[channel]
        input_image_numpy[channel] /= std_vec[channel]
    # Convert to float tensor
    input_image_tensor = torch.from_numpy(input_image_numpy).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    input_image_tensor.unsqueeze_(0)
    # Convert to Pytorch variable
    input_image_variable = Variable(input_image_tensor, requires_grad=False)
    return input_image_variable


def inverse_preprocessing_image_ImageNet(input_image_variable, initial_mean=[0.485,0.456,0.406], initial_std=[0.229,0.224,0.225]):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    #TODO: generalize this to both numpy
    #Reverse Transform:
    reverse_mean = list(-np.array(initial_mean))
    reverse_std = list(1/np.array(initial_std))
    input_image_numpy = copy.copy(input_image_variable.data.numpy()[0])
    #Inverse Preprocessing:
    for channel in range(3):
        input_image_numpy[channel] /= reverse_std[channel]
        input_image_numpy[channel] -= reverse_mean[channel]
    #Clip Values to between 0 and 1:
    input_image_numpy[input_image_numpy > 1] = 1
    input_image_numpy[input_image_numpy < 0] = 0
    #Stretch to between [0,255] uint8:
    input_image_numpy = np.round(input_image_numpy * 255)
    input_image_numpy = np.uint8(input_image_numpy).transpose(1, 2, 0)
    #Convert RBG to GBR (RBG????.... when did we get RBG?... i thought we get BGR)
    if input_image_numpy.shape[0] == 3:
        input_image_numpy = input_image_numpy[..., [2,1,0]] #before it was input_image_numpy[..., ::-1] which is more general becasue input_image_numpy can indeed have 3 channels BUT it can also be grayscale
    return input_image_numpy

############################################################################################################



############################################################################################################
#Saliency Maps:
def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency
############################################################################################################



############################################################################################################
def get_parameters_for_tests(example_index):
    #This Function is relevant for the Use Examples presented here....
    #When i'll use these visualization functions in my own model i will need to get the returned variables according to my framework
    """
        Gets used variables for almost all visualizations, like the image, model etc.
    Args:
        example_index (int): Image id to use from examples
    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    original_images_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
    example_list = [[os.path.join(original_images_folder,'snake.jpg'), 56],
                    [os.path.join(original_images_folder,'cat_dog.png'), 243],
                    [os.path.join(original_images_folder,'spider.png'), 72]]
    # selected_example = example_index
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = cv2.imread(img_path, 1)
    # Process image
    preprocessed_image = preprocess_image_ImageNet(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    return (original_image,
            preprocessed_image,
            target_class,
            file_name_to_export,
            pretrained_model)
########################################################################################################################################################################################################





#(3). Deep Visualizers- Class Activation Maximization:
#######################################################################################################################################################################################################
class Visualizer_Class_Activation_Maximizing_Image_Generator():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    #TODO: i can optimize not only the input image but any layer output really... generalize to this
    #TODO: make this more modular and applicable - transfer target_class from __init__ to generatr(), also initial_min_value and initial_max_value
    def __init__(self, model, folder_to_save_visualization_images, image_to_synthesize_size=(224,224,3), rand_image_initial_min_max = (0,255)):
        self.model = model
        self.save_folder = folder_to_save_visualization_images
        #Initialize created_image with random noise:
        #TODO: try and see what happens if i use speckles/blobs instead of absolute random noise
        self.initial_min_value = list(rand_image_initial_min_max)[0];
        self.initial_max_value = list(rand_image_initial_min_max)[1];
        self.created_image = np.uint8(np.random.uniform(self.initial_min_value, self.initial_max_value, image_to_synthesize_size))



    def generate(self, target_class, number_of_iterations=150, initial_learning_rate=6, save_frequency=5):
        self.model.eval()  # Put in Evaluation/Test Mode
        self.created_image = np.uint8(np.random.uniform(self.initial_min_value, self.initial_max_value, image_to_synthesize_size))
        # self.created_image = create_speckles_of_certain_size_in_pixels_multichannel(10, image_to_synthesize_size, (self.initial_min_value, self.initial_max_value));
        for i in range(1, number_of_iterations):
            # Process image and return variable
            self.processed_image = preprocess_image_ImageNet(self.created_image)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, target_class]
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = inverse_preprocessing_image_ImageNet(self.processed_image)
            # Save image
            if i % save_frequency == 0 or i==number_of_iterations-1:
                cv2.imwrite(self.save_folder + '/c_specific_iteration_'+str(i)+'.jpg', self.created_image)
            self.model.train()
        return self.processed_image




class Visualizer_Class_Activation_Maximizing_Image_Modifier():
    """
        Modifies an image to maximize a certain class with gradient ascent
    """

    # TODO: i can optimize not only the input image but any layer output really... generalize to this
    # TODO: make this more modular and applicable - transfer target_class from __init__ to generatr(), also initial_min_value and initial_max_value


    #TODO: add perceptual loss to the below defined class loss to change the image in a way which will maximize model confidence in a certain decision but keep it realistic!!!
    #TODO: should probably add regularization like L2!!!!
    def __init__(self, model, folder_to_save_visualization_images):
        self.model = model
        self.save_folder = folder_to_save_visualization_images


    def generate(self, input_image, target_class, number_of_iterations=150, initial_learning_rate=6, save_frequency=5):
        self.model.eval()  # Put in Evaluation/Test Mode
        # Define optimizer for the image
        optimizer = SGD([input_image], lr=initial_learning_rate)
        for i in range(1, number_of_iterations):
            # Forward
            output = self.model(input_image)
            # Target specific class
            class_loss = -output[0, target_class]
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Save image
            if i % save_frequency == 0 or i == number_of_iterations - 1:
                cv2.imwrite(self.save_folder + '/c_specific_iteration_' + str(i) + '.jpg', inverse_preprocessing_image_ImageNet(input_image))
        self.model.train()




# #Visualizer_Class_Activation_Maximizing_Image_Generator
# #Example Of Use:
# target_class = 130  # Flamingo
# folder_to_save_visualization_images = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\Visualizations'
# pretrained_model = models.alexnet(pretrained=True)
# image_to_synthesize_size = (224,224,3);
# rand_image_initial_min_max = (0,255)
# number_of_iterations = 300;
# initial_learning_rate = 6;
# save_frequency = 100;
# CA_maximization_image_generator_object = Visualizer_Class_Activation_Maximizing_Image_Generator(pretrained_model, folder_to_save_visualization_images, image_to_synthesize_size, rand_image_initial_min_max)
# CA_maximization_image_generator_object.generate(target_class, number_of_iterations,initial_learning_rate,save_frequency)
#
#
# #Visualizer_Class_Activation_Maximizing_Image_Modifier
# # Example Of Use:
# folder_to_save_visualization_images = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\Visualizations'
# original_images_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
# input_image_name = 'snake.jpg';
# full_image_path = os.path.join(original_images_folder, input_image_name)
# save_filename = 'snake';
#
# input_image_target_class = 56  # Snake
# pretrained_model = models.alexnet(pretrained=True)
# number_of_iterations = 10;
# initial_learning_rate = 10;
# save_frequency = 1;
#
# input_image = cv2.imread(full_image_path,1); #return BGR
# preprocessed_image = preprocess_image_ImageNet(input_image)
#
# CA_maximization_image_generator_object = Visualizer_Class_Activation_Maximizing_Image_Modifier(pretrained_model,
#                                                                                                 folder_to_save_visualization_images)
# CA_maximization_image_generator_object.generate(preprocessed_image, input_image_target_class, number_of_iterations, initial_learning_rate, save_frequency)
#######################################################################################################################################################################################################













#######################################################################################################################################################################################################
#(4). Deep Visualizer - BackPropagation Object To Get Image Pixels Gradients from Class Activation:
class Visualizer_VanillaBackprop_Image_Pixels_Gradients_From_Class_Activations():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    #Takes the classifier, force output to be a one_hot vector with the wanted/real target class, and back propagate to see which pixels are most active in making the model have that be the output.
    #Notice we're not optimizing anything here so no optimizer is initialized... we only use the gradient
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        #Register hook to the first layer (the hook is to the first layer to the grad_in because this is the image itself...and we wish to see which pixels in the image had the largest gradient):
        first_layer = get_network_layers_list_flat(self.model)[0];
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        self.model.eval()
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_numpy = self.gradients.data.numpy()[0]
        self.model.train()
        return gradients_numpy




class Visualizer_GuidedBackprop_Image_Pixel_Gradients_From_Class_Activation():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    #"Guided" Here Basically & Practically means the same as vanilla backpropagation but with all the relus or activations only backpropagating Positive Gradients... negative gradients are clipped to zero
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.clamp_relu_gradients_to_zero()
        self.hook_layers()

    def hook_layers(self):
        def get_gradients(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = get_network_layers_list_flat(self.model)[0];
        first_layer.register_backward_hook(get_gradients)

    def clamp_relu_gradients_to_zero(self):
        # Updates relu activation functions so that it only returns positive gradients
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU): #TODO: generalize to all other activations!!! such as leaky relu etc'
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        #TODO: having each model have a .featrues attribute is actually really smart and makes us have to think much less on the order by which we define our layer... add this fact to my functions like this
        #TODO: in functions like this is would be expressed as enumerate(get_network_layers_list_flat(self.model.features)).
        #TODO: understand if using get_network_layer_list_flat is smart because maybe i would like to use BLOCK_index instead of layer_index... perhapse i should incorporate this into my function.
        #TODO: i should probably have as possible inputs to the functions both layer_index (which would be ABSOLUTE index) and BLOCK_indices which would be a list or tuple of layered indices for block structures
        #TODO: NO WAIT!!!!.... actually that line of thinking probably only applies when i want to get a SPECIFIC LAYER... if i want to go through all the layers one by one i think this is still a good option
        for pos, module in enumerate(get_network_layers_list_flat(self.model)):
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_hook_function)


    def generate_gradients(self, input_image, target_class):
        self.model.eval()
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_numpy = self.gradients.data.numpy()[0]
        self.model.train()
        return gradients_numpy



def generate_smooth_grad(backpropagation_object, preprocessed_image, target_class, number_of_smoothing_noise_generations, noise_sigma_multiplier):
    """
        Generates smooth gradients of given Backprop type. You can use this with both vanilla
        and guided backprop
    Args:
        Backprop (class): Backprop type
        prep_img (torch Variable): preprocessed image
        target_class (int): target class of imagenet
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
    """
    # Generate an empty image/matrix
    smooth_grad = np.zeros(preprocessed_image.size()[1:])

    #Generate several noisy images and average resultant gradient image.
    #TODO: Parallelize!
    mean = 0
    sigma = noise_sigma_multiplier / (torch.max(preprocessed_image) - torch.min(preprocessed_image)).item()
    for x in range(number_of_smoothing_noise_generations):
        # Generate noise
        noise = Variable(preprocessed_image.data.new(preprocessed_image.size()).normal_(mean, sigma**2))
        # Add noise to the image
        noisy_img = preprocessed_image + noise
        # Calculate gradients
        current_grads = backpropagation_object.generate_gradients(noisy_img, target_class)
        # Add gradients to smooth_grad
        smooth_grad = smooth_grad + current_grads
    # Average it out
    smooth_grad = smooth_grad / number_of_smoothing_noise_generations
    return smooth_grad



# #Visualizer_VanillaBackprop_Image_Pixels_Gradients_From_Class_Activations
# # Get params
# folder_to_save_visualization_images = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\Visualizations'
# original_images_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
# pretrained_model = models.alexnet(pretrained=True)
# input_image_name = 'snake.jpg';
# full_image_path = os.path.join(original_images_folder, input_image_name)
# save_filename = 'snake';
#
# input_image_target_class = 56  # Snake
# input_image = cv2.imread(full_image_path,1); #NOTICE, here i NEED to use cv2.imread instead of my vanilla read_image_cv2 because the model expects the image to be BGR and not RGB
# # input_image = read_image_cv2(full_image_path,0,0)
# # imshow(input_image)
# preprocessed_image = preprocess_image_ImageNet(input_image)
#
#
# # Vanilla backprop
# vanilla_backprop_object = Visualizer_VanillaBackprop_Image_Pixels_Gradients_From_Class_Activations(pretrained_model)
# # Generate gradients
# vanilla_grads = vanilla_backprop_object.generate_gradients(preprocessed_image, input_image_target_class)
# # Save colored gradients
# save_gradient_images(vanilla_grads, folder_to_save_visualization_images, save_filename + '_Vanilla_BP_color')
# # Convert to grayscale
# grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
# # Save grayscale gradients
# save_gradient_images(vanilla_grads, folder_to_save_visualization_images, save_filename + '_Vanilla_BP_gray')
# print('Vanilla backprop completed')





# #Visualizer_GuidedBackprop_Image_Pixel_Gradients_From_Class_Activation
# # Get params
# folder_to_save_visualization_images = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\Visualizations'
# original_images_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
# pretrained_model = models.alexnet(pretrained=True)
# input_image_name = 'snake.jpg';
# full_image_path = os.path.join(original_images_folder, input_image_name)
# save_filename = 'snake';
#
# input_image_target_class = 56  # Snake
# input_image = cv2.imread(full_image_path,1); #NOTICE, here i NEED to use cv2.imread instead of my vanilla read_image_cv2 because the model expects the image to be BGR and not RGB
# # input_image = read_image_cv2(full_image_path,0,0)
# # imshow(input_image)
# preprocessed_image = preprocess_image_ImageNet(input_image)
#
#
# # Guided backprop
# guided_backprop_object = Visualizer_GuidedBackprop_Image_Pixel_Gradients_From_Class_Activation(pretrained_model)
# # Get gradients
# guided_grads = guided_backprop_object.generate_gradients(preprocessed_image, input_image_target_class)
# # Save colored gradients
# save_gradient_images(guided_grads, folder_to_save_visualization_images, save_filename + '_Guided_BP_color')
# # Convert to grayscale
# grayscale_guided_grads = convert_to_grayscale(guided_grads)
# # Save grayscale gradients
# save_gradient_images(grayscale_guided_grads, folder_to_save_visualization_images, save_filename + '_Guided_BP_gray')
# # Positive and negative saliency maps
# pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
# save_gradient_images(pos_sal, folder_to_save_visualization_images, save_filename + '_pos_sal')
# save_gradient_images(neg_sal, folder_to_save_visualization_images, save_filename + '_neg_sal')
# print('Guided backprop completed')




# #generate_smooth_grad
# # Get params
# folder_to_save_visualization_images = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\Visualizations'
# original_images_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
# pretrained_model = models.alexnet(pretrained=True)
# input_image_name = 'snake.jpg';
# full_image_path = os.path.join(original_images_folder, input_image_name)
# save_filename = 'snake';
#
# input_image_target_class = 56  # Snake
# input_image = cv2.imread(full_image_path,1); #NOTICE, here i NEED to use cv2.imread instead of my vanilla read_image_cv2 because the model expects the image to be BGR and not RGB
# # input_image = read_image_cv2(full_image_path,0,0)
# # imshow(input_image)
# preprocessed_image = preprocess_image_ImageNet(input_image)
#
# #Initialize Back-Propgagation Objects:
# vanilla_backprop_object = Visualizer_Pixels_Most_Dominant_In_Class_Activation_VanillaBackprop(pretrained_model)
# guided_backprop_object = Visualizer_Pixels_Most_Dominant_In_Class_Activation_GuidedBackprop(pretrained_model)
#
# #Smooth Gradients Parameters and Generation:
# number_of_smoothing_noise_generations = 50
# noise_sigma_multiplier = 4
# smooth_grad_vanilla_backprop = generate_smooth_grad(vanilla_backprop_object,
#                                    preprocessed_image,
#                                    input_image_target_class,
#                                    number_of_smoothing_noise_generations,
#                                    noise_sigma_multiplier)
# smooth_grad_guided_backprop = generate_smooth_grad(guided_backprop_object,
#                                    preprocessed_image,
#                                    input_image_target_class,
#                                    number_of_smoothing_noise_generations,
#                                    noise_sigma_multiplier)
#
# #Save Vanilla BackProp images:
# # Save colored gradients
# save_gradient_images(smooth_grad_vanilla_backprop, folder_to_save_visualization_images, save_filename + '_VanillaSmoothGrad_color')
# # Convert to grayscale
# grayscale_smooth_grad = convert_to_grayscale(smooth_grad_vanilla_backprop)
# # Save grayscale gradients
# save_gradient_images(grayscale_smooth_grad, folder_to_save_visualization_images, save_filename + '_VanillaSmoothGrad_gray')
# print('Smooth grad completed')
#
# #Save Guided BackProp images:
# # Save colored gradients
# save_gradient_images(smooth_grad_guided_backprop, folder_to_save_visualization_images, save_filename + '_GuidedSmoothGrad_color')
# # Convert to grayscale
# grayscale_smooth_grad = convert_to_grayscale(smooth_grad_guided_backprop)
# # Save grayscale gradients
# save_gradient_images(grayscale_smooth_grad, folder_to_save_visualization_images, save_filename + '_GuidedSmoothGrad_gray')
# print('Smooth grad completed')
#######################################################################################################################################################################################################




#
# #######################################################################################################################################################################################################
# #(5). Class Activation Map (CAM):
# class Visualizer_Class_Activation_Map_From_Layer_Outputs_Gradient_Weighted():
#     """
#         Produces class activation map
#     """
#     #Basically what this does is: you pick a layer and an input image (you can also choose a specific target class with the default being the network results),
#     #                             and then
#     def __init__(self, model):
#         self.model = model
#         self.layer_gradients = None;
#         self.layer_outputs = None
#
#     # Define Hook Layer which records inputs and outputs of dataflow (suitable for forward and backward hooks):
#     def hook_target_layer_outputs(self,target_layer):
#         def record_layer_output(module, grad_in, grad_out):  # forward hook function inputs: (module, inputs_to_layer, layer_outputs)
#             # Gets the conv output of the selected filter (from selected layer)
#             self.layer_outputs = grad_out[0]
#         # Hook the selected layer
#         get_network_layers_list_flat(self.model)[target_layer].register_forward_hook(record_layer_output);
#
#
#     def hook_target_layer_gradients(self, target_layer):
#         def get_gradients(module, grad_in, grad_out):
#             self.layer_gradients = grad_out[0][0] #grad_in[0] returns the gradients in tensor form (1,C,H,W) and grad_in[0][0] return (C,H,W). TODO: probably should change to grad_out
#         # Register hook to the first layer
#         get_network_layers_list_flat(self.model)[target_layer].register_backward_hook(get_gradients)
#
#
#     def generate_cam(self, input_image, target_layer, target_class=None):
#         # TODO:    FUTURE MORE ADVANCED MODEL DAGs:
#         # TODO: what about multiple input models? what about multiple output models?
#         # TODO: multiple input models can probably be handled by manually feeding an input list/tuple and have the model itself detect and handle such situations?
#         # TODO: it seems that for now the models that these functions are ready to accept are Sequential Models, preferably in the form of simple stacked layers only form  without real distinct
#         # TODO: functionality in the .forward() function of the model.
#
#         # TODO: i can perhapse like the loop the same way but if i get a BLOCK_indices variable i can CALCULATE layer_index
#         # TODO: i can have a variable such that if the model has a .features block then that is what is assigned to it and if not then the model itself is assigned to it. the only question is what about
#         # TODO: memory allocation??? as far as i know in python all assignments accept with the most basic types (int, float etc') are pointers... so i should probably not worry about memory allocation
#
#
#         # Full forward pass
#         self.model.eval()
#         self.hook_target_layer_gradients(target_layer)
#         self.hook_target_layer_outputs(target_layer)
#         model_output = self.model(input_image);
#         if target_class is None:
#             target_class = np.argmax(model_output.data.numpy())
#         # One-Hot Target for backprop
#         one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
#         one_hot_output[0][target_class] = 1
#         # Zero grads
#         self.model.zero_grad()
#         # Backward pass with specified target
#         model_output.backward(gradient=one_hot_output, retain_graph=True)
#         # Get hooked gradients
#         guided_gradients = self.layer_gradients.data.numpy()
#         # Get convolution outputs
#         target_layer_output = self.layer_outputs.data.numpy()
#
#         # Get weights from gradients
#         gradient_average_as_weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
#         # Create empty numpy array for cam
#         cam = np.ones(target_layer_output.shape[1:], dtype=np.float32)
#         # Multiply each weight with its conv output and then, sum
#         for i, w in enumerate(gradient_average_as_weights):
#             cam += w * target_layer_output[i, :, :]
#         # Make the CAM from the target layer the size of the original image for superimposition and stretch the dynamic range:
#         cam = cv2.resize(cam, (224, 224))
#         cam = np.maximum(cam, 0)
#         cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
#         cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
#         self.model.train()
#         return cam
#
#
#
# #Guided grad cam:
# def Weigh_Gradient_By_CAM(grad_cam_mask, guided_backprop_mask):
#     """
#         Guided grad cam is just pointwise multiplication of cam mask and
#         guided backprop mask
#     Args:
#         grad_cam_mask (np_arr): Class activation map mask
#         guided_backprop_mask (np_arr):Guided backprop mask
#     """
#     cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
#     return cam_gb
#
#
#
#
#
# #Visualizer_Class_Activation_Map_From_Layer_Outputs_Gradient_Weighted
# # Get params
# target_example = 2  # Snake
# (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#     get_parameters_for_tests(target_example)
# # Grad cam
# grad_cam_object = Visualizer_Class_Activation_Map_From_Layer_Outputs_Gradient_Weighted(pretrained_model)
# # Generate cam mask
# cam = grad_cam_object.generate_cam(prep_img, target_layer=11, target_class=target_class) #usually what is used here is the last layer before the classification head
# # Save mask
# save_class_activation_on_image(original_image, cam, folder_to_save_visualization_images,'blibli')
# print('Grad cam completed')
#
#
#
#
# #Weigh_Gradient_By_CAM
# # Get params
# folder_to_save_visualization_images = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\Visualizations'
# original_images_folder = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
# pretrained_model = models.alexnet(pretrained=True)
# input_image_name = 'snake.jpg';
# full_image_path = os.path.join(original_images_folder, input_image_name)
# save_filename = 'snake';
#
# input_image_target_class = 56  # Snake
# input_image = cv2.imread(full_image_path,1); #NOTICE, here i NEED to use cv2.imread instead of my vanilla read_image_cv2 because the model expects the image to be BGR and not RGB
# # input_image = read_image_cv2(full_image_path,0,0)
# # imshow(input_image)
# preprocessed_image = preprocess_image_ImageNet(input_image)
#
#
# # Grad cam
# grad_cam_object = Visualizer_Class_Activation_Map_From_Layer_Outputs_Gradient_Weighted(pretrained_model)
# # Generate cam mask
# cam = grad_cam_object.generate_cam(prep_img, target_layer=11, target_class=target_class)
# print('Grad cam completed')
#
# # Guided backprop
# guided_backprop_object = Visualizer_GuidedBackprop_Image_Pixel_Gradients_From_Class_Activation(pretrained_model)
# # Get gradients
# guided_grads = guided_backprop_object.generate_gradients(prep_img, target_class)
# print('Guided backpropagation completed')
#
# # Guided Grad cam
# cam_weighted_grad = Weigh_Gradient_By_CAM(cam, guided_grads)
# save_gradient_images(cam_weighted_grad, folder_to_save_visualization_images, '_GGrad_Cam')
# grayscale_cam_gb = convert_to_grayscale(cam_weighted_grad)
# save_gradient_images(grayscale_cam_gb, folder_to_save_visualization_images, '_GGrad_Cam_gray')
# print('Guided grad cam completed')
# #######################################################################################################################################################################################################
#
#
#
#
#
#
#
# #######################################################################################################################################################################################################
# #(5). Deep Dream:
# class Visualizer_DeepDream():
#     """
#         Produces an image that minimizes the loss of a convolution
#         operation for a specific layer and filter
#     """
#     def __init__(self, model, folder_to_save_visualization_images):
#         self.model = model
#         self.conv_output = 0
#         self.created_image = 0;
#         self.save_folder = folder_to_save_visualization_images;
#         self.layers_list = get_network_layers_list_flat(self.model);
#
#
#     def hook_target_layer_filter_output(self, layer_index, filter_index):
#         def record_layer_filter_output(module, grad_in, grad_out):
#             # Gets the conv output of the selected filter (from selected layer)
#             self.conv_output = grad_out[0, filter_index]
#         # Hook the selected layer
#         self.layers_list[layer_index].register_forward_hook(record_layer_filter_output)
#
#
#     def dream(self, input_image, layer_index, filter_index, number_of_iterations=251, save_frequency=20, lr=12, weight_decay=1e-4):
#         self.hook_target_layer_filter_output(layer_index, filter_index)
#         self.model.eval()
#         # Define optimizer for the image - input to optimizer is input_image, which is also what is being changed by the optimizer
#         # Earlier layers need higher learning rates to visualize whereas layer layers need less
#         optimizer = SGD([input_image], lr=lr,  weight_decay=weight_decay)
#         for i in range(1, number_of_iterations):
#             optimizer.zero_grad()
#             # Assign create image to a variable to move forward in the model
#             x = pretrained_model(input_image);
#             # Loss function is the mean of the output of the selected layer/filter
#             # We try to minimize the mean of the output of that specific filter
#             loss = -torch.mean(self.conv_output)
#             print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
#             # Backward
#             loss.backward()
#             # Update image
#             optimizer.step()
#             # Save image every save_frequency iterations
#             if i % save_frequency == 0 or i==number_of_iterations-1:
#                 # Recreate image
#                 created_image = inverse_preprocessing_image_ImageNet(input_image)
#                 cv2.imwrite(self.save_folder + '/ddream_l' + str(layer_index) +
#                             '_f' + str(filter_index) + '_iter'+str(i)+'.jpg',
#                             created_image)
#                 if i==number_of_iterations-1:
#                     return created_image;
#         self.model.train();


# # THIS OPERATION IS MEMORY HUNGRY! #
# # Because of the selected image is very large
# # If it gives out of memory error or locks the computer
# # Try it with a smaller image
# layer_index = -4
# filter_index = 94
# number_of_iterations = 100
# save_frequency = 20
# learning_rate = 100;
# weight_decay = 1e-4;
#
# folder_to_save_visualization_images = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\SR_GAN\Visualizations'
# save_name = ''
# folder_name = 'C:/Users\dkarl\PycharmProjects\dudy\KERAS\pytorch-cnn-visualizations-master\input_images';
# image_name = 'dd_tree.jpg';
# full_image_path = os.path.join(folder_name,image_name);
# input_image = cv2.imread(full_image_path);
# processed_image = preprocess_image_ImageNet(input_image, True)
#
# # Fully connected layer is not needed
# # pretrained_model = models.vgg19(pretrained=True).features
# pretrained_model = models.resnet18(True)
# dd = DeepDream(pretrained_model, folder_to_save_visualization_images)
# # This operation can also be done without Pytorch hooks
# # See layer visualisation for the implementation without hooks
# created_image = dd.dream(processed_image,layer_index,filter_index, number_of_iterations, save_frequency, learning_rate, weight_decay);
#######################################################################################################################################################################################################





#######################################################################################################################################################################################################
#(5). Occluder HeatMap:



#######################################################################################################################################################################################################









































#######################################################################################################################################################################################################
#Old visualization method - using ConvTranspose/DeConv with layer max activation searching to find out what the model is "most" looking at at the different layers (what's most active in the map) and then "backwalk" using convtranspose to the original image
#Actually i think this can prove an Important Visualization Method....The problem is how do you generalize this to other architectures instead of doing something specifically for VGG16?

class Vgg16Conv(nn.Module):
    """
    vgg16 convolution network architecture
    """

    def __init__(self, num_cls=1000):
        """
        Input
            number of class, default is 1k.
        """
        super(Vgg16Conv, self).__init__()

        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_cls),
            nn.Softmax(dim=1)
        )

        # index of conv
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        # feature maps
        self.feature_maps = OrderedDict()
        # switch
        self.pool_locs = OrderedDict()
        # initial weight
        self.init_weights()

    def init_weights(self):
        """
        initial weights from preptrained model by vgg16
        """
        vgg16_pretrained = models.vgg16(pretrained=True)
        # fine-tune Conv2d
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
        # fine-tune Linear
        for idx, layer in enumerate(vgg16_pretrained.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data

    def check(self):
        model = models.vgg16(pretrained=True)
        return model

    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
                # self.pool_locs[idx] = location
            else:
                x = layer(x)

        # reshape to (1, 512 * 7 * 7)
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output


class Vgg16Deconv(nn.Module):
    """
    vgg16 transpose convolution network architecture
    """

    def __init__(self):
        super(Vgg16Deconv, self).__init__()

        self.features = nn.Sequential(
            # deconv1
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),

            # deconv2
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, padding=1),

            # deconv3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, padding=1),

            # deconv4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, padding=1),

            # deconv5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, padding=1)
        )

        self.conv2deconv_indices = {
            0: 30, 2: 28, 5: 25, 7: 23,
            10: 20, 12: 18, 14: 16, 17: 13,
            19: 11, 21: 9, 24: 6, 26: 4, 28: 2
        }

        self.unpool2pool_indices = {
            26: 4, 21: 9, 14: 16, 7: 23, 0: 30
        }

        self.init_weight()

    def init_weight(self):
        vgg16_pretrained = models.vgg16(pretrained=True)
        for idx, layer in enumerate(vgg16_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data
                # self.features[self.conv2deconv_indices[idx]].bias.data\
                # = layer.bias.data

    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx] \
                    (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x



def decode_predictions(preds, top=5):
    """Decode the prediction of an ImageNet model
    # Arguments
        preds: torch tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return
    # Return
        A list of lists of top class prediction tuples
        One list of turples per sample in batch input.
    """


    class_index_path = 'https://s3.amazonaws.com\
/deep-learning-models/image-models/imagenet_class_index.json'

    class_index_dict = None

    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects a batch of predciton'
        '(i.e. a 2D array of shape (samples, 1000)).'
        'Found array with shape: ' + str(preds.shape)
        )

    if not os.path.exists('./data/imagenet_class_index.json'):
        r = requests.get(class_index_path).content
        with open('./data/imagenet_class_index.json', 'w+') as f:
            f.write(r.content)
    with open('./data/imagenet_class_index.json') as f:
        class_index_dict = json.load(f)

    results = []
    for pred in preds:
        top_value, top_indices = torch.topk(pred, top)
        result = [tuple(class_index_dict[str(i.item())]) + (pred[i].item(),) \
                for i in top_indices]
        result = [tuple(class_index_dict[str(i.item())]) + (j.item(),) \
        for (i, j) in zip(top_indices, top_value)]
        results.append(result)

    return results





def load_images(img_path):
    # imread from img_path
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # pytorch must normalize the pic by
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = transform(img)
    img.unsqueeze_(0)
    # img_s = img.numpy()
    # img_s = np.transpose(img_s, (1, 2, 0))
    # cv2.imshow("test img", img_s)
    # cv2.waitKey()
    return img


def store(model):
    """
    make hook for feature map
    """

    def hook(module, input, output, key):
        if isinstance(module, nn.MaxPool2d):
            model.feature_maps[key] = output[0]
            model.pool_locs[key] = output[1]
        else:
            model.feature_maps[key] = output

    for idx, layer in enumerate(model._modules.get('features')):
        # _modules returns an OrderedDict
        layer.register_forward_hook(partial(hook, key=idx))


def vis_layer(layer, vgg16_conv, vgg16_deconv):
    """
    visualing the layer deconv result
    """
    num_feat = vgg16_conv.feature_maps[layer].shape[1]

    # set other feature map activations to zero
    new_feat_map = vgg16_conv.feature_maps[layer].clone()

    # choose the max activations map
    act_lst = []
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :]
        activation = torch.max(choose_map)
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)
    mark = np.argmax(act_lst)

    choose_map = new_feat_map[0, mark, :, :]
    max_activation = torch.max(choose_map)

    # make zeros for other feature maps
    if mark == 0:
        new_feat_map[:, 1:, :, :] = 0
    else:
        new_feat_map[:, :mark, :, :] = 0
        if mark != vgg16_conv.feature_maps[layer].shape[1] - 1:
            new_feat_map[:, mark + 1:, :, :] = 0

    choose_map = torch.where(choose_map == max_activation,
                             choose_map,
                             torch.zeros(choose_map.shape)
                             )

    # make zeros for ther activations
    new_feat_map[0, mark, :, :] = choose_map

    # print(torch.max(new_feat_map[0, mark, :, :]))
    print(max_activation)

    deconv_output = vgg16_deconv(new_feat_map, layer, mark, vgg16_conv.pool_locs)

    new_img = deconv_output.data.numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    # cv2.imshow('reconstruction img ' + str(layer), new_img)
    # cv2.waitKey()
    return new_img, int(max_activation)


if __name__ == '__main__':

    img_path = './data/cat.jpg'

    # forward processing
    img = load_images(img_path)
    vgg16_conv = Vgg16Conv()
    vgg16_conv.eval()
    store(vgg16_conv)
    conv_output = vgg16_conv(img)
    pool_locs = vgg16_conv.pool_locs
    print('Predicted:', decode_predictions(conv_output, top=3)[0])

    # backward processing
    vgg16_deconv = Vgg16Deconv()
    vgg16_deconv.eval()
    plt.figure(num=None, figsize=(16, 12), dpi=80)
    plt.subplot(2, 4, 1)
    plt.title('original picture')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    plt.imshow(img)
    for idx, layer in enumerate([14, 17, 19, 21, 24, 26, 28]):
        # for idx, layer in enumerate(vgg16_conv.conv_layer_indices):
        plt.subplot(2, 4, idx + 2)
        img, activation = vis_layer(layer, vgg16_conv, vgg16_deconv)
        plt.title(str(layer) + ' layer, the max activations is ' + str(activation))
        # img = img[112:,112:,:]
        plt.imshow(img)
        # plt.colorbar()

    # plt.show()
    plt.savefig('result.jpg')
    print('result picture has save at ./result.jpg')




#######################################################################################################################################################################################################























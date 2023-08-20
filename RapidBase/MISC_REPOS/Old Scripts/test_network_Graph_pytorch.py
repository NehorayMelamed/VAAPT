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
from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))


def make_dot(output_variable, parameters_dictionary=None):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        output_variable: output Variable
        parameters_dictionary: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """

    # input_size = (3, 224, 224)
    # inputs = torch.randn(1, *input_size);
    # network = torchvision.models.resnet18()
    # outputs = network(Variable(inputs));
    # output_variable = outputs
    # parameters_dictionary = dict(network.named_parameters());


    ########################################################################################################################################################################
    #Make sure all parameters_dictionary values are of type Variable:
    if parameters_dictionary is not None:
        assert all(isinstance(current_parameter_value, Variable) for current_parameter_value in parameters_dictionary.values())
        #Build a dictionary "map" with the "keys" being UNIQUE IDENTIFIERS for the parameter value, the the "value" being the parameter name
        param_map = {id(v): k for k, v in parameters_dictionary.items()} # {unique_id_number : parameter_name}
    ########################################################################################################################################################################


    ########################################################################################################################################################################
    #Initialize "Graph" Diagraph object
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    diagraph_object = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    ########################################################################################################################################################################


    ########################################################################################################################################################################
    #Initialize a "set" object (an "unordered collection of unique elements"):
    seen_objects_set = set()

    #Function to "pretty print" size array:
    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    #The initial input variable to the make_dot function is what we term output_nodes (we map the graph by going BACKWARD)
    if not isinstance(output_variable,tuple):
        output_nodes = (output_variable.grad_fn,)
    else:
        output_nodes = tuple(v.grad_fn for v in output_variable)

    ########################################################################################################################################################################


    ###############################################################################################################################################################################
    #A recursive function which accepts a variable/stage-of-the-network and "back-travels" to map the network graph:
    def add_nodes(current_variable):
        #Check current_variable is not in seen_obejcts_set, meaning we didn't map it still, and if we did map it then do nothing (TODO: what about iterative networks?):
        if current_variable not in seen_objects_set:


            if torch.is_tensor(current_variable):
                #If this is a tensor then add a node showing tensor size (DOESN'T SEEM TO WORK NOW- MAKE IT HAPPEN!):
                diagraph_object.node(str(id(current_variable)), size_to_str(current_variable.size()), fillcolor='orange') # note: this used to show .saved_tensors in pytorch0.2, but stopped working as it was moved to ATen and Variable-Tensor merged
            elif hasattr(current_variable, 'variable'): #this means current_variable is something like AccumulateGrad "object"....meaning it's a parameter basically
                #If current_variable has an attribute "variable" then get it's name and size and add it to the graph
                #TODO: when i have something like this (will be either bias or weight) - i need to know how to add it to the appropriate place like in keras!!!! too many wires as it is
                u = current_variable.variable
                name = param_map[id(u)] if parameters_dictionary is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                diagraph_object.node(str(id(current_variable)), node_name, fillcolor='lightblue')
            elif current_variable in output_nodes:
                #If current_variable is in output_nodes defined at the beginning to be the initial input variable's .grad_fn then add it to the graph with a green color
                diagraph_object.node(str(id(current_variable)), str(type(current_variable).__name__), fillcolor='darkolivegreen1')
            else:
                #If current_variable is any other thing (not a tensor and doesn't have an attribute .variable) the add it to the graph as is:
                diagraph_object.node(str(id(current_variable)), str(type(current_variable).__name__))

            #Add current_variable to seen_objects_set
            seen_objects_set.add(current_variable)

            #Decide what to do next:
            if hasattr(current_variable, 'next_functions'):
                #If current_variable has an attribute .next_functions (meaning it's a .grad_fn), which are basically calculation inputs which are functions themselves, then loop over them
                for u in current_variable.next_functions:
                    #now for each of the .next_functions
                    if u[0] is not None:
                        diagraph_object.edge(tail_name=str(id(u[0])), head_name=str(id(current_variable))) #Add edge(???) with no label
                        add_nodes(u[0]) #call the add_nodes recursive function on each of the .next_functions
            if hasattr(current_variable, 'saved_tensors'):
                #TODO: i don't know of any object which has an attribute "saved_tensors"????
                for t in current_variable.saved_tensors:
                    #loop over each of the saved tensors and add an edge
                    diagraph_object.edge(str(id(t)), str(id(current_variable)))
                    add_nodes(t) #call the add_nodes recursive function on each of the .saved_tensors
    #######################################################################################################################################################################################################################################

    ########################################################
    #Activate the recursive add_nodes function on each of the input variables
    if isinstance(output_variable, tuple):
        for v in output_variable:
            add_nodes(v.grad_fn)
    else:
        add_nodes(output_variable.grad_fn)
    #########################################################

    #########################################################
    #Resize graph to fit in width(?):
    resize_graph(diagraph_object)
    #########################################################

    return diagraph_object


# #Example of use:
# inputs = torch.randn(1,*input_size);
# outputs = network(Variable(inputs));
# network_graph = make_dot(outputs, parameters_dictionary=dict(network.named_parameters()))
# network_graph.view(filename,directory);




# For traces

def replace(name, scope):
    return '/'.join([scope[name], name])


def parse(graph):
    scope = {}
    for n in graph.nodes():
        inputs = [i.uniqueName() for i in n.inputs()]
        for i in range(1, len(inputs)):
            scope[inputs[i]] = n.scopeName()

        uname = next(n.outputs()).uniqueName()
        assert n.scopeName() != '', '{} has empty scope name'.format(n)
        scope[uname] = n.scopeName()
    scope['0'] = 'input'

    nodes = []
    for n in graph.nodes():
        attrs = {k: n[k] for k in n.attributeNames()}
        attrs = str(attrs).replace("'", ' ')
        inputs = [replace(i.uniqueName(), scope) for i in n.inputs()]
        uname = next(n.outputs()).uniqueName()
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': n.kind(),
                             'inputs': inputs,
                             'attr': attrs}))

    for n in graph.inputs():
        uname = n.uniqueName()
        if uname not in scope.keys():
            scope[uname] = 'unused'
        nodes.append(Node(**{'name': replace(uname, scope),
                             'op': 'Parameter',
                             'inputs': [],
                             'attr': str(n.type())}))

    return nodes


def make_dot_from_trace(trace):
    # """ Produces graphs of torch.jit.trace outputs
    # Example:
    # >>> trace, = torch.jit.trace(model, args=(x,))
    # >>> dot = make_dot_from_trace(trace)
    # """
    # from tensorboardX
    if LooseVersion(torch.__version__) >= LooseVersion("0.4.1"):
        torch.onnx._optimize_trace(trace, torch._C._onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    elif LooseVersion(torch.__version__) >= LooseVersion("0.4"):
        torch.onnx._optimize_trace(trace, False)
    else:
        torch.onnx._optimize_trace(trace)
    graph = trace.graph()
    list_of_nodes = parse(graph)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for node in list_of_nodes:
        dot.node(node.name, label=node.name.replace('/', '\n'))
        if node.inputs:
            for inp in node.inputs:
                dot.edge(inp, node.name)

    resize_graph(dot)

    return dot


def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


import os
os.environ["PATH"] += os.pathsep + 'C:/Users\dkarl\AppData\Local\conda\conda\envs\dudy_test\Library/bin/graphviz/'
input_size = (3, 224, 224)
inputs = torch.randn(1, *input_size);
network = torchvision.models.resnet18()
outputs = network(Variable(inputs));
output_variable = outputs
parameters_dictionary = dict(network.named_parameters());
network_graph = make_dot(output_variable, parameters_dictionary=dict(network.named_parameters()))

filename = 'blabla';
directory  = os.getcwd();
network_graph.view(filename, directory);




# if __name__ == '__main__':
#     input_size = (3, 224, 224)
#     inputs = torch.randn(1, *input_size);
#     network = torchvision.models.resnet18()
#     outputs = network(Variable(inputs));
#     output_variable = outputs
#     parameters_dictionary = dict(network.named_parameters());
#     network_graph = make_dot(outputs, params=dict(network.named_parameters()))
#     network_graph.view(filename, directory);







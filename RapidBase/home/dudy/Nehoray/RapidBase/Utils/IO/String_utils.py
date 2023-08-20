
import glob
import os
import fnmatch
#from path import path
import sys
from fnmatch import filter
from functools import partial
from itertools import chain
from os import path, walk
import pathlib
from os import mkdir
# import lmdb
import numpy as np 
import cv2
import torch
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import *
from RapidBase.Utils.MISCELENEOUS import get_random_start_stop_indices_for_crop
# import numpngw

# Default Image: #
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import get_full_shape_torch

def scientific_notation(input_number,number_of_digits_after_point=2):
    format_string = '{:.' + str(number_of_digits_after_point) + 'e}'
    return format_string.format(input_number)

def decimal_notation(input_number, number_of_digits_after_point=2):
    if input_number == np.inf:
        return 'inf'
    elif input_number is np.nan:
        return 'NaN'
    else:
        output_number = int(input_number*(10**number_of_digits_after_point))/(10**number_of_digits_after_point)
        return str(output_number)





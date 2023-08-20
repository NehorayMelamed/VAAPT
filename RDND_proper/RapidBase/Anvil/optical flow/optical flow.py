"""
A unified API for optical flow
"""
import os

import numpy as np
from PIL import Image

from wrappers.wrappers_dict import get_optical_flow_algorithms_wrappers_dict
from temporary_utils import pick_class


def get_optical_flow_algorithm(algorithm_name: str = 'classic', checkpoint_path: str = None):
    possible_classes = get_optical_flow_algorithms_wrappers_dict()
    algorithm = pick_class(possible_classes, algorithm_name)
    return algorithm




folder = '/home/elisheva/Projects/IMOD general/data/DIV2K/DIV2K_train_HR_shifted/clean/Shift_5/0001'
images = []
for filename in os.listdir(folder):
    img = Image.open(os.path.join(folder, filename))
    if img is not None:
        images += [np.array(img)]
images = np.array(images)
model = get_optical_flow_algorithm('FlowNet1Simple')
optical_flow = model(images)
print(optical_flow)

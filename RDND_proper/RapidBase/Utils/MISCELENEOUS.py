from RapidBase.Basic_Import_Libs import *
import numpy as np
import collections
import os

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_dict(obj, name):
    with open(name.split('.pkl')[0] + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

def string_rjust(original_number, final_number_size):
    return str.rjust(str(original_number), final_number_size, '0')

def scientific_notation(input_number,number_of_digits_after_point=2):
    format_string = '{:.' + str(number_of_digits_after_point) + 'e}'
    return format_string.format(input_number)

def decimal_notation(input_number, number_of_digits_after_point=2):
    output_number = int(input_number*(10**number_of_digits_after_point))/(10**number_of_digits_after_point)
    return str(output_number)

def print_list(input_list):
    for i in input_list:
        print(i)

def print_dict(input_dict):
    for k in input_dict.keys():
        print(k + ':   ' + str(input_dict[k]))

def get_elements_from_list_by_indices(input_list, indices):
    return [input_list[i] for i in indices]

def get_random_start_stop_indices_for_crop(crop_size, max_number):
    start_index = random.randint(0, max(0, max_number-crop_size))
    stop_index = start_index + min(crop_size,max_number)
    return start_index, stop_index

_all_ = ['as_variable', 'as_numpy', 'mark_volatile']    #### NOTE!!!!!!@#!#! understand why when i put __all__ instead of _all_ i can't use the functions here!@#@!?#@!?#?#!?

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    if isinstance(obj, collections.Sequence):
        return [as_variable(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_variable(v) for k, v in obj.items()}
    else:
        return Variable(obj)

def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)

def mark_volatile(obj):
    if torch.is_tensor(obj):
        obj = Variable(obj)
    if isinstance(obj, Variable):
        obj.no_grad = True
        return obj
    elif isinstance(obj, collections.Mapping):
        return {k: mark_volatile(o) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [mark_volatile(o) for o in obj]
    else:
        return obj



def to_list_of_certain_size(input_number, number_of_elements):
    if type(input_number)==tuple or type(input_number)==list:
        if len(input_number)==1:
            return input_number*number_of_elements;
        else:
            return input_number;
    else:
        return [input_number]*number_of_elements

def to_tuple_of_certain_size(input_number, number_of_elements):
    if type(input_number)==tuple or type(input_number)==list:
        if len(input_number)==1:
            return tuple(list(input_number)*number_of_elements)
        else:
            return tuple(input_number)
    else:
        return tuple([input_number]*number_of_elements)

def permute_tuple_indices(input_tuple, new_positions):
    return tuple(input_tuple[i] for i in new_positions)

def permute_list_indices(input_tuple, new_positions):
    return list(input_tuple[i] for i in new_positions)

def make_list_of_certain_size(n_layers, *args):
    args_list = []
    for current_arg in args:
        if type(current_arg) != list:
            args_list.append([current_arg] * n_layers)
        else:
            assert len(current_arg) == n_layers, str(current_arg) + ' must have the same length as n_layers'
            args_list.append(current_arg)
    return args_list


# def combined_several_checkpoints_of_different_parts_of_the_network_into_one():
#     ### If Network has several parts, each coresponding to a different part of the network, like DVDNet, compine them to a single one: ###
#     model_spatial_file = path_fix_path_for_linux(r'/home/mafat\PycharmProjects\IMOD\models\DVDNet\dvdnet/model_spatial.pth')
#     model_temp_file = path_fix_path_for_linux(r'/home/mafat\PycharmProjects\IMOD\models\DVDNet\dvdnet/model_temp.pth')
#     model_combined_final_save_name = path_fix_path_for_linux(r'/home/mafat\PycharmProjects\IMOD\models\DVDNet\dvdnet/model_combined.pth')
#     state_spatial_dict = torch.load(model_spatial_file, map_location=torch.device('cuda'))
#     state_temp_dict = torch.load(model_temp_file, map_location=torch.device('cuda'))
#     state_spatial_dict = remove_dataparallel_wrapper(state_spatial_dict)
#     state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
#     model_DVDNet = DVDNet(temp_psz=5, mc_algo='DeepFlow')
#     model_DVDNet.model_spatial.load_state_dict(state_spatial_dict)
#     model_DVDNet.model_temporal.load_state_dict(state_temp_dict)
#
#     dictionary_to_save = {}
#     dictionary_to_save['model_state_dict'] = model_DVDNet.state_dict()
#     dictionary_to_save['variables_dictionary'] = None
#     dictionary_to_save['optimizer_state_dict'] = None
#     dictionary_to_save['complete_network'] = None
#     dictionary_to_save['complete_optimzier'] = None
#     torch.save(dictionary_to_save, model_combined_final_save_name)
#     combined_state_dict = torch.load(model_combined_final_save_name, map_location=torch.device('cuda'))
#     model_DVDNet.load_state_dict(combined_state_dict['model_state_dict'])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


class AverageMeter_Dict(object):
    def __init__(self):
        self.inner_dict = EasyDict()  # a dictionary containing the average value
        self.average_meters_dict = EasyDict()  # a dictionary containing the average_meter classes for every key
    def update_key(self, key, value):
        if key in self.inner_dict.keys():
            self.average_meters_dict[key].update(value)
            setattr(self, key, self.average_meters_dict[key].avg)
            self.inner_dict[key] = getattr(self, key)
        else:
            self.average_meters_dict[key] = AverageMeter()
            self.average_meters_dict[key].update(value)
            setattr(self, key, self.average_meters_dict[key].avg)
            self.inner_dict[key] = getattr(self, key)

    def update_dict(self, input_dict):
        for key, value in input_dict.items():
            self.update_key(key, value)
        return self.inner_dict

    def keys(self):
        return self.inner_dict.keys()

    def items(self):
        return self.inner_dict.items()

    def __getitem__(self, key):
        return self.inner_dict[key]


class KeepValuesHistory_Dict(object):
    def __init__(self):
        self.inner_dict = EasyDict()  # a dictionary containing the average value
        self.average_meters_dict = EasyDict()  # a dictionary containing the average_meter classes for every key

    def update_key(self, key, value):
        if key in self.inner_dict.keys():
            self.inner_dict[key].append(value)
            setattr(self, key, self.inner_dict[key])
        else:
            self.inner_dict[key] = []
            self.inner_dict[key].append(value)
            setattr(self, key, self.inner_dict[key])

    def update_dict(self, input_dict):
        for key, value in input_dict.items():
            self.update_key(key, value)
        return self.inner_dict

    def keys(self):
        return self.inner_dict.keys()

    def items(self):
        return self.inner_dict.items()

    def __getitem__(self, key):
        return self.inner_dict[key]


def update_dict(main_dict, input_dict):
    # update main dict with input dict
    default_dict = EasyDict(main_dict)
    if input_dict is None:
        input_dict = EasyDict()
    default_dict.update(input_dict)
    return default_dict

def convert_range(old_mat, new_range):
    old_min = old_mat.min()
    old_max = old_mat.max()
    new_min = new_range[0]
    new_max = new_range[1]
    old_range_delta = old_max-old_min
    new_range_delta = new_max-new_min
    new_mat = (old_mat-old_min)*(new_range_delta/old_range_delta) + new_min
    return new_mat

def get_random_number_in_range(min_num, max_num, array_size=(1)):
    return (np.random.random(array_size)*(max_num-min_num) + min_num).astype('float32')

import fnmatch
def string_match_pattern(input_string, input_pattern):
    return fnmatch.fnmatch(input_string, input_pattern)

def path_make_path_if_none_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def clean_up_filenames_list(input_list):
    i = 0
    flag_loop = True
    while flag_loop:
        if input_list[i] == []:
            del input_list[i]
        else:
            i += 1
        flag_loop = i < len(input_list)
    return input_list


import random
def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors, pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0, 100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def get_n_colors(number_of_colors=10, pastel_factor=0.6):
    colors = []
    for i in np.arange(number_of_colors):
        colors.append(generate_new_color(colors, pastel_factor=pastel_factor))

    colors = [list(np.array(current_color)*255) for current_color in colors]
    return colors


def create_folder_if_needed(folder_full_path):
    if not os.path.exists(folder_full_path):
        os.makedirs(folder_full_path)

def create_folder_if_doesnt_exist(folder_full_path):
    if not os.path.exists(folder_full_path):
        os.makedirs(folder_full_path)


def assign_attributes_from_dict(input_object, input_dict):
    for key in input_dict:
        setattr(input_object, key, input_dict[key])

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary
	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict




def create_empty_list_of_lists(number_of_elements):
    return [[] for x in np.arange(number_of_elements)]



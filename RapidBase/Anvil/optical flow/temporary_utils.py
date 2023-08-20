import os
from typing import Dict, Callable

import numpy as np
import torch


class InvalidTypeError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


def pick_class(classes: Dict[str, Callable], class_name: str, *args, **kwargs):  # can return anything
    # pass classes in dictionary, and constructor args in args and kwargs. Saves many lines of code
    # prune args of Null arguments
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("+++++++++++++++++++++++++")
    """
    args = [s_arg for s_arg in args if s_arg is not None]  # s_arg = single_arg
    kwargs = {arg_key: kwargs[arg_key] for arg_key in kwargs if kwargs[arg_key] is not None}
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("\n\n\n\n\n********************************************\n\n\n\n\n")
    """

    if class_name in classes.keys():
        return classes[class_name](*args, **kwargs)
    else:
        raise InvalidTypeError("Given optical flow algorithm not valid")


# Not my code I promise! Will clean this if I have some spare time
def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary


    Args:
        state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict


# Also not my code. Will clean this if I have some spare time
def load_model_from_checkpoint(Train_dict=None, model=None):
    ### Unpack: ###
    models_folder = Train_dict.models_folder
    load_Network_filename = Train_dict.load_Network_filename

    ### Check Whether we inserted a general full-filename to checkpoint, or "infrastructure compatible" filename: ###
    flag_is_filename_infrastructure_compatible = ('Step' in load_Network_filename) and ('.tar' in load_Network_filename)

    if flag_is_filename_infrastructure_compatible == False:
        ### Update Dict Network_checkpoint_step: ###
        Network_checkpoint_step = 0
        Train_dict.Network_checkpoint_step = Network_checkpoint_step

        if hasattr(model, 'load_parameters'):
            model.load_parameters(load_Network_filename)  # this network has a unique weights loading function
        else:
            # (*). Load Dictionary ((*). includes everything saved to checkpoint file):
            args_from_checkpoint = torch.load(load_Network_filename)

            ### state_dict: ###
            if args_from_checkpoint is not None:
                if 'model_state_dict' in args_from_checkpoint.keys():
                    pretrained_dict = args_from_checkpoint['model_state_dict']
                elif 'state_dict' in args_from_checkpoint.keys():
                    pretrained_dict = args_from_checkpoint['state_dict']
                else:
                    pretrained_dict = args_from_checkpoint
                if Train_dict.flag_remove_dataparallel_wrapper_from_checkpoint:
                    pretrained_dict = remove_dataparallel_wrapper(pretrained_dict)
                current_network_weights = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in current_network_weights}
                current_network_weights.update(pretrained_dict)
                model.load_state_dict(current_network_weights)  # pretrained_dict or model_dict?


    else:  # checkpoint's name is infrastructure compatible
        ### network checkpoint is encoded in checkpoint's name: ###
        Network_checkpoint_step = int(load_Network_filename.split('.tar')[0].split('Step')[1])

        ### New Convention: ###
        # (*). Get Specific Model Folder According To Convention
        # (models_folder_path->specific_model_named_folder->specific_checkpoint_name):
        path_Network = os.path.join(models_folder, load_Network_filename.split('_TEST')[0])
        path_Network = os.path.join(path_Network, str(load_Network_filename))

        # (*). Looad Dictionary ((*). includes everything saved to checkpoint file):
        args_from_checkpoint = torch.load(path_Network)

        ### state_dict: ###
        network_weights_from_checkpoint = args_from_checkpoint['model_state_dict']
        if network_weights_from_checkpoint is not None:
            pretrained_dict = network_weights_from_checkpoint
            current_network_weights = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in current_network_weights}
            current_network_weights.update(pretrained_dict)
            model.load_state_dict(current_network_weights)  # pretrained_dict or model_dict?

        ### variables_dictioanry: ###
        if args_from_checkpoint['variables_dictionary'] is not None:
            variables_dictionary = args_from_checkpoint['variables_dictionary']
        else:
            variables_dictionary = None

        Train_dict.Network_checkpoint_step = Network_checkpoint_step

    return model, Train_dict


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)

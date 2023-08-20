import os

import numpy as np
import torch
import shutil


def Network_to_CPU(Network, netF=None):
    Network = Network.to('cpu')
    # Network.hidden_states_to_device(device_cpu);
    # Network.reset_hidden_states()
    if netF:
        netF = netF.to('cpu');
    return Network;

def Network_to_GPU(Network,device=None):
    current_device = device
    Network = Network.to(current_device);
    return Network;


#################################################################################################################################################################################################################################################################################
##############################################   Allocation, Saving, Initialization & Auxiliary For Networks: ###########################################################################
from RapidBase.Utils.Tensor_Manipulation.Pytorch_Numpy_Utils import remove_dataparallel_wrapper
def load_model_from_checkpoint(Train_dict=None, model=None):
    ### Unpack: ###
    models_folder = Train_dict.models_folder
    load_Network_filename = Train_dict.load_Network_filename

    ### Check Whether we inserted a general full-filename to checkpoint, or "infrastructure compatible" filename: ###
    flag_is_filename_infrastructure_compatible = ('Step' in load_Network_filename) and ('.tar' in load_Network_filename)

    if flag_is_filename_infrastructure_compatible==False:
        ### Update Dict Network_checkpoint_step: ###
        Network_checkpoint_step = 0
        Train_dict.Network_checkpoint_step = Network_checkpoint_step

        if hasattr(model, 'load_parameters'):
            model.load_parameters(load_Network_filename)  #this network has a unique weights loading function
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


    else: #checkpoint's name is infrastructure compatible
        ### network checkpoint is encoded in checkpoint's name: ###
        Network_checkpoint_step = int(load_Network_filename.split('.tar')[0].split('Step')[1])

        ### New Convention: ###
        #(*). Get Specific Model Folder According To Convention
        # (models_folder_path->specific_model_named_folder->specific_checkpoint_name):
        path_Network = os.path.join(models_folder, load_Network_filename.split('_TEST')[0])
        path_Network = os.path.join(path_Network, str(load_Network_filename))

        #(*). Looad Dictionary ((*). includes everything saved to checkpoint file):
        args_from_checkpoint = torch.load(path_Network)

        ### state_dict: ###
        network_weights_from_checkpoint = args_from_checkpoint['model_state_dict']
        if network_weights_from_checkpoint is not None:
            pretrained_dict = network_weights_from_checkpoint
            current_network_weights = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in current_network_weights}
            current_network_weights.update(pretrained_dict)
            model.load_state_dict(current_network_weights)  #pretrained_dict or model_dict?

        ### variables_dictioanry: ###
        if args_from_checkpoint['variables_dictionary'] is not None:
            variables_dictionary = args_from_checkpoint['variables_dictionary']
        else:
            variables_dictionary = None

        Train_dict.Network_checkpoint_step = Network_checkpoint_step

    return model, Train_dict


def load_optimizer_from_checkpoint(Optimizer, Train_dict):
    ### Unpack: ###
    models_folder = Train_dict.models_folder
    load_Network_filename = Train_dict.load_Network_filename

    ### New Convention: ###
    # (*). Get Specific Model Folder According To Convention (models_folder_path->specific_model_named_folder->specific_checkpoint_name):
    path_Network = os.path.join(models_folder, load_Network_filename.split('_TEST')[0])
    path_Network = os.path.join(path_Network, str(load_Network_filename))

    # (*). Looad Dictionary ((*). includes everything saved to checkpoint file):
    args_from_checkpoint = torch.load(path_Network)

    ### Get Model Args: ###
    optimizer_state_dict = args_from_checkpoint['optimizer_state_dict']
    Optimizer.load_state_dict(optimizer_state_dict)

    return Optimizer


def save_dictionary_from_the_wild_to_conform_with_infrastructure():
    ### Load model from the web: ###
    current_path = r'/home/mafat/PycharmProjects/IMOD/models/GitHub/BasicSR/experiments/pretrained_models/EDVR/EDVR_L_deblur_REDS_official-ca46bd8c.pth'
    loaded_dict = torch.load(current_path)

    ### put state dict in the "correct" format for the infrastructure: ###
    model_state_dict = loaded_dict['params']  #TODO - NOTICE!!!!: can change from model to model found on the web
    dict_to_save = {}
    dict_to_save['model_state_dict'] = model_state_dict

    ### save new checkpoint: ###
    path_to_save = r'/home/mafat/PycharmProjects/IMOD/models/GitHub/BasicSR/experiments/pretrained_models/EDVR/EDVR_L_deblur_REDS.tar'
    torch.save(dict_to_save, path_to_save)


# def save_Network_parts_to_checkpoint(Network,folder,basic_filename,flag_save_dict_or_whole='whole'):
def save_Network_parts_to_checkpoint(Network, Network_checkpoint_folder, Optimizer, Train_dict):
    ### Unpack Dict: ###
    variables_dictionary = Train_dict.variables_dictionary
    basic_Network_filename = Train_dict.basic_Network_filename

    ### New Saving Convention: ###
    # current_device = Network.parameters().__iter__().__next__().device  # Not needed anymore
    # Network = Network_to_CPU(Network)

    ### Get Paths Needed: ###
    specific_network_checkpoints_folder = os.path.join(Network_checkpoint_folder, basic_Network_filename.split('_TEST1')[0])
    path_Network = os.path.join(specific_network_checkpoints_folder, str(basic_Network_filename)) + '.tar'

    ### Make Folders If They Don't Exist:
    path_make_path_if_none_exists(Network_checkpoint_folder)
    path_make_path_if_none_exists(specific_network_checkpoints_folder)

    ### .PTH / .TAR File With Dictionaries: ###
    dictionary_to_save = {}

    ### Save Network .state_dict: ###
    dictionary_to_save['model_state_dict'] = Network.state_dict()

    ### Variables I Want to save (HyperParameters etc'): ###
    if variables_dictionary is not None:
        dictionary_to_save['variables_dictionary'] = variables_dictionary
    else:
        dictionary_to_save['variables_dictionary'] = None


    ### Optimizer state_dict: ###
    if Optimizer is not None:
        dictionary_to_save['optimizer_state_dict'] = Optimizer.state_dict()
    else:
        dictionary_to_save['optimizer_state_dict'] = None

    ### Complete Network And Optimizer: ###
    dictionary_to_save['complete_network'] = None
    dictionary_to_save['complete_optimzier'] = None

    #### Save To File: ####
    torch.save(dictionary_to_save, path_Network)

    #### Network to GPU again: ####
    # Network = Network_to_GPU(Network, device=current_device) # Not needed anymore









#################################################################################################################################################################################################################################################################################
def PreTraining_Aux(Train_dict, Network_checkpoint_folder, project_path):
    ### Unpack Save/Load Dict: ###
    flag_use_new_or_checkpoint_network = Train_dict.flag_use_new_or_checkpoint_network
    flag_use_new_hyperparameters_or_checkpoint = 'new'  #TODO: this was responsible for taking the parameters from the current script or from checkpoint. now that there is no parameters file this means we must create another way to recreate the expirement without copy-pasting
    load_Network_filename = Train_dict.load_Network_filename
    Network_checkpoint_prefix = Train_dict.Network_checkpoint_prefix

    ### Adjust Network Checkpoint string according to script variables/prefixes/postfixes: ###
    ### DEFINE SAVE filenames and folders: ###
    setpoint_string = 'TEST1' + ''
    models_folder = Network_checkpoint_folder
    Network_checkpoint_postfix = setpoint_string + '.tar'

    ### Correct save name in case we're checkpoint mode: ###
    if flag_use_new_or_checkpoint_network == 'checkpoint':
        flag_is_filename_infrastructure_compatible = ('Step' in load_Network_filename) and ('.tar' in load_Network_filename)
        if flag_is_filename_infrastructure_compatible == False:
            # if the checkpoint is a general one from outside the infrastructure, simply take it's name: #
            if Train_dict.flag_continue_checkpoint_naming_convention:
                Network_checkpoint_prefix = os.path.split(load_Network_filename)[-1].split('.')[0]
            else:
                Network_checkpoint_prefix = Network_checkpoint_prefix + '_Checkpoint'
        else:
            # is the name given when saving, which will appear before the _TEST part of the saved name + any addition: ###
            if Train_dict.flag_continue_checkpoint_naming_convention:
                Network_checkpoint_prefix = load_Network_filename.split('_TEST')[0]
                Network_checkpoint_prefix = os.path.split(Network_checkpoint_prefix)[-1]
            else:
                Network_checkpoint_prefix = Network_checkpoint_prefix + '_Checkpoint'

    ### Append To Dict: ###
    Train_dict.setpoint_string = setpoint_string
    Train_dict.models_folder = models_folder
    Train_dict.Network_checkpoint_postfix = Network_checkpoint_postfix
    Train_dict.Network_checkpoint_prefix = Network_checkpoint_prefix
    Train_dict.Network_checkpoint_folder = Network_checkpoint_folder

    # ### Save Environment Scripts For Later Reproducibility (if this is the first time training the network): ###
    # copy_env_files(flag_use_new_or_checkpoint_network, Network_checkpoint_folder, Network_checkpoint_prefix, project_path, Train_dict)

    ### Save Main File For Later Review: ###
    path_specific_checkpoint = Network_checkpoint_folder + '/' + Network_checkpoint_prefix
    filename_main_to_save = path_specific_checkpoint + '/' + os.path.split(Train_dict.main_file_to_save_for_later)[-1]
    path_make_path_if_none_exists(Network_checkpoint_folder)
    path_make_path_if_none_exists(path_specific_checkpoint)
    # shutil.copy(Train_dict.main_file_to_save_for_later, filename_main_to_save)

    ### Write Down Description To TXT File: ###
    with open(os.path.join(path_specific_checkpoint,'network_description.txt'), 'w') as f:
        f.write(Train_dict.network_free_flow_description)

    return Train_dict



def Get_Network_Save_Basic_Names(global_step, Train_dict):
    ### Unpack Dict: ###
    Network_checkpoint_postfix = Train_dict.Network_checkpoint_postfix
    Network_checkpoint_prefix = Train_dict.Network_checkpoint_prefix
    Network_checkpoint_step = Train_dict.Network_checkpoint_step

    current_Network_iteration_string = 'Step' + str(global_step + Network_checkpoint_step)
    basic_Network_filename = Network_checkpoint_prefix + '_' + Network_checkpoint_postfix.split('.')[0] + '_' + current_Network_iteration_string
    return current_Network_iteration_string, basic_Network_filename


def Get_HyperParameters_Save_Flags(flag_first_save, Train_dict):
    ### Unpack Save/Load Dict: ###
    flag_use_new_or_checkpoint_network = Train_dict.flag_use_new_or_checkpoint_network
    flag_use_new_hyperparameters_or_checkpoint = 'new'
    load_Network_filename = Train_dict.load_Network_filename


    ### Decide Which Hyper-Parmaters To Save - From Project Or From Previous Checkpoint: ###
    if flag_use_new_or_checkpoint_network == 'new' and flag_first_save == True:
        flag_save_hyperparameters_from_project_or_checkpoint = 'project'
        explicit_hyperparameters_full_filename_to_save = ''
    elif flag_use_new_or_checkpoint_network == 'checkpoint' and flag_use_new_hyperparameters_or_checkpoint == 'new' and flag_first_save == True:
        flag_save_hyperparameters_from_project_or_checkpoint = 'project'
        explicit_hyperparameters_full_filename_to_save = ''
    elif flag_use_new_or_checkpoint_network == 'checkpoint' and flag_use_new_hyperparameters_or_checkpoint == 'checkpoint' and flag_first_save == True:
        flag_save_hyperparameters_from_project_or_checkpoint = 'checkpoint'
        explicit_hyperparameters_full_filename_to_save = os.path.join(Network_checkpoint_folder, load_Network_filename.split('.')[0].split('_TEST1')[0], (load_Network_filename.split('.')[0] + '_Parameters.py'))
    else:
        flag_save_hyperparameters_from_project_or_checkpoint = 'checkpoint'
        explicit_hyperparameters_full_filename_to_save = ''

    ### Append To Dict: ###
    Train_dict.explicit_hyperparameters_full_filename_to_save = explicit_hyperparameters_full_filename_to_save
    Train_dict.flag_save_hyperparameters_from_project_or_checkpoint = flag_save_hyperparameters_from_project_or_checkpoint
    return Train_dict
#######################################################################################################################################################################################################################################








#### Utils For Training Script: ###
#(*). Adjust checkpoint filename with the proper prefixes and postfixes for saving
def adjust_checkpoint_prefix(flag_use_new_or_checkpoint_network, load_Network_filename, Network_checkpoint_prefix_addition, Network_checkpoint_folder, Network_checkpoint_prefix):
    if flag_use_new_or_checkpoint_network == 'new':
        flag_use_new_or_checkpoint_optimizer = 'new'

    ### DEFINE SAVE filenames and folders: ###
    setpoint_string = 'TEST1' + ''
    models_folder = Network_checkpoint_folder
    Network_checkpoint_postfix = setpoint_string + '.tar'

    ### Correct save name in case we're checkpoint mode: ###
    if flag_use_new_or_checkpoint_network == 'checkpoint':
        ### Network_checkpoint_prefix, the name to be saved as from now on,
        # is the name given when saving, which will appear before the _TEST part of the saved name + any addition: ###
        Network_checkpoint_prefix = load_Network_filename.split('_TEST')[0]
        Network_checkpoint_prefix += Network_checkpoint_prefix_addition

    return setpoint_string, models_folder, Network_checkpoint_postfix, Network_checkpoint_prefix, Network_checkpoint_folder


#(*). get hyperparameters import string
def get_hyper_parameters_from_file(flag_use_new_or_checkpoint_network, flag_use_new_hyperparameters_or_checkpoint, load_Network_filename, Network_checkpoint_folder):
    ################################################################
    ### Get Hyper-Parameters From External "X_Paramaeters" File: ###
    if flag_use_new_or_checkpoint_network == 'new':
        # import DeepStereo_Training_Parameters
        # from DeepStereo_Training_Parameters import *
        parameters_file_import_string = 'DS5_Enhanced' + '_Training_Parameters'  #TODO: remember what, if anything, i'm doing with this string when we're not in checkpoint mode...maybe i don't need to do anything because the script by default draws the hyperparameters needed
    else:
        if flag_use_new_hyperparameters_or_checkpoint == 'checkpoint':
            # My_DeepStereo.
            # Model_New_Checkpoints.
            # Network Checkpoints' Folder
            # Checkpoint_name + '_Parameters
            parameters_file_import_string = project_path.split('/')[-1] + '.' + \
                                            os.path.split(Network_checkpoint_folder)[-1] + '.' + \
                                            load_Network_filename.split('.')[0].split('_TEST1')[0] + '.' + (
                                                        load_Network_filename.split('.')[0] + '_Parameters')
            execute_string = 'from ' + parameters_file_import_string + ' import *'
            exec(execute_string)  #actually load the HyperParameters file
        else:
            # import DeepStereo_Training_Parameters
            # from DeepStereo_Training_Parameters import *
            # parameters_file_import_string = global_project_name + '_Training_Parameters'
            parameters_file_import_string = 'Magic' + '_Training_Parameters'
    return parameters_file_import_string



### for reproducibility purposes simply copy the environment files to be able to always run this checkpoint when needed if it doesn't work in your current environmenet: ###
# from RapidBase.TrainingCore.misc.misc import path_get_files_recursively, path_get_residual_of_path_after_initial_path
def path_get_files_recursively(main_path, string_to_find='', flag_full_path=True):
    files = []
    for r,d,f in os.walk(main_path):
        for file in f:
            if string_to_find in file:
                if flag_full_path:
                    files.append(os.path.join(r, file))
                else:
                    files.append(file)
    return files
def path_get_residual_of_path_after_initial_path(full_path, initial_path):
    # return os.path.join(*str.split(full_path, initial_path)[1].split('\\')[1:])
    full_path = path_fix_path_for_linux(full_path)
    initial_path = path_fix_path_for_linux(initial_path)
    return os.path.join(str.split(full_path, initial_path)[1].split('/')[-1])
def path_fix_path_for_linux(path):
    return path.replace('\\','/')

def copy_env_files(flag_use_new_or_checkpoint_network, Network_checkpoint_folder, Network_checkpoint_prefix, project_path, Train_dict=None):
    ######################
    ### Copy environment files: ###
    ######################
    if flag_use_new_or_checkpoint_network == 'new':
        filenames_initial = path_get_files_recursively(project_path, '.py', True)
        filenames = [path_get_residual_of_path_after_initial_path(full_filename, project_path) for full_filename in filenames_initial]

        python_files = []
        python_files += [file for file in filenames if file.endswith('.py')]

        path_specific_checkpoint = Network_checkpoint_folder + '/' + Network_checkpoint_prefix
        path_saved_env_files = path_specific_checkpoint + '/env_files'

        path_make_path_if_none_exists(Network_checkpoint_folder)
        path_make_path_if_none_exists(path_specific_checkpoint)
        path_make_path_if_none_exists(path_saved_env_files)

        for file in python_files:
            if file.split('\\')[0] != 'Checkpoints':
                path_make_path_if_none_exists(os.path.join(path_saved_env_files, os.path.split(file)[0]))

        for file in python_files:
            import shutil
            if file.split('\\')[0] != 'Checkpoints':
                try:
                    shutil.copy(project_path + '/' + file, path_saved_env_files + '/' + file)
                except:
                    1

        ### Write down the file we're running: ###
        # with open('run_file.txt', 'w') as f:
            # f.write(Train_dict.running_file)


def path_make_path_if_none_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)



def minibatch_crop_center(variable, tw, th):
    _, _, w, h = variable.shape
    x1 = max(int(round((w - tw) / 2.)), 0)
    y1 = max(int(round((h - th) / 2.)), 0)

    x2 = min(x1 + tw, w-1)
    y2 = min(y1 + th, h-1)

    return variable[:, :, x1:x2, y1:y2]




def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)































import klepto
from klepto.archives import file_archive
from numpy import *
# from matplotlib import *
import os

import RapidBase.Basic_Import_Libs
from RapidBase.Basic_Import_Libs import *

# ### Other Utils: ###
# import RapidBase.Utils.Pytorch_Utils
# import RapidBase.Utils.Adding_Noise_To_Image
# import RapidBase.Utils.GPU_Utils
# import RapidBase.Utils.Imshow_and_Plots
# # import RapidBase.Utils.Klepto
# import RapidBase.Utils.linspace_arange
# import RapidBase.Utils.MISCELENEOUS
# import RapidBase.Utils.Path_and_Reading_utils
# import RapidBase.Utils.tic_toc
# from RapidBase.Utils.Pytorch_Utils import *
# from RapidBase.Utils.Adding_Noise_To_Image import *
# from RapidBase.Utils.IO.GPU_Utils import *
# from RapidBase.Utils.IO.Imshow_and_Plots import *
# # from RapidBase.Utils.IO.Klepto import *
# from RapidBase.Utils.Tensor_Manipulation.linspace_arange import *
# from RapidBase.Utils.MISCELENEOUS import *
# from RapidBase.Utils.IO.Path_and_Reading_utils import *
# from RapidBase.Utils.IO.tic_toc import *


def remove_variables_from_klepto_file(file_name,variable_names_list):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name + '.txt');

    if type(variable_names_list) == str:
        #input is a string:
        k = variable_names_list;
        db.archive.pop(k);
    else:
        #input is a list:
        for k in variable_names_list:
            db.archive.pop(k)


def load_variables_from_klepto_file(file_name):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name + '.txt');
    db.load();
    db_keys_list = list(db.keys());
    db_values_list = list(db.values());
    execution_string = '';
    for k,v in zip(db_keys_list,db_values_list):
        if type(v) != str:
            execution_string = execution_string + '\n' + k + '=' + str(v);
        else:
            execution_string = execution_string + '\n' + k + '=\'' + v + '\'';

    #Way of Using: exec(load_variables_from_klepto_file)
    return execution_string;


def save_variables_to_klepto_file(file_name,variables_dict):
    #do i need to do a different function to UPDATE variables or is it that if the variable exists
    #in the file that it automatically updates

    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name + '.txt');
    counter = 0;
    counter_total = 0;
    for k,v in variables_dict.items():
        #Try and pickle and see if was successfull:
        db[k] = v;
        try:
            db.dump();
        except:
            print(k);
            db.pop(k);
            counter += 1;

    db.dump();

    #Delete all garbage files saved parasitically to folder:
    garbage_files = search_file('I_*')
    for file_string in garbage_files:
        os.remove(file_string)

    return db;



    # variables_list = list();
    # for k,v in variables_dict.items():
    #     variables_list.append(type(v));
    # variables_list = set(variables_list);

def update_klepto(file_name,variables_dict):
    #do i need to do a different function to UPDATE variables or is it that if the variable exists
    #in the file that it automatically updates

    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name + '.txt');
    for k,v in variables_dict.items():
        db[k] = v;
    db.dump();


#### Save tic tocs:
def save_tic(baseline_name_finale):


    get_baseline_variables_str = '_baseline_variables = locals().copy();' \
                                   '_baseline_variables_keys = list(_baseline_variables.keys());' \
                                   '_baseline_variables_values = list(_baseline_variables.values());' \
                                   '_baseline_variables_keys_set' + str(baseline_name_finale) + ' = set(_baseline_variables_keys);'
    return get_baseline_variables_str;



def join_string_list_with_delimiter_in_between(string_list,delimiter):
    joint_string = '';
    for i in range(0,len(string_list),1):
        if i>0:
            joint_string += delimiter;
        joint_string += string_list[i];
    return joint_string

def save_toc(base_finale, post_finale, prefix_to_delete=None):


    saved_dictionary_name = 'saved_dictionary' + str(post_finale);

    check_if_dictionary_already_exists_string = 'try:\n' \
                       '    ' + saved_dictionary_name + '\n' \
                       'except NameError:\n' \
                       '    var_exists = False\n' \
                       'else:\n' \
                       '    var_exists = True\n'

    # #Possible Scope Hierarchy:
    # def temp1():
    #     bla4 = 4
    #     def temp2():
    #         bla1 = 1
    #         bla2 = 2
    #         return locals();
    #     bla3 = 3
    #     return locals();
    # bla = temp1();
    # blo = list(bla.items());
    # for i in range(0,len(blo),1):
    #     if str(type(blo[i][1]))=='<class \'function\'>':
    #         print(blo[i])


    # get_variables_declared_so_far_str = \
    #            '_final_variables = locals().copy();\n' \
    #            '_final_variables_keys = list(_final_variables.keys());\n' \
    #            '_final_variables_values = list(_final_variables.values());\n' \
    #            '_final_variables_keys_set = set(_final_variables_keys);\n' \
    #            'final_variables_dictionary = dict();\n' \
    #            'for k,v in zip(_final_variables_keys,_final_variables_values):\n' \
    #            '    final_variables_dictionary[k] = v;\n' \
    #            'relevant_keys_set = _final_variables_keys_set ^ _baseline_variables_keys_set' + str(base_finale) + ';\n' \
    #            'relevant_keys_set = [element for element in relevant_keys_set if element.startswith(\'_\')==False]\n' \
    #            'relevant_keys_set = list(relevant_keys_set);\n' \
    #            'saved_dictionary' + str(post_finale) + '= {key:final_variables_dictionary[key] for key in relevant_keys_set}\n';

    get_variables_declared_so_far_str = \
        '_final_variables = locals().copy();\n' \
        '_final_variables_keys = list(_final_variables.keys());\n' \
        '_final_variables_values = list(_final_variables.values());\n' \
        '_final_variables_keys_set = set(_final_variables_keys);\n' \
        'final_variables_dictionary = dict();\n' \
        'for k,v in zip(_final_variables_keys,_final_variables_values):\n' \
        '    final_variables_dictionary[k] = v;\n' \
        'relevant_keys_set = _final_variables_keys_set ^ _baseline_variables_keys_set' + str(base_finale) + ';\n' \
        'relevant_keys_set = [element for element in relevant_keys_set if element.startswith(\'_\')==False]\n' \
        'relevant_keys_set = list(relevant_keys_set);\n'   \
        'relevant_keys_set_modified = relevant_keys_set.copy(); \n' \
        'for i in range(0,len(relevant_keys_set_modified),1): \n' \
        '   if relevant_keys_set_modified[i].split(\'_\')[0]==\'' + str(prefix_to_delete) + '\': \n' \
        '       relevant_keys_set_modified[i] = join_string_list_with_delimiter_in_between(relevant_keys_set_modified[i].split(\'_\')[1:],\'_\') \n' \
        'saved_dictionary' + str(post_finale) + '= {key_modified:final_variables_dictionary[key_original] for key_modified,key_original in zip(relevant_keys_set_modified,relevant_keys_set)} \n';
        # 'saved_dictionary' + str(post_finale) + '= {key:final_variables_dictionary[key] for key in relevant_keys_set}\n';






    execution_string = 'if var_exists:\n' \
                       '    for k in ' + saved_dictionary_name + ':\n' \
                       '        ' + saved_dictionary_name + '[k] = k;\n' \
                       'else:\n'  + add_tab_before_each_row_in_string(get_variables_declared_so_far_str);

    execution_string = check_if_dictionary_already_exists_string + execution_string



    return execution_string;


def reload_variables(variable_dictionray):
    for k,v in variable_dictionray.items():
        if type(v) != str:
            execution_string = execution_string + '\n' + k + '=' + str(v);
        else:
            execution_string = execution_string + '\n' + k + '=\'' + v + '\'';

    #Way of Using: exec(load_variables_from_klepto_file)
    return execution_string;


def add_tab_before_each_row_in_string(input_string):
    input_string_split = input_string.split('\n');
    new_string = '';
    for counter in arange(0,len(input_string_split)):
        new_string += '    ' + input_string_split[counter] + '\n';

    return new_string;



################################################################################################################################################################################################################
# #get variables as dictionary and then as lists of keys and values
# _baseline_variables = locals().copy();
# _baseline_variables_keys = list(_baseline_variables.keys());
# _baseline_variables_values = list(_baseline_variables.values());
# _baseline_variables_keys_set = set(_baseline_variables_keys);
#
# _final_variables = locals().copy();
# _final_variables_keys = list(_final_variables.keys());
# _final_variables_values = list(_final_variables.values());
# _final_variables_keys_set = set(_final_variables_keys);
# #get dictionary with relevant variable names and their values:
# final_variables_dictionary = dict();
# for k,v in zip(_final_variables_keys,_final_variables_values):
#         final_variables_dictionary[k] = v;
# #get only relevant keys by removing union:
# relevant_keys_set = _final_variables_keys_set ^ _baseline_variables_keys_set;
# relevant_keys_set = [element for element in relevant_keys_set if element.startswith('_')==False]
# relevant_keys_set = list(relevant_keys_set);
# relevant_variables_dictionary = {key:final_variables_dictionary[key] for key in relevant_keys_set}
# #
################################################################################################################################################################################################################





def save_variables_to_klepto_file(file_name,variables_dict):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name);
    for k,v in variables_dict:
        db[k] = v;
    db.dump();



def remove_variables_from_klepto_file(file_name,variable_names_list):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name);

    if type(variable_names_list) == str:
        #input is a string:
        k = variable_names_list;
        db.archive.pop(k);
    else:
        #input is a list:
        for k in variable_names_list:
            db.archive.pop(k)




def load_variables_from_klepto_file(file_name):
    #Requires all klepto key names to be convertable to strings:
    db = file_archive(file_name);
    db.load();
    db_keys_list = list(db.keys());
    db_values_list = list(db.values());
    for k,v in zip(db_keys_list,db_values_list):
        exec( str(k) + '=' + str(v));



































def write_params_to_txt_file(params):
    ### Get parameters from params dict: ###
    FileName = params['FileName']
    DistFold = params['DistFold']
    ResultsFold = params['ResultsFold']
    roi = params['roi']
    utype = params['utype']

    ### make resuls path if none exists: ###
    filename_itself = os.path.split(FileName)[-1]
    experiment_folder = os.path.split(FileName)[0]
    Res_dir = os.path.join(experiment_folder, 'Results')
    text_full_filename = os.path.join(Res_dir, 'params.txt')
    create_folder_if_doesnt_exist(Res_dir)

    ### write down current parameters into params.txt format: ###
    open(text_full_filename, 'w+').write(str(params))

def split_string_to_name_and_value(input_string):
    split_index = str.find(input_string, ':')
    variable_name = input_string[0:split_index].replace(' ', '')
    variable_value = input_string[split_index + 1:].replace(' ', '')
    return variable_name, variable_value


def get_experiment_description_from_txt_file(params):
    ### Get Lines From Description File: ###
    experiment_description_text_file_full_filename = os.path.join(params.experiment_folder, 'Description.txt')
    experiment_description_string_list = open(experiment_description_text_file_full_filename).readlines()
    for line_index in np.arange(len(experiment_description_string_list)):
        if '\n' in experiment_description_string_list[line_index]:
            experiment_description_string_list[line_index] = experiment_description_string_list[line_index][0:-1]  # get rid of \n (line break)
        else:
            experiment_description_string_list[line_index] = experiment_description_string_list[line_index]
        key, value = split_string_to_name_and_value(experiment_description_string_list[line_index])
        print(key + ':    ' + str(value))
        params[key] = value

    return params






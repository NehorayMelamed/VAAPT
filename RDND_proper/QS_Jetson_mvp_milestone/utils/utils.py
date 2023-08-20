import argparse
import yaml
import numpy as np
import vpi


class MainMessage:
    """
    Represents a message from subprocess to the main process
    Every message has a header from constants.MainMessages
    """
    def __init__(self, header, data=None):
        self.__header = header
        self.__data = data

    def __str__(self):
        return f"header = {self.__header}\ndata = {self.__data}\n"

    def get_header(self):
        return self.__header

    def get_data(self):
        return self.__data


def parse_config(config_file_path):
    with open(config_file_path, 'r') as config_file:
        program_config = yaml.safe_load(config_file)

    input_config = program_config["input_kaya"]
    bgs_config = program_config["bgs"]
    ransac_config = program_config["ransac"]
    flask_config = program_config["flask"]
    return input_config, bgs_config, ransac_config, flask_config


def select_vpi_dtype(input_dtype):
    if input_dtype == np.uint8:
        return vpi.U8
    elif input_dtype == np.uint16:
        return vpi.U16
    elif input_dtype == np.float32:
        return vpi.F32
    elif input_dtype == np.float64:
        return vpi.F64


def parse_args():
    """
    Argument parser for command line arguments
    Possible args:
        - config_file: relative path to config file from config directory
    """
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('config_file', type=str, default="default_config.yml",
                        help='config file name inside config directory')

    program_command_line_args = parser.parse_args()
    return program_command_line_args


def get_data_from_queue(queue, cond):
    cond.acquire()
    while queue.empty():
        cond.wait()
    result = queue.get()
    cond.release()
    return result


def push_data_to_queue(data, queue, cond):
    cond.acquire()
    queue.put(data)
    cond.notify()
    cond.release()

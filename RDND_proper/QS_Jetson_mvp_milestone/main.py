# Python imports
import os
import argparse
from multiprocessing import Manager, Process

# Import from project specific packages
from flask_server import FlaskManager
from utils import KayaManager, Constants
from utils import parse_args, parse_config, select_vpi_dtype
from bg_sub import Subtractor


def main():
    """
    Main App
    Will initialize config and manage all the subprocesses in the system

    Subprocess 1 - Flask Listener for incoming HTTP messages
    Subprocess 2 - KAYA/Filesystem frame acquisition
    """

    working_directory = os.path.dirname(os.path.realpath(__file__))
    subprocess_dict = dict()

    # Read all program arguments and configuration
    program_cli_args = parse_args()
    input_config, bgs_config, ransac_config, flask_config = parse_config(os.path.join(working_directory, "config",
                                                                                      program_cli_args["config_file"]))

    # Multiprocessing Manager
    mp_manager = Manager()

    bgs_input_queue = mp_manager.Queue()
    bgs_input_cond = mp_manager.Condition(mp_manager.Lock())

    ransac_input_queue = mp_manager.Queue()
    ransac_input_cond = mp_manager.Condition(mp_manager.Lock())

    output_queue = mp_manager.Queue()
    output_cond = mp_manager.Condition(mp_manager.Lock())

    main_msg_queue = mp_manager.Queue()
    main_msg_cond = mp_manager.Condition(mp_manager.Lock())

    # Create Kaya Program Manager
    program_manager = KayaManager(bgs_input_queue, bgs_input_cond, ransac_input_queue, ransac_input_cond,
                                  output_queue, output_cond, main_msg_queue, main_msg_cond)

    input_width = int(input_config["height"])
    input_height = int(input_config["width"])
    input_fps = float(input_config["fps"])
    input_dtype = int(input_config["dtype"])
    roi_mode = int(input_config["multi_roi_mode"])

    offset_x = int(input_config["offset_x"])
    offset_y = int(input_config["offset_y"])
    max_gray_level = int(input_config["max_gray_level"])
    max_saturated_pixels = int(input_config["max_saturated_pixels"])
    saturation_range = int(input_config["saturation_range"])
    exposure_time = float(input_config["exposure_time"])

    # Background subtraction
    vpi_dtype = select_vpi_dtype(input_dtype)
    bgs_learn_rate = float(bgs_config["bgs_learn_rate"])
    bgs_threshold = float(bgs_config["bgs_threshold"])
    bgs_shadow = float(bgs_config["bgs_shadow"])
    bgs_n_sigma = float(bgs_config["bgs_n_sigma"])
    use_custom_bgs = bool(bgs_config["use_custom_bgs"])

    subtractor = Subtractor(program_manager, input_width, input_height, vpi_dtype,
                            bgs_learn_rate, bgs_threshold, bgs_shadow, bgs_n_sigma, use_custom_bgs)
    subtractor_process = Process(target=subtractor.bgs_main, name=Constants.ProcessNames.BGS, daemon=False)
    subprocess_dict.update({Constants.ProcessNames.BGS: subtractor_process})

    # Ransac

    # Flask remote listener
    flask_host = flask_config["flask_host"]
    flask_port = int(flask_config["flask_port"])
    gui_host = flask_config["remote_gui_host"]
    gui_port = int(flask_config["remote_gui_port"])
    flask_manager = FlaskManager(name=Constants.ProcessNames.FLASK, manager=program_manager,
                                 gui_host=gui_host, gui_port=gui_port, host=flask_host, port=flask_port)

    flask_process = Process(target=flask_manager.run, daemon=False, name=Constants.ProcessNames.FLASK)
    subprocess_dict.update({Constants.ProcessNames.FLASK: flask_process})

    # Start all program subprocesses
    for process_name, process in subprocess_dict.items():
        print(f"Starting subprocess {process_name}...")
        process.start()

    try:
        while True:
            message = program_manager.get_main_message()
            print(message)
            message_header = message.get_header()
            message_data = message.get_data()

            if message_header == Constants.MainMessages.BATCH_READY:
                print("Main Handling message BATCH_READY!")
            elif message_header == Constants.MainMessages.M_2:
                print("Main handling message M_2!")
            else:
                print("Main received an unknown message!")
                print("Ignoring... Please fix this!")

    except KeyboardInterrupt:
        print("Main App caught a keyboard interrupt. Exiting")
        exit(0)
    except Exception as e:
        print("Main App caught an unknown exception.")
        print(e)
    finally:
        for process_name, process in subprocess_dict.items():
            if process.is_alive():
                print(f"Stopping subprocess {process_name}...")
                process.terminate()
                process.join()

        print("Program successfully finished!")


if __name__ == "__main__":
    main()

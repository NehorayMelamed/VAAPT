# pip install requests
import os
import threading
import time
docker_image_name = "platerecognizer/alpr:latest"


def run_command_and_get_status(command, text_flag_which_should_appear_for_success=None, as_thread=False, wait_time=0):
    # output_file_name = ".output_docker_run_command.txt"
    #
    # import subprocess
    # proc = subprocess.Popen('echo "to stdout"', shell=True, stdout=subprocess.PIPE, )
    # output = proc.communicate()[0]
    output_command = ''

    def run_command():
        # os.system(f"{command} > {output_file_name}")
        # time.sleep(wait_time)

        nonlocal output_command
        output_command = os.popen(command).read()

    if as_thread is True:
        tread_runner = threading.Thread(target=run_command)
        tread_runner.start()
    else:
        run_command()

    if text_flag_which_should_appear_for_success is not None:
        # with open(output_file_name, "r") as pointer_to_output_file:
        #     content = pointer_to_output_file.read()
        #     if text_flag_which_should_appear_for_success in content:
        if text_flag_which_should_appear_for_success in output_command:
            return True
        else:
            return False
    return True


def is_docker_up():

    # docker run --rm -t -p 8080:8080 -v license:/license -e LICENSE_KEY=8zZTYPGdNf -e TOKEN=ef341d9eea7d3d3545e9c0d9aaef2c2cb19ebb97 platerecognizer/alpr:latest

    docker_already_up_command = f"docker ps  | grep {docker_image_name}"
    return run_command_and_get_status(docker_already_up_command,
                                      text_flag_which_should_appear_for_success=docker_image_name,wait_time=1)


def wake_up_docker_system():
    # ToDo do it  more secured !!!!!!
    docker_run_command = f"docker run --rm -t -p 8080:8080 -v license:/license -e LICENSE_KEY=8zZTYPGdNf -e TOKEN=ef341d9eea7d3d3545e9c0d9aaef2c2cb19ebb97 {docker_image_name}"
    flag_to_find = "[worker-0] Ready!"

    if is_docker_up() is True:
        return True
    if run_command_and_get_status(docker_run_command, flag_to_find, as_thread=True, wait_time=1) is False:
        raise OSError(f"Failed to run docker command - {docker_run_command}")
    else:
        return True


def shutdown_docker_system():
    def stop_docker_by_docker_name():
        docker_stop_command_by_docker_name = f"docker stop $(docker ps -a -q  --filter ancestor={docker_image_name})"
        print(docker_stop_command_by_docker_name)
        return run_command_and_get_status(docker_stop_command_by_docker_name)

    return stop_docker_by_docker_name()


# wake_up_docker_system()


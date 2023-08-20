import requests
from utils import Constants, MainMessage
import json
import yaml

server_host = 'localhost'
server_port = 8000


def get_json():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.GET_JSON}"
    response = requests.post(url=url)


def set_json():
    # TODO: get the yaml file here
    yaml_path = ""
    # new_yaml = json.dumps(yaml.safe_load(yaml_path))
    new_yaml = json.dumps("this is some good stuff: its working")
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.SET_JSON}"
    response = requests.post(url=url, json={'new_yaml': new_yaml})


def get_raw_frame():
    # TODO: get the frame number
    frame_number = 0
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.RAW_FRAME}?frame_num={frame_number}"
    response = requests.post(url=url)


def get_point_cloud():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.PCL_DATA}"
    response = requests.post(url=url, )


def get_suspect_list():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.SUSPECT_LIST}"
    response = requests.post(url=url)


def get_suspect_picture():
    # TODO: get the suspect number
    suspect_number = 0
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.SUSPECT_PICTURE}?suspect_num={suspect_number}"
    response = requests.post(url=url)


def get_buffer_counter():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.BUFFER_COUNTER}"
    response = requests.post(url=url)


def stop():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.STOP_APP}"
    response = requests.post(url=url)


def start():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.START_APP}"
    response = requests.post(url=url)


def save():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.SAVE}"
    response = requests.post(url=url)


def profile():
    url = f"http://{server_host}:{server_port}/{Constants.FlaskURLS.PROFILE_APP}"
    response = requests.post(url=url)


get_json()
set_json()
get_raw_frame()
get_point_cloud()
get_suspect_list()
get_suspect_picture()
get_buffer_counter()
stop()
start()
save()
profile()

from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from utils import Constants, MainMessage
import yaml
import json
import requests


class EndpointAction(object):

    def __init__(self, action):
        self.action = action
        self.response = Response(status=200)

    def __call__(self, *args):
        self.action()
        return self.response


class FlaskManager(object):
    app = None

    def __init__(self, name, manager, gui_host, gui_port, host="0.0.0.0", port=5000):
        self.app = Flask(name)
        CORS(self.app)
        self.manager = manager
        self.__host = host
        self.__port = port

        self.gui_host = gui_host
        self.gui_port = gui_port

        self.config_path = "/home/efcom/Desktop/QS_Jetson_V2/config/default_config.yml"

        get_json_endpoint = "/" + Constants.FlaskURLS.GET_JSON
        self.add_endpoint(endpoint=get_json_endpoint,
                          endpoint_name=Constants.FlaskURLS.GET_JSON,
                          handler=self.get_json,
                          methods=["POST", "GET"])

        set_json_endpoint = "/" + Constants.FlaskURLS.SET_JSON
        self.add_endpoint(endpoint=set_json_endpoint,
                          endpoint_name=Constants.FlaskURLS.SET_JSON,
                          handler=self.set_json,
                          methods=["POST", "GET"])

        get_raw_frame_endpoint = "/" + Constants.FlaskURLS.RAW_FRAME
        self.add_endpoint(endpoint=get_raw_frame_endpoint,
                          endpoint_name=Constants.FlaskURLS.RAW_FRAME,
                          handler=self.get_raw_frame,
                          methods=["POST", "GET"])

        get_point_cloud_endpoint = "/" + Constants.FlaskURLS.PCL_DATA
        self.add_endpoint(endpoint=get_point_cloud_endpoint,
                          endpoint_name=Constants.FlaskURLS.PCL_DATA,
                          handler=self.get_point_cloud,
                          methods=["POST", "GET"])

        get_suspect_list_endpoint = "/" + Constants.FlaskURLS.SUSPECT_LIST
        self.add_endpoint(endpoint=get_suspect_list_endpoint,
                          endpoint_name=Constants.FlaskURLS.SUSPECT_LIST,
                          handler=self.get_suspect_list,
                          methods=["POST", "GET"])

        get_suspect_picture_endpoint = "/" + Constants.FlaskURLS.SUSPECT_PICTURE
        self.add_endpoint(endpoint=get_suspect_picture_endpoint,
                          endpoint_name=Constants.FlaskURLS.SUSPECT_PICTURE,
                          handler=self.get_suspect_picture,
                          methods=["POST", "GET"])

        get_buffer_counter_endpoint = "/" + Constants.FlaskURLS.BUFFER_COUNTER
        self.add_endpoint(endpoint=get_buffer_counter_endpoint,
                          endpoint_name=Constants.FlaskURLS.BUFFER_COUNTER,
                          handler=self.get_buffer_counter,
                          methods=["POST", "GET"])

        stop_endpoint = "/" + Constants.FlaskURLS.STOP_APP
        self.add_endpoint(endpoint=stop_endpoint,
                          endpoint_name=Constants.FlaskURLS.STOP_APP,
                          handler=self.stop,
                          methods=["POST", "GET"])

        start_endpoint = "/" + Constants.FlaskURLS.START_APP
        self.add_endpoint(endpoint=start_endpoint,
                          endpoint_name=Constants.FlaskURLS.START_APP,
                          handler=self.start,
                          methods=["POST", "GET"])

        save_endpoint = "/" + Constants.FlaskURLS.SAVE
        self.add_endpoint(endpoint=save_endpoint,
                          endpoint_name=Constants.FlaskURLS.SAVE,
                          handler=self.save,
                          methods=["POST", "GET"])

        profile_endpoint = "/" + Constants.FlaskURLS.PROFILE_APP
        self.add_endpoint(endpoint=profile_endpoint,
                          endpoint_name=Constants.FlaskURLS.PROFILE_APP,
                          handler=self.profile,
                          methods=["POST", "GET"])

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=None):
        if methods is None:
            methods = ["POST"]

        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler), methods=methods)

    def get_json(self):
        with open(self.config_path, 'r') as yaml_file:
            json_config = json.dumps(yaml.safe_load(yaml_file))
        url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.GET_JSON}"
        response = requests.post(url=url, json={'config_file': json_config})

    def set_json(self):
        yaml_data = request.get_json()['new_yaml']
        # with open(self.config_path, 'r') as yaml_file:
        with open("new_yml.yml", 'w') as yaml_file:
            yaml.safe_dump(json.loads(yaml_data), yaml_file, allow_unicode=True)  # yaml_object will be a list or a dict
        print("save new yaml")

    def get_raw_frame(self):
        frame_number = int(request.args.get("frame_num", default=-1))
        # TODO: get the buffer
        raw_frames_buffer = []

        # TODO: review this logic
        if len(raw_frames_buffer) > frame_number:
            # TODO: json the raw_frame
            raw_frame = raw_frames_buffer[frame_number]
            url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.RAW_FRAME}?frame_num={frame_number}"
            response = requests.post(url=url, json={'raw_frame': raw_frame})
        else:
            # TODO: handle for specific errors? for now just general error response.
            url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.BAD_REQUEST}"
            response = requests.post(url=url)

    def get_point_cloud(self):
        # TODO: get the data
        pcl_format_data = []

        json_pcl_data = json.dumps(pcl_format_data)

        url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.PCL_DATA}"
        response = requests.post(url=url, json={'pcl_data': json_pcl_data})

    def get_suspect_list(self):
        # TODO: get the data
        suspect_list = []

        json_suspect_list = json.dumps(suspect_list)

        url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.SUSPECT_LIST}"
        response = requests.post(url=url, json={'suspect_list': json_suspect_list})

    def get_suspect_picture(self):
        suspect_number = int(request.args.get("suspect_num"))

        suspect_image_list = []

        if len(suspect_image_list) > suspect_number:
            raw_frame = suspect_image_list[suspect_number]
            url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.SUSPECT_PICTURE}?suspect_num={suspect_number}"
            response = requests.post(url=url, json={'suspect_picture': raw_frame})
        else:
            # TODO: handle for specific errors? for now just general error response.
            url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.BAD_REQUEST}"
            response = requests.post(url=url)

    def get_buffer_counter(self):
        # TODO: get the data
        grabbed_buffer_number = 0
        processed_buffer_number = 0
        pipeline_state = 1

        json_buffer_counter = json.dumps((grabbed_buffer_number, processed_buffer_number, pipeline_state))

        url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.BUFFER_COUNTER}"
        response = requests.post(url=url, json={'buffer_counter': json_buffer_counter})

    # TODO: handle for this 4 messages from main
    def stop(self):
        message = MainMessage(header=Constants.MainMessages.STOP_APP)
        print("stopped")
        # self.manager.put_message_for_main(message)

    def start(self):
        message = MainMessage(header=Constants.MainMessages.START_APP)
        print("started")
        # self.manager.put_message_for_main(message)

    def save(self):
        # TODO: handle somewhere else (different process? main?)
        # need to save current data (raw buffer, point cloud file, suspect freqency pivtures, suspect list JSON)
        message = MainMessage(header=Constants.MainMessages.SAVE)
        print("saved")
        # self.manager.put_message_for_main(message)

    def profile(self):
        # TODO: if handle from main:
        message = MainMessage(header=Constants.MainMessages.PROFILE_APP)
        print("profiled")
        # self.manager.put_message_for_main(message)
        # TODO: elif its easy to access this data from this process:
        blocks_runtimes = []

        json_blocks_runtimes = json.dumps(blocks_runtimes)
        url = f"http://{self.gui_host}:{self.gui_port}/{Constants.FlaskURLS.PROFILE_APP}"
        response = requests.post(url=url, json={'blocks_runtimes': json_blocks_runtimes})

    def run(self):
        try:
            # Run the app
            self.app.run(host=self.__host, port=self.__port)
        except KeyboardInterrupt:
            self.manager.logger.info("Flask app caught keyboard interrupt!")
        except Exception as e:
            self.manager.logger.error("Flask app caught an unknown exception!")
            self.manager.logger.error(e)


if __name__ == '__main__':
    server_host = 'localhost'
    server_port = 8000
    gui_host = "localhost"
    gui_port = 8001
    program_manager = ""
    flask_manager = FlaskManager(name=Constants.ProcessNames.FLASK, manager=program_manager,
                                 gui_host=gui_host, gui_port=gui_port,
                                 host=server_host, port=server_port)
    flask_manager.run()

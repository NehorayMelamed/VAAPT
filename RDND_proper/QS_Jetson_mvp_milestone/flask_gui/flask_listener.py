"""
BIG notice:
this is the GUI side
its a demo thats meant to demonstrate how to use the flask

"""
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

    def __init__(self, name, manager, host="0.0.0.0", port=5000):
        self.app = Flask(name)
        CORS(self.app)
        self.manager = manager
        self.__host = host
        self.__port = port

        get_json_endpoint = "/" + Constants.FlaskURLS.GET_JSON
        self.add_endpoint(endpoint=get_json_endpoint,
                          endpoint_name=Constants.FlaskURLS.GET_JSON,
                          handler=self.get_json,
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

        profile_endpoint = "/" + Constants.FlaskURLS.PROFILE_APP
        self.add_endpoint(endpoint=profile_endpoint,
                          endpoint_name=Constants.FlaskURLS.PROFILE_APP,
                          handler=self.profile,
                          methods=["POST", "GET"])

        bad_request_endpoint = "/" + Constants.FlaskURLS.BAD_REQUEST
        self.add_endpoint(endpoint=bad_request_endpoint,
                          endpoint_name=Constants.FlaskURLS.BAD_REQUEST,
                          handler=self.bad_request,
                          methods=["POST", "GET"])

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=None):
        if methods is None:
            methods = ["POST"]

        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler), methods=methods)

    def get_json(self):
        config_file = request.get_json()['config_file']
        print(f"got config_file")

    def get_raw_frame(self):
        frame_number = int(request.args.get("frame_num"))
        raw_frame = request.get_json()['raw_frame']
        print(f"got raw_frame number {frame_number}")

    def get_point_cloud(self):
        pcl_data = request.get_json()['pcl_data']
        print(f"got pcl_data")

    def get_suspect_list(self):
        suspect_list = request.get_json()['suspect_list']
        print(f"got suspect_list")

    def get_suspect_picture(self):
        suspect_number = int(request.args.get("suspect_num"))
        suspect_picture = request.get_json()['suspect_picture']
        print(f"got suspect_picture")

    def get_buffer_counter(self):
        buffer_counter = request.get_json()['buffer_counter']
        print(f"got buffer_counter")


    def profile(self):
        blocks_runtimes = request.get_json()['blocks_runtimes']
        print(f"got blocks_runtimes")

    def bad_request(self):
        # all requests that cant be completed (bad index etc..) will get here for now.
        print(f"got BAD_REQUEST")

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

    gui_host = "localhost"
    gui_port = 8001
    program_manager = ""
    flask_manager = FlaskManager(name=Constants.ProcessNames.FLASK, manager=program_manager,
                                 host=gui_host, port=gui_port)
    flask_manager.run()

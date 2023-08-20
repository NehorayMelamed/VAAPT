
from plate_recognizer_company.backend_api import *

from plate_recognizer_company_inner_api import *


# class VehicleAndLicense()
license_plate_template = {
        "Plate": {
            "Bbox": {},
            "License list": [],
            "License": None
        }
    }

vehicle_template = {
    "Vehicle": {
        "Type": None,
        "Color": None,
        "Orientation": None,
        "Family": None,
        "Model": None,
        "Region": None,
        "Make": None,

    }
}

image_path = "/home/dudy/Downloads/WhatsApp Image 2023-03-19 at 20.17.33.jpeg"

#
# def get_js_license_plate_data():
#     pass
#
# def get_js_vehicle_data():
#     pass
#
# def js_vehicle_and_license_plate_data():
#     pass
#
# output_data_js = get_objects_detection_api_call_data_of(image_path)
#
#
# try:
#     license_plate_template["Plate"]["Bbox"] = output_data_js[0]["plate"]["box"]
#     license_plate_template["Plate"]["License list"] = output_data_js[0]["plate"]["props"]["plate"]
#     license_plate_template["Plate"]["License"] = output_data_js[0]["plate"]["props"]["plate"][0]["value"]
#
#     vehicle_template["Type"] = output_data_js[0]["vehicle"]["type"]
# except Exception:
#     pass
#
#
# final_json_output = {vehicle_template, license_plate_template}

output_data_js = get_objects_detection_api_call_data_of(image_path)
output_data_js = {"PlateRecognizer-Company": output_data_js[0]}
print(1)
print(1)
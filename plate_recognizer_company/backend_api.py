import json
from enum import Enum
from typing import Union, List, Tuple
import requests

from plate_recognizer_company.docker_manager import wake_up_docker_system


# base on --> https://docs.platerecognizer.com/#on-premise-sdk



class OptionalObjectDetection(Enum):
    Vehicles = "Vehicles"
    Plates = "Plates"
    All = "All"


class PlateOrVehicleCoordinates(Enum):
    Vehicle = "Vehicle"
    Plate = "Plate"


class OptionalInfo(Enum):
    Nothing = "Nothing"
    Everything = "Everything"
    BestPlate = "BestPlate"
    AllPlates = "AllPlates"
    VehicleType = "Type"
    RegionCode = " RegionCode"
    RegionScore = "RegionScore"
    PlateDetectionScore = "PlateDetectionScore"
    PlateTextReadingScore = "PlateTextReadingScore"
    VehicleTypeDetectionScore = "VehicleTypeDetectionScore"


class OptionalVehicleType(Enum):
    BigTrack = "Big Truck"
    Bus = "Bus"
    Motorcycle = "Motorcycle"
    Pickup = "Pickup"
    Truck = "Truck"
    Sedan = "Sedan"
    SUV = "SUV"
    Van = "Van"
    Unknown = "Unknown"


class OptionalRegionLicense(Enum):
    Argentina = "ar"

    Armenia = "am"

    Australia = "au"

    Austria = "at"

    Azerbaijan = "az"

    Belarus = "by"

    Belgium = "be"

    Brazil = "br"

    Bulgaria = "bg"

    Canada = "ca"

    Chile = "cl"

    Colombia = "co"

    CostaRica = "cr"

    Croatia = "hr"

    Czechia = "cz"

    Denmark = "dk"

    Estonia = "ee"

    Finland = "fi"

    France = "fr"

    Georgia = "ge"

    Germany = "de"

    Greece = "gr"

    Hungary = "hu"

    India = "in"

    Israel = "il"

    Kazakhstan = "kz"

    Latvia = "lv"

    Lithuania = "lt"

    Luxembourg = "lu"

    MoldovaRepublic = "md"

    Monaco = "mc"

    Montenegro = "me"

    Netherlands = "nl"

    NewZealand = "nz"

    Norway = "no"

    Poland = "pl"

    Portugal = "pt"

    Romania = "ro"

    Serbia = "rs"

    Slovakia = "sk"

    SouthAfric = "za"

    Spain = "es"

    Sweden = "se"

    Switzerland = "ch"

    Thailand = "th"

    Turkey = "tr"

    Ukraine = "ua"

    UnitedArabEmirates = "ae"

    USA = "us"

    Uzbekistan = "uz"

    Vietnam = "vn"


def send_api_request(source, regions=[OptionalRegionLicense.Israel.value], text_formats_regular_ex=["[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"]) -> Union[bool, dict]:
    # ToDo do it more secured
    endpoint = 'http://localhost:8080/v1/plate-reader/'
    token = "ef341d9eea7d3d3545e9c0d9aaef2c2cb19ebb97"
    # region = "strict"
    config_params = json.dumps(dict(detection_mode="vehicle",
                                    text_formats=text_formats_regular_ex, # https://www.programiz.com/python-programming/regex
                                   ))
    # text_formats = json.dumps(dict(text_formats=["[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"]))
    try:
        with open(source, 'rb') as fp:
            response = requests.post(
                endpoint,
                data=dict(regions=regions, mmc=True, config=config_params),  # Optional
                files=dict(upload=fp),
                headers={'Authorization': 'ef341d9eea7d3d3545e9c0d9aaef2c2cb19ebb97'})
            # print("/")
            print(response.json())
            return response.json()
    except Exception as e:
        print(f"Failed to send api request, got an error - {e}")
        return False


def system_start_up_api_validation():
    if wake_up_docker_system() is False:
        print("Failed to start up system - docker is not running")
        return False
    return True


def get_objects_detection_api_call_data_of(source,
                                           specific_object: OptionalObjectDetection = OptionalObjectDetection.All,
                                           use_advanced_data_detector=False,
                                           regions=[OptionalRegionLicense.Israel.value]) -> List[dict]:
    response = send_api_request(source, regions)
    # print(response)
    output_list = []
    try:
        mainly_data_dict_response = response["results"]
    except Exception as e:
        print(f"Failed to get results via api call request - er - {e}")
        print("MAYBE THERE IS NO DATA FOR NOW")
        return False

    if specific_object == OptionalObjectDetection.Vehicles:
        for item in mainly_data_dict_response:
            for key, value in (item.items()):
                if key == "vehicle":
                    output_list.append(value)  # Append only vehicle data
        return output_list
    elif specific_object == OptionalObjectDetection.Plates:
        for item in mainly_data_dict_response:
            out_dict = {}
            for key, value in (item.items()):
                if key != "vehicle":
                    out_dict[key] = value
            output_list.append(out_dict)
        return output_list
    else:
        return mainly_data_dict_response


def get_object_info_and_coordinates_by_terms(source,
                                             plate_or_vehicle_coordinates: PlateOrVehicleCoordinates,
                                             desired_info: List[OptionalInfo] = [OptionalInfo.Everything],
                                             vehicle_type: OptionalVehicleType = None,
                                             min_vehicle_type_detection_score=None,
                                             region_license_plate_code: OptionalRegionLicense = None,
                                             min_region_license_plate_score=None,
                                             min_plate_detection_score=None,
                                             min_plate_text_reading_score=None,

                                             color=None, min_color_score=None,
                                             vehicle_orientation=None, min_vehicle_orientation_score=None,
                                             prediction_make=None, min_prediction_make_score=None,
                                             prediction_model=None,
                                             min_prediction_model_score=None) -> \
        Union[List[Tuple[dict, Tuple[int, int, int, int]]], List[Tuple[int, int, int, int]]]:
    if type(desired_info) != list:
        print(f"Invalid input argument - describer, expect to List, Got {type(desired_info)}")
        return []

    all_data_list = get_objects_detection_api_call_data_of(source)
    if all_data_list is False:
        return False
    out_list = []
    desired_info = [new_itm.value for new_itm in desired_info]

    for itm in all_data_list:
        # Decide to get vehicle or plate coordinates
        if plate_or_vehicle_coordinates == PlateOrVehicleCoordinates.Vehicle:
            try:
                box = itm["vehicle"]["box"]
            except Exception:
                pass
        elif plate_or_vehicle_coordinates == PlateOrVehicleCoordinates.Plate:
            try:
                box = itm["box"]
            except Exception:
                pass
        else:
            print("Invalid plate ot vehicle desired coordinates")
            return []

        if min_vehicle_type_detection_score is not None:
            if itm["vehicle"]["score"] < min_vehicle_type_detection_score:
                continue
        if vehicle_type is not None:
            if itm["vehicle"]['type'] != vehicle_type.value:
                continue
        if region_license_plate_code is not None:
            if itm['region']['code'] != region_license_plate_code.value:
                continue
        if min_region_license_plate_score is not None:
            if itm['region']['score'] < min_region_license_plate_score:
                continue
        if min_plate_detection_score is not None:
            if itm["score"] < min_plate_detection_score:
                continue
        if min_plate_text_reading_score is not None:
            if itm["dscore"] < min_plate_text_reading_score:
                continue

        out_info = {}
        try:
            # Building the info
            if OptionalInfo.Nothing.value in desired_info:
                # If we just want a coordinates, so we will append and return a list of Tuples
                out_list.append((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
            else:
                if OptionalInfo.AllPlates.value in desired_info or OptionalInfo.Everything.value in desired_info:
                    out_info["Plates"] = [optional_plate["plate"] for optional_plate in itm['candidates']].__str__()

                elif OptionalInfo.BestPlate.value in desired_info or OptionalInfo.Everything.value in desired_info:
                    out_info["BestPlate"] = itm['plate']

                if OptionalInfo.VehicleType.value in desired_info or OptionalInfo.Everything.value in desired_info:
                    try:
                        out_info["VehicleType"] = itm['vehicle']['type']
                    except Exception:
                        pass
                try:
                    if OptionalInfo.VehicleTypeDetectionScore.value in desired_info or OptionalInfo.Everything.value in desired_info:
                        out_info["VehicleTypeDetectionScore"] = itm['vehicle']['score']
                except Exception:
                    pass
                if OptionalInfo.RegionCode.value in desired_info or OptionalInfo.Everything.value in desired_info:
                    out_info["RegionCode"] = itm["region"]["code"]

                if OptionalInfo.PlateDetectionScore.value in desired_info or OptionalInfo.Everything.value in desired_info:
                    out_info["PlateDetectionScore"] = itm["score"]

                if OptionalInfo.PlateTextReadingScore.value in desired_info or OptionalInfo.Everything.value in desired_info:
                    out_info["PlateTextReadingScore"] = itm["dscore"]

                if OptionalInfo.RegionScore.value in desired_info or OptionalInfo.Everything.value in desired_info:
                    out_info["RegionCodeScore"] = itm['region']['score']

                out_list.append((out_info, (box['xmin'], box['ymin'], box['xmax'], box['ymax'])))
        except Exception:
            continue
    return out_list


# if "__main__" == __name__:
    # system_start_up_api_validation()
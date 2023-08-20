from plate_recognizer_company.backend_api import *


def get_total_vehicle(source) -> int:
    return len(get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Vehicle,
                                                        desired_info=[OptionalInfo.Nothing]))


def get_total_plats(source) -> int:
    return len(get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Plate,
                                                        desired_info=[OptionalInfo.Nothing]))


def get_vehicles_coordinates_with_best_plate_text(source) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.BestPlate],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Vehicle)
    new_result = []
    for itm in result:
        new_result.append((itm[0]["BestPlate"], itm[1]))
    return new_result


def get_plate_coordinates_and_confidence_and_class_name_of_object(source):
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.BestPlate,
                                                                    OptionalInfo.PlateDetectionScore],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Plate)
    new_result = []
    if not result:
        return []
    for itm in result:
        coordinates = list(itm[1])
        best_plate = itm[0]["BestPlate"]
        confidence = float(itm[0]["PlateDetectionScore"])
        new_result.append((coordinates, confidence, best_plate))
    print("new_result", new_result)
    return new_result


def get_vehicle_coordinates_and_confidence_and_class_name(source):
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.VehicleType,
                                                                    OptionalInfo.VehicleTypeDetectionScore],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Vehicle)
    new_result = []
    if result == []:
        return []
    for itm in result:
        coordinates = list(itm[1])
        vehicle_type = itm[0]["VehicleType"]
        confidence = float(itm[0]["VehicleTypeDetectionScore"])
        new_result.append((coordinates, confidence, vehicle_type))
    print("get_vehicle_coordinates_and_confidence_and_class_name", new_result)
    return new_result


def get_plate_coordinates_with_plates_text(source) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    result = get_object_info_and_coordinates_by_terms(source,
                                                      desired_info=[OptionalInfo.BestPlate],
                                                      plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Plate)
    new_result = []
    for itm in result:
        new_result.append((itm[0]["BestPlate"], itm[1]))
    return new_result


def get_vehicles_coordinates(source) -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Vehicle,
                                                    desired_info=[OptionalInfo.Nothing])


def get_plates_coordinates(source) -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Plate,
                                                    desired_info=[OptionalInfo.Nothing])


def get_plates_coordinates_with_best_plate(source) -> List[Tuple]:
    result = get_object_info_and_coordinates_by_terms(source, PlateOrVehicleCoordinates.Plate,
                                                      desired_info=[OptionalInfo.BestPlate])
    if result is False:
        return False
    new_result = []
    for itm in result:
        new_result.append((itm[0]["BestPlate"], itm[1]))
    return new_result


def get_vehicle_coordinates_by_terms(source,
                                     vehicle_type: OptionalVehicleType = None,
                                     min_vehicle_type_detection_score=None,
                                     region_license_plate_code: OptionalRegionLicense = None,
                                     min_region_license_plate_score=None,
                                     min_plate_detection_score=None,
                                     min_plate_text_reading_score=None, ) \
        -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source,
                                                    desired_info=[OptionalInfo.Nothing],
                                                    plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Vehicle,
                                                    vehicle_type=vehicle_type,
                                                    min_vehicle_type_detection_score=min_vehicle_type_detection_score,
                                                    region_license_plate_code=region_license_plate_code,
                                                    min_region_license_plate_score=min_region_license_plate_score,
                                                    min_plate_detection_score=min_plate_detection_score,
                                                    min_plate_text_reading_score=min_plate_text_reading_score)


def get_plate_coordinates_by_terms(source,
                                   vehicle_type: OptionalVehicleType = None,
                                   min_vehicle_type_detection_score=None,
                                   region_license_plate_code: OptionalRegionLicense = None,
                                   min_region_license_plate_score=None,
                                   min_plate_detection_score=None,
                                   min_plate_text_reading_score=None, ) \
        -> List[Tuple]:
    return get_object_info_and_coordinates_by_terms(source,
                                                    desired_info=[OptionalInfo.Nothing],
                                                    plate_or_vehicle_coordinates=PlateOrVehicleCoordinates.Plate,
                                                    vehicle_type=vehicle_type,
                                                    min_vehicle_type_detection_score=min_vehicle_type_detection_score,
                                                    region_license_plate_code=region_license_plate_code,
                                                    min_region_license_plate_score=min_region_license_plate_score,
                                                    min_plate_detection_score=min_plate_detection_score,
                                                    min_plate_text_reading_score=min_plate_text_reading_score)


# img = "/home/mafat/PycharmProjects/pythonProject1/data/Images/car_1.jpg"
# #
# print(get_plate_coordinates_and_confidence_and_class_name_of_object(img))

# print(1)

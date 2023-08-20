import os.path

import sys

import PARAMETER

# sys.path.append("/home/nehoray/PycharmProjects/Shaback/yolov8_tracking")
sys.path.append(f"{PARAMETER.BASE_PROJECT}/yolov8_tracking/")
# sys.path.append("/home/nehoray/PycharmProjects/Shaback/yolov8_tracking/yolov8_tracking_old/yolov8")

from flask import Flask, render_template, request, redirect, url_for


from yolov8_tracking.yolov8_tracking.backup_tracker import PlateRecognizerOptionalCarType, \
    PlateRecognizerOptionalCarOrientation, PlateRecognizerOptionalCarColor, interface_run

app = Flask(__name__)


def your_function(form_data):
    # Extract the form data
    source_format = form_data.get('source_format')
    reid_algorithm = form_data.get('reid_algorithm')
    segmentation_algorithm = form_data.get('segmentation_algorithm')
    speed_accuracy = form_data.get('speed_accuracy')
    show_realtime = form_data.get('show_realtime')
    tracker = form_data.get('tracker')
    car_type = form_data.getlist('vehicle_type[]')
    car_orientation = form_data.get('vehicle_orientation')
    car_color = form_data.getlist('vehicle_color[]')
    vehicle_model = form_data.get('vehicle_model')
    vehicle_make = form_data.get('vehicle_make')
    plate_color = form_data.get('license_plate_color')
    plate_regex = form_data.get('license_plate_regex')
    save_option = form_data.get('save_option')
    clean_database = form_data.get('clean_database')
    create_videos = form_data.get('create_videos')
    export_results = form_data.get('export_results')
    video_path = form_data.get('video_path')
    use_deblur = form_data.get('use_deblur')
    use_denoise = form_data.get('use_denoise')
    use_crop_video = form_data.get('use_crop_video')
    denoise_frames_stride = form_data.get('denoise_frames_stride')
    window_size_temporal = form_data.get('window_size_temporal')
    upload_file_format = form_data.get('upload_file_format')
    vehicle_confidence = request.form.get('vehicle_confidence')

    # Print the filename of the uploaded file
    if 'upload_file' in request.files:
        file = request.files['upload_file']
        print('Uploaded filename:', file.filename)
    else:
        print("Cannot found file uploading")
        exit(1)

    if source_format == None:  # ToDO: support other source
        exit("please insert file format")
        pass
    else:
        base_data_video = os.path.join(PARAMETER.BASE_PROJECT, "data", "videos")
        input_full_path_video = os.path.join(base_data_video, file.filename)
        source = input_full_path_video

    ## ToDo: make it more general
    # ToDo: make more option
    if reid_algorithm is None:
        input_reid_algorithm = "osnet_x0_25_market1501.pt"
    else:
        input_reid_algorithm = "osnet_x0_25_market1501.pt"


    input_yolo_weights = "last_best_1.pt"

    if speed_accuracy == "0":
        if segmentation_algorithm is None or segmentation_algorithm == "no":
            input_yolo_weights = "last_best_1.pt"
        else:
            input_yolo_weights= "yolov8n-seg.pt"

    elif speed_accuracy == "1":
        if segmentation_algorithm is None or segmentation_algorithm == "no":
            input_yolo_weights = "last_best_1.pt"
        else:
            input_yolo_weights = "yolov8n-seg.pt"

    elif speed_accuracy == "3":
        if segmentation_algorithm is None or segmentation_algorithm == "no":
            input_yolo_weights = "last_best_1.pt"
        else:
            input_yolo_weights = "yolov8x-seg.pt"

    if show_realtime is None or show_realtime == "no":
        input_show_realtime = False
    else:
        input_show_realtime = True

    input_tracker_method = tracker

    if "All" in car_type:
        input_cars_type_list = [car_type.value for car_type in PlateRecognizerOptionalCarType]
    else:
        input_cars_type_list = car_type

    if "All" in car_orientation:
        input_cars_orientation_list = [car_or.value for car_or in PlateRecognizerOptionalCarOrientation]
    else:
        input_cars_orientation_list = car_orientation

    if "All" in car_color:
        input_cars_color_list = [car_color.value for car_color in PlateRecognizerOptionalCarColor]
    else:
        input_cars_color_list = car_color

    # ToDo : support
    if vehicle_make == "":
        pass

    input_vehicle_make = None

    # ToDo : support
    if vehicle_model == "":
        input_vehicle_model = None
    input_vehicle_model = None

    # ToDo : support
    if plate_color == "all":
        input_plate_color = None
    else:
        input_plate_color = None

    # ToDo support
    if plate_regex == "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]":
        input_plate_regex = None
    else:
        input_plate_regex = None

    # ToDo spit it
    if save_option is not None or save_option == "yes":
        input_save_txt = True  # save results to *.txt
        input_save_conf = True  # save confidences in --save-txt labels
        input_save_crop = True  # save cropped prediction boxes
        input_save_trajectories = True  # save trajectories for each track
        input_save_vid = True
    else:
        input_save_txt = False  # save results to *.txt
        input_save_conf = False  # save confidences in --save-txt labels
        input_save_crop = False  # save cropped prediction boxes
        input_save_trajectories = False  # save trajectories for each track
        input_save_vid = False

    input_vehicle_confidence = 0.2
    # if vehicle_confidence is not None:
    #     try:
    #         input_vehicle_confidence = float(vehicle_confidence)
    #     except:
    #         input_vehicle_confidence = 0.5

    ##3 Save only the most common data base
    if clean_database is not None and clean_database != "no":
        input_clean_data_set = True
    else:
        input_clean_data_set = False

    if create_videos is not None and create_videos != "no":
        input_create_crop_videos = True
    else:
        input_create_crop_videos = True

    # ToDo support
    if export_results == "no":
        pass

    # Print the form data to the console
    print('Source format:', source_format)
    print('Re-identification algorithm:', reid_algorithm)
    print('Segmentation mask algorithm:', segmentation_algorithm)
    print('Speed vs accuracy:', speed_accuracy)
    print('Show result real time:', show_realtime)
    print('Tracker:', tracker)
    print('Car type:', car_type)
    print('Car orientation:', car_orientation)
    print('Car color:', car_color)
    print('Vehicle model:', vehicle_model)
    print('Vehicle make:', vehicle_make)
    print('License plate color:', plate_color)
    print('License plate regex:', plate_regex)
    print('Save option:', save_option)
    print('Clean database:', clean_database)
    print('Create videos:', create_videos)
    print('Export results:', export_results)
    print('Video path:', video_path)
    print('Use deblur:', use_deblur)
    print('Use denoise:', use_denoise)
    print('Use crop video:', use_crop_video)
    print('Denoise frames stride:', denoise_frames_stride)
    print('Window size temporal:', window_size_temporal)
    print('Upload file format:', upload_file_format)

    interface_run(source=source,
                  input_reid_algorithm=input_reid_algorithm,
                  input_yolo_weights=input_yolo_weights,
                  input_show_realtime=input_show_realtime,
                  input_tracker_method=input_tracker_method,
                  input_cars_type_list=input_cars_type_list,
                  input_cars_orientation_list=input_cars_orientation_list,
                  input_cars_color_list=input_cars_color_list,
                  input_vehicle_make=input_vehicle_make,
                  input_vehicle_model=input_vehicle_model,
                  input_plate_color=input_plate_color,
                  input_plate_regex=input_plate_regex,
                  input_save_txt=input_save_txt,
                  input_save_conf=input_save_conf,
                  input_save_crop=input_save_crop,
                  input_save_trajectories=input_save_trajectories,
                  input_save_vid=input_save_vid,
                  input_clean_data_set=input_clean_data_set,
                  input_create_crop_videos=input_create_crop_videos,
                  export_results=export_results,
                  input_vehicle_confidence=input_vehicle_confidence)

    # Return the result (in this case, we don't need to return anything)
    return


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def submit_form():
    form_data = request.form

    # Call your function here, passing in the form data as arguments
    your_function(form_data)

    # Redirect the user to the result page
    return render_template('result.html')



def run_flask_web_page():
    app.run(debug=False)

#
# if __name__ == '__main__':
#     run_flask_web_page()
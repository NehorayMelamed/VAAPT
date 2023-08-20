import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, TextBox, Button, Slider
from ipywidgets import Dropdown
from tkinter import Tk, filedialog
from collections import defaultdict

def bounding_box_distance(bbox1, bbox2):
    left1, top1, width1, height1 = bbox1
    right1, bottom1 = left1 + width1, top1 + height1

    left2, top2, width2, height2 = bbox2
    right2, bottom2 = left2 + width2, top2 + height2

    dx = max(left1 - right2, left2 - right1, 0)
    dy = max(top1 - bottom2, top2 - bottom1, 0)

    return np.sqrt(dx * dx + dy * dy)
def parse_line(line, class_map):
    parts = line.split()
    return {
        'frame': int(parts[0]),
        'id': int(parts[1]),
        'class_name': class_map[int(float(parts[2]))],
        'bbox_left': float(parts[3]),
        'bbox_top': float(parts[4]),
        'bbox_w': float(parts[5]),
        'bbox_h': float(parts[6]),
    }

def read_data(file_path, parse_line_func, class_map):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [parse_line_func(line, class_map) for line in lines]

def get_centroid(bbox):
    return bbox['bbox_left'] + bbox['bbox_w'] / 2, bbox['bbox_top'] + bbox['bbox_h'] / 2

def get_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def filter_data(data, target_id):
    """
    Filter the data to only include frames that contain the target object ID.
    """
    target_object_data = {}
    for frame_number, frame_data in data.items():
        for obj in frame_data:
            if obj['id'] == target_id:
                target_object_data[frame_number] = frame_data
                break
    return target_object_data



def save_images(video_path, data, target_id, distance_threshold, target_classes, output_dir, start_frame, end_frame):
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    frame_index = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    objects_closest_to_target = {}  # dict to hold the closest object to the target for each id
    while success:
        if start_frame <= frame_index <= end_frame:
            objects_in_frame = [d for d in data if d['frame'] == frame_index]

            target_object = None
            for obj in objects_in_frame:
                if obj['id'] == target_id:
                    target_object = obj
                    break

            if target_object is not None:
                for obj in objects_in_frame:
                    if obj['id'] != target_id and obj['class_name'] in target_classes:
                        distance = bounding_box_distance((target_object['bbox_left'], target_object['bbox_top'],
                                                          target_object['bbox_w'], target_object['bbox_h']), (
                                                         obj['bbox_left'], obj['bbox_top'], obj['bbox_w'],
                                                         obj['bbox_h']))

                        if distance <= distance_threshold:
                            bbox_color = (0, 255, 0)  # Green color for bounding box
                            cv2.rectangle(frame, (int(obj['bbox_left']), int(obj['bbox_top'])),
                                          (int(obj['bbox_left'] + obj['bbox_w']), int(obj['bbox_top'] + obj['bbox_h'])),
                                          bbox_color, 2)

                            # Draw a line between the objects
                            target_center = (int(target_object['bbox_left'] + target_object['bbox_w'] / 2),
                                             int(target_object['bbox_top'] + target_object['bbox_h'] / 2))
                            obj_center = (int(obj['bbox_left'] + obj['bbox_w'] / 2),
                                          int(obj['bbox_top'] + obj['bbox_h'] / 2))
                            cv2.line(frame, target_center, obj_center, (255, 0, 0), 2)

                            # Display the distance
                            distance_text_position = (int((target_center[0] + obj_center[0]) / 2),
                                                      int((target_center[1] + obj_center[1]) / 2))
                            cv2.putText(frame, f"{distance:.2f}", distance_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 1)

                            # Save image with the specified format
                            image_name = f"{distance:.2f}.jpg"
                            object_dir = os.path.join(output_dir, f"{obj['id']}_{obj['class_name']}")
                            os.makedirs(object_dir, exist_ok=True)
                            image_path = os.path.join(object_dir, image_name)

                            # Check if the image already exists and is closer to the target
                            if os.path.exists(image_path):
                                existing_distance = float(os.path.splitext(os.path.basename(image_path))[0].split('_')[1])
                                if existing_distance < distance:
                                    continue

                            cv2.imwrite(image_path, frame)

        success, frame = video.read()
        frame_index += 1

    video.release()

def get_target_objects(data):
    """
    Get a list of all objects in the data.
    """
    objects = []
    for frame_data in data.values():
        for obj in frame_data:
            if obj not in objects:
                objects.append(obj)
    return objects


def main():
    file_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/exp5/tracks/Downtest_ch0007_00010000262000000.txt'
    video_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/data/videos/shabak_videos/shabak_short_Downtest_ch0007_00010000262000000.mp4'
    output_dir = 'closes_object_dir'

    class_map = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        7: 'truck',
        5: 'bus',
    }

    data = read_data(file_path, parse_line, class_map)

    target_id = int(input("Enter target object ID: "))
    distance_threshold = float(input("Enter distance threshold: "))
    target_classes = input("Enter target classes separated by commas (e.g. person,car): ").split(',')
    start_frame = int(input("Enter start frame: "))
    end_frame = int(input("Enter end frame: "))

    save_images(video_path, data, target_id, distance_threshold, target_classes, output_dir,start_frame=start_frame,end_frame=end_frame)


if __name__ == '__main__':
    main()
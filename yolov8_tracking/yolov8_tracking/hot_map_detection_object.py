import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, TextBox


h, w = None, None


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

def filter_data(data, selected_classes, start_frame, end_frame):
    return [d for d in data if d['class_name'] in selected_classes and start_frame <= d['frame'] <= end_frame]

def get_heatmap_data(data, h):
    heatmap_data = {}
    for d in data:
        center_x = d['bbox_left'] + d['bbox_w'] / 2
        center_y = h - (d['bbox_top'] + d['bbox_h'] / 2)  # Flip the y-coordinate
        if d['class_name'] not in heatmap_data:
            heatmap_data[d['class_name']] = []
        heatmap_data[d['class_name']].append((center_x, center_y))
    return heatmap_data

def update_heatmap(ax, heatmap_data, selected_classes):
    ax.clear()
    background_image = cv2.imread(background_image_path)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    h, w, _ = background_image.shape

    ax.imshow(background_image, extent=[0, w, 0, h], aspect='auto')

    colors = plt.cm.get_cmap('hsv', len(selected_classes) + 1)
    for i, class_name in enumerate(selected_classes):
        if class_name in heatmap_data:
            points = np.array(heatmap_data[class_name])
            ax.scatter(points[:, 0], points[:, 1], color=colors(i), label=class_name)

    ax.legend()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal', adjustable='box')


def plot_heatmap_with_inputs():
    global selected_classes, start_frame, end_frame, h

    filtered_data = filter_data(data, [classes[i] for i in range(len(classes)) if selected_classes[i]], start_frame, end_frame)
    heatmap_data = get_heatmap_data(filtered_data, h)
    update_heatmap(ax, heatmap_data, [classes[i] for i in range(len(classes)) if selected_classes[i]])
    plt.draw()

def toggle_visibility(label):
    index = classes.index(label)
    selected_classes[index] = not selected_classes[index]
    plot_heatmap_with_inputs()

def submit_start_frame(start_text):
    global start_frame
    start_frame = int(start_text)
    plot_heatmap_with_inputs()

def submit_end_frame(end_text):
    global end_frame
    end_frame = int(end_text)
    plot_heatmap_with_inputs()

def main():
    global data, background_image_path, classes, selected_classes, start_frame, end_frame, fig, ax
    global  h, w
    file_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/exp/tracks/Downtest_ch0007_00010000262000000.txt'
    background_image_path = '/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/First_image.jpg'
    h, w, _ = cv2.imread(background_image_path).shape

    class_map = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        7: 'truck',
        5: 'bus',
    }

    classes = list(class_map.values())
    selected_classes = [True] * len(classes)
    start_frame = 0
    end_frame = 1000

    data = read_data(file_path, parse_line, class_map)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    rax = plt.axes([0.05, 0.05, 0.15, 0.15])
    check = CheckButtons(rax, classes, selected_classes)
    check.on_clicked(toggle_visibility)

    start_textbox = TextBox(plt.axes([0.3, 0.05, 0.1, 0.075]), 'Start Frame', initial=str(start_frame))
    start_textbox.on_submit(submit_start_frame)

    end_textbox = TextBox(plt.axes([0.5, 0.05, 0.1, 0.075]), 'End Frame', initial=str(end_frame))
    end_textbox.on_submit(submit_end_frame)

    plot_heatmap_with_inputs()

    plt.show()


if __name__ == '__main__':
    main()



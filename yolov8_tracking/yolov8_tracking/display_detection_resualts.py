import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons

class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog'
}

def plot_detection_results(input_file_path):
    global frame_interval

    class_counts = {}
    with open(input_file_path, 'r') as f:
        for line in f:
            fields = line.strip().split(' ')
            frame_index, object_id, class_num, bbox_left, bbox_top, bbox_w, bbox_h = fields
            class_name = class_names[int(float(class_num))]
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(int(frame_index))

    frame_interval = 50

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    def plot_frame_counts(selected_classes):
        global frame_interval
        ax.clear()
        for class_name, frame_indices in class_counts.items():
            if class_name in selected_classes:
                frame_counts = [frame_indices.count(i) for i in range(min(frame_indices), max(frame_indices) + 1, frame_interval)]
                time_counts = [i for i in range(len(frame_counts))]
                ax.plot(time_counts, frame_counts, label=class_name)

        ax.legend()
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Count')
        ax.set_title('Detection Results')
        plt.draw()

    plot_frame_counts(list(class_names.values()))

    # Add button to change the frame interval
    button_ax = plt.axes([0.8, 0.2, 0.1, 0.075])
    button = Button(button_ax, 'Set Interval')

    # Add text input box for setting the frame interval
    text_box_ax = plt.axes([0.8, 0.1, 0.1, 0.075])
    text_box = TextBox(text_box_ax, 'Interval', initial='1')

    def update_interval(new_interval):
        global frame_interval
        try:
            interval = int(new_interval)
            if interval > 0:
                frame_interval = interval
                plot_frame_counts(check.get_status())
            else:
                text_box.set_val(str(frame_interval))
        except ValueError:
            text_box.set_val(str(frame_interval))

    def set_frame_interval(event):
        update_interval(text_box.text)

    button.on_clicked(set_frame_interval)

    # Add checkboxes for selecting classes
    class_checkbox_ax = plt.axes([0.8, 0.4, 0.15, 0.4])
    selected_classes = [class_name for class_name in class_names.values()]
    check = CheckButtons(class_checkbox_ax, list(class_names.values()), [True]*len(class_names), )

    def update_classes(label):
        selected_classes = [class_names[i] for i, status in enumerate(check.get_status()) if status]
        plot_frame_counts(selected_classes)

    check.on_clicked(update_classes)
    plt.show()


txt_file = '/home/dudy/Nehoray/SHABACK_POC_NEW/object_tracking/git_repos/yolov8_tracking/runs/track/exp/tracks/Downtest_ch0007_00010000262000000.txt'
plot_detection_results(txt_file)

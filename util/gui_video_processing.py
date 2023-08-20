from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QProgressBar, \
    QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QFileDialog, QTextEdit, QSizePolicy
import os
import cv2


import numpy as np


import os
import site

from cut_video_by_gui import VideoEditor

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(site.getsitepackages()[0], "PyQt5", "Qt", "plugins", "platforms")
os.environ["QT_QPA_PLATFORM"] = "xcb"


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Processing App")
        self.directory_path = None
        self.video_path = None
        self.resize(800, 400)  # Adjust window size here

        mainLayout = QVBoxLayout()

        # Browse base directory button
        self.browse_directory_button = QPushButton("Browse base directory", self)
        self.browse_directory_button.clicked.connect(self.browse_directory)
        self.browse_directory_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        mainLayout.addWidget(self.browse_directory_button)

        # Browse video button
        self.browse_video_button = QPushButton("Browse video", self)
        self.browse_video_button.clicked.connect(self.browse_video)
        self.browse_video_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        mainLayout.addWidget(self.browse_video_button)

        # Video stabilization drop-down list
        self.stabilization_combobox = QComboBox(self)
        self.stabilization_combobox.addItems(["ECC", "Cross Correlation", "ECC entire image"])
        self.stabilization_combobox.setItemData(0, "ECC algorithm explanation", Qt.ToolTipRole)
        self.stabilization_combobox.setItemData(1, "Cross Correlation algorithm explanation", Qt.ToolTipRole)
        self.stabilization_combobox.setItemData(2, "ECC entire image algorithm explanation", Qt.ToolTipRole)
        mainLayout.addWidget(self.stabilization_combobox)

        # Display stabilized video button
        self.display_stabilized_button = QPushButton("Display Stabilized Video", self)
        self.display_stabilized_button.clicked.connect(self.display_stabilized_video)
        self.display_stabilized_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        mainLayout.addWidget(self.display_stabilized_button)

        # Video enhancement drop-down list
        self.enhancement_combobox = QComboBox(self)
        self.enhancement_combobox.addItems(["ECC base optical flow", "ECC base segmentation mask", "Denoise", "Deblur", "De-jpeg", "Dfocuse"])
        # Add tooltips to each item in the enhancement combobox
        self.enhancement_combobox.setItemData(0, "ECC base optical flow explanation\n this is for stabilize the "
                                                 "object, it can help when the object moves\n also, it base on "
                                                 "optical flow so u will need\n to select an area to stabilize",
                                              Qt.ToolTipRole)
        self.enhancement_combobox.setItemData(1, "ECC base segmentation mask explanation", Qt.ToolTipRole)
        self.enhancement_combobox.setItemData(2, "Denoise", Qt.ToolTipRole)
        self.enhancement_combobox.setItemData(3, "Deblur", Qt.ToolTipRole)
        self.enhancement_combobox.setItemData(4, "De-jpeg", Qt.ToolTipRole)
        self.enhancement_combobox.setItemData(5, "Dfocuse", Qt.ToolTipRole)

        # ... repeat this for the other items
        mainLayout.addWidget(self.enhancement_combobox)

        # Display enhanced video button
        self.display_enhanced_button = QPushButton("Display Enhanced Video", self)
        self.display_enhanced_button.clicked.connect(self.display_enhanced_video)
        self.display_enhanced_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        mainLayout.addWidget(self.display_enhanced_button)

        # Log box
        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(200)  # Adjust log box height here
        mainLayout.addWidget(self.log_box)

        self.setLayout(mainLayout)

    def log(self, message):
        self.log_box.append(message)  # append the message to the log box

    def browse_directory(self):
        self.directory_path = QFileDialog.getExistingDirectory(self, 'Select base directory')
        self.log(f"Base directory: {self.directory_path}")

    def browse_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Select video file')
        self.log(f"Video file: {self.video_path}")

        # Get the base name and extension of the input video
        video_base_name = os.path.basename(self.video_path)
        video_base_name_without_ext = os.path.splitext(video_base_name)[0]

        # Define the output directory and path
        output_directory = self.directory_path  # Use the selected output directory
        output_path = os.path.join(output_directory, video_base_name_without_ext + "_processed.mp4")

        # Instantiate VideoEditor after video file selection
        self.video_editor = VideoEditor(self.video_path, output_path)
        self.video_editor.process_video()  # or wherever you need to call process_video

    def save_video(self, video_capture, output_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        self.log("Video processing finished. Saved to " + output_path)

    def stabiliz_video(self, video_capture, output_path):
        # You should insert your implementation for the stabilization here
        self.save_video(video_capture, output_path)

    def enhance_video(self, video_capture, output_path):
        # You should insert your implementation for the enhancement here
        self.save_video(video_capture, output_path)

    def display_stabilized_video(self):
        if self.video_path is None:
            self.log("Please select a video file first")
            return
        stabilization_method = self.stabilization_combobox.currentText()
        output_path = os.path.join(self.directory_path, os.path.basename(self.video_path).split('.')[0] + '__stabilization_' + stabilization_method + '.mp4')

        video_capture = cv2.VideoCapture(self.video_path)
        self.log("Stabilizing video... This might take a while.")
        self.stabiliz_video(video_capture, output_path)
        video_capture.release()


        # Instantiate VideoEditor for stabilized video
        self.video_editor = VideoEditor(output_path, output_path, self)
        self.video_editor.process_video()  # or wherever you need to call process_video


    def display_enhanced_video(self):
        if self.video_path is None:
            self.log("Please select a video file first")
            return
        enhancement_method = self.enhancement_combobox.currentText()
        output_path = os.path.join(self.directory_path, os.path.basename(self.video_path).split('.')[0] + '__enhancement_' + enhancement_method + '.mp4')

        video_capture = cv2.VideoCapture(self.video_path)
        self.log("Enhancing video... This might take a while.")
        self.enhance_video(video_capture, output_path)
        video_capture.release()


app = QApplication([])
demo = AppDemo()
demo.show()
app.exit(app.exec_())

import os
import site

import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import time
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(site.getsitepackages()[0], "PyQt5", "Qt", "plugins", "platforms")
os.environ["QT_QPA_PLATFORM"] = "xcb"




class VideoEditor:
    def __init__(self, input_video_path, output_video_path):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.cap = VideoFileClip(self.input_video_path)
        self.fps = self.cap.fps
        self.frame_count = int(self.cap.duration * self.fps)
        self.is_playing = False
        self.last_moved = 'Navigate'

        plt.figure('Video')
        plt.subplot(111)
        plt.axis('off')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)

        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        self.start_line = plt.axvline(0, color='green')
        self.end_line = plt.axvline(0, color='red')

        self.start_bar = plt.axvline(0, color='green', linewidth=3)
        self.end_bar = plt.axvline(0, color='red', linewidth=3)
        self.navigate_bar = plt.axvline(0, color='blue', linewidth=3)

        self.navigate_bar_visible = False

        self.start_text = plt.text(0, -20, 'Start', fontsize=12, color='green', ha='center', va='top')
        self.end_text = plt.text(0, -20, 'End', fontsize=12, color='red', ha='center', va='top')
        self.navigate_text = plt.text(0, -20, 'Navigate', fontsize=12, color='blue', ha='center', va='top')

        self.start_coord = 0
        self.end_coord = self.frame_count - 1

    def process_video(self):
        def on_key_press(event):
            if event.key == 'p':
                self.is_playing = not self.is_playing
            elif event.key == 's':
                self.save_video()
                plt.close()
            elif event.key == 'q':
                plt.close()

        def on_mouse_move(event):
            if event.inaxes:
                x = event.xdata
                if self.start_bar.get_visible():
                    self.last_moved = 'Start'
                    self.start_coord = max(0, int(x))
                elif self.end_bar.get_visible():
                    self.last_moved = 'End'
                    self.end_coord = min(self.frame_count - 1, int(x))
                elif self.navigate_bar_visible:
                    self.last_moved = 'Navigate'
                    self.navigate_bar.set_x(x)
                    plt.draw()

        def on_mouse_click(event):
            if event.inaxes:
                x = event.xdata
                if self.start_bar.get_visible():
                    self.last_moved = 'Start'
                    self.start_coord = max(0, int(x))
                elif self.end_bar.get_visible():
                    self.last_moved = 'End'
                    self.end_coord = min(self.frame_count - 1, int(x))
                elif self.navigate_bar.get_visible():
                    self.last_moved = 'Navigate'
                    self.is_playing = not self.is_playing

        def update_visualization():
            self.start_line.set_xdata(self.start_coord)
            self.end_line.set_xdata(self.end_coord)
            self.start_bar.set_xdata(self.start_coord)
            self.end_bar.set_xdata(self.end_coord)

            if self.is_playing:
                self.navigate_bar_visible = False
                self.start_bar.set_visible(False)
                self.end_bar.set_visible(False)
                self.navigate_bar.set_visible(True)
            else:
                self.navigate_bar_visible = True
                self.start_bar.set_visible(True)
                self.end_bar.set_visible(True)
                self.navigate_bar.set_visible(False)

            plt.draw()

        plt.connect('key_press_event', on_key_press)
        plt.connect('motion_notify_event', on_mouse_move)
        plt.connect('button_press_event', on_mouse_click)

        update_visualization()

        plt.show()

    def save_video(self):
        clip = self.cap.subclip(self.start_coord / self.fps, self.end_coord / self.fps)
        clip.write_videofile(self.output_video_path)


if __name__ == "__main__":
    editor = VideoEditor('/home/nehoray/PycharmProjects/Shaback/data/videos/toem.mp4', 'output.mp4')
    editor.process_video()

import os
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import time

class VideoEditor:
    def __init__(self, input_video_path, output_video_path, UiManager=None):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.cap = cv2.VideoCapture(self.input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_playing = False
        self.last_moved = 'Navigate'
        self.UiManager = UiManager

        if not self.cap.isOpened():
            self.log("Error opening video stream or file")

        self.window_name = f'Video - {self.get_video_name()}'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Start', self.window_name, 0, self.frame_count - 1, self.nothing)
        cv2.createTrackbar('End', self.window_name, 0, self.frame_count - 1, self.nothing)
        cv2.createTrackbar('Navigate', self.window_name, 0, self.frame_count - 1, self.nothing)
        cv2.resizeWindow(self.window_name, 800, 600)  # Adjust the size as desired

    def log(self, msg):
        if self.UiManager is None:
            print(msg)
        else:
            self.UiManager.log(msg)

    @staticmethod
    def nothing(x):
        pass

    def process_video(self):
        while True:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            start_frame = cv2.getTrackbarPos('Start', self.window_name)
            end_frame = cv2.getTrackbarPos('End', self.window_name)

            if self.is_playing:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame + 1 >= end_frame:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    current_frame = start_frame
                else:
                    current_frame = current_frame + 1
            else:
                if self.last_moved == 'Start':
                    current_frame = start_frame
                elif self.last_moved == 'End':
                    current_frame = end_frame
                else:  # last_moved == 'Navigate'
                    current_frame = cv2.getTrackbarPos('Navigate', self.window_name)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

            ret, frame = self.cap.read()
            if not ret:
                break

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKeyEx(1) & 0xFF  # Use waitKeyEx instead of waitKey

            if self.is_playing:
                cv2.setTrackbarPos('Navigate', self.window_name, current_frame)

            if key == ord('p'):
                self.is_playing = not self.is_playing

            elif key == ord('s'):
                self.save_video(start_frame, end_frame)
                break

            elif key == ord('g'):
                self.cap.release()
                cv2.destroyAllWindows()
                self.log(f"Returning start frame {start_frame} and end frame {end_frame}.")
                return start_frame, end_frame

            elif key == ord('q'):
                break

            if cv2.getTrackbarPos('Start', self.window_name) != start_frame:
                self.last_moved = 'Start'
            elif cv2.getTrackbarPos('End', self.window_name) != end_frame:
                self.last_moved = 'End'
            elif cv2.getTrackbarPos('Navigate', self.window_name) != current_frame:
                self.last_moved = 'Navigate'

            # Adding a delay based on FPS of video to maintain original speed
            if self.is_playing:
                time.sleep(1 / self.fps)

        self.cap.release()
        cv2.destroyAllWindows()

    def save_video(self, start_frame, end_frame):
        self.cap.release()
        cv2.destroyAllWindows()

        clip = VideoFileClip(self.input_video_path)
        start_sec = start_frame / self.fps
        end_sec = end_frame / self.fps
        subclip = clip.subclip(start_sec, end_sec)
        subclip.write_videofile(self.output_video_path)
        self.log(f"Success to cut and save video - {self.output_video_path}")
        self.cap = cv2.VideoCapture(self.input_video_path)

    def get_video_name(self):
        return os.path.basename(self.input_video_path).split('.')[0]


# # Usage example
# editor = VideoEditor('/home/nehoray/PycharmProjects/Shaback/data/videos/toem.mp4', 'output.mp4')
# start, end = editor.process_video()
# print(f"Start frame: {start}, End frame: {end}")









# import os
# import cv2
# import numpy as np
# from moviepy.video.io.VideoFileClip import VideoFileClip
# import time
#
# class VideoEditor:
#     def __init__(self, input_video_path, output_video_path, UiManager=None):
#         self.input_video_path = input_video_path
#         self.output_video_path = output_video_path
#         self.cap = cv2.VideoCapture(self.input_video_path)
#         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
#         self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.is_playing = False
#         self.last_moved = 'Navigate'
#         self.UiManager = UiManager
#
#         if not self.cap.isOpened():
#             self.log("Error opening video stream or file")
#
#         self.window_name = f'Video - {self.get_video_name()}'
#         cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
#         cv2.createTrackbar('Start', self.window_name, 0, self.frame_count - 1, self.nothing)
#         cv2.createTrackbar('End', self.window_name, 0, self.frame_count - 1, self.nothing)
#         cv2.createTrackbar('Navigate', self.window_name, 0, self.frame_count - 1, self.nothing)
#         cv2.resizeWindow(self.window_name, 800, 600)  # Adjust the size as desired
#
#     def log(self, msg):
#         if self.UiManager is None:
#             print(msg)
#         else:
#             self.UiManager.log(msg)
#
#     @staticmethod
#     def nothing(x):
#         pass
#
#     def overlay_instructions(self, frame):
#         """Overlay instructions on the video frame."""
#         instructions = [
#             "Press 'p' to Play/Pause",
#             "Press 's' to Save the video",
#             "Press 'g' to Get the start and end frames and close",
#             "Press 'q' to Quit without saving",
#             "Use the 'Start', 'End', and 'Navigate' trackbars to select frames"
#         ]
#
#         y0, dy = 30, 30  # Starting y position and line spacing
#
#         for i, line in enumerate(instructions):
#             y = y0 + i * dy
#             cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
#
#
#     def process_video(self):
#         while True:
#             if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
#                 break
#
#             start_frame = cv2.getTrackbarPos('Start', self.window_name)
#             end_frame = cv2.getTrackbarPos('End', self.window_name)
#
#             if self.is_playing:
#                 current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
#                 if current_frame + 1 >= end_frame:
#                     self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#                     current_frame = start_frame
#                 else:
#                     current_frame = current_frame + 1
#             else:
#                 if self.last_moved == 'Start':
#                     current_frame = start_frame
#                 elif self.last_moved == 'End':
#                     current_frame = end_frame
#                 else:  # last_moved == 'Navigate'
#                     current_frame = cv2.getTrackbarPos('Navigate', self.window_name)
#
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
#
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#
#
#             # Overlay instructions
#             self.overlay_instructions(frame)
#
#
#             cv2.imshow(self.window_name, frame)
#
#
#             key = cv2.waitKeyEx(1) & 0xFF  # Use waitKeyEx instead of waitKey
#
#             if self.is_playing:
#                 cv2.setTrackbarPos('Navigate', self.window_name, current_frame)
#
#             if key == ord('p'):
#                 self.is_playing = not self.is_playing
#
#             elif key == ord('s'):
#                 self.save_video(start_frame, end_frame)
#                 break
#
#             elif key == ord('g'):
#                 self.cap.release()
#                 cv2.destroyAllWindows()
#                 self.log(f"Returning start frame {start_frame} and end frame {end_frame}.")
#                 return start_frame, end_frame
#
#             elif key == ord('q'):
#                 break
#
#             if cv2.getTrackbarPos('Start', self.window_name) != start_frame:
#                 self.last_moved = 'Start'
#             elif cv2.getTrackbarPos('End', self.window_name) != end_frame:
#                 self.last_moved = 'End'
#             elif cv2.getTrackbarPos('Navigate', self.window_name) != current_frame:
#                 self.last_moved = 'Navigate'
#
#             # Adding a delay based on FPS of video to maintain original speed
#             if self.is_playing:
#                 time.sleep(1 / self.fps)
#
#         self.cap.release()
#         cv2.destroyAllWindows()
#
#     def save_video(self, start_frame, end_frame):
#         self.cap.release()
#         cv2.destroyAllWindows()
#
#         clip = VideoFileClip(self.input_video_path)
#         start_sec = start_frame / self.fps
#         end_sec = end_frame / self.fps
#         subclip = clip.subclip(start_sec, end_sec)
#         subclip.write_videofile(self.output_video_path)
#         self.log(f"Success to cut and save video - {self.output_video_path}")
#         self.cap = cv2.VideoCapture(self.input_video_path)
#
#     def get_video_name(self):
#         return os.path.basename(self.input_video_path).split('.')[0]
#
#
# # # Usage example
# # editor = VideoEditor('/home/nehoray/PycharmProjects/Shaback/data/videos/toem.mp4', 'output.mp4')
# # start, end = editor.process_video()
# # print(f"Start frame: {start}, End frame: {end}")
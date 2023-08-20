import cv2
import numpy
import numpy as np
import pyautogui
import torch


class ImageCropper:
    def __init__(self, source, window_name="image"):
        if str(source).endswith(".jpg") or str(source).endswith(".png"):
            self.image = cv2.imread(source)
        elif torch.is_tensor(source) or type(source) == numpy.ndarray:
            if len(source.shape) == 3:
                self.image = source
        else:
            raise RuntimeError("Failed to init ImageCropper")
        self.mouse_pts = []
        self.selected_roi = None
        self.window_name = window_name

    def get_coordinates_by_ROI_area(self):
        if len(self.mouse_pts) == 2:
            x1, y1 = self.mouse_pts[0]
            x2, y2 = self.mouse_pts[1]
            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        else:
            raise RuntimeError('Cannot return new crop image')

    def get_new_crop_image_by_ROI_area(self, new_image_tensor):
        if len(self.mouse_pts) == 2:
            x1, y1 = self.mouse_pts[0]
            x2, y2 = self.mouse_pts[1]
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

            return new_image_tensor[y1:y2, x1:x2]
        else:
            raise RuntimeError('Cannot return new crop image')

    def crop_image(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pts = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pts.append((x, y))
            cv2.rectangle(self.image, self.mouse_pts[0], self.mouse_pts[1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)

    def get_tensor_crop(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.crop_image)

        # Add text overlay on the image
        text = "Please select the ROI using the mouse, then press Enter"
        text_position = (10, 30)
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_font_scale = 0.8
        text_color = (0, 0, 255)
        text_thickness = 2
        cv2.putText(self.image, text, text_position, text_font, text_font_scale, text_color, text_thickness)

        cv2.imshow(self.window_name, self.image)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break

        cv2.destroyAllWindows()

        if len(self.mouse_pts) == 2:
            x1, y1 = self.mouse_pts[0]
            x2, y2 = self.mouse_pts[1]
            x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

            self.selected_roi = self.image[y1:y2, x1:x2]
            return self.selected_roi

        else:
            print("Error: Two points were not selected")


# Example usage
# image_path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/photos/netivey_israel/Screenshot from 2023-02-23 10-21-15.png"
# image_cropper = ImageCropper(image_path)
# cropped_image = image_cropper.get_tensor_crop()
# print(cropped_image)
# # #

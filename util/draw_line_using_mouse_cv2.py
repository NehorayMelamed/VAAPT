from typing import List, Tuple

import cv2


class DrawLineWidget(object):
    def __init__(self, original_image, image_name_window):
        self.image_name_window = str(image_name_window)
        self.original_image = original_image
        self.clone = self.original_image.copy()

        cv2.namedWindow(self.image_name_window)
        cv2.setMouseCallback(self.image_name_window, self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = self.start_end_points_lines_images_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))
            print('Starting: {}, Ending: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            self.start_end_points_lines_images_coordinates.append((self.image_coordinates[0], self.image_coordinates[1]))
            # Draw line
            cv2.line(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36, 255, 12), 4)
            cv2.imshow(self.image_name_window, self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()



    def get_lines_coordinates_list(self) -> List[Tuple[Tuple, Tuple]]:
        return self.start_end_points_lines_images_coordinates

    def show_image(self):
        return self.clone

# if __name__ == '__main__':
#     draw_line_widget = DrawLineWidget()
#     while True:
#         cv2.imshow('image', draw_line_widget.show_image())
#         key = cv2.waitKey(1)
#
#         # Close program with keyboard 'q'
#         if key == ord('q'):
#             cv2.destroyAllWindows()
#             exit(1)

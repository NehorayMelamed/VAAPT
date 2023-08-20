import cv2
import tkinter as tk
from tkinter import simpledialog

list_of_polygons = []


class DrawPolygon:
    def _init_(self, original_image, image_name_window):
        self.image_name_window = str(image_name_window)
        self.original_image = original_image
        self.clone = self.original_image.copy()
        self.ix = self.iy = self.sx = self.sy = -1
        self.__list_of_points = []
        self.list_of_polygons = []
        self.final_list_of_polygons = []
        self.list_of_polygons_numbers = []
        # self.list_of_polygons_attributes = []
        self.points = []

        cv2.namedWindow(self.image_name_window)
        cv2.setMouseCallback(self.image_name_window, self.draw_lines)

    # mouse callback function
    def draw_lines(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        if event == cv2.EVENT_LBUTTONDOWN:
            # draw circle of 2px
            print(x,y)
            cv2.circle(self.clone, (x, y), 3, (0, 0, 127), -1)
            self.points.append((x,y))
            self.__list_of_points.append((x, y))
            if self.ix != -1:  # if ix and iy are not first points, then draw a line
                cv2.line(self.clone, (self.ix, self.iy), (x, y), (0, 0, 127), 2, cv2.LINE_AA)
            else:  # if ix and iy are first points, store as starting points
                self.sx, self.sy = x, y
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_RBUTTONDOWN:
            print()
            # self.points.append((self.sx, self.sy))
            # cv2.line(self.clone, (x, y), (self.sx, self.sy), (0, 0, 127), 2, cv2.LINE_AA)
            self.ix, self.iy = -1, -1  # reset ix and iy
            if len(self.points) == 1:
                return
            # self.polygon_name, self.polygon_type = self.set_polygon_name_and_type()
            self.polygon_number = self.set_polygon_number_and_type()
            # print("Polygon name:", self.polygon_name)
            self.final_list_of_polygons.append(self.points)
            self.list_of_polygons_numbers.append(int(self.polygon_number))
            # self.list_of_polygons_type.append(self.polygon_type)

            self.points = []
            # if flags == 33:  # if alt key is pressed, create line between start and end points to create polygon


    def get_list_of_polygons(self):
        return self.final_list_of_polygons

    def get_list_of_polygons_numbers(self):
        return self.list_of_polygons_numbers

    def show_image(self):
        return self.clone
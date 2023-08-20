import cv2
import numpy as np

list_of_polygons = []


class DrawPolygon:
    def __init__(self, original_image, image_name_window):
        self.image_name_window = str(image_name_window)
        self.original_image = original_image
        self.clone = self.original_image.copy()
        self.ix = self.iy = self.sx = self.sy = -1
        self.__list_of_points = []
        self.list_of_polygons = []
        self.final_list_of_polygons = []
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
            self.ix, self.iy = -1, -1  # reset ix and iy
            if not self.points:
                return
            self.final_list_of_polygons.append(self.points)
            self.points = []
            if flags == 33:  # if alt key is pressed, create line between start and end points to create polygon
                cv2.line(self.clone, (x, y), (self.sx, self.sy), (0, 0, 127), 2, cv2.LINE_AA)

    def get_list_of_polygons(self):
        return self.final_list_of_polygons

    def show_image(self):
        return self.clone




# if __name__ == '__main__':
    # path = "/home/dudy/Nehoray/SHABACK_POC_NEW/data/photos/netivey_israel/Screenshot from 2023-02-23 10-21-00.png"
    # first_frame = cv.imread(path)
    # draw_line_widget = DrawPolygon(original_image=first_frame, image_name_window=path)
    # while True:
    #     cv2.imshow(path, draw_line_widget.show_image())
    #     # Close program with keyboard 'q'
    #     key = cv2.waitKey(1)
    #     # save results and continue by enter
    #     if key == 13:
    #         print("sdfsdfds")
    #         # print(draw_line_widget.get_list_of_polygons())
    #         list_of_polygons_traffic_coordinates = draw_line_widget.get_list_of_polygons()
    #         print(list_of_polygons_traffic_coordinates)
    #         break
    #         # cv.destroyAllWindows()
    #
    # print("eefw")
    #
    # for polygon in list_of_polygons_traffic_coordinates:
    #     i = 0
    #     while i < len(polygon) -1:
    #        cv2.line(first_frame, polygon[i], polygon[i+1], (36, 255, 12), 2)
    #        i += 1
    #
    # cv2.imshow(path, first_frame)
    #         # Close program with keyboard 'q'
    # cv2.waitKey(0)


 # i = 0
 #        while i < len(polygon) -1:
 #            cv2.line(first_frame, polygon[i], polygon[i+1], (36, 255, 12), 2)
 #            i += 1

# read image from path and add callback
# img = cv.resize(
#     cv.imread("/home/dudy/Nehoray/SHABACK_POC_NEW/data/photos/netivey_israel/Screenshot from 2023-02-23 10-21-00.png"),
#     (1280, 720))
# cv.namedWindow('image')
# cv.setMouseCallback('image', draw_lines)
#
# while (1):
#     cv.imshow('image', img)
#     if cv.waitKey(20) & 0xFF == 27:
#         break
#
# cv.destroyAllWindows()
#
# out = []
# up_to_4 = 0
#
# for i in range(len(list_of_polygons)):
#     if up_to_4 == 4:
#         up_to_4 = 0
#         continue
#     else:
#         up_to_4 += 1
#         out.append(list_of_polygons[i])
#
# out = [[out[i:i + 4] for i in range(0, len(out), 4)]]
#
# print("out", out)

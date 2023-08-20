import cv2
from RapidBase.import_all import *

class BoundingBoxWidget(object):
    def __init__(self, input_image):
        self.original_image = input_image
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            # self.image_coordinates = [(x,y)]
            self.current_image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            # self.image_coordinates.append((x,y))
            self.current_image_coordinates.append((x,y))

            ### Update Image Coordinates: ###
            point_1 = self.current_image_coordinates[0]
            point_2 = self.current_image_coordinates[1]
            point_1_x, point_1_y = point_1
            point_2_x, point_2_y = point_2
            start_x = min(point_1_x, point_2_x)
            stop_x = max(point_1_x, point_2_x)
            start_y = min(point_1_y, point_2_y)
            stop_y = max(point_1_y, point_2_y)

            new_point_1 = (start_x, start_y)
            new_point_2 = (stop_x, start_y)
            new_point_3 = (stop_x, stop_y)
            new_point_4 = (start_x, stop_y)

            # self.image_coordinates.append(self.current_image_coordinates)
            self.image_coordinates.append([(new_point_1, new_point_2, new_point_3, new_point_4)])


            print('top left: {}, bottom right: {}'.format(self.current_image_coordinates[0], self.current_image_coordinates[1]))
            print('x,y,w,h : ({}, {}, {}, {})'.format(self.current_image_coordinates[0][0],
                                                      self.current_image_coordinates[0][1],
                                                      self.current_image_coordinates[1][0] - self.current_image_coordinates[0][0],
                                                      self.current_image_coordinates[1][1] - self.current_image_coordinates[0][1]))

            # Draw rectangle
            cv2.rectangle(self.clone, self.current_image_coordinates[0], self.current_image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

if __name__ == '__main__':
    ### Get Input Image: ###
    # filepath = r'E:\Quickshot\06_07_2022_FLIR\converted_bin\large_bin_files_with_avi_movies\8-900m_matrice300_night_flir_640x512_800fps_26000frames_converted\Results\Sequences\seq0/Mov_BG.npy'
    # filepath = r'E:\Quickshot\13.07.22_bin\one_big_bin_file\22-20000frames500fps950m50mmlensLeftRight__CONVERTED\Results/Mov_BG.npy'
    # filepath = r'E:\Quickshot\13.07.22_bin\one_big_bin_file\18-20000frames500fps950m100mmlensLeftRight__CONVERTED\Results/Mov_BG.npy'
    # filepath = r'E:\Quickshot\13.07.22_bin\one_big_bin_file\10-20000frames500fps750m50mmlensRightLeft__CONVERTED\Results/Mov_BG.npy'
    # filepath = r'E:\Quickshot\13.07.22_bin\one_big_bin_file\2-20000frames500fps525m100mmlensRightLeft__CONVERTED\Results/Mov_BG.npy'
    filepath = r'E:\Quickshot\19_7_22_flir_exp_Bin\one_big_bin_file\8-1000m_2matrice300_1martice600_1mavic2_640x_converted\Results/Mov_BG.npy'
    input_image = np.load(filepath, allow_pickle=True)
    input_image = scale_array_stretch_hist(input_image)

    ### Get Folder To Save Result At: ###
    folder_to_save_at = r'E:\Quickshot\19_7_22_flir_exp_Bin\one_big_bin_file\8-1000m_2matrice300_1martice600_1mavic2_640x_converted\Results'
    filename = 'BB_regions_to_disregard.npy'
    full_filename = os.path.join(folder_to_save_at, filename)

    ### Take Bounding-Boxes: ###
    boundingbox_widget = BoundingBoxWidget(input_image)

    # cv2.imshow('image', boundingbox_widget.show_image())
    # key = cv2.waitKey(1)
    #
    # # Close program with keyboard 'q'
    # if key == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit(1)

    1

    while True:
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            print(boundingbox_widget.image_coordinates)
            np.save(full_filename, boundingbox_widget.image_coordinates, allow_pickle=True)
            cv2.destroyAllWindows()
            exit(1)



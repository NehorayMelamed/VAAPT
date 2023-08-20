from RapidBase.import_all import *

import cv2


class BoundingBoxWidgetVideo(object):
    """
    This class is an interactive bounding boxes generator for videos.

    Instructions:
    The object receives a video path.
    when calling generate_all_bounding_boxes, each batch is presented as a video. The video will freeze after
    batch_size frames. Then mark a bounding box with the mouse. Right click to cancel a drawing. If multiple boxes
    are drawn, only the latest is saved.
    If no bounding box drawn or all drawn boxes are canceled, an empty box is saved.
    When you are satisfied with your box, press space key to continue to the next batch. press 'q' key to quit

    Returned output is two lists:
        flag_bbox_found: list that determines for each batch whether it contains a box
        per_frame_bbox_list: a list containing batch_size bounding boxes for every batch (all identical)
    """
    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self.bbox_coordinates = []
        self.batch_bboxes_list = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        """An event triggered function for bbox creation

        :param event: triggering mouse event
        :param x: mouse x coordinate in image
        :param y: mouse y coordinate in image
        :param flags:
        :param parameters:
        :return:
        """
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bbox_coordinates = [(x, y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            # Complete bbox coordinates
            x0, y0 = self.bbox_coordinates[0]
            self.bbox_coordinates += [(x, y0), (x, y), (x0, y)]

            # Draw rectangle
            cv2.rectangle(self.clone, self.bbox_coordinates[0], self.bbox_coordinates[2], (36, 255, 12), 2)
            cv2.imshow("video batch", self.clone)

            # Add bbox to bboxes list
            self.batch_bboxes_list.append(self.bbox_coordinates)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.bbox_coordinates = []
            self.batch_bboxes_list = []
            self.clone = self.original_image.copy()
            cv2.imshow("video batch", self.clone)

    def get_batch_bounding_box(self, frames_batch: np.array,
                               max_bboxes_in_batch: int = 1,
                               stride: int = 5,
                               fps: int = 10) -> Tuple[List, str]:
        """Allows the user to mark bounding box over a video batch and returns it

        :param frames_batch: frames to show
        :param max_bboxes_in_batch: max number of bounding boxes to mark
        :param stride: frames stride when presenting the video
        :param fps: fps when presenting the video
        :return: batches bounding box + requested next step
        """
        batch_size = len(frames_batch)
        window_name = 'video batch'

        for frame_idx in range(0, batch_size, stride):
            cv2.imshow(window_name, frames_batch[frame_idx])
            cv2.waitKey(int(1000 / int(fps)))
        print('Done Showing Sequence')

        self.image_coordinates = []
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.extract_coordinates)

        self.original_image = frames_batch[-1].copy()
        self.clone = self.original_image.copy()
        self.batch_bboxes_list = []
        cv2.imshow(window_name, self.original_image)

        # Wait for a key to determine next steps:
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                return self.batch_bboxes_list[:max_bboxes_in_batch], 'quit'
            elif key == 32:  # space
                cv2.destroyAllWindows()
                return self.batch_bboxes_list[:max_bboxes_in_batch], 'move to next'

    def generate_all_bounding_boxes(self, num_bboxes_in_batch: int = 1,
                                    batch_size: int = 500,
                                    video_stride: int = 5,
                                    video_fps: int = 10):
        """Generates bboxes for all video in batches

        :param batch_size: batch size
        :param num_bboxes_in_batch: number of returned bboxes (if not enough to return, padded with empty boxes)
        :param video_stride: frames stride when presenting the video
        :param video_fps: fps when presenting the video
        :return: flag_bbox_found: list that determines for each batch whether it contains a box
                 per_frame_bbox_list: a list containing batch_size bounding boxes for every batch (all identical)
        """
        cap = cv2.VideoCapture(self.video_path)
        frames_batch = []
        batch_idx = 0
        bbox_list = []

        frame = 0
        while True:
            # read image
            success, img = cap.read()
            if not success:
                break
            frame += 1
            frames_batch.append(img)
            # for every batch get a bounding box
            if frame % batch_size == 0:
                batch_idx += 1
                frames_batch = frames_batch
                # get bboxes and next action
                batch_bboxes, user_feedback = self.get_batch_bounding_box(frames_batch, video_stride, video_fps)
                # pad bboxes list to wanted size
                batch_bboxes += [[]] * (num_bboxes_in_batch - len(batch_bboxes))
                bbox_list += [batch_bboxes]
                # determine next action
                if user_feedback == 'quit':
                    break
                elif user_feedback == 'move to next':
                    frames_batch = []
                    frame = 0

        cap.release()

        return bbox_list

    def show_image(self):
        return self.clone


def generate_and_save_flashlight_bboxes(load_video_path: str,
                                        save_has_flashlight_path: str = 'flag_flashlight_found_list.npy',
                                        save_bboxes_path: str = 'flashlight_BB_list.npy',
                                        max_bboxes_in_batch: int = 1,
                                        batch_size: int = 500,
                                        video_show_stride: int = 50,
                                        video_show_fps: int = 50):
    boundingbox_widget = BoundingBoxWidgetVideo(load_video_path)
    batch_bbox_list = boundingbox_widget.generate_all_bounding_boxes(num_bboxes_in_batch=max_bboxes_in_batch,
                                                                     batch_size=batch_size,
                                                                     video_stride=video_show_stride,
                                                                     video_fps=video_show_fps)

    # return a list indicating whether there is a bounding box and a list of bounding boxes for each batch
    flag_bbox_found = np.array([any(bbox) for bbox in batch_bbox_list])
    batch_bbox_list = np.array(batch_bbox_list, dtype=object)

    np.save(save_has_flashlight_path, flag_bbox_found)
    np.save(save_bboxes_path, batch_bbox_list)

    return flag_bbox_found, batch_bbox_list


if __name__ == '__main__':
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    super_folder = r'E:\Quickshot\12.4.2022 - natznatz experiments'
    avi_filenames = get_filenames_from_folder_string_pattern(super_folder, flag_recursive=True, string_pattern_to_search='*.avi')
    avi_filenames_list = [
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\10_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\11_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\12_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\14_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\15_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\19_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\1_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\20_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\21_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\22_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\23_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\24_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\25_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\26_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\27_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\28_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\29_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\30_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\31_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\32_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\33_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\34_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\35_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\36_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\38_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\3_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\43_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\44_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\45_day_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                         # r'E:\\Quickshot\\12.4.2022 - natznatz experiments\\6_night_500fps_20000frames_640x320\\Results\\Original_Movie.avi',
                          ]


    ### Use List Of Files: ###
    # video_path = r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments\1_640_512_800fps_10000frames_1000_meter_flir/Original_Movie.avi'
    video_path = avi_filenames_list[0]
    folder, filename, filename_without_extension, filename_extension = path_get_folder_filename_and_extension(video_path)
    flashlight_path = os.path.join(folder, 'Flashlight')
    path_create_path_if_none_exists(flashlight_path)
    has_flashlight_path = os.path.join(folder, 'Flashlight', 'flag_flashlight_found_list.npy')
    bboxes_path = os.path.join(folder, 'Flashlight', 'flashlight_BB_list.npy')

    ### Use Predermined Path: ###
    # has_flashlight_path = r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments\1_640_512_800fps_10000frames_1000_meter_flir/Results/Flashlight/flag_flashlight_found_list.npy'
    # bboxes_path = r'C:\Users\dudyk\Desktop\dudy_karl\QS_experiments\1_640_512_800fps_10000frames_1000_meter_flir/Results/Flashlight/flashlight_BB_list.npy'
    flags, bboxes = generate_and_save_flashlight_bboxes(video_path, has_flashlight_path, bboxes_path,
                                                        max_bboxes_in_batch=4)

    ### Look At Resulst: ###
    print(np.load(bboxes_path, allow_pickle=True))
    print(flags)
    print(bboxes)

    ### Correct For Flags Stuff: ###
    for flag_index in np.arange(len(flags)):
        current_flag = flags[flag_index]
        if current_flag != False:
            flags[flag_index] = True
    np.save(has_flashlight_path, flags, allow_pickle=True)

    ### Load Arrays For Checking: ###
    flashlight_BB_list = np.load(bboxes_path, allow_pickle=True)
    flag_flashlight_found_list = np.load(has_flashlight_path, allow_pickle=True)
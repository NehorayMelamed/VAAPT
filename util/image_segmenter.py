import cv2
import numpy as np
# from RapidBase.import_all import *
class ImageSegmenter:
    def __init__(self, img_path):
        # Read the original image
        self.original_img = cv2.imread(img_path)

        # Initialize the state and points list
        self.points = []

    def select_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add the point to the list
            self.points.append((x, y))
            # Display the points
            cv2.circle(self.img, (x, y), 3, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(self.img, self.points[-2], self.points[-1], (0, 255, 0), 1)

    def get_mask(self):
        while True:
            # Start with a fresh copy of the image for each loop
            self.img = self.original_img.copy()
            self.mask = np.zeros_like(self.img)

            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.select_roi)

            while True:
                img_with_instructions = self.img.copy()
                cv2.putText(img_with_instructions, "Press Enter when finished selecting ROI",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('image', img_with_instructions)
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    break

            if len(self.points) > 1:
                # Draw the polygon on the mask
                cv2.fillPoly(self.mask, [np.array(self.points)], (255, 255, 255))
                # Bitwise AND operation to black out regions of the image outside the mask
                result = cv2.bitwise_and(self.img, self.mask)

                while True:
                    result_with_instructions = result.copy()
                    cv2.putText(result_with_instructions, "Press Enter to confirm, ESC to select new ROI",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('result', result_with_instructions)
                    key = cv2.waitKey(0)
                    if key == 13:  # Enter key
                        cv2.destroyAllWindows()
                        mask_to_return = self.mask[:, :, 0]
                        return mask_to_return, result
                    elif key == 27:  # ESC key
                        # Clear points and restart the loop
                        self.points = []
                        break

# image_path = "/home/dudy/Nehoray/segment_anything_base_dir/Grounded-Segment-Anything/outputs/raw_image.jpg"
# img_segmenter = ImageSegmenter(image_path)
# mask, result = img_segmenter.get_mask()
#
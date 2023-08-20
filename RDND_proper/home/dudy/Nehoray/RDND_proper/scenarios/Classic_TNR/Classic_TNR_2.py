import cv2
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import atexit

## DEPENDENCIES
# python3
# opencv3, numpy
# pip3 install matplotlib
# sudo apt install python3-tk

INPUT_MODE = 1  # 0 - webcam, 1 - video, 2 - image sequence
ALIGN = True  # if False, just average
PREDICTION = True
DEBUG = 1
RESIZE = 1.0  # .0 0.5 gives half of every dim, and so on...
BORDER = 0.1

IN_DIR = "./input/"
DARK_IN_DIR = "./input_darkframe/"
DARKFRAME_COEFF = 1.0
OUT_DIR = "./output/"

stabilized_average, divider_mask, imagenames, outname = None, None, None, None


## Returns 0.0 ... 65535.0 darkframe loaded from 16bpp image
def get_darkframe():
    mypath = DARK_IN_DIR

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]

    if len(images_paths) == 0:
        return None

    darkframe = cv2.imread(mypath + images_paths[0])

    print("get_darkframe(): loaded darkframe.")

    return np.float32(darkframe) / 255 * DARKFRAME_COEFF


def get_frame_from_main_camera():
    cap = cv2.VideoCapture(0)

    counter = 0

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        yield np.float32(frame) / 255, "webcam_{}".format(counter)
        counter += 1


def get_next_frame_from_video():
    cap = cv2.VideoCapture(0)

    mypath = IN_DIR
    outpath = OUT_DIR

    videos_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    videos_paths.sort()

    counter = 0
    for vidpath in videos_paths:
        cap = cv2.VideoCapture(mypath + vidpath)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            yield np.float32(frame) / 255, "{}_{}".format(vidpath, counter)
            counter += 1


def get_next_frame_from_file():
    mypath = IN_DIR
    outpath = OUT_DIR

    images_paths = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f[0] is not "."]
    images_paths.sort()

    counter = 0
    for imgpath in images_paths:
        yield np.float32(cv2.imread(mypath + imgpath)), imgpath
        counter += 1


def get_next_frame_from_chosen_source(_src=1):
    if _src == 0:
        return get_frame_from_main_camera()
    if _src == 1:
        return get_next_frame_from_video()
    if _src == 2:
        return get_next_frame_from_file()


def compute_homography(_input_image, _base, pxdistcoeff=0.9, dscdistcoeff=0.9, _cache=None):
    k_cnt = "counter"
    k_kp2 = "kp2"
    k_des2 = "des2"
    if _cache is not None:
        if k_cnt in _cache:
            _cache[k_cnt] += 1
        else:
            _cache[k_cnt] = 0

    # Initiate ORB detector
    # alg = cv2.ORB_create()
    alg = cv2.AKAZE_create()

    # find the keypoints and descriptors with orb
    kp1 = alg.detect(_input_image, None)
    kp1, des1 = alg.compute(_input_image, kp1)

    kp2, des2 = None, None
    if _cache is not None:
        if _cache[k_cnt] < 10 or _cache[k_cnt] % 10 == 0:  #if we're before the 10th frame and every 10th frame, detect new keypoints
            kp2 = alg.detect(_base, None)
            kp2, des2 = alg.compute(_base, kp2)
            _cache[k_kp2] = kp2
            _cache[k_des2] = des2
            print("    > found and cached {} kp2".format(len(kp2)))
        else:
            kp2, des2 = _cache[k_kp2], _cache[k_des2]  #in between every 10 frames use already detected keypoints
            print("    > reused {} kp2".format(len(kp2)))
    else:
        kp2 = alg.detect(_base, None)
        kp2, des2 = alg.compute(_base, kp2)
        print("    > found {} kp2".format(len(kp2)))

    print("    > found {} kp1".format(len(kp1)))

    # initialize matcher
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # matching
    matches = bfm.match(des1, des2)
    print("    > found {} matches".format(len(matches)))

    # filtering matches
    avg_dist = 0
    for match in matches:
        input_image_idx = match.queryIdx
        img2_idx = match.trainIdx

        im1_x, im1_y = np.int32(kp1[input_image_idx].pt)
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)

        a = np.array((im1_x, im1_y))
        b = np.array((im2_x, im2_y))

        avg_dist += np.linalg.norm(a - b)
    avg_dist /= len(matches)

    dist_diffs = []
    for match in matches:
        input_image_idx = match.queryIdx
        img2_idx = match.trainIdx

        im1_x, im1_y = np.int32(kp1[input_image_idx].pt)
        im2_x, im2_y = np.int32(kp2[img2_idx].pt)

        a = np.array((im1_x, im1_y))
        b = np.array((im2_x, im2_y))

        dist = np.linalg.norm(a - b)
        dist_diff = np.linalg.norm(avg_dist - dist)
        dist_diffs.append(dist_diff)

    # sort img1,img2 keypoint "normalized distances" / according to how different their distance is from the average keypoint distance
    sorted_y_idx_list = sorted(range(len(dist_diffs)), key=lambda x: dist_diffs[x])
    good = [matches[i] for i in sorted_y_idx_list][0:int(len(matches) * pxdistcoeff)]

    good = sorted(good, key=lambda x: x.distance)
    good = good[0:int(len(good) * dscdistcoeff)]

    print("    > after filtration {} matches left\n".format(len(good)))

    # getting source & destination points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if DEBUG >= 2:
        matchesimg = cv2.drawMatches(_input_image, kp1, _base, kp2, good, flags=4, outImg=None)
        cv2.imshow("matchesimg", matchesimg)
        cv2.waitKey(1)

    # finding homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return H


def transform_image_to_base_image(_input_image, _H):
    h, w, b = _input_image.shape

    dst = cv2.warpPerspective(_input_image, _H, (w, h))

    return dst


def make_mask_for_image(_img, _border_coeff=0):
    return make_border_for_image(np.ones_like(_img), _border_coeff)


def make_border_for_image(_img, _border_coeff=0):
    h, w, b = _img.shape
    hborder = int(h * _border_coeff)
    wborder = int(w * _border_coeff)
    map1 = cv2.copyMakeBorder(_img, hborder, hborder, wborder, wborder, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return map1


def exit_handler():
    global stabilized_average, divider_mask, imagenames, outname

    stabilized_average /= divider_mask  # dividing sum by number of images

    outname = OUT_DIR + imagenames[0] + '-' + imagenames[-1] + '.png'

    print("Saving file " + outname)

    stabilized_average[stabilized_average < 0] = 0
    stabilized_average[stabilized_average > 1] = 1

    cv2.imwrite(outname, np.uint16(stabilized_average * 65535))

    cv2.waitKey();


def main():
    global stabilized_average, divider_mask, imagenames, outname

    atexit.register(exit_handler)

    shape_original = None
    shape_no_border = None
    shape = None

    stabilized_average = None
    darkframe_image_no_border = None  # get_darkframe()
    # darkframe_image = None
    mask_image_no_border = None
    mask_image = None
    divider_mask_no_border = None
    divider_mask = None
    transformed_image = None
    transformed_mask = None

    H_state_x_old = None
    H_state_v_old = None
    H_state_a_old = None

    H_pred = None

    keypoint_cache = dict()

    counter = 0
    imagenames = []
    for input_image_, imgname in get_next_frame_from_chosen_source(INPUT_MODE):
        input_image_no_border = None
        input_image = None

        try:
            shape_original = input_image_.shape
        except Exception as e:
            print("Nope")
            if counter > 0:
                return
            continue

        if darkframe_image_no_border is None:
            darkframe_image_no_border = get_darkframe()
            darkframe_image_no_border = cv2.resize(darkframe_image_no_border, (shape_original[1], shape_original[0]),
                                                   interpolation=cv2.INTER_LINEAR)

        if darkframe_image_no_border is not None:
            input_image_no_border = input_image_ - darkframe_image_no_border

        if shape is None:
            input_image_no_border = cv2.resize(input_image_no_border, (0, 0), fx=RESIZE, fy=RESIZE,
                                               interpolation=cv2.INTER_LINEAR)
        else:
            input_image_no_border = cv2.resize(input_image_no_border, (shape_no_border[1], shape_no_border[0]),
                                               interpolation=cv2.INTER_LINEAR)

        input_image = make_border_for_image(input_image_no_border, BORDER)

        shape_no_border = input_image_no_border.shape
        shape = input_image.shape

        if stabilized_average is None:
            stabilized_average = np.float32(input_image)

        if mask_image is None:
            mask_image_no_border = make_mask_for_image(input_image_no_border)
            mask_image = make_border_for_image(mask_image_no_border, BORDER)

        if divider_mask is None:
            divider_mask = make_mask_for_image(mask_image_no_border, BORDER)
            divider_mask[:] = 1

        imagenames.append(imgname)
        imgpath = IN_DIR + imgname

        input_image = cv2.resize(input_image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        transformed_image = input_image #initialize transformed image with input image
        H = None

        if ALIGN:
            base = (stabilized_average / divider_mask)
            base[base <= 0] = 0
            base[base >= 1] = 1
            H = compute_homography(input_image, np.array(base, dtype=input_image.dtype), pxdistcoeff=0.9,
                                   dscdistcoeff=0.9, _cache=keypoint_cache)

            if H_pred is not None:
                transformed_image_nopred = transform_image_to_base_image(transformed_image, H) #transformed according to "currently found Homography"
                transformed_image_pred = transform_image_to_base_image(transformed_image, H_pred)  #transformed according to "model homography" / "predicted homography"
                cv2.imshow("transformed_image_nopred", transformed_image_nopred)
                cv2.imshow("transformed_image_pred", transformed_image_pred)
                cv2.waitKey(1)
                # New, final Homography is the average between found and predicted homography: ###
                H = (H + H_pred) / 2

            ### Use Homography to get transformed image and mask: ###
            transformed_image = transform_image_to_base_image(transformed_image, H)
            transformed_mask = transform_image_to_base_image(mask_image, H)

            if PREDICTION and (INPUT_MODE == 1 or INPUT_MODE == 0):
                H_state_x_current = H
                H_state_v_current = 0
                H_state_a_current = 0
                k = 0.5  # denoising

                if H_state_x_old is not None:
                    H_state_v_current = (H_state_x_current - H_state_x_old) * k + H_state_v_old * (1 - k)

                if H_state_v_old is not None:
                    H_state_a_current = (H_state_v_current - H_state_v_old) * k + H_state_a_old * (1 - k)
                    H_pred = H_state_x_current + H_state_v_current + H_state_a_current / 2

                H_state_x_old = H_state_x_current
                H_state_v_old = H_state_v_current
                H_state_a_old = H_state_a_current

        stabilized_average += np.float32(transformed_image)
        divider_mask += transformed_mask

        if DEBUG >= 1:
            cv2.imshow("stabilized_average/256/divider_mask", stabilized_average / divider_mask);
            cv2.imshow("transformed_image", transformed_image);
            cv2.imshow("divider_mask", divider_mask / (counter + 1));
            cv2.waitKey(1);

        if DEBUG >= 3:
            cv2.imwrite(OUT_DIR + imgname + '_DEBUG.jpg', transformed_image)
            cv2.imwrite(OUT_DIR + imgname + '_divider_maskDEBUG.png', np.uint16(divider_mask * 255 / (counter + 1)))
        counter += 1


# exit_handler()

# main
if __name__ == '__main__':
    main()

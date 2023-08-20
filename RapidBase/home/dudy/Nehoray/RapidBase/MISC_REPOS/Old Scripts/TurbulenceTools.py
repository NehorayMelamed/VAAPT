import numpy as np
import cv2

class Tools:
    def turbFilter(I):

        ### TODO: why the .copy() ??? ###
        I = I.copy()


        ### Parameters: ###
        h = 100
        Cn2 = 7e-14
        # Cn2 = 7e-17
        wvl = 5.3e-7
        IFOV = 4e-7
        R = 1000
        VarTiltX = 3.34e-6
        VarTiltY = 3.21e-6
        k = 2 * np.pi / wvl
        r0 = (0.423 * k ** 2 * Cn2 * h) ** (-3 / 5)
        PixSize = IFOV * R
        PatchSize = 2 * r0 / PixSize

        ### Get Current Image Shape And Appropriate Meshgrid: ###
        PatchNumRow = int(np.round(I.shape[0] / PatchSize))
        PatchNumCol = int(np.round(I.shape[1] / PatchSize))
        shape = I.shape
        [X0, Y0] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        if I.dtype == 'uint8':
            mv = 255
        else:
            mv = 1


        ### Get Random Motion Field: ###
        ShiftMatX0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltX * R / PixSize)
        ShiftMatY0 = np.random.randn(PatchNumRow, PatchNumCol) * (VarTiltY * R / PixSize)

        ### Resize (small) Random Motion Field To Image Size: ###
        ShiftMatX = cv2.resize(ShiftMatX0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        ShiftMatY = cv2.resize(ShiftMatY0, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)

        ### Add Rescaled Flow Field To Meshgrid: ###
        X = X0 + ShiftMatX
        Y = Y0 + ShiftMatY

        ### Resample According To Motion Field: ###
        I = cv2.remap(I, X.astype('float32'), Y.astype('float32'), interpolation=cv2.INTER_CUBIC)


        ### Clip Result: ###
        I = np.minimum(I, mv)
        I = np.maximum(I, 0)

        return I

    def segmentImage(I1, I2, minSpeed = 2):
        I1 = Tools.convertToGray(I1)
        I2 = Tools.convertToGray(I2)
        flow = cv2.calcOpticalFlowFarneback(I1, I2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        velocity = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        moving = velocity > minSpeed

        Y = cv2.cvtColor(I1, cv2.COLOR_GRAY2BGR)
        Y = cv2.cvtColor(Y, cv2.COLOR_BGR2YCrCb)
        Y[:, :, 1] = moving * 255
        Y[:, :, 2] =  np.logical_not(moving) * 255
        Y = cv2.cvtColor(Y, cv2.COLOR_YCrCb2BGR)
        return Y,moving

    def convertToGray(I):
        if len(I.shape) == 3:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        return I

    # def getSupportedDevice():
    #     from tensorflow.python.client import device_lib
    #     return device_lib.list_local_devices()


# import ESRGAN_utils
# from ESRGAN_utils import *
# I = read_image_default()
# tic()
# Y = Tools.turbFilter(I)
# toc()
#
# plt.figure()
# plt.imshow(I, cmap='gray')
# plt.figure()
# plt.imshow(Y, cmap='gray')
# plt.show()

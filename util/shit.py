import time

import cv2

img = cv2.imread("/home/dudy/Nehoray/SHABACK_POC_NEW/util/output_directory/Jun07_Shaback/toem/frame_0010.png")
cv2.imshow('window_name', img)

time.sleep(1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import cv2
import numpy as np
# frame_1 = "/home/avraham/mafat/fay/dog_cat_data/croc.jpg"
# # frame_1 = "/home/caiman/backupped/croc.jpg"
# frame_2 = "/home/avraham/Downloads/caiman.jpg"
# print( cv2.imread(frame_1).shape)
# shape = (620, 414)
# a = cv2.imread(frame_1)#[:shape[0], :shape[1],:]
# # b = cv2.imread(frame_2)[100:300, 100:300, 0]
# print(a.shape)
# # print(b.shape)
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# vw = cv2.VideoWriter("croc.avi", fourcc, float(2), shape)
# for i in range(50):
#     # frame = cv2.cvtColor(a, cv2.CO)
#     vw.write(a)
#     # vw.write(b)
# vw.release()
from RapidBase.Utils.Tensor_Manipulation.Array_Tensor_Manipulation import BW2RGB

fr = np.zeros((3, 512, 640))
fr=BW2RGB(fr).transpose((1,2,0))
print(fr.shape)
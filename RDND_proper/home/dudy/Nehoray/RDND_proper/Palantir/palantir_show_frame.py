import matplotlib.pyplot as plt
import torch, os, cv2
import numpy as np
frame_dir = "/home/avraham/mafat/QS/PalantirResults/14/frames"
frames = []
for i in range(39):
    current_name = os.path.join(frame_dir, str(i).rjust(6, '0')+".pt")
    fr = torch.load(current_name).squeeze().numpy()
    fr = fr/fr.max().clip(0, 255)
    fr = fr * 255
    frames.append(fr.astype(np.uint8))
for i in range(39):
    cv2.imshow("frames", frames[i])
    cv2.waitKey(200)

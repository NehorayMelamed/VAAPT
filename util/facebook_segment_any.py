import cv2
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry["vit_h"](checkpoint="/home/nehoray/Downloads/sam_vit_h_4b8939.pth")

device = "cuda"

sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

image_path = "/home/nehoray/PycharmProjects/Shaback/yolo_tracking/examples/runs/track/exp2/crops/2/1/toem2.jpg"
image = plt.imread(image_path)
masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()
# mask_generator = SamAutomaticMaskGenerator(sam)
# sam = sam_model_registry["vit_h"](checkpoint="/home/nehoray/Downloads/sam_vit_h_4b8939.pth")
# predictor = SamPredictor(sam)
# img = cv2.imread(image_path)
# predictor.set_image(img)

# masks, _, _ = predictor.predict()
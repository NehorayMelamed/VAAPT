import torch
from torch import Tensor


def manhattan_distance_from_zero(H: int, W: int) -> Tensor:
    width_range = torch.arange(W)
    width_extended = width_range.repeat(H, 1)
    height_range = torch.arange(H)
    height_extended = height_range.repeat(W, 1).T
    ret= width_extended+height_extended
    return ret


def tshow(matrix, viewer='plt'):
    img = matrix.clip(0,255).permute((1,2,0)).to(torch.uint8).numpy()
    if viewer == 'plt':
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()
    else:
        import cv2
        cv2.imshow("croc", img)
        cv2.waitKey(0)


def tshowm(matrices, viewer='plt'):
    matrices=torch.cat(matrices, dim=-1)
    tshow(matrices, viewer)

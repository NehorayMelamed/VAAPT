import math

import cv2
import torch
from torch import Tensor

from transforms import rotate_matrix, blur_rotate_matrix


def shift_croc():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_back = rotate_matrix(croc_input, [0.2], warp_method='fft').permute(1, 2, 0)
    both: Tensor = torch.zeros((H,W*2,C), dtype=torch.uint8)

    both[:, :W, :] = Tensor(croc).to(torch.uint8)
    #both[:, W:, :] = CenterCrop((H,W))()
    croc_back = croc_back.to(torch.uint8)
    cv2.imshow("croc", croc_back.numpy())
    cv2.waitKey(0)
    #cv2.imshow('croc', croc_back.to(torch.uint8).numpy().reshape(H, W, C))


def blur_shift_croc():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    # croc_input = torch.stack([croc_input, croc_input, croc_input])
    croc_back = blur_rotate_matrix(croc_input, Tensor([5*math.pi/180, 0.2, 0.4])[0],  N=10, warp_method='bicubic').permute(1, 2, 0)
    both: Tensor = torch.zeros((H, W * 2, C), dtype=torch.uint8)

    both[:, :W, :] = Tensor(croc).to(torch.uint8)
    # both[:, W:, :] = CenterCrop((H,W))()
    croc_back = croc_back.to(torch.uint8)
    cv2.imshow("croc", croc_back.numpy())
    cv2.waitKey(0)


if __name__ == '__main__':
    blur_shift_croc()

import cv2
import torch
from torch import Tensor

from transforms import blur_shift_matrix


def shift_croc():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_back = blur_shift_matrix(croc_input, Tensor([200]), 100, 50, warp_method='fft').permute(1, 2, 0)
    both: Tensor = torch.zeros((H,W*2,C), dtype=torch.uint8)

    both[:, :W, :] = Tensor(croc).to(torch.uint8)
    both[:, W:, :] = croc_back.to(torch.uint8)
    cv2.imshow("croc", both.numpy())
    cv2.waitKey(0)
    #cv2.imshow('croc', croc_back.to(torch.uint8).numpy().reshape(H, W, C))


if __name__ == '__main__':
    shift_croc()

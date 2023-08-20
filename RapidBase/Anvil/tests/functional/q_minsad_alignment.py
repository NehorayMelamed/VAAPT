

db=False
if not db:
    import cv2

from transforms import affine_warp_matrix
from alignments import min_SAD_detect_shifts

import torch
from torch import Tensor


def crop(matrix: Tensor, target_H: int, target_W: int) -> Tensor:
    C,H,W = matrix.shape
    excess_rows = H - target_H
    excess_columns = W - target_W
    start_y = int(excess_rows/2)
    start_x = int(excess_columns/2)
    stop_y = start_y+target_H
    stop_x = start_x+target_W
    return matrix[:, start_y:stop_y, start_x:stop_x]


def shift_then_unshift_croc():
    croc_input = torch.load("data/croc.pt")
    # H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    # croc_input = Tensor(croc).permute((2, 0, 1))
    C,H,W = croc_input.shape
    shifts_x = [2,4,6,8,10]
    shifts_y = [0,30,60,90,120]
    thetas = [-0.01, 0.0, 0.01, 0.02, 0.03]
    scales = [0.80, 0.85,0.90, 0.95, 1.00]
    sy = shifts_y[3]
    sx = shifts_x[2]
    r = thetas[1]
    sc = scales[1]
    shifted_croc = affine_warp_matrix(croc_input, -sy, -sx, r, warp_method='bicubic')

    shifted_croc = crop(shifted_croc, H, W)

    shifts = min_SAD_detect_shifts(shifted_croc, croc_input, shifts_y, shifts_x, thetas, warp_method='bicubic')
    print(shifts[:-1])
    croc_back = affine_warp_matrix(shifted_croc, shifts[0], shifts[1], shifts[2], shifts[3], warp_method='bicubic')
    frame = torch.zeros((C, H, W*3))
    frame[:, :, :W] = croc_input
    frame[:, :, W:2*W] = shifted_croc
    frame[:, :, 2*W:] = crop(croc_back, H, W)
    # plt.imshow('croc', frame.to(torch.uint8).permute((1, 2, 0)).numpy())
    if not db:
        cv2.imshow('croc', frame.permute((1,2,0)).clip(0,255).to(torch.uint8).numpy())
        cv2.waitKey(0)
    # plt.show()


if __name__ == '__main__':
    shift_then_unshift_croc()

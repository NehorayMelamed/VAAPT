from icecream import ic

from transforms import shift_matrix_subpixel
from alignments import normalized_cc_shifts, circular_cc_shifts

import torch
from torch import Tensor


def shift_then_unshift_croc():
    ma = True
    if ma:
        import cv2
        croc = cv2.imread('data/eye.jpg')
    else:
        croc = (torch.rand((447,640,3))*255)
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    print((H,W,C))
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_input = croc_input.to(torch.uint8)
    shifted_croc = shift_matrix_subpixel(croc_input, 20, 30, None, warp_method="fft")

    # croc_ref = shift_matrix_subpixel(croc_input, 20.1, 10.4)
    shifts = normalized_cc_shifts(shifted_croc, croc_input, 100)
    ic(shifts)
    croc_back = shift_matrix_subpixel(shifted_croc, -1*shifts[0][0], -1*shifts[1][0], warp_method='fft')
    frame = torch.zeros((C, H, W*3))
    frame[:, :, :W] = croc_input
    frame[:, :, W:2*W] = shifted_croc
    frame[:, :, 2*W:] = croc_back
    if ma:
        cv2.imshow("croc", frame.permute((1,2,0)).clip(0,255).to(torch.uint8).numpy())
        cv2.waitKey(0)

def sc():
    ma = True
    if ma:
        import cv2
        croc = cv2.imread('data/eye.jpg')
    else:
        croc = (torch.rand((447,640,3))*255)
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    print((H,W,C))
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_input = croc_input.to(torch.uint8)
    shifted_croc = shift_matrix_subpixel(croc_input, 60, 30, None, warp_method="fft")

    # croc_ref = shift_matrix_subpixel(croc_input, 20.1, 10.4)
    shifts = circular_cc_shifts(shifted_croc, croc_input)
    ic(shifts)
    croc_back = shift_matrix_subpixel(shifted_croc, -1*shifts[0][0], -1*shifts[1][0], warp_method='fft')
    frame = torch.zeros((C, H, W*3))
    frame[:, :, :W] = croc_input
    frame[:, :, W:2*W] = shifted_croc
    frame[:, :, 2*W:] = croc_back
    if ma:
        cv2.imshow("croc", frame.permute((1,2,0)).clip(0,255).to(torch.uint8).numpy())
        cv2.waitKey(0)


if __name__ == '__main__':
    shift_then_unshift_croc()

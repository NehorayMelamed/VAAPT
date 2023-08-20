import cv2, torch

from torch import Tensor

from transforms import shift_matrix_subpixel

from alignments import circular_cross_correlation, normalized_cross_correlation


def ccc_croc():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_ref = shift_matrix_subpixel(croc_input, 20.1, 10.4)
    croc_back = circular_cross_correlation(croc_input, croc_ref)
    print(croc_back.mean())


def n_ccc_croc():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_ref = shift_matrix_subpixel(croc_input, 20.1, 10.4)
    croc_back = normalized_cross_correlation(croc_input, croc_ref, 10)
    print(croc_back.mean())


def wierd_dims():
    target = torch.rand((10,30,3,10,50))
    target_sh = shift_matrix_subpixel(target, 10, 20)
    target_cc = circular_cross_correlation(target, target_sh)
    print(target_cc.shape)


if __name__ == '__main__':
    n_ccc_croc()

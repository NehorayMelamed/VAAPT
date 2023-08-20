import cv2
import numpy as np
import torch
from icecream import ic
from torch import Tensor

from transforms import shift_matrix_subpixel
from tests.testing_utils.type_transforms import tensor_to_iterable, convert_singleton_to_type


def test_no_crash_batch_shifts():
    test_tensor = torch.rand((5, 3, 100, 151))
    valid_methods = ['nearest', 'bilinear', 'bicubic', 'fft']
    valid_shift_types = [Tensor, list, tuple]
    shift_H = [-4.1, 3.2, 7.2, 100.5, 90.5]
    shift_W = [0, 0, 45.5, 23.1, 93.1]
    for method in valid_methods:
        mat_fft = torch.fft.fftn(test_tensor, dim=[-2, -1])
        for shift_type_H in valid_shift_types:
            for shift_type_W in valid_shift_types:
                for matrix_type in [list, tuple, np.array]:
                    if method != 'fft':
                        mat_fft = None
                    _ = shift_matrix_subpixel(tensor_to_iterable(test_tensor, matrix_type),
                                              shift_type_H(shift_H),
                                              shift_type_W(shift_W), mat_fft, method)
                    #print("finished one")


def test_fft():
    test_tensor = torch.rand((5, 3, 100, 151))
    shift_H = [-4.1, 3.2, 7.2, 100.5, 90.5]
    shift_W = [0, 0, 45.5, 23.1, 93.1]
    _ = shift_matrix_subpixel(test_tensor,
                              shift_H,
                              shift_W, None, 'fft')


def test_no_crash():  # verify correctness
    test_tensor = torch.zeros(10, 10)
    valid_methods = ['nearest', 'bilinear', 'bicubic', 'fft']
    valid_shift_types = [Tensor, list, tuple, int]
    shift_H = 20.1
    shift_W = 15
    for method in valid_methods:
        for shift_type_H in valid_shift_types:
            for shift_type_W in valid_shift_types:
                for matrix_type in [list, tuple, np.array]:
                    _ = shift_matrix_subpixel(tensor_to_iterable(test_tensor, matrix_type), convert_singleton_to_type(shift_H, shift_type_H),
                                                           convert_singleton_to_type(shift_W, shift_type_W), None, method)


def crash_tests():
    test_fft()
    ic("FFT TEST: PASSED")
    test_no_crash()
    ic("NO CRASH TEST: PASSED")
    test_no_crash_batch_shifts()
    ic("NO CRASH BATCH TEST: PASSED")


def shift_croc():
    croc = cv2.imread('data/eye.jpg')
    H, W, C = croc.shape[0], croc.shape[1], croc.shape[2]
    croc_input = Tensor(croc).permute((2, 0, 1))
    croc_back = shift_matrix_subpixel(croc_input, 100.5, Tensor([23])[0], warp_method='fft').clip(0,255).permute(1, 2, 0)
    both: Tensor = torch.zeros((H,W*2,C), dtype=torch.uint8)

    both[:, :W, :] = Tensor(croc).to(torch.uint8)
    both[:, W:, :] = croc_back.to(torch.uint8)
    cv2.imshow("croc", both.numpy())
    cv2.waitKey(0)
    #cv2.imshow('croc', croc_back.to(torch.uint8).numpy().reshape(H, W, C))


if __name__ == '__main__':
    shift_croc()

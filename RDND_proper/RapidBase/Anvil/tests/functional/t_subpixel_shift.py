import unittest

import numpy as np
import torch
from torch import Tensor

from transforms import shift_matrix_subpixel
from tests.testing_utils.matrix_creations import manhattan_distance_from_zero
from tests.testing_utils.type_transforms import tensor_to_iterable, convert_singleton_to_type


class Test_subpixel_shift(unittest.TestCase):
    def test_testing(self):
        self.assertTrue(True, "Testing launch or syntax failed")

    def test_no_crash(self):  # verify correctness
        test_tensor = manhattan_distance_from_zero(10, 10)
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

    def test_no_crash_batch_shifts(self):
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
                        print("finished one")

    def test_fft(self):
        test_tensor = torch.rand((5, 3, 100, 151))
        shift_H = [-4.1, 3.2, 7.2, 100.5, 90.5]
        shift_W = [0, 0, 45.5, 23.1, 93.1]
        _ = shift_matrix_subpixel(test_tensor,
                                  shift_H,
                                  shift_W, None, 'fft')


if __name__ == '__main__':
    unittest.main()

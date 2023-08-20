import unittest

import numpy as np
from torch import Tensor

import transforms
from tests.testing_utils.matrix_creations import manhattan_distance_from_zero
from tests.testing_utils.type_transforms import tensor_to_iterable, convert_singleton_to_type


class Test_integer_shift(unittest.TestCase):
    def test_testing(self):
        self.assertTrue(True, "Testing launch or syntax failed")

    def test_actual_shifts(self):  # verify correctness
        test_tensor = manhattan_distance_from_zero(10, 10)
        transformed_tensor = transforms.shift_matrix_integer_pixels(test_tensor, 3, 2)
        self.assertEqual(transformed_tensor[5][4], 4)
        self.assertEqual(transformed_tensor[6][5], 6)

    def test_formatting(self):
        test_tensor = manhattan_distance_from_zero(10, 10)
        shift_h = 3
        shift_w = 2
        correct = transforms.shift_matrix_integer_pixels(test_tensor, 3, 2) # previously verified
        shift_types = [Tensor, list, tuple, int]
        for matrix_type in [list, tuple, np.array]:
            for shift_type_H in shift_types:
                for shift_type_W in shift_types:
                    cur_tensor = tensor_to_iterable(test_tensor, matrix_type, dtype=int)
                    cur_shift_h = convert_singleton_to_type(shift_h, shift_type_H)
                    cur_shift_w = convert_singleton_to_type(shift_w, shift_type_W)
                    exotic_type_transform = transforms.shift_matrix_integer_pixels(cur_tensor, cur_shift_h, cur_shift_w)
                    self.assertTrue((correct==exotic_type_transform).all())


if __name__ == '__main__':
    unittest.main()

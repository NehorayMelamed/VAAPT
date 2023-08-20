from typing import Union, Tuple

import torch
from torch import Tensor


def is_integer_builtin_iterable(iterable: Union[list, Tuple]) -> bool:
    for element in iterable:
        if type(element) != int:
            return False
    return True


def is_integer_iterable(iterable: Union[Tensor, list, Tuple]) -> bool:
    if type(iterable) == tuple or type(iterable) == tuple:
        return is_integer_builtin_iterable(iterable)
    elif type(iterable) == Tensor:
        return len(torch.nonzero(iterable-iterable.to(torch.int64))) == 0
    else:
        raise TypeError("Unsupported type for is_integer_iterable")

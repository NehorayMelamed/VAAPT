import numpy as np
import torch
from torch import Tensor

from typing import Union


def tensor_to_tuple_list(a: Tensor, iterable_type=list, dtype=float) -> Union[list, tuple]:
    try:
        return iterable_type([dtype(a[i]) for i in range(a.shape[0])])
    except:
        return iterable_type([tensor_to_iterable(a[i]) for i in range(a.shape[0])])


def tensor_to_iterable(a: Tensor, iterable_type=list, dtype=float) -> list:
    if iterable_type == np.array:
        return a.numpy()
    else:
        return tensor_to_tuple_list(a, iterable_type, dtype)


def convert_singleton_to_type(element, container):
    element = float(element)
    if container in [float, int]:
        return container(element)
    else:  # probably iterable
        return container([element])

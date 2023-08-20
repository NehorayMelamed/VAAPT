from typing import List, Tuple, Union
from torch import Tensor

from RapidBase.Anvil._internal_utils.iterable_utils import is_integer_iterable
from RapidBase.Anvil._internal_utils.test_exceptions import raise_if_not


def type_check_parameters(arguments: List[Tuple]) -> None:
    """
    :param arguments: List of tuples of params and what their types can be
    """
    for num_arg, arg in enumerate(arguments):
        if type(arg[1]) in [list, tuple]:
            raise_if_not(type(arg[0]) in arg[1], message=f"Type {type(arg[0])} not valid for argument {num_arg}. Valid types: {arg[1]}")
        else:
            raise_if_not(type(arg[0]) == arg[1], message=f"Type {type(arg[0])} not valid for argument {num_arg}. Valid type: {arg[1]}")


def validate_warp_method(method: str, valid_methods=['bilinear', 'bicubic', 'nearest', 'fft']) -> None:
    raise_if_not(method in valid_methods, message="Invalid method")


def is_integer_argument(argument: Union[int, float, Tensor, list, Tuple]):
    if type(argument) == int:
        return True
    elif type(argument) in [list, tuple, Tensor]:
        return is_integer_iterable(argument)
    else:
        return False

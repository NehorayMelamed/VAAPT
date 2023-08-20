from typing import Union


def raise_if(condition: bool, error: Exception = RuntimeError, message: str = "RuntimeError") -> None:
    if condition:
        raise error(message)


def raise_if_not(condition: bool, error: Exception = RuntimeError, message: str = "RuntimeError") -> None:
    if not condition:
        raise error(message)


def raise_if_not_close(a: Union[int, float], b: Union[float, int], error: Exception = RuntimeError,
                       message: str = "RuntimeError", closeness_distance: float = 1e-7) -> None:
    if closeness_distance < abs(a-b):
        raise error(message)

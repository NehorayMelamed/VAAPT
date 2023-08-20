from typing import Dict, Callable

from RapidBase.Anvil._internal_utils.invalid_method import InvalidMethodError


def pick_method(functions: Dict[str, Callable], method: str, *args, **kwargs): # can return anything
    # pass functions in dictionary, and args in args and kwargs. Saves many lines of code
    # prune args of Null arguments
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("+++++++++++++++++++++++++")
    """
    args = [s_arg for s_arg in args if s_arg is not None]  # s_arg = single_arg
    kwargs = {arg_key: kwargs[arg_key] for arg_key in kwargs if kwargs[arg_key] is not None}
    """
    for s in args:
        print(type(s))
    for s in kwargs:
        print(f"{s}: {kwargs[s]}")
    print("\n\n\n\n\n********************************************\n\n\n\n\n")
    """

    if method in functions.keys():
        return functions[method](*args, **kwargs)
    else:
        raise InvalidMethodError("Given method not valid")


def not_implemented():
    raise NotImplementedError
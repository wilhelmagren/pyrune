from rune import criterion  # noqa
from rune import datautil  # noqa
from rune import nn  # noqa
from rune import optim  # noqa
from .tensor import Tensor


def tensor(*args, **kwargs) -> Tensor:
    """"""
    return Tensor(*args, **kwargs)

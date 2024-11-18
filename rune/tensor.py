from __future__ import annotations

import numpy as np

from ._function import Function
from typing import (
    Iterable,
    Optional,
    Union,
)


class Tensor:
    """A tensor is an algebraic object that describes a multilinear relationships between sets
    of algebraic objects related to a vector space [1]. The tensor is represented as a (potentially
    multidimensional) array. The total number of indices (m) required to identify each component
    of the tensor uniquely is equal to the dimension (dim) of an array. The total number of indices
    is referred to as order, degree, or rank of the tensor.

    [1] Wikipedia, Tensor, https://en.wikipedia.org/wiki/Tensor.
    """

    def __init__(
        self,
        data: Union[Iterable[Union[int, float]], np.ndarray],
        device: Optional[str] = None,
        requires_grad: bool = False,
    ) -> None:
        """"""

        if isinstance(data, (tuple, list)):
            data = np.array(data)

        self._data = data
        self._device: Optional[str] = device
        self._requires_grad: bool = requires_grad
        self._grad: Optional[Tensor] = None
        self._ctx: Optional[Function]

    def __repr__(self) -> str:
        """Get a developer suited representation of the ``Tensor``."""
        return f"tensor({self._data}, device={self._device}, grad_fn={self._ctx})"

    @property
    def data(self) -> np.ndarray:
        """"""
        return self._data
    
    @property
    def device(self) -> str:
        """"""
        return self._device

    @property
    def grad(self) -> Optional[Tensor]:
        """"""
        return self._grad

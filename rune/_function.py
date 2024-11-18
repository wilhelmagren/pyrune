from __future__ import annotations

from .tensor import Tensor
from typing import (
    Iterable,
    List,
    Optional,
    Type,
)


class Function:
    """
    """

    def __init__(self, *tensors: List[Tensor], device: str) -> None:
        self._children: List[Tensor] = tensors
        self._device: str = device
        self._saved_tensors: List[Tensor] = []
        self._requires_grad = any(t.requires_grad for t in tensors)

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"forward pass not implemented for {type(self)}"
        )
    
    def backward(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"backward pass not implemented for {type(self)}"
        )

    @classmethod
    def apply(
        cls: Type[Function],
        *tensors: Iterable[Tensor],
        device: Optional[str] = None,
    ) -> Tensor:
        """
        """

        if not isinstance(tensors, (tuple, list)):
            tensors = list(tensors)
        
        if device is None:
            device = tensors[0].device
        
        ctx = cls(*tensors, device=device)
        result = Tensor(
            ctx.forward(*tensors),
            device=ctx._device,
            requires_grad=ctx._requires_grad,
        )
        result._ctx = ctx
        return result


class Add(Function):
    pass
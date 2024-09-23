from __future__ import annotations

import abc
from typing import Iterable, Generic, List, TypeVar

import tassadar


T = TypeVar("T")

U = TypeVar("U")


class Proxy(Generic[T, U], abc.ABC):
    def __init__(self, batch_size: int = 64, callbacks: List[tassadar.Callback] = None):
        self.callbacks = callbacks or []
        self.coordinator = tassadar.Coordinator(proxy=self, batch_size=batch_size)

    @abc.abstractmethod
    async def _inference(self, arguments: Iterable[T]) -> Iterable[U]:
        pass

    async def inference(self, argument: T) -> U:
        return await self.coordinator.schedule(argument)

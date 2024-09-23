from __future__ import annotations

import abc
from typing import Iterable, List, Any


class Callback(abc.ABC):
    @abc.abstractmethod
    def on_batch_complete(self, batch: List[Any], results: Iterable[Any]):
        pass

    @abc.abstractmethod
    def on_batch_start(self, batch: List[Any]):
        pass

    @abc.abstractmethod
    def on_exception(self, exception: Exception):
        pass

    @abc.abstractmethod
    def on_schedule(self, argument: Any):
        pass

    @abc.abstractmethod
    def on_start(self):
        pass

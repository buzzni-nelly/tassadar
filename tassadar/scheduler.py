from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

import tassadar


class Scheduler:

    def __init__(self, proxy: tassadar.Proxy, batch_size: int = 64):
        if batch_size < 1:
            raise ValueError("Not allowed batch size under 1.")

        self.proxy = proxy
        self.batch_size = batch_size
        self.queue = asyncio.Queue()
        self.callbacks = self.proxy.callbacks

    async def schedule(self, future: concurrent.futures.Future, argument: Any):
        await self.queue.put((future, argument))
        for callback in self.callbacks:
            callback.on_schedule(argument)

    async def join(self) -> list[tuple[asyncio.Future, Any]]:
        argument = await self.queue.get()
        batch = [argument]
        for _ in range(self.batch_size - 1):
            if self.queue.empty():
                break
            args = self.queue.get_nowait()
            batch.append(args)
        return batch

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any

import tassadar


class Coordinator:

    def __init__(self, proxy: tassadar.Proxy, batch_size: int = 64):
        self.proxy = proxy
        self.callbacks = self.proxy.callbacks

        self.scheduler = tassadar.Scheduler(proxy=self.proxy, batch_size=batch_size)
        self.engine = tassadar.Engine(proxy=self.proxy, scheduler=self.scheduler)

        self.engine_event_loop = asyncio.new_event_loop()
        self.engine_thread = threading.Thread(target=self.run_engine, daemon=True)
        self.engine_thread.start()

    def run_engine(self):
        if threading.current_thread() is threading.main_thread():
            raise Exception("run_engine must not be called from the main thread")
        asyncio.set_event_loop(self.engine_event_loop)
        self.engine_event_loop.create_task(self.engine.run())
        self.engine_event_loop.run_forever()

    async def schedule(self, argument: Any) -> Any:
        future = concurrent.futures.Future()
        coro = self.scheduler.schedule(future, argument)
        asyncio.run_coroutine_threadsafe(coro, self.engine_event_loop)
        wrapped_future = asyncio.wrap_future(future)
        result = await wrapped_future
        return result

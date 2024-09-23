import asyncio
import itertools
from typing import Any
from typing import Iterable

import pytest

import tassadar


class DummyCallback(tassadar.Callback):
    def __init__(self):
        self.events = []

    def on_batch_complete(self, batch: list[Any], results: Iterable[Any]):
        self.events.append(("on_batch_complete", batch, list(results)))

    def on_batch_start(self, batch: list[Any]):
        self.events.append(("on_batch_start", batch))

    def on_exception(self, exception: Exception):
        self.events.append(("on_exception", exception))

    def on_schedule(self, argument: Any):
        self.events.append(("on_schedule", argument))

    def on_start(self):
        self.events.append(("on_start",))


class DummyProxy(tassadar.Proxy[str, str]):
    def __init__(
        self,
        delay: float = 0.1,
        batch_size: int = 64,
        raise_exception_on: Iterable[str] = None,
        callbacks: list[tassadar.Callback] = None,
    ):
        super().__init__(batch_size=batch_size, callbacks=callbacks)
        self.delay = delay
        self.batches = []
        self.raise_exception_on = (
            set(raise_exception_on) if raise_exception_on else set()
        )

    @property
    def flatten_batches(self):
        return list(itertools.chain.from_iterable(self.batches))

    async def _inference(self, arguments: Iterable[str]) -> Iterable[str]:
        self.batches.append(list(arguments))
        await asyncio.sleep(self.delay)
        results = []
        for arg in arguments:
            if arg in self.raise_exception_on:
                raise ValueError(f"Invalid input: {arg}")
            results.append(f"progressed_{arg}")
        return results


@pytest.mark.asyncio
async def test_single_inference():
    dummy_callbacks = DummyCallback()
    dummy_proxy = DummyProxy(delay=0.1, callbacks=[dummy_callbacks])
    input_data = "test_input"
    expected_output = "progressed_test_input"
    result = await dummy_proxy.inference(input_data)
    assert result == expected_output

    assert ("on_start",) in dummy_callbacks.events
    assert ("on_schedule", input_data) in dummy_callbacks.events

    batch_events = [
        event for event in dummy_callbacks.events if event[0] == "on_batch_start"
    ]
    assert len(batch_events) == 1
    assert input_data in batch_events[0][1][0][1]  # (future, argument)


@pytest.mark.asyncio
async def test_batch_processing():
    dummy_callbacks = DummyCallback()
    batch_size = 5
    dummy_proxy = DummyProxy(
        delay=0.1, batch_size=batch_size, callbacks=[dummy_callbacks]
    )
    inputs = [f"test_input_{i}" for i in range(10)]
    expected_outputs = [f"progressed_test_input_{i}" for i in range(10)]
    results = await asyncio.gather(*[dummy_proxy.inference(inp) for inp in inputs])
    assert results == expected_outputs

    assert len(dummy_proxy.batches) >= 2
    assert dummy_proxy.flatten_batches == inputs

    assert ("on_start",) in dummy_callbacks.events
    schedule_events = [
        event for event in dummy_callbacks.events if event[0] == "on_schedule"
    ]
    assert len(schedule_events) == 10

    batch_start_events = [
        event for event in dummy_callbacks.events if event[0] == "on_batch_start"
    ]
    assert len(batch_start_events) >= 2


@pytest.mark.asyncio
async def test_batch_size_one():
    dummy_callbacks = DummyCallback()
    batch_size = 1
    dummy_proxy = DummyProxy(
        delay=0.1, batch_size=batch_size, callbacks=[dummy_callbacks]
    )
    inputs = [f"test_input_{i}" for i in range(3)]
    expected_outputs = [f"progressed_test_input_{i}" for i in range(3)]
    results = await asyncio.gather(*[dummy_proxy.inference(inp) for inp in inputs])
    assert results == expected_outputs

    assert len(dummy_proxy.batches) >= 3
    assert dummy_proxy.flatten_batches == inputs

    batch_start_events = [
        event for event in dummy_callbacks.events if event[0] == "on_batch_start"
    ]
    assert len(batch_start_events) == 3


@pytest.mark.asyncio
async def test_large_batch_size():
    dummy_callbacks = DummyCallback()
    batch_size = 100
    dummy_proxy = DummyProxy(
        delay=0.1, batch_size=batch_size, callbacks=[dummy_callbacks]
    )
    inputs = [f"test_input_{i}" for i in range(50)]
    expected_outputs = [f"progressed_test_input_{i}" for i in range(50)]
    results = await asyncio.gather(*[dummy_proxy.inference(inp) for inp in inputs])
    assert results == expected_outputs

    assert len(dummy_proxy.batches) >= 1
    assert dummy_proxy.flatten_batches == inputs


def test_invalid_batch_size():
    class DummyProxy(tassadar.Proxy):
        async def _inference(self, arguments: Iterable[Any]) -> Iterable[Any]:
            pass

    with pytest.raises(ValueError) as exc_info:
        DummyProxy(batch_size=0)
    assert "Not allowed batch size under 1." in str(exc_info.value)


def test_negative_batch_size():
    class DummyProxy(tassadar.Proxy):
        async def _inference(self, arguments: Iterable[Any]) -> Iterable[Any]:
            pass

    with pytest.raises(ValueError) as exc_info:
        DummyProxy(batch_size=-5)
    assert "Not allowed batch size under 1." in str(exc_info.value)


@pytest.mark.asyncio
async def test_inference_exception():
    dummy_callbacks = DummyCallback()
    inputs = ["good_input", "bad_input", "another_good_input"]
    dummy_proxy = DummyProxy(
        delay=0.1, raise_exception_on=["bad_input"], callbacks=[dummy_callbacks]
    )
    tasks = [asyncio.create_task(dummy_proxy.inference(inp)) for inp in inputs]
    exceptions = []
    for task in tasks:
        try:
            _ = await task
        except Exception as e:
            exceptions.append(f"Exception: {e}")

    assert exceptions

    exception_events = [
        event for event in dummy_callbacks.events if event[0] == "on_exception"
    ]
    assert len(exception_events) >= 1


@pytest.mark.asyncio
async def test_concurrent_inference():
    dummy_callbacks = DummyCallback()
    dummy_proxy = DummyProxy(delay=0.1, batch_size=10, callbacks=[dummy_callbacks])
    inputs = [f"input_{i}" for i in range(100)]
    expected_outputs = [f"progressed_input_{i}" for i in range(100)]
    results = await asyncio.gather(*[dummy_proxy.inference(inp) for inp in inputs])
    assert results == expected_outputs

    assert len(dummy_proxy.batches) >= 10
    assert dummy_proxy.flatten_batches == inputs


@pytest.mark.asyncio
async def test_inference_with_delays():
    dummy_callbacks = DummyCallback()
    dummy_proxy = DummyProxy(delay=0.1, batch_size=5, callbacks=[dummy_callbacks])

    async def delayed_inference(proxy, input_data, delay):
        await asyncio.sleep(delay)
        return await proxy.inference(input_data)

    inputs = [f"input_{i}" for i in range(10)]
    delays = [0.05 * i for i in range(10)]
    tasks = [
        asyncio.create_task(delayed_inference(dummy_proxy, inp, delay))
        for inp, delay in zip(inputs, delays)
    ]
    results = await asyncio.gather(*tasks)
    expected_outputs = [f"progressed_input_{i}" for i in range(10)]
    assert results == expected_outputs

    batch_start_events = [
        event for event in dummy_callbacks.events if event[0] == "on_batch_start"
    ]
    assert len(batch_start_events) >= 2

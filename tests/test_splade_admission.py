import asyncio
from pathlib import Path

import pytest

from conftest import load_flat_service


REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "splade-service"
splade = load_flat_service("splade_service", SERVICE, "admission")
admission = splade.admission


def test_priority_waiter_takes_the_next_slot():
    async def body():
        semaphore = admission.PrioritySemaphore(1)
        await semaphore.acquire()
        order = []

        async def enter(name, priority=False):
            async with semaphore.slot(priority=priority):
                order.append(name)

        document = asyncio.create_task(enter("document"))
        await asyncio.sleep(0)
        query = asyncio.create_task(enter("query", priority=True))
        await asyncio.sleep(0)

        semaphore.release()
        await asyncio.gather(document, query)

        assert order == ["query", "document"]

    asyncio.run(body())


def test_cancelling_a_priority_waiter_passes_the_slot_on():
    async def body():
        semaphore = admission.PrioritySemaphore(1)
        await semaphore.acquire()
        entered = asyncio.Event()

        async def wait_for_query():
            async with semaphore.slot(priority=True):
                raise AssertionError("cancelled query entered")

        async def wait_for_document():
            async with semaphore.slot():
                entered.set()

        query = asyncio.create_task(wait_for_query())
        document = asyncio.create_task(wait_for_document())
        await asyncio.sleep(0)

        query.cancel()
        with pytest.raises(asyncio.CancelledError):
            await query
        semaphore.release()

        await asyncio.wait_for(entered.wait(), 0.1)
        await document

    asyncio.run(body())

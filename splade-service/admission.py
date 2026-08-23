import asyncio
from collections import deque
from contextlib import asynccontextmanager


class PrioritySemaphore:
    """A semaphore whose priority waiters take the next released slot.

    Running work is never interrupted. Priority only changes which queued caller
    starts next. Waiters retain first-in, first-out order within their class, and
    cancelling a waiter passes an assigned slot to the next caller.
    """

    def __init__(self, value):
        if value < 1:
            raise ValueError("semaphore value must be at least one")
        self._capacity = value
        self._value = value
        self._priority_waiters = deque()
        self._regular_waiters = deque()

    async def acquire(self, priority=False):
        if (
            self._value > 0
            and not self._priority_waiters
            and not self._regular_waiters
        ):
            self._value -= 1
            return

        future = asyncio.get_running_loop().create_future()
        queue = self._priority_waiters if priority else self._regular_waiters
        queue.append(future)
        try:
            await future
        except BaseException:
            if future.done() and not future.cancelled():
                self.release()
            else:
                future.cancel()
            raise

    def release(self):
        for queue in (self._priority_waiters, self._regular_waiters):
            while queue:
                future = queue.popleft()
                if future.done():
                    continue
                future.set_result(None)
                return
        self._value += 1
        if self._value > self._capacity:
            self._value -= 1
            raise ValueError("semaphore released too many times")

    @asynccontextmanager
    async def slot(self, priority=False):
        await self.acquire(priority=priority)
        try:
            yield
        finally:
            self.release()

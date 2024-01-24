"""
Example of using async generators.

async generators or the async for syntax does not automatically parallelize execution.
These are syntactic sugar that helps in cases like TCP streams or piped logs from processes,
where from time to time, the current program could be blocked on waiting for data.
"""

import asyncio
import time


async def my_async_generator():
    for i in range(10):
        yield i
        await asyncio.sleep(1)


async def main():
    now = time.perf_counter()
    async for i in my_async_generator():
        print(f'time elapsed: {time.perf_counter() - now:0.2f} seconds. i: {i}')


asyncio.run(main())

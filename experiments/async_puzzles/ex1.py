"""
First example of using async.

async is most helpful for IO-bound tasks, not CPU-bound tasks.
A function defined with async def is called a coroutine. When called, it returns a coroutine object.
The coroutine object does not do anything until you schedule its execution,
e.g., with asyncio.run() or asyncio.create_task().
coroutines don't block the event loop while waiting for the result.
"""
import asyncio

import aiohttp


async def scrape(url: str):
    async with aiohttp.ClientSession() as session:
        resp = await session.request(method="GET", url=url)
        resp.raise_for_status()
        html = await resp.text()
        return html


asyncio.run(scrape(url="https://www.google.com"))

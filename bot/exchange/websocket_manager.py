from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import aiohttp

from bot.utils.logger import logger


class WebsocketManager:
    def __init__(self, symbol: str, stream: str = 'trade') -> None:
        normalized = symbol.replace('/', '').lower()
        self.url = f'wss://stream.binance.com:9443/ws/{normalized}@{stream}'
        self._running = False

    async def stream_messages(self) -> AsyncIterator[dict]:
        self._running = True
        backoff = 1
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.url, heartbeat=30) as ws:
                        logger.info('Websocket connected: {}', self.url)
                        backoff = 1
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                yield json.loads(msg.data)
                            elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                                break
            except Exception as exc:
                logger.warning('Websocket reconnect in {}s due to {}', backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def stop(self) -> None:
        self._running = False


__all__ = ['WebsocketManager']

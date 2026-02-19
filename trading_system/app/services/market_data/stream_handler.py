from __future__ import annotations

import asyncio
import json
import logging
import random
import time

import websockets

from trading_system.app.config.settings import Settings
from trading_system.app.core.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


class MarketStreamHandler:
    def __init__(self, settings: Settings, bus: EventBus, ws_base: str) -> None:
        symbol = settings.symbol.lower()
        streams = '/'.join([f'{symbol}@kline_1m', f'{symbol}@trade', f'{symbol}@depth20@100ms'])
        self.url = ws_base + streams
        self.bus = bus
        self.heartbeat_interval = 15
        self.last_message_at = time.monotonic()

    async def _heartbeat(self, ws: websockets.WebSocketClientProtocol) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            stale_for = time.monotonic() - self.last_message_at
            if stale_for > self.heartbeat_interval * 2:
                logger.warning('WS heartbeat stale_for=%ss, forzando reconnect', round(stale_for, 2))
                await ws.close()
                return
            pong_waiter = await ws.ping()
            await asyncio.wait_for(pong_waiter, timeout=8)

    async def run(self) -> None:
        retry = 0
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=None, close_timeout=3) as ws:
                    logger.info('WS conectado a Binance')
                    retry = 0
                    self.last_message_at = time.monotonic()
                    hb_task = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for message in ws:
                            self.last_message_at = time.monotonic()
                            msg = json.loads(message)
                            stream = msg.get('stream', '')
                            payload = msg.get('data', {})
                            await self.bus.publish(Event(topic=f'market.{stream}', payload=payload))
                    finally:
                        hb_task.cancel()
                        await asyncio.gather(hb_task, return_exceptions=True)
            except Exception as exc:  # noqa: BLE001
                retry += 1
                sleep = min(60.0, (2**retry) + random.uniform(0, 0.5 * retry))
                logger.warning('WS desconectado: %s. Reintento en %.2fs', exc, sleep)
                await asyncio.sleep(sleep)

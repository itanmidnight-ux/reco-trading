from __future__ import annotations

import asyncio
import time

from fastapi import WebSocket

from trading_system.app.dashboard.service import DashboardService


class DashboardWebSocketStreamer:
    def __init__(self, dashboard_service: DashboardService) -> None:
        self.dashboard_service = dashboard_service

    async def stream(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                payload = await self.dashboard_service.get_dashboard_payload()
                payload['server_ts'] = int(time.time() * 1000)
                await websocket.send_json(payload)
                await asyncio.sleep(2)
        except Exception:
            await websocket.close()

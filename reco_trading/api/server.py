from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from reco_trading.config.settings import Settings
from reco_trading.database.repository import Repository


class APIServer:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings = settings
        self.repository = repository
        self.app = FastAPI(
            title="Reco-Trading API",
            description="API for Reco-Trading Bot",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        self._setup_routes()
        self._setup_websockets()
        self._setup_middleware()

    def _setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        @self.app.get("/")
        async def root():
            return {
                "name": "Reco-Trading API",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

        @self.app.get("/api/v1/status")
        async def status():
            return {
                "bot_running": True,
                "mode": os.getenv("ENVIRONMENT", "testnet"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        @self.app.get("/api/v1/balance")
        async def balance():
            try:
                bal = await self.repository.get_balance()
                return {
                    "total": bal.total,
                    "free": bal.free,
                    "locked": bal.locked,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/trades")
        async def trades(limit: int = 50):
            try:
                trades_list = await self.repository.get_recent_trades(limit=limit)
                return {
                    "trades": [
                        {
                            "id": t.id,
                            "symbol": t.symbol,
                            "side": t.side,
                            "quantity": t.quantity,
                            "entry_price": t.entry_price,
                            "exit_price": t.exit_price,
                            "pnl": t.pnl,
                            "status": t.status,
                            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                        }
                        for t in trades_list
                    ],
                    "count": len(trades_list),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/trades")
        async def trades(symbol: str | None = None, limit: int = 50):
            return {"trades": [], "count": 0}

        @self.app.get("/api/v1/open-trades")
        async def open_trades():
            try:
                trades = await self.repository.get_open_trades()
                return {
                    "trades": [
                        {
                            "id": t.id,
                            "symbol": t.symbol,
                            "side": t.side,
                            "quantity": t.quantity,
                            "entry_price": t.entry_price,
                            "unrealized_pnl": 0.0,
                            "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                        }
                        for t in trades
                    ],
                    "count": len(trades),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/stats")
        async def stats():
            try:
                session_pnl = await self.repository.get_session_pnl()
                daily_pnl = await self.repository.get_daily_pnl()
                return {
                    "session_pnl": session_pnl,
                    "daily_pnl": daily_pnl,
                    "open_trades": 0,
                    "closed_trades": 0,
                    "win_rate": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/config")
        async def config():
            return {
                "trading_symbol": self.settings.trading_symbol,
                "timeframe": self.settings.timeframe,
                "binance_testnet": self.settings.binance_testnet,
                "max_trades_per_day": self.settings.max_trades_per_day,
                "risk_per_trade_fraction": self.settings.risk_per_trade_fraction,
            }

    def _setup_websockets(self) -> None:
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
            except WebSocketDisconnect:
                pass

    def get_app(self) -> FastAPI:
        return self.app


def create_app(settings: Settings, repository: Repository) -> FastAPI:
    server = APIServer(settings, repository)
    return server.get_app()

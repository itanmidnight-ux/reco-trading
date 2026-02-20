from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from trading_system.app.database.models import Base, Candle, EquitySnapshot, OrderExecution, TradeSignal


class Repository:
    def __init__(self, dsn: str) -> None:
        self.engine: AsyncEngine = create_async_engine(dsn, pool_pre_ping=True)
        self._session = async_sessionmaker(self.engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self._session() as sess:
            yield sess

    async def save_candle(self, payload: dict[str, Any]) -> None:
        async with self.session() as s:
            s.add(Candle(**payload))
            await s.commit()

    async def save_signal(self, payload: dict[str, Any]) -> None:
        async with self.session() as s:
            s.add(TradeSignal(**payload))
            await s.commit()

    async def save_execution(self, payload: dict[str, Any]) -> None:
        async with self.session() as s:
            s.add(OrderExecution(**payload))
            await s.commit()

    async def save_equity_snapshot(self, payload: dict[str, Any]) -> None:
        async with self.session() as s:
            s.add(EquitySnapshot(**payload))
            await s.commit()

    async def save(self, table: str, payload: dict[str, Any]) -> None:
        if table == 'candles':
            await self.save_candle(payload)
        elif table == 'trade_signals':
            await self.save_signal(payload)
        elif table == 'order_executions':
            await self.save_execution(payload)
        elif table == 'equity_snapshots':
            await self.save_equity_snapshot(payload)
        else:
            raise ValueError(f'tabla no soportada: {table}')

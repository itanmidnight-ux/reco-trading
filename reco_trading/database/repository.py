from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reco_trading.database.models import Base, BotLog, MarketData, Signal, Trade


class Repository:
    """Data access layer for all database operations."""

    def __init__(self, dsn: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.engine = create_async_engine(dsn, echo=False, future=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def setup(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def record_log(self, level: str, state: str, message: str) -> None:
        await self._persist(BotLog(level=level, state=state, message=message, timestamp=datetime.utcnow()))

    async def record_signal(self, symbol: str, payload: Mapping[str, str], confidence: float, action: str) -> None:
        await self._persist(
            Signal(
                symbol=symbol,
                trend=payload["trend"],
                momentum=payload["momentum"],
                volume=payload["volume"],
                volatility=payload["volatility"],
                structure=payload["structure"],
                confidence=confidence,
                action=action,
            )
        )

    async def record_market_candle(self, symbol: str, timeframe: str, candle: Mapping[str, float]) -> None:
        await self._persist(
            MarketData(
                symbol=symbol,
                timeframe=timeframe,
                open=float(candle["open"]),
                high=float(candle["high"]),
                low=float(candle["low"]),
                close=float(candle["close"]),
                volume=float(candle["volume"]),
            )
        )

    async def create_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        order_id: str | None,
    ) -> Trade:
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=order_id,
        )
        await self._persist(trade)
        return trade

    async def close_trade(self, trade_id: int, exit_price: float, pnl: float, status: str) -> None:
        async with self.session_factory() as session:
            trade = await session.get(Trade, trade_id)
            if not trade:
                return
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.status = status
            await session.commit()

    async def get_daily_pnl(self) -> float:
        async with self.session_factory() as session:
            q = select(func.coalesce(func.sum(Trade.pnl), 0.0)).where(Trade.status != "OPEN")
            result = await session.execute(q)
            return float(result.scalar_one())

    async def _persist(self, obj: Base) -> None:
        try:
            async with self.session_factory() as session:
                session.add(obj)
                await session.commit()
        except Exception as exc:
            self.logger.error("db_persist_error: %s", exc)

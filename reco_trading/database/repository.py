from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from sqlalchemy import func, inspect, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reco_trading.database.models import Base, BotLog, ErrorLog, MarketData, Signal, StateChange, Trade

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def safe_db_call(default: Any = None) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(self: "Repository", *args: Any, **kwargs: Any) -> Any:
            try:
                return await func(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("db_error in %s: %s", func.__name__, exc)
                return default

        return wrapper  # type: ignore[return-value]

    return decorator


class Repository:
    """Data access layer for all database operations."""

    def __init__(self, dsn: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.engine = create_async_engine(dsn, echo=False, future=True)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    @safe_db_call()
    async def setup(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            columns = await conn.run_sync(lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("trades")})
            if "close_timestamp" not in columns:
                await conn.execute(text("ALTER TABLE trades ADD COLUMN close_timestamp DATETIME"))

    @safe_db_call()
    async def record_log(self, level: str, state: str, message: str) -> None:
        await self._persist(BotLog(level=level, state=state, message=message, timestamp=datetime.utcnow()))

    @safe_db_call()
    async def record_error(self, state: str, category: str, message: str) -> None:
        await self._persist(ErrorLog(state=state, category=category, message=message, timestamp=datetime.utcnow()))

    @safe_db_call()
    async def record_state_change(self, from_state: str, to_state: str, context: str = "") -> None:
        await self._persist(StateChange(from_state=from_state, to_state=to_state, context=context, timestamp=datetime.utcnow()))

    @safe_db_call()
    async def record_signal(self, symbol: str, payload: Mapping[str, str], confidence: float, action: str) -> None:
        await self._persist(
            Signal(
                symbol=symbol,
                trend=payload["trend"],
                momentum=payload["momentum"],
                volume=payload["volume"],
                volatility=payload["volatility"],
                structure=payload["structure"],
                order_flow=payload.get("order_flow", "NEUTRAL"),
                regime=payload.get("regime", "NORMAL_VOLATILITY"),
                confidence=confidence,
                action=action,
            )
        )

    @safe_db_call()
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

    @safe_db_call()
    async def close_trade(self, trade_id: int, exit_price: float, pnl: float, status: str) -> None:
        async with self.session_factory() as session:
            trade = await session.get(Trade, trade_id)
            if not trade:
                return
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.status = status
            trade.close_timestamp = datetime.utcnow()
            await session.commit()

    @safe_db_call(default=0.0)
    async def get_session_pnl(self) -> float:
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
        day_end = day_start + timedelta(days=1)
        async with self.session_factory() as session:
            q = (
                select(func.coalesce(func.sum(Trade.pnl), 0.0))
                .where(Trade.status != "OPEN")
                .where(func.coalesce(Trade.close_timestamp, Trade.timestamp) >= day_start)
                .where(func.coalesce(Trade.close_timestamp, Trade.timestamp) < day_end)
            )
            result = await session.execute(q)
            return float(result.scalar_one())

    @safe_db_call(default=0.0)
    async def get_daily_pnl(self) -> float:
        return await self.get_session_pnl()

    async def _persist(self, obj: Base) -> None:
        async with self.session_factory() as session:
            session.add(obj)
            await session.commit()

from __future__ import annotations

import logging
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from sqlalchemy import func, inspect, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reco_trading.database.models import Base, BotLog, ErrorLog, MarketData, RuntimeSetting, Signal, StateChange, Trade

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
                await conn.execute(text("ALTER TABLE trades ADD COLUMN IF NOT EXISTS close_timestamp TIMESTAMP WITH TIME ZONE"))
            if "entry_slippage_ratio" not in columns:
                await conn.execute(text("ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_slippage_ratio DOUBLE PRECISION"))
            if "exit_slippage_ratio" not in columns:
                await conn.execute(text("ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_slippage_ratio DOUBLE PRECISION"))
            await self._migrate_signals_columns(conn)
            await self._migrate_symbol_columns(conn)

    async def _migrate_signals_columns(self, conn: Any) -> None:
        existing_columns = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_columns("signals"))
        existing_column_names = {col["name"] for col in existing_columns}
        signals_table = Signal.__table__
        dialect = postgresql.dialect()
        preparer = dialect.identifier_preparer

        for column in signals_table.columns:
            if column.name in existing_column_names:
                continue
            if column.primary_key:
                continue

            column_name = preparer.quote(column.name)
            column_type = column.type.compile(dialect=dialect)
            migration_sql = text(
                f"ALTER TABLE {preparer.quote(signals_table.name)} "
                f"ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
            )
            await conn.execute(migration_sql)
            self.logger.info("Database migration applied: added column %s to signals table", column.name)

        order_flow_column = next((col for col in existing_columns if col["name"] == "order_flow"), None)
        if order_flow_column and getattr(order_flow_column.get("type"), "length", None) != 32:
            await conn.execute(text("ALTER TABLE signals ALTER COLUMN order_flow TYPE VARCHAR(32)"))
            self.logger.info("Database migration applied: widened signals.order_flow to VARCHAR(32)")

    async def _migrate_symbol_columns(self, conn: Any) -> None:
        await conn.execute(text("ALTER TABLE bot_logs ADD COLUMN IF NOT EXISTS symbol VARCHAR(20)"))
        await conn.execute(text("ALTER TABLE error_logs ADD COLUMN IF NOT EXISTS symbol VARCHAR(20)"))
        await conn.execute(text("ALTER TABLE state_changes ADD COLUMN IF NOT EXISTS symbol VARCHAR(20)"))

    @safe_db_call()
    async def record_log(self, level: str, state: str, message: str, symbol: str | None = None) -> None:
        await self._persist(BotLog(level=level, state=state, message=message, symbol=symbol, timestamp=datetime.utcnow()))

    @safe_db_call()
    async def record_error(self, state: str, category: str, message: str, symbol: str | None = None) -> None:
        await self._persist(ErrorLog(state=state, category=category, message=message, symbol=symbol, timestamp=datetime.utcnow()))

    @safe_db_call()
    async def record_state_change(self, from_state: str, to_state: str, context: str = "", symbol: str | None = None) -> None:
        await self._persist(StateChange(from_state=from_state, to_state=to_state, context=context, symbol=symbol, timestamp=datetime.utcnow()))

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
        entry_slippage_ratio: float | None = None,
    ) -> Trade:
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            order_id=order_id,
            entry_slippage_ratio=entry_slippage_ratio,
        )
        await self._persist(trade)
        return trade

    @safe_db_call()
    async def close_trade(self, trade_id: int, exit_price: float, pnl: float, status: str, exit_slippage_ratio: float | None = None) -> None:
        async with self.session_factory() as session:
            trade = await session.get(Trade, trade_id)
            if not trade:
                return
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.status = status
            trade.close_timestamp = datetime.utcnow()
            trade.exit_slippage_ratio = exit_slippage_ratio
            await session.commit()

    @safe_db_call(default={})
    async def get_runtime_settings(self) -> dict[str, Any]:
        async with self.session_factory() as session:
            result = await session.execute(select(RuntimeSetting))
            payload: dict[str, Any] = {}
            for row in result.scalars().all():
                try:
                    payload[row.key] = json.loads(row.value)
                except (TypeError, ValueError):
                    payload[row.key] = row.value
            return payload

    @safe_db_call()
    async def set_runtime_setting(self, key: str, value: Any) -> None:
        encoded = json.dumps(value)
        async with self.session_factory() as session:
            result = await session.execute(select(RuntimeSetting).where(RuntimeSetting.key == key))
            setting = result.scalar_one_or_none()
            if setting is None:
                setting = RuntimeSetting(key=key, value=encoded, updated_at=datetime.utcnow())
                session.add(setting)
            else:
                setting.value = encoded
                setting.updated_at = datetime.utcnow()
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

    @safe_db_call(default=[])
    async def get_open_trades(self, symbol: str) -> list[Trade]:
        async with self.session_factory() as session:
            q = (
                select(Trade)
                .where(Trade.symbol == symbol)
                .where(Trade.status == "OPEN")
                .order_by(Trade.timestamp.desc())
            )
            result = await session.execute(q)
            return list(result.scalars().all())

    @safe_db_call(default=[])
    async def get_recent_trades(self, limit: int = 200) -> list[Trade]:
        capped_limit = max(1, int(limit))
        async with self.session_factory() as session:
            q = select(Trade).order_by(Trade.timestamp.desc()).limit(capped_limit)
            result = await session.execute(q)
            return list(result.scalars().all())

    @safe_db_call(default=[])
    async def get_recent_logs(self, limit: int = 400) -> list[BotLog]:
        capped_limit = max(1, int(limit))
        async with self.session_factory() as session:
            q = select(BotLog).order_by(BotLog.timestamp.desc()).limit(capped_limit)
            result = await session.execute(q)
            return list(result.scalars().all())

    async def close(self) -> None:
        await self.engine.dispose()

    async def _persist(self, obj: Base) -> None:
        async with self.session_factory() as session:
            session.add(obj)
            await session.commit()

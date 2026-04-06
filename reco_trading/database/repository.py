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

from reco_trading.database.models import Base, BotLog, CustomData, DailyStats, BotState, ErrorLog, MarketData, Order, PairLock, RuntimeSetting, Signal, StateChange, Trade

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

    async def setup(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Only run PostgreSQL-specific migrations for PostgreSQL
            dsn = str(self.engine.url)
            if "postgresql" in dsn or "postgres" in dsn:
                try:
                    columns = await conn.run_sync(lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("trades")})
                    if "close_timestamp" not in columns:
                        await conn.execute(text("ALTER TABLE trades ADD COLUMN IF NOT EXISTS close_timestamp TIMESTAMP WITH TIME ZONE"))
                    if "entry_slippage_ratio" not in columns:
                        await conn.execute(text("ALTER TABLE trades ADD COLUMN IF NOT EXISTS entry_slippage_ratio DOUBLE PRECISION"))
                    if "exit_slippage_ratio" not in columns:
                        await conn.execute(text("ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_slippage_ratio DOUBLE PRECISION"))
                    await self._migrate_signals_columns(conn)
                    await self._migrate_market_data_columns(conn)
                    await self._migrate_orders_columns(conn)
                except Exception as e:
                    self.logger.warning(f"Column migration skipped: {e}")

    async def verify_connectivity(self) -> None:
        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

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

    async def _migrate_market_data_columns(self, conn: Any) -> None:
        existing_columns = await conn.run_sync(lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("market_data")})
        if "candle_timestamp" not in existing_columns:
            await conn.execute(text("ALTER TABLE market_data ADD COLUMN IF NOT EXISTS candle_timestamp TIMESTAMP WITH TIME ZONE"))
            self.logger.info("Database migration applied: added market_data.candle_timestamp")

    async def _migrate_orders_columns(self, conn: Any) -> None:
        existing_columns = await conn.run_sync(lambda sync_conn: {col["name"] for col in inspect(sync_conn).get_columns("orders")})
        order_migrations = {
            "ft_trade_id": "INTEGER REFERENCES trades(id)",
            "ft_order_side": "VARCHAR(25) NOT NULL",
            "ft_pair": "VARCHAR(25) NOT NULL",
            "ft_is_open": "BOOLEAN DEFAULT TRUE",
            "ft_amount": "DOUBLE PRECISION NOT NULL",
            "ft_price": "DOUBLE PRECISION NOT NULL",
            "ft_cancel_reason": "VARCHAR(160)",
            "order_id": "VARCHAR(255)",
            "order_type": "VARCHAR(50)",
            "average": "DOUBLE PRECISION",
            "filled": "DOUBLE PRECISION",
            "remaining": "DOUBLE PRECISION",
            "cost": "DOUBLE PRECISION",
            "stop_price": "DOUBLE PRECISION",
            "order_date": "TIMESTAMP WITH TIME ZONE",
            "order_filled_date": "TIMESTAMP WITH TIME ZONE",
            "order_update_date": "TIMESTAMP WITH TIME ZONE",
            "funding_fee": "DOUBLE PRECISION",
            "ft_fee_base": "DOUBLE PRECISION",
            "ft_order_tag": "VARCHAR(160)",
        }
        for col_name, col_type in order_migrations.items():
            if col_name not in existing_columns:
                await conn.execute(text(f"ALTER TABLE orders ADD COLUMN IF NOT EXISTS {col_name} {col_type}"))
                self.logger.info("Database migration applied: added orders.%s", col_name)

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
    async def record_signal(
        self,
        symbol: str,
        payload: Mapping[str, str],
        confidence: float,
        action: str,
        *,
        factor_scores: Mapping[str, float] | None = None,
        gating: Mapping[str, Any] | None = None,
        decision_reason: str = "UNKNOWN",
    ) -> None:
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
                factor_scores_json=json.dumps(dict(factor_scores or {})),
                gating_json=json.dumps(dict(gating or {})),
                decision_reason=decision_reason[:160],
            )
        )

    @safe_db_call()
    async def record_market_candle(self, symbol: str, timeframe: str, candle: Mapping[str, Any]) -> None:
        candle_timestamp = candle.get("timestamp")
        async with self.session_factory() as session:
            existing = None
            if isinstance(candle_timestamp, datetime):
                query = select(MarketData).where(
                    MarketData.symbol == symbol,
                    MarketData.timeframe == timeframe,
                    MarketData.candle_timestamp == candle_timestamp,
                )
                existing = (await session.execute(query)).scalar_one_or_none()

            if existing is None:
                session.add(
                    MarketData(
                        symbol=symbol,
                        timeframe=timeframe,
                        candle_timestamp=candle_timestamp if isinstance(candle_timestamp, datetime) else None,
                        open=float(candle["open"]),
                        high=float(candle["high"]),
                        low=float(candle["low"]),
                        close=float(candle["close"]),
                        volume=float(candle["volume"]),
                    )
                )
            else:
                existing.open = float(candle["open"])
                existing.high = float(candle["high"])
                existing.low = float(candle["low"])
                existing.close = float(candle["close"])
                existing.volume = float(candle["volume"])
                existing.timestamp = datetime.utcnow()
            await session.commit()

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
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
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
    async def get_trades(self, limit: int = 200, offset: int = 0, status_filter: str | None = None, symbol: str | None = None) -> list[Trade]:
        async with self.session_factory() as session:
            q = select(Trade)
            if status_filter:
                q = q.where(Trade.status == status_filter)
            if symbol:
                q = q.where(Trade.symbol == symbol)
            q = q.order_by(Trade.timestamp.desc()).limit(max(1, int(limit))).offset(max(0, int(offset)))
            result = await session.execute(q)
            return list(result.scalars().all())

    @safe_db_call(default={"total": 0, "closed": 0, "open": 0, "realized_pnl": 0.0, "win_rate": 0.0})
    async def get_trade_summary(self) -> dict[str, Any]:
        async with self.session_factory() as session:
            total_q = select(func.count(Trade.id))
            total_result = await session.execute(total_q)
            total = total_result.scalar_one() or 0

            closed_q = select(func.count(Trade.id)).where(Trade.status != "OPEN")
            closed_result = await session.execute(closed_q)
            closed = closed_result.scalar_one() or 0

            open_q = select(func.count(Trade.id)).where(Trade.status == "OPEN")
            open_result = await session.execute(open_q)
            open_count = open_result.scalar_one() or 0

            pnl_q = select(func.coalesce(func.sum(Trade.pnl), 0.0)).where(Trade.status != "OPEN")
            pnl_result = await session.execute(pnl_q)
            realized_pnl = float(pnl_result.scalar_one() or 0.0)

            wins_q = select(func.count(Trade.id)).where(Trade.status != "OPEN").where(Trade.pnl > 0)
            wins_result = await session.execute(wins_q)
            wins = wins_result.scalar_one() or 0

            win_rate = (wins / closed * 100) if closed > 0 else 0.0

            return {
                "total": total,
                "closed": closed,
                "open": open_count,
                "realized_pnl": realized_pnl,
                "win_rate": win_rate,
                "wins": wins,
                "losses": closed - wins,
            }

    @safe_db_call(default=[])
    async def get_recent_logs(self, limit: int = 400) -> list[BotLog]:
        capped_limit = max(1, int(limit))
        async with self.session_factory() as session:
            q = select(BotLog).order_by(BotLog.timestamp.desc()).limit(capped_limit)
            result = await session.execute(q)
            return list(result.scalars().all())

    @safe_db_call()
    async def cleanup_old_logs(self, keep_days: int = 7) -> None:
        """Elimina registros antiguos para controlar el tamaño de la DB."""
        cutoff = datetime.utcnow() - timedelta(days=max(int(keep_days), 1))
        async with self.session_factory() as session:
            await session.execute(text("DELETE FROM bot_logs WHERE timestamp < :cutoff"), {"cutoff": cutoff})
            await session.execute(text("DELETE FROM market_data WHERE timestamp < :cutoff"), {"cutoff": cutoff})
            await session.execute(text("DELETE FROM state_changes WHERE timestamp < :cutoff"), {"cutoff": cutoff})
            await session.commit()

    async def close(self) -> None:
        await self.engine.dispose()

    async def _persist(self, obj: Base) -> None:
        async with self.session_factory() as session:
            session.add(obj)
            await session.commit()

    @safe_db_call(default=[])
    async def create_order(
        self,
        trade_id: int,
        order_id: str,
        ft_pair: str,
        ft_order_side: str,
        ft_amount: float,
        ft_price: float,
        order_type: str | None = None,
        side: str | None = None,
        status: str | None = None,
        symbol: str | None = None,
        price: float | None = None,
        amount: float | None = None,
        filled: float | None = None,
        order_date: datetime | None = None,
    ) -> Order:
        order = Order(
            ft_trade_id=trade_id,
            order_id=order_id,
            ft_pair=ft_pair,
            ft_order_side=ft_order_side,
            ft_amount=ft_amount,
            ft_price=ft_price,
            order_type=order_type,
            side=side,
            status=status,
            symbol=symbol,
            price=price,
            amount=amount,
            filled=filled,
            order_date=order_date,
        )
        await self._persist(order)
        return order

    @safe_db_call(default=[])
    async def update_order(self, order_id: str, **kwargs: Any) -> None:
        async with self.session_factory() as session:
            query = select(Order).where(Order.order_id == order_id)
            order = (await session.execute(query)).scalar_one_or_none()
            if order:
                for key, value in kwargs.items():
                    if hasattr(order, key):
                        setattr(order, key, value)
                await session.commit()

    @safe_db_call(default=[])
    async def get_orders_for_trade(self, trade_id: int) -> list[Order]:
        async with self.session_factory() as session:
            query = select(Order).where(Order.ft_trade_id == trade_id).order_by(Order.order_date)
            result = await session.execute(query)
            return list(result.scalars().all())

    @safe_db_call(default=[])
    async def get_open_orders(self) -> list[Order]:
        async with self.session_factory() as session:
            query = select(Order).where(Order.ft_is_open == True).order_by(Order.order_date)
            result = await session.execute(query)
            return list(result.scalars().all())

    @safe_db_call()
    async def create_pairlock(
        self,
        pair: str,
        lock_time: datetime,
        lock_end_time: datetime,
        side: str = "*",
        reason: str | None = None,
    ) -> PairLock:
        pairlock = PairLock(
            pair=pair,
            side=side,
            reason=reason,
            lock_time=lock_time,
            lock_end_time=lock_end_time,
            active=True,
        )
        await self._persist(pairlock)
        return pairlock

    @safe_db_call(default=[])
    async def get_active_pairlocks(self, pair: str | None = None, side: str | None = None) -> list[PairLock]:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            query = select(PairLock).where(
                PairLock.lock_end_time > now,
                PairLock.active == True,
            )
            if pair:
                query = query.where(PairLock.pair == pair)
            if side and side != "*":
                query = query.where((PairLock.side == side) | (PairLock.side == "*"))
            result = await session.execute(query)
            return list(result.scalars().all())

    @safe_db_call(default=[])
    async def get_all_pairlocks(self) -> list[PairLock]:
        async with self.session_factory() as session:
            query = select(PairLock).order_by(PairLock.lock_time.desc())
            result = await session.execute(query)
            return list(result.scalars().all())

    @safe_db_call()
    async def unlock_pair(self, pairlock_id: int) -> None:
        async with self.session_factory() as session:
            pairlock = await session.get(PairLock, pairlock_id)
            if pairlock:
                pairlock.active = False
                await session.commit()

    @safe_db_call()
    async def set_custom_data(self, trade_id: int, key: str, value: Any) -> None:
        value_type = type(value).__name__
        unserialized_types = ["bool", "float", "int", "str"]

        if value_type not in unserialized_types:
            try:
                value_db = json.dumps(value)
            except TypeError as e:
                self.logger.warning(f"could not serialize {key} value due to {e}")
                return
        else:
            value_db = str(value)

        async with self.session_factory() as session:
            query = select(CustomData).where(
                CustomData.ft_trade_id == trade_id,
                CustomData.cd_key == key,
            )
            existing = (await session.execute(query)).scalar_one_or_none()

            if existing:
                existing.cd_value = value_db
                existing.cd_type = value_type
                existing.updated_at = datetime.now(timezone.utc)
            else:
                custom_data = CustomData(
                    ft_trade_id=trade_id,
                    cd_key=key,
                    cd_type=value_type,
                    cd_value=value_db,
                    created_at=datetime.now(timezone.utc),
                )
                session.add(custom_data)
            await session.commit()

    @safe_db_call(default=[])
    async def get_custom_data(self, trade_id: int, key: str | None = None) -> list[CustomData]:
        async with self.session_factory() as session:
            query = select(CustomData).where(CustomData.ft_trade_id == trade_id)
            if key:
                query = query.where(CustomData.cd_key == key)
            result = await session.execute(query)
            return list(result.scalars().all())

    @safe_db_call()
    async def delete_custom_data(self, trade_id: int, key: str | None = None) -> None:
        async with self.session_factory() as session:
            query = select(CustomData).where(CustomData.ft_trade_id == trade_id)
            if key:
                query = query.where(CustomData.cd_key == key)
            await session.execute(query)
            await session.commit()

    # ========================
    # Daily Stats Methods
    # ========================

    from reco_trading.database.models import DailyStats, BotState

    @safe_db_call()
    async def save_daily_stats(
        self,
        symbol: str,
        daily_pnl: float,
        session_pnl: float,
        trades_count: int,
        wins: int,
        losses: int,
        starting_balance: float | None = None,
        ending_balance: float | None = None,
        peak_balance: float | None = None,
        max_drawdown: float = 0.0,
    ) -> None:
        """Save daily trading statistics."""
        from reco_trading.database.models import DailyStats
        
        # Convert timezone-aware to naive for database compatibility
        today_aware = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        today_naive = today_aware.replace(tzinfo=None)
        
        async with self.session_factory() as session:
            result = await session.execute(
                select(DailyStats).where(DailyStats.date == today_naive, DailyStats.symbol == symbol)
            )
            stats = result.scalar_one_or_none()
            
            win_rate = (wins / trades_count * 100) if trades_count > 0 else 0.0
            
            if stats:
                stats.daily_pnl = daily_pnl
                stats.session_pnl = session_pnl
                stats.trades_count = trades_count
                stats.wins = wins
                stats.losses = losses
                stats.win_rate = win_rate
                if starting_balance is not None:
                    stats.starting_balance = starting_balance
                if ending_balance is not None:
                    stats.ending_balance = ending_balance
                if peak_balance is not None:
                    stats.peak_balance = peak_balance
                stats.max_drawdown = max_drawdown
                stats.updated_at = datetime.utcnow()
            else:
                stats = DailyStats(
                    date=today_naive,
                    symbol=symbol,
                    daily_pnl=daily_pnl,
                    session_pnl=session_pnl,
                    trades_count=trades_count,
                    wins=wins,
                    losses=losses,
                    win_rate=win_rate,
                    starting_balance=starting_balance,
                    ending_balance=ending_balance,
                    peak_balance=peak_balance,
                    max_drawdown=max_drawdown,
                )
                session.add(stats)
            await session.commit()

    @safe_db_call(default=None)
    async def get_daily_stats(self, symbol: str | None = None, days: int = 30) -> list[dict]:
        """Get daily stats for the last N days."""
        from reco_trading.database.models import DailyStats
        
        # Use naive datetime for database query
        cutoff = datetime.utcnow() - timedelta(days=days)
        async with self.session_factory() as session:
            query = select(DailyStats).where(DailyStats.date >= cutoff)
            if symbol:
                query = query.where(DailyStats.symbol == symbol)
            query = query.order_by(DailyStats.date.desc())
            result = await session.execute(query)
            return [
                {
                    "date": stat.date.isoformat() if stat.date else None,
                    "symbol": stat.symbol,
                    "daily_pnl": float(stat.daily_pnl or 0),
                    "session_pnl": float(stat.session_pnl or 0),
                    "trades_count": stat.trades_count or 0,
                    "wins": stat.wins or 0,
                    "losses": stat.losses or 0,
                    "win_rate": float(stat.win_rate or 0),
                    "starting_balance": float(stat.starting_balance or 0),
                    "ending_balance": float(stat.ending_balance or 0),
                    "peak_balance": float(stat.peak_balance or 0),
                    "max_drawdown": float(stat.max_drawdown or 0),
                }
                for stat in result.scalars().all()
            ]

    @safe_db_call(default=None)
    async def get_latest_daily_stats(self, symbol: str) -> dict | None:
        """Get the most recent daily stats for a symbol."""
        stats = await self.get_daily_stats(symbol=symbol, days=1)
        return stats[0] if stats else None

    # ========================
    # Bot State Methods
    # ========================

    @safe_db_call()
    async def save_bot_state(self, key: str, value: dict | str | float | int | bool) -> None:
        """Save bot state for recovery."""
        from reco_trading.database.models import BotState
        
        encoded = json.dumps(value) if not isinstance(value, str) else value
        async with self.session_factory() as session:
            result = await session.execute(select(BotState).where(BotState.key == key))
            state = result.scalar_one_or_none()
            if state:
                state.value = encoded
                state.updated_at = datetime.utcnow()
            else:
                state = BotState(key=key, value=encoded, updated_at=datetime.utcnow())
                session.add(state)
            await session.commit()

    @safe_db_call(default=None)
    async def get_bot_state(self, key: str) -> dict | str | None:
        """Get bot state by key."""
        from reco_trading.database.models import BotState
        
        async with self.session_factory() as session:
            result = await session.execute(select(BotState).where(BotState.key == key))
            state = result.scalar_one_or_none()
            if state:
                try:
                    return json.loads(state.value)
                except (TypeError, ValueError):
                    return state.value
        return None

    @safe_db_call(default={})
    async def get_all_bot_state(self) -> dict[str, Any]:
        """Get all bot state."""
        from reco_trading.database.models import BotState
        
        async with self.session_factory() as session:
            result = await session.execute(select(BotState))
            payload: dict[str, Any] = {}
            for row in result.scalars().all():
                try:
                    payload[row.key] = json.loads(row.value)
                except (TypeError, ValueError):
                    payload[row.key] = row.value
            return payload

    @safe_db_call()
    async def save_config(self, config: dict[str, Any]) -> None:
        """Save complete bot configuration."""
        await self.save_bot_state("bot_config", config)

    @safe_db_call(default={})
    async def load_config(self) -> dict[str, Any]:
        """Load complete bot configuration."""
        config = await self.get_bot_state("bot_config")
        return config if isinstance(config, dict) else {}

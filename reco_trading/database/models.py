from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base model for ORM tables."""
    
    # Add __table_args__ for better PostgreSQL compatibility
    __table_args__ = {'extend_existing': True}


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index('ix_trades_symbol_timestamp', 'symbol', 'timestamp'),
        Index('ix_trades_status_timestamp', 'status', 'timestamp'),
        Index('ix_trades_close_timestamp', 'close_timestamp'),
        {'extend_existing': True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(8))  # BUY or SELL
    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")  # OPEN, CLOSED, CANCELLED
    order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_slippage_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_slippage_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entry_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="trade", lazy="immediate")
    custom_data: Mapped[list["CustomData"]] = relationship("CustomData", back_populates="trade", lazy="immediate")


class DailyStats(Base):
    """Daily trading statistics for persistence across restarts."""
    __tablename__ = "daily_stats"
    __table_args__ = (
        Index('ix_daily_stats_date_symbol', 'date', 'symbol'),
        {'extend_existing': True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    daily_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    session_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    starting_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    ending_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class BotLog(Base):
    """Bot logs for dashboard and persistence."""
    __tablename__ = "bot_logs"
    __table_args__ = (
        Index('ix_bot_logs_level_timestamp', 'level', 'timestamp'),
        Index('ix_bot_logs_state_timestamp', 'state', 'timestamp'),
        {'extend_existing': True},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)
    level: Mapped[str] = mapped_column(String(10), index=True)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    state: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON additional details
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)


class BotState(Base):
    """Bot state persistence for recovery after restart."""
    __tablename__ = "bot_state"
    __table_args__ = {'extend_existing': True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)


class RuntimeSetting(Base):
    __tablename__ = "runtime_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    trend: Mapped[str] = mapped_column(String(10))
    momentum: Mapped[str] = mapped_column(String(10))
    volume: Mapped[str] = mapped_column(String(10))
    volatility: Mapped[str] = mapped_column(String(10))
    structure: Mapped[str] = mapped_column(String(10))
    order_flow: Mapped[str] = mapped_column(String(32), default="NEUTRAL")
    regime: Mapped[str] = mapped_column(String(24), default="NORMAL_VOLATILITY")
    confidence: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(10))
    factor_scores_json: Mapped[str] = mapped_column(Text, default="{}")
    gating_json: Mapped[str] = mapped_column(Text, default="{}")
    decision_reason: Mapped[str] = mapped_column(String(160), default="UNKNOWN")


class MarketData(Base):
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)
    candle_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)


class StateChange(Base):
    __tablename__ = "state_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)
    from_state: Mapped[str] = mapped_column(String(40))
    to_state: Mapped[str] = mapped_column(String(40), index=True)
    context: Mapped[str] = mapped_column(Text, default="")


class ErrorLog(Base):
    __tablename__ = "error_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, index=True)
    state: Mapped[str] = mapped_column(String(40), index=True)
    category: Mapped[str] = mapped_column(String(32), index=True)
    message: Mapped[str] = mapped_column(Text)





class Order(Base):
    """
    Order database model.
    Keeps a record of all orders placed on the exchange.

    One to many relationship with Trades:
      - One trade can have many orders
      - One Order can only be associated with one Trade
    """

    __tablename__ = "orders"

    __table_args__ = (UniqueConstraint("ft_pair", "order_id", name="_order_pair_order_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ft_trade_id: Mapped[int] = mapped_column(Integer, ForeignKey("trades.id"), index=True)

    trade: Mapped["Trade"] = relationship("Trade", back_populates="orders", lazy="immediate")

    ft_order_side: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_pair: Mapped[str] = mapped_column(String(25), nullable=False)
    ft_is_open: Mapped[bool] = mapped_column(default=True, index=True)
    ft_amount: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_price: Mapped[float] = mapped_column(Float(), nullable=False)
    ft_cancel_reason: Mapped[str | None] = mapped_column(String(160), nullable=True)

    order_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str | None] = mapped_column(String(255), nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(25), nullable=True)
    order_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    side: Mapped[str | None] = mapped_column(String(25), nullable=True)
    price: Mapped[float | None] = mapped_column(Float(), nullable=True)
    average: Mapped[float | None] = mapped_column(Float(), nullable=True)
    amount: Mapped[float | None] = mapped_column(Float(), nullable=True)
    filled: Mapped[float | None] = mapped_column(Float(), nullable=True)
    remaining: Mapped[float | None] = mapped_column(Float(), nullable=True)
    cost: Mapped[float | None] = mapped_column(Float(), nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float(), nullable=True)
    order_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    order_filled_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    order_update_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    funding_fee: Mapped[float | None] = mapped_column(Float(), nullable=True)

    ft_fee_base: Mapped[float | None] = mapped_column(Float(), nullable=True)
    ft_order_tag: Mapped[str | None] = mapped_column(String(160), nullable=True)

    @property
    def safe_amount(self) -> float:
        return self.amount or self.ft_amount

    @property
    def safe_placement_price(self) -> float:
        return self.price or self.stop_price or self.ft_price

    @property
    def safe_price(self) -> float:
        return self.average or self.price or self.stop_price or self.ft_price

    @property
    def safe_filled(self) -> float:
        return self.filled if self.filled is not None else 0.0

    @property
    def safe_cost(self) -> float:
        return self.cost or 0.0

    @property
    def safe_remaining(self) -> float:
        return self.remaining if self.remaining is not None else self.safe_amount - (self.filled or 0.0)

    @property
    def safe_fee_base(self) -> float:
        return self.ft_fee_base or 0.0


class PairLock(Base):
    """
    Pair Locks database model.
    Prevents trading on specific pairs for a period of time.
    """

    __tablename__ = "pairlocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    pair: Mapped[str] = mapped_column(String(25), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(25), nullable=False, default="*")
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lock_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lock_end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    active: Mapped[bool] = mapped_column(default=True, index=True)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pair": self.pair,
            "lock_time": self.lock_time.isoformat() if self.lock_time else None,
            "lock_end_time": self.lock_end_time.isoformat() if self.lock_end_time else None,
            "reason": self.reason,
            "side": self.side,
            "active": self.active,
        }


class CustomData(Base):
    """
    CustomData database model.
    Keeps records of metadata as key/value store for trades or global values.
    """

    __tablename__ = "trade_custom_data"

    __table_args__ = (UniqueConstraint("ft_trade_id", "cd_key", name="_trade_id_cd_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ft_trade_id: Mapped[int] = mapped_column(Integer, ForeignKey("trades.id"), index=True)

    trade: Mapped["Trade"] = relationship("Trade", back_populates="custom_data", lazy="immediate")

    cd_key: Mapped[str] = mapped_column(String(255), nullable=False)
    cd_type: Mapped[str] = mapped_column(String(25), nullable=False)
    cd_value: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    value: ClassVar[Any] = None

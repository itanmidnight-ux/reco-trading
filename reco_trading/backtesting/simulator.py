from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass(slots=True)
class SimulatedTrade:
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None
    pnl: float = 0.0
    status: str = "OPEN"


class TradeSimulator:
    """Simple execution simulator for historical replay."""

    def __init__(self, fee_rate: float = 0.001) -> None:
        self.fee_rate = max(fee_rate, 0.0)
        self.logger = logging.getLogger(__name__)
        self.trades: list[SimulatedTrade] = []
        self.open_trade: SimulatedTrade | None = None

    def open_position(self, side: str, quantity: float, price: float, timestamp: datetime) -> SimulatedTrade | None:
        if self.open_trade is not None or quantity <= 0:
            return None
        trade = SimulatedTrade(side=side, quantity=quantity, entry_price=price, entry_time=timestamp)
        self.open_trade = trade
        self.trades.append(trade)
        return trade

    def close_position(self, price: float, timestamp: datetime, reason: str = "SIGNAL_FLIP") -> SimulatedTrade | None:
        if self.open_trade is None:
            return None
        trade = self.open_trade
        gross = (price - trade.entry_price) * trade.quantity
        if trade.side == "SELL":
            gross *= -1
        fees = (trade.entry_price * trade.quantity + price * trade.quantity) * self.fee_rate
        trade.pnl = gross - fees
        trade.exit_price = price
        trade.exit_time = timestamp
        trade.status = reason
        self.open_trade = None
        return trade

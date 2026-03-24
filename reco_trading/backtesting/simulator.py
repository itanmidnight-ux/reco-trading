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
    expected_fill_price: float | None = None
    realized_fill_price: float | None = None
    filled_quantity: float = 0.0
    gross_pnl: float = 0.0
    commission_paid: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0


class TradeSimulator:
    """Simple execution simulator for historical replay."""

    def __init__(
        self,
        maker_fee_rate: float = 0.0002,
        taker_fee_rate: float = 0.0007,
        base_spread_ratio: float = 0.0005,
        base_slippage_ratio: float = 0.0004,
    ) -> None:
        self.maker_fee_rate = max(maker_fee_rate, 0.0)
        self.taker_fee_rate = max(taker_fee_rate, 0.0)
        self.base_spread_ratio = max(base_spread_ratio, 0.0)
        self.base_slippage_ratio = max(base_slippage_ratio, 0.0)
        self.logger = logging.getLogger(__name__)
        self.trades: list[SimulatedTrade] = []
        self.open_trade: SimulatedTrade | None = None

    def open_position(
        self,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        *,
        volatility_ratio: float = 0.0,
        liquidity_ratio: float = 1.0,
        aggressor: bool = True,
    ) -> SimulatedTrade | None:
        if self.open_trade is not None or quantity <= 0:
            return None
        fill_ratio = self._fill_ratio(volatility_ratio=volatility_ratio, liquidity_ratio=liquidity_ratio)
        filled_qty = quantity * fill_ratio
        if filled_qty <= 0:
            return None
        expected_fill = price
        realized_fill, spread_cost, slippage_cost = self._realized_price(
            side=side,
            expected_price=price,
            volatility_ratio=volatility_ratio,
            liquidity_ratio=liquidity_ratio,
        )
        fee_rate = self.taker_fee_rate if aggressor else self.maker_fee_rate
        commission = realized_fill * filled_qty * fee_rate
        trade = SimulatedTrade(
            side=side,
            quantity=quantity,
            entry_price=realized_fill,
            entry_time=timestamp,
            expected_fill_price=expected_fill,
            realized_fill_price=realized_fill,
            filled_quantity=filled_qty,
            commission_paid=commission,
            spread_cost=spread_cost * filled_qty,
            slippage_cost=slippage_cost * filled_qty,
        )
        self.open_trade = trade
        self.trades.append(trade)
        return trade

    def close_position(
        self,
        price: float,
        timestamp: datetime,
        reason: str = "SIGNAL_FLIP",
        *,
        volatility_ratio: float = 0.0,
        liquidity_ratio: float = 1.0,
        aggressor: bool = True,
    ) -> SimulatedTrade | None:
        if self.open_trade is None:
            return None
        trade = self.open_trade
        qty = trade.filled_quantity if trade.filled_quantity > 0 else trade.quantity
        expected_exit = price
        realized_exit, spread_cost, slippage_cost = self._realized_price(
            side="SELL" if trade.side == "BUY" else "BUY",
            expected_price=price,
            volatility_ratio=volatility_ratio,
            liquidity_ratio=liquidity_ratio,
        )
        gross = (realized_exit - trade.entry_price) * qty
        if trade.side == "SELL":
            gross *= -1
        fee_rate = self.taker_fee_rate if aggressor else self.maker_fee_rate
        fees = trade.commission_paid + (realized_exit * qty * fee_rate)
        trade.gross_pnl = gross
        trade.pnl = gross - fees
        trade.commission_paid = fees
        trade.exit_price = realized_exit
        trade.exit_time = timestamp
        trade.status = reason
        trade.expected_fill_price = trade.expected_fill_price or expected_exit
        trade.realized_fill_price = trade.realized_fill_price or trade.entry_price
        trade.spread_cost += spread_cost * qty
        trade.slippage_cost += slippage_cost * qty
        self.open_trade = None
        return trade

    def _fill_ratio(self, *, volatility_ratio: float, liquidity_ratio: float) -> float:
        penalty = max(0.0, volatility_ratio) * 0.2 + max(0.0, 1 - liquidity_ratio) * 0.4
        return max(0.25, min(1.0, 1.0 - penalty))

    def _realized_price(
        self,
        *,
        side: str,
        expected_price: float,
        volatility_ratio: float,
        liquidity_ratio: float,
    ) -> tuple[float, float, float]:
        spread_ratio = self.base_spread_ratio * (1 + max(volatility_ratio, 0.0) * 1.5)
        slippage_ratio = self.base_slippage_ratio * (1 + max(volatility_ratio, 0.0) * 2.0 + max(0.0, 1 - liquidity_ratio))
        total_ratio = spread_ratio + slippage_ratio
        sign = 1 if side.upper() == "BUY" else -1
        realized = expected_price * (1 + (sign * total_ratio))
        return max(realized, 1e-9), expected_price * spread_ratio, expected_price * slippage_ratio

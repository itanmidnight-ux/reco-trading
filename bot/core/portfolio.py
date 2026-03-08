from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float


@dataclass
class Portfolio:
    quote_balance: float = 0.0
    base_balance: float = 0.0
    open_position: Position | None = None
    realized_pnl: float = 0.0
    daily_trades: int = 0
    daily_loss: float = 0.0
    seen_client_order_ids: set[str] = field(default_factory=set)

    def reset_daily_counters(self) -> None:
        self.daily_trades = 0
        self.daily_loss = 0.0


__all__ = ['Portfolio', 'Position']

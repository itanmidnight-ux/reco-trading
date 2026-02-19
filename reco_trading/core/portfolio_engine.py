from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PortfolioState:
    equity: float = 1000.0
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    last_signal: str = 'HOLD'


class PortfolioEngine:
    def __init__(self, state: PortfolioState | None = None) -> None:
        self.state = state or PortfolioState()

    def register_trade_result(self, pnl: float) -> None:
        self.state.daily_pnl += pnl
        self.state.equity += pnl
        self.state.consecutive_losses = self.state.consecutive_losses + 1 if pnl < 0 else 0

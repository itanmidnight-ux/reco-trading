from dataclasses import dataclass


@dataclass
class PortfolioState:
    equity: float
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 3
    trading_blocked: bool = False


class PortfolioEngine:
    def __init__(self, state: PortfolioState) -> None:
        self.state = state

    def register_trade_result(self, pnl: float) -> None:
        if self.state.trading_blocked:
            return

        self.state.daily_pnl += pnl
        self.state.equity += pnl

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        if self.state.consecutive_losses >= self.state.max_consecutive_losses:
            self.state.trading_blocked = True

    def can_trade(self) -> bool:
        return not self.state.trading_blocked

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_score: float = 0.0
    samples: int = 0
    recent_returns: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / abs(self.gross_loss) if self.gross_loss < 0 else 99.0

    def update_score(self, score: float) -> None:
        self.samples += 1
        self.avg_score += (score - self.avg_score) / self.samples

    def register_trade(self, pnl: float) -> None:
        self.trades += 1
        self.pnl += pnl
        self.peak_pnl = max(self.peak_pnl, self.pnl)
        dd = self.peak_pnl - self.pnl
        self.max_drawdown = max(self.max_drawdown, dd)
        if pnl >= 0:
            self.wins += 1
            self.gross_profit += pnl
        else:
            self.losses += 1
            self.gross_loss += pnl
        self.recent_returns.append(pnl)
        if len(self.recent_returns) > 200:
            self.recent_returns.pop(0)


class MonitoringService:
    def __init__(self) -> None:
        self.metrics = Metrics()
        self.live_mode_enabled = False

    def set_live_mode(self, enabled: bool) -> None:
        self.live_mode_enabled = enabled

    def snapshot(self) -> dict[str, float | int]:
        return {
            'trades': self.metrics.trades,
            'win_rate': round(self.metrics.win_rate, 4),
            'profit_factor': round(self.metrics.profit_factor, 4),
            'pnl': round(self.metrics.pnl, 6),
            'drawdown': round(self.metrics.max_drawdown, 6),
            'avg_score': round(self.metrics.avg_score, 4),
        }

    def alerts(self, weight_usage: int, weight_limit: int, drawdown: float, ws_stale_seconds: float) -> list[str]:
        messages: list[str] = []
        if weight_usage >= int(weight_limit * 0.8):
            messages.append('Rate limit alto')
        if drawdown > 0.12:
            messages.append('Drawdown crítico')
        if ws_stale_seconds > 25:
            messages.append('WebSocket stale/sin datos recientes')
        if self.live_mode_enabled:
            messages.append('CRITICAL: Sistema operando en LIVE')
        return messages

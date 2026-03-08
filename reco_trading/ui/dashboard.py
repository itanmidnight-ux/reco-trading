from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table


@dataclass
class DashboardSnapshot:
    state: str = "STARTING"
    pair: str = "BTCUSDT"
    timeframe: str = "5m/15m"
    price: float = 0.0
    trend: str = "NEUTRAL"
    signals: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    signal_grade: str = "WEAK"
    volatility_regime: str = "NORMAL_VOLATILITY"
    order_flow_signal: str = "NEUTRAL"
    balance: float = 0.0
    available_capital: float = 0.0
    open_position: str = "NONE"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    daily_pnl: float = 0.0


class TerminalDashboard:
    """Renderable Rich dashboard component."""

    def render(self, snap: DashboardSnapshot) -> Any:
        status = Table.grid(expand=True)
        status.add_column()
        status.add_column()
        status.add_row("Bot Status", "RUNNING")
        status.add_row("State", snap.state)
        status.add_row("Pair", snap.pair)
        status.add_row("Price", f"{snap.price:.2f}")
        status.add_row("Timeframe", snap.timeframe)
        status.add_row("Confidence", f"{snap.confidence:.2%} ({snap.signal_grade})")
        status.add_row("Volatility Regime", snap.volatility_regime)
        status.add_row("Order Flow", snap.order_flow_signal)

        portfolio = Table.grid(expand=True)
        portfolio.add_column()
        portfolio.add_column()
        portfolio.add_row("Account Balance", f"{snap.balance:.4f} USDT")
        portfolio.add_row("Available Balance", f"{snap.available_capital:.4f} USDT")
        portfolio.add_row("Open Position", snap.open_position)
        portfolio.add_row("Entry Price", f"{snap.entry_price:.2f}")
        portfolio.add_row("Stop Loss", f"{snap.stop_loss:.2f}")
        portfolio.add_row("Take Profit", f"{snap.take_profit:.2f}")
        portfolio.add_row("Position Size", f"{snap.position_size:.8f}")
        portfolio.add_row("Daily PnL", f"{snap.daily_pnl:.4f} USDT")

        signal_table = Table(title="Signal Engines", expand=True)
        signal_table.add_column("Engine")
        signal_table.add_column("Signal")
        for key in ["trend", "momentum", "volume", "volatility", "structure", "order_flow"]:
            signal_table.add_row(key, snap.signals.get(key, "NEUTRAL"))

        return Group(
            Panel(status, title="Reco Trading Bot"),
            Panel(portfolio, title="Portfolio & Risk"),
            Panel(signal_table),
        )

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
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column()
        table.add_row("State", snap.state)
        table.add_row("Pair", snap.pair)
        table.add_row("Price", f"{snap.price:.2f}")
        table.add_row("Timeframe", snap.timeframe)
        table.add_row("Trend", snap.trend)
        table.add_row("Confidence", f"{snap.confidence:.2%}")
        table.add_row("Balance", f"{snap.balance:.4f} USDT")
        table.add_row("Available", f"{snap.available_capital:.4f} USDT")
        table.add_row("Open Position", snap.open_position)
        table.add_row("Entry", f"{snap.entry_price:.2f}")
        table.add_row("Stop Loss", f"{snap.stop_loss:.2f}")
        table.add_row("Take Profit", f"{snap.take_profit:.2f}")
        table.add_row("Position Size", f"{snap.position_size:.8f} BTC")
        table.add_row("Daily PnL", f"{snap.daily_pnl:.4f} USDT")

        signal_table = Table(title="Signal Engines", expand=True)
        signal_table.add_column("Engine")
        signal_table.add_column("Signal")
        for key in ["trend", "momentum", "volume", "volatility", "structure"]:
            signal_table.add_row(key, snap.signals.get(key, "NEUTRAL"))

        return Group(Panel(table, title="Reco Trading Bot"), Panel(signal_table))

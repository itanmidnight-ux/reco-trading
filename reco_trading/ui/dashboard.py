from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table


@dataclass
class DashboardSnapshot:
    state: str = "INITIALIZING"
    pair: str = ""
    timeframe: str = ""
    price: float | None = None
    spread: float | None = None
    trend: str | None = None
    adx: float | None = None
    volatility_regime: str | None = None
    order_flow: str | None = None
    signal: str | None = None
    confidence: float | None = None
    balance: float | None = None
    equity: float | None = None
    daily_pnl: float | None = None
    operable_capital_usdt: float | None = None
    capital_profile: str | None = None
    trades_today: int = 0
    win_rate: float | None = None
    last_trade: str | None = None
    cooldown: str | None = None
    signals: dict[str, str] = field(default_factory=dict)
    decision_trace: dict[str, Any] = field(default_factory=dict)
    decision_gating: dict[str, Any] = field(default_factory=dict)
    decision_reason: str | None = None
    api_latency_p95_ms: float | None = None
    stale_market_data_ratio: float | None = None
    exchange_reconnections: int = 0
    circuit_breaker_trips: int = 0
    database_status: str | None = None
    exchange_status: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DashboardSnapshot":
        return cls(
            state=str(data.get("status", "INITIALIZING")),
            pair=str(data.get("pair", "")),
            timeframe=str(data.get("timeframe", "")),
            price=_to_float(data.get("price")),
            spread=_to_float(data.get("spread")),
            trend=_to_text(data.get("trend")),
            adx=_to_float(data.get("adx")),
            volatility_regime=_to_text(data.get("volatility_regime")),
            order_flow=_to_text(data.get("order_flow")),
            signal=_to_text(data.get("signal")),
            confidence=_to_float(data.get("confidence")),
            balance=_to_float(data.get("balance")),
            equity=_to_float(data.get("equity")),
            daily_pnl=_to_float(data.get("daily_pnl")),
            operable_capital_usdt=_to_float(data.get("operable_capital_usdt")),
            capital_profile=_to_text(data.get("capital_profile")),
            trades_today=int(data.get("trades_today", 0) or 0),
            win_rate=_to_float(data.get("win_rate")),
            last_trade=_to_text(data.get("last_trade")),
            cooldown=_to_text(data.get("cooldown")),
            signals=dict(data.get("signals", {}) or {}),
            decision_trace=dict(data.get("decision_trace", {}) or {}),
            decision_gating=dict(data.get("decision_gating", {}) or {}),
            decision_reason=_to_text(data.get("decision_reason")),
            api_latency_p95_ms=_to_float(data.get("api_latency_p95_ms")),
            stale_market_data_ratio=_to_float(data.get("stale_market_data_ratio")),
            exchange_reconnections=int(data.get("exchange_reconnections", 0) or 0),
            circuit_breaker_trips=int(data.get("circuit_breaker_trips", 0) or 0),
            database_status=_to_text(data.get("database_status")),
            exchange_status=_to_text(data.get("exchange_status")),
        )


class TerminalDashboard:
    """Renderable Rich dashboard component."""

    def __init__(self) -> None:
        self.console = Console()

    def render(self, snapshot: DashboardSnapshot | Mapping[str, Any]) -> Any:
        try:
            snap = DashboardSnapshot.from_mapping(snapshot) if isinstance(snapshot, Mapping) else snapshot

            headline = Table.grid(expand=True)
            headline.add_column(ratio=3)
            headline.add_column(ratio=2, justify="right")
            headline.add_row(
                f"[bold cyan]{snap.pair or '-'}[/bold cyan]  •  [white]{snap.timeframe or '-'}[/white]",
                f"[bold]{snap.state}[/bold]",
            )
            headline.add_row(
                f"Signal [bold]{snap.signal or '-'}[/bold]  •  Trend [bold]{snap.trend or '-'}[/bold]",
                f"Confidence [bold green]{_fmt_pct(snap.confidence)}[/bold green]",
            )

            status = Table.grid(expand=True)
            status.add_column()
            status.add_column()
            status.add_row("Bot Status", snap.state)
            status.add_row("Pair", snap.pair or "-")
            status.add_row("Timeframe", snap.timeframe or "-")
            status.add_row("Price", _fmt_num(snap.price, 2))
            status.add_row("Spread", _fmt_num(snap.spread, 6))
            status.add_row("Trend", snap.trend or "-")
            status.add_row("ADX", _fmt_num(snap.adx, 2))
            status.add_row("Signal", snap.signal or "-")
            status.add_row("Confidence", _fmt_pct(snap.confidence))
            status.add_row("Volatility Regime", snap.volatility_regime or "-")
            status.add_row("Order Flow", snap.order_flow or "-")

            portfolio = Table.grid(expand=True)
            portfolio.add_column()
            portfolio.add_column()
            portfolio.add_row("Balance", f"{_fmt_num(snap.balance, 4)} USDT")
            portfolio.add_row("Equity", f"{_fmt_num(snap.equity, 4)} USDT")
            portfolio.add_row("Operable Capital", f"{_fmt_num(snap.operable_capital_usdt, 4)} USDT")
            portfolio.add_row("Capital Profile", snap.capital_profile or "-")
            portfolio.add_row("Daily PnL", f"{_fmt_num(snap.daily_pnl, 4)} USDT")
            portfolio.add_row("Trades Today", str(snap.trades_today))
            portfolio.add_row("Win Rate", _fmt_pct(snap.win_rate))
            portfolio.add_row("Last Trade", snap.last_trade or "-")
            portfolio.add_row("Cooldown", snap.cooldown or "-")

            signal_table = Table(title="Signal Engines", expand=True)
            signal_table.add_column("Engine")
            signal_table.add_column("Signal")
            for key in ["trend", "momentum", "volume", "volatility", "structure", "order_flow"]:
                signal_table.add_row(key, str(snap.signals.get(key, "-")))

            health = Table(title="System Health", expand=True)
            health.add_column("Metric")
            health.add_column("Value")
            health.add_row("API latency p95", f"{_fmt_num(snap.api_latency_p95_ms, 2)} ms")
            health.add_row("Stale market data", _fmt_pct(snap.stale_market_data_ratio))
            health.add_row("Reconnections", str(snap.exchange_reconnections))
            health.add_row("Circuit breaker trips", str(snap.circuit_breaker_trips))
            health.add_row("DB", snap.database_status or "-")
            health.add_row("Exchange", snap.exchange_status or "-")

            decision = Table(title="Decision Trace", expand=True)
            decision.add_column("Field")
            decision.add_column("Value")
            for factor, score in (snap.decision_trace.get("factor_scores") or {}).items():
                decision.add_row(f"factor:{factor}", f"{float(score):+.3f}")
            for gate, value in snap.decision_gating.items():
                decision.add_row(f"gate:{gate}", str(value))
            decision.add_row("reason", snap.decision_reason or "-")

            layout = Layout()
            layout.split_column(
                Layout(Panel(headline, title="Executive Snapshot", border_style="bright_blue"), ratio=1),
                Layout(Panel(status, title="Reco Trading Bot", border_style="cyan"), ratio=2),
                Layout(Panel(portfolio, title="Portfolio & Risk"), ratio=2),
                Layout(Panel(signal_table, border_style="magenta"), ratio=3),
                Layout(Panel(health, border_style="green"), ratio=2),
                Layout(Panel(decision, border_style="yellow"), ratio=3),
            )
            return Group(layout)
        except Exception as exc:  # noqa: BLE001
            return Panel(f"Dashboard render error: {exc}", title="Reco Trading Bot", border_style="red")


def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _fmt_num(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"

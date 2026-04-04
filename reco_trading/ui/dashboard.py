from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rich.align import Align
from rich.box import ROUNDED, SIMPLE_HEAVY
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


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
    exit_intelligence_score: float | None = None
    exit_intelligence_threshold: float | None = None
    exit_intelligence_reason: str | None = None
    exit_intelligence_codes: list[str] = field(default_factory=list)
    exit_intelligence_events: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    unrealized_pnl: float | None = None
    open_position_side: str | None = None
    open_position_entry: float | None = None
    open_position_qty: float | None = None
    open_position_sl: float | None = None
    open_position_tp: float | None = None
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    llm_mode: str | None = None
    llm_trade_confirmator: dict[str, Any] = field(default_factory=dict)

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
            exit_intelligence_score=_to_float(data.get("exit_intelligence_score")),
            exit_intelligence_threshold=_to_float(data.get("exit_intelligence_threshold")),
            exit_intelligence_reason=_to_text(data.get("exit_intelligence_reason")),
            exit_intelligence_codes=[str(item) for item in (data.get("exit_intelligence_codes") or [])],
            exit_intelligence_events=[dict(item) for item in (data.get("exit_intelligence_log") or []) if isinstance(item, Mapping)],
            logs=[dict(item) for item in (data.get("logs") or []) if isinstance(item, Mapping)],
            unrealized_pnl=_to_float(data.get("unrealized_pnl")),
            open_position_side=_to_text(data.get("open_position_side")),
            open_position_entry=_to_float(data.get("open_position_entry")),
            open_position_qty=_to_float(data.get("open_position_qty")),
            open_position_sl=_to_float(data.get("open_position_sl")),
            open_position_tp=_to_float(data.get("open_position_tp")),
            open_positions=[dict(item) for item in (data.get("open_positions") or []) if isinstance(item, Mapping)],
            llm_mode=_to_text(data.get("llm_mode")),
            llm_trade_confirmator=dict(data.get("llm_trade_confirmator", {}) or {}),
        )


class TerminalDashboard:
    def __init__(self) -> None:
        self.console = Console()

    def render(self, snapshot: DashboardSnapshot | Mapping[str, Any]) -> Any:
        try:
            snap = DashboardSnapshot.from_mapping(snapshot) if isinstance(snapshot, Mapping) else snapshot

            header = Table.grid(expand=True)
            header.add_column(ratio=4)
            header.add_column(ratio=2, justify="right")
            header.add_row(Text("Reco Trading Terminal TUI", style="bold bright_cyan"), _status_badge(snap.state))
            header.add_row(Text(f"{snap.pair or '-'}  •  {snap.timeframe or '-'}", style="cyan"), _signal_badge(snap.signal))

            status = Table.grid(expand=True)
            status.add_column()
            status.add_column()
            status.add_row("Price", f"[bold]{_fmt_num(snap.price, 2)}[/bold]")
            status.add_row("Spread", _fmt_num(snap.spread, 6))
            status.add_row("Trend", _trend_badge(snap.trend))
            status.add_row("ADX", _fmt_num(snap.adx, 2))
            status.add_row("Signal", _signal_badge(snap.signal))
            status.add_row("Confidence", _confidence_bar(snap.confidence))
            status.add_row("Volatility", snap.volatility_regime or "-")
            status.add_row("Order Flow", snap.order_flow or "-")
            status.add_row("Cooldown", snap.cooldown or "-")

            portfolio = Table.grid(expand=True)
            portfolio.add_column()
            portfolio.add_column()
            portfolio.add_row("Balance", f"{_fmt_num(snap.balance, 4)} USDT")
            portfolio.add_row("Equity", f"{_fmt_num(snap.equity, 4)} USDT")
            portfolio.add_row("Operable Capital", f"{_fmt_num(snap.operable_capital_usdt, 4)} USDT")
            portfolio.add_row("Daily PnL", _styled_pnl(snap.daily_pnl))
            portfolio.add_row("Unrealized PnL", _styled_pnl(snap.unrealized_pnl))
            portfolio.add_row("Trades Today", str(snap.trades_today))
            portfolio.add_row("Win Rate", _fmt_pct(snap.win_rate))
            portfolio.add_row("Last Trade", snap.last_trade or "-")

            live_trades = Table(title="Live Trades", expand=True, box=ROUNDED)
            live_trades.add_column("Side", justify="center")
            live_trades.add_column("Qty", justify="right")
            live_trades.add_column("Entry", justify="right")
            live_trades.add_column("Mark", justify="right")
            live_trades.add_column("PnL", justify="right")
            live_trades.add_column("SL/TP", justify="right")
            if snap.open_positions:
                for position in snap.open_positions[:4]:
                    live_trades.add_row(
                        _signal_value_style(str(position.get("side", "-")).upper()),
                        _fmt_num(_to_float(position.get("quantity")), 6),
                        _fmt_num(_to_float(position.get("entry_price")), 4),
                        _fmt_num(snap.price, 4),
                        _styled_pnl(_to_float(position.get("unrealized_pnl"))),
                        f"{_fmt_num(_to_float(position.get('stop_loss')), 4)} / {_fmt_num(_to_float(position.get('take_profit')), 4)}",
                    )
            elif snap.open_position_side:
                live_trades.add_row(
                    _signal_value_style((snap.open_position_side or "-").upper()),
                    _fmt_num(snap.open_position_qty, 6),
                    _fmt_num(snap.open_position_entry, 4),
                    _fmt_num(snap.price, 4),
                    _styled_pnl(snap.unrealized_pnl),
                    f"{_fmt_num(snap.open_position_sl, 4)} / {_fmt_num(snap.open_position_tp, 4)}",
                )
            else:
                live_trades.add_row("-", "-", "-", "-", "[dim]No active trade[/dim]", "-")

            llm_gate = Table(title="LLM Final Gate", expand=True, box=ROUNDED)
            llm_gate.add_column("Field")
            llm_gate.add_column("Value")
            llm_gate.add_row("Mode", (snap.llm_mode or "base").upper())
            llm_gate.add_row("Analyzed", str(snap.llm_trade_confirmator.get("total_analyzed", 0)))
            llm_gate.add_row("Confirmed", str(snap.llm_trade_confirmator.get("confirmed", 0)))
            llm_gate.add_row("Rejected", str(snap.llm_trade_confirmator.get("rejected", 0)))
            llm_gate.add_row("Avg Latency", f"{_fmt_num(_to_float(snap.llm_trade_confirmator.get('avg_analysis_time_ms')), 2)} ms")

            event_log = Table(title="Execution Feed", expand=True, box=ROUNDED)
            event_log.add_column("T", width=8, no_wrap=True)
            event_log.add_column("L", width=7, no_wrap=True)
            event_log.add_column("Event", overflow="fold")
            for log in (snap.logs or [])[-8:]:
                level = str(log.get("level", "INFO")).upper()
                event_log.add_row(str(log.get("time", "--:--:--")), _log_level_badge(level), str(log.get("message", ""))[:200])

            decision = Table(title="Decision Trace", expand=True, box=SIMPLE_HEAVY)
            decision.add_column("Field")
            decision.add_column("Value")
            for factor, score in (snap.decision_trace.get("factor_scores") or {}).items():
                decision.add_row(f"factor:{factor}", f"{float(score):+.3f}")
            for gate, value in snap.decision_gating.items():
                decision.add_row(f"gate:{gate}", str(value))
            decision.add_row("reason", snap.decision_reason or "-")

            width = self.console.size.width
            layout = Layout(name="root")
            layout.split_column(
                Layout(Panel(header, border_style="bright_blue"), ratio=1),
                Layout(name="main", ratio=13),
                Layout(Panel(Align.center(Text("Press Ctrl+C to stop • Reco Trading TUI Runtime", style="dim white")), border_style="grey37"), ratio=1),
            )

            if width < 110:
                # Compact mode for Android/mobile terminals and narrow SSH sessions.
                layout["main"].split_column(
                    Layout(Panel(status, title="Market", border_style="cyan"), ratio=3),
                    Layout(Panel(portfolio, title="Portfolio", border_style="green"), ratio=3),
                    Layout(Panel(live_trades, border_style="bright_cyan"), ratio=4),
                    Layout(Panel(llm_gate, border_style="bright_yellow"), ratio=3),
                    Layout(Panel(event_log, border_style="white"), ratio=4),
                )
            else:
                layout["main"].split_row(
                    Layout(Panel(status, title="Market", border_style="cyan"), ratio=3),
                    Layout(name="center", ratio=5),
                    Layout(Panel(decision, title="Decision", border_style="yellow"), ratio=4),
                )
                layout["main"]["center"].split_column(
                    Layout(Panel(portfolio, title="Portfolio", border_style="green"), ratio=3),
                    Layout(Panel(live_trades, border_style="bright_cyan"), ratio=4),
                    Layout(Panel(llm_gate, border_style="bright_yellow"), ratio=3),
                    Layout(Panel(event_log, border_style="white"), ratio=4),
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


def _status_badge(state: str | None) -> str:
    value = (state or "UNKNOWN").upper()
    if value in {"RUNNING", "READY"}:
        return f"[bold black on green] {value} [/bold black on green]"
    if value in {"ERROR", "CRASHED"}:
        return f"[bold white on red] {value} [/bold white on red]"
    if value in {"PAUSED", "USER_PAUSED"}:
        return f"[bold black on yellow] {value} [/bold black on yellow]"
    return f"[bold black on cyan] {value} [/bold black on cyan]"


def _signal_badge(signal: str | None) -> str:
    value = (signal or "HOLD").upper()
    if value == "BUY":
        return "[bold black on green] BUY [/bold black on green]"
    if value == "SELL":
        return "[bold white on red] SELL [/bold white on red]"
    return "[bold black on yellow] HOLD [/bold black on yellow]"


def _trend_badge(trend: str | None) -> str:
    value = (trend or "NEUTRAL").upper()
    if value == "BULLISH":
        return "[green]BULLISH[/green]"
    if value == "BEARISH":
        return "[red]BEARISH[/red]"
    return "[yellow]NEUTRAL[/yellow]"


def _signal_value_style(value: str) -> str:
    upper = (value or "").upper()
    if "BUY" in upper or "BULL" in upper:
        return f"[green]{value}[/green]"
    if "SELL" in upper or "BEAR" in upper:
        return f"[red]{value}[/red]"
    return f"[yellow]{value}[/yellow]"


def _confidence_bar(value: float | None, width: int = 18) -> str:
    clamped = _normalize_confidence(value)
    filled = int(clamped * width)
    empty = width - filled
    color = "green" if clamped >= 0.7 else "yellow" if clamped >= 0.45 else "red"
    return f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {clamped * 100:5.1f}%"


def _normalize_confidence(value: float | None) -> float:
    if value is None:
        return 0.0
    normalized = float(value)
    if normalized > 1.0:
        normalized = normalized / 100.0 if normalized <= 100.0 else 1.0
    return max(0.0, min(1.0, normalized))


def _styled_pnl(value: float | None) -> str:
    if value is None:
        return "-"
    if value > 0:
        return f"[bold green]+{value:.4f} USDT[/bold green]"
    if value < 0:
        return f"[bold red]{value:.4f} USDT[/bold red]"
    return "[white]0.0000 USDT[/white]"


def _log_level_badge(level: str) -> str:
    if level in {"ERROR", "CRITICAL"}:
        return "[bold white on red] ERROR [/bold white on red]"
    if level in {"WARNING", "WARN"}:
        return "[bold black on yellow] WARN [/bold black on yellow]"
    if level == "DEBUG":
        return "[bold black on white] DEBUG [/bold black on white]"
    return "[bold black on green] INFO [/bold black on green]"

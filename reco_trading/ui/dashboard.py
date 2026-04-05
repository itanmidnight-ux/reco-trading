from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

from rich.align import Align
from rich.box import ROUNDED, SIMPLE_HEAD, MINIMAL_DOUBLE_HEAD
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


@dataclass
class DashboardSnapshot:
    state: str = "INITIALIZING"
    pair: str = ""
    timeframe: str = ""
    price: float | None = None
    spread: float | None = None
    bid: float | None = None
    ask: float | None = None
    trend: str | None = None
    adx: float | None = None
    rsi: float | None = None
    volatility_regime: str | None = None
    order_flow: str | None = None
    signal: str | None = None
    confidence: float | None = None
    balance: float | None = None
    equity: float | None = None
    daily_pnl: float | None = None
    session_pnl: float | None = None
    operable_capital_usdt: float | None = None
    capital_profile: str | None = None
    trades_today: int = 0
    win_rate: float | None = None
    last_trade: str | None = None
    cooldown: str | None = None
    consecutive_losses: int = 0
    signals: dict[str, str] = field(default_factory=dict)
    decision_trace: dict[str, Any] = field(default_factory=dict)
    decision_gating: dict[str, Any] = field(default_factory=dict)
    decision_reason: str | None = None
    autonomous_filters: dict[str, Any] = field(default_factory=dict)
    autonomous_market_condition: str | None = None
    api_latency_p95_ms: float | None = None
    stale_market_data_ratio: float | None = None
    exchange_reconnections: int = 0
    circuit_breaker_trips: int = 0
    database_status: str | None = None
    exchange_status: str | None = None
    exit_intelligence_score: float | None = None
    exit_intelligence_reason: str | None = None
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
    session_recommendation: str | None = None
    auto_improve_win_rate: float | None = None
    auto_improve_total_trades: int = 0
    investment_mode: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DashboardSnapshot":
        return cls(
            state=str(data.get("status", "INITIALIZING")),
            pair=str(data.get("pair", "")),
            timeframe=str(data.get("timeframe", "")),
            price=_to_float(data.get("price")),
            spread=_to_float(data.get("spread")),
            bid=_to_float(data.get("bid")),
            ask=_to_float(data.get("ask")),
            trend=_to_text(data.get("trend")),
            adx=_to_float(data.get("adx")),
            rsi=_to_float(data.get("rsi")),
            volatility_regime=_to_text(data.get("volatility_regime")),
            order_flow=_to_text(data.get("order_flow")),
            signal=_to_text(data.get("signal")),
            confidence=_to_float(data.get("confidence")),
            balance=_to_float(data.get("balance")),
            equity=_to_float(data.get("equity")),
            daily_pnl=_to_float(data.get("daily_pnl")),
            session_pnl=_to_float(data.get("session_pnl")),
            operable_capital_usdt=_to_float(data.get("operable_capital_usdt")),
            capital_profile=_to_text(data.get("capital_profile")),
            trades_today=int(data.get("trades_today", 0) or 0),
            win_rate=_to_float(data.get("win_rate")),
            last_trade=_to_text(data.get("last_trade")),
            cooldown=_to_text(data.get("cooldown")),
            consecutive_losses=int(data.get("consecutive_losses", 0) or 0),
            signals=dict(data.get("signals", {}) or {}),
            decision_trace=dict(data.get("decision_trace", {}) or {}),
            decision_gating=dict(data.get("decision_gating", {}) or {}),
            decision_reason=_to_text(data.get("decision_reason")),
            autonomous_filters=dict(data.get("autonomous_filters", {}) or {}),
            autonomous_market_condition=_to_text(data.get("autonomous_market_condition")),
            api_latency_p95_ms=_to_float(data.get("api_latency_p95_ms")),
            stale_market_data_ratio=_to_float(data.get("stale_market_data_ratio")),
            exchange_reconnections=int(data.get("exchange_reconnections", 0) or 0),
            circuit_breaker_trips=int(data.get("circuit_breaker_trips", 0) or 0),
            database_status=_to_text(data.get("database_status")),
            exchange_status=_to_text(data.get("exchange_status")),
            exit_intelligence_score=_to_float(data.get("exit_intelligence_score")),
            exit_intelligence_reason=_to_text(data.get("exit_intelligence_reason")),
            logs=[dict(i) for i in (data.get("logs") or []) if isinstance(i, Mapping)],
            unrealized_pnl=_to_float(data.get("unrealized_pnl")),
            open_position_side=_to_text(data.get("open_position_side")),
            open_position_entry=_to_float(data.get("open_position_entry")),
            open_position_qty=_to_float(data.get("open_position_qty")),
            open_position_sl=_to_float(data.get("open_position_sl")),
            open_position_tp=_to_float(data.get("open_position_tp")),
            open_positions=[dict(i) for i in (data.get("open_positions") or []) if isinstance(i, Mapping)],
            llm_mode=_to_text(data.get("llm_mode")),
            llm_trade_confirmator=dict(data.get("llm_trade_confirmator", {}) or {}),
            session_recommendation=_to_text(data.get("session_recommendation")),
            auto_improve_win_rate=_to_float(data.get("auto_improve_win_rate")),
            auto_improve_total_trades=int(data.get("auto_improve_total_trades", 0) or 0),
            investment_mode=_to_text(data.get("investment_mode")),
        )


class TerminalDashboard:
    """
    Rich TUI Dashboard — Full rebuild.
    Adapts between wide (>= 110 cols) and compact mode.
    """

    def __init__(self) -> None:
        self.console = Console()
        self._width_cache: int = 0

    def render(self, snapshot: DashboardSnapshot | Mapping[str, Any]) -> Any:
        try:
            snap = DashboardSnapshot.from_mapping(snapshot) if isinstance(snapshot, Mapping) else snapshot
            width = self.console.size.width

            # ── HEADER ──────────────────────────────────────────────
            header_grid = Table.grid(expand=True)
            header_grid.add_column(ratio=5)
            header_grid.add_column(ratio=1, justify="right")
            header_grid.add_row(
                Text.assemble(
                    ("◈ RECO TRADING ", "bold bright_cyan"),
                    ("TUI", "bold blue"),
                    ("  │  ", "dim"),
                    (snap.pair or "—", "bold white"),
                    ("  ", ""),
                    (f"[{snap.timeframe or '—'}]", "dim cyan"),
                ),
                _status_badge(snap.state),
            )
            header_grid.add_row(
                Text.assemble(
                    _signal_badge(snap.signal),
                    ("  conf:", "dim"),
                    (f" {(snap.confidence or 0)*100:.1f}%", "bold yellow" if (snap.confidence or 0) < 0.5 else "bold green"),
                    ("  │  market: ", "dim"),
                    (snap.autonomous_market_condition or "—", "cyan"),
                    ("  │  mode: ", "dim"),
                    (snap.investment_mode or "—", "magenta"),
                ),
                Text(f"latency {_fmt(snap.api_latency_p95_ms, 1)}ms", style="dim"),
            )

            # ── MARKET PANEL ────────────────────────────────────────
            market = Table.grid(expand=True, padding=(0, 1))
            market.add_column(style="dim", min_width=14)
            market.add_column(style="bold white")
            market.add_row("Price", f"[bold bright_white]{_fmt(snap.price, 2)}[/]  USDT")
            market.add_row("Bid / Ask", f"[green]{_fmt(snap.bid, 4)}[/] / [red]{_fmt(snap.ask, 4)}[/]")
            market.add_row("Spread", f"[yellow]{_fmt(snap.spread, 6)}[/]")
            market.add_row("ADX", _adx_styled(snap.adx))
            market.add_row("RSI", _rsi_styled(snap.rsi))
            market.add_row("Trend", _trend_badge(snap.trend))
            market.add_row("Regime", _regime_badge(snap.volatility_regime))
            market.add_row("Order Flow", _flow_badge(snap.order_flow))
            market.add_row("Cooldown", Text(snap.cooldown or "READY", style="green" if (snap.cooldown or "READY") in ("READY", "ready") else "yellow"))

            # ── SIGNALS PANEL ────────────────────────────────────────
            sigs = snap.signals or {}
            sig_grid = Table(box=MINIMAL_DOUBLE_HEAD, expand=True, show_header=True, padding=(0, 1))
            sig_grid.add_column("Signal", style="dim", width=12)
            sig_grid.add_column("5m", justify="center", width=8)
            sig_grid.add_column("15m", justify="center", width=8)
            for name, key in [("Trend","trend"),("Momentum","momentum"),("Volume","volume"),
                               ("Volatility","volatility"),("Structure","structure"),("OrderFlow","order_flow")]:
                val = sigs.get(key, "NEUTRAL")
                sig_grid.add_row(name, _signal_cell(val), "—")

            # Confidence bar
            conf_pct = int((snap.confidence or 0) * 20)
            conf_bar = "█" * conf_pct + "░" * (20 - conf_pct)
            conf_color = "green" if (snap.confidence or 0) >= 0.65 else "yellow" if (snap.confidence or 0) >= 0.45 else "red"
            conf_row = Text.assemble(("Conf: ", "dim"), (conf_bar, conf_color), (f" {(snap.confidence or 0)*100:.1f}%", f"bold {conf_color}"))

            # ── PORTFOLIO PANEL ─────────────────────────────────────
            portfolio = Table.grid(expand=True, padding=(0, 1))
            portfolio.add_column(style="dim", min_width=18)
            portfolio.add_column(style="bold white")
            portfolio.add_row("Balance", f"[bold]{_fmt(snap.balance, 4)}[/] USDT")
            portfolio.add_row("Equity", f"[bold]{_fmt(snap.equity, 4)}[/] USDT")
            portfolio.add_row("Operable", f"[cyan]{_fmt(snap.operable_capital_usdt, 4)}[/] USDT")
            pnl = snap.daily_pnl or snap.session_pnl or 0.0
            portfolio.add_row("Session PnL", _styled_pnl(pnl))
            portfolio.add_row("Unrealized", _styled_pnl(snap.unrealized_pnl))
            portfolio.add_row("Win Rate", _fmt_pct(snap.win_rate))
            portfolio.add_row("Trades Today", str(snap.trades_today))
            portfolio.add_row("Consec. Losses", _losses_styled(snap.consecutive_losses))
            portfolio.add_row("Capital Profile", Text(snap.capital_profile or "—", style="magenta"))
            portfolio.add_row("Session Rec.", Text(snap.session_recommendation or "—",
                style="green" if snap.session_recommendation == "TRADE" else "yellow" if snap.session_recommendation == "CAUTION" else "red"))

            # ── OPEN POSITION ────────────────────────────────────────
            pos_table = Table(box=ROUNDED, expand=True, padding=(0, 1))
            pos_table.add_column("Side", justify="center", width=6)
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Entry", justify="right")
            pos_table.add_column("Mark", justify="right")
            pos_table.add_column("PnL", justify="right")
            pos_table.add_column("SL", justify="right")
            pos_table.add_column("TP", justify="right")

            positions = snap.open_positions or []
            if not positions and snap.open_position_side:
                positions = [{"side": snap.open_position_side, "quantity": snap.open_position_qty,
                              "entry_price": snap.open_position_entry, "stop_loss": snap.open_position_sl,
                              "take_profit": snap.open_position_tp, "unrealized_pnl": snap.unrealized_pnl}]
            if positions:
                for pos in positions[:3]:
                    side = str(pos.get("side", "")).upper()
                    side_style = "bold green" if side == "BUY" else "bold red"
                    pos_table.add_row(
                        Text(side, style=side_style),
                        _fmt(pos.get("quantity"), 6),
                        _fmt(pos.get("entry_price"), 4),
                        _fmt(snap.price, 4),
                        _styled_pnl(_to_float(pos.get("unrealized_pnl"))),
                        Text(_fmt(pos.get("stop_loss"), 4), style="red"),
                        Text(_fmt(pos.get("take_profit"), 4), style="green"),
                    )
            else:
                pos_table.add_row(Text("—", style="dim"), "—", "—", "—", Text("No position", style="dim"), "—", "—")

            # ── LLM GATE PANEL ───────────────────────────────────────
            llm = snap.llm_trade_confirmator or {}
            llm_mode = (snap.llm_mode or "base").upper()
            llm_health = llm.get("local_endpoint_healthy")
            health_icon = "🟢" if llm_health is True else "🔴" if llm_health is False else "⚪"
            llm_table = Table.grid(expand=True, padding=(0, 1))
            llm_table.add_column(style="dim", min_width=16)
            llm_table.add_column(style="bold white")
            llm_table.add_row("Mode", Text(llm_mode, style="bold magenta"))
            llm_table.add_row("Ollama", Text(health_icon + (" Online" if llm_health else " Offline" if llm_health is False else " —")))
            llm_table.add_row("Analyzed", str(llm.get("total_analyzed", 0)))
            llm_table.add_row("Confirmed", Text(str(llm.get("confirmed", 0)), style="green"))
            llm_table.add_row("Rejected", Text(str(llm.get("rejected", 0)), style="red"))
            rate = llm.get("confirmation_rate", 0)
            llm_table.add_row("Rate", Text(f"{_fmt_float(rate, 1)}%", style="cyan"))
            llm_table.add_row("Avg Latency", f"{_fmt_float(llm.get('avg_analysis_time_ms', 0), 1)} ms")

            # ── FILTER STATUS ─────────────────────────────────────────
            af = snap.autonomous_filters
            filter_table = Table.grid(expand=True, padding=(0, 1))
            filter_table.add_column(style="dim", min_width=16)
            filter_table.add_column(style="bold cyan")
            if af:
                filter_table.add_row("ADX ≥", _fmt_float(af.get("adx_threshold"), 1))
                filter_table.add_row("RSI Buy ≥", _fmt_float(af.get("rsi_buy_threshold"), 1))
                filter_table.add_row("RSI Sell ≤", _fmt_float(af.get("rsi_sell_threshold"), 1))
                filter_table.add_row("Min Conf", f"{_fmt_float((af.get('min_confidence') or 0) * 100, 1)}%")
                filter_table.add_row("Vol Buy ≥", _fmt_float(af.get("volume_buy_threshold"), 2))
            else:
                filter_table.add_row("Filters", "loading...")

            # ── EVENT LOG ────────────────────────────────────────────
            log_table = Table(box=None, expand=True, padding=(0, 0), show_header=False)
            log_table.add_column("T", width=8, no_wrap=True, style="dim")
            log_table.add_column("L", width=8, no_wrap=True)
            log_table.add_column("Msg", overflow="fold")
            for log in (snap.logs or [])[-10:]:
                level = str(log.get("level", "INFO")).upper()
                style = {"WARNING": "yellow", "ERROR": "bold red", "DEBUG": "dim"}.get(level, "white")
                log_table.add_row(
                    str(log.get("time", "--:--"))[:8],
                    Text(level[:4], style=style),
                    Text(str(log.get("message", ""))[:150], style=style if level in ("WARNING","ERROR") else "dim white"),
                )

            # ── SYSTEM STATUS BAR ─────────────────────────────────────
            sys_status = Table.grid(expand=True)
            sys_status.add_column(ratio=1)
            sys_status.add_column(ratio=1)
            sys_status.add_column(ratio=1)
            sys_status.add_column(ratio=1)
            exch_style = "green" if (snap.exchange_status or "").upper() in ("CONNECTED", "OK") else "red"
            db_style = "green" if (snap.database_status or "").upper() in ("CONNECTED", "SQLITE_FALLBACK") else "red"
            sys_status.add_row(
                Text.assemble(("EX:", "dim"), (f" {snap.exchange_status or '—'}", exch_style)),
                Text.assemble(("DB:", "dim"), (f" {snap.database_status or '—'}", db_style)),
                Text.assemble(("CBT:", "dim"), (f" {snap.circuit_breaker_trips}", "yellow" if snap.circuit_breaker_trips > 0 else "green")),
                Text.assemble(("RECONN:", "dim"), (f" {snap.exchange_reconnections}", "yellow" if snap.exchange_reconnections > 0 else "green")),
            )

            footer = Text.assemble(
                ("◈ Reco Trading TUI  ", "dim"),
                ("│  ", "dim"),
                ("Ctrl+C", "bold white"),
                (" to stop  ", "dim"),
                ("│  ", "dim"),
                ("Web: ", "dim"),
                ("http://localhost:9000", "bright_cyan"),
            )

            # ── LAYOUT ASSEMBLY ───────────────────────────────────────
            layout = Layout(name="root")

            if width < 110:
                # Compact / mobile SSH mode
                layout.split_column(
                    Layout(Panel(header_grid, border_style="bright_blue", padding=(0, 1)), size=4),
                    Layout(name="body", ratio=1),
                    Layout(Panel(sys_status, border_style="grey27", padding=(0, 1)), size=3),
                    Layout(Panel(footer, border_style="grey27"), size=3),
                )
                layout["body"].split_column(
                    Layout(Panel(Group(market), title="Market", border_style="cyan"), ratio=3),
                    Layout(Panel(conf_row, border_style="blue"), size=3),
                    Layout(Panel(portfolio, title="Portfolio", border_style="green"), ratio=4),
                    Layout(Panel(pos_table, title="Position", border_style="bright_cyan"), ratio=3),
                    Layout(Panel(llm_table, title="LLM Gate", border_style="magenta"), ratio=3),
                    Layout(Panel(log_table, title="Feed", border_style="white"), ratio=4),
                )
            else:
                # Wide mode: 3-column layout
                layout.split_column(
                    Layout(Panel(header_grid, border_style="bright_blue", padding=(0, 1)), size=4),
                    Layout(name="body", ratio=1),
                    Layout(Panel(sys_status, border_style="grey27", padding=(0, 0)), size=3),
                    Layout(Panel(Align.center(footer), border_style="grey27"), size=3),
                )
                layout["body"].split_row(
                    Layout(name="left", ratio=3),
                    Layout(name="center", ratio=4),
                    Layout(name="right", ratio=3),
                )
                layout["body"]["left"].split_column(
                    Layout(Panel(market, title="📈 Market", border_style="cyan"), ratio=5),
                    Layout(Panel(sig_grid, title="📡 Signals", border_style="blue"), ratio=5),
                )
                layout["body"]["center"].split_column(
                    Layout(Panel(Group(conf_row, portfolio), title="💼 Portfolio", border_style="green"), ratio=5),
                    Layout(Panel(pos_table, title="🔵 Open Position", border_style="bright_cyan"), ratio=3),
                    Layout(Panel(log_table, title="📄 Feed", border_style="white"), ratio=4),
                )
                layout["body"]["right"].split_column(
                    Layout(Panel(llm_table, title="🤖 LLM Gate", border_style="magenta"), ratio=4),
                    Layout(Panel(filter_table, title="🔧 Active Filters", border_style="yellow"), ratio=4),
                    Layout(Panel(_build_decision_panel(snap), title="🔍 Decision", border_style="yellow"), ratio=4),
                )

            return Group(layout)

        except Exception as exc:
            return Panel(
                Text.assemble(
                    ("Dashboard render error:\n", "bold red"),
                    (str(exc), "white"),
                ),
                title="Reco Trading TUI",
                border_style="red",
            )


def _build_decision_panel(snap: DashboardSnapshot) -> Table:
    t = Table.grid(expand=True, padding=(0, 1))
    t.add_column(style="dim", min_width=14)
    t.add_column(style="bold white")
    dt = snap.decision_trace or {}
    factors = dt.get("factor_scores") or {}
    gating = snap.decision_gating or {}
    for k, v in list(factors.items())[:5]:
        vf = _to_float(v) or 0.0
        style = "green" if vf > 0 else "red" if vf < 0 else "dim"
        t.add_row(f"▸ {k}", Text(f"{vf:+.3f}", style=style))
    for k, v in list(gating.items())[:3]:
        t.add_row(f"• {k}", Text(str(v)[:20], style="cyan"))
    t.add_row("reason", Text((snap.decision_reason or "—")[:40], style="white"))
    return t


# ── HELPER FORMATTERS ─────────────────────────────────────────────────────────

def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

def _to_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return str(value) if str(value).strip() else None

def _fmt(value: Any, dec: int = 2) -> str:
    f = _to_float(value)
    return f"{f:.{dec}f}" if f is not None else "—"

def _fmt_float(value: Any, dec: int = 2) -> str:
    f = _to_float(value)
    return f"{f:.{dec}f}" if f is not None else "—"

def _fmt_pct(value: Any) -> str:
    f = _to_float(value)
    return f"{f*100:.1f}%" if f is not None else "—"

def _styled_pnl(value: Any) -> Text:
    f = _to_float(value)
    if f is None:
        return Text("—", style="dim")
    sign = "+" if f > 0 else ""
    style = "bold green" if f > 0 else "bold red" if f < 0 else "dim"
    return Text(f"{sign}{f:.4f} USDT", style=style)

def _losses_styled(n: int) -> Text:
    if n == 0:
        return Text("0", style="green")
    if n <= 2:
        return Text(str(n), style="yellow")
    return Text(str(n), style="bold red")

def _adx_styled(adx: float | None) -> Text:
    f = _to_float(adx)
    if f is None:
        return Text("—", style="dim")
    style = "bold green" if f >= 25 else "yellow" if f >= 15 else "red"
    return Text(f"{f:.2f}", style=style)

def _rsi_styled(rsi: float | None) -> Text:
    f = _to_float(rsi)
    if f is None:
        return Text("—", style="dim")
    style = "red" if f > 70 else "green" if f < 30 else "cyan" if f > 55 else "white"
    return Text(f"{f:.1f}", style=style)

def _status_badge(state: str) -> Text:
    styles = {
        "RUNNING": ("RUNNING", "bold green"),
        "PAUSED": ("PAUSED", "bold yellow"),
        "INITIALIZING": ("INIT", "bold blue"),
        "WAITING_MARKET_DATA": ("WAITING", "blue"),
        "CONNECTING_EXCHANGE": ("CONNECTING", "blue"),
        "POSITION_OPEN": ("IN TRADE", "bold bright_green"),
        "COOLDOWN": ("COOLDOWN", "yellow"),
        "ERROR": ("ERROR", "bold red"),
    }
    label, style = styles.get(state.upper(), (state[:10], "white"))
    return Text(f"[{label}]", style=style)

def _signal_badge(signal: str | None) -> Text:
    s = (signal or "HOLD").upper()
    styles = {"BUY": "bold green", "SELL": "bold red", "HOLD": "dim", "NEUTRAL": "dim"}
    return Text(s, style=styles.get(s, "white"))

def _signal_cell(val: str) -> Text:
    v = (val or "NEUTRAL").upper()
    styles = {"BUY": "bold green", "SELL": "bold red", "NEUTRAL": "dim", "HOLD": "dim"}
    return Text(v, style=styles.get(v, "white"))

def _trend_badge(trend: str | None) -> Text:
    t = (trend or "NEUTRAL").upper()
    styles = {"BULLISH": "bold green", "BEARISH": "bold red", "BUY": "green", "SELL": "red", "NEUTRAL": "dim"}
    return Text(t, style=styles.get(t, "white"))

def _regime_badge(regime: str | None) -> Text:
    r = (regime or "NORMAL").upper()
    styles = {
        "LOW_VOLATILITY": "dim cyan",
        "NORMAL_VOLATILITY": "green",
        "HIGH_VOLATILITY": "bold yellow",
    }
    return Text(r.replace("_", " "), style=styles.get(r, "white"))

def _flow_badge(flow: str | None) -> Text:
    f = (flow or "NEUTRAL").upper()
    styles = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "dim"}
    return Text(f, style=styles.get(f, "white"))



def _confidence_bar(value: Any) -> str:
    """Backward-compatible helper used by tests and legacy callers."""
    f = _to_float(value)
    if f is None:
        pct = 0.0
    else:
        pct = f * 100 if f <= 1 else f
    pct = max(0.0, min(100.0, pct))
    bars = int(round((pct / 100.0) * 20))
    return f"{'█' * bars}{'░' * (20 - bars)} {pct:.1f}%"

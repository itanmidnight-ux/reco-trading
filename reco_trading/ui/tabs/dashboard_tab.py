from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from reco_trading.ui.chart_widget import CandlestickChartWidget
from reco_trading.ui.state_manager import StateManager
from reco_trading.ui.widgets.stat_card import StatCard



class AnimatedButton(QPushButton):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(label, parent)
        self._base_min_width = 118
        self.setMinimumWidth(self._base_min_width)
        self._hover_anim = QPropertyAnimation(self, b"minimumWidth", self)
        self._hover_anim.setDuration(140)
        self._hover_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._animate_width(self._base_min_width + 10)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._animate_width(self._base_min_width)
        super().leaveEvent(event)

    def _animate_width(self, target: int) -> None:
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self.minimumWidth())
        self._hover_anim.setEndValue(target)
        self._hover_anim.start()

class DashboardTab(QWidget):
    def __init__(self, state_manager: StateManager | None = None) -> None:
        super().__init__()
        self.state_manager = state_manager
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Executive Dashboard")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        self.top_bar = QLabel("BTC/USDT | - | NEUTRAL | INITIALIZING")
        self.top_bar.setObjectName("statusRibbon")
        root.addWidget(self.top_bar)

        self.hero_panel = self._panel()
        hero_layout = QGridLayout(self.hero_panel)
        hero_layout.setContentsMargins(10, 10, 10, 10)
        hero_layout.setSpacing(10)
        self.hero_cards = {
            "price": StatCard("Market Price"),
            "signal": StatCard("Signal Quality"),
            "daily_pnl": StatCard("Session PnL"),
            "exposure": StatCard("Current Exposure"),
        }
        for i, card in enumerate(self.hero_cards.values()):
            hero_layout.addWidget(card, 0, i)
        root.addWidget(self.hero_panel)

        self.position_panel = self._panel()
        position_layout = QGridLayout(self.position_panel)
        position_layout.setContentsMargins(10, 10, 10, 10)
        position_layout.setSpacing(8)
        position_layout.addWidget(self._title("Open Position"), 0, 0, 1, 4)
        self.pos_cards = {
            "side": StatCard("Side", compact=True),
            "entry": StatCard("Entry", compact=True),
            "current": StatCard("Current", compact=True),
            "pnl": StatCard("Unrealized PnL", compact=True),
            "sl": StatCard("Stop Loss", compact=True),
            "tp": StatCard("Take Profit", compact=True),
            "size": StatCard("Size", compact=True),
        }
        for i, card in enumerate(self.pos_cards.values()):
            position_layout.addWidget(card, 1 + i // 4, i % 4)
        self.pnl_bar = QProgressBar()
        self.pnl_bar.setRange(-100, 100)
        self.pnl_bar.setValue(0)
        self.pnl_bar.setTextVisible(False)
        position_layout.addWidget(self.pnl_bar, 2, 0, 1, 4)
        root.addWidget(self.position_panel)
        self.position_panel.setVisible(False)

        controls = self._build_controls()
        root.addWidget(controls)

        body = QGridLayout()
        body.setSpacing(10)
        root.addLayout(body)

        self.market_panel = self._panel()
        self.market_cards = {
            "spread": StatCard("Spread", compact=True),
            "adx": StatCard("ADX", compact=True),
            "volatility_regime": StatCard("Volatility Regime", compact=True),
            "order_flow": StatCard("Order Flow", compact=True),
        }
        market_layout = QGridLayout(self.market_panel)
        market_layout.setContentsMargins(10, 10, 10, 10)
        market_layout.addWidget(self._title("Market Information"), 0, 0, 1, 2)
        for i, card in enumerate(self.market_cards.values()):
            market_layout.addWidget(card, (i // 2) + 1, i % 2)
        self.signal_badge = QLabel("NEUTRAL")
        self.confidence_label = QLabel("Confidence")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_anim = QPropertyAnimation(self.confidence_bar, b"value", self)
        self.confidence_anim.setDuration(300)
        self.confidence_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        market_layout.addWidget(self.signal_badge, 3, 0)
        market_layout.addWidget(self.confidence_label, 3, 1)
        market_layout.addWidget(self.confidence_bar, 4, 0, 1, 2)
        self.market_context = QLabel("Waiting for signal context")
        self.market_context.setObjectName("metricLabel")
        self.market_context.setWordWrap(True)
        market_layout.addWidget(self.market_context, 5, 0, 1, 2)

        self.account_panel = self._panel()
        self.account_cards = {
            "balance": StatCard("Balance", compact=True),
            "equity": StatCard("Equity", compact=True),
            "btc_balance": StatCard("BTC Balance", compact=True),
            "btc_value": StatCard("BTC Value", compact=True),
            "total_equity": StatCard("Total Equity", compact=True),
            "daily_pnl": StatCard("Daily PnL", compact=True),
            "trades_today": StatCard("Trades Today", compact=True),
            "win_rate": StatCard("Win Rate", compact=True),
        }
        account_layout = QGridLayout(self.account_panel)
        account_layout.setContentsMargins(10, 10, 10, 10)
        account_layout.addWidget(self._title("Account Performance"), 0, 0, 1, 2)
        for i, card in enumerate(self.account_cards.values()):
            account_layout.addWidget(card, (i // 2) + 1, i % 2)

        self.activity_panel = self._panel()
        activity_layout = QVBoxLayout(self.activity_panel)
        activity_layout.setContentsMargins(10, 10, 10, 10)
        activity_layout.addWidget(self._title("Bot Activity"))
        self.feed = QLabel("[--:--] Waiting for events")
        self.feed.setWordWrap(True)
        self.feed.setObjectName("smallMetricValue")
        activity_layout.addWidget(self.feed)
        self.feed_meta = QLabel("No alerts yet")
        self.feed_meta.setObjectName("metricLabel")
        activity_layout.addWidget(self.feed_meta)

        self.chart_panel = self._panel()
        chart_layout = QVBoxLayout(self.chart_panel)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_layout.addWidget(self._title("Realtime Chart"))
        self.chart = CandlestickChartWidget()
        chart_layout.addWidget(self.chart)

        body.addWidget(self.market_panel, 0, 0)
        body.addWidget(self.account_panel, 0, 1)
        body.addWidget(self.activity_panel, 1, 0)
        body.addWidget(self.chart_panel, 1, 1)

    def _build_controls(self) -> QFrame:
        panel = self._panel()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = self._title("Bot Controls")
        layout.addWidget(title)
        layout.addStretch()

        self.start_btn = AnimatedButton("Start Bot")
        self.pause_btn = AnimatedButton("Pause Bot")
        self.resume_btn = AnimatedButton("Resume Bot")
        self.emergency_btn = AnimatedButton("Emergency Stop")
        self.emergency_btn.setStyleSheet("QPushButton { background:#ea3943; color:#e6e8ee; }")
        self.close_active_trade_btn = AnimatedButton("CLOSE ACTIVE TRADE")
        self.close_active_trade_btn.setStyleSheet("QPushButton { background:#f0b90b; color:#111827; font-weight:700; }")
        self.close_active_trade_btn.setVisible(False)

        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.resume_btn)
        layout.addWidget(self.emergency_btn)
        layout.addWidget(self.close_active_trade_btn)

        if self.state_manager:
            self.start_btn.clicked.connect(self.state_manager.request_start)
            self.pause_btn.clicked.connect(self.state_manager.request_pause)
            self.resume_btn.clicked.connect(self.state_manager.request_resume)
            self.emergency_btn.clicked.connect(self.state_manager.request_emergency_stop)
            self.close_active_trade_btn.clicked.connect(self.state_manager.request_force_close)

        self._sync_control_buttons("INITIALIZING")
        return panel

    def _sync_control_buttons(self, status: str) -> None:
        normalized = str(status).lower()
        running_states = {
            "connecting_exchange",
            "syncing_symbol",
            "syncing_rules",
            "waiting_market_data",
            "analyzing_market",
            "signal_generated",
            "placing_order",
            "position_open",
            "cooldown",
        }
        paused_states = {"paused"}

        running = normalized in running_states
        paused = normalized in paused_states

        self.pause_btn.setVisible(running)
        self.start_btn.setVisible(not running and not paused)
        self.resume_btn.setVisible(paused)

    def _panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panelCard")
        return panel

    def _title(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("metricLabel")
        return label

    def update_state(self, state: dict[str, Any]) -> None:
        pair = state.get("pair", "BTC/USDT")
        price = _fmt_num(state.get("current_price", state.get("price")), 2)
        trend = str(state.get("trend", "NEUTRAL"))
        status = str(state.get("status", "-"))
        self._sync_control_buttons(status)
        self.close_active_trade_btn.setVisible(bool(state.get("has_open_position", False)))
        has_position = bool(state.get("has_open_position", False))
        self.position_panel.setVisible(has_position)
        if has_position:
            price_value = float(state.get("current_price", state.get("price", 0)) or 0)
            entry = float(state.get("open_position_entry") or 0)
            qty = float(state.get("open_position_qty") or 0)
            sl = float(state.get("open_position_sl") or 0)
            tp = float(state.get("open_position_tp") or 0)
            side_value = str(state.get("open_position_side") or "-").upper()
            upnl = float(state.get("unrealized_pnl") or 0)
            tone_pnl = "positive" if upnl >= 0 else "negative"
            self.pos_cards["side"].set_value(side_value, tone="positive" if side_value == "BUY" else "negative")
            self.pos_cards["entry"].set_value(_fmt_num(entry, 2))
            self.pos_cards["current"].set_value(_fmt_num(price_value, 2))
            self.pos_cards["pnl"].set_value(f"{upnl:+.4f} USDT", tone=tone_pnl)
            self.pos_cards["sl"].set_value(_fmt_num(sl, 2), tone="negative")
            self.pos_cards["tp"].set_value(_fmt_num(tp, 2), tone="positive")
            self.pos_cards["size"].set_value(_fmt_num(qty, 6))
            if tp != sl:
                pct = int(((price_value - sl) / (tp - sl)) * 100)
                pct = max(-100, min(100, pct))
                self.pnl_bar.setValue(pct)
                color = "#16c784" if pct > 50 else "#f0b90b" if pct > 0 else "#ea3943"
                self.pnl_bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
            else:
                self.pnl_bar.setValue(0)
        else:
            self.pnl_bar.setValue(0)

        signal = str(state.get("signal", "NEUTRAL")).upper()
        confidence = max(0, min(100, int(float(state.get("confidence", 0)) * 100)))
        self.top_bar.setText(
            f"{pair}  •  Price {price}  •  {signal} {confidence}%  •  {status.replace('_', ' ').title()}"
        )
        self.top_bar.setStyleSheet(f"color: {status_color(status)};")

        exposure = _as_float(state.get("risk_metrics", {}).get("current_exposure"), 0.0)
        self.hero_cards["price"].set_value(f"{price} USDT", tone="info", badge=pair)
        self.hero_cards["signal"].set_value(
            f"{signal} • {confidence}%",
            tone=_signal_tone(signal),
            badge=trend,
        )
        daily_pnl = float(state.get("daily_pnl", 0) or 0)
        self.hero_cards["daily_pnl"].set_value(
            f"{daily_pnl:.2f} USDT",
            tone="positive" if daily_pnl >= 0 else "negative",
            badge="Session",
        )
        self.hero_cards["exposure"].set_value(
            f"{exposure * 100:.1f}%",
            tone="warning" if exposure >= 0.6 else "neutral",
            badge=str(state.get("cooldown", "READY")),
        )

        self.market_cards["spread"].set_value(_fmt_num(state.get("spread"), 6))
        self.market_cards["adx"].set_value(_fmt_num(state.get("adx"), 2))
        self.market_cards["volatility_regime"].set_value(str(state.get("volatility_regime", "-")))
        self.market_cards["order_flow"].set_value(str(state.get("order_flow", "-")))

        self.signal_badge.setText(f"Signal: {signal}")
        self.signal_badge.setStyleSheet(
            f"padding:4px 10px; border-radius:10px; background:{signal_color(signal)}; color:#e6e8ee;"
        )
        self.confidence_label.setText(f"Confidence {confidence}%")
        self.confidence_anim.stop()
        self.confidence_anim.setStartValue(self.confidence_bar.value())
        self.confidence_anim.setEndValue(confidence)
        self.confidence_anim.start()
        self.market_context.setText(
            " • ".join(
                [
                    f"Regime {state.get('volatility_regime', '-')}",
                    f"Order flow {state.get('order_flow', '-')}",
                    f"ADX {_fmt_num(state.get('adx'), 2)}",
                    f"Cooldown {state.get('cooldown', 'READY')}",
                ]
            )
        )

        self.account_cards["balance"].set_value(f"{_fmt_num(state.get('balance'), 2)} USDT")
        self.account_cards["equity"].set_value(f"{_fmt_num(state.get('equity'), 2)} USDT")
        self.account_cards["btc_balance"].set_value(f"{_fmt_num(state.get('btc_balance'), 6)} BTC")
        self.account_cards["btc_value"].set_value(f"{_fmt_num(state.get('btc_value'), 2)} USDT")
        self.account_cards["total_equity"].set_value(f"{_fmt_num(state.get('total_equity'), 2)} USDT")
        daily_pnl = float(state.get("daily_pnl", 0) or 0)
        self.account_cards["daily_pnl"].set_value(f"{daily_pnl:.2f} USDT")
        self.account_cards["daily_pnl"].value.setStyleSheet(
            f"color: {'#16c784' if daily_pnl >= 0 else '#ea3943'}; font-size:14px; font-weight:600;"
        )
        self.account_cards["trades_today"].set_value(str(state.get("trades_today", "-")))
        self.account_cards["win_rate"].set_value(f"{float(state.get('win_rate', 0) or 0)*100:.1f}%")

        logs = state.get("logs", [])[-8:]
        feed_lines = [_format_feed_entry(entry) for entry in logs] or ["<span style='color:#9fb2d9;'>[--:--] Waiting for events</span>"]
        self.feed.setText("<br>".join(feed_lines))
        system = state.get("system", {}) or {}
        lag_text = "LAG" if system.get("ui_lag_detected") else "UI OK"
        self.feed_meta.setText(
            f"Latency {_fmt_num(system.get('api_latency_ms'), 0)} ms • "
            f"UI {_fmt_num(system.get('ui_render_ms'), 0)} ms • "
            f"Stale {_fmt_num(system.get('ui_staleness_ms'), 0)} ms • "
            f"{lag_text} • "
            f"Mode {state.get('runtime_settings', {}).get('investment_mode', 'Balanced')} • "
            f"Cap {_fmt_num(state.get('runtime_settings', {}).get('capital_limit_usdt'), 2)} USDT"
        )
        self.chart.update_from_snapshot(state)


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def signal_color(signal: str) -> str:
    return {"BUY": "#16c784", "SELL": "#ea3943"}.get(signal, "#667085")


def _signal_tone(signal: str) -> str:
    return {"BUY": "positive", "SELL": "negative"}.get(signal, "neutral")


def _format_feed_entry(entry: dict[str, Any]) -> str:
    level = str(entry.get("level", "INFO")).upper()
    color = {"ERROR": "#ea3943", "WARNING": "#f0b90b", "INFO": "#5a8dff"}.get(level, "#9fb2d9")
    return (
        f"<span style='color:{color}; font-weight:700;'>●</span> "
        f"<span style='color:#9fb2d9;'>[{entry.get('time', '--:--')}]</span> "
        f"<span style='color:#edf2ff;'>{entry.get('message', '-')}</span>"
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def status_color(status: str) -> str:
    normalized = str(status).lower()
    if normalized in {
        "connecting_exchange",
        "syncing_symbol",
        "syncing_rules",
        "analyzing_market",
        "signal_generated",
        "placing_order",
        "position_open",
    }:
        return "#16c784"
    if normalized in {"waiting_market_data", "cooldown", "paused"}:
        return "#f0b90b"
    if normalized == "error":
        return "#ea3943"
    return "#9aa4b2"

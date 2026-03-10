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

        title = QLabel("Professional Trading Dashboard")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        self.top_bar = QLabel("BTC/USDT | - | NEUTRAL | INITIALIZING")
        self.top_bar.setObjectName("metricValue")
        root.addWidget(self.top_bar)

        self.capital_strip = self._panel()
        cap_layout = QHBoxLayout(self.capital_strip)
        cap_layout.setContentsMargins(12, 10, 12, 10)
        cap_layout.setSpacing(18)
        self.usdt_capital = QLabel("USDT Capital: -")
        self.usdt_capital.setObjectName("metricValue")
        self.btc_capital = QLabel("BTC Capital: -")
        self.btc_capital.setObjectName("metricValue")
        self.total_capital = QLabel("Total Equity: -")
        self.total_capital.setObjectName("metricValue")
        cap_layout.addWidget(self.usdt_capital)
        cap_layout.addWidget(self.btc_capital)
        cap_layout.addWidget(self.total_capital)
        root.addWidget(self.capital_strip)

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
            "atr": StatCard("ATR", compact=True),
            "change_24h": StatCard("24h Change", compact=True),
        }
        market_layout = QGridLayout(self.market_panel)
        market_layout.setContentsMargins(10, 10, 10, 10)
        market_layout.addWidget(self._title("Market Intelligence"), 0, 0, 1, 2)
        for i, card in enumerate(self.market_cards.values()):
            market_layout.addWidget(card, (i // 2) + 1, i % 2)
        self.signal_badge = QLabel("NEUTRAL")
        self.confidence_label = QLabel("Confidence")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_anim = QPropertyAnimation(self.confidence_bar, b"value", self)
        self.confidence_anim.setDuration(300)
        self.confidence_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        market_layout.addWidget(self.signal_badge, 4, 0)
        market_layout.addWidget(self.confidence_label, 4, 1)
        market_layout.addWidget(self.confidence_bar, 5, 0, 1, 2)

        self.account_panel = self._panel()
        self.account_cards = {
            "balance": StatCard("USDT Balance", compact=True),
            "btc_balance": StatCard("BTC Balance", compact=True),
            "btc_value": StatCard("BTC Value (USDT)", compact=True),
            "total_equity": StatCard("Total Equity", compact=True),
            "equity": StatCard("Net Equity", compact=True),
            "daily_pnl": StatCard("Daily PnL", compact=True),
            "trades_today": StatCard("Trades Today", compact=True),
            "win_rate": StatCard("Win Rate", compact=True),
            "position_side": StatCard("Position Side", compact=True),
            "entry_price": StatCard("Entry Price", compact=True),
            "position_size": StatCard("Position Size", compact=True),
            "unrealized_pnl": StatCard("Unrealized PnL", compact=True),
            "bot_mode": StatCard("Bot Mode", compact=True),
        }
        account_layout = QGridLayout(self.account_panel)
        account_layout.setContentsMargins(10, 10, 10, 10)
        account_layout.addWidget(self._title("Portfolio & Execution State"), 0, 0, 1, 2)
        for i, card in enumerate(self.account_cards.values()):
            account_layout.addWidget(card, (i // 2) + 1, i % 2)

        self.activity_panel = self._panel()
        activity_layout = QVBoxLayout(self.activity_panel)
        activity_layout.setContentsMargins(10, 10, 10, 10)
        activity_layout.addWidget(self._title("Execution Feed"))
        self.feed = QLabel("[--:--] Waiting for events")
        self.feed.setWordWrap(True)
        self.feed.setObjectName("smallMetricValue")
        activity_layout.addWidget(self.feed)

        self.chart_panel = self._panel()
        chart_layout = QVBoxLayout(self.chart_panel)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_layout.addWidget(self._title("Realtime Candle Matrix"))
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

        layout.addWidget(self.start_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.resume_btn)
        layout.addWidget(self.emergency_btn)

        if self.state_manager:
            self.start_btn.clicked.connect(self.state_manager.request_start)
            self.pause_btn.clicked.connect(self.state_manager.request_pause)
            self.resume_btn.clicked.connect(self.state_manager.request_resume)
            self.emergency_btn.clicked.connect(self.state_manager.request_emergency_stop)

        self._sync_control_buttons("INITIALIZING")
        return panel

    def _sync_control_buttons(self, status: str) -> None:
        normalized = status.upper()
        running = normalized in {"RUNNING", "ACTIVE", "TRADING", "POSITION_OPEN", "PLACING_ORDER"}
        paused = normalized in {"PAUSED", "COOLDOWN"}

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
        self.top_bar.setText(f"{pair} | {price} | {trend} | {status}")
        self.top_bar.setStyleSheet(f"color: {status_color(status)};")

        balance = _to_float(state.get("balance"))
        btc_balance = _to_float(state.get("btc_balance"))
        btc_value = _to_float(state.get("btc_value"))
        total_equity = _to_float(state.get("total_equity"))
        self.usdt_capital.setText(f"USDT Capital: {_fmt_num(balance, 2)} USDT")
        self.btc_capital.setText(f"BTC Capital: {_fmt_num(btc_balance, 8)} BTC ({_fmt_num(btc_value, 2)} USDT)")
        self.total_capital.setText(f"Total Equity: {_fmt_num(total_equity, 2)} USDT")

        self.market_cards["spread"].set_value(_fmt_num(state.get("spread"), 6))
        self.market_cards["adx"].set_value(_fmt_num(state.get("adx"), 2))
        self.market_cards["volatility_regime"].set_value(str(state.get("volatility_regime", "-")))
        self.market_cards["order_flow"].set_value(str(state.get("order_flow", "-")))
        self.market_cards["atr"].set_value(_fmt_num(state.get("atr"), 4))
        change_24h = _to_float(state.get("change_24h"))
        self.market_cards["change_24h"].set_value("-" if change_24h is None else f"{change_24h:.2f}%")

        signal = str(state.get("signal", "NEUTRAL")).upper()
        self.signal_badge.setText(f"Signal: {signal}")
        self.signal_badge.setStyleSheet(
            f"padding:4px 10px; border-radius:10px; background:{signal_color(signal)}; color:#e6e8ee;"
        )

        confidence_raw = state.get("confidence", 0)
        try:
            confidence = max(0, min(100, int(float(confidence_raw or 0) * 100)))
        except (TypeError, ValueError):
            confidence = 0
        self.confidence_label.setText(f"Confidence {confidence}%")
        self.confidence_anim.stop()
        self.confidence_anim.setStartValue(self.confidence_bar.value())
        self.confidence_anim.setEndValue(confidence)
        self.confidence_anim.start()

        self.account_cards["balance"].set_value(f"{_fmt_num(balance, 2)} USDT")
        self.account_cards["btc_balance"].set_value(f"{_fmt_num(btc_balance, 8)} BTC")
        self.account_cards["btc_value"].set_value(f"{_fmt_num(btc_value, 2)} USDT")
        self.account_cards["total_equity"].set_value(f"{_fmt_num(total_equity, 2)} USDT")
        self.account_cards["equity"].set_value(f"{_fmt_num(state.get('equity'), 2)} USDT")

        daily_pnl = _to_float(state.get("daily_pnl")) or 0.0
        self.account_cards["daily_pnl"].set_value(f"{daily_pnl:.2f} USDT")
        self.account_cards["daily_pnl"].value.setStyleSheet(
            f"color: {'#16c784' if daily_pnl >= 0 else '#ea3943'}; font-size:14px; font-weight:600;"
        )

        self.account_cards["trades_today"].set_value(str(state.get("trades_today", "-")))
        try:
            win_rate = float(state.get("win_rate", 0) or 0)
        except (TypeError, ValueError):
            win_rate = 0.0
        self.account_cards["win_rate"].set_value(f"{win_rate*100:.1f}%")
        self.account_cards["position_side"].set_value(str(state.get("position_side", "NONE")))
        self.account_cards["entry_price"].set_value(f"{_fmt_num(state.get('entry_price'), 2)}")
        self.account_cards["position_size"].set_value(f"{_fmt_num(state.get('position_size'), 8)}")
        unrl = float(state.get("unrealized_pnl", 0) or 0)
        self.account_cards["unrealized_pnl"].set_value(f"{unrl:.4f} USDT")
        self.account_cards["unrealized_pnl"].value.setStyleSheet(
            f"color: {'#16c784' if unrl >= 0 else '#ea3943'}; font-size:14px; font-weight:600;"
        )
        system = state.get("system", {})
        self.account_cards["bot_mode"].set_value(str(system.get("bot_mode", state.get("bot_mode", "-"))))

        logs = state.get("logs", [])[-8:]
        lines = [f"[{entry.get('time', '--:--')}] {entry.get('message', '-') }" for entry in logs] or ["[--:--] Waiting for events"]
        self.feed.setText("\n".join(lines))
        self.chart.update_from_snapshot(state)


def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: Any, digits: int) -> str:
    parsed = _to_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.{digits}f}"


def signal_color(signal: str) -> str:
    return {"BUY": "#16c784", "SELL": "#ea3943", "HOLD": "#5a8dff"}.get(signal, "#667085")


def status_color(status: str) -> str:
    status = status.upper()
    if status in {"RUNNING", "POSITION_OPEN", "PLACING_ORDER"}:
        return "#16c784"
    if status in {"WAITING_DATA", "WAITING_MARKET_DATA", "ANALYZING_MARKET", "SIGNAL_GENERATED", "COOLDOWN"}:
        return "#f0b90b"
    if status == "ERROR":
        return "#ea3943"
    return "#9aa4b2"

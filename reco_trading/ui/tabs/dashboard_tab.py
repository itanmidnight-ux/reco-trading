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
        self.top_bar.setObjectName("metricValue")
        root.addWidget(self.top_bar)

        self.capital_strip = QLabel("USDT Capital: - | BTC Holdings: - | BTC Value: - | Allocation: -")
        self.capital_strip.setObjectName("smallMetricValue")
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
            "bid": StatCard("Best Bid", compact=True),
            "ask": StatCard("Best Ask", compact=True),
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
        market_layout.addWidget(self.signal_badge, 4, 0)
        market_layout.addWidget(self.confidence_label, 4, 1)
        market_layout.addWidget(self.confidence_bar, 5, 0, 1, 2)

        self.account_panel = self._panel()
        self.account_cards = {
            "balance": StatCard("USDT Capital", compact=True),
            "equity": StatCard("Equity", compact=True),
            "btc_balance": StatCard("BTC Holdings", compact=True),
            "btc_value": StatCard("BTC Value", compact=True),
            "total_equity": StatCard("Total Equity", compact=True),
            "daily_pnl": StatCard("Daily PnL", compact=True),
            "trades_today": StatCard("Trades Today", compact=True),
            "win_rate": StatCard("Win Rate", compact=True),
            "position_side": StatCard("Position Side", compact=True),
            "entry_price": StatCard("Entry Price", compact=True),
            "position_size": StatCard("Position Size", compact=True),
            "unrealized_pnl": StatCard("Unrealized PnL", compact=True),
            "bot_mode": StatCard("Bot Mode", compact=True),
            "exchange_status": StatCard("Exchange", compact=True),
            "database_status": StatCard("Database", compact=True),
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

        self.pause_btn.setVisible(running)
        self.start_btn.setVisible(not running)
        self.resume_btn.setVisible(False)

    def _panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panelCard")
        return panel

    def _title(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("metricLabel")
        return label

    def update_state(self, state: dict[str, Any]) -> None:
        pair = str(state.get("pair", "BTC/USDT"))
        price = _fmt_num(state.get("current_price", state.get("price")), 2)
        trend = str(state.get("trend", "NEUTRAL"))
        status = str(state.get("status", "-"))
        self._sync_control_buttons(status)
        self.top_bar.setText(f"{pair} | {price} | {trend} | {status}")
        self.top_bar.setStyleSheet(f"color: {status_color(status)};")

        usdt_balance = _as_float(state.get("balance"))
        btc_balance = _as_float(state.get("btc_balance"))
        btc_value = _as_float(state.get("btc_value"))
        total_equity = _as_float(state.get("total_equity"))
        btc_allocation = (btc_value / total_equity) if total_equity > 0 else 0.0
        self.capital_strip.setText(
            f"USDT Capital: {usdt_balance:.2f} USDT | BTC Holdings: {btc_balance:.6f} BTC | "
            f"BTC Value: {btc_value:.2f} USDT | BTC Allocation: {_fmt_pct(btc_allocation)}"
        )

        self.market_cards["spread"].set_value(_fmt_num(state.get("spread"), 6))
        self.market_cards["adx"].set_value(_fmt_num(state.get("adx"), 2))
        self.market_cards["volatility_regime"].set_value(str(state.get("volatility_regime", "-")))
        self.market_cards["order_flow"].set_value(str(state.get("order_flow", "-")))
        self.market_cards["bid"].set_value(_fmt_num(state.get("bid"), 2))
        self.market_cards["ask"].set_value(_fmt_num(state.get("ask"), 2))

        signal = str(state.get("signal", "NEUTRAL")).upper()
        self.signal_badge.setText(f"Signal: {signal}")
        self.signal_badge.setStyleSheet(
            f"padding:4px 10px; border-radius:10px; background:{signal_color(signal)}; color:#e6e8ee;"
        )

        confidence = max(0, min(100, int(_as_float(state.get("confidence")) * 100)))
        self.confidence_label.setText(f"Confidence {confidence}%")
        self.confidence_anim.stop()
        self.confidence_anim.setStartValue(self.confidence_bar.value())
        self.confidence_anim.setEndValue(confidence)
        self.confidence_anim.start()

        self.account_cards["balance"].set_value(f"{usdt_balance:.2f} USDT")
        self.account_cards["equity"].set_value(f"{_as_float(state.get('equity')):.2f} USDT")
        self.account_cards["btc_balance"].set_value(f"{btc_balance:.6f} BTC")
        self.account_cards["btc_value"].set_value(f"{btc_value:.2f} USDT")
        self.account_cards["total_equity"].set_value(f"{total_equity:.2f} USDT")

        daily_pnl = _as_float(state.get("daily_pnl"))
        self.account_cards["daily_pnl"].set_value(f"{daily_pnl:.2f} USDT")
        self.account_cards["daily_pnl"].value.setStyleSheet(
            f"color: {'#16c784' if daily_pnl >= 0 else '#ea3943'}; font-size:14px; font-weight:600;"
        )

        self.account_cards["trades_today"].set_value(str(state.get("trades_today", "-")))
        win_rate = _as_float(state.get("win_rate"))
        self.account_cards["win_rate"].set_value(_fmt_pct(win_rate))

        self.account_cards["position_side"].set_value(str(state.get("position_side", "NONE")))
        self.account_cards["entry_price"].set_value(_fmt_num(state.get("entry_price"), 2))
        self.account_cards["position_size"].set_value(_fmt_num(state.get("position_size"), 8))
        unrl = _as_float(state.get("unrealized_pnl"))
        self.account_cards["unrealized_pnl"].set_value(f"{unrl:.4f} USDT")
        self.account_cards["unrealized_pnl"].value.setStyleSheet(
            f"color: {'#16c784' if unrl >= 0 else '#ea3943'}; font-size:14px; font-weight:600;"
        )

        system = state.get("system", {}) if isinstance(state.get("system"), dict) else {}
        self.account_cards["bot_mode"].set_value(str(system.get("bot_mode", state.get("bot_mode", "-"))))
        self.account_cards["exchange_status"].set_value(str(system.get("exchange_status", "UNKNOWN")))
        self.account_cards["database_status"].set_value(str(system.get("database_status", "UNKNOWN")))

        logs = state.get("logs", [])[-8:]
        lines = [f"[{entry.get('time', '--:--')}] {entry.get('message', '-')}" for entry in logs] or ["[--:--] Waiting for events"]
        self.feed.setText("\n".join(lines))
        self.chart.update_from_snapshot(state)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: float | Any) -> str:
    return f"{_as_float(value) * 100:.1f}%"


def signal_color(signal: str) -> str:
    return {"BUY": "#16c784", "SELL": "#ea3943", "HOLD": "#667085"}.get(signal, "#667085")


def status_color(status: str) -> str:
    status = status.upper()
    if status in {"RUNNING", "POSITION_OPEN", "PLACING_ORDER"}:
        return "#16c784"
    if status in {"WAITING_DATA", "WAITING_MARKET_DATA", "ANALYZING_MARKET", "SIGNAL_GENERATED", "COOLDOWN"}:
        return "#f0b90b"
    if status in {"PAUSED", "STOPPED"}:
        return "#9aa4b2"
    if status == "ERROR":
        return "#ea3943"
    return "#9aa4b2"

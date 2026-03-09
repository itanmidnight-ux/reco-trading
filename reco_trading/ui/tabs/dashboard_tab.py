from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from reco_trading.ui.chart_widget import CandlestickChartWidget
from reco_trading.ui.widgets.stat_card import StatCard


class DashboardTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        self.top_bar = QLabel("BTC/USDT | - | NEUTRAL | INITIALIZING")
        self.top_bar.setObjectName("metricValue")
        root.addWidget(self.top_bar)

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

        self.account_panel = self._panel()
        self.account_cards = {
            "balance": StatCard("Balance", compact=True),
            "equity": StatCard("Equity", compact=True),
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
        self.top_bar.setText(f"{pair} | {price} | {trend} | {status}")
        self.top_bar.setStyleSheet(f"color: {status_color(status)};")

        self.market_cards["spread"].set_value(_fmt_num(state.get("spread"), 6))
        self.market_cards["adx"].set_value(_fmt_num(state.get("adx"), 2))
        self.market_cards["volatility_regime"].set_value(str(state.get("volatility_regime", "-")))
        self.market_cards["order_flow"].set_value(str(state.get("order_flow", "-")))

        signal = str(state.get("signal", "NEUTRAL")).upper()
        self.signal_badge.setText(f"Signal: {signal}")
        self.signal_badge.setStyleSheet(
            f"padding:4px 10px; border-radius:10px; background:{signal_color(signal)}; color:#e6e8ee;"
        )

        confidence = max(0, min(100, int(float(state.get("confidence", 0)) * 100)))
        self.confidence_label.setText(f"Confidence {confidence}%")
        self.confidence_anim.stop()
        self.confidence_anim.setStartValue(self.confidence_bar.value())
        self.confidence_anim.setEndValue(confidence)
        self.confidence_anim.start()

        self.account_cards["balance"].set_value(f"{_fmt_num(state.get('balance'), 2)} USDT")
        self.account_cards["equity"].set_value(f"{_fmt_num(state.get('equity'), 2)} USDT")
        daily_pnl = float(state.get("daily_pnl", 0) or 0)
        self.account_cards["daily_pnl"].set_value(f"{daily_pnl:.2f} USDT")
        self.account_cards["daily_pnl"].value.setStyleSheet(
            f"color: {'#16c784' if daily_pnl >= 0 else '#ea3943'}; font-size:14px; font-weight:600;"
        )
        self.account_cards["trades_today"].set_value(str(state.get("trades_today", "-")))
        self.account_cards["win_rate"].set_value(f"{float(state.get('win_rate', 0) or 0)*100:.1f}%")

        logs = state.get("logs", [])[-8:]
        lines = [f"[{entry.get('time', '--:--')}] {entry.get('message', '-') }" for entry in logs] or ["[--:--] Waiting for events"]
        self.feed.setText("\n".join(lines))
        self.chart.update_from_snapshot(state)


def _fmt_num(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def signal_color(signal: str) -> str:
    return {"BUY": "#16c784", "SELL": "#ea3943"}.get(signal, "#667085")


def status_color(status: str) -> str:
    status = status.upper()
    if status == "RUNNING":
        return "#16c784"
    if status == "WAITING_DATA":
        return "#f0b90b"
    if status == "ERROR":
        return "#ea3943"
    return "#9aa4b2"

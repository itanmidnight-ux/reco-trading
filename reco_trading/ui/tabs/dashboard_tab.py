from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.chart_widget import CandlestickChartWidget
from reco_trading.ui.theme import NEGATIVE, POSITIVE, status_color
from reco_trading.ui.widgets.stat_card import StatCard


class DashboardTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(10)

        self.top = QLabel("BTC/USDT | - | NEUTRAL | INITIALIZING")
        self.top.setStyleSheet("font-size: 18px; font-weight: 700; padding: 6px 4px;")
        root.addWidget(self.top)

        body = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        self.market_cards = self._build_cards([("spread", "Spread"), ("adx", "ADX"), ("volatility_regime", "Volatility"), ("order_flow", "Order Flow"), ("signal", "Signal")])
        left.addLayout(self.market_cards["layout"])

        confidence_wrap = QFrame()
        confidence_wrap.setObjectName("card")
        conf_l = QVBoxLayout(confidence_wrap)
        conf_l.addWidget(QLabel("Confidence"))
        self.confidence = QProgressBar()
        self.confidence.setRange(0, 100)
        self.confidence.setValue(0)
        conf_l.addWidget(self.confidence)
        self.confidence_anim = QPropertyAnimation(self.confidence, b"value")
        self.confidence_anim.setDuration(300)
        self.confidence_anim.setEasingCurve(QEasingCurve.InOutCubic)
        left.addWidget(confidence_wrap)

        self.account_cards = self._build_cards([("balance", "Balance"), ("equity", "Equity"), ("daily_pnl", "Daily PnL"), ("trades_today", "Trades"), ("win_rate", "Win Rate")])
        left.addLayout(self.account_cards["layout"])

        self.activity = QListWidget()
        self.activity.setMinimumHeight(200)
        right.addWidget(QLabel("Bot Activity"))
        right.addWidget(self.activity)

        self.chart = CandlestickChartWidget()
        right.addWidget(self.chart)

        body.addLayout(left, 2)
        body.addLayout(right, 3)
        root.addLayout(body)

    def _build_cards(self, entries: list[tuple[str, str]]) -> dict:
        grid = QGridLayout()
        cards: dict[str, StatCard] = {}
        for i, (key, title) in enumerate(entries):
            card = StatCard(title)
            cards[key] = card
            grid.addWidget(card, i // 3, i % 3)
        cards["layout"] = grid
        return cards

    def update_state(self, state: dict) -> None:
        pair = state.get("pair", "BTC/USDT")
        price = state.get("current_price", state.get("price", "-"))
        trend = state.get("trend", "NEUTRAL")
        status = state.get("status", "-")
        self.top.setText(f"{pair} | {self._fmt(price, 2)} | {trend} | {status}")
        self.top.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {status_color(str(status))}; padding: 6px 4px;")

        for key, card in self.market_cards.items():
            if key == "layout":
                continue
            value = state.get(key, "-")
            if key == "signal":
                txt = str(value or "NEUTRAL")
                card.value.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {'#16c784' if txt.upper() == 'BUY' else '#ea3943' if txt.upper() == 'SELL' else '#9aa4b2'}")
                card.set_value(txt)
            else:
                card.set_value(self._fmt(value, 4))

        for key, card in self.account_cards.items():
            if key == "layout":
                continue
            val = state.get(key, "-")
            if key == "win_rate":
                card.set_value(self._fmt_pct(val))
            elif key == "daily_pnl":
                num = self._to_float(val)
                card.value.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {POSITIVE if (num or 0) >= 0 else NEGATIVE};")
                card.set_value(self._fmt(val, 4))
            else:
                card.set_value(self._fmt(val, 4))

        conf = int(max(0.0, min(1.0, self._to_float(state.get("confidence")) or 0.0)) * 100)
        self.confidence_anim.stop()
        self.confidence_anim.setStartValue(self.confidence.value())
        self.confidence_anim.setEndValue(conf)
        self.confidence_anim.start()

        logs = state.get("logs", [])[-8:]
        self.activity.clear()
        for entry in logs:
            self.activity.addItem(f"[{entry.get('time', '--:--')}] {entry.get('message', '-')}")

        self.chart.update_from_snapshot(state)

    def _fmt(self, v: object, digits: int) -> str:
        n = self._to_float(v)
        return "-" if n is None else f"{n:.{digits}f}"

    def _fmt_pct(self, v: object) -> str:
        n = self._to_float(v)
        return "-" if n is None else f"{n:.2%}"

    def _to_float(self, v: object) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

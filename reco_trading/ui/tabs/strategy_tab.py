from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class StrategyTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_signature: tuple[object, ...] | None = None

        root = QVBoxLayout(self)
        title = QLabel("Strategy Monitor")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        subtitle = QLabel("Signal posture, confidence, session stats and recent strategy decisions")
        subtitle.setObjectName("metricLabel")
        root.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)

        self.signal_panel = QFrame()
        self.signal_panel.setObjectName("metricCard")
        signal_layout = QGridLayout(self.signal_panel)
        signal_layout.setContentsMargins(10, 10, 10, 10)
        signal_layout.setSpacing(8)

        self.signal_cards = {
            "signal": StatCard("Signal", compact=True),
            "trend": StatCard("Trend", compact=True),
            "order_flow": StatCard("Order Flow", compact=True),
            "volatility": StatCard("Volatility", compact=True),
        }
        for i, card in enumerate(self.signal_cards.values()):
            signal_layout.addWidget(card, 0, i)

        self.confidence_value = QLabel("Confidence 0%")
        self.confidence_value.setObjectName("smallMetricValue")
        signal_layout.addWidget(self.confidence_value, 1, 0, 1, 2)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        signal_layout.addWidget(self.confidence_bar, 1, 2, 1, 2)
        self._confidence_anim = QPropertyAnimation(self.confidence_bar, b"value", self)
        self._confidence_anim.setDuration(280)
        self._confidence_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.signal_context = QLabel("Waiting for strategy state...")
        self.signal_context.setObjectName("metricLabel")
        self.signal_context.setWordWrap(True)
        signal_layout.addWidget(self.signal_context, 2, 0, 1, 4)
        panel_layout.addWidget(self.signal_panel)

        stats_grid = QGridLayout()
        self.session_cards = {
            "total_trades": StatCard("Session Trades", compact=True),
            "win_rate": StatCard("Session Win Rate", compact=True),
            "streak": StatCard("Current Streak", compact=True),
            "recommendation": StatCard("Recommendation", compact=True),
            "profit_factor": StatCard("Profit Factor", compact=True),
            "sharpe": StatCard("Sharpe", compact=True),
        }
        for i, card in enumerate(self.session_cards.values()):
            stats_grid.addWidget(card, i // 3, i % 3)
        panel_layout.addLayout(stats_grid)

        self.signal_history = QTableWidget(0, 2)
        self.signal_history.setHorizontalHeaderLabels(["Signal Metric", "Value"])
        self.signal_history.horizontalHeader().setStretchLastSection(True)
        panel_layout.addWidget(self.signal_history)

        self.decision_history = QListWidget()
        self.decision_history.addItems(["No recent strategy events."])
        panel_layout.addWidget(self.decision_history)

    def update_state(self, state: dict[str, Any]) -> None:
        analytics = dict(state.get("analytics", {}) or {})
        session_stats = dict(analytics.get("session_stats", {}) or state.get("session_stats", {}) or {})
        signals = dict(state.get("signals", {}) or {})
        confidence_raw = float(state.get("confidence", 0.0) or 0.0)

        signature = (
            str(state.get("signal", "-")),
            round(confidence_raw, 4),
            str(state.get("trend", "-")),
            str(state.get("order_flow", "-")),
            str(state.get("volatility_regime", "-")),
            tuple(sorted((str(k), str(v)) for k, v in signals.items())),
            tuple(sorted((str(k), str(session_stats.get(k, "-"))) for k in ("total_trades", "win_rate", "streak", "recommendation", "profit_factor", "sharpe"))),
            str(state.get("pair", "-")),
            str(state.get("timeframe", "-")),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        signal = str(state.get("signal", "HOLD"))
        trend = str(state.get("trend", "NEUTRAL"))
        order_flow = str(state.get("order_flow", "NEUTRAL"))
        volatility = str(state.get("volatility_regime", "NORMAL"))
        tone = _signal_tone(signal)

        self.signal_cards["signal"].set_value(signal, tone=tone)
        self.signal_cards["trend"].set_value(trend, tone=_trend_tone(trend))
        self.signal_cards["order_flow"].set_value(order_flow, tone=_trend_tone(order_flow))
        self.signal_cards["volatility"].set_value(volatility, tone="warning" if volatility.upper() in {"HIGH", "EXTREME"} else "info")

        confidence_pct = max(0, min(100, int(confidence_raw * 100)))
        self.confidence_value.setText(f"Confidence {confidence_pct}%")
        self._confidence_anim.stop()
        self._confidence_anim.setStartValue(self.confidence_bar.value())
        self._confidence_anim.setEndValue(confidence_pct)
        self._confidence_anim.start()

        pair = str(state.get("pair", "-"))
        timeframe = str(state.get("timeframe", "-"))
        self.signal_context.setText(
            f"{pair} • {timeframe} • Signal {signal} with {confidence_pct}% confidence. "
            f"Trend: {trend}. Order flow: {order_flow}. Volatility: {volatility}."
        )

        for key, card in self.session_cards.items():
            value = session_stats.get(key, "-")
            if key == "win_rate":
                try:
                    rate = float(value)
                    card.set_value(f"{rate * 100:.2f}%", tone=_win_rate_tone(rate))
                except (TypeError, ValueError):
                    card.set_value("-", tone="neutral")
            elif key in {"profit_factor", "sharpe"}:
                try:
                    num = float(value)
                    card.set_value(f"{num:.3f}", tone="positive" if num > 0 else "warning")
                except (TypeError, ValueError):
                    card.set_value(str(value))
            elif key == "recommendation":
                card.set_value(str(value), tone=_recommendation_tone(str(value)))
            else:
                card.set_value(str(value))

        metric_rows = [
            ("Primary Signal", signal),
            ("Confidence", f"{confidence_raw:.2f}"),
            ("Pair", pair),
            ("Timeframe", timeframe),
            ("Confluence Score", str(state.get("confluence_score", "-"))),
            ("Confluence Aligned", str(state.get("confluence_aligned", "-"))),
            ("Session Recommendation", str(session_stats.get("recommendation", "-"))),
            ("Session Streak", str(session_stats.get("streak", state.get("session_streak", "-")))),
        ]
        if signals:
            for key, value in sorted(signals.items()):
                metric_rows.append((str(key).replace("_", " ").title(), str(value)))

        self.signal_history.setRowCount(0)
        for row, (label, value) in enumerate(metric_rows):
            self.signal_history.insertRow(row)
            self.signal_history.setItem(row, 0, QTableWidgetItem(label))
            self.signal_history.setItem(row, 1, QTableWidgetItem(str(value)))

        self.decision_history.clear()
        history_entries = [
            f"Signal: {signal}",
            f"Confidence: {confidence_pct}%",
            f"Session recommendation: {session_stats.get('recommendation', state.get('session_recommendation', '-'))}",
            f"Current streak: {session_stats.get('streak', state.get('session_streak', '-'))}",
        ]
        if signals:
            history_entries.extend(f"{str(key).replace('_', ' ').title()}: {value}" for key, value in sorted(signals.items()))
        else:
            history_entries.append("No signal components available yet.")
        self.decision_history.addItems(history_entries)


def _signal_tone(signal: str) -> str:
    normalized = signal.strip().upper()
    if normalized in {"BUY", "LONG", "INCREASE"}:
        return "positive"
    if normalized in {"SELL", "SHORT", "REDUCE", "REDUCE_SIZE", "PAUSE"}:
        return "warning"
    return "neutral"


def _trend_tone(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"BULLISH", "UP", "BUY"}:
        return "positive"
    if normalized in {"BEARISH", "DOWN", "SELL"}:
        return "negative"
    return "info"


def _win_rate_tone(rate: float) -> str:
    if rate >= 0.60:
        return "positive"
    if rate >= 0.45:
        return "warning"
    return "negative"


def _recommendation_tone(value: str) -> str:
    normalized = value.strip().upper()
    if normalized == "INCREASE":
        return "positive"
    if normalized in {"PAUSE", "REDUCE", "REDUCE_SIZE"}:
        return "warning"
    return "neutral"

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.widgets.stat_card import StatCard


class RiskTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_signature: tuple[object, ...] | None = None
        root = QVBoxLayout(self)
        title = QLabel("Risk Center")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        subtitle = QLabel("Exposure, drawdown and protection controls")
        subtitle.setObjectName("metricLabel")
        root.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        root.addWidget(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)

        layout = QGridLayout()
        self.cards = {}
        keys = [
            "risk_per_trade",
            "max_concurrent_trades",
            "daily_drawdown",
            "current_exposure",
            "consecutive_losses",
            "capital_profile",
            "operable_capital_usdt",
            "setup_quality_score",
            "adaptive_size_multiplier",
        ]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title(), compact=True)
            self.cards[key] = card
            layout.addWidget(card, i // 3, i % 3)
        panel_layout.addLayout(layout)

        self.status_badge = QLabel("Risk posture: NORMAL")
        self.status_badge.setObjectName("smallMetricValue")
        panel_layout.addWidget(self.status_badge)

        self.exposure_bar = QProgressBar()
        self.exposure_bar.setRange(0, 100)
        panel_layout.addWidget(self.exposure_bar)
        self.alerts = QListWidget()
        self.alerts.addItems(["No risk alerts."])
        panel_layout.addWidget(self.alerts)
        self._anim = QPropertyAnimation(self.exposure_bar, b"value", self)
        self._anim.setDuration(300)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.live_position_label = QLabel("No open position")
        self.live_position_label.setObjectName("smallMetricValue")
        self.live_position_label.setWordWrap(True)
        panel_layout.addWidget(self.live_position_label)

    def update_state(self, state: dict) -> None:
        metrics = state.get("risk_metrics", {})
        signature = (
            metrics.get("risk_per_trade", "-"),
            metrics.get("max_concurrent_trades", "-"),
            metrics.get("daily_drawdown", "-"),
            metrics.get("current_exposure", "-"),
            metrics.get("consecutive_losses", "-"),
            metrics.get("capital_profile", "-"),
            metrics.get("operable_capital_usdt", "-"),
            metrics.get("setup_quality_score", "-"),
            metrics.get("adaptive_size_multiplier", "-"),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        for key, card in self.cards.items():
            card.set_value(str(metrics.get(key, "-")))
        try:
            exposure = int(float(metrics.get("current_exposure", 0)) * 100)
        except (TypeError, ValueError):
            exposure = 0
        self._anim.stop()
        self._anim.setStartValue(self.exposure_bar.value())
        self._anim.setEndValue(max(0, min(100, exposure)))
        self._anim.start()

        self.alerts.clear()
        daily_drawdown_raw = float(metrics.get("daily_drawdown", 0) or 0)
        daily_drawdown = abs(daily_drawdown_raw)
        if exposure >= 80:
            self.alerts.addItem("High exposure detected, consider reducing position sizes.")
        if daily_drawdown >= 0.03:
            self.alerts.addItem("Drawdown exceeded 3%, pause aggressive entries.")
        if str(metrics.get("capital_profile", "")).upper() == "MICRO":
            self.alerts.addItem("MICRO profile active, only premium setups should pass.")
        if self.alerts.count() == 0:
            self.alerts.addItem("No risk alerts.")
            self.status_badge.setText("Risk posture: NORMAL")
            self.status_badge.setStyleSheet("color:#16c784;")
        else:
            self.status_badge.setText("Risk posture: CAUTION")
            self.status_badge.setStyleSheet("color:#f0b90b;")

        unrealized = float(state.get("unrealized_pnl") or 0)
        has_pos = bool(state.get("has_open_position", False))
        if has_pos:
            side = str(state.get("open_position_side") or "-").upper()
            entry = float(state.get("open_position_entry") or 0)
            sl = float(state.get("open_position_sl") or 0)
            color = "#16c784" if unrealized >= 0 else "#ea3943"
            self.live_position_label.setText(
                f"{side} @ {entry:.2f}  |  Unrealized: {unrealized:+.4f} USDT  |  SL: {sl:.2f}"
            )
            self.live_position_label.setStyleSheet(f"color: {color};")
        else:
            self.live_position_label.setText("No open position")
            self.live_position_label.setStyleSheet("color: #9fb2d9;")

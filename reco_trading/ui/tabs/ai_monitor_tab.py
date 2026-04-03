from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class AIMonitorTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("AI/LLM Monitor")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        self.status_bar = QLabel("AI Systems Status: Initializing...")
        self.status_bar.setObjectName("statusRibbon")
        self.status_bar.setStyleSheet(
            "padding:8px 12px; border-radius:12px; "
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1f2a44, stop:1 #111827);"
            "color:#d9e6ff; border:1px solid #2f3b59;"
        )
        root.addWidget(self.status_bar)

        ml_panel = self._panel()
        ml_layout = QGridLayout(ml_panel)
        ml_layout.setContentsMargins(10, 10, 10, 10)
        ml_layout.setSpacing(8)
        ml_layout.addWidget(self._title("ML Models"), 0, 0, 1, 4)

        self.ml_direction = self._info_item("ML Direction", "-")
        ml_layout.addWidget(self.ml_direction, 1, 0)
        self.ml_confidence = self._info_item("ML Confidence", "-")
        ml_layout.addWidget(self.ml_confidence, 1, 1)
        self.tft_direction = self._info_item("TFT Direction", "-")
        ml_layout.addWidget(self.tft_direction, 1, 2)
        self.tft_confidence = self._info_item("TFT Confidence", "-")
        ml_layout.addWidget(self.tft_confidence, 1, 3)
        self.nbeats_direction = self._info_item("NBEATS Direction", "-")
        ml_layout.addWidget(self.nbeats_direction, 2, 0)
        self.nbeats_confidence = self._info_item("NBEATS Confidence", "-")
        ml_layout.addWidget(self.nbeats_confidence, 2, 1)
        self.market_regime = self._info_item("Market Regime", "-")
        ml_layout.addWidget(self.market_regime, 2, 2)
        self.market_sentiment = self._info_item("Market Sentiment", "-")
        ml_layout.addWidget(self.market_sentiment, 2, 3)

        root.addWidget(ml_panel)

        brain_panel = self._panel()
        brain_layout = QGridLayout(brain_panel)
        brain_layout.setContentsMargins(10, 10, 10, 10)
        brain_layout.setSpacing(8)
        brain_layout.addWidget(self._title("Autonomous Brain"), 0, 0, 1, 4)

        self.active_pair = self._info_item("Active Pair", "-")
        brain_layout.addWidget(self.active_pair, 1, 0)
        self.trading_mode = self._info_item("Trading Mode", "-")
        brain_layout.addWidget(self.trading_mode, 1, 1)
        self.leverage = self._info_item("Leverage", "-")
        brain_layout.addWidget(self.leverage, 1, 2)
        self.side = self._info_item("Side", "-")
        brain_layout.addWidget(self.side, 1, 3)
        self.all_modules = self._info_item("All Modules Init", "-")
        brain_layout.addWidget(self.all_modules, 2, 0)
        self.ws_connected = self._info_item("WS Connected", "-")
        brain_layout.addWidget(self.ws_connected, 2, 1)
        self.auto_trades = self._info_item("Auto-Improver Trades", "-")
        brain_layout.addWidget(self.auto_trades, 2, 2)
        self.auto_wr = self._info_item("Auto-Improver Win Rate", "-")
        brain_layout.addWidget(self.auto_wr, 2, 3)

        root.addWidget(brain_panel)

        health_panel = self._panel()
        health_layout = QGridLayout(health_panel)
        health_layout.setContentsMargins(10, 10, 10, 10)
        health_layout.setSpacing(8)
        health_layout.addWidget(self._title("System Health"), 0, 0, 1, 4)

        self.resilience = self._info_item("Resilience", "-")
        health_layout.addWidget(self.resilience, 1, 0)
        self.failures = self._info_item("Consecutive Failures", "-")
        health_layout.addWidget(self.failures, 1, 1)
        self.emergency = self._info_item("Emergency System", "-")
        health_layout.addWidget(self.emergency, 1, 2)
        self.consec_losses = self._info_item("Consecutive Losses", "-")
        health_layout.addWidget(self.consec_losses, 1, 3)
        self.opt_count = self._info_item("Optimization Count", "-")
        health_layout.addWidget(self.opt_count, 2, 0)

        root.addWidget(health_panel)

        params_panel = self._panel()
        params_layout = QVBoxLayout(params_panel)
        params_layout.setContentsMargins(10, 10, 10, 10)
        params_layout.setSpacing(4)
        params_layout.addWidget(self._title("Auto-Optimized Parameters"))
        self.auto_params_label = QLabel("Waiting for data...")
        self.auto_params_label.setWordWrap(True)
        self.auto_params_label.setObjectName("smallMetricValue")
        self.auto_params_label.setStyleSheet("color:#94a3b8;")
        params_layout.addWidget(self.auto_params_label)

        root.addWidget(params_panel)
        root.addStretch()

    def _panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panelCard")
        panel.setStyleSheet(
            "QFrame#panelCard {"
            "background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #131c2e, stop:1 #0f172a);"
            "border:1px solid #243049; border-radius:14px;"
            "}"
        )
        return panel

    def _title(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("metricLabel")
        return label

    def _info_item(self, label: str, value: str) -> QLabel:
        widget = QLabel()
        widget.setText(f"<b style='color:#64748b;'>{label}:</b> <span style='color:#e2e8f0;'>{value}</span>")
        widget.setWordWrap(True)
        return widget

    def update_state(self, state: dict[str, Any]) -> None:
        active = 0
        total = 8

        ml_dir = str(state.get("ml_direction", "-") or "-")
        ml_conf = state.get("ml_confidence", 0) or 0
        self.ml_direction.setText(
            f"<b style='color:#64748b;'>ML Direction:</b> <span style='color:#e2e8f0;'>{ml_dir}</span>"
        )
        self.ml_confidence.setText(
            f"<b style='color:#64748b;'>ML Confidence:</b> <span style='color:#e2e8f0;'>{float(ml_conf)*100:.1f}%</span>"
        )
        if ml_dir != "-":
            active += 1

        tft_dir = str(state.get("tft_direction", "-") or "-")
        tft_conf = state.get("tft_confidence", 0) or 0
        self.tft_direction.setText(
            f"<b style='color:#64748b;'>TFT Direction:</b> <span style='color:#e2e8f0;'>{tft_dir}</span>"
        )
        self.tft_confidence.setText(
            f"<b style='color:#64748b;'>TFT Confidence:</b> <span style='color:#e2e8f0;'>{float(tft_conf)*100:.1f}%</span>"
        )
        if tft_dir != "-":
            active += 1

        nbeats_dir = str(state.get("nbeats_direction", "-") or "-")
        nbeats_conf = state.get("nbeats_confidence", 0) or 0
        self.nbeats_direction.setText(
            f"<b style='color:#64748b;'>NBEATS Direction:</b> <span style='color:#e2e8f0;'>{nbeats_dir}</span>"
        )
        self.nbeats_confidence.setText(
            f"<b style='color:#64748b;'>NBEATS Confidence:</b> <span style='color:#e2e8f0;'>{float(nbeats_conf)*100:.1f}%</span>"
        )
        if nbeats_dir != "-":
            active += 1

        m_regime = str(state.get("market_regime", "-") or "-")
        m_sentiment = str(state.get("market_sentiment", "-") or "-")
        self.market_regime.setText(
            f"<b style='color:#64748b;'>Market Regime:</b> <span style='color:#e2e8f0;'>{m_regime}</span>"
        )
        self.market_sentiment.setText(
            f"<b style='color:#64748b;'>Market Sentiment:</b> <span style='color:#e2e8f0;'>{m_sentiment}</span>"
        )
        if m_regime != "-":
            active += 1

        a_pair = str(state.get("active_pair", "-") or "-")
        t_mode = str(state.get("trading_mode", "-") or "-")
        t_lev = str(state.get("trading_leverage", "-") or "-")
        t_side = str(state.get("trading_side", "-") or "-")
        self.active_pair.setText(
            f"<b style='color:#64748b;'>Active Pair:</b> <span style='color:#e2e8f0;'>{a_pair}</span>"
        )
        self.trading_mode.setText(
            f"<b style='color:#64748b;'>Trading Mode:</b> <span style='color:#e2e8f0;'>{t_mode}</span>"
        )
        self.leverage.setText(
            f"<b style='color:#64748b;'>Leverage:</b> <span style='color:#e2e8f0;'>{t_lev}</span>"
        )
        self.side.setText(
            f"<b style='color:#64748b;'>Side:</b> <span style='color:#e2e8f0;'>{t_side}</span>"
        )

        all_mod = str(state.get("all_modules_initialized", "-") or "-")
        ws_conn = str(state.get("ws_connected", "-") or "-")
        self.all_modules.setText(
            f"<b style='color:#64748b;'>All Modules Init:</b> <span style='color:#e2e8f0;'>{all_mod}</span>"
        )
        self.ws_connected.setText(
            f"<b style='color:#64748b;'>WS Connected:</b> <span style='color:#e2e8f0;'>{ws_conn}</span>"
        )

        auto_trades = str(state.get("auto_improve_total_trades", "-") or "-")
        auto_wr_val = state.get("auto_improve_win_rate", 0) or 0
        self.auto_trades.setText(
            f"<b style='color:#64748b;'>Auto-Improver Trades:</b> <span style='color:#e2e8f0;'>{auto_trades}</span>"
        )
        self.auto_wr.setText(
            f"<b style='color:#64748b;'>Auto-Improver Win Rate:</b> <span style='color:#e2e8f0;'>{float(auto_wr_val)*100:.1f}%</span>"
        )

        resilience_healthy = str(state.get("resilience_healthy", "-") or "-")
        consec_failures = str(state.get("resilience_consecutive_failures", "-") or "-")
        emergency_sys = str(state.get("emergency_system", "-") or "-")
        consec_losses = str(state.get("auto_improve_consecutive_losses", "-") or "-")
        opt_count = str(state.get("auto_improve_optimization_count", "-") or "-")
        self.resilience.setText(
            f"<b style='color:#64748b;'>Resilience:</b> <span style='color:#e2e8f0;'>{resilience_healthy}</span>"
        )
        self.failures.setText(
            f"<b style='color:#64748b;'>Consecutive Failures:</b> <span style='color:#e2e8f0;'>{consec_failures}</span>"
        )
        self.emergency.setText(
            f"<b style='color:#64748b;'>Emergency System:</b> <span style='color:#e2e8f0;'>{emergency_sys}</span>"
        )
        self.consec_losses.setText(
            f"<b style='color:#64748b;'>Consecutive Losses:</b> <span style='color:#e2e8f0;'>{consec_losses}</span>"
        )
        self.opt_count.setText(
            f"<b style='color:#64748b;'>Optimization Count:</b> <span style='color:#e2e8f0;'>{opt_count}</span>"
        )

        auto_params = state.get("auto_optimized_params", {}) or {}
        if auto_params:
            import json
            self.auto_params_label.setText(json.dumps(auto_params, indent=2))
        else:
            self.auto_params_label.setText("No auto-optimized parameters available yet.")

        status_text = f"AI Systems: {active}/{total} active"
        self.status_bar.setText(status_text)

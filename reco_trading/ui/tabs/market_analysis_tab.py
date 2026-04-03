from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class MarketAnalysisTab(QWidget):
    analysis_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._analysis_running = False
        self._analysis_complete = False
        self._pending_pair_switch = False

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

        title = QLabel("Market Analysis Center")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        status_bar = QLabel("Ready")
        status_bar.setObjectName("statusRibbon")
        status_bar.setStyleSheet(
            "padding:8px 12px; border-radius:12px; "
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1f2a44, stop:1 #111827);"
            "color:#d9e6ff; border:1px solid #2f3b59;"
        )
        self.status_bar = status_bar
        root.addWidget(status_bar)

        info_panel = self._panel()
        info_layout = QGridLayout(info_panel)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(8)

        self.current_pair_label = self._info_item("Current Pair", "BTC/USDT")
        info_layout.addWidget(self.current_pair_label, 0, 0)

        self.ai_status_label = self._info_item("AI Connection", "UNKNOWN")
        info_layout.addWidget(self.ai_status_label, 0, 1)

        self.markets_to_scan_label = self._info_item("Markets to Scan", "0")
        info_layout.addWidget(self.markets_to_scan_label, 1, 0)

        self.analyzed_label = self._info_item("Analyzed", "0")
        info_layout.addWidget(self.analyzed_label, 1, 1)

        self.best_pair_label = self._info_item("Best Pair Found", "--")
        info_layout.addWidget(self.best_pair_label, 2, 0)

        self.last_analysis_label = self._info_item("Last Analysis", "Never")
        info_layout.addWidget(self.last_analysis_label, 2, 1)

        root.addWidget(info_panel)

        controls_panel = self._panel()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)

        controls_header = QLabel("Analysis Controls")
        controls_header.setObjectName("metricLabel")
        controls_layout.addWidget(controls_header)

        market_count_layout = QHBoxLayout()
        market_count_layout.addWidget(QLabel("Markets to analyze:"))
        self.market_count_spin = QSpinBox()
        self.market_count_spin.setRange(1, 500)
        self.market_count_spin.setValue(50)
        self.market_count_spin.setMinimumWidth(80)
        market_count_layout.addWidget(self.market_count_spin)
        market_count_layout.addStretch()
        controls_layout.addLayout(market_count_layout)

        btn_layout = QHBoxLayout()
        self.start_analysis_btn = QPushButton("Iniciar Analisis de Mejor Mercado")
        self.start_analysis_btn.setMinimumHeight(40)
        self.start_analysis_btn.setStyleSheet(
            "QPushButton { background:#16c784; color:#0a0f14; font-weight:700; font-size:13px; "
            "border-radius:8px; padding:8px 16px; }"
            "QPushButton:hover { background:#1ae09a; }"
            "QPushButton:pressed { background:#12a86e; }"
        )
        self.start_analysis_btn.clicked.connect(self._on_start_analysis)
        btn_layout.addWidget(self.start_analysis_btn)

        self.cancel_analysis_btn = QPushButton("Cancelar Analisis")
        self.cancel_analysis_btn.setMinimumHeight(40)
        self.cancel_analysis_btn.setStyleSheet(
            "QPushButton { background:#ea3943; color:#e6e8ee; font-weight:700; font-size:13px; "
            "border-radius:8px; padding:8px 16px; }"
            "QPushButton:hover { background:#f0555e; }"
            "QPushButton:pressed { background:#c42d37; }"
        )
        self.cancel_analysis_btn.setVisible(False)
        self.cancel_analysis_btn.clicked.connect(self._on_cancel_analysis)
        btn_layout.addWidget(self.cancel_analysis_btn)

        btn_layout.addStretch()
        controls_layout.addLayout(btn_layout)

        self.change_pair_btn = QPushButton("Cambiar a la Par/Moneda")
        self.change_pair_btn.setMinimumHeight(40)
        self.change_pair_btn.setStyleSheet(
            "QPushButton { background:#6366f1; color:#e6e8ee; font-weight:700; font-size:13px; "
            "border-radius:8px; padding:8px 16px; }"
            "QPushButton:hover { background:#818cf8; }"
            "QPushButton:pressed { background:#4f46e5; }"
        )
        self.change_pair_btn.setVisible(False)
        self.change_pair_btn.clicked.connect(self._on_change_pair)
        controls_layout.addWidget(self.change_pair_btn)

        root.addWidget(controls_panel)

        progress_panel = self._panel()
        progress_layout = QVBoxLayout(progress_panel)
        progress_layout.setContentsMargins(10, 10, 10, 10)
        progress_layout.setSpacing(6)

        progress_header = QLabel("Analysis Progress")
        progress_header.setObjectName("metricLabel")
        progress_layout.addWidget(progress_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(24)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background:#1e293b; border-radius:12px; border:1px solid #2f3b59; }"
            "QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0, "
            "stop:0 #16c784, stop:0.5 #22d3ee, stop:1 #16c784); border-radius:10px; }"
        )
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("0.0%")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color:#16c784; font-size:16px; font-weight:700; font-family:'JetBrains Mono',monospace;")
        progress_layout.addWidget(self.progress_label)

        root.addWidget(progress_panel)

        results_panel = self._panel()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(10, 10, 10, 10)
        results_layout.setSpacing(4)

        results_header = QLabel("Analysis Results")
        results_header.setObjectName("metricLabel")
        results_layout.addWidget(results_header)

        self.results_label = QLabel("No analysis results yet. Start an analysis to scan all markets.")
        self.results_label.setWordWrap(True)
        self.results_label.setObjectName("smallMetricValue")
        self.results_label.setStyleSheet("color:#94a3b8;")
        results_layout.addWidget(self.results_label)

        root.addWidget(results_panel)

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

    def _info_item(self, label: str, value: str) -> QLabel:
        widget = QLabel()
        widget.setText(f"<b style='color:#64748b;'>{label}:</b> <span style='color:#e2e8f0;'>{value}</span>")
        widget.setWordWrap(True)
        return widget

    def _on_start_analysis(self) -> None:
        self._analysis_running = True
        self._analysis_complete = False
        self.start_analysis_btn.setVisible(False)
        self.cancel_analysis_btn.setVisible(True)
        self.change_pair_btn.setVisible(False)
        self.market_count_spin.setEnabled(False)
        self.analysis_requested.emit("start")

    def _on_cancel_analysis(self) -> None:
        self._analysis_running = False
        self._analysis_complete = False
        self.start_analysis_btn.setVisible(True)
        self.cancel_analysis_btn.setVisible(False)
        self.change_pair_btn.setVisible(False)
        self.market_count_spin.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("0.0%")
        self.status_bar.setText("Analysis cancelled")
        self.analysis_requested.emit("cancel")

    def _on_change_pair(self) -> None:
        self._pending_pair_switch = True
        self.analysis_requested.emit("change_pair")

    def update_state(self, state: dict[str, Any]) -> None:
        ma = state.get("market_analysis", {}) if isinstance(state, dict) else {}

        current_pair = state.get("pair", "BTC/USDT")
        self.current_pair_label.setText(
            f"<b style='color:#64748b;'>Current Pair:</b> <span style='color:#e2e8f0;'>{current_pair}</span>"
        )

        ai_connected = ma.get("ai_connected", False)
        ai_status = "Connected" if ai_connected else "Disconnected"
        ai_color = "#16c784" if ai_connected else "#ea3943"
        self.ai_status_label.setText(
            f"<b style='color:#64748b;'>AI Connection:</b> <span style='color:{ai_color};'>{ai_status}</span>"
        )

        total_markets = ma.get("total_markets", 0)
        analyzed_markets = ma.get("analyzed_markets", 0)
        self.markets_to_scan_label.setText(
            f"<b style='color:#64748b;'>Markets to Scan:</b> <span style='color:#e2e8f0;'>{total_markets}</span>"
        )
        self.analyzed_label.setText(
            f"<b style='color:#64748b;'>Analyzed:</b> <span style='color:#e2e8f0;'>{analyzed_markets}</span>"
        )

        best_pair = ma.get("best_pair", "--")
        best_score = ma.get("best_score", 0)
        if best_pair and best_pair != "--":
            self.best_pair_label.setText(
                f"<b style='color:#64748b;'>Best Pair Found:</b> <span style='color:#22d3ee;'>{best_pair} (score: {best_score:.2f})</span>"
            )
        else:
            self.best_pair_label.setText(
                f"<b style='color:#64748b;'>Best Pair Found:</b> <span style='color:#e2e8f0;'>--</span>"
            )

        last_analysis = ma.get("last_analysis_time", "Never")
        self.last_analysis_label.setText(
            f"<b style='color:#64748b;'>Last Analysis:</b> <span style='color:#e2e8f0;'>{last_analysis}</span>"
        )

        progress_pct = ma.get("progress_pct", 0.0)
        progress_int = int(progress_pct * 10)
        self.progress_bar.setValue(progress_int)
        self.progress_label.setText(f"{progress_pct:.1f}%")

        analysis_status = ma.get("status", "idle")

        if analysis_status == "running":
            if not self._analysis_running:
                self._analysis_running = True
                self.start_analysis_btn.setVisible(False)
                self.cancel_analysis_btn.setVisible(True)
                self.change_pair_btn.setVisible(False)
                self.market_count_spin.setEnabled(False)
            self.status_bar.setText(
                f"Analyzing markets... {analyzed_markets}/{total_markets} ({progress_pct:.1f}%)"
            )
            self.status_bar.setStyleSheet(
                "padding:8px 12px; border-radius:12px; "
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1a3a2a, stop:1 #111827);"
                "color:#16c784; border:1px solid #2f3b59;"
            )
        elif analysis_status == "completed":
            self._analysis_running = False
            self._analysis_complete = True
            self.start_analysis_btn.setVisible(True)
            self.cancel_analysis_btn.setVisible(False)
            self.change_pair_btn.setVisible(True)
            self.market_count_spin.setEnabled(True)
            self.status_bar.setText(
                f"Analysis complete! Best pair: {best_pair} (score: {best_score:.2f})"
            )
            self.status_bar.setStyleSheet(
                "padding:8px 12px; border-radius:12px; "
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1a3a4a, stop:1 #111827);"
                "color:#22d3ee; border:1px solid #2f3b59;"
            )
            results_text = (
                f"Analysis completed at {last_analysis}. "
                f"Scanned {total_markets} markets. "
                f"Best pair: {best_pair} with score {best_score:.2f}. "
                f"Click 'Cambiar a la Par/Moneda' to switch to the best pair."
            )
            self.results_label.setText(results_text)
        elif analysis_status == "cancelled":
            self._analysis_running = False
            self._analysis_complete = False
            self.start_analysis_btn.setVisible(True)
            self.cancel_analysis_btn.setVisible(False)
            self.change_pair_btn.setVisible(False)
            self.market_count_spin.setEnabled(True)
            self.status_bar.setText("Analysis cancelled")
            self.status_bar.setStyleSheet(
                "padding:8px 12px; border-radius:12px; "
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #3a1a1a, stop:1 #111827);"
                "color:#ea3943; border:1px solid #2f3b59;"
            )
        else:
            if not self._analysis_running:
                self.start_analysis_btn.setVisible(True)
                self.cancel_analysis_btn.setVisible(False)
                self.change_pair_btn.setVisible(False)
                self.market_count_spin.setEnabled(True)
            self.status_bar.setText("Ready")
            self.status_bar.setStyleSheet(
                "padding:8px 12px; border-radius:12px; "
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1f2a44, stop:1 #111827);"
                "color:#d9e6ff; border:1px solid #2f3b59;"
            )

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QFrame, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget


class SettingsTab(QWidget):
    settings_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Interface Studio")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        description = QLabel("Customize visual behavior and refresh speed")
        description.setObjectName("metricLabel")
        layout.addWidget(description)

        panel = QFrame()
        panel.setObjectName("panelCard")
        layout.addWidget(panel)
        panel_layout = QVBoxLayout(panel)

        form = QFormLayout()
        self.refresh_rate = QSpinBox()
        self.refresh_rate.setRange(250, 5000)
        self.refresh_rate.setValue(1000)
        self.chart_visible = QCheckBox()
        self.chart_visible.setChecked(True)
        self.theme = QComboBox()
        self.theme.addItems(["Dark", "Dark+Contrast"])
        self.log_verbosity = QComboBox()
        self.log_verbosity.addItems(["INFO", "WARNING", "ERROR"])
        self.default_pair = QComboBox()
        self.default_pair.addItems(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        self.default_tf = QComboBox()
        self.default_tf.addItems(["1m / 5m", "5m / 15m", "15m / 1h"])

        form.addRow("Refresh rate (ms)", self.refresh_rate)
        form.addRow("Chart visibility", self.chart_visible)
        form.addRow("Theme", self.theme)
        form.addRow("Log verbosity", self.log_verbosity)
        form.addRow("Default pair", self.default_pair)
        form.addRow("Default timeframe", self.default_tf)
        panel_layout.addLayout(form)

        self.status_hint = QLabel("Changes are applied in realtime.")
        self.status_hint.setObjectName("metricLabel")
        panel_layout.addWidget(self.status_hint)

        self.apply_btn = QPushButton("Apply now")
        self.apply_btn.clicked.connect(self._emit)
        panel_layout.addWidget(self.apply_btn)

        self.refresh_rate.valueChanged.connect(self._emit)
        self.chart_visible.stateChanged.connect(self._emit)
        self.theme.currentTextChanged.connect(self._emit)
        self.log_verbosity.currentTextChanged.connect(self._emit)
        self.default_pair.currentTextChanged.connect(self._emit)
        self.default_tf.currentTextChanged.connect(self._emit)

    def _emit(self) -> None:
        self.settings_changed.emit(
            {
                "refresh_rate_ms": self.refresh_rate.value(),
                "chart_visible": self.chart_visible.isChecked(),
                "theme": self.theme.currentText(),
                "log_verbosity": self.log_verbosity.currentText(),
                "default_pair": self.default_pair.currentText(),
                "default_timeframe": self.default_tf.currentText(),
            }
        )

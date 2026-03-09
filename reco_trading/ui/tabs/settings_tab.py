from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QSpinBox, QVBoxLayout, QWidget


class SettingsTab(QWidget):
    settings_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Interface Studio")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        layout.addWidget(QLabel("UI settings (impact only visual behaviour)"))

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

        form.addRow("Refresh rate (ms)", self.refresh_rate)
        form.addRow("Chart visibility", self.chart_visible)
        form.addRow("Theme", self.theme)
        form.addRow("Log verbosity", self.log_verbosity)
        layout.addLayout(form)

        self.refresh_rate.valueChanged.connect(self._emit)
        self.chart_visible.stateChanged.connect(self._emit)
        self.theme.currentTextChanged.connect(self._emit)
        self.log_verbosity.currentTextChanged.connect(self._emit)

    def _emit(self) -> None:
        self.settings_changed.emit(
            {
                "refresh_rate_ms": self.refresh_rate.value(),
                "chart_visible": self.chart_visible.isChecked(),
                "theme": self.theme.currentText(),
                "log_verbosity": self.log_verbosity.currentText(),
            }
        )

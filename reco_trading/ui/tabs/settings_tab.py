from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class SettingsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.layout = QVBoxLayout(self)
        form = QFormLayout()

        self.refresh_rate = QComboBox()
        self.refresh_rate.addItems(["500 ms", "1000 ms", "2000 ms", "5000 ms"])
        self.chart_visible = QCheckBox("Show chart")
        self.chart_visible.setChecked(True)
        self.theme_toggle = QComboBox()
        self.theme_toggle.addItems(["Dark", "Dark High Contrast"])
        self.log_verbosity = QComboBox()
        self.log_verbosity.addItems(["INFO", "WARNING", "ERROR", "DEBUG"])

        form.addRow("Refresh Rate", self.refresh_rate)
        form.addRow("Chart Visibility", self.chart_visible)
        form.addRow("Theme", self.theme_toggle)
        form.addRow("Log Verbosity", self.log_verbosity)

        self.layout.addLayout(form)
        self.apply_btn = QPushButton("Apply UI Settings")
        self.layout.addWidget(self.apply_btn)
        self.status = QLabel("UI-only settings. Trading engine remains untouched.")
        self.layout.addWidget(self.status)

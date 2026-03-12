from __future__ import annotations

import os
from pathlib import Path

from dotenv import set_key
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsTab(QWidget):
    settings_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Interface Studio")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        description = QLabel("Customize visual behavior, refresh and active API credentials")
        description.setObjectName("metricLabel")
        layout.addWidget(description)

        panel = QFrame()
        panel.setObjectName("panelCard")
        layout.addWidget(panel)
        panel_layout = QVBoxLayout(panel)

        visual_panel = QFrame()
        visual_panel.setObjectName("metricCard")
        visual_layout = QVBoxLayout(visual_panel)
        visual_layout.addWidget(self._section_title("Visual & Runtime"))

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
        visual_layout.addLayout(form)

        creds_panel = QFrame()
        creds_panel.setObjectName("metricCard")
        creds_layout = QVBoxLayout(creds_panel)
        creds_layout.addWidget(self._section_title("Active API Keys"))

        creds_form = QGridLayout()
        self.api_key = QLineEdit()
        self.api_key.setPlaceholderText("BINANCE API KEY")
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)

        self.api_secret = QLineEdit()
        self.api_secret.setPlaceholderText("BINANCE API SECRET")
        self.api_secret.setEchoMode(QLineEdit.EchoMode.Password)

        self.toggle_key_btn = QPushButton("Show")
        self.toggle_secret_btn = QPushButton("Show")
        self.toggle_key_btn.clicked.connect(lambda: self._toggle_visibility(self.api_key, self.toggle_key_btn))
        self.toggle_secret_btn.clicked.connect(lambda: self._toggle_visibility(self.api_secret, self.toggle_secret_btn))

        creds_form.addWidget(QLabel("API Key"), 0, 0)
        creds_form.addWidget(self.api_key, 0, 1)
        creds_form.addWidget(self.toggle_key_btn, 0, 2)
        creds_form.addWidget(QLabel("API Secret"), 1, 0)
        creds_form.addWidget(self.api_secret, 1, 1)
        creds_form.addWidget(self.toggle_secret_btn, 1, 2)
        creds_layout.addLayout(creds_form)

        creds_actions = QHBoxLayout()
        self.load_keys_btn = QPushButton("Load current keys")
        self.save_keys_btn = QPushButton("Save keys")
        creds_actions.addWidget(self.load_keys_btn)
        creds_actions.addWidget(self.save_keys_btn)
        creds_actions.addStretch(1)
        creds_layout.addLayout(creds_actions)

        panel_layout.addWidget(visual_panel)
        panel_layout.addWidget(creds_panel)

        self.status_hint = QLabel("Changes are applied in realtime. API keys can be edited and persisted to .env")
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

        self.load_keys_btn.clicked.connect(self._load_keys_from_env)
        self.save_keys_btn.clicked.connect(self._save_keys_to_env)

        self._load_keys_from_env()

    def _section_title(self, text: str) -> QLabel:
        title = QLabel(text)
        title.setObjectName("smallMetricValue")
        return title

    def _toggle_visibility(self, field: QLineEdit, button: QPushButton) -> None:
        hidden = field.echoMode() == QLineEdit.EchoMode.Password
        field.setEchoMode(QLineEdit.EchoMode.Normal if hidden else QLineEdit.EchoMode.Password)
        button.setText("Hide" if hidden else "Show")

    def _load_keys_from_env(self) -> None:
        self.api_key.setText(os.getenv("BINANCE_API_KEY", ""))
        self.api_secret.setText(os.getenv("BINANCE_API_SECRET", ""))
        self.status_hint.setText("API keys loaded from environment.")
        self._emit()

    def _save_keys_to_env(self) -> None:
        key = self.api_key.text().strip()
        secret = self.api_secret.text().strip()
        os.environ["BINANCE_API_KEY"] = key
        os.environ["BINANCE_API_SECRET"] = secret

        env_path = Path(".env")
        try:
            set_key(str(env_path), "BINANCE_API_KEY", key)
            set_key(str(env_path), "BINANCE_API_SECRET", secret)
            self.status_hint.setText("API keys saved to .env and active in current session.")
        except Exception as exc:  # noqa: BLE001
            self.status_hint.setText(f"Keys updated in memory only (could not write .env): {exc}")

        self._emit()

    def _emit(self) -> None:
        self.settings_changed.emit(
            {
                "refresh_rate_ms": self.refresh_rate.value(),
                "chart_visible": self.chart_visible.isChecked(),
                "theme": self.theme.currentText(),
                "log_verbosity": self.log_verbosity.currentText(),
                "default_pair": self.default_pair.currentText(),
                "default_timeframe": self.default_tf.currentText(),
                "binance_api_key": self.api_key.text().strip(),
                "binance_api_secret": self.api_secret.text().strip(),
            }
        )

from __future__ import annotations

import os
from pathlib import Path

from dotenv import set_key
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class SettingsTab(QWidget):
    settings_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._applying_state = False
        self._symbol_capital_limits: dict[str, float] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        header = QFrame()
        header.setObjectName("panelCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        title = QLabel("Settings Studio")
        title.setObjectName("sectionTitle")
        description = QLabel("Professional runtime configuration, risk profile and credential management")
        description.setObjectName("metricLabel")
        header_layout.addWidget(title)
        header_layout.addWidget(description)
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        layout.addLayout(grid)

        runtime_panel = self._build_runtime_panel()
        profile_panel = self._build_profile_panel()
        risk_panel = self._build_risk_panel()
        creds_panel = self._build_credentials_panel()

        grid.addWidget(runtime_panel, 0, 0)
        grid.addWidget(profile_panel, 0, 1)
        grid.addWidget(risk_panel, 1, 0)
        grid.addWidget(creds_panel, 1, 1)

        footer = QFrame()
        footer.setObjectName("panelCard")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 10, 12, 10)

        self.status_hint = QLabel("Changes are applied in realtime and synchronized to runtime controls.")
        self.status_hint.setObjectName("metricLabel")
        footer_layout.addWidget(self.status_hint, 1)

        self.apply_btn = QPushButton("Apply settings now")
        self.apply_btn.clicked.connect(self._emit)
        footer_layout.addWidget(self.apply_btn)
        layout.addWidget(footer)

        self._bind_signals()

        self._pulse = QPropertyAnimation(self.apply_btn, b"windowOpacity", self)
        self._pulse.setDuration(240)
        self._pulse.setStartValue(0.6)
        self._pulse.setEndValue(1.0)
        self._pulse.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._load_keys_from_env()
        self._on_default_pair_changed(self.default_pair.currentText())
        self._apply_investment_preset(self.investment_mode.currentText())

    def _build_runtime_panel(self) -> QFrame:
        panel = self._panel("Runtime Experience")
        layout = panel.layout()
        assert isinstance(layout, QVBoxLayout)

        self.refresh_rate = QSpinBox()
        self.refresh_rate.setRange(250, 5000)
        self.refresh_rate.setValue(1000)

        self.chart_visible = QCheckBox("Enable chart rendering")
        self.chart_visible.setChecked(True)

        self.theme = QComboBox()
        self.theme.addItems(["Dark", "Dark+Contrast"])

        self.log_verbosity = QComboBox()
        self.log_verbosity.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])

        form.addRow("Refresh rate (ms)", self.refresh_rate)
        form.addRow("Theme", self.theme)
        form.addRow("Log verbosity", self.log_verbosity)
        form.addRow("Chart", self.chart_visible)
        layout.addLayout(form)
        return panel

    def _build_profile_panel(self) -> QFrame:
        panel = self._panel("Trading Profile")
        layout = panel.layout()
        assert isinstance(layout, QVBoxLayout)

        form = QFormLayout()
        self.default_pair = QComboBox()
        self.default_pair.addItems(["BTC/USDT", "ETH/USDT", "SOL/USDT"])

        self.default_tf = QComboBox()
        self.default_tf.addItems(["1m / 5m", "5m / 15m", "15m / 1h"])

        self.investment_mode = QComboBox()
        self.investment_mode.addItems(["Conservative", "Balanced", "Aggressive", "Custom"])

        form.addRow("Default pair", self.default_pair)
        form.addRow("Default timeframe", self.default_tf)
        form.addRow("Investment mode", self.investment_mode)
        layout.addLayout(form)
        return panel

    def _build_risk_panel(self) -> QFrame:
        panel = self._panel("Risk & Capital Controls")
        layout = panel.layout()
        assert isinstance(layout, QVBoxLayout)

        form = QFormLayout()

        self.capital_limit = QDoubleSpinBox()
        self.capital_limit.setRange(0.0, 10_000_000.0)
        self.capital_limit.setDecimals(2)
        self.capital_limit.setValue(0.0)
        self.capital_limit.setSuffix(" USDT")

        self.symbol_budget = QDoubleSpinBox()
        self.symbol_budget.setRange(0.0, 10_000_000.0)
        self.symbol_budget.setDecimals(2)
        self.symbol_budget.setValue(0.0)
        self.symbol_budget.setSuffix(" USDT")

        self.risk_per_trade = QDoubleSpinBox()
        self.risk_per_trade.setRange(0.1, 10.0)
        self.risk_per_trade.setDecimals(2)
        self.risk_per_trade.setValue(1.0)
        self.risk_per_trade.setSuffix(" %")

        self.max_allocation = QDoubleSpinBox()
        self.max_allocation.setRange(1.0, 100.0)
        self.max_allocation.setDecimals(1)
        self.max_allocation.setValue(20.0)
        self.max_allocation.setSuffix(" %")

        form.addRow("Capital limit", self.capital_limit)
        form.addRow("Per-pair budget", self.symbol_budget)
        form.addRow("Risk per trade", self.risk_per_trade)
        form.addRow("Max allocation", self.max_allocation)
        layout.addLayout(form)

        self.simulation_hint = QLabel("Estimated max order: 0.00 USDT")
        self.simulation_hint.setObjectName("smallMetricValue")
        layout.addWidget(self.simulation_hint)
        return panel

    def _build_credentials_panel(self) -> QFrame:
        panel = self._panel("API Credentials")
        layout = panel.layout()
        assert isinstance(layout, QVBoxLayout)

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
        layout.addLayout(creds_form)

        actions = QHBoxLayout()
        self.load_keys_btn = QPushButton("Load current keys")
        self.save_keys_btn = QPushButton("Save keys")
        actions.addWidget(self.load_keys_btn)
        actions.addWidget(self.save_keys_btn)
        actions.addStretch(1)
        layout.addLayout(actions)
        return panel

    def _panel(self, title_text: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panelCard")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)
        title = QLabel(title_text)
        title.setObjectName("smallMetricValue")
        panel_layout.addWidget(title)
        return panel

    def _bind_signals(self) -> None:
        self.refresh_rate.valueChanged.connect(self._emit)
        self.chart_visible.stateChanged.connect(self._emit)
        self.theme.currentTextChanged.connect(self._emit)
        self.log_verbosity.currentTextChanged.connect(self._emit)
        self.default_pair.currentTextChanged.connect(self._on_default_pair_changed)
        self.default_tf.currentTextChanged.connect(self._emit)
        self.investment_mode.currentTextChanged.connect(self._apply_investment_preset)
        self.capital_limit.valueChanged.connect(self._emit)
        self.symbol_budget.valueChanged.connect(self._emit)
        self.risk_per_trade.valueChanged.connect(self._emit)
        self.max_allocation.valueChanged.connect(self._emit)

        self.load_keys_btn.clicked.connect(self._load_keys_from_env)
        self.save_keys_btn.clicked.connect(self._save_keys_to_env)

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

    def _apply_investment_preset(self, mode: str) -> None:
        normalized = mode.strip().lower()
        if normalized == "conservative":
            self.risk_per_trade.setValue(0.5)
            self.max_allocation.setValue(10.0)
        elif normalized == "balanced":
            self.risk_per_trade.setValue(1.0)
            self.max_allocation.setValue(20.0)
        elif normalized == "aggressive":
            self.risk_per_trade.setValue(2.0)
            self.max_allocation.setValue(35.0)
        self._emit()

    def _on_default_pair_changed(self, pair: str) -> None:
        budget = float(self._symbol_capital_limits.get(pair.strip(), 0.0))
        self.symbol_budget.blockSignals(True)
        self.symbol_budget.setValue(max(budget, 0.0))
        self.symbol_budget.blockSignals(False)
        self._emit()

    def _emit(self) -> None:
        if self._applying_state:
            return

        pair = self.default_pair.currentText().strip()
        budget_value = self.symbol_budget.value()
        if pair:
            if budget_value > 0:
                self._symbol_capital_limits[pair] = budget_value
            elif pair in self._symbol_capital_limits:
                self._symbol_capital_limits.pop(pair)

        capital_limit = self.capital_limit.value()
        effective_capital = budget_value if budget_value > 0 else capital_limit
        estimated_order = effective_capital * (self.max_allocation.value() / 100.0)
        self.simulation_hint.setText(f"Estimated max order: {estimated_order:.2f} USDT")

        payload = {
            "refresh_rate_ms": self.refresh_rate.value(),
            "chart_visible": self.chart_visible.isChecked(),
            "theme": self.theme.currentText(),
            "log_verbosity": self.log_verbosity.currentText(),
            "default_pair": self.default_pair.currentText(),
            "default_timeframe": self.default_tf.currentText(),
            "investment_mode": self.investment_mode.currentText(),
            "capital_limit_usdt": self.capital_limit.value(),
            "symbol_capital_limits": dict(self._symbol_capital_limits),
            "risk_per_trade_fraction": self.risk_per_trade.value() / 100.0,
            "max_trade_balance_fraction": self.max_allocation.value() / 100.0,
            "binance_api_key": self.api_key.text().strip(),
            "binance_api_secret": self.api_secret.text().strip(),
        }
        self.settings_changed.emit(payload)
        if hasattr(self, "_pulse"):
            self._pulse.stop()
            self._pulse.start()

        payload = {
            "refresh_rate_ms": self.refresh_rate.value(),
            "chart_visible": self.chart_visible.isChecked(),
            "theme": self.theme.currentText(),
            "log_verbosity": self.log_verbosity.currentText(),
            "default_pair": self.default_pair.currentText(),
            "default_timeframe": self.default_tf.currentText(),
            "investment_mode": self.investment_mode.currentText(),
            "capital_limit_usdt": self.capital_limit.value(),
            "symbol_capital_limits": dict(self._symbol_capital_limits),
            "risk_per_trade_fraction": self.risk_per_trade.value() / 100.0,
            "max_trade_balance_fraction": self.max_allocation.value() / 100.0,
            "binance_api_key": self.api_key.text().strip(),
            "binance_api_secret": self.api_secret.text().strip(),
        }
        self.settings_changed.emit(payload)
        self._pulse.stop()
        self._pulse.start()

    def update_state(self, state: dict) -> None:
        runtime = state.get("runtime_settings")
        if not isinstance(runtime, dict) or not runtime:
            return

        self._applying_state = True
        try:
            mode = str(runtime.get("investment_mode", "")).strip()
            if mode and mode in {self.investment_mode.itemText(i) for i in range(self.investment_mode.count())}:
                self.investment_mode.setCurrentText(mode)

            theme = str(runtime.get("theme", "")).strip()
            if theme and theme in {self.theme.itemText(i) for i in range(self.theme.count())}:
                self.theme.setCurrentText(theme)

            verbosity = str(runtime.get("log_verbosity", "")).strip()
            if verbosity and verbosity in {self.log_verbosity.itemText(i) for i in range(self.log_verbosity.count())}:
                self.log_verbosity.setCurrentText(verbosity)

            default_pair = str(runtime.get("default_pair", "")).strip()
            if default_pair and default_pair in {self.default_pair.itemText(i) for i in range(self.default_pair.count())}:
                self.default_pair.setCurrentText(default_pair)

            timeframe = str(runtime.get("default_timeframe", "")).strip()
            if timeframe and timeframe in {self.default_tf.itemText(i) for i in range(self.default_tf.count())}:
                self.default_tf.setCurrentText(timeframe)

            capital_limit = float(runtime.get("capital_limit_usdt", 0.0) or 0.0)
            self.capital_limit.setValue(max(capital_limit, 0.0))

            risk_fraction = float(runtime.get("risk_per_trade_fraction", 0.01) or 0.01)
            self.risk_per_trade.setValue(max(risk_fraction * 100.0, 0.1))

            max_trade_fraction = float(runtime.get("max_trade_balance_fraction", 0.2) or 0.2)
            self.max_allocation.setValue(max(max_trade_fraction * 100.0, 1.0))

            chart_visible = runtime.get("chart_visible")
            if isinstance(chart_visible, bool):
                self.chart_visible.setChecked(chart_visible)

            refresh_rate = int(runtime.get("refresh_rate_ms", 1000) or 1000)
            self.refresh_rate.setValue(max(250, min(refresh_rate, 5000)))

            symbol_limits = runtime.get("symbol_capital_limits", {})
            if isinstance(symbol_limits, dict):
                self._symbol_capital_limits = {str(k): float(v) for k, v in symbol_limits.items() if float(v) > 0}
                self._on_default_pair_changed(self.default_pair.currentText())
        except Exception:
            pass
        finally:
            self._applying_state = False
        self._emit()

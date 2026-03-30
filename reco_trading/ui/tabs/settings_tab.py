from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
        self._applying_state = False
        self._symbol_capital_limits: dict[str, float] = {}
        self._base_assets = ["BTC", "ETH", "SOL", "BNB", "XRP"]
        self._symbol_filter_configs: dict[str, dict] = {}
        self._load_default_filter_configs()
        layout = QVBoxLayout(self)
        title = QLabel("Interface Studio")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        description = QLabel("Customize visual behavior, refresh cadence and session-safe controls")
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
        self.quote_currency = QComboBox()
        self.quote_currency.addItems(["USDT", "USDC", "FDUSD", "BUSD", "TUSD", "DAI", "EUR", "BTC", "ETH", "BNB"])
        self.default_pair.addItems([f"{asset}/USDT" for asset in self._base_assets[:3]])
        self.default_tf = QComboBox()
        self.default_tf.addItems(["1m / 5m", "5m / 15m", "15m / 1h"])
        self.investment_mode = QComboBox()
        self.investment_mode.addItems(["Conservative", "Balanced", "Aggressive", "Auto-Optimized", "Custom"])
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
        self.reserve_ratio = QDoubleSpinBox()
        self.reserve_ratio.setRange(0.0, 90.0)
        self.reserve_ratio.setDecimals(1)
        self.reserve_ratio.setValue(15.0)
        self.reserve_ratio.setSuffix(" %")
        self.cash_buffer = QDoubleSpinBox()
        self.cash_buffer.setRange(0.0, 1_000_000.0)
        self.cash_buffer.setDecimals(2)
        self.cash_buffer.setValue(10.0)
        self.cash_buffer.setSuffix(" USDT")

        form.addRow("Refresh rate (ms)", self.refresh_rate)
        form.addRow("Chart visibility", self.chart_visible)
        form.addRow("Theme", self.theme)
        form.addRow("Log verbosity", self.log_verbosity)
        form.addRow("Account currency", self.quote_currency)
        form.addRow("Default pair", self.default_pair)
        form.addRow("Default timeframe", self.default_tf)
        form.addRow("Investment mode", self.investment_mode)
        form.addRow("Capital limit", self.capital_limit)
        form.addRow("Per-pair budget", self.symbol_budget)
        form.addRow("Risk per trade", self.risk_per_trade)
        form.addRow("Max allocation", self.max_allocation)
        form.addRow("Reserve ratio", self.reserve_ratio)
        form.addRow("Cash buffer", self.cash_buffer)
        visual_layout.addLayout(form)

        self.simulation_hint = QLabel("Estimated max order: 0.00 USDT")
        self.simulation_hint.setObjectName("metricLabel")
        visual_layout.addWidget(self.simulation_hint)
        self.optimization_hint = QLabel("Optimizer: waiting for account snapshot")
        self.optimization_hint.setObjectName("metricLabel")
        visual_layout.addWidget(self.optimization_hint)
        self._filter_info_label = QLabel("Filters: ADX≥22 | RSI:45/55 | Vol:0.65/1.10 | ATR:0.003/0.018 | Conf≥55%")
        self._filter_info_label.setObjectName("metricLabel")
        self._filter_info_label.setStyleSheet("color: #5a8dff;")
        visual_layout.addWidget(self._filter_info_label)

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
        self.save_keys_btn = QPushButton("Apply to session")
        creds_actions.addWidget(self.load_keys_btn)
        creds_actions.addWidget(self.save_keys_btn)
        creds_actions.addStretch(1)
        creds_layout.addLayout(creds_actions)

        panel_layout.addWidget(visual_panel)
        panel_layout.addWidget(creds_panel)

        self.status_hint = QLabel("Changes are applied in realtime. API credentials are never broadcast in runtime settings.")
        self.status_hint.setObjectName("metricLabel")
        panel_layout.addWidget(self.status_hint)

        self.apply_btn = QPushButton("Apply now")
        self.apply_btn.clicked.connect(self._emit)
        panel_layout.addWidget(self.apply_btn)

        self.refresh_rate.valueChanged.connect(self._emit)
        self.chart_visible.stateChanged.connect(self._emit)
        self.theme.currentTextChanged.connect(self._emit)
        self.log_verbosity.currentTextChanged.connect(self._emit)
        self.quote_currency.currentTextChanged.connect(self._on_quote_currency_changed)
        self.default_pair.currentTextChanged.connect(self._on_default_pair_changed)
        self.default_tf.currentTextChanged.connect(self._emit)
        self.investment_mode.currentTextChanged.connect(self._apply_investment_preset)
        self.capital_limit.valueChanged.connect(self._emit)
        self.symbol_budget.valueChanged.connect(self._emit)
        self.risk_per_trade.valueChanged.connect(self._emit)
        self.max_allocation.valueChanged.connect(self._emit)
        self.reserve_ratio.valueChanged.connect(self._emit)
        self.cash_buffer.valueChanged.connect(self._emit)

        self.load_keys_btn.clicked.connect(self._load_keys_from_env)
        self.save_keys_btn.clicked.connect(self._save_keys_to_env)

        self._load_keys_from_env()
        self._on_quote_currency_changed(self.quote_currency.currentText())
        self._on_default_pair_changed(self.default_pair.currentText())
        self._apply_investment_preset(self.investment_mode.currentText())
        self._load_default_filter_configs()

    def _load_default_filter_configs(self) -> None:
        self._symbol_filter_configs = {
            "BTCUSDT": {
                "adx_threshold": 22.0,
                "rsi_buy_threshold": 55,
                "rsi_sell_threshold": 45,
                "volume_buy_threshold": 1.10,
                "volume_sell_threshold": 0.65,
                "atr_low_threshold": 0.003,
                "atr_high_threshold": 0.018,
                "stop_loss_atr_multiplier": 1.5,
                "take_profit_atr_multiplier": 2.5,
                "min_confidence": 0.55,
            },
            "ETHUSDT": {
                "adx_threshold": 22.0,
                "rsi_buy_threshold": 54,
                "rsi_sell_threshold": 46,
                "volume_buy_threshold": 1.05,
                "volume_sell_threshold": 0.70,
                "atr_low_threshold": 0.004,
                "atr_high_threshold": 0.020,
                "stop_loss_atr_multiplier": 1.6,
                "take_profit_atr_multiplier": 2.6,
                "min_confidence": 0.52,
            },
            "SOLUSDT": {
                "adx_threshold": 24.0,
                "rsi_buy_threshold": 58,
                "rsi_sell_threshold": 42,
                "volume_buy_threshold": 1.15,
                "volume_sell_threshold": 0.60,
                "atr_low_threshold": 0.005,
                "atr_high_threshold": 0.025,
                "stop_loss_atr_multiplier": 1.8,
                "take_profit_atr_multiplier": 3.0,
                "min_confidence": 0.58,
            },
            "BNBUSDT": {
                "adx_threshold": 22.0,
                "rsi_buy_threshold": 55,
                "rsi_sell_threshold": 45,
                "volume_buy_threshold": 1.08,
                "volume_sell_threshold": 0.68,
                "atr_low_threshold": 0.0035,
                "atr_high_threshold": 0.022,
                "stop_loss_atr_multiplier": 1.5,
                "take_profit_atr_multiplier": 2.5,
                "min_confidence": 0.54,
            },
            "XRPUSDT": {
                "adx_threshold": 20.0,
                "rsi_buy_threshold": 52,
                "rsi_sell_threshold": 48,
                "volume_buy_threshold": 1.00,
                "volume_sell_threshold": 0.75,
                "atr_low_threshold": 0.004,
                "atr_high_threshold": 0.025,
                "stop_loss_atr_multiplier": 2.0,
                "take_profit_atr_multiplier": 3.2,
                "min_confidence": 0.50,
            },
            "default": {
                "adx_threshold": 22.0,
                "rsi_buy_threshold": 55,
                "rsi_sell_threshold": 45,
                "volume_buy_threshold": 1.10,
                "volume_sell_threshold": 0.65,
                "atr_low_threshold": 0.003,
                "atr_high_threshold": 0.018,
                "stop_loss_atr_multiplier": 1.5,
                "take_profit_atr_multiplier": 2.5,
                "min_confidence": 0.55,
            },
        }

    def _section_title(self, text: str) -> QLabel:
        title = QLabel(text)
        title.setObjectName("smallMetricValue")
        return title

    def _toggle_visibility(self, field: QLineEdit, button: QPushButton) -> None:
        hidden = field.echoMode() == QLineEdit.EchoMode.Password
        field.setEchoMode(QLineEdit.EchoMode.Normal if hidden else QLineEdit.EchoMode.Password)
        button.setText("Hide" if hidden else "Show")

    def _load_keys_from_env(self) -> None:
        key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_API_SECRET", "")
        self.api_key.clear()
        self.api_secret.clear()
        self.api_key.setPlaceholderText(_masked_secret_hint("BINANCE API KEY", key))
        self.api_secret.setPlaceholderText(_masked_secret_hint("BINANCE API SECRET", secret))
        self.status_hint.setText("API credential presence loaded from environment. Values stay local to this form.")

    def _save_keys_to_env(self) -> None:
        key = self.api_key.text().strip()
        secret = self.api_secret.text().strip()
        if key:
            os.environ["BINANCE_API_KEY"] = key
        if secret:
            os.environ["BINANCE_API_SECRET"] = secret

        self.api_key.clear()
        self.api_secret.clear()
        self.api_key.setPlaceholderText(_masked_secret_hint("BINANCE API KEY", os.getenv("BINANCE_API_KEY", "")))
        self.api_secret.setPlaceholderText(_masked_secret_hint("BINANCE API SECRET", os.getenv("BINANCE_API_SECRET", "")))
        self.status_hint.setText(
            f"Credentials applied to current process only. Persist them manually in {Path('.env')} if you need restart durability."
        )

    def _apply_investment_preset(self, mode: str) -> None:
        normalized = mode.strip().lower()
        if normalized == "conservative":
            self.risk_per_trade.setValue(0.5)
            self.max_allocation.setValue(10.0)
            self.reserve_ratio.setValue(25.0)
            self.cash_buffer.setValue(15.0)
        elif normalized == "balanced":
            self.risk_per_trade.setValue(1.0)
            self.max_allocation.setValue(20.0)
            self.reserve_ratio.setValue(15.0)
            self.cash_buffer.setValue(10.0)
        elif normalized == "aggressive":
            self.risk_per_trade.setValue(2.0)
            self.max_allocation.setValue(35.0)
            self.reserve_ratio.setValue(8.0)
            self.cash_buffer.setValue(5.0)
        elif normalized == "auto-optimized":
            self.risk_per_trade.setValue(0.8)
            self.max_allocation.setValue(18.0)
            self.reserve_ratio.setValue(18.0)
            self.cash_buffer.setValue(12.0)
        self._emit()

    def _on_default_pair_changed(self, pair: str) -> None:
        budget = float(self._symbol_capital_limits.get(pair.strip(), 0.0))
        self.symbol_budget.blockSignals(True)
        self.symbol_budget.setValue(max(budget, 0.0))
        self.symbol_budget.blockSignals(False)
        
        self._apply_filter_config_for_pair(pair)
        self._emit()
    
    def _apply_filter_config_for_pair(self, pair: str) -> None:
        normalized_pair = pair.strip().replace("/", "").upper()
        config = self._symbol_filter_configs.get(normalized_pair, self._symbol_filter_configs["default"])
        
        self.min_confidence_filter = config.get("min_confidence", 0.55)
        self.adx_filter = config.get("adx_threshold", 22.0)
        self.rsi_buy_filter = config.get("rsi_buy_threshold", 55)
        self.rsi_sell_filter = config.get("rsi_sell_threshold", 45)
        self.volume_buy_filter = config.get("volume_buy_threshold", 1.10)
        self.volume_sell_filter = config.get("volume_sell_threshold", 0.65)
        self.atr_low_filter = config.get("atr_low_threshold", 0.003)
        self.atr_high_filter = config.get("atr_high_threshold", 0.018)
        
        if hasattr(self, '_filter_info_label'):
            self._filter_info_label.setText(
                f"Filters: ADX≥{config['adx_threshold']} | RSI:{config['rsi_sell_threshold']}/{config['rsi_buy_threshold']} | "
                f"Vol:{config['volume_sell_threshold']}/{config['volume_buy_threshold']} | "
                f"ATR:{config['atr_low_threshold']:.3f}/{config['atr_high_threshold']:.3f} | "
                f"Conf≥{config['min_confidence']:.0%}"
            )

    def _on_quote_currency_changed(self, currency: str) -> None:
        quote = currency.strip().upper() or "USDT"
        current_pair = self.default_pair.currentText().strip()
        current_base = current_pair.split("/", 1)[0] if "/" in current_pair else ""
        available_pairs = [f"{asset}/{quote}" for asset in self._base_assets if asset != quote]
        if not available_pairs:
            available_pairs = [f"BTC/{quote}"]

        self.default_pair.blockSignals(True)
        self.default_pair.clear()
        self.default_pair.addItems(available_pairs)
        if current_base and f"{current_base}/{quote}" in available_pairs:
            self.default_pair.setCurrentText(f"{current_base}/{quote}")
        self.default_pair.blockSignals(False)

        currency_suffix = f" {quote}"
        self.capital_limit.setSuffix(currency_suffix)
        self.symbol_budget.setSuffix(currency_suffix)
        self.cash_buffer.setSuffix(currency_suffix)
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
        account_currency = self.quote_currency.currentText().strip().upper() or "USDT"
        self.simulation_hint.setText(f"Estimated max order: {estimated_order:.2f} {account_currency}")

        pair_normalized = pair.replace("/", "").upper()
        filter_config = self._symbol_filter_configs.get(pair_normalized, self._symbol_filter_configs["default"])

        self.settings_changed.emit(
            {
                "refresh_rate_ms": self.refresh_rate.value(),
                "chart_visible": self.chart_visible.isChecked(),
                "theme": self.theme.currentText(),
                "log_verbosity": self.log_verbosity.currentText(),
                "account_currency": account_currency,
                "default_pair": self.default_pair.currentText(),
                "default_timeframe": self.default_tf.currentText(),
                "investment_mode": self.investment_mode.currentText(),
                "dynamic_exit_enabled": self.investment_mode.currentText().strip().lower() == "auto-optimized",
                "capital_limit_usdt": self.capital_limit.value(),
                "symbol_capital_limits": dict(self._symbol_capital_limits),
                "risk_per_trade_fraction": self.risk_per_trade.value() / 100.0,
                "max_trade_balance_fraction": self.max_allocation.value() / 100.0,
                "capital_reserve_ratio": self.reserve_ratio.value() / 100.0,
                "min_cash_buffer_usdt": self.cash_buffer.value(),
                "filter_config": filter_config,
            }
        )

    def update_state(self, state: dict) -> None:
        runtime = state.get("runtime_settings")
        if not isinstance(runtime, dict) or not runtime:
            return
        self._applying_state = True
        try:
            mode = str(runtime.get("investment_mode", "")).strip()
            if mode and mode in {self.investment_mode.itemText(i) for i in range(self.investment_mode.count())}:
                self.investment_mode.setCurrentText(mode)
            account_currency = str(runtime.get("account_currency", "")).strip().upper()
            if account_currency and account_currency in {self.quote_currency.itemText(i) for i in range(self.quote_currency.count())}:
                self.quote_currency.setCurrentText(account_currency)
            capital_limit = float(runtime.get("capital_limit_usdt", 0.0) or 0.0)
            self.capital_limit.setValue(max(capital_limit, 0.0))
            risk_fraction = float(runtime.get("risk_per_trade_fraction", 0.01) or 0.01)
            self.risk_per_trade.setValue(max(risk_fraction * 100.0, 0.1))
            max_trade_fraction = float(runtime.get("max_trade_balance_fraction", 0.2) or 0.2)
            self.max_allocation.setValue(max(max_trade_fraction * 100.0, 1.0))
            reserve_ratio = float(runtime.get("capital_reserve_ratio", 0.15) or 0.15)
            self.reserve_ratio.setValue(max(reserve_ratio * 100.0, 0.0))
            cash_buffer = float(runtime.get("min_cash_buffer_usdt", 10.0) or 10.0)
            self.cash_buffer.setValue(max(cash_buffer, 0.0))
            symbol_limits = runtime.get("symbol_capital_limits", {})
            if isinstance(symbol_limits, dict):
                self._symbol_capital_limits = {str(k): float(v) for k, v in symbol_limits.items() if float(v) > 0}
                self._on_default_pair_changed(self.default_pair.currentText())
            optimized_risk = runtime.get("risk_per_trade_fraction")
            optimized_alloc = runtime.get("max_trade_balance_fraction")
            optimized_capital = runtime.get("capital_limit_usdt")
            if optimized_risk is not None and optimized_alloc is not None:
                self.optimization_hint.setText(
                    "Optimizer: "
                    f"risk {float(optimized_risk)*100:.2f}% | "
                    f"allocation {float(optimized_alloc)*100:.2f}% | "
                    f"capital {float(optimized_capital or 0.0):.2f} {self.quote_currency.currentText().strip().upper() or 'USDT'}"
                )
        except Exception:
            pass
        finally:
            self._applying_state = False
        self._emit()

    def set_default_pair(self, pair: str) -> None:
        normalized = pair.strip()
        if not normalized:
            return
        if self.default_pair.findText(normalized) < 0:
            self.default_pair.addItem(normalized)
        self._applying_state = True
        try:
            self.default_pair.setCurrentText(normalized)
            self._on_default_pair_changed(normalized)
        finally:
            self._applying_state = False


def _masked_secret_hint(label: str, value: str) -> str:
    if not value:
        return f"{label} (not configured)"
    trimmed = value.strip()
    if len(trimmed) <= 8:
        return f"{label} (configured)"
    return f"{label} ({trimmed[:4]}…{trimmed[-4:]})"

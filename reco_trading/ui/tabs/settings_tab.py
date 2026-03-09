from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFormLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from reco_trading.config.settings import Settings


class SettingsTab(QWidget):
    FIELDS = ["TRADING_SYMBOL", "PRIMARY_TIMEFRAME", "RISK_PER_TRADE_FRACTION", "COOLDOWN_MINUTES"]

    def __init__(self) -> None:
        super().__init__()
        self.inputs: dict[str, QLineEdit] = {}
        settings = Settings()
        defaults = {
            "TRADING_SYMBOL": settings.trading_symbol,
            "PRIMARY_TIMEFRAME": settings.primary_timeframe,
            "RISK_PER_TRADE_FRACTION": str(settings.risk_per_trade_fraction),
            "COOLDOWN_MINUTES": str(settings.cooldown_minutes),
        }
        layout = QVBoxLayout(self)
        form = QFormLayout()
        for field in self.FIELDS:
            inp = QLineEdit(defaults.get(field, ""))
            form.addRow(field, inp)
            self.inputs[field] = inp
        layout.addLayout(form)
        save = QPushButton("Save to .env")
        save.clicked.connect(self.save_env)
        layout.addWidget(save)

    def save_env(self) -> None:
        env_path = Path(".env")
        existing: dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    existing[k] = v
        for k, widget in self.inputs.items():
            if widget.text().strip():
                existing[k] = widget.text().strip()
        env_path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")

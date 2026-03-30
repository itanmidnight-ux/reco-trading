from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Any


@dataclass(slots=True)
class ConfigBundle:
    trading: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """YAML-backed configuration loader with defaults and validation."""

    def __init__(self, base_dir: str | Path = "config") -> None:
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)

    def load(self) -> ConfigBundle:
        return ConfigBundle(
            trading=self._validate_trading(self._read_yaml("trading.yaml")),
            risk=self._validate_risk(self._read_yaml("risk.yaml")),
            strategy=self._validate_strategy(self._read_yaml("strategy.yaml")),
        )

    def _read_yaml(self, name: str) -> dict[str, Any]:
        path = self.base_dir / name
        if not path.exists():
            self.logger.warning("config_missing file=%s", path)
            return {}
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:  # noqa: BLE001
            self.logger.error("config_load_failed file=%s error=%s", path, exc)
            return {}

    @staticmethod
    def _validate_trading(raw: dict[str, Any]) -> dict[str, Any]:
        defaults = {"symbol": "BTCUSDT", "timeframe": "5m", "max_concurrent_trades": 1, "loop_sleep_seconds": 10}
        out = {**defaults, **raw}
        out["max_concurrent_trades"] = max(int(out.get("max_concurrent_trades", 1)), 1)
        out["loop_sleep_seconds"] = max(int(out.get("loop_sleep_seconds", 10)), 1)
        return out

    @staticmethod
    def _validate_risk(raw: dict[str, Any]) -> dict[str, Any]:
        defaults = {"risk_per_trade_fraction": 0.01, "max_daily_loss_percent": 3.0, "max_drawdown_percent": 10.0}
        out = {**defaults, **raw}
        out["risk_per_trade_fraction"] = min(max(float(out.get("risk_per_trade_fraction", 0.01)), 0.0), 1.0)
        return out

    @staticmethod
    def _validate_strategy(raw: dict[str, Any]) -> dict[str, Any]:
        defaults = {"min_signal_confidence": 0.60, "hold_threshold": 0.45, "use_liquidity_filter": True}
        out = {**defaults, **raw}
        out["min_signal_confidence"] = min(max(float(out.get("min_signal_confidence", 0.60)), 0.0), 1.0)
        return out

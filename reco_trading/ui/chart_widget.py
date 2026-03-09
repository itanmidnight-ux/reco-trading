from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from typing import Any

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # noqa: BLE001
    Figure = None
    FigureCanvas = None


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._candles: deque[dict[str, float]] = deque(maxlen=120)
        self._last_ts: datetime | None = None
        self._last_signature: tuple[int, float, float] | None = None
        layout = QVBoxLayout(self)
        if Figure is None or FigureCanvas is None:
            layout.addWidget(QLabel("Chart unavailable (matplotlib not installed)."))
            self.figure = None
            self.canvas = None
            self.ax_price = None
            self.ax_volume = None
            return
        self.figure = Figure(figsize=(7, 4), facecolor="#1a1d26")
        self.ax_price = self.figure.add_subplot(211)
        self.ax_volume = self.figure.add_subplot(212, sharex=self.ax_price)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        if self.figure is None or self.canvas is None:
            return
        changed = False
        candles = snapshot.get("candles_5m") or snapshot.get("candles") or []
        if isinstance(candles, list) and candles:
            parsed: list[dict[str, float]] = []
            for item in candles[-120:]:
                try:
                    parsed.append(
                        {
                            "open": float(item.get("open")),
                            "high": float(item.get("high")),
                            "low": float(item.get("low")),
                            "close": float(item.get("close")),
                            "volume": float(item.get("volume", 0)),
                        }
                    )
                except (TypeError, ValueError, AttributeError):
                    continue
            if parsed:
                signature = self._build_signature(parsed)
                if signature != self._last_signature:
                    self._candles = deque(parsed, maxlen=120)
                    self._last_signature = signature
                    changed = True
        else:
            changed = self._append_synthetic(snapshot)
        if changed:
            self._draw()

    def _append_synthetic(self, snapshot: dict[str, Any]) -> bool:
        price = snapshot.get("current_price", snapshot.get("price"))
        try:
            close = float(price)
        except (TypeError, ValueError):
            return False
        now = datetime.utcnow().replace(second=0, microsecond=0)
        now -= timedelta(minutes=now.minute % 5)
        if self._last_ts == now and self._candles:
            candle = self._candles[-1]
            before = (candle["high"], candle["low"], candle["close"], candle["volume"])
            candle["high"] = max(candle["high"], close)
            candle["low"] = min(candle["low"], close)
            candle["close"] = close
            candle["volume"] += float(snapshot.get("volume", 0) or 0)
            after = (candle["high"], candle["low"], candle["close"], candle["volume"])
            self._last_signature = self._build_signature(list(self._candles))
            return before != after
        prev = self._candles[-1]["close"] if self._candles else close
        self._candles.append({"open": prev, "high": max(prev, close), "low": min(prev, close), "close": close, "volume": float(snapshot.get("volume", 0) or 0)})
        self._last_ts = now
        self._last_signature = self._build_signature(list(self._candles))
        return True

    def _build_signature(self, candles: list[dict[str, float]]) -> tuple[int, float, float] | None:
        if not candles:
            return None
        last = candles[-1]
        return (len(candles), float(last.get("close", 0.0)), float(last.get("volume", 0.0)))

    def _draw(self) -> None:
        if not self._candles:
            return
        self.ax_price.clear()
        self.ax_volume.clear()
        self.ax_price.set_facecolor("#131722")
        self.ax_volume.set_facecolor("#131722")
        width = 0.6
        for i, c in enumerate(self._candles):
            color = "#16c784" if c["close"] >= c["open"] else "#ea3943"
            self.ax_price.vlines(i, c["low"], c["high"], color=color, linewidth=1)
            bottom = min(c["open"], c["close"])
            height = max(abs(c["close"] - c["open"]), 1e-8)
            self.ax_price.bar(i, height, width=width, bottom=bottom, color=color, align="center")
            self.ax_volume.bar(i, c["volume"], width=width, color=color, align="center", alpha=0.5)
        self.ax_price.set_title("BTC/USDT Candles (5m)", color="#e6e8ee")
        self.ax_price.tick_params(colors="#9aa4b2")
        self.ax_volume.tick_params(colors="#9aa4b2")
        self.ax_price.grid(color="#2a2f3a", linestyle="--", linewidth=0.5)
        self.ax_volume.grid(color="#2a2f3a", linestyle="--", linewidth=0.5)
        self.figure.tight_layout()
        self.canvas.draw_idle()

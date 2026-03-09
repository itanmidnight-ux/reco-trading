from __future__ import annotations

import json
from typing import Any
from urllib.request import urlopen

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # noqa: BLE001
    Figure = None
    FigureCanvas = None


class CandleFetchWorker(QThread):
    fetched = Signal(list)

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 120) -> None:
        super().__init__()
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

    def run(self) -> None:
        url = (
            "https://api.binance.com/api/v3/klines"
            f"?symbol={self.symbol}&interval={self.interval}&limit={self.limit}"
        )
        try:
            with urlopen(url, timeout=6) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            candles = [
                {
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
                for row in payload
                if len(row) >= 6
            ]
            self.fetched.emit(candles)
        except Exception:
            self.fetched.emit([])


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._candles: list[dict[str, float]] = []
        layout = QVBoxLayout(self)
        self._status = QLabel("Syncing market candles…")
        self._status.setObjectName("metricLabel")
        layout.addWidget(self._status)

        if Figure is None or FigureCanvas is None:
            layout.addWidget(QLabel("Chart unavailable: matplotlib backend missing"))
            self.figure = None
            self.canvas = None
            self.ax_price = None
            self.ax_volume = None
            return

        self.figure = Figure(figsize=(8, 4), facecolor="#141d35")
        self.ax_price = self.figure.add_subplot(211)
        self.ax_volume = self.figure.add_subplot(212, sharex=self.ax_price)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.fetch_timer = QTimer(self)
        self.fetch_timer.timeout.connect(self._request_public_candles)
        self.fetch_timer.start(15000)
        self._request_public_candles()

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        candles = snapshot.get("candles_5m", [])
        if candles:
            self._candles = [
                {
                    "open": float(c.get("open", 0.0)),
                    "high": float(c.get("high", 0.0)),
                    "low": float(c.get("low", 0.0)),
                    "close": float(c.get("close", 0.0)),
                    "volume": float(c.get("volume", 0.0)),
                }
                for c in candles[-120:]
            ]
            self._status.setText("Live candles from engine stream")
            self._plot_candles(self._candles)

    def _request_public_candles(self) -> None:
        self._worker = CandleFetchWorker()
        self._worker.fetched.connect(self._on_public_candles)
        self._worker.start()

    def _on_public_candles(self, candles: list[dict[str, float]]) -> None:
        if candles:
            self._candles = candles[-120:]
            self._status.setText("Public Binance sync active (auto-refresh 15s)")
            self._plot_candles(self._candles)
        elif not self._candles:
            self._status.setText("Waiting for candle data from engine/public endpoint")

    def _plot_candles(self, candles: list[dict[str, float]]) -> None:
        if not candles or not self.ax_price or not self.ax_volume or not self.canvas:
            return

        opens, highs, lows, closes, volumes = [], [], [], [], []
        for c in candles[-80:]:
            opens.append(float(c.get("open", 0)))
            highs.append(float(c.get("high", 0)))
            lows.append(float(c.get("low", 0)))
            closes.append(float(c.get("close", 0)))
            volumes.append(float(c.get("volume", 0)))

        self.ax_price.clear()
        self.ax_volume.clear()
        self.ax_price.set_facecolor("#141d35")
        self.ax_volume.set_facecolor("#141d35")

        for i, (o, h, l, cl) in enumerate(zip(opens, highs, lows, closes)):
            color = "#22d39b" if cl >= o else "#ff5f7b"
            self.ax_price.plot([i, i], [l, h], color=color, linewidth=1)
            lower = min(o, cl)
            height = max(abs(cl - o), 1e-8)
            self.ax_price.bar(i, height, bottom=lower, color=color, width=0.65)
            self.ax_volume.bar(i, volumes[i], color=color, width=0.65, alpha=0.45)

        ema = _ema(closes, 20)
        if ema:
            self.ax_price.plot(range(len(ema)), ema, color="#5a8dff", linewidth=1.2, label="EMA 20")
            self.ax_price.legend(loc="upper left", fontsize=8, frameon=False)

        self.ax_price.set_title("BTC/USDT 5m Candles", color="#edf2ff", fontsize=10)
        self.ax_price.tick_params(colors="#9fb2d9", labelsize=8)
        self.ax_volume.tick_params(colors="#9fb2d9", labelsize=7)
        self.figure.tight_layout()
        self.canvas.draw_idle()


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (period + 1)
    out = [values[0]]
    for value in values[1:]:
        out.append((value * alpha) + (out[-1] * (1 - alpha)))
    return out

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QPainter, QPen, QPicture
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlestickItem(pg.GraphicsObject):
    def __init__(self) -> None:
        super().__init__()
        self._picture = QPicture()

    def set_data(self, candles: list[Candle]) -> None:
        pic = QPicture()
        painter = QPainter(pic)
        for i, c in enumerate(candles):
            up = c.close >= c.open
            color = pg.mkColor("#22c55e" if up else "#ef4444")
            painter.setPen(QPen(color))
            painter.drawLine(i, c.low, i, c.high)
            painter.setBrush(QBrush(color))
            painter.drawRect(QRectF(i - 0.3, min(c.open, c.close), 0.6, max(abs(c.close - c.open), 0.01)))
        painter.end()
        self.prepareGeometryChange()
        self._picture = pic
        self.update()

    def paint(self, painter: QPainter, *args: Any) -> None:  # type: ignore[override]
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return QRectF(self._picture.boundingRect())


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.status = QLabel("Waiting for market candles…")
        self.status.setObjectName("metricLabel")
        layout.addWidget(self.status)

        pg.setConfigOptions(antialias=False, background="#0f172a", foreground="#dbeafe")
        self.canvas = pg.GraphicsLayoutWidget()
        layout.addWidget(self.canvas)

        self.price_plot = self.canvas.addPlot(row=0, col=0)
        self.price_plot.showGrid(x=True, y=True, alpha=0.2)
        self.price_item = CandlestickItem()
        self.price_plot.addItem(self.price_item)
        self.ma_fast = self.price_plot.plot(pen=pg.mkPen("#38bdf8", width=1.2))
        self.ma_slow = self.price_plot.plot(pen=pg.mkPen("#f59e0b", width=1.2))

        self.volume_plot = self.canvas.addPlot(row=1, col=0)
        self.volume_plot.setMaximumHeight(120)
        self.volume_plot.showGrid(x=True, y=True, alpha=0.15)
        self.volume_bars = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#334155")
        self.volume_plot.addItem(self.volume_bars)

        self._sig: tuple[tuple[float, float, float, float, float], ...] = tuple()

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        raw = snapshot.get("candles_5m") or snapshot.get("candles") or []
        if not raw:
            return
        candles = [
            Candle(float(c.get("open", 0)), float(c.get("high", 0)), float(c.get("low", 0)), float(c.get("close", 0)), float(c.get("volume", 0)))
            for c in raw[-180:]
        ]
        sig = tuple((c.open, c.high, c.low, c.close, c.volume) for c in candles)
        if sig == self._sig:
            return
        self._sig = sig
        self.status.setText("Live 5m candles · auto-updating")

        x = np.arange(len(candles), dtype=float)
        closes = np.array([c.close for c in candles])
        vols = np.array([c.volume for c in candles])
        self.price_item.set_data(candles)
        self.ma_fast.setData(x, _ema(closes, 9))
        self.ma_slow.setData(x, _ema(closes, 21))
        self.volume_plot.removeItem(self.volume_bars)
        self.volume_bars = pg.BarGraphItem(x=x, height=vols, width=0.6, brush="#334155")
        self.volume_plot.addItem(self.volume_bars)
        self.price_plot.setXRange(max(0, len(candles) - 120), len(candles))
        self.volume_plot.setXRange(max(0, len(candles) - 120), len(candles))


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    alpha = 2 / (period + 1)
    out = np.empty(values.size)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = values[i] * alpha + out[i - 1] * (1 - alpha)
    return out

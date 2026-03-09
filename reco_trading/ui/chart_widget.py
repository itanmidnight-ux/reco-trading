from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QPainter, QPen, QPicture
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

BG_COLOR = "#131722"
GRID_COLOR = "#2a2f3a"
TEXT_COLOR = "#e6e8ee"
BULL_COLOR = "#16c784"
BEAR_COLOR = "#ea3943"


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, body_width: float = 0.68) -> None:
        super().__init__()
        self._body_width = body_width
        self._picture = QPicture()
        self._candles: list[Candle] = []

    def set_candles(self, candles: list[Candle]) -> None:
        self._candles = candles
        self._rebuild_picture()
        self.update()

    def _rebuild_picture(self) -> None:
        picture = QPicture()
        painter = QPainter(picture)
        bull_pen = QPen(pg.mkColor(BULL_COLOR))
        bear_pen = QPen(pg.mkColor(BEAR_COLOR))
        bull_brush = QBrush(pg.mkColor(BULL_COLOR))
        bear_brush = QBrush(pg.mkColor(BEAR_COLOR))

        for idx, c in enumerate(self._candles):
            is_bull = c.close >= c.open
            pen = bull_pen if is_bull else bear_pen
            brush = bull_brush if is_bull else bear_brush
            painter.setPen(pen)
            painter.drawLine(idx, c.low, idx, c.high)

            body_low = min(c.open, c.close)
            body_high = max(c.open, c.close)
            body_height = max(body_high - body_low, 1e-8)
            rect = QRectF(idx - self._body_width / 2, body_low, self._body_width, body_height)
            painter.fillRect(rect, brush)
            painter.drawRect(rect)

        painter.end()
        self.prepareGeometryChange()
        self._picture = picture

    def paint(self, painter: QPainter, *args: Any) -> None:  # type: ignore[override]
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return QRectF(self._picture.boundingRect())


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._candles: list[Candle] = []
        self._last_signature: tuple[tuple[float, float, float, float, float], ...] = tuple()

        layout = QVBoxLayout(self)
        self._status = QLabel("Waiting for engine candle stream…")
        self._status.setObjectName("metricLabel")
        layout.addWidget(self._status)

        pg.setConfigOptions(antialias=False, background=BG_COLOR, foreground=TEXT_COLOR)

        self._graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphics)

        self._price_plot = self._graphics.addPlot(row=0, col=0)
        self._price_plot.showGrid(x=True, y=True, alpha=0.25)
        self._price_plot.setLabel("left", "Price")
        self._price_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._price_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._price_plot.getAxis("left").setPen(pg.mkColor(GRID_COLOR))
        self._price_plot.getAxis("bottom").setPen(pg.mkColor(GRID_COLOR))
        self._price_plot.getViewBox().setBackgroundColor(BG_COLOR)

        self._volume_plot = self._graphics.addPlot(row=1, col=0)
        self._volume_plot.setXLink(self._price_plot)
        self._volume_plot.showGrid(x=True, y=True, alpha=0.2)
        self._volume_plot.setLabel("left", "Volume")
        self._volume_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._volume_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._volume_plot.getAxis("left").setPen(pg.mkColor(GRID_COLOR))
        self._volume_plot.getAxis("bottom").setPen(pg.mkColor(GRID_COLOR))
        self._volume_plot.getViewBox().setBackgroundColor(BG_COLOR)

        self._rsi_plot = self._graphics.addPlot(row=2, col=0)
        self._rsi_plot.setXLink(self._price_plot)
        self._rsi_plot.showGrid(x=True, y=True, alpha=0.2)
        self._rsi_plot.setLabel("left", "RSI")
        self._rsi_plot.setYRange(0, 100)
        self._rsi_plot.addLine(y=70, pen=pg.mkPen("#f0b90b", width=1))
        self._rsi_plot.addLine(y=30, pen=pg.mkPen("#f0b90b", width=1))
        self._rsi_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._rsi_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._rsi_plot.getAxis("left").setPen(pg.mkColor(GRID_COLOR))
        self._rsi_plot.getAxis("bottom").setPen(pg.mkColor(GRID_COLOR))
        self._rsi_plot.getViewBox().setBackgroundColor(BG_COLOR)

        self._candles_item = CandlestickItem()
        self._price_plot.addItem(self._candles_item)

        self._ema9_line = self._price_plot.plot(pen=pg.mkPen("#4da3ff", width=1.2), name="EMA 9")
        self._ema21_line = self._price_plot.plot(pen=pg.mkPen("#f0b90b", width=1.2), name="EMA 21")
        self._ema50_line = self._price_plot.plot(pen=pg.mkPen("#c678dd", width=1.2), name="EMA 50")

        self._volume_item = pg.BarGraphItem(x=np.array([]), height=np.array([]), width=0.68, brushes=[])
        self._volume_plot.addItem(self._volume_item)
        self._rsi_line = self._rsi_plot.plot(pen=pg.mkPen("#ff9f43", width=1.2))

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        raw_candles = snapshot.get("candles_5m", [])
        if not raw_candles:
            return

        normalized = [
            Candle(
                open=float(c.get("open", 0.0)),
                high=float(c.get("high", 0.0)),
                low=float(c.get("low", 0.0)),
                close=float(c.get("close", 0.0)),
                volume=float(c.get("volume", 0.0)),
            )
            for c in raw_candles[-120:]
        ]
        signature = tuple((c.open, c.high, c.low, c.close, c.volume) for c in normalized)
        if signature == self._last_signature:
            return

        self._candles = normalized
        self._last_signature = signature
        self._status.setText("Live candles from engine stream")
        self._update_plot_items()

    def _update_plot_items(self) -> None:
        if not self._candles:
            return

        x = np.arange(len(self._candles), dtype=float)
        closes = np.array([c.close for c in self._candles], dtype=float)
        volumes = np.array([c.volume for c in self._candles], dtype=float)

        self._candles_item.set_candles(self._candles)

        self._ema9_line.setData(x, _ema(closes, 9))
        self._ema21_line.setData(x, _ema(closes, 21))
        self._ema50_line.setData(x, _ema(closes, 50))

        brushes = [pg.mkBrush(BULL_COLOR if c.close >= c.open else BEAR_COLOR) for c in self._candles]
        self._volume_plot.removeItem(self._volume_item)
        self._volume_item = pg.BarGraphItem(x=x, height=volumes, width=0.68, brushes=brushes)
        self._volume_plot.addItem(self._volume_item)

        self._rsi_line.setData(x, _rsi(closes, 14))

        self._price_plot.setXRange(max(0, len(self._candles) - 120), len(self._candles))


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    alpha = 2.0 / (period + 1)
    ema = np.empty(values.size, dtype=float)
    ema[0] = values[0]
    for i in range(1, values.size):
        ema[i] = (values[i] * alpha) + (ema[i - 1] * (1.0 - alpha))
    return ema


def _rsi(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])

    rsi = np.full(values.size, 50.0, dtype=float)
    if values.size <= period:
        return rsi

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, values.size):
        if i > period:
            avg_gain = ((avg_gain * (period - 1)) + gains[i - 1]) / period
            avg_loss = ((avg_loss * (period - 1)) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

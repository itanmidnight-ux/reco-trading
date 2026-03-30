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
ACCENT_COLOR = "#5a8dff"


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: float = 50.0
    macd_diff: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    ema9: float = 0.0


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

        self._candles_item = CandlestickItem()
        self._price_plot.addItem(self._candles_item)

        self._ema9_line = self._price_plot.plot(pen=pg.mkPen("#4da3ff", width=1.2), name="EMA 9")
        self._ema21_line = self._price_plot.plot(pen=pg.mkPen("#f0b90b", width=1.2), name="EMA 21")
        self._ema50_line = self._price_plot.plot(pen=pg.mkPen("#c678dd", width=1.2), name="EMA 50")
        self._last_price_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(ACCENT_COLOR, width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._price_plot.addItem(self._last_price_line)
        self._graphics.ci.layout.setRowStretchFactor(0, 6)
        self._graphics.nextRow()
        self._rsi_plot = self._graphics.addPlot(row=1, col=0)
        self._rsi_plot.showGrid(x=True, y=True, alpha=0.20)
        self._rsi_plot.setLabel("left", "RSI")
        self._rsi_plot.setYRange(0, 100)
        self._rsi_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._rsi_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._rsi_plot.getViewBox().setBackgroundColor(BG_COLOR)
        self._graphics.ci.layout.setRowStretchFactor(1, 2)
        for level, color in [(70, "#ea3943"), (30, "#16c784"), (50, "#9fb2d9")]:
            self._rsi_plot.addItem(
                pg.InfiniteLine(pos=level, angle=0, movable=False,
                                pen=pg.mkPen(color, width=0.8,
                                             style=pg.QtCore.Qt.PenStyle.DashLine))
            )
        self._rsi_line = self._rsi_plot.plot(pen=pg.mkPen("#ffffff", width=1.2))
        self._graphics.nextRow()
        self._macd_plot = self._graphics.addPlot(row=2, col=0)
        self._macd_plot.showGrid(x=True, y=True, alpha=0.20)
        self._macd_plot.setLabel("left", "MACD")
        self._macd_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._macd_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._macd_plot.getViewBox().setBackgroundColor(BG_COLOR)
        self._graphics.ci.layout.setRowStretchFactor(2, 2)
        self._macd_line = self._macd_plot.plot(pen=pg.mkPen("#5a8dff", width=1.0))
        self._signal_line = self._macd_plot.plot(pen=pg.mkPen("#f0b90b", width=1.0))
        self._macd_hist = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#ea3943")
        self._macd_plot.addItem(self._macd_hist)
        self._rsi_plot.setXLink(self._price_plot)
        self._macd_plot.setXLink(self._price_plot)

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
                rsi=float(c.get("rsi", 50.0)),
                macd_diff=float(c.get("macd_diff", 0.0)),
                macd=float(c.get("macd", 0.0)),
                macd_signal=float(c.get("macd_signal", 0.0)),
                ema9=float(c.get("ema9", 0.0)),
            )
            for c in raw_candles[-120:]
        ]
        signature = tuple(
            (c.open, c.high, c.low, c.close, c.volume,
             round(c.rsi, 1), round(c.macd_diff, 6))
            for c in normalized
        )
        if signature == self._last_signature:
            return

        self._candles = normalized
        self._last_signature = signature
        last_close = normalized[-1].close
        last_open = normalized[-1].open
        direction = "Bullish" if last_close >= last_open else "Bearish"
        self._status.setText(f"Live candles • last close {last_close:,.2f} • {direction}")
        self._update_plot_items()

    def _update_plot_items(self) -> None:
        if not self._candles:
            return

        n = len(self._candles)
        x = np.arange(n, dtype=float)
        closes = np.array([c.close for c in self._candles], dtype=float)
        self._candles_item.set_candles(self._candles)

        self._ema9_line.setData(x, _ema(closes, 9))
        self._ema21_line.setData(x, _ema(closes, 21))
        self._ema50_line.setData(x, _ema(closes, 50))
        self._last_price_line.setPos(float(closes[-1]))
        rsi_vals = np.array([c.rsi for c in self._candles], dtype=float)
        macd_vals = np.array([c.macd for c in self._candles], dtype=float)
        sig_vals = np.array([c.macd_signal for c in self._candles], dtype=float)
        hist_vals = np.array([c.macd_diff for c in self._candles], dtype=float)
        self._rsi_line.setData(x, rsi_vals)
        self._macd_line.setData(x, macd_vals)
        self._signal_line.setData(x, sig_vals)
        brushes = ["#16c784" if v >= 0 else "#ea3943" for v in hist_vals]
        self._macd_hist.setOpts(x=x, height=hist_vals, width=0.6, brushes=brushes)
        view_start = max(0, n - 120)
        self._price_plot.setXRange(view_start, n, padding=0)


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    alpha = 2.0 / (period + 1)
    ema = np.empty(values.size, dtype=float)
    ema[0] = values[0]
    for i in range(1, values.size):
        ema[i] = (values[i] * alpha) + (ema[i - 1] * (1.0 - alpha))
    return ema

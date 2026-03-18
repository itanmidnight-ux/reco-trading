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
EMA_FAST_COLOR = "#4da3ff"
EMA_MID_COLOR = "#f0b90b"
EMA_SLOW_COLOR = "#c678dd"
VOLUME_BULL = "#1fbf8f"
VOLUME_BEAR = "#b8425d"


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


class VolumeBarItem(pg.GraphicsObject):
    def __init__(self, bar_width: float = 0.72) -> None:
        super().__init__()
        self._bar_width = bar_width
        self._picture = QPicture()
        self._candles: list[Candle] = []

    def set_candles(self, candles: list[Candle]) -> None:
        self._candles = candles
        self._rebuild_picture()
        self.update()

    def _rebuild_picture(self) -> None:
        picture = QPicture()
        painter = QPainter(picture)
        bull_brush = QBrush(pg.mkColor(VOLUME_BULL))
        bear_brush = QBrush(pg.mkColor(VOLUME_BEAR))
        bull_pen = QPen(pg.mkColor(VOLUME_BULL))
        bear_pen = QPen(pg.mkColor(VOLUME_BEAR))

        for idx, candle in enumerate(self._candles):
            is_bull = candle.close >= candle.open
            painter.setPen(bull_pen if is_bull else bear_pen)
            painter.setBrush(bull_brush if is_bull else bear_brush)
            rect = QRectF(idx - self._bar_width / 2, 0.0, self._bar_width, max(candle.volume, 1e-8))
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
        self._last_market_price: float | None = None

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
        self._price_plot.showAxis("right")
        self._price_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._price_plot.getAxis("right").setTextPen(pg.mkColor(TEXT_COLOR))
        self._price_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._price_plot.getAxis("left").setPen(pg.mkColor(GRID_COLOR))
        self._price_plot.getAxis("right").setPen(pg.mkColor(GRID_COLOR))
        self._price_plot.getAxis("bottom").setPen(pg.mkColor(GRID_COLOR))
        self._price_plot.getViewBox().setBackgroundColor(BG_COLOR)
        self._price_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self._price_plot.hideButtons()
        self._price_plot.setMenuEnabled(False)

        self._candles_item = CandlestickItem()
        self._price_plot.addItem(self._candles_item)

        self._ema9_line = self._price_plot.plot(pen=pg.mkPen(EMA_FAST_COLOR, width=1.35), name="EMA 9")
        self._ema21_line = self._price_plot.plot(pen=pg.mkPen(EMA_MID_COLOR, width=1.35), name="EMA 21")
        self._ema50_line = self._price_plot.plot(pen=pg.mkPen(EMA_SLOW_COLOR, width=1.2), name="EMA 50")
        self._last_price_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(ACCENT_COLOR, width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._price_plot.addItem(self._last_price_line)
        self._price_marker = pg.TextItem(anchor=(0, 0.5))
        self._price_plot.addItem(self._price_marker)
        self._legend = self._price_plot.addLegend(offset=(12, 10))

        self._graphics.nextRow()
        self._volume_plot = self._graphics.addPlot(row=1, col=0)
        self._volume_plot.showGrid(x=True, y=True, alpha=0.18)
        self._volume_plot.setMaximumHeight(120)
        self._volume_plot.setLabel("left", "Vol")
        self._volume_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._volume_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._volume_plot.getAxis("left").setPen(pg.mkColor(GRID_COLOR))
        self._volume_plot.getAxis("bottom").setPen(pg.mkColor(GRID_COLOR))
        self._volume_plot.getViewBox().setBackgroundColor(BG_COLOR)
        self._volume_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self._volume_plot.hideButtons()
        self._volume_plot.setMenuEnabled(False)
        self._volume_plot.setXLink(self._price_plot)
        self._volume_item = VolumeBarItem()
        self._volume_plot.addItem(self._volume_item)

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
        market_price = float(snapshot.get("current_price", snapshot.get("price", normalized[-1].close)) or normalized[-1].close)
        signature = tuple((c.open, c.high, c.low, c.close, c.volume) for c in normalized)
        if signature == self._last_signature and self._last_market_price == market_price:
            return

        self._candles = normalized
        self._last_signature = signature
        self._last_market_price = market_price
        last_close = market_price
        last_open = normalized[-1].open
        direction = "Bullish" if last_close >= last_open else "Bearish"
        self._status.setText(
            f"Professional stream • last {last_close:,.2f} • {direction} • "
            f"vol {_fmt_compact(normalized[-1].volume)}"
        )
        self._update_plot_items()

    def _update_plot_items(self) -> None:
        if not self._candles:
            return

        x = np.arange(len(self._candles), dtype=float)
        closes = np.array([c.close for c in self._candles], dtype=float)
        highs = np.array([c.high for c in self._candles], dtype=float)
        lows = np.array([c.low for c in self._candles], dtype=float)
        volumes = np.array([c.volume for c in self._candles], dtype=float)
        self._candles_item.set_candles(self._candles)
        self._volume_item.set_candles(self._candles)

        ema9 = _ema(closes, 9)
        ema21 = _ema(closes, 21)
        ema50 = _ema(closes, 50)
        self._ema9_line.setData(x, ema9)
        self._ema21_line.setData(x, ema21)
        self._ema50_line.setData(x, ema50)
        current_price = float(self._last_market_price or closes[-1])
        self._last_price_line.setPos(current_price)
        self._price_marker.setHtml(
            (
                "<div style='background-color: rgba(90,141,255,0.22);"
                " border: 1px solid rgba(90,141,255,0.65); border-radius: 8px;"
                " padding: 4px 8px; color: #edf2ff; font-size: 11px; font-weight: 700;'>"
                f"{current_price:,.2f}</div>"
            )
        )
        self._price_marker.setPos(len(self._candles) + 1.2, current_price)

        visible = min(len(self._candles), 90)
        left = max(0, len(self._candles) - visible)
        right = len(self._candles) + 4
        self._price_plot.setXRange(left - 0.5, right, padding=0)
        self._volume_plot.setXRange(left - 0.5, right, padding=0)

        visible_high = highs[left:] if left < len(highs) else highs
        visible_low = lows[left:] if left < len(lows) else lows
        if visible_high.size and visible_low.size:
            ymin = float(np.min(visible_low))
            ymax = float(np.max(visible_high))
            pad = max((ymax - ymin) * 0.08, ymax * 0.0025, 1e-6)
            self._price_plot.setYRange(ymin - pad, ymax + pad, padding=0)

        max_volume = float(np.max(volumes[left:])) if volumes[left:].size else 1.0
        self._volume_plot.setYRange(0, max(max_volume * 1.25, 1.0), padding=0)


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    alpha = 2.0 / (period + 1)
    ema = np.empty(values.size, dtype=float)
    ema[0] = values[0]
    for i in range(1, values.size):
        ema[i] = (values[i] * alpha) + (ema[i - 1] * (1.0 - alpha))
    return ema


def _fmt_compact(value: float) -> str:
    thresholds = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    absolute = abs(value)
    for threshold, suffix in thresholds:
        if absolute >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:.2f}"

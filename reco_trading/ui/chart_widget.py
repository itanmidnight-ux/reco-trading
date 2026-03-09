from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
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

        self._candles_item = CandlestickItem()
        self._price_plot.addItem(self._candles_item)

        self._ema9_line = self._price_plot.plot(pen=pg.mkPen("#4da3ff", width=1.2), name="EMA 9")
        self._ema21_line = self._price_plot.plot(pen=pg.mkPen("#f0b90b", width=1.2), name="EMA 21")
        self._ema50_line = self._price_plot.plot(pen=pg.mkPen("#c678dd", width=1.2), name="EMA 50")

        self._buy_scatter = pg.ScatterPlotItem(symbol="t", size=12, brush=pg.mkBrush(BULL_COLOR), pen=pg.mkPen(BULL_COLOR))
        self._sell_scatter = pg.ScatterPlotItem(symbol="t1", size=12, brush=pg.mkBrush(BEAR_COLOR), pen=pg.mkPen(BEAR_COLOR))
        self._price_plot.addItem(self._buy_scatter)
        self._price_plot.addItem(self._sell_scatter)

        self._entry_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#4da3ff", width=1.2, style=Qt.PenStyle.DashLine), label="ENTRY", labelOpts={"color": "#4da3ff", "position": 0.95})
        self._sl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(BEAR_COLOR, width=1.2, style=Qt.PenStyle.DashLine), label="SL", labelOpts={"color": BEAR_COLOR, "position": 0.95})
        self._tp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(BULL_COLOR, width=1.2, style=Qt.PenStyle.DashLine), label="TP", labelOpts={"color": BULL_COLOR, "position": 0.95})
        self._price_plot.addItem(self._entry_line)
        self._price_plot.addItem(self._sl_line)
        self._price_plot.addItem(self._tp_line)
        self._entry_line.hide()
        self._sl_line.hide()
        self._tp_line.hide()

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

        self._candles = normalized
        if signature != self._last_signature:
            self._last_signature = signature
            self._status.setText("Live candles from engine stream")
            self._update_plot_items()

        self._update_trade_markers(snapshot)
        self._update_position_levels(snapshot)

    def _update_plot_items(self) -> None:
        if not self._candles:
            return

        x = np.arange(len(self._candles), dtype=float)
        closes = np.array([c.close for c in self._candles], dtype=float)
        self._candles_item.set_candles(self._candles)

        self._ema9_line.setData(x, _ema(closes, 9))
        self._ema21_line.setData(x, _ema(closes, 21))
        self._ema50_line.setData(x, _ema(closes, 50))

        self._price_plot.setXRange(max(0, len(self._candles) - 120), len(self._candles))

    def _update_trade_markers(self, snapshot: dict[str, Any]) -> None:
        if not self._candles:
            return
        trade_history = snapshot.get("trade_history", []) or []
        buy_pts: list[dict[str, float]] = []
        sell_pts: list[dict[str, float]] = []
        window = self._candles[-60:]

        for idx, trade in enumerate(reversed(trade_history[-120:])):
            side = str(trade.get("side", trade.get("signal", ""))).upper()
            price = _to_float(trade.get("price"))
            if price is None:
                continue
            anchor = next((c for c in window if c.low <= price <= c.high), window[min(idx, len(window)-1)])
            x_val = len(self._candles) - len(window) + min(idx, len(window)-1)
            if side == "BUY":
                buy_pts.append({"pos": (x_val, anchor.low * 0.9985)})
            elif side == "SELL":
                sell_pts.append({"pos": (x_val, anchor.high * 1.0015)})

        self._buy_scatter.setData(buy_pts)
        self._sell_scatter.setData(sell_pts)

    def _update_position_levels(self, snapshot: dict[str, Any]) -> None:
        entry = _to_float(snapshot.get("entry_price"))
        sl = _to_float(snapshot.get("stop_loss"))
        tp = _to_float(snapshot.get("take_profit"))
        side = str(snapshot.get("position_side", snapshot.get("open_position", "NONE"))).upper()
        has_position = side not in {"NONE", "", "FLAT", "CLOSED"}

        if has_position and entry is not None:
            self._entry_line.setValue(entry)
            self._entry_line.show()
        else:
            self._entry_line.hide()

        if sl is not None:
            self._sl_line.setValue(sl)
            self._sl_line.show()
        else:
            self._sl_line.hide()

        if tp is not None:
            self._tp_line.setValue(tp)
            self._tp_line.show()
        else:
            self._tp_line.hide()


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    alpha = 2.0 / (period + 1)
    ema = np.empty(values.size, dtype=float)
    ema[0] = values[0]
    for i in range(1, values.size):
        ema[i] = (values[i] * alpha) + (ema[i - 1] * (1.0 - alpha))
    return ema


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

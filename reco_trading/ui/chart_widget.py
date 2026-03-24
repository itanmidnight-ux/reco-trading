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
    def __init__(self, body_width: float = 0.66) -> None:
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
        bull_pen.setWidthF(1.15)
        bear_pen = QPen(pg.mkColor(BEAR_COLOR))
        bear_pen.setWidthF(1.15)
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


_FALLBACK_DARK = {
    "background": "#0b1020",
    "border": "#273658",
    "text_primary": "#edf2ff",
    "text_secondary": "#9fb2d9",
    "info": "#5a8dff",
    "warning": "#ffcc66",
    "accent": "#7b61ff",
}

_FALLBACK_LIGHT = {
    "background": "#f5f7fb",
    "border": "#c7d3ea",
    "text_primary": "#12213f",
    "text_secondary": "#425a85",
    "info": "#2c5bd8",
    "warning": "#b7791f",
    "accent": "#4b54e6",
}


def _resolve_theme_colors(theme: str) -> dict[str, str]:
    normalized = str(theme or "Dark").strip().lower()
    return dict(_FALLBACK_LIGHT if normalized in {"light", "white", "blanco"} else _FALLBACK_DARK)


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._candles: list[Candle] = []
        self._last_signature: tuple[tuple[float, float, float, float, float], ...] = tuple()
        self._theme_name = "Dark"

        layout = QVBoxLayout(self)
        self._status = QLabel("Waiting for engine candle stream…")
        self._status.setObjectName("metricLabel")
        layout.addWidget(self._status)

        pg.setConfigOptions(antialias=True, background=BG_COLOR, foreground=TEXT_COLOR)

        self._graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphics)

        self._price_plot = self._graphics.addPlot(row=0, col=0)
        self._price_plot.showGrid(x=True, y=True, alpha=0.25)
        self._price_plot.setMenuEnabled(False)
        self._price_plot.hideButtons()
        self._price_plot.setLabel("left", "Price")
        self._price_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._price_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._price_plot.getAxis("left").setPen(pg.mkColor(GRID_COLOR))
        self._price_plot.getAxis("bottom").setPen(pg.mkColor(GRID_COLOR))
        self._price_plot.getViewBox().setBackgroundColor(BG_COLOR)

        self._candles_item = CandlestickItem()
        self._price_plot.addItem(self._candles_item)

        self._ema9_line = self._price_plot.plot(pen=pg.mkPen(EMA_FAST_COLOR, width=1.5), name="EMA 9")
        self._ema21_line = self._price_plot.plot(pen=pg.mkPen(EMA_MID_COLOR, width=1.4), name="EMA 21")
        self._ema50_line = self._price_plot.plot(pen=pg.mkPen(EMA_SLOW_COLOR, width=1.2), name="EMA 50")
        self._ema_spread_fill = pg.FillBetweenItem(
            self._ema9_line,
            self._ema21_line,
            brush=pg.mkBrush(90, 141, 255, 35),
        )
        self._price_plot.addItem(self._ema_spread_fill)
        self._last_price_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(ACCENT_COLOR, width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._price_plot.addItem(self._last_price_line)
        self._last_price_label = pg.TextItem("", anchor=(0, 0.5), color=TEXT_COLOR)
        self._price_plot.addItem(self._last_price_label)
        self._graphics.ci.layout.setRowStretchFactor(0, 6)
        self._graphics.nextRow()
        self._rsi_plot = self._graphics.addPlot(row=1, col=0)
        self._rsi_plot.showGrid(x=True, y=True, alpha=0.20)
        self._rsi_plot.setMenuEnabled(False)
        self._rsi_plot.hideButtons()
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
        self._macd_plot.setMenuEnabled(False)
        self._macd_plot.hideButtons()
        self._macd_plot.setLabel("left", "MACD")
        self._macd_plot.getAxis("left").setTextPen(pg.mkColor(TEXT_COLOR))
        self._macd_plot.getAxis("bottom").setTextPen(pg.mkColor(TEXT_COLOR))
        self._macd_plot.getViewBox().setBackgroundColor(BG_COLOR)
        self._graphics.ci.layout.setRowStretchFactor(2, 2)
        self._macd_line = self._macd_plot.plot(pen=pg.mkPen("#5a8dff", width=1.0))
        self._signal_line = self._macd_plot.plot(pen=pg.mkPen("#f0b90b", width=1.0))
        self._macd_hist = pg.BarGraphItem(x=[], height=[], width=0.6, brush="#ea3943")
        self._macd_plot.addItem(self._macd_hist)
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#7f93bf", width=0.8, style=pg.QtCore.Qt.PenStyle.DotLine))
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#7f93bf", width=0.8, style=pg.QtCore.Qt.PenStyle.DotLine))
        self._price_plot.addItem(self._crosshair_v, ignoreBounds=True)
        self._price_plot.addItem(self._crosshair_h, ignoreBounds=True)
        self._graphics.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._rsi_plot.setXLink(self._price_plot)
        self._macd_plot.setXLink(self._price_plot)
        self.set_theme("Dark")


    def set_theme(self, theme: str = "Dark") -> None:
        self._theme_name = theme or "Dark"
        colors = _resolve_theme_colors(self._theme_name)
        background = colors.get("background", BG_COLOR)
        border = colors.get("border", GRID_COLOR)
        text_primary = colors.get("text_primary", TEXT_COLOR)
        text_secondary = colors.get("text_secondary", TEXT_COLOR)
        info = colors.get("info", ACCENT_COLOR)
        warning = colors.get("warning", EMA_MID_COLOR)
        accent = colors.get("accent", EMA_SLOW_COLOR)

        pg.setConfigOptions(antialias=True, background=background, foreground=text_primary)
        self._graphics.setBackground(background)

        for plot in (self._price_plot, self._rsi_plot, self._macd_plot):
            plot.getViewBox().setBackgroundColor(background)
            plot.showGrid(x=True, y=True, alpha=0.22)
            plot.getAxis("left").setTextPen(pg.mkColor(text_primary))
            plot.getAxis("bottom").setTextPen(pg.mkColor(text_primary))
            plot.getAxis("left").setPen(pg.mkColor(border))
            plot.getAxis("bottom").setPen(pg.mkColor(border))

        self._ema9_line.setPen(pg.mkPen(info, width=1.5))
        self._ema21_line.setPen(pg.mkPen(warning, width=1.4))
        self._ema50_line.setPen(pg.mkPen(accent, width=1.2))
        self._last_price_line.setPen(pg.mkPen(info, width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
        self._last_price_label.setColor(pg.mkColor(text_primary))
        self._rsi_line.setPen(pg.mkPen(text_primary, width=1.2))
        self._macd_line.setPen(pg.mkPen(info, width=1.0))
        self._signal_line.setPen(pg.mkPen(warning, width=1.0))
        self._crosshair_v.setPen(pg.mkPen(text_secondary, width=0.8, style=pg.QtCore.Qt.PenStyle.DotLine))
        self._crosshair_h.setPen(pg.mkPen(text_secondary, width=0.8, style=pg.QtCore.Qt.PenStyle.DotLine))

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
        session_low = min(c.low for c in normalized)
        session_high = max(c.high for c in normalized)
        self._status.setText(
            f"Live candles • close {last_close:,.2f} • {direction} • range {session_low:,.2f}-{session_high:,.2f}"
        )
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
        self._last_price_label.setText(f" {closes[-1]:,.2f}")
        self._last_price_label.setPos(max(n - 1.5, 0), float(closes[-1]))
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

    def _on_mouse_moved(self, pos: object) -> None:
        if not self._price_plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = self._price_plot.getViewBox().mapSceneToView(pos)
        self._crosshair_v.setPos(mouse_point.x())
        self._crosshair_h.setPos(mouse_point.y())


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    alpha = 2.0 / (period + 1)
    ema = np.empty(values.size, dtype=float)
    ema[0] = values[0]
    for i in range(1, values.size):
        ema[i] = (values[i] * alpha) + (ema[i - 1] * (1.0 - alpha))
    return ema

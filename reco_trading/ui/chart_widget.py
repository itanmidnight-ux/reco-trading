from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QPainter, QPen, QPicture
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

BG_COLOR = "#131722"
GRID_COLOR = "#2a2f3a"
TEXT_COLOR = "#e6e8ee"
BULL_COLOR = "#16c784"
BEAR_COLOR = "#ea3943"

try:
    import pyqtgraph as pg

    HAS_PG = True
except Exception:  # noqa: BLE001
    pg = None
    HAS_PG = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    HAS_MPL = True
except Exception:  # noqa: BLE001
    Figure = None
    FigureCanvasQTAgg = None
    HAS_MPL = False


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandlestickItem(pg.GraphicsObject if HAS_PG else object):
    def __init__(self, body_width: float = 0.68) -> None:
        if not HAS_PG:
            return
        super().__init__()
        self._body_width = body_width
        self._picture = QPicture()
        self._candles: list[Candle] = []

    def set_candles(self, candles: list[Candle]) -> None:
        if not HAS_PG:
            return
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
            rect = QRectF(idx - self._body_width / 2, body_low, self._body_width, max(body_high - body_low, 1e-8))
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

        if HAS_PG:
            self._init_pyqtgraph(layout)
        elif HAS_MPL:
            self._init_matplotlib(layout)
        else:
            self._chart_backend = "none"
            layout.addWidget(QLabel("No chart backend available (install pyqtgraph or matplotlib)."))

    def _init_pyqtgraph(self, layout: QVBoxLayout) -> None:
        self._chart_backend = "pyqtgraph"
        pg.setConfigOptions(antialias=False, background=BG_COLOR, foreground=TEXT_COLOR)
        self._graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphics, 1)

        self._price_plot = self._graphics.addPlot(row=0, col=0)
        self._volume_plot = self._graphics.addPlot(row=1, col=0)
        self._price_plot.showGrid(x=True, y=True, alpha=0.25)
        self._volume_plot.showGrid(x=True, y=True, alpha=0.2)
        self._volume_plot.setMaximumHeight(140)
        self._price_plot.setXLink(self._volume_plot)

        self._candles_item = CandlestickItem()
        self._price_plot.addItem(self._candles_item)
        self._ma20_line = self._price_plot.plot(pen=pg.mkPen("#4da3ff", width=1.2), name="MA 20")
        self._ma50_line = self._price_plot.plot(pen=pg.mkPen("#f0b90b", width=1.2), name="MA 50")

        self._volume_bars = pg.BarGraphItem(x=[], height=[], width=0.7, brushes=[])
        self._volume_plot.addItem(self._volume_bars)

        self._buy_scatter = pg.ScatterPlotItem(symbol="t", size=11, brush=pg.mkBrush(BULL_COLOR), pen=pg.mkPen(BULL_COLOR))
        self._sell_scatter = pg.ScatterPlotItem(symbol="t1", size=11, brush=pg.mkBrush(BEAR_COLOR), pen=pg.mkPen(BEAR_COLOR))
        self._price_plot.addItem(self._buy_scatter)
        self._price_plot.addItem(self._sell_scatter)

        self._entry_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#4da3ff", style=Qt.PenStyle.DashLine))
        self._sl_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(BEAR_COLOR, style=Qt.PenStyle.DashLine))
        self._tp_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(BULL_COLOR, style=Qt.PenStyle.DashLine))
        for line in (self._entry_line, self._sl_line, self._tp_line):
            self._price_plot.addItem(line)
            line.hide()

    def _init_matplotlib(self, layout: QVBoxLayout) -> None:
        self._chart_backend = "matplotlib"
        self._fig = Figure(facecolor=BG_COLOR)
        self._ax_price = self._fig.add_subplot(2, 1, 1)
        self._ax_volume = self._fig.add_subplot(2, 1, 2, sharex=self._ax_price)
        self._canvas = FigureCanvasQTAgg(self._fig)
        layout.addWidget(self._canvas, 1)

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        try:
            raw_candles = snapshot.get("candles_5m") or snapshot.get("candles") or []
            if not raw_candles:
                return
            normalized: list[Candle] = []
            for c in raw_candles[-180:]:
                try:
                    normalized.append(
                        Candle(
                            open=float(c.get("open", 0.0)),
                            high=float(c.get("high", 0.0)),
                            low=float(c.get("low", 0.0)),
                            close=float(c.get("close", 0.0)),
                            volume=float(c.get("volume", 0.0)),
                        )
                    )
                except Exception:
                    continue
            if not normalized:
                return

            signature = tuple((c.open, c.high, c.low, c.close, c.volume) for c in normalized)
            self._candles = normalized
            if signature != self._last_signature:
                self._last_signature = signature
                self._status.setText("Live OHLC stream")
                if self._chart_backend == "pyqtgraph":
                    self._update_plot_items_pg()
                elif self._chart_backend == "matplotlib":
                    self._update_plot_items_mpl()

            if self._chart_backend == "pyqtgraph":
                self._update_trade_markers_pg(snapshot)
                self._update_position_levels_pg(snapshot)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Chart update protected: {exc}")

    def _update_plot_items_pg(self) -> None:
        x = np.arange(len(self._candles), dtype=float)
        closes = np.array([c.close for c in self._candles], dtype=float)
        volumes = np.array([c.volume for c in self._candles], dtype=float)
        brushes = [pg.mkBrush(BULL_COLOR if c.close >= c.open else BEAR_COLOR) for c in self._candles]

        self._candles_item.set_candles(self._candles)
        self._ma20_line.setData(x, _sma(closes, 20))
        self._ma50_line.setData(x, _sma(closes, 50))
        self._volume_plot.removeItem(self._volume_bars)
        self._volume_bars = pg.BarGraphItem(x=x, height=volumes, width=0.7, brushes=brushes)
        self._volume_plot.addItem(self._volume_bars)

        self._price_plot.setXRange(max(0, len(self._candles) - 120), len(self._candles))
        self._volume_plot.setXRange(max(0, len(self._candles) - 120), len(self._candles))

    def _update_plot_items_mpl(self) -> None:
        x = np.arange(len(self._candles))
        closes = np.array([c.close for c in self._candles])
        self._ax_price.clear()
        self._ax_volume.clear()
        self._ax_price.set_facecolor(BG_COLOR)
        self._ax_volume.set_facecolor(BG_COLOR)
        for i, candle in enumerate(self._candles):
            color = BULL_COLOR if candle.close >= candle.open else BEAR_COLOR
            self._ax_price.vlines(i, candle.low, candle.high, color=color, linewidth=1)
            low = min(candle.open, candle.close)
            high = max(candle.open, candle.close)
            self._ax_price.add_patch(Rectangle((i - 0.34, low), 0.68, max(high - low, 1e-8), color=color))
            self._ax_volume.bar(i, candle.volume, color=color, width=0.7)
        self._ax_price.plot(x, _sma(closes, 20), color="#4da3ff", linewidth=1.0)
        self._ax_price.plot(x, _sma(closes, 50), color="#f0b90b", linewidth=1.0)
        self._ax_price.grid(color=GRID_COLOR, linestyle="--", alpha=0.25)
        self._ax_volume.grid(color=GRID_COLOR, linestyle="--", alpha=0.2)
        self._canvas.draw_idle()

    def _update_trade_markers_pg(self, snapshot: dict[str, Any]) -> None:
        trades = snapshot.get("trade_history", []) or []
        buy_pts: list[dict[str, float]] = []
        sell_pts: list[dict[str, float]] = []
        for idx, trade in enumerate(trades[-80:]):
            side = str(trade.get("side", trade.get("signal", ""))).upper()
            price = _to_float(trade.get("price"))
            if price is None:
                continue
            x_val = max(0, len(self._candles) - min(80, len(trades)) + idx)
            if side == "BUY":
                buy_pts.append({"pos": (x_val, price * 0.998)})
            elif side == "SELL":
                sell_pts.append({"pos": (x_val, price * 1.002)})
        self._buy_scatter.setData(buy_pts)
        self._sell_scatter.setData(sell_pts)

    def _update_position_levels_pg(self, snapshot: dict[str, Any]) -> None:
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

        for line, value in ((self._sl_line, sl), (self._tp_line, tp)):
            if has_position and value is not None:
                line.setValue(value)
                line.show()
            else:
                line.hide()


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return np.array([])
    out = np.full(values.size, np.nan)
    if values.size < period:
        return out
    csum = np.cumsum(values, dtype=float)
    csum[period:] = csum[period:] - csum[:-period]
    out[period - 1 :] = csum[period - 1 :] / period
    return out


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

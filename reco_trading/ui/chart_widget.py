from __future__ import annotations

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
        layout = QVBoxLayout(self)
        if Figure is None or FigureCanvas is None:
            layout.addWidget(QLabel("Chart unavailable: matplotlib backend missing"))
            self.figure = None
            self.canvas = None
            self.ax_price = None
            self.ax_volume = None
            return

        self.figure = Figure(figsize=(8, 4), facecolor="#1a1d26")
        self.ax_price = self.figure.add_subplot(211)
        self.ax_volume = self.figure.add_subplot(212, sharex=self.ax_price)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        candles = snapshot.get("candles_5m", [])
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
        self.ax_price.set_facecolor("#1a1d26")
        self.ax_volume.set_facecolor("#1a1d26")

        for i, (o, h, l, cl) in enumerate(zip(opens, highs, lows, closes)):
            color = "#16c784" if cl >= o else "#ea3943"
            self.ax_price.plot([i, i], [l, h], color=color, linewidth=1)
            lower = min(o, cl)
            height = max(abs(cl - o), 1e-8)
            self.ax_price.bar(i, height, bottom=lower, color=color, width=0.65)
            self.ax_volume.bar(i, volumes[i], color=color, width=0.65, alpha=0.5)

        if closes:
            ema = _ema(closes, 20)
            self.ax_price.plot(range(len(ema)), ema, color="#3a7afe", linewidth=1.2, label="EMA 20")
            self.ax_price.legend(loc="upper left", fontsize=8)

        self.ax_price.set_title("BTC/USDT 5m Candles", color="#e6e8ee", fontsize=10)
        self.ax_price.tick_params(colors="#9aa4b2", labelsize=8)
        self.ax_volume.tick_params(colors="#9aa4b2", labelsize=7)
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

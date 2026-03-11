from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except Exception:  # noqa: BLE001
    Figure = None
    FigureCanvas = None


class PnlChart(QWidget):
    def __init__(self, title: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.title = title
        if Figure is None or FigureCanvas is None:
            layout.addWidget(QLabel(f"{title}: matplotlib no disponible"))
            self.figure = None
            self.axes = None
            self.canvas = None
            return
        self.figure = Figure(figsize=(4, 2))
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(title)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

    def plot(self, values: list[float]) -> None:
        if not self.axes or not self.canvas:
            return
        self.axes.clear()
        self.axes.set_title(self.title)
        self.axes.plot(values)
        self.canvas.draw_idle()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ChartBridge(QObject):
    """WebChannel bridge for pushing candle payloads from Python to chart JS."""

    candleDataUpdated = Signal(str)


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_signature: tuple[tuple[int, float, float, float, float, float], ...] = tuple()
        self._last_series: list[dict[str, float | int]] = []

        layout = QVBoxLayout(self)
        self._status = QLabel("Loading TradingView chart…")
        self._status.setObjectName("metricLabel")
        layout.addWidget(self._status)

        self._web_view = QWebEngineView(self)
        layout.addWidget(self._web_view)

        self._bridge = ChartBridge()
        self._channel = QWebChannel(self._web_view.page())
        self._channel.registerObject("chartBridge", self._bridge)
        self._web_view.page().setWebChannel(self._channel)

        chart_path = Path(__file__).resolve().parent / "assets" / "chart.html"
        self._web_view.load(QUrl.fromLocalFile(str(chart_path)))
        self._web_view.loadFinished.connect(self._on_load_finished)

    def _on_load_finished(self, ok: bool) -> None:
        self._status.setText("Live candles from engine stream" if ok else "Failed to load chart.html")

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        raw_candles = snapshot.get("candles_5m", [])
        if not raw_candles:
            return

        normalized: list[dict[str, float | int]] = []
        for idx, c in enumerate(raw_candles[-240:]):
            normalized.append(
                {
                    "time": int(c.get("timestamp") or c.get("time") or idx),
                    "open": float(c.get("open", 0.0)),
                    "high": float(c.get("high", 0.0)),
                    "low": float(c.get("low", 0.0)),
                    "close": float(c.get("close", 0.0)),
                    "volume": float(c.get("volume", 0.0)),
                }
            )

        signature = tuple(
            (int(c["time"]), float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]), float(c["volume"]))
            for c in normalized
        )
        if signature == self._last_signature:
            return

        self._last_signature = signature

        mode = "set"
        payload_series = normalized
        if self._last_series:
            if len(normalized) == len(self._last_series):
                mode = "update"
                payload_series = [normalized[-1]]
            elif len(normalized) == (len(self._last_series) + 1):
                mode = "append"
                payload_series = [normalized[-1]]

        payload = json.dumps({"mode": mode, "candles": payload_series})
        self._bridge.candleDataUpdated.emit(payload)
        self._last_series = normalized

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

MAX_CANDLES = 400


class _ChartBridge(QObject):
    updateCandles = Signal("QVariantMap")


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._candles: list[dict[str, float | int]] = []
        self._base_time = 1_700_000_000
        self._web_ready = False
        self._bridge = _ChartBridge()

        layout = QVBoxLayout(self)
        self._status = QLabel("Waiting for engine candle stream…")
        self._status.setObjectName("metricLabel")
        layout.addWidget(self._status)

        self._web_view: QWidget | None = None

        try:
            from PySide6.QtWebChannel import QWebChannel
            from PySide6.QtWebEngineWidgets import QWebEngineView

            self._web_view = QWebEngineView(self)
            self._channel = QWebChannel(self._web_view.page())
            self._channel.registerObject("bridge", self._bridge)
            self._web_view.page().setWebChannel(self._channel)
            self._web_view.loadFinished.connect(self._on_load_finished)
            layout.addWidget(self._web_view)

            chart_path = Path(__file__).with_name("assets") / "chart.html"
            self._web_view.load(QUrl.fromLocalFile(str(chart_path.resolve())))
        except Exception:
            self._status.setText("Chart engine unavailable (WebEngine not installed)")

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        raw_candles = snapshot.get("candles_5m", [])
        if not raw_candles:
            return

        normalized: list[dict[str, float | int]] = []
        for idx, candle in enumerate(raw_candles[-MAX_CANDLES:]):
            resolved_time = self._resolve_time(candle, idx)
            normalized.append(
                {
                    "time": resolved_time,
                    "open": float(candle.get("open", 0.0)),
                    "high": float(candle.get("high", 0.0)),
                    "low": float(candle.get("low", 0.0)),
                    "close": float(candle.get("close", 0.0)),
                    "volume": float(candle.get("volume", 0.0)),
                }
            )

        if not normalized:
            return

        payload: dict[str, Any]
        if self._candles and normalized[-1]["time"] == self._candles[-1]["time"]:
            payload = {"mode": "append", "candles": [normalized[-1]]}
            self._candles[-1] = normalized[-1]
        elif self._candles and len(normalized) >= len(self._candles):
            tail = normalized[len(self._candles) - 1 :]
            payload = {"mode": "append", "candles": tail}
            self._candles = normalized
        else:
            payload = {"mode": "reset", "candles": normalized}
            self._candles = normalized

        self._status.setText("Live candles from engine stream")

        if self._web_ready:
            self._bridge.updateCandles.emit(payload)

    def _resolve_time(self, candle: dict[str, Any], idx: int) -> int:
        raw_time = candle.get("time", candle.get("timestamp"))
        if raw_time is None:
            return self._base_time + idx * 300

        value = float(raw_time)
        if value > 10_000_000_000:
            return int(value // 1000)
        return int(value)

    def _on_load_finished(self, ok: bool) -> None:
        self._web_ready = bool(ok)
        if not ok:
            self._status.setText("Chart failed to load")
            return

        if self._candles:
            self._bridge.updateCandles.emit({"mode": "reset", "candles": self._candles})

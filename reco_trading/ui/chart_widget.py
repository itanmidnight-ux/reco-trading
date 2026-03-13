from __future__ import annotations

import json
import os
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ChartBridge(QObject):
    fullSnapshot = Signal(str)
    candleUpdate = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._is_ready = False
        self._pending_full: str | None = None

    @Slot()
    def js_ready(self) -> None:
        self._is_ready = True
        if self._pending_full is not None:
            self.fullSnapshot.emit(self._pending_full)
            self._pending_full = None

    def set_full_snapshot(self, candles: list[dict[str, float | int]]) -> None:
        payload = json.dumps(candles, separators=(",", ":"))
        if self._is_ready:
            self.fullSnapshot.emit(payload)
        else:
            self._pending_full = payload

    def update_last_candle(self, candle: dict[str, float | int]) -> None:
        if not self._is_ready:
            return
        self.candleUpdate.emit(json.dumps(candle, separators=(",", ":")))


class CandlestickChartWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_candles: list[dict[str, float | int]] = []

        layout = QVBoxLayout(self)
        self._status = QLabel("Loading TradingView chart…")
        self._status.setObjectName("metricLabel")
        layout.addWidget(self._status)

        self._chart_view = QWebEngineView(self)
        self._chart_view.setObjectName("tradingChartView")
        layout.addWidget(self._chart_view)

        self._bridge = ChartBridge(self)
        self._web_channel = QWebChannel(self._chart_view.page())
        self._web_channel.registerObject("chartBridge", self._bridge)
        self._chart_view.page().setWebChannel(self._web_channel)
        self._chart_view.settings().setAttribute(self._chart_view.settings().WebAttribute.Accelerated2dCanvasEnabled, True)
        self._chart_view.settings().setAttribute(self._chart_view.settings().WebAttribute.WebGLEnabled, True)

        chart_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "assets",
                "chart.html",
            )
        )
        self._chart_view.load(QUrl.fromLocalFile(chart_path))

    def update_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        raw_candles = snapshot.get("candles_5m", [])
        if not raw_candles:
            return

        normalized = [self._normalize_candle(c) for c in raw_candles[-500:]]
        if not normalized:
            return

        if normalized == self._last_candles:
            return

        self._status.setText("Live candles from engine stream")

        if self._should_send_incremental(normalized):
            self._bridge.update_last_candle(normalized[-1])
        else:
            self._bridge.set_full_snapshot(normalized)

        self._last_candles = normalized

    def _should_send_incremental(self, normalized: list[dict[str, float | int]]) -> bool:
        if not self._last_candles:
            return False

        previous = self._last_candles
        if len(normalized) == len(previous):
            return normalized[:-1] == previous[:-1]

        if len(normalized) == len(previous) + 1:
            return normalized[:-1] == previous

        return False

    def _normalize_candle(self, raw: dict[str, Any]) -> dict[str, float | int]:
        timestamp = self._normalize_time(raw)
        open_price = float(raw.get("open", 0.0))
        high = float(raw.get("high", open_price))
        low = float(raw.get("low", open_price))
        close = float(raw.get("close", open_price))
        volume = float(raw.get("volume", 0.0))

        return {
            "time": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    def _normalize_time(self, raw: dict[str, Any]) -> int:
        for key in ("time", "timestamp", "open_time", "ts"):
            value = raw.get(key)
            if value is None:
                continue
            try:
                ts = int(float(value))
            except (TypeError, ValueError):
                continue
            return ts // 1000 if ts > 10_000_000_000 else ts
        return 0

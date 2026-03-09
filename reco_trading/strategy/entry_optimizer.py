from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EntryState:
    waiting_entry: bool = False
    signal_side: str | None = None
    signal_price: float | None = None
    lowest_price: float | None = None
    highest_price: float | None = None
    wait_cycles: int = 0


class EntryOptimizer:
    """Delays entries until rebound confirmation or timeout."""

    def __init__(
        self,
        rebound_pct: float = 0.002,
        timeout_cycles: int = 10,
        falling_knife_threshold: float = -0.015,
        adx_trend_threshold: float = 25.0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.rebound_pct = rebound_pct
        self.timeout_cycles = timeout_cycles
        self.falling_knife_threshold = falling_knife_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.state = EntryState()

    def process(self, signal: str, market_data: dict[str, Any]) -> str | None:
        side = str(signal).upper()
        price = float(market_data.get("price") or 0.0)
        if side not in {"BUY", "SELL"} or price <= 0:
            self._reset()
            return None

        if side == "BUY" and self._buy_blocked(market_data):
            self._reset()
            self.logger.info("FALLING KNIFE BLOCK")
            return None

        if not self.state.waiting_entry or self.state.signal_side != side:
            self.state.waiting_entry = True
            self.state.signal_side = side
            self.state.signal_price = price
            self.state.lowest_price = price
            self.state.highest_price = price
            self.state.wait_cycles = 0
            self.logger.info("ENTRY WAITING")
            return None

        self.state.wait_cycles += 1

        if side == "BUY":
            self.state.lowest_price = min(self.state.lowest_price or price, price)
            rebound_level = (self.state.lowest_price or price) * (1 + self.rebound_pct)
            if price > rebound_level:
                self.logger.info("ENTRY REBOUND DETECTED")
                self._reset()
                return "BUY"
        else:
            self.state.highest_price = max(self.state.highest_price or price, price)
            rebound_level = (self.state.highest_price or price) * (1 - self.rebound_pct)
            if price < rebound_level:
                self.logger.info("ENTRY REBOUND DETECTED")
                self._reset()
                return "SELL"

        if self.state.wait_cycles >= self.timeout_cycles:
            self.logger.info("ENTRY TIMEOUT EXECUTION")
            timed_out_side = side
            self._reset()
            return timed_out_side

        return None

    def _buy_blocked(self, market_data: dict[str, Any]) -> bool:
        if self._is_falling_knife(market_data):
            return True
        if self._is_strong_downtrend(market_data):
            return True
        return False

    def _is_falling_knife(self, market_data: dict[str, Any]) -> bool:
        frame5 = market_data.get("frame5")
        if frame5 is None or len(frame5) < 3:
            return False
        close_now = float(frame5["close"].iloc[-1])
        close_10m_ago = float(frame5["close"].iloc[-3])
        if close_10m_ago <= 0:
            return False
        change = (close_now - close_10m_ago) / close_10m_ago
        return change < self.falling_knife_threshold

    def _is_strong_downtrend(self, market_data: dict[str, Any]) -> bool:
        frame5 = market_data.get("frame5")
        adx = float(market_data.get("adx") or 0.0)
        if frame5 is None or len(frame5) < 2:
            return False

        row = frame5.iloc[-1]
        ema20 = float(row.get("ema20", 0.0))
        ema50 = float(row.get("ema50", 0.0))
        close_price = float(row.get("close", 0.0))
        return adx > self.adx_trend_threshold and ema20 < ema50 and close_price < ema20

    def _reset(self) -> None:
        self.state = EntryState()

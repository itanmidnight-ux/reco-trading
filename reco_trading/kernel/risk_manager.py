from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RiskDecision:
    blocked: bool
    reason: str
    position_size: float


class RiskManager:
    def __init__(
        self,
        max_drawdown: float,
        max_latency_ms: float,
        max_spread_bps: float,
        max_risk_fraction: float,
        max_daily_loss: float | None = None,
        max_trades_per_hour: int | None = None,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_latency_ms = max_latency_ms
        self.max_spread_bps = max_spread_bps
        self.max_risk_fraction = max_risk_fraction
        self.max_daily_loss = float(max_daily_loss) if max_daily_loss is not None else None
        self.max_trades_per_hour = int(max_trades_per_hour) if max_trades_per_hour is not None else None
        self.kill_switch_active = False

    def evaluate_kill_switch(
        self,
        drawdown: float,
        latency_ms: float,
        spread_bps: float,
        execution_error: bool = False,
        daily_pnl: float = 0.0,
        trades_last_hour: int = 0,
    ) -> tuple[bool, str]:
        if self.kill_switch_active:
            return True, 'kill_switch_active'
        if drawdown >= self.max_drawdown:
            self.kill_switch_active = True
            return True, 'drawdown_limit'
        if latency_ms >= self.max_latency_ms:
            self.kill_switch_active = True
            return True, 'latency_limit'
        if spread_bps >= self.max_spread_bps:
            return True, 'spread_anomaly'
        if self.max_daily_loss is not None and daily_pnl <= -abs(self.max_daily_loss):
            self.kill_switch_active = True
            return True, 'daily_loss_limit'
        if self.max_trades_per_hour is not None and trades_last_hour >= self.max_trades_per_hour:
            return True, 'max_trades_per_hour'
        if execution_error:
            self.kill_switch_active = True
            return True, 'execution_error'
        return False, 'ok'

    def compute_position_size(self, equity: float, confidence: float, atr: float) -> float:
        safe_atr = max(float(atr), 1e-9)
        risk_budget = float(equity) * float(self.max_risk_fraction) * float(np.clip(confidence, 0.0, 1.0))
        return max(risk_budget / safe_atr, 0.0)

    def validate_trade(
        self,
        *,
        drawdown: float,
        latency_ms: float,
        spread_bps: float,
        equity: float,
        confidence: float,
        atr: float,
        daily_pnl: float = 0.0,
        trades_last_hour: int = 0,
    ) -> RiskDecision:
        blocked, reason = self.evaluate_kill_switch(
            drawdown,
            latency_ms,
            spread_bps,
            daily_pnl=daily_pnl,
            trades_last_hour=trades_last_hour,
        )
        if blocked:
            return RiskDecision(True, reason, 0.0)
        position_size = self.compute_position_size(equity, confidence, atr)
        if position_size <= 0.0:
            return RiskDecision(True, 'zero_position_size', 0.0)
        return RiskDecision(False, 'ok', position_size)

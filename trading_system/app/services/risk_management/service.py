from __future__ import annotations

from dataclasses import dataclass

from trading_system.app.config.settings import Settings
from trading_system.app.services.decision_engine.service import Decision
from trading_system.app.services.feature_engineering.pipeline import FeatureVector


@dataclass
class RiskPlan:
    allow: bool
    qty: float
    stop_loss: float
    take_profit: float
    trailing: float
    reason: str


class RiskManagementService:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.peak = 1.0
        self.equity = 1.0
        self.losses = 0

    def update(self, pnl: float) -> None:
        self.equity *= 1 + pnl
        self.peak = max(self.peak, self.equity)
        self.losses = self.losses + 1 if pnl < 0 else 0

    def plan(self, decision: Decision, f: FeatureVector, price: float) -> RiskPlan:
        drawdown = 1 - self.equity / self.peak
        if drawdown >= self.s.max_drawdown:
            return RiskPlan(False, 0, 0, 0, 0, 'kill-switch drawdown')
        if self.losses >= self.s.max_consecutive_losses:
            return RiskPlan(False, 0, 0, 0, 0, 'cooldown por pérdidas')
        if decision.signal == 'HOLD' or price <= 0:
            return RiskPlan(False, 0, 0, 0, 0, 'sin trade')

        atr = max(f.atr, price * 0.002)
        qty = max(0.0, (self.s.risk_per_trade * (1 + decision.confidence)) / max(atr, 1e-9))
        if decision.signal == 'LONG':
            sl, tp = price - 1.8 * atr, price + 2.6 * atr
        else:
            sl, tp = price + 1.8 * atr, price - 2.6 * atr
        return RiskPlan(True, qty, sl, tp, 1.2 * atr, 'ok')

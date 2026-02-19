from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    size_btc: float
    stop_price: float


class RiskEngine:
    def __init__(
        self,
        risk_per_trade: float,
        max_daily_drawdown: float,
        max_consecutive_losses: int,
        atr_multiplier: float,
        volatility_target: float,
        circuit_breaker_volatility: float,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.max_daily_drawdown = max_daily_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.atr_multiplier = atr_multiplier
        self.volatility_target = volatility_target
        self.circuit_breaker_volatility = circuit_breaker_volatility

    def evaluate(
        self,
        equity: float,
        daily_pnl: float,
        consecutive_losses: int,
        price: float,
        atr: float,
        annualized_vol: float,
        side: str,
    ) -> RiskDecision:
        if abs(daily_pnl) / max(equity, 1) >= self.max_daily_drawdown and daily_pnl < 0:
            return RiskDecision(False, 'max_daily_drawdown_hit', 0.0, price)
        if consecutive_losses >= self.max_consecutive_losses:
            return RiskDecision(False, 'max_consecutive_losses_hit', 0.0, price)
        if annualized_vol >= self.circuit_breaker_volatility:
            return RiskDecision(False, 'volatility_circuit_breaker_hit', 0.0, price)
        if atr <= 0:
            return RiskDecision(False, 'invalid_atr', 0.0, price)

        stop_distance = atr * self.atr_multiplier
        risk_usdt = equity * self.risk_per_trade
        raw_size = risk_usdt / stop_distance
        vol_adjust = min(1.0, self.volatility_target / max(annualized_vol, 1e-6))
        size = raw_size * vol_adjust
        stop_price = price - stop_distance if side == 'BUY' else price + stop_distance

        return RiskDecision(True, 'ok', max(size, 0.0), stop_price)

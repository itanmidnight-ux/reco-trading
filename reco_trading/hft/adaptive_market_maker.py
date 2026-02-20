from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MarketMakingState:
    inventory: float = 0.0
    skew: float = 0.0
    risk_aversion_gamma: float = 0.1
    max_inventory: float = 100.0
    inventory_neutralization_ratio: float = 0.7
    base_order_size: float = 1.0
    min_order_size: float = 0.05
    unwind_ratio: float = 0.25


@dataclass(slots=True)
class QuoteDecision:
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    spread: float
    reservation_price: float
    should_unwind: bool
    unwind_size: float


class AdaptiveMarketMaker:
    def __init__(
        self,
        state: MarketMakingState,
        spread_factor: float = 1.0,
        volatility_high_threshold: float = 0.03,
        vpin_threshold: float = 0.65,
        high_vol_multiplier: float = 1.35,
        liquidity_shock_multiplier: float = 1.6,
        vpin_multiplier: float = 1.25,
        inventory_risk_coefficient: float = 1.0,
    ) -> None:
        self.state = state
        self.spread_factor = float(max(spread_factor, 0.0))
        self.volatility_high_threshold = float(max(volatility_high_threshold, 0.0))
        self.vpin_threshold = float(max(vpin_threshold, 0.0))
        self.high_vol_multiplier = float(max(high_vol_multiplier, 1.0))
        self.liquidity_shock_multiplier = float(max(liquidity_shock_multiplier, 1.0))
        self.vpin_multiplier = float(max(vpin_multiplier, 1.0))
        self.inventory_risk_coefficient = float(max(inventory_risk_coefficient, 0.0))

    def _dynamic_spread_multiplier(self, volatility: float, liquidity_shock: bool, vpin: float) -> float:
        multiplier = 1.0
        if volatility >= self.volatility_high_threshold:
            multiplier *= self.high_vol_multiplier
        if liquidity_shock:
            multiplier *= self.liquidity_shock_multiplier
        if vpin >= self.vpin_threshold:
            multiplier *= self.vpin_multiplier
        return multiplier

    def _avellaneda_stoikov(self, mid_price: float, sigma: float, time_horizon: float) -> tuple[float, float]:
        sigma_sq_t = max(sigma, 0.0) ** 2 * max(time_horizon, 0.0)
        optimal_spread = self.state.risk_aversion_gamma * sigma_sq_t

        inv_ratio = 0.0
        if self.state.max_inventory > 0:
            inv_ratio = self.state.inventory / self.state.max_inventory

        inventory_adjustment = self.inventory_risk_coefficient * self.state.risk_aversion_gamma * sigma_sq_t * inv_ratio
        reservation_price = mid_price - inventory_adjustment + self.state.skew

        return reservation_price, optimal_spread

    def _neutralization_adjustments(self, bid: float, ask: float, bid_size: float, ask_size: float) -> tuple[float, float, float, float, bool, float]:
        threshold = self.state.max_inventory * self.state.inventory_neutralization_ratio
        if threshold <= 0 or abs(self.state.inventory) < threshold:
            return bid, ask, bid_size, ask_size, False, 0.0

        pressure = min(abs(self.state.inventory) / max(self.state.max_inventory, 1e-9), 1.0)
        size_scale = max(0.25, 1.0 - pressure)

        if self.state.inventory > 0:
            bid *= 1.0 - 0.0008 * pressure
            ask *= 1.0 - 0.0003 * pressure
            bid_size *= size_scale
            ask_size *= min(1.5, 1.0 + pressure * 0.4)
        else:
            bid *= 1.0 + 0.0003 * pressure
            ask *= 1.0 + 0.0008 * pressure
            bid_size *= min(1.5, 1.0 + pressure * 0.4)
            ask_size *= size_scale

        unwind_size = max(abs(self.state.inventory) * self.state.unwind_ratio, self.state.min_order_size)
        return bid, ask, bid_size, ask_size, True, unwind_size

    def compute_quotes(
        self,
        mid_price: float,
        atr: float,
        volatility: float,
        sigma: float,
        time_horizon: float,
        vpin: float,
        liquidity_shock: bool,
    ) -> QuoteDecision:
        if mid_price <= 0:
            raise ValueError('mid_price debe ser positivo')

        dynamic_multiplier = self._dynamic_spread_multiplier(volatility=volatility, liquidity_shock=liquidity_shock, vpin=vpin)
        volatility_adjusted_atr = max(atr, 0.0) * dynamic_multiplier

        reservation_price, optimal_spread = self._avellaneda_stoikov(
            mid_price=mid_price,
            sigma=sigma,
            time_horizon=time_horizon,
        )

        spread_component = self.spread_factor * volatility_adjusted_atr
        total_half_spread = max(spread_component + optimal_spread, 1e-9)

        bid = reservation_price - total_half_spread
        ask = reservation_price + total_half_spread

        bid_size = self.state.base_order_size
        ask_size = self.state.base_order_size

        bid, ask, bid_size, ask_size, should_unwind, unwind_size = self._neutralization_adjustments(
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
        )

        return QuoteDecision(
            bid=max(bid, 1e-9),
            ask=max(ask, 1e-9),
            bid_size=max(bid_size, self.state.min_order_size),
            ask_size=max(ask_size, self.state.min_order_size),
            spread=max(ask - bid, 1e-9),
            reservation_price=reservation_price,
            should_unwind=should_unwind,
            unwind_size=unwind_size,
        )

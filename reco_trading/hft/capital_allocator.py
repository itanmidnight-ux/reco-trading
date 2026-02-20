from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AllocationLimits:
    max_global_notional: float
    max_per_exchange_notional: dict[str, float] = field(default_factory=dict)
    max_single_opportunity_notional: float = float('inf')
    max_concentration: float = 0.40
    correlation_penalty_floor: float = 0.20


@dataclass(slots=True)
class AllocationRequest:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    mid_price: float
    expected_edge_bps: float
    correlation_score: float = 0.0


@dataclass(slots=True)
class AllocationResult:
    allowed: bool
    reason: str
    units: float
    notional: float
    debug: dict[str, float]


class CapitalAllocator:
    def __init__(
        self,
        limits: AllocationLimits,
        *,
        capital_isolation: dict[str, float] | None = None,
        max_cross_exchange_notional: float = float('inf'),
    ) -> None:
        self.limits = limits
        self.capital_isolation = capital_isolation or {}
        self.max_cross_exchange_notional = max_cross_exchange_notional

    def allocate(
        self,
        request: AllocationRequest,
        *,
        equity: float,
        inventory_by_exchange: dict[str, float],
        notionals_by_exchange: dict[str, float],
        notionals_by_asset: dict[str, float],
    ) -> AllocationResult:
        if request.mid_price <= 0:
            return AllocationResult(False, 'invalid_mid_price', 0.0, 0.0, {})

        absolute_exchange_notionals = {k: abs(v) for k, v in notionals_by_exchange.items()}
        cross_exchange_notional = sum(absolute_exchange_notionals.values())
        if cross_exchange_notional >= self.max_cross_exchange_notional:
            return AllocationResult(False, 'max_cross_exchange_notional_hit', 0.0, 0.0, {'cross_exchange_notional': cross_exchange_notional})

        global_remaining = max(0.0, self.limits.max_global_notional - cross_exchange_notional)
        if global_remaining <= 0:
            return AllocationResult(False, 'max_global_notional_hit', 0.0, 0.0, {'global_remaining': 0.0})

        buy_notional = absolute_exchange_notionals.get(request.buy_exchange, 0.0)
        sell_notional = absolute_exchange_notionals.get(request.sell_exchange, 0.0)

        buy_limit = self.limits.max_per_exchange_notional.get(request.buy_exchange, self.limits.max_global_notional)
        sell_limit = self.limits.max_per_exchange_notional.get(request.sell_exchange, self.limits.max_global_notional)

        buy_remaining = max(0.0, buy_limit - buy_notional)
        sell_remaining = max(0.0, sell_limit - sell_notional)

        if request.buy_exchange in self.capital_isolation:
            buy_remaining = min(buy_remaining, max(0.0, equity * self.capital_isolation[request.buy_exchange] - buy_notional))
        if request.sell_exchange in self.capital_isolation:
            sell_remaining = min(sell_remaining, max(0.0, equity * self.capital_isolation[request.sell_exchange] - sell_notional))

        exchange_remaining = min(buy_remaining, sell_remaining)
        if exchange_remaining <= 0:
            return AllocationResult(False, 'max_exchange_notional_hit', 0.0, 0.0, {'exchange_remaining': 0.0})

        base_notional = min(exchange_remaining, global_remaining, self.limits.max_single_opportunity_notional)

        inventory_penalty = max(0.20, 1.0 - abs(inventory_by_exchange.get(request.buy_exchange, 0.0) - inventory_by_exchange.get(request.sell_exchange, 0.0)))

        symbol_notional = abs(notionals_by_asset.get(request.symbol, 0.0))
        concentration = 0.0 if cross_exchange_notional <= 0 else symbol_notional / max(cross_exchange_notional, 1e-9)
        concentration_penalty = 1.0
        if concentration > self.limits.max_concentration:
            excess = concentration - self.limits.max_concentration
            concentration_penalty = max(0.20, 1.0 - excess)

        correlation_penalty = max(self.limits.correlation_penalty_floor, 1.0 - abs(request.correlation_score))

        edge_multiplier = min(max(request.expected_edge_bps / 10.0, 0.25), 1.5)
        adjusted_notional = base_notional * inventory_penalty * concentration_penalty * correlation_penalty * edge_multiplier
        final_notional = min(adjusted_notional, base_notional)

        if final_notional <= 0:
            return AllocationResult(False, 'allocator_capped_to_zero', 0.0, 0.0, {'base_notional': base_notional})

        units = final_notional / request.mid_price
        return AllocationResult(
            True,
            'ok',
            units,
            final_notional,
            {
                'base_notional': base_notional,
                'inventory_penalty': inventory_penalty,
                'concentration_penalty': concentration_penalty,
                'correlation_penalty': correlation_penalty,
                'edge_multiplier': edge_multiplier,
            },
        )

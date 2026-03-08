from __future__ import annotations

from dataclasses import dataclass
import math

from reco_trading.exchange.binance_client import BinanceClient


@dataclass(slots=True)
class SymbolRules:
    min_qty: float
    step_size: float
    min_notional: float
    tick_size: float


class OrderManager:
    """Validates order quantities against Binance exchange filters."""

    def __init__(self, client: BinanceClient, symbol: str) -> None:
        self.client = client
        self.symbol = symbol
        self.rules: SymbolRules | None = None

    async def sync_rules(self) -> SymbolRules:
        markets = await self.client.load_markets()
        market = markets[self.symbol]
        limits = market.get("limits", {})
        amount = limits.get("amount", {})
        cost = limits.get("cost", {})
        precision = market.get("precision", {})

        step = 10 ** (-int(precision.get("amount", 6)))
        tick = 10 ** (-int(precision.get("price", 2)))
        self.rules = SymbolRules(
            min_qty=float(amount.get("min") or step),
            step_size=float(step),
            min_notional=float(cost.get("min") or 5.0),
            tick_size=float(tick),
        )
        return self.rules

    def normalize_quantity(self, quantity: float) -> float:
        if not self.rules:
            raise RuntimeError("symbol_rules_not_loaded")
        if quantity < self.rules.min_qty:
            return 0.0
        steps = math.floor(quantity / self.rules.step_size)
        return round(steps * self.rules.step_size, 8)

    def validate_notional(self, quantity: float, price: float) -> bool:
        if not self.rules:
            raise RuntimeError("symbol_rules_not_loaded")
        return quantity * price >= self.rules.min_notional

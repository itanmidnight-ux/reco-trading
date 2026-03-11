from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from reco_trading.config.symbols import normalize_symbol
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
        self.symbol = normalize_symbol(symbol)
        self.rules: SymbolRules | None = None

    async def sync_rules(self) -> SymbolRules:
        markets = await self.client.load_markets()
        normalized_symbol = normalize_symbol(self.symbol)
        self.symbol = normalized_symbol
        market = markets.get(normalized_symbol)
        if market is None:
            raise KeyError("Trading symbol not found in exchange markets.")
        filters = {flt.get("filterType"): flt for flt in market.get("info", {}).get("filters", [])}

        lot_size = filters.get("LOT_SIZE", {})
        min_notional_filter = filters.get("MIN_NOTIONAL", {})
        price_filter = filters.get("PRICE_FILTER", {})

        step_size = float(lot_size.get("stepSize") or (10 ** (-int(market.get("precision", {}).get("amount", 6)))))
        tick_size = float(price_filter.get("tickSize") or (10 ** (-int(market.get("precision", {}).get("price", 2)))))
        self.rules = SymbolRules(
            min_qty=float(lot_size.get("minQty") or market.get("limits", {}).get("amount", {}).get("min") or step_size),
            step_size=step_size,
            min_notional=float(min_notional_filter.get("minNotional") or market.get("limits", {}).get("cost", {}).get("min") or 5.0),
            tick_size=tick_size,
        )
        return self.rules

    def normalize_quantity(self, quantity: float) -> float:
        if not self.rules:
            raise RuntimeError("symbol_rules_not_loaded")
        if quantity < self.rules.min_qty:
            return 0.0
        return self._round_to_step(quantity, self.rules.step_size)

    def normalize_price(self, price: float) -> float:
        if not self.rules:
            raise RuntimeError("symbol_rules_not_loaded")
        return self._round_to_step(price, self.rules.tick_size)

    def validate_notional(self, quantity: float, price: float) -> bool:
        if not self.rules:
            raise RuntimeError("symbol_rules_not_loaded")
        return quantity * price >= self.rules.min_notional

    @staticmethod
    def validate_spread(bid: float, ask: float, price: float, max_spread_ratio: float) -> bool:
        spread = max(float(ask) - float(bid), 0.0)
        safe_price = max(float(price), 1e-9)
        return (spread / safe_price) <= max(float(max_spread_ratio), 0.0)

    @staticmethod
    def _round_to_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        dec_value = Decimal(str(value))
        dec_step = Decimal(str(step))
        rounded = (dec_value / dec_step).quantize(Decimal("1"), rounding=ROUND_DOWN) * dec_step
        return float(rounded)

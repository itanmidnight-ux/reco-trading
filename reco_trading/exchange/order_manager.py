from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN

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
            min_notional=float(min_notional_filter.get("minNotional") or market.get("limits", {}).get("cost", {}).get("min") or 10.0),
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


    def normalize_order_quantity(
        self,
        symbol: str,
        price: float,
        quantity: float,
        equity: float,
        max_trade_balance_fraction: float,
    ) -> float | None:
        if not self.rules:
            raise RuntimeError("symbol_rules_not_loaded")

        normalized_symbol = normalize_symbol(symbol)
        if normalized_symbol != self.symbol:
            raise ValueError(f"symbol_rules_mismatch symbol={normalized_symbol} expected={self.symbol}")

        safe_price = max(float(price), 1e-9)
        dec_price = Decimal(str(safe_price))

        # 1) load filter values already synced from Binance exchangeInfo.
        min_qty = Decimal(str(self.rules.min_qty))
        step_size = Decimal(str(self.rules.step_size))
        min_notional = Decimal(str(self.rules.min_notional))

        # 2) quantity required by notional.
        qty_by_notional = min_notional / dec_price

        # 3) minimum valid quantity considering both LOT_SIZE and MIN_NOTIONAL.
        min_valid_qty = max(min_qty, qty_by_notional)

        # 4) apply step size ceiling.
        min_valid_qty = self._ceil_to_step(float(min_valid_qty), float(step_size))
        min_valid_qty_dec = Decimal(str(min_valid_qty))

        # 5) bring the calculated quantity up if below minimum.
        normalized_quantity = Decimal(str(quantity))
        if normalized_quantity < min_valid_qty_dec:
            normalized_quantity = min_valid_qty_dec

        # 6) normalize final quantity to step size.
        normalized_quantity = Decimal(str(self._round_to_step(float(normalized_quantity), float(step_size))))

        # 7) re-check minQty.
        if normalized_quantity < min_qty:
            normalized_quantity = min_qty

        if normalized_quantity <= 0:
            return None

        # 8) final notional.
        notional = dec_price * normalized_quantity

        # 9) enforce risk protection.
        max_notional = Decimal(str(max(float(equity), 0.0))) * Decimal(str(max(float(max_trade_balance_fraction), 0.0)))
        if notional > max_notional:
            return None

        # 10) return normalized quantity.
        return float(normalized_quantity)

    def adjust_quantity_for_min_notional(
        self,
        quantity: float,
        price: float,
        safety_margin: float = 0.02,
    ) -> tuple[float, bool]:
        if not self.rules:
            raise RuntimeError("symbol_rules_not_loaded")

        safe_price = max(float(price), 1e-9)
        normalized_quantity = self.normalize_quantity(float(quantity))
        if normalized_quantity <= 0:
            return 0.0, False
        if self.validate_notional(normalized_quantity, safe_price):
            return normalized_quantity, False

        target_notional = self.rules.min_notional * (1.0 + max(float(safety_margin), 0.0))
        required_quantity = target_notional / safe_price
        adjusted_quantity = self.normalize_quantity(required_quantity)
        if adjusted_quantity <= 0:
            return 0.0, True
        if not self.validate_notional(adjusted_quantity, safe_price):
            adjusted_quantity = self.normalize_quantity(adjusted_quantity + self.rules.step_size)

        return adjusted_quantity, True

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

    @staticmethod
    def _ceil_to_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        dec_value = Decimal(str(value))
        dec_step = Decimal(str(step))
        rounded = (dec_value / dec_step).quantize(Decimal("1"), rounding=ROUND_CEILING) * dec_step
        return float(rounded)

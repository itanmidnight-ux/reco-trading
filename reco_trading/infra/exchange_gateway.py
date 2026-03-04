from __future__ import annotations

from typing import Any


class ExchangeGateway:
    """Single governed access gateway to the exchange client.

    Keeps all trading-related exchange calls behind one indirection layer,
    so execution/risk modules don't touch raw CCXT internals directly.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    async def fetch_ticker(self, symbol: str) -> Any:
        return await self.client.fetch_ticker(symbol)

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Any:
        return await self.client.fetch_order_book(symbol, limit=limit)

    async def fetch_balance(self) -> Any:
        return await self.client.fetch_balance()

    async def sanitize_order_quantity(self, symbol: str, amount: float, reference_price: float) -> float:
        return await self.client.sanitize_order_quantity(symbol, amount=amount, reference_price=reference_price)

    async def submit_market_order(self, symbol: str, side: str, amount: float, client_order_id: str) -> Any:
        return await self.client.create_market_order_with_client_id(
            symbol,
            side,
            amount,
            client_order_id=client_order_id,
            firewall_checked=True,
        )

    async def fetch_order(self, symbol: str, order_id: str) -> Any:
        if not hasattr(self.client, 'fetch_order'):
            if hasattr(self.client, 'wait_for_fill'):
                try:
                    return await self.client.wait_for_fill(symbol, order_id, timeout=1)
                except TypeError:
                    return await self.client.wait_for_fill(symbol, order_id)
            return None
        return await self.client.fetch_order(symbol, order_id)

    async def fetch_order_by_client_order_id(self, symbol: str, client_order_id: str) -> Any:
        return await self.client.fetch_order_by_client_order_id(symbol, client_order_id)

    async def fetch_open_orders(self, symbol: str) -> Any:
        return await self.client.fetch_open_orders(symbol)

    async def wait_for_fill(self, symbol: str, order_id: str, timeout: int = 45) -> Any:
        try:
            return await self.client.wait_for_fill(symbol, order_id, timeout=timeout)
        except TypeError:
            return await self.client.wait_for_fill(symbol, order_id)

    async def get_symbol_rules(self, symbol: str) -> dict[str, float | None]:
        return await self.client.get_symbol_rules(symbol)

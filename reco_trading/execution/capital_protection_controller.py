from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CapitalProtectionDecision:
    allowed: bool
    reason: str


class CapitalProtectionController:
    def __init__(self, client: Any, order_service: Any) -> None:
        self.client = client
        self.order_service = order_service

    async def evaluate(self, *, symbol: str, position_qty: float) -> CapitalProtectionDecision:
        if self.order_service.active_trade_in_progress():
            return CapitalProtectionDecision(False, 'ACTIVE_TRADE_IN_PROGRESS')
        open_orders = await self.client.fetch_open_orders(symbol)
        if open_orders:
            return CapitalProtectionDecision(False, 'ACTIVE_TRADE_IN_PROGRESS')
        if float(position_qty) > 0.0:
            return CapitalProtectionDecision(False, 'ACTIVE_TRADE_IN_PROGRESS')
        return CapitalProtectionDecision(True, 'OK')

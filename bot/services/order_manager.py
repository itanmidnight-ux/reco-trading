from __future__ import annotations

import uuid
from dataclasses import dataclass


@dataclass
class OrderIntent:
    symbol: str
    side: str
    quantity: float
    price: float | None = None


class OrderManager:
    @staticmethod
    def build_client_order_id(symbol: str, side: str) -> str:
        compact = symbol.replace('/', '')
        return f'{compact}-{side}-{uuid.uuid4().hex[:12]}'


__all__ = ['OrderManager', 'OrderIntent']

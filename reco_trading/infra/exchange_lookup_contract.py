from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExchangeOrderLookup:
    ok: bool
    transient_error: bool
    error_type: str
    status: str
    exchange_order_id: str

    @property
    def exists(self) -> bool:
        return bool(self.status)

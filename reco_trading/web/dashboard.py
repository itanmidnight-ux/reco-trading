from __future__ import annotations

from typing import Any


class DashboardService:
    """Backwards-compatible dashboard service helpers.

    This compatibility class preserves utility behavior used by tests and legacy
    integrations while the dashboard implementation lives in `trading_system`.
    """

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _extract_binance_usdt_balance(self, balance_payload: dict[str, Any] | None) -> float:
        if not isinstance(balance_payload, dict):
            return 0.0

        total_bucket = balance_payload.get('total')
        if isinstance(total_bucket, dict) and 'USDT' in total_bucket:
            return self._to_float(total_bucket.get('USDT'))

        free_bucket = balance_payload.get('free')
        if isinstance(free_bucket, dict) and 'USDT' in free_bucket:
            return self._to_float(free_bucket.get('USDT'))

        usdt_bucket = balance_payload.get('USDT')
        if isinstance(usdt_bucket, dict) and 'free' in usdt_bucket:
            return self._to_float(usdt_bucket.get('free'))

        return 0.0

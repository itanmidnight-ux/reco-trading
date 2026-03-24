from __future__ import annotations

import asyncio

from reco_trading.exchange.order_manager import OrderManager


class _ClientWithNotional:
    async def load_markets(self) -> dict:
        return {
            "BTC/USDT": {
                "precision": {"amount": 6, "price": 2},
                "limits": {"amount": {"min": 0.0001}, "cost": {"min": 5.0}},
                "info": {
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.00010000", "stepSize": "0.00010000"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                        {"filterType": "NOTIONAL", "minNotional": "25.00000000"},
                    ]
                },
            }
        }


def test_sync_rules_prefers_notional_filter_when_available() -> None:
    manager = OrderManager(client=_ClientWithNotional(), symbol="BTCUSDT")  # type: ignore[arg-type]

    rules = asyncio.run(manager.sync_rules())

    assert rules.min_notional == 25.0
    assert rules.step_size == 0.0001
    assert rules.tick_size == 0.01

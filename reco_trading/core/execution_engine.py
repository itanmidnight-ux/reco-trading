from __future__ import annotations

import asyncio
from loguru import logger

from reco_trading.infra.binance_client import BinanceClient


class ExecutionEngine:
    def __init__(self, client: BinanceClient, symbol: str) -> None:
        self.client = client
        self.symbol = symbol

    async def execute_market_order(self, side: str, amount: float, max_retries: int = 3) -> dict | None:
        if amount <= 0:
            return None

        for attempt in range(1, max_retries + 1):
            try:
                balance = await self.client.fetch_balance()
                if side == 'BUY' and balance.get('USDT', {}).get('free', 0.0) <= 10:
                    logger.warning('Saldo insuficiente para compra')
                    return None

                order = await self.client.create_market_order(self.symbol, side, amount)
                order_id = order.get('id')
                if not order_id:
                    raise RuntimeError('No order id returned')

                status = await self.client.wait_for_fill(self.symbol, order_id)
                if status:
                    return status
            except Exception as exc:
                logger.exception(f'Order attempt {attempt} failed: {exc}')
                await asyncio.sleep(attempt)
        return None

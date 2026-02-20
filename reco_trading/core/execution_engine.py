from __future__ import annotations

import asyncio

from loguru import logger

from reco_trading.core.microstructure import MicrostructureSnapshot
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database


class ExecutionEngine:
    def __init__(self, client: BinanceClient, symbol: str, db: Database) -> None:
        self.client = client
        self.symbol = symbol
        self.db = db

    async def execute_market_order(
        self,
        side: str,
        amount: float,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        microstructure: MicrostructureSnapshot | None = None,
    ) -> dict | None:
        if amount <= 0:
            return None

        if microstructure:
            amount *= max(0.25, 1.0 - microstructure.vpin)
            if microstructure.liquidity_shock:
                amount *= 0.35

        for attempt in range(1, max_retries + 1):
            try:
                balance = await asyncio.wait_for(self.client.fetch_balance(), timeout=timeout_seconds)
                usdt = float(balance.get('USDT', {}).get('free', 0.0))
                btc = float(balance.get('BTC', {}).get('free', 0.0))

                if side == 'BUY' and usdt <= 15:
                    logger.warning('Saldo USDT insuficiente para compra.')
                    return None
                if side == 'SELL' and btc <= 0.00001:
                    logger.warning('Saldo BTC insuficiente para venta.')
                    return None

                order = await asyncio.wait_for(
                    self.client.create_market_order(self.symbol, side, amount), timeout=timeout_seconds
                )
                await self.db.record_order(order)
                order_id = order.get('id')
                if not order_id:
                    raise RuntimeError('Binance no devolvió order id')

                fill = await asyncio.wait_for(
                    self.client.wait_for_fill(self.symbol, str(order_id)), timeout=timeout_seconds + 20
                )
                if fill:
                    await self.db.record_fill(fill)
                    return fill
                logger.warning(f'Orden {order_id} no confirmó fill dentro del timeout.')
            except Exception as exc:
                logger.exception(f'Intento {attempt}/{max_retries} de orden falló: {exc}')
                await asyncio.sleep(attempt)
        return None

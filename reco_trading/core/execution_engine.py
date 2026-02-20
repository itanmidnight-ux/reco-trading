from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Any

import redis
from loguru import logger

from reco_trading.core.rate_limit_controller import AdaptiveRateLimitController
from reco_trading.core.microstructure import MicrostructureSnapshot
from reco_trading.infra.binance_client import BinanceClient
from reco_trading.infra.database import Database


class ExecutionEngine:
    def __init__(
        self,
        client: BinanceClient,
        symbol: str,
        db: Database,
        redis_url: str = 'redis://localhost:6379/0',
        redis_key: str = 'reco_trading:last_execution',
        max_order_size: float = 100_000.0,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.db = db
        self.max_order_size = max_order_size
        self._rate_limiter = AdaptiveRateLimitController(max_calls=5, period_seconds=1.0)
        self._redis_key = redis_key
        try:
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
            self._redis.ping()
        except Exception:
            self._redis = None

    def _validate_order(self, side: str, amount: float) -> bool:
        if side not in {'BUY', 'SELL'}:
            logger.warning('Orden rechazada: side inválido', side=side)
            return False
        if amount <= 0 or amount > self.max_order_size:
            logger.warning('Orden rechazada: amount inválido', amount=amount)
            return False
        return True

    @staticmethod
    def _validate_microstructure(microstructure: MicrostructureSnapshot | None) -> bool:
        if microstructure is None:
            return True
        if not isinstance(microstructure, MicrostructureSnapshot):
            logger.warning('Microstructure inválido: se esperaba MicrostructureSnapshot', type_received=type(microstructure).__name__)
            return False
        if not 0.0 <= microstructure.vpin <= 1.0:
            logger.warning('Microstructure inválido: vpin fuera de rango', vpin=microstructure.vpin)
            return False
        return True

    async def _has_sufficient_balance(self, side: str, amount: float) -> bool:
        balance = await self.client.fetch_balance()
        usdt = float(balance.get('USDT', {}).get('free', 0.0))
        btc = float(balance.get('BTC', {}).get('free', 0.0))

        if side == 'BUY' and usdt <= 15:
            logger.warning('Saldo USDT insuficiente para compra.', usdt=usdt)
            return False
        if side == 'SELL' and btc < amount:
            logger.warning('Saldo BTC insuficiente para venta.', btc=btc, required=amount)
            return False
        return True

    def _persist_execution(self, payload: dict) -> None:
        if not self._redis:
            return
        try:
            self._redis.set(self._redis_key, json.dumps(payload, ensure_ascii=False))
        except Exception:
            logger.warning('No se pudo persistir el estado de ejecución en Redis')

    async def execute_market_order(
        self,
        side: str,
        amount: float,
        max_retries: int = 5,
        microstructure: MicrostructureSnapshot | None = None,
        timeout_seconds: float = 8.0,
    ) -> dict[str, Any] | None:
        if not self._validate_order(side, amount):
            return None

        if max_retries < 1:
            logger.warning('max_retries inválido, debe ser >= 1', max_retries=max_retries)
            return None

        if timeout_seconds <= 0:
            logger.warning('timeout_seconds inválido, debe ser > 0', timeout_seconds=timeout_seconds)
            return None

        if not self._validate_microstructure(microstructure):
            return None

        if microstructure:
            amount *= max(0.25, 1.0 - microstructure.vpin)
            if microstructure.liquidity_shock:
                amount *= 0.35

        for attempt in range(1, max_retries + 1):
            try:
                await self._rate_limiter.acquire()

                try:
                    has_balance = await asyncio.wait_for(
                        self._has_sufficient_balance(side, amount), timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        'Timeout verificando balance previo a la orden',
                        timeout_seconds=timeout_seconds,
                        attempt=attempt,
                    )
                    continue

                if not has_balance:
                    return None

                try:
                    order = await asyncio.wait_for(
                        self.client.create_market_order(self.symbol, side, amount), timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        'Timeout creando market order',
                        symbol=self.symbol,
                        side=side,
                        timeout_seconds=timeout_seconds,
                        attempt=attempt,
                    )
                    continue

                await self.db.record_order(order)
                order_id = order.get('id')
                if not order_id:
                    raise RuntimeError('Binance no devolvió order id')

                try:
                    fill = await asyncio.wait_for(
                        self.client.wait_for_fill(self.symbol, str(order_id)), timeout=timeout_seconds + 20
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        'Timeout esperando fill de la orden',
                        symbol=self.symbol,
                        order_id=str(order_id),
                        timeout_seconds=timeout_seconds + 20,
                        attempt=attempt,
                    )
                    continue

                if fill:
                    await self.db.record_fill(fill)
                    self._persist_execution(
                        {
                            'symbol': self.symbol,
                            'side': side,
                            'amount': amount,
                            'order_id': str(order_id),
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    return fill
                logger.warning(f'Orden {order_id} no confirmó fill dentro del timeout.')
            except Exception as exc:
                sleep_seconds = min(2 ** (attempt - 1) + random.uniform(0, 0.25), 30)
                logger.exception(f'Intento {attempt}/{max_retries} de orden falló: {exc}')
                await asyncio.sleep(sleep_seconds)
        return None

    async def execute(self, side: str, amount: float) -> dict | None:
        return await self.execute_market_order(side=side, amount=amount)

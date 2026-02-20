from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Any

import redis
from loguru import logger

from reco_trading.core.rate_limit_controller import AdaptiveRateLimitController
from reco_trading.execution.smart_order_router import SmartOrderRouter, VenueSnapshot
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
        institutional_order_threshold: float = 25.0,
        sor: SmartOrderRouter | None = None,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.db = db
        self.max_order_size = max_order_size
        self.institutional_order_threshold = institutional_order_threshold
        self._sor = sor or SmartOrderRouter()
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

    async def _execute_child_order(
        self,
        side: str,
        amount: float,
        timeout_seconds: int = 30,
        max_retries: int = 5,
    ) -> dict | None:
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

    async def execute_market_order(self, side: str, amount: float, max_retries: int = 5) -> dict | None:
        if not self._validate_order(side, amount):
            return None
        return await self._execute_child_order(side=side, amount=amount, max_retries=max_retries)

    async def _execute_institutional_order(self, side: str, amount: float) -> dict | None:
        order_book = await self.client.fetch_order_book(self.symbol, limit=10)
        bids = order_book.get('bids') or []
        asks = order_book.get('asks') or []
        top_bid = float(bids[0][0]) if bids else 0.0
        top_ask = float(asks[0][0]) if asks else top_bid
        spread_bps = ((top_ask - top_bid) / max(top_bid, 1e-9)) * 10_000 if top_bid > 0 else 0.0
        depth = float(sum(level[1] for level in (bids[:5] if side == 'SELL' else asks[:5])))

        venues = [
            VenueSnapshot(
                venue='binance_spot',
                spread_bps=spread_bps,
                depth=max(depth, 1e-6),
                latency_ms=40.0,
                fee_bps=10.0,
                fill_ratio=0.98,
                liquidity=max(depth * 4.0, amount),
            ),
            VenueSnapshot(
                venue='binance_alt',
                spread_bps=spread_bps * 1.03,
                depth=max(depth * 0.8, 1e-6),
                latency_ms=22.0,
                fee_bps=12.0,
                fill_ratio=0.95,
                liquidity=max(depth * 2.0, amount * 0.5),
            ),
        ]

        route = self._sor.route_order(
            amount=amount,
            venues=venues,
            strategy='VWAP',
            slices=5,
            expected_volume_profile=[0.15, 0.20, 0.25, 0.20, 0.20],
        )

        fills: list[dict] = []
        for child in route:
            child_amount = float(child['amount'])
            fill = await self._execute_child_order(side=side, amount=child_amount, max_retries=3)
            if fill:
                fills.append(fill)

        if not fills:
            return None
        return {
            'status': 'institutional_completed',
            'fills': fills,
            'routed_children': len(route),
            'filled_children': len(fills),
        }

    async def execute(self, side: str, amount: float) -> dict | None:
        if amount >= self.institutional_order_threshold:
            return await self._execute_institutional_order(side=side, amount=amount)
        return await self.execute_market_order(side=side, amount=amount)

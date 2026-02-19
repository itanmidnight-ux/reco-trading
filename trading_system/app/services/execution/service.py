from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Awaitable, Callable

from trading_system.app.config.settings import Settings
from trading_system.app.core.rate_limiter import exponential_backoff
from trading_system.app.services.market_data.binance_client import BinanceClient

logger = logging.getLogger(__name__)

OnExecuted = Callable[[dict], Awaitable[None]]


@dataclass
class OrderRequest:
    symbol: str
    side: str
    qty: float
    ref_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_delta: float = 0.0
    trailing_activation: float = 0.0


@dataclass
class ProtectionState:
    symbol: str
    entry_side: str
    qty: float
    stop_loss: float
    take_profit: float
    trailing_delta: float
    trailing_activation: float
    stop_order_id: int = 0
    take_profit_order_id: int = 0
    active: bool = True


class ExecutionService:
    def __init__(self, settings: Settings, binance: BinanceClient, on_executed: OnExecuted | None = None) -> None:
        self.settings = settings
        self.binance = binance
        self.on_executed = on_executed
        self.queue: asyncio.Queue[OrderRequest] = asyncio.Queue()
        self.protections: dict[str, ProtectionState] = {}
        self._reconcile_interval = 5.0

    async def submit(self, req: OrderRequest) -> None:
        await self.queue.put(req)

    async def run(self) -> None:
        last_reconcile = 0.0
        while True:
            try:
                req = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except TimeoutError:
                req = None

            if req is not None:
                try:
                    if self.settings.mode == 'paper':
                        logger.info('[PAPER] %s %s %.6f', req.symbol, req.side, req.qty)
                        await self._emit_event(
                            {
                                'symbol': req.symbol,
                                'side': req.side,
                                'qty': req.qty,
                                'price': req.ref_price,
                                'status': 'entry_filled',
                            }
                        )
                    else:
                        await self._execute_live(req)
                except Exception as exc:  # noqa: BLE001
                    logger.exception('Fallo de ejecución: %s', exc)
                finally:
                    self.queue.task_done()

            now = asyncio.get_running_loop().time()
            if now - last_reconcile >= self._reconcile_interval:
                last_reconcile = now
                if self.settings.mode != 'paper':
                    await self._reconcile_protections()

    async def _emit_event(self, payload: dict) -> None:
        if self.on_executed:
            await self.on_executed(payload)

    async def _execute_live(self, req: OrderRequest) -> None:
        payload = {
            'symbol': req.symbol,
            'side': 'BUY' if req.side == 'LONG' else 'SELL',
            'type': 'MARKET',
            'quantity': f'{req.qty:.6f}',
        }
        order_resp = await self.binance.create_order(payload)
        order_id = int(order_resp.get('orderId', 0))
        if order_id <= 0:
            raise RuntimeError(f'Orden sin orderId válido: {order_resp}')

        for attempt in range(5):
            status = await self.binance.order_status(req.symbol, order_id)
            s = status.get('status')
            if s in {'FILLED', 'PARTIALLY_FILLED'}:
                executed_qty = float(status.get('executedQty') or req.qty)
                quote_qty = float(status.get('cummulativeQuoteQty') or 0.0)
                avg_price = quote_qty / executed_qty if executed_qty > 0 and quote_qty > 0 else float(status.get('price') or req.ref_price or 0.0)
                logger.info('[LIVE] %s %s qty=%.6f status=%s order_id=%s', req.symbol, req.side, req.qty, s, order_id)
                await self._emit_event(
                    {
                        'symbol': req.symbol,
                        'side': req.side,
                        'qty': executed_qty,
                        'price': avg_price,
                        'status': 'entry_filled',
                        'entry_order_id': order_id,
                    }
                )

                if req.stop_loss > 0 or req.take_profit > 0:
                    await self._arm_protection(req, executed_qty)
                return
            if s in {'REJECTED', 'EXPIRED', 'CANCELED'}:
                raise RuntimeError(f'Orden no ejecutable status={s} order_id={order_id}')
            await exponential_backoff(attempt, base=0.2, cap=3)

        raise RuntimeError(f'Timeout confirmando ejecución order_id={order_id}')

    async def _arm_protection(self, req: OrderRequest, executed_qty: float) -> None:
        close_side = 'SELL' if req.side == 'LONG' else 'BUY'
        state = ProtectionState(
            symbol=req.symbol,
            entry_side=req.side,
            qty=executed_qty,
            stop_loss=req.stop_loss,
            take_profit=req.take_profit,
            trailing_delta=req.trailing_delta,
            trailing_activation=req.trailing_activation,
        )

        if req.stop_loss > 0 and req.take_profit > 0:
            try:
                oco_resp = await self.binance.request(
                    'POST',
                    '/api/v3/order/oco',
                    params={
                        'symbol': req.symbol,
                        'side': close_side,
                        'quantity': f'{executed_qty:.6f}',
                        'price': f'{req.take_profit:.8f}',
                        'stopPrice': f'{req.stop_loss:.8f}',
                        'stopLimitPrice': f'{req.stop_loss:.8f}',
                        'stopLimitTimeInForce': 'GTC',
                    },
                    private=True,
                    weight=1,
                )
                reports = oco_resp.get('orderReports') or []
                for rep in reports:
                    oid = int(rep.get('orderId', 0))
                    otype = str(rep.get('type', ''))
                    if otype.startswith('STOP'):
                        state.stop_order_id = oid
                    else:
                        state.take_profit_order_id = oid
            except Exception:  # noqa: BLE001
                logger.warning('No se pudo crear OCO para %s; fallback a órdenes individuales', req.symbol, exc_info=True)

        if req.stop_loss > 0 and state.stop_order_id <= 0:
            stop_payload = {
                'symbol': req.symbol,
                'side': close_side,
                'type': 'STOP_LOSS_LIMIT',
                'timeInForce': 'GTC',
                'quantity': f'{executed_qty:.6f}',
                'price': f'{req.stop_loss:.8f}',
                'stopPrice': f'{req.stop_loss:.8f}',
            }
            if req.trailing_delta > 0:
                stop_payload['trailingDelta'] = int(req.trailing_delta)
            if req.trailing_activation > 0:
                stop_payload['activationPrice'] = f'{req.trailing_activation:.8f}'
            stop_resp = await self.binance.create_order(stop_payload)
            state.stop_order_id = int(stop_resp.get('orderId', 0))

        if req.take_profit > 0 and state.take_profit_order_id <= 0:
            tp_resp = await self.binance.create_order(
                {
                    'symbol': req.symbol,
                    'side': close_side,
                    'type': 'LIMIT',
                    'timeInForce': 'GTC',
                    'quantity': f'{executed_qty:.6f}',
                    'price': f'{req.take_profit:.8f}',
                }
            )
            state.take_profit_order_id = int(tp_resp.get('orderId', 0))

        self.protections[req.symbol] = state
        logger.info(
            '[LIVE] Protección armada %s stop_id=%s tp_id=%s sl=%.8f tp=%.8f',
            req.symbol,
            state.stop_order_id,
            state.take_profit_order_id,
            req.stop_loss,
            req.take_profit,
        )
        await self._emit_event(
            {
                'symbol': req.symbol,
                'side': req.side,
                'qty': executed_qty,
                'price': req.take_profit or req.stop_loss,
                'status': 'protection_armed',
                'stop_loss': req.stop_loss,
                'take_profit': req.take_profit,
                'stop_order_id': state.stop_order_id,
                'take_profit_order_id': state.take_profit_order_id,
            }
        )

    async def _reconcile_protections(self) -> None:
        for symbol, state in list(self.protections.items()):
            if not state.active:
                continue
            open_orders = await self.binance.open_orders(symbol)
            open_ids = {int(o.get('orderId', 0)) for o in open_orders}

            close_price = 0.0
            close_status = ''
            close_order_id = 0

            for oid in [state.stop_order_id, state.take_profit_order_id]:
                if oid > 0 and oid not in open_ids:
                    ord_status = await self.binance.order_status(symbol, oid)
                    s = str(ord_status.get('status', ''))
                    if s == 'FILLED':
                        close_status = 'stop_loss_filled' if oid == state.stop_order_id else 'take_profit_filled'
                        qty = float(ord_status.get('executedQty') or state.qty)
                        quote_qty = float(ord_status.get('cummulativeQuoteQty') or 0.0)
                        close_price = quote_qty / qty if qty > 0 and quote_qty > 0 else float(ord_status.get('price') or 0.0)
                        close_order_id = oid
                        break

            if close_status:
                sibling = state.take_profit_order_id if close_order_id == state.stop_order_id else state.stop_order_id
                if sibling > 0 and sibling in open_ids:
                    await self.binance.cancel_order(symbol, sibling)
                state.active = False
                logger.info('[LIVE] Cierre por protección %s status=%s order_id=%s', symbol, close_status, close_order_id)
                await self._emit_event(
                    {
                        'symbol': symbol,
                        'side': state.entry_side,
                        'qty': state.qty,
                        'price': close_price,
                        'status': close_status,
                        'close_order_id': close_order_id,
                    }
                )
                continue

            missing_stop = state.stop_order_id > 0 and state.stop_order_id not in open_ids
            missing_tp = state.take_profit_order_id > 0 and state.take_profit_order_id not in open_ids
            if missing_stop or missing_tp:
                logger.warning('[LIVE] Protección faltante detectada en %s (stop=%s tp=%s), reponiendo', symbol, missing_stop, missing_tp)
                req = OrderRequest(
                    symbol=symbol,
                    side=state.entry_side,
                    qty=state.qty,
                    stop_loss=state.stop_loss,
                    take_profit=state.take_profit,
                    trailing_delta=state.trailing_delta,
                    trailing_activation=state.trailing_activation,
                )
                await self._arm_protection(req, state.qty)

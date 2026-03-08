from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot.config import BotConfig
from bot.core.portfolio import Portfolio, Position
from bot.core.risk_manager import RiskManager
from bot.exchange.binance_client import BinanceClient
from bot.services.order_manager import OrderIntent, OrderManager
from bot.utils.helpers import floor_step
from bot.utils.logger import logger


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ''


class ExecutionEngine:
    def __init__(
        self,
        config: BotConfig,
        client: BinanceClient,
        risk_manager: RiskManager,
    ) -> None:
        self.config = config
        self.client = client
        self.risk_manager = risk_manager

    async def _extract_filters(self, market: dict[str, Any]) -> tuple[float, float, float]:
        limits = market.get('limits', {})
        precision = market.get('precision', {})
        lot_step = 10 ** (-(precision.get('amount', 6)))
        min_notional = limits.get('cost', {}).get('min', 5.0) or 5.0
        tick_size = 10 ** (-(precision.get('price', 2)))
        return lot_step, min_notional, tick_size

    async def validate_order(self, intent: OrderIntent, portfolio: Portfolio, expected_move: float) -> ValidationResult:
        market = await self.client.fetch_market(intent.symbol)
        lot_step, min_notional, _ = await self._extract_filters(market)
        ticker = await self.client.fetch_ticker(intent.symbol)
        reference_price = intent.price or float(ticker['last'])

        adjusted_qty = floor_step(intent.quantity, lot_step)
        if adjusted_qty <= 0:
            return ValidationResult(False, 'invalid quantity after LOT_SIZE rounding')

        notional = adjusted_qty * reference_price
        if notional < min_notional:
            return ValidationResult(False, f'below MIN_NOTIONAL={min_notional}')

        if portfolio.quote_balance < notional and intent.side == 'buy':
            return ValidationResult(False, 'insufficient quote balance')

        fee_bps = (self.config.maker_fee + self.config.taker_fee) * 10_000
        spread_bps = 5.0
        edge_bps = expected_move * 10_000
        if edge_bps <= (fee_bps + spread_bps + self.config.min_expected_edge_bps):
            return ValidationResult(False, 'expected profit does not exceed fees + spread buffer')

        return ValidationResult(True)

    async def execute_signal(self, symbol: str, side: str, expected_move: float, portfolio: Portfolio) -> None:
        risk = self.risk_manager.validate_new_trade(portfolio)
        if not risk.approved:
            logger.info('Risk blocked trade: {}', risk.reason)
            return

        ticker = await self.client.fetch_ticker(symbol)
        last_price = float(ticker['last'])
        stop_distance = last_price * 0.01
        stop_loss = last_price - stop_distance if side == 'buy' else last_price + stop_distance
        take_profit = last_price + stop_distance * 1.5 if side == 'buy' else last_price - stop_distance * 1.5

        equity = max(portfolio.quote_balance, self.config.order_quote_size_usd)
        quantity = min(
            self.config.order_quote_size_usd / last_price,
            self.risk_manager.position_size_from_risk(equity, last_price, stop_loss),
        )
        intent = OrderIntent(symbol=symbol, side=side, quantity=quantity)

        validation = await self.validate_order(intent, portfolio, expected_move)
        if not validation.ok:
            logger.info('Order blocked: {}', validation.reason)
            return

        client_order_id = OrderManager.build_client_order_id(symbol, side)
        if client_order_id in portfolio.seen_client_order_ids:
            logger.warning('Duplicate order id prevented: {}', client_order_id)
            return

        order = await self.client.create_order(symbol=symbol, side=side, amount=quantity)
        portfolio.seen_client_order_ids.add(client_order_id)
        portfolio.daily_trades += 1
        portfolio.open_position = Position(
            symbol=symbol,
            side=side,
            entry_price=last_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        logger.info('Order executed id={} side={} qty={} price={}', order.get('id'), side, quantity, last_price)

    def update_position_mark_to_market(self, portfolio: Portfolio, mark_price: float) -> None:
        pos = portfolio.open_position
        if not pos:
            return
        pnl_pct = ((mark_price - pos.entry_price) / pos.entry_price) if pos.side == 'buy' else ((pos.entry_price - mark_price) / pos.entry_price)
        if mark_price <= pos.stop_loss or mark_price >= pos.take_profit:
            pnl = pnl_pct
            portfolio.realized_pnl += pnl
            if pnl < 0:
                portfolio.daily_loss += abs(pnl)
            logger.info('Position closed side={} pnl_pct={:.4f}', pos.side, pnl)
            portfolio.open_position = None


__all__ = ['ExecutionEngine', 'ValidationResult']

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger


@dataclass(slots=True)
class FirewallDecision:
    allowed: bool
    reason: str
    risk_snapshot: dict[str, Any]
    recommended_size: float


class ExecutionFirewall:
    def __init__(
        self,
        *,
        max_total_exposure: float = 500_000.0,
        max_asset_exposure: float = 250_000.0,
        max_exchange_exposure: float = 500_000.0,
        max_daily_loss: float = 10_000.0,
        max_daily_notional: float = 1_000_000.0,
        max_slippage_bps: float = 35.0,
        min_liquidity_coverage: float = 2.5,
        exchange_name: str = 'binance',
    ) -> None:
        self.max_total_exposure = float(max_total_exposure)
        self.max_asset_exposure = float(max_asset_exposure)
        self.max_exchange_exposure = float(max_exchange_exposure)
        self.max_daily_loss = float(max_daily_loss)
        self.max_daily_notional = float(max_daily_notional)
        self.max_slippage_bps = float(max_slippage_bps)
        self.min_liquidity_coverage = float(min_liquidity_coverage)
        self.exchange_name = exchange_name

        self._asset_exposure: dict[str, float] = {}
        self._exchange_exposure: dict[str, float] = {}
        self._total_exposure = 0.0
        self._daily_notional = 0.0
        self._daily_realized_pnl = 0.0
        self._day_key = self._today_key()

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _roll_day_if_needed(self) -> None:
        now_key = self._today_key()
        if now_key == self._day_key:
            return
        self._day_key = now_key
        self._daily_notional = 0.0
        self._daily_realized_pnl = 0.0

    @staticmethod
    def _base_asset(symbol: str) -> str:
        if '/' in symbol:
            return symbol.split('/', 1)[0]
        return symbol

    async def evaluate(
        self,
        *,
        client: Any,
        symbol: str,
        side: str,
        amount: float,
    ) -> FirewallDecision:
        self._roll_day_if_needed()
        base_asset = self._base_asset(symbol)

        order_book = await client.fetch_order_book(symbol, limit=20)
        bids = order_book.get('bids') or []
        asks = order_book.get('asks') or []
        top_bid = float(bids[0][0]) if bids else 0.0
        top_ask = float(asks[0][0]) if asks else top_bid
        mid_price = (top_bid + top_ask) / 2 if top_bid > 0 and top_ask > 0 else max(top_bid, top_ask, 1.0)
        side_levels = bids if side == 'SELL' else asks
        depth = float(sum(float(level[1]) for level in side_levels[:10]))
        spread_bps = ((top_ask - top_bid) / max(mid_price, 1e-9)) * 10_000 if top_ask and top_bid else 0.0
        impact_bps = (amount / max(depth, 1e-9)) * 100.0 if depth > 0 else 10_000.0
        estimated_slippage_bps = spread_bps / 2.0 + impact_bps

        notional = amount * mid_price
        projected_asset_exposure = self._asset_exposure.get(base_asset, 0.0) + notional
        projected_exchange_exposure = self._exchange_exposure.get(self.exchange_name, 0.0) + notional
        projected_total_exposure = self._total_exposure + notional
        projected_daily_notional = self._daily_notional + notional

        balance = await client.fetch_balance()
        usdt = float(balance.get('USDT', {}).get('free', 0.0))
        base_free = float(balance.get(base_asset, {}).get('free', 0.0))

        liquidity_capacity = max(depth / max(self.min_liquidity_coverage, 1e-9), 0.0)
        recommended_size = min(amount, liquidity_capacity)
        exposure_room_notional = min(
            max(self.max_total_exposure - self._total_exposure, 0.0),
            max(self.max_asset_exposure - self._asset_exposure.get(base_asset, 0.0), 0.0),
            max(self.max_exchange_exposure - self._exchange_exposure.get(self.exchange_name, 0.0), 0.0),
            max(self.max_daily_notional - self._daily_notional, 0.0),
        )
        if mid_price > 0:
            recommended_size = min(recommended_size, exposure_room_notional / mid_price)

        snapshot = {
            'symbol': symbol,
            'side': side,
            'amount': float(amount),
            'notional': float(notional),
            'price': float(mid_price),
            'spread_bps': float(spread_bps),
            'estimated_slippage_bps': float(estimated_slippage_bps),
            'available_liquidity': float(depth),
            'daily_notional': float(self._daily_notional),
            'daily_realized_pnl': float(self._daily_realized_pnl),
            'projected_total_exposure': float(projected_total_exposure),
            'projected_asset_exposure': float(projected_asset_exposure),
            'projected_exchange_exposure': float(projected_exchange_exposure),
            'limits': {
                'max_total_exposure': self.max_total_exposure,
                'max_asset_exposure': self.max_asset_exposure,
                'max_exchange_exposure': self.max_exchange_exposure,
                'max_daily_loss': self.max_daily_loss,
                'max_daily_notional': self.max_daily_notional,
                'max_slippage_bps': self.max_slippage_bps,
                'min_liquidity_coverage': self.min_liquidity_coverage,
            },
        }

        if side == 'BUY' and usdt < notional:
            return FirewallDecision(False, 'insufficient_quote_balance', snapshot, max(0.0, usdt / max(mid_price, 1e-9)))
        if side == 'SELL' and base_free < amount:
            return FirewallDecision(False, 'insufficient_base_balance', snapshot, max(0.0, base_free))
        if projected_total_exposure > self.max_total_exposure:
            return FirewallDecision(False, 'total_exposure_limit', snapshot, max(0.0, recommended_size))
        if projected_asset_exposure > self.max_asset_exposure:
            return FirewallDecision(False, 'asset_exposure_limit', snapshot, max(0.0, recommended_size))
        if projected_exchange_exposure > self.max_exchange_exposure:
            return FirewallDecision(False, 'exchange_exposure_limit', snapshot, max(0.0, recommended_size))
        if -self._daily_realized_pnl >= self.max_daily_loss:
            return FirewallDecision(False, 'daily_loss_limit', snapshot, 0.0)
        if projected_daily_notional > self.max_daily_notional:
            return FirewallDecision(False, 'daily_notional_limit', snapshot, max(0.0, recommended_size))
        if estimated_slippage_bps > self.max_slippage_bps:
            return FirewallDecision(False, 'slippage_limit', snapshot, max(0.0, recommended_size))
        if depth < amount * self.min_liquidity_coverage:
            return FirewallDecision(False, 'insufficient_liquidity', snapshot, max(0.0, recommended_size))

        return FirewallDecision(True, 'allowed', snapshot, float(amount))

    def register_fill(
        self,
        *,
        symbol: str,
        notional: float,
        realized_pnl: float = 0.0,
    ) -> None:
        self._roll_day_if_needed()
        asset = self._base_asset(symbol)
        notional = max(float(notional), 0.0)

        self._asset_exposure[asset] = self._asset_exposure.get(asset, 0.0) + notional
        self._exchange_exposure[self.exchange_name] = self._exchange_exposure.get(self.exchange_name, 0.0) + notional
        self._total_exposure += notional
        self._daily_notional += notional
        self._daily_realized_pnl += float(realized_pnl)
        logger.bind(component='execution_firewall').info(
            'Exposición del firewall actualizada',
            symbol=symbol,
            asset=asset,
            notional=notional,
            total_exposure=self._total_exposure,
            daily_notional=self._daily_notional,
        )

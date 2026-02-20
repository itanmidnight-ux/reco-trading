from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


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
        min_liquidity_coverage: float = 2.0,
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

    @staticmethod
    def _base_asset(symbol: str) -> str:
        return symbol.split('/', 1)[0] if '/' in symbol else symbol

    @staticmethod
    def _today_key() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _roll_day_if_needed(self) -> None:
        now_key = self._today_key()
        if now_key != self._day_key:
            self._day_key = now_key
            self._daily_notional = 0.0
            self._daily_realized_pnl = 0.0

    async def _min_size(self, client: Any, symbol: str) -> float:
        markets = getattr(client.exchange, 'markets', None) if hasattr(client, 'exchange') else None
        if not markets:
            load = getattr(client.exchange, 'load_markets', None) if hasattr(client, 'exchange') else None
            if load:
                await load()
                markets = client.exchange.markets
        if not markets or symbol not in markets:
            return 0.0
        limits = markets[symbol].get('limits', {})
        amount_limit = limits.get('amount', {}).get('min')
        if amount_limit is None:
            return 0.0
        return float(amount_limit)

    async def evaluate(self, *, client: Any, symbol: str, side: str, amount: float) -> FirewallDecision:
        self._roll_day_if_needed()
        side = side.upper()
        base_asset = self._base_asset(symbol)

        order_book = await client.fetch_order_book(symbol, limit=20)
        bids = order_book.get('bids') or []
        asks = order_book.get('asks') or []
        top_bid = float(bids[0][0]) if bids else 0.0
        top_ask = float(asks[0][0]) if asks else 0.0
        mid = (top_bid + top_ask) / 2 if top_bid > 0 and top_ask > 0 else max(top_bid, top_ask, 1.0)

        side_levels = asks if side == 'BUY' else bids
        depth = float(sum(float(x[1]) for x in side_levels[:10]))
        spread_bps = ((top_ask - top_bid) / max(mid, 1e-9)) * 10_000 if top_ask and top_bid else 0.0
        impact_bps = (amount / max(depth, 1e-9)) * 100.0 if depth else 10_000.0
        estimated_slippage_bps = spread_bps / 2 + impact_bps

        min_size = await self._min_size(client, symbol)
        notional = amount * mid
        proj_asset = self._asset_exposure.get(base_asset, 0.0) + notional
        proj_exchange = self._exchange_exposure.get(self.exchange_name, 0.0) + notional
        proj_total = self._total_exposure + notional
        proj_daily = self._daily_notional + notional

        balance = await client.fetch_balance()
        usdt = float(balance.get('USDT', {}).get('free', 0.0))
        base_free = float(balance.get(base_asset, {}).get('free', 0.0))

        recommended = min(amount, max(depth / max(self.min_liquidity_coverage, 1e-9), 0.0))
        snap = {
            'symbol': symbol,
            'side': side,
            'amount': float(amount),
            'min_size': float(min_size),
            'mid_price': float(mid),
            'notional': float(notional),
            'spread_bps': float(spread_bps),
            'estimated_slippage_bps': float(estimated_slippage_bps),
            'available_liquidity': float(depth),
            'daily_realized_pnl': float(self._daily_realized_pnl),
            'projected_total_exposure': float(proj_total),
            'projected_asset_exposure': float(proj_asset),
            'projected_exchange_exposure': float(proj_exchange),
            'projected_daily_notional': float(proj_daily),
        }

        if amount < min_size:
            return FirewallDecision(False, 'binance_min_size', snap, max(min_size, recommended))
        if side == 'BUY' and usdt < notional:
            return FirewallDecision(False, 'insufficient_quote_balance', snap, max(0.0, usdt / max(mid, 1e-9)))
        if side == 'SELL' and base_free < amount:
            return FirewallDecision(False, 'insufficient_base_balance', snap, max(0.0, base_free))
        if proj_total > self.max_total_exposure:
            return FirewallDecision(False, 'total_exposure_limit', snap, max(0.0, recommended))
        if proj_asset > self.max_asset_exposure:
            return FirewallDecision(False, 'asset_exposure_limit', snap, max(0.0, recommended))
        if proj_exchange > self.max_exchange_exposure:
            return FirewallDecision(False, 'exchange_exposure_limit', snap, max(0.0, recommended))
        if -self._daily_realized_pnl >= self.max_daily_loss:
            return FirewallDecision(False, 'daily_loss_limit', snap, 0.0)
        if proj_daily > self.max_daily_notional:
            return FirewallDecision(False, 'daily_notional_limit', snap, max(0.0, recommended))
        if estimated_slippage_bps > self.max_slippage_bps:
            return FirewallDecision(False, 'slippage_limit', snap, max(0.0, recommended))
        if depth < amount * self.min_liquidity_coverage:
            return FirewallDecision(False, 'insufficient_liquidity', snap, max(0.0, recommended))
        return FirewallDecision(True, 'allowed', snap, float(amount))

    def register_fill(self, *, symbol: str, notional: float, realized_pnl: float = 0.0) -> None:
        self._roll_day_if_needed()
        asset = self._base_asset(symbol)
        n = max(float(notional), 0.0)
        self._asset_exposure[asset] = self._asset_exposure.get(asset, 0.0) + n
        self._exchange_exposure[self.exchange_name] = self._exchange_exposure.get(self.exchange_name, 0.0) + n
        self._total_exposure += n
        self._daily_notional += n
        self._daily_realized_pnl += float(realized_pnl)

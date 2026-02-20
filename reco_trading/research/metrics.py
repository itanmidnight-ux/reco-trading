from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from scipy.stats import kurtosis, skew


@dataclass(frozen=True)
class NormalizedFill:
    timestamp: datetime
    side: str
    price: float
    quantity: float
    exchange: str = 'UNKNOWN'
    strategy: str = 'default'
    fee: float = 0.0


@dataclass(frozen=True)
class NormalizedQuote:
    timestamp: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    exchange: str = 'UNKNOWN'
    strategy: str = 'default'


@dataclass(frozen=True)
class NormalizedMidpricePoint:
    timestamp: datetime
    midprice: float
    exchange: str = 'UNKNOWN'
    strategy: str = 'default'


@dataclass(frozen=True)
class NormalizedCancelEvent:
    timestamp: datetime
    side: str
    quantity: float
    exchange: str = 'UNKNOWN'
    strategy: str = 'default'


def _to_datetime(value) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return datetime.now(timezone.utc)


def normalize_fills(fills: Iterable[dict | NormalizedFill]) -> list[NormalizedFill]:
    normalized = []
    for item in fills:
        if isinstance(item, NormalizedFill):
            normalized.append(item)
            continue
        side = str(item.get('side', '')).upper()
        if side not in {'BUY', 'SELL'}:
            continue
        normalized.append(
            NormalizedFill(
                timestamp=_to_datetime(item.get('timestamp') or item.get('created_at')),
                side=side,
                price=float(item.get('price') or item.get('fill_price') or item.get('average') or 0.0),
                quantity=abs(float(item.get('quantity') or item.get('fill_amount') or item.get('filled') or 0.0)),
                exchange=str(item.get('exchange') or 'UNKNOWN'),
                strategy=str(item.get('strategy') or 'default'),
                fee=float((item.get('fee') or {}).get('cost', 0.0)) if isinstance(item.get('fee'), dict) else float(item.get('fee') or 0.0),
            )
        )
    return sorted(normalized, key=lambda f: f.timestamp)


def normalize_quotes(quotes: Iterable[dict | NormalizedQuote]) -> list[NormalizedQuote]:
    normalized = []
    for item in quotes:
        if isinstance(item, NormalizedQuote):
            normalized.append(item)
            continue
        bid = float(item.get('bid') or 0.0)
        ask = float(item.get('ask') or 0.0)
        normalized.append(
            NormalizedQuote(
                timestamp=_to_datetime(item.get('timestamp') or item.get('created_at')),
                bid=bid,
                ask=ask,
                bid_size=abs(float(item.get('bid_size') or 0.0)),
                ask_size=abs(float(item.get('ask_size') or 0.0)),
                exchange=str(item.get('exchange') or 'UNKNOWN'),
                strategy=str(item.get('strategy') or 'default'),
            )
        )
    return sorted(normalized, key=lambda q: q.timestamp)


def normalize_midprice_path(path: Iterable[dict | NormalizedMidpricePoint]) -> list[NormalizedMidpricePoint]:
    normalized = []
    for item in path:
        if isinstance(item, NormalizedMidpricePoint):
            normalized.append(item)
            continue
        normalized.append(
            NormalizedMidpricePoint(
                timestamp=_to_datetime(item.get('timestamp') or item.get('created_at')),
                midprice=float(item.get('midprice') or item.get('mid') or item.get('price') or 0.0),
                exchange=str(item.get('exchange') or 'UNKNOWN'),
                strategy=str(item.get('strategy') or 'default'),
            )
        )
    return sorted(normalized, key=lambda p: p.timestamp)


def normalize_cancel_events(events: Iterable[dict | NormalizedCancelEvent]) -> list[NormalizedCancelEvent]:
    normalized = []
    for item in events:
        if isinstance(item, NormalizedCancelEvent):
            normalized.append(item)
            continue
        side = str(item.get('side', '')).upper()
        if side not in {'BUY', 'SELL'}:
            continue
        normalized.append(
            NormalizedCancelEvent(
                timestamp=_to_datetime(item.get('timestamp') or item.get('created_at')),
                side=side,
                quantity=abs(float(item.get('quantity') or item.get('amount') or 0.0)),
                exchange=str(item.get('exchange') or 'UNKNOWN'),
                strategy=str(item.get('strategy') or 'default'),
            )
        )
    return sorted(normalized, key=lambda e: e.timestamp)


def _signed_qty(fill: NormalizedFill) -> float:
    return fill.quantity if fill.side == 'BUY' else -fill.quantity


def inventory_turnover(fills: Iterable[dict | NormalizedFill]) -> float:
    f = normalize_fills(fills)
    if not f:
        return 0.0
    signed = np.asarray([_signed_qty(x) for x in f], dtype=float)
    inv = np.cumsum(signed)
    avg_inventory = np.mean(np.abs(inv))
    return float(np.sum(np.abs(signed)) / (avg_inventory + 1e-12))


def quote_fill_ratio(fills: Iterable[dict | NormalizedFill], quotes: Iterable[dict | NormalizedQuote]) -> float:
    f = normalize_fills(fills)
    q = normalize_quotes(quotes)
    filled = float(sum(x.quantity for x in f))
    quoted = float(sum(x.bid_size + x.ask_size for x in q))
    return float(filled / (quoted + 1e-12))


def _mid_lookup(midprice_path: list[NormalizedMidpricePoint], ts: datetime, horizon_steps: int = 0) -> float:
    if not midprice_path:
        return 0.0
    times = np.array([p.timestamp.timestamp() for p in midprice_path], dtype=float)
    idx = int(np.searchsorted(times, ts.timestamp(), side='left'))
    idx = min(max(idx, 0), len(midprice_path) - 1)
    idx = min(idx + max(horizon_steps, 0), len(midprice_path) - 1)
    return float(midprice_path[idx].midprice)


def adverse_selection_cost(
    fills: Iterable[dict | NormalizedFill],
    midprice_path: Iterable[dict | NormalizedMidpricePoint],
    horizon_steps: int = 1,
) -> float:
    f = normalize_fills(fills)
    mids = normalize_midprice_path(midprice_path)
    if not f or not mids:
        return 0.0
    total_qty = 0.0
    total_cost = 0.0
    for fill in f:
        signed = 1.0 if fill.side == 'BUY' else -1.0
        future_mid = _mid_lookup(mids, fill.timestamp, horizon_steps=horizon_steps)
        total_cost += -signed * (future_mid - fill.price) * fill.quantity
        total_qty += fill.quantity
    return float(total_cost / (total_qty + 1e-12))


def realized_spread(
    fills: Iterable[dict | NormalizedFill],
    midprice_path: Iterable[dict | NormalizedMidpricePoint],
    horizon_steps: int = 1,
) -> float:
    return float(2.0 * adverse_selection_cost(fills, midprice_path, horizon_steps=horizon_steps))


def unrealized_spread(
    fills: Iterable[dict | NormalizedFill],
    midprice_path: Iterable[dict | NormalizedMidpricePoint],
) -> float:
    f = normalize_fills(fills)
    mids = normalize_midprice_path(midprice_path)
    if not f or not mids:
        return 0.0
    total_qty = 0.0
    total = 0.0
    for fill in f:
        signed = 1.0 if fill.side == 'BUY' else -1.0
        current_mid = _mid_lookup(mids, fill.timestamp, horizon_steps=0)
        total += signed * (fill.price - current_mid) * 2.0 * fill.quantity
        total_qty += fill.quantity
    return float(total / (total_qty + 1e-12))


def microstructure_alpha_decay(
    fills: Iterable[dict | NormalizedFill],
    midprice_path: Iterable[dict | NormalizedMidpricePoint],
    horizons: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    horizons = tuple(sorted({max(int(h), 1) for h in horizons}))
    base_edge = abs(adverse_selection_cost(fills, midprice_path, horizon_steps=1))
    if base_edge <= 1e-12:
        return {f'h{h}': 0.0 for h in horizons}
    return {f'h{h}': float(abs(adverse_selection_cost(fills, midprice_path, horizon_steps=h)) / base_edge) for h in horizons}


def aggregate_execution_quality(
    fills: Iterable[dict | NormalizedFill],
    quotes: Iterable[dict | NormalizedQuote],
    midprice_path: Iterable[dict | NormalizedMidpricePoint],
    cancel_events: Iterable[dict | NormalizedCancelEvent] | None = None,
) -> dict:
    f = normalize_fills(fills)
    q = normalize_quotes(quotes)
    mids = normalize_midprice_path(midprice_path)
    cancels = normalize_cancel_events(cancel_events or [])

    def _compute(group_fills, group_quotes, group_mids, group_cancels):
        return {
            'inventory_turnover': inventory_turnover(group_fills),
            'quote_fill_ratio': quote_fill_ratio(group_fills, group_quotes),
            'adverse_selection_cost': adverse_selection_cost(group_fills, group_mids),
            'realized_spread': realized_spread(group_fills, group_mids),
            'unrealized_spread': unrealized_spread(group_fills, group_mids),
            'microstructure_alpha_decay': microstructure_alpha_decay(group_fills, group_mids),
            'cancelled_quantity': float(sum(c.quantity for c in group_cancels)),
        }

    result = {'global': _compute(f, q, mids, cancels), 'by_exchange': {}, 'by_strategy': {}}
    exchanges = sorted({x.exchange for x in f + q + mids + cancels})
    strategies = sorted({x.strategy for x in f + q + mids + cancels})

    for exchange in exchanges:
        result['by_exchange'][exchange] = _compute(
            [x for x in f if x.exchange == exchange],
            [x for x in q if x.exchange == exchange],
            [x for x in mids if x.exchange == exchange],
            [x for x in cancels if x.exchange == exchange],
        )
    for strategy in strategies:
        result['by_strategy'][strategy] = _compute(
            [x for x in f if x.strategy == strategy],
            [x for x in q if x.strategy == strategy],
            [x for x in mids if x.strategy == strategy],
            [x for x in cancels if x.strategy == strategy],
        )
    return result


def sharpe(returns, periods: int = 365 * 24 * 12):
    r = np.asarray(returns, dtype=float)
    return float(np.sqrt(periods) * r.mean() / (r.std() + 1e-12))


def sortino(returns, periods: int = 365 * 24 * 12):
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    return float(np.sqrt(periods) * r.mean() / (downside.std() + 1e-12))


def information_ratio(returns, benchmark_returns, periods: int = 365 * 24 * 12):
    r = np.asarray(returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    n = min(r.size, b.size)
    if n == 0:
        return 0.0
    active = r[-n:] - b[-n:]
    return float(np.sqrt(periods) * active.mean() / (active.std() + 1e-12))


def cvar(returns, alpha: float = 0.95):
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    q = np.quantile(r, 1.0 - alpha)
    tail = r[r <= q]
    return float(abs(tail.mean()) if tail.size else abs(q))


def max_drawdown(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / (peak + 1e-12)
    return float(dd.min())


def calmar(returns, equity_curve, periods: int = 365 * 24 * 12):
    annual_return = float(np.mean(returns) * periods)
    mdd = abs(max_drawdown(equity_curve)) + 1e-12
    return annual_return / mdd


def ulcer_index(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(ec)
    drawdown_pct = ((ec / (peak + 1e-12)) - 1.0) * 100.0
    return float(np.sqrt(np.mean(np.square(np.minimum(drawdown_pct, 0.0)))))


def equity_curve_smoothness(equity_curve):
    ec = np.asarray(equity_curve, dtype=float)
    if ec.size < 3:
        return 0.0
    second_diff = np.diff(ec, n=2)
    return float(1.0 / (1.0 + np.mean(np.abs(second_diff))))


def return_skewness(returns):
    r = np.asarray(returns, dtype=float)
    return float(skew(r, bias=False)) if r.size > 2 else 0.0


def return_kurtosis(returns):
    r = np.asarray(returns, dtype=float)
    return float(kurtosis(r, fisher=True, bias=False)) if r.size > 3 else 0.0


def expectancy(returns):
    r = np.asarray(returns, dtype=float)
    wins = r[r > 0]
    losses = r[r < 0]
    if r.size == 0:
        return 0.0
    win_rate = wins.size / r.size
    loss_rate = losses.size / r.size
    avg_win = wins.mean() if wins.size else 0.0
    avg_loss = abs(losses.mean()) if losses.size else 0.0
    return float((win_rate * avg_win) - (loss_rate * avg_loss))


def profit_factor(returns):
    r = np.asarray(returns, dtype=float)
    gain = r[r > 0].sum()
    loss = abs(r[r < 0].sum()) + 1e-12
    return float(gain / loss)

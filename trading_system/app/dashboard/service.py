from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import Select, select

from trading_system.app.database.models import EquitySnapshot, OrderExecution, TradeSignal
from trading_system.app.database.repository import Repository


@dataclass
class DashboardState:
    latest_price: float = 0.0
    regime: str = 'UNKNOWN'
    signal: str = 'HOLD'
    binance_status: str = 'unknown'
    latency_ms: float = 0.0
    risk_active: bool = False
    active_exposure: float = 0.0
    capital_real_usdt: float = 0.0
    account_equity_usdt: float = 0.0


class DashboardService:
    def __init__(self, repository: Repository, state_provider: Callable[[], DashboardState | dict[str, Any]]) -> None:
        self.repository = repository
        self.state_provider = state_provider

    async def _fetch_all(self, stmt: Select[Any]) -> list[Any]:
        async with self.repository.session() as session:
            rows = await session.execute(stmt)
            return list(rows.scalars().all())

    async def get_recent_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self._fetch_all(select(OrderExecution).order_by(OrderExecution.ts.desc()).limit(limit))
        return [
            {
                'ts': row.ts,
                'symbol': row.symbol,
                'side': row.side,
                'qty': row.qty,
                'price': row.price,
                'status': row.status,
                'pnl': row.pnl,
            }
            for row in rows
        ]

    async def get_activity_feed(self, limit: int = 100) -> list[dict[str, Any]]:
        trades = await self.get_recent_trades(limit=limit)
        signals_rows = await self._fetch_all(select(TradeSignal).order_by(TradeSignal.ts.desc()).limit(limit))

        entries: list[dict[str, Any]] = []
        for trade in trades:
            entries.append(
                {
                    'ts': int(trade['ts']),
                    'type': 'execution',
                    'title': f"{trade['status']} {trade['side']} {trade['symbol']}",
                    'detail': f"qty={float(trade['qty']):.6f} price={float(trade['price']):.6f} pnl={float(trade['pnl']):+.6f}",
                }
            )
        for signal in signals_rows:
            entries.append(
                {
                    'ts': int(signal.ts),
                    'type': 'signal',
                    'title': f"signal {signal.signal} {signal.symbol}",
                    'detail': f"score={float(signal.score):.4f} ev={float(signal.expected_value):+.6f} reason={str(signal.reason)[:120]}",
                }
            )

        entries.sort(key=lambda item: int(item['ts']), reverse=True)
        return entries[:limit]

    async def get_equity_curve(self, limit: int = 500) -> list[dict[str, float | int]]:
        rows = await self._fetch_all(select(EquitySnapshot).order_by(EquitySnapshot.ts.asc()).limit(limit))
        return [
            {
                'timestamp': row.ts,
                'equity': row.equity,
                'drawdown': row.drawdown,
                'pnl_total': row.pnl_total,
            }
            for row in rows
        ]

    async def get_last_signal(self) -> str:
        rows = await self._fetch_all(select(TradeSignal).order_by(TradeSignal.ts.desc()).limit(1))
        return rows[0].signal if rows else 'HOLD'

    async def get_metrics(self) -> dict[str, Any]:
        trades = await self.get_recent_trades(limit=1000)
        equity_curve = await self.get_equity_curve(limit=2000)

        def _normalize_ts_ms(value: int | float | str | None) -> int:
            ts = int(float(value or 0))
            # Compatibilidad histórica: algunos productores usan segundos y otros milisegundos.
            return ts * 1000 if ts < 10_000_000_000 else ts

        total_trades = len(trades)
        wins = sum(1 for trade in trades if trade['pnl'] > 0)
        losses = sum(1 for trade in trades if trade['pnl'] < 0)
        win_rate = (wins / total_trades) if total_trades else 0.0
        pnl_total_from_trades = sum(float(trade['pnl']) for trade in trades)

        now_ms = int(time.time() * 1000)
        day_ms = 24 * 60 * 60 * 1000
        pnl_daily = sum(
            float(trade['pnl'])
            for trade in trades
            if (now_ms - _normalize_ts_ms(trade['ts'])) <= day_ms
        )

        pnl_total = float(equity_curve[-1]['pnl_total']) if equity_curve else pnl_total_from_trades

        returns = [float(trade['pnl']) for trade in trades if float(trade['pnl']) != 0.0]
        expectancy = pnl_total / total_trades if total_trades else 0.0
        if len(returns) > 1 and statistics.pstdev(returns) > 0:
            sharpe = (statistics.fmean(returns) / statistics.pstdev(returns)) * math.sqrt(len(returns))
        else:
            sharpe = 0.0

        capital = float(equity_curve[-1]['equity']) if equity_curve else 0.0
        drawdown = max((float(point['drawdown']) for point in equity_curve), default=0.0)

        return {
            'capital': capital,
            'balance_real': capital,
            'pnl_total': pnl_total,
            'pnl_daily': pnl_daily,
            'drawdown': drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'sharpe': sharpe,
            'equity_curve': equity_curve,
            'losses': losses,
        }

    async def get_dashboard_payload(self) -> dict[str, Any]:
        metrics = await self.get_metrics()
        state = self.state_provider()
        if isinstance(state, dict):
            state_data = state
        else:
            state_data = state.__dict__

        state_data['signal'] = state_data.get('signal') or await self.get_last_signal()

        capital_real_usdt = float(state_data.get('capital_real_usdt', state_data.get('balance_real', metrics.get('capital', 0.0))) or 0.0)
        account_equity_usdt = float(state_data.get('account_equity_usdt', metrics.get('capital', capital_real_usdt)) or 0.0)
        merged = {**metrics, **state_data}
        merged['capital'] = capital_real_usdt
        merged['balance_real'] = capital_real_usdt
        merged['capital_real_usdt'] = capital_real_usdt
        merged['account_equity_usdt'] = account_equity_usdt
        return merged

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

        total_trades = len(trades)
        wins = sum(1 for trade in trades if trade['pnl'] > 0)
        losses = sum(1 for trade in trades if trade['pnl'] < 0)
        win_rate = (wins / total_trades) if total_trades else 0.0
        pnl_total = sum(float(trade['pnl']) for trade in trades)

        now_ms = int(time.time() * 1000)
        day_ms = 24 * 60 * 60 * 1000
        pnl_daily = sum(float(trade['pnl']) for trade in trades if (now_ms - int(trade['ts'])) <= day_ms)

        returns = [float(trade['pnl']) for trade in trades if float(trade['pnl']) != 0.0]
        expectancy = pnl_total / total_trades if total_trades else 0.0
        if len(returns) > 1 and statistics.pstdev(returns) > 0:
            sharpe = (statistics.fmean(returns) / statistics.pstdev(returns)) * math.sqrt(len(returns))
        else:
            sharpe = 0.0

        capital = equity_curve[-1]['equity'] if equity_curve else 1.0
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
        return {**metrics, **state_data}

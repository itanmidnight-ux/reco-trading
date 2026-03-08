from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from bot.core.portfolio import Portfolio


class StateManager:
    def __init__(self, state_file: str = 'bot_state.json') -> None:
        self.path = Path(state_file)

    def save(self, portfolio: Portfolio) -> None:
        payload = asdict(portfolio)
        if payload.get('open_position') is not None:
            payload['open_position'] = asdict(portfolio.open_position)
        payload['seen_client_order_ids'] = list(portfolio.seen_client_order_ids)
        self.path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def load(self) -> Portfolio:
        if not self.path.exists():
            return Portfolio()
        payload = json.loads(self.path.read_text(encoding='utf-8'))
        portfolio = Portfolio(
            quote_balance=payload.get('quote_balance', 0.0),
            base_balance=payload.get('base_balance', 0.0),
            realized_pnl=payload.get('realized_pnl', 0.0),
            daily_trades=payload.get('daily_trades', 0),
            daily_loss=payload.get('daily_loss', 0.0),
            seen_client_order_ids=set(payload.get('seen_client_order_ids', [])),
        )
        open_position = payload.get('open_position')
        if open_position:
            from bot.core.portfolio import Position

            portfolio.open_position = Position(**open_position)
        return portfolio


__all__ = ['StateManager']

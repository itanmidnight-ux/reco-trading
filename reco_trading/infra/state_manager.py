from __future__ import annotations

import json
from pathlib import Path

import redis

from reco_trading.core.portfolio_engine import PortfolioState


class StateManager:
    def __init__(self, redis_url: str, key: str = 'reco_trading:portfolio_state', backup_file: str = 'state/portfolio_state.json') -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.key = key
        self.backup = Path(backup_file)
        self.backup.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: PortfolioState) -> None:
        payload = json.dumps(state.__dict__)
        try:
            self.client.set(self.key, payload)
        except Exception:
            pass
        self.backup.write_text(payload, encoding='utf-8')

    def load(self) -> PortfolioState:
        raw = None
        try:
            raw = self.client.get(self.key)
        except Exception:
            raw = None

        if not raw and self.backup.exists():
            raw = self.backup.read_text(encoding='utf-8')

        if not raw:
            return PortfolioState()

        return PortfolioState(**json.loads(raw))

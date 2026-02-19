from __future__ import annotations

import json
import redis

from reco_trading.core.portfolio_engine import PortfolioState


class StateManager:
    def __init__(self, redis_url: str, key: str = 'reco_trading:portfolio_state') -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.key = key

    def save(self, state: PortfolioState) -> None:
        self.client.set(self.key, json.dumps(state.__dict__))

    def load(self) -> PortfolioState:
        raw = self.client.get(self.key)
        if not raw:
            return PortfolioState()
        payload = json.loads(raw)
        return PortfolioState(**payload)

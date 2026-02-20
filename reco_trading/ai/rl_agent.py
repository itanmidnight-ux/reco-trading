from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import redis


@dataclass(slots=True)
class RLAction:
    size_multiplier: float
    threshold_shift: float
    risk_shift: float
    pause_trading: bool


class TradingRLAgent:
    """Agente RL online con Q-learning tabular discretizado."""

    ACTIONS: tuple[RLAction, ...] = (
        RLAction(size_multiplier=1.10, threshold_shift=-0.01, risk_shift=0.001, pause_trading=False),
        RLAction(size_multiplier=0.80, threshold_shift=0.01, risk_shift=-0.001, pause_trading=False),
        RLAction(size_multiplier=1.00, threshold_shift=-0.02, risk_shift=0.000, pause_trading=False),
        RLAction(size_multiplier=0.00, threshold_shift=0.00, risk_shift=-0.003, pause_trading=True),
    )

    def __init__(self, redis_url: str, redis_key: str = "reco_trading:rl_agent_state") -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.redis_key = redis_key
        self.alpha = 0.12
        self.gamma = 0.95
        self.epsilon = 0.08
        self.lambda_drawdown = 2.0
        self.q_table: dict[str, list[float]] = {}
        self._last_state_key: str | None = None
        self._last_action_idx: int | None = None

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _state_key(self, state: dict[str, float | str]) -> str:
        regime = str(state.get("regime", "unknown"))
        bins = {
            "volatility": int(np.clip(float(state.get("volatility", 0.0)) * 20, 0, 20)),
            "win_rate": int(np.clip(float(state.get("win_rate", 0.5)) * 20, 0, 20)),
            "drawdown": int(np.clip(float(state.get("drawdown", 0.0)) * 30, 0, 30)),
            "sharpe": int(np.clip((float(state.get("sharpe", 0.0)) + 2.0) * 5, 0, 20)),
            "obi": int(np.clip((float(state.get("obi", 0.0)) + 1.0) * 10, 0, 20)),
            "spread": int(np.clip(float(state.get("spread", 0.0)) * 10_000, 0, 50)),
        }
        return f"{regime}|{bins['volatility']}|{bins['win_rate']}|{bins['drawdown']}|{bins['sharpe']}|{bins['obi']}|{bins['spread']}"

    def select_action(self, state: dict[str, float | str]) -> RLAction:
        state_key = self._state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0 for _ in self.ACTIONS]

        if np.random.random() < self.epsilon:
            action_idx = int(np.random.randint(0, len(self.ACTIONS)))
        else:
            action_idx = int(np.argmax(self.q_table[state_key]))

        self._last_state_key = state_key
        self._last_action_idx = action_idx
        return self.ACTIONS[action_idx]

    def update_policy(self, next_state: dict[str, float | str], delta_equity: float, drawdown: float) -> None:
        if self._last_state_key is None or self._last_action_idx is None:
            return

        reward = float(delta_equity - self.lambda_drawdown * max(drawdown, 0.0))
        next_key = self._state_key(next_state)
        self.q_table.setdefault(next_key, [0.0 for _ in self.ACTIONS])

        old_q = self.q_table[self._last_state_key][self._last_action_idx]
        td_target = reward + self.gamma * max(self.q_table[next_key])
        self.q_table[self._last_state_key][self._last_action_idx] = float(old_q + self.alpha * (td_target - old_q))
        self.epsilon = float(np.clip(self.epsilon * 0.9995, 0.01, 0.25))

    def save_state(self) -> None:
        payload = json.dumps({"q_table": self.q_table, "epsilon": self.epsilon})
        try:
            self.client.set(self.redis_key, payload)
        except Exception:
            return

    def load_state(self) -> None:
        try:
            raw = self.client.get(self.redis_key)
        except Exception:
            raw = None
        if not raw:
            return
        decoded = json.loads(raw)
        self.q_table = {k: [float(vv) for vv in v] for k, v in decoded.get("q_table", {}).items()}
        self.epsilon = self._clip01(float(decoded.get("epsilon", self.epsilon)))

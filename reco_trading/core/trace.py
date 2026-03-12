from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DecisionTrace:
    """Lightweight observability payload for one signal-to-decision cycle."""

    signal_side: str = "HOLD"
    confidence: float = 0.0

    risk_validation: str = "UNKNOWN"

    volatility_filter: dict[str, Any] = field(default_factory=dict)
    liquidity_filter: dict[str, Any] = field(default_factory=dict)
    regime_filter: dict[str, Any] = field(default_factory=dict)
    range_filter: dict[str, Any] = field(default_factory=dict)

    final_decision: str = "HOLD"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

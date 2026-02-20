from __future__ import annotations

import asyncio

from reco_trading.core.capital_governor import CapitalGovernor
from reco_trading.os.quant_kernel import QuantKernel
from reco_trading.validation.live_validator import LiveValidationInput, LiveValidator


class _DummyDb:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def persist_validation_event(self, payload: dict) -> None:
        self.events.append(payload)


class _DummyRedis:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, str]]] = []

    async def xadd(self, stream: str, fields: dict[str, str], maxlen: int | None = None, approximate: bool = True):
        self.events.append((stream, fields))
        return '1-0'


def _make_input(i: int, *, weak: bool = False) -> LiveValidationInput:
    signal = 0.15 if weak else 0.6
    realized = -0.03 if weak else 0.02
    expected = 0.05
    confidence = 0.35 if weak else 0.85
    return LiveValidationInput(
        ts=float(i),
        features={'f1': 2.5 if weak else 0.5, 'f2': 3.0 if weak else 0.4},
        signal=signal,
        confidence=confidence,
        realized_return=realized,
        expected_return=expected,
        shadow_return=realized,
    )


def test_live_validator_escalates_and_notifies_kernel() -> None:
    governor = CapitalGovernor(base_capital=1000)
    kernel = QuantKernel(governor, strategy_limit=8)
    db = _DummyDb()
    redis = _DummyRedis()
    validator = LiveValidator(quant_kernel=kernel, db=db, redis_client=redis, rolling_window=30)

    async def _run() -> None:
        for i in range(15):
            await validator.validate_tick(_make_input(i, weak=False))
        snapshot = None
        for i in range(15, 36):
            snapshot = await validator.validate_tick(_make_input(i, weak=True))
        assert snapshot is not None
        assert snapshot.contract.severity in {'degradado', 'critico'}

    asyncio.run(_run())

    assert kernel.notifications
    assert governor.deployable_capital < 1000
    assert kernel.mode.active
    assert db.events
    assert redis.events

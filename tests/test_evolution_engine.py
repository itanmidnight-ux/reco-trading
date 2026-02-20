from __future__ import annotations

import asyncio

from reco_trading.core.evolution_engine import EvolutionEngine


class FakeDatabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def create_config_version(self, version, configuration, signature, reason=None, metadata_payload=None):
        self.calls.append(('create_config_version', version))
        return 101

    async def register_deployment(self, version_id, status='pending', reason=None):
        self.calls.append(('register_deployment', status))
        return 202

    async def activate_config_version(self, version, reason=None):
        self.calls.append(('activate_config_version', version))

    async def register_config_failure(self, version, reason):
        self.calls.append(('register_config_failure', version))

    async def execute_rollback(self, from_version, reason, deployment_id=None):
        self.calls.append(('execute_rollback', from_version))
        return 'v-previous'

    async def complete_deployment(self, deployment_id, status, reason=None):
        self.calls.append(('complete_deployment', status))


def test_deploy_new_configuration_active_when_metrics_are_healthy() -> None:
    db = FakeDatabase()
    engine = EvolutionEngine(db)

    result = asyncio.run(
        engine.deploy_new_configuration(
            version='v-2',
            configuration={'risk_per_trade': 0.01},
            signature='any',
            validation_metrics={'sharpe': 1.4, 'drawdown': 0.06, 'execution_error_rate': 0.01},
        )
    )

    assert result['status'] == 'active'
    assert ('complete_deployment', 'active') in db.calls
    assert ('execute_rollback', 'v-2') not in db.calls


def test_deploy_new_configuration_rolls_back_when_metrics_degrade() -> None:
    db = FakeDatabase()
    engine = EvolutionEngine(db)

    result = asyncio.run(
        engine.deploy_new_configuration(
            version='v-3',
            configuration={'risk_per_trade': 0.02},
            signature='any',
            validation_metrics={'sharpe': 0.4, 'drawdown': 0.2, 'execution_error_rate': 0.05},
        )
    )

    assert result['status'] == 'rolled_back'
    assert result['rolled_back_to'] == 'v-previous'
    assert ('register_config_failure', 'v-3') in db.calls
    assert ('execute_rollback', 'v-3') in db.calls

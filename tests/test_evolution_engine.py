from __future__ import annotations

import asyncio

import pytest

from reco_trading.core.evolution_engine import EvolutionEngine
from reco_trading.security.signing import ConfigurationSigner


class FakeDatabase:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def create_config_version(self, version, configuration, signature, reason=None, metadata_payload=None, actor='system'):
        self.calls.append(('create_config_version', version))
        return 101

    async def register_deployment(self, version_id, status='pending', reason=None, actor='system', signature=None, deployment_hash=None):
        self.calls.append(('register_deployment', status))
        return 202

    async def activate_config_version(self, version, reason=None, actor='system'):
        self.calls.append(('activate_config_version', version))

    async def register_config_failure(self, version, reason, actor='system'):
        self.calls.append(('register_config_failure', version))

    async def execute_rollback(self, from_version, reason, deployment_id=None, actor='system'):
        self.calls.append(('execute_rollback', from_version))
        return 'v-previous'

    async def complete_deployment(self, deployment_id, status, reason=None, actor='system'):
        self.calls.append(('complete_deployment', status))

    async def append_audit_event(self, **kwargs):
        self.calls.append(('append_audit_event', kwargs.get('event_type')))
        return 1


def test_deploy_new_configuration_active_when_metrics_are_healthy() -> None:
    db = FakeDatabase()
    signer = ConfigurationSigner('reco_trading')
    engine = EvolutionEngine(db, signer=signer)
    configuration = {'risk_per_trade': 0.01}
    signature = signer.sign(configuration).signature

    result = asyncio.run(
        engine.deploy_new_configuration(
            version='v-2',
            configuration=configuration,
            signature=signature,
            validation_metrics={'sharpe': 1.4, 'drawdown': 0.06, 'execution_error_rate': 0.01},
        )
    )

    assert result['status'] == 'active'
    assert ('complete_deployment', 'active') in db.calls
    assert ('execute_rollback', 'v-2') not in db.calls


def test_deploy_new_configuration_rolls_back_when_metrics_degrade() -> None:
    db = FakeDatabase()
    signer = ConfigurationSigner('reco_trading')
    engine = EvolutionEngine(db, signer=signer)
    configuration = {'risk_per_trade': 0.02}
    signature = signer.sign(configuration).signature

    result = asyncio.run(
        engine.deploy_new_configuration(
            version='v-3',
            configuration=configuration,
            signature=signature,
            validation_metrics={'sharpe': 0.4, 'drawdown': 0.2, 'execution_error_rate': 0.05},
            actor_role='platform_admin',
        )
    )

    assert result['status'] == 'rolled_back'
    assert result['rolled_back_to'] == 'v-previous'
    assert ('register_config_failure', 'v-3') in db.calls
    assert ('execute_rollback', 'v-3') in db.calls


def test_deploy_requires_valid_signature() -> None:
    db = FakeDatabase()
    engine = EvolutionEngine(db)

    with pytest.raises(Exception):
        asyncio.run(
            engine.deploy_new_configuration(
                version='v-4',
                configuration={'risk_per_trade': 0.03},
                signature='invalid-signature',
                validation_metrics={'sharpe': 1.4, 'drawdown': 0.02, 'execution_error_rate': 0.005},
            )
        )

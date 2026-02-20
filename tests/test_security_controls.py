import asyncio

import pytest

from reco_trading.core.security import (
    APIKeyVault,
    AuthenticatedEncryption,
    CircuitBreaker,
    CircuitState,
    ClusterOperation,
    EnvironmentVault,
    KeyRotationManager,
    NodeRateLimiter,
    RBACAuthorizer,
    SecurityError,
)
from reco_trading.distributed.coordinator import ClusterCoordinator
from reco_trading.distributed.models import TaskEnvelope, WorkerRegistration
from reco_trading.security.rbac import CriticalOperation, CriticalRBAC
from reco_trading.security.secrets_vault import InMemorySecretStore, SecretsVault
from reco_trading.security.signing import ConfigurationSigner
from tests.test_distributed_cluster import FakeRedis


@pytest.mark.parametrize('plaintext', ['super-secret', 'clave-ñ'])
def test_authenticated_encryption_and_rotation(plaintext):
    manager = KeyRotationManager(passphrase='test-pass', active_key_id='v1')
    crypto = AuthenticatedEncryption(manager)
    token_v1 = crypto.encrypt(plaintext)
    assert crypto.decrypt(token_v1) == plaintext

    manager.rotate('v2')
    token_v2 = crypto.encrypt(plaintext)
    assert crypto.decrypt(token_v2) == plaintext
    assert crypto.decrypt(token_v1) == plaintext


def test_rbac_and_rate_limiter():
    authz = RBACAuthorizer()
    assert authz.authorize('admin', ClusterOperation.DRAIN)
    assert not authz.authorize('viewer', ClusterOperation.DISPATCH)

    limiter = NodeRateLimiter(max_requests=2, window_seconds=10)
    assert limiter.allow('node-a', now=1)
    assert limiter.allow('node-a', now=2)
    assert not limiter.allow('node-a', now=3)


def test_circuit_breaker_open_and_recovery():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=5, half_open_success_threshold=1)
    assert breaker.allow_request(now=0)
    breaker.record_failure(now=1)
    breaker.record_failure(now=2)
    assert breaker.state == CircuitState.OPEN
    assert not breaker.allow_request(now=3)
    assert breaker.allow_request(now=8)
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_api_key_vault_with_env_backend(monkeypatch):
    manager = KeyRotationManager(passphrase='vault-pass')
    crypto = AuthenticatedEncryption(manager)
    encrypted = crypto.encrypt('api-123')
    monkeypatch.setenv('RECO_SECRET_BINANCE_MOMENTUM_API_KEY', encrypted)

    vault = APIKeyVault(EnvironmentVault(), encryption=crypto)
    out = asyncio.run(vault.get_api_key('binance', 'momentum'))
    assert out == 'api-123'


def test_cluster_coordinator_security_ops():
    async def _run():
        redis = FakeRedis()
        coordinator = ClusterCoordinator(redis, heartbeat_ttl_s=5, monitor_interval_s=0.05)

        await coordinator.register_worker(WorkerRegistration(worker_id='w1', task_types={'features'}), role='admin')

        with pytest.raises(SecurityError):
            await coordinator.register_worker(
                WorkerRegistration(worker_id='w2', task_types={'features'}),
                role='viewer',
            )

        coordinator.circuit_breakers['binance'] = CircuitBreaker(recovery_timeout_s=10**12)
        t1 = TaskEnvelope(task_type='features', payload={'exchange': 'binance'})
        await coordinator.dispatch_task(t1, role='operator', node_id='dispatcher-a')
        await coordinator.cancel_task(t1.task_id, role='admin')
        assert (await redis.hgetall(coordinator._task_key(t1.task_id))).get('status') == 'cancelled'

        breaker = coordinator.circuit_breakers['binance']
        for _ in range(breaker.failure_threshold):
            breaker.record_failure(now=1)

        with pytest.raises(SecurityError):
            await coordinator.dispatch_task(TaskEnvelope(task_type='features', payload={'exchange': 'binance'}))

    asyncio.run(_run())


def test_secrets_vault_persists_key_version_and_decrypts():
    async def _run():
        manager = KeyRotationManager(passphrase='vault-pass', active_key_id='v1')
        crypto = AuthenticatedEncryption(manager)
        store = InMemorySecretStore()
        vault = SecretsVault(store, crypto)

        record_v1 = await vault.put_secret('binance_api_key', 'k-123')
        assert record_v1.key_version == 'v1'
        assert await vault.get_secret('binance_api_key') == 'k-123'

        manager.rotate('v2')
        record_v2 = await vault.put_secret('binance_api_key', 'k-456')
        assert record_v2.key_version == 'v2'
        assert await vault.get_secret('binance_api_key') == 'k-456'

    asyncio.run(_run())


def test_critical_rbac_and_signing():
    rbac = CriticalRBAC()
    assert rbac.authorize('platform_admin', CriticalOperation.KILL_SWITCH)
    assert not rbac.authorize('operator', CriticalOperation.ROLLBACK)

    signer = ConfigurationSigner('reco_trading')
    payload = {'mode': 'safe', 'max_risk': 0.01}
    envelope = signer.sign(payload)
    assert signer.verify(payload, envelope.signature)

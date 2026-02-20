from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from reco_trading.security.rbac import CriticalOperation, CriticalRBAC
from reco_trading.security.secrets_vault import AuthenticatedEncryption, KeyRotationManager, SecurityError


class VaultBackend(Protocol):
    async def get_secret(self, secret_name: str) -> str | None: ...


class EnvironmentVault:
    def __init__(self, *, prefix: str = 'RECO_SECRET_') -> None:
        self.prefix = prefix

    async def get_secret(self, secret_name: str) -> str | None:
        return os.getenv(f'{self.prefix}{secret_name.upper()}')


class RedisVault:
    def __init__(self, redis_client: Any, *, key_prefix: str = 'reco:vault') -> None:
        self.redis = redis_client
        self.key_prefix = key_prefix

    async def get_secret(self, secret_name: str) -> str | None:
        value = await self.redis.get(f'{self.key_prefix}:{secret_name}')
        if isinstance(value, bytes):
            return value.decode('utf-8')
        return value


class InMemorySecretManagerVault:
    def __init__(self, secrets_store: dict[str, str] | None = None) -> None:
        self.secrets_store = secrets_store or {}

    async def get_secret(self, secret_name: str) -> str | None:
        return self.secrets_store.get(secret_name)


class APIKeyVault:
    def __init__(self, backend: VaultBackend, encryption: AuthenticatedEncryption | None = None) -> None:
        self.backend = backend
        self.encryption = encryption

    async def get_api_key(self, exchange: str, strategy: str) -> str:
        secret_name = f'{exchange}_{strategy}_api_key'.lower()
        raw = await self.backend.get_secret(secret_name)
        if not raw:
            raise SecurityError(f'Missing API key for {exchange}/{strategy}')
        if self.encryption is None:
            return raw
        return self.encryption.decrypt(raw)


class ClusterOperation(str, Enum):
    REGISTER = 'register'
    DISPATCH = 'dispatch'
    CANCEL = 'cancel'
    DRAIN = 'drain'


class RBACAuthorizer:
    def __init__(self) -> None:
        self._legacy = {
            'viewer': set(),
            'operator': {ClusterOperation.REGISTER, ClusterOperation.DISPATCH},
            'admin': set(ClusterOperation),
        }

    def authorize(self, role: str, operation: ClusterOperation) -> bool:
        return operation in self._legacy.get(role, set())

    def require(self, role: str, operation: ClusterOperation) -> None:
        if not self.authorize(role, operation):
            raise SecurityError(f'RBAC denied: role={role} operation={operation.value}')


@dataclass(slots=True)
class ClusterChannelPolicy:
    tls_enabled: bool = True
    mtls_required: bool = True
    min_tls_version: str = 'TLSv1.2'

    def validate(self, *, tls_active: bool, client_cert_present: bool) -> None:
        if self.tls_enabled and not tls_active:
            raise SecurityError('TLS is required for intra-cluster traffic')
        if self.mtls_required and not client_cert_present:
            raise SecurityError('mTLS client certificate required')


class NodeRateLimiter:
    def __init__(self, *, max_requests: int = 120, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, node_id: str, now: float | None = None) -> bool:
        timestamp = now or time.time()
        bucket = self._requests[node_id]
        while bucket and timestamp - bucket[0] > self.window_seconds:
            bucket.popleft()
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(timestamp)
        return True


class CircuitState(str, Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'


@dataclass(slots=True)
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    half_open_success_threshold: int = 2
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _half_open_successes: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow_request(self, now: float | None = None) -> bool:
        current = now or time.time()
        if self._state == CircuitState.OPEN:
            if self._opened_at is not None and current - self._opened_at >= self.recovery_timeout_s:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_success_threshold:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._half_open_successes = 0
                self._opened_at = None
            return
        self._failures = 0

    def record_failure(self, now: float | None = None) -> None:
        self._failures += 1
        self._half_open_successes = 0
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now or time.time()


def encrypt_secret(plaintext: str, passphrase: str) -> str:
    return AuthenticatedEncryption(KeyRotationManager(passphrase=passphrase)).encrypt(plaintext)


def decrypt_secret(ciphertext: str, passphrase: str) -> str:
    try:
        return AuthenticatedEncryption(KeyRotationManager(passphrase=passphrase)).decrypt(ciphertext)
    except SecurityError as exc:
        raise SecurityError('Unable to decrypt secret. Ensure passphrase and key rotation state match.') from exc


__all__ = [
    'APIKeyVault',
    'AuthenticatedEncryption',
    'CircuitBreaker',
    'CircuitState',
    'ClusterChannelPolicy',
    'ClusterOperation',
    'CriticalOperation',
    'CriticalRBAC',
    'EnvironmentVault',
    'InMemorySecretManagerVault',
    'KeyRotationManager',
    'NodeRateLimiter',
    'RBACAuthorizer',
    'RedisVault',
    'SecurityError',
    'decrypt_secret',
    'encrypt_secret',
]

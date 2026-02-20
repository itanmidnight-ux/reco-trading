from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class SecurityError(RuntimeError):
    pass


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(passphrase.encode('utf-8'))


@dataclass(slots=True)
class KeyMaterial:
    key_id: str
    key: bytes
    algorithm: str = 'aesgcm'
    created_at: float = field(default_factory=time.time)


class KeyRotationManager:
    def __init__(self, *, passphrase: str, active_key_id: str = 'v1') -> None:
        self._passphrase = passphrase
        self._keys: dict[str, KeyMaterial] = {}
        self._active_key_id = active_key_id
        self.rotate(active_key_id)

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    def rotate(self, new_key_id: str | None = None) -> str:
        key_id = new_key_id or f'v{len(self._keys) + 1}'
        salt = hashlib.sha256(f'{self._passphrase}:{key_id}'.encode('utf-8')).digest()[:16]
        key = _derive_key(self._passphrase, salt)
        self._keys[key_id] = KeyMaterial(key_id=key_id, key=key)
        self._active_key_id = key_id
        return key_id

    def get(self, key_id: str) -> KeyMaterial:
        if key_id not in self._keys:
            raise SecurityError(f'Unknown key_id={key_id}')
        return self._keys[key_id]

    def active(self) -> KeyMaterial:
        return self.get(self._active_key_id)


class AuthenticatedEncryption:
    """AES-GCM authenticated encryption with key rotation support."""

    def __init__(self, key_manager: KeyRotationManager) -> None:
        self._key_manager = key_manager

    def encrypt(self, plaintext: str, *, associated_data: bytes | None = None) -> str:
        material = self._key_manager.active()
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(material.key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode('utf-8'), associated_data)
        envelope = {
            'alg': material.algorithm,
            'key_id': material.key_id,
            'nonce': base64.urlsafe_b64encode(nonce).decode('utf-8'),
            'ct': base64.urlsafe_b64encode(ciphertext).decode('utf-8'),
        }
        return base64.urlsafe_b64encode(json.dumps(envelope).encode('utf-8')).decode('utf-8')

    def decrypt(self, token: str, *, associated_data: bytes | None = None) -> str:
        try:
            raw = base64.urlsafe_b64decode(token.encode('utf-8'))
            envelope = json.loads(raw.decode('utf-8'))
            key_id = envelope['key_id']
            nonce = base64.urlsafe_b64decode(envelope['nonce'].encode('utf-8'))
            ciphertext = base64.urlsafe_b64decode(envelope['ct'].encode('utf-8'))
            material = self._key_manager.get(key_id)
            aesgcm = AESGCM(material.key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data)
            return plaintext.decode('utf-8')
        except (KeyError, ValueError, InvalidTag, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecurityError('Invalid encrypted payload') from exc


class FernetEncryption:
    """Alternative authenticated encryption backend based on Fernet."""

    def __init__(self, passphrase: str) -> None:
        salt = b'reco_trading_static_salt'  # constant to recover previous encrypted values per env secret.
        key = base64.urlsafe_b64encode(_derive_key(passphrase, salt))
        self._fernet = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode('utf-8')).decode('utf-8')

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode('utf-8')).decode('utf-8')
        except (InvalidToken, UnicodeDecodeError) as exc:
            raise SecurityError('Invalid fernet payload') from exc


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


@dataclass(slots=True)
class RoleBinding:
    role: str
    allowed_operations: set[ClusterOperation]


class RBACAuthorizer:
    def __init__(self) -> None:
        self._bindings: dict[str, RoleBinding] = {
            'viewer': RoleBinding(role='viewer', allowed_operations=set()),
            'operator': RoleBinding(role='operator', allowed_operations={ClusterOperation.REGISTER, ClusterOperation.DISPATCH}),
            'admin': RoleBinding(role='admin', allowed_operations=set(ClusterOperation)),
        }

    def authorize(self, role: str, operation: ClusterOperation) -> bool:
        binding = self._bindings.get(role)
        return bool(binding and operation in binding.allowed_operations)

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
    manager = KeyRotationManager(passphrase=passphrase)
    return AuthenticatedEncryption(manager).encrypt(plaintext)


def decrypt_secret(ciphertext: str, passphrase: str) -> str:
    try:
        # current format with embedded key_id/envelope
        manager = KeyRotationManager(passphrase=passphrase)
        return AuthenticatedEncryption(manager).decrypt(ciphertext)
    except SecurityError as exc:
        raise SecurityError('Unable to decrypt secret. Ensure passphrase and key rotation state match.') from exc

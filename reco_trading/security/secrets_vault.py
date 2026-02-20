from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class SecurityError(RuntimeError):
    """Security-related operation failed."""


@dataclass(slots=True)
class KeyMaterial:
    key_id: str
    key: bytes
    algorithm: str = 'AES256-GCM'
    created_at: float = field(default_factory=time.time)


class KeyRotationManager:
    """Derives and rotates key material deterministically from a passphrase."""

    def __init__(self, *, passphrase: str, active_key_id: str = 'v1') -> None:
        self._passphrase = passphrase
        self._keys: dict[str, KeyMaterial] = {}
        self._active_key_id = active_key_id
        self.rotate(active_key_id)

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    @staticmethod
    def _derive_key(passphrase: str, key_id: str) -> bytes:
        salt = hashlib.sha256(f'{passphrase}:{key_id}'.encode('utf-8')).digest()[:16]
        kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
        return kdf.derive(passphrase.encode('utf-8'))

    def rotate(self, new_key_id: str | None = None) -> str:
        key_id = new_key_id or f'v{len(self._keys) + 1}'
        self._keys[key_id] = KeyMaterial(key_id=key_id, key=self._derive_key(self._passphrase, key_id))
        self._active_key_id = key_id
        return key_id

    def get(self, key_id: str) -> KeyMaterial:
        material = self._keys.get(key_id)
        if material is None:
            raise SecurityError(f'Unknown key_id={key_id}')
        return material

    def active(self) -> KeyMaterial:
        return self.get(self._active_key_id)


class AuthenticatedEncryption:
    def __init__(self, key_manager: KeyRotationManager) -> None:
        self._key_manager = key_manager

    @property
    def active_key_id(self) -> str:
        return self._key_manager.active_key_id

    def encrypt(self, plaintext: str, *, associated_data: bytes | None = None) -> str:
        material = self._key_manager.active()
        nonce = secrets.token_bytes(12)
        payload = AESGCM(material.key).encrypt(nonce, plaintext.encode('utf-8'), associated_data)
        envelope = {
            'alg': material.algorithm,
            'key_id': material.key_id,
            'nonce': base64.urlsafe_b64encode(nonce).decode('utf-8'),
            'ct': base64.urlsafe_b64encode(payload).decode('utf-8'),
        }
        return base64.urlsafe_b64encode(json.dumps(envelope, separators=(',', ':')).encode('utf-8')).decode('utf-8')

    def decrypt(self, token: str, *, associated_data: bytes | None = None) -> str:
        try:
            envelope = json.loads(base64.urlsafe_b64decode(token.encode('utf-8')).decode('utf-8'))
            key_id = envelope['key_id']
            nonce = base64.urlsafe_b64decode(envelope['nonce'].encode('utf-8'))
            ciphertext = base64.urlsafe_b64decode(envelope['ct'].encode('utf-8'))
            material = self._key_manager.get(key_id)
            plaintext = AESGCM(material.key).decrypt(nonce, ciphertext, associated_data)
            return plaintext.decode('utf-8')
        except (KeyError, ValueError, InvalidTag, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecurityError('Invalid encrypted payload') from exc


@dataclass(slots=True)
class SecretRecord:
    name: str
    ciphertext: str
    key_version: str
    created_at: float = field(default_factory=time.time)


class SecretStoreBackend(Protocol):
    async def set(self, secret_name: str, record: SecretRecord) -> None: ...

    async def get(self, secret_name: str) -> SecretRecord | None: ...


class InMemorySecretStore:
    def __init__(self) -> None:
        self._store: dict[str, SecretRecord] = {}

    async def set(self, secret_name: str, record: SecretRecord) -> None:
        self._store[secret_name] = record

    async def get(self, secret_name: str) -> SecretRecord | None:
        return self._store.get(secret_name)


class SecretsVault:
    """Store/retrieve encrypted secrets with explicit key version metadata."""

    def __init__(self, backend: SecretStoreBackend, encryption: AuthenticatedEncryption) -> None:
        self.backend = backend
        self.encryption = encryption

    async def put_secret(self, name: str, plaintext: str) -> SecretRecord:
        key_version = self.encryption.active_key_id
        ciphertext = self.encryption.encrypt(plaintext, associated_data=name.encode('utf-8'))
        record = SecretRecord(name=name, ciphertext=ciphertext, key_version=key_version)
        await self.backend.set(name, record)
        return record

    async def get_secret(self, name: str) -> str:
        record = await self.backend.get(name)
        if record is None:
            raise SecurityError(f'Secret not found: {name}')
        return self.encryption.decrypt(record.ciphertext, associated_data=name.encode('utf-8'))

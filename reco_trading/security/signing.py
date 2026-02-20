from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SignedEnvelope:
    payload_hash: str
    signature: str
    key_id: str
    signed_at: int


class ConfigurationSigner:
    """Deterministic hash + HMAC signing for configs/deployments."""

    def __init__(self, secret: str, *, key_id: str = 'sig-v1') -> None:
        self._secret = secret.encode('utf-8')
        self.key_id = key_id

    @staticmethod
    def payload_hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(canonical).hexdigest()

    def sign(self, payload: dict[str, Any]) -> SignedEnvelope:
        digest = self.payload_hash(payload)
        signature = hmac.new(self._secret, digest.encode('utf-8'), hashlib.sha256).hexdigest()
        return SignedEnvelope(payload_hash=digest, signature=signature, key_id=self.key_id, signed_at=int(time.time()))

    def verify(self, payload: dict[str, Any], signature: str) -> bool:
        digest = self.payload_hash(payload)
        expected = hmac.new(self._secret, digest.encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

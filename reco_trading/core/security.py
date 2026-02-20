from __future__ import annotations

import base64
import hashlib


def _derive_key(secret: str) -> bytes:
    return hashlib.sha256(secret.encode('utf-8')).digest()


def encrypt_secret(plaintext: str, passphrase: str) -> str:
    key = _derive_key(passphrase)
    data = plaintext.encode('utf-8')
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    return base64.urlsafe_b64encode(encrypted).decode('utf-8')


def decrypt_secret(ciphertext: str, passphrase: str) -> str:
    key = _derive_key(passphrase)
    data = base64.urlsafe_b64decode(ciphertext.encode('utf-8'))
    decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    return decrypted.decode('utf-8')

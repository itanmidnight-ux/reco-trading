from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlencode


def sign_params(secret: str, params: dict) -> str:
    query = urlencode(params)
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def mask_secret(value: str) -> str:
    if len(value) <= 8:
        return '***'
    return f"{value[:4]}***{value[-4:]}"

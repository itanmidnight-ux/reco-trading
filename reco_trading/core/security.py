"""
Security module for Reco-Trading.
Provides encryption, sanitization, and secure handling of sensitive data.
Based on FreqTrade's config_secrets.py
"""

import base64
import hashlib
import os
import secrets
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any

from cryptography.fernet import Fernet


_SENSITIVE_KEYS = [
    "exchange.key",
    "exchange.api_key",
    "exchange.apiKey",
    "exchange.secret",
    "exchange.password",
    "exchange.uid",
    "exchange.account_id",
    "exchange.accountId",
    "exchange.wallet_address",
    "exchange.walletAddress",
    "exchange.private_key",
    "exchange.privateKey",
    "telegram.token",
    "telegram.chat_id",
    "discord.webhook_url",
    "api_server.password",
    "api_server.jwt_secret_key",
    "api_server.ws_token",
    "webhook.url",
]


class SecureConfig:
    """Handles secure configuration management."""

    def __init__(self, encryption_key: bytes | None = None):
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            key = os.environ.get("RECO_TRADING_CONFIG_KEY")
            if key:
                self.fernet = Fernet(key)
            else:
                self.fernet = None

    @staticmethod
    def generate_encryption_key() -> str:
        """Generate a new Fernet encryption key."""
        return Fernet.generate_key().decode()

    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        if not self.fernet:
            return value
        return self.fernet.encrypt(value.encode()).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value."""
        if not self.fernet:
            return encrypted_value
        return self.fernet.decrypt(encrypted_value.encode()).decode()

    def sanitize_config(self, config: dict[str, Any], *, show_sensitive: bool = False) -> dict[str, Any]:
        """
        Remove sensitive information from the config.
        
        Args:
            config: Configuration dictionary
            show_sensitive: Show sensitive information (for debugging)
            
        Returns:
            Sanitized configuration
        """
        if show_sensitive:
            return config
        
        config = deepcopy(config)
        for key in _SENSITIVE_KEYS:
            if "." in key:
                nested_keys = key.split(".")
                nested_config = config
                for nested_key in nested_keys[:-1]:
                    if isinstance(nested_config, dict):
                        nested_config = nested_config.get(nested_key, {})
                    else:
                        break
                if isinstance(nested_config, dict) and nested_keys[-1] in nested_config:
                    nested_config[nested_keys[-1]] = "REDACTED"
            else:
                if key in config:
                    config[key] = "REDACTED"
        
        return config

    def remove_exchange_credentials(
        self, exchange_config: dict[str, Any], dry_run: bool
    ) -> dict[str, Any]:
        """
        Removes exchange keys from the configuration for dry-run mode.
        
        Args:
            exchange_config: Exchange configuration
            dry_run: If True, remove sensitive keys
            
        Returns:
            Modified exchange config
        """
        if not dry_run:
            return exchange_config
        
        exchange_config = deepcopy(exchange_config)
        for key in _SENSITIVE_KEYS:
            if key.startswith("exchange."):
                key1 = key.removeprefix("exchange.")
                if key1 in exchange_config:
                    exchange_config[key1] = ""
        
        return exchange_config


class JWTManager:
    """JWT token management for API authentication."""

    def __init__(self, secret_key: str | None = None):
        self.secret_key = secret_key or os.environ.get("RECO_TRADING_JWT_SECRET", secrets.token_hex(32))
        self.algorithm = "HS256"

    def create_token(self, data: dict[str, Any], expires_delta_hours: int = 24) -> str:
        """Create a JWT token."""
        try:
            import jwt
            from datetime import datetime, timedelta, timezone
            
            expire = datetime.now(timezone.utc) + timedelta(hours=expires_delta_hours)
            to_encode = data.copy()
            to_encode.update({"exp": expire})
            return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        except ImportError:
            return ""

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify and decode a JWT token."""
        try:
            import jwt
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except ImportError:
            return None
        except Exception:
            return None


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, bytes]:
    """
    Hash a password using PBKDF2.
    
    Returns:
        Tuple of (hash_hex, salt)
    """
    if salt is None:
        salt = os.urandom(32)
    
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100000,
    )
    return key.hex(), salt


def verify_password(password: str, hashed: str, salt: bytes) -> bool:
    """Verify a password against its hash."""
    new_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(new_hash, hashed)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def check_weak_secrets(config: dict[str, Any]) -> list[str]:
    """
    Check for weak or default secrets in configuration.
    
    Returns:
        List of warnings
    """
    warnings = []
    
    weak_secrets = {
        "api_server": {
            "jwt_secret_key": ["super-secret", "changeme", "password", "secret"],
            "ws_token": ["super-secret", "changeme", "password", "secret"],
        },
        "telegram": {
            "token": ["changeme", "password", "secret"],
        },
    }
    
    for section, keys in weak_secrets.items():
        if section not in config:
            continue
        section_config = config[section]
        if not isinstance(section_config, dict):
            continue
            
        for key, weak_values in keys.items():
            if key in section_config:
                value = str(section_config[key]).lower()
                if any(w in value for w in weak_values):
                    warnings.append(f"SECURITY WARNING - `{section}.{key}` seems to be default or weak")
    
    return warnings


class RateLimiter:
    """Rate limiter for API endpoints."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = time.time()
        minute_ago = now - 60
        
        client_requests = self.requests[client_id]
        client_requests[:] = [t for t in client_requests if t > minute_ago]
        
        if len(client_requests) >= self.requests_per_minute:
            return False
        
        client_requests.append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        minute_ago = now - 60
        
        client_requests = self.requests[client_id]
        client_requests[:] = [t for t in client_requests if t > minute_ago]
        
        return max(0, self.requests_per_minute - len(client_requests))

    def reset(self, client_id: str) -> None:
        """Reset rate limit for client."""
        if client_id in self.requests:
            del self.requests[client_id]

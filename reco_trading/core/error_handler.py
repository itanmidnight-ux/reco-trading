from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from typing import Any, TypeVar

import ccxt

T = TypeVar("T")


@dataclass(slots=True)
class RetryPolicy:
    retries: int = 3
    backoff_seconds: float = 0.5


class ErrorHandler:
    """Centralized async-safe error handling and retry utility."""

    RETRYABLE = (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable)

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    async def guard(self, operation: Callable[[], Awaitable[T]], fallback: T, context: str = "") -> T:
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("guarded_operation_failed context=%s error=%s", context, exc)
            return fallback

    async def with_retry(self, operation: Callable[[], Awaitable[T]], policy: RetryPolicy | None = None, context: str = "") -> T:
        policy = policy or RetryPolicy()
        delay = policy.backoff_seconds
        last_exc: Exception | None = None

        for attempt in range(policy.retries):
            try:
                return await operation()
            except self.RETRYABLE as exc:
                last_exc = exc
                self.logger.warning("retryable_error context=%s attempt=%s error=%s", context, attempt + 1, exc)
                if attempt == policy.retries - 1:
                    break
                await asyncio.sleep(delay)
                delay *= 2

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("retry_exhausted_without_exception")

    async def log_to_repository(self, repository: Any, state: str, phase: str, error: Exception) -> None:
        try:
            await repository.record_error(state, phase, str(error))
        except Exception as repo_exc:  # noqa: BLE001
            self.logger.error("error_logging_failed phase=%s error=%s", phase, repo_exc)

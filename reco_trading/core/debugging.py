from __future__ import annotations

import asyncio
import functools
import logging
import traceback
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar

import ccxt

from reco_trading.core.error_handler import ErrorHandler, RetryPolicy

T = TypeVar("T")


class DebugLevel(Enum):
    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"


@dataclass
class DebugContext:
    operation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = False
    error: str | None = None
    traceback: str | None = None
    retry_count: int = 0
    recovery_attempted: bool = False
    recovery_success: bool = False


class AutoDebugger:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: DebugLevel = DebugLevel.NORMAL,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.level = level
        self.error_handler = ErrorHandler(self.logger)
        self._debug_history: list[DebugContext] = []
        self._error_count: dict[str, int] = {}
        self._max_history = 1000

    def track_error(self, operation: str) -> None:
        self._error_count[operation] = self._error_count.get(operation, 0) + 1

    def get_error_count(self, operation: str) -> int:
        return self._error_count.get(operation, 0)

    def get_recent_errors(self, limit: int = 10) -> list[DebugContext]:
        return self._debug_history[-limit:]

    def clear_history(self) -> None:
        self._debug_history.clear()
        self._error_count.clear()

    @asynccontextmanager
    async def debug_context(self, operation: str, reraise: bool = True):
        context = DebugContext(operation=operation)
        try:
            yield context
            context.success = True
        except Exception as exc:
            context.success = False
            context.error = str(exc)
            context.traceback = traceback.format_exc()
            self.track_error(operation)
            self.logger.exception(f"Debug context failed: {operation} - {exc}")
            if reraise:
                raise

        finally:
            self._debug_history.append(context)
            if len(self._debug_history) > self._max_history:
                self._debug_history = self._debug_history[-self._max_history:]

    def safe_execute(self, func: Callable[..., T], *args, default: T | None = None, **kwargs) -> T | None:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug(f"Safe execute failed for {func.__name__}: {exc}")
            return default

    async def safe_async_execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        default: T | None = None,
        **kwargs,
    ) -> T | None:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug(f"Safe async execute failed for {func.__name__}: {exc}")
            return default


class RetryableErrors:
    NETWORK_ERRORS = (
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.DDoSProtection,
    )

    RATE_LIMIT_ERRORS = (
        ccxt.RateLimitExceeded,
        ccxt.DDoSProtection,
    )

    TEMPORARY_ERRORS = (
        ccxt.ExchangeNotAvailable,
        ccxt.RequestTimeout,
        ccxt.NetworkError,
    )

    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        return isinstance(error, (*cls.NETWORK_ERRORS, *cls.RATE_LIMIT_ERRORS, *cls.TEMPORARY_ERRORS))

    @classmethod
    def get_error_category(cls, error: Exception) -> str:
        if isinstance(error, ccxt.NetworkError):
            return "network"
        if isinstance(error, (ccxt.RateLimitExceeded, ccxt.DDoSProtection)):
            return "rate_limit"
        if isinstance(error, ccxt.ExchangeNotAvailable):
            return "exchange_unavailable"
        if isinstance(error, ccxt.RequestTimeout):
            return "timeout"
        if isinstance(error, ccxt.InsufficientFunds):
            return "insufficient_funds"
        if isinstance(error, ccxt.InvalidOrder):
            return "invalid_order"
        return "unknown"


class SmartRetry:
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        context: str = "",
        retryable_only: bool = True,
    ) -> T:
        last_exception: Exception | None = None
        delay = self.base_delay

        for attempt in range(self.max_retries):
            try:
                return await operation()
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                is_retryable = RetryableErrors.is_retryable(exc)

                if retryable_only and not is_retryable:
                    self.logger.error(
                        f"Non-retryable error in {context}: {exc}",
                    )
                    raise

                category = RetryableErrors.get_error_category(exc)
                self.logger.warning(
                    f"Retryable error in {context} (attempt {attempt + 1}/{self.max_retries}): "
                    f"category={category} error={exc}",
                )

                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * self.exponential_base, self.max_delay)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("SmartRetry: exhausted without exception")


def with_auto_debug(level: DebugLevel = DebugLevel.NORMAL):
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            logger = logging.getLogger(func.__module__)
            debugger = AutoDebugger(logger=logger, level=level)
            operation_name = f"{func.__module__}.{func.__name__}"

            async with debugger.debug_context(operation_name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class ValidationResult:
    def __init__(
        self,
        valid: bool,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []

    def __bool__(self) -> bool:
        return self.valid


class SystemValidator:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.checks: list[ValidationResult] = []

    def add_check(self, result: ValidationResult) -> None:
        self.checks.append(result)

    async def validate_exchange(self, exchange: ccxt.Exchange) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        try:
            await exchange.fetch_time()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Exchange time fetch failed: {exc}")

        try:
            markets = await exchange.load_markets()
            if not markets:
                warnings.append("No markets loaded")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Market loading failed: {exc}")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    async def validate_database(self, repository: Any) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        try:
            await repository.verify_connectivity()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Database connectivity failed: {exc}")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    async def validate_configuration(self, settings: Any) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if not settings.binance_api_key:
            errors.append("BINANCE_API_KEY is required")

        if not settings.binance_api_secret:
            errors.append("BINANCE_API_SECRET is required")

        if not settings.postgres_dsn:
            errors.append("POSTGRES_DSN is required")

        if settings.binance_testnet is None:
            warnings.append("Running on testnet (not production)")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)

    def is_valid(self) -> bool:
        return all(check.valid for check in self.checks)

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.valid),
            "failed": sum(1 for c in self.checks if not c.valid),
            "all_valid": self.is_valid(),
            "errors": [e for c in self.checks for e in c.errors],
            "warnings": [w for c in self.checks for w in c.warnings],
        }
